# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for Prefill physical layout and byte-copy semantics."""

import ast
import importlib.util
import sys
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[3]


class DeviceView:
    """Use CPU storage to verify the exact byte spans of NPU task descriptors."""

    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return DeviceView(self.tensor[index])


class PrepackedPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "prefill_host_sync", REPO / "vllm_ascend/expert_offload/host_sync.py"
        )
        cls.host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.host)
        tree = ast.parse((REPO / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8"))
        names = {"_alloc_prefill_pool_slot", "_build_prefill_h2d_tasks"}
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.namespace = {
            "torch": torch,
            "torch_npu": SimpleNamespace(get_npu_format=lambda _: 2),
            "H2DCopyTask": make_dataclass("Task", ["source", "destination", "nbytes", "name"]),
            "_expert_weight": getattr,
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), "prefill_manager", "exec"), cls.namespace)

    @staticmethod
    def layer(rows=8, columns=4):
        # Matches MXFP's dense transposed [K/2, N, 2] expert layout.
        scale = torch.empty((2, rows, columns, 2), dtype=torch.uint8).transpose(1, 2)
        return SimpleNamespace(
            w13_weight=torch.empty((2, 8), dtype=torch.uint8),
            w2_weight=torch.empty((2, 4), dtype=torch.uint8),
            w13_weight_scale=scale,
            w2_weight_scale=scale.clone(),
            w13_weight_scale_fp32=torch.empty((2, 3), dtype=torch.float32),
        )

    def manager(self, backend="mte", mode="prepacked"):
        manager = SimpleNamespace(
            offload_config=SimpleNamespace(host_sync_backend=backend, host_sync_quant_copy=mode),
            scale_cpu_buffers={},
            offset_cpu_buffers={},
            scale_bias_cpu_buffers={},
            _shard_local=lambda eid: eid,
            w13_expert_size_bytes=8,
            w2_expert_size_bytes=4,
        )
        for name in (
            "w13",
            "w2",
            "w13_scale",
            "w2_scale",
            "w13_scale_fp32",
            "w13_offset",
            "w2_offset",
            "w13_scale_bias",
            "w2_scale_bias",
        ):
            setattr(manager, "_prefill_" + name, [])
        return manager

    def allocate(self, manager, layer, count=3):
        with patch.dict(sys.modules, {"vllm_ascend.expert_offload.host_sync": self.host}):
            self.namespace["_alloc_prefill_pool_slot"](manager, layer, "cpu", torch.uint8, count)

    def test_only_mte_prepacked_uint8_scales_change_layout(self):
        layer = self.layer()
        for backend, mode in (
            ("callback", "prepacked"),
            ("callback", "storage"),
            ("mte", "storage"),
            ("mte", "tensor"),
            ("mte", "prepacked"),
        ):
            with self.subTest(backend=backend, mode=mode):
                manager = self.manager(backend, mode)
                self.allocate(manager, layer)
                expected_packed = backend == "mte" and mode == "prepacked"
                for name in ("w13", "w2"):
                    pool = getattr(manager, f"_prefill_{name}_scale")[0]
                    reference = getattr(layer, f"{name}_weight_scale")
                    self.assertEqual(pool.is_contiguous(), not expected_packed)
                    if expected_packed:
                        self.assertEqual(pool.stride()[1:], reference.stride()[1:])
                    self.assertEqual(pool.untyped_storage().nbytes(), pool.numel())
                self.assertTrue(manager._prefill_w13[0].is_contiguous())
                self.assertTrue(manager._prefill_w13_scale_fp32[0].is_contiguous())

    def test_real_scale_shapes_copy_values_without_touching_neighbors(self):
        for k in (64, 32):
            reference = torch.empty((1, 4096, k, 2), dtype=torch.uint8).transpose(1, 2)
            pool = self.host.allocate_prepacked_scale_pool(reference, 3, "cpu")
            pool.zero_()
            source = torch.empty_strided(reference.shape[1:], reference.stride()[1:], dtype=torch.uint8)
            values = (torch.arange(source.numel()) % 251).to(torch.uint8).reshape(source.shape)
            source.copy_(values)
            task = self.namespace["H2DCopyTask"](source, DeviceView(pool[1]), source.numel(), "scale")
            converted = self.host.storage_quant_copy(task, lambda _: 2, allow_pack=False)
            converted.destination.copy_(converted.source)
            self.assertTrue(torch.equal(pool[1], values))
            self.assertEqual(pool[0].count_nonzero().item(), 0)
            self.assertEqual(pool[2].count_nonzero().item(), 0)
            self.assertEqual(pool.untyped_storage().nbytes(), 3 * source.numel())

    def test_manager_tasks_preserve_logical_values_and_destination_offsets(self):
        layer = self.layer()
        manager = self.manager()
        self.allocate(manager, layer)
        for name in ("w13", "w2"):
            pool = getattr(manager, f"_prefill_{name}_scale")[0]
            pool.zero_()
            setattr(manager, f"_prefill_{name}_scale", [DeviceView(pool)])
            reference = getattr(layer, f"{name}_weight_scale")[0]
            source = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype)
            source.copy_(torch.arange(source.numel()).reshape(source.shape))
            manager.scale_cpu_buffers[f"{name}_weight_scale"] = [[source]]
            getattr(manager, f"_prefill_{name}")[0].zero_()
        manager._expert_src_storage = lambda layer, eid, name: torch.full(
            (getattr(manager, f"{name}_expert_size_bytes"),), 7, dtype=torch.uint8
        ).untyped_storage()
        with patch.dict(sys.modules, {"vllm_ascend.expert_offload.host_sync": self.host}):
            tasks = self.namespace["_build_prefill_h2d_tasks"](manager, 0, 0, 0, 1)
        self.assertEqual(len(tasks), 4)
        for task in tasks:
            self.assertIsInstance(task.source, torch.UntypedStorage)
            self.assertIsInstance(task.destination, torch.UntypedStorage)
            task.destination.copy_(task.source)
        for name in ("w13", "w2"):
            pool = getattr(manager, f"_prefill_{name}_scale")[0].tensor
            self.assertTrue(torch.equal(pool[1], manager.scale_cpu_buffers[f"{name}_weight_scale"][0][0]))
            self.assertEqual(pool[0].count_nonzero().item(), 0)
            self.assertEqual(pool[2].count_nonzero().item(), 0)
            self.assertTrue(torch.all(getattr(manager, f"_prefill_{name}")[0][1] == 7))

    def test_invalid_or_mismatched_layout_fails_without_runtime_pack(self):
        for reference, count in (
            (torch.empty(2, 4), 3),
            (torch.empty(2, 4, dtype=torch.uint8), 0),
            (torch.empty(2, 8, dtype=torch.uint8)[:, ::2], 3),
        ):
            with self.assertRaises(ValueError):
                self.host.allocate_prepacked_scale_pool(reference, count, "cpu")
        pool = self.host.allocate_prepacked_scale_pool(self.layer().w13_weight_scale, 3, "cpu")
        source = torch.zeros(pool.shape[1:], dtype=pool.dtype)
        task = self.namespace["H2DCopyTask"](source, DeviceView(pool[0]), source.numel(), "scale")
        with self.assertRaisesRegex(RuntimeError, "Prepacked quant layout mismatch"):
            self.host.storage_quant_copy(task, lambda _: 2, allow_pack=False)


if __name__ == "__main__":
    unittest.main()
