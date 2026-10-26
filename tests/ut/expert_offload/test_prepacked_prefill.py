# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for Prefill physical layout and byte-copy semantics."""

import ast
import contextlib
import importlib.util
import sys
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
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
        names = {
            "_alloc_prefill_pool_slot",
            "_build_prefill_h2d_tasks",
            "_prepacked_quant_task",
            "_build_quant_attr_h2d_tasks",
        }
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.namespace = {
            "torch": torch,
            "torch_npu": SimpleNamespace(get_npu_format=lambda _: 2),
            "H2DCopyTask": make_dataclass("Task", ["source", "destination", "nbytes", "name"]),
            "_expert_weight": getattr,
            "callback_stage": lambda *args: contextlib.nullcontext(),
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
        manager._prepacked_quant_task = MethodType(self.namespace["_prepacked_quant_task"], manager)
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

    def test_cached_views_reuse_endpoints_and_read_current_values(self):
        layer = self.layer()
        pool = self.host.allocate_prepacked_scale_pool(layer.w13_weight_scale, 3, "cpu")
        pool.zero_()
        sources = [torch.empty_strided(pool.shape[1:], pool.stride()[1:], dtype=pool.dtype) for _ in range(2)]
        cache = self.host.PrepackedQuantViews()
        for step in range(20):
            eid, slot = step % 2, step % 3
            sources[eid].fill_(step)
            src, dst, nbytes = cache.prepare(sources[eid], DeviceView(pool), slot, eid, "pool", lambda _: 2)
            again = cache.prepare(sources[eid], DeviceView(pool), slot, eid, "pool", lambda _: 2)
            self.assertIs(again[0], src)
            self.assertIs(again[1], dst)
            self.assertEqual(nbytes, sources[eid].numel())
            dst.copy_(src)
            self.assertTrue(torch.all(pool[slot] == step))
        self.assertEqual(len(cache.sources), 2)
        self.assertEqual(len(cache.destinations), 1)
        self.assertEqual(len(cache.destinations["pool"][3]), 3)

    def test_cached_views_invalidate_replaced_allocations_and_keep_old_views_alive(self):
        cache = self.host.PrepackedQuantViews()
        source, pool = torch.full((2, 4), 3, dtype=torch.uint8), torch.zeros((3, 2, 4), dtype=torch.uint8)
        old = cache.prepare(source, DeviceView(pool), 1, "src", "dst", lambda _: 2)
        new_source, new_pool = source.clone().fill_(8), torch.zeros_like(pool)
        new = cache.prepare(new_source, DeviceView(new_pool), 1, "src", "dst", lambda _: 2)
        self.assertNotEqual(old[0].data_ptr(), new[0].data_ptr())
        self.assertNotEqual(old[1].data_ptr(), new[1].data_ptr())
        old[1].copy_(old[0])
        new[1].copy_(new[0])
        self.assertTrue(torch.all(pool[1] == 3))
        self.assertTrue(torch.all(new_pool[1] == 8))
        self.assertEqual(len(cache.sources), 1)
        self.assertEqual(len(cache.destinations), 1)

    def test_cached_views_recheck_layout_format_and_storage_bounds(self):
        cache = self.host.PrepackedQuantViews()
        source, pool = torch.arange(8, dtype=torch.uint8).reshape(2, 4), torch.zeros((3, 2, 4), dtype=torch.uint8)
        cache.prepare(source, DeviceView(pool), 1, "src", "dst", lambda _: 2)
        source.transpose_(0, 1)
        pool.transpose_(1, 2)
        src, dst, _ = cache.prepare(source, DeviceView(pool), 1, "src", "dst", lambda _: 2)
        dst.copy_(src)
        self.assertTrue(torch.equal(pool[1], source))
        with self.assertRaisesRegex(RuntimeError, "NPU ND"):
            cache.prepare(source, DeviceView(pool), 1, "src", "dst", lambda _: 29)
        with self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            cache.prepare(source.contiguous(), DeviceView(pool), 1, "src", "dst", lambda _: 2)
        for slot in (-1, 3):
            with self.assertRaises(ValueError):
                cache.prepare(source, DeviceView(pool), slot, "src", "dst", lambda _: 2)
        wrapped = DeviceView(pool)
        wrapped.untyped_storage = lambda: torch.empty(1, dtype=torch.uint8).untyped_storage()
        with self.assertRaisesRegex(RuntimeError, "exceeds storage"):
            cache.prepare(source, wrapped, 1, "src", "dst", lambda _: 2)

    def test_cached_views_preserve_nonzero_offsets_and_reject_overlapping_slots(self):
        source_backing = torch.arange(16, dtype=torch.uint8)
        source = source_backing[4:12].reshape(2, 4)
        backing = torch.zeros(40, dtype=torch.uint8)
        pool = backing[8:32].reshape(3, 2, 4)
        cache = self.host.PrepackedQuantViews()
        src, dst, _ = cache.prepare(source, DeviceView(pool), 1, 0, 0, lambda _: 2)
        dst.copy_(src)
        self.assertTrue(torch.equal(pool[1], source))
        self.assertEqual(backing[:16].count_nonzero().item(), 0)
        self.assertEqual(backing[24:].count_nonzero().item(), 0)
        overlap = pool.as_strided(pool.shape, (0, 4, 1))
        with self.assertRaisesRegex(RuntimeError, "non-overlapping"):
            cache.prepare(source, DeviceView(overlap), 1, 0, 0, lambda _: 2)

    def test_decode_builder_uses_cached_views_for_prepacked_scale(self):
        layer = self.layer()
        manager = self.manager()
        source = torch.empty_strided(
            layer.w13_weight_scale.shape[1:], layer.w13_weight_scale.stride()[1:], dtype=torch.uint8
        ).fill_(17)
        manager.scale_cpu_buffers = {"w13_weight_scale": [[source]]}
        layer.w13_weight_scale = DeviceView(layer.w13_weight_scale)
        with patch.dict(sys.modules, {"vllm_ascend.expert_offload.host_sync": self.host}):
            first = self.namespace["_build_quant_attr_h2d_tasks"](manager, layer, 0, 0, 1)
            second = self.namespace["_build_quant_attr_h2d_tasks"](manager, layer, 0, 0, 1)
        self.assertEqual(len(first), 1)
        self.assertIs(first[0].source, second[0].source)
        self.assertIs(first[0].destination, second[0].destination)
        first[0].destination.copy_(first[0].source)
        self.assertTrue(torch.all(layer.w13_weight_scale.tensor[1] == 17))


if __name__ == "__main__":
    unittest.main()
