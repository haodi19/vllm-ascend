# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native tests require both optional extensions built."""

import ast
import ctypes
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "native_update_config", ROOT / "vllm_ascend/expert_offload/native_update.py"
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
try:
    import _vllm_ascend_lrc as lrc
    import vllm_ascend_host_sync as hs
except ImportError:
    lrc = hs = None


class LayoutAndConfig(unittest.TestCase):
    def test_default_unchanged(self):
        config.validate_native_update_config({})
        config.validate_native_update_config({"host_update_backend": "python", "moe_offload_debug": True})

    def test_dense_transpose_and_reject_alias(self):
        self.assertEqual(config.dense_span((4, 3), (1, 4)), 12)
        for strides in ((0, 1), (1, 1), (4, 1)):
            with self.assertRaises(ValueError):
                config.dense_span((4, 3), strides)

    def test_unsupported_config(self):
        good = dict(
            host_update_backend="native", host_sync_backend="mte", cache_policy_backend="cpp", cache_policy_enabled=True
        )
        config.validate_native_update_config(good)
        for key, value in (
            ("moe_offload_debug", True),
            ("callback_timing_port", 18155),
            ("enable_multi_card", True),
            ("expert_prefetch_enabled", True),
            ("h2d_backend", "memfabric"),
        ):
            with self.assertRaises(ValueError):
                config.validate_native_update_config(dict(good, **{key: value}))

    def test_packed_bytes_match_transposed_destination(self):
        import torch

        class CpuFactory:
            @staticmethod
            def empty_strided(shape, strides, **kwargs):
                kwargs.pop("pin_memory")
                return torch.empty_strided(shape, strides, **kwargs)

        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        destination = torch.empty_strided((4, 3), (1, 4), dtype=torch.float64)
        packed = config.pack_scale(CpuFactory, source, destination)
        ctypes.memmove(destination.data_ptr(), packed.data_ptr(), packed.numel() * packed.element_size())
        torch.testing.assert_close(destination, source.to(torch.float64))


class ScaleLoading(unittest.TestCase):
    """Run actual loader/layout methods on CPU, without importing vLLM/NPU."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        layout_name = "vllm_ascend.quantization.methods.w4a8_mxfp4"
        cls.layout_module = ModuleType(layout_name)
        cls.layout_module.torch = torch
        tree = ast.parse((ROOT / "vllm_ascend/quantization/methods/w4a8_mxfp4.py").read_text(encoding="utf-8"))
        helper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_mxfp4_weight_scale_layout"
        )
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "layout", "exec"), cls.layout_module.__dict__)
        tree = ast.parse((ROOT / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8"))
        manager = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExpertOffloadManager")
        names = {
            "_init_layer_scale_buffers",
            "_copy_scale_direct",
            "_copy_scale_assembled",
            "_assert_native_reload_safe",
            "load_w13",
            "load_w2",
            "_load_scale_shard",
            "_finalize_offload",
            "process_weights_after_loading",
        }
        manager.body = [n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[manager], type_ignores=[]), "manager", "exec"), scope)
        cls.Manager = scope["ExpertOffloadManager"]

    def test_single_buffer_mxfp4_and_original_eager_prefill_parity(self):
        torch = self.torch
        with patch.dict(sys.modules, {self.layout_module.__name__: self.layout_module}):
            for assembled in (False, True):
                raw = (torch.arange(16 * 8).reshape(16, 8) % 251).to(torch.uint8)
                expected = self.layout_module.apply_mxfp4_weight_scale_layout(raw)
                for backend in ("python", "native"):
                    manager = self.Manager()
                    manager.offload_config = SimpleNamespace(shard_per_rank=False, host_update_backend=backend)
                    manager.ep_size = 1
                    manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers = {}, {}, {}
                    manager._allocate_expert_host_tensor = lambda shape, dtype: torch.empty(shape, dtype=dtype)
                    layer = SimpleNamespace(w13_weight_scale=torch.empty((2, 16, 8), dtype=torch.uint8))
                    manager._init_layer_scale_buffers(layer, 0, 2)
                    source = manager.scale_cpu_buffers["w13_weight_scale"][0][1]
                    if assembled:
                        manager._copy_scale_assembled(source, raw[:8], raw[8:])
                    else:
                        manager._copy_scale_direct(source, raw)
                    torch.testing.assert_close(source, expected)
                    # Existing eager copy and contiguous prefill target keep the same values.
                    for target in (torch.empty_like(expected), torch.empty(expected.shape, dtype=torch.uint8)):
                        target.copy_(source.reshape(target.shape))
                        torch.testing.assert_close(target, expected)
                    if backend == "python":
                        self.assertTrue(source.is_contiguous())
                    else:
                        self.assertEqual(source.stride(), expected.stride())
                        # This CPU test substitutes pinning only; the NPU test uses actual pinned memory.
                        with patch.object(torch.Tensor, "is_pinned", return_value=True):
                            packed, reused = config.prepare_scale(torch, source, expected)
                        self.assertTrue(reused)
                        self.assertIs(packed, source)
                        self.assertEqual(
                            ctypes.string_at(source.data_ptr(), raw.numel()),
                            ctypes.string_at(raw.data_ptr(), raw.numel()),
                        )

    def test_scale_fallback_conversion_and_offset_reuse(self):
        torch = self.torch

        class CpuFactory:
            @staticmethod
            def empty_strided(shape, strides, **kwargs):
                kwargs.pop("pin_memory")
                return torch.empty_strided(shape, strides, **kwargs)

        slab = torch.arange(40, dtype=torch.float32)
        source = slab[5:17].view(3, 4).t()
        target = torch.empty_like(source)
        with patch.object(torch.Tensor, "is_pinned", return_value=True):
            packed, reused = config.prepare_scale(CpuFactory, source, target)
            self.assertTrue(reused)
            self.assertEqual(packed.data_ptr(), slab.data_ptr() + 5 * slab.element_size())
            converted, reused = config.prepare_scale(CpuFactory, source, target.to(torch.float64))
            self.assertFalse(reused)
            torch.testing.assert_close(converted, source.to(torch.float64))
            repacked, reused = config.prepare_scale(CpuFactory, source.contiguous(), target)
            self.assertFalse(reused)
            self.assertEqual(repacked.stride(), target.stride())
            torch.testing.assert_close(repacked, source)
        with patch.object(torch.Tensor, "is_pinned", return_value=False):
            _, reused = config.prepare_scale(CpuFactory, source, target)
            self.assertFalse(reused)
        with self.assertRaises(ValueError):
            config.prepare_scale(CpuFactory, source[:1], target)

    def test_reload_rejected_before_any_source_write(self):
        manager = self.Manager()
        manager._native_update = SimpleNamespace(tables=[object()])
        operations = [
            lambda: manager.load_w13(0, 0, None, "w1"),
            lambda: manager.load_w2(0, 0, None),
            lambda: manager._load_scale_shard(0, 0, "w2_weight_scale", "w2", None),
            lambda: manager.process_weights_after_loading(),
            lambda: manager._finalize_offload(None),
        ]
        for operation in operations:
            with self.assertRaisesRegex(RuntimeError, "native expert buffers are registered"):
                operation()
        manager._assert_native_reload_safe(1)  # A later, unregistered layer may still load.
        manager._native_update = None
        manager._assert_native_reload_safe()  # Original backend unchanged.

    def test_rebound_destination_rejected_before_task_install(self):
        torch = self.torch
        original = torch.empty((2, 3))
        layer = SimpleNamespace(w13_weight=original.clone())
        runtime = config.NativeUpdateRuntime.__new__(config.NativeUpdateRuntime)
        runtime.manager = SimpleNamespace(moe_layers=[layer])
        runtime.tables = [object()]
        runtime.targets = [{"w13_weight": (original, config.tensor_layout(original))}]
        runtime.expert_weight = getattr
        with self.assertRaisesRegex(RuntimeError, "destination changed"):
            runtime.task((None, None, layer, 0, None, False))


@unittest.skipUnless(lrc is not None and hs is not None, "build lrc_policy and host_sync extensions")
class NativeExecution(unittest.TestCase):
    def test_transfer_rejects_pointer_range_wraparound(self):
        with self.assertRaisesRegex(ValueError, "range overflow"):
            lrc.NativeTransfers([[ctypes.c_void_p(-1).value - 3]], [[16]], [16], 0, "", True)

    def fixture(self):
        self.policy = lrc.Policy(1, 8, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
        self.source = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
        self.destination = np.zeros((4, 16), dtype=np.uint8)
        self.table = lrc.NativeTransfers(
            [[row.ctypes.data] for row in self.source],
            [[row.ctypes.data] for row in self.destination],
            [16],
            0,
            "",
            True,
        )
        self.mapping = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
        self.ids = np.array([[4, 5]], dtype=np.int64)

    def task(self, prefetch=False):
        return lrc.native_update_task(
            self.policy, self.table, 0, self.ids, None, self.mapping, prefetch, 1, (self.source, self.destination)
        )

    def test_replay_reads_new_ids_and_updates_policy(self):
        self.fixture()
        capsule = self.task()
        for ids in ([4, 5], [6, 7], [6, 7], [0, 1]):
            self.ids[0] = ids
            lrc.run_native_update_for_test(capsule)
            for eid in ids:
                self.assertGreaterEqual(self.mapping[eid], 0)
                np.testing.assert_array_equal(self.destination[self.mapping[eid]], self.source[eid])
            self.assertEqual(len(set(self.mapping[self.mapping >= 0])), 4)
        self.assertEqual(self.policy.step(0), 4)

    def test_prefetch_cap_and_no_observation(self):
        self.fixture()
        lrc.run_native_update_for_test(self.task(True))
        self.assertEqual(sum(self.mapping[4:] >= 0), 1)
        self.assertEqual(self.policy.step(0), 0)

    def test_random_replay_matches_existing_planner_residency_and_lrc(self):
        self.fixture()
        self.destination[:] = self.source[:4]
        reference = lrc.Policy(1, 8, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
        reference_map = self.mapping.copy()
        scores = np.zeros((1, 2), dtype=np.float32)
        tasks = [
            lrc.native_update_task(
                self.policy, self.table, 0, self.ids, scores, self.mapping, prefetch, 1, (self.source, self.destination)
            )
            for prefetch in (False, True)
        ]
        random = np.random.default_rng(123)
        for step in range(200):
            self.ids[:] = random.integers(0, 8, size=(1, 2))
            scores[:] = random.random((1, 2))
            prefetch = bool(step % 2)
            reference.plan(0, self.ids, scores, reference_map, prefetch, 1, False)
            lrc.run_native_update_for_test(tasks[int(prefetch)])
            np.testing.assert_array_equal(self.mapping >= 0, reference_map >= 0)
            self.assertEqual(self.policy.snapshot(0), reference.snapshot(0))
            for eid in np.flatnonzero(self.mapping >= 0):
                np.testing.assert_array_equal(self.destination[self.mapping[eid]], self.source[eid])

    def test_worker_failure_does_not_publish_success(self):
        self.fixture()
        self.mapping[0] = 100
        with hs.Session(slots=1, backend="cpu") as session:
            session.register_task(0, lambda: self.fail("unexpected Python fallback"))
            session.install_native_task(0, self.task())
            session.start()
            session.submit_for_test(0, 1)
            with self.assertRaises(RuntimeError):
                session.wait(0, 1)
            state = session.snapshot(0)
            self.assertEqual(state["done_generation"], 0)
            self.assertEqual(state["error_generation"], 1)

    def test_invalid_mapping_not_committed(self):
        self.fixture()
        self.mapping[0] = 9
        before = self.mapping.copy()
        with self.assertRaises(RuntimeError):
            lrc.run_native_update_for_test(self.task())
        np.testing.assert_array_equal(self.mapping, before)
        self.assertFalse(self.destination.any())

    def test_shortfall_no_partial_copy(self):
        self.fixture()
        self.ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        before = self.mapping.copy()
        with self.assertRaises(RuntimeError):
            lrc.run_native_update_for_test(self.task())
        np.testing.assert_array_equal(self.mapping, before)
        self.assertFalse(self.destination.any())

    def test_worker_completes_while_python_holds_gil(self):
        self.fixture()
        with hs.Session(slots=1, backend="cpu") as session:
            session.register_task(0, lambda: self.fail("native task called Python"))
            session.install_native_task(0, self.task())
            session.start()
            # Directly inspect completion memory without Session.wait (which releases GIL).
            info = session.info()
            done = ctypes.c_uint64.from_address(info["host_address"] + info["done_offset"])
            old_interval = sys.getswitchinterval()
            try:
                sys.setswitchinterval(1.0)
                session.submit_for_test(0, 1)
                deadline = time.perf_counter() + 0.5
                while not done.value and time.perf_counter() < deadline:
                    pass
                self.assertEqual(done.value, 1)
            finally:
                sys.setswitchinterval(old_interval)
            with self.assertRaises(RuntimeError):
                session.install_native_task(0, self.task())


if __name__ == "__main__":
    unittest.main()
