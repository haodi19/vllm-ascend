# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native tests require both optional extensions built."""

import ast
import contextlib
import ctypes
import gc
import importlib.util
import sys
import time
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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
        config.validate_native_update_config(dict(good, host_update_backend="native_full"))
        config.validate_native_update_config(dict(good, callback_timing_port=18156, callback_mstx_enabled=True))
        for key, value in (
            ("moe_offload_debug", True),
            ("callback_timing_port", 18155),
            ("enable_multi_card", True),
            ("expert_prefetch_enabled", True),
            ("h2d_backend", "memfabric"),
        ):
            with self.assertRaises(ValueError):
                config.validate_native_update_config(dict(good, **{key: value}))
            with self.assertRaises(ValueError):
                config.validate_native_update_config(dict(good, host_update_backend="native_full", **{key: value}))

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
            "_alloc_prefill_pool_slot",
            "_build_prefill_h2d_tasks",
            "_build_quant_attr_h2d_tasks",
            "_quant_copy_views",
            "_load_expert_weights_into_slots",
            "_prefill_load_layer",
        }
        manager.body = [n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {
            "torch": torch,
            "torch_npu": SimpleNamespace(
                get_npu_format=lambda t: 2, npu=SimpleNamespace(stream=lambda s: contextlib.nullcontext())
            ),
            "_expert_weight": getattr,
            "H2DCopyTask": SimpleNamespace,
        }
        exec(compile(ast.Module(body=[manager], type_ignores=[]), "manager", "exec"), scope)
        cls.Manager = scope["ExpertOffloadManager"]

    def setUp(self):
        modules = {self.layout_module.__name__: self.layout_module, "vllm_ascend.expert_offload.native_update": config}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_manager_uses_full_batch_only_for_the_new_backend_and_keeps_prefill_sync(self):
        for enabled in (False, True):
            m = self.Manager()
            m._native_update = Mock(full_submit=enabled)
            m.num_device_layers, m.num_total_experts = 1, 3
            m._prefill_w13 = [self.torch.empty((3, 4), dtype=self.torch.uint8)]
            m.load_stream, m._debug = object(), False
            m._get_h2d_transport = Mock()
            m._synchronize_h2d = Mock()
            m._build_prefill_h2d_tasks = Mock(return_value=[object()])
            m._refresh_prefill_fp32_scale = Mock()
            m._build_expert_h2d_tasks = Mock(return_value=[object()])
            m._refresh_expert_fp32_scale = Mock()
            m._prefill_load_layer(0, None)
            m._synchronize_h2d.assert_called_once()
            m._load_expert_weights_into_slots(None, 0, [(1, 2)])
            if enabled:
                m._native_update.enqueue_prefill.assert_called_once_with(0)
                m._native_update.enqueue_eager.assert_called_once_with(0, [(1, 2)])
                m._get_h2d_transport.assert_not_called()
                m._build_prefill_h2d_tasks.assert_not_called()
                m._build_expert_h2d_tasks.assert_not_called()
            else:
                m._native_update.enqueue_prefill.assert_not_called()
                m._native_update.enqueue_eager.assert_not_called()
                self.assertEqual(m._build_prefill_h2d_tasks.call_count, 3)
                m._build_expert_h2d_tasks.assert_called_once_with(None, 0, 2, 1)
                self.assertEqual(m._get_h2d_transport().copy_batch.call_count, 2)

    def test_native_loads_final_layout_once_and_original_backend_stays_contiguous(self):
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
                    if backend == "native":
                        self.assertEqual(source.stride(), expected.stride())
                        with (
                            patch.object(torch.Tensor, "is_pinned", return_value=True),
                            patch.object(config, "pack_scale", side_effect=AssertionError("second scale allocation")),
                        ):
                            packed, reused = config.prepare_scale(torch, source, expected, require_reuse=True)
                        self.assertTrue(reused)
                        self.assertIs(packed, source)
                        self.assertEqual(
                            ctypes.string_at(packed.data_ptr(), raw.numel()),
                            ctypes.string_at(raw.data_ptr(), raw.numel()),
                        )
                    else:
                        self.assertTrue(source.is_contiguous())

    def test_eager_and_prefill_share_scale_storage_and_preserve_slot_offsets(self):
        torch = self.torch
        for backend in ("python", "native"):
            with self.subTest(backend=backend):
                manager = self.Manager()
                manager.offload_config = SimpleNamespace(shard_per_rank=False, host_update_backend=backend)
                manager.ep_size = 1
                manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers = {}, {}, {}
                manager._allocate_expert_host_tensor = lambda shape, dtype: torch.empty(shape, dtype=dtype)
                manager._shard_local = lambda eid: eid
                raw = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
                layer = SimpleNamespace(
                    w13_weight_scale=torch.empty((2, 16, 8), dtype=torch.uint8),
                    w13_weight=torch.empty((2, 2, 4), dtype=torch.uint8),
                    w2_weight=torch.empty((2, 2, 4), dtype=torch.uint8),
                )
                manager._init_layer_scale_buffers(layer, 0, 3)
                for eid, source in enumerate(manager.scale_cpu_buffers["w13_weight_scale"][0]):
                    manager._copy_scale_direct(source, raw + eid)
                layer.w13_weight_scale = self.layout_module.apply_mxfp4_weight_scale_layout(layer.w13_weight_scale)
                for name in (
                    "w13",
                    "w2",
                    "w13_scale",
                    "w13_scale_fp32",
                    "w13_offset",
                    "w2_scale",
                    "w2_offset",
                    "w13_scale_bias",
                    "w2_scale_bias",
                ):
                    setattr(manager, "_prefill_" + name, [])
                manager._alloc_prefill_pool_slot(layer, "cpu", torch.uint8, 3)
                pool = manager._prefill_w13_scale[0]
                if backend == "native":
                    self.assertEqual(pool.stride(), layer.w13_weight_scale.stride())
                else:
                    self.assertTrue(pool.is_contiguous())
                manager.w13_expert_size_bytes = manager.w2_expert_size_bytes = 8
                weight = torch.ones(8, dtype=torch.uint8)
                manager._expert_src_storage = lambda *args, weight=weight: weight.untyped_storage()
                pool.zero_()
                # Deliberately use different source expert and nonzero target slot.
                tasks = manager._build_prefill_h2d_tasks(0, 0, 2, 1)
                quant = tasks[-1]
                src = manager.scale_cpu_buffers["w13_weight_scale"][0][2]
                self.assertEqual(quant.source.data_ptr(), src.data_ptr())
                self.assertEqual(quant.destination.data_ptr(), pool[1].data_ptr())
                if backend == "native":
                    self.assertIsInstance(quant.source, torch.UntypedStorage)
                    self.assertIsInstance(quant.destination, torch.UntypedStorage)
                    self.assertEqual(quant.source.nbytes(), 128)
                else:
                    self.assertIsInstance(quant.source, torch.Tensor)
                quant.destination.copy_(quant.source)
                expected = self.layout_module.apply_mxfp4_weight_scale_layout(raw + 2)
                torch.testing.assert_close(pool[1], expected)
                self.assertFalse(pool[0].any())
                self.assertFalse(pool[2].any())
                eager = manager._build_quant_attr_h2d_tasks(layer, 0, 2, 1)[0]
                self.assertEqual(eager.source.data_ptr(), src.data_ptr())
                eager.destination.copy_(eager.source)
                torch.testing.assert_close(layer.w13_weight_scale[1], expected)

    def test_checkpoint_shards_load_in_either_order_without_another_bank(self):
        torch = self.torch
        for order in (("w1", "w3"), ("w3", "w1")):
            manager = self.Manager()
            manager.offload_config = SimpleNamespace(shard_per_rank=False, host_update_backend="native")
            manager.ep_size = 1
            manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers = {}, {}, {}
            allocations = []

            def allocate(shape, dtype, allocations=allocations):
                result = torch.empty(shape, dtype=dtype)
                allocations.append(result.data_ptr())
                return result

            manager._allocate_expert_host_tensor = allocate
            layer = SimpleNamespace(
                w13_weight_scale=torch.empty((2, 16, 8), dtype=torch.uint8),
                w2_weight_scale=torch.empty((2, 8, 8), dtype=torch.uint8),
            )
            manager._init_layer_scale_buffers(layer, 0, 3)
            self.assertEqual(len(allocations), 6)
            manager._shard_local = lambda eid: eid
            manager._weight_load_calls = 0
            manager._scale_shard_temp = {}
            # Queue work to ensure checkpoint views can be modified/unmapped
            # before the actual copy. Each submitted job must own its inputs.
            jobs = []
            manager._get_load_pool = lambda jobs=jobs: SimpleNamespace(
                submit=lambda function, *arguments: jobs.append((function, arguments))
            )
            manager._track_load_future = lambda future: None
            raw = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
            for shard in order:
                checkpoint = (raw[:8] if shard == "w1" else raw[8:]).clone()
                manager._load_scale_shard(0, 1, "w13_weight_scale", shard, checkpoint)
                checkpoint.zero_()
            checkpoint = raw[:8].clone()
            manager._load_scale_shard(0, 2, "w2_weight_scale", "w2", checkpoint)
            checkpoint.zero_()
            for fn, args in jobs:
                fn(*args)
            self.assertFalse(manager._scale_shard_temp)
            for name, eid, expected in (("w13_weight_scale", 1, raw), ("w2_weight_scale", 2, raw[:8])):
                tensor = manager.scale_cpu_buffers[name][0][eid]
                self.assertIn(tensor.data_ptr(), allocations)
                torch.testing.assert_close(tensor, self.layout_module.apply_mxfp4_weight_scale_layout(expected))
                self.assertEqual(
                    ctypes.string_at(tensor.data_ptr(), tensor.numel()),
                    ctypes.string_at(expected.data_ptr(), expected.numel()),
                )

    def test_scale_byte_copy_rejects_mismatched_or_aliasing_layouts(self):
        torch = self.torch
        source = torch.arange(12).reshape(3, 4).t()
        for target in (source.contiguous(), source.to(torch.float32), torch.empty(12, dtype=source.dtype)):
            left, right = config.scale_copy_views(source, target)
            self.assertIs(left, source)
            self.assertIs(right, target)
        source = torch.zeros(1).expand(3, 4)
        target = torch.zeros(1).expand(3, 4)
        self.assertIs(config.scale_copy_views(source, target)[0], source)

    def test_mxfp_layout_mismatch_cannot_silently_allocate_duplicate(self):
        torch = self.torch
        source = torch.zeros((4, 3), dtype=torch.uint8)
        target = torch.empty_strided((4, 3), (1, 4), dtype=torch.uint8)
        with patch.object(torch.Tensor, "is_pinned", return_value=True), patch.object(config, "pack_scale") as pack:
            with self.assertRaisesRegex(ValueError, "refusing a duplicate"):
                config.prepare_scale(torch, source, target, require_reuse=True)
            pack.assert_not_called()

    @unittest.skipUnless(lrc is not None, "build lrc_policy extension")
    def test_loaded_scale_addresses_survive_runtime_registration_and_native_execution(self):
        self.check_runtime("native")

    @unittest.skipUnless(lrc is not None, "build lrc_policy extension")
    def test_full_submission_reuses_sources_for_eager_prefill_and_graph(self):
        self.check_runtime("native_full")

    def check_runtime(self, backend):
        torch = self.torch
        manager = self.Manager()
        manager.offload_config = SimpleNamespace(
            shard_per_rank=False, host_update_backend=backend, callback_mstx_enabled=False
        )
        manager.ep_size = 1
        manager.scale_cpu_buffers, manager.offset_cpu_buffers, manager.scale_bias_cpu_buffers = {}, {}, {}
        manager._allocate_expert_host_tensor = lambda shape, dtype: torch.empty(shape, dtype=dtype)
        layer = SimpleNamespace(
            w13_weight=torch.zeros((2, 4), dtype=torch.uint8),
            w2_weight=torch.zeros((2, 4), dtype=torch.uint8),
            w13_weight_scale=torch.empty((2, 16, 8), dtype=torch.uint8),
        )
        manager._init_layer_scale_buffers(layer, 0, 3)
        raw = torch.arange(128, dtype=torch.uint8).reshape(16, 8)
        for eid, tensor in enumerate(manager.scale_cpu_buffers["w13_weight_scale"][0]):
            manager._copy_scale_direct(tensor, raw + eid)
        layer.w13_weight_scale = self.layout_module.apply_mxfp4_weight_scale_layout(layer.w13_weight_scale)
        manager.moe_layers = [layer]
        manager.num_device_layers = 1
        manager._prefill_w13 = [torch.zeros((3, 4), dtype=torch.uint8)]
        manager._prefill_w2 = [torch.zeros((3, 4), dtype=torch.uint8)]
        shape, strides = layer.w13_weight_scale.shape[1:], layer.w13_weight_scale.stride()[1:]
        manager._prefill_w13_scale = [torch.empty_strided((3, *shape), (128, *strides), dtype=torch.uint8)]
        manager.num_total_experts = 3
        manager.num_device_experts_for_layer = lambda index: 2
        manager.w13_expert_size_bytes = manager.w2_expert_size_bytes = 4
        manager._shard_local = lambda eid: eid
        weights = [torch.full((4,), eid + 10, dtype=torch.uint8) for eid in range(3)]
        manager._expert_src_storage = lambda index, eid, name: weights[eid].untyped_storage()
        manager.load_stream = SimpleNamespace(npu_stream=1, synchronize=lambda: None)
        manager.prefetch_topk = 1
        manager.cache_policy = SimpleNamespace(_native=lrc.Policy(1, 3, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True))
        before = [t.data_ptr() for t in manager.scale_cpu_buffers["w13_weight_scale"][0]]
        tables = []

        def transfers(src, dst, sizes, stream, library, cpu):
            tables.append((src, dst, sizes))
            return lrc.NativeTransfers(src, dst, sizes, 0, "", True)

        transfers.enqueue_all = True  # Proxy constructor retains the extension capability contract.
        modules = {
            "torch": torch,
            "torch_npu": SimpleNamespace(get_npu_format=lambda t: 2),
            "_vllm_ascend_lrc": SimpleNamespace(native_update_task=lrc.native_update_task, NativeTransfers=transfers),
        }
        with (
            patch.object(config.importlib, "import_module", side_effect=modules.__getitem__),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(config, "pack_scale", side_effect=AssertionError("second scale bank")),
        ):
            runtime = config.NativeUpdateRuntime(manager, getattr)
            ids = torch.tensor([[2]], dtype=torch.int64)
            mapping = np.array([0, 1, -1], dtype=np.int32)
            task = runtime.task((ids, mapping, layer, 0, None, False))
            lrc.run_native_update_for_test(task)
            if backend == "native_full":
                runtime.enqueue_prefill(0)
                for eid in range(3):
                    torch.testing.assert_close(manager._prefill_w13[0][eid], weights[eid])
                    torch.testing.assert_close(manager._prefill_w2[0][eid], weights[eid])
                    torch.testing.assert_close(
                        manager._prefill_w13_scale[0][eid],
                        self.layout_module.apply_mxfp4_weight_scale_layout(raw + eid),
                    )
                # Repeated calls read the registered source, including a nonzero slot.
                runtime.enqueue_eager(0, [(1, 2)])
                torch.testing.assert_close(layer.w13_weight[1], weights[2])
                unchanged = layer.w13_weight.clone()
                with self.assertRaisesRegex(RuntimeError, "outside table"):
                    runtime.enqueue_eager(0, [(0, 1), (2, 0)])
                torch.testing.assert_close(layer.w13_weight, unchanged)
                runtime.enqueue_eager(0, [])
                manager._prefill_w13[0] = manager._prefill_w13[0].clone()
                with self.assertRaisesRegex(RuntimeError, "destination changed"):
                    runtime.enqueue_prefill(0)
        self.assertEqual([row[-1] for row in tables[0][0]], before)
        self.assertEqual(runtime.reused_bytes, 3 * raw.numel())
        self.assertEqual(runtime.packed_bytes + runtime.replaced_bytes + runtime.derived_bytes, 0)
        torch.testing.assert_close(layer.w13_weight[mapping[2]], weights[2])
        torch.testing.assert_close(
            layer.w13_weight_scale[mapping[2]], self.layout_module.apply_mxfp4_weight_scale_layout(raw + 2)
        )

    def test_generic_scale_repack_replaces_canonical_buffer_without_retaining_old(self):
        torch = self.torch
        runtime = config.NativeUpdateRuntime.__new__(config.NativeUpdateRuntime)
        runtime.torch = torch
        runtime.npu = SimpleNamespace(get_npu_format=lambda t: 2)
        runtime.owners = []
        runtime.packed_bytes = runtime.reused_bytes = runtime.replaced_bytes = runtime.derived_bytes = 0
        old = torch.arange(12, dtype=torch.float32).reshape(4, 3).clone()
        old_ref = weakref.ref(old)
        canonical = [old]
        target = torch.empty_strided((1, 4, 3), (12, 1, 4), dtype=torch.float32)
        expected = old.clone()
        sources, destinations, sizes = [[]], [[]], []
        del old

        def pack(factory, src, dst):
            result = torch.empty_strided(dst.shape, dst.stride(), dtype=dst.dtype)
            result.copy_(src.reshape(dst.shape))
            return result

        with patch.object(config, "pack_scale", side_effect=pack):
            runtime._add_scale(target, canonical, sources, destinations, sizes)
        gc.collect()
        self.assertIsNone(old_ref())
        torch.testing.assert_close(canonical[0], expected)
        self.assertEqual(sources, [[canonical[0].data_ptr()]])
        self.assertEqual(runtime.packed_bytes, 0)
        self.assertEqual(runtime.replaced_bytes, 48)
        ctypes.memmove(destinations[0][0], sources[0][0], sizes[0])
        torch.testing.assert_close(target[0], expected)

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
