# SPDX-License-Identifier: Apache-2.0
"""CPU checks for optional MoE dispatch without NPU imports."""

import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class MoeHostSyncTests(unittest.TestCase):
    def test_prepacked_loader_matches_original_values_without_runtime_pack(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        names = {"_init_layer_scale_buffers", "_copy_scale_assembled", "_copy_scale_direct"}
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in methods:
            node.decorator_list = []
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "loader", "exec"), namespace)
        layout_tree = ast.parse(
            (self.root / "vllm_ascend/quantization/methods/w4a8_mxfp4.py").read_text(encoding="utf-8")
        )
        layout = next(
            node
            for node in layout_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "apply_mxfp4_weight_scale_layout"
        )
        exec(compile(ast.Module(body=[layout], type_ignores=[]), "layout", "exec"), namespace)
        task_type = make_dataclass("Task", ["source", "destination", "nbytes", "name"])

        class DeviceView:
            device = SimpleNamespace(type="npu")

            def __init__(self, tensor):
                self.tensor = tensor

            def __getattr__(self, name):
                return getattr(self.tensor, name)

        modules = {
            "vllm_ascend.quantization.methods.w4a8_mxfp4": SimpleNamespace(
                apply_mxfp4_weight_scale_layout=namespace[layout.name]
            ),
            "vllm_ascend.expert_offload.host_sync": self.module,
        }
        with patch.dict(sys.modules, modules):
            for n, k in ((6, 8), (4096, 128), (4096, 64)):
                layer = SimpleNamespace(
                    w13_weight_scale=torch.empty((1, n, k), dtype=torch.uint8),
                    w2_weight_scale=torch.empty((1, n, k), dtype=torch.uint8),
                )
                managers = []
                for mode in ("storage", "prepacked"):
                    manager = SimpleNamespace(
                        offload_config=SimpleNamespace(
                            shard_per_rank=False, host_sync_backend="mte", host_sync_quant_copy=mode
                        ),
                        scale_cpu_buffers={},
                        offset_cpu_buffers={},
                        scale_bias_cpu_buffers={},
                        ep_size=1,
                        _allocate_expert_host_tensor=lambda shape, dtype: torch.empty(shape, dtype=dtype),
                    )
                    namespace["_init_layer_scale_buffers"](manager, layer, 0, 2)
                    managers.append(manager)
                values = (torch.arange(n * k) % 251).to(torch.uint8).reshape(n, k)
                for manager in managers:
                    for expert in range(2):
                        namespace["_copy_scale_assembled"](
                            manager.scale_cpu_buffers["w13_weight_scale"][0][expert], values[: n // 2], values[n // 2 :]
                        )
                        namespace["_copy_scale_direct"](manager.scale_cpu_buffers["w2_weight_scale"][0][expert], values)
                for name in ("w13_weight_scale", "w2_weight_scale"):
                    original = managers[0].scale_cpu_buffers[name][0][0]
                    packed = managers[1].scale_cpu_buffers[name][0][0]
                    self.assertTrue(torch.equal(original, packed))
                    self.assertEqual(original.untyped_storage().nbytes(), packed.untyped_storage().nbytes())
                    backing = torch.zeros((2, n, k), dtype=torch.uint8)
                    dst = namespace[layout.name](backing[1])
                    allocator = Mock(side_effect=AssertionError("runtime packing forbidden"))
                    task = task_type(packed, DeviceView(dst), packed.numel(), name)
                    copy = self.module.storage_quant_copy(task, lambda _: 2, allocator, allow_pack=False)
                    copy.destination.copy_(copy.source)
                    self.assertTrue(torch.equal(dst, original))
                    self.assertEqual(backing[0].count_nonzero().item(), 0)
                    allocator.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "Prepacked quant layout mismatch"):
                        self.module.storage_quant_copy(
                            task_type(original, DeviceView(dst), original.numel(), name),
                            lambda _: 2,
                            allocator,
                            allow_pack=False,
                        )

    def test_quant_tensor_mode_bypasses_only_storage_adapter(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_quant_attr_h2d_tasks"
        )
        task_type = make_dataclass("Task", ["source", "destination", "nbytes", "name"])
        namespace = {
            "H2DCopyTask": task_type,
            "torch_npu": Mock(),
            "callback_stage": lambda *args: contextlib.nullcontext(),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "quant_tasks", "exec"), namespace)
        for backend, mode, adapter_calls in (
            ("callback", "storage", 0),
            ("callback", "prepacked", 0),
            ("mte", "storage", 1),
            ("mte", "tensor", 0),
            ("mte", "prepacked", 1),
        ):
            with self.subTest(backend=backend, mode=mode):
                source = torch.arange(12).reshape(3, 4)
                destination = torch.zeros((1, 3, 4), dtype=source.dtype)
                manager = SimpleNamespace(
                    scale_cpu_buffers={"scale": [[source]]},
                    offset_cpu_buffers={},
                    scale_bias_cpu_buffers={},
                    _shard_local=lambda eid: 0,
                    offload_config=SimpleNamespace(host_sync_backend=backend, host_sync_quant_copy=mode),
                    _mte_pending_quant_sources=[],
                )
                adapter = Mock(side_effect=lambda task, *args, **kwargs: task)
                with patch.dict(
                    sys.modules, {"vllm_ascend.expert_offload.host_sync": SimpleNamespace(storage_quant_copy=adapter)}
                ):
                    tasks = namespace[method.name](manager, SimpleNamespace(scale=destination), 0, 0, 0)
                self.assertEqual(adapter.call_count, adapter_calls)
                if adapter_calls:
                    self.assertEqual(adapter.call_args.kwargs["allow_pack"], mode != "prepacked")
                self.assertEqual(len(manager._mte_pending_quant_sources), adapter_calls)
                self.assertEqual(len(tasks), 1)
                self.assertTrue(torch.equal(tasks[0].source, source))
                self.assertEqual(tasks[0].destination.data_ptr(), destination.data_ptr())

    def test_quant_copy_config_validation(self):
        for mode in ("storage", "tensor", "prepacked"):
            self.module.validate_host_sync_config({"host_sync_quant_copy": mode})
        with self.assertRaisesRegex(ValueError, "host_sync_quant_copy"):
            self.module.validate_host_sync_config({"host_sync_quant_copy": "invalid"})

    def test_packed_sources_live_until_successful_h2d_sync(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_synchronize_h2d")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "h2d_sync", "exec"), namespace)
        pending = [object()]
        transport = Mock()
        manager = SimpleNamespace(
            _mte_pending_quant_sources=pending, load_stream="load", _get_h2d_transport=lambda: transport
        )
        transport.synchronize.side_effect = RuntimeError("incomplete")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            namespace["_synchronize_h2d"](manager)
        self.assertEqual(len(pending), 1)
        transport.synchronize.side_effect = lambda stream: self.assertEqual(len(pending), 1)
        namespace["_synchronize_h2d"](manager)
        self.assertEqual(pending, [])

    def test_transposed_quant_storage_matches_tensor_copy(self):
        class DeviceView:
            device = SimpleNamespace(type="npu")

            def __init__(self, tensor):
                self.tensor = tensor

            def __getattr__(self, name):
                return getattr(self.tensor, name)

        for matching_layout in (False, True):
            # Real MXFP4 reshape/transpose pattern, with a nonzero slot offset.
            backing = torch.zeros((3, 4, 6, 2), dtype=torch.uint8)
            destination = backing[1].transpose(0, 1)
            source = torch.arange(48, dtype=torch.uint8).reshape(4, 6, 2).transpose(0, 1)
            if not matching_layout:
                source = source.contiguous()
            task_type = make_dataclass("Task", ["source", "destination", "nbytes", "name"])
            task = task_type(source, DeviceView(destination), 48, "scale")

            def allocate(shape, stride, **kwargs):
                kwargs["pin_memory"] = False  # CPU-only validation; NPU run uses pinned storage.
                return torch.empty_strided(shape, stride, **kwargs)

            allocator = Mock(side_effect=allocate)
            stages = []

            @contextlib.contextmanager
            def stage(name, stages=stages):
                stages.append(name)
                yield

            converted = self.module.storage_quant_copy(task, lambda tensor: 2, allocator, stage=stage)
            converted.destination.copy_(converted.source)
            self.assertTrue(torch.equal(destination, source))
            self.assertEqual(backing[0].sum().item(), 0)
            self.assertEqual(backing[2].sum().item(), 0)
            self.assertEqual(allocator.call_count, 0 if matching_layout else 1)
            self.assertEqual(stages.count("quant_pack"), 0 if matching_layout else 1)
            self.assertEqual(stages.count("quant_allocate"), allocator.call_count)
            self.assertEqual(stages.count("quant_source_view"), 1)
            self.assertEqual(stages.count("quant_destination_view"), 1)

    def quant_task(self):
        class Storage(bytearray):
            def nbytes(self):
                return len(self)

        def tensor(device, offset):
            return SimpleNamespace(
                device=SimpleNamespace(type=device),
                dtype="float32",
                shape=(2,),
                stride=lambda: (1,),
                is_contiguous=lambda: True,
                numel=lambda: 2,
                element_size=lambda: 4,
                storage_offset=lambda: offset,
                untyped_storage=lambda: Storage(range(32)),
            )

        task_type = make_dataclass("Task", ["source", "destination", "nbytes", "name", "non_blocking"])
        return task_type(tensor("cpu", 1), tensor("npu", 3), 8, "scale", True)

    def test_quant_storage_copy_preserves_byte_offsets_and_size(self):
        task = self.quant_task()
        converted = self.module.storage_quant_copy(task, lambda tensor: 2)
        self.assertEqual(converted.source, bytearray(range(4, 12)))
        self.assertEqual(converted.destination, bytearray(range(12, 20)))
        self.assertEqual(converted.nbytes, 8)
        self.assertTrue(converted.non_blocking)
        self.assertEqual(task.source.dtype, "float32")

    def test_quant_storage_copy_rejects_unsafe_layout_or_conversion(self):
        for mode in ("dtype", "shape", "stride", "format", "bytes", "bounds"):
            task = self.quant_task()
            if mode == "dtype":
                task.destination.dtype = "float16"
            elif mode == "shape":
                task.destination.shape = (1, 2)
            elif mode == "stride":
                task.destination.stride = lambda: (2,)
            elif mode == "bytes":
                task.nbytes = 4
            elif mode == "bounds":
                task.destination.storage_offset = lambda: 8
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                self.module.storage_quant_copy(task, lambda tensor, mode=mode: 29 if mode == "format" else 2)

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "host_sync_test", cls.root / "vllm_ascend/expert_offload/host_sync.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location("moe_runner_test", cls.root / "script/run_moe_host_sync.py")
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)

    def test_default_needs_no_extension(self):
        self.module.validate_host_sync_config({})
        self.module.validate_host_sync_config({"host_sync_backend": "callback"})

    def test_load_diagnosis_disabled_does_not_touch_stream(self):
        obj = self.dispatcher()
        obj.manager.offload_config = SimpleNamespace()
        self.assertFalse(obj.diagnose_load([object()], [(0, 1)], None, 1))

    def test_load_diagnosis_preserves_copies_and_scale_order(self):
        obj = self.dispatcher()
        events = Mock()
        obj.manager.offload_config = SimpleNamespace(host_sync_diagnose_load=True)
        obj.active_callback = (2, 123)
        obj.load_diagnostic_batches = 0
        obj.manager.load_stream = events.stream
        obj.manager._get_h2d_transport = lambda: events.transport
        obj.manager._refresh_expert_fp32_scale = events.refresh
        task = SimpleNamespace(name="weight", nbytes=64)
        with patch("builtins.print"):
            self.assertTrue(obj.diagnose_load([task], [(3, 9)], "layer", 1))
        self.assertEqual(
            [c[0] for c in events.mock_calls],
            ["stream.synchronize", "transport.copy_batch", "stream.synchronize", "refresh", "stream.synchronize"],
        )
        events.transport.copy_batch.assert_called_once_with([task])
        events.refresh.assert_called_once_with("layer", 3)
        obj.load_diagnostic_batches = 8
        self.assertFalse(obj.diagnose_load([task], [(3, 9)], "layer", 1))

    def test_diagnostic_failed_copy_does_not_run_scale_refresh(self):
        obj = self.dispatcher()
        obj.manager.offload_config = SimpleNamespace(host_sync_diagnose_load=True)
        obj.active_callback, obj.load_diagnostic_batches = (2, 123), 0
        obj.manager.load_stream = Mock()
        transport = Mock()
        transport.copy_batch.side_effect = RuntimeError("copy failure")
        obj.manager._get_h2d_transport = lambda: transport
        obj.manager._refresh_expert_fp32_scale = Mock()
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "copy failure"):
            obj.diagnose_load([SimpleNamespace(name="weight", nbytes=64)], [(3, 9)], None, 1)
        obj.manager._refresh_expert_fp32_scale.assert_not_called()

    def test_callback_keeps_ordinary_load_stream(self):
        npu = SimpleNamespace(Stream=Mock(return_value="ordinary"))
        self.assertEqual(self.module.create_offload_load_stream(npu, "callback"), "ordinary")
        npu.Stream.assert_called_once_with()

    def test_mte_uses_direct_submission_load_stream(self):
        npu = SimpleNamespace(Stream=Mock(), SyncLaunchStream=Mock(return_value="direct"))
        self.assertEqual(self.module.create_offload_load_stream(npu, "mte"), "direct")
        npu.SyncLaunchStream.assert_called_once_with()
        npu.Stream.assert_not_called()

    def test_missing_sync_stream_cannot_silently_fallback(self):
        npu = SimpleNamespace(Stream=Mock())
        with self.assertRaisesRegex(RuntimeError, "SyncLaunchStream"):
            self.module.create_offload_load_stream(npu, "mte")
        npu.Stream.assert_not_called()

    def test_manager_initializes_selected_load_stream(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        assignments = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "load_stream" for t in n.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].value.func.id, "create_offload_load_stream")

    def test_invalid_configuration(self):
        for config in (
            {"host_sync_backend": "typo"},
            {"host_sync_backend": "mte", "enable_multi_card": True},
            {"host_sync_backend": "mte", "h2d_backend": "acl"},
            {"host_sync_backend": "mte"},
            {"host_sync_backend": "mte", "expert_offload": True, "host_sync_library": "relative.so"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.module.validate_host_sync_config(config)

    def test_explicit_library(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "lib.so"
            library.touch()
            self.module.validate_host_sync_config(
                {"host_sync_backend": "mte", "expert_offload": True, "host_sync_library": str(library)}
            )

    def dispatcher(self):
        obj = self.module.MteHostDispatcher.__new__(self.module.MteHostDispatcher)
        obj.closed, obj.next_slot = False, 1
        obj.entries = [None] * 4
        obj.SLOT_CAPACITY = 4
        obj.alias, obj.stride, obj.done_offset, obj.state_base = 10000, 1024, 512, 20000
        obj.publish, obj.wait = Mock(), Mock()
        obj.session = Mock()
        obj.manager, obj.completed = SimpleNamespace(_cb_failures=0), 0
        obj.active_callback = None
        obj.load_checkpoint = None
        return obj

    def test_capture_slots_and_submission_order(self):
        obj, calls = self.dispatcher(), []

        class Stream:
            @property
            def npu_stream(self):
                calls.append("drain")
                return 55

        obj.publish.side_effect = lambda *args: calls.append("publish")
        obj.wait.side_effect = lambda *args: calls.append("wait")
        obj.session.set_poll_slot_count.side_effect = lambda count: calls.append(f"enable:{count}")
        callback = Mock()
        for value in ("prefetch", "demand"):
            obj.launch(Stream(), callback, value)
        self.assertEqual(calls, ["enable:2", "drain", "publish", "wait", "enable:3", "drain", "publish", "wait"])
        self.assertEqual(obj.entries[1:3], [(callback, "prefetch"), (callback, "demand")])
        obj.publish.assert_called_with(55, 12048, 20064)
        obj.wait.assert_called_with(55, 12560, 20064)
        with patch("builtins.print"):
            obj._run(1)
            obj._run(2)
        self.assertEqual([call.args for call in callback.call_args_list], [("prefetch",), ("demand",)])
        self.assertEqual(obj.completed, 2)

    def test_capacity_and_closed(self):
        for closed, next_slot in ((True, 1), (False, 4)):
            obj = self.dispatcher()
            obj.closed, obj.next_slot = closed, next_slot
            with self.assertRaises(RuntimeError):
                obj.launch(None, None, None)
            obj.publish.assert_not_called()

    def test_guarded_failure_is_fatal(self):
        obj = self.dispatcher()
        obj.entries[1] = (lambda args: setattr(obj.manager, "_cb_failures", 1), None)
        obj._fatal = Mock(side_effect=RuntimeError("fatal"))
        with self.assertRaisesRegex(RuntimeError, "fatal"):
            obj._run(1)
        obj._fatal.assert_called_once()
        self.assertEqual(obj.completed, 0)

    def test_callback_exception_is_fatal(self):
        obj = self.dispatcher()
        obj.entries[1] = (Mock(side_effect=ValueError("copy failed")), None)
        obj._fatal = Mock(side_effect=RuntimeError("fatal"))
        with self.assertRaisesRegex(RuntimeError, "fatal"):
            obj._run(1)
        self.assertIn("copy failed", obj._fatal.call_args.args[0])
        self.assertEqual(obj.completed, 0)

    def test_watchdog_detects_unfinished_generation(self):
        obj = self.dispatcher()
        obj.next_slot = 2
        obj.stopping = Mock()
        obj.stopping.wait.side_effect = [False, False, True]
        obj.session = Mock()
        obj.session.snapshot.return_value = {"request_generation": 1, "done_generation": 0, "error_generation": 0}
        obj._fatal = Mock()
        with patch.object(self.module.time, "monotonic", side_effect=[0, 121]):
            obj._watch()
        obj._fatal.assert_called_once()
        self.assertIn("exceeded", obj._fatal.call_args.args[0])

    def test_stall_diagnostics_once_before_fatal_timeout(self):
        obj = self.dispatcher()
        obj.next_slot = 2
        obj.stopping = Mock()
        obj.stopping.wait.side_effect = [False, False, False, True]
        obj.session = Mock()
        obj.session.snapshot.return_value = {"request_generation": 1, "done_generation": 0, "error_generation": 0}
        obj._dump_stall, obj._fatal = Mock(), Mock()
        with patch.object(self.module.time, "monotonic", side_effect=[0, 11, 12]):
            obj._watch()
        obj._dump_stall.assert_called_once_with(1, obj.session.snapshot.return_value)
        obj._fatal.assert_not_called()

    def test_diagnostic_includes_worker_stack_without_device_queries(self):
        obj = self.dispatcher()
        obj.entries[1] = (lambda args: None, (None, None, None, 7))
        obj.next_slot = 2
        obj.session = Mock()
        with patch("builtins.print") as output:
            obj._dump_stall(1, {"request_generation": 1})
        text = output.call_args.args[0]
        self.assertIn("DIAGNOSTIC BEGIN", text)
        self.assertIn("layer=7", text)
        self.assertIn("Python thread=", text)
        obj.publish.assert_not_called()
        obj.wait.assert_not_called()

    def test_error_summary_keeps_diagnostic_before_shutdown_tail(self):
        block = "[MOE-MTE DIAGNOSTIC BEGIN]\nworker stack\n[MOE-MTE DIAGNOSTIC END]\n"
        result = self.runner.failure_excerpt(block + "shutdown noise\n" * 100)
        self.assertIn("worker stack", result)
        self.assertEqual(result.count("shutdown noise"), 70)

    def test_error_summary_keeps_engine_startup_root_before_statistics(self):
        text = (
            "(EngineCore pid=1) Traceback (most recent call last):\n"
            "(EngineCore pid=1) RuntimeError: invalid tensor layout\n"
            + "final statistics\n" * 180
            + "(APIServer) engine initialization failed\n"
        )
        self.assertIn("invalid tensor layout", self.runner.failure_excerpt(text))

    def test_read_existing_log_does_not_launch_or_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("RuntimeError: saved failure", encoding="utf-8")
            with (
                patch.object(sys, "argv", ["run_moe_host_sync.py", "--read-log", str(path)]),
                patch.object(self.runner, "run") as run,
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                self.assertEqual(self.runner.main(), 0)
                self.assertIn("saved failure", output.getvalue())
                run.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_close_synchronizes_before_releasing_lease(self):
        obj = self.dispatcher()
        events = Mock()
        obj.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=events.synchronize))
        obj.stopping, obj.watchdog, obj.session = events.stopping, events.watchdog, events.session
        with patch.object(self.module.atexit, "unregister"):
            obj.close()
            obj.close()
        self.assertEqual(
            [str(call) for call in events.mock_calls],
            [
                "call.synchronize()",
                "call.stopping.set()",
                "call.watchdog.join()",
                "call.session.end_device_access(stream_synchronized=True)",
                "call.session.stop()",
                "call.session.close()",
            ],
        )

    def test_original_and_optional_dispatch(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_launch_offload_callback"
        )
        original = Mock()
        namespace = {"torch_npu": SimpleNamespace(npu=SimpleNamespace(_launch_host_func=original))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dispatch", "exec"), namespace)
        launch, manager = namespace["_launch_offload_callback"], SimpleNamespace(offload_config=SimpleNamespace())
        launch(manager, "stream", "fn", "args")
        original.assert_called_once_with("stream", "fn", "args")
        manager._host_dispatcher = Mock()
        launch(manager, "stream", "fn", "args")
        manager._host_dispatcher.launch.assert_called_once_with("stream", "fn", "args")
        original.assert_called_once()

    def test_runner_preserves_model_settings(self):
        source = (self.root / "script/test_graph2.sh").read_text(encoding="utf-8")
        _, before, _, before_config = self.runner.launch_spec(source, "python", 1, None, 18155)
        updated = self.runner.configured_source(source, "mte", "/tmp/libhost_sync_device.so")
        _, after, _, after_config = self.runner.launch_spec(updated, "python", 1, None, 18155)
        self.assertEqual(before_config, after_config)
        self.assertEqual(before[2], after[2])
        old = json.loads(before[before.index("--additional-config") + 1])["expert_offload_config"]
        new = json.loads(after[after.index("--additional-config") + 1])["expert_offload_config"]
        self.assertEqual(new.pop("host_sync_backend"), "mte")
        self.assertEqual(new.pop("host_sync_library"), "/tmp/libhost_sync_device.so")
        self.assertFalse(new.pop("host_sync_diagnose_load"))
        self.assertEqual(new.pop("host_sync_quant_copy"), "storage")
        self.assertFalse(new.pop("host_sync_prefetch_callback"))
        self.assertEqual(new, old)
        self.assertFalse(any(not word.strip() for word in after))

    def test_prefetch_ablation_keeps_demand_mte_and_subscribes_once(self):
        tree = ast.parse(
            (self.root / "vllm_ascend/expert_offload/expert_offload_manager.py").read_text(encoding="utf-8")
        )
        method = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_launch_offload_callback"
        )
        npu, subscribed = SimpleNamespace(_launch_host_func=Mock(), _subscribe_report=Mock()), set()
        namespace = {"torch_npu": SimpleNamespace(npu=npu), "get_subscribed_compute_streams": lambda: subscribed}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dispatch", "exec"), namespace)
        manager = SimpleNamespace(
            _host_dispatcher=Mock(), offload_config=SimpleNamespace(host_sync_prefetch_callback=True)
        )
        launch = namespace["_launch_offload_callback"]
        prefetch = (None, None, None, 1, None, True)
        demand = (None, None, None, 1, None, False, None, None, None)
        launch(manager, "prefetch_stream", "fn", prefetch)
        launch(manager, "prefetch_stream", "fn", prefetch)
        launch(manager, "main_stream", "fn", demand)
        npu._subscribe_report.assert_called_once_with("prefetch_stream")
        self.assertEqual(npu._launch_host_func.call_count, 2)
        manager._host_dispatcher.launch.assert_called_once_with("main_stream", "fn", demand)

    def test_prefetch_ablation_rejects_invalid_configuration(self):
        for config in ({"host_sync_prefetch_callback": "true"}, {"host_sync_prefetch_callback": True}):
            with self.assertRaises(ValueError):
                self.module.validate_host_sync_config(config)

    def test_report_thread_load_stall_emits_native_marker(self):
        obj = self.dispatcher()
        obj.next_slot = 1
        obj.stopping = Mock()
        obj.stopping.wait.side_effect = [False, True]
        obj.load_checkpoint = (0, "scale_copy", 12345, 0)
        obj._fatal = Mock()
        with patch.object(self.module.time, "monotonic", return_value=11), patch("builtins.print") as output:
            obj._watch()
        text = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("worker_tid=12345", text)
        self.assertIn("scale_copy", text)
        obj._fatal.assert_not_called()

    def test_runner_tensor_option_only_affects_mte(self):
        source = (self.root / "script/test_graph2.sh").read_text(encoding="utf-8")
        for backend, expected in (("callback", "storage"), ("mte", "tensor")):
            updated = self.runner.configured_source(source, backend, "/tmp/lib.so", quant_copy="tensor")
            _, command, _, _ = self.runner.launch_spec(updated, "python", 1, None, 18155)
            config = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
            self.assertEqual(config["host_sync_quant_copy"], expected)

    @staticmethod
    def weight_comparison_reports():
        reports = {}
        for backend, means in (
            ("callback", (20, 30, 5, 60, 100)),
            ("mte", (10, 20, 5, 40, 80)),
        ):
            reports[backend] = {
                "weight_timing": {
                    "digests": ["same-output"],
                    "usage": [{"prompt_tokens": 5, "completion_tokens": 5}],
                    "groups": [
                        {
                            "kind": "reactive",
                            "token_rows": 1,
                            "count": 3,
                            "loads": {"mean": 2},
                            "metrics": {
                                name: {"mean": mean}
                                for name, mean in zip(
                                    ("load_prepare", "copy_submit", "scale_refresh", "load_submit", "synchronize"),
                                    means,
                                )
                            },
                        }
                    ],
                }
            }
        return reports

    def test_weight_comparison_reports_stage_delta_without_double_counting(self):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.runner.compare_weight_timings(self.weight_comparison_reports())
        output = captured.getvalue()
        self.assertIn("count/load match=True", output)
        self.assertIn("total_loaded_experts=6/6", output)
        rows = {
            words[0]: words[1:]
            for line in output.splitlines()
            if len(words := line.split()) == 5 and words[0] in ("load_prepare", "load_and_wait")
        }
        self.assertEqual(rows["load_prepare"], ["20.000", "10.000", "-10.000", "-50.0%"])
        self.assertEqual(rows["load_and_wait"], ["160.000", "120.000", "-40.000", "-25.0%"])
        self.assertIn("do not add load_prepare/copy_submit again", output)

    def test_weight_comparison_flags_different_counts_loads_or_missing_group(self):
        for count, loads in ((4, 1.5), (3, 3)):
            with self.subTest(count=count, loads=loads):
                reports = self.weight_comparison_reports()
                row = reports["mte"]["weight_timing"]["groups"][0]
                row["count"], row["loads"]["mean"] = count, loads
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.runner.compare_weight_timings(reports)
                self.assertIn("count/load match=False", captured.getvalue())
                self.assertIn("do not establish a speedup", captured.getvalue())
        reports = self.weight_comparison_reports()
        reports["mte"]["weight_timing"]["groups"] = []
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.runner.compare_weight_timings(reports)
        self.assertIn("missing in one backend; not comparable", captured.getvalue())
        self.assertNotIn("delta_us", captured.getvalue())

    def test_weight_comparison_rejects_different_output_or_usage(self):
        for key, value in (("digests", ["different-output"]), ("usage", [{"completion_tokens": 4}])):
            with self.subTest(key=key):
                reports = self.weight_comparison_reports()
                reports["mte"]["weight_timing"][key] = value
                with self.assertRaisesRegex(RuntimeError, "different outputs/usage"):
                    self.runner.compare_weight_timings(reports)

    def test_prefill_comparison_keeps_full_expert_loads_separate(self):
        reports = self.weight_comparison_reports()
        for backend, total in (("callback", 100000), ("mte", 150000)):
            reports[backend]["weight_timing"]["prefill_load"] = dict(count=2, sum_us=total)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.runner.compare_weight_timings(reports)
        self.assertIn("PREFILL ms/layer: callback=50.000; mte=75.000; delta=+25.000", captured.getvalue())
        self.assertIn("separate from Decode below", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
