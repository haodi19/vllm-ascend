# SPDX-License-Identifier: Apache-2.0
"""CPU checks for optional MoE dispatch without NPU imports."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class MoeHostSyncTests(unittest.TestCase):
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
        obj.manager, obj.completed = SimpleNamespace(_cb_failures=0), 0
        obj.active_callback = None
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
        callback = Mock()
        for value in ("prefetch", "demand"):
            obj.launch(Stream(), callback, value)
        self.assertEqual(calls, ["drain", "publish", "wait"] * 2)
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
        launch, manager = namespace["_launch_offload_callback"], SimpleNamespace()
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
        self.assertEqual(new, old)
        self.assertFalse(any(not word.strip() for word in after))


if __name__ == "__main__":
    unittest.main()
