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

    def test_vector_limit_is_diagnostic_only_and_binds_after_set(self):
        for value in (True, -1, 57):
            with self.assertRaises(ValueError):
                self.module.validate_host_sync_config({"host_sync_diagnose_load_vector_cores": value})
        with self.assertRaises(ValueError):
            self.module.validate_host_sync_config({"host_sync_diagnose_load_vector_cores": 28})
        library = SimpleNamespace(
            aclrtSetStreamResLimit=Mock(return_value=0), aclrtUseStreamResInCurrentThread=Mock(return_value=0)
        )
        self.module.apply_load_vector_limit(library, 123, 28)
        library.aclrtSetStreamResLimit.assert_called_once_with(123, 1, 28)
        library.aclrtUseStreamResInCurrentThread.assert_called_once_with(123)
        library.aclrtSetStreamResLimit.return_value = 1
        library.aclrtUseStreamResInCurrentThread.reset_mock()
        with self.assertRaises(RuntimeError):
            self.module.apply_load_vector_limit(library, 123, 28)
        library.aclrtUseStreamResInCurrentThread.assert_not_called()

    def test_bounded_wait_cannot_be_enabled_without_completion_guard(self):
        with self.assertRaisesRegex(ValueError, "completion guard"):
            self.module.validate_host_sync_config({"host_sync_diagnose_prefetch_poll_budget": 1000})
        for value in (-1, True, 10_000_001):
            with self.assertRaises(ValueError):
                self.module.validate_host_sync_config({"host_sync_diagnose_prefetch_poll_budget": value})

    def test_profile_parser_failure_overrides_export_success(self):
        report = {"profile_status": "complete", "request_us": 123, "mstx_trace_occurrences": 0}
        log = "profiler.py: Task [CANNTimelineParser] run failed.\nEngineDeadError"
        self.runner.assess_profile_report(report, log)
        self.assertEqual(report["profile_status"], "invalid")
        self.assertEqual(report["request_us"], 123)
        self.assertTrue(report["engine_error_seen"])
        self.assertEqual(len(report["profile_errors"]), 1)

    def test_missing_mstx_is_unverified_not_success(self):
        report = {"profile_status": "complete", "mstx_trace_occurrences": 0}
        self.runner.assess_profile_report(report, "")
        self.assertEqual(report["profile_status"], "unverified")
        report = {"profile_status": "timeout", "mstx_trace_occurrences": 0}
        self.runner.assess_profile_report(report, "")
        self.assertEqual(report["profile_status"], "timeout")

    def test_acl_error_detail_handles_null_and_missing_api(self):
        for raw, expected in ((b"task detail", "task detail"), (None, "<no recent ACL error>")):
            library = SimpleNamespace(aclGetRecentErrMsg=Mock(return_value=raw))
            self.assertEqual(self.module.recent_acl_error(library), expected)
        self.assertIn("unavailable", self.module.recent_acl_error(SimpleNamespace()))

    def test_runtime_stream_id_is_distinct_from_pointer(self):
        def query(pointer, output):
            self.assertEqual(pointer, 12345)
            output._obj.value = 27
            return 0

        library = SimpleNamespace(aclrtStreamGetId=Mock(side_effect=query))
        self.assertEqual(
            self.module.runtime_stream_identity(library, 12345), {"pointer": 12345, "code": 0, "stream_id": 27}
        )
        self.assertIn("query_error", self.module.runtime_stream_identity(SimpleNamespace(), 12345))

    def test_last_task_query_handles_success_failure_and_missing_symbol(self):
        def query(output):
            output._obj.value = 773
            return 0

        library = SimpleNamespace(aclrtGetThreadLastTaskId=Mock(side_effect=query))
        result = self.module.runtime_last_task(library)
        self.assertEqual(result["task_id"], 773)
        self.assertGreater(result["host_tid"], 0)
        library.aclrtGetThreadLastTaskId = Mock(return_value=1)
        self.assertIsNone(self.module.runtime_last_task(library)["task_id"])
        self.assertIn("query_error", self.module.runtime_last_task(SimpleNamespace()))

    def test_stall_lists_pending_slots_without_device_access(self):
        obj = self.dispatcher()
        obj.next_slot = 3
        obj.entries[1] = obj.entries[2] = (lambda args: None, (None, None, None, 7))
        obj.entry_streams[1:3] = [111, 222]
        obj.session.snapshot.side_effect = lambda slot: {
            "request_generation": 1,
            "done_generation": 1 if slot == 1 else 0,
        }
        with patch("builtins.print") as output:
            obj._dump_stall(2, {})
        text = output.call_args.args[0]
        pending = next(line for line in text.splitlines() if line.startswith("[MOE-MTE PENDING]"))
        records = json.loads(pending.split("] ", 1)[1])
        self.assertEqual([record["slot"] for record in records], [2])
        self.assertEqual(records[0]["stream_pointer"], 222)
        obj.publish.assert_not_called()
        obj.wait.assert_not_called()

    def test_context_probes_stop_on_first_failed_checkpoint(self):
        obj = self.dispatcher()
        contiguous, strided = Mock(), Mock()
        task = SimpleNamespace(name="scale", destination=Mock())
        task.destination.stride.return_value = strided.stride.return_value = (2, 128, 1)
        obj.torch = SimpleNamespace(
            empty_like=Mock(side_effect=[contiguous, strided]),
            contiguous_format="contiguous",
            preserve_format="preserve",
        )
        checkpoint = Mock(side_effect=RuntimeError("wait failed"))
        with (
            patch.object(self.module, "describe_npu_tensor", return_value={}),
            patch("builtins.print"),
            self.assertRaisesRegex(RuntimeError, "wait failed"),
        ):
            obj._diagnose_scale_targets(task, Mock(), "scale", checkpoint)
        self.assertEqual(obj.torch.empty_like.call_count, 1)
        task.destination.copy_.assert_not_called()
        strided.copy_.assert_not_called()

    def test_context_probes_preserve_original_target(self):
        obj = self.dispatcher()
        contiguous, strided, temporary = Mock(), Mock(), Mock()
        task = SimpleNamespace(name="scale", destination=Mock())
        task.destination.stride.return_value = strided.stride.return_value = (2, 128, 1)
        obj.torch = SimpleNamespace(
            empty_like=Mock(side_effect=[contiguous, strided]),
            contiguous_format="contiguous",
            preserve_format="preserve",
        )
        labels = []

        def checkpoint(label, operation):
            labels.append(label)
            operation()

        with patch.object(self.module, "describe_npu_tensor", return_value={}), patch("builtins.print"):
            obj._diagnose_scale_targets(task, temporary, "scale", checkpoint)
        self.assertEqual(labels, ["scale D2D_CONTIGUOUS", "scale D2D_INDEPENDENT_STRIDED"])
        contiguous.copy_.assert_called_once_with(temporary, non_blocking=True)
        strided.copy_.assert_called_once_with(temporary, non_blocking=True)
        task.destination.copy_.assert_not_called()

    def test_load_diagnosis_disabled_does_not_touch_stream(self):
        obj = self.dispatcher()
        obj.manager.offload_config = SimpleNamespace()
        self.assertFalse(obj.diagnose_load([object()], [(0, 1)], None, 1))

    def test_native_wait_split_and_error_stops_progress(self):
        for fail in (False, True):
            obj = self.dispatcher()
            obj.manager.offload_config = SimpleNamespace(
                host_sync_diagnose_load=True, host_sync_diagnose_native_wait=True
            )
            obj.active_callback, obj.load_diagnostic_batches = (1, 123), 0
            obj.entry_streams[1] = 55
            stream = Mock(npu_stream=77)
            obj.manager.load_stream = stream
            transport = Mock()
            obj.manager._get_h2d_transport = lambda transport=transport: transport
            obj.manager._refresh_expert_fp32_scale = Mock()
            obj._dump_stall = Mock()
            obj._fatal = Mock(side_effect=RuntimeError("terminated"))
            native = Mock(side_effect=[0, 123] if fail else [0, 0, 0])
            lib = SimpleNamespace(aclrtSynchronizeStreamWithTimeout=native)
            npu = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(npu_stream=77)))
            task = SimpleNamespace(name="scale", nbytes=64, non_blocking=True, source=object(), destination=object())
            with (
                patch.dict(sys.modules, torch_npu=npu),
                patch.object(self.module.ctypes, "CDLL", return_value=lib),
                patch("builtins.print"),
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "terminated"):
                        obj.diagnose_load([task], [(0, 1)], None, 1)
                    self.assertEqual(stream.synchronize.call_count, 1)
                    obj.manager._refresh_expert_fp32_scale.assert_not_called()
                else:
                    self.assertTrue(obj.diagnose_load([task], [(0, 1)], None, 1))
                    self.assertEqual(stream.synchronize.call_count, 3)
                native.assert_called_with(77, 10000)

    def test_copy_description_reads_metadata_not_values(self):
        value = SimpleNamespace(
            shape=(2, 3),
            dtype="uint8",
            device="cpu",
            stride=lambda: (1, 2),
            storage_offset=lambda: 4,
            is_contiguous=lambda: False,
            data_ptr=lambda: 100,
            is_pinned=lambda: True,
            cpu=Mock(side_effect=AssertionError),
            tolist=Mock(side_effect=AssertionError),
        )
        data = self.module.describe_copy_object(value)
        self.assertEqual(data["stride"], (1, 2))
        self.assertEqual(data["shape"], [2, 3])
        self.assertTrue(data["is_pinned"])
        value.cpu.assert_not_called()
        value.tolist.assert_not_called()

    def test_split_scale_stops_before_destination_on_h2d_failure(self):
        for fail in (False, True):
            obj = self.dispatcher()
            obj.manager.offload_config = SimpleNamespace()
            temporary = Mock()
            obj.torch = SimpleNamespace(empty_like=Mock(return_value=temporary), contiguous_format="contiguous")
            task = SimpleNamespace(source=object(), destination=Mock(), non_blocking=True)
            labels = []

            def checkpoint(label, operation, labels=labels, fail=fail):
                labels.append(label)
                operation()
                if fail:
                    raise RuntimeError("native wait failed")

            with patch("builtins.print"):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "native wait failed"):
                        obj._diagnose_split_scale(task, "scale", checkpoint)
                    task.destination.copy_.assert_not_called()
                else:
                    obj._diagnose_split_scale(task, "scale", checkpoint)
                    task.destination.copy_.assert_called_once_with(temporary, non_blocking=True)
                    self.assertEqual(labels, ["scale H2D_CONTIGUOUS", "scale D2D_TARGET_LAYOUT"])
            temporary.copy_.assert_called_once_with(task.source, non_blocking=True)
            obj.torch.empty_like.assert_called_once_with(task.destination, memory_format="contiguous")

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
        obj.entry_streams = [None] * 4
        obj.SLOT_CAPACITY = 4
        obj.alias, obj.stride, obj.done_offset, obj.state_base = 10000, 1024, 512, 20000
        obj.publish, obj.wait = Mock(), Mock()
        obj.manager, obj.completed = SimpleNamespace(_cb_failures=0, offload_config=SimpleNamespace()), 0
        obj.report_wait_streams = set()
        obj.report_wait_samples = 0
        obj.warmup_wait = Mock()
        obj.wait_states_h = {1: Mock(), 2: Mock()}
        obj.states = {1: object(), 2: object()}
        obj.wait_exit_counts = {"completed": 0, "budget": 0}
        obj.active_callback = None
        obj.allocated_scan = False
        obj.session = Mock()
        return obj

    def test_report_wait_control_changes_prefetch_wait_only(self):
        obj = self.dispatcher()
        prefetch, reactive = SimpleNamespace(npu_stream=55), SimpleNamespace(npu_stream=66)
        obj.manager._prefetch_stream = prefetch
        obj.manager.offload_config.host_sync_diagnose_prefetch_report_wait = True
        npu = SimpleNamespace(_subscribe_report=Mock(), _launch_host_func=Mock())
        callback = Mock()
        with patch.dict(sys.modules, torch_npu=SimpleNamespace(npu=npu)), patch("builtins.print"):
            obj.launch(prefetch, callback, "prefetch args")
            obj.launch(reactive, callback, "reactive args")
        self.assertEqual(obj.publish.call_count, 2)
        obj.wait.assert_called_once_with(
            66, obj.alias + 2 * obj.stride + obj.done_offset, obj.state_base + 2 * obj.STATE_BYTES
        )
        npu._subscribe_report.assert_called_once_with(prefetch)
        npu._launch_host_func.assert_called_once_with(prefetch, obj._report_wait_completion, 1)
        self.assertEqual(obj.entries[1], (callback, "prefetch args"))
        callback.assert_not_called()

    def test_report_wait_requires_real_worker_completion(self):
        obj = self.dispatcher()
        obj.session.snapshot.return_value = {"request_generation": 3}
        obj.session.wait.return_value = {"done_generation": 3}
        obj._fatal = Mock(side_effect=RuntimeError("terminated"))
        obj._report_wait_completion(2)
        obj.session.wait.assert_called_once_with(2, 3, timeout_ms=30000)
        obj._fatal.assert_not_called()
        for result in ({"done_generation": 2}, TimeoutError("pending")):
            obj.session.wait.side_effect = result if isinstance(result, Exception) else None
            obj.session.wait.return_value = result
            with self.assertRaisesRegex(RuntimeError, "terminated"):
                obj._report_wait_completion(2)
        obj.publish.assert_not_called()
        obj.wait.assert_not_called()

    def test_bounded_prefetch_always_has_following_report_guard(self):
        obj = self.dispatcher()
        prefetch = SimpleNamespace(npu_stream=55)
        obj.manager._prefetch_stream = prefetch
        obj.manager.offload_config = SimpleNamespace(
            host_sync_diagnose_prefetch_report_wait=True, host_sync_diagnose_prefetch_poll_budget=1000000
        )
        events = Mock()
        obj.publish, obj.warmup_wait = events.publish, events.bounded_wait
        obj.wait_states_h[1] = SimpleNamespace(copy_=events.exit_record_copy)
        npu = SimpleNamespace(_subscribe_report=events.subscribe, _launch_host_func=events.report_guard)
        with patch.dict(sys.modules, torch_npu=SimpleNamespace(npu=npu)), patch("builtins.print"):
            obj.launch(prefetch, Mock(), ())
        names = [entry[0] for entry in events.mock_calls]
        self.assertEqual(names, ["publish", "bounded_wait", "exit_record_copy", "subscribe", "report_guard"])
        events.exit_record_copy.assert_called_once_with(obj.states[1], non_blocking=True)
        self.assertEqual(events.bounded_wait.call_args.args[-1], 1000000)
        obj.wait.assert_not_called()

    def test_bounded_wait_reports_pending_then_matching_completion(self):
        obj = self.dispatcher()
        obj.manager.offload_config.host_sync_diagnose_prefetch_poll_budget = 1000
        obj.session.snapshot.return_value = {"request_generation": 1, "done_generation": 0}
        obj.session.wait.return_value = {"done_generation": 1}
        obj.wait_states_h[2].tolist.return_value = [1, 3, 0, 1000]
        with patch("builtins.print") as output:
            obj._report_wait_completion(2)
        text = "\n".join(entry.args[0] for entry in output.call_args_list)
        self.assertIn('"done_at_report_entry": 0', text)
        self.assertIn('"done": 1', text)
        obj.session.wait.assert_called_once_with(2, 1, timeout_ms=30000)

    def test_bad_device_exit_record_never_releases_report_guard(self):
        obj = self.dispatcher()
        obj.manager.offload_config.host_sync_diagnose_prefetch_poll_budget = 1000
        obj.session.snapshot.return_value = {"request_generation": 1, "done_generation": 0}
        obj.wait_states_h[2].tolist.return_value = [2, 3, 0, 1000]
        obj._fatal = Mock(side_effect=RuntimeError("terminated"))
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            obj._report_wait_completion(2)
        obj.session.wait.assert_not_called()

    def test_allocated_prefix_published_after_entry_before_device(self):
        obj = self.dispatcher()
        obj.allocated_scan = True
        calls = []
        callback = Mock()

        def expand(count):
            self.assertEqual(obj.entries[count - 1], (callback, count - 1))
            calls.append(("expand", count))

        obj.session.expand_scan_prefix.side_effect = expand
        obj.publish.side_effect = lambda *args: calls.append(("publish",))
        obj.wait.side_effect = lambda *args: calls.append(("wait",))
        for slot in (1, 2):
            obj.launch(SimpleNamespace(npu_stream=55), callback, slot)
        self.assertEqual(calls, [("expand", 2), ("publish",), ("wait",), ("expand", 3), ("publish",), ("wait",)])

    def test_worker_policy_defaults_and_opt_in(self):
        session = Mock()
        self.assertFalse(self.module.configure_mte_worker(session, SimpleNamespace()))
        self.assertEqual(session.mock_calls, [])
        config = SimpleNamespace(
            host_sync_scan_mode="allocated", host_sync_idle_probe="atomic", host_sync_idle_mode="spin"
        )
        self.assertTrue(self.module.configure_mte_worker(session, config))
        session.configure_scan_prefix.assert_called_once_with(1)
        session.configure_idle_probe.assert_called_once_with("atomic")
        session.configure_polling.assert_called_once_with("spin", False)
        with self.assertRaisesRegex(RuntimeError, "Rebuild"):
            self.module.configure_mte_worker(object(), config)
        for key in ("host_sync_scan_mode", "host_sync_idle_probe", "host_sync_idle_mode"):
            with self.assertRaises(ValueError):
                self.module.validate_host_sync_config({key: "invalid"})

    def test_prefix_failure_prevents_device_publication(self):
        obj = self.dispatcher()
        obj.allocated_scan = True
        obj.session.expand_scan_prefix.side_effect = RuntimeError("failed")
        with self.assertRaisesRegex(RuntimeError, "failed"):
            obj.launch(SimpleNamespace(npu_stream=55), Mock(), None)
        obj.publish.assert_not_called()
        obj.wait.assert_not_called()
        self.assertEqual(obj.next_slot, 1)

    def test_runtime_artifacts_include_package_and_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            package = source / "vllm_ascend"
            binary = package / "_cann_ops_custom/vendors/custom_transformer/kernel.bin"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"device binary")
            (package / "_build_info.py").write_text("build_info = 1")
            (package / "_C.so").write_bytes(b"extension")
            (package / "host_sync.py").write_text("must not copy source")
            target.mkdir()
            with patch("builtins.print"):
                self.runner.prepare_runtime_artifacts(source, target)
            self.assertEqual((target / binary.relative_to(source)).read_bytes(), b"device binary")
            self.assertFalse((target / "vllm_ascend/host_sync.py").exists())
            (target / "vllm_ascend/_build_info.py").write_text("keep")
            with patch("builtins.print"):
                self.runner.prepare_runtime_artifacts(source, target)
            self.assertEqual((target / "vllm_ascend/_build_info.py").read_text(), "keep")

    def test_runner_worker_switches_only_apply_to_mte(self):
        source = (self.root / "script/test_graph2.sh").read_text(encoding="utf-8")
        for backend in ("mte", "callback"):
            updated = self.runner.configured_source(
                source, backend, "/tmp/lib.so", False, "allocated", "atomic", "spin"
            )
            _, command, _, _ = self.runner.launch_spec(updated, "python", 1, None, 18155)
            config = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
            expected = ("allocated", "atomic", "spin") if backend == "mte" else ("all", "locked", "yield")
            self.assertEqual(
                tuple(config[key] for key in ("host_sync_scan_mode", "host_sync_idle_probe", "host_sync_idle_mode")),
                expected,
            )

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
        block = "[MOE-MTE DIAGNOSTIC BEGIN]\nworker stack\n[MOE-MTE FATAL] failed\n"
        result = self.runner.failure_excerpt(block + "shutdown noise\n" * 100)
        self.assertIn("DIAGNOSTIC BEGIN", result)
        self.assertIn("FATAL", result)
        self.assertNotIn("shutdown noise", result)

    def test_error_summary_bounds_successful_load_output(self):
        block = "[MOE-MTE LOAD] success COMPLETE\n" * 1000
        block += "[MOE-MTE LOAD] D2D_TARGET_LAYOUT ACL_WAIT_RETURN code=507046\n[MOE-MTE FATAL] failed"
        result = self.runner.failure_excerpt(block)
        self.assertLessEqual(len(result.splitlines()), 20)
        self.assertIn("D2D_TARGET_LAYOUT", result)

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
        self.assertEqual(new.pop("host_sync_scan_mode"), "all")
        self.assertEqual(new.pop("host_sync_idle_probe"), "locked")
        self.assertEqual(new.pop("host_sync_idle_mode"), "yield")
        self.assertEqual(new, old)
        self.assertFalse(any(not word.strip() for word in after))


if __name__ == "__main__":
    unittest.main()
