# SPDX-License-Identifier: Apache-2.0
"""CPU tests for capture-evidence checks and the single-run diagnostic wrapper."""

import contextlib
import importlib.util
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class WaitPlacementRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "wait_runner_test", cls.root / "script/diagnose_mte_wait_placement.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
            runner_spec = importlib.util.spec_from_file_location(
                "moe_runner_placement_test", cls.root / "script/run_moe_host_sync.py"
            )
            cls.runner = importlib.util.module_from_spec(runner_spec)
            runner_spec.loader.exec_module(cls.runner)
            mapping_spec = importlib.util.spec_from_file_location(
                "mapping_runner_test", cls.root / "script/diagnose_mte_graph_mapping.py"
            )
            cls.mapping = importlib.util.module_from_spec(mapping_spec)
            mapping_spec.loader.exec_module(cls.mapping)
            order_spec = importlib.util.spec_from_file_location(
                "wait_order_test", cls.root / "script/diagnose_mte_wait_order.py"
            )
            cls.order = importlib.util.module_from_spec(order_spec)
            order_spec.loader.exec_module(cls.order)

    @staticmethod
    def node(stream, task, name, kind="KERNEL_AIVEC"):
        return {"name": name, "args": {"Model Id": 63, "Stream Id": stream, "Task Id": task, "Task Type": kind}}

    def graph(self):
        return [
            self.node(21, 5, "host_sync_publish"),
            self.node(21, 6, "EVENT_WAIT_40", "EVENT_WAIT"),
            self.node(21, 7, "host_sync_wait"),
            self.node(21, 8, "MEMCPY_ASYNC", "MEMCPY_ASYNC"),
            self.node(21, 9, "EVENT_RECORD_41", "EVENT_RECORD"),
            self.node(23, 195, "KvCompressEpilog"),
            self.node(23, 200, "EVENT_RECORD_40", "EVENT_RECORD"),
            self.node(23, 201, "EVENT_WAIT_41", "EVENT_WAIT"),
        ]

    def test_unique_record_before_wait_protects_mapping_and_join(self):
        pairs = self.module.gated_wait_pairs(self.graph())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["strict_wait"], (63, 21, 7))
        self.assertEqual(pairs[0]["mapping_tasks"], [(63, 21, 8)])

    def test_early_wait_graph_requires_mapping_to_stay_after_gate(self):
        graph = self.graph()
        graph[1]["args"]["Task Id"], graph[2]["args"]["Task Id"] = 7, 6
        self.assertEqual(self.module.gated_wait_pairs(graph), [])
        pairs = self.module.gated_wait_pairs(graph, wait_before_gate=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["strict_wait"], (63, 21, 6))
        self.assertEqual(pairs[0]["gate_wait"], (63, 21, 7))
        graph[1]["args"]["Task Id"], graph[3]["args"]["Task Id"] = 8, 7
        self.assertEqual(self.module.gated_wait_pairs(graph, wait_before_gate=True), [])

    def test_wait_order_facts_use_matching_capture_and_only_active_worker_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mte/graph_debug/capture/graph_dump"
            path.parent.mkdir(parents=True)
            graph = self.graph()
            graph[1]["args"]["Task Id"], graph[2]["args"]["Task Id"] = 7, 6
            path.write_text(json.dumps({"traceEvents": graph}), encoding="utf-8")
            ledger = dict(
                phase="capture_end", published_slots=[2], joined_slots=[2], pending_waits=0, pending_layers=[]
            )
            pending = [dict(slot=2, state=dict(request_generation=1, done_generation=0))]
            prefix = "[MOE-MTE WAIT_PLACEMENT] " + json.dumps(ledger) + "\n[MOE-MTE DIAGNOSTIC BEGIN]\n"
            prefix += "active_callback=(2, 123); completed=1\n[MOE-MTE PENDING] " + json.dumps(pending) + "\n"
            service = root / "mte/inference_python.log"
            for worker_sync in (False, True):
                service.write_text(
                    prefix
                    + "(EngineCore pid=7) Python thread=123; active_worker=True\n"
                    + ("_synchronize_h2d\n" if worker_sync else "cache_plan\n")
                    + "(EngineCore pid=7) Python thread=456; active_worker=False\n_synchronize_h2d\n"
                    + "[MOE-MTE DIAGNOSTIC END]\n",
                    encoding="utf-8",
                )
                facts = self.order.case_facts(root, 1, True)
                self.assertTrue(facts["expected_structure"])
                self.assertTrue(facts["active_prefetch_pending"])
                self.assertEqual(facts["worker_in_load_sync"], worker_sync)
                self.assertTrue(facts["pending_snapshot_valid"])
                self.assertEqual(facts["active_prefetch_snapshot"]["slot"], 2)
                self.assertFalse(facts["watchdog_timeout_for_active_prefetch"])
                self.assertFalse(self.order.case_facts(root, 1, False)["expected_structure"])
            with service.open("a", encoding="utf-8") as log:
                log.write("[MOE-MTE FATAL] slot=2 generation=1 exceeded 120s\n")
            self.assertTrue(self.order.case_facts(root, 1, True)["watchdog_timeout_for_active_prefetch"])
            ledger["pending_layers"] = [2]
            service.write_text("[MOE-MTE WAIT_PLACEMENT] " + json.dumps(ledger), encoding="utf-8")
            self.assertFalse(self.order.case_facts(root, 0, True)["expected_structure"])
            self.assertIsNone(self.order.case_facts(root, 0, True)["pending_prefetch_count"])

    def test_wait_order_does_not_run_early_case_without_valid_control(self):
        for control_ok, passed, timed_out in (
            (False, False, True),
            (True, False, True),
            (True, False, False),
            (True, True, False),
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                control = dict(request_outcome="finite_requests_passed", expected_structure=control_ok)
                target = dict(
                    request_outcome="finite_requests_passed" if passed else "inconclusive",
                    expected_structure=True,
                    active_prefetch_pending=True,
                    worker_in_load_sync=True,
                    watchdog_timeout_for_active_prefetch=timed_out,
                )
                output = io.StringIO()
                with (
                    patch.object(self.order, "run_once", return_value=0) as run,
                    patch.object(self.order, "append_evidence"),
                    patch.object(self.order, "case_facts", side_effect=[control, target]),
                ):
                    self.order.run_comparison(root, root, 7, output)
                self.assertEqual(run.call_count, 2 if control_ok else 1)
                expected = (
                    "CONTROL_NOT_ESTABLISHED"
                    if not control_ok
                    else "EARLY_WAIT_LATE_MAPPING_PASSED"
                    if passed
                    else "EARLY_WAIT_LATE_MAPPING_PENDING_AT_LOAD_SYNC"
                    if timed_out
                    else "INCONCLUSIVE"
                )
                self.assertIn("BLOCKING_PROGRESS=" + expected, output.getvalue())

    def test_early_wait_or_ambiguous_event_is_not_accepted(self):
        graph = self.graph()
        for changed in (
            [n for n in graph if n["name"] != "EVENT_WAIT_40"],
            [*graph, self.node(24, 300, "EVENT_RECORD_40")],
            [*graph, self.node(24, 301, "EVENT_WAIT_41")],
            [*graph, self.node(24, 300, "EVENT_RECORD_41")],
            [*graph, self.node(24, 301, "EVENT_WAIT_40")],
            [n for n in graph if n["name"] != "MEMCPY_ASYNC"],
        ):
            with self.subTest(changed=changed):
                self.assertEqual(self.module.gated_wait_pairs(changed), [])

    def test_reversed_main_edge_or_duplicate_task_identity_rejected(self):
        graph = self.graph()
        graph[-1]["args"]["Task Id"] = 199
        self.assertEqual(self.module.gated_wait_pairs(graph), [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.module.gated_wait_pairs([*self.graph(), self.graph()[0]])

    def test_request_outcome_is_separate_from_capture_evidence(self):
        self.assertEqual(self.module.request_outcome("RUNNER_EXIT=0\nMOE_RESULT {}", 0), "finite_requests_passed")
        self.assertEqual(self.module.request_outcome("RUNNER_EXIT=0\nMOE_RESULT {}", 1), "inconclusive")
        self.assertEqual(
            self.module.request_outcome("w13_weight_scale COMPLETE\nother ACL_WAIT_RETURN code=507046", 1),
            "inconclusive",
        )
        self.assertEqual(
            self.module.request_outcome("w13_weight_scale ACL_WAIT_RETURN code=507046", 1), "scale_wait_reproduced"
        )

    def test_graph_schema_failure_and_missing_files_remain_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("UNVERIFIED", self.module.graph_evidence(root)[0])
            path = root / "mte/graph_debug/capture/graph_dump.json"
            path.parent.mkdir(parents=True)
            path.write_text("invalid", encoding="utf-8")
            self.assertIn("UNVERIFIED", self.module.graph_evidence(root)[0])
            path.write_text(json.dumps({"traceEvents": self.graph()}), encoding="utf-8")
            self.assertIn("structural_pairs=1", self.module.graph_evidence(root)[0])

    def test_single_launch_without_core_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.wait.return_value = 1
            with patch.object(self.module.subprocess, "Popen", return_value=process) as launch:
                self.assertEqual(self.module.run_once(root, root / "consumer_join", 7), 1)
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertIn("--prefetch-wait-at-join", command)
            self.assertNotIn("--compute-cube-cores", command)
            self.assertNotIn("--dsa-cube-cores", command)
            self.assertNotIn("--trace-startup", command)

    def test_verification_routes_both_backends_and_preserves_evidence_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(wait=Mock(return_value=0))
            with patch.object(self.module.subprocess, "Popen", return_value=process) as launch:
                self.module.run_once(root, root / "case", 7, True, True, True)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--backend") + 1], "both")
            self.assertEqual(launch.call_args.kwargs["env"]["ASCEND_MODULE_LOG_LEVEL"], "RUNTIME=0")
            for flag in (
                "--verification-suite",
                "--no-load-diagnostics",
                "--diagnose-prefetch-wait-at-join",
                "--skip-profile",
            ):
                self.assertIn(flag, command)
            (root / "summary.txt").write_text(
                "=== REAL MOE HOST BACKEND: callback ===\nEffective serve command: vllm serve /model\n"
                'MOE_RESULT {"host_backend": "callback"}\n'
                "=== REAL MOE HOST BACKEND: mte ===\nEffective serve command: vllm serve /model\n"
                'MOE_RESULT {"host_backend": "mte"}\nPAIR_RESULT {"status": "passed"}\n',
                encoding="utf-8",
            )
            for code, expected in ((0, "finite_output_match"), (1, "inconclusive_or_mismatch")):
                log = root / "callback/inference_python.log"
                log.parent.mkdir(exist_ok=True)
                log.write_text("RuntimeError: synthetic callback failure", encoding="utf-8")
                output = io.StringIO()
                self.module.append_evidence(output, root, code, verify_callback=True)
                self.assertIn("RESULT=" + expected, output.getvalue())
                self.assertEqual(output.getvalue().count("EFFECTIVE_CONFIG "), 2)
                self.assertIn("GATE_GRAPH_UNVERIFIED", output.getvalue())
                if code:
                    self.assertIn("synthetic callback failure", output.getvalue())
            (root / "summary.txt").write_text('PAIR_RESULT {"status":', encoding="utf-8")
            output = io.StringIO()
            self.module.append_evidence(output, root, 0, verify_callback=True)
            self.assertIn("RESULT=inconclusive_or_mismatch", output.getvalue())

    def test_startup_trace_still_launches_one_wait_placement_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.wait.return_value = 1
            with patch.object(self.module.subprocess, "Popen", return_value=process) as launch:
                self.assertEqual(
                    self.module.run_once(root, root / "consumer_join", 7, trace_startup=True, no_load_diagnostics=True),
                    1,
                )
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertIn("--trace-startup", command)
            self.assertIn("--no-load-diagnostics", command)
            self.assertIn("--prefetch-wait-at-join", command)
            self.assertNotIn("--native-stream-limits", command)

    def test_no_graph_dump_reaches_runner_and_summary_does_not_infer_graph_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(self.module.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=0))) as launch:
                self.module.run_once(root, root / "case", 7, True, True, True, True)
            command = launch.call_args.args[0]
            self.assertIn("--no-graph-dump", command)
            self.assertNotIn("--diagnose-graph-dump", command)
            with (
                patch.object(sys, "argv", [*command[2:], "--output-dir", str(root / "parsed")]),
                patch.object(sys, "platform", "linux"),
                patch.object(self.runner, "run", return_value=0) as parsed,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.runner.main(), 0)
            self.assertTrue(parsed.call_args.args[0].no_graph_dump)
            self.assertFalse(parsed.call_args.args[0].diagnose_graph_dump)
            (root / "summary.txt").write_text('PAIR_RESULT {"status": "passed"}', encoding="utf-8")
            output = io.StringIO()
            with patch.object(self.module, "graph_evidence", side_effect=AssertionError("graph debug disabled")):
                self.module.append_evidence(output, root, 0, verify_callback=True, no_graph_dump=True)
            self.assertIn("RESULT=finite_output_match", output.getvalue())
            self.assertIn("GATE_GRAPH=NOT_REQUESTED", output.getvalue())
            self.assertNotIn("structural_pairs=", output.getvalue())
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.runner.configured_source("unused", "mte", "/lib.so", diagnose_graph_dump=True, no_graph_dump=True)

    def test_graph_runner_removes_native_diagnostic_request_and_keeps_workload(self):
        graph = SimpleNamespace(enable_debug_mode=lambda: None, debug_dump=lambda self, path: None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            def launch(command, **kwargs):
                output.mkdir()
                return contextlib.nullcontext(Mock(wait=Mock(return_value=0)))

            with (
                patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu=SimpleNamespace(NPUGraph=graph))}),
                patch.object(
                    sys,
                    "argv",
                    [
                        "mapping",
                        "--npu",
                        "7",
                        "--prefetch-wait-at-join",
                        "--prefetch-wait-before-gate",
                        "--trace-startup",
                        "--no-load-diagnostics",
                        "--output-dir",
                        str(output),
                    ],
                ),
                patch.object(self.mapping.subprocess, "Popen", side_effect=launch) as launched,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.mapping.main(), 0)
            launched.assert_called_once()
            command = launched.call_args.args[0]
            self.assertNotIn("--diagnose-native-wait", command)
            for flag in (
                "--no-load-diagnostics",
                "--diagnose-prefetch-wait-at-join",
                "--diagnose-prefetch-wait-before-gate",
                "--diagnose-graph-dump",
                "--skip-profile",
                "--trace-startup",
            ):
                self.assertIn(flag, command)
            self.assertEqual(command[command.index("--requests") + 1], "3")
            self.assertEqual(command[command.index("--warmup-requests") + 1], "1")

    def test_no_load_diagnostics_overrides_inherited_flags_without_changing_other_config(self):
        offload = dict(
            expert_offload=True,
            expert_prefetch_enabled=True,
            host_sync_diagnose_graph_dump=True,
            host_sync_diagnose_load=True,
            host_sync_diagnose_native_wait=True,
        )
        source = "vllm serve /model --additional-config " + shlex.quote(json.dumps(dict(expert_offload_config=offload)))
        configs = []
        for disabled in (False, True):
            words = shlex.split(
                self.runner.configured_source(
                    source, "mte", "/lib.so", diagnose_prefetch_wait_at_join=True, no_load_diagnostics=disabled
                )
            )
            config = json.loads(words[words.index("--additional-config") + 1])["expert_offload_config"]
            self.assertEqual(config.pop("host_sync_diagnose_load"), not disabled)
            self.assertEqual(config.pop("host_sync_diagnose_native_wait"), not disabled)
            self.assertTrue(config["host_sync_diagnose_prefetch_wait_at_join"])
            configs.append(config)
        self.assertEqual(configs[0], configs[1])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.runner.configured_source(source, "mte", "/lib.so", no_load_diagnostics=True, diagnose_native_wait=True)

    def test_wait_order_control_resets_inherited_early_wait_and_only_changes_wait_flag(self):
        config = dict(
            expert_offload_config=dict(
                expert_offload=True, expert_prefetch_enabled=True, host_sync_diagnose_prefetch_wait_before_gate=True
            )
        )
        source = "vllm serve /model --additional-config " + shlex.quote(json.dumps(config))
        configs = []
        for early in (False, True):
            words = shlex.split(
                self.runner.configured_source(
                    source,
                    "mte",
                    "/lib.so",
                    diagnose_prefetch_wait_at_join=True,
                    diagnose_graph_dump=True,
                    no_load_diagnostics=True,
                    diagnose_prefetch_wait_before_gate=early,
                )
            )
            result = json.loads(words[words.index("--additional-config") + 1])["expert_offload_config"]
            self.assertEqual(result.pop("host_sync_diagnose_prefetch_wait_before_gate"), early)
            configs.append(result)
        self.assertEqual(configs[0], configs[1])

    def test_no_load_option_rejects_conflicting_diagnostics_before_launch(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "runner",
                    "--npu",
                    "7",
                    "--backend",
                    "mte",
                    "--diagnose-prefetch-wait-at-join",
                    "--no-load-diagnostics",
                    "--diagnose-native-wait",
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(self.runner, "run") as run,
            self.assertRaises(SystemExit),
        ):
            self.runner.main()
        run.assert_not_called()

    def test_wrapper_cleanup_survives_diagnostic_log_failure(self):
        log, process = Mock(), Mock(pid=123)
        log.write.side_effect = OSError("diagnostic output unavailable")
        with (
            patch.object(Path, "open", return_value=contextlib.nullcontext(log)),
            patch.object(self.module.subprocess, "Popen", return_value=process),
            patch.object(self.module, "CASE_TIMEOUT_SECONDS", 0),
            patch.object(self.module, "stop_owned_process_group") as cleanup,
        ):
            self.assertIsNone(self.module.run_once(Path("tree"), Path("tree/case"), 7, trace_startup=True))
        cleanup.assert_called_once_with(process)
        log.write.assert_called_once()

    def test_failed_observer_preflight_stops_before_build_or_model(self):
        output = io.StringIO()
        with (
            patch.object(self.runner, "run_step", return_value=False),
            patch.object(self.runner, "prepare_runtime_artifacts") as artifacts,
            patch.object(self.runner, "inference") as inference,
            contextlib.redirect_stdout(output),
        ):
            code = self.runner.run(SimpleNamespace(trace_startup=True), Path("unused"))
        self.assertEqual(code, 1)
        self.assertIn("FAILED: startup observation preflight", output.getvalue())
        self.assertIn("model not launched", output.getvalue())
        artifacts.assert_not_called()
        inference.assert_not_called()

    def test_startup_option_requires_the_same_mte_wait_placement_case(self):
        for backend in ("callback", "mte"):
            with (
                self.subTest(backend=backend),
                patch.object(sys, "argv", ["runner", "--npu", "7", "--backend", backend, "--trace-startup"]),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(self.runner, "run") as run,
                self.assertRaises(SystemExit),
            ):
                self.runner.main()
            run.assert_not_called()

    def test_success_summary_keeps_absent_structure_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph_diagnosis.txt").write_text("RUNNER_EXIT=0\nMOE_RESULT {}", encoding="utf-8")
            output = io.StringIO()
            self.module.append_evidence(output, root, 0)
            self.assertIn("finite_requests_passed", output.getvalue())
            self.assertIn("GATE_GRAPH_UNVERIFIED", output.getvalue())

    def test_startup_observation_and_request_result_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph_diagnosis.txt").write_text("RUNNER_EXIT=0\nMOE_RESULT {}", encoding="utf-8")
            (root / "mte").mkdir()
            service = "(EngineCore pid=123) loading\n[MOE-MTE PROCESS] " + json.dumps(
                dict(event="exit", child_pid=123, observer_pid=12, returncode=-15, signal="SIGTERM", monotonic_ns=30)
            )
            service += "\n[MOE-MTE RUNNER] " + json.dumps(dict(event="cleanup_begin", api_pid=12, monotonic_ns=20))
            (root / "mte/inference_python.log").write_text(service, encoding="utf-8")
            output = io.StringIO()
            self.module.append_evidence(output, root, 0, trace_startup=True)
            summary = output.getvalue()
        self.assertIn("RESULT=finite_requests_passed", summary)
        self.assertIn('"returncode": -15', summary)
        self.assertIn('"observed_after_runner_cleanup": true', summary)
        self.assertIn("STARTUP_TRACE_STATUS=UNVERIFIED", summary)  # No arm markers: do not claim complete coverage.

    def test_runner_config_enables_only_mte_and_preserves_model(self):
        source = 'vllm serve /model --additional-config \'{"expert_offload_config": {"expert_offload": true}}\''
        for backend, expected in (("mte", True), ("callback", False)):
            configured = self.runner.configured_source(source, backend, "/lib.so", diagnose_prefetch_wait_at_join=True)
            self.assertIn('"host_sync_diagnose_prefetch_wait_at_join": ' + str(expected).lower(), configured)
            self.assertIn("/model", configured)

    def test_effective_configuration_uses_actual_command_and_rejects_ambiguity(self):
        command = "Effective serve command: vllm serve /actual/model --api-key secret --compilation-config '{}'"
        text = self.module.effective_configuration(command)
        self.assertIn("/actual/model", text)
        self.assertIn("--compilation-config", text)
        self.assertNotIn("secret", text)
        self.assertIn("UNAVAILABLE", self.module.effective_configuration(command + "\n" + command))
        self.assertIn("UNAVAILABLE", self.module.effective_configuration(command + " '"))
        traced = command.replace("vllm serve", "python -B /tree/script/mte_startup_bootstrap.py --diagnostic serve")
        self.assertEqual(self.module.effective_configuration(command), self.module.effective_configuration(traced))


if __name__ == "__main__":
    unittest.main()
