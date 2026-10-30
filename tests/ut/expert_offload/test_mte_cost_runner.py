# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration checks for the controlled consumer-join cost comparison."""

import contextlib
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class MteCostRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location("cost_runner_test", cls.root / "script/run_moe_host_sync.py")
            cls.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.runner)
            cls.inference = sys.modules["run_verified_cpp_tests"]
        cls.source = (cls.root / "script/test_graph2.sh").read_text(encoding="utf-8")

    def parse(self, root, *options):
        with (
            patch.object(sys, "argv", ["runner", "--npu", "7", "--output-dir", str(root), *options]),
            patch.object(sys, "platform", "linux"),
            patch.object(self.runner, "run_cost_comparison", return_value=0) as cost,
            patch.object(self.runner, "run", return_value=0) as original,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(self.runner.main(), 0)
        invoked = cost if "--compare-join-cost" in options else original
        invoked.assert_called_once()
        return invoked.call_args.args[0]

    def test_cost_preset_and_original_defaults_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            cost = self.parse(Path(directory) / "cost", "--compare-join-cost")
            original = self.parse(Path(directory) / "original")
        self.assertEqual((cost.requests, cost.warmup_requests), (10, 3))
        self.assertEqual(cost.backend, "both")
        self.assertTrue(cost.diagnose_prefetch_wait_at_join)
        self.assertTrue(cost.no_load_diagnostics)
        self.assertTrue(cost.no_graph_dump)
        self.assertTrue(cost.skip_profile)
        self.assertFalse(cost.trace_startup)
        self.assertFalse(cost.diagnose_prefetch_wait_before_gate)
        self.assertEqual((cost.worker_scan, cost.worker_probe, cost.worker_idle), ("all", "locked", "yield"))
        self.assertEqual((original.requests, original.warmup_requests), (3, 1))
        self.assertFalse(original.diagnose_prefetch_wait_at_join)
        self.assertFalse(original.skip_profile)

    def test_cost_request_counts_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.parse(
                Path(directory) / "cost", "--compare-join-cost", "--requests", "4", "--warmup-requests", "2"
            )
        self.assertEqual((args.requests, args.warmup_requests), (4, 2))

    def test_cost_rejects_confounding_options_before_running(self):
        variants = (
            ("--diagnose-native-wait",),
            ("--diagnose-graph-dump",),
            ("--diagnose-prefetch-wait-before-gate",),
            ("--diagnose-prefetch-report-wait",),
            ("--diagnose-prefetch-poll-budget", "100"),
            ("--diagnose-dsa-cube-cores", "14"),
            ("--diagnose-routing-vector-cores", "5"),
            ("--trace-startup",),
            ("--verification-suite",),
            ("--worker-idle", "spin"),
            ("--worker-probe", "atomic"),
            ("--worker-scan", "allocated"),
            ("--backend", "mte"),
            ("--requests", "0"),
        )
        for options in variants:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit) as error:
                    self.parse(Path(directory) / "cost", "--compare-join-cost", *options)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((Path(directory) / "cost").exists())

    def inference_case(self, *, failure=None, active=False, mstx=False, cost=True):
        args = SimpleNamespace(
            npu=7,
            port=8155,
            timing_port=18155,
            startup_timeout=10,
            profile_run_dir=Path("profiles/test"),
            profile_timeout=10,
            request_timeout=10,
            warmup_requests=1,
            requests=2,
            max_tokens=5,
            skip_profile=True,
            mstx=False,
            compare_join_cost=cost,
        )
        sequence, completions = [], []
        measured = False

        def request(opener, port, path, payload=None, timeout=None):
            nonlocal measured
            sequence.append(path)
            if path in ("/start_profile", "/stop_profile"):
                self.fail("Cost windows must not invoke profiler endpoints")
            if path == "/status":
                return dict(backend="python", load_backend="python", active=active, mstx_enabled=mstx)
            if path == "/start":
                measured = True
            if path == "/v1/completions":
                number = len(completions)
                completions.append(payload)
                if (failure == "inactive" and number == 1) or (failure == "timed" and measured):
                    raise RuntimeError("injected completion failure")
                return dict(
                    choices=[dict(text=f"{'timed' if measured else 'inactive'}-{number}")],
                    usage=dict(completion_tokens=5, total_tokens=10),
                )
            if path == "/stop":
                return dict(count=4, errors=0, dropped=0, groups=[], mstx_enabled=False)
            return {}

        process = Mock(pid=123, returncode=None)
        process.poll.return_value = None
        cleanup, ports = Mock(), Mock()
        log = io.StringIO()
        result, error = None, None
        with (
            patch.object(self.inference, "request", side_effect=request),
            patch.object(self.inference, "assert_port_free"),
            patch.object(self.inference, "stop_owned_service", cleanup),
            patch.object(self.inference, "wait_service_ports_free", ports),
            patch.object(self.inference.subprocess, "Popen", return_value=process),
            patch.object(self.inference, "finish_profile", side_effect=AssertionError("No profile export")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                result = self.inference._inference(args, self.source, "python", {}, log)
            except RuntimeError as exc:
                error = exc
        cleanup.assert_called_once_with(process)
        ports.assert_called_once_with(args)
        return result, error, sequence, log.getvalue()

    def test_cost_uses_two_windows_in_one_service_without_profiler(self):
        result, error, sequence, log = self.inference_case()
        self.assertIsNone(error)
        self.assertEqual(
            sequence,
            ["/health", "/status"] + ["/v1/completions"] * 3 + ["/start"] + ["/v1/completions"] * 2 + ["/stop"],
        )
        self.assertEqual(result["untimed"]["count"], 2)
        self.assertEqual(
            result["untimed"]["digests"], [hashlib.sha256(f"inactive-{i}".encode()).hexdigest() for i in (1, 2)]
        )
        self.assertEqual(result["digests"], [hashlib.sha256(f"timed-{i}".encode()).hexdigest() for i in (3, 4)])
        self.assertEqual(len(result["untimed"]["usage"]), 2)
        self.assertEqual(len(result["usage"]), 2)
        self.assertEqual(result["profile_status"], "skipped")
        self.assertEqual(result["request_stats_us"]["mean"], result["request_us"])
        self.assertEqual(result["untimed"]["stats_us"]["mean"], result["untimed"]["request_us"])
        self.assertIn('"event": "cleanup_begin"', log)
        self.assertIn('"reason": "completed"', log)

    def test_inactive_window_failure_does_not_start_timers(self):
        result, error, sequence, log = self.inference_case(failure="inactive")
        self.assertIsNone(result)
        self.assertRegex(str(error), "injected completion failure")
        self.assertNotIn("/start", sequence)
        self.assertNotIn("/stop", sequence)
        self.assertIn('"reason": "RuntimeError"', log)

    def test_timed_window_failure_drains_collector_and_cleans_up(self):
        result, error, sequence, log = self.inference_case(failure="timed")
        self.assertIsNone(result)
        self.assertRegex(str(error), "injected completion failure")
        self.assertEqual(sequence[-1], "/stop")
        self.assertIn('"reason": "RuntimeError"', log)

    def test_invalid_collector_state_cannot_produce_cost_baseline(self):
        for settings in (dict(active=True), dict(mstx=True)):
            with self.subTest(settings=settings):
                result, error, sequence, _ = self.inference_case(**settings)
                self.assertIsNone(result)
                self.assertRegex(str(error), "inactive timers and disabled MSTX")
                self.assertNotIn("/start", sequence)
                self.assertEqual(sequence.count("/v1/completions"), 1)  # Warmup only.

    def test_original_inference_retains_single_measurement_window(self):
        result, error, sequence, log = self.inference_case(cost=False)
        self.assertIsNone(error)
        self.assertNotIn("untimed", result)
        self.assertEqual(sequence.count("/v1/completions"), 3)
        self.assertEqual(sequence.index("/start"), 3)
        self.assertEqual(log, "")

    def test_request_phase_engine_error_stops_before_next_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.parse(root / "parsed", "--compare-join-cost")
            args.test_script = self.root / "script/test_graph2.sh"
            calls = []

            def fake_inference(run_args, source, backend, env):
                calls.append(run_args)
                (run_args.output_dir / "inference_python.log").write_text(
                    'EngineDeadError: request phase\n[MOE-MTE RUNNER] {"event": "cleanup_begin"}\n',
                    encoding="utf-8",
                )
                return dict(profile_status="skipped", request_us=1)

            with (
                patch.object(self.runner, "run_step", return_value=True),
                patch.object(self.runner, "select_build_directory", return_value=root / "build"),
                patch.object(self.runner, "inference", side_effect=fake_inference),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "request-phase EngineDeadError"),
            ):
                self.runner.run(args, root)
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0].mstx)
            self.assertTrue(calls[0].skip_profile)
            self.assertTrue(calls[0].compare_join_cost)
            self.assertTrue((root / "callback/cost_report.json").is_file())
            self.assertFalse((root / "mte").exists())


if __name__ == "__main__":
    unittest.main()
