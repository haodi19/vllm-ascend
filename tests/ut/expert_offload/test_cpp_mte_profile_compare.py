# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the C++ policy/two-service profile orchestration."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class CppMteProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "cpp_profile_test", cls.root / "script/run_cpp_mte_profile_compare.py"
            )
            cls.compare = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.compare)
            cls.runner = sys.modules["run_moe_host_sync"]
            cls.infer = sys.modules["run_verified_cpp_tests"]
        cls.source = (cls.root / "script/test_graph2.sh").read_text(encoding="utf-8")

    def args(self, *extra, profile=True):
        prefix = ["--npu", "7", "--cache-policy-backend", "cpp"] if profile else ["--npu", "7"]
        with patch.object(sys, "platform", "linux"):
            return self.runner.parse_args([*prefix, *extra], profile_compare=profile)

    def setUp(self):
        self.storage = self.enterContext(patch.object(self.runner, "ensure_profile_storage"))
        self.inference_storage = self.enterContext(patch.object(self.infer, "ensure_profile_storage"))

    def report(self, status="complete"):
        return dict(
            units="us",
            count=2,
            errors=0,
            dropped=0,
            mstx_enabled=True,
            profile_status=status,
            request_us=1000.0,
            request_stats_us=dict(p95=1100.0),
            request_specs=[dict(prompt_sha256="prompt", max_tokens=5, temperature=0)],
            digests=[hashlib.sha256(b"text").hexdigest()],
            usage=[dict(prompt_tokens=5, completion_tokens=5)],
            engine_error_log_positions=dict(cleanup_marker_unique=True, before_cleanup=0, after_cleanup=0),
            groups=[
                dict(
                    kind="prefetch",
                    token_rows=1,
                    count=2,
                    layers=[1, 2],
                    loads=dict.fromkeys(self.compare.LOAD_STATS, 1.0),
                    metrics={metric: dict(mean=10.0, p95=15.0) for metric in self.compare.METRICS},
                )
            ],
        )

    def test_profile_defaults_and_legacy_python_defaults(self):
        args, legacy = self.args(), self.args(profile=False)
        self.assertEqual((args.cache_policy_backend, args.requests, args.warmup_requests), ("cpp", 1, 3))
        self.assertEqual((legacy.cache_policy_backend, legacy.requests, legacy.warmup_requests), ("python", 3, 1))
        self.assertTrue(args.diagnose_prefetch_wait_at_join and args.no_load_diagnostics and args.no_graph_dump)
        self.assertFalse(args.skip_profile or args.compare_join_cost)
        self.assertEqual(args.large_gap_us, 100.0)
        self.assertEqual(args.profile_timeout, 120)
        self.assertFalse(legacy.diagnose_prefetch_wait_at_join)

    def test_profile_rejects_confounded_settings(self):
        for options in (
            ("--cache-policy-backend", "python"),
            ("--backend", "mte"),
            ("--skip-profile",),
            ("--compare-join-cost",),
            ("--diagnose-native-wait",),
            ("--diagnose-graph-dump",),
            ("--worker-idle", "spin"),
            ("--diagnose-compute-cube-cores", "14"),
            ("--large-gap-us", "nan"),
            ("--large-gap-us", "0"),
        ):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.args(*options)
        args = self.args("--requests", "2", "--large-gap-us", "80")
        self.assertEqual((args.requests, args.large_gap_us), (2, 80.0))

    def test_two_cpp_services_build_only_lrc_and_host_extensions_and_keep_policy_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, calls = self.args(), []

            def fake_inference(run_args, source, policy, env):
                calls.append((run_args, policy))
                self.assertEqual(policy, "cpp")
                self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "7")
                self.assertEqual((run_args.requests, run_args.warmup_requests, run_args.max_tokens), (1, 3, 5))
                self.assertTrue(run_args.mstx and run_args.profile_compare)
                self.assertFalse(run_args.skip_profile or run_args.compare_join_cost)
                prefix, command, _, _ = self.runner.launch_spec(source, policy, 7, None, 18155)
                settings = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
                self.assertEqual(settings["cache_policy_backend"], "cpp")
                self.assertEqual(settings["h2d_backend"], "torch")
                self.assertFalse(settings["moe_offload_debug"])
                self.assertFalse(settings["host_sync_diagnose_load"])
                self.assertFalse(settings["host_sync_diagnose_graph_dump"])
                self.assertEqual(
                    settings["host_sync_diagnose_prefetch_wait_at_join"], settings["host_sync_backend"] == "mte"
                )
                self.assertIn("unset ASCEND_LAUNCH_BLOCKING", prefix)
                (run_args.output_dir / "inference_cpp.log").write_text(
                    "[MOE-MTE WORKER] scan=all probe=locked idle=yield\n"
                    "[MOE-MTE] first real offload callback completed\n"
                    '[MOE-MTE RUNNER] {"event": "cleanup_begin"}\n',
                    encoding="utf-8",
                )
                trace = run_args.profile_run_dir / "cpp/worker/ASCEND_PROFILER_OUTPUT/trace_view.json"
                trace.parent.mkdir(parents=True)
                trace.write_text("[]", encoding="utf-8")
                return self.report()

            checked = dict(json_complete=True, usable=True, issues=[], host_gaps=[], memcpy_gaps=[])
            with (
                patch.object(self.runner, "run_step", return_value=True) as steps,
                patch.object(self.runner, "select_build_directory", return_value=root / "build"),
                patch.object(self.runner, "scan_trace", return_value=checked) as scan,
                patch.object(self.runner, "inference", side_effect=fake_inference),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                reports = self.runner.run(args, root)
            self.assertEqual(set(reports), {"callback", "mte"})
            self.assertEqual(len(calls), 2)
            commands = [call.args[1] for call in steps.call_args_list]
            self.assertEqual(
                [command[command.index("install") + 1] for command in commands if "install" in command],
                ["./csrc/lrc_policy", "./csrc/host_sync"],
            )
            self.assertTrue(any("--require-cpp" in command for command in commands))
            self.assertEqual(scan.call_count, 2)
            for call, backend in zip(scan.call_args_list, ("callback", "mte")):
                self.assertIn(f"profile/{backend}/cpp/", call.args[0].as_posix())
                self.assertEqual(call.kwargs, dict(require_mte=backend == "mte", large_gap_us=100.0))
                self.assertTrue((root / backend / "profile_report.json").exists())
                self.assertEqual(len(reports[backend]["profile_trace_data"]), 1)

    def test_stream_comparison_only_changes_load_stream_and_preserves_baseline_on_timeout(self):
        for fail_ordinary, three_way in ((False, False), (True, False), (False, True), (True, True)):
            with tempfile.TemporaryDirectory() as directory:
                root, settings, calls = Path(directory), [], []
                labels = ("callback", "sync_launch", "ordinary") if three_way else ("sync_launch", "ordinary")
                with patch.object(sys, "platform", "linux"):
                    args = self.runner.parse_args(
                        ["--npu", "7", "--cache-policy-backend", "cpp"],
                        profile_compare=True,
                        stream_compare=True,
                        three_way=three_way,
                    )

                def fake_inference(
                    run_args,
                    source,
                    policy,
                    env,
                    *,
                    calls=calls,
                    settings=settings,
                    root=root,
                    fail_ordinary=fail_ordinary,
                ):
                    mode = run_args.output_dir.name
                    calls.append(mode)
                    _, command, _, _ = self.runner.launch_spec(source, policy, 7, None, 18155)
                    config = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
                    self.assertEqual(config.pop("host_sync_load_stream"), "ordinary" if mode == "callback" else mode)
                    self.assertEqual(config["host_sync_backend"], "callback" if mode == "callback" else "mte")
                    self.assertEqual(config["host_sync_diagnose_prefetch_wait_at_join"], mode != "callback")
                    self.assertFalse(config["host_sync_diagnose_load"])
                    self.assertEqual(policy, "cpp")
                    settings.append(config)
                    self.assertEqual(run_args.profile_run_dir, root / "profile" / mode)
                    stream_type = "SyncLaunchStream" if mode == "sync_launch" else "Stream"
                    (run_args.output_dir / "inference_cpp.log").write_text(
                        f"[MOE-MTE] weight load stream={stream_type}; mode={mode}; H2D wait retained\n"
                        "[MOE-MTE WORKER] scan=all probe=locked idle=yield\n"
                        "[MOE-MTE] first real offload callback completed\n"
                        '[MOE-MTE RUNNER] {"event": "cleanup_begin"}\n',
                        encoding="utf-8",
                    )
                    if fail_ordinary and mode == "ordinary":
                        raise TimeoutError("ordinary request timed out")
                    return self.report("incomplete")

                with (
                    patch.object(self.runner, "run_step", return_value=True) as steps,
                    patch.object(self.runner, "select_build_directory", return_value=root / "build"),
                    patch.object(self.runner, "inference", side_effect=fake_inference),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if fail_ordinary:
                        with self.assertRaisesRegex(TimeoutError, "ordinary request"):
                            self.runner.run(args, root)
                    else:
                        reports = self.runner.run(args, root)
                        self.assertEqual(set(reports), set(labels))
                        result = self.compare.print_comparison(reports, root, labels)
                        self.assertTrue(result["outputs_match"])
                self.assertEqual(calls, list(labels))
                self.assertEqual(settings[-2], settings[-1])
                if three_way:
                    callback, mte = copy.deepcopy(settings[0]), copy.deepcopy(settings[1])
                    for key in ("host_sync_backend", "host_sync_library", "host_sync_diagnose_prefetch_wait_at_join"):
                        callback.pop(key)
                        mte.pop(key)
                    self.assertEqual(callback, mte)
                    self.assertTrue((root / "callback/profile_report.json").is_file())
                self.assertTrue((root / "sync_launch/profile_report.json").is_file())
                installs = [call.args[1] for call in steps.call_args_list if "install" in call.args[1]]
                self.assertEqual(len(installs), 2)  # one build per extension, not per service

    def test_three_way_summary_compares_all_outputs_and_workloads(self):
        labels = ("callback", "sync_launch", "ordinary")
        reports = {name: self.report("incomplete") for name in labels}
        for name, report in reports.items():
            report.update(
                host_backend="callback" if name == "callback" else "mte",
                load_stream_mode="ordinary" if name == "callback" else name,
                cache_policy_backend="cpp",
            )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.compare.print_comparison(reports, Path("run"), labels)
        self.assertTrue(result["outputs_match"] and result["workload_summaries_match"])
        self.assertIn("ordinary-callback=", output.getvalue())
        self.assertIn("ordinary-sync_launch=", output.getvalue())
        self.assertEqual(output.getvalue().count("PROFILE ordinary:"), 1)
        reports["ordinary"]["groups"][0]["count"] = 3
        reports["ordinary"]["count"] = 3
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.compare.print_comparison(reports, Path("run"), labels)
        self.assertFalse(result["workload_summaries_match"])
        reports["ordinary"]["digests"] = [hashlib.sha256(b"different").hexdigest()]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "输出不一致"):
            self.compare.print_comparison(reports, Path("run"), labels)

    def test_three_way_entry_uses_three_labels_and_keeps_completed_reports_on_failure(self):
        def fake_run(args, root):
            self.assertTrue(args.three_way and args.stream_compare and args.profile_compare)
            for name in ("callback", "sync_launch"):
                directory = root / name
                directory.mkdir()
                (directory / "profile_report.json").write_text(json.dumps(self.report()), encoding="utf-8")
            raise TimeoutError("ordinary timed out")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "platform", "linux"),
            patch.object(self.compare, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory) / "run"
            self.assertEqual(self.compare.main(["--npu", "7", "--output-dir", str(root)], three_way=True), 1)
            summary = (root / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("PARTIAL_METRICS callback", summary)
            self.assertIn("PARTIAL_METRICS sync_launch", summary)
            self.assertIn("PARTIAL ordinary: 未取得完成报告", summary)

    def test_stream_comparison_rejects_wrong_runtime_identity(self):
        reports = {name: self.report() for name in ("sync_launch", "ordinary")}
        for name, report in reports.items():
            report.update(host_backend="mte", load_stream_mode=name, cache_policy_backend="cpp")
        reports["ordinary"]["load_stream_mode"] = "sync_launch"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "identity"):
            self.compare.print_comparison(reports, Path("run"), ("sync_launch", "ordinary"))

    def test_stream_entry_preserves_first_report_and_includes_failure_excerpt(self):
        def fake_run(args, root):
            self.assertTrue(args.stream_compare and args.profile_compare)
            first = root / "sync_launch"
            first.mkdir()
            (first / "profile_report.json").write_text(json.dumps(self.report()), encoding="utf-8")
            second = root / "ordinary"
            second.mkdir()
            (second / "inference_cpp.log").write_text("[MOE-MTE FATAL] stalled\n", encoding="utf-8")
            raise TimeoutError("ordinary timeout")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "platform", "linux"),
            patch.object(self.compare, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory) / "run"
            result = self.compare.main(["--npu", "7", "--output-dir", str(root)], stream_compare=True)
            self.assertEqual(result, 1)
            summary = (root / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("PARTIAL_METRICS sync_launch", summary)
            self.assertIn("FAILED_CASE ordinary", summary)
            self.assertIn("[MOE-MTE FATAL] stalled", summary)

    def test_lrc_build_or_parity_failure_prevents_model_start(self):
        for returns in ([False], [True, False]):
            with (
                self.subTest(returns=returns),
                tempfile.TemporaryDirectory() as directory,
                patch.object(self.runner, "run_step", side_effect=returns),
                patch.object(self.runner, "inference") as inference,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.runner.run(self.args(), Path(directory)), 1)
                inference.assert_not_called()

    def test_failed_callback_profile_still_runs_mte_service(self):
        calls = []

        def fake_inference(args, source, policy, env):
            calls.append(args.output_dir.name)
            (args.output_dir / "inference_cpp.log").write_text(
                "[MOE-MTE WORKER] scan=all probe=locked idle=yield\n"
                "[MOE-MTE] first real offload callback completed\n"
                '[MOE-MTE RUNNER] {"event": "cleanup_begin"}\n',
                encoding="utf-8",
            )
            return self.report("incomplete")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(self.runner, "run_step", return_value=True),
            patch.object(self.runner, "select_build_directory", return_value=Path(directory) / "build"),
            patch.object(self.runner, "inference", side_effect=fake_inference),
            patch.object(self.runner, "scan_trace") as scan,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            reports = self.runner.run(self.args(), Path(directory))
            self.assertEqual(calls, ["callback", "mte"])
            self.assertEqual(reports["callback"]["profile_status"], "incomplete")
            self.assertEqual(reports["mte"]["request_us"], 1000.0)
            scan.assert_not_called()

    def test_profile_inference_does_not_copy_or_rescan_trace_and_keeps_failed_export_report(self):
        args = SimpleNamespace(
            npu=7,
            port=8155,
            timing_port=18155,
            startup_timeout=10,
            request_timeout=10,
            profile_timeout=10,
            strict_profile=False,
            profile_run_dir=Path("profiles/test"),
            warmup_requests=3,
            requests=1,
            max_tokens=5,
            skip_profile=False,
            mstx=True,
            profile_compare=True,
            compare_join_cost=False,
        )
        sequence = []
        self.inference_storage.side_effect = lambda path, stage: sequence.append("STORAGE")

        def request(opener, port, path, payload=None, timeout=None):
            sequence.append(path)
            if path == "/status":
                return dict(backend="cpp", load_backend="python", active=False, mstx_enabled=True)
            if path == "/v1/completions":
                return dict(choices=[dict(text="text")], usage=dict(completion_tokens=5, total_tokens=10))
            if path == "/stop":
                return dict(count=2, errors=0, dropped=0, groups=[], mstx_enabled=True)
            return {}

        process = Mock(pid=123, returncode=None)
        process.poll.return_value = None
        log = io.StringIO()
        with (
            patch.object(self.infer, "request", side_effect=request),
            patch.object(self.infer, "assert_port_free"),
            patch.object(self.infer, "stop_owned_service") as stop,
            patch.object(self.infer, "wait_service_ports_free"),
            patch.object(self.infer.subprocess, "Popen", return_value=process),
            patch.object(self.infer, "finish_profile", return_value="incomplete") as finish,
            patch.object(self.infer, "inspect_mstx_trace", side_effect=AssertionError("duplicate scan")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = self.infer._inference(args, self.source, "cpp", {}, log)
        self.assertEqual(
            sequence,
            ["/health", "/status"]
            + ["/v1/completions"] * 3
            + ["STORAGE", "/start_profile", "/start", "/v1/completions", "/stop"],
        )
        self.assertEqual(result["profile_status"], "incomplete")
        self.assertEqual(len(result["digests"]), 1)
        self.assertNotIn("untimed", result)
        finish.assert_called_once()
        stop.assert_called_once_with(process)
        self.assertIn('"event": "cleanup_begin"', log.getvalue())

    def test_failed_profile_keeps_cpu_comparison_without_gap_claim(self):
        first, second = self.report("incomplete"), self.report("invalid")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.compare.print_comparison(dict(callback=first, mte=second), Path("run"))
        self.assertTrue(result["outputs_match"])
        self.assertFalse(result["both_profiles_usable"])
        self.assertIn("load_submit", output.getvalue())
        self.assertIn("无可验证的完整时间线", output.getvalue())
        self.assertNotIn("MEMCPY_GAP", output.getvalue())

    def test_output_mismatch_rejects_comparison(self):
        second = self.report()
        second["digests"] = [hashlib.sha256(b"other").hexdigest()]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "输出不一致"):
            self.compare.print_comparison(dict(callback=self.report(), mte=second), Path("run"))

    def test_workload_or_cleanup_mismatch_is_not_accepted_as_gain(self):
        second = self.report()
        second["groups"][0]["loads"]["mean"] = 2.0
        second["engine_error_log_positions"]["before_cleanup"] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.compare.print_comparison(dict(callback=self.report(), mte=second), Path("run"))
        self.assertFalse(result["workload_summaries_match"])
        self.assertFalse(result["no_precleanup_engine_error"])

    def test_threshold_statistics_and_multiple_request_scope_are_printed(self):
        report = self.report()
        row = dict(
            count=3,
            mean_us=60,
            large_gap_threshold_us=100,
            large_gaps=dict(count=1, mean_us=150),
            large_gap_fraction=1 / 3,
        )
        report["profile_trace_data"] = [
            dict(
                json_complete=True,
                usable=True,
                file="trace_view.json",
                host_gaps=[dict(direction="prefetch->reactive", **row)],
                memcpy_gaps=[dict(stream=1, **row)],
            )
        ]
        for field in ("request_specs", "digests", "usage"):
            report[field] *= 2
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.compare.print_comparison(dict(callback=report, mte=copy.deepcopy(report)), Path("run"))
        self.assertIn('"large_gap_threshold_us":100', output.getvalue())
        self.assertIn("HOST_CALLBACK_GAP", output.getvalue())
        self.assertIn("MEMCPY_GAP", output.getvalue())
        self.assertIn("REQUEST_BOUNDARIES", output.getvalue())

    def test_entry_preserves_concise_summary_separate_from_detailed_output(self):
        def fake_run(args, root):
            self.assertEqual(args.cache_policy_backend, "cpp")
            print("verbose data only for details")
            return dict(callback=self.report("incomplete"), mte=self.report("incomplete"))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "platform", "linux"),
            patch.object(self.compare, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory) / "run"
            self.assertEqual(self.compare.main(["--npu", "7", "--output-dir", str(root)]), 0)
            self.assertNotIn("verbose data", (root / "summary.txt").read_text(encoding="utf-8"))
            self.assertIn("verbose data", (root / "run_details.log").read_text(encoding="utf-8"))
            self.assertIn("load_submit", (root / "summary.txt").read_text(encoding="utf-8"))

    def test_mte_failure_keeps_completed_callback_in_summary(self):
        def fake_run(args, root):
            (root / "callback").mkdir()
            (root / "callback/profile_report.json").write_text(json.dumps(self.report()), encoding="utf-8")
            raise RuntimeError("MTE startup failed")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "platform", "linux"),
            patch.object(self.compare, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory) / "run"
            self.assertEqual(self.compare.main(["--npu", "7", "--output-dir", str(root)]), 1)
            summary = (root / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("MTE startup failed", summary)
            self.assertIn("PARTIAL_METRICS callback", summary)
            self.assertIn("load_submit", summary)
            self.assertIn("PARTIAL mte: 未取得完成报告", summary)
            self.assertIn("不能据此比较收益", summary)

    def test_display_rounding_does_not_mutate_full_precision_report(self):
        report = dict(
            mean_us=3.12345678, large_gap_fraction=0.12345678, scope="verbose", large_gaps=dict(mean_us=100.12345678)
        )
        original = copy.deepcopy(report)
        output = self.compare.display_numbers(report)
        self.assertEqual(output["mean_us"], 3.123)
        self.assertEqual(output["large_gap_fraction"], 0.123457)
        self.assertNotIn("scope", output)
        self.assertEqual(report, original)

    def worker_reports(self):
        reports = {}
        for label, scan, probe, idle in self.runner.WORKER_PROFILE_CASES:
            reports[label] = self.report()
            reports[label].update(
                host_backend="mte",
                load_stream_mode="ordinary",
                cache_policy_backend="cpp",
                worker_policy=dict(scan=scan, probe=probe, idle=idle),
            )
        return reports

    def test_worker_candidates_change_one_option_and_stop_on_failure(self):
        for failure in (None, "timeout", "policy", "output", "dropped", "capacity"):
            self.storage.reset_mock(side_effect=True)
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root, calls, configs = Path(directory), [], []
                with patch.object(sys, "platform", "linux"):
                    args = self.runner.parse_args(
                        ["--npu", "7", "--cache-policy-backend", "cpp"],
                        profile_compare=True,
                        worker_compare=True,
                    )

                def fake_inference(
                    run_args, source, policy, env, *, calls=calls, root=root, configs=configs, failure=failure
                ):
                    label = run_args.output_dir.name
                    calls.append(label)
                    if label == "baseline" and failure == "capacity":
                        self.storage.side_effect = RuntimeError("Insufficient profile storage")
                    self.assertEqual((policy, run_args.max_tokens, run_args.requests), ("cpp", 5, 1))
                    self.assertEqual(run_args.profile_run_dir, root / "profile" / label)
                    self.assertFalse(run_args.skip_profile)
                    _, command, _, _ = self.runner.launch_spec(source, policy, 7, None, 18155)
                    config = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
                    self.assertEqual(config["host_sync_backend"], "mte")
                    self.assertEqual(config["host_sync_load_stream"], "ordinary")
                    self.assertTrue(config["host_sync_diagnose_prefetch_wait_at_join"])
                    self.assertFalse(config["host_sync_diagnose_load"])
                    configs.append(config)
                    scan, probe, idle = (config[f"host_sync_{key}"] for key in ("scan_mode", "idle_probe", "idle_mode"))
                    if label == "allocated" and failure == "policy":
                        scan = "all"
                    (run_args.output_dir / "inference_cpp.log").write_text(
                        "[MOE-MTE] weight load stream=Stream; mode=ordinary; H2D wait retained\n"
                        f"[MOE-MTE WORKER] scan={scan} probe={probe} idle={idle}\n"
                        "[MOE-MTE] first real offload callback completed\n"
                        '[MOE-MTE RUNNER] {"event": "cleanup_begin"}\n',
                        encoding="utf-8",
                    )
                    report = self.report("incomplete")
                    if label == "allocated":
                        if failure == "timeout":
                            raise TimeoutError("worker request timeout")
                        if failure == "output":
                            report["digests"] = [hashlib.sha256(b"different").hexdigest()]
                        if failure == "dropped":
                            report["dropped"] = 1
                    return report

                with (
                    patch.object(self.runner, "run_step", return_value=True) as steps,
                    patch.object(self.runner, "select_build_directory", return_value=root / "build"),
                    patch.object(self.runner, "inference", side_effect=fake_inference),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if failure:
                        with self.assertRaises((RuntimeError, TimeoutError)):
                            self.runner.run(args, root)
                        self.assertEqual(calls, ["baseline"] if failure == "capacity" else ["baseline", "allocated"])
                        self.assertTrue((root / "baseline/profile_report.json").exists())
                    else:
                        reports = self.runner.run(args, root)
                        self.assertEqual(calls, [case[0] for case in self.runner.WORKER_PROFILE_CASES])
                        self.assertEqual(configs[0], configs[-1])
                        for config in configs[1:-1]:
                            self.assertEqual(sum(config[key] != configs[0][key] for key in config), 1)
                        for label, scan, probe, idle in self.runner.WORKER_PROFILE_CASES:
                            self.assertEqual(reports[label]["worker_policy"], dict(scan=scan, probe=probe, idle=idle))
                        commands = [call.args[1] for call in steps.call_args_list]
                        self.assertEqual(sum("install" in command for command in commands), 2)
                        self.assertTrue(any("tests/ut/expert_offload/test_host_sync_session.py" in c for c in commands))

    def test_worker_summary_checks_all_outputs_and_experiment_identity(self):
        labels = tuple(case[0] for case in self.runner.WORKER_PROFILE_CASES)
        for changed in labels:
            reports = self.worker_reports()
            reports[changed]["digests"] = [hashlib.sha256(b"wrong").hexdigest()]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "输出不一致"):
                self.compare.print_comparison(reports, Path("run"), labels)
        reports = self.worker_reports()
        reports["spin"]["worker_policy"]["idle"] = "yield"
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "identity"):
            self.compare.print_comparison(reports, Path("run"), labels)

    def test_worker_entry_preserves_profiles_metrics_and_repeated_baseline(self):
        def fake_run(args, root):
            self.assertTrue(args.worker_compare and args.diagnose_prefetch_wait_at_join)
            return self.worker_reports()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "platform", "linux"),
            patch.object(self.compare, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory) / "run"
            self.assertEqual(self.compare.main(["--npu", "7", "--output-dir", str(root)], worker_compare=True), 0)
            summary = (root / "summary.txt").read_text(encoding="utf-8")
            for word in ("baseline_repeat", "allocated", "atomic", "spin", "copy_submit", "last_baseline"):
                self.assertIn(word, summary)
            self.assertIn(str(root / "profile/spin/cpp"), summary)

    def test_worker_preset_rejects_confounded_modes(self):
        with self.assertRaises(ValueError):
            self.runner.parse_args([], worker_compare=True)
        with self.assertRaises(ValueError):
            self.runner.parse_args([], profile_compare=True, stream_compare=True, worker_compare=True)
        with (
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.runner.parse_args(
                ["--npu", "7", "--cache-policy-backend", "cpp", "--worker-idle", "spin"],
                profile_compare=True,
                worker_compare=True,
            )

    def test_low_storage_stops_before_build(self):
        self.storage.side_effect = RuntimeError("Insufficient profile storage")
        with patch.object(self.runner, "run_step") as build, self.assertRaisesRegex(RuntimeError, "storage"):
            self.runner.run(self.args(), Path("unused-output"))
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
