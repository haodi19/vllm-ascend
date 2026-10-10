"""Standalone CPU tests for inference instrumentation and test orchestration."""

import contextlib
import io
import json
import runpy
import sys
import tempfile
import threading
import types
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[3]


class TestCallbackTimings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))

    def make_call(self):
        now = [0]
        collector = self.module["CallbackTimings"]("cpp", clock=lambda: now[0])

        def advance(ns):
            now[0] += ns

        observe = collector.wrap("observe", lambda: advance(10000))
        choose = collector.wrap("choose_victims", lambda: advance(20000))
        prepare = collector.wrap("load_prepare", lambda: advance(30000))
        copy = collector.wrap("copy_submit", lambda: advance(40000))
        scale = collector.wrap("scale_refresh", lambda: advance(5000))
        sync = collector.wrap("synchronize", lambda: advance(70000))

        def load(*args):
            prepare()
            copy()
            scale()
            advance(5000)

        submit = collector.wrap("load_submit", load, count_loads=True)

        def update(args):
            if not args[5]:
                observe()
            choose()
            submit(None, 1, [(0, 2), (1, 3)])
            sync()
            advance(25000)
            return "original return"

        return collector, collector.wrap_update(update)

    @staticmethod
    def args(prefetch=False, rows=1):
        return (types.SimpleNamespace(shape=(rows, 6)), None, None, 2, None, prefetch)

    def test_window_excludes_warmup_and_preserves_formulas(self):
        collector, call = self.make_call()
        self.assertEqual(call(self.args()), "original return")
        self.assertEqual(collector.stop()["count"], 0)
        collector.start()
        with self.assertRaises(ValueError):
            collector.start()
        call(self.args())
        call(self.args(True, 3))
        report = collector.stop()
        self.assertEqual(report["count"], 2)
        groups = {g["kind"]: g for g in report["groups"]}
        metrics = groups["reactive"]["metrics"]
        for key, value in {
            "total": 205,
            "observe": 10,
            "choose_victims": 20,
            "synchronize": 70,
            "outer": 105,
            "host_without_sync": 135,
            "load_submit": 80,
            "load_prepare": 30,
            "copy_submit": 40,
            "scale_refresh": 5,
            "load_other": 5,
        }.items():
            self.assertEqual(metrics[key]["mean"], value)
        self.assertEqual(groups["prefetch"]["metrics"]["observe"]["mean"], 0)
        self.assertEqual(groups["prefetch"]["token_rows"], 3)
        self.assertEqual(groups["reactive"]["loads"]["mean"], 2)
        call(self.args())
        self.assertEqual(collector.stop()["count"], 2)
        collector.start()
        self.assertEqual(collector.stop()["count"], 0)

    def test_errors_are_reported_and_not_swallowed(self):
        collector, _ = self.make_call()

        def fail(args):
            raise RuntimeError("callback failed")

        collector.start()
        with self.assertRaisesRegex(RuntimeError, "callback failed"):
            collector.wrap_update(fail)(self.args())
        result = collector.stop()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["count"], 0)
        self.assertEqual(collector.inflight, 0)

    def test_http_controller_and_bounded_storage(self):
        collector, call = self.make_call()
        collector.serve(0)
        self.addCleanup(collector.close)
        root = f"http://127.0.0.1:{collector.server.server_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(path, data=None):
            with opener.open(urllib.request.Request(root + path, data=data), timeout=3) as response:
                return json.load(response)

        self.assertEqual(request("/status")["backend"], "cpp")
        self.assertTrue(request("/start", b"{}")["started"])
        with patch.dict(collector.wrap_update.__globals__, {"MAX_SAMPLES": 1}):
            call(self.args())
            call(self.args())
        result = request("/stop", b"{}")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["dropped"], 1)

    def test_stop_drains_inflight_callback(self):
        collector, _ = self.make_call()
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

        def callback(args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test callback timed out")

        reports = []

        def stop():
            reports.append(collector.stop())
            stopped.set()

        collector.start()
        worker = threading.Thread(target=collector.wrap_update(callback), args=(self.args(),))
        worker.start()
        self.assertTrue(entered.wait(3))
        stopper = threading.Thread(target=stop)
        stopper.start()
        try:
            self.assertFalse(stopped.wait(0.05))
        finally:
            release.set()
            worker.join(3)
            stopper.join(3)
        self.assertTrue(stopped.is_set())
        self.assertEqual(reports[0]["count"], 1)


class TestVerifiedRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(REPO / "script/run_verified_cpp_tests.py"))
        cls.source = (REPO / "script/test_graph2.sh").read_text(encoding="utf-8")

    def test_subprocess_noise_goes_to_log_and_metrics_to_summary(self):
        fn = self.module["run_console"]
        with tempfile.TemporaryDirectory() as root:
            log = Path(root) / "child.log"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                means = fn([sys.executable, "-c", "print('noisy log'); print('total 1 2 3 4 5 6')"], None, log)
            self.assertEqual(means, {"total": 1})
            self.assertNotIn("noisy log", console.getvalue())
            self.assertIn("total 1 2 3 4 5 6", console.getvalue())
            self.assertIn("noisy log", log.read_text())

    def test_failed_run_retains_summary(self):
        fn = self.module["main"]

        def fail(args):
            print("completed timing: total=12")
            raise RuntimeError("injected export failure")

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new_run"
            with (
                patch.dict(fn.__globals__, {"os": types.SimpleNamespace(name="posix"), "run_tests": fail}),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                fn(["--output-dir", str(output)])
            self.assertEqual(raised.exception.code, 1)
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("completed timing: total=12", summary)
            self.assertIn("FAILED: RuntimeError: injected export failure", summary)

    def test_service_configs_differ_only_in_policy(self):
        commands = []
        for backend in ("python", "cpp"):
            prefix, words, model, config_path = self.module["launch_spec"](self.source, backend, 4, 8155, 18155)
            self.assertNotIn("rm -rf", prefix)
            self.assertNotIn("sysctl", prefix)
            self.assertIn("--profiler-config", words)
            self.assertIn("export ASCEND_RT_VISIBLE_DEVICES=4", prefix)
            self.assertEqual(model, "glm-5")
            self.assertEqual(config_path.name, "config.json")
            index = words.index("--additional-config") + 1
            config = json.loads(words[index])
            self.assertEqual(config["expert_offload_config"].pop("cache_policy_backend"), backend)
            self.assertEqual(config["expert_offload_config"]["h2d_backend"], "torch")
            self.assertTrue(config["expert_offload_config"]["moe_offload_debug"])
            words[index] = json.dumps(config)
            commands.append(words)
        self.assertEqual(*commands)
        with self.assertRaises(ValueError):
            self.module["launch_spec"](self.source, "cpp_fused", 4, 8155, 18155)

    def test_preserves_original_settings_except_explicit_instrumentation(self):
        fn = self.module["launch_spec"]
        prefix, words, _, _ = fn(self.source, "cpp", None, None, 18155, Path("profiles/cpp"))
        self.assertIn('export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"', prefix)
        self.assertEqual(words[words.index("--host") + 1], "0.0.0.0")
        self.assertEqual(words[words.index("--port") + 1], "8155")
        original = self.module["shlex"].split(self.source[self.source.index("vllm serve ") :] + "\n", comments=True)
        original = [w for w in original if w.strip()]
        config_index = words.index("--additional-config") + 1
        config = json.loads(words[config_index])
        for key in ("cache_policy_backend", "h2d_backend", "callback_timing_port"):
            config["expert_offload_config"].pop(key)
        self.assertEqual(config, json.loads(original[config_index]))
        words[config_index] = original[config_index]
        profile_index = words.index("--profiler-config") + 1
        profile = json.loads(words[profile_index])
        self.assertEqual(Path(profile.pop("torch_profiler_dir")), Path("profiles/cpp"))
        original_profile = json.loads(original[profile_index])
        original_profile.pop("torch_profiler_dir")
        self.assertEqual(profile, original_profile)
        words[profile_index] = original[profile_index]
        self.assertEqual(words, original)

    def test_profile_requires_completed_export(self):
        fn = self.module["wait_profile"]
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            with self.assertRaisesRegex(RuntimeError, "not confirmed"):
                fn(directory, 0)
            output = directory / "worker" / "ASCEND_PROFILER_OUTPUT"
            output.mkdir(parents=True)
            (output / "trace_view.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                fn(directory, 0)
            (output / "analyse.done").touch()
            with contextlib.redirect_stdout(io.StringIO()):
                fn(directory, 0)

    def test_inference_window_and_cleanup(self):
        fn = self.module["inference"]
        args = types.SimpleNamespace(
            npu=4,
            port=8155,
            timing_port=18155,
            startup_timeout=20,
            request_timeout=10,
            warmup_requests=2,
            requests=3,
            max_tokens=5,
            profile_run_dir=Path("profiles/test"),
            profile_timeout=10,
        )
        sequence = []

        def request(opener, port, path, payload=None, timeout=None):
            sequence.append(path)
            if path == "/status":
                return dict(backend="cpp", load_backend="python")
            if path == "/v1/completions":
                return dict(choices=[{"text": "synthetic"}], usage={"completion_tokens": 5})
            if path == "/stop":
                return dict(count=5, errors=0, dropped=0, groups=[])
            if path == "/stop_profile":
                self.assertIn("INFERENCE cpp: callbacks=5", captured.getvalue())
            return {}

        process = Mock()
        process.poll.return_value = None
        popen, cleanup = Mock(return_value=process), Mock()
        captured = io.StringIO()
        with (
            patch.dict(
                fn.__globals__,
                {"request": request, "assert_port_free": Mock(), "stop_owned_service": cleanup, "wait_profile": Mock()},
            ),
            patch("subprocess.Popen", popen),
            contextlib.redirect_stdout(captured),
        ):
            result = fn(args, self.source, "cpp", {})
        self.assertEqual(
            sequence,
            [
                "/health",
                "/status",
                *["/v1/completions"] * 2,
                "/start_profile",
                "/start",
                *["/v1/completions"] * 3,
                "/stop",
                "/stop_profile",
            ],
        )
        cleanup.assert_called_once_with(process)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(len(result["digests"]), 3)
        self.assertIn("INFERENCE cpp: callbacks=5", captured.getvalue())

    def test_request_failure_still_cleans_up_owned_service(self):
        fn = self.module["inference"]
        args = types.SimpleNamespace(
            npu=4, port=8155, timing_port=18155, startup_timeout=20, profile_run_dir=Path("profiles/test")
        )
        process = Mock()
        process.poll.return_value = None
        cleanup = Mock()
        with (
            patch.dict(
                fn.__globals__,
                {
                    "request": Mock(side_effect=RuntimeError("injected failure")),
                    "assert_port_free": Mock(),
                    "stop_owned_service": cleanup,
                },
            ),
            patch("subprocess.Popen", return_value=process),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "injected failure"),
        ):
            fn(args, self.source, "python", {})
        cleanup.assert_called_once_with(process)

    def test_profile_is_stopped_when_measured_request_fails(self):
        fn = self.module["inference"]
        args = types.SimpleNamespace(
            npu=None,
            port=8155,
            timing_port=18155,
            startup_timeout=20,
            profile_run_dir=Path("profiles/test"),
            profile_timeout=10,
            request_timeout=10,
            warmup_requests=0,
            requests=1,
            max_tokens=5,
        )
        sequence = []

        def request(opener, port, path, payload=None, timeout=None):
            sequence.append(path)
            if path == "/status":
                return dict(backend="python", load_backend="python")
            if path == "/v1/completions":
                raise RuntimeError("measured request failed")
            return {}

        process = Mock()
        process.poll.return_value = None
        cleanup, export = Mock(), Mock()
        with (
            patch.dict(
                fn.__globals__,
                {"request": request, "assert_port_free": Mock(), "stop_owned_service": cleanup, "wait_profile": export},
            ),
            patch("subprocess.Popen", return_value=process),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "measured request failed"),
        ):
            fn(args, self.source, "python", {})
        self.assertEqual(sequence[-2:], ["/stop", "/stop_profile"])
        export.assert_called_once()
        cleanup.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
