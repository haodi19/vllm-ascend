# SPDX-License-Identifier: Apache-2.0
import contextlib
import errno
import importlib.util
import io
import json
import shlex
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
spec = importlib.util.spec_from_file_location("native_runner", ROOT / "script/run_native_moe_compare.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeRunnerTests(unittest.TestCase):
    def test_pipeline_suite_preserves_baseline_and_uses_one_service_per_case(self):
        with tempfile.TemporaryDirectory() as temp:
            code, output, launched, count, profiles = self.run_mock_comparison(Path(temp), "pipeline")
            self.assertEqual(code, 0)
            self.assertEqual(launched, ["python", "native", "native_pipeline"])
            self.assertEqual(count, 9)
            self.assertEqual(profiles.call_count, 3)
            self.assertTrue((output / "native_pipeline/timing_result.json").is_file())
        self.assertEqual(runner.selected_cases("all"), ("python", "native", "native_full"))

    def run_mock_comparison(
        self, directory, case="both", *, varied_outputs=False, marker_leak=False, profile_error=False
    ):
        """Exercise the complete runner without processes, Torch or NPU."""
        library = directory / "libhost_sync_device.so"
        library.write_bytes(b"fixture, never loaded")
        output = directory / "result"
        argv = [
            "runner",
            "--case",
            case,
            "--npu",
            "6",
            "--skip-build",
            "--library",
            str(library),
            "--warmup-requests",
            "1",
            "--requests",
            "2",
            "--output-dir",
            str(output),
        ]
        sample = dict(
            digest="same",
            usage=dict(completion_tokens=5),
            completion_tokens=5,
            content_chunks=3,
            decode_us=2000,
            ttft_us=3000,
            output_latency_us=5000,
            tpot_us=500,
            http_wall_us=5100,
        )
        launched = []
        services = []
        stopped = []

        def start(command, **kwargs):
            child = Mock(pid=1234, poll=lambda: None, wait=lambda **kwargs: 0)
            if "vllm.entrypoints.cli.main" in command:
                services.append(child)
                config = json.loads(command[command.index("--additional-config") + 1])["expert_offload_config"]
                launched.append(config["host_update_backend"])
                if config["host_update_backend"] in ("native", "native_full", "native_pipeline"):
                    kwargs["stdout"].write("[MOE-NATIVE] first native graph task completed without Python\n")
                    kwargs["stdout"].write(
                        "[MOE-NATIVE] SCALE_MEMORY "
                        + json.dumps(
                            dict(
                                policy="shared_final_layout_v1",
                                layers=43,
                                duplicate_bytes=0,
                                reused_bytes=100,
                                replaced_bytes=0,
                                derived_bytes=0,
                            )
                        )
                        + "\n"
                    )
                    kwargs["stdout"].write(
                        "[MOE-NATIVE] FULL_SUBMIT ready;\n[MOE-NATIVE] FULL_SUBMIT executed=prefill;\n"
                    )
                    kwargs["stdout"].flush()
            return child

        def capture(*args, **kwargs):
            self.assertIs(kwargs["live_service"], services[-1])
            self.assertNotIn(services[-1], stopped)
            self.assertEqual(requests.call_count, len(services) * 3)
            self.assertEqual(kwargs["timing_status"]["mstx_marker_attempts"], 0)
            if profile_error:
                raise TimeoutError("profile export failed")
            return {}

        opener = Mock()
        opener.open.side_effect = lambda *args, **kwargs: contextlib.nullcontext(
            io.BytesIO(b'{"data":[{"id":"test-model"}]}')
        )
        streamed = [dict(sample) for _ in range(9)]
        if varied_outputs:
            streamed[2]["digest"] = "different"
        status = dict(
            pid=4321,
            active=False,
            mstx_active=False,
            mstx_enabled=True,
            mstx_marker_attempts=0,
            inflight=0,
            native=True,
            native_count=0,
            native_marker_errors=0,
        )
        statuses = [dict(status) for _ in range(9)]
        if marker_leak:
            statuses[2]["mstx_marker_attempts"] = 1
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "platform", "linux"),
            patch.object(runner.urllib.request, "build_opener", return_value=opener),
            patch.object(runner.subprocess, "Popen", side_effect=start),
            patch.object(runner, "streaming_completion", side_effect=streamed) as requests,
            patch.object(runner, "wait_port_free"),
            patch.object(runner, "stop_process_group", side_effect=stopped.append),
            patch.object(runner, "check_host_memory"),
            patch.object(runner, "after_service_memory"),
            patch.object(runner, "log_memory"),
            patch.object(runner, "control_request", side_effect=statuses),
            patch.object(runner, "ensure_profile_storage"),
            patch.object(runner, "profile_cases", side_effect=capture) as profiles,
            patch.object(runner, "report_profiles"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = runner.main()
        self.assertEqual(stopped, services)
        return code, output, launched, requests.call_count, profiles

    def test_marker_leak_rejects_timing_before_profile_or_result_acceptance(self):
        for case in ("python", "native", "native_full"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                code, output, launched, count, profiles = self.run_mock_comparison(Path(temp), case, marker_leak=True)
                self.assertEqual(code, 1)
                self.assertEqual(launched, [case])
                self.assertEqual(count, 3)
                profiles.assert_not_called()
                self.assertFalse((output / case / "timing_result.json").exists())

    def test_saved_gated_timing_requires_unchanged_native_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            code, output, _, _, _ = self.run_mock_comparison(Path(temp), "native")
            self.assertEqual(code, 0)
            path = output / "native/timing_result.json"
            record = json.loads(path.read_text())
            record["instrumentation"]["after"]["native_count"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "measurements were enabled"):
                runner.load_completed_timing_run(output, 8157, ("native",))

    def test_profile_failure_preserves_completed_timing_and_cleans_service_once(self):
        with tempfile.TemporaryDirectory() as temp:
            code, output, launched, count, profiles = self.run_mock_comparison(Path(temp), "native", profile_error=True)
            self.assertEqual(code, 1)
            self.assertEqual(launched, ["native"])
            self.assertEqual(count, 3)
            profiles.assert_called_once()
            records, _, _, _ = runner.load_completed_timing_run(output, 8157, ("native",))
            self.assertEqual(records["native"]["requests"], 2)
            self.assertFalse((output / "native/native_partial.json").exists())
            self.assertIn("PROFILE_FAILED native", (output / "run_details.log").read_text(encoding="utf-8"))

    def test_independent_complete_runs_preserve_timing_and_selected_profile(self):
        for case in ("python", "native", "native_full"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                code, output, launched, requests, profiles = self.run_mock_comparison(Path(temporary), case)
                self.assertEqual(code, 0)
                self.assertEqual(launched, [case])
                self.assertEqual(requests, 3)  # warmup + two timed requests
                self.assertEqual(list(profiles.call_args.args[1]), [case])
                self.assertFalse(profiles.call_args.args[0].skip_profile)
                text = (output / "run_details.log").read_text(encoding="utf-8")
                rows = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT ")]
                self.assertEqual([row["case"] for row in rows], [case])
                self.assertEqual(rows[0]["requests"], 2)
                self.assertEqual(rows[0]["output_match_scope"], "within_case")
                self.assertIn("cross-version output parity and speedup not evaluated", text)
                self.assertNotIn("mean change=", text)
                other = "python" if case == "native" else "native"
                self.assertFalse((output / other).exists())
                self.assertTrue((output / case / "inference.log").is_file())
                self.assertTrue((output / case / "command.json").is_file())
                self.assertFalse((output / f"{case}_command.json").exists())
                self.assertEqual(json.loads((output / case / "timing_result.json").read_text()), rows[0])
                # A single-case directory can also resume just its own profile.
                records, commands, _, tokens = runner.load_completed_timing_run(output, 8157, (case,))
                self.assertEqual(list(commands), [case])
                self.assertEqual(tokens, 5)
                self.assertEqual(records[case]["requests"], 2)

    def test_three_way_run_and_profile_keep_distinct_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            code, output, launched, count, profiles = self.run_mock_comparison(Path(temp), "all")
            self.assertEqual(code, 0)
            self.assertEqual(launched, ["python", "native", "native_full"])
            self.assertEqual(count, 9)
            self.assertEqual([next(iter(call.args[1])) for call in profiles.call_args_list], launched)
            text = (output / "run_details.log").read_text(encoding="utf-8")
            self.assertIn("native_full vs native", text)
            self.assertIn("native_full vs python", text)
            self.assertEqual({p.name for p in output.iterdir() if p.is_dir()}, {*launched, "profile"})
            for case in launched:
                row = json.loads((output / case / "timing_result.json").read_text())
                self.assertEqual(row["case"], case)
                command = json.loads((output / case / "command.json").read_text())
                config = json.loads(command[command.index("--additional-config") + 1])
                self.assertEqual(config["expert_offload_config"]["host_update_backend"], case)

    def test_independent_cases_append_to_same_output_without_replacing_previous_results(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            previous = {}
            for case in ("python", "native", "native_full"):
                code, output, launched, _, _ = self.run_mock_comparison(directory, case)
                self.assertEqual(code, 0)
                self.assertEqual(launched, [case])
                for path, content in previous.items():
                    self.assertEqual(path.read_bytes(), content)
                previous.update({p: p.read_bytes() for p in (output / case).iterdir() if p.is_file()})
            records, commands, _, _ = runner.load_completed_timing_run(
                output, 8157, ("python", "native", "native_full")
            )
            self.assertEqual(set(records), {"python", "native", "native_full"})
            self.assertEqual(set(commands), set(records))
            before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                self.run_mock_comparison(directory, "python")
            self.assertEqual(before, {p: p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_shared_output_rejects_saved_profile_legacy_command_and_completed_result(self):
        for evidence in ("profile", "legacy", "summary"):
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                if evidence == "profile":
                    (root / "profile/native").mkdir(parents=True)
                elif evidence == "legacy":
                    (root / "native_command.json").write_text("[]")
                else:
                    (root / "summary.txt").write_text('RESULT {"case":"native"}\n')
                with self.assertRaises(FileExistsError):
                    runner.prepare_output(root, ("native",), append_single_case=True)
                self.assertFalse((root / "native").exists())

    def test_combined_run_still_launches_both_and_compares(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, output, launched, requests, profiles = self.run_mock_comparison(Path(temporary))
            self.assertEqual(code, 0)
            self.assertEqual(launched, ["python", "native"])
            self.assertEqual(requests, 6)
            self.assertEqual([next(iter(call.args[1])) for call in profiles.call_args_list], launched)
            text = (output / "run_details.log").read_text(encoding="utf-8")
            self.assertIn("decode mean change=0.000%", text)
            self.assertNotIn("SINGLE_CASE_TIMING_COMPLETE", text)

    def test_single_case_rejects_inconsistent_outputs_before_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, output, _, _, profiles = self.run_mock_comparison(Path(temporary), "python", varied_outputs=True)
            self.assertEqual(code, 1)
            profiles.assert_not_called()
            text = (output / "run_details.log").read_text(encoding="utf-8")
            self.assertIn("outputs/usage differ between timed requests", text)
            self.assertNotIn("RESULT ", text)
            self.assertTrue((output / "python/python_partial.json").is_file())
            self.assertFalse((output / "python/timing_result.json").exists())

    def saved_timing_fixture(self, directory, tokens=128):
        library = directory / "libhost_sync_device.so"
        library.write_bytes(b"test library; never loaded")
        source = "vllm serve /model --port 8155 --additional-config " + shlex.quote(
            '{"expert_offload_config":{"num_device_experts":24}}'
        )
        records = []
        for case in ("python", "native", "native_full"):
            command = runner.service_command(source, case, library, 8156)
            (directory / f"{case}_command.json").write_text(json.dumps(command), encoding="utf-8")
            sample = dict(
                digest="same",
                usage=dict(completion_tokens=tokens),
                completion_tokens=tokens,
                content_chunks=3,
                decode_us=2000,
                ttft_us=3000,
                output_latency_us=5000,
                tpot_us=2000 / (tokens - 1),
                http_wall_us=5100,
            )
            records.append(dict(case=case, requests=2, output_match=True, stream_samples=[sample, sample]))
        (directory / "summary.txt").write_text(
            "\n".join("RESULT " + json.dumps(row) for row in records) + "\nFAILED profile disk full\n", encoding="utf-8"
        )
        return records

    def test_profile_continuation_keeps_saved_settings_and_does_not_rebuild_or_retime(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "timing"
            source.mkdir()
            self.saved_timing_fixture(source)
            output = Path(temporary) / "continued"
            argv = [
                "runner",
                "--npu",
                "7",
                "--port",
                "8158",
                "--min-profile-free-gib",
                "5",
                "--profile-only-from",
                str(source),
                "--test-script",
                "must-not-read.sh",
                "--output-dir",
                str(output),
            ]
            original = (source / "summary.txt").read_bytes()
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "platform", "linux"),
                patch("run_verified_cpp_tests.ensure_profile_storage") as storage,
                patch("native_moe_profile.profile_cases") as profiles,
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(runner.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 0)
            popen.assert_not_called()
            run.assert_not_called()
            args, commands = profiles.call_args.args[:2]
            self.assertEqual(args.max_tokens, 128)
            self.assertEqual(args.min_profile_free_gib, 5)
            storage.assert_called_once_with(
                output.resolve(), stage="before profile continuation; no model started", min_free_gib=5
            )
            self.assertEqual(set(commands), {"python", "native"})
            for command in commands.values():
                self.assertEqual(command[command.index("--port") + 1], "8158")
                config = json.loads(command[command.index("--additional-config") + 1])
                self.assertEqual(config["expert_offload_config"]["num_device_experts"], 24)
            self.assertEqual((source / "summary.txt").read_bytes(), original)
            summary = (output / "run_details.log").read_text(encoding="utf-8")
            self.assertIn("TIMING_SOURCE=", summary)
            self.assertIn("| native |", summary)
            for case, command in commands.items():
                self.assertEqual(json.loads((output / case / "command.json").read_text()), command)
                self.assertEqual(json.loads((output / case / "timing_result.json").read_text())["case"], case)

    def test_profile_continuation_rejects_incomplete_and_mismatched_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = self.saved_timing_fixture(directory)
            summary = directory / "summary.txt"
            for invalid in (records[:1], records + records[:1]):
                summary.write_text("\n".join("RESULT " + json.dumps(row) for row in invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_completed_timing_run(directory, 8156)
            records[1]["stream_samples"][0]["digest"] = "different"
            summary.write_text("\n".join("RESULT " + json.dumps(row) for row in records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outputs/usage"):
                runner.load_completed_timing_run(directory, 8156)

    def test_profile_continuation_runs_only_selected_case(self):
        for case in ("python", "native", "native_full"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "timing"
                source.mkdir()
                self.saved_timing_fixture(source, tokens=5)
                output = Path(temporary) / "single"
                argv = [
                    "runner",
                    "--npu",
                    "7",
                    "--profile-only-from",
                    str(source),
                    "--profile-case",
                    case,
                    "--output-dir",
                    str(output),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(sys, "platform", "linux"),
                    patch("run_verified_cpp_tests.ensure_profile_storage"),
                    patch("native_moe_profile.profile_cases") as profiles,
                    patch.object(runner.subprocess, "Popen") as popen,
                    patch.object(runner.subprocess, "run") as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(runner.main(), 0)
                popen.assert_not_called()
                run.assert_not_called()
                args, commands = profiles.call_args.args[:2]
                self.assertEqual(list(commands), [case])
                self.assertEqual(args.max_tokens, 5)
                summary = (output / "run_details.log").read_text(encoding="utf-8")
                rows = [json.loads(line[7:]) for line in summary.splitlines() if line.startswith("RESULT ")]
                self.assertEqual([row["case"] for row in rows], [case])
                self.assertIn(f"cases={case}", summary)
                self.assertIn("TIMING_SOURCE=", summary)

    def test_single_case_selector_requires_profile_continuation(self):
        with (
            patch.object(sys, "argv", ["runner", "--npu", "7", "--profile-case", "native", "--output-dir", "unused"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            runner.main()
        self.assertEqual(error.exception.code, 2)

    def test_profile_continuation_rejects_missing_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.saved_timing_fixture(directory)
            (directory / "libhost_sync_device.so").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "handshake library"):
                runner.load_completed_timing_run(directory, 8156)

    def test_profile_continuation_checks_disk_before_starting_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.saved_timing_fixture(directory)
            output = directory / "continuation"
            output.mkdir()
            args = SimpleNamespace(
                profile_only_from=directory,
                profile_case="both",
                port=8156,
                npu=7,
                warmup_requests=5,
                profile_requests=1,
                min_profile_free_gib=20,
            )
            with (
                patch("run_verified_cpp_tests.ensure_profile_storage", side_effect=RuntimeError("disk full")),
                patch("native_moe_profile.profile_cases") as profiles,
                self.assertRaisesRegex(RuntimeError, "disk full"),
            ):
                runner.continue_profile(args, output, {}, lambda _: None)
            profiles.assert_not_called()

    def test_handshake_rebuild_uses_fresh_directory_and_retains_old_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "persistent_build"
            parent.mkdir()
            stale = parent / "handshake.cpp.o"
            stale.write_bytes(b"previous merge output")
            first = runner.fresh_handshake_build(parent)
            (first / "handshake.cpp.o").write_bytes(b"first build evidence")
            # An occupied filename is also skipped; it is never replaced.
            (parent / "build_02").write_text("reserved", encoding="utf-8")
            second = runner.fresh_handshake_build(parent)
            self.assertEqual(first.name, "build_01")
            self.assertEqual(second.name, "build_03")
            self.assertEqual(list(second.iterdir()), [])
            self.assertEqual(stale.read_bytes(), b"previous merge output")
            self.assertEqual((first / "handshake.cpp.o").read_bytes(), b"first build evidence")
            for build in (first, second):
                command = runner.handshake_configure_command(build, Path("cann root"), "ascend950pr_957c")
                self.assertEqual(command[command.index("-B") + 1], str(build))
                words = runner.service_command(
                    (ROOT / "script/test_graph2.sh").read_text(encoding="utf-8"),
                    "native",
                    build / "lib/libhost_sync_device.so",
                    8156,
                )
                config = json.loads(words[words.index("--additional-config") + 1])
                self.assertEqual(
                    config["expert_offload_config"]["host_sync_library"], str(build / "lib/libhost_sync_device.so")
                )

    def test_fresh_handshake_build_creates_missing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "build" / "handshake"
            created = runner.fresh_handshake_build(parent)
            self.assertEqual(created.parent, parent.resolve())
            self.assertEqual(created.name, "build_01")
            self.assertTrue(created.is_dir())

    def test_port_check_rejects_live_listener_and_accepts_released_port(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, f"Port {port} is unavailable"):
                runner.wait_port_free(port)
        runner.wait_port_free(port)

    @unittest.skipUnless(sys.platform == "linux", "Linux TCP TIME_WAIT semantics")
    def test_port_check_accepts_server_time_wait_without_live_listener(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            client.connect(("127.0.0.1", port))
            accepted, _ = server.accept()
            # Server actively closes, leaving its local port in TIME_WAIT.
            accepted.shutdown(socket.SHUT_WR)
            self.assertEqual(client.recv(1), b"")
            client.shutdown(socket.SHUT_WR)
            self.assertEqual(accepted.recv(1), b"")
            accepted.close()
        with socket.socket() as old_probe:
            with self.assertRaises(OSError) as raised:
                old_probe.bind(("127.0.0.1", port))
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
        runner.wait_port_free(port)

    def test_port_check_waits_for_listener_but_has_deadline(self):
        probe = Mock()
        probe.connect_ex.side_effect = [0, 1]
        sock = Mock()
        sock.__enter__ = Mock(return_value=probe)
        sock.__exit__ = Mock(return_value=False)
        with patch.object(runner.socket, "socket", return_value=sock), patch.object(runner.time, "sleep") as sleep:
            runner.wait_port_free(8156, timeout=10)
            sleep.assert_called_once()
        probe.connect_ex.side_effect = None
        probe.connect_ex.return_value = 0
        with (
            patch.object(runner.socket, "socket", return_value=sock),
            patch.object(runner.time, "monotonic", side_effect=[0, 1, 11]),
            patch.object(runner.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "after 10s"):
                runner.wait_port_free(8156, timeout=10)
            sleep.assert_called_once()

    def test_busy_port_fails_before_build_or_npu_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            argv = ["runner", "--npu", "1", "--output-dir", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "platform", "linux"),
                patch.object(runner, "wait_port_free", side_effect=RuntimeError("Port 8156 is unavailable")),
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(runner.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 1)
            popen.assert_not_called()
            run.assert_not_called()
            self.assertIn("Port 8156 is unavailable", (output / "run_details.log").read_text(encoding="utf-8"))

    def test_service_failure_keeps_partial_samples_and_engine_error(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "python.log"
            log.write_text("EngineCore root cause: device error\n", encoding="utf-8")
            lines = []
            sample = {"decode_us": 1000, "completion_tokens": 5}
            runner.report_service_failure(
                "python", log, [sample], 6, Mock(poll=lambda: None), ValueError("SSE error"), lines.append
            )
            partial = json.loads((Path(directory) / "python_partial.json").read_text(encoding="utf-8"))
            self.assertEqual(partial["status"], "incomplete")
            self.assertEqual(partial["stream_samples"], [sample])
            self.assertIn("completed_timed_requests=1", lines[0])
            self.assertIn("EngineCore root cause: device error", lines)
            self.assertIn("SERVICE_LOG_BEFORE_CLEANUP=", "\n".join(lines))

    def test_repository_launch_script_with_and_without_final_newline(self):
        source = (ROOT / "script/test_graph2.sh").read_text(encoding="utf-8").rstrip("\n")
        for case in ("python", "native", "native_full"):
            expected = runner.service_command(source + "\n", case, "/lib/handshake.so", 8156)
            self.assertEqual(runner.service_command(source, case, "/lib/handshake.so", 8156), expected)
            self.assertEqual(expected[:4], [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"])
            config = json.loads(expected[expected.index("--additional-config") + 1])
            self.assertEqual(config["expert_offload_config"]["host_update_backend"], case)
            self.assertNotIn("--profiler-config", expected)
            self.assertTrue(all(word.strip() for word in expected))

    def test_eof_continuation_preserves_quoted_backslashes(self):
        config = {"expert_offload_config": {"hot_experts_file": "path\\with\\escapes.json"}}
        source = 'vllm serve "/model with spaces" --port 8155 --additional-config '
        source += shlex.quote(json.dumps(config))
        expected = runner.service_command(source, "native", "/lib/test.so", 8156)
        for ending in (" \\", " \\\n", " \\\n# comment at EOF"):
            self.assertEqual(runner.service_command(source + ending, "native", "/lib/test.so", 8156), expected)
        with self.assertRaisesRegex(ValueError, "No closing quotation"):
            runner.service_command(source + " 'unfinished", "native", "/lib/test.so", 8156)

    def test_bad_launch_script_fails_before_build_or_npu_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.sh"
            source.write_text("vllm serve '/unfinished", encoding="utf-8")
            output = Path(directory) / "result"
            argv = ["runner", "--npu", "1", "--test-script", str(source), "--output-dir", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "platform", "linux"),
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(runner.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 1)
            popen.assert_not_called()
            run.assert_not_called()
            self.assertIn("No closing quotation", (output / "run_details.log").read_text(encoding="utf-8"))

    def test_handshake_configure_preserves_soc_and_sets_build_type(self):
        for soc in ("ascend950pr_957c", "ascend950dt_9582"):
            command = runner.handshake_configure_command(Path("build with spaces"), Path("cann root"), soc)
            self.assertIn("-DCMAKE_BUILD_TYPE=Release", command)
            self.assertIn(f"-DSOC_VERSION={soc}", command)
            self.assertIn("-DASCEND_COMPUTE_UNIT=ascend950", command)
            self.assertEqual(command[command.index("-B") + 1], "build with spaces")
            self.assertIn(f"-DASCEND_CANN_PACKAGE_PATH={Path('cann root').resolve()}", command)

    def test_failed_step_summary_contains_failing_command_not_earlier_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "build.log"
            earlier = b"previous successful build\n" * 500
            failure = b"COMMAND bash merge_mix_obj.sh --build-type\nError 1\n"
            log.write_bytes(earlier + failure)
            lines = []
            runner.report_failed_step(log, len(earlier), lines.append)
            self.assertIn("COMMAND bash merge_mix_obj.sh --build-type", lines)
            self.assertIn("Error 1", lines)
            self.assertNotIn("previous successful build", "\n".join(lines))
            log.write_bytes(b"x" * 40000 + b"\nlast error\n")
            lines.clear()
            runner.report_failed_step(log, 0, lines.append)
            self.assertIn("last error", lines)
            self.assertLess(len("\n".join(lines)), 33000)

    def test_preserves_workload_and_separates_case(self):
        config = {"expert_offload_config": {"num_device_experts": 24, "expert_prefetch_enabled": True}}
        source = "rm -rf do-not-run\nvllm serve /model --port 8155 --speculative-config "
        source += shlex.quote('{"num_speculative_tokens":2}')
        source += " --additional-config " + shlex.quote(json.dumps(config))
        source += " --profiler-config '{}'"
        for case in ("python", "native", "native_full"):
            words = runner.service_command(source, case, "/lib/handshake.so", 8156)
            self.assertNotIn("rm", words)
            self.assertNotIn("--profiler-config", words)
            self.assertEqual(words[words.index("--port") + 1], "8156")
            self.assertEqual(json.loads(words[words.index("--speculative-config") + 1])["num_speculative_tokens"], 2)
            actual = json.loads(words[words.index("--additional-config") + 1])["expert_offload_config"]
            self.assertEqual(actual["num_device_experts"], 24)
            self.assertEqual(actual["host_update_backend"], case)
            self.assertTrue(actual["host_sync_diagnose_prefetch_wait_at_join"])
            self.assertEqual(actual["callback_timing_port"], 0)
            self.assertFalse(actual["callback_mstx_enabled"])

    def test_streaming_timing_excludes_ttft_and_handles_single_token(self):
        sample = dict(
            decode_us=6000,
            ttft_us=10000,
            tpot_us=2000,
            output_latency_us=16000,
            http_wall_us=16100,
            completion_tokens=4,
            content_chunks=2,
        )
        timing = runner.summarize_streams([sample])
        # Two SSE chunks do not divide the three output-token intervals.
        self.assertEqual(runner.timing_row("native", timing, 16.1), "| native | 6.000 | 10.000 | 2.000 | 16.100 |")
        sample.update(decode_us=0, tpot_us=None, completion_tokens=1, content_chunks=1)
        timing = runner.summarize_streams([sample])
        self.assertEqual(runner.timing_row("native", timing, 16.1), "| native | 0.000 | 10.000 | undefined | 16.100 |")

    def test_refuses_missing_serve(self):
        with self.assertRaises(ValueError):
            runner.service_command("echo no model", "native", "/lib/test.so", 8156)


if __name__ == "__main__":
    unittest.main()
