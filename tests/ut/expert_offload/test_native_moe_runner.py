# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
spec = importlib.util.spec_from_file_location("native_runner", ROOT / "script/run_native_moe_compare.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeRunnerTests(unittest.TestCase):
    def test_repository_launch_script_with_and_without_final_newline(self):
        source = (ROOT / "script/test_graph2.sh").read_text(encoding="utf-8").rstrip("\n")
        for case in ("python", "native"):
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
            self.assertIn("No closing quotation", (output / "summary.txt").read_text(encoding="utf-8"))

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
        for case in ("python", "native"):
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
