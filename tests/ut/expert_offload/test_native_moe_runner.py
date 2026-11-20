# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import shlex
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
spec = importlib.util.spec_from_file_location("native_runner", ROOT / "script/run_native_moe_compare.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeRunnerTests(unittest.TestCase):
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

    def test_refuses_missing_serve(self):
        with self.assertRaises(ValueError):
            runner.service_command("echo no model", "native", "/lib/test.so", 8156)


if __name__ == "__main__":
    unittest.main()
