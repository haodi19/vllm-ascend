# SPDX-License-Identifier: Apache-2.0
"""Retry ordering, bounded deletion and process ownership without a model."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import run_native_moe_memory_retry as retry  # noqa: E402


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.owner = "test-owner"
        (self.root / retry.OWNER_FILE).write_text(json.dumps(dict(owner=self.owner)))
        self.args = SimpleNamespace(
            output_root=self.root, owner=self.owner, case="native_pipeline", npu=6, port=8156, timing_port=18156
        )
        self.node = dict(node=1, free_mb=150000, cpus=[64, 65], workers=(64, 65))

    def saved_attempt(self, tokens, success, tag):
        directory = self.root / f"token{tokens}"
        case = directory / "native_pipeline"
        case.mkdir(parents=True)
        (case / "inference.log").write_text(tag)
        (case / "run_status.json").write_text(json.dumps(dict(exit_code=0 if success else 1)))
        (case / "timing_result.json").write_text(
            json.dumps(dict(output_match=True, stream_samples=[dict(completion_tokens=tokens)] * 20))
        )
        (directory / "summary.txt").write_text("summary " + tag)
        (directory / "profile").mkdir()
        (directory / "profile/trace.json").write_text(tag)

    def test_parse_and_strict_threshold_with_permitted_nodes(self):
        nodes = retry.parse_nodes("""available: 2 nodes (0-1)
node 0 cpus: 0 1
node 0 size: 250000 MB
node 0 free: 100000 MB
node 1 cpus: 64 65
node 1 free: 100001 MB
node distances:
node 0 1
""")
        with patch.object(retry, "resolve_host_cpus", side_effect=lambda _, allowed: tuple(allowed)):
            self.assertEqual(retry.choose_node(nodes, {0, 1}, {0, 1, 64, 65})["node"], 1)
            self.assertIsNone(retry.choose_node(nodes, {0}, {0, 1, 64, 65}))
            self.assertIsNone(retry.choose_node(nodes, {0, 1}, {0, 1}))
        self.assertEqual(retry.cpu_list("0-2,4"), {0, 1, 2, 4})
        with self.assertRaises(ValueError):
            retry.parse_nodes("node 0 cpus: 0 1")

    def test_wait_retry_5_then_128_preserves_successful_5(self):
        attempts = []
        stop = threading.Event()
        stop.wait = Mock(return_value=False)

        def run(args, tokens, node, _stop):
            self.assertFalse((self.root / f"token{tokens}").exists())
            attempts.append(tokens)
            success = attempts.count(tokens) == 2
            if tokens == 128:
                self.assertEqual((self.root / "token5/profile/trace.json").read_text(), "5-success")
            self.saved_attempt(tokens, success, f"{tokens}-{'success' if success else 'failed'}")
            return 0 if success else 1

        def drop():
            failed_tokens = attempts[-1]
            self.assertFalse((self.root / f"token{failed_tokens}").exists())
            if failed_tokens == 128:
                self.assertEqual((self.root / "token5/profile/trace.json").read_text(), "5-success")
            return failed_tokens == 128  # A failed cache drop must not prevent retrying token5.

        with (
            patch.object(
                retry,
                "read_available_node",
                side_effect=[
                    None,
                    subprocess.TimeoutExpired("numactl", 15),
                    self.node,
                    self.node,
                    self.node,
                    self.node,
                ],
            ),
            patch.object(retry, "run_attempt", side_effect=run),
            patch.object(retry, "drop_page_cache_after_failure", side_effect=drop) as drop_cache,
            patch.object(retry, "rebuild"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(retry.retry_loop(self.args, stop), 0)
        self.assertEqual(attempts, [5, 5, 128, 128])
        self.assertEqual(drop_cache.call_count, 2)
        state = json.loads((self.root / "retry_state.json").read_text())
        self.assertEqual(state["completed"], [5, 128])
        self.assertEqual(state["attempts"], {"5": 2, "128": 2})
        self.assertEqual((self.root / "token5/profile/trace.json").read_text(), "5-success")
        self.assertEqual((self.root / "token128/profile/trace.json").read_text(), "128-success")
        self.assertIn("128-failed", (self.root / "last_failure.log").read_text())

    def test_cleanup_cannot_remove_sibling_unknown_directory_or_wrong_owner(self):
        self.saved_attempt(5, True, "keep")
        self.saved_attempt(128, False, "remove")
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            retry.clear_failed_token(self.root, "wrong", 128)
        with self.assertRaises(ValueError):
            retry.clear_failed_token(self.root, self.owner, 129)
        with patch.object(retry.os.path, "ismount", return_value=True), self.assertRaisesRegex(RuntimeError, "mounted"):
            retry.clear_failed_token(self.root, self.owner, 128)
        retry.clear_failed_token(self.root, self.owner, 128)
        self.assertTrue((self.root / "token5/profile/trace.json").is_file())
        self.assertFalse((self.root / "token128").exists())

    def test_symlink_target_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("unchanged")
        try:
            (self.root / "token128").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation not allowed")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            retry.clear_failed_token(self.root, self.owner, 128)
        self.assertEqual((outside / "keep").read_text(), "unchanged")

    def test_interruption_keeps_partial_results_and_stops(self):
        stop = threading.Event()

        def run(*_):
            self.saved_attempt(5, False, "interrupted")
            stop.set()
            return -15

        with (
            patch.object(retry, "read_available_node", return_value=self.node),
            patch.object(retry, "run_attempt", side_effect=run),
            patch.object(retry, "clear_failed_token") as clear,
            patch.object(retry, "drop_page_cache_after_failure") as drop_cache,
        ):
            self.assertEqual(retry.retry_loop(self.args, stop), 130)
        clear.assert_not_called()
        drop_cache.assert_not_called()
        self.assertTrue((self.root / "token5/profile/trace.json").is_file())

    def test_unverified_cleanup_never_deletes_or_retries(self):
        self.saved_attempt(128, False, "leftover")
        with (
            patch.object(retry, "read_available_node", return_value=self.node),
            patch.object(retry, "run_attempt", side_effect=retry.ProcessCleanupError("still running")),
            patch.object(retry, "clear_failed_token") as clear,
            patch.object(retry, "drop_page_cache_after_failure") as drop_cache,
            self.assertRaises(retry.ProcessCleanupError),
        ):
            retry.retry_loop(self.args, threading.Event())
        clear.assert_not_called()
        drop_cache.assert_not_called()

    def test_cache_drop_writes_only_one(self):
        target = self.root / "drop_caches"
        with patch.object(retry, "DROP_CACHES_PATH", target), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(retry.drop_page_cache_after_failure())
        self.assertEqual(target.read_bytes(), b"1\n")
        self.assertIn("DROP_CACHES_PASS", output.getvalue())

    def test_cache_drop_permission_failure_is_reported_without_stopping_retry(self):
        target = Mock()
        target.write_bytes.side_effect = PermissionError("read-only proc sysctl")
        with patch.object(retry, "DROP_CACHES_PATH", target), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(retry.drop_page_cache_after_failure())
        self.assertIn("DROP_CACHES_FAILED", output.getvalue())

    def test_command_preserves_workload_and_limits_cpu_memory_to_selected_node(self):
        command = retry.attempt_command(self.args, 128, self.node)
        self.assertIn("--membind=1", command)
        self.assertIn("--physcpubind=64,65", command)
        self.assertEqual(command[command.index("--host-cpus") + 1], "64,65")
        self.assertEqual(command[command.index("--max-tokens") + 1], "128")
        self.assertEqual(command[command.index("--requests") + 1], "20")
        self.assertEqual(command[-1], str(self.root / "token128"))
        self.assertNotIn("--skip-profile", command)

    def test_pid_reuse_is_never_signalled(self):
        with (
            patch.object(retry, "process_identity", return_value="new-process"),
            patch.object(retry.os, "kill") as kill,
        ):
            retry.signal_owned(999, "old-process", 15)
        kill.assert_not_called()

    def test_missing_or_incomplete_results_are_not_success(self):
        with self.assertRaises(OSError):
            retry.verify_success(self.root, 5, "native_pipeline")
        self.saved_attempt(5, True, "good")
        retry.verify_success(self.root, 5, "native_pipeline")
        path = self.root / "token5/native_pipeline/timing_result.json"
        result = json.loads(path.read_text())
        result["stream_samples"][0]["completion_tokens"] = 4
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            retry.verify_success(self.root, 5, "native_pipeline")

    def test_cleanup_is_required_even_when_runner_exits_zero(self):
        process = Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            patch.object(retry, "attempt_command", return_value=["dummy"]),
            patch.object(retry.subprocess, "Popen", return_value=process),
            patch.object(retry, "clean_owned_processes", side_effect=OSError("cannot inspect children")),
            self.assertRaises(retry.ProcessCleanupError),
        ):
            retry.run_attempt(self.args, 5, self.node, threading.Event())

    @unittest.skipUnless(sys.platform == "linux", "Linux subreaper and /proc")
    def test_detached_orphan_is_reaped_in_isolated_supervisor(self):
        child_code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'], start_new_session=True)"
        )
        code = """
import ctypes,sys,threading
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import run_native_moe_memory_retry as r
assert ctypes.CDLL(None).prctl(r.PR_SET_CHILD_SUBREAPER,1,0,0,0) == 0
with patch.object(r,'attempt_command',return_value=[sys.executable,'-c',sys.argv[2]]):
    assert r.run_attempt(None,None,None,threading.Event()) == 0
assert r.descendants() == {}
print('CLEANUP_PASS')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(retry.ROOT / "script"), child_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLEANUP_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
