"""隔离实验的分类和超时保护测试；不模拟 NPU 性能结论。"""

import contextlib
import io
import runpy
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/probe_mte_compute_overlap.py"))


class OverlapProbeTests(unittest.TestCase):
    def test_completion_before_callback_is_overlap(self):
        self.assertEqual(MODULE["classify"](dict(compute_done_ns=10, callback_exit_ns=20)), "OVERLAP_CONFIRMED")

    def test_late_or_equal_completion_does_not_prove_resource_conflict(self):
        for end in (20, 30):
            self.assertEqual(MODULE["classify"](dict(compute_done_ns=end, callback_exit_ns=20)), "NO_OVERLAP_OBSERVED")

    def test_timeout_stops_subsequent_backends(self):
        args = ["probe", "--npu", "4", "--library", "/test/lib.so"]
        with (
            patch("sys.platform", "linux"),
            patch("sys.argv", args),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 90)) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(MODULE["main"](), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4")


if __name__ == "__main__":
    unittest.main()
