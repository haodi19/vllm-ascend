# SPDX-License-Identifier: Apache-2.0
"""Check detection of stale routes, maps and weights without loading the NPU."""

import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class DataflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            module = runpy.run_path(str(scripts / "host_sync_dataflow.py"))
            runner = runpy.run_path(str(scripts / "run_host_sync_stage5.py"))
        cls.expected = staticmethod(module["expected_output"])
        cls.check = staticmethod(module["check_output"])
        cls.runner = staticmethod(runner["main"])

    def test_consumer_matches_changing_experts(self):
        self.check([2002, 2015, 1, 1], 1, 16)
        self.check([1008, 1015, 4, 4], 4, 16)

    def test_previous_round_even_same_expert_is_rejected(self):
        for previous in (1, 4):
            with self.assertRaises(RuntimeError):
                self.check(self.expected(previous, 16), 5, 16)

    def test_each_consumer_field_is_checked(self):
        for index in range(4):
            values = self.expected(3, 16)
            values[index] -= 1
            with self.assertRaises(RuntimeError):
                self.check(values, 3, 16)

    def test_regression_failure_prevents_dataflow(self):
        step = Mock()
        with (
            patch.dict(self.runner.__globals__, run_regression=Mock(return_value=1), run_step=step),
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.runner(["--npu", "1"]), 1)
        step.assert_not_called()

    def test_dataflow_failure_does_not_print_pass_summary(self):
        step = Mock(side_effect=[True, False])
        output = io.StringIO()
        with (
            patch.dict(self.runner.__globals__, run_regression=Mock(return_value=0), run_step=step),
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(self.runner(["--npu", "1"]), 1)
        self.assertNotIn("STAGE 5 SUMMARY", output.getvalue())
        self.assertEqual(step.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
