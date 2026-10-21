# SPDX-License-Identifier: Apache-2.0
"""CPU checks for stage 2 result validation; no fake device success."""

import argparse
import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class DeviceResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/host_sync_device_probe.py"
        module = runpy.run_path(str(path))
        cls.check_result = staticmethod(module["check_result"])
        cls.check_capture = staticmethod(module["check_capture_unchanged"])

    def test_capture_preserves_warmup_state(self):
        self.check_capture([1, 1, 1, 4], [1, 1, 1, 4], [1])

    def test_capture_rejects_device_or_host_execution(self):
        for after, calls in (([2, 1, 2, 4], [1, 2]), ([1, 1, 1, 4], [1, 2]), ([1, 1, 1, 4], [])):
            with self.assertRaises(RuntimeError):
                self.check_capture([1, 1, 1, 4], after, calls)

    def test_matching_success(self):
        self.check_result([7, 1, 7, 4], 7, 1)

    def test_stale_generation_and_completion_rejected(self):
        for result in ([6, 1, 6, 4], [7, 1, 6, 4], [7, 1, 7, 0]):
            with self.assertRaises(RuntimeError):
                self.check_result(result, 7, 1)

    def test_failure_and_timeout_cannot_be_success(self):
        for code in (2, 3, 4):
            with self.assertRaises(RuntimeError):
                self.check_result([7, code, 6, 4], 7, 1)
        self.check_result([7, 2, 6, 4], 7, 2)
        self.check_result([7, 3, 6, 4], 7, 3)


class BuildTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            module = runpy.run_path(str(scripts / "run_host_sync_stage2.py"))
        cls.target_args = staticmethod(module["target_cmake_args"])

    def test_concrete_soc_and_compute_architecture_are_distinct(self):
        self.assertEqual(
            self.target_args("Ascend950PR_9579"),
            ["-DSOC_VERSION=ascend950pr_9579", "-DASCEND_COMPUTE_UNIT=ascend950"],
        )

    def test_architecture_is_not_a_concrete_soc(self):
        for value in ("ascend950", "ascend910b1", "ascend950pr_"):
            with self.assertRaises(argparse.ArgumentTypeError):
                self.target_args(value)


class GraphRunnerTests(unittest.TestCase):
    def setUp(self):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            self.main = runpy.run_path(str(scripts / "run_host_sync_stage3.py"))["main"]

    def invoke(self, eager_result, graph_results):
        eager = Mock(return_value=eager_result)
        step = Mock(side_effect=graph_results)
        output = io.StringIO()
        with (
            patch.dict(self.main.__globals__, run_eager_regression=eager, run_step=step),
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stdout(output),
        ):
            result = self.main(["--npu", "4", "--rounds", "3", "--skip-build"])
        return result, eager, step, output.getvalue()

    def test_eager_failure_prevents_graph_tests(self):
        result, _, step, output = self.invoke(1, [])
        self.assertEqual(result, 1)
        step.assert_not_called()
        self.assertNotIn("STAGE 3 SUMMARY", output)

    def test_graph_failure_stops_remaining_cases(self):
        result, _, step, output = self.invoke(0, [True, False])
        self.assertEqual(result, 1)
        self.assertEqual(step.call_count, 2)
        self.assertNotIn("STAGE 3 SUMMARY", output)

    def test_graph_mode_and_device_forwarded_to_each_case(self):
        result, eager, step, output = self.invoke(0, [True, True, True])
        self.assertEqual(result, 0)
        self.assertIn("--skip-build", eager.call_args.args[0])
        self.assertIn("STAGE 3 SUMMARY", output)
        for invocation, case in zip(step.call_args_list, ("normal", "callback-error", "poll-limit")):
            command = invocation.args[1]
            self.assertEqual(command[command.index("--mode") + 1], "graph")
            self.assertEqual(command[command.index("--case") + 1], case)
            self.assertEqual(command[command.index("--rounds") + 1], "3")
            self.assertEqual(invocation.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
