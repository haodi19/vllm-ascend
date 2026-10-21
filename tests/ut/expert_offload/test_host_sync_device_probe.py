# SPDX-License-Identifier: Apache-2.0
"""CPU checks for stage 2 result validation; no fake device success."""

import argparse
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class DeviceResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "script/host_sync_device_probe.py"
        cls.check_result = staticmethod(runpy.run_path(str(path))["check_result"])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
