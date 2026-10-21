# SPDX-License-Identifier: Apache-2.0
"""CPU checks for stage 2 result validation; no fake device success."""

import runpy
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
