# SPDX-License-Identifier: Apache-2.0
"""CPU-only decisions for the bounded, two-case resource diagnosis."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ResourceScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "resource_scope_test", scripts / "diagnose_mte_resource_scope.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def known_failure(self):
        return "\n".join(
            [
                "RESOURCE_API=torch",
                "RUNNER_EXIT=1",
                "[MOE-MTE ACL_DETAIL] stream timeout",
                "w13_weight_scale ACL_WAIT_RETURN code=507046",
                'GRAPH_KERNEL [23, "HcPre_1000", 14, 0]',
                'GRAPH_KERNEL [23, "MoeInitRoutingV3_10030000", 5, 1]',
                'GRAPH_KERNEL [21, "host_sync_wait_2", 1, 0]',
            ]
        )

    def test_matching_failure_allows_auxiliary_control(self):
        self.assertEqual(self.module.classify_report(self.known_failure(), 1), "scale_wait_reproduced")

    def test_wrong_or_missing_actual_limits_stop_experiment(self):
        text = self.known_failure()
        for changed in (text.replace("14, 0", "28, 0"), text.replace("5, 1", "56, 1"), text.replace("1, 0", "2, 0")):
            with self.subTest(changed=changed):
                self.assertEqual(self.module.classify_report(changed, 1), "inconclusive")
        self.assertEqual(self.module.classify_report("RuntimeError: startup failed", 1), "inconclusive")

    def test_request_success_requires_report_and_exit_code(self):
        text = 'RUNNER_EXIT=0\nMOE_RESULT {"profile_status": "invalid"}'
        self.assertEqual(self.module.classify_report(text, 0), "request_passed")
        self.assertEqual(self.module.classify_report(text, 1), "inconclusive")
        self.assertEqual(self.module.classify_report("RUNNER_EXIT=0", 0), "inconclusive")

    def test_compact_summary_keeps_resource_api_and_observed_task(self):
        text = 'RESOURCE_API=torch\nMATCH {"name":"MatMulV3"}\nnoise'
        result = self.module.concise_report(text)
        self.assertIn("RESOURCE_API=torch", result)
        self.assertIn("MatMulV3", result)
        self.assertNotIn("noise", result)


if __name__ == "__main__":
    unittest.main()
