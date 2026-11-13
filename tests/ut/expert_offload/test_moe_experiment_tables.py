# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of automatically generated experiment tables."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ExperimentTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[3] / "script"), *sys.path]):
            cls.module = importlib.import_module("moe_experiment_tables")

    def test_recovered_host_data_and_missing_tpot_are_distinct(self):
        gap = dict(direction="prefetch->reactive", outliers_us=self.module.outlier_summary([30, 50]))
        report = dict(
            profile_status="recovered_intervals",
            profile_export_status="incomplete",
            count=3,
            request_window={"samples_us": [1000000, 2000000]},
            profile_trace_data=[dict(usable=True, json_complete=True, host_gaps=[gap])],
        )
        result = self.module.render_tables({"mte_combined": report}, ["mte_combined"])
        self.assertIn("| mte_combined | 端到端 | 2 | 1500.000", result)
        self.assertIn("| mte_combined | TPOT | — | —", result)
        self.assertIn("| mte_combined | prefetch->reactive | 2 | 40.000", result)
        self.assertIn("incomplete | recovered_intervals", result)
        report["profile_status"] = "invalid"
        result = self.module.render_tables({"mte_combined": report}, ["mte_combined"])
        self.assertNotIn("| 40.000", result)

    def test_bubbles_keep_scopes_counts_totals_and_partial_direction_warning(self):
        bubbles = "CASE=mte_combined; traces=1\n"
        for scope, mean in (("full_capture", 100), ("callback_window", 80)):
            bubbles += (
                "TARGET_GAPS_US "
                + json.dumps(
                    dict(
                        scope=scope,
                        excluded_short=9,
                        excluded_long=2,
                        kept=dict(n=10, mean=mean, p50=mean, p95=mean, total=10 * mean),
                    )
                )
                + "\n"
            )
        bubbles += 'TARGET_DIRECTION_GAPS_US {"direction":"prefetch->reactive","kept":{"n":2,"mean":50}}\n'
        bubbles += 'OFFLOAD_STATUS {"partial":true,"reasons":{"cross_boundary":12}}\n'
        result = self.module.render_tables({}, ["mte_combined"], bubbles=bubbles)
        self.assertIn("full_capture | 10 | 100.000 | 100.000 | 100.000 | 1.000 | 9 | 2", result)
        self.assertIn("callback_window | 10 | 80.000", result)
        self.assertIn("子集：存在跨边界/歧义任务", result)
        self.assertIn("prefetch->reactive | 2 | 50.000", result)

    def test_failure_still_creates_document_with_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = self.module.write_tables(root, {}, ["callback_legacy", "mte_combined"], failure="startup failed")
            self.assertEqual((root / "experiment_tables.md").read_text(encoding="utf-8"), text)
            self.assertIn("startup failed", text)
            self.assertIn("Python LRC", text)
            self.assertIn("allocated/atomic/yield", text)
            self.assertNotIn("0.000", text)


if __name__ == "__main__":
    unittest.main()
