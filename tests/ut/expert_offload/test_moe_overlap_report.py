# SPDX-License-Identifier: Apache-2.0
"""Check integration keeps uncertainty, missing evidence and both restat schemes."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[3] / "script"), *sys.path]):
    import moe_overlap_report as report
    from analyze_prefetch_compute_overlap import restat_host_gaps, summarize


class OverlapReportTests(unittest.TestCase):
    def test_missing_capture_does_not_become_zero_or_use_raw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report, "analyze") as analyze:
            root = Path(directory)
            text = report.overlap_summary({"mte": dict(host_backend="mte")}, root, ["mte"])
            self.assertIn("OVERLAP_UNAVAILABLE mte", text)
            analyze.assert_not_called()

    def test_same_capture_and_classification_scopes(self):
        rows = [
            dict(gap_us=10, **{"class": "early"}, zero_margin_class="early", prefetch_loads=1, reactive_loads=2),
            dict(
                gap_us=100,
                **{"class": "late"},
                zero_margin_class="late",
                prefetch_loads=0,
                reactive_loads=2,
                readiness_minus_prefetch_end_bounds_us=[70, 80],
            ),
            dict(
                gap_us=20,
                **{"class": "uncertain"},
                zero_margin_class="uncertain",
                prefetch_loads=None,
                reactive_loads=None,
            ),
            dict(gap_us=30, **{"class": "unmapped"}, prefetch_loads=None, reactive_loads=None),
        ]
        result = dict(
            rows=rows,
            classification=summarize(rows),
            direct_classification=summarize(rows),
            conservative=restat_host_gaps(rows),
            direct=restat_host_gaps(rows, "zero_margin_class"),
            sensitivity_margin_us=10,
            load_issues=[],
            reasons={"missing": 1},
        )
        reports = {
            "callback_python": dict(host_backend="callback"),
            "mte_combined": dict(
                host_backend="mte",
                profile_status="complete",
                profile_trace_data=[dict(file="capture/trace_view.json", usable=True, json_complete=True)],
            ),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(report, "analyze", return_value=result) as analyze,
        ):
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                text = report.overlap_summary(reports, root, list(reports))
            analyze.assert_called_once_with(Path("capture/trace_view_aligned.json"))
            saved = json.loads((root / "prefetch_overlap.json").read_text(encoding="utf-8"))["mte_combined"]
            self.assertEqual(saved["conservative"]["all_samples"]["original"]["mean_us"], 40)
            self.assertEqual(saved["conservative"]["all_samples"]["method1_drop_late"]["mean_us"], 20)
            self.assertEqual(
                saved["conservative"]["all_samples"]["method2_subtract_remaining_compute"]["mean_us"], 21.25
            )
            self.assertIn("1 / 100.000%", text)
            self.assertIn("decidable_only", text)


if __name__ == "__main__":
    unittest.main()
