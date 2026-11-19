# SPDX-License-Identifier: Apache-2.0
"""Offline refresh must use corrected captures and keep a valid summary on failure."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import refresh_moe_aligned_statistics as refresh
import run_cpp_mte_profile_compare as compare


class RefreshTests(unittest.TestCase):
    def test_bubbles_reject_raw_and_select_validated_corrected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, aligned = root / "trace_view.json", root / "trace_view_aligned.json"
            raw.write_text("[]", encoding="utf-8")
            aligned.write_text("[]", encoding="utf-8")
            reports = dict(
                callback=dict(
                    profile_status="complete", profile_trace_data=[dict(file=str(raw), usable=True, json_complete=True)]
                )
            )
            with patch.object(compare, "inspect_offload_trace", return_value=True) as inspect:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = compare.loading_bubble_summary(reports, root, ["callback"], "Stream 56")
                self.assertIn("BUBBLES_UNAVAILABLE", result)
                inspect.assert_not_called()
                raw.with_name("clock_alignment.json").write_text(
                    json.dumps(
                        dict(
                            status="calibration_validated",
                            aligned_trace=str(aligned),
                            held_out_after=dict(outside_bracket_1us_tolerance=0),
                        )
                    ),
                    encoding="utf-8",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    compare.loading_bubble_summary(reports, root, ["callback"], "Stream 56")
                self.assertEqual(inspect.call_args.args[0], aligned.resolve())

    def test_refresh_preserves_timing_and_does_not_replace_summary_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            case = run / "callback"
            case.mkdir()
            timings = dict(samples_us=[10, 20])
            report_file = case / "profile_report.json"
            report_file.write_text(json.dumps(dict(request_window=timings, host_backend="callback")), encoding="utf-8")
            (case / "inference_cpp.log").write_text("", encoding="utf-8")
            root = run / "profile/callback/cpp/test_ascend_pt"
            output = root / refresh.OUTPUT_NAME
            output.mkdir(parents=True)
            (output / "clock_alignment.json").write_text("{}", encoding="utf-8")
            (run / "summary.txt").write_text("previous complete summary", encoding="utf-8")

            def assess(report, *args, **kwargs):
                self.assertTrue(report["profile_clock_required"])
                report["profile_trace_data"] = [dict(file=str(output / "trace_view.json"), usable=True)]

            with (
                patch.object(refresh, "publish_aligned_capture", return_value=dict(status="exported")) as publish,
                patch.object(refresh, "assess_profile_report", side_effect=assess),
                patch.object(refresh, "print_comparison"),
                patch.object(refresh, "loading_bubble_summary", return_value="BUBBLES_UNAVAILABLE") as bubbles,
                patch.object(refresh, "print_tpot_comparison"),
                patch.object(refresh, "overlap_summary", return_value="overlap"),
                patch.object(refresh, "write_tables", return_value="corrected tables"),
            ):
                with self.assertRaisesRegex(ValueError, "bubbles unavailable"):
                    refresh.refresh(run, "Stream 56")
                self.assertEqual((run / "summary.txt").read_text(encoding="utf-8"), "previous complete summary")
                self.assertEqual(json.loads(report_file.read_text(encoding="utf-8"))["request_window"], timings)
                publish.assert_called_once()
                bubbles.return_value = "aligned bubbles"
                with contextlib.redirect_stdout(io.StringIO()):
                    refresh.refresh(run, "Stream 56")
                self.assertIn("corrected tables", (run / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
