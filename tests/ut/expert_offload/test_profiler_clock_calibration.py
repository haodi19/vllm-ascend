# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import calibrate_moe_profiler_clock as calibration


class ClockCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "test_ascend_pt"
        self.metadata = self.root / "PROF_test/host/start_info"
        self.metadata.parent.mkdir(parents=True)
        self.trace = self.root / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        self.trace.parent.mkdir()
        self.anchor_ns = 100_000_000_000
        self.epoch_us = 1_794_296_000_000_000
        self.offset = Decimal(self.epoch_us) - Decimal(self.anchor_ns) / 1000
        self.metadata.write_text(
            repr({"collectionTimeBegin": str(self.epoch_us), "clockMonotonicRaw": str(self.anchor_ns)}),
            encoding="utf-8",
        )
        self.rows = []
        for index in range(3):
            begin = self.anchor_ns + 1_000_000_000 * index
            self.rows.append(
                {
                    "sample": index,
                    "start_before_ns": begin,
                    "start_after_ns": begin + 2000,
                    "submit_before_ns": begin + 3000,
                    "sync_after_ns": begin + 15000,
                    "end_before_ns": begin + 16000,
                    "end_after_ns": begin + 18000,
                    "event_elapsed_us": index * 1_000_000,
                }
            )

    def write_trace(self, host_rate=0, device_offset=0, device_rate=0, device_copies=1):
        values = [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Python"}},
            {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        ]
        for row in self.rows:
            index = row["sample"]
            begin = self.offset + Decimal(row["start_before_ns"]) / 1000
            values.append(
                {
                    "ph": "X",
                    "name": f"{calibration.MARKER_PREFIX}{index}",
                    "pid": 1,
                    "tid": 9,
                    "ts": str(begin + 1 + index * host_rate),
                    "dur": "16",
                }
            )
            values.append({"ph": "X", "name": "aten::add_", "pid": 1, "tid": 9, "ts": str(begin), "dur": 6})
            for _ in range(device_copies):
                values.append(
                    {
                        "ph": "X",
                        "name": "aclnnInplaceAdd_Add",
                        "pid": 2,
                        "tid": 5,
                        "ts": str(begin + 6 + device_offset + index * device_rate),
                        "dur": "4",
                    }
                )
        self.trace.write_text(json.dumps({"traceEvents": values}), encoding="utf-8")
        return values

    def test_correct_alignment_preserves_epoch_precision_and_ignores_cpu_add(self):
        self.write_trace()
        result = calibration.analyze_trace(self.root, self.rows)
        self.assertEqual(result["mstx_start_vs_raw"]["outside_bracket"], 0)
        self.assertEqual(result["mstx_end_vs_raw"]["outside_bracket"], 0)
        self.assertEqual(result["mstx_start_vs_raw"]["first_offset_bounds_us"], ["-1", "1"])
        self.assertEqual(result["device_add_candidates"]["count"], 3)
        self.assertEqual(result["device_offset_necessary_bounds_us"]["zero_offset_violations"], 0)
        self.assertEqual(result["event_elapsed_vs_host_bounds_us"]["violations"], 0)

    def test_host_drift_is_detected_independently_of_device_intervals(self):
        self.write_trace(host_rate=29)
        result = calibration.analyze_trace(self.root, self.rows)
        stats = result["mstx_start_vs_raw"]
        self.assertEqual(stats["outside_bracket"], 2)
        self.assertEqual(stats["drift_bounds_us_per_second"], ["28", "30"])
        self.assertEqual(result["device_offset_necessary_bounds_us"]["zero_offset_violations"], 0)

    def test_early_device_timestamps_detected_with_correct_host_clock(self):
        self.write_trace(device_offset=-200)
        result = calibration.analyze_trace(self.root, self.rows)
        self.assertEqual(result["mstx_start_vs_raw"]["outside_bracket"], 0)
        stats = result["device_offset_necessary_bounds_us"]
        self.assertEqual(stats["zero_offset_violations"], 3)
        self.assertEqual(stats["constant_offset_intersection"], ["-205", "-197"])
        self.assertTrue(stats["constant_offset_possible"])

    def test_device_drift_has_no_constant_offset_solution(self):
        self.write_trace(device_rate=29)
        result = calibration.analyze_trace(self.root, self.rows)
        self.assertFalse(result["device_offset_necessary_bounds_us"]["constant_offset_possible"])

    def test_ambiguous_device_mapping_retains_host_check_only(self):
        self.write_trace(device_copies=2)
        result = calibration.analyze_trace(self.root, self.rows)
        self.assertEqual(result["mstx_start_vs_raw"]["outside_bracket"], 0)
        self.assertNotIn("device_offset_necessary_bounds_us", result)
        self.assertIn("unproven", result["device_mapping"])

    def test_incomplete_or_duplicate_markers_cannot_pass_calibration(self):
        values = self.write_trace()
        marker = next(e for e in values if e["name"].startswith(calibration.MARKER_PREFIX))
        for changed in (values + [marker], [e for e in values if e is not marker]):
            with self.subTest(duplicates=len(changed) > len(values)):
                self.trace.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    calibration.analyze_trace(self.root, self.rows)

    def test_invalid_direct_clock_order_rejected(self):
        self.write_trace()
        self.rows[1]["sync_after_ns"] = self.rows[1]["submit_before_ns"] - 1
        with self.assertRaisesRegex(ValueError, "RAW measurement order"):
            calibration.analyze_trace(self.root, self.rows)

    def test_missing_export_keeps_error_in_summary(self):
        args = SimpleNamespace(output_dir=self.base / "new_run", npu=1, samples=3, interval=0.2)
        process = mock.Mock(returncode=0)
        process.poll.return_value = 0
        with (
            mock.patch.object(calibration.subprocess, "Popen", return_value=process),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(calibration.run_parent(args), 1)
        self.assertIn("ANALYSIS_FAILED mode=eager", (args.output_dir / "summary.txt").read_text(encoding="utf-8"))
        self.assertFalse((args.output_dir / "graph").exists())

    def test_existing_output_is_not_overwritten(self):
        args = SimpleNamespace(output_dir=self.root)
        self.write_trace()
        original = self.trace.read_bytes()
        with self.assertRaises(FileExistsError):
            calibration.run_parent(args)
        self.assertEqual(self.trace.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
