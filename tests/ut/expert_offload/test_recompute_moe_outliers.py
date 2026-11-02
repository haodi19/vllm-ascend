# SPDX-License-Identifier: Apache-2.0
"""只读离群值分析的合成数据回归，不依赖Torch/NPU。"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RecomputeOutliersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location("outlier_test", root / "script/recompute_moe_outliers.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_extreme_value_removed_and_original_unchanged(self):
        values = [100, 101, 102, 103, 104, 105, 106, 107, 2395]
        original = list(values)
        raw, clean, removed, bounds, reason = self.module.filtered_stats(values)
        self.assertEqual(values, original)
        self.assertEqual(raw["n"], 9)
        self.assertEqual(clean["n"], 8)
        self.assertEqual(clean["mean"], 103.5)
        self.assertEqual(removed, [(9, 2395)])
        self.assertEqual(reason, "ok")
        self.assertLess(bounds[1], 2395)

    def test_four_way_request_sample(self):
        values = [
            3143.446,
            3146.241,
            3146.352,
            3165.375,
            3135.934,
            3162.3,
            3151.797,
            3163.012,
            3161.425,
            3142.168,
            3149.089,
            3184.372,
            3150.701,
            3139.618,
            3155.141,
            3161.527,
            3139.422,
            3159.57,
            4921.186,
            3139.629,
        ]
        raw, clean, removed, _, _ = self.module.filtered_stats(values)
        self.assertEqual(removed, [(19, 4921.186)])
        self.assertAlmostEqual(raw["mean"], 3240.91525)
        self.assertAlmostEqual(clean["mean"], 3152.479947368421)
        self.assertAlmostEqual(clean["p95"], 3167.2747)

    def test_small_zero_iqr_empty_and_invalid_samples(self):
        for values, reason in (([1, 2, 100], "small_n"), ([1] * 10 + [100], "zero_iqr"), ([], "small_n")):
            raw, clean, removed, _, actual = self.module.filtered_stats(values)
            self.assertEqual(raw, clean)
            self.assertFalse(removed)
            self.assertEqual(actual, reason)
        for value in (True, -1, float("nan"), float("inf"), "3"):
            with self.assertRaises(ValueError):
                self.module.filtered_stats([value])

    def test_exact_derived_metric_filters_paired_sum_not_sum_of_filtered_means(self):
        group = dict(
            count=9,
            metric_samples_us={
                "load_submit": [1, 2, 3, 4, 5, 6, 7, 8, 1000],
                "synchronize": [1000, 20, 30, 40, 50, 60, 70, 80, 90],
            },
        )
        before = json.dumps(group)
        result = self.module.group_outliers(group)
        self.assertEqual(result["load_and_wait"]["filtered"]["n"], 7)
        self.assertEqual(result["load_and_wait"]["filtered"]["mean"], 55)
        self.assertNotEqual(
            result["load_and_wait"]["filtered"]["mean"],
            sum(result[key]["filtered"]["mean"] for key in group["metric_samples_us"]),
        )
        self.assertEqual(json.dumps(group), before)
        group["count"] = 10
        with self.assertRaises(ValueError):
            self.module.group_outliers(group)
        self.assertEqual(self.module.group_outliers({"metrics": {"total": {"mean": 1}}}), {})

    def test_new_report_exact_samples_without_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile_report.json"
            values = [100, 101, 102, 103, 104, 105, 106, 107, 2395]
            path.write_text(
                json.dumps(
                    dict(groups=[dict(kind="reactive", token_rows=3, count=9, metric_samples_us={"outer": values})])
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.module.analyze_report(path, 100)
            self.assertIn("EXACT_TIMER reactive/rows=3/outer [us] n=9->8", output.getvalue())
            self.assertNotIn("MISSING_EXACT_TIMERS", output.getvalue())

    @staticmethod
    def trace():
        def host(stage, call, kind, start, dur, pid=1):
            return dict(
                ph="X",
                name=f"moe_offload/{stage}|call={call}|kind={kind}|rows=1|layer=2",
                ts=start,
                dur=dur,
                pid=pid,
                tid=10,
            )

        return [
            host("host_callback", 1, "prefetch", 0, 20),
            host("update_weights", 1, "prefetch", 1, 18),
            host("load_submit", 1, "prefetch", 2, 3),
            host("synchronize", 1, "prefetch", 6, 5),
            host("host_callback", 2, "reactive", 120, 20),
            host("update_weights", 2, "reactive", 121, 18),
            host("host_callback", 1, "reactive", 30, 20, pid=2),
            host("update_weights", 1, "reactive", 31, 18, pid=2),
            dict(ph="X", name="MEMCPY_ASYNC", ts=0, dur=100, pid=100, tid=27),
            dict(ph="X", name="MEMCPY_ASYNC", ts=10, dur=5, pid=100, tid=27),
            dict(ph="X", name="MEMCPY_ASYNC", ts=150, dur=5, pid=100, tid=27),
            dict(ph="X", name="MEMCPY_ASYNC", ts=0, dur=10, pid=999, tid=27),
            dict(ph="M", name="process_name", pid=100, args={"name": "Ascend Hardware"}),
            dict(ph="M", name="thread_name", pid=100, tid=27, args={"name": "Stream 27"}),
        ]

    def read(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return self.module.read_trace(path)

    def test_pid_isolation_metadata_last_and_nested_copies(self):
        durations, gaps, _, device, _, count = self.read(self.trace())
        self.assertEqual(count, 3)
        self.assertEqual(dict(gaps), {("1", "prefetch->reactive"): [100]})
        self.assertEqual(durations["1", "prefetch", "1", "load_and_wait_mstx"], [8])
        self.assertEqual(set(device), {("100", "27")})
        self.assertEqual(self.module.copy_gaps(device["100", "27"]), ([50], 1))

    def test_duplicate_and_missing_markers_rejected(self):
        for data in (self.trace() + [self.trace()[0]], self.trace()[1:]):
            with self.assertRaises(ValueError):
                self.read(data)

    def test_truncated_trace_does_not_emit_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(self.trace())[:-1], encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(ValueError):
                self.module.analyze_trace(path, 100)
            self.assertNotIn("HOST_GAP", output.getvalue())
            self.assertNotIn("TRACE_JSON_COMPLETE", output.getvalue())

    def test_recursive_reports_deduplicated_and_no_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "run" / "callback"
            group.mkdir(parents=True)
            report = group / "profile_report.json"
            report.write_text(json.dumps({"request_window": {"count": 8, "samples_us": [1000] * 8}}), encoding="utf-8")
            (group / "cost_report.json").write_text("{}", encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(self.module.discover([root, root / "run"]), [report.resolve()])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(self.module.main([str(root)]), 0)
            self.assertIn("REQUEST_NO_PROFILE [ms] n=8->8; raw=1.000", output.getvalue())
            self.assertIn("MISSING", output.getvalue())
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_request_window_survives_missing_profile_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "request_window.json"
            path.write_text(
                json.dumps(dict(count=9, samples_us=[100, 101, 102, 103, 104, 105, 106, 107, 2395])), encoding="utf-8"
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(self.module.main([str(root)]), 0)
            self.assertIn("REQUEST_NO_PROFILE [ms] n=9->8", output.getvalue())


if __name__ == "__main__":
    unittest.main()
