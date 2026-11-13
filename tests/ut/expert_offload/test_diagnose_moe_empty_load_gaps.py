# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for empty-load evidence and multi-callback device gaps."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))

from diagnose_moe_empty_load_gaps import Coverage, analyze, compare_pairs, discover, main, render


def interval(name, start, end, pid="host", tid="worker"):
    return dict(ph="X", name=name, ts=start, dur=end - start, pid=pid, tid=tid)


def callback(call, start, kind, layer, loads=0):
    suffix = f"|backend=cpp|call={call}|kind={kind}|layer={layer}|rows=3"
    points = [
        interval(f"moe_offload/host_callback{suffix}", start, start + 40),
        interval(f"moe_offload/update_weights{suffix}", start + 1, start + 39),
        interval(f"moe_offload/load_submit{suffix}", start + 10, start + 15),
        interval(f"moe_offload/copy_submit{suffix}", start + 12, start + 13),
        interval(f"moe_offload/synchronize{suffix}", start + 20, start + 22),
    ]
    for _ in range(loads):
        points.extend(
            [
                interval(f"moe_offload/load_prepare{suffix}", start + 10, start + 11),
                interval(f"moe_offload/scale_refresh{suffix}", start + 13, start + 14),
            ]
        )
    return points


def fixture():
    metadata = [
        dict(ph="M", pid="device", name="process_name", args=dict(name="Ascend Hardware")),
        dict(ph="M", pid="device", tid="987", name="thread_name", args=dict(name="Stream 56")),
    ]
    device = [
        interval("MEMCPY_ASYNC", 0, 10, "device", "987"),
        interval("MEMCPY_ASYNC", 2510, 2520, "device", "987"),
        interval("MatMul", 150, 500, "device", "compute"),
        interval("Activation", 200, 400, "device", "other"),
        interval("DAVID_EVENT_WAIT", 140, 600, "device", "wait"),
    ]
    host = callback(1, 100, "prefetch", 1) + callback(3, 500, "reactive", 1) + callback(5, 600, "prefetch", 2)
    report = dict(
        count=3,
        errors=0,
        dropped=0,
        mstx_errors=0,
        mstx_enabled=True,
        load_backend="python",
        groups=[
            dict(kind="prefetch", token_rows=3, count=2, loads=dict(mean=0)),
            dict(kind="reactive", token_rows=3, count=1, loads=dict(mean=0)),
        ],
    )
    return metadata + device + host, report


class EmptyLoadGapTests(unittest.TestCase):
    def run_analysis(self, events, report):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_view.json"
            path.write_text(json.dumps(dict(traceEvents=events)), encoding="utf-8")
            return analyze(path, "Stream 56", report, progress=lambda _: None)

    def test_episode_contains_short_empty_callbacks_and_compute(self):
        trace, report = fixture()
        result = self.run_analysis(trace, report)
        self.assertEqual(result["integrity_issues"], [])
        self.assertEqual(len(result["episodes"]), 1)
        episode = result["episodes"][0]
        self.assertEqual(episode["gap_us"], 2500)
        self.assertEqual(episode["classes"], {"zero_prepare": 3})
        self.assertEqual([p["gap_us"] for p in result["pairs"]], [360, 60])
        self.assertEqual([p["compute_overlap_us"] for p in result["pairs"]], [350, 0])
        self.assertTrue(all(p["in_multi_callback_gap"] for p in result["pairs"]))
        self.assertIn("other_compute_union=350.000", "\n".join(render(result)))

    def test_short_duration_does_not_make_loaded_callback_empty(self):
        trace, report = fixture()
        trace = [e for e in trace if "call=3|" not in e["name"]] + callback(3, 500, "reactive", 1, loads=1)
        report["groups"][1]["loads"]["mean"] = 1
        result = self.run_analysis(trace, report)
        self.assertEqual(result["callbacks"][1]["load_class"], "with_prepare")
        self.assertEqual(result["callbacks"][1]["load_wait_us"], 7)
        self.assertEqual(result["pairs"][0]["class_pair"], "zero_prepare->with_prepare")

    def test_missing_stages_are_unknown_not_zero_work(self):
        trace, report = fixture()
        trace = [e for e in trace if not e["name"].startswith("moe_offload/copy_submit|backend=cpp|call=3|")]
        result = self.run_analysis(trace, report)
        self.assertEqual(result["callbacks"][1]["load_class"], "unknown")
        self.assertTrue(result["integrity_issues"])
        self.assertIsNone(result["callbacks"][1]["load_wait_us"])

    def test_no_prepare_is_not_verified_when_report_expects_loads(self):
        trace, report = fixture()
        report["groups"][1]["loads"]["mean"] = 2
        result = self.run_analysis(trace, report)
        self.assertTrue(any("准备次数" in issue for issue in result["integrity_issues"]))

    def test_transpose_occupancy_splits_copy_gap(self):
        trace, report = fixture()
        trace.append(interval("aclnInplaceCopy_TransposeAiCore_Transpose", 400, 700, "device", "987"))
        result = self.run_analysis(trace, report)
        self.assertEqual(result["episodes"], [])
        self.assertFalse(any(p["in_multi_callback_gap"] for p in result["pairs"]))

    def test_long_gaps_are_audited_but_not_in_regular_band(self):
        trace, report = fixture()
        for event in trace:
            if event.get("ts") == 2510:
                event["ts"] = 6010
        result = self.run_analysis(trace, report)
        self.assertFalse(result["episodes"][0]["in_bubble_band"])
        self.assertIn("count_ge_5000us=1", "\n".join(render(result)))

    def test_duplicate_ids_rejected(self):
        trace, report = fixture()
        trace += callback(1, 900, "reactive", 2)
        with self.assertRaisesRegex(ValueError, "重复"):
            self.run_analysis(trace, report)

    def test_wrong_thread_stage_not_silently_assigned(self):
        trace, report = fixture()
        for event in trace:
            if event["name"].startswith("moe_offload/load_submit|backend=cpp|call=3|"):
                event["tid"] = "wrong"
        result = self.run_analysis(trace, report)
        self.assertEqual(result["callbacks"][1]["load_class"], "unknown")

    def test_coverage_merges_overlapping_tasks(self):
        cov = Coverage([(10, 20), (12, 17), (18, 22), (30, 40)])
        self.assertEqual(cov.overlap(15, 35), 12)
        self.assertEqual(cov.overlap(22, 30), 0)
        self.assertEqual(cov.overlap(0, 100), 22)

    def test_file_mode_and_never_overwrites_summary(self):
        trace, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_view.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            self.assertEqual(discover(path), [("single_trace", path, None)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path)]), 1)  # Missing report explicitly incomplete.
            with self.assertRaisesRegex(ValueError, "已存在"):
                main([str(path)])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), trace)

    def test_both_token_lengths_discovered_without_mixing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tokens in (5, 128):
                directory = root / f"tokens{tokens}" / "profile" / "callback" / "cpp"
                directory.mkdir(parents=True)
                (directory / "trace_view.json").write_text("[]", encoding="utf-8")
            found = discover(root)
            self.assertEqual(len(found), 8)
            self.assertEqual(sum(path is not None for _, path, _ in found), 2)

    def test_matched_comparison_does_not_mix_load_counts_or_missing_output(self):
        trace, report = fixture()
        baseline = self.run_analysis(trace, report)
        candidate = self.run_analysis(trace, report)
        for result in (baseline, candidate):
            result["identity"] = dict(request_specs=["same prompt"], digests=["same text"], usage=[5])
        candidate["pairs"][0]["prepared_pair"] = [0, 1]
        candidate["pairs"][1]["gap_us"] = 50
        output = "\n".join(compare_pairs(baseline, candidate))
        self.assertIn("excluded_pairs=1", output)
        self.assertIn("delta=-10.000", output)
        del candidate["identity"]["digests"]
        self.assertIn("skipped", "\n".join(compare_pairs(baseline, candidate)))

    def test_four_way_cli_writes_one_complete_summary_with_comparisons(self):
        trace, report = fixture()
        report.update(request_specs=["same"], digests=["same"], usage=[5])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in ("callback_legacy", "callback", "mte_default", "mte_combined"):
                policy = "python" if case == "callback_legacy" else "cpp"
                profile = root / "profile" / case / policy
                profile.mkdir(parents=True)
                (profile / "trace_view.json").write_text(json.dumps(trace), encoding="utf-8")
                (root / case).mkdir()
                (root / case / "profile_report.json").write_text(json.dumps(report), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root)]), 0)
            output = (root / "empty_load_gap_summary.txt").read_text(encoding="utf-8")
            self.assertEqual(output.count("MARKER_INTEGRITY PASS"), 4)
            self.assertEqual(output.count("MATCHED_COMPARE excluded_pairs=0"), 3)
            self.assertIn("SUMMARY traces=4 incomplete=0", output)


if __name__ == "__main__":
    unittest.main()
