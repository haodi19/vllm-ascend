# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))

from analyze_prefetch_compute_overlap import (
    analyze,
    bind_streams,
    classify,
    consumer_boundary,
    load_annotations,
    load_breakdown,
    load_pair,
    restat_host_gaps,
    summarize,
)


def task(name, start, end):
    return dict(name=name, start=start, end=end)


class OverlapTests(unittest.TestCase):
    def segment(self):
        return [
            task("MoeRouting", 10, 30),
            task("EVENT_RECORD", 30, 31),
            task("DAVID_EVENT_WAIT", 32, 500),
            task("MEMCPY_ASYNC", 501, 503),
            task("MEMCPY_ASYNC", 504, 505),
        ]

    def test_join_start_not_end_is_upper_boundary(self):
        (compute, join), reason = consumer_boundary(self.segment())
        self.assertEqual(reason, "ok")
        self.assertEqual(classify(compute["end"], join["start"], 100, 10), "early")
        self.assertGreater(join["end"], 100)

    def test_partial_or_altered_tail_not_guessed(self):
        for segment in (
            self.segment()[:-1],
            self.segment() + [task("Transpose", 506, 507)],
            self.segment()[2:],
            self.segment()[:2] + self.segment()[3:],
            self.segment() + [task("MEMCPY_ASYNC", 506, 507), task("MEMCPY_ASYNC", 508, 509)],
        ):
            self.assertIsNone(consumer_boundary(segment)[0])

    def test_three_copy_lrc_tail_from_real_capture(self):
        segment = self.segment() + [task("MEMCPY_ASYNC", 506, 507)]
        (compute, join), reason = consumer_boundary(segment)
        self.assertEqual(reason, "ok")
        self.assertEqual(compute["end"], 30)
        self.assertEqual(join["start"], 32)

    def test_unmapped_is_not_reported_as_zero_late_rate(self):
        result = load_breakdown([dict(gap_us=300, load_pair="both_zero", **{"class": "unmapped"})])
        self.assertIsNone(result["late_rate_by_load_pair"]["both_zero"]["late_pct"])
        self.assertEqual(result["late_rate_by_load_pair"]["both_zero"]["unmapped_count"], 1)

    def test_uncertainty_not_forced_into_early_or_late(self):
        self.assertEqual(classify(90, 110, 100, 0), "uncertain")
        self.assertEqual(classify(105, 110, 100, 10), "uncertain")
        self.assertEqual(classify(111, 120, 100, 10), "late")

    def test_long_tail_contribution_all_samples_retained(self):
        rows = [dict(gap_us=50, **{"class": "early"}) for _ in range(9)]
        rows += [dict(gap_us=550, **{"class": "late"})]
        result = summarize(rows)
        self.assertEqual(result["all"]["mean_us"], 100)
        self.assertEqual(result["late"]["fraction_pct"], 10)
        self.assertEqual(result["late"]["total_gap_share_pct"], 55)
        self.assertEqual(result["late_vs_early"]["late_excess_contribution_to_all_mean_us"], 50)

    def test_unmapped_in_denominator(self):
        result = summarize(
            [
                dict(gap_us=10, **{"class": "early"}),
                dict(gap_us=30, **{"class": "unmapped"}),
            ]
        )
        self.assertEqual(result["all"]["mean_us"], 20)
        self.assertEqual(result["early"]["fraction_pct"], 50)
        self.assertEqual(result["unmapped"]["total_gap_share_pct"], 75)

    def test_bad_marker_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid Host markers"):
            bind_streams(dict(issues={"missing_sync": 1}, invalid_kinds=set()))

    def test_empty_group_has_no_invented_ratio(self):
        self.assertNotIn("late_vs_early", summarize([]))

    def test_restat_keeps_unknown_visible_and_subtracts_only_post_prefetch_time(self):
        rows = [
            dict(gap_us=40, **{"class": "early"}),
            dict(gap_us=300, readiness_minus_prefetch_end_bounds_us=[250, 260], **{"class": "late"}),
            dict(gap_us=20, **{"class": "uncertain"}),
            dict(gap_us=400, **{"class": "unmapped"}),
        ]
        result = restat_host_gaps(rows)
        all_rows = result["all_samples"]
        self.assertEqual(all_rows["method1_drop_late"]["count"], 3)
        self.assertAlmostEqual(all_rows["method1_drop_late"]["mean_us"], 460 / 3)
        self.assertEqual(all_rows["method2_subtract_remaining_compute"]["mean_us"], 126.25)
        self.assertEqual(all_rows["subtracted_total_us"], 255)
        known = result["decidable_only"]
        self.assertEqual(known["excluded"], 2)
        self.assertEqual(known["method2_subtract_remaining_compute"]["mean_us"], 42.5)
        self.assertEqual(known["method2_aligned_boundary_low"]["mean_us"], 40)
        self.assertEqual(known["method2_aligned_boundary_high"]["mean_us"], 45)

    def test_restat_does_not_clamp_invalid_deduction_to_zero(self):
        result = restat_host_gaps(
            [dict(gap_us=10, readiness_minus_prefetch_end_bounds_us=[20, 30], **{"class": "late"})]
        )["all_samples"]
        self.assertEqual(result["method2_subtract_remaining_compute"]["mean_us"], 10)
        self.assertEqual(result["counts"]["invalid_subtraction"], 1)

    def test_restat_zero_margin_is_separate(self):
        row = dict(
            gap_us=50,
            readiness_minus_prefetch_end_bounds_us=[10, 10],
            zero_margin_class="late",
            **{"class": "uncertain"},
        )
        self.assertEqual(restat_host_gaps([row])["all_samples"]["method2_subtract_remaining_compute"]["mean_us"], 50)
        result = restat_host_gaps([row], "zero_margin_class")["all_samples"]
        self.assertEqual(result["method1_drop_late"]["count"], 0)
        self.assertEqual(result["method2_subtract_remaining_compute"]["mean_us"], 40)

    def test_load_pair_unknown_is_not_empty(self):
        self.assertEqual(load_pair(None, 0), "unknown")
        self.assertEqual(load_pair(0, None), "unknown")
        self.assertEqual(load_pair(0, 1), "prefetch_zero_only")
        self.assertEqual(load_pair(1, 0), "reactive_zero_only")
        self.assertEqual(load_pair(0, 0), "both_zero")
        self.assertEqual(load_pair(1, 2), "both_loaded")

    def test_late_load_composition_and_conditional_rate_have_different_denominators(self):
        rows = [
            dict(gap_us=300, load_pair="both_zero", **{"class": "late"}),
            dict(gap_us=100, load_pair="both_loaded", **{"class": "late"}),
            dict(gap_us=50, load_pair="both_loaded", **{"class": "early"}),
            dict(gap_us=50, load_pair="both_loaded", **{"class": "uncertain"}),
        ]
        result = load_breakdown(rows)
        self.assertEqual(result["late"]["both_zero"]["pct_of_timing_group"], 50)
        self.assertEqual(result["late"]["both_zero"]["gap_sum_share_pct"], 75)
        self.assertEqual(result["late_rate_by_load_pair"]["both_zero"]["late_pct"], 100)
        self.assertAlmostEqual(result["late_rate_by_load_pair"]["both_loaded"]["late_pct"], 100 / 3)

    def annotation_fixture(self):
        host = dict(
            pid="host", tid="worker", start=0, end=100, index=0, context=dict(call="1", layer="15", kind="prefetch")
        )
        meta = dict(call="1", layer="15", kind="prefetch", rows="1")
        stages = {
            1: {
                name: [dict(start=a, end=b, tid="worker", meta=meta)]
                for name, a, b in (
                    ("update_weights", 1, 99),
                    ("load_submit", 10, 50),
                    ("copy_submit", 20, 30),
                    ("synchronize", 60, 90),
                )
            }
        }
        report = dict(
            count=1,
            errors=0,
            dropped=0,
            mstx_errors=0,
            mstx_enabled=True,
            load_backend="python",
            h2d_backend="torch",
            groups=[dict(kind="prefetch", token_rows=1, count=1, loads=dict(mean=0))],
        )
        return [host], stages, report

    def test_confirmed_empty_does_not_depend_on_duration(self):
        hosts, stages, report = self.annotation_fixture()
        loads, issues = load_annotations(hosts, stages, report)
        self.assertEqual(issues, [])
        self.assertEqual(loads, {"1": 0})

    def test_missing_report_or_lost_markers_cannot_prove_empty(self):
        hosts, stages, report = self.annotation_fixture()
        self.assertIsNone(load_annotations(hosts, stages, None)[0]["1"])
        report["groups"][0]["loads"]["mean"] = 1
        loads, issues = load_annotations(hosts, stages, report)
        self.assertIsNone(loads["1"])
        self.assertTrue(issues)

    def test_actual_one_expert_count_is_loaded(self):
        hosts, stages, report = self.annotation_fixture()
        for name in ("load_prepare", "scale_refresh"):
            stages[1][name] = [
                dict(start=11, end=12, tid="worker", meta=dict(call="1", layer="15", kind="prefetch", rows="1"))
            ]
        report["groups"][0]["loads"]["mean"] = 1
        loads, issues = load_annotations(hosts, stages, report)
        self.assertFalse(issues)
        self.assertEqual(loads["1"], 1)

    def test_saved_capture_end_to_end(self):
        records = [dict(ph="M", name="process_name", pid="device", args=dict(name="Ascend Hardware"))]

        def add(name, start, end, tid, pid="device"):
            records.append(dict(ph="X", name=name, ts=start, dur=end - start, pid=pid, tid=tid))

        def host(kind, layer, call, start, end):
            suffix = f"|backend=cpp|kind={kind}|layer={layer}|rows=3|call={call}"
            add("moe_offload/host_callback" + suffix, start, end, "worker", "host")
            add("moe_offload/synchronize" + suffix, start + 1, end - 1, "worker", "host")

        def handshake(tid, publish, wait, end):
            add("host_sync_publish_1", publish, publish + 1, tid)
            add("host_sync_wait_2", wait, end, tid)

        host("reactive", 0, 1, 10, 20)
        host("prefetch", 1, 2, 100, 200)
        host("reactive", 1, 3, 212, 240)
        host("prefetch", 2, 4, 300, 320)
        host("reactive", 2, 5, 375, 400)
        handshake("main", 5, 6, 22)
        handshake("prefetch", 80, 155, 205)
        handshake("main", 211, 212, 242)
        handshake("prefetch", 280, 360, 365)
        handshake("main", 371, 372, 402)
        for start, end, join, finish in ((50, 145, 150, 206), (250, 350, 355, 366)):
            add("MoeRouting", start, end, "main")
            add("DAVID_EVENT_WAIT", join, finish, "main")
            add("MEMCPY_ASYNC", finish + 1, finish + 2, "main")
            add("MEMCPY_ASYNC", finish + 3, finish + 4, "main")
            add("MEMCPY_ASYNC", finish + 4.25, finish + 4.5, "main")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace_view_aligned.json"
            path.write_text(json.dumps(dict(traceEvents=records)), encoding="utf-8")
            path.with_name("clock_alignment.json").write_text(
                json.dumps(
                    dict(
                        status="calibration_validated",
                        calibration_layout="model_request_brackets",
                        aligned_trace=str(path),
                        trace_origin_us="0",
                        host_span_us=["0", "500"],
                        endpoint_mapping_envelope_width_us=["1", "1"],
                    )
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                analyze(path)
            result = next(line[7:] for line in output.getvalue().splitlines() if line.startswith("RESULT "))
            result = json.loads(result)
            self.assertEqual(result["all"]["count"], 2)
            self.assertEqual(result["early"]["mean_us"], 12)
            self.assertEqual(result["late"]["mean_us"], 55)
            self.assertEqual(result["unmapped"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
