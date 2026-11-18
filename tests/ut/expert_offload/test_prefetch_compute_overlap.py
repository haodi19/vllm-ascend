# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))

from analyze_prefetch_compute_overlap import analyze, bind_streams, classify, consumer_boundary, summarize


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
        ):
            self.assertIsNone(consumer_boundary(segment)[0])

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
