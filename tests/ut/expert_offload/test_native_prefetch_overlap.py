# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import reanalyze_native_prefetch_overlap as runner
from analyze_prefetch_compute_overlap import bind_streams
from diagnose_mte_wait_timeline import read_trace


def capture(root, case, *, bad_tail=False):
    path = root / f"profile/{case}/capture/ASCEND_PROFILER_OUTPUT/trace_view.json"
    path.parent.mkdir(parents=True)
    origin = Decimal("1795507048074000.043")
    records = [dict(ph="M", name="process_name", pid="device", args=dict(name="Ascend Hardware"))]

    def add(name, start, end, tid, pid="device"):
        records.append(dict(ph="X", name=name, ts=str(origin + Decimal(str(start))), dur=end - start, pid=pid, tid=tid))

    def host(kind, layer, call, start, end):
        suffix = f"|backend=cpp|kind={kind}|layer={layer}|rows=3|call={call}"
        for stage, a, b in (
            ("host_callback", start, end),
            ("update_weights", start + 0.1, end - 0.05),
            ("load_submit", start + 0.2, start + 0.6),
            ("copy_submit", start + 0.3, start + 0.5),
            ("synchronize", start + 1, end - 0.1),
        ):
            add("moe_offload/" + stage + suffix, a, b, "worker", "host")

    def handshake(tid, pub, wait, end):
        add("host_sync_publish_1", pub, pub + 1, tid)
        add("host_sync_wait_2", wait, end, tid)

    host("reactive", 0, 1, 10, 20)
    host("prefetch", 1, 2, 100, 160 if case == "native" else 200)
    host("reactive", 1, 3, 212, 240)
    host("prefetch", 2, 4, 300, 320 if case == "native" else 360)
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
        add("UnexpectedCompute" if bad_tail and start == 50 else "MEMCPY_ASYNC", finish + 3, finish + 4, "main")
        add("MEMCPY_ASYNC", finish + 4.25, finish + 4.5, "main")
    # Preserve the actual decimal JSON numbers at an epoch where float loses sub-us precision.
    serialized = re.sub(r'"ts": "([0-9.]+)"', r'"ts": \1', json.dumps(dict(traceEvents=records)))
    path.write_text(serialized, encoding="utf-8")
    path.with_name("clock_alignment.json").write_text(
        json.dumps(
            dict(
                status="calibration_validated",
                calibration_layout="model_request_brackets",
                held_out_after=dict(outside_bracket_1us_tolerance=0),
                aligned_trace=str(path),
                trace_origin_us=str(origin),
                host_span_us=["0", "500"],
                endpoint_mapping_envelope_width_us=["1", "1"],
            )
        ),
        encoding="utf-8",
    )
    sample = dict(digest="same", usage=dict(completion_tokens=5))
    (root / "summary.txt").write_text(
        "PROFILE_ONLY: physical_npu=6\nRESULT "
        + json.dumps(dict(case=case, output_match=True, stream_samples=[sample])),
        encoding="utf-8",
    )
    (root / f"{case}_profile_samples.json").write_text(json.dumps([sample]), encoding="utf-8")
    counters = dict(count=5, mstx_errors=0, errors=0, dropped=0, mstx_enabled=True)
    if case == "native":
        counters.update(native=True, groups=[])
    else:
        counters.update(
            load_backend="python",
            h2d_backend="torch",
            groups=[
                dict(kind="prefetch", token_rows=3, count=2, loads=dict(mean=0)),
                dict(kind="reactive", token_rows=3, count=3, loads=dict(mean=0)),
            ],
        )
    (root / f"{case}_profile_counters.json").write_text(json.dumps(counters), encoding="utf-8")
    return path


class NativePrefetchOverlapTests(unittest.TestCase):
    def test_publish_after_host_start_is_not_accepted_as_valid_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "native", Path(temp) / "analysis_01"
            path = capture(root, "native")
            data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
            publish = next(
                e for e in data["traceEvents"] if e.get("tid") == "prefetch" and e["name"] == "host_sync_publish_1"
            )
            publish["ts"] += 22  # prefetch Host starts at 100, publish now at 102
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 0)
            result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["native"]
            self.assertEqual(result["rows"][0]["class"], "unmapped")
            check = next(c for c in result["mte_order_checks"] if not c["valid"])
            self.assertEqual(check["call"], "2")
            self.assertEqual(check["publish_start_minus_host_start_us"], 2)
            self.assertEqual(result["host_timings"]["all"]["host_callback"]["count"], 5)

    def test_protocol_failures_keep_host_timings_and_mark_only_dependent_rows_unmapped(self):
        # (stream, ordinal wait, new end, expected affected prefetch calls)
        for stream, ordinal, end, affected in (
            ("prefetch", 0, 158, ["2"]),
            ("main", 0, 18, ["2"]),  # previous reactive defines the compute segment
            ("main", 1, 238, ["2", "4"]),  # current reactive AND next segment's anchor
        ):
            with self.subTest(stream=stream, ordinal=ordinal), tempfile.TemporaryDirectory() as temp:
                root, out = Path(temp) / "native", Path(temp) / "analysis_01"
                path = capture(root, "native")
                data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
                waits = [e for e in data["traceEvents"] if e.get("tid") == stream and e["name"] == "host_sync_wait_2"]
                target = waits[ordinal]
                origin = Decimal("1795507048074000.043")
                target["dur"] = origin + end - Decimal(target["ts"])
                path.write_text(json.dumps(data, default=str), encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "aligned protocol order check failed"):
                    bind_streams(read_trace(path, time_origin=origin))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 1 if len(affected) == 2 else 0)
                self.assertEqual(before, path.read_bytes())
                result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["native"]
                self.assertEqual(result["host_timings"]["all"]["host_callback"]["count"], 5)
                self.assertAlmostEqual(result["host_timings"]["prefetch"]["host_callback"]["mean_us"], 40)
                bad = [c for c in result["mte_order_checks"] if not c["valid"]]
                self.assertEqual(len(bad), 1)
                self.assertEqual(bad[0]["host_end_minus_wait_end_us"], 2)
                self.assertEqual([r["prefetch_call"] for r in result["rows"] if r["class"] == "unmapped"], affected)
                for row in result["rows"]:
                    if row["prefetch_call"] in affected:
                        self.assertEqual(row["reason"], "mte_protocol_host_order_mismatch")
                        self.assertNotIn("readiness_minus_prefetch_end_bounds_us", row)
                stats = result["direct"]["all_samples"]
                self.assertEqual(stats["original"]["count"], 2)
                self.assertAlmostEqual(stats["original"]["mean_us"], 53.5)
                self.assertAlmostEqual(
                    stats["method2_subtract_remaining_compute"]["mean_us"], 53.5 if len(affected) == 2 else 37.25
                )
                summary = (out / "summary.txt").read_text(encoding="utf-8")
                self.assertIn("ORDER_FAILURE_EXAMPLES", summary)
                self.assertIn('"host_end_minus_wait_end_us": 2.0', summary)
                self.assertIn("HOST_FUNCTION_TIMES", summary)

    def test_structural_stream_failure_retains_independent_host_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "native", Path(temp) / "analysis_01"
            path = capture(root, "native")
            # Device handshake name unavailable, but all Host markers are valid.
            path.write_text(
                path.read_text(encoding="utf-8").replace("host_sync_publish_1", "unknown_publish"), encoding="utf-8"
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 1)
            result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["native"]
            self.assertEqual(result["classification"]["unmapped"]["count"], 2)
            self.assertTrue(result["binding_error"])
            self.assertAlmostEqual(result["host_timings"]["all"]["update_weights"]["mean_us"], 28.45)
            self.assertEqual(result["direct"]["all_samples"]["method1_drop_late"]["count"], 2)
            self.assertAlmostEqual(
                result["direct"]["all_samples"]["method2_subtract_remaining_compute"]["mean_us"], 53.5
            )

    def test_invalid_host_nesting_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "native", Path(temp) / "analysis_01"
            path = capture(root, "native")
            data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
            sync = next(e for e in data["traceEvents"] if e["name"].startswith("moe_offload/synchronize"))
            sync["dur"] = 1000
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 1)
            self.assertEqual(json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8")), {})

    def test_missing_python_load_metadata_is_unknown_not_zero_or_missing_host_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = [base / c for c in ("python", "native")]
            for root in roots:
                capture(root, root.name)
            path = roots[0] / "python_profile_counters.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["load_backend"]
            del data["h2d_backend"]
            path.write_text(json.dumps(data), encoding="utf-8")
            out = base / "analysis_01"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([*map(str, roots), "--output-dir", str(out)]), 0)
            result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["python"]
            self.assertTrue(result["load_issues"])
            self.assertTrue(all(r["prefetch_loads"] is None for r in result["rows"]))
            self.assertEqual(result["host_timings"]["all"]["host_callback"]["count"], 5)
            self.assertEqual(result["classification"]["unmapped"]["count"], 0)

    def test_offline_actual_capture_analysis_and_both_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = [base / c for c in ("python", "native")]
            for root in roots:
                capture(root, root.name)
            before = {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}
            out = base / "analysis_01"
            with (
                patch("subprocess.Popen") as popen,
                patch("subprocess.run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main([*map(str, roots), "--output-dir", str(out)]), 0)
            popen.assert_not_called()
            run.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()})
            results = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))
            native = results["native"]
            self.assertEqual(native["host_timing_expected_counts"], dict(all=5, prefetch=2, reactive=3))
            self.assertEqual(native["host_timings"]["all"]["host_callback"]["count"], 5)
            self.assertAlmostEqual(native["host_timings"]["all"]["host_callback"]["mean_us"], 28.6)
            self.assertAlmostEqual(results["python"]["host_timings"]["all"]["host_callback"]["mean_us"], 44.6)
            self.assertAlmostEqual(native["host_timings"]["all"]["update_weights"]["mean_us"], 28.45)
            stats = native["direct"]["all_samples"]
            self.assertAlmostEqual(stats["original"]["mean_us"], 53.5)
            self.assertAlmostEqual(stats["method1_drop_late"]["mean_us"], 52)
            self.assertAlmostEqual(stats["method2_subtract_remaining_compute"]["mean_us"], 37.25)
            self.assertEqual(native["direct_classification"]["late"]["count"], 1)
            self.assertTrue(all(r["prefetch_loads"] is None for r in native["rows"]))
            self.assertTrue(all(r["prefetch_loads"] == 0 for r in results["python"]["rows"]))
            pairs, rejected = runner.comparison_rows(results["python"], native)
            self.assertFalse(rejected)
            for pair in pairs:
                self.assertAlmostEqual(pair["right"]["host"] - pair["left"]["host"], -40)
                self.assertAlmostEqual(pair["right"]["compute"] - pair["left"]["compute"], 0)
                self.assertAlmostEqual(pair["right"]["gap"] - pair["left"]["gap"], 40)
            summary = (out / "summary.txt").read_text(encoding="utf-8")
            self.assertIn('"early->late": 1', summary)
            self.assertIn("OVERLAP_COMPLETE", summary)
            self.assertIn("HOST_FUNCTION_TIMES", summary)

    def test_unrecognized_boundary_is_preserved_not_corrected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "native", Path(temp) / "analysis_01"
            capture(root, "native", bad_tail=True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 0)
            result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["native"]
            self.assertEqual(result["classification"]["unmapped"]["count"], 1)
            self.assertAlmostEqual(
                result["direct"]["all_samples"]["method2_subtract_remaining_compute"]["mean_us"], 37.25
            )

    def test_missing_function_marker_has_no_zero_duration_or_bogus_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "native", Path(temp) / "analysis_01"
            path = capture(root, "native")
            path.write_text(path.read_text().replace("moe_offload/update_weights", "unrelated"), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 0)
            result = json.loads((out / "prefetch_overlap.json").read_text(encoding="utf-8"))["native"]
            stat = result["host_timings"]["all"]["update_weights"]
            self.assertEqual(stat["count"], 0)
            self.assertIsNone(stat["mean_us"])
            self.assertIsNone(stat["outliers_us"]["filtered"])

    def test_bad_alignment_counters_or_runtime_identity_rejected(self):
        for error in ("alignment", "count", "backend", "output"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp:
                root, out = Path(temp) / "native", Path(temp) / "analysis_01"
                path = capture(root, "native")
                if error == "alignment":
                    path.with_name("clock_alignment.json").write_text('{"status":"bad"}', encoding="utf-8")
                elif error == "output":
                    (root / "native_profile_samples.json").write_text(
                        '[{"digest":"wrong","usage":{}}]', encoding="utf-8"
                    )
                else:
                    counter_path = root / "native_profile_counters.json"
                    counters = json.loads(counter_path.read_text())
                    counters["count" if error == "count" else "native"] = 4 if error == "count" else False
                    counter_path.write_text(json.dumps(counters), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main([str(root), "--output-dir", str(out)]), 1)
                self.assertIn("OVERLAP_FAILED", (out / "summary.txt").read_text(encoding="utf-8"))

    def test_output_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture(root, "native")
            with self.assertRaises(FileExistsError):
                runner.main([str(root), "--output-dir", str(root)])

    def test_relative_decimal_reader_handles_real_sub_microsecond_nesting(self):
        suffix = "|backend=cpp|layer=26|kind=reactive|rows=3|call=138"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            path.write_text(
                '[{"ph":"X","pid":1,"tid":2,"name":"moe_offload/host_callback'
                + suffix
                + '","ts":1795507048074376.043,"dur":523.856},'
                '{"ph":"X","pid":1,"tid":2,"name":"moe_offload/synchronize'
                + suffix
                + '","ts":1795507048074376.664,"dur":523.135}]',
                encoding="utf-8",
            )
            self.assertTrue(read_trace(path)["issues"])
            self.assertFalse(read_trace(path, time_origin=Decimal("1795507048074000"))["issues"])
            path.write_text(path.read_text().replace('"dur":523.135', '"dur":523.335'), encoding="utf-8")
            self.assertTrue(read_trace(path, time_origin=Decimal("1795507048074000"))["issues"])


if __name__ == "__main__":
    unittest.main()
