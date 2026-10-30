# SPDX-License-Identifier: Apache-2.0
"""Synthetic, CPU-only checks for read-only MoE visualization auditing."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MoeProfileAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location("profile_audit_test", root / "script/inspect_moe_profile.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    @staticmethod
    def trace():
        def host(name, call, kind, start, duration, pid=7):
            return dict(
                name=f"moe_offload/{name}|call={call}|kind={kind}|layer=1",
                ph="X",
                pid=pid,
                tid=9,
                ts=start,
                dur=duration,
            )

        def device(name, start, duration):
            return dict(name=name, ph="X", pid=100, tid=21, ts=start, dur=duration)

        return [
            host("host_callback", 1, "prefetch", 10, 20),
            host("update_weights", 1, "prefetch", 12, 15),
            host("host_callback", 2, "reactive", 50, 20),
            host("update_weights", 2, "reactive", 52, 15),
            device("host_sync_publish_1", 4, 1),
            device("host_sync_wait_2", 5, 30),
            device("MEMCPY_ASYNC", 20, 10),
            device("Transpose_int8", 31, 4),
            device("EVENT_WAIT_9", 35, 2),
            dict(name="thread_name", ph="M", pid="100", tid="21", args=dict(name="Stream 21")),
            dict(name="process_name", ph="M", pid="100", args=dict(name="Ascend Hardware")),
        ]

    def scan(self, data, *, require_mte=True, wrapped=False, large_gap_us=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            value = dict(traceEvents=data, displayTimeUnit="ms") if wrapped else data
            path.write_text(json.dumps(value), encoding="utf-8")
            return self.module.scan_trace(path, require_mte=require_mte, large_gap_us=large_gap_us)

    def test_bare_and_wrapped_events_with_metadata_last(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                report = self.scan(self.trace(), wrapped=wrapped)
                self.assertTrue(report["json_complete"])
                self.assertTrue(report["usable"])
                self.assertEqual(report["hardware_pids"], ["100"])
                self.assertEqual(report["nesting"]["matched"], 2)
                self.assertEqual(report["device_counts"]["copy"], 1)
                self.assertEqual(report["device_counts"]["compute_candidate"], 1)
                self.assertEqual(report["device_counts"]["other"], 1)
                self.assertEqual(report["streams"][0]["name"], "Stream 21")
                self.assertEqual(report["host_gaps"][0]["mean_us"], 20)

    def test_numeric_strings_are_normalized_without_mutating_original(self):
        data = self.trace()
        for event in data:
            if event["ph"] == "X":
                event["ts"], event["dur"] = str(event["ts"]), str(event["dur"])
        report = self.scan(data)
        self.assertTrue(report["usable"])
        self.assertEqual(report["streams"][0]["tasks"]["copy"]["mean_us"], 10)
        self.assertIsInstance(data[0]["dur"], str)

    def test_nonfinite_negative_or_boolean_time_is_rejected(self):
        for field, value in (("dur", "nan"), ("ts", "inf"), ("dur", -1), ("ts", -1), ("dur", True)):
            with self.subTest(field=field, value=value):
                data = self.trace()
                data[0][field] = value
                with self.assertRaises(ValueError):
                    self.scan(data)

    def test_missing_metadata_or_device_content_is_not_usable(self):
        for excluded in ("Ascend Hardware", "MEMCPY_ASYNC", "Transpose_int8"):
            with self.subTest(excluded=excluded):
                data = [event for event in self.trace() if excluded not in json.dumps(event)]
                report = self.scan(data)
                self.assertTrue(report["json_complete"])
                self.assertFalse(report["usable"])
                self.assertTrue(report["issues"])

    def test_mstx_strings_in_non_x_events_do_not_validate_ranges(self):
        data = self.trace()
        for event in data:
            if event["name"].startswith("moe_offload/"):
                event["ph"] = "B"
        report = self.scan(data)
        self.assertFalse(report["usable"])
        self.assertEqual(report["mstx_phases"], {"B": 4})
        self.assertEqual(report["host_callbacks"], 0)

    def test_outside_or_wrong_thread_nesting_is_reported(self):
        for mutation in (dict(ts=1), dict(tid=10)):
            with self.subTest(mutation=mutation):
                data = self.trace()
                data[1].update(mutation)
                report = self.scan(data)
                self.assertFalse(report["usable"])
                self.assertEqual(report["nesting"]["invalid"], 1)

    def test_pairing_never_crosses_host_processes(self):
        data = self.trace()
        data[1]["pid"] = 8
        data[3]["pid"] = 8
        report = self.scan(data)
        self.assertFalse(report["usable"])
        self.assertEqual(report["nesting"]["matched"], 0)
        self.assertEqual(report["nesting"]["update_only"], 2)

    def test_mte_requires_both_handshake_kernels_on_hardware_track(self):
        data = [event for event in self.trace() if "host_sync_wait" not in event["name"]]
        self.assertTrue(self.scan(data, require_mte=False)["usable"])
        self.assertFalse(self.scan(data, require_mte=True)["usable"])
        fake = dict(name="host_sync_wait_2", ph="X", pid=7, tid=9, ts=1, dur=1)
        self.assertFalse(self.scan([*data, fake], require_mte=True)["usable"])

    def test_overlap_is_counted_but_not_a_negative_gap(self):
        data = self.trace()
        data[2]["ts"], data[3]["ts"] = 20, 22
        report = self.scan(data)
        self.assertTrue(report["usable"])
        gap = report["host_gaps"][0]
        self.assertEqual(gap["count"], 0)
        self.assertEqual(gap["overlaps_excluded"], 1)
        self.assertIsNone(gap["mean_us"])

    def test_host_gaps_are_separate_by_process(self):
        data = self.trace()
        other = copy.deepcopy(data[:4])
        for event in other:
            event["pid"] = 8
            event["ts"] += 1000
        report = self.scan([*data, *other])
        self.assertEqual(len(report["host_gaps"]), 2)
        self.assertTrue(all(gap["mean_us"] == 20 for gap in report["host_gaps"]))

    def test_duplicate_ids_do_not_overwrite_valid_intervals(self):
        data = self.trace()
        report = self.scan([*data, copy.deepcopy(data[0])])
        self.assertFalse(report["usable"])
        self.assertEqual(report["nesting"]["duplicate_ids"], 1)

    def test_truncated_or_trailing_json_fails_after_events(self):
        for suffix in ("missing_bracket", " extra"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.json"
                raw = json.dumps(self.trace())
                path.write_text(raw[:-1] if suffix == "missing_bracket" else raw + suffix, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.module.scan_trace(path)

    def test_storage_cap_is_enforced(self):
        with patch.object(self.module, "MAX_INTERVALS", 2), self.assertRaises(ValueError):
            self.scan(self.trace())

    @staticmethod
    def memcpy(pid, tid, start, duration):
        return dict(name="MEMCPY_ASYNC", ph="X", pid=pid, tid=tid, ts=start, dur=duration)

    def without_copies(self):
        return [event for event in self.trace() if event["name"] != "MEMCPY_ASYNC"]

    def test_memcpy_gaps_sort_unordered_events_and_include_zero(self):
        data = self.without_copies()
        data[-2:-2] = [
            self.memcpy(100, 21, "50", "5"),
            self.memcpy(100, 21, "5", "5"),
            self.memcpy(100, 21, "0", "5"),
            self.memcpy(100, 21, "20", "0"),
        ]
        report = self.scan(data)
        gap = report["memcpy_gaps"][0]
        self.assertEqual(gap["copy_count"], 4)
        self.assertEqual(gap["count"], 3)
        self.assertEqual(gap["overlaps_excluded"], 0)
        self.assertEqual(gap["mean_us"], 40 / 3)
        self.assertEqual(gap["p50_us"], 10)
        self.assertAlmostEqual(gap["p95_us"], 28)
        self.assertEqual(gap["max_us"], 30)
        self.assertEqual(gap["name"], "Stream 21")
        self.assertTrue(report["usable"])

    def test_memcpy_gaps_never_mix_streams_processes_or_host_copies(self):
        data = self.without_copies()
        data[-2:-2] = [
            self.memcpy(100, 21, 0, 5),
            self.memcpy(100, 21, 15, 5),
            self.memcpy(100, 22, 0, 100),
            self.memcpy(100, 22, 150, 10),
            self.memcpy(200, 21, 1000, 1),
            self.memcpy(200, 21, 1004, 1),
            self.memcpy(7, 21, 0, 9999),
            self.memcpy(7, 21, 10000, 1),
        ]
        data.append(dict(name="process_name", ph="M", pid=200, args=dict(name="Ascend Hardware")))
        report = self.scan(data)
        gaps = {(gap["pid"], gap["tid"]): gap for gap in report["memcpy_gaps"]}
        self.assertEqual(set(gaps), {("100", "21"), ("100", "22"), ("200", "21")})
        self.assertEqual(gaps[("100", "21")]["mean_us"], 10)
        self.assertEqual(gaps[("100", "22")]["mean_us"], 50)
        self.assertEqual(gaps[("200", "21")]["mean_us"], 3)

    def test_nested_copy_does_not_create_false_idle_interval(self):
        data = self.without_copies()
        data[-2:-2] = [
            self.memcpy(100, 21, 0, 100),
            self.memcpy(100, 21, 10, 10),
            self.memcpy(100, 21, 30, 10),
            self.memcpy(100, 21, 90, 20),
            self.memcpy(100, 21, 110, 0),
            self.memcpy(100, 21, 120, 5),
        ]
        gap = self.scan(data)["memcpy_gaps"][0]
        self.assertEqual(gap["overlaps_excluded"], 3)
        self.assertEqual(gap["copy_count"], 6)
        self.assertEqual(gap["count"], 2)
        self.assertEqual(gap["mean_us"], 5)
        self.assertEqual(gap["max_us"], 10)

    def test_single_copy_reports_no_gap_sample_instead_of_zero_latency(self):
        gap = self.scan(self.trace())["memcpy_gaps"][0]
        self.assertEqual(gap["copy_count"], 1)
        self.assertEqual(gap["count"], 0)
        self.assertEqual(gap["overlaps_excluded"], 0)
        for field in ("mean_us", "p50_us", "p95_us", "max_us"):
            self.assertIsNone(gap[field])
        self.assertEqual(self.scan(self.without_copies())["memcpy_gaps"], [])

    def test_memcpy_retained_intervals_count_against_storage_cap(self):
        # Four Host intervals, two tracks, one copy and two metadata entries.
        with patch.object(self.module, "MAX_INTERVALS", 9):
            self.assertTrue(self.scan(self.trace())["usable"])
        with patch.object(self.module, "MAX_INTERVALS", 8), self.assertRaises(ValueError):
            self.scan(self.trace())

    def test_large_memcpy_gaps_keep_full_stats_and_include_threshold_equality(self):
        data = self.without_copies()
        data.extend(
            self.memcpy(100, 21, start, duration)
            for start, duration in ((0, 10), (5, 2), (10, 10), (70, 10), (180, 10), (390, 10))
        )
        original = self.scan(data)["memcpy_gaps"][0]
        filtered = self.scan(data, large_gap_us=100)["memcpy_gaps"][0]
        self.assertEqual({key: filtered[key] for key in original}, original)
        self.assertEqual(filtered["count"], 4)  # Zero/50/100/200; overlap remains excluded.
        self.assertEqual(filtered["overlaps_excluded"], 1)
        self.assertEqual(filtered["large_gap_threshold_us"], 100)
        self.assertEqual(filtered["large_gap_fraction"], 0.5)
        self.assertEqual(
            filtered["large_gaps"],
            dict(count=2, total_us=300, mean_us=150, p50_us=150, p95_us=195, max_us=200),
        )
        self.assertNotIn("large_gaps", original)

    def test_large_host_gaps_remain_separate_by_direction(self):
        data = [event for event in self.trace() if not event["name"].startswith("moe_offload/")]
        for call, start in enumerate((10, 50, 170, 390, 415, 535), start=1):
            kind = "prefetch" if call % 2 else "reactive"
            for name, offset, duration in (("host_callback", 0, 20), ("update_weights", 2, 15)):
                data.append(
                    dict(
                        name=f"moe_offload/{name}|call={call}|kind={kind}|layer=1",
                        ph="X",
                        pid=7,
                        tid=9,
                        ts=start + offset,
                        dur=duration,
                    )
                )
        original = {item["direction"]: item for item in self.scan(data)["host_gaps"]}
        filtered = {item["direction"]: item for item in self.scan(data, large_gap_us=100)["host_gaps"]}
        self.assertEqual(set(original), set(filtered))
        for direction, values in original.items():
            self.assertEqual({key: filtered[direction][key] for key in values}, values)
        first, second = filtered["prefetch->reactive"], filtered["reactive->prefetch"]
        self.assertEqual(first["count"], 3)
        self.assertEqual(first["large_gaps"]["count"], 2)
        self.assertEqual(first["large_gaps"]["total_us"], 300)
        self.assertEqual(first["large_gap_fraction"], 2 / 3)
        self.assertEqual(second["count"], 2)
        self.assertEqual(second["large_gaps"]["count"], 1)
        self.assertEqual(second["large_gaps"]["total_us"], 100)
        self.assertEqual(second["large_gap_fraction"], 0.5)

    def test_large_gap_zero_matches_and_no_samples_are_distinct(self):
        report = self.scan(self.trace(), large_gap_us=100)
        host, copy_gap = report["host_gaps"][0], report["memcpy_gaps"][0]
        self.assertEqual(host["count"], 1)
        self.assertEqual(host["large_gap_fraction"], 0)
        self.assertEqual(copy_gap["count"], 0)
        self.assertIsNone(copy_gap["large_gap_fraction"])
        for gap in (host, copy_gap):
            self.assertEqual(gap["large_gaps"]["count"], 0)
            self.assertEqual(gap["large_gaps"]["total_us"], 0)
            self.assertIsNone(gap["large_gaps"]["mean_us"])
            self.assertIsNone(gap["large_gaps"]["p95_us"])
        data = self.trace()
        data[2]["ts"], data[3]["ts"] = 20, 22
        overlap = self.scan(data, large_gap_us=100)["host_gaps"][0]
        self.assertEqual(overlap["overlaps_excluded"], 1)
        self.assertIsNone(overlap["large_gap_fraction"])

    def test_invalid_large_gap_threshold_fails_before_reading_trace(self):
        for value in (0, -1, float("nan"), float("inf"), float("-inf"), True, "bad"):
            with (
                self.subTest(value=value),
                patch.object(self.module, "events", side_effect=AssertionError("Must validate before file read")),
                self.assertRaisesRegex(ValueError, "正的有限"),
            ):
                self.module.scan_trace("does-not-exist.json", large_gap_us=value)

    def test_cli_large_gap_threshold_is_shared_by_both_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("callback", "mte"):
                path = root / "profile" / backend / "python/worker/trace_view.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(self.trace()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = self.module.main([str(root), "--gaps-only", "--large-gap-us", "20"])
            self.assertEqual(code, 0)
            host_rows, copy_rows = [], []
            for line in output.getvalue().splitlines():
                if line.startswith("HOST_CALLBACK_GAP "):
                    host_rows.append(json.loads(line.split(" ", 1)[1]))
                elif line.startswith("MEMCPY_GAP "):
                    copy_rows.append(json.loads(line.split(" ", 1)[1]))
            self.assertEqual(len(host_rows), 2)
            self.assertEqual(len(copy_rows), 2)
            self.assertTrue(all(row["large_gap_threshold_us"] == 20 for row in host_rows + copy_rows))
            self.assertTrue(all(row["large_gaps"]["count"] == 1 for row in host_rows))
            self.assertIn("仅按时长 >= 阈值筛选", output.getvalue())

    def test_cli_rejects_nonpositive_and_nonfinite_large_gap_threshold(self):
        for text in ("0", "-2", "nan", "inf", "abc"):
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.module.main(["unused", f"--large-gap-us={text}"])

    def test_gaps_only_keeps_results_without_verbose_tracks_or_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("callback", "mte"):
                path = root / "profile" / backend / "python/worker/trace_view.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(self.trace()), encoding="utf-8")
            output = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                patch.object(self.module, "service_warning_context", side_effect=AssertionError("Do not read logs")),
            ):
                code = self.module.main([str(root), "--gaps-only"])
            self.assertEqual(code, 0)
            self.assertIn("HOST_CALLBACK_GAP ", output.getvalue())
            self.assertIn("MEMCPY_GAP ", output.getvalue())
            self.assertNotIn("DEVICE_STREAM ", output.getvalue())
            self.assertNotIn("SERVICE_WARNING ", output.getvalue())
            for line in output.getvalue().splitlines():
                if line.startswith("TRACE_CHECK "):
                    self.assertEqual(
                        set(json.loads(line.split(" ", 1)[1])),
                        {"json_complete", "usable", "issues", "host_callbacks"},
                    )

    def test_cli_and_warning_context_are_read_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("callback", "mte"):
                path = root / "profile" / backend / "python/worker/ASCEND_PROFILER_OUTPUT/trace_view.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(self.trace()), encoding="utf-8")
                log = root / backend / "inference_python.log"
                log.parent.mkdir()
                log.write_text(
                    "INFO request prompt=SECRET\n"
                    "ERROR External init callback must run in same thread as registerClient (1 != 2)\n"
                    "INFO args=SECRET\n"
                    "[MOE-MTE RUNNER] cleanup_begin\n"
                    "EngineDeadError PRIVATE\n"
                    "INFO body=SECRET\n",
                    encoding="utf-8",
                )
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = self.module.main([str(root)])
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(code, 0)
            self.assertEqual(before, after)
            self.assertIn("BASIC_VISUALIZATION=PASS", output.getvalue())
            self.assertEqual(sum(line.startswith("MEMCPY_GAP ") for line in output.getvalue().splitlines()), 2)
            for line in output.getvalue().splitlines():
                if line.startswith("TRACE_CHECK "):
                    self.assertNotIn("memcpy_gaps", json.loads(line.split(" ", 1)[1]))
            self.assertNotIn("SECRET", output.getvalue())
            self.assertNotIn("PRIVATE", output.getvalue())
            contexts = self.module.service_warning_context(root / "mte/inference_python.log")
            self.assertEqual(len(contexts), 2)
            self.assertTrue(all(len(context["context"]) <= 5 for context in contexts))

    def test_cli_selects_cpp_profile_and_logs_and_keeps_python_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for backend in ("callback", "mte"):
                for policy in ("python", "cpp"):
                    path = root / "profile" / backend / policy / "worker/trace_view.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(self.trace()), encoding="utf-8")
            for options, policy in (([], "python"), (["--cache-policy-backend", "cpp"], "cpp")):
                with (
                    self.subTest(policy=policy),
                    contextlib.redirect_stdout(io.StringIO()),
                    patch.object(self.module, "scan_trace", wraps=self.module.scan_trace) as scan,
                    patch.object(self.module, "service_warning_context", return_value=[]) as warnings,
                ):
                    self.assertEqual(self.module.main([str(root), *options]), 0)
                    self.assertEqual(
                        [call.args[0] for call in scan.call_args_list],
                        [
                            root / "profile" / backend / policy / "worker/trace_view.json"
                            for backend in ("callback", "mte")
                        ],
                    )
                    self.assertEqual(
                        [call.args[0] for call in warnings.call_args_list],
                        [root / backend / f"inference_{policy}.log" for backend in ("callback", "mte")],
                    )


if __name__ == "__main__":
    unittest.main()
