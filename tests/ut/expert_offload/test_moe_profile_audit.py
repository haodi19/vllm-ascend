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

    def scan(self, data, *, require_mte=True, wrapped=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            value = dict(traceEvents=data, displayTimeUnit="ms") if wrapped else data
            path.write_text(json.dumps(value), encoding="utf-8")
            return self.module.scan_trace(path, require_mte=require_mte)

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
            self.assertNotIn("SECRET", output.getvalue())
            self.assertNotIn("PRIVATE", output.getvalue())
            contexts = self.module.service_warning_context(root / "mte/inference_python.log")
            self.assertEqual(len(contexts), 2)
            self.assertTrue(all(len(context["context"]) <= 5 for context in contexts))


if __name__ == "__main__":
    unittest.main()
