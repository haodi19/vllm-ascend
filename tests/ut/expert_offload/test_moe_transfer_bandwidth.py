# SPDX-License-Identifier: Apache-2.0
"""Offline bandwidth units, direction joins, coverage and read-only guarantees."""

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


class TransferBandwidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location(
                "transfer_bandwidth_test", scripts / "inspect_moe_transfer_bandwidth.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.pop(0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "ASCEND_PROFILER_OUTPUT"
        self.output.mkdir()
        self.path = self.output / "ascend_pytorch_profiler_0.db"
        self.stdout = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.stdout)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def database(self):
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE TASK (deviceId INTEGER,streamId INTEGER,globalTaskId INTEGER,
                                   startNs INTEGER,endNs INTEGER);
                CREATE TABLE MEMCPY_INFO (globalTaskId INTEGER,size INTEGER,memcpyOperation INTEGER);
                CREATE TABLE ENUM_MEMCPY_OPERATION (id INTEGER,name TEXT);
                INSERT INTO ENUM_MEMCPY_OPERATION VALUES (17,'host to device');
                INSERT INTO MEMCPY_INFO VALUES (99,8388608,17);
                INSERT INTO TASK VALUES (0,56,99,1000000000000000000,1000000000000100000);
                INSERT INTO TASK VALUES (0,56,99,1000000000001000000,1000000000001300000);
            """)

    def mutate_database(self, query):
        with contextlib.closing(sqlite3.connect(self.path)) as db:
            db.executescript(query)

    def trace(self, extra=None):
        data = [
            {
                "ph": "X",
                "name": "MEMCPY_ASYNC",
                "pid": 5,
                "tid": 900,
                "dur": 100,
                "args": {"Size(bytes)": 8388608, "Direction": "H2D"},
            },
            {
                "ph": "X",
                "name": "MEMCPY_ASYNC",
                "pid": 6,
                "tid": 900,
                "dur": 1,
                "args": {"Size(bytes)": 8388608, "Direction": "H2D"},
            },
            {
                "ph": "X",
                "name": "Transpose",
                "pid": 5,
                "tid": 900,
                "dur": 100,
                "args": {"Size(bytes)": 8388608, "Direction": "H2D"},
            },
            {"ph": "M", "name": "process_name", "pid": 5, "args": {"name": "Ascend Hardware"}},
            {"ph": "M", "name": "process_name", "pid": 6, "args": {"name": "Host"}},
            {"ph": "M", "name": "thread_name", "pid": 5, "tid": 900, "args": {"name": "Stream 56"}},
        ]
        data.extend(extra or [])
        path = self.output / "trace_view.json"
        path.write_text(json.dumps({"traceEvents": data}), encoding="utf-8")
        return path

    def test_ns_bytes_weighted_rate_replays_and_read_only(self):
        self.database()
        before = self.path.read_bytes()
        result = self.module.inspect_capture(self.root, "old")
        stats = result[("0", "56", "H2D", 8388608)]
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["mean_us"], 200)
        self.assertAlmostEqual(stats["gbps"], 41.94304)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.output.iterdir()), [self.path])

    def test_duplicate_info_and_device_intervals_rejected(self):
        self.database()
        self.mutate_database("INSERT INTO MEMCPY_INFO SELECT * FROM MEMCPY_INFO;")
        with self.assertRaisesRegex(ValueError, "multiplying join"):
            self.module.read_database(self.path)
        self.mutate_database("DELETE FROM MEMCPY_INFO WHERE rowid=2; INSERT INTO TASK SELECT * FROM TASK LIMIT 1;")
        with self.assertRaisesRegex(ValueError, "double counting"):
            self.module.read_database(self.path)

    def test_direction_conflict_and_unknown_codes_not_guessed(self):
        self.database()
        self.mutate_database(
            "CREATE TABLE STRING_IDS (id INTEGER,value TEXT); INSERT INTO STRING_IDS VALUES(17,'D2H');"
        )
        self.assertFalse(self.module.read_database(self.path))
        self.mutate_database(
            "DELETE FROM STRING_IDS; DELETE FROM ENUM_MEMCPY_OPERATION; UPDATE MEMCPY_INFO SET memcpyOperation=1;"
        )
        self.assertFalse(self.module.read_database(self.path))

    def test_invalid_duration_or_size_or_lane_excluded(self):
        self.database()
        self.mutate_database("UPDATE TASK SET endNs=startNs;")
        self.assertFalse(self.module.read_database(self.path))
        self.mutate_database("UPDATE TASK SET endNs=startNs+100; UPDATE MEMCPY_INFO SET size=0;")
        self.assertFalse(self.module.read_database(self.path))
        self.mutate_database("UPDATE MEMCPY_INFO SET size=8388608; UPDATE TASK SET deviceId=NULL;")
        self.assertFalse(self.module.read_database(self.path))

    def test_json_device_only_explicit_bytes_and_actual_display_lane(self):
        extra = [
            {
                "ph": "X",
                "name": "MEMCPY_ASYNC",
                "pid": 5,
                "tid": 900,
                "dur": 100,
                "args": {"Size": 8388608, "Direction": "H2D"},
            },
            {
                "ph": "X",
                "name": "MEMCPY_ASYNC",
                "pid": 5,
                "tid": 900,
                "dur": 100,
                "args": {"bytes": 8388608, "Direction": 1},
            },
        ]
        result = self.module.summarize(self.module.read_json_trace(self.trace(extra)))
        self.assertEqual(len(result), 1)
        stats = result[("5", "Stream 56", "H2D", 8388608)]
        self.assertEqual(stats["n"], 1)
        self.assertAlmostEqual(stats["gbps"], 83.88608)

    def test_bad_schema_falls_back_to_trace(self):
        self.mutate_database("CREATE TABLE unrelated (id INTEGER);")
        self.trace()
        self.assertTrue(self.module.inspect_capture(self.root, "old"))
        self.assertIn("DB_UNAVAILABLE", self.stdout.getvalue())

    def test_truncated_trace_is_not_partial_success(self):
        path = self.trace()
        path.write_text(path.read_text(encoding="utf-8")[:-1], encoding="utf-8")
        with self.assertRaises(ValueError):
            self.module.read_json_trace(path)

    def test_multiple_streams_not_automatically_paired(self):
        single = {("0", "56", "H2D", 8388608): {"n": 1, "gbps": 80}}
        other = {**single, ("0", "27", "H2D", 8388608): {"n": 1, "gbps": 40}}
        self.module.compare(single, other)
        self.assertIn("H2D_PAIR_UNRESOLVED", self.stdout.getvalue())
        self.assertIn("COMPARE_UNAVAILABLE", self.stdout.getvalue())

    def test_journal_not_opened(self):
        self.database()
        Path(str(self.path) + "-wal").write_bytes(b"active")
        with self.assertRaisesRegex(ValueError, "activity/consistency unknown"):
            self.module.read_database(self.path)

    def test_aligned_string_duration_and_nonfinite_rejection(self):
        path = self.trace()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["traceEvents"][0]["dur"] = "100.00000000000001"
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = self.module.summarize(self.module.read_json_trace(path))
        self.assertAlmostEqual(result[("5", "Stream 56", "H2D", 8388608)]["gbps"], 83.88608)
        for invalid in (True, None, "NaN", "Infinity", "-1", "0", "wrong", []):
            with self.subTest(value=invalid):
                self.assertIsNone(self.module.duration_number(invalid))
        self.assertIn("'str': 1", self.stdout.getvalue())

    def test_root_mapping_and_exported_manifest_not_raw_replacement(self):
        self.database()
        (self.root / "clock_alignment.json").write_text(
            json.dumps({"status": "calibration_validated", "device_scale": "0.99997"}), encoding="utf-8"
        )
        (self.root / "aligned_profile_manifest.json").write_text(json.dumps({"status": "exported"}), encoding="utf-8")
        self.module.inspect_capture(self.output, "new")
        self.assertIn("ALIGNMENT_MANIFEST=exported; RAW_CAPTURE_REPLACED=False", self.stdout.getvalue())
        self.assertIn("status=calibration_validated", self.stdout.getvalue())

    def test_raw_inventory_is_read_only_and_reports_schema(self):
        raw = self.root / "PROF_example" / "device_0" / "sqlite"
        raw.mkdir(parents=True)
        db_path = raw / "runtime.db"
        with contextlib.closing(sqlite3.connect(db_path)) as db:
            db.executescript("CREATE TABLE MemcpyRecords (task_id INTEGER, size INTEGER, copy_kind INTEGER);")
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.module.inventory_capture(self.output)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertIn("CANDIDATE_SCHEMA MemcpyRecords", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
