# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sqlite3
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import align_moe_profile_clock as alignment
import export_moe_aligned_profile as export
import test_profile_clock_alignment as fixtures


class AlignedDirectoryTests(unittest.TestCase):
    def setUp(self):
        # Reuse the realistic epoch-sized synthetic calibration, not its test suite.
        fixture = fixtures.ProfileAlignmentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.mapping = alignment.align_capture(self.root, fixture.rows)
        self.destination = self.root.parent / "views" / self.root.name
        self.db_path = fixture.trace.parent / "ascend_pytorch_profiler_0.db"
        self.epoch = Decimal("1794496000000000000")

    def database(self):
        with sqlite3.connect(self.db_path) as db:
            db.executescript(
                "CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY,value TEXT);"
                "CREATE TABLE MSTX_EVENTS(startNs INTEGER,endNs INTEGER,message INTEGER,globalTid INTEGER);"
                "CREATE TABLE TASK(startNs INTEGER,endNs INTEGER,deviceId INTEGER,streamId INTEGER,"
                "globalTaskId INTEGER,taskId INTEGER,connectionId INTEGER,depth INTEGER);"
                "CREATE TABLE COMPUTE_TASK_INFO(globalTaskId INTEGER,name INTEGER);"
                "CREATE TABLE CANN_API(startNs INTEGER,endNs INTEGER,name INTEGER);"
                "CREATE TABLE OVERLAP_ANALYSIS(startNs INTEGER,endNs INTEGER,deviceId INTEGER,type INTEGER);"
            )
            for index, event in enumerate(self.fixture.items):
                if event.get("ph") != "X":
                    continue
                start = int(Decimal(event["ts"]) * 1000 - self.epoch)
                end = int(start + Decimal(event["dur"]) * 1000)
                db.execute("INSERT INTO STRING_IDS VALUES(?,?)", (index, event["name"]))
                if event["pid"] == 1:
                    db.execute("INSERT INTO MSTX_EVENTS VALUES(?,?,?,?)", (start, end, index, event["tid"]))
                else:
                    db.execute(
                        "INSERT INTO TASK VALUES(?,?,?,?,?,?,?,?)",
                        (start, end, 1, event["tid"], index, index, index, 0),
                    )
                    db.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(?,?)", (index, index))
            db.execute("INSERT INTO CANN_API VALUES(100,200,3)")
            db.execute("INSERT INTO OVERLAP_ANALYSIS SELECT startNs,endNs,deviceId,0 FROM TASK LIMIT 1")

    def read(self, path, table):
        with contextlib.closing(sqlite3.connect(path)) as db:
            return db.execute(f'SELECT * FROM "{table}"').fetchall()

    def add_rows(self, db):
        return db.execute(
            "SELECT t.* FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId "
            "JOIN STRING_IDS s ON s.id=c.name WHERE s.value=? ORDER BY t.startNs,t.endNs",
            (self.mapping["calibration_stream"][2],),
        ).fetchall()

    def test_model_adds_on_other_streams_do_not_hide_calibration_lane(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            probes = self.add_rows(db)
            # DB stream numbers need not be the JSON lane numbers. Another model
            # lane can even have exactly the probe count, but different timestamps.
            db.execute("UPDATE TASK SET streamId=81 WHERE streamId=9")
            for stream, subset in ((82, probes), (83, probes[:5])):
                for row in subset:
                    db.execute(
                        "INSERT INTO TASK VALUES(?,?,?,?,?,?,?,?)",
                        (row[0] + 1000000, row[1] + 1000000, row[2], stream, *row[4:]),
                    )
        original = self.db_path.read_bytes()
        report = export.export_capture(self.root, self.mapping, self.destination)
        for check in ("before_check", "after_check"):
            self.assertEqual(report["database"][check]["calibration_stream_id"], 81)
            self.assertEqual(report["database"][check]["device_anchors"], len(probes))
            self.assertEqual(report["database"][check]["same_name_streams"], 3)
        target = self.destination / export.OUTPUT_NAME / self.db_path.name
        for old, new in zip(self.read(self.db_path, "TASK"), self.read(target, "TASK")):
            self.assertEqual(old[2:], new[2:])
            self.assertEqual(new[:2], tuple(export.convert_device_ns(v, self.mapping, self.epoch) for v in old[:2]))
        self.assertEqual(self.db_path.read_bytes(), original)

    def test_two_full_matching_lanes_are_not_guessed_even_if_one_has_json_tid(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            for row in self.add_rows(db):
                db.execute("INSERT INTO TASK VALUES(?,?,?,?,?,?,?,?)", (*row[:3], 19, *row[4:]))
        original = self.db_path.read_bytes()
        with self.assertRaisesRegex(ValueError, '"timestamp_matched_streams": 2'):
            export.export_capture(self.root, self.mapping, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.db_path.read_bytes(), original)

    def test_incomplete_or_duplicate_probe_lane_is_not_matched_as_a_subset(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            row = self.add_rows(db)[0]
            db.execute("DELETE FROM TASK WHERE globalTaskId=?", (row[4],))
        with self.assertRaisesRegex(ValueError, '"count": 15'):
            export.export_capture(self.root, self.mapping, self.destination)
        with sqlite3.connect(self.db_path) as db:
            db.executemany("INSERT INTO TASK VALUES(?,?,?,?,?,?,?,?)", [row, row])
        with self.assertRaisesRegex(ValueError, '"count": 17'):
            export.export_capture(self.root, self.mapping, self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_name_reports_available_add_names_without_guessing(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE STRING_IDS SET value='different_Add_kernel' WHERE value=?",
                (self.mapping["calibration_stream"][2],),
            )
        original = self.db_path.read_bytes()
        with self.assertRaisesRegex(ValueError, '"available_add_names": \\[\\["different_Add_kernel", 16\\]\\]'):
            export.export_capture(self.root, self.mapping, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.db_path.read_bytes(), original)

    def test_database_layout_and_host_preserved_device_converted_original_untouched(self):
        self.database()
        originals = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = export.export_capture(self.root, self.mapping, self.destination)
        self.assertEqual(report["format"], "db")
        target = self.destination / export.OUTPUT_NAME / self.db_path.name
        for table in ("MSTX_EVENTS", "CANN_API", "STRING_IDS", "COMPUTE_TASK_INFO"):
            self.assertEqual(self.read(self.db_path, table), self.read(target, table))
        before, after = self.read(self.db_path, "TASK"), self.read(target, "TASK")
        for old, new in zip(before, after):
            self.assertEqual(old[2:], new[2:])  # IDs, streams, nesting/depth, connections.
            expected = tuple(export.convert_device_ns(v, self.mapping, self.epoch) for v in old[:2])
            self.assertEqual(new[:2], expected)
        self.assertNotEqual(before[0][:2], after[0][:2])
        self.assertNotEqual(self.read(self.db_path, "OVERLAP_ANALYSIS"), self.read(target, "OVERLAP_ANALYSIS"))
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        self.assertEqual(
            (self.destination / export.OUTPUT_NAME / "trace_view.json").read_bytes(),
            Path(self.mapping["aligned_trace"]).read_bytes(),
        )

    def test_text_directory_keeps_lane_metadata_omits_cached_db_and_raw_binary(self):
        (self.root / "profiler_info_0.json").write_text('{"rank":0}', encoding="utf-8")
        output = self.fixture.trace.parent
        (output / "mindstudio_insight_data.db").write_bytes(b"stale GUI cache")
        (output / "analysis.db").write_bytes(b"stale derived summaries")
        (output / "kernel_details.csv").write_text("Name,Duration(us)\nAdd,4\n", encoding="utf-8")
        report = export.export_capture(self.root, self.mapping, self.destination)
        self.assertEqual(report["format"], "text")
        self.assertEqual(list(self.destination.rglob("*.db")), [])
        self.assertFalse((self.destination / "PROF_test").exists())
        self.assertTrue((self.destination / "profiler_info_0.json").is_file())
        trace = json.loads((self.destination / export.OUTPUT_NAME / "trace_view.json").read_text())
        self.assertEqual(trace["traceEvents"][:2], self.fixture.items[:2])

    def test_device_units_mismatch_fails_without_publishing(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE TASK SET startNs=startNs+100000,endNs=endNs+100000")
        original = self.db_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "device timestamps differ"):
            export.export_capture(self.root, self.mapping, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.db_path.read_bytes(), original)
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_host_mismatch_and_ambiguous_devices_rejected(self):
        self.database()
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE MSTX_EVENTS SET endNs=endNs+100000 WHERE rowid=1")
        with self.assertRaisesRegex(ValueError, "Host timestamps"):
            export.export_capture(self.root, self.mapping, self.destination)
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE MSTX_EVENTS SET endNs=endNs-100000 WHERE rowid=1")
            db.execute("UPDATE TASK SET deviceId=2 WHERE rowid=1")
        with self.assertRaisesRegex(ValueError, "single identified device"):
            export.export_capture(self.root, self.mapping, self.destination)

    def test_no_overwrite_no_nested_destination_no_unvalidated_alignment(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            export.export_capture(self.root, self.mapping, self.root / "copy_ascend_pt")
        with self.assertRaisesRegex(ValueError, "independent validation"):
            export.export_capture(self.root, dict(self.mapping, status="failed"), self.destination)
        export.export_capture(self.root, self.mapping, self.destination)
        with self.assertRaisesRegex(ValueError, "new"):
            export.export_capture(self.root, self.mapping, self.destination)

    def test_active_journal_and_unsupported_domain_rejected(self):
        self.database()
        journal = Path(str(self.db_path) + "-journal")
        journal.write_bytes(b"active")
        with self.assertRaisesRegex(ValueError, "active journal"):
            export.export_capture(self.root, self.mapping, self.destination)
        journal.unlink()
        with sqlite3.connect(self.db_path) as db:
            db.executescript("CREATE TABLE COMMUNICATION_OP(startNs INTEGER); INSERT INTO COMMUNICATION_OP VALUES(1)")
        with self.assertRaisesRegex(ValueError, "unsupported populated"):
            export.export_capture(self.root, self.mapping, self.destination)

    def test_offline_run_export(self):
        run = self.root.parent / "run"
        case = run / "profile/mte_combined/cpp"
        case.mkdir(parents=True)
        self.root.rename(case / self.root.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(export.main(["--run", str(run)]), 0)
        target = run / "profile_aligned/mte_combined/cpp" / self.root.name
        self.assertTrue((target / export.OUTPUT_NAME / "trace_view.json").is_file())

    def test_stale_aligned_json_is_not_packaged_as_corrected(self):
        aligned = Path(self.mapping["aligned_trace"])
        aligned.write_bytes(self.fixture.trace.read_bytes())
        with self.assertRaisesRegex(ValueError, "does not implement"):
            export.export_capture(self.root, self.mapping, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
