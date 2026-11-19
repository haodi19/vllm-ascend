# SPDX-License-Identifier: Apache-2.0
"""Offline recovery exports only a verified copy and distinguishes failure from missing data."""

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch


class BandwidthRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location(
                "bandwidth_recovery_test", scripts / "recover_moe_transfer_bandwidth.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            sys.path.pop(0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.capture = self.base / "original"
        self.raw = self.capture / "PROF_saved"
        self.data = self.raw / "device_0" / "data"
        self.data.mkdir(parents=True)
        (self.data / "ffts_profile.data.0.slice_0").write_bytes(b"original binary data")
        self.output = self.base / "recovered"
        self.stdout = io.StringIO()
        redirect = contextlib.redirect_stdout(self.stdout)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def fake_export(self, msprof, copied, log, timeout):
        self.assertNotEqual(copied, self.raw)
        self.assertEqual(copied.parent, self.output)
        self.assertEqual(self.module.fingerprint_tree(copied), self.module.fingerprint_tree(self.raw))
        with contextlib.closing(sqlite3.connect(copied / "msprof_123.db")) as db:
            db.executescript("""
                CREATE TABLE TASK (deviceId INTEGER,streamId INTEGER,globalTaskId INTEGER,
                                   startNs INTEGER,endNs INTEGER);
                CREATE TABLE MEMCPY_INFO (globalTaskId INTEGER,size INTEGER,memcpyOperation INTEGER);
                CREATE TABLE ENUM_MEMCPY_OPERATION (id INTEGER,name TEXT);
                INSERT INTO ENUM_MEMCPY_OPERATION VALUES (1,'host to device');
                INSERT INTO MEMCPY_INFO VALUES (99,8388608,1);
                INSERT INTO TASK VALUES (0,56,99,100000,200000);
            """)
        log.write_text("offline export completed\n", encoding="utf-8")
        return 0

    def test_recovers_h2d_from_copy_source_unchanged_and_summary_complete(self):
        before = self.module.fingerprint_tree(self.raw)
        with patch.object(self.module, "run_export", side_effect=self.fake_export):
            code = self.module.recover(self.capture, self.output, Path("msprof"), 20)
        self.assertEqual(code, 0)
        self.assertEqual(self.module.fingerprint_tree(self.raw), before)
        report = json.loads((self.output / "recovery_result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "H2D_RECOVERED")
        self.assertAlmostEqual(report["groups"][0]["gbps"], 83.88608)
        summary = (self.output / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("SOURCE_UNCHANGED=True", summary)
        self.assertIn("83.886", summary)
        self.assertEqual(summary, self.stdout.getvalue())

    def test_export_failure_not_claimed_missing_data(self):
        with patch.object(self.module, "run_export", return_value=7):
            code = self.module.recover(self.capture, self.output, Path("msprof"), 20)
        self.assertEqual(code, 1)
        self.assertIn("RECOVERY_STATUS=EXPORT_FAILED", self.stdout.getvalue())
        self.assertNotIn("NO_USABLE_H2D_RECORDS", self.stdout.getvalue())

    def test_export_success_without_fields_is_not_bandwidth_zero(self):
        def export_without_fields(msprof, copied, log, timeout):
            with contextlib.closing(sqlite3.connect(copied / "msprof_123.db")) as db:
                db.execute("CREATE TABLE TASK (taskId INTEGER)")
                db.commit()
            return 0

        with patch.object(self.module, "run_export", side_effect=export_without_fields):
            code = self.module.recover(self.capture, self.output, Path("msprof"), 20)
        self.assertEqual(code, 2)
        self.assertIn("NO_USABLE_H2D_RECORDS", self.stdout.getvalue())

    def test_output_cannot_overlap_source_or_existing_directory(self):
        for output in (self.capture, self.capture / "result", self.base):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.module.recover(self.capture, output, Path("msprof"), 20)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.module.recover(self.capture, self.output, Path("msprof"), 20)

    def test_multiple_raw_captures_rejected(self):
        (self.capture / "PROF_another").mkdir()
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.module.source_capture(self.capture)

    def test_copy_corruption_prevents_export(self):
        original = self.module.shutil.copytree

        def bad_copy(source, target, *args, **kwargs):
            result = original(source, target, *args, **kwargs)
            if Path(source) == self.raw:
                (target / "device_0" / "data" / "ffts_profile.data.0.slice_0").write_bytes(b"corrupt")
            return result

        with (
            patch.object(self.module.shutil, "copytree", side_effect=bad_copy),
            patch.object(self.module, "run_export") as run,
        ):
            code = self.module.recover(self.capture, self.output, Path("msprof"), 20)
        self.assertEqual(code, 1)
        run.assert_not_called()

    def test_prior_exports_not_labeled_new_recovery(self):
        (self.raw / "msprof_old.db").write_bytes(b"old export")
        with patch.object(self.module, "run_export") as run:
            code = self.module.recover(self.capture, self.output, Path("msprof"), 20)
        self.assertEqual(code, 1)
        run.assert_not_called()

    def test_only_offline_arguments(self):
        command = self.module.export_command(PurePosixPath("/opt/msprof"), PurePosixPath("/scratch/PROF_copy"))
        self.assertEqual(
            command, ["/opt/msprof", "--export=on", "--output=/scratch/PROF_copy", "--type=text", "--clear=off"]
        )
        with self.assertRaises(ValueError):
            self.module.export_command(PurePosixPath("/opt/msprof"), PurePosixPath("/scratch/bad;path"))

    def test_export_monitor_saves_real_child_output(self):
        log = self.base / "export.log"
        with patch.object(self.module, "export_command", return_value=[sys.executable, "-c", "print('export output')"]):
            self.assertEqual(self.module.run_export(Path("msprof"), self.raw, log, 10), 0)
        self.assertIn("export output", log.read_text(encoding="utf-8"))

    def test_export_timeout_stops_child(self):
        log = self.base / "export.log"
        with (
            patch.object(
                self.module, "export_command", return_value=[sys.executable, "-c", "import time; time.sleep(60)"]
            ),
            self.assertRaisesRegex(TimeoutError, "timed out"),
        ):
            self.module.run_export(Path("msprof"), self.raw, log, 0.1)


if __name__ == "__main__":
    unittest.main()
