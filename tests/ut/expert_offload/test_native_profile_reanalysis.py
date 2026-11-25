# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_native_moe_profile import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import reanalyze_native_moe_profiles as runner


class NativeReanalysisTests(unittest.TestCase):
    def test_grouped_artifacts_support_reanalysis_and_do_not_fall_back_when_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            self.saved(root, "native")
            artifacts = root / "native"
            artifacts.mkdir()
            for filename in ("profile_samples.json", "profile_counters.json"):
                legacy = root / f"native_{filename}"
                (artifacts / filename).write_bytes(legacy.read_bytes())
                legacy.unlink()
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = Path(directory) / "analysis"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(output)]), 0)
            self.assertTrue((output / "native/profile_report.json").is_file())
            self.assertTrue((output / "native/bubble_details.log").is_file())
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            # A stale legacy success must not mask an invalid current report.
            (root / "native_profile_counters.json").write_bytes((artifacts / "profile_counters.json").read_bytes())
            (artifacts / "profile_counters.json").write_text(
                '{"count":3,"mstx_errors":0,"errors":0,"dropped":1}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incomplete saved Host counters"):
                runner.saved_case(root, "native")

    def saved(self, root, case):
        trace = root / f"profile/{case}/capture/ASCEND_PROFILER_OUTPUT/trace_view.json"
        trace.parent.mkdir(parents=True)
        data = fixture()
        data["traceEvents"].append(
            dict(ph="X", name="moe_offload/loading_stream|schema=1|stream_id=56|device=0", pid=1, tid=8, ts=0, dur=1)
        )
        trace.write_text(json.dumps(data), encoding="utf-8")
        trace.with_name("clock_alignment.json").write_text(
            json.dumps(
                dict(
                    status="calibration_validated",
                    held_out_after=dict(outside_bracket_1us_tolerance=0),
                    aligned_trace=str(trace),
                )
            ),
            encoding="utf-8",
        )
        sample = dict(digest="same", usage=dict(completion_tokens=5))
        record = dict(case=case, output_match=True, stream_samples=[sample])
        (root / "summary.txt").write_text(
            "PROFILE_ONLY: physical_npu=6\nRESULT " + json.dumps(record), encoding="utf-8"
        )
        (root / f"{case}_profile_samples.json").write_text(json.dumps([sample]), encoding="utf-8")
        (root / f"{case}_profile_counters.json").write_text(
            '{"count": 3, "mstx_errors": 0, "errors": 0, "dropped": 0}', encoding="utf-8"
        )
        return trace

    def test_two_saved_runs_recomputed_without_subprocess_or_original_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = [base / case for case in ("python", "native")]
            for root in roots:
                self.saved(root, root.name)
            before = {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()}
            output = base / "analysis_01"
            with (
                patch("subprocess.Popen") as popen,
                patch("subprocess.run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main([*map(str, roots), "--output-dir", str(output)]), 0)
            popen.assert_not_called()
            run.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()})
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn('"same_physical_npu": true', summary)
            self.assertIn("| native | Host prefetch->reactive", summary)
            self.assertIn("| python | Loading bubbles full_capture", summary)
            self.assertIn("REANALYSIS_PASS", summary)

    def test_bad_alignment_outputs_counters_or_real_nesting_fail(self):
        for error in ("alignment", "digest", "dropped", "nesting"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root, output = base / "native", base / "analysis_01"
                trace = self.saved(root, "native")
                if error == "alignment":
                    trace.with_name("clock_alignment.json").write_text('{"status": "failed"}', encoding="utf-8")
                elif error == "digest":
                    (root / "native_profile_samples.json").write_text(
                        '[{"digest":"wrong", "usage":{"completion_tokens":5}}]', encoding="utf-8"
                    )
                elif error == "dropped":
                    (root / "native_profile_counters.json").write_text(
                        '{"count":3,"mstx_errors":0,"errors":0,"dropped":1}', encoding="utf-8"
                    )
                else:
                    data = fixture()
                    data["traceEvents"][5]["dur"] = 200
                    trace.write_text(json.dumps(data), encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main([str(root), "--output-dir", str(output)]), 1)
                summary = (output / "summary.txt").read_text(encoding="utf-8")
                self.assertIn("REANALYSIS_FAILED", summary)
                self.assertNotIn("REANALYSIS_PASS", summary)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.txt"
            summary.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.main([str(root), "--output-dir", str(root)])
            self.assertEqual(summary.read_text(encoding="utf-8"), "original")

    def test_changed_stream_name_is_listed_read_only_then_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "native"
            trace = self.saved(root, "native")
            data = fixture()
            # Display name intentionally differs from tid. Do not guess by tid.
            data["traceEvents"][1]["args"]["name"] = "Stream 54"
            trace.write_text(json.dumps(data), encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            stdout = io.StringIO()
            with (
                patch("subprocess.Popen") as popen,
                patch("subprocess.run") as run,
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(runner.main([str(root), "--list-streams"]), 0)
            popen.assert_not_called()
            run.assert_not_called()
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertEqual(list(base.iterdir()), [root])
            row = next(line for line in stdout.getvalue().splitlines() if line.startswith("COPY_TRACK "))
            track = json.loads(row.split(" ", 1)[1])
            self.assertEqual((track["name"], track["tid"], track["copies"]), ("Stream 54", "56", 4))
            self.assertEqual(track["missing_bytes"], 4)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([str(root), "--output-dir", str(base / "wrong_01")]), 1)
                self.assertEqual(
                    runner.main(
                        [str(root), "--bubble-stream-name", "Stream 54", "--output-dir", str(base / "analysis_01")]
                    ),
                    0,
                )
            summary = (base / "analysis_01/summary.txt").read_text(encoding="utf-8")
            self.assertIn("Loading bubbles full_capture | 2 | 95", summary)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_list_rejects_conflicting_track_names_without_printing_partial_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.saved(root, "native")
            data = fixture()
            data["traceEvents"].append(dict(name="thread_name", pid=2, tid=56, ph="M", args=dict(name="Stream 54")))
            trace.write_text(json.dumps(data), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(runner.main([str(root), "--list-streams"]), 1)
            self.assertIn("INVENTORY_FAILED", stdout.getvalue())
            self.assertNotIn("COPY_TRACK ", stdout.getvalue())
            self.assertNotIn("INVENTORY_COMPLETE", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
