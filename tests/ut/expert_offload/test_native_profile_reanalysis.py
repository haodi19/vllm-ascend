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
    def saved(self, root, case):
        trace = root / f"profile/{case}/capture/ASCEND_PROFILER_OUTPUT/trace_view.json"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps(fixture()), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
