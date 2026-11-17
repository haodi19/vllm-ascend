# SPDX-License-Identifier: Apache-2.0
import builtins
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import align_moe_profile_clock as alignment


class ProfileAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model_ascend_pt"
        metadata = self.root / "PROF_test/host/start_info"
        metadata.parent.mkdir(parents=True)
        self.origin = Decimal("1794496322300000")
        self.raw_origin = 200_000_000_000
        metadata.write_text(
            repr({"collectionTimeBegin": str(self.origin), "clockMonotonicRaw": self.raw_origin}), encoding="utf-8"
        )
        self.trace = self.root / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        self.trace.parent.mkdir()
        self.scale = Decimal("0.999971")
        self.offset = Decimal("-140")
        self.rows = []
        self.items = [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Python"}},
            {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        ]
        for i in range(16):
            relative = Decimal(i * 10000 + (10_000_000 if i >= 8 else 0))
            raw = self.raw_origin + int(relative * 1000)
            self.rows.append(
                {
                    "sample": i,
                    "phase": "before" if i < 8 else "after",
                    "start_before_ns": raw - 4000,
                    "start_after_ns": raw - 2000,
                    "submit_before_ns": raw,
                    "sync_after_ns": raw + 10000,
                    "end_before_ns": raw + 11000,
                    "end_after_ns": raw + 13000,
                }
            )
            self.items.append(
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 7,
                    "name": f"{alignment.MARKER_PREFIX}{i}",
                    "ts": str(self.origin + relative - 3),
                    "dur": "15",
                }
            )
            self.items.append(
                {
                    "ph": "X",
                    "pid": 2,
                    "tid": 9,
                    "name": "aclnnAdds_Add",
                    "ts": str(self.origin + (relative + 3) * self.scale + self.offset),
                    "dur": str(4 * self.scale),
                }
            )
        self.host = {
            "ph": "X",
            "pid": 1,
            "tid": 4,
            "name": "moe_offload/host_callback|kind=prefetch",
            "ts": str(self.origin + 5_000_000),
            "dur": "500",
        }
        self.items.append(self.host)
        self.wait = {
            "ph": "X",
            "pid": 2,
            "tid": 25,
            "name": "host_sync_wait_2",
            "ts": str(self.origin + 5_000_500 * self.scale + self.offset),
            "dur": str(4 * self.scale),
        }
        self.items.append(self.wait)
        self.save()

    def save(self):
        self.trace.write_text(
            json.dumps({"displayTimeUnit": "us", "traceEvents": self.items, "other_metadata": {"kept": True}}),
            encoding="utf-8",
        )

    def test_alignment_corrects_all_device_streams_and_preserves_host_and_source(self):
        original = self.trace.read_bytes()
        result = alignment.align_capture(self.root, self.rows)
        self.assertEqual(result["status"], "calibration_validated")
        self.assertEqual(result["held_out_samples"], 8)
        self.assertEqual(self.trace.read_bytes(), original)
        exported = json.loads(Path(result["aligned_trace"]).read_text(encoding="utf-8"))
        self.assertEqual(exported["other_metadata"], {"kept": True})
        self.assertEqual(exported["traceEvents"][-2], self.host)
        wait = exported["traceEvents"][-1]
        self.assertLess(abs(Decimal(wait["ts"]) - (self.origin + 5_000_500)), Decimal("1"))
        self.assertLess(abs(Decimal(wait["dur"]) - 4), Decimal("0.0001"))

    def test_held_out_failure_never_creates_aligned_trace(self):
        marker = [e for e in self.items if e.get("name") == "aclnnAdds_Add"][3]
        marker["ts"] = str(Decimal(marker["ts"]) + 100)
        self.save()
        with self.assertRaisesRegex(ValueError, "held-out"):
            alignment.align_capture(self.root, self.rows)
        self.assertFalse(self.trace.with_name("trace_view_aligned.json").exists())

    def test_other_capture_raw_measurements_are_rejected(self):
        for row in self.rows:
            for key in tuple(row):
                if key.endswith("_ns"):
                    row[key] += 1_000_000_000
        with self.assertRaisesRegex(ValueError, "Host MSTX"):
            alignment.fit_alignment(self.root, self.rows)

    def test_ambiguous_calibration_stream_rejected(self):
        self.items.extend(dict(e, tid=19) for e in list(self.items) if e.get("name") == "aclnnAdds_Add")
        self.save()
        with self.assertRaisesRegex(ValueError, "uniquely"):
            alignment.fit_alignment(self.root, self.rows)

    def test_correction_does_not_use_wait_or_callback_positions_to_fit(self):
        first = alignment.fit_alignment(self.root, self.rows)
        self.wait["ts"] = str(self.origin + 1_000_000)
        self.host["dur"] = "999999"
        self.save()
        second = alignment.fit_alignment(self.root, self.rows)
        for key in ("device_scale", "device_offset_us"):
            self.assertEqual(first[key], second[key])

    def test_existing_output_and_original_are_never_overwritten(self):
        result = alignment.align_capture(self.root, self.rows)
        output = Path(result["aligned_trace"])
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "new file"):
            alignment.align_capture(self.root, self.rows)
        self.assertEqual(original, output.read_bytes())

    def test_flow_endpoint_and_counter_follow_their_device_domain(self):
        mapping = alignment.fit_alignment(self.root, self.rows)
        counts = alignment.collections.Counter()
        host_flow = {"ph": "s", "pid": 1, "ts": str(self.origin), "id": 10}
        device_flow = {"ph": "f", "pid": 2, "ts": str(self.origin + self.offset), "id": 10}
        self.assertEqual(alignment.transform_event(host_flow, mapping, counts), host_flow)
        aligned = alignment.transform_event(device_flow, mapping, counts)
        self.assertLess(abs(Decimal(aligned["ts"]) - self.origin), 1)
        self.assertEqual(aligned["id"], 10)

    def test_standalone_does_not_relax_model_capture_requirements(self):
        rows = [{key: value for key, value in row.items() if key != "phase"} for row in self.rows]
        with self.assertRaisesRegex(ValueError, "before/after"):
            alignment.fit_alignment(self.root, rows)
        result = alignment.fit_alignment(self.root, rows, require_model_phases=False)
        self.assertEqual(result["calibration_layout"], "standalone_calibration")
        self.assertEqual(result["held_out_before"]["outside_bracket_1us_tolerance"], 8)
        self.assertEqual(result["held_out_after"]["outside_bracket_1us_tolerance"], 0)
        self.assertEqual(result["held_out_after"]["max_distance_outside_us"], "0")
        with self.assertRaisesRegex(ValueError, "at least 16"):
            alignment.fit_alignment(self.root, rows[:8], require_model_phases=False)

    def saved_calibration(self):
        run = Path(self.temp.name) / "saved_run"
        rows = [{key: value for key, value in row.items() if key != "phase"} for row in self.rows]
        for mode in ("eager", "graph"):
            case = run / mode
            shutil.copytree(self.root, case / "profile" / self.root.name)
            (case / "host_brackets.json").write_text(json.dumps(rows), encoding="utf-8")
        return run, Path(self.temp.name) / "offline_result"

    def test_saved_calibration_command_preserves_inputs_without_torch(self):
        run, output = self.saved_calibration()
        originals = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
        real_import = builtins.__import__

        def check_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torch_npu"):
                raise AssertionError("offline alignment must not import device libraries")
            return real_import(name, *args, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()), mock.patch("builtins.__import__", side_effect=check_import):
            result = alignment.main(["--calibration-run", str(run), "--output-dir", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual({path: path.read_bytes() for path in run.rglob("*") if path.is_file()}, originals)
        for mode in ("eager", "graph"):
            self.assertTrue((output / mode / "trace_view_aligned.json").is_file())
            mapping = json.loads((output / mode / "clock_alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["held_out_after"]["outside_bracket_1us_tolerance"], 0)
        self.assertIn("PASS", (output / "summary.txt").read_text(encoding="utf-8"))

    def test_saved_calibration_failure_is_reported_without_overwriting_inputs(self):
        run, output = self.saved_calibration()
        samples = run / "graph/host_brackets.json"
        samples.write_text("[]", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(alignment.align_saved_calibration(run, output), 1)
        self.assertTrue((output / "eager/trace_view_aligned.json").is_file())
        self.assertFalse((output / "graph/trace_view_aligned.json").exists())
        self.assertEqual(samples.read_text(encoding="utf-8"), "[]")
        self.assertIn("ALIGNMENT_FAILED mode=graph", (output / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
