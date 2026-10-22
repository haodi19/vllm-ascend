"""CPU regression coverage for trace container formats."""

import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path


class TraceReaderTests(unittest.TestCase):
    def test_mixed_time_types_and_original_preserved(self):
        analyze = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/analyze_callback_gaps.py"))[
            "analyze"
        ]
        records = [
            dict(name="moe_offload/host_callback|call=1|kind=prefetch", ph="X", pid=1, ts="0", dur=15.0),
            dict(name="moe_offload/update_weights|call=1|kind=prefetch", ph="X", pid=1, ts=2, dur="10"),
            dict(name="moe_offload/host_callback|call=2|kind=reactive", ph="X", pid=1, ts="25", dur="15"),
            dict(name="moe_offload/update_weights|call=2|kind=reactive", ph="X", pid=1, ts="28", dur=10),
            dict(name="MEMCPY_ASYNC", ph="X", ts="14", dur=6.0),
            dict(name="MEMCPY_ASYNC", ph="X", ts=18, dur="4"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            for index, payload in enumerate((records, {"traceEvents": records})):
                source.write_text(json.dumps(payload), encoding="utf-8")
                original = source.read_bytes()
                output = Path(directory) / f"overlay{index}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = analyze(source, output)
                self.assertEqual(rows[0]["gap_us"], 16)
                self.assertEqual(rows[0]["memcpy_overlap_us"], 8)
                self.assertEqual(rows[0]["callback_tail_us"], 3)
                self.assertEqual(rows[0]["between_callbacks_us"], 10)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(json.loads(output.read_text())["traceEvents"][:6], records)

    def test_invalid_time_rejected(self):
        analyze = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/analyze_callback_gaps.py"))[
            "analyze"
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "trace.json", Path(directory) / "overlay.json"
            for key in ("ts", "dur"):
                for value in (None, True, "bad", "NaN", "Infinity", []):
                    with self.subTest(key=key, value=value):
                        event = dict(name="MEMCPY_ASYNC", ph="X", ts=1, dur=2)
                        event[key] = value
                        source.write_text(json.dumps([event]), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, f"Invalid trace {key}"):
                            analyze(source, output)
                        self.assertFalse(output.exists())

    def test_optional_analysis_failure_does_not_abort(self):
        safe = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/run_verified_cpp_tests.py"))[
            "analyze_gaps_safely"
        ]

        def fail(trace, output):
            raise TypeError("mixed time types")

        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            safe(fail, Path("trace.json"), Path("overlay.json"))
        self.assertIn("GAP ANALYSIS NOT VERIFIED: TypeError", console.getvalue())

    def test_array_and_object_exports(self):
        read = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/analyze_callback_gaps.py"))["events"]
        records = [{"name": "moe_offload/example", "ph": "X", "args": {"text": "x" * 70000}}] * 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            for payload in (records, {"traceEvents": records}, []):
                path.write_text(json.dumps(payload), encoding="utf-8-sig")
                self.assertEqual(list(read(path)), payload if isinstance(payload, list) else records)
            path.write_text('[{"name":"incomplete"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read(path))


if __name__ == "__main__":
    unittest.main()
