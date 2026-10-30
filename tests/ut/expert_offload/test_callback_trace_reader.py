# SPDX-License-Identifier: Apache-2.0
"""Validate CANN trace containers before using their events for gap analysis."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TraceReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "callback_trace_test", Path(__file__).resolve().parents[3] / "script/analyze_callback_gaps.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def read(self, text, *, bom=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
            return list(self.module.events(path))

    def test_root_array_and_object_exports_with_bom(self):
        records = [{"name": "算子", "ph": "X", "args": {"text": "x" * 70000}}] * 16
        for payload in (records, {"traceEvents": records, "displayTimeUnit": "ms"}):
            for bom in (False, True):
                with self.subTest(container=type(payload).__name__, bom=bom):
                    self.assertEqual(self.read(json.dumps(payload, ensure_ascii=False), bom=bom), records)

    def test_empty_valid_event_arrays(self):
        for text in ("[]", '{"traceEvents":[]}', ' {"other":1,"traceEvents":[],"tail":{}} \n'):
            with self.subTest(text=text):
                self.assertEqual(self.read(text), [])

    def test_metadata_before_events_can_exceed_one_megabyte(self):
        record = {"name": "valid"}
        payload = {"metadata": {"nested": {"traceEvents": ["not the root events"]}, "text": "x" * 1100000}}
        payload["traceEvents"] = [record]
        self.assertEqual(self.read(json.dumps(payload)), [record])

    def test_chunk_boundaries_cover_unicode_escapes_and_scalar_metadata(self):
        text = '{"before":-1.25e+3,"traceEvents":[{"name":"中文\\"\\n\\u03bb"}],"after":false}'
        expected = json.loads(text)["traceEvents"]
        for size in range(1, 13):
            with self.subTest(size=size), patch.object(self.module, "READ_CHUNK_CHARS", size):
                self.assertEqual(self.read(text), expected)

    def test_malformed_and_truncated_containers_are_rejected(self):
        cases = (
            "",
            "  \n",
            "{}",
            '"traceEvents"',
            '{"metadata":{"traceEvents":[]}}',
            '{"traceEvents":{}}',
            '[{"name":"incomplete"}',
            '[{"name":"incomplete}',
            "[{},]",
            "[,{}]",
            "[{}{}]",
            '{"traceEvents":[]',
            '{"traceEvents":[],}',
            '{"traceEvents":[],"tail":',
            '{"traceEvents":[],"tail":1e}',
            '{"traceEvents":[],"traceEvents":[]}',
            '{"traceEvents":[],"tail":NaN}',
            '[{"ts":Infinity}]',
            '[{"x":1,}]',
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_root_end_requires_eof_even_after_valid_events(self):
        for text in ("[{}]garbage", "[{}][]", '{"traceEvents":[{}]} extra', '{"traceEvents":[]}{}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)

    def test_event_entries_must_be_objects(self):
        for value in (None, True, 1, "event", []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "事件必须"):
                self.read(json.dumps([value]))

    def test_size_limit_is_per_value_not_whole_trace(self):
        with patch.object(self.module, "MAX_VALUE_CHARS", 80), patch.object(self.module, "READ_CHUNK_CHARS", 16):
            self.assertEqual(self.read(json.dumps([{"n": i} for i in range(100)])), [{"n": i} for i in range(100)])
            with self.assertRaisesRegex(ValueError, "大小上限"):
                self.read(json.dumps([{"name": "x" * 100}]))
            with self.assertRaisesRegex(ValueError, "大小上限"):
                self.read(json.dumps({"metadata": "x" * 100, "traceEvents": []}))

    def test_reads_use_bounded_chunks(self):
        class BoundedReader(io.StringIO):
            def read(self, size=-1):
                if size < 0 or size > self.limit:
                    raise AssertionError("unbounded read")
                return super().read(size)

        records = [{"name": f"event-{i}"} for i in range(1000)]
        with patch.object(self.module, "READ_CHUNK_CHARS", 32):
            source = BoundedReader(json.dumps(records))
            source.limit = 32
            with patch.object(Path, "open", return_value=source):
                self.assertEqual(list(self.module.events(Path("fake.json"))), records)

    def test_mixed_times_and_overlay_preserve_original_event_payloads(self):
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
                    rows = self.module.analyze(source, output)
                self.assertEqual(rows[0]["gap_us"], 16)
                self.assertEqual(rows[0]["memcpy_overlap_us"], 8)
                self.assertEqual(rows[0]["callback_tail_us"], 3)
                self.assertEqual(rows[0]["between_callbacks_us"], 10)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["traceEvents"][:6], records)

    def test_invalid_times_do_not_create_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "trace.json", Path(directory) / "overlay.json"
            for key in ("ts", "dur"):
                for value in (None, True, "bad", "NaN", "Infinity", []):
                    with self.subTest(key=key, value=value):
                        event = dict(name="MEMCPY_ASYNC", ph="X", ts=1, dur=2)
                        event[key] = value
                        source.write_text(json.dumps([event]), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, f"Invalid trace {key}"):
                            self.module.analyze(source, output)
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
