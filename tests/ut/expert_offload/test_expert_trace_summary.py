# SPDX-License-Identifier: Apache-2.0
"""Read-only compact reports must preserve counts and expose partial input."""

import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
WRITER = importlib.import_module("moe_expert_trace_report")
READER = importlib.import_module("summarize_moe_expert_trace")
write_trace = WRITER.write_trace
main, read_summary, report = READER.main, READER.read_summary, READER.report
sys.path.pop(0)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.tmp) / "expert_trace.txt"
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def fixture(self, index=1):
        return dict(
            complete=True,
            errors=0,
            dropped=0,
            metadata={"moe_layers": 1},
            request={"index": index},
            records=[
                dict(
                    event_id=1,
                    kind="prefill_pool",
                    layer_visit=1,
                    layer=0,
                    status="completed",
                    loads=[],
                    copies=[],
                    full_pool_experts=256,
                    logical_h2d_bytes=4096,
                ),
                dict(
                    event_id=2,
                    kind="reactive",
                    layer_visit=1,
                    layer=0,
                    status="completed",
                    loads=[[2, 42]],
                    copies=[],
                    hit_ids=[3, 4],
                    missing_ids=[42],
                    loaded_ids=[42],
                    logical_h2d_bytes=128,
                    route_rows=[[3, 4, 42]],
                ),
            ],
        )

    def test_compact_report_preserves_totals_and_ids(self):
        write_trace(self.fixture(), self.path)
        before = self.path.read_bytes()
        data = read_summary(self.path)
        self.assertEqual(report(data, 1), 0)
        self.assertEqual(data["counts"]["reactive", 1, 0]["loaded_experts"], 1)
        self.assertEqual(data["counts"]["prefill_pool", 1, 0]["loaded_experts"], 256)
        text = self.output.getvalue()
        self.assertIn("00 2/1/1", text)
        self.assertIn('"loaded_ids":[42]', text)
        self.assertNotIn("resident_before", text)
        self.assertEqual(before, self.path.read_bytes())

    def test_truncated_or_corrupt_event_rejected(self):
        write_trace(self.fixture(), self.path)
        source = self.path.read_text(encoding="utf-8")
        self.path.write_text(source.split("TOTAL ")[0], encoding="utf-8")
        self.assertEqual(report(read_summary(self.path), 1), 1)
        self.path.write_text(source.replace('"event_id": 2', '"event_id": INVALID'), encoding="utf-8")
        self.assertEqual(report(read_summary(self.path), 1), 1)

    def test_combined_requests_not_double_counted(self):
        write_trace(self.fixture(1), self.path)
        first = self.path.read_text(encoding="utf-8")
        second = self.fixture(2)
        second["records"][1]["logical_h2d_bytes"] = 256
        write_trace(second, self.path)
        self.path.write_text(first + self.path.read_text(encoding="utf-8"), encoding="utf-8")
        data = read_summary(self.path, 2)
        self.assertEqual(data["event_count"], 2)
        self.assertEqual(data["headers"]["REQUEST"]["index"], 2)
        self.assertEqual(data["counts"]["reactive", 1, 0]["h2d_bytes"], 256)
        self.assertEqual(report(data, 2), 0)
        self.assertEqual(report(read_summary(self.path, 3), 3), 1)

    def test_directory_input_and_missing_file(self):
        self.assertEqual(main([str(self.path)]), 1)
        write_trace(self.fixture(), self.path)
        self.assertEqual(main([self.tmp]), 0)


if __name__ == "__main__":
    unittest.main()
