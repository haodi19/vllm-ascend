"""CPU regression coverage for trace container formats."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path


class TraceReaderTests(unittest.TestCase):
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
