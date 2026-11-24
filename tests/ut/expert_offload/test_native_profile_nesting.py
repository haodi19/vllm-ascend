# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
import inspect_native_profile_nesting as audit


class NativeNestingTests(unittest.TestCase):
    def pair(self):
        return [
            dict(name=f"moe_offload/{stage}|kind=prefetch|call=1", pid=1, tid=2, ph="X", ts=ts, dur=dur)
            for stage, ts, dur in (("host_callback", 10, 20), ("update_weights", 12, 15))
        ]

    def check(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            before = path.read_bytes()
            result = audit.audit_host_nesting(path)
            self.assertEqual(before, path.read_bytes())
            return result

    def test_valid_pair_and_unpaired_ranges(self):
        result = self.check(self.pair())
        self.assertEqual(result["paired"], 1)
        self.assertEqual(result["invalid_exact"], 0)
        self.assertEqual(result["examples"], [])
        result = self.check(self.pair()[:1])
        self.assertEqual(result["callback_only"], 1)

    def test_actual_overrun_and_wrong_thread_remain_invalid(self):
        for mutation in (dict(dur=19), dict(tid=3)):
            with self.subTest(mutation=mutation):
                pair = self.pair()
                pair[1].update(mutation)
                result = self.check(pair)
                self.assertEqual(result["invalid_exact"], 1)
                self.assertEqual(result["float_only_invalid"], 0)
                self.assertFalse(result["examples"][0]["exact_ok"])

    def test_duplicates_and_missing_ids_reported_separately(self):
        pair = self.pair()
        missing = dict(pair[0], pid=None)
        result = self.check([*pair, pair[0], missing])
        self.assertEqual(result["duplicate_ids"], 1)
        self.assertEqual(result["missing_ids"], 1)
        self.assertEqual(result["callbacks"], 1)

    def test_large_epoch_float_false_overrun_with_exact_numeric_json(self):
        # Both exact endpoints equal .51; float addition makes the inner end
        # .75 and the outer end .50. Diagnose without adding a tolerance.
        pair = self.pair()
        pair[0].update(ts="1795507043360466", dur="0.51")
        pair[1].update(ts="1795507043360466.13", dur="0.38")
        raw = json.dumps(pair)
        for event in pair:
            for key in ("ts", "dur"):
                raw = raw.replace(json.dumps(event[key]), event[key])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(raw, encoding="utf-8")
            result = audit.audit_host_nesting(path)
        self.assertEqual(result["invalid_float"], 1)
        self.assertEqual(result["invalid_exact"], 0)
        self.assertEqual(result["float_only_invalid"], 1)
        self.assertEqual(result["examples"][0]["float_end_overrun_us"], 0.25)

    def test_cli_prints_counters_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "profile/native/capture/ASCEND_PROFILER_OUTPUT/trace_view.json"
            trace.parent.mkdir(parents=True)
            trace.write_text(json.dumps(self.pair()), encoding="utf-8")
            (root / "native_profile_counters.json").write_text('{"count": 1}', encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(audit.main([str(root)]), 0)
            self.assertIn('"expected_callbacks": 1', output.getvalue())
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_invalid_or_truncated_json_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(self.pair())[:-1], encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.audit_host_nesting(path)


if __name__ == "__main__":
    unittest.main()
