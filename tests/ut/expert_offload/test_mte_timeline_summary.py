"""Read-only timeline accounting and device classification regression tests."""

import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path


class MteTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/summarize_mte_timeline.py"))

    def summarize(self, events, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            original = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.module["summarize"](path, **kwargs)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            return result

    @staticmethod
    def event(name, pid, ts, dur):
        return dict(ph="X", name=name, pid=pid, ts=ts, dur=dur, cat="kernel")

    def test_device_metadata_after_events_and_numeric_strings(self):
        result = self.summarize(
            [
                self.event("host_sync_wait", 1, "0", "1000"),
                self.event("moe_offload/update_weights|kind=prefetch|call=1", 1, "5", "10"),
                self.event("host_sync_publish", 2, "10", "5"),
                self.event("host_sync_wait", "2", 12, "20"),
                self.event("MEMCPY_ASYNC", 2, 30, 10),
                self.event("MatMul", 2, 50, 10),
                dict(ph="M", name="process_name", pid=1, args={"name": "CPU"}),
                dict(ph="M", name="process_name", pid=2, args={"name": "Ascend Hardware"}),
            ]
        )
        self.assertEqual(result["device_events"], 4)
        self.assertEqual(result["busy_us"], 40)
        self.assertEqual(result["span_us"], 50)
        self.assertEqual(result["handoff"]["host_sync_wait"], [1, 20, 20])
        self.assertEqual(result["updates"]["prefetch"], [1, 10, 10])

    def test_unknown_hardware_does_not_guess_from_kernel_name(self):
        result = self.summarize([self.event("host_sync_wait", 1, 0, 100)])
        self.assertEqual(result["device_events"], 0)
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            self.module["report"](result)
        self.assertIn("DEVICE UNKNOWN", console.getvalue())

    def test_interval_limit_is_strict(self):
        records = [dict(ph="M", name="process_name", pid=2, args={"name": "Ascend Hardware"})]
        records += [self.event("kernel", 2, 1, 2)] * 2
        with self.assertRaisesRegex(ValueError, "设备区间超过上限"):
            self.summarize(records, max_intervals=1)

    def test_invalid_time_rejected(self):
        for key in ("ts", "dur"):
            for value in (True, "NaN", "Infinity", None, []):
                with self.subTest(key=key, value=value):
                    event = self.event("moe_offload/update_weights|kind=reactive", 1, 1, 2)
                    event[key] = value
                    with self.assertRaisesRegex(ValueError, f"Invalid trace {key}"):
                        self.summarize([event])

    def test_partial_trace_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text('[{"ph":"X"}', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                self.module["summarize"](path)

    def test_non_x_events_and_negative_durations_reported(self):
        result = self.summarize(
            [
                dict(ph="M", name="process_name", pid=2, args={"name": "Ascend Hardware 0"}),
                dict(ph="B", name="kernel", pid=2, ts=1),
                self.event("kernel", 2, 1, -1),
            ]
        )
        self.assertEqual(result["unsupported"], 1)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["device_events"], 0)


if __name__ == "__main__":
    unittest.main()
