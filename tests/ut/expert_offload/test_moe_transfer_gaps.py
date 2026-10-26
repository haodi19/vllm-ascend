"""搬运间隔诊断的合成时间线测试，不依赖 NPU。"""

import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[3] / "script/diagnose_moe_transfer_gaps.py"))


def callback(start, end, kind, layer):
    return dict(pid="cpu", start=start, end=end, fields=dict(kind=kind, layer=str(layer)))


def device(name, start, end):
    return dict(name=name, start=start, end=end, dur=end - start, tid="25")


class TransferGapTests(unittest.TestCase):
    def test_publish_partition_without_wait_endpoint(self):
        pairs, _ = MODULE["pair_callbacks"]([callback(0, 10, "prefetch", 2), callback(50, 60, "reactive", 2)])
        MODULE["split_pairs"](pairs, [device("host_sync_wait_2", 0, 9), device("host_sync_publish_1", 30, 35)])
        self.assertFalse(pairs[0]["resolved"])
        self.assertTrue(pairs[0]["publish_resolved"])
        self.assertEqual(
            [pairs[0][key] for key in ("before_publish_us", "publish_duration_us", "after_publish_us")], [20, 5, 15]
        )

    def test_multiple_publish_candidates_not_forced(self):
        pairs, _ = MODULE["pair_callbacks"]([callback(0, 10, "prefetch", 2), callback(50, 60, "reactive", 2)])
        MODULE["split_pairs"](pairs, [device("host_sync_publish", 20, 25), device("host_sync_publish", 30, 35)])
        self.assertFalse(pairs[0]["publish_resolved"])

    def test_neighbor_outside_gap_is_shown_without_association(self):
        pairs, _ = MODULE["pair_callbacks"]([callback(0, 10, "prefetch", 2), callback(50, 60, "reactive", 2)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            MODULE["nearby_handoffs"](pairs[0], [device("host_sync_wait", 0, 9)])
        self.assertIn("end_from_left_end=-1.000", output.getvalue())

    def test_layer_pairing(self):
        rows = [
            callback(0, 10, "prefetch", 2),
            callback(20, 30, "reactive", 2),
            callback(40, 50, "prefetch", 3),
            callback(60, 70, "reactive", 4),
        ]
        pairs, skipped = MODULE["pair_callbacks"](rows)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(skipped, 1)

    def test_overlapping_callbacks_rejected(self):
        pairs, _ = MODULE["pair_callbacks"]([callback(0, 20, "prefetch", 2), callback(10, 30, "reactive", 2)])
        self.assertFalse(pairs)

    def test_unique_candidates_partition_interval(self):
        pairs, _ = MODULE["pair_callbacks"]([callback(0, 10, "prefetch", 2), callback(50, 60, "reactive", 2)])
        MODULE["split_pairs"](pairs, [device("host_sync_wait", 0, 20), device("host_sync_publish", 30, 35)])
        result = pairs[0]
        self.assertTrue(result["resolved"])
        self.assertEqual(
            [
                result[k]
                for k in (
                    "completion_to_wait_exit_us",
                    "after_wait_to_publish_us",
                    "publish_us",
                    "publish_exit_to_callback_us",
                )
            ],
            [10, 10, 5, 15],
        )

    def test_ambiguous_and_overlapping_publish_rejected(self):
        for events in (
            [device("host_sync_wait", 0, 20), device("host_sync_wait", 0, 21), device("host_sync_publish", 30, 35)],
            [device("host_sync_wait", 0, 20), device("host_sync_publish", 30, 55)],
            [],
        ):
            pairs, _ = MODULE["pair_callbacks"]([callback(0, 10, "prefetch", 2), callback(50, 60, "reactive", 2)])
            MODULE["split_pairs"](pairs, events)
            self.assertFalse(pairs[0]["resolved"])

    def test_metadata_after_events_and_host_kernel_excluded(self):
        events = [
            dict(ph="X", name="host_sync_wait", pid=1, tid=2, ts="0", dur="10"),
            dict(ph="X", name="host_sync_publish", pid=2, tid=25, ts="20", dur="5"),
            dict(ph="M", name="process_name", pid=2, args=dict(name="Ascend Hardware")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            before = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, hardware_events, _, _ = MODULE["load_trace"](path)
            self.assertEqual(len(hardware_events), 1)
            self.assertEqual(hardware_events[0]["start"], 20)
            self.assertEqual(path.read_bytes(), before)

    def test_truncated_trace_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text('[{"ph":"X"}', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                MODULE["load_trace"](path)


if __name__ == "__main__":
    unittest.main()
