# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "script"))
from diagnose_mte_wait_timeline import (
    analyze,
    candidate,
    containment,
    discover,
    neighbors,
    read_trace,
    spanning_callbacks,
    task_pairs,
)


def event(name, start, duration, pid="host", tid="worker", **kwargs):
    return dict(name=name, ts=start, dur=duration, ph="X", pid=pid, tid=tid, **kwargs)


def meta(name, label, pid, tid=0):
    return dict(name=name, ph="M", pid=pid, tid=tid, args=dict(name=label))


def host_events(call, start, kind="prefetch", layer=1, duration=20, sync_end=None):
    context = f"backend=cpp|layer={layer}|kind={kind}|rows=1|call={call}"
    sync_end = start + duration - 2 if sync_end is None else sync_end
    return [
        event("moe_offload/host_callback|" + context, start, duration),
        event("moe_offload/synchronize|" + context, start + 5, sync_end - start - 5),
    ]


def hardware_events(start=0, end=25, tid="25"):
    return [
        event("host_sync_publish_1", start, 1, "hardware", tid),
        event("host_sync_wait_2", start + 2, end - start - 2, "hardware", tid),
    ]


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace_view.json"

    def tearDown(self):
        self.temp.cleanup()

    def read(self, extra):
        data = [
            meta("process_name", "Ascend Hardware", "hardware"),
            meta("process_name", "Python", "host"),
            meta("thread_name", "Stream 25", "hardware", "25"),
            *extra,
        ]
        self.path.write_text(json.dumps(dict(traceEvents=data)), encoding="utf-8")
        return read_trace(self.path)

    def test_cpu_names_excluded_even_if_tid_matches(self):
        data = self.read([*host_events(1, 5), *hardware_events(), event("host_sync_wait_2", 0, 100, "host", "25")])
        self.assertEqual(data["excluded"], 1)
        self.assertEqual(list(data["tasks"]), [("hardware", "25")])
        self.assertEqual(data["groups"]["prefetch"][0]["sync"]["end"], 23)

    def test_metadata_can_follow_events(self):
        raw = [*hardware_events(), meta("process_name", "Ascend Hardware", "hardware")]
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(len(read_trace(self.path)["tasks"]), 1)

    def test_wrong_context_sync_not_matched_by_time(self):
        records = host_events(1, 5)
        records[1]["name"] = records[1]["name"].replace("call=1", "call=2")
        data = self.read(records + hardware_events())
        self.assertIn("prefetch", data["invalid_kinds"])
        self.assertEqual(data["issues"]["orphan_sync"], 1)
        self.assertFalse(data["groups"])

    def test_wrong_thread_sync_not_matched(self):
        records = host_events(1, 5)
        records[1]["tid"] = "other_worker"
        data = self.read(records)
        self.assertIn("prefetch", data["invalid_kinds"])

    def test_duplicate_sync_blocks_pairing(self):
        records = host_events(1, 5)
        data = self.read([*records, records[1]])
        self.assertIn("prefetch", data["invalid_kinds"])

    def test_same_tid_two_hardware_processes_remain_separate(self):
        data = self.read(
            [
                *hardware_events(),
                meta("process_name", "Ascend Hardware", "second"),
                event("host_sync_wait_2", 2, 20, "second", "25"),
            ]
        )
        self.assertEqual(len(data["tasks"]), 2)

    def test_negative_delta_and_feasible_constant_shift(self):
        data = self.read([*host_events(1, 100), *hardware_events(0, 30)])
        pairs, reason = task_pairs(data["tasks"]["hardware", "25"])
        result = candidate(pairs, data["groups"]["prefetch"])
        self.assertEqual(reason, "ok")
        self.assertEqual(result["wait_end_before_sync_end"], 1)
        self.assertEqual(result["constant_device_shift_necessary_bounds_us"], [90, 100])
        self.assertTrue(result["constant_shift_feasible"])
        self.assertEqual(result["examples"][0]["wait_end_minus_sync_end_us"], -88)

    def test_inconsistent_offsets_not_accepted(self):
        data = self.read(
            [*host_events(1, 100), *host_events(2, 150), *hardware_events(0, 30), *hardware_events(100, 130)]
        )
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        result = candidate(pairs, data["groups"]["prefetch"])
        self.assertEqual(result["constant_device_shift_necessary_bounds_us"], [90, 50])
        self.assertFalse(result["constant_shift_feasible"])

    def test_count_mismatch_does_not_zip_shorter_prefix(self):
        data = self.read([*host_events(1, 5), *host_events(2, 50), *hardware_events()])
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        self.assertEqual(candidate(pairs, data["groups"]["prefetch"])["status"], "count_mismatch")

    def test_wait_publish_bad_order_blocks_pairs(self):
        data = self.read(
            [event("host_sync_wait_2", 0, 1, "hardware", "25"), event("host_sync_publish_1", 2, 1, "hardware", "25")]
        )
        pairs, reason = task_pairs(data["tasks"]["hardware", "25"])
        self.assertEqual(pairs, [])
        self.assertEqual(reason, "not_alternating_publish_wait")

    def test_overlap_observation_does_not_assume_identity(self):
        data = self.read([*host_events(1, 0, duration=50), *hardware_events(10, 30)])
        wait = data["tasks"]["hardware", "25"][1]
        self.assertEqual(
            containment([wait], data["groups"]["prefetch"]),
            dict(wait_fully_inside_sync=1, wait_start_inside_sync=1, wait_end_inside_sync=1),
        )

    def test_wait_spans_multiple_host_callbacks(self):
        data = self.read([*host_events(1, 10), *host_events(2, 40, kind="reactive"), *hardware_events(0, 100)])
        waits = [data["tasks"]["hardware", "25"][1]]
        self.assertEqual(spanning_callbacks(waits, data["all_hosts"])["maximum_callbacks"], 2)

    def test_invalid_and_truncated_json_fail(self):
        with self.assertRaises(ValueError):
            self.read([event("host_sync_wait_2", True, 10, "hardware", "25")])
        self.path.write_text('{"traceEvents": [', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_trace(self.path)

    def test_wait_starts_prefetch_sync_and_ends_reactive_sync(self):
        data = self.read(
            [
                *host_events(1, 0, duration=50),
                *host_events(2, 60, kind="reactive", duration=50),
                *hardware_events(10, 90),
            ]
        )
        waits = [data["tasks"]["hardware", "25"][1]]
        self.assertEqual(
            containment(waits, data["groups"]["prefetch"]),
            dict(wait_fully_inside_sync=0, wait_start_inside_sync=1, wait_end_inside_sync=0),
        )
        self.assertEqual(
            containment(waits, data["groups"]["reactive"]),
            dict(wait_fully_inside_sync=0, wait_start_inside_sync=0, wait_end_inside_sync=1),
        )

    def test_unknown_orphan_sync_blocks_both_kinds(self):
        orphan = host_events(7, 60)[1]
        orphan["name"] = orphan["name"].replace("kind=prefetch", "kind=unknown")
        data = self.read([*host_events(1, 0), orphan])
        self.assertEqual(data["invalid_kinds"], {"prefetch", "reactive"})

    def test_discovery_deduplicates_without_writing(self):
        self.read([])
        self.assertEqual(discover([self.temp.name, str(self.path)]), [self.path.resolve()])

    def test_neighbors_use_timestamps_not_json_order(self):
        data = self.read(
            [
                event("after_later", 35, 2, "hardware", "25"),
                event("host_sync_wait_2", 10, 10, "hardware", "25"),
                event("before_near", 8, 2, "hardware", "25"),
                event("overlap", 9, 3, "hardware", "25"),
                event("before_earlier", 3, 2, "hardware", "25"),
                event("after_near", 21, 1, "hardware", "25"),
                event("host_sync_publish_1", 0, 1, "hardware", "25"),
            ]
        )
        sample = neighbors(self.path, data["tasks"])["hardware", "25"][0]
        self.assertEqual([r["name"] for r in sample["before"]], ["before_earlier", "before_near"])
        self.assertEqual([r["name"] for r in sample["after"]], ["after_near", "after_later"])
        self.assertEqual(sample["overlapping_task_count"], 1)

    def test_separate_prefetch_reactive_threads_block_single_worker_pairing(self):
        reactive = host_events(2, 50, kind="reactive")
        for item in reactive:
            item["tid"] = "second_worker"
        data = self.read([*host_events(1, 5), *reactive])
        self.assertEqual(data["invalid_kinds"], {"prefetch", "reactive"})

    def test_shift_bounds_remain_separate_for_hardware_pids(self):
        other = hardware_events(100, 130)
        for item in other:
            item["pid"] = "second"
        self.read([*host_events(1, 200), *hardware_events(), meta("process_name", "Ascend Hardware", "second"), *other])
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            analyze(self.path)
        results = [
            json.loads(line.split(" ", 1)[1])
            for line in capture.getvalue().splitlines()
            if line.startswith("CROSS_STREAM_SHIFT_HYPOTHESIS ")
        ]
        self.assertEqual({r["hardware_pid"] for r in results}, {"hardware", "second"})


if __name__ == "__main__":
    unittest.main()
