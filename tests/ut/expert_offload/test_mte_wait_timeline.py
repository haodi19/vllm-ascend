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
    compare_aligned,
    containment,
    cross_kind_overlap,
    discover,
    layer_cycles,
    main,
    neighbors,
    nonnegative_us,
    order_audit,
    read_trace,
    shift_diagnostics,
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
        self.assertTrue(all(row["coverage"] == "single_stream_only" for row in results))

    def test_order_audit_reports_all_edge_overlaps_and_preserves_strict_result(self):
        data = self.read(
            [
                event("host_sync_publish_1", 0, 2.25, "hardware", "27"),
                event("host_sync_wait_2", 2, 20, "hardware", "27"),
                event("host_sync_publish_1", 21.75, 2, "hardware", "27"),
                event("host_sync_wait_2", 23.5, 10, "hardware", "27"),
            ]
        )
        tasks = data["tasks"]["hardware", "27"]
        before = json.dumps(tasks)
        report = order_audit(tasks, 1.0)
        self.assertEqual(report["strict_result"], "publish_wait_overlap_on_same_stream")
        self.assertEqual(report["tolerance_result"], "ok")
        self.assertEqual(report["publish_to_wait"]["overlap_us"]["count"], 2)
        self.assertEqual(report["wait_to_next_publish"]["overlap_us"]["count"], 1)
        self.assertEqual(report["publish_to_wait"]["overlap_us"]["max_us"], 0.25)
        self.assertEqual(len(task_pairs(tasks, 1.0)[0]), 2)
        self.assertEqual(task_pairs(tasks)[0], [])
        self.assertEqual(json.dumps(tasks), before)

    def test_tolerance_does_not_accept_large_overlap(self):
        data = self.read(
            [event("host_sync_publish_1", 0, 4, "hardware", "27"), event("host_sync_wait_2", 2, 20, "hardware", "27")]
        )
        report = order_audit(data["tasks"]["hardware", "27"], 1)
        self.assertNotEqual(report["tolerance_result"], "ok")
        self.assertEqual(report["publish_to_wait"]["above_tolerance"], 1)

    def test_real_epoch_quarter_microsecond_overlap_preserves_strict_result(self):
        epoch = 1_794_000_000_000_000.0
        data = self.read(
            [
                event("host_sync_publish_1", epoch, 2.25, "hardware", "27"),
                event("host_sync_wait_2", epoch + 2, 20, "hardware", "27"),
            ]
        )
        tasks = data["tasks"]["hardware", "27"]
        before = json.dumps(tasks)
        report = order_audit(tasks, 1.0)
        self.assertEqual(report["publish_to_wait"]["overlap_us"]["max_us"], 0.25)
        self.assertEqual(report["strict_result"], "publish_wait_overlap_on_same_stream")
        self.assertEqual(report["tolerance_result"], "ok")
        self.assertEqual(task_pairs(tasks)[0], [])
        self.assertEqual(len(task_pairs(tasks, 1.0)[0]), 1)
        self.assertEqual(json.dumps(tasks), before)

    def test_tolerance_does_not_accept_reversed_task_types(self):
        data = self.read(
            [event("host_sync_wait_2", 0, 2, "hardware", "27"), event("host_sync_publish_1", 2, 2, "hardware", "27")]
        )
        self.assertEqual(
            order_audit(data["tasks"]["hardware", "27"], 1)["tolerance_result"], "not_alternating_publish_wait"
        )

    def test_analysis_labels_tolerated_candidate_separately(self):
        self.read(
            [
                *host_events(1, 10),
                event("host_sync_publish_1", 0, 2.25, "hardware", "27"),
                event("host_sync_wait_2", 2, 40, "hardware", "27"),
            ]
        )
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            analyze(self.path)
        output = capture.getvalue()
        self.assertIn("ORDER_TOLERATED_ORDINAL_CANDIDATE", output)
        self.assertIn('"strict_result":"publish_wait_overlap_on_same_stream"', output)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            analyze(self.path, 0)
        self.assertNotIn("ORDER_TOLERATED_ORDINAL_CANDIDATE", capture.getvalue())

    def test_shift_windows_expose_time_variation_without_filtering(self):
        rows = []
        for i, shift in enumerate((50, 50, 120, 120)):
            rows.extend(host_events(i, 100 * i + shift, layer=i % 2))
            rows.extend(hardware_events(100 * i, 100 * i + 30))
        data = self.read(rows)
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        hosts = data["groups"]["prefetch"]
        full = candidate(pairs, hosts)
        diagnostics = shift_diagnostics(pairs, hosts, 2)
        self.assertFalse(full["constant_shift_feasible"])
        self.assertEqual(diagnostics["windows"]["total_windows"], 2)
        self.assertEqual(diagnostics["windows"]["infeasible_windows"], 0)
        self.assertEqual(diagnostics["binding_lower_sample"]["ordinal"], 2)
        self.assertEqual(diagnostics["binding_upper_sample"]["ordinal"], 0)
        self.assertEqual(diagnostics["binding_lower_sample"]["context"]["layer"], "0")
        self.assertEqual(diagnostics["layers"]["infeasible_layers"], 2)

    def test_diagnostic_rows_bounded_but_totals_complete(self):
        rows = []
        for i in range(160):
            rows.extend(host_events(i, 100 * i + 50, layer=i % 40))
            rows.extend(hardware_events(100 * i, 100 * i + 30))
        data = self.read(rows)
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        diagnostics = shift_diagnostics(pairs, data["groups"]["prefetch"], 2)
        self.assertEqual(diagnostics["windows"]["total_windows"], 80)
        self.assertEqual(diagnostics["layers"]["total_layers"], 40)
        self.assertLessEqual(len(diagnostics["windows"]["rows"]), 5)
        self.assertLessEqual(len(diagnostics["layers"]["rows"]), 5)

    def test_same_wait_cross_kind_overlap_count_not_independent_counts(self):
        data = self.read(
            [
                *host_events(1, 0, duration=50),
                *host_events(2, 60, kind="reactive", duration=50),
                *hardware_events(10, 90),
            ]
        )
        waits = [data["tasks"]["hardware", "25"][1]]
        report = cross_kind_overlap(waits, data["groups"])
        self.assertEqual(report["wait_start_prefetch_and_end_reactive"], 1)
        self.assertEqual(report["same_layer_context"], 1)
        self.assertEqual(report["examples"][0]["start_prefetch_context"]["call"], "1")
        self.assertEqual(report["examples"][0]["end_reactive_context"]["call"], "2")
        split = [dict(start=10, end=40), dict(start=70, end=90)]
        self.assertEqual(cross_kind_overlap(split, data["groups"])["wait_start_prefetch_and_end_reactive"], 0)

    def test_tolerance_converter_rejects_negative_nonfinite(self):
        for value in ("nan", "inf", "-0.1"):
            with self.assertRaises(ValueError):
                order_audit([], float(value))
        self.assertEqual(nonnegative_us("0"), 0)

    def test_complete_layer_cycles_each_feasible_without_global_intersection(self):
        rows = []
        for i, (layer, shift) in enumerate(((3, 50), (4, 50), (3, 120), (4, 120))):
            rows.extend(host_events(i, 100 * i + shift, layer=layer))
            rows.extend(hardware_events(100 * i, 100 * i + 30))
        data = self.read(rows)
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        result = layer_cycles(pairs, data["groups"]["prefetch"])
        self.assertEqual(result["status"], "validated_complete_repetitions")
        self.assertEqual(result["observed_layer_order"], [3, 4])
        self.assertEqual(result["total_cycles"], 2)
        self.assertEqual(result["infeasible_cycles"], 0)
        self.assertEqual(result["global_necessary_bounds_us"], [110, 50])
        self.assertEqual([row[0] for row in result["rows"]], [0, 1])

    def test_missing_layer_in_one_cycle_not_silently_trimmed(self):
        rows = []
        for i, layer in enumerate((0, 1, 2, 0, 2)):
            rows.extend(host_events(i, 100 * i + 50, layer=layer))
            rows.extend(hardware_events(100 * i, 100 * i + 30))
        data = self.read(rows)
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        result = layer_cycles(pairs, data["groups"]["prefetch"])
        self.assertEqual(result["status"], "incomplete_or_nonrepeating_layer_sequence")
        self.assertEqual(result["samples"], 5)
        self.assertNotIn("rows", result)
        self.assertEqual(result["first_mismatches"][0], dict(ordinal=4, actual=2, expected=1))

    def test_partial_prefix_and_suffix_reported(self):
        rows = []
        for i, layer in enumerate((1, 2, 0, 1)):
            rows.extend(host_events(i, 100 * i + 50, layer=layer))
            rows.extend(hardware_events(100 * i, 100 * i + 30))
        data = self.read(rows)
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        result = layer_cycles(pairs, data["groups"]["prefetch"])
        self.assertTrue(result["partial_first_cycle"])
        self.assertTrue(result["partial_last_cycle"])
        self.assertNotIn("total_cycles", result)

    def test_non_numeric_layer_cannot_claim_cycles(self):
        data = self.read([*host_events(1, 50, layer="unknown"), *hardware_events(0, 30)])
        pairs, _ = task_pairs(data["tasks"]["hardware", "25"])
        self.assertEqual(layer_cycles(pairs, data["groups"]["prefetch"])["status"], "non_numeric_or_negative_layers")

    def aligned_fixture(self, rows=None, shift=90):
        self.read(rows if rows is not None else [*host_events(1, 100), *hardware_events(0, 30)])
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for item in payload["traceEvents"]:
            if item["pid"] == "hardware" and item["ph"] == "X":
                item["ts"] += shift
        aligned_path = self.path.with_name("trace_view_aligned.json")
        aligned_path.write_text(json.dumps(payload), encoding="utf-8")
        mapping = dict(
            status="calibration_validated",
            calibration_layout="model_request_brackets",
            source=str(self.path.resolve()),
            aligned_trace=str(aligned_path.resolve()),
            device_scale="1",
            device_offset_us=str(-shift),
            trace_origin_us="0",
            device_pids=["hardware"],
        )
        self.path.with_name("clock_alignment.json").write_text(json.dumps(mapping), encoding="utf-8")
        return aligned_path

    def test_aligned_audit_reports_before_after_without_refitting(self):
        self.aligned_fixture()
        result = compare_aligned(self.path)
        self.assertEqual(result["remaining_negative_relations_over_1us"], 0)
        stream = result["streams"][0]
        self.assertEqual(stream["kind"], "prefetch")
        self.assertEqual(stream["raw"]["wait_end_minus_sync_end_us"]["negative_count"], 1)
        self.assertEqual(stream["aligned"]["wait_end_minus_sync_end_us"]["negative_count"], 0)
        self.assertEqual(stream["aligned"]["wait_end_minus_sync_end_us"]["min_us"], 2)

    def test_aligned_audit_retains_residual_contradictions(self):
        self.aligned_fixture(shift=50)
        result = compare_aligned(self.path)
        self.assertEqual(result["status"], "residual_negative_relations")
        self.assertEqual(result["streams"][0]["aligned"]["wait_end_minus_sync_end_us"]["min_us"], -38)

    def test_aligned_audit_does_not_guess_kind_when_counts_match_both(self):
        self.aligned_fixture([*host_events(1, 100), *host_events(2, 200, kind="reactive"), *hardware_events(0, 30)])
        result = compare_aligned(self.path)
        self.assertEqual(result["status"], "incomplete_pairing")
        self.assertEqual(result["streams"][0]["status"], "no_unique_complete_ordinal_candidate")

    def test_aligned_audit_rejects_changed_host_or_incorrect_device_transform(self):
        for domain, pattern in (("host", "Host callback"), ("hardware", "coordinates disagree")):
            aligned_path = self.aligned_fixture()
            payload = json.loads(aligned_path.read_text(encoding="utf-8"))
            item = next(
                item
                for item in payload["traceEvents"]
                if item["pid"] == domain
                and item["ph"] == "X"
                and (domain == "host" or item["name"].startswith("host_sync_wait"))
            )
            item["ts"] += 3
            aligned_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, pattern):
                compare_aligned(self.path)

    def test_aligned_cli_writes_one_new_summary_and_preserves_traces(self):
        aligned = self.aligned_fixture()
        originals = {path: path.read_bytes() for path in (self.path, aligned)}
        summary = self.path.with_name("audit.txt")
        argv = [str(self.path), "--compare-aligned", "--summary-file", str(summary)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv), 0)
        content = summary.read_text(encoding="utf-8")
        self.assertIn("WAIT_COMPARISON", content)
        self.assertIn("no_negative_relations_over_1us_in_ordinal_candidates", content)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        with self.assertRaises(FileExistsError):
            main(argv)

    def test_two_streams_for_same_host_sequence_remain_ambiguous(self):
        self.aligned_fixture([*host_events(1, 100), *hardware_events(0, 30), *hardware_events(0, 30, tid="27")])
        result = compare_aligned(self.path)
        self.assertEqual(result["status"], "incomplete_pairing")
        self.assertEqual(result["ambiguous_or_uncovered_kinds"], ["prefetch"])


if __name__ == "__main__":
    unittest.main()
