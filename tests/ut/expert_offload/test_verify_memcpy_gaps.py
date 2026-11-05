# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for explicit device-copy track verification."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class VerifyMemcpyGapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location("verify_memcpy_test", root / "script/verify_memcpy_gaps.py")
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    @staticmethod
    def copy(start, duration, *, pid=100, tid=27, args=None):
        return dict(ph="X", name="MEMCPY_ASYNC", pid=pid, tid=tid, ts=start, dur=duration, args=args or {})

    @staticmethod
    def metadata(pid=100, tid=27, name="Stream 56"):
        return [
            dict(ph="M", name="process_name", pid=pid, args=dict(name="Ascend Hardware")),
            dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=name)),
        ]

    def invoke(self, data, *, pid=None, stream="Stream 56", wrapped=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(dict(traceEvents=data) if wrapped else data), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.module.inspect_trace(path, stream, 100.0, pid)
            return output.getvalue()

    @staticmethod
    def rows(output, label):
        prefix = label + " "
        return [json.loads(line[len(prefix) :]) for line in output.splitlines() if line.startswith(prefix)]

    def test_selects_metadata_name_not_tid_and_accepts_metadata_last(self):
        data = [self.copy(0, 10), self.copy(210, 5), self.copy(0, 1000, tid=56), *self.metadata()]
        data.append(dict(ph="M", name="thread_name", pid=100, tid=56, args=dict(name="Stream 27")))
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                selected = self.rows(self.invoke(data, wrapped=wrapped), "SELECTED")[0]
                self.assertEqual(selected["tid"], "27")
                self.assertEqual(selected["name"], "Stream 56")
                self.assertEqual(selected["copies"], 2)
                self.assertEqual(selected["large_gaps"], 1)

    def test_multiple_devices_same_name_requires_pid(self):
        data = [self.copy(0, 1), *self.metadata(), self.copy(0, 1, pid=200), *self.metadata(pid=200)]
        with self.assertRaisesRegex(ValueError, "不唯一"):
            self.invoke(data)
        self.assertEqual(self.rows(self.invoke(data, pid="200"), "SELECTED")[0]["pid"], "200")

    def test_missing_name_does_not_fall_back_to_tid_or_count(self):
        data = [self.copy(0, 1, tid=56), *self.metadata(tid=56, name="Stream 27")]
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.invoke(data)

    def test_covered_endpoint_overlap_zero_and_unsorted_records(self):
        data = [self.copy(250, 10), self.copy(20, 10), self.copy(0, 100), self.copy(100, 10), *self.metadata()]
        output = self.invoke(data)
        selected = self.rows(output, "SELECTED")[0]
        self.assertEqual(selected["overlaps_excluded"], 1)
        self.assertEqual(selected["zero_gaps"], 1)
        self.assertEqual(selected["gaps"], 2)
        boundary = self.rows(output, "FIRST_LARGE_GAP")[0]
        self.assertEqual(boundary["left"]["end_us"], 110)
        self.assertEqual(boundary["right"]["ts_us"], 250)
        self.assertEqual(boundary["gap_us"], 140)
        self.assertEqual(self.rows(output, "ALL_GAPS_US")[0]["raw"]["mean"], 70)

    def test_nested_copy_keeps_longest_covering_event_as_left_boundary(self):
        data = [self.copy(0, 100), self.copy(10, 5), self.copy(250, 1), *self.metadata()]
        boundary = self.rows(self.invoke(data), "FIRST_LARGE_GAP")[0]
        self.assertEqual(boundary["left"]["event_index"], 1)
        self.assertEqual(boundary["left"]["end_us"], 100)
        self.assertEqual(boundary["gap_us"], 150)

    def test_host_or_non_x_copy_not_selected(self):
        data = [self.copy(0, 10), self.copy(200, 10), *self.metadata()]
        data.extend([self.copy(0, 1000, pid=7), dict(ph="B", name="MEMCPY_ASYNC", pid=100, tid=27, ts=5)])
        data.append(dict(ph="M", name="process_name", pid=7, args=dict(name="CPU Host")))
        selected = self.rows(self.invoke(data), "SELECTED")[0]
        self.assertEqual(selected["copies"], 2)

    def test_arbitrary_args_and_addresses_are_not_printed(self):
        secret = "PRIVATE_SECRET_0x123456"
        args = {
            "src": secret,
            "dst": secret,
            "request": secret,
            "task_id": secret,
            "Stream Id": "56",
            "Copy Size": 8192,
            "Direction": "H2D",
        }
        output = self.invoke([self.copy(0, 10, args=args), self.copy(200, 10, args=args), *self.metadata()])
        self.assertNotIn(secret, output)
        boundary = self.rows(output, "FIRST_LARGE_GAP")[0]
        self.assertEqual(boundary["left"]["bytes"], 8192)
        self.assertEqual(boundary["left"]["direction"], "H2D")
        self.assertNotIn("task_id", boundary["left"])

    def test_truncated_json_rejects_partial_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps([self.copy(0, 1), *self.metadata()])[:-1], encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = self.module.main([str(path), "--stream-name", "Stream 56"])
            self.assertEqual(result, 1)
            self.assertNotIn("SELECTED ", output.getvalue())
            self.assertNotIn("ALL_GAPS_US ", output.getvalue())

    def test_four_groups_continue_after_missing_group_and_exit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case, policy in self.module.CASES:
                if case == "callback":
                    continue
                target = root / "profile" / case / policy / "worker" / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps([self.copy(0, 1), *self.metadata()]), encoding="utf-8")
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = self.module.main([str(root), "--stream-name", "Stream 56"])
            self.assertEqual(result, 1)
            self.assertEqual(len(self.rows(output.getvalue(), "SELECTED")), 3)
            self.assertIn("CASE=mte_combined", output.getvalue())
            self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob("*")))

    def test_retained_copy_limit(self):
        with patch.object(self.module, "MAX_INTERVALS", 2), self.assertRaisesRegex(ValueError, "上限"):
            self.invoke([self.copy(0, 1), self.copy(2, 1), *self.metadata()])

    def test_single_copy_has_no_gap_samples(self):
        output = self.invoke([self.copy(0, 1), *self.metadata()])
        self.assertEqual(self.rows(output, "SELECTED")[0]["gaps"], 0)
        self.assertIsNone(self.rows(output, "ALL_GAPS_US")[0]["raw"])

    def test_conflicting_metadata_is_rejected(self):
        data = [self.copy(0, 1), *self.metadata()]
        conflicts = (
            dict(ph="M", name="thread_name", pid=100, tid=27, args=dict(name="Stream 27")),
            dict(ph="M", name="process_name", pid=100, args=dict(name="CPU Host")),
        )
        for conflict in conflicts:
            with self.subTest(name=conflict["name"]), self.assertRaisesRegex(ValueError, "冲突"):
                self.invoke([*data, conflict])

    def test_boundary_examples_are_deduplicated(self):
        data = [self.copy(start, 10) for start in (0, 110, 320, 630, 1040)]
        output = self.invoke([*data, *self.metadata()])
        first = self.rows(output, "FIRST_LARGE_GAP")
        longest = self.rows(output, "LONGEST_GAP")
        self.assertEqual([item["gap_us"] for item in first], [100, 200, 300])
        self.assertEqual([item["gap_us"] for item in longest], [400])

    @staticmethod
    def callback(start, duration, call, kind, layer, pid=7):
        return dict(
            ph="X",
            name=f"moe_offload/host_callback|call={call}|kind={kind}|layer={layer}",
            pid=pid,
            tid=8,
            ts=start,
            dur=duration,
        )

    def bubble_trace(self):
        return [
            self.copy(0, 10),
            dict(ph="X", name="npu_fx_compiler_inference", pid=7, tid=8, ts=0, dur=100),
            self.callback(200, 100, 1, "prefetch", 1),
            self.copy(210, 10),
            dict(ph="X", name="Transpose_int8", pid=100, tid=27, ts=220, dur=30),
            self.callback(400, 100, 2, "reactive", 1),
            self.copy(410, 10),
            self.copy(600, 10),
            *self.metadata(),
        ]

    def invoke_bubbles(self, data, *extra_args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = self.module.main([str(path), "--stream-name", "Stream 56", "--offload-bubbles", *extra_args])
            return status, output.getvalue()

    def test_offload_mode_excludes_window_outside_and_includes_transpose(self):
        status, output = self.invoke_bubbles(self.bubble_trace())
        self.assertEqual(status, 0)
        phases = self.rows(output, "COPY_CALLBACK_WINDOWS")[0]
        self.assertEqual(phases, dict(before_callbacks=1, inside_unique=2, after_callbacks=1))
        example = self.rows(output, "NORMAL_PAIR_EXAMPLE")[0]
        self.assertEqual(example["gap_us"], 160)
        self.assertEqual(example["copy_only_gap_us"], 190)
        self.assertEqual(example["last_device_task"]["kind"], "transpose")
        compiler = self.rows(output, "COMPILER_RANGE")[0]
        self.assertEqual(compiler["dur_ms"], 0.1)
        self.assertIn("READ 2/2", output)
        self.assertFalse(self.rows(output, "OFFLOAD_STATUS")[0]["partial"])

    def test_offload_mode_multiple_host_processes_require_explicit_selection(self):
        data = [*self.bubble_trace(), self.callback(1000, 100, 1, "prefetch", 1, pid=9)]
        status, output = self.invoke_bubbles(data)
        self.assertEqual(status, 1)
        self.assertIn("--host-pid", output)
        status, output = self.invoke_bubbles(data, "--host-pid", "7")
        self.assertEqual(status, 0)
        self.assertEqual(self.rows(output, "SELECTED_OFFLOAD_TRACK")[0]["host_pid"], "7")

    def test_offload_mode_does_not_retain_unrelated_host_x_events(self):
        host_events = [dict(ph="X", name="Event::synchronize", pid=7, tid=8, ts=index, dur=1) for index in range(100)]
        with patch.object(self.module, "MAX_INTERVALS", 20):
            status, output = self.invoke_bubbles([*host_events, *self.bubble_trace()])
        self.assertEqual(status, 0)
        self.assertEqual(self.rows(output, "SELECTED_OFFLOAD_TRACK")[0]["tasks"], 5)

    def test_offload_mode_partial_is_reported_without_claiming_full_pass(self):
        data = [*self.bubble_trace(), self.callback(205, 80, 3, "reactive", 1)]
        status, output = self.invoke_bubbles(data)
        self.assertEqual(status, 1)
        report = self.rows(output, "OFFLOAD_STATUS")[0]
        self.assertTrue(report["partial"])
        self.assertIn("overlapping_callbacks", report["reasons"])
        self.assertNotIn("PASS：", output)

    def test_offload_mode_preserves_short_gaps_in_main_regular_band(self):
        data = self.bubble_trace()
        data[5]["ts"] = 260
        data[6]["ts"] = 270
        status, output = self.invoke_bubbles(data)
        # Keep Host windows disjoint while giving a short physical stream gap.
        self.assertEqual(status, 1)
        data[2]["dur"] = 50
        status, output = self.invoke_bubbles(data)
        self.assertEqual(status, 0)
        distributions = [
            row for row in self.rows(output, "OFFLOAD_GAPS_US") if row["direction"] == "prefetch->reactive"
        ]
        by_band = {row["band"]: row for row in distributions}
        self.assertEqual(by_band["regular_band"]["raw"]["mean"], 20)
        self.assertIsNone(by_band["large_band"]["raw"])

    def test_offload_mode_rejects_invalid_band_thresholds(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.module.parse_args(
                ["missing", "--stream-name", "Stream 56", "--offload-bubbles", "--long-gap-us", "100"]
            )

    def test_offload_mode_does_not_echo_private_event_arguments(self):
        data = self.bubble_trace()
        for event in data:
            if event["ph"] == "X":
                event["args"] = dict(src="PRIVATE_POINTER_0x1234", secret="PRIVATE_TOKEN")
        status, output = self.invoke_bubbles(data)
        self.assertEqual(status, 0)
        self.assertNotIn("PRIVATE", output)

    def test_compiler_markers_on_other_synthetic_pid_remain_visible_without_linking(self):
        data = self.bubble_trace()
        data[1]["pid"] = 99
        data.append(dict(ph="M", name="process_name", pid=99, args=dict(name="PyTorch")))
        status, output = self.invoke_bubbles(data)
        self.assertEqual(status, 0)
        compiler = self.rows(output, "COMPILER_RANGE")[0]
        self.assertEqual(compiler["pid"], "99")
        self.assertEqual(compiler["process_name"], "PyTorch")
        self.assertEqual(compiler["scope"], "trace_wide_unlinked")
        self.assertEqual(self.rows(output, "COMPILER_MARKERS")[0]["max_duration_us"], 100)


if __name__ == "__main__":
    unittest.main()
