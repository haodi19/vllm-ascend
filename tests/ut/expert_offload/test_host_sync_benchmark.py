# SPDX-License-Identifier: Apache-2.0
"""CPU validation of benchmark statistics and comparison boundaries."""

import contextlib
import copy
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            bench = runpy.run_path(str(scripts / "host_sync_benchmark.py"))
            runner = runpy.run_path(str(scripts / "run_host_sync_stage4.py"))
        cls.summarize = staticmethod(bench["summarize"])
        cls.register_slots = staticmethod(bench["register_benchmark_slots"])
        cls.summarize_worker = staticmethod(bench["summarize_worker"])
        cls.inspect_profile = staticmethod(bench["inspect_profile"])
        cls.compare = staticmethod(runner["compare_results"])
        cls.runner = staticmethod(runner["main"])

    def test_all_slots_registered_but_only_last_slot_may_execute(self):
        for count in (1, 2048):
            session = Mock()
            callback = Mock()
            self.register_slots(session, count, callback)
            registered = session.register_task.call_args_list
            self.assertEqual([call.args[0] for call in registered], list(range(count)))
            self.assertIs(registered[-1].args[1], callback)
            for call in registered[:-1]:
                with self.assertRaisesRegex(RuntimeError, "Inactive benchmark slot"):
                    call.args[1]()
            callback.assert_not_called()

    def test_statistics_including_nearest_rank_percentile(self):
        result = self.summarize(list(range(1, 21)))
        self.assertEqual(result, {"count": 20, "mean": 10.5, "p50": 10.5, "p95": 19, "min": 1, "max": 20})

    def test_worker_statistics_exclude_warmup_and_convert_units(self):
        raw = dict(
            timing_enabled=True,
            task_samples_dropped=0,
            task_columns=["slot", "generation", "gil_wait_ns"],
            task_samples=[[3, 1, 999000], [3, 2, 2000]],
            idle_columns=["scan_ns"],
            idle_samples=[[1000]],
            idle_sample_stride=1024,
            idle_samples_dropped=0,
        )
        result = self.summarize_worker(raw, 1, 1)
        self.assertEqual(result["task"]["gil_wait_us"]["mean"], 2)
        self.assertEqual(result["idle"]["scan_us"]["mean"], 1)
        with self.assertRaises(RuntimeError):
            self.summarize_worker(raw, 1, 2)
        raw["task_samples_dropped"] = 1
        with self.assertRaises(RuntimeError):
            self.summarize_worker(raw, 1, 1)

    def test_invalid_samples_rejected(self):
        for values in ([], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                self.summarize(values)

    @staticmethod
    def records():
        return [
            {
                "backend": backend,
                "rounds": 200,
                "unit": "us",
                "profiler_enabled": False,
                **{metric: {"mean": mean} for metric in ("event_interval", "replay_sync_host", "callback_body")},
            }
            for backend, mean in (("callback", 100), ("mte", 60), ("mte", 80), ("callback", 120))
        ]

    def test_balanced_comparison_keeps_difference_sign(self):
        result = self.compare(self.records())
        self.assertEqual(
            result["event_interval"],
            {
                "callback_mean_us": 110,
                "mte_mean_us": 70,
                "mte_minus_callback_us": -40,
            },
        )

    def test_missing_backend_or_repeat_rejected(self):
        for records in ([], self.records()[:1], self.records()[:3]):
            with self.assertRaises(ValueError):
                self.compare(records)

    def test_profile_units_and_count_mismatch_rejected(self):
        for field, value in (("profiler_enabled", True), ("unit", "ms"), ("rounds", 1)):
            records = copy.deepcopy(self.records())
            records[0][field] = value
            with self.assertRaises(ValueError):
                self.compare(records)

    def run_mock_runner(self, steps, extra_args=()):
        capture = Mock(side_effect=steps)
        output = io.StringIO()
        with (
            patch.dict(
                self.runner.__globals__,
                capture_step=capture,
                run_graph_regression=Mock(return_value=0),
                run_step=Mock(return_value=True),
            ),
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stdout(output),
        ):
            result = self.runner(["--npu", "4", *extra_args])
        return result, capture, output.getvalue()

    def test_profile_failure_preserves_comparison_and_runs_other_profile(self):
        steps = [(True, "BENCH_RESULT " + json.dumps(record)) for record in self.records()]
        result, capture, output = self.run_mock_runner([*steps, (False, "timeout"), (True, "profile ok")])
        self.assertEqual(result, 1)
        self.assertEqual(capture.call_count, 6)
        self.assertIn("COMPARISON ", output)
        self.assertIn('"callback": "INCOMPLETE", "mte": "PASS"', output)
        for invocation in capture.call_args_list[:4]:
            self.assertNotIn("--profile-dir", invocation.args[1])
        for invocation in capture.call_args_list[4:]:
            self.assertIn("--profile-dir", invocation.args[1])

    def test_timing_failure_prevents_comparison(self):
        result, capture, output = self.run_mock_runner([(False, "timing failed")])
        self.assertEqual(result, 1)
        self.assertEqual(capture.call_count, 1)
        self.assertNotIn("COMPARISON ", output)

    def test_profile_only_does_not_generate_timing_results(self):
        result, capture, output = self.run_mock_runner([(True, "ok"), (True, "ok")], ("--only", "profile"))
        self.assertEqual(result, 0)
        self.assertEqual(capture.call_count, 2)
        self.assertNotIn("COMPARISON ", output)
        self.assertIn("NOT RUN (--only profile)", output)
        for invocation in capture.call_args_list:
            self.assertIn("--profile-dir", invocation.args[1])

    def test_profile_requires_expected_mstx_ranges_not_just_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace_view.json"
            trace.write_text('{"traceEvents": []}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.inspect_profile(root, "mte", 1)
            events = [{"ph": "X", "name": f"host_sync/mte/{label}"} for label in ("callback_body", "replay_and_wait")]
            trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            result = self.inspect_profile(root, "mte", 1)
            self.assertEqual(result["mstx_ranges"], {"callback_body": 1, "replay_and_wait": 1})
            with self.assertRaises(RuntimeError):
                self.inspect_profile(root, "mte", 2)

    def test_profile_end_events_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {"ph": phase, "name": f"host_sync/callback/{label}"}
                for label in ("callback_body", "replay_and_wait")
                for phase in ("B", "E")
            ]
            (root / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
            self.assertEqual(self.inspect_profile(root, "callback", 1)["expected_ranges"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
