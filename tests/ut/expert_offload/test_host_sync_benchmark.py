# SPDX-License-Identifier: Apache-2.0
"""CPU validation of benchmark statistics and comparison boundaries."""

import contextlib
import copy
import io
import json
import runpy
import sys
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
        cls.compare = staticmethod(runner["compare_results"])
        cls.runner = staticmethod(runner["main"])

    def test_statistics_including_nearest_rank_percentile(self):
        result = self.summarize(list(range(1, 21)))
        self.assertEqual(result, {"count": 20, "mean": 10.5, "p50": 10.5, "p95": 19, "min": 1, "max": 20})

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

    def run_mock_runner(self, steps):
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
            result = self.runner(["--npu", "4"])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
