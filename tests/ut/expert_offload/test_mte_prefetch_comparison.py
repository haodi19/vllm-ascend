"""三组串行对照的输出与同口径样本检查。"""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
SPEC = importlib.util.spec_from_file_location("prefetch_comparison", ROOT / "script/run_mte_prefetch_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    def test_partial_log_only_uses_completed_cpu_window(self):
        text = (
            "=== WEIGHT TIMING: CPU timers only; no active profiler/MSTX ===\n"
            "INFERENCE python: callbacks=2; errors=0; dropped=0\n"
            "kind=prefetch token_rows=1 callbacks=2 layers=[1] loads_per_callback={'mean': 1.0}\n"
            "load_submit 10.0 9.0 11.0 12.0 8.0 13.0\n"
            "Request usage: [{'completion_tokens': 5}]\n"
            "load_submit 999.0 9.0 11.0 12.0 8.0 13.0\n"
        )
        rows, usage = MODULE.timing_from_log(text)
        self.assertEqual(rows[("prefetch", 1)]["metrics"]["load_submit"], 10.0)
        self.assertEqual(usage, [{"completion_tokens": 5}])
        with self.assertRaises(ValueError):
            MODULE.timing_from_log(text.split("Request usage:")[0])
        with self.assertRaises(ValueError):
            MODULE.timing_from_log(text.replace("errors=0", "errors=1"))

    def test_failure_summary_preserves_runner_error_before_service_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "callback_runner.log"
            path.write_text(
                "WEIGHT TIMING\nFAILED: KeyError: missing_metric\nSERVICE LOG TAIL: service.log\n"
                + "EngineDeadError during cleanup\n" * 100,
                encoding="utf-8",
            )
            output = []
            MODULE.failure_summary(Path(directory), output.append)
            text = "\n".join(output)
            self.assertIn("FAILED: KeyError: missing_metric", text)
            self.assertNotIn("EngineDeadError", text)

    def test_failure_summary_missing_logs(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            MODULE.failure_summary(Path(directory), lambda text: None)

    def test_compare_checks_both_windows(self):
        report = dict(
            digests=["same"],
            usage=[dict(completion_tokens=5)],
            weight_timing=dict(digests=["same"], usage=[dict(completion_tokens=5)]),
        )
        MODULE.check_outputs(report, copy.deepcopy(report))
        different = copy.deepcopy(report)
        different["weight_timing"]["digests"] = ["different"]
        with self.assertRaises(ValueError):
            MODULE.check_outputs(report, different)

    def test_cycles_use_common_layer_rows_and_equal_counts(self):
        base = dict(key=(1, ("3", "3", "3")), **{key: 10 for key in MODULE.METRICS})
        extra = dict(base, key=(2, ("3", "3", "3")))
        groups = dict(
            callback=[base, extra], mte=[dict(base, to_reactive_us=400)], hybrid=[dict(base, to_reactive_us=20)]
        )
        output = []
        MODULE.compare_cycles(groups, output.append)
        text = "\n".join(output)
        self.assertIn("samples/group=1", text)
        self.assertIn("to_reactive_us 10.000 400.000 20.000 -380.000", text)

    def test_three_variants_and_no_tensor_for_full_mte(self):
        self.assertEqual(
            MODULE.CASES, (("callback", "callback", "mte"), ("mte", "mte", "mte"), ("hybrid", "mte", "callback"))
        )

    def test_final_round_excludes_full_mte(self):
        self.assertEqual(MODULE.experiment_cases(False), MODULE.CASES)
        self.assertEqual(MODULE.experiment_cases(True), (MODULE.CASES[0], MODULE.CASES[2]))

    def test_two_group_cycles_compare_to_original_callback(self):
        base = dict(key=(1, ("3", "1", "3")), **{key: 10 for key in MODULE.METRICS})
        output = []
        MODULE.compare_cycles(dict(callback=[base], hybrid=[dict(base, to_reactive_us=20)]), output.append)
        self.assertIn("to_reactive_us 10.000 20.000 +10.000", "\n".join(output))


if __name__ == "__main__":
    unittest.main()
