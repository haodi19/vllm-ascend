# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for independent profile trials and honest aggregation."""

import contextlib
import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProfileTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[3] / "script"), *sys.path]):
            cls.module = importlib.import_module("run_moe_profile_trials")

    def entry(self, trial=1, value=100, count=8):
        report = dict(
            count=count,
            errors=0,
            dropped=0,
            request_specs=["fixed"],
            digests=["same"],
            usage=[5],
            profile_status="recovered_intervals",
            profile_export_status="incomplete",
            engine_error_log_positions=dict(cleanup_marker_unique=True, before_cleanup=0),
            request_window=dict(
                count=20, request_specs=["fixed"], digests=["same"], usage=[5], samples_us=[1000000] * 20
            ),
            profile_trace_data=[
                dict(
                    usable=True,
                    json_complete=True,
                    host_gaps=[
                        dict(direction=d, outliers_us=self.module.outlier_summary([value] * count))
                        for d in self.module.DIRECTIONS
                    ],
                )
            ],
        )
        bubbles = []
        for label in self.module.LABELS:
            for scope in ("full_capture", "callback_window"):
                bubbles.append(
                    (
                        label,
                        "TARGET_GAPS_US",
                        dict(
                            scope=scope,
                            excluded_short=50,
                            excluded_long=4,
                            kept=dict(n=count, mean=value, p50=value, p95=value, total=value * count),
                        ),
                    )
                )
            bubbles += [
                (label, "OFFLOAD_STATUS", {"partial": True}),
                (
                    label,
                    "TARGET_DIRECTION_GAPS_US",
                    dict(direction="prefetch->reactive", kept=dict(n=2, mean=500, p95=600)),
                ),
            ]
        return dict(
            trial=trial,
            exit_code=0,
            reports={label: copy.deepcopy(report) for label in self.module.LABELS},
            bubbles=bubbles,
            problems=[],
        )

    def test_rotations_balance_positions_without_changing_configuration(self):
        reference = self.module.handoff_profile_cases(True)
        orders = [self.module.handoff_profile_cases(True, i) for i in range(4)]
        for position in range(4):
            self.assertEqual({order[position] for order in orders}, set(reference))

    def test_equal_weight_trials_not_event_pooling_and_no_removed_trials(self):
        entries = [self.entry(value=100, count=1000), self.entry(trial=2, value=300, count=8)]
        text = self.module.render_trials(entries, 4, {})
        self.assertIn("Host prefetch->reactive/us | 2 | 100.000, 300.000 | 200.000", text)
        self.assertIn("已结束 2/4", text)
        self.assertIn("incomplete | recovered_intervals", text)
        self.assertIn("partial", text)
        aggregate = text.split("## 各次均值的波动", 1)[1]
        self.assertNotIn("subset", aggregate)

    def test_mean_improves_but_total_worsens_are_reported_separately(self):
        entry = self.entry()
        for label, kind, value in entry["bubbles"]:
            if label == "mte_combined" and kind == "TARGET_GAPS_US":
                value["kept"].update(mean=80, n=20, total=1600)
        text = self.module.render_trials([entry], 4, {})
        self.assertIn("Bubble callback_window/us mean | 1 | -20.00%", text)
        self.assertIn("Bubble callback_window/us total | 1 | +100.00%", text)

    def test_invalid_or_ambiguous_profiles_do_not_supply_gap_values(self):
        entry = self.entry()
        entry["reports"]["mte_combined"]["profile_status"] = "invalid"
        entry["reports"]["callback"]["profile_trace_data"] *= 2
        rows = self.module.observations(entry)
        for label in ("mte_combined", "callback"):
            self.assertFalse(any(r["case"] == label and not r["metric"].endswith("/ms") for r in rows))

    def test_mismatched_outputs_or_failed_round_not_used_in_aggregate(self):
        for failure in ("output", "exit", "engine"):
            entry = self.entry()
            if failure == "output":
                entry["reports"]["mte_combined"]["digests"] = ["wrong"]
            elif failure == "exit":
                entry["exit_code"] = 1
            else:
                entry["reports"]["callback"]["engine_error_log_positions"]["before_cleanup"] = 1
            text = self.module.render_trials([entry], 4, {})
            self.assertIn("100.000", text.split("## 各次均值的波动")[0])
            self.assertNotIn("100.000", text.split("## 各次均值的波动")[1])

    def test_runner_uses_separate_directories_profiles_and_rotates_then_stops_on_failure(self):
        for fail_at in (None, 2):
            calls = []

            def fake_run(argv, *, handoff_compare, calls=calls, fail_at=fail_at):
                self.assertTrue(handoff_compare)
                root = Path(argv[argv.index("--output-dir") + 1])
                root.mkdir()
                rotation = int(argv[argv.index("--case-rotation") + 1])
                self.assertIn("--measure-tpot", argv)
                self.assertIn("--include-legacy-callback", argv)
                self.assertEqual(argv[argv.index("--profile-timeout") + 1], "600")
                calls.append((root.name, rotation))
                entry = self.entry(len(calls))
                for label, report in entry["reports"].items():
                    (root / label).mkdir()
                    (root / label / "profile_report.json").write_text(json.dumps(report), encoding="utf-8")
                lines = []
                for label, kind, record in entry["bubbles"]:
                    lines.extend([f"CASE={label}; traces=1", kind + " " + json.dumps(record)])
                (root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
                return int(len(calls) == fail_at)

            with tempfile.TemporaryDirectory() as directory, patch.object(sys, "platform", "linux"):
                root = Path(directory) / "run"
                with (
                    patch.object(self.module, "run_comparison", side_effect=fake_run),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = self.module.main(
                        ["--output-dir", str(root), "--npu", "1", "--max-tokens", "128", "--ignore-eos"]
                    )
                self.assertEqual(result, int(fail_at is not None))
                expected = fail_at or 4
                self.assertEqual(calls, [(f"trial_{i:02d}", i - 1) for i in range(1, expected + 1)])
                self.assertEqual((root / "summary.txt").read_bytes(), (root / "experiment_tables.md").read_bytes())
                self.assertIn(f"已结束 {expected}/4", (root / "summary.txt").read_text(encoding="utf-8"))

    def test_multiple_requests_in_one_trace_rejected(self):
        with (
            patch.object(sys, "platform", "linux"),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.module.main(["--output-dir", "unused", "--npu", "1", "--requests", "4"])

    def test_token_sweep_separates_eos_and_outputs_and_stops_on_failure(self):
        for fail in (False, True):
            calls = []

            def fake_main(argv, *, calls=calls, fail=fail):
                calls.append(argv)
                path = Path(argv[argv.index("--output-dir") + 1])
                path.mkdir()
                (path / "summary.txt").write_text(path.name + " results", encoding="utf-8")
                return int(fail)

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with (
                    patch.object(self.module, "main", side_effect=fake_main),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = self.module.run_token_lengths(root, 4, ["--npu", "1"])
                self.assertEqual(code, int(fail))
                self.assertEqual(len(calls), 1 if fail else 2)
                self.assertNotIn("--ignore-eos", calls[0])
                self.assertEqual(calls[0][calls[0].index("--max-tokens") + 1], "5")
                text = (root / "summary.txt").read_text(encoding="utf-8")
                self.assertIn("tokens5 results", text)
                if not fail:
                    self.assertIn("--ignore-eos", calls[1])
                    self.assertEqual(calls[1][calls[1].index("--max-tokens") + 1], "128")
                    self.assertIn("tokens128 results", text)

    def test_sweep_rejects_conflicting_length_flags(self):
        for flags in (["--max-tokens=128"], ["--max-tokens", "5"], ["--ignore-eos"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.module.main(["--both-token-lengths", "--npu", "1", *flags])

    def test_cross_trial_output_change_excluded_and_missing_gap_coverage_visible(self):
        first, second = self.entry(), self.entry(trial=2, value=300)
        for report in second["reports"].values():
            report["digests"] = ["other"]
        second["reports"]["mte_combined"]["profile_trace_data"] = []
        text = self.module.render_trials([first, second], 4, {})
        self.assertIn("mte_combined | Host prefetch->reactive/us | 1/4 | 2", text)
        self.assertNotIn("300.000", text.split("## 各次均值的波动")[1])


if __name__ == "__main__":
    unittest.main()
