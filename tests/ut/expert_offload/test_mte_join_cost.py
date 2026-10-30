# SPDX-License-Identifier: Apache-2.0
"""CPU checks for cost-run isolation and honest workload comparisons."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class JoinCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location("join_cost_test", cls.root / "script/mte_join_cost.py")
        cls.cost = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cost)
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "cost_launch_test", cls.root / "script/run_verified_cpp_tests.py"
            )
            cls.launch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.launch)

    @staticmethod
    def source():
        config = {
            "expert_offload_config": {
                "expert_offload": True,
                "cache_policy_enabled": True,
                "expert_prefetch_num": 1,
                "host_sync_diagnose_load": True,
                "host_sync_diagnose_prefetch_wait_before_gate": True,
                "host_sync_diagnose_compute_cube_cores": 14,
                "host_sync_diagnose_new_unknown_flag": True,
                "moe_offload_debug": True,
            },
            "unrelated": {"keep": 7},
        }
        command = [
            "vllm",
            "serve",
            "/model path",
            "--additional-config",
            json.dumps(config),
            "--speculative-config",
            '{"method":"mtp","num_speculative_tokens":2}',
        ]
        return (
            "export ASCEND_LAUNCH_BLOCKING=1\n"
            "export ASCEND_MODULE_LOG_LEVEL='RUNTIME=0'\n"
            "export ASCEND_GLOBAL_LOG_LEVEL=0\n"
            "export KEEP_THIS=hello\n" + shlex.join(command) + "\n"
        )

    @staticmethod
    def report():
        loads = dict(mean=1.0, p50=1.0, p95=1.0, p99=1.0, min=1.0, max=1.0)
        metrics = {
            name: {"mean": 10.0, "p95": 20.0}
            for name in (
                "total",
                "observe",
                "choose_victims",
                "load_prepare",
                "copy_submit",
                "scale_refresh",
                "load_submit",
                "synchronize",
            )
        }
        return {
            "units": "us",
            "count": 2,
            "errors": 0,
            "dropped": 0,
            "mstx_enabled": False,
            "profile_status": "skipped",
            "request_us": 1000.0,
            "request_stats_us": {"p95": 1200.0},
            "groups": [
                {"kind": "prefetch", "token_rows": 1, "count": 2, "layers": [1, 2], "loads": loads, "metrics": metrics}
            ],
            "request_specs": [{"prompt_sha256": "prompt", "max_tokens": 5, "temperature": 0}],
            "digests": [hashlib.sha256(b"output").hexdigest()],
            "usage": [{"prompt_tokens": 5, "completion_tokens": 5}],
            "untimed": {
                "count": 1,
                "request_us": 900.0,
                "stats_us": {"p95": 1000.0},
                "digests": [hashlib.sha256(b"output").hexdigest()],
                "usage": [{"prompt_tokens": 5, "completion_tokens": 5}],
            },
        }

    def compare(self, first=None, second=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.cost.print_cost_comparison(
                {"callback": first or self.report(), "mte": second or self.report()}
            )
        return result, output.getvalue()

    def test_remove_all_diagnoses_preserve_model_mtp_and_unrelated_config(self):
        result = self.cost.prepare_cost_source(self.source())
        _, words, _, _ = self.launch.launch_spec(result, "python", 7, 8155, 18155)
        offload = json.loads(words[words.index("--additional-config") + 1])
        self.assertFalse(any(key.startswith("host_sync_diagnose_") for key in offload["expert_offload_config"]))
        self.assertFalse(offload["expert_offload_config"]["moe_offload_debug"])
        self.assertEqual(offload["expert_offload_config"]["expert_prefetch_num"], 1)
        self.assertEqual(offload["unrelated"], {"keep": 7})
        self.assertEqual(words[2], "/model path")
        self.assertEqual(json.loads(words[words.index("--speculative-config") + 1])["num_speculative_tokens"], 2)

    def test_environment_overrides_survive_launch_spec_after_inherited_debug(self):
        result = self.cost.prepare_cost_source(self.source())
        prefix, _, _, _ = self.launch.launch_spec(result, "python", 7, 8155, 18155)
        self.assertGreater(
            prefix.rindex("unset ASCEND_LAUNCH_BLOCKING"), prefix.index("export ASCEND_LAUNCH_BLOCKING=1")
        )
        self.assertGreater(prefix.rindex("export ASCEND_MODULE_LOG_LEVEL='RUNTIME=3'"), prefix.index("RUNTIME=0"))
        self.assertGreater(
            prefix.rindex("export ASCEND_GLOBAL_LOG_LEVEL=3"), prefix.index("export ASCEND_GLOBAL_LOG_LEVEL=0")
        )
        self.assertIn("export KEEP_THIS=hello", prefix)

    def test_current_test_script_with_trailing_continuation_is_supported(self):
        source = (self.root / "script/test_graph2.sh").read_text(encoding="utf-8")
        self.assertIn("vllm serve", self.cost.prepare_cost_source(source))

    def test_malformed_sources_are_rejected(self):
        sources = [
            "echo hello",
            self.source() + "vllm serve /other\n",
            self.source() + "echo hello\n",
            self.source().rstrip() + "; echo hello\n",
            "vllm serve /model --additional-config '{}'\n",
            "vllm serve /model --additional-config 'unterminated\n",
        ]
        for source in sources:
            with self.subTest(source=source[-70:]), self.assertRaises(ValueError):
                self.cost.prepare_cost_source(source)

    def test_equal_workload_and_mean_only_sum(self):
        result, output = self.compare()
        group = result["groups"][0]
        self.assertTrue(result["comparable"])
        self.assertEqual(group["estimated_experts"], [2.0, 2.0])
        self.assertEqual(
            group["metrics"]["load_and_wait"], {"callback_mean": 20.0, "mte_mean": 20.0, "mean_delta": 0.0}
        )
        self.assertNotIn("p95", group["metrics"]["load_and_wait"])
        self.assertIn("p95不可由分项相加", output)
        self.assertIn("内部计时关闭", output)
        self.assertIn("内部计时开启", output)
        self.assertIn("OUTPUT_MATCH=PASS", output)
        self.assertEqual(result["request_windows"]["untimed"]["callback_mean"], 900.0)

    def test_unequal_sample_counts_remain_descriptive(self):
        report = self.report()
        report["count"] = report["groups"][0]["count"] = 3
        result, output = self.compare(second=report)
        self.assertFalse(result["comparable"])
        self.assertFalse(result["groups"][0]["samples_match"])
        self.assertIn("仅描述", output)

    def test_equal_expert_total_does_not_hide_unequal_workload_distribution(self):
        report = self.report()
        report["groups"][0]["loads"]["p95"] = 2.0
        result, _ = self.compare(second=report)
        self.assertFalse(result["comparable"])
        self.assertEqual(result["groups"][0]["estimated_experts"], [2.0, 2.0])

    def test_missing_group_is_not_a_comparable_pair(self):
        report = self.report()
        report["groups"][0]["kind"] = "reactive"
        result, _ = self.compare(second=report)
        self.assertFalse(result["comparable"])
        self.assertEqual(len(result["groups"]), 2)
        self.assertTrue(all(group["missing"] for group in result["groups"]))

    def test_output_or_request_mismatch_rejected_before_printing_comparison(self):
        for field in ("digests", "usage", "request_specs"):
            report = self.report()
            report[field] = ["different"]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.compare(second=report)

    def test_untimed_output_mismatch_is_not_ignored(self):
        report = self.report()
        report["untimed"]["digests"] = [hashlib.sha256(b"different").hexdigest()]
        with self.assertRaisesRegex(ValueError, "计时关闭"):
            self.compare(second=report)

    def test_empty_or_malformed_text_hash_rejected_in_both_windows(self):
        for untimed in (False, True):
            for digest in (hashlib.sha256(b"").hexdigest(), "", "not-a-digest", "g" * 64):
                report = self.report()
                window = report["untimed"] if untimed else report
                window["digests"] = [digest]
                with self.subTest(untimed=untimed, digest=digest), self.assertRaisesRegex(ValueError, "SHA256"):
                    self.compare(second=report)

    def test_zero_or_missing_tokens_rejected_in_both_windows(self):
        for untimed in (False, True):
            for usage in ({"completion_tokens": 0}, {}, {"completion_tokens": True}, "invalid"):
                report = self.report()
                window = report["untimed"] if untimed else report
                window["usage"] = [usage]
                with (
                    self.subTest(untimed=untimed, usage=usage),
                    self.assertRaisesRegex(ValueError, "completion_tokens"),
                ):
                    self.compare(second=report)

    def test_untimed_count_must_equal_request_plan_even_if_lists_match(self):
        report = self.report()
        report["untimed"]["count"] = 2
        report["untimed"]["digests"] *= 2
        report["untimed"]["usage"] *= 2
        with self.assertRaisesRegex(ValueError, "请求计划"):
            self.compare(second=report)

    def test_incomplete_reports_rejected(self):
        for field, value in (
            ("count", 0),
            ("errors", 1),
            ("dropped", 1),
            ("mstx_enabled", True),
            ("profile_status", "complete"),
            ("groups", []),
            ("digests", []),
            ("usage", []),
            ("request_specs", []),
            ("untimed", {}),
            ("request_stats_us", {}),
        ):
            report = self.report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare(second=report)

    def test_missing_or_nonfinite_metric_rejected(self):
        for change in ("missing", "nan", "negative"):
            report = self.report()
            if change == "missing":
                del report["groups"][0]["metrics"]["copy_submit"]
            else:
                report["groups"][0]["metrics"]["copy_submit"]["mean"] = float("nan") if change == "nan" else -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare(second=report)

    def test_group_count_total_and_duplicate_keys_checked(self):
        report = self.report()
        report["count"] = 3
        with self.assertRaises(ValueError):
            self.compare(second=report)
        report = self.report()
        report["groups"].append(copy.deepcopy(report["groups"][0]))
        report["count"] = 4
        with self.assertRaisesRegex(ValueError, "重复"):
            self.compare(second=report)

    def test_cleanup_error_position_is_not_treated_as_request_error(self):
        report = self.report()
        report["engine_error_log_positions"] = {"cleanup_marker_unique": True, "before_cleanup": 0, "after_cleanup": 1}
        result, output = self.compare(second=report)
        self.assertTrue(result["comparable"])
        self.assertIn("before_cleanup=0 after_cleanup=1", output)
        report["engine_error_log_positions"]["before_cleanup"] = 1
        with self.assertRaisesRegex(ValueError, "清理"):
            self.compare(second=report)


if __name__ == "__main__":
    unittest.main()
