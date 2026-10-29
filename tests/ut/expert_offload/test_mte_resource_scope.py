# SPDX-License-Identifier: Apache-2.0
"""CPU-only decisions for the bounded, two-case resource diagnosis."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ResourceScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[3] / "script"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "resource_scope_test", scripts / "diagnose_mte_resource_scope.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def known_failure(self):
        return "\n".join(
            [
                "RESOURCE_API=torch",
                "RUNNER_EXIT=1",
                "[MOE-MTE ACL_DETAIL] stream timeout",
                "w13_weight_scale ACL_WAIT_RETURN code=507046",
                'GRAPH_KERNEL [23, "HcPre_1000", 14, 0]',
                'GRAPH_KERNEL [23, "MoeInitRoutingV3_10030000", 5, 1]',
                'GRAPH_KERNEL [21, "host_sync_wait_2", 1, 0]',
            ]
        )

    def test_matching_failure_allows_auxiliary_control(self):
        text = self.known_failure()
        self.assertEqual(self.module.classify_report(text, 1), "scale_wait_reproduced")
        self.assertEqual(self.module.classify_resource_evidence(text, 0)["main_targets"], "VERIFIED")

    def test_outcome_does_not_claim_resource_controls_were_verified(self):
        text = self.known_failure()
        for changed in (text.replace("14, 0", "28, 0"), text.replace("5, 1", "56, 1"), text.replace("1, 0", "2, 0")):
            with self.subTest(changed=changed):
                self.assertEqual(self.module.classify_report(changed, 1), "scale_wait_reproduced")
                self.assertEqual(self.module.classify_resource_evidence(changed, 0)["main_targets"], "UNVERIFIED")
        self.assertEqual(self.module.classify_report("RuntimeError: startup failed", 1), "inconclusive")

    def test_scale_name_and_timeout_must_belong_to_same_record(self):
        text = self.known_failure().replace(
            "w13_weight_scale ACL_WAIT_RETURN code=507046",
            "w13_weight_scale ACL_WAIT_RETURN code=0\nscale_refresh ACL_WAIT_RETURN code=507046",
        )
        self.assertEqual(self.module.classify_report(text, 1), "inconclusive")

    def test_arbitrary_matmul_count_never_proves_auxiliary_stream_identity(self):
        text = self.known_failure() + (
            '\nGRAPH_KERNEL [22, "MatMulV3_bf16_100", 14, 1]\n'
            "[MOE-MTE COMPUTE_LIMIT] api=torch stream=dsa_aux pointer=1234 cube=14 vector=28\n"
        )
        evidence = self.module.classify_resource_evidence(text, 14)
        self.assertEqual(evidence["main_targets"], "VERIFIED")
        self.assertEqual(evidence["dsa_aux"], "UNVERIFIED")
        self.assertEqual(evidence["matmul_candidates"][0]["stream"], 22)
        self.assertEqual(evidence["matmul_candidates"][0]["blocks"], 14)
        self.assertIn("明确关联", evidence["dsa_aux_reason"])

    def test_malformed_and_missing_graph_records_are_not_resource_evidence(self):
        malformed = 'RESOURCE_API=torch\nGRAPH_KERNEL [23, "HcPre", true, 0]\nGRAPH_KERNEL null\nGRAPH_KERNEL {bad}'
        evidence = self.module.classify_resource_evidence(malformed, 0)
        self.assertEqual(evidence["main_targets"], "UNVERIFIED")
        self.assertEqual(evidence["target_status"]["hcpre"], "UNVERIFIED")
        self.assertEqual(evidence["dsa_aux"], "NOT_REQUESTED")
        text = self.known_failure().replace("RESOURCE_API=torch", "RESOURCE_API=raw_acl")
        self.assertEqual(self.module.classify_resource_evidence(text, 0)["main_targets"], "UNVERIFIED")

    def test_request_success_requires_report_and_exit_code(self):
        text = 'RUNNER_EXIT=0\nMOE_RESULT {"profile_status": "invalid"}'
        self.assertEqual(self.module.classify_report(text, 0), "request_passed")
        self.assertEqual(self.module.classify_resource_evidence(text, 14)["dsa_aux"], "UNVERIFIED")
        self.assertEqual(self.module.classify_report(text, 1), "inconclusive")
        self.assertEqual(self.module.classify_report("RUNNER_EXIT=0", 0), "inconclusive")

    def test_compact_summary_keeps_resource_api_and_observed_task(self):
        text = 'RESOURCE_API=torch\nMATCH {"name":"MatMulV3"}\nnoise'
        result = self.module.concise_report(text)
        self.assertIn("RESOURCE_API=torch", result)
        self.assertIn("MatMulV3", result)
        self.assertNotIn("noise", result)

    def test_engine_trace_is_not_lost_behind_api_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text(
                "(EngineCore pid=1) Traceback (most recent call last):\n"
                "(EngineCore pid=1)   File helper.py, line 7\n"
                "(EngineCore pid=1) ValueError: invalid stream\n"
                + "ordinary log\n"
                * 100
                + "(APIServer pid=2) Traceback (most recent call last):\n"
                "(APIServer pid=2) RuntimeError: Engine core initialization failed\n",
                encoding="utf-8",
            )
            text = self.module.failure_context(path)
            self.assertIn("File helper.py, line 7", text)
            self.assertIn("ValueError: invalid stream", text)
            self.assertLess(len(text.splitlines()), 70)

    def test_truncated_trace_keeps_terminal_error_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            self.assertIn("LOG MISSING", self.module.failure_context(path))
            path.write_text(
                "Traceback (most recent call last):\n" + " frame\n" * 70 + "RuntimeError: final cause\n",
                encoding="utf-8",
            )
            self.assertIn("RuntimeError: final cause", self.module.failure_context(path))

    def test_read_only_mode_never_runs_case_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = root / "native_main_aux/mte/inference_python.log"
            service.parent.mkdir(parents=True)
            service.write_text("RuntimeError: startup cause\n", encoding="utf-8")
            before = set(root.rglob("*"))
            with (
                patch.object(sys, "argv", ["diagnose", "--inspect", str(root)]),
                patch.object(self.module, "run_case") as run,
                patch.object(self.module, "kernel_failure_context", return_value="KERNEL unavailable"),
                patch("builtins.print") as output,
            ):
                self.module.main()
            run.assert_not_called()
            self.assertIn("startup cause", str(output.call_args_list))
            self.assertEqual(before, set(root.rglob("*")))

    def test_api_trace_does_not_hide_silent_engine_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text(
                "(EngineCore pid=9) Loading checkpoint shards\n"
                "Engine process exited with code -9\n"
                "(APIServer pid=10) Traceback (most recent call last):\n"
                + "(APIServer pid=10) secondary frame\n" * 60
                + "RuntimeError: Engine core initialization failed\n",
                encoding="utf-8",
            )
            text = self.module.failure_context(path)
            self.assertIn("Loading checkpoint shards", text)
            self.assertIn("exited with code -9", text)
            self.assertNotIn("secondary frame\n" * 40, text)

    def test_kernel_denial_and_oom_are_not_automatic_diagnoses(self):
        with patch.object(sys, "platform", "linux"), patch.object(self.module.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = "Operation not permitted"
            self.assertIn("UNAVAILABLE", self.module.kernel_failure_context())
            run.return_value.returncode = 0
            run.return_value.stdout = "normal boot\n[date] Killed process 123 (python)\n"
            text = self.module.kernel_failure_context()
            self.assertIn("Killed process 123", text)
            self.assertNotIn("normal boot", text)
            self.assertIn("不能自动关联", text)

    def test_aux_retry_runs_only_aux_group(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_script = Path(directory) / "script/diagnose_mte_resource_scope.py"
            with (
                patch.object(self.module, "__file__", str(fake_script)),
                patch.object(sys, "argv", ["diagnose", "--npu", "7", "--case", "native_main_aux"]),
                patch.object(self.module, "run_case", return_value=(0, "RUNNER_EXIT=0\nMOE_RESULT {}")) as run,
                patch.object(self.module, "host_memory_snapshot", return_value="HOST_MEMORY test"),
                patch("builtins.print"),
            ):
                self.module.main()
            run.assert_called_once()
            _, case_path, npu, cores = run.call_args.args
            self.assertEqual((case_path.name, npu, cores), ("native_main_aux", 7, 14))
            summary = (case_path.parent / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("单独补跑 native_main_aux", summary)
            self.assertIn("request_passed", summary)
            self.assertIn('"dsa_aux": "UNVERIFIED"', summary)

    def test_two_case_run_stops_on_scale_failure_with_unverified_main_control(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_script = Path(directory) / "script/diagnose_mte_resource_scope.py"
            changed = self.known_failure().replace("14, 0", "28, 0")
            with (
                patch.object(self.module, "__file__", str(fake_script)),
                patch.object(sys, "argv", ["diagnose", "--npu", "7"]),
                patch.object(self.module, "run_case", return_value=(1, changed)) as run,
                patch.object(self.module, "host_memory_snapshot", return_value="HOST_MEMORY test"),
                patch("builtins.print"),
            ):
                self.module.main()
            run.assert_called_once()
            case_path = run.call_args.args[1]
            summary = (case_path.parent / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("RESULT=scale_wait_reproduced", summary)
            self.assertIn("主流目标配置未验证", summary)


if __name__ == "__main__":
    unittest.main()
