# SPDX-License-Identifier: Apache-2.0
"""CPU tests for capture-evidence checks and the single-run diagnostic wrapper."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class WaitPlacementRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        with patch.object(sys, "path", [str(cls.root / "script"), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "wait_runner_test", cls.root / "script/diagnose_mte_wait_placement.py"
            )
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
            runner_spec = importlib.util.spec_from_file_location(
                "moe_runner_placement_test", cls.root / "script/run_moe_host_sync.py"
            )
            cls.runner = importlib.util.module_from_spec(runner_spec)
            runner_spec.loader.exec_module(cls.runner)

    @staticmethod
    def node(stream, task, name, kind="KERNEL_AIVEC"):
        return {"name": name, "args": {"Model Id": 63, "Stream Id": stream, "Task Id": task, "Task Type": kind}}

    def graph(self):
        return [
            self.node(21, 5, "host_sync_publish"),
            self.node(21, 6, "EVENT_WAIT_40", "EVENT_WAIT"),
            self.node(21, 7, "host_sync_wait"),
            self.node(21, 8, "MEMCPY_ASYNC", "MEMCPY_ASYNC"),
            self.node(21, 9, "EVENT_RECORD_41", "EVENT_RECORD"),
            self.node(23, 195, "KvCompressEpilog"),
            self.node(23, 200, "EVENT_RECORD_40", "EVENT_RECORD"),
            self.node(23, 201, "EVENT_WAIT_41", "EVENT_WAIT"),
        ]

    def test_unique_record_before_wait_protects_mapping_and_join(self):
        pairs = self.module.gated_wait_pairs(self.graph())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["strict_wait"], (63, 21, 7))
        self.assertEqual(pairs[0]["mapping_tasks"], [(63, 21, 8)])

    def test_early_wait_or_ambiguous_event_is_not_accepted(self):
        graph = self.graph()
        for changed in (
            [n for n in graph if n["name"] != "EVENT_WAIT_40"],
            [*graph, self.node(24, 300, "EVENT_RECORD_40")],
            [*graph, self.node(24, 301, "EVENT_WAIT_41")],
            [*graph, self.node(24, 300, "EVENT_RECORD_41")],
            [*graph, self.node(24, 301, "EVENT_WAIT_40")],
            [n for n in graph if n["name"] != "MEMCPY_ASYNC"],
        ):
            with self.subTest(changed=changed):
                self.assertEqual(self.module.gated_wait_pairs(changed), [])

    def test_reversed_main_edge_or_duplicate_task_identity_rejected(self):
        graph = self.graph()
        graph[-1]["args"]["Task Id"] = 199
        self.assertEqual(self.module.gated_wait_pairs(graph), [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.module.gated_wait_pairs([*self.graph(), self.graph()[0]])

    def test_request_outcome_is_separate_from_capture_evidence(self):
        self.assertEqual(self.module.request_outcome("RUNNER_EXIT=0\nMOE_RESULT {}", 0), "finite_requests_passed")
        self.assertEqual(self.module.request_outcome("RUNNER_EXIT=0\nMOE_RESULT {}", 1), "inconclusive")
        self.assertEqual(
            self.module.request_outcome("w13_weight_scale COMPLETE\nother ACL_WAIT_RETURN code=507046", 1),
            "inconclusive",
        )
        self.assertEqual(
            self.module.request_outcome("w13_weight_scale ACL_WAIT_RETURN code=507046", 1), "scale_wait_reproduced"
        )

    def test_graph_schema_failure_and_missing_files_remain_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("UNVERIFIED", self.module.graph_evidence(root)[0])
            path = root / "mte/graph_debug/capture/graph_dump.json"
            path.parent.mkdir(parents=True)
            path.write_text("invalid", encoding="utf-8")
            self.assertIn("UNVERIFIED", self.module.graph_evidence(root)[0])
            path.write_text(json.dumps({"traceEvents": self.graph()}), encoding="utf-8")
            self.assertIn("structural_pairs=1", self.module.graph_evidence(root)[0])

    def test_single_launch_without_core_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock()
            process.wait.return_value = 1
            with patch.object(self.module.subprocess, "Popen", return_value=process) as launch:
                self.assertEqual(self.module.run_once(root, root / "consumer_join", 7), 1)
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertIn("--prefetch-wait-at-join", command)
            self.assertNotIn("--compute-cube-cores", command)
            self.assertNotIn("--dsa-cube-cores", command)

    def test_success_summary_keeps_absent_structure_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph_diagnosis.txt").write_text("RUNNER_EXIT=0\nMOE_RESULT {}", encoding="utf-8")
            output = io.StringIO()
            self.module.append_evidence(output, root, 0)
            self.assertIn("finite_requests_passed", output.getvalue())
            self.assertIn("GATE_GRAPH_UNVERIFIED", output.getvalue())

    def test_runner_config_enables_only_mte_and_preserves_model(self):
        source = 'vllm serve /model --additional-config \'{"expert_offload_config": {"expert_offload": true}}\''
        for backend, expected in (("mte", True), ("callback", False)):
            configured = self.runner.configured_source(source, backend, "/lib.so", diagnose_prefetch_wait_at_join=True)
            self.assertIn('"host_sync_diagnose_prefetch_wait_at_join": ' + str(expected).lower(), configured)
            self.assertIn("/model", configured)

    def test_effective_configuration_uses_actual_command_and_rejects_ambiguity(self):
        command = "Effective serve command: vllm serve /actual/model --api-key secret --compilation-config '{}'"
        text = self.module.effective_configuration(command)
        self.assertIn("/actual/model", text)
        self.assertIn("--compilation-config", text)
        self.assertNotIn("secret", text)
        self.assertIn("UNAVAILABLE", self.module.effective_configuration(command + "\n" + command))
        self.assertIn("UNAVAILABLE", self.module.effective_configuration(command + " '"))


if __name__ == "__main__":
    unittest.main()
