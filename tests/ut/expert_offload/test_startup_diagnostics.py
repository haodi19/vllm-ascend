# SPDX-License-Identifier: Apache-2.0
"""CPU tests for startup tracing without importing vLLM, Torch, or NPU modules."""

import ast
import importlib.util
import io
import json
import sys
import unittest
from contextlib import nullcontext, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class StartupDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.helper_path = cls.root / "vllm_ascend/expert_offload/startup_diagnostics.py"
        spec = importlib.util.spec_from_file_location("startup_diagnostics_test", cls.helper_path)
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def config(self, enabled=True):
        return SimpleNamespace(
            additional_config={"expert_offload_config": {"host_sync_diagnose_prefetch_wait_at_join": enabled}},
            quant_config=None,
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)),
        )

    def records(self, output):
        prefix = "[MOE-MTE STARTUP] "
        return [json.loads(line.split(prefix, 1)[1]) for line in output.getvalue().splitlines() if prefix in line]

    def method(self, relative_path, name, namespace):
        tree = ast.parse((self.root / relative_path).read_text(encoding="utf-8"))
        method = next(
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        method.decorator_list = []
        namespace.update(StartupDiagnostics=self.helper.StartupDiagnostics, nn=SimpleNamespace(Module=object))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(self.root / relative_path), "exec"), namespace)
        return namespace[name]

    def test_disabled_and_missing_configuration_have_no_observable_side_effects(self):
        configs = [None, SimpleNamespace(), SimpleNamespace(additional_config=None), self.config(False)]
        configs.extend(SimpleNamespace(additional_config=value) for value in ({}, [], {"expert_offload_config": None}))
        with (
            patch.object(self.helper.faulthandler, "enable") as arm,
            patch.object(self.helper.time, "time_ns") as wall,
            patch.object(self.helper.time, "monotonic_ns") as monotonic,
            patch.object(self.helper.os, "getpid") as pid,
            patch.object(self.helper.threading, "get_native_id") as tid,
            redirect_stderr(io.StringIO()) as output,
        ):
            for config in configs:
                trace = self.helper.StartupDiagnostics.from_config(config)
                trace.arm()
                trace.mark("point", "begin")
                with trace.stage("phase"):
                    pass
            for effect in (arm, wall, monotonic, pid, tid):
                effect.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_enabled_markers_have_process_thread_and_both_clocks(self):
        output = io.StringIO()
        with (
            redirect_stderr(output),
            self.helper.StartupDiagnostics.from_config(self.config()).stage("draft.share_topk_indices"),
        ):
            pass
        records = self.records(output)
        self.assertEqual([record["event"] for record in records], ["begin", "end"])
        for record in records:
            self.assertEqual(record["phase"], "draft.share_topk_indices")
            for field in ("pid", "tid", "wall_time_ns", "monotonic_ns"):
                self.assertIsInstance(record[field], int)

    def test_exception_is_rethrown_without_stringifying_or_success_end(self):
        class PrivateError(Exception):
            def __str__(self):
                raise AssertionError("exception contents must not be inspected")

        failure = PrivateError()
        output = io.StringIO()
        with (
            redirect_stderr(output),
            self.assertRaises(PrivateError) as caught,
            self.helper.StartupDiagnostics(True).stage("draft.lm_head_equal"),
        ):
            raise failure
        self.assertIs(caught.exception, failure)
        records = self.records(output)
        self.assertEqual([record["event"] for record in records], ["begin", "exception"])
        self.assertEqual(records[-1]["exception_type"], "PrivateError")

    def test_faulthandler_uses_service_stderr_and_all_threads(self):
        with patch.object(self.helper.faulthandler, "enable") as enable, redirect_stderr(io.StringIO()) as output:
            self.helper.StartupDiagnostics(True).arm("runner.faulthandler")
        enable.assert_called_once_with(file=2, all_threads=True)
        self.assertEqual([record["event"] for record in self.records(output)], ["begin", "end"])

    def test_marker_write_failure_cannot_change_success_or_replace_model_exception(self):
        failure = RuntimeError("original model failure")
        stderr = SimpleNamespace(write=Mock(side_effect=OSError("closed diagnostic stream")))
        with patch.object(self.helper.sys, "stderr", stderr):
            with self.helper.StartupDiagnostics(True).stage("success"):
                pass
            with self.assertRaises(RuntimeError) as caught, self.helper.StartupDiagnostics(True).stage("failure"):
                raise failure
        self.assertIs(caught.exception, failure)

    def test_faulthandler_failure_is_explicit(self):
        with (
            patch.object(self.helper.faulthandler, "enable", side_effect=OSError("no fd")),
            redirect_stderr(io.StringIO()) as output,
            self.assertRaises(OSError),
        ):
            self.helper.StartupDiagnostics(True).arm()
        records = self.records(output)
        self.assertEqual([record["event"] for record in records], ["begin", "exception"])
        self.assertEqual(records[-1]["exception_type"], "OSError")

    def lm_head_case(self, enabled, outcome=True):
        class PrivateWeight:
            def __repr__(self):
                raise AssertionError("weights must not be inspected")

        target = SimpleNamespace(lm_head=SimpleNamespace(weight=PrivateWeight()))
        original = SimpleNamespace(weight=PrivateWeight())
        layer = SimpleNamespace(shared_head=SimpleNamespace(head=original))
        proposer = SimpleNamespace(
            method="mtp",
            model=SimpleNamespace(model=SimpleNamespace(layers={"0": layer})),
            vllm_config=self.config(enabled),
            use_cuda_graph=False,
        )
        equal = Mock(side_effect=outcome if isinstance(outcome, Exception) else None, return_value=outcome)
        method = self.method(
            "vllm_ascend/spec_decode/llm_base_proposer.py",
            "_maybe_share_lm_head",
            {"torch": SimpleNamespace(equal=equal)},
        )
        return method, proposer, target, original, layer, equal

    def test_lm_head_tracing_preserves_one_comparison_and_sharing(self):
        for enabled in (False, True):
            for equal_result in (False, True):
                method, proposer, target, original, layer, equal = self.lm_head_case(enabled, equal_result)
                with redirect_stderr(io.StringIO()) as output:
                    method(proposer, target)
                equal.assert_called_once_with(original.weight, target.lm_head.weight)
                self.assertIs(layer.shared_head.head, target.lm_head if equal_result else original)
                records = self.records(output)
                self.assertEqual([record["event"] for record in records], ["begin", "end"] if enabled else [])

    def test_lm_head_comparison_failure_has_no_end_and_does_not_share(self):
        method, proposer, target, original, layer, equal = self.lm_head_case(True, RuntimeError("comparison failed"))
        with redirect_stderr(io.StringIO()) as output, self.assertRaises(RuntimeError):
            method(proposer, target)
        equal.assert_called_once()
        self.assertIs(layer.shared_head.head, original)
        self.assertEqual([record["event"] for record in self.records(output)], ["begin", "exception"])

    def runner_case(self, enabled, exit_error=None):
        calls = []

        class MemoryProfiler:
            consumed_memory = 1024

            def __enter__(self):
                calls.append("profile_enter")
                return self

            def __exit__(self, *exc):
                calls.append("profile_exit")
                if exit_error:
                    raise exit_error

        model = SimpleNamespace(named_parameters=lambda: [])
        runner = SimpleNamespace(
            vllm_config=self.config(enabled),
            model_config=SimpleNamespace(model="test", multimodal_config=None),
            ascend_config=SimpleNamespace(mix_placement=False),
            eplb_enable=False,
            drafter=SimpleNamespace(load_model=lambda model: calls.append("draft_load")),
            use_aux_hidden_state_outputs=False,
            lora_config=None,
            use_eagle=False,
            enable_enpu=False,
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)),
            get_model=lambda: model,
            _start_dump_data=lambda: calls.append("dump"),
        )
        namespace = {
            "time": SimpleNamespace(perf_counter=lambda: 1),
            "logger": Mock(),
            "DeviceMemoryProfiler": MemoryProfiler,
            "get_model": lambda **kwargs: calls.append("model_load") or model,
            "get_tp_context": lambda draft: nullcontext(),
            "get_pp_group": lambda: SimpleNamespace(world_size=1),
            "get_offloader": lambda: SimpleNamespace(post_init=lambda: calls.append("post_init")),
            "supports_multimodal_pruning": lambda model: False,
            "torch": SimpleNamespace(npu=SimpleNamespace(Stream=lambda: calls.append("stream"))),
            "ACLGraphWrapper": lambda *args, **kwargs: calls.append("wrapper") or model,
            "CUDAGraphMode": SimpleNamespace(FULL="FULL", NONE=None),
        }
        method = self.method("vllm_ascend/worker/model_runner_v1.py", "load_model", namespace)
        gate_module = SimpleNamespace(verify_moe_gate_override=lambda model: calls.append("verify"))
        return method, runner, calls, gate_module

    def test_runner_tracing_preserves_loading_and_wrapper_order(self):
        expected = [
            "profile_enter",
            "model_load",
            "verify",
            "draft_load",
            "profile_exit",
            "post_init",
            "stream",
            "wrapper",
            "dump",
        ]
        for enabled in (False, True):
            method, runner, calls, gate_module = self.runner_case(enabled)
            with (
                patch.dict(sys.modules, {"vllm_ascend.models.moe_gate_override": gate_module}),
                patch.object(self.helper.faulthandler, "enable") as arm,
                redirect_stderr(io.StringIO()) as output,
            ):
                method(runner)
            self.assertEqual(calls, expected)
            self.assertEqual(arm.call_count, int(enabled))
            records = self.records(output)
            if not enabled:
                self.assertEqual(records, [])
            else:
                phases = [(record["phase"], record["event"]) for record in records]
                self.assertIn(("runner.drafter_load_model", "end"), phases)
                self.assertLess(
                    phases.index(("runner.memory_profile_exit", "begin")),
                    phases.index(("runner.memory_profile_exit", "end")),
                )
                self.assertLess(
                    phases.index(("runner.memory_profile_exit", "end")),
                    phases.index(("runner.full_graph_wrapper", "begin")),
                )

    def test_memory_profiler_exit_failure_is_distinct_from_draft_loading(self):
        method, runner, calls, gate_module = self.runner_case(True, RuntimeError("profile exit"))
        with (
            patch.dict(sys.modules, {"vllm_ascend.models.moe_gate_override": gate_module}),
            patch.object(self.helper.faulthandler, "enable"),
            redirect_stderr(io.StringIO()) as output,
            self.assertRaises(RuntimeError),
        ):
            method(runner)
        phases = [(record["phase"], record["event"]) for record in self.records(output)]
        self.assertIn(("runner.drafter_load_model", "end"), phases)
        self.assertIn(("runner.memory_profile_exit", "begin"), phases)
        self.assertNotIn(("runner.memory_profile_exit", "end"), phases)
        self.assertIn(("runner.model_load_profile", "exception"), phases)
        self.assertNotIn("wrapper", calls)

    def test_proposer_share_calls_are_covered_by_independent_stages(self):
        tree = ast.parse((self.root / "vllm_ascend/spec_decode/llm_base_proposer.py").read_text(encoding="utf-8"))
        stages = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                context = item.context_expr
                if (
                    isinstance(context, ast.Call)
                    and isinstance(context.func, ast.Attribute)
                    and context.func.attr == "stage"
                ):
                    stages[context.args[0].value] = node
        for phase, call in (
            ("draft.share_topk_indices", "_maybe_share_topk_indices"),
            ("draft.share_lm_head", "_maybe_share_lm_head"),
        ):
            calls = [
                node
                for node in ast.walk(stages[phase])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == call
            ]
            self.assertEqual(len(calls), 1)

    def test_helper_has_no_device_imports_or_tensor_operations(self):
        tree = ast.parse(self.helper_path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        self.assertLessEqual(
            imports, {"faulthandler", "json", "os", "sys", "threading", "time", "collections", "contextlib"}
        )
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertFalse(attributes & {"synchronize", "cpu", "item", "numpy", "tolist"})


if __name__ == "__main__":
    unittest.main()
