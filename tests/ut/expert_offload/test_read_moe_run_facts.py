# SPDX-License-Identifier: Apache-2.0
"""日志提取不运行模型，不用当前脚本配置冒充历史启动参数。"""

import contextlib
import importlib.util
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path


class RunFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / "script/read_moe_run_facts.py"
        spec = importlib.util.spec_from_file_location("read_moe_run_facts_test", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def command(self, model):
        return "Effective serve command: " + shlex.join(
            [
                "vllm",
                "serve",
                str(model),
                "--compilation-config",
                '{"cudagraph_capture_sizes":[1,2,3]}',
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":2}',
                "--additional-config",
                '{"expert_offload_config":{"num_device_experts":24}}',
                "--profiler-config",
                '{"torch_profiler_dir":"/results/profile/mte_combined/cpp"}',
            ]
        )

    def test_quoted_actual_command(self):
        result = self.module.parse_command(self.command("/models/DeepSeek V4"))
        self.assertEqual(result["model_path"], "/models/DeepSeek V4")
        self.assertEqual(result["compilation-config"]["cudagraph_capture_sizes"], [1, 2, 3])
        self.assertEqual(result["speculative-config"]["num_speculative_tokens"], 2)
        self.assertEqual(result["expert_offload_config"]["num_device_experts"], 24)

    def test_missing_logs_not_inferred_from_current_script(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(directory)
            (root / "test_graph2.sh").write_text(self.command("/unused"), encoding="utf-8")
            self.module.main([str(root)])
            self.assertIn("未找到实际启动命令", output.getvalue())
            self.assertNotIn("SERVE line=", output.getvalue())

    def test_read_only_extraction_and_malformed_line_recovery(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"n_routed_experts": 256}), encoding="utf-8")
            command = self.command(model)
            (root / "run_details.log").write_text(
                'Effective serve command: "unterminated\n' + command + "\n" + command + "\n", encoding="utf-8"
            )
            case = root / "mte_combined"
            case.mkdir()
            (case / "inference_cpp.log").write_text(
                "Actual usage: 33 GiB for weights, 2 GiB for peak activation. Current KV cache memory: 50 GiB.\n"
                "on-demand loads /layer/step 4.5\nSpecDecoding metrics: Mean acceptance length: 2.00\n",
                encoding="utf-8",
            )
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.module.main([str(root)])
            printed = output.getvalue()
            self.assertIn("命令解析失败", printed)
            self.assertEqual(printed.count("SERVE line="), 1)
            self.assertIn("Current KV cache memory: 50 GiB", printed)
            self.assertIn("on-demand loads /layer/step", printed)
            self.assertIn("Mean acceptance length: 2.00", printed)
            self.assertIn('"n_routed_experts": 256', printed)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
