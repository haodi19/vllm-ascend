"""Standalone benchmark checks; run this file directly without vLLM/NPU."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path


class TestBenchmarkInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = runpy.run_path(str(
            Path(__file__).resolve().parents[3] / "bench_lrc_policy.py"
        ))

    def test_commented_continuation_does_not_hide_active_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            config.write_text(
                "class ExpertOffloadConfig:\n"
                "    _defaults = {'num_device_experts': 32, 'cache_router_weight': 0.3}\n",
                encoding="utf-8",
            )
            script = root / "test.sh"
            script.write_text(
                "# example with trailing continuation \\\n"
                "vllm serve model \\\n"
                " --additional-config '{\"expert_offload_config\": "
                "{\"num_device_experts\": 84}}'\n"
                "# --additional-config '{}'\n",
                encoding="utf-8",
            )
            settings = self.bench["project_settings"](script, config)
            self.assertEqual(settings["num_device_experts"], 84)
            self.assertEqual(settings["cache_router_weight"], 0.3)

    def test_nested_model_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"text_config": {
                "n_routed_experts": 128, "num_experts_per_tok": 4,
            }}), encoding="utf-8")
            self.assertEqual(self.bench["model_dimensions"](path), (128, 4))

    def test_plural_selection_and_protection(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload/lrc_policy.py"
        policy_class, _ = self.bench["load_policy"](path)
        policy = policy_class(num_layers=1, num_experts=8, cache_size=4, topk=2)
        needed = policy.observe(0, [[0, 5], [0, 6]])
        slots = {0: 0, 1: 1, 2: 2, 3: 3}
        misses = needed - set(slots.values())
        selected = policy.choose_victims(0, slots, needed, len(misses))
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)
        self.assertTrue(set(selected) <= {1, 2, 3})
        self.assertEqual(policy.choose_victims(0, slots, needed, 0), [])


if __name__ == "__main__":
    unittest.main()
