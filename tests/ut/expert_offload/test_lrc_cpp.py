"""Standalone differential tests: python -B this_file.py --require-cpp."""

import importlib.util
import json
import random
import runpy
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from bench_lrc_policy import load_policy  # noqa: E402


class TestBackendSelection(unittest.TestCase):
    def test_python_default_and_explicit_selection(self):
        policy, _ = load_policy(REPO / "vllm_ascend/expert_offload/lrc_policy.py")
        module = sys.modules[policy.__module__]
        args = dict(num_layers=1, num_experts=8, cache_size=4, topk=2)
        self.assertIsInstance(module.create_lrc_policy(**args), policy)
        self.assertIsInstance(module.create_lrc_policy(backend="python", **args), policy)
        with self.assertRaises(ValueError):
            module.create_lrc_policy(backend="invalid", **args)
        with patch.dict(sys.modules, {"_vllm_ascend_lrc": None}):
            self.assertIsInstance(module.create_lrc_policy(**args), policy)
            with self.assertRaisesRegex(ImportError, r"C\+\+ LRC requested"):
                module.create_lrc_policy(backend="cpp", **args)

    def test_launcher_only_changes_backend(self):
        select = runpy.run_path(str(REPO / "script/run_lrc_graph.py"))["select_backend"]
        source = (REPO / "script/test_graph2.sh").read_text(encoding="utf-8")
        original = shlex.split(source + "\n", comments=True)
        before = json.loads(original[original.index("--additional-config") + 1])
        updated = shlex.split(select(source, "cpp"), comments=True)
        after = json.loads(updated[updated.index("--additional-config") + 1])
        self.assertEqual(after["expert_offload_config"].pop("cache_policy_backend"), "cpp")
        self.assertEqual(after, before)
        self.assertEqual((REPO / "script/test_graph2.sh").read_text(encoding="utf-8"), source)
        with self.assertRaises(ValueError):
            select("# --additional-config '{}'", "cpp")


@unittest.skipUnless(importlib.util.find_spec("_vllm_ascend_lrc"), "optional C++ extension not installed")
class TestCppParity(unittest.TestCase):
    def pair(self, **overrides):
        args = dict(num_layers=2, num_experts=32, cache_size=12, topk=4, recent_window=7)
        args.update(overrides)
        path = REPO / "vllm_ascend/expert_offload/lrc_policy.py"
        python_cls, _ = load_policy(path)
        cpp_cls, _ = load_policy(path, "cpp")
        return python_cls(**args), cpp_cls(**args)

    def assert_states(self, reference, native):
        self.assertEqual(len(reference.layer_states), len(native.layer_states))
        for index, (a, b) in enumerate(zip(reference.layer_states, native.layer_states)):
            self.assertEqual(a.step, b.step)
            self.assertEqual(a.freq, b.freq)
            self.assertEqual(a.last_used, b.last_used)
            self.assertEqual(a.router_score, b.router_score)
            self.assertEqual(a.recent_queue, b.recent_queue)
            np.testing.assert_array_equal(a.ema, b.ema)
            np.testing.assert_array_equal(reference.hotness_array(index), native.hotness_array(index))
            for eid in range(reference.num_experts):
                self.assertEqual(float(reference.hotness(index, eid)), native.hotness(index, eid))

    def test_repeated_routes_windows_scores_and_eviction(self):
        rng = random.Random(20261010)
        for beta in (0.0, 0.5, 0.9, 0.999):
            reference, native = self.pair(ema_beta=beta)
            for index in range(180):
                layer = index % 2
                # Includes duplicates, invalid IDs, multiple tokens and empty rows.
                rows = [[rng.randrange(-2, 34) for _ in range(index % 5)] for _ in range(index % 4 + 1)]
                scores = [[rng.random() for _ in row] for row in rows] if index % 3 else None
                protected = reference.observe(layer, rows, scores)
                self.assertEqual(protected, native.observe(layer, rows, scores))
                residents = dict(enumerate(rng.sample(range(32), 12)))
                count = index % 15
                self.assertEqual(
                    reference.choose_victims(layer, residents, protected, count),
                    native.choose_victims(layer, residents, protected, count),
                )
                self.assert_states(reference, native)

    def test_seed_add_layer_ties_and_loading(self):
        reference, native = self.pair()
        for obj in (reference, native):
            obj.seed_layer_hotness(0, {1: 0.9, 2: 0.45, 3: 0.0})
            obj.seed_layer_hotness(1, {})
        self.assertEqual(reference.add_layer(), native.add_layer())
        self.assert_states(reference, native)
        residents = {4: 7, 1: 5, 8: 9, 2: 2}
        for protected in (set(), {7}, set(residents.values())):
            for loading in (None, {5, 7}, set(residents.values())):
                self.assertEqual(
                    reference.choose_victim(0, residents, protected, loading),
                    native.choose_victim(0, residents, protected, loading),
                )
            for count in (0, 1, 99):
                self.assertEqual(
                    reference.choose_victims(0, residents, protected, count),
                    native.choose_victims(0, residents, protected, count),
                )
        for obj in (reference, native):
            with self.assertRaises(ValueError):
                obj.choose_victims(0, residents, set(), -1)

    def test_global_counts(self):
        reference, native = self.pair()
        rng = np.random.default_rng(42)
        for index in range(100):
            counts = rng.integers(0, 6, size=32)
            scores = rng.random(32).astype(np.float32) if index % 2 else None
            self.assertEqual(
                reference.observe_global_counts(0, counts, scores), native.observe_global_counts(0, counts, scores)
            )
            self.assert_states(reference, native)
        for obj in (reference, native):
            for counts, scores in (([1], None), ([-1] * 32, None), ([0] * 32, [0])):
                with self.assertRaises(ValueError):
                    obj.observe_global_counts(0, counts, scores)

    def test_score_errors_preserve_reference_state(self):
        for rows, scores, exception in (([[1, 2]], [[0.1]], ValueError), ([[1], [2]], [[0.1]], IndexError)):
            reference, native = self.pair()
            for obj in (reference, native):
                with self.assertRaises(exception):
                    obj.observe(0, rows, scores)
            self.assert_states(reference, native)


if __name__ == "__main__":
    if "--require-cpp" in sys.argv:
        sys.argv.remove("--require-cpp")
        if importlib.util.find_spec("_vllm_ascend_lrc") is None:
            sys.exit("C++ extension missing: build csrc/lrc_policy first (tests NOT passed).")
    unittest.main()
