"""Compare the actual manager's Python and fused planning paths; console only."""

import ast
import contextlib
import importlib.util
import logging
import random
import sys
import threading
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from bench_lrc_policy import load_policy  # noqa: E402


def make_manager(backend, experts=256, capacity=24, stats_enabled=True):
    path = REPO / "vllm_ascend/expert_offload/expert_offload_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExpertOffloadManager")
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_update_weights", "_record_cache_stats")
    ]
    namespace = {
        "torch_npu": types.SimpleNamespace(npu=types.SimpleNamespace(stream=lambda _: contextlib.nullcontext())),
        "logger": logging.getLogger("plan_test"),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    manager = namespace["ExpertOffloadManager"]()
    policy, _ = load_policy(REPO / "vllm_ascend/expert_offload/lrc_policy.py", backend)
    manager.cache_policy = policy(num_layers=1, num_experts=experts, cache_size=capacity, topk=6, recent_window=7)
    manager.offload_config = types.SimpleNamespace(experts_pruning_enabled=False, cache_stats_log_interval=3)
    manager.load_stream = None
    manager.num_total_experts = experts
    manager.prefetch_topk = 1
    manager._debug = False
    manager._pf_wait_timing = False
    manager._prefetch_state_lock = threading.Lock()
    manager._prefetch_stats_pending = {}
    manager.records = []
    manager.shortfalls = []
    manager._stats = (
        types.SimpleNamespace(
            collecting=True,
            record_layer=lambda *args, **kwargs: manager.records.append((args, kwargs)),
            note_shortfall=lambda: manager.shortfalls.append(1),
        )
        if stats_enabled
        else None
    )
    for name in ("cache_calls", "cache_requests", "cache_hits", "cache_misses"):
        setattr(manager, name, [0])
    manager.last_hit_experts = [[]]
    manager.last_miss_experts = [[]]
    manager.loads = []
    manager._load_expert_weights_into_slots = lambda layer, layer_idx, loads: manager.loads.append(list(loads))
    manager.syncs = []
    manager._synchronize_h2d = lambda: manager.syncs.append(1)
    return manager


@unittest.skipUnless(importlib.util.find_spec("_vllm_ascend_lrc"), "C++ extension not installed")
class TestFusedPlan(unittest.TestCase):
    def compare(self, left, right, mapping, rows, scores, prefetch, exact_slots=True):
        a, b = mapping.copy(), mapping.copy()
        left._update_weights((rows, a, None, 0, scores, prefetch))
        right._update_weights((rows, b, None, 0, scores, prefetch))
        if exact_slots:
            np.testing.assert_array_equal(a, b)
            self.assertEqual(left.loads, right.loads)
        else:
            # Phase-1 observe returns an unordered_set through pybind11: set
            # membership is equal but iteration/physical slot assignment can
            # differ. Phase 2 follows the original Python ordering exactly.
            np.testing.assert_array_equal(a >= 0, b >= 0)
            self.assertEqual(
                [[sorted(eid for _, eid in batch)] for batch in left.loads],
                [[sorted(eid for _, eid in batch)] for batch in right.loads],
            )
        for name in (
            "records",
            "shortfalls",
            "syncs",
            "cache_calls",
            "cache_requests",
            "cache_hits",
            "cache_misses",
            "last_hit_experts",
            "last_miss_experts",
            "_prefetch_stats_pending",
        ):
            self.assertEqual(getattr(left, name), getattr(right, name), name)
        x, y = left.cache_policy.layer_states[0], right.cache_policy.layer_states[0]
        for name in ("step", "freq", "router_score", "last_used", "recent_queue"):
            self.assertEqual(getattr(x, name), getattr(y, name), name)
        np.testing.assert_array_equal(x.ema, y.ema)
        return a

    def test_random_sequence_exact_plans_and_stats(self):
        rng = random.Random(20261010)
        for baseline in ("python", "cpp"):
            for dtype in (np.int32, np.int64):
                left = make_manager(baseline)
                right = make_manager("cpp_fused")
                mapping = np.full(256, -1, dtype=dtype)
                mapping[rng.sample(range(256), 24)] = np.arange(24)
                for i in range(240):
                    rows = np.array([[rng.randrange(256) for _ in range(6)] for _ in range(i % 4 + 1)], dtype=dtype)
                    scores = np.array([[rng.random() for _ in row] for row in rows], dtype=np.float32)
                    if i % 3 == 0:
                        scores = None
                    left.prefetch_topk = right.prefetch_topk = i % 6 + 1
                    with self.subTest(baseline=baseline, dtype=dtype, iteration=i):
                        mapping = self.compare(
                            left, right, mapping, rows, scores, i % 2 == 0, exact_slots=baseline == "python"
                        )

    def test_ties_duplicates_hit_shortfall_and_strides(self):
        for resident_count in (0, 2, 24):
            left, right = make_manager("python"), make_manager("cpp_fused")
            for manager in (left, right):
                manager.cache_policy.seed_layer_hotness(0, {1: 0.9, 2: 0.3})
            mapping = np.full(256, -1, dtype=np.int32)
            mapping[:resident_count] = np.arange(resident_count)
            rows = np.array([[1, 65, 129, 193, 1, 0], [0, 1, 2, 3, 4, 5]], dtype=np.int32)
            scores = np.ones(rows.shape, dtype=np.float32) / 6
            for prefetch in (True, False):
                for r, w in ((rows, scores), (rows[:, ::-1], scores[:, ::-1]), (np.zeros_like(rows), None)):
                    self.compare(left, right, mapping, r, w, prefetch)
            invalid = np.array([[-1, 256, 1, 1, 2, 0]], dtype=np.int32)
            self.compare(left, right, mapping, invalid, None, False)

    def test_validation_before_state_or_mapping_mutation(self):
        manager = make_manager("cpp_fused")
        policy = manager.cache_policy
        mapping = np.full(256, -1, dtype=np.int32)
        mapping[:24] = np.arange(24)
        rows = np.array([[30, 31]], dtype=np.int32)
        with self.assertRaises(TypeError):
            policy.plan_update(0, rows, None, mapping.tolist(), False, 1)
        readonly = mapping.copy()
        readonly.flags.writeable = False
        duplicate = mapping.copy()
        duplicate[1] = duplicate[0]
        for ids, scores, m, prefetch in (
            (rows.astype(np.float32), None, mapping, False),
            (rows, np.ones((1, 3), dtype=np.float32), mapping, False),
            (rows, None, readonly, False),
            (rows, None, duplicate, False),
            (np.array([[-1]], dtype=np.int32), None, mapping, True),
        ):
            before = m.copy()
            with self.assertRaises(ValueError):
                policy.plan_update(0, ids, scores, m, prefetch, 1)
            np.testing.assert_array_equal(m, before)
            self.assertEqual(policy.layer_step(0), 0)

    def test_production_timers_off_and_prefetch_no_observe(self):
        policy = make_manager("cpp_fused").cache_policy
        mapping = np.full(256, -1, dtype=np.int32)
        mapping[:24] = np.arange(24)
        rows = np.array([[30, 31]], dtype=np.int32)
        plan = policy.plan_update(0, rows, None, mapping, False, 1)
        self.assertEqual((plan["observe_ns"], plan["choose_ns"]), (0, 0))
        step = policy.layer_step(0)
        plan = policy.plan_update(0, rows, None, mapping, True, 1, measure=True)
        self.assertEqual(policy.layer_step(0), step)
        self.assertEqual(plan["observe_ns"], 0)
        self.assertGreaterEqual(plan["choose_ns"], 0)

    def test_debug_logging_with_and_without_decode_stats(self):
        for enabled in (True, False):
            left = make_manager("python", stats_enabled=enabled)
            right = make_manager("cpp_fused", stats_enabled=enabled)
            left._debug = right._debug = True
            mapping = np.full(256, -1, dtype=np.int32)
            mapping[:2] = [0, 1]
            rows = np.array([[0, 30, 31, 32, 33, 34]], dtype=np.int32)
            with self.assertLogs("plan_test", level="INFO") as messages:
                self.compare(left, right, mapping, rows, None, False)
            self.assertTrue(any("NO SLOTS" in message for message in messages.output))


if __name__ == "__main__":
    if "--require-cpp" in sys.argv:
        sys.argv.remove("--require-cpp")
        if importlib.util.find_spec("_vllm_ascend_lrc") is None:
            sys.exit("C++ extension missing: rebuild csrc/lrc_policy first.")
    unittest.main()
