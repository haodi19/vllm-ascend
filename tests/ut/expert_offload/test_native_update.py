# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native tests require both optional extensions built."""

import ctypes
import importlib.util
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "native_update_config", ROOT / "vllm_ascend/expert_offload/native_update.py"
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
try:
    import _vllm_ascend_lrc as lrc
    import vllm_ascend_host_sync as hs
except ImportError:
    lrc = hs = None


class LayoutAndConfig(unittest.TestCase):
    def test_default_unchanged(self):
        config.validate_native_update_config({})
        config.validate_native_update_config({"host_update_backend": "python", "moe_offload_debug": True})

    def test_dense_transpose_and_reject_alias(self):
        self.assertEqual(config.dense_span((4, 3), (1, 4)), 12)
        for strides in ((0, 1), (1, 1), (4, 1)):
            with self.assertRaises(ValueError):
                config.dense_span((4, 3), strides)

    def test_unsupported_config(self):
        good = dict(
            host_update_backend="native", host_sync_backend="mte", cache_policy_backend="cpp", cache_policy_enabled=True
        )
        config.validate_native_update_config(good)
        for key, value in (
            ("moe_offload_debug", True),
            ("callback_timing_port", 18155),
            ("enable_multi_card", True),
            ("expert_prefetch_enabled", True),
            ("h2d_backend", "memfabric"),
        ):
            with self.assertRaises(ValueError):
                config.validate_native_update_config(dict(good, **{key: value}))

    def test_packed_bytes_match_transposed_destination(self):
        import torch

        class CpuFactory:
            @staticmethod
            def empty_strided(shape, strides, **kwargs):
                kwargs.pop("pin_memory")
                return torch.empty_strided(shape, strides, **kwargs)

        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        destination = torch.empty_strided((4, 3), (1, 4), dtype=torch.float64)
        packed = config.pack_scale(CpuFactory, source, destination)
        ctypes.memmove(destination.data_ptr(), packed.data_ptr(), packed.numel() * packed.element_size())
        torch.testing.assert_close(destination, source.to(torch.float64))


@unittest.skipUnless(lrc is not None and hs is not None, "build lrc_policy and host_sync extensions")
class NativeExecution(unittest.TestCase):
    def fixture(self):
        self.policy = lrc.Policy(1, 8, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
        self.source = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
        self.destination = np.zeros((4, 16), dtype=np.uint8)
        self.table = lrc.NativeTransfers(
            [[row.ctypes.data] for row in self.source],
            [[row.ctypes.data] for row in self.destination],
            [16],
            0,
            "",
            True,
        )
        self.mapping = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
        self.ids = np.array([[4, 5]], dtype=np.int64)

    def task(self, prefetch=False):
        return lrc.native_update_task(
            self.policy, self.table, 0, self.ids, None, self.mapping, prefetch, 1, (self.source, self.destination)
        )

    def test_replay_reads_new_ids_and_updates_policy(self):
        self.fixture()
        capsule = self.task()
        for ids in ([4, 5], [6, 7], [6, 7], [0, 1]):
            self.ids[0] = ids
            lrc.run_native_update_for_test(capsule)
            for eid in ids:
                self.assertGreaterEqual(self.mapping[eid], 0)
                np.testing.assert_array_equal(self.destination[self.mapping[eid]], self.source[eid])
            self.assertEqual(len(set(self.mapping[self.mapping >= 0])), 4)
        self.assertEqual(self.policy.step(0), 4)

    def test_prefetch_cap_and_no_observation(self):
        self.fixture()
        lrc.run_native_update_for_test(self.task(True))
        self.assertEqual(sum(self.mapping[4:] >= 0), 1)
        self.assertEqual(self.policy.step(0), 0)

    def test_random_replay_matches_existing_planner_residency_and_lrc(self):
        self.fixture()
        self.destination[:] = self.source[:4]
        reference = lrc.Policy(1, 8, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)
        reference_map = self.mapping.copy()
        scores = np.zeros((1, 2), dtype=np.float32)
        tasks = [
            lrc.native_update_task(
                self.policy, self.table, 0, self.ids, scores, self.mapping, prefetch, 1, (self.source, self.destination)
            )
            for prefetch in (False, True)
        ]
        random = np.random.default_rng(123)
        for step in range(200):
            self.ids[:] = random.integers(0, 8, size=(1, 2))
            scores[:] = random.random((1, 2))
            prefetch = bool(step % 2)
            reference.plan(0, self.ids, scores, reference_map, prefetch, 1, False)
            lrc.run_native_update_for_test(tasks[int(prefetch)])
            np.testing.assert_array_equal(self.mapping >= 0, reference_map >= 0)
            self.assertEqual(self.policy.snapshot(0), reference.snapshot(0))
            for eid in np.flatnonzero(self.mapping >= 0):
                np.testing.assert_array_equal(self.destination[self.mapping[eid]], self.source[eid])

    def test_worker_failure_does_not_publish_success(self):
        self.fixture()
        self.mapping[0] = 100
        with hs.Session(slots=1, backend="cpu") as session:
            session.register_task(0, lambda: self.fail("unexpected Python fallback"))
            session.install_native_task(0, self.task())
            session.start()
            session.submit_for_test(0, 1)
            with self.assertRaises(RuntimeError):
                session.wait(0, 1)
            state = session.snapshot(0)
            self.assertEqual(state["done_generation"], 0)
            self.assertEqual(state["error_generation"], 1)

    def test_invalid_mapping_not_committed(self):
        self.fixture()
        self.mapping[0] = 9
        before = self.mapping.copy()
        with self.assertRaises(RuntimeError):
            lrc.run_native_update_for_test(self.task())
        np.testing.assert_array_equal(self.mapping, before)
        self.assertFalse(self.destination.any())

    def test_shortfall_no_partial_copy(self):
        self.fixture()
        self.ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        before = self.mapping.copy()
        with self.assertRaises(RuntimeError):
            lrc.run_native_update_for_test(self.task())
        np.testing.assert_array_equal(self.mapping, before)
        self.assertFalse(self.destination.any())

    def test_worker_completes_while_python_holds_gil(self):
        self.fixture()
        with hs.Session(slots=1, backend="cpu") as session:
            session.register_task(0, lambda: self.fail("native task called Python"))
            session.install_native_task(0, self.task())
            session.start()
            # Directly inspect completion memory without Session.wait (which releases GIL).
            info = session.info()
            done = ctypes.c_uint64.from_address(info["host_address"] + info["done_offset"])
            old_interval = sys.getswitchinterval()
            try:
                sys.setswitchinterval(1.0)
                session.submit_for_test(0, 1)
                deadline = time.perf_counter() + 0.5
                while not done.value and time.perf_counter() < deadline:
                    pass
                self.assertEqual(done.value, 1)
            finally:
                sys.setswitchinterval(old_interval)
            with self.assertRaises(RuntimeError):
                session.install_native_task(0, self.task())


if __name__ == "__main__":
    unittest.main()
