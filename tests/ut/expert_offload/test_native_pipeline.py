# SPDX-License-Identifier: Apache-2.0
"""Real C++ queue/ownership/order tests with CPU byte transport; no NPU."""

import ctypes
import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

try:
    import _vllm_ascend_lrc as native
except ImportError:
    native = None

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipIf(native is None or not hasattr(native, "NativePipeline"), "build current lrc_policy first")
class PipelineCPU(unittest.TestCase):
    def setUp(self):
        self.engine = native.NativePipeline("", True)
        self.owners = []
        self.policy = native.Policy(2, 4, 32, 0.9, 1.0, 0.5, 0.3, 0.01, True)

    def tearDown(self):
        self.engine.pause_for_test(False)
        self.engine.close()
        self.owners.clear()

    @unittest.skipUnless(sys.platform == "linux", "Linux thread affinity")
    def test_bound_submitter_moves_bytes_without_changing_caller_affinity(self):
        allowed = os.sched_getaffinity(0)
        cpu = max(allowed)
        self.engine.close()
        self.engine = native.NativePipeline("", True, cpu)
        self.assertEqual(self.engine.worker_info()["worker_affinity"], [cpu])
        self.assertEqual(os.sched_getaffinity(0), allowed)
        item = self.channel(0)
        self.submit(item)
        self.wait(item)
        self.assertEqual(os.sched_getaffinity(0), allowed)

    def test_invalid_cpu_rejected_before_worker_starts(self):
        with self.assertRaises(ValueError):
            native.NativePipeline("", True, -2)
        if sys.platform != "linux":
            with self.assertRaisesRegex(ValueError, "requires Linux"):
                native.NativePipeline("", True, 0)

    @unittest.skipUnless(sys.platform == "linux", "Linux thread affinity")
    def test_disallowed_cpu_is_reported_at_construction_without_abort(self):
        excluded = next((c for c in range(1024) if c not in os.sched_getaffinity(0)), None)
        if excluded is None:
            self.skipTest("all supported CPUs allowed")
        with self.assertRaisesRegex(RuntimeError, "outside inherited allowed set"):
            native.NativePipeline("", True, excluded)

    def channel(self, layer, prefetch=False, dependency=-1):
        source = np.arange(128, dtype=np.uint8).reshape(4, 32)
        target = source[:2].copy()
        mapping = np.array([0, 1, -1, -1], dtype=np.int32)
        device_map = mapping.copy()
        ids = np.array([[2]], dtype=np.int64)
        request, ready, completed = [np.zeros(4, dtype=np.uint64) for _ in range(3)]
        table = native.NativeTransfers(
            [[a.ctypes.data] for a in source], [[a.ctypes.data] for a in target], [32], 0, "", True
        )
        old = native.native_update_task(self.policy, table, layer, ids, None, mapping, prefetch, 1, (source, target))
        capsule, index = native.native_pipeline_task(
            old,
            self.engine,
            request.ctypes.data,
            ready.ctypes.data,
            completed.ctypes.data,
            device_map.ctypes.data,
            dependency,
        )
        item = SimpleNamespace(
            source=source,
            target=target,
            mapping=mapping,
            device_map=device_map,
            ids=ids,
            request=request,
            ready=ready,
            completed=completed,
            table=table,
            old=old,
            capsule=capsule,
            index=index,
        )
        self.owners.append(item)
        return item

    def submit(self, item):
        item.request[0] += np.uint64(1)
        native.run_native_update_for_test(item.capsule)

    def wait(self, item):
        deadline = time.monotonic() + 3
        while item.completed[0] != item.request[0] and time.monotonic() < deadline:
            time.sleep(0.001)
        self.engine.check_progress(3)
        self.assertEqual(item.completed[0], item.request[0])
        self.assertEqual(item.ready[0], item.request[0])
        np.testing.assert_array_equal(item.mapping, item.device_map)
        for eid, slot in enumerate(item.device_map):
            if slot >= 0:
                np.testing.assert_array_equal(item.target[slot], item.source[eid])

    def test_admission_does_not_release_consumer_and_no_gil_needed(self):
        item = self.channel(0)
        self.engine.pause_for_test(True)
        self.submit(item)
        self.assertEqual(item.ready[0], 0)
        self.assertEqual(item.device_map[2], -1)
        self.engine.pause_for_test(False)
        previous = sys.getswitchinterval()
        try:
            sys.setswitchinterval(10)
            done = ctypes.c_uint64.from_address(item.completed.ctypes.data)
            until = time.monotonic() + 3
            while done.value != 1 and time.monotonic() < until:
                pass
            self.assertEqual(done.value, 1)
        finally:
            sys.setswitchinterval(previous)
        self.wait(item)

    def test_dependency_and_zero_load_jobs_never_overtake(self):
        demand = self.channel(0)
        prefetch = self.channel(1, True, demand.index)
        prefetch.ids[0, 0] = 0  # Zero loads must still respect demand admission.
        self.submit(prefetch)
        time.sleep(0.02)
        self.assertEqual(prefetch.ready[0], 0)
        self.assertEqual(self.engine.submitted_count(), 0)
        self.submit(demand)
        self.wait(demand)
        self.wait(prefetch)
        self.assertEqual(self.engine.submitted_count(), 2)

    def test_multiple_replays_and_mapping_matches_completed_bytes(self):
        demand = self.channel(0)
        prefetch = self.channel(1, True, demand.index)
        for eid in (2, 3, 3, 0, 1, 2):
            for item in (demand, prefetch):
                item.mapping[:] = item.device_map
                item.ids[0, 0] = eid
                self.submit(item)
            self.wait(demand)
            self.wait(prefetch)

    def test_duplicate_generation_is_rejected(self):
        item = self.channel(0)
        self.submit(item)
        self.wait(item)
        with self.assertRaisesRegex(RuntimeError, "sequence"):
            native.run_native_update_for_test(item.capsule)

    def test_invalid_ids_do_not_enqueue_or_release_consumer(self):
        item = self.channel(1, True)
        item.ids[0, 0] = 999
        with self.assertRaisesRegex(RuntimeError, "invalid prefetch expert"):
            self.submit(item)
        self.assertEqual(self.engine.submitted_count(), 0)
        self.assertEqual(item.ready[0], 0)

    def test_dependency_must_be_an_existing_demand(self):
        with self.assertRaisesRegex(ValueError, "earlier registered demand"):
            self.channel(1, True, 0)
        demand = self.channel(0)
        prefetch = self.channel(1, True, demand.index)
        with self.assertRaisesRegex(ValueError, "connect demand to prefetch"):
            self.channel(1, True, prefetch.index)


class EarlyPrediction(unittest.TestCase):
    def test_predict_publish_before_demand_wait(self):
        spec = importlib.util.spec_from_file_location(
            "pipeline", ROOT / "vllm_ascend/expert_offload/native_pipeline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        order = []
        main = SimpleNamespace(record_event=lambda: "ready")
        context = Mock(__enter__=Mock(), __exit__=Mock(return_value=False))
        manager = SimpleNamespace(
            offload_config=SimpleNamespace(cache_router_weight=0, expert_prefetch_enabled=True),
            _skip_prefill=False,
            moe_layers=[None, SimpleNamespace(log2phy="map")],
            _prefetch_stream=SimpleNamespace(wait_event=lambda event: order.append("inputs_ready")),
        )
        manager.predict_next_layer_experts_npu = lambda *args: order.append("predict") or ("scores", "ids")
        dispatcher = SimpleNamespace(
            publish_pipeline_demand=lambda *args: order.append("demand_publish") or (4,),
            wait_pipeline_demand=lambda *args: order.append("demand_wait"),
            defer_prefetch=lambda *args, **kwargs: order.append(("prefetch_publish", kwargs["dependency"])),
            pipeline_indices={4: 9},
            join_events=[],
        )
        manager._host_dispatcher = dispatcher
        runtime = SimpleNamespace(
            manager=manager,
            torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: main, stream=lambda stream: context)),
        )
        module.stage_route = lambda *args: ("args", "mapping")
        module.update_and_prefetch(runtime, 0, "ids", None, "hidden")
        self.assertEqual(order, ["demand_publish", "predict", "inputs_ready", ("prefetch_publish", 9), "demand_wait"])


if __name__ == "__main__":
    unittest.main()
