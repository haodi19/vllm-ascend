# SPDX-License-Identifier: Apache-2.0
"""CPU topology and runtime evidence; no Torch or model required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "script"))
from moe_host_affinity import read_host_affinity, resolve_host_cpus  # noqa: E402


class HostAffinityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for cpu, core, node in ((0, 0, 0), (1, 1, 0), (2, 2, 1), (4, 0, 0), (5, 1, 0)):
            topology = self.root / f"cpu{cpu}/topology"
            topology.mkdir(parents=True)
            (topology / "physical_package_id").write_text("0")
            (topology / "core_id").write_text(str(core))
            (topology.parent / f"node{node}").mkdir()
        self.allowed = {0, 1, 2, 4, 5}

    def resolve(self, value, allowed=None):
        return resolve_host_cpus(value, allowed=self.allowed if allowed is None else allowed, sysfs=self.root)

    def test_auto_prefers_same_node_distinct_cores_within_affinity(self):
        self.assertEqual(self.resolve("auto"), (1, 0))
        self.assertEqual(self.resolve("auto", {4, 5}), (5, 4))
        self.assertEqual(self.resolve("auto", {1, 2}), (2, 1))

    def test_reject_smt_same_cpu_outside_cpuset_and_unknown_topology(self):
        for value in ("0,4", "1,1", "0,99", "-1,2", "1", "1,2,4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)
        with self.assertRaisesRegex(ValueError, "two allowed physical"):
            self.resolve("auto", {0, 4})
        with self.assertRaisesRegex(ValueError, "identify physical core"):
            self.resolve("auto", {99})

    def test_none_never_reads_topology(self):
        self.assertIsNone(resolve_host_cpus("none", allowed=set(), sysfs=self.root / "missing"))

    def test_actual_masks_are_required_before_accepting_bound_run(self):
        path = self.root / "inference.log"
        record = dict(
            planner=dict(worker_cpu=1, worker_affinity=[1]),
            dma=dict(worker_cpu=2, worker_affinity=[2]),
            idle_mode="yield",
        )
        marker = "(EngineCore pid=123) [MOE-NATIVE] HOST_AFFINITY "
        path.write_text(marker + json.dumps(record), encoding="utf-8")
        self.assertEqual(read_host_affinity(path, "native_pipeline", (1, 2)), record)
        record["dma"]["worker_affinity"] = [1, 2]
        path.write_text(marker + json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Actual dma affinity"):
            read_host_affinity(path, "native_pipeline", (1, 2))
        path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Expected one verified"):
            read_host_affinity(path, "native_pipeline", (1, 2))
        self.assertIsNone(read_host_affinity(path, "native_pipeline", None))


if __name__ == "__main__":
    unittest.main()
