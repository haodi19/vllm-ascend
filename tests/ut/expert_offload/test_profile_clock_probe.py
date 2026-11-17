# SPDX-License-Identifier: Apache-2.0
import contextlib
import itertools
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload/clock_probe.py"))


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def add_(self, value):
        self.value += value

    def cpu(self):
        return self


class ClockProbeTests(unittest.TestCase):
    def test_prepare_before_after_lifecycle_and_numerical_check(self):
        stream = Mock()
        npu = SimpleNamespace(set_device=Mock(), Stream=lambda: stream, stream=lambda stream: contextlib.nullcontext())
        torch = SimpleNamespace(
            npu=npu,
            float32="float32",
            zeros=lambda *a, **k: FakeTensor(0),
            full=lambda shape, value, **k: FakeTensor(value),
            equal=lambda a, b: a.value == b.value,
        )
        mstx = SimpleNamespace(range_start=Mock(return_value=1), range_end=Mock())
        counter = itertools.count()
        with (
            patch.dict(sys.modules, {"torch": torch, "torch_npu.npu": SimpleNamespace(mstx=mstx)}),
            patch("time.sleep"),
            patch("time.clock_gettime_ns", side_effect=lambda clock: next(counter), create=True),
            patch("time.CLOCK_MONOTONIC_RAW", 0, create=True),
        ):
            probe = MODULE["ProfileClockProbe"](0)
            with self.assertRaisesRegex(ValueError, "phase"):
                probe.measure("before")
            probe.prepare()
            with self.assertRaisesRegex(ValueError, "already prepared"):
                probe.prepare()
            before = probe.measure("before")
            after = probe.measure("after")
            self.assertEqual(len(before["samples"]), MODULE["PROBE_SAMPLES"])
            self.assertEqual(len(after["samples"]), 2 * MODULE["PROBE_SAMPLES"])
            self.assertEqual([r["sample"] for r in after["samples"]], list(range(2 * MODULE["PROBE_SAMPLES"])))
            self.assertEqual(mstx.range_start.call_count, 2 * MODULE["PROBE_SAMPLES"])
            with self.assertRaisesRegex(ValueError, "phase"):
                probe.measure("after")
            with self.assertRaisesRegex(ValueError, "phase"):
                probe.measure("unknown")


if __name__ == "__main__":
    unittest.main()
