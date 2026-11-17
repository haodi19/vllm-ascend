# SPDX-License-Identifier: Apache-2.0
"""Explicit profiling-only clock probes, executed outside inference requests."""

import time

PROBE_SAMPLES = 24
PROBE_INTERVAL_SECONDS = 0.002
PROBE_ELEMENTS = 256
PROBE_WARMUP = 10
MARKER_PREFIX = "moe_clock_calibration|sample="


class ProfileClockProbe:
    def __init__(self, device):
        self.device = device
        self.stream = None
        self.tensor = None
        self.samples = []
        self.phase = "new"

    def prepare(self):
        if self.phase != "new":
            raise ValueError("clock probe already prepared")
        # Import only in the explicitly requested control operation.
        import torch

        torch.npu.set_device(self.device)
        self.stream = torch.npu.Stream()
        with torch.npu.stream(self.stream):
            self.tensor = torch.zeros(PROBE_ELEMENTS, device=f"npu:{self.device}", dtype=torch.float32)
            for _ in range(PROBE_WARMUP):
                self.tensor.add_(1.0)
        self.stream.synchronize()
        self.phase = "prepared"
        return {"prepared": True}

    def measure(self, phase):
        expected = {"before": "prepared", "after": "before"}
        if phase not in expected or self.phase != expected[phase]:
            raise ValueError("clock probe phase out of order")
        import torch
        from torch_npu.npu import mstx

        torch.npu.set_device(self.device)
        self.phase = "running"
        for _ in range(PROBE_SAMPLES):
            row = {"sample": len(self.samples), "phase": phase}
            row["start_before_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            marker = mstx.range_start(f"{MARKER_PREFIX}{row['sample']}")
            row["start_after_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            with torch.npu.stream(self.stream):
                row["submit_before_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                self.tensor.add_(1.0)
            self.stream.synchronize()
            row["sync_after_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            row["end_before_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            mstx.range_end(marker)
            row["end_after_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            self.samples.append(row)
            time.sleep(PROBE_INTERVAL_SECONDS)
        if not torch.equal(
            self.tensor.cpu(), torch.full((PROBE_ELEMENTS,), PROBE_WARMUP + len(self.samples), dtype=torch.float32)
        ):
            raise ValueError("clock probe Add result mismatch")
        self.phase = phase
        return {"phase": phase, "samples": list(self.samples), "device": self.device}
