# SPDX-License-Identifier: Apache-2.0
"""Resolve two distinct physical cores without changing process/NUMA policy."""

import json
import os
from pathlib import Path


def resolve_host_cpus(value, *, allowed=None, sysfs=Path("/sys/devices/system/cpu")):
    """Return planner/submitter logical CPU IDs, or None for inherited affinity.

    Auto prefers two cores on the same NUMA node within the inherited cpuset.
    It is a deterministic topology choice, not a claim that the CPUs are idle
    or nearest to the NPU. Explicit IDs also must represent different cores.
    """
    if value == "none":
        return None
    if allowed is None:
        if not hasattr(os, "sched_getaffinity"):
            raise ValueError("Host CPU binding requires Linux")
        allowed = os.sched_getaffinity(0)
    allowed = set(allowed)

    def topology(cpu):
        root = sysfs / f"cpu{cpu}"
        try:
            package = int((root / "topology/physical_package_id").read_text())
            core = int((root / "topology/core_id").read_text())
            if min(package, core) < 0:
                raise ValueError("unknown physical core")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot identify physical core for CPU {cpu}: {exc}") from exc
        nodes = sorted(root.glob("node[0-9]*"))
        node = nodes[0].name if len(nodes) == 1 else f"package{package}"
        return (package, core), node

    if value == "auto":
        cores = {}
        for cpu in sorted(allowed):
            identity, node = topology(cpu)
            cores.setdefault(identity, (cpu, node))  # One SMT sibling per core.
        choices = sorted(cores.values(), reverse=True)
        if len(choices) < 2:
            raise ValueError("Host binding needs at least two allowed physical cores")
        pair = next(
            ((a[0], b[0]) for i, a in enumerate(choices) for b in choices[i + 1 :] if a[1] == b[1]),
            (choices[0][0], choices[1][0]),
        )
    else:
        words = value.split(",")
        if len(words) != 2 or any(not word.isdecimal() for word in words):
            raise ValueError("--host-cpus must be none, auto, or PLANNER_CPU,DMA_CPU")
        pair = tuple(map(int, words))
        if not set(pair) <= allowed:
            raise ValueError("Requested Host CPUs are outside inherited allowed set")
        if topology(pair[0])[0] == topology(pair[1])[0]:
            raise ValueError("Planner and DMA CPUs must be on different physical cores, not SMT siblings")
    return pair


def read_host_affinity(log_path, case, cpus):
    """Require actual worker masks, not merely the requested launch config."""
    if cpus is None:
        return None
    marker = "[MOE-NATIVE] HOST_AFFINITY "
    records = [
        json.loads(line.split(marker, 1)[1])
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if marker in line
    ]
    if len(records) != 1:
        raise RuntimeError(f"Expected one verified Host affinity record for {case}; found {len(records)}")
    record = records[0]
    expected = {"planner": cpus[0]}
    if case == "native_pipeline":
        expected["dma"] = cpus[1]
    for role, cpu in expected.items():
        info = record.get(role) or {}
        if info.get("worker_cpu") != cpu or info.get("worker_affinity") != [cpu]:
            raise RuntimeError(f"Actual {role} affinity does not match requested CPU {cpu}")
    return record
