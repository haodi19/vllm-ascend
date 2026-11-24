# SPDX-License-Identifier: Apache-2.0
"""Host/NUMA diagnostics between services; optional one-shot page-cache reclaim."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

KIB_PER_GIB = 1024**2
# Trigger levels for this benchmark, not driver pinned-allocation guarantees.
DEFAULT_MIN_HOST_FREE_GIB = 160
DEFAULT_MIN_NUMA_FREE_GIB = 160
GROUP_EXIT_TIMEOUT = 15
GROUP_POLL_SECONDS = 0.2
NUMACTL_TIMEOUT = 10
RECLAIM_TIMEOUT = 120
MEMORY_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Cached",
    "FilePages",
    "AnonPages",
    "Shmem",
    "Slab",
    "SReclaimable",
    "SUnreclaim",
    "Mlocked",
    "Unevictable",
    "Dirty",
    "Writeback",
    "AnonHugePages",
    "SwapFree",
)


def memory_fields(text):
    return {
        name: int(value) / KIB_PER_GIB
        for name, value in re.findall(r"(?:^|\s)(\w+):\s+(\d+)\s+kB", text)
        if name in MEMORY_FIELDS
    }


def expand_nodes(value):
    nodes = set()
    for part in value.split(","):
        ends = [int(number) for number in part.split("-")]
        nodes.update(range(ends[0], ends[-1] + 1))
    return nodes


def memory_snapshot(proc_root=Path("/proc"), node_root=Path("/sys/devices/system/node")):
    host = memory_fields((proc_root / "meminfo").read_text())
    status = (proc_root / "self/status").read_text()
    allowed = next(
        (line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("Mems_allowed_list:")), None
    )
    nodes = {}
    for path in sorted(node_root.glob("node[0-9]*/meminfo")):
        nodes[path.parent.name] = memory_fields(path.read_text())
    return dict(unit="GiB", host=host, nodes=nodes, allowed_nodes=sorted(expand_nodes(allowed)) if allowed else None)


def cgroup_snapshot():
    # Read limits/counters for the visible container hierarchy, including parents.
    roots = {Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            base = Path("/sys/fs/cgroup") / ("memory" if controllers else "")
            child = base / relative.lstrip("/")
            if ".." not in child.parts:
                while child != base:
                    roots.add(child)
                    child = child.parent
    result = {}
    for root in sorted(roots):
        fields = {}
        for name in (
            "memory.current",
            "memory.max",
            "memory.high",
            "memory.events",
            "memory.usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.failcnt",
            "memory.oom_control",
        ):
            path = root / name
            if path.is_file():
                fields[name] = path.read_text().strip()
        if fields:
            result[str(root)] = fields
    return result


def log_memory(stage, report):
    try:
        result = subprocess.run(["numactl", "-H"], capture_output=True, text=True, timeout=NUMACTL_TIMEOUT)
        report(f"NUMACTL stage={stage}; exit={result.returncode}\n{(result.stdout + result.stderr).strip()}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        report(f"NUMACTL_UNAVAILABLE stage={stage}; {exc}; using /proc and /sys")
    snapshot = memory_snapshot()
    report("HOST_MEMORY " + json.dumps(dict(stage=stage, **snapshot)))
    try:
        report("HOST_CGROUP " + json.dumps(dict(stage=stage, limits=cgroup_snapshot())))
    except OSError as exc:
        report(f"HOST_CGROUP_UNAVAILABLE {exc}")
    return snapshot


def group_members(pgid, proc_root=Path("/proc")):
    members = []
    for path in proc_root.glob("[0-9]*/stat"):
        try:
            # comm may contain spaces/parentheses. Fields after ')' start at state.
            fields = path.read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            members.append(int(path.parent.name))
    return members


def wait_group_exit(pgid, report):
    deadline = time.monotonic() + GROUP_EXIT_TIMEOUT
    members = group_members(pgid)
    if members:
        report(f"HOST_MEMORY_WAIT pgid={pgid}; remaining_pids={members}; timeout={GROUP_EXIT_TIMEOUT}s")
    while members and time.monotonic() < deadline:
        time.sleep(GROUP_POLL_SECONDS)
        members = group_members(pgid)
    if members:
        raise RuntimeError(f"Service group {pgid} still has live processes {members}; no cache reclaim or next service")
    report(f"HOST_SERVICE_GROUP_EXITED pgid={pgid}")


def reclaim_reasons(snapshot, host_threshold, node_threshold):
    reasons = []
    if snapshot["host"]["MemFree"] < host_threshold:
        reasons.append(f"host MemFree<{host_threshold:g}GiB")
    allowed = snapshot["allowed_nodes"]
    for node, fields in snapshot["nodes"].items():
        if allowed is not None and int(node[4:]) in allowed and fields["MemFree"] < node_threshold:
            reasons.append(f"{node} MemFree<{node_threshold:g}GiB")
    return reasons


def drop_page_cache():
    # Only called by an explicit --auto-drop-caches request, outside measurements.
    # Never write 3, change huge pages, compact memory, or stop unrelated processes.
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import os; from pathlib import Path; os.sync(); Path('/proc/sys/vm/drop_caches').write_text('1\\n')",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=RECLAIM_TIMEOUT,
    )


def check_host_memory(args, stage, report, *, service_pgid=None, before_start=False, reclaim=True):
    if service_pgid is not None:
        wait_group_exit(service_pgid, report)
    try:
        before = log_memory(stage, report)
    except OSError as exc:
        report(f"HOST_MEMORY_UNAVAILABLE stage={stage}; {exc}")
        if args.auto_drop_caches:
            raise RuntimeError("Cannot decide automatic cache reclaim without Host memory readings") from exc
        return
    if not args.auto_drop_caches or not reclaim:
        return before
    reasons = reclaim_reasons(before, args.min_host_free_gib, args.min_numa_free_gib)
    after = before
    if reasons:
        report(f"HOST_CACHE_RECLAIM stage={stage}; reasons={reasons}; sync then drop_caches=1; host-wide file cache")
        try:
            drop_page_cache()
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            report(f"HOST_CACHE_RECLAIM_FAILED {detail}; no automatic retry")
            raise RuntimeError("Page-cache reclaim failed; check container permission or read-only /proc") from exc
        after = log_memory(stage + "/after_drop", report)
        report(
            "HOST_CACHE_DELTA "
            + json.dumps(
                dict(
                    stage=stage,
                    host_free_gib=after["host"]["MemFree"] - before["host"]["MemFree"],
                    node_free_gib={
                        node: fields["MemFree"] - before["nodes"][node]["MemFree"]
                        for node, fields in after["nodes"].items()
                        if node in before["nodes"]
                    },
                )
            )
        )
    remaining = reclaim_reasons(after, args.min_host_free_gib, args.min_numa_free_gib)
    report(
        f"HOST_MEMORY_DECISION stage={stage}; reclaimed={bool(reasons)}; remaining_low_free={remaining}; "
        "node free is a reclaim trigger, not proof of allocation failure or driver NUMA placement"
    )
    if before_start and after["host"]["MemAvailable"] < args.min_host_free_gib:
        raise RuntimeError("Host MemAvailable remains below configured threshold; next service not started")
    return after


def after_service_memory(args, stage, report, pgid):
    """Called from finally, after stopping the group; preserve the original error."""
    already_failed = sys.exc_info()[0] is not None
    try:
        check_host_memory(args, stage, report, service_pgid=pgid, reclaim=not already_failed)
    except Exception as exc:
        if not already_failed:
            raise
        report(f"HOST_MEMORY_DIAGNOSTIC_FAILED {type(exc).__name__}: {exc}")
