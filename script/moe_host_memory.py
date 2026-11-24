# SPDX-License-Identifier: Apache-2.0
"""Host/NUMA diagnostics and a fixed page-cache drop after the Python service."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

KIB_PER_GIB = 1024**2
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


def drop_page_cache():
    # Literal command requested for the Python -> native transition; no interpolation.
    subprocess.run(
        ["sh", "-c", "echo 1 > /proc/sys/vm/drop_caches"],
        check=True,
        capture_output=True,
        text=True,
        timeout=RECLAIM_TIMEOUT,
    )


def check_host_memory(stage, report, *, service_pgid=None):
    if service_pgid is not None:
        wait_group_exit(service_pgid, report)
    try:
        return log_memory(stage, report)
    except OSError as exc:
        report(f"HOST_MEMORY_UNAVAILABLE stage={stage}; {exc}")
        return None


def after_service_memory(stage, report, pgid, *, drop_cache=False):
    """Called from finally, after stopping the group; preserve the original error."""
    already_failed = sys.exc_info()[0] is not None
    try:
        before = check_host_memory(stage, report, service_pgid=pgid)
        if drop_cache and not already_failed:
            report(
                f"HOST_CACHE_RECLAIM stage={stage}; fixed Python -> native transition; "
                "command=echo 1 > /proc/sys/vm/drop_caches"
            )
            try:
                drop_page_cache()
            except (OSError, subprocess.SubprocessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                report(f"HOST_CACHE_RECLAIM_FAILED {detail}; next service not started")
                raise RuntimeError("Fixed page-cache drop failed; next service not started") from exc
            report(f"HOST_CACHE_RECLAIM_PASS stage={stage}; exit=0")
            after = check_host_memory(stage + "/after_drop", report)
            if before is not None and after is not None:
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
    except Exception as exc:
        if not already_failed:
            raise
        report(f"HOST_MEMORY_DIAGNOSTIC_FAILED {type(exc).__name__}: {exc}")
