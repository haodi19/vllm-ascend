# SPDX-License-Identifier: Apache-2.0
"""Read existing Host/device timelines without assuming their clock alignment."""

import argparse
import bisect
import collections
import json
import re
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from inspect_moe_profile import _interval, _stats

MAX_TRACES = 16
MAX_EXAMPLES = 3
MAX_NEIGHBORS = 2
PROGRESS_EVENTS = 500000
CONTEXT_FIELDS = ("backend", "layer", "kind", "rows", "call")
HANDSHAKE = re.compile(r"^host_sync_(publish|wait)(?:_\d+)?$")
IDENTITY_FIELDS = frozenset(("streamid", "taskid", "connectionid", "correlationid", "modelid"))


def identity_fields(args):
    return {
        key: value
        for key, value in args.items()
        if re.sub(r"[^a-z0-9]", "", key.lower()) in IDENTITY_FIELDS and isinstance(value, (str, int, float))
    }


def record(event, index):
    start, end = _interval(event)
    return dict(
        index=index,
        name=str(event.get("name", "")),
        pid=str(event.get("pid")),
        tid=str(event.get("tid")),
        start=start,
        end=end,
        duration=end - start,
        ids=identity_fields(event.get("args", {})),
    )


def host_key(event):
    context = fields(event)
    return (str(event.get("pid")), str(event.get("tid")), *(context.get(key) for key in CONTEXT_FIELDS))


def read_trace(path):
    """Keep only protocol tasks and Host boundaries; validate the complete JSON."""
    processes, threads = {}, {}
    candidates, hosts, syncs = [], collections.defaultdict(list), collections.defaultdict(list)
    phases = collections.Counter()
    retained = 0
    for index, event in enumerate(events(path)):
        if index and index % PROGRESS_EVENTS == 0:
            print(f"READ=1/2; scanned_events={index}", flush=True)
        name = str(event.get("name", ""))
        pid, tid = str(event.get("pid")), str(event.get("tid"))
        if event.get("ph") == "M":
            target = processes if name == "process_name" else threads if name == "thread_name" else None
            if target is not None:
                key = pid if name == "process_name" else (pid, tid)
                value = str(event.get("args", {}).get("name", ""))
                if key in target and target[key] != value:
                    raise ValueError(f"conflicting metadata: {key}")
                target[key] = value
        stage = name.split("|", 1)[0]
        if stage in ("moe_offload/host_callback", "moe_offload/synchronize"):
            phases[f"{stage}:{event.get('ph')}"] += 1
            if event.get("ph") != "X":
                continue
            rec = record(event, index)
            rec["context"] = fields(event)
            (hosts if stage.endswith("host_callback") else syncs)[host_key(event)].append(rec)
            retained += 1
        elif HANDSHAKE.fullmatch(name) and event.get("ph") == "X":
            candidates.append(record(event, index))
            retained += 1
        if retained + len(processes) + len(threads) > MAX_INTERVALS:
            raise ValueError("retained events exceed analysis limit; no partial statistics")
    hardware = {pid for pid, name in processes.items() if name.strip().lower() == "ascend hardware"}
    tasks = collections.defaultdict(list)
    excluded = 0
    for task in candidates:
        if task["pid"] in hardware:
            tasks[task["pid"], task["tid"]].append(task)
        else:
            excluded += 1
    groups, issues = collections.defaultdict(list), collections.Counter()
    invalid_kinds = set()
    for key, entries in hosts.items():
        kind = key[4]
        if None in key[2:] or kind not in ("prefetch", "reactive"):
            issues["missing_or_invalid_host_context"] += len(entries)
            invalid_kinds.update(("prefetch", "reactive"))
            continue
        corresponding = syncs.get(key, [])
        if len(entries) != 1 or len(corresponding) != 1:
            issues["duplicate_host_or_missing_duplicate_sync"] += len(entries)
            invalid_kinds.add(kind)
            continue
        host, sync = entries[0], corresponding[0]
        if not host["start"] <= sync["start"] <= sync["end"] <= host["end"]:
            issues["sync_outside_matching_host"] += 1
            invalid_kinds.add(kind)
            continue
        host["sync"] = sync
        groups[kind].append(host)
    for key, entries in syncs.items():
        if key not in hosts:
            issues["orphan_sync"] += len(entries)
            if key[4] in ("prefetch", "reactive"):
                invalid_kinds.add(key[4])
            else:
                invalid_kinds.update(("prefetch", "reactive"))
    if any(not name.endswith(":X") for name in phases):
        issues["unsupported_host_marker_phase"] += 1
        invalid_kinds.update(("prefetch", "reactive"))
    all_hosts = sorted((entry for entries in hosts.values() for entry in entries), key=lambda item: item["start"])
    if len({(host["pid"], host["tid"]) for host in all_hosts}) > 1:
        issues["multiple_host_actors_for_single_worker"] += 1
        invalid_kinds.update(("prefetch", "reactive"))
    for kind in groups:
        groups[kind].sort(key=lambda item: (item["start"], item["index"]))
        actors = {(item["pid"], item["tid"]) for item in groups[kind]}
        if len(actors) != 1:
            issues[f"multiple_{kind}_host_threads"] += 1
            invalid_kinds.add(kind)
    for left, right in zip(all_hosts, all_hosts[1:]):
        if left["end"] > right["start"]:
            issues["overlapping_host_callbacks"] += 1
            invalid_kinds.update(("prefetch", "reactive"))
    for task_list in tasks.values():
        task_list.sort(key=lambda item: (item["start"], item["index"]))
    return dict(
        tasks=tasks,
        groups=groups,
        issues=dict(issues),
        invalid_kinds=invalid_kinds,
        threads=threads,
        hardware=sorted(hardware),
        excluded=excluded,
        host_count=sum(map(len, hosts.values())),
        phases=dict(phases),
        all_hosts=all_hosts,
    )


def task_pairs(tasks):
    """A stream must alternate publish/wait, without overlapping executions."""
    if len(tasks) % 2:
        return [], "odd_handshake_count"
    pairs = []
    for index in range(0, len(tasks), 2):
        publish, wait = tasks[index : index + 2]
        if not publish["name"].startswith("host_sync_publish") or not wait["name"].startswith("host_sync_wait"):
            return [], "not_alternating_publish_wait"
        if publish["end"] > wait["start"]:
            return [], "publish_wait_overlap_on_same_stream"
        if index and tasks[index - 1]["end"] > publish["start"]:
            return [], "successive_tasks_overlap_on_same_stream"
        pairs.append((publish, wait))
    return pairs, "ok" if pairs else "empty"


def candidate(pairs, hosts, valid=True):
    """Ordinal hypotheses only: equal counts do not establish task identity."""
    if not valid:
        return dict(status="incomplete_or_ambiguous_host_markers", pairs=0)
    if not pairs or len(pairs) != len(hosts):
        return dict(status="count_mismatch", device_pairs=len(pairs), host_callbacks=len(hosts))
    pub_to_host, sync_to_wait, host_to_wait, examples = [], [], [], []
    lowers, uppers = [], []
    previous_prefetch_deltas = []
    for (pub, wait), host in zip(pairs, hosts):
        sync = host["sync"]
        a, b, c = host["start"] - pub["start"], wait["end"] - sync["end"], wait["end"] - host["end"]
        pub_to_host.append(a)
        sync_to_wait.append(b)
        host_to_wait.append(c)
        lowers.append(-c)
        uppers.append(a)
        if "previous_prefetch_end" in host:
            previous_prefetch_deltas.append(pub["start"] - host["previous_prefetch_end"])
        if (a < 0 or b < 0 or c < 0) and len(examples) < MAX_EXAMPLES:
            examples.append(
                dict(
                    context=host["context"],
                    host_pid=host["pid"],
                    host_tid=host["tid"],
                    host_start=host["start"],
                    host_end=host["end"],
                    sync_start=sync["start"],
                    sync_end=sync["end"],
                    publish_start=pub["start"],
                    publish_end=pub["end"],
                    wait_start=wait["start"],
                    wait_end=wait["end"],
                    publish_ids=pub["ids"],
                    wait_ids=wait["ids"],
                    wait_end_minus_sync_end_us=b,
                    wait_end_minus_host_end_us=c,
                )
            )
    lower, upper = max(lowers), min(uppers)
    return dict(
        status="ordinal_candidate_not_identity_proof",
        pairs=len(pairs),
        host_start_minus_publish_start_us=_stats(pub_to_host),
        wait_end_minus_sync_end_us=_stats(sync_to_wait),
        wait_end_minus_host_end_us=_stats(host_to_wait),
        host_before_publish=sum(x < 0 for x in pub_to_host),
        wait_end_before_sync_end=sum(x < 0 for x in sync_to_wait),
        wait_end_before_host_end=sum(x < 0 for x in host_to_wait),
        reactive_publish_minus_previous_prefetch_host_end_us=_stats(previous_prefetch_deltas),
        reactive_previous_same_layer_prefetch_not_available=(
            len(hosts) - len(previous_prefetch_deltas) if hosts[0]["context"]["kind"] == "reactive" else None
        ),
        constant_device_shift_necessary_bounds_us=[lower, upper],
        constant_shift_feasible=lower <= upper,
        examples=examples,
    )


def containment(waits, hosts):
    """Compare raw exported coordinates; never interpret overlap as a pairing."""
    syncs = sorted((item["sync"] for item in hosts), key=lambda item: item["start"])
    starts = [item["start"] for item in syncs]
    maximum_ends = []
    for item in syncs:
        maximum_ends.append(max(item["end"], maximum_ends[-1] if maximum_ends else item["end"]))
    full, start_inside, end_inside = 0, 0, 0
    for wait in waits:
        index = bisect.bisect_right(starts, wait["end"]) - 1
        if index >= 0 and wait["end"] <= maximum_ends[index]:
            end_inside += 1
        index = bisect.bisect_right(starts, wait["start"]) - 1
        if index >= 0 and wait["start"] <= maximum_ends[index]:
            start_inside += 1
            full += wait["end"] <= maximum_ends[index]
    return dict(wait_fully_inside_sync=full, wait_start_inside_sync=start_inside, wait_end_inside_sync=end_inside)


def spanning_callbacks(waits, hosts):
    starts = [item["start"] for item in hosts]
    counts, examples = [], []
    for wait in waits:
        left, right = bisect.bisect_left(starts, wait["start"]), bisect.bisect_right(starts, wait["end"])
        within = [host for host in hosts[left:right] if host["end"] <= wait["end"]]
        counts.append(len(within))
        if len(within) >= 2 and len(examples) < MAX_EXAMPLES:
            examples.append(
                dict(
                    wait_start=wait["start"],
                    wait_end=wait["end"],
                    count=len(within),
                    first_context=within[0]["context"],
                    last_context=within[-1]["context"],
                )
            )
    return dict(
        waits_covering_two_or_more_host_callbacks=sum(n >= 2 for n in counts),
        maximum_callbacks=max(counts, default=0),
        examples=examples,
    )


def neighbors(path, tasks):
    """Nearest nonoverlapping time intervals, independent of JSON event order."""
    found = {}
    for key, records in tasks.items():
        waits = [item for item in records if item["name"].startswith("host_sync_wait")][:MAX_NEIGHBORS]
        found[key] = [dict(before=[], wait=wait, after=[], overlapping_task_count=0) for wait in waits]
    for index, event in enumerate(events(path)):
        if index and index % PROGRESS_EVENTS == 0:
            print(f"READ=2/2; scanned_events={index}", flush=True)
        key = (str(event.get("pid")), str(event.get("tid")))
        if key not in found or event.get("ph") != "X":
            continue
        rec = record(event, index)
        brief = dict(name=rec["name"], start=rec["start"], end=rec["end"], ids=rec["ids"])
        for sample in found[key]:
            wait = sample["wait"]
            if index == wait["index"]:
                continue
            if rec["end"] <= wait["start"]:
                sample["before"].append(brief)
                sample["before"] = sorted(sample["before"], key=lambda r: (r["end"], r["start"]), reverse=True)[
                    :MAX_NEIGHBORS
                ]
            elif rec["start"] >= wait["end"]:
                sample["after"].append(brief)
                sample["after"] = sorted(sample["after"], key=lambda r: (r["start"], r["end"]))[:MAX_NEIGHBORS]
            else:
                sample["overlapping_task_count"] += 1
    for samples in found.values():
        for sample in samples:
            sample["before"].sort(key=lambda r: (r["start"], r["end"]))
            sample["wait"] = {key: sample["wait"][key] for key in ("name", "start", "end", "ids")}
            sample["basis"] = "nearest nonoverlapping exported device intervals, not causal neighbors"
    return found


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def analyze(path):
    print(f"TRACE={path}; READ=1/2", flush=True)
    data = read_trace(path)
    emit(
        "HOST",
        dict(
            count=data["host_count"],
            hardware_pids=data["hardware"],
            same_named_nonhardware_excluded=data["excluded"],
            issues=data["issues"],
            groups={
                kind: dict(count=len(rows), actors=sorted({(r["pid"], r["tid"]) for r in rows}))
                for kind, rows in data["groups"].items()
            },
        ),
    )
    if not data["tasks"]:
        print("SKIP: no hardware host_sync tasks; no MTE comparison", flush=True)
        return
    previous = None
    for host in data["all_hosts"]:
        if host["context"].get("kind") == "reactive" and previous is not None:
            if (
                previous["context"].get("kind") == "prefetch"
                and (previous["pid"], previous["tid"]) == (host["pid"], host["tid"])
                and previous["context"].get("layer") == host["context"].get("layer")
            ):
                host["previous_prefetch_end"] = previous["end"]
        previous = host
    print("READ=2/2; collecting device neighbors, not assuming stream numbers", flush=True)
    nearby = neighbors(path, data["tasks"])
    device_bounds = collections.defaultdict(list)
    for key, tasks in sorted(data["tasks"].items()):
        waits = [item for item in tasks if item["name"].startswith("host_sync_wait")]
        pairs, reason = task_pairs(tasks)
        emit(
            "DEVICE_STREAM",
            dict(
                pid=key[0],
                tid=key[1],
                name=data["threads"].get(key),
                publishes=len(tasks) - len(waits),
                waits=len(waits),
                wait_duration_us=_stats([w["duration"] for w in waits]),
                order=reason,
            ),
        )
        emit("RAW_WAIT_SPANS_NOT_PAIRING", dict(stream=key, **spanning_callbacks(waits, data["all_hosts"])))
        for item in nearby[key]:
            emit("DEVICE_NEIGHBORS", item)
        usable = []
        for kind in ("prefetch", "reactive"):
            hosts = data["groups"].get(kind, [])
            emit("RAW_OVERLAP_NOT_PAIRING", dict(stream=key, kind=kind, **containment(waits, hosts)))
            result = (
                candidate(pairs, hosts, kind not in data["invalid_kinds"]) if reason == "ok" else dict(status=reason)
            )
            examples = result.pop("examples", [])
            emit("ORDINAL_CANDIDATE", dict(stream=key, kind=kind, **result))
            for example in examples:
                emit("CONTRADICTION_EXAMPLE", dict(stream=key, kind=kind, **example))
            if result["status"] == "ordinal_candidate_not_identity_proof":
                usable.append((kind, result))
        if len(usable) == 1:
            kind, result = usable[0]
            lower, upper = result["constant_device_shift_necessary_bounds_us"]
            device_bounds[key[0]].append((key, kind, lower, upper))
    for pid, bounds in device_bounds.items():
        lower, upper = max(row[2] for row in bounds), min(row[3] for row in bounds)
        emit(
            "CROSS_STREAM_SHIFT_HYPOTHESIS",
            dict(
                hardware_pid=pid,
                candidate_stream_kinds=[(row[0], row[1]) for row in bounds],
                necessary_bounds_us=[lower, upper],
                feasible=lower <= upper,
                prerequisite="pure MTE strict successful path required; kernel name alone is insufficient",
                meaning="add this constant to DEVICE timestamps only; diagnostic bounds, not calibration",
            ),
        )
    print(
        "解释：ordinal 是顺序候选，未证明 slot/generation 配对；"
        "shift 只检查常量偏移能否满足必要因果约束，不校正原数据、不证明时钟故障。",
        flush=True,
    )


def discover(paths):
    found = set()
    for arg in paths:
        path = Path(arg)
        if not path.exists():
            raise ValueError(f"path does not exist: {path}")
        candidates = (path,) if path.is_file() else path.rglob("trace_view.json")
        for item in candidates:
            if item.is_file():
                found.add(item.resolve())
                if len(found) > MAX_TRACES:
                    raise ValueError(f"more than {MAX_TRACES} traces; choose narrower run directories")
    if not found:
        raise ValueError("no trace_view.json found")
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="existing trace_view.json files or run/profile directories")
    args = parser.parse_args()
    print("只读已有 trace；不导入 Torch、不启动模型/NPU、不写文件。全部数值为原导出坐标，单位 us。", flush=True)
    print("因果约束以本轮配置确认为纯 MTE 严格等待且成功完成为前提；同名算子不能排除有界诊断变体。", flush=True)
    try:
        paths = discover(args.paths)
        for path in paths:
            analyze(path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}; incomplete input is not evidence of protocol success", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
