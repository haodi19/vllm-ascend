# SPDX-License-Identifier: Apache-2.0
"""Identify native callback device gates without applying MTE publish/wait rules."""

import collections
import json
import re

from analyze_callback_gaps import MAX_INTERVALS, events
from diagnose_mte_wait_timeline import record

NATIVE_GATE = re.compile(r"(?:DAVID_)?(?:HOST_FUNC|HOST_CPU|HOST_CALLBACK|HOSTFUNC_CALLBACK)(?:_\d+)?", re.IGNORECASE)
NOTIFY_GATE = re.compile(r"(?:DAVID_)?NOTIFY_WAIT(?:_\d+)?", re.IGNORECASE)
ORDER_US = 1.0
MAX_CHECK_EXAMPLES = 2


def gate_check(tasks, hosts):
    """Keep failed ordinal matches inspectable without guessing another pairing."""
    result = dict(device_count=len(tasks), host_count=len(hosts))
    if not hosts or len(tasks) != len(hosts):
        return dict(result, reason="count_mismatch")
    if any(left["end"] > right["start"] + ORDER_US for left, right in zip(tasks, tasks[1:])):
        return dict(result, reason="overlapping_device_gates")
    failures = []
    failed_count = 0
    for index, (task, host) in enumerate(zip(tasks, hosts)):
        start_delta = task["start"] - host["start"]
        end_delta = task["end"] - host["end"]
        if start_delta > ORDER_US or end_delta < -ORDER_US:
            failed_count += 1
            if len(failures) < MAX_CHECK_EXAMPLES:
                failures.append(
                    dict(
                        ordinal=index,
                        name=task["name"],
                        device_start_minus_host_start_us=round(start_delta, 3),
                        device_end_minus_host_end_us=round(end_delta, 3),
                    )
                )
    return dict(
        result,
        reason="host_not_contained" if failed_count else "ok",
        failed_count=failed_count,
        examples=failures,
    )


def native_gate(event):
    return task_matches(event, NATIVE_GATE)


def task_matches(event, pattern):
    args = event.get("args") or {}
    names = [str(event.get("name", ""))]
    if isinstance(args, dict):
        names.extend(str(args[key]) for key in ("Task Type", "task_type") if key in args)
    return any(pattern.fullmatch(name) for name in names)


def callback_pairs(tasks):
    """Use adjacent launch/notify tasks, never a nearest-time notify match.

    Tasks include every device interval on this stream, in execution order.
    Some exports instead have one native task spanning the entire callback;
    that layout still requires full Host containment in the caller.
    """
    launches = [i for i, task in enumerate(tasks) if task["native"]]
    waits = {i for i, task in enumerate(tasks) if task["notify"]}
    if not waits:
        return [(tasks[i], tasks[i]) for i in launches], "single_native_task"
    pairs, used = [], set()
    for i in launches:
        if i + 1 not in waits:
            return [], "native_task_not_followed_by_notify_wait"
        launch, wait = tasks[i], tasks[i + 1]
        if launch["end"] > wait["start"] + ORDER_US:
            return [], "overlapping_launch_and_notify_wait"
        if i + 2 < len(tasks) and wait["end"] > tasks[i + 2]["start"] + ORDER_US:
            return [], "notify_wait_overlaps_following_task"
        pairs.append((launch, wait))
        used.add(i + 1)
    if used != waits:
        return [], "unpaired_notify_wait"
    return pairs, "native_task_then_notify_wait"


def bind_callback_streams(path, data):
    # Native callbacks may use one report thread per stream. Same-kind multi-actor,
    # missing markers and overlapping callbacks are still rejected.
    issues = set(data["issues"]) - {"multiple_host_actors_for_single_worker"}
    if issues or (data["invalid_kinds"] and not data["issues"]):
        raise ValueError(f"invalid callback Host markers: {data['issues']}")
    if data["tasks"]:
        raise ValueError("callback analysis received an MTE handshake trace")
    gates = collections.defaultdict(list)
    names = collections.Counter()
    retained = 0
    for index, event in enumerate(events(path)):
        if event.get("ph") != "X" or str(event.get("pid")) not in data["hardware"]:
            continue
        name = str(event.get("name", ""))
        if any(word in name.upper() for word in ("HOST", "NOTIFY", "EVENT", "CALLBACK")):
            names[str(event.get("tid")), name] += 1
        if native_gate(event):
            item = record(event, index)
            gates[item["pid"], item["tid"]].append(item)
            retained += 1
        if retained + len(names) > MAX_INTERVALS:
            raise ValueError("too many native callback gate records")
    print(
        "CALLBACK_DEVICE_TASKS "
        + json.dumps(
            [{"tid": tid, "name": name, "count": count} for (tid, name), count in names.most_common(32)],
            ensure_ascii=False,
        ),
        flush=True,
    )
    bindings = {}
    checks = []
    # A HOSTFUNC_CALLBACK record can end before the Host work. Read its real
    # successor on the same stream; filtering out other task types here could
    # silently associate a later, unrelated NOTIFY_WAIT.
    stream_tasks = {stream: [] for stream in gates}
    retained = 0
    for index, event in enumerate(events(path)):
        stream = (str(event.get("pid")), str(event.get("tid")))
        if event.get("ph") != "X" or stream not in stream_tasks:
            continue
        item = record(event, index)
        item.update(native=native_gate(event), notify=task_matches(event, NOTIFY_GATE))
        stream_tasks[stream].append(item)
        retained += 1
        if retained > MAX_INTERVALS:
            raise ValueError("too many native callback stream tasks; refuse partial mapping")
    for stream, tasks in stream_tasks.items():
        tasks.sort(key=lambda item: (item["start"], item["index"]))
        pairs, layout = callback_pairs(tasks)
        if not pairs:
            examples = []
            for i, item in enumerate(tasks):
                if item["native"] and len(examples) < MAX_CHECK_EXAMPLES:
                    examples.append([{key: row[key] for key in ("name", "start", "end")} for row in tasks[i : i + 3]])
            checks.append(dict(pid=stream[0], tid=stream[1], reason=layout, examples=examples))
            continue
        envelopes = [dict(launch, end=wait["end"]) for launch, wait in pairs]
        kinds = []
        for kind, hosts in data["groups"].items():
            check = dict(pid=stream[0], tid=stream[1], kind=kind, layout=layout, **gate_check(envelopes, hosts))
            checks.append(check)
            if check["reason"] == "ok":
                kinds.append(kind)
        if not kinds:
            continue
        if len(kinds) != 1 or kinds[0] in bindings:
            raise ValueError("ambiguous native callback gate mapping")
        bindings[kinds[0]] = (stream, pairs)
    print("CALLBACK_GATE_CHECKS " + json.dumps(checks, ensure_ascii=False), flush=True)
    data["callback_gate_checks"] = checks
    if set(bindings) != {"prefetch", "reactive"}:
        raise ValueError(
            "native callback device gates not uniquely identifiable; "
            "need explicit native callback tasks (including HOSTFUNC_CALLBACK) with complete count/order/containment; "
            "EVENT_WAIT or nearest timestamps alone are not callback identity evidence; "
            "checks=" + json.dumps(checks, ensure_ascii=False) + "; "
            "observed="
            + json.dumps(
                [{"tid": tid, "name": name, "n": count} for (tid, name), count in names.most_common(12)],
                ensure_ascii=False,
            )
        )
    return bindings
