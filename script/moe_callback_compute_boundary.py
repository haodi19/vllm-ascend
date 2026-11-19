# SPDX-License-Identifier: Apache-2.0
"""Identify native callback device gates without applying MTE publish/wait rules."""

import collections
import json
import re

from analyze_callback_gaps import MAX_INTERVALS, events
from diagnose_mte_wait_timeline import record

NATIVE_GATE = re.compile(r"(?:DAVID_)?(?:HOST_FUNC|HOST_CPU|HOST_CALLBACK)(?:_\d+)?", re.IGNORECASE)
ORDER_US = 1.0


def native_gate(event):
    args = event.get("args") or {}
    names = [str(event.get("name", ""))]
    if isinstance(args, dict):
        names.extend(str(args[key]) for key in ("Task Type", "task_type") if key in args)
    return any(NATIVE_GATE.fullmatch(name) for name in names)


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
    for stream, tasks in gates.items():
        tasks.sort(key=lambda item: (item["start"], item["index"]))
        kinds = []
        for kind, hosts in data["groups"].items():
            if len(tasks) != len(hosts) or not hosts:
                continue
            if any(left["end"] > right["start"] + ORDER_US for left, right in zip(tasks, tasks[1:])):
                continue
            if all(
                task["start"] <= host["start"] + ORDER_US and task["end"] >= host["end"] - ORDER_US
                for task, host in zip(tasks, hosts)
            ):
                kinds.append(kind)
        if not kinds:
            continue
        if len(kinds) != 1 or kinds[0] in bindings:
            raise ValueError("ambiguous native callback gate mapping")
        # One native gate corresponds to a Host callback, not two handshake tasks.
        bindings[kinds[0]] = (stream, [(gate, gate) for gate in tasks])
    if set(bindings) != {"prefetch", "reactive"}:
        raise ValueError(
            "native callback device gates not uniquely identifiable; "
            "need explicit HOST_FUNC/HOST_CPU/HOST_CALLBACK tasks with complete count/order/containment; "
            "EVENT_WAIT or nearest timestamps alone are not callback identity evidence; "
            "observed="
            + json.dumps(
                [{"tid": tid, "name": name, "n": count} for (tid, name), count in names.most_common(12)],
                ensure_ascii=False,
            )
        )
    return bindings
