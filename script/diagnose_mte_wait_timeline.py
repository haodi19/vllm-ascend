# SPDX-License-Identifier: Apache-2.0
"""Read existing Host/device timelines without assuming their clock alignment."""

import argparse
import bisect
import collections
import contextlib
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from inspect_moe_profile import _interval, _stats

MAX_TRACES = 16
MAX_EXAMPLES = 3
MAX_NEIGHBORS = 2
PROGRESS_EVENTS = 500000
DEFAULT_ORDER_TOLERANCE_US = 1.0
SHIFT_WINDOW_PAIRS = 64
MAX_DIAGNOSTIC_ROWS = 5
CONTEXT_FIELDS = ("backend", "layer", "kind", "rows", "call")
HANDSHAKE = re.compile(r"^host_sync_(publish|wait)(?:_\d+)?$")
IDENTITY_FIELDS = frozenset(("streamid", "taskid", "connectionid", "correlationid", "modelid"))


def identity_fields(args):
    return {
        key: value
        for key, value in args.items()
        if re.sub(r"[^a-z0-9]", "", key.lower()) in IDENTITY_FIELDS and isinstance(value, (str, int, float))
    }


def record(event, index, *, time_origin=None):
    if time_origin is None:
        start, end = _interval(event)
        duration = end - start
    else:
        # Subtract the epoch in decimal before converting to float. Native
        # nested ranges can end only 0.1 us apart at ~1.8e15 us timestamps.
        values = [Decimal(str(event.get(key))) for key in ("ts", "dur")]
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("invalid exact event interval")
        timestamp, elapsed = values
        start, end = float(timestamp - time_origin), float(timestamp + elapsed - time_origin)
        duration = float(elapsed)
    return dict(
        index=index,
        name=str(event.get("name", "")),
        pid=str(event.get("pid")),
        tid=str(event.get("tid")),
        start=start,
        end=end,
        duration=duration,
        ids=identity_fields(event.get("args", {})),
    )


def host_key(event):
    context = fields(event)
    return (str(event.get("pid")), str(event.get("tid")), *(context.get(key) for key in CONTEXT_FIELDS))


def read_trace(path, *, time_origin=None, allow_pipeline=False):
    """Keep only protocol tasks and Host boundaries; validate the complete JSON."""
    processes, threads = {}, {}
    candidates, hosts, syncs = [], collections.defaultdict(list), collections.defaultdict(list)
    enqueues = collections.defaultdict(list)
    phases = collections.Counter()
    retained = 0
    for index, event in enumerate(events(path, parse_float=Decimal if time_origin is not None else float)):
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
        if stage in ("moe_offload/host_callback", "moe_offload/synchronize") or (
            allow_pipeline and stage == "moe_offload/pipeline_enqueue"
        ):
            phases[f"{stage}:{event.get('ph')}"] += 1
            if event.get("ph") != "X":
                continue
            rec = record(event, index, time_origin=time_origin)
            rec["context"] = fields(event)
            target = (
                hosts if stage.endswith("host_callback") else enqueues if stage.endswith("pipeline_enqueue") else syncs
            )
            target[host_key(event)].append(rec)
            retained += 1
        elif HANDSHAKE.fullmatch(name) and event.get("ph") == "X":
            candidates.append(record(event, index, time_origin=time_origin))
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
        if allow_pipeline and len(entries) == 1 and entries[0]["context"].get("pipeline") == "1":
            host = entries[0]
            queued = enqueues.get(key, [])
            if (
                corresponding
                or len(queued) != 1
                or not (host["start"] <= queued[0]["start"] <= queued[0]["end"] <= host["end"])
            ):
                issues["invalid_pipeline_enqueue_range"] += 1
                invalid_kinds.add(kind)
                continue
            host["sync"] = None
            host["enqueue"] = queued[0]
            groups[kind].append(host)
            continue
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


def task_pairs(tasks, tolerance_us=0.0):
    """A stream must alternate publish/wait, without overlapping executions."""
    if len(tasks) % 2:
        return [], "odd_handshake_count"
    pairs = []
    for index in range(0, len(tasks), 2):
        publish, wait = tasks[index : index + 2]
        if not publish["name"].startswith("host_sync_publish") or not wait["name"].startswith("host_sync_wait"):
            return [], "not_alternating_publish_wait"
        if publish["end"] - wait["start"] > tolerance_us:
            return [], "publish_wait_overlap_on_same_stream"
        if index and tasks[index - 1]["end"] - publish["start"] > tolerance_us:
            return [], "successive_tasks_overlap_on_same_stream"
        pairs.append((publish, wait))
    return pairs, "ok" if pairs else "empty"


def order_audit(tasks, tolerance_us):
    """Retain the strict result and report every overlap, without adjusting times."""
    if not math.isfinite(tolerance_us) or tolerance_us < 0:
        raise ValueError("order tolerance must be finite and nonnegative")
    _, strict_reason = task_pairs(tasks)
    _, tolerant_reason = task_pairs(tasks, tolerance_us)
    groups = {"publish_to_wait": [], "wait_to_next_publish": []}
    for left, right in zip(tasks, tasks[1:]):
        a, b = HANDSHAKE.fullmatch(left["name"]), HANDSHAKE.fullmatch(right["name"])
        if a is None or b is None:
            continue
        key = (
            "publish_to_wait"
            if (a[1], b[1]) == ("publish", "wait")
            else "wait_to_next_publish"
            if (a[1], b[1]) == ("wait", "publish")
            else None
        )
        if key is None or left["end"] <= right["start"]:
            continue
        groups[key].append(
            dict(
                overlap_us=left["end"] - right["start"],
                left_index=left["index"],
                left_name=left["name"],
                left_start=left["start"],
                left_end=left["end"],
                right_index=right["index"],
                right_name=right["name"],
                right_start=right["start"],
                right_end=right["end"],
            )
        )
    report = dict(
        strict_result=strict_reason,
        tolerance_us=tolerance_us,
        tolerance_result=tolerant_reason,
        timestamps_modified=False,
    )
    for name, rows in groups.items():
        values = [row["overlap_us"] for row in rows]
        report[name] = dict(
            overlap_us=_stats(values),
            min_us=min(values, default=None),
            above_tolerance=sum(value > tolerance_us for value in values),
            examples=sorted(rows, key=lambda row: row["overlap_us"], reverse=True)[:MAX_EXAMPLES],
        )
    return report


def shift_diagnostics(pairs, hosts, window_size=SHIFT_WINDOW_PAIRS):
    """All ordinal samples retained; local bounds expose time/layer dependence."""
    if window_size < 1:
        raise ValueError("shift window must be positive")
    if len(pairs) != len(hosts):
        raise ValueError("shift diagnostics require equal pair and host counts")
    samples = []
    for ordinal, ((publish, wait), host) in enumerate(zip(pairs, hosts)):
        samples.append(
            dict(
                ordinal=ordinal,
                context=host["context"],
                host_start=host["start"],
                host_end=host["end"],
                publish_start=publish["start"],
                wait_start=wait["start"],
                wait_end=wait["end"],
                lower_us=host["end"] - wait["end"],
                upper_us=host["start"] - publish["start"],
            )
        )
    if not samples:
        return {}
    origin = samples[0]["host_start"]
    windows = []
    for first in range(0, len(samples), window_size):
        rows = samples[first : first + window_size]
        lower, upper = max(row["lower_us"] for row in rows), min(row["upper_us"] for row in rows)
        windows.append(
            [
                first,
                first + len(rows) - 1,
                round((rows[0]["host_start"] - origin) / 1000, 3),
                round((rows[-1]["host_start"] - origin) / 1000, 3),
                lower,
                upper,
                lower <= upper,
            ]
        )
    layers = collections.defaultdict(list)
    for row in samples:
        layers[row["context"]["layer"]].append(row)
    layer_rows = []
    for layer, rows in sorted(layers.items(), key=lambda entry: (len(str(entry[0])), str(entry[0]))):
        lower_values = [row["lower_us"] for row in rows]
        stats = _stats(lower_values)
        lower, upper = max(lower_values), min(row["upper_us"] for row in rows)
        layer_rows.append(
            [
                layer,
                len(rows),
                round(stats["mean_us"], 3),
                stats["p50_us"],
                stats["p95_us"],
                lower,
                upper,
                lower <= upper,
            ]
        )
    window_indices = sorted(
        {
            0,
            len(windows) // 2,
            len(windows) - 1,
            max(range(len(windows)), key=lambda i: windows[i][4]),
            min(range(len(windows)), key=lambda i: windows[i][5]),
        }
    )
    worst_layers = sorted(range(len(layer_rows)), key=lambda i: layer_rows[i][5] - layer_rows[i][6], reverse=True)[:3]
    layer_indices = sorted(
        set(
            worst_layers
            + [
                min(range(len(layer_rows)), key=lambda i: layer_rows[i][2]),
                max(range(len(layer_rows)), key=lambda i: layer_rows[i][2]),
            ]
        )
    )
    return dict(
        binding_lower_sample=max(samples, key=lambda row: row["lower_us"]),
        binding_upper_sample=min(samples, key=lambda row: row["upper_us"]),
        first_sample=samples[0],
        last_sample=samples[-1],
        windows=dict(
            pair_count=window_size,
            total_windows=len(windows),
            infeasible_windows=sum(not row[6] for row in windows),
            lower_range_us=[min(row[4] for row in windows), max(row[4] for row in windows)],
            upper_range_us=[min(row[5] for row in windows), max(row[5] for row in windows)],
            selection="first/middle/last/max-lower/min-upper windows; all windows computed",
            columns=[
                "first_ordinal",
                "last_ordinal",
                "first_rel_host_ms",
                "last_rel_host_ms",
                "lower_us",
                "upper_us",
                "feasible",
            ],
            rows=[windows[i] for i in window_indices][:MAX_DIAGNOSTIC_ROWS],
        ),
        layers=dict(
            columns=[
                "layer",
                "n",
                "lower_mean_us",
                "lower_p50_us",
                "lower_p95_us",
                "lower_max_us",
                "upper_min_us",
                "feasible",
            ],
            total_layers=len(layer_rows),
            infeasible_layers=sum(not row[7] for row in layer_rows),
            selection="three worst lower-minus-upper gaps plus min/max lower-mean layers; all layers computed",
            rows=[layer_rows[i] for i in layer_indices][:MAX_DIAGNOSTIC_ROWS],
        ),
        scope="same ordinal hypothesis; all samples retained; bounds are necessary, not calibrated offsets",
    )


def layer_cycles(pairs, hosts):
    """Validate whole ascending repetitions of the observed numeric layer set."""
    if not hosts or len(pairs) != len(hosts):
        return dict(status="count_mismatch_or_empty", device_pairs=len(pairs), host_callbacks=len(hosts))
    values = [host["context"].get("layer") for host in hosts]
    if any(not re.fullmatch(r"\d+", str(value)) for value in values):
        return dict(status="non_numeric_or_negative_layers")
    layers = [int(value) for value in values]
    order = sorted(set(layers))
    common = dict(
        observed_layer_order=order,
        layer_count=len(order),
        samples=len(hosts),
        scope="complete repetitions of observed layers only; not proof of model-wide layer coverage",
    )
    if len(order) < 2:
        return dict(status="insufficient_layer_span", **common)
    mismatches = [index for index, layer in enumerate(layers) if layer != order[index % len(order)]]
    if len(layers) % len(order) or mismatches:
        return dict(
            status="incomplete_or_nonrepeating_layer_sequence",
            **common,
            partial_first_cycle=layers[0] != order[0],
            partial_last_cycle=layers[-1] != order[-1],
            sample_count_remainder=len(layers) % len(order),
            mismatched_positions=len(mismatches),
            first_mismatches=[
                dict(ordinal=index, actual=layers[index], expected=order[index % len(order)])
                for index in mismatches[:MAX_EXAMPLES]
            ],
        )
    rows = []
    origin = hosts[0]["start"]
    for cycle, first in enumerate(range(0, len(hosts), len(order))):
        selected_pairs, selected_hosts = pairs[first : first + len(order)], hosts[first : first + len(order)]
        lower = max(host["end"] - wait["end"] for (_, wait), host in zip(selected_pairs, selected_hosts))
        upper = min(host["start"] - publish["start"] for (publish, _), host in zip(selected_pairs, selected_hosts))
        rows.append(
            [
                cycle,
                first,
                first + len(order) - 1,
                round((selected_hosts[0]["start"] - origin) / 1000, 3),
                round((selected_hosts[-1]["start"] - origin) / 1000, 3),
                lower,
                upper,
                lower <= upper,
            ]
        )
    indices = sorted(
        {
            0,
            len(rows) // 2,
            len(rows) - 1,
            max(range(len(rows)), key=lambda i: rows[i][5] - rows[i][6]),
            max(range(len(rows)), key=lambda i: rows[i][5]),
        }
    )
    return dict(
        status="validated_complete_repetitions",
        **common,
        total_cycles=len(rows),
        infeasible_cycles=sum(not row[7] for row in rows),
        global_necessary_bounds_us=[max(row[5] for row in rows), min(row[6] for row in rows)],
        selection="first/middle/last/worst-gap/max-lower cycles; all cycles computed; zero-based cycle index",
        columns=[
            "cycle",
            "first_ordinal",
            "last_ordinal",
            "first_rel_host_ms",
            "last_rel_host_ms",
            "lower_us",
            "upper_us",
            "feasible",
        ],
        rows=[rows[index] for index in indices][:MAX_DIAGNOSTIC_ROWS],
    )


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
        shift_diagnostics=shift_diagnostics(pairs, hosts),
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


def cross_kind_overlap(waits, groups):
    """The SAME wait begins in a prefetch sync and ends in a reactive sync."""
    lookups = {}
    for kind in ("prefetch", "reactive"):
        rows = sorted(groups.get(kind, []), key=lambda row: row["sync"]["start"])
        lookups[kind] = ([row["sync"]["start"] for row in rows], rows)

    def containing(kind, value):
        starts, rows = lookups[kind]
        index = bisect.bisect_right(starts, value) - 1
        if index >= 0 and value <= rows[index]["sync"]["end"]:
            return rows[index]
        return None

    count, same_layer, examples = 0, 0, []
    for wait in waits:
        pre, react = containing("prefetch", wait["start"]), containing("reactive", wait["end"])
        if pre is None or react is None:
            continue
        count += 1
        same_layer += pre["context"]["layer"] == react["context"]["layer"]
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                dict(
                    wait_start=wait["start"],
                    wait_end=wait["end"],
                    start_prefetch_context=pre["context"],
                    end_reactive_context=react["context"],
                    prefetch_sync=[pre["sync"]["start"], pre["sync"]["end"]],
                    reactive_sync=[react["sync"]["start"], react["sync"]["end"]],
                )
            )
    return dict(
        wait_start_prefetch_and_end_reactive=count,
        same_layer_context=same_layer,
        examples=examples,
        basis="raw-coordinate overlap for same wait; not task identity",
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


def analyze(path, order_tolerance_us=DEFAULT_ORDER_TOLERANCE_US):
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
        audit = order_audit(tasks, order_tolerance_us)
        chosen_reason = reason
        pairing_basis = "strict_device_order"
        if reason != "ok" and audit["tolerance_result"] == "ok":
            pairs, chosen_reason = task_pairs(tasks, order_tolerance_us)
            pairing_basis = "device_order_with_explicit_tolerance"
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
        emit("ORDER_AUDIT", dict(stream=key, **audit))
        emit("RAW_CROSS_KIND_OVERLAP_NOT_PAIRING", dict(stream=key, **cross_kind_overlap(waits, data["groups"])))
        emit("RAW_WAIT_SPANS_NOT_PAIRING", dict(stream=key, **spanning_callbacks(waits, data["all_hosts"])))
        for item in nearby[key]:
            emit("DEVICE_NEIGHBORS", item)
        usable = []
        for kind in ("prefetch", "reactive"):
            hosts = data["groups"].get(kind, [])
            emit("RAW_OVERLAP_NOT_PAIRING", dict(stream=key, kind=kind, **containment(waits, hosts)))
            result = (
                candidate(pairs, hosts, kind not in data["invalid_kinds"])
                if chosen_reason == "ok"
                else dict(status=reason)
            )
            examples = result.pop("examples", [])
            diagnostics = result.pop("shift_diagnostics", None)
            label = (
                "ORDINAL_CANDIDATE" if pairing_basis == "strict_device_order" else "ORDER_TOLERATED_ORDINAL_CANDIDATE"
            )
            emit(label, dict(stream=key, kind=kind, pairing_basis=pairing_basis, **result))
            if diagnostics:
                emit(
                    "SHIFT_BINDING_SAMPLES",
                    dict(
                        stream=key,
                        kind=kind,
                        **{
                            name: diagnostics[name]
                            for name in ("binding_lower_sample", "binding_upper_sample", "first_sample", "last_sample")
                        },
                    ),
                )
                emit("SHIFT_WINDOWS", dict(stream=key, kind=kind, **diagnostics["windows"]))
                emit("SHIFT_LAYERS", dict(stream=key, kind=kind, **diagnostics["layers"], scope=diagnostics["scope"]))
                emit("SHIFT_LAYER_CYCLES", dict(stream=key, kind=kind, **layer_cycles(pairs, hosts)))
            for example in examples:
                emit("CONTRADICTION_EXAMPLE", dict(stream=key, kind=kind, **example))
            if result["status"] == "ordinal_candidate_not_identity_proof":
                usable.append((kind, result))
        if len(usable) == 1:
            kind, result = usable[0]
            lower, upper = result["constant_device_shift_necessary_bounds_us"]
            device_bounds[key[0]].append((key, kind, lower, upper, pairing_basis))
    for pid, bounds in device_bounds.items():
        lower, upper = max(row[2] for row in bounds), min(row[3] for row in bounds)
        emit(
            "CROSS_STREAM_SHIFT_HYPOTHESIS",
            dict(
                hardware_pid=pid,
                candidate_stream_count=len(bounds),
                coverage="multiple_streams" if len(bounds) > 1 else "single_stream_only",
                candidate_stream_kinds=[(row[0], row[1]) for row in bounds],
                candidate_order_basis=[(row[0], row[4]) for row in bounds],
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


def compare_aligned(path, order_tolerance_us=DEFAULT_ORDER_TOLERANCE_US):
    """Check an existing export; never fit clocks using the MoE ordering."""
    path = path.resolve()
    aligned_path = path.with_name("trace_view_aligned.json")
    mapping = json.loads(path.with_name("clock_alignment.json").read_text(encoding="utf-8"))
    if (
        mapping.get("status") != "calibration_validated"
        or mapping.get("calibration_layout") != "model_request_brackets"
    ):
        raise ValueError("need successful same-model-capture alignment")
    if Path(mapping["source"]).resolve() != path or Path(mapping["aligned_trace"]).resolve() != aligned_path:
        raise ValueError("clock mapping points to another trace")
    raw, aligned = read_trace(path), read_trace(aligned_path)
    if raw["all_hosts"] != aligned["all_hosts"] or raw["groups"] != aligned["groups"]:
        raise ValueError("Host callback/synchronize records changed in aligned trace")
    if raw["hardware"] != aligned["hardware"] or sorted(mapping["device_pids"]) != raw["hardware"]:
        raise ValueError("device time domains do not match mapping")
    if raw["tasks"].keys() != aligned["tasks"].keys():
        raise ValueError("handshake stream set changed")
    if not raw["tasks"]:
        raise ValueError("no MTE handshake tasks; select mte_default/mte_combined")
    scale, offset, origin = (Decimal(mapping[key]) for key in ("device_scale", "device_offset_us", "trace_origin_us"))
    if scale <= 0:
        raise ValueError("invalid clock scale")
    results, incomplete, residual = [], 0, 0

    def identity(record):
        return tuple(record[key] for key in ("index", "name", "pid", "tid", "ids"))

    for stream, tasks in sorted(raw["tasks"].items()):
        corrected = aligned["tasks"][stream]
        if len(tasks) != len(corrected) or any(identity(a) != identity(b) for a, b in zip(tasks, corrected)):
            raise ValueError("handshake event identities/order changed")
        for a, b in zip(tasks, corrected):
            for edge in ("start", "end"):
                expected = origin + (Decimal(str(a[edge])) - origin - offset) / scale
                # Existing reader uses float at absolute epoch; allow <=1 us
                # for conversion/quantization, not a MoE causal tolerance.
                if abs(Decimal(str(b[edge])) - expected) > DEFAULT_ORDER_TOLERANCE_US:
                    raise ValueError("handshake coordinates disagree with saved clock mapping")
        pairs, raw_order = task_pairs(tasks, order_tolerance_us)
        aligned_pairs, aligned_order = task_pairs(corrected, order_tolerance_us)
        kinds = [
            kind
            for kind, hosts in raw["groups"].items()
            if kind not in raw["invalid_kinds"] and pairs and len(pairs) == len(hosts)
        ]
        result = dict(
            stream=stream,
            name=raw["threads"].get(stream),
            raw_order=raw_order,
            aligned_order=aligned_order,
            strict_raw_order=task_pairs(tasks)[1],
            strict_aligned_order=task_pairs(corrected)[1],
        )
        if raw_order != "ok" or aligned_order != "ok" or len(kinds) != 1:
            incomplete += 1
            result.update(status="no_unique_complete_ordinal_candidate", candidate_kinds=kinds)
            results.append(result)
            continue
        kind = kinds[0]
        hosts = raw["groups"][kind]
        result.update(kind=kind, pairs=len(pairs), status="ordinal_candidate_not_slot_identity_proof")
        before_after = {}
        for label, selected in (("raw", pairs), ("aligned", aligned_pairs)):
            deltas = {
                "host_start_minus_publish_start_us": [
                    host["start"] - pub["start"] for (pub, wait), host in zip(selected, hosts)
                ],
                "wait_end_minus_sync_end_us": [
                    wait["end"] - host["sync"]["end"] for (pub, wait), host in zip(selected, hosts)
                ],
                "wait_end_minus_host_end_us": [wait["end"] - host["end"] for (pub, wait), host in zip(selected, hosts)],
            }
            summary = {
                name: dict(
                    min_us=min(values),
                    mean_us=sum(values) / len(values),
                    negative_count=sum(value < 0 for value in values),
                    below_minus_1us_count=sum(value < -DEFAULT_ORDER_TOLERANCE_US for value in values),
                )
                for name, values in deltas.items()
            }
            summary["wait_fully_inside_own_sync"] = sum(
                host["sync"]["start"] <= wait["start"] <= wait["end"] <= host["sync"]["end"]
                for (_, wait), host in zip(selected, hosts)
            )
            summary["wait_starts_in_prefetch_sync_ends_in_reactive_sync"] = cross_kind_overlap(
                [wait for _, wait in selected], raw["groups"]
            )["wait_start_prefetch_and_end_reactive"]
            if label == "aligned":
                residual += sum(value["below_minus_1us_count"] for value in summary.values() if isinstance(value, dict))
            before_after[label] = summary
        result.update(before_after)
        # First and last instances aid visual lookup without changing the fit.
        result["view_examples"] = [
            dict(
                context=hosts[index]["context"],
                host_start_from_calibration_origin_us=hosts[index]["start"] - float(origin),
                raw_wait_end_minus_sync_end_us=pairs[index][1]["end"] - hosts[index]["sync"]["end"],
                aligned_wait_end_minus_sync_end_us=aligned_pairs[index][1]["end"] - hosts[index]["sync"]["end"],
            )
            for index in sorted({0, len(hosts) - 1})
        ]
        results.append(result)
    kind_counts = collections.Counter(row["kind"] for row in results if "kind" in row)
    ambiguous_kinds = sorted(
        kind
        for kind in set(raw["groups"]) | raw["invalid_kinds"]
        if kind_counts[kind] != 1 or kind in raw["invalid_kinds"]
    )
    return dict(
        source=str(path),
        aligned_trace=str(aligned_path),
        host_issues=raw["issues"],
        host_counts={kind: len(hosts) for kind, hosts in raw["groups"].items()},
        mapping_envelope_width_us=mapping.get("endpoint_mapping_envelope_width_us"),
        streams=results,
        unpaired_streams=incomplete,
        ambiguous_or_uncovered_kinds=ambiguous_kinds,
        remaining_negative_relations_over_1us=residual,
        status="incomplete_pairing"
        if incomplete or ambiguous_kinds
        else "residual_negative_relations"
        if residual
        else "no_negative_relations_over_1us_in_ordinal_candidates",
        scope="ordinal candidates only; all samples retained; no extra clock fit or IQR filtering",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="existing trace_view.json files or run/profile directories")
    parser.add_argument(
        "--compare-aligned", action="store_true", help="Compare original/aligned MTE waits using saved calibration"
    )
    parser.add_argument(
        "--summary-file", type=Path, help="Write a new compact alignment audit; never overwrite an existing file"
    )
    parser.add_argument(
        "--order-tolerance-us",
        type=nonnegative_us,
        default=DEFAULT_ORDER_TOLERANCE_US,
        help="explicit tolerance for additional device-order candidates; strict result stays reported",
    )
    args = parser.parse_args(argv)
    if args.summary_file and not args.compare_aligned:
        parser.error("--summary-file requires --compare-aligned")
    if args.compare_aligned:
        with (
            args.summary_file.open("x", encoding="utf-8")
            if args.summary_file
            else contextlib.nullcontext(sys.stdout) as report
        ):

            def output(label, value):
                line = label + " " + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                print(line, file=report, flush=True)
                if report is not sys.stdout:
                    print(line, flush=True)

            output("SCOPE", "只读本轮原图/对齐图；不导入Torch、不启动模型/NPU、不修改profile。仅写指定的新摘要。")
            output("PAIRING", "按流内顺序和完整数量形成唯一候选，不按时间重叠配对；未证明slot/generation身份。")
            output(
                "TOLERANCE", dict(device_order_us=args.order_tolerance_us, causal_report_us=DEFAULT_ORDER_TOLERANCE_US)
            )
            try:
                for path in discover(args.paths):
                    result = compare_aligned(path, args.order_tolerance_us)
                    streams = result.pop("streams")
                    output("TRACE_COMPARISON", result)
                    for stream in streams:
                        output("WAIT_COMPARISON", stream)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                output("FAILED", f"{type(exc).__name__}: {exc}")
                return 1
            output("DONE", "检查完成；负差值表示图上仍提前。零计数只说明候选序列无此矛盾，不证明协议或零误差。")
        return 0
    print("只读已有 trace；不导入 Torch、不启动模型/NPU、不写文件。全部数值为原导出坐标，单位 us。", flush=True)
    print(
        f"额外顺序容差={args.order_tolerance_us:g} us，仅为诊断假设，不是已验证的计时精度；"
        "严格检查结果保留，不修改任何时间戳。",
        flush=True,
    )
    print("因果约束以本轮配置确认为纯 MTE 严格等待且成功完成为前提；同名算子不能排除有界诊断变体。", flush=True)
    try:
        paths = discover(args.paths)
        for path in paths:
            analyze(path, args.order_tolerance_us)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}; incomplete input is not evidence of protocol success", flush=True)
        return 1
    return 0


def nonnegative_us(value):
    try:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be finite nonnegative microseconds") from exc


if __name__ == "__main__":
    raise SystemExit(main())
