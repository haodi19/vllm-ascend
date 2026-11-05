# SPDX-License-Identifier: Apache-2.0
"""从已选定设备流和Host回调窗口，保守关联加载批次及批次间空闲区间。"""

import bisect
import collections

from moe_outlier_stats import outlier_summary


def pair_direction(left, right):
    """Use explicit callback metadata, never the gap duration, to classify a pair."""
    if left["layer"] is None or right["layer"] is None:
        return "unknown"
    if left["kind"] == "prefetch" and right["kind"] == "reactive" and left["layer"] == right["layer"]:
        return "prefetch->reactive"
    if left["kind"] == "reactive" and right["kind"] == "prefetch" and right["layer"] == left["layer"] + 1:
        return "reactive->prefetch"
    if right["layer"] < left["layer"]:
        return "round_boundary_candidate"
    return "other_structure"


def distributions(values, lower_us, long_us):
    """Preserve all samples; explicit time bands are descriptive, not exclusions."""
    return dict(
        all=outlier_summary(values),
        regular_band=outlier_summary([value for value in values if value < long_us]),
        large_band=outlier_summary([value for value in values if lower_us <= value < long_us]),
        long_band=outlier_summary([value for value in values if value >= long_us]),
        zero_gaps=sum(value == 0 for value in values),
        below_large_threshold=sum(value < lower_us for value in values),
    )


def analyze_offload_batches(tasks, callbacks, *, lower_us, long_us):
    """Associate only fully contained tasks; ambiguous/busy pairs never become idle samples.

    Input timestamps share the exported trace coordinate system. Containment is
    a temporal association, not a proved CPU-to-device flow/correlation link.
    """
    if not callbacks:
        raise ValueError("没有选中Host进程的host_callback完整区间，不能关联加载批次")
    tasks = sorted(tasks, key=lambda task: (task["start"], task["end"], task["event_index"]))
    callbacks = sorted(callbacks, key=lambda callback: (callback["start"], callback["end"], callback["event_index"]))
    starts = [callback["start"] for callback in callbacks]
    callback_max_end = []
    for callback in callbacks:
        callback_max_end.append(max(callback["end"], callback_max_end[-1] if callback_max_end else callback["end"]))
    first_start, last_end = starts[0], callback_max_end[-1]
    contaminated = set()
    overlapping_callbacks = set()
    invalid_callbacks = set()
    seen_calls = {}
    active = []
    for index, callback in enumerate(callbacks):
        active = [previous for previous in active if callbacks[previous]["end"] > callback["start"]]
        if active:
            overlapping_callbacks.update(active)
            overlapping_callbacks.add(index)
        active.append(index)
        if callback["kind"] not in ("prefetch", "reactive") or callback["layer"] is None or callback["call"] is None:
            invalid_callbacks.add(index)
        if callback["call"] is not None:
            if callback["call"] in seen_calls:
                invalid_callbacks.update((index, seen_calls[callback["call"]]))
            seen_calls[callback["call"]] = index

    assigned = collections.defaultdict(list)
    copy_phases, task_matches = collections.Counter(), collections.Counter()
    activity_phases = collections.defaultdict(collections.Counter)
    for task_index, task in enumerate(tasks):
        # Prefix maximum ends allow skipping all earlier nonintersecting windows.
        first = bisect.bisect_left(callback_max_end, task["start"])
        last = bisect.bisect_right(starts, task["end"])
        containing, intersecting = [], []
        for callback_index in range(first, last):
            callback = callbacks[callback_index]
            if callback["start"] <= task["start"] and task["end"] <= callback["end"]:
                containing.append(callback_index)
            if callback["start"] < task["end"] and task["start"] < callback["end"]:
                intersecting.append(callback_index)
        if len(containing) == 1:
            assigned[containing[0]].append(task_index)
            category = "inside_unique"
        elif len(containing) > 1:
            category = "ambiguous"
            contaminated.update(containing)
        elif intersecting:
            category = "cross_boundary"
            contaminated.update(intersecting)
        elif task["end"] <= first_start:
            category = "before_callbacks"
        elif task["start"] >= last_end:
            category = "after_callbacks"
        else:
            category = "between_callbacks"
        task_matches[category] += 1
        activity_phases[category][task["kind"]] += 1
        if task["kind"] == "copy":
            copy_phases[category] += 1

    batches = {}
    for callback_index, indices in assigned.items():
        copies = [index for index in indices if tasks[index]["kind"] == "copy"]
        if not copies:
            continue
        batches[callback_index] = dict(
            first_task=min(indices, key=lambda index: (tasks[index]["start"], tasks[index]["event_index"])),
            last_task=max(indices, key=lambda index: (tasks[index]["end"], tasks[index]["event_index"])),
            first_copy=min(copies, key=lambda index: (tasks[index]["start"], tasks[index]["event_index"])),
            last_copy=max(copies, key=lambda index: (tasks[index]["end"], tasks[index]["event_index"])),
            copies=len(copies),
            tasks=len(indices),
            transpose_tasks=sum(tasks[index]["kind"] == "transpose" for index in indices),
        )

    task_starts = [task["start"] for task in tasks]
    task_max_end = []
    for task in tasks:
        task_max_end.append(max(task["end"], task_max_end[-1] if task_max_end else task["end"]))
    pairs, round_pairs = [], []
    excluded = collections.Counter()
    for left_index in range(len(callbacks) - 1):
        right_index = left_index + 1
        indices = {left_index, right_index}
        if indices & invalid_callbacks:
            excluded["invalid_callback_identity"] += 1
            continue
        if indices & overlapping_callbacks:
            excluded["overlapping_callbacks"] += 1
            continue
        if indices & contaminated:
            excluded["ambiguous_or_crossing_device_task"] += 1
            continue
        if left_index not in batches or right_index not in batches:
            excluded["empty_or_unmatched_callback"] += 1
            continue
        left_callback, right_callback = callbacks[left_index], callbacks[right_index]
        direction = pair_direction(left_callback, right_callback)
        if direction not in ("prefetch->reactive", "reactive->prefetch", "round_boundary_candidate"):
            excluded[direction] += 1
            continue
        left_batch, right_batch = batches[left_index], batches[right_index]
        left_task, right_task = tasks[left_batch["last_task"]], tasks[right_batch["first_task"]]
        gap = right_task["start"] - left_task["end"]
        if gap < 0:
            excluded["device_batches_overlap"] += 1
            continue
        # Check *all* selected-stream tasks, including unmatched tasks and tasks
        # starting before the left boundary. A busy interval cannot be an idle gap.
        cut = bisect.bisect_left(task_starts, right_task["start"])
        if gap > 0 and cut and task_max_end[cut - 1] > left_task["end"]:
            excluded["occupied_by_other_device_task"] += 1
            continue
        left_copy, right_copy = tasks[left_batch["last_copy"]], tasks[right_batch["first_copy"]]
        copy_gap = right_copy["start"] - left_copy["end"]
        record = dict(
            direction=direction,
            left_callback=left_callback,
            right_callback=right_callback,
            left_task=left_task,
            right_task=right_task,
            left_copy=left_copy,
            right_copy=right_copy,
            left_batch=left_batch,
            right_batch=right_batch,
            gap_us=gap,
            copy_only_gap_us=copy_gap,
            copy_only_excess_us=copy_gap - gap,
        )
        (round_pairs if direction == "round_boundary_candidate" else pairs).append(record)

    partial_reasons = {}
    if invalid_callbacks:
        partial_reasons["invalid_callback_identity"] = len(invalid_callbacks)
    if overlapping_callbacks:
        partial_reasons["overlapping_callbacks"] = len(overlapping_callbacks)
    for reason in ("ambiguous", "cross_boundary"):
        if task_matches[reason]:
            partial_reasons[reason] = task_matches[reason]
    for reason in ("device_batches_overlap", "occupied_by_other_device_task"):
        if excluded[reason]:
            partial_reasons[reason] = excluded[reason]
    if not pairs:
        partial_reasons["no_normal_pairs"] = 1
    return dict(
        tasks=tasks,
        callbacks=callbacks,
        batches=batches,
        pairs=pairs,
        round_pairs=round_pairs,
        copy_phases=dict(copy_phases),
        activity_phases={phase: dict(kinds) for phase, kinds in activity_phases.items()},
        task_matches=dict(task_matches),
        excluded_pairs=dict(excluded),
        partial=bool(partial_reasons),
        partial_reasons=partial_reasons,
        directions={
            direction: dict(
                **distributions(
                    [pair["gap_us"] for pair in pairs if pair["direction"] == direction], lower_us, long_us
                ),
                copy_only=outlier_summary(
                    [pair["copy_only_gap_us"] for pair in pairs if pair["direction"] == direction]
                ),
                copy_only_excess=outlier_summary(
                    [pair["copy_only_excess_us"] for pair in pairs if pair["direction"] == direction]
                ),
            )
            for direction in ("prefetch->reactive", "reactive->prefetch")
        },
        round_gaps=distributions([pair["gap_us"] for pair in round_pairs], lower_us, long_us),
    )
