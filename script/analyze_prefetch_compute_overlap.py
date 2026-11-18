# SPDX-License-Identifier: Apache-2.0
"""Analyze saved, aligned MTE captures; never load Torch or run inference."""

import argparse
import collections
import contextlib
import json
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events
from diagnose_mte_wait_timeline import read_trace, record, task_pairs
from inspect_moe_profile import _stats

ORDER_US = 1.0
MAX_EXAMPLES = 8


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False), flush=True)


def task_type(name):
    name = name.upper()
    if "EVENT" in name and "WAIT" in name:
        return "event_wait"
    if "MEMCPY" in name:
        return "copy"
    if any(word in name for word in ("EVENT", "NOTIFY", "MEMSET", "HOST_SYNC")):
        return "control"
    return "compute"


def bind_streams(data):
    """Complete ordinal mapping, independent of the overlap being tested."""
    if data["issues"] or data["invalid_kinds"]:
        raise ValueError(f"invalid Host markers: {data['issues']}")
    bindings = {}
    for stream, tasks in data["tasks"].items():
        pairs, reason = task_pairs(tasks, ORDER_US)
        kinds = [k for k, hosts in data["groups"].items() if pairs and len(hosts) == len(pairs)]
        if reason != "ok" or len(kinds) != 1 or kinds[0] in bindings:
            raise ValueError("no unique complete ordinal stream mapping")
        kind = kinds[0]
        for (publish, wait), host in zip(pairs, data["groups"][kind]):
            if publish["start"] > host["start"] + ORDER_US or wait["end"] < host["end"] - ORDER_US:
                raise ValueError("aligned protocol order check failed")
        bindings[kind] = (stream, pairs)
    if set(bindings) != {"prefetch", "reactive"}:
        raise ValueError("need both MTE streams; callback backend is not classified with MTE rules")
    return bindings


def consumer_boundary(segment):
    """Bracket readiness using the last compute and subsequent consumer join.

    Require the baseline decode tail: Event wait, exactly two D2H task
    records, reactive publish. Extra compute (e.g. substitution/pruning) or
    absent task records make this boundary unavailable rather than guessed.
    The Event wait START is an upper bound, not its completion time.
    """
    joins = [i for i, row in enumerate(segment) if task_type(row["name"]) == "event_wait"]
    if not joins:
        return None, "missing_consumer_event_wait"
    index = joins[-1]
    tail = [task_type(row["name"]) for row in segment[index + 1 :]]
    if tail.count("copy") != 2 or any(kind not in ("copy", "control") for kind in tail):
        return None, "consumer_tail_not_two_copies_and_controls"
    previous = [row for row in segment[:index] if task_type(row["name"]) == "compute"]
    if not previous:
        return None, "missing_prejoin_compute"
    last, join = previous[-1], segment[index]
    if last["end"] > join["start"] + ORDER_US:
        return None, "overlapping_compute_and_join"
    return (last, join), "ok"


def classify(lower, upper, prefetch_end, margin):
    if upper < prefetch_end - margin:
        return "early"
    if lower > prefetch_end + margin:
        return "late"
    return "uncertain"


def summarize(rows):
    total = sum(row["gap_us"] for row in rows)
    groups = {}
    for kind in ("all", "early", "late", "uncertain", "unmapped"):
        subset = rows if kind == "all" else [row for row in rows if row["class"] == kind]
        gaps = [row["gap_us"] for row in subset]
        groups[kind] = dict(
            **_stats(gaps),
            fraction_pct=100 * len(subset) / len(rows) if rows else None,
            total_gap_us=sum(gaps),
            total_gap_share_pct=100 * sum(gaps) / total if total else None,
        )
    early, late = groups["early"], groups["late"]
    if early["count"] and late["count"]:
        groups["late_vs_early"] = {
            "mean_ratio": late["mean_us"] / early["mean_us"] if early["mean_us"] else None,
            "all_mean_minus_early_mean_us": groups["all"]["mean_us"] - early["mean_us"],
            "late_excess_contribution_to_all_mean_us": (
                late["count"] / len(rows) * (late["mean_us"] - early["mean_us"])
            ),
            "meaning": "descriptive decomposition, not the speedup from removing computation",
        }
    return groups


def analyze(path):
    mapping = json.loads(path.with_name("clock_alignment.json").read_text(encoding="utf-8"))
    if (
        mapping.get("status") != "calibration_validated"
        or mapping.get("calibration_layout") != "model_request_brackets"
    ):
        raise ValueError("need validated before/after calibration from this model capture")
    if Path(mapping["aligned_trace"]).resolve() != path.resolve():
        raise ValueError("alignment metadata refers to a different trace")
    data = read_trace(path)
    bindings = bind_streams(data)
    stream, reactive_pairs = bindings["reactive"]
    prefetch_pairs = bindings["prefetch"][1]
    tasks = []
    for index, event in enumerate(events(path)):
        if event.get("ph") == "X" and (str(event.get("pid")), str(event.get("tid"))) == stream:
            tasks.append(record(event, index))
            if len(tasks) > MAX_INTERVALS:
                raise ValueError("too many tasks; refuse partial analysis")
    tasks.sort(key=lambda row: (row["start"], row["index"]))
    positions = {row["index"]: i for i, row in enumerate(tasks)}
    reactive = {host["index"]: i for i, host in enumerate(data["groups"]["reactive"])}
    prefetch = {host["index"]: i for i, host in enumerate(data["groups"]["prefetch"])}
    # This is a sensitivity margin, NOT a certified confidence interval.
    margin = max(map(float, mapping["endpoint_mapping_envelope_width_us"])) + ORDER_US
    origin = float(mapping["trace_origin_us"])
    span = [origin + float(x) for x in mapping["host_span_us"]]
    rows, reasons = [], collections.Counter()
    for left, right in zip(data["all_hosts"], data["all_hosts"][1:]):
        if left["context"]["kind"] != "prefetch" or right["context"]["kind"] != "reactive":
            continue
        if left["context"]["layer"] != right["context"]["layer"]:
            reasons["adjacent_layer_mismatch"] += 1
            continue
        row = dict(
            layer=left["context"]["layer"],
            prefetch_call=left["context"]["call"],
            reactive_call=right["context"]["call"],
            gap_us=right["start"] - left["end"],
            prefetch_duration_us=left["duration"],
            **{"class": "unmapped"},
        )
        rows.append(row)
        rindex = reactive[right["index"]]
        pub = reactive_pairs[rindex][0]
        start = positions[reactive_pairs[rindex - 1][1]["index"]] + 1 if rindex else 0
        segment = tasks[start : positions[pub["index"]]]
        boundary, reason = consumer_boundary(segment)
        if boundary is None:
            reasons[reason] += 1
            row["reason"] = reason
            row["tail_names"] = [item["name"] for item in segment[-8:]]
            continue
        compute, join = boundary
        prefetch_wait = prefetch_pairs[prefetch[left["index"]]][1]
        if join["end"] < prefetch_wait["end"] - ORDER_US:
            reasons["consumer_join_ends_before_prefetch_wait"] += 1
            row["reason"] = "consumer_join_ends_before_prefetch_wait"
            continue
        # gate precedes both consumer join and the strict prefetch wait.
        upper = min(join["start"], prefetch_wait["start"])
        lower = compute["end"]
        if lower > upper + ORDER_US or not span[0] <= lower <= upper <= span[1]:
            reasons["boundary_order_or_calibration_span_invalid"] += 1
            continue
        row.update(
            **{"class": classify(lower, upper, left["end"], margin)},
            zero_margin_class=classify(lower, upper, left["end"], 0),
            readiness_minus_prefetch_end_bounds_us=[lower - left["end"], upper - left["end"]],
            last_compute_name=compute["name"],
            join_name=join["name"],
            join_ids=join["ids"],
        )
    emit("TRACE", str(path))
    emit(
        "BOUNDARY",
        dict(
            main_stream=stream,
            prefetch_stream=bindings["prefetch"][0],
            prefetch_callbacks=len(prefetch_pairs),
            adjacent_same_layer_pairs=len(rows),
            unmatched_prefetch=len(prefetch_pairs) - len(rows),
            sensitivity_margin_us=margin,
            reasons=dict(reasons),
            pairing="complete ordinal candidates; not independent slot/generation identity proof",
        ),
    )
    emit("RESULT", summarize(rows))
    emit(
        "ZERO_MARGIN_SENSITIVITY",
        summarize([dict(row, **{"class": row.get("zero_margin_class", "unmapped")}) for row in rows]),
    )
    emit("LARGEST_GAPS", sorted(rows, key=lambda row: row["gap_us"], reverse=True)[:MAX_EXAMPLES])
    emit("UNMAPPED_EXAMPLES", [row for row in rows if row["class"] == "unmapped"][:MAX_EXAMPLES])
    print("TABLE class | n | percent | mean_us | p50_us | p95_us | gap_sum_share_percent")
    for kind, values in summarize(rows).items():
        if "count" in values:
            print(
                kind
                + " | "
                + " | ".join(
                    str(values.get(key))
                    for key in ("count", "fraction_pct", "mean_us", "p50_us", "p95_us", "total_gap_share_pct")
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.run / "profile").glob("mte*/**/ASCEND_PROFILER_OUTPUT/trace_view_aligned.json"))
    if not paths:
        parser.error("no aligned MTE traces under RUN/profile; no model rerun was attempted")
    with args.output.open("x", encoding="utf-8") as output, contextlib.redirect_stdout(output):
        print("只读已有对齐profile；不导入Torch、不启动模型。仅写本摘要。单位us。")
        print("预取结束=外层host_callback结束；Host间隔=同层相邻按需开始-预取结束。")
        print("计算边界=消费者gate前主流推进（含真实路由及依赖），不是纯MoE+Attention算子耗时。")
        print("区间下界=join前最后计算任务结束；上界=min(主流join开始,预取严格wait开始)。")
        print("主流join必须有两次copy后接按需publish；不符合则unmapped，不猜测。")
        print("early/late采用校准端点包络宽度加1us做敏感性余量；不是严格误差保证。")
        print("全部保留零加载和长尾，不做IQR、不设5ms上限；uncertain/unmapped均计入总数。")
        print("适用前提：单worker、纯MTE、消费者处延后严格等待；只统计MTE组，不套用回调组。")
        for path in paths:
            try:
                analyze(path)
            except (ValueError, KeyError, OSError) as exc:
                emit("FAILED", dict(path=str(path), error=str(exc)))
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
