# SPDX-License-Identifier: Apache-2.0
"""Analyze saved, aligned offload captures; never load Torch or run inference."""

import argparse
import collections
import contextlib
import json
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from diagnose_moe_empty_load_gaps import STAGES, annotate_callbacks, validate_report
from diagnose_mte_wait_timeline import read_trace, record, task_pairs
from export_moe_aligned_profile import aligned_trace_path
from inspect_moe_profile import _stats
from moe_callback_compute_boundary import bind_callback_streams
from moe_outlier_stats import outlier_summary

ORDER_US = 1.0
MAX_EXAMPLES = 8
# ID and log2phy staging, plus optional router-weight staging for LRC.
SUPPORTED_STAGING_COPY_COUNTS = frozenset((2, 3))


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


def bind_streams(data, *, retain_unmapped=False):
    """Complete ordinal mapping, independent of the overlap being tested."""
    if data["issues"] or data["invalid_kinds"]:
        raise ValueError(f"invalid Host markers: {data['issues']}")
    bindings = {}
    checks = data["mte_order_checks"] = []
    for stream, tasks in data["tasks"].items():
        pairs, reason = task_pairs(tasks, ORDER_US)
        kinds = [k for k, hosts in data["groups"].items() if pairs and len(hosts) == len(pairs)]
        if reason != "ok" or len(kinds) != 1 or kinds[0] in bindings:
            raise ValueError(
                f"no unique complete ordinal stream mapping: stream={stream}; "
                f"reason={reason}; tasks={len(tasks)}; pairs={len(pairs)}; candidate_kinds={kinds}"
            )
        kind = kinds[0]
        for ordinal, ((publish, wait), host) in enumerate(zip(pairs, data["groups"][kind])):
            publish_overrun = publish["start"] - host["start"]
            wait_shortfall = host["end"] - wait["end"]
            valid = publish_overrun <= ORDER_US and wait_shortfall <= ORDER_US
            publish["host_coverage_valid"] = valid
            check = dict(
                kind=kind,
                stream=list(stream),
                ordinal=ordinal,
                call=host["context"]["call"],
                layer=host["context"]["layer"],
                valid=valid,
                publish_start_minus_host_start_us=publish_overrun,
                host_end_minus_wait_end_us=wait_shortfall,
                sync_end_minus_wait_end_us=host["sync"]["end"] - wait["end"],
                host_duration_us=host["duration"],
                wait_duration_us=wait["duration"],
                publish_ids=publish.get("ids", {}),
                wait_ids=wait.get("ids", {}),
            )
            checks.append(check)
            if not valid and not retain_unmapped:
                raise ValueError(f"aligned protocol order check failed: {check}")
        bindings[kind] = (stream, pairs)
    if set(bindings) != {"prefetch", "reactive"}:
        raise ValueError("need both MTE streams; callback backend is not classified with MTE rules")
    return bindings


def consumer_boundary(segment):
    """Bracket readiness using the last compute and subsequent consumer join.

    Require the baseline decode tail: Event wait, two or three copy task
    records, reactive publish. The third is compatible with optional LRC
    router-weight D2H staging; task names alone do not prove tensor identity.
    Extra compute (e.g. substitution/pruning) or
    absent task records make this boundary unavailable rather than guessed.
    The Event wait START is an upper bound, not its completion time.
    """
    joins = [i for i, row in enumerate(segment) if task_type(row["name"]) == "event_wait"]
    if not joins:
        return None, "missing_consumer_event_wait"
    index = joins[-1]
    tail = [task_type(row["name"]) for row in segment[index + 1 :]]
    if tail.count("copy") not in SUPPORTED_STAGING_COPY_COUNTS or any(kind not in ("copy", "control") for kind in tail):
        return None, "unsupported_consumer_staging_tail"
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


def restat_host_gaps(rows, classification_key="class"):
    """Descriptive residuals, not a measurement of hardware compute utilization.

    Keep undecidable samples unchanged in the all-sample result and show a
    separate decidable-only result. Never silently remove or correct them.
    Bounds reflect the saved aligned coordinates, not physical clock certainty.
    """
    output = {}
    for scope in ("all_samples", "decidable_only"):
        selected = [r for r in rows if scope == "all_samples" or r.get(classification_key) in ("early", "late")]
        kept, adjusted, low_residual, high_residual = [], [], [], []
        counts = collections.Counter()
        removed_us = 0.0
        for row in selected:
            kind = row.get(classification_key, "unmapped")
            counts[kind] += 1
            gap = row["gap_us"]
            if kind != "late":
                kept.append(gap)
                adjusted.append(gap)
                low_residual.append(gap)
                high_residual.append(gap)
                continue
            lower, upper = row["readiness_minus_prefetch_end_bounds_us"]
            if not 0 <= lower <= upper <= gap:
                # Retain the raw value if the candidate correction is invalid.
                counts["invalid_subtraction"] += 1
                adjusted.append(gap)
                low_residual.append(gap)
                high_residual.append(gap)
                continue
            deduction = (lower + upper) / 2
            adjusted.append(gap - deduction)
            low_residual.append(gap - upper)
            high_residual.append(gap - lower)
            removed_us += deduction
        output[scope] = dict(
            selected=len(selected),
            excluded=len(rows) - len(selected),
            counts=dict(counts),
            original=_stats([row["gap_us"] for row in selected]),
            method1_drop_late=_stats(kept),
            method2_subtract_remaining_compute=_stats(adjusted),
            method2_aligned_boundary_low=_stats(low_residual),
            method2_aligned_boundary_high=_stats(high_residual),
            subtracted_total_us=removed_us,
            unknown_policy="all_samples retains uncertain/unmapped unchanged; decidable_only excludes both",
        )
    return output


def render_restat(rows):
    print("HOST_RESTAT：间隔=按需开始-预取结束；晚完成扣除=消费位置时间-预取结束。")
    print("扣除的是等待计算推进的经过时间（含路由/依赖），不是累加算子busy时长；结果不是实测加速收益。")
    print("方案2取已对齐边界区间中点；low/high仅是该区间敏感性结果，不含完整时钟校准误差。")
    print(
        "TABLE_RESTAT 判定 | 范围 | 原样本数 | 晚完成 | 不确定 | 无法识别 | 原均值 | "
        "方案1 n/均值/P95 | 方案2 n/均值/P95"
    )
    for label, key in (("保守判定", "class"), ("对齐坐标直接判定", "zero_margin_class")):
        results = restat_host_gaps(rows, key)
        emit("HOST_RESTAT_RESULT", dict(classification=label, **results))
        for scope, item in results.items():

            def cell(name, group=item):
                stat = group[name]
                return f"{stat['count']}/{stat['mean_us']}/{stat['p95_us']}"

            counts = item["counts"]
            print(
                f"{label} | {scope} | {item['selected']} | {counts.get('late', 0)} | "
                f"{counts.get('uncertain', 0)} | {counts.get('unmapped', 0)} | "
                f"{item['original']['mean_us']} | {cell('method1_drop_late')} | "
                f"{cell('method2_subtract_remaining_compute')}"
            )


def load_annotations(hosts, stages, report):
    callbacks = [
        dict(
            pid=host["pid"],
            tid=host["tid"],
            start=host["start"],
            end=host["end"],
            event_index=host["index"],
            call=int(host["context"]["call"]),
            layer=int(host["context"]["layer"]),
            kind=host["context"]["kind"],
        )
        for host in hosts
    ]
    try:
        annotated = annotate_callbacks(callbacks, stages)
    except ValueError as exc:
        return {str(cb["call"]): None for cb in callbacks}, [str(exc)]
    for callback in annotated:
        updates = stages.get(callback["call"], {}).get("update_weights", [])
        callback["rows"] = updates[0]["meta"].get("rows") if len(updates) == 1 else None
    issues = validate_report(report, annotated)
    # Missing markers must never be interpreted as zero loads.
    return {str(cb["call"]): None if issues else cb["prepared"] for cb in annotated}, issues


def load_pair(prefetch_loads, reactive_loads):
    if prefetch_loads is None or reactive_loads is None:
        return "unknown"
    if prefetch_loads == 0:
        return "both_zero" if reactive_loads == 0 else "prefetch_zero_only"
    return "reactive_zero_only" if reactive_loads == 0 else "both_loaded"


def load_breakdown(rows):
    """Cross-tabulation retains unknown load status and all long-tail samples."""
    result = {}
    for timing in ("all", "early", "late", "uncertain", "unmapped"):
        subset = rows if timing == "all" else [row for row in rows if row["class"] == timing]
        total = sum(row["gap_us"] for row in subset)
        groups = {}
        for cohort in ("both_loaded", "prefetch_zero_only", "reactive_zero_only", "both_zero", "unknown"):
            values = [row["gap_us"] for row in subset if row["load_pair"] == cohort]
            groups[cohort] = dict(
                **_stats(values),
                pct_of_timing_group=100 * len(values) / len(subset) if subset else None,
                gap_sum_share_pct=100 * sum(values) / total if total else None,
            )
        result[timing] = groups
    # P(late | cohort) complements P(cohort | late); they answer different questions.
    result["late_rate_by_load_pair"] = {
        cohort: dict(
            total_count=result["all"][cohort]["count"],
            late_count=result["late"][cohort]["count"],
            unmapped_count=result["unmapped"][cohort]["count"],
            late_pct=(
                100 * result["late"][cohort]["count"] / result["all"][cohort]["count"]
                if result["all"][cohort]["count"] and not result["unmapped"][cohort]["count"]
                else None
            ),
        )
        for cohort in result["all"]
    }
    return result


def host_stage_durations(hosts, stages):
    """Retain every Host task, including initial/final and zero-load tasks."""
    result = {}
    for host in hosts:
        call = host["context"]["call"]
        parts = stages.get(int(call), {})
        durations = {"host_callback": host["duration"]}
        for stage in ("update_weights", "load_submit", "copy_submit", "synchronize"):
            entries = parts.get(stage, [])
            valid = (
                len(entries) == 1
                and entries[0]["tid"] == host["tid"]
                and all(entries[0]["meta"].get(k) == host["context"].get(k) for k in ("kind", "layer", "rows"))
                and host["start"] <= entries[0]["start"] <= entries[0]["end"] <= host["end"]
            )
            durations[stage] = entries[0]["end"] - entries[0]["start"] if valid else None
        result[call] = durations
    return result


def analyze(
    path,
    backend="mte",
    *,
    profile_report=None,
    update_backend="python",
    exact_relative=False,
    retain_unmapped=False,
    estimate_order_mismatches=False,
):
    if backend not in ("mte", "callback"):
        raise ValueError("unknown Host backend")
    if update_backend not in ("python", "native") or (exact_relative and backend != "mte"):
        raise ValueError("unsupported update backend or relative callback analysis")
    if estimate_order_mismatches and (backend != "mte" or not retain_unmapped or not exact_relative):
        raise ValueError("coordinate estimates require exact relative MTE analysis with retained strict diagnostics")
    mapping = json.loads(path.with_name("clock_alignment.json").read_text(encoding="utf-8"))
    if (
        mapping.get("status") != "calibration_validated"
        or mapping.get("calibration_layout") != "model_request_brackets"
    ):
        raise ValueError("need validated before/after calibration from this model capture")
    if Path(mapping["aligned_trace"]).resolve() != path.resolve():
        raise ValueError("alignment metadata refers to a different trace")
    origin = Decimal(mapping["trace_origin_us"]) if exact_relative else None
    data = read_trace(path, time_origin=origin) if exact_relative else read_trace(path)
    if profile_report is not None and profile_report.get("count") != data["host_count"]:
        raise ValueError("saved counters and trace Host counts differ")
    # Invalid Host identities/nesting still reject the capture. Cross-clock
    # mapping failures must not discard independently valid Host durations.
    if data["issues"] or data["invalid_kinds"] or not data["all_hosts"]:
        raise ValueError(f"invalid Host markers: {data['issues']}")
    binding_error = None
    try:
        bindings = (
            bind_streams(data, retain_unmapped=retain_unmapped)
            if backend == "mte"
            else bind_callback_streams(path, data)
        )
    except ValueError as exc:
        if not retain_unmapped or backend != "mte":
            raise
        binding_error = str(exc)
        bindings = {}
    stream, reactive_pairs = bindings.get("reactive", (None, []))
    prefetch_stream, prefetch_pairs = bindings.get("prefetch", (None, []))
    tasks = []
    stages = collections.defaultdict(lambda: collections.defaultdict(list))
    host_pid = data["all_hosts"][0]["pid"]
    retained_stages = 0
    for index, event in enumerate(events(path, parse_float=Decimal if exact_relative else float)):
        if event.get("ph") == "X" and (str(event.get("pid")), str(event.get("tid"))) == stream:
            tasks.append(record(event, index, time_origin=origin))
            if len(tasks) > MAX_INTERVALS:
                raise ValueError("too many tasks; refuse partial analysis")
        name = str(event.get("name", ""))
        stage = name.split("|", 1)[0].removeprefix("moe_offload/")
        if (
            event.get("ph") == "X"
            and str(event.get("pid")) == host_pid
            and name.startswith("moe_offload/")
            and stage in STAGES
        ):
            meta = fields(event)
            rec = record(event, index, time_origin=origin)
            stages[int(meta["call"])][stage].append(
                dict(
                    start=rec["start"],
                    end=rec["end"],
                    tid=rec["tid"],
                    meta=meta,
                )
            )
            retained_stages += 1
            if retained_stages + len(tasks) > MAX_INTERVALS:
                raise ValueError("too many retained tasks/stages")
    # RUN/profile/CASE/POLICY/CAPTURE/ASCEND_PROFILER_OUTPUT/trace_view_aligned.json
    report_path = path.parents[5] / path.parents[3].name / "profile_report.json" if len(path.parents) > 5 else None
    report = profile_report
    if report is None and report_path and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    if update_backend == "native":
        # Native submit has one range per batch, no per-expert prepare/refresh
        # ranges. Their absence cannot be interpreted as zero experts loaded.
        loads = {host["context"]["call"]: None for host in data["all_hosts"]}
        load_issues = ["native capture has no per-expert load counts; load status remains unknown"]
    else:
        loads, load_issues = load_annotations(data["all_hosts"], stages, report)
    emit(
        "LOAD_EVIDENCE",
        dict(
            report="supplied capture counters" if profile_report is not None else str(report_path),
            issues=load_issues,
            basis="per-call load_prepare/scale_refresh counts checked against report loads",
        ),
    )
    tasks.sort(key=lambda row: (row["start"], row["index"]))
    durations = host_stage_durations(data["all_hosts"], stages)
    host_timings, host_timing_counts = {}, {}
    for kind in ("all", "prefetch", "reactive"):
        selected = [h for h in data["all_hosts"] if kind == "all" or h["context"]["kind"] == kind]
        host_timing_counts[kind] = len(selected)
        host_timings[kind] = {}
        for stage in ("host_callback", "update_weights", "load_submit", "copy_submit", "synchronize"):
            values = [value for h in selected if (value := durations[h["context"]["call"]][stage]) is not None]
            host_timings[kind][stage] = dict(**_stats(values), outliers_us=outlier_summary(values))
    positions = {row["index"]: i for i, row in enumerate(tasks)}
    reactive = {host["index"]: i for i, host in enumerate(data["groups"]["reactive"])}
    prefetch = {host["index"]: i for i, host in enumerate(data["groups"]["prefetch"])}
    # This is a sensitivity margin, NOT a certified confidence interval.
    margin = max(map(float, mapping["endpoint_mapping_envelope_width_us"])) + ORDER_US
    epoch = 0.0 if exact_relative else float(mapping["trace_origin_us"])
    span = [epoch + float(x) for x in mapping["host_span_us"]]
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
            prefetch_rows=left["context"]["rows"],
            reactive_rows=right["context"]["rows"],
            gap_us=right["start"] - left["end"],
            prefetch_duration_us=left["duration"],
            reactive_duration_us=right["duration"],
            **{"class": "unmapped"},
            prefetch_loads=loads[left["context"]["call"]],
            reactive_loads=loads[right["context"]["call"]],
        )
        row["load_pair"] = load_pair(row["prefetch_loads"], row["reactive_loads"])
        for role, host in (("prefetch", left), ("reactive", right)):
            row[role + "_stages_us"] = durations[host["context"]["call"]]
        rows.append(row)
        if binding_error:
            row["reason"] = "mte_stream_mapping_unavailable"
            reasons[row["reason"]] += 1
            continue
        rindex = reactive[right["index"]]
        pindex = prefetch[left["index"]]
        if backend == "callback" or retain_unmapped:
            required = [("prefetch", prefetch_pairs[pindex]), ("reactive", reactive_pairs[rindex])]
            if rindex:
                required.append(("previous_reactive", reactive_pairs[rindex - 1]))
            invalid = [role for role, pair in required if not pair[0]["host_coverage_valid"]]
            if invalid:
                row["reason"] = (
                    "callback_gate_host_order_mismatch" if backend == "callback" else "mte_protocol_host_order_mismatch"
                )
                row["invalid_gate_roles"] = invalid
                if not estimate_order_mismatches:
                    reasons[row["reason"]] += 1
                    continue
                # Separate, explicitly conditional coordinate estimate. The
                # strict result and failed protocol checks remain unchanged.
                row["coordinate_estimate_only"] = True
                row["order_warning"] = row.pop("reason")
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
        prefetch_wait = prefetch_pairs[pindex][1]
        if join["end"] < prefetch_wait["end"] - ORDER_US:
            reasons["consumer_join_ends_before_prefetch_wait"] += 1
            row["reason"] = "consumer_join_ends_before_prefetch_wait"
            continue
        # Native callback begins on prefetch stream before its consumer is reached;
        # its gate START cannot bound main-stream readiness. Only MTE late-wait can.
        upper = min(join["start"], prefetch_wait["start"]) if backend == "mte" else join["start"]
        lower = compute["end"]
        row["boundary_diagnostic_us"] = dict(
            lower_minus_prefetch_end=lower - left["end"],
            upper_minus_prefetch_end=upper - left["end"],
            main_join_start_minus_prefetch_end=join["start"] - left["end"],
            prefetch_device_gate_start_minus_prefetch_end=prefetch_wait["start"] - left["end"],
            prefetch_wait_start_minus_prefetch_end=(prefetch_wait["start"] - left["end"] if backend == "mte" else None),
            lower_minus_calibration_start=lower - span[0],
            calibration_end_minus_upper=span[1] - upper,
        )
        failure = (
            "inverted_readiness_bounds"
            if lower > upper
            else "outside_calibration_span"
            if not span[0] <= lower <= upper <= span[1]
            else None
        )
        if failure:
            reasons[failure] += 1
            row["reason"] = failure
            continue
        row.update(
            **{"class": classify(lower, upper, left["end"], margin)},
            zero_margin_class=classify(lower, upper, left["end"], 0),
            readiness_minus_prefetch_end_bounds_us=[lower - left["end"], upper - left["end"]],
            last_compute_name=compute["name"],
            join_name=join["name"],
            join_ids=join["ids"],
            staging_copy_count=sum(task_type(item["name"]) == "copy" for item in segment[segment.index(join) + 1 :]),
        )
        if rindex:
            anchor = data["groups"]["reactive"][rindex - 1]["end"]
            row["previous_reactive_anchor_us"] = dict(
                prefetch_start=left["start"] - anchor,
                prefetch_end=left["end"] - anchor,
                compute_lower=lower - anchor,
                compute_upper=upper - anchor,
                reactive_start=right["start"] - anchor,
            )
    emit("TRACE", str(path))
    mapped_count = sum(row["class"] != "unmapped" for row in rows)
    emit(
        "CLASSIFICATION_STATUS",
        dict(
            backend=backend,
            status="complete" if mapped_count == len(rows) and rows else "partial" if mapped_count else "unavailable",
            mapped=mapped_count,
            total=len(rows),
            meaning="unmapped samples have no early/late decision; zero recognized late is not zero actual late",
        ),
    )
    emit(
        "BOUNDARY",
        dict(
            main_stream=stream,
            prefetch_stream=prefetch_stream,
            prefetch_callbacks=len(data["groups"]["prefetch"]),
            adjacent_same_layer_pairs=len(rows),
            unmatched_prefetch=len(data["groups"]["prefetch"]) - len(rows),
            binding_error=binding_error,
            sensitivity_margin_us=margin,
            reasons=dict(reasons),
            pairing="complete ordinal candidates; not independent slot/generation identity proof",
        ),
    )
    emit("RESULT", summarize(rows))
    emit("LOAD_CROSS_TAB", load_breakdown(rows))
    emit(
        "ZERO_MARGIN_LATE_LOADS",
        load_breakdown([dict(row, **{"class": row.get("zero_margin_class", "unmapped")}) for row in rows])["late"],
    )
    print("LATE_LOAD_TABLE 状态 | 数量 | 占晚完成样本% | 间隔均值us | P95us | 占晚完成间隔总时长%")
    for cohort, values in load_breakdown(rows)["late"].items():
        print(
            cohort
            + " | "
            + " | ".join(
                str(values.get(key))
                for key in ("count", "pct_of_timing_group", "mean_us", "p95_us", "gap_sum_share_pct")
            )
        )
    late = [row for row in rows if row["class"] == "late"]
    zero_prefetch = [row for row in late if row["prefetch_loads"] == 0]
    zero_either = [row for row in late if row["prefetch_loads"] == 0 or row["reactive_loads"] == 0]
    emit(
        "LATE_LOAD_COUNTS",
        dict(
            late_total=len(late),
            prefetch_zero=len(zero_prefetch),
            prefetch_zero_pct_of_all_late=100 * len(zero_prefetch) / len(late) if late else None,
            reactive_zero=sum(row["reactive_loads"] == 0 for row in late),
            either_zero=len(zero_either),
            either_zero_pct_of_all_late=100 * len(zero_either) / len(late) if late else None,
            both_zero=sum(row["load_pair"] == "both_zero" for row in late),
            unknown=sum(row["load_pair"] == "unknown" for row in late),
            meaning="association, not proof that zero loads caused late computation",
        ),
    )
    emit("LATE_ZERO_EXAMPLES", sorted(zero_either, key=lambda row: row["gap_us"], reverse=True)[:MAX_EXAMPLES])
    emit(
        "ZERO_MARGIN_SENSITIVITY",
        summarize([dict(row, **{"class": row.get("zero_margin_class", "unmapped")}) for row in rows]),
    )
    emit("LARGEST_GAPS", sorted(rows, key=lambda row: row["gap_us"], reverse=True)[:MAX_EXAMPLES])
    emit("UNMAPPED_EXAMPLES", [row for row in rows if row["class"] == "unmapped"][:MAX_EXAMPLES])
    render_restat(rows)
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

    return dict(
        backend=backend,
        update_backend=update_backend,
        estimate_order_mismatches=estimate_order_mismatches,
        time_arithmetic="decimal epoch subtraction then relative float" if exact_relative else "absolute float",
        host_timings=host_timings,
        host_timing_expected_counts=host_timing_counts,
        callback_gate_checks=data.get("callback_gate_checks", []),
        mte_order_checks=data.get("mte_order_checks", []),
        binding_error=binding_error,
        trace=str(path),
        rows=rows,
        classification=summarize(rows),
        direct_classification=summarize(
            [dict(row, **{"class": row.get("zero_margin_class", "unmapped")}) for row in rows]
        ),
        late_loads=load_breakdown(rows)["late"],
        conservative=restat_host_gaps(rows),
        direct=restat_host_gaps(rows, "zero_margin_class"),
        sensitivity_margin_us=margin,
        load_issues=load_issues,
        reasons=dict(reasons),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.run / "profile").glob("mte*/**/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if not paths:
        parser.error("no aligned MTE traces under RUN/profile; no model rerun was attempted")
    with args.output.open("x", encoding="utf-8") as output, contextlib.redirect_stdout(output):
        print("只读已有对齐profile；不导入Torch、不启动模型。仅写本摘要。单位us。")
        print("预取结束=外层host_callback结束；Host间隔=同层相邻按需开始-预取结束。")
        print("计算边界=消费者gate前主流推进（含真实路由及依赖），不是纯MoE+Attention算子耗时。")
        print("区间下界=join前最后计算任务结束；上界=min(主流join开始,预取严格wait开始)。")
        print("主流join后识别2/3次copy再接按需publish；第三次与LRC路由权重回传路径相容，不按名称猜张量身份。")
        print("early/late采用校准端点包络宽度加1us做敏感性余量；不是严格误差保证。")
        print("全部保留零加载和长尾，不做IQR、不设5ms上限；uncertain/unmapped均计入总数。")
        print("零加载按同次采集call ID的准备/scale标记及report加载总数核验，不按时长猜测；缺证据=unknown。")
        print("重点看预取零加载占晚完成样本比例；后续按需零加载仅为关联信息，不能解释为此前间隔的原因。")
        print(
            "零加载四类：both_loaded双加载、prefetch_zero_only仅预取零加载、reactive_zero_only仅按需零加载、both_zero双零加载。"
        )
        print("适用前提：单worker、纯MTE、消费者处延后严格等待；只统计MTE组，不套用回调组。")
        for path in paths:
            try:
                analyze(aligned_trace_path(path))
            except (ValueError, KeyError, OSError) as exc:
                emit("FAILED", dict(path=str(path), error=str(exc)))
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
