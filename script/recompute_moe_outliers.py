# SPDX-License-Identifier: Apache-2.0
"""只读重算已有实验的离群值敏感性统计，不启动模型、不写文件。"""

import argparse
import collections
import json
import math
import time
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from inspect_moe_profile import _interval
from moe_outlier_stats import MAX_REMOVED_DISPLAY, filtered_stats, group_outliers

PROGRESS_SECONDS = 30
STAGES = frozenset(
    (
        "host_callback",
        "update_weights",
        "observe",
        "choose_victims",
        "synchronize",
        "load_submit",
        "load_prepare",
        "copy_submit",
        "scale_refresh",
        "fused_plan",
        "gap_mapping",
        "gap_missing",
        "gap_hit_stats",
        "gap_cache_stats",
        "gap_slots",
        "gap_debug",
    )
)


def emit(label, values, unit="us"):
    raw, clean, removed, bounds, reason = filtered_stats(values)
    if raw is None:
        print(f"{label}: NO_SAMPLES", flush=True)
        return

    def pack(row):
        return "/".join(f"{row[k]:.3f}" for k in ("mean", "p50", "p95", "max", "sd", "total"))

    limits = "none" if bounds is None else f"{bounds[0]:.3f},{bounds[1]:.3f}"
    print(
        f"{label} [{unit}] n={raw['n']}->{clean['n']}; raw={pack(raw)}; clean={pack(clean)}; "
        f"bounds={limits}; rule={reason}",
        flush=True,
    )
    if removed:
        top = sorted(removed, key=lambda x: x[1], reverse=True)[:MAX_REMOVED_DISPLAY]
        print(f"  removed={len(removed)}; largest(index,value)={[(i, round(v, 3)) for i, v in top]}", flush=True)


def copy_gaps(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return [], 0
    covered = ordered[0][1]
    gaps, overlaps = [], 0
    for start, end in ordered[1:]:
        if start < covered:
            overlaps += 1
        else:
            gaps.append(start - covered)
        covered = max(covered, end)
    return gaps, overlaps


def read_trace(path):
    """全文件验证通过才返回统计样本；不把截断文件的前半段作为完整实验。"""
    processes, threads, calls = {}, {}, {}
    copies = collections.defaultdict(list)
    retained = 0
    last_progress = time.monotonic()
    for event in events(path):
        name, phase = str(event.get("name", "")), event.get("ph")
        pid = "" if event.get("pid") is None else str(event["pid"])
        tid = "" if event.get("tid") is None else str(event["tid"])
        if phase == "M":
            args = event.get("args", {})
            if name == "process_name":
                processes[pid] = str(args.get("name", ""))
            elif name == "thread_name":
                threads[pid, tid] = str(args.get("name", ""))
        elif phase == "X" and name.startswith("moe_offload/"):
            stage = name.split("|", 1)[0].removeprefix("moe_offload/")
            if stage in STAGES:
                meta = fields(event)
                if not pid or not tid or not meta.get("call") or meta.get("kind") not in ("prefetch", "reactive"):
                    raise ValueError("MSTX缺少pid/call/kind，无法可靠配对")
                identity = meta["kind"], meta.get("rows", "unknown"), meta.get("layer", "unknown"), tid
                record = calls.setdefault((pid, meta["call"]), {"identity": identity, "stages": {}})
                if record["identity"] != identity:
                    raise ValueError("同一pid/call出现不一致的线程或分组身份")
                intervals = record["stages"].setdefault(stage, [])
                if stage in ("host_callback", "update_weights") and intervals:
                    raise ValueError("重复host_callback/update_weights标记")
                intervals.append(_interval(event))
                retained += 1
        elif phase == "X" and "MEMCPY_ASYNC" in name.upper():
            copies[pid, tid].append(_interval(event))
            retained += 1
        if retained + len(processes) + len(threads) > MAX_INTERVALS:
            raise ValueError("相关区间超过内存上限；未输出部分统计")
        if time.monotonic() - last_progress >= PROGRESS_SECONDS:
            print(f"READING {path.parent.parent.name}; retained={retained}", flush=True)
            last_progress = time.monotonic()

    durations = collections.defaultdict(list)
    callbacks = collections.defaultdict(list)
    for (pid, call), record in calls.items():
        kind, rows, _, _ = record["identity"]
        stages = record["stages"]
        if "host_callback" not in stages or "update_weights" not in stages:
            raise ValueError("回调与update_weights标记未完整配对")
        host_start, host_end = stages["host_callback"][0]
        start, end = stages["update_weights"][0]
        if not host_start <= start <= end <= host_end:
            raise ValueError("回调嵌套关系无效")
        callbacks[pid].append((host_start, call, host_end, kind))
        sums = {}
        for stage, intervals in stages.items():
            if stage not in ("host_callback", "update_weights") and any(a < start or b > end for a, b in intervals):
                raise ValueError("分项越出update_weights区间")
            _, overlaps = copy_gaps(intervals)
            if overlaps:
                raise ValueError("同一回调同名分项有重叠，不能相加")
            sums[stage] = sum(b - a for a, b in intervals)
            durations[pid, kind, rows, stage].append(sums[stage])
        # 仅从同一回调内完整存在的标记推导，绝不以汇总均值伪造样本。
        if "load_submit" in sums and "synchronize" in sums:
            durations[pid, kind, rows, "load_and_wait_mstx"].append(sums["load_submit"] + sums["synchronize"])

    gaps = collections.defaultdict(list)
    overlaps = collections.Counter()
    for pid, records in callbacks.items():
        ordered = sorted(records)
        for previous, current in zip(ordered, ordered[1:]):
            direction = f"{previous[3]}->{current[3]}"
            if previous[3] == current[3]:
                continue
            gap = current[0] - previous[2]
            if gap < 0:
                overlaps[pid, direction] += 1
            else:
                gaps[pid, direction].append(gap)
    hardware = {pid for pid, name in processes.items() if "ascend hardware" in name.lower()}
    device = {key: value for key, value in copies.items() if key[0] in hardware}
    return durations, gaps, overlaps, device, threads, len(calls)


def analyze_trace(path, threshold):
    print(f"TRACE {path}; bytes={path.stat().st_size}", flush=True)
    durations, gaps, overlaps, device, threads, count = read_trace(path)
    print(f"TRACE_JSON_COMPLETE; paired_callbacks={count}; copy_tracks={len(device)}", flush=True)
    if not count or not device:
        print("MISSING: 缺少有效MSTX或设备轨道，本文件不构成完整分项/间隔分析", flush=True)
    for (pid, direction), values in sorted(gaps.items()):
        print(f"HOST_SCOPE pid={pid}; direction={direction}; overlaps_excluded={overlaps[pid, direction]}")
        emit(f"HOST_GAP {direction}", values)
    for (pid, tid), intervals in sorted(device.items()):
        values, overlap_count = copy_gaps(intervals)
        print(
            f"COPY_SCOPE pid={pid} tid={tid} name={threads.get((pid, tid), '')}; "
            f"copies={len(intervals)} overlaps={overlap_count}"
        )
        emit("MEMCPY_GAP_ALL", values)
        emit(f"MEMCPY_GAP_GE_{threshold:g}", [x for x in values if x >= threshold])
    for (pid, kind, rows, stage), values in sorted(durations.items()):
        emit(f"MSTX pid={pid} {kind}/rows={rows}/{stage}", values)
    print("MSTX含标记开销，与perf_counter分开列；仅统计存在的标记，没有标记不视为0。")


def discover(roots):
    reports = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"目录不存在：{root}")
        reports.update(path.resolve() for path in root.rglob("profile_report.json"))
        reports.update(
            path.resolve()
            for path in root.rglob("cost_report.json")
            if not path.with_name("profile_report.json").exists()
        )
        reports.update(
            path.resolve()
            for path in root.rglob("request_window.json")
            if not any(path.with_name(name).exists() for name in ("profile_report.json", "cost_report.json"))
        )
    return sorted(reports)


def analyze_report(path, threshold):
    print(f"\nREPORT {path}", flush=True)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(
        "IDENTITY "
        + json.dumps(
            {
                key: report.get(key)
                for key in (
                    "host_backend",
                    "cache_policy_backend",
                    "moe_offload_debug",
                    "worker_policy",
                    "profile_status",
                    "errors",
                    "dropped",
                )
            },
            ensure_ascii=False,
        )
    )
    window = report.get("request_window")
    if window is None:
        sidecar = path.with_name("request_window.json")
        if sidecar.is_file():
            window = report if path.name == sidecar.name else json.loads(sidecar.read_text(encoding="utf-8"))
    if isinstance(window, dict) and isinstance(window.get("samples_us"), list):
        if window.get("count") != len(window["samples_us"]):
            raise ValueError("请求样本数与count不符")
        if any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
            for x in window["samples_us"]
        ):
            raise ValueError("逐请求时延不是有限正数")
        emit("REQUEST_NO_PROFILE", [x / 1000 for x in window["samples_us"]], "ms")
    else:
        print("MISSING: 无逐请求无profiler样本，不能重算端到端离群值。")
    if isinstance(report.get("request_samples_us"), list):
        emit("REQUEST_WITH_TIMERS", [x / 1000 for x in report["request_samples_us"]], "ms")
    missing_exact = 0
    for group in report.get("groups", []):
        rows = group_outliers(group)
        if not rows:
            missing_exact += 1
            continue
        columns = {**group["metric_samples_us"], **group.get("gap_samples_us", {})}
        if "load_and_wait" in rows:
            columns["load_and_wait"] = [a + b for a, b in zip(columns["load_submit"], columns["synchronize"])]
        for metric, values in sorted(columns.items()):
            emit(f"EXACT_TIMER {group['kind']}/rows={group['token_rows']}/{metric}", values)
    if missing_exact or not report.get("groups"):
        print(
            "MISSING_EXACT_TIMERS: 旧报告缺少perf_counter逐回调样本；"
            "outer/host_without_sync/load_other等原计时列无法精确重算，不能用MSTX替换。"
        )
    if report.get("errors", 0) or report.get("dropped", 0):
        print("INVALID_COMPARISON: 原报告存在错误或丢样；以下只能作诊断，不能认定收益。")
    print("统计按实验/服务/worker/方向或kind+rows分别进行，不跨实验合并，不自动证明输出一致。")
    # 标准runner目录结构；忽略报告中可能已经过期的绝对路径。
    profile = path.parent.parent / "profile" / path.parent.name
    traces = sorted(profile.rglob("ASCEND_PROFILER_OUTPUT/trace_view.json")) if profile.exists() else []
    if not traces:
        print(f"MISSING: 无完整时间线候选，预期目录={profile}")
    failures = 0
    for trace in traces:
        try:
            analyze_trace(trace, threshold)
        except (OSError, UnicodeError, ValueError) as exc:
            failures += 1
            print(f"TRACE_FAILED: {trace}: {type(exc).__name__}: {exc}; 不采用部分文件样本", flush=True)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="实验目录或包含多个历史实验的父目录，可一次提供多个")
    parser.add_argument("--large-gap-us", type=float, default=100.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.large_gap_us) or args.large_gap_us <= 0:
        parser.error("--large-gap-us必须为有限正数")
    print("只读；不导入Torch、不占NPU、不写文件。所有结果输出控制台。", flush=True)
    print("各组同一规则：3×IQR；n<8或IQR=0不剔除。事后敏感性分析，离群不等于无效。")
    print("raw/clean列顺序=mean/p50/p95/max/总体标准差/sum；bounds为原始样本计算的剔除界限。")
    print("逐指标独立筛选，保留样本可能不同，clean分项不能相加；大间隔先按阈值筛选再独立计算界限。")
    reports = discover(args.roots)
    print(f"REPORTS={len(reports)}", flush=True)
    if not reports:
        print("MISSING: 未找到profile_report.json/cost_report.json/request_window.json；仅有汇总无法恢复逐次样本。")
        return 1
    failures = 0
    for path in reports:
        try:
            failures += analyze_report(path, args.large_gap_us)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            failures += 1
            print(f"REPORT_FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"完成：reports={len(reports)}; parse_failures={failures}。缺样本/时间线失败项不可声称完成剔除。")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
