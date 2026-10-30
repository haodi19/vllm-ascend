# SPDX-License-Identifier: Apache-2.0
"""只读核对已有 MoE 时间线的可视化内容；不启动模型、不写文件。"""

import argparse
import collections
import json
import math
import re
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields

MAX_TRACE_FILES = 32
MAX_PRINT_STREAMS = 32
MAX_LOG_LINE_CHARS = 4096


def _interval(event):
    values = []
    for key in ("ts", "dur"):
        value = event.get(key)
        try:
            number = float(value) if not isinstance(value, bool) else float("nan")
        except (ValueError, TypeError):
            number = float("nan")
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"X 型事件的 {key} 不是有限非负数")
        values.append(number)
    if not math.isfinite(sum(values)):
        raise ValueError("X 型事件结束时间溢出")
    return values[0], values[0] + values[1]


def _task_kind(name):
    upper = name.upper()
    if "HOST_SYNC_PUBLISH" in upper:
        return "publish"
    if "HOST_SYNC_WAIT" in upper:
        return "wait"
    if "MEMCPY_ASYNC" in upper:
        return "copy"
    if not name or any(word in upper for word in ("EVENT", "WAIT", "MEMCPY", "MEMSET", "PUBLISH", "NOTIFY")):
        return "other"
    return "compute_candidate"


def _observe(summary, duration):
    summary["count"] += 1
    summary["total_us"] += duration
    summary["max_us"] = max(summary["max_us"], duration)
    if not math.isfinite(summary["total_us"]):
        raise ValueError("时间线时长汇总溢出")


def _stats(values):
    if not values:
        return dict(count=0, mean_us=None, p50_us=None, p95_us=None, max_us=None)
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return dict(
        count=len(ordered),
        mean_us=sum(ordered) / len(ordered),
        p50_us=percentile(0.5),
        p95_us=percentile(0.95),
        max_us=ordered[-1],
    )


def scan_trace(path, *, require_mte=False):
    """完整消费 JSON；核对基本轨道/嵌套，不证明所有数据或时钟关联正确。"""
    path = Path(path)
    process_names, thread_names, tracks = {}, {}, {}
    callbacks, updates = {}, {}
    phases = collections.Counter()
    issues = []
    missing_identity = duplicates = event_count = 0
    for event in events(path):
        event_count += 1
        name = str(event.get("name", ""))
        pid = "" if event.get("pid") is None else str(event["pid"])
        tid = "" if event.get("tid") is None else str(event["tid"])
        phase = str(event.get("ph", ""))
        if phase == "M":
            args = event.get("args", {})
            if isinstance(args, dict):
                if name == "process_name":
                    process_names[pid] = str(args.get("name", ""))
                elif name == "thread_name":
                    thread_names[(pid, tid)] = str(args.get("name", ""))
        if name.startswith("moe_offload/"):
            phases[phase] += 1
        if phase == "X":
            start, end = _interval(event)
            track = tracks.setdefault((pid, tid), {})
            kind = _task_kind(name)
            summary = track.setdefault(kind, dict(count=0, total_us=0.0, max_us=0.0))
            _observe(summary, end - start)
            target = None
            if name.startswith("moe_offload/host_callback|"):
                target = callbacks
            elif name.startswith("moe_offload/update_weights|"):
                target = updates
            if target is not None:
                info = fields(event)
                call = info.get("call")
                if not pid or not call:
                    missing_identity += 1
                elif (pid, call) in target:
                    duplicates += 1
                else:
                    target[(pid, call)] = dict(start=start, end=end, tid=tid, kind=info.get("kind", "unknown"))
        if len(callbacks) + len(updates) + len(tracks) + len(process_names) + len(thread_names) > MAX_INTERVALS:
            raise ValueError("轨道或回调区间超过分析上限；请使用更短采集")

    # Metadata may be emitted after all device tasks. Select tracks only after
    # the entire JSON has been parsed, including closing delimiters and EOF.
    hardware = {pid for pid, name in process_names.items() if "ascend hardware" in name.lower()}
    device_counts = collections.Counter()
    streams = []
    for (pid, tid), task_groups in sorted(tracks.items()):
        if pid not in hardware:
            continue
        for kind, summary in task_groups.items():
            device_counts[kind] += summary["count"]
            summary["mean_us"] = summary["total_us"] / summary["count"]
        streams.append(dict(pid=pid, tid=tid, name=thread_names.get((pid, tid), ""), tasks=task_groups))

    matched = invalid_nesting = 0
    for key in callbacks.keys() & updates.keys():
        outer, inner = callbacks[key], updates[key]
        if outer["tid"] == inner["tid"] and outer["start"] <= inner["start"] <= inner["end"] <= outer["end"]:
            matched += 1
        else:
            invalid_nesting += 1
    nesting = dict(
        matched=matched,
        invalid=invalid_nesting,
        callback_only=len(callbacks.keys() - updates.keys()),
        update_only=len(updates.keys() - callbacks.keys()),
        duplicate_ids=duplicates,
        missing_ids=missing_identity,
    )
    host_by_pid = collections.defaultdict(list)
    for (pid, call), event in callbacks.items():
        host_by_pid[pid].append((event["start"], call, event))
    gaps = []
    for pid, records in sorted(host_by_pid.items()):
        by_direction, overlaps = collections.defaultdict(list), collections.Counter()
        records.sort(key=lambda item: (item[0], item[1]))
        for (_, _, previous), (_, _, current) in zip(records, records[1:]):
            direction = f"{previous['kind']}->{current['kind']}"
            if direction not in ("prefetch->reactive", "reactive->prefetch"):
                continue
            gap = current["start"] - previous["end"]
            if gap < 0:
                overlaps[direction] += 1
            else:
                by_direction[direction].append(gap)
        for direction in sorted(set(by_direction) | set(overlaps)):
            gaps.append(
                dict(
                    pid=pid,
                    direction=direction,
                    overlaps_excluded=overlaps[direction],
                    **_stats(by_direction[direction]),
                )
            )

    if not hardware:
        issues.append("没有 Ascend Hardware 进程元数据，无法确认设备轨道")
    if not device_counts["copy"]:
        issues.append("设备轨道没有有效的 X 型 MEMCPY_ASYNC")
    if not device_counts["compute_candidate"]:
        issues.append("设备轨道没有可识别的计算候选任务")
    if not callbacks or not updates:
        issues.append("缺少 X 型 host_callback 或 update_weights；标签字符串出现不等于完整区间")
    if not matched:
        issues.append("没有按进程和 call 配对且时间范围正确的 callback/update 嵌套")
    if invalid_nesting or duplicates or missing_identity:
        issues.append("存在无效嵌套、重复 call 或缺失进程/call 标识")
    if require_mte and (not device_counts["publish"] or not device_counts["wait"]):
        issues.append("MTE 设备轨道缺少实际 X 型 host_sync_publish 或 host_sync_wait")
    return dict(
        json_complete=True,
        usable=not issues,
        issues=issues,
        events=event_count,
        hardware_pids=sorted(hardware),
        device_counts=dict(device_counts),
        mstx_phases=dict(phases),
        host_callbacks=len(callbacks),
        updates=len(updates),
        nesting=nesting,
        streams=streams,
        host_gaps=gaps,
        scope="basic visualization content only; Host gaps are not NPU bubbles or causal attribution",
    )


def _safe_log_label(line):
    # Never echo request bodies, tensor arguments or arbitrary service text.
    if "External init callback must run in same thread" in line:
        match = re.search(r"\(-?\d+ != -?\d+\)", line)
        return "External init callback/registerClient 线程不一致" + (" " + match[0] if match else "")
    for token in ("EngineDeadError", "cleanup_begin", "start_profile", "stop_profile", "profiler.py"):
        if token in line:
            return token
    level = re.search(r"\b(INFO|WARNING|ERROR|DEBUG)\b", line)
    return (level[0] if level else "日志行") + "（原文省略）"


def service_warning_context(path):
    """At most two preceding and two following sanitized lines per error kind."""
    if not path.is_file():
        return []
    previous = collections.deque(maxlen=2)
    contexts = {}
    with path.open(encoding="utf-8", errors="replace") as source:
        number = 0
        while True:
            line = source.readline(MAX_LOG_LINE_CHARS)
            if not line:
                break
            number += 1
            while not line.endswith("\n") and len(line) == MAX_LOG_LINE_CHARS:
                remainder = source.readline(MAX_LOG_LINE_CHARS)
                if not remainder or remainder.endswith("\n") or len(remainder) < MAX_LOG_LINE_CHARS:
                    break
            item = dict(line=number, label=_safe_log_label(line))
            for record in contexts.values():
                if number <= record["line"] + 2:
                    record["context"].append(item)
            for kind, needle in (
                ("external_init", "External init callback must run in same thread"),
                ("engine_dead", "EngineDeadError"),
            ):
                if needle in line and kind not in contexts:
                    contexts[kind] = dict(kind=kind, line=number, context=[*previous, item])
            previous.append(item)
    return list(contexts.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    print(f"READ ONLY PROFILE AUDIT: {root}", flush=True)
    print("不启动模型、不导入 Torch、不修改采集；JSON 会读到文件末尾。单位 us。", flush=True)
    complete = True
    for backend in ("callback", "mte"):
        directory = root / "profile" / backend / "python"
        traces = []
        for path in directory.rglob("trace_view.json"):
            traces.append(path)
            if len(traces) > MAX_TRACE_FILES:
                break
        print(f"\nBACKEND={backend}; trace_files={len(traces)}", flush=True)
        if not traces or len(traces) > MAX_TRACE_FILES:
            print("UNUSABLE: 没有时间线，或文件数超过只读检查上限。", flush=True)
            complete = False
            continue
        for path in sorted(traces):
            print(f"TRACE {path.relative_to(root)}; bytes={path.stat().st_size}", flush=True)
            try:
                report = scan_trace(path, require_mte=backend == "mte")
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"UNUSABLE: {type(exc).__name__}: {exc}", flush=True)
                complete = False
                continue
            facts = {key: value for key, value in report.items() if key not in ("streams", "host_gaps", "scope")}
            print("TRACE_CHECK " + json.dumps(facts, ensure_ascii=False), flush=True)
            for stream in report["streams"][:MAX_PRINT_STREAMS]:
                print("DEVICE_STREAM " + json.dumps(stream, ensure_ascii=False), flush=True)
            if len(report["streams"]) > MAX_PRINT_STREAMS:
                print(f"DEVICE_STREAM omitted={len(report['streams']) - MAX_PRINT_STREAMS}", flush=True)
            for gap in report["host_gaps"]:
                print("HOST_CALLBACK_GAP " + json.dumps(gap), flush=True)
            complete = complete and report["usable"]
        for context in service_warning_context(root / backend / "inference_python.log"):
            print("SERVICE_WARNING " + json.dumps(context, ensure_ascii=False), flush=True)
    print("Host 间隔仅按同一进程的时间相邻回调统计，重叠不计入；不是设备空泡，也不跨时钟归因。", flush=True)
    print("日志警告不能独立判定时间线无效；EngineDeadError 的清理阶段归属需另核对。", flush=True)
    print("BASIC_VISUALIZATION=" + ("PASS" if complete else "PARTIAL_OR_UNVERIFIED"), flush=True)
    print("PASS 只表示基本轨道和区间可用，不证明全部采集完整、CPU/NPU关联或推理正确。", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
