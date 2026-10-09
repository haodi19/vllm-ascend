#!/usr/bin/env python3
"""Read Ascend profile exports and print a console-only summary.

Usage: python profile_console_summary.py /path/to/run_ascend_pt
Uses only the Python standard library. Does not modify the input or write reports.
"""

import argparse
import collections
import csv
import json
import math
import pathlib
import re

READ_CHUNK = 1024 * 1024
MAX_HEADER = 8 * READ_CHUNK
MAX_EVENT = 32 * READ_CHUNK
CSV_NAMES = (
    "kernel_details.csv", "operator_details.csv", "op_statistic.csv",
    "api_statistic.csv", "task_time.csv", "step_trace_time.csv",
)
SAFE_COLUMNS = frozenset({
    "Name", "Type", "Op Name", "OP Type", "Op Type", "Task Type",
    "Accelerator Core", "Stream ID", "Stream Id", "Task ID", "Task Id",
    "Step ID", "Step Id", "Duration(us)", "Task Duration(us)",
    "Wait Time(us)", "Task Wait Time(us)", "Count",
    "Host Self Duration(us)", "Host Total Duration(us)",
    "Device Self Duration(us)", "Device Total Duration(us)",
})


def show(value):
    print(json.dumps(value, ensure_ascii=False, default=str))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def events(path):
    """Stream a top-level event array or a traceEvents array in an object."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as source:
        buf = ""
        while True:
            chunk = source.read(READ_CHUNK)
            if not chunk:
                raise ValueError("未找到 traceEvents 数组")
            buf += chunk
            stripped = buf.lstrip()
            if stripped.startswith("["):
                buf = stripped[1:]
                break
            match = re.search(r'"traceEvents"\s*:\s*\[', buf)
            if match:
                buf = buf[match.end():]
                break
            if len(buf) > MAX_HEADER:
                raise ValueError("JSON 头部过大，停止自动解析")

        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " \r\n\t,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                buf = buf[pos:]
                pos = 0
                chunk = source.read(READ_CHUNK)
                if not chunk:
                    raise ValueError("事件 JSON 不完整或格式不支持") from None
                buf += chunk
                if len(buf) > MAX_EVENT:
                    raise ValueError("单个事件过大或 JSON 格式异常") from None
                continue
            pos = end
            if isinstance(item, dict):
                yield item


def summarize_csv(root):
    print("=== CSV 字段与摘要 ===")
    for filename in CSV_NAMES:
        paths = sorted(root.rglob(filename))
        print("\nFILE:", filename, "发现数量:", len(paths))
        if not paths:
            continue
        if len(paths) > 1:
            print("仅分析排序后的第一个文件；建议指定单次、单 rank 目录")
        try:
            with paths[0].open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                fields = reader.fieldnames or []
                show({"columns": fields})
                duration_key = next((key for key in (
                    "Duration(us)", "Task Duration(us)"
                ) if key in fields), None)
                name_key = next((key for key in (
                    "Type", "OP Type", "Op Type", "Name", "Op Name"
                ) if key in fields), None)
                totals = collections.defaultdict(lambda: [0, 0.0])
                count = 0
                for row in reader:
                    count += 1
                    if count <= 2:
                        show({"sample": {
                            key: str(value)[:180] for key, value in row.items()
                            if key in SAFE_COLUMNS
                        }})
                    if duration_key and name_key:
                        duration = number(row.get(duration_key))
                        if duration is not None and duration >= 0:
                            stat = totals[row.get(name_key, "?")]
                            stat[0] += 1
                            stat[1] += duration
                print("rows:", count)
                if duration_key and name_key:
                    print("按", duration_key, "累计排序；不是请求延迟：")
                    for name, (n, total) in sorted(
                        totals.items(), key=lambda item: item[1][1], reverse=True
                    )[:10]:
                        show({"name": name, "rows": n, "sum_us": round(total, 3)})
        except (OSError, UnicodeError, csv.Error) as exc:
            print("读取失败:", type(exc).__name__, str(exc)[:160])


def summarize_trace(root):
    traces = sorted(root.rglob("trace_view.json"))
    print("\n=== TRACE ===")
    print("trace_view.json 数量:", len(traces))
    if not traces:
        print("未找到 trace_view.json；请返回上面的 CSV 输出")
        return
    if len(traces) > 1:
        print("仅分析第一个；多卡对比请分别指定各 rank 的 ascend_pt 目录")
    path = traces[0]
    print("文件大小 MiB:", round(path.stat().st_size / READ_CHUNK, 2))
    phases = collections.Counter()
    categories = collections.Counter()
    arg_keys = collections.defaultdict(set)
    process_names = {}
    thread_names = {}
    lanes = collections.Counter()
    totals = collections.defaultdict(lambda: [0, 0.0, 0.0])
    first = last = None
    try:
        for event in events(path):
            phase = str(event.get("ph", "?"))
            category = str(event.get("cat", ""))
            name = str(event.get("name", ""))
            pid = str(event.get("pid", ""))
            tid = str(event.get("tid", ""))
            args = event.get("args")
            args = args if isinstance(args, dict) else {}
            phases[phase] += 1
            categories[category] += 1
            arg_keys[category].update(args.keys())
            if phase == "M":
                if name == "process_name":
                    process_names[pid] = str(args.get("name", ""))
                elif name == "thread_name":
                    thread_names[(pid, tid)] = str(args.get("name", ""))
                continue
            lanes[(pid, tid)] += 1
            timestamp = number(event.get("ts"))
            duration = number(event.get("dur"))
            if phase != "X" or timestamp is None or duration is None or duration < 0:
                continue
            first = timestamp if first is None else min(first, timestamp)
            end = timestamp + duration
            last = end if last is None else max(last, end)
            stat = totals[(pid, tid, category, name)]
            stat[0] += 1
            stat[1] += duration
            stat[2] = max(stat[2], duration)
    except (OSError, UnicodeError, ValueError) as exc:
        print("TRACE 解析未完成，以下只是部分数据:",
              type(exc).__name__, str(exc)[:160])

    show({"ph_counts": dict(phases)})
    show({"category_counts": dict(categories)})
    if first is not None:
        print("X事件首尾跨度 ms（含空闲，不是请求延迟）:",
              round((last - first) / 1000, 3))
    print("\n=== 泳道，最多 40 条 ===")
    for (pid, tid), count in lanes.most_common(40):
        show({"pid": pid, "tid": tid,
              "process": process_names.get(pid, ""),
              "thread": thread_names.get((pid, tid), ""), "events": count})
    print("\n=== 各类事件 args 的键名，不输出值 ===")
    for category, _ in categories.most_common(25):
        show({"category": category, "arg_keys": sorted(arg_keys[category])})
    print("\n=== 完整事件 X 的累计耗时 Top 25 ===")
    print("按标准 Chrome Trace 的微秒时间单位解释。")
    print("跨层级可能重复计时；B/E事件未配对，不纳入排行榜。")
    for (pid, tid, category, name), (n, total, maximum) in sorted(
        totals.items(), key=lambda item: item[1][1], reverse=True
    )[:25]:
        show({"process": process_names.get(pid, pid),
              "thread": thread_names.get((pid, tid), tid),
              "category": category, "name": name, "count": n,
              "sum_us": round(total, 3), "mean_us": round(total / n, 3),
              "max_us": round(maximum, 3)})


def merge_intervals(intervals):
    """Union intervals to avoid double-counting concurrent tasks."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(left, right):
    """Intersect sorted, nonoverlapping interval lists."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def span_ms(intervals):
    return round(sum(end - start for start, end in intervals) / 1000, 6)


def detailed_csv(root):
    print("=== 补充 CSV 统计 ===")
    specs = (
        ("api_statistic.csv", "API Name", "Time(us)"),
        ("operator_details.csv", "Name", "Host Self Duration(us)"),
        ("task_time.csv", "kernel_type", "task_time(us)"),
    )
    for filename, name_key, time_key in specs:
        paths = sorted(root.rglob(filename))
        if not paths:
            continue
        print("\nFILE:", filename, "文件数:", len(paths), "分析第一个")
        grouped = collections.defaultdict(lambda: [0, 0.0])
        with paths[0].open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                duration = number(row.get(time_key))
                if duration is None:
                    continue
                stat = grouped[row.get(name_key, "?")]
                stat[0] += 1
                stat[1] += duration
        print("时间字段:", time_key, "；rows 是 CSV 行数，不一定是调用次数")
        for name, (count, total) in sorted(
            grouped.items(), key=lambda item: item[1][1], reverse=True
        )[:12]:
            show({"name": name, "rows": count, "sum_ms": round(total / 1000, 6)})
    for path in sorted(root.rglob("step_trace_time.csv"))[:1]:
        print("\n=== step_trace_time 前 3 行原始值（不推断无标注字段的单位）===")
        with path.open(encoding="utf-8-sig", newline="") as source:
            for index, row in enumerate(csv.DictReader(source)):
                if index >= 3:
                    break
                show(row)


def detailed_trace(root):
    paths = sorted(root.rglob("trace_view.json"))
    if len(paths) != 1:
        raise ValueError("--details 需要目录内恰好一份 trace_view.json，请指定单次单 rank 目录")
    path = paths[0]
    print("\n=== 扫描 1/2：泳道与事件类型 ===", flush=True)
    process_names = {}
    thread_names = {}
    phases = collections.defaultdict(collections.Counter)
    origin = None
    for event in events(path):
        phase = str(event.get("ph", ""))
        phases[str(event.get("cat", ""))][phase] += 1
        pid, tid = str(event.get("pid", "")), str(event.get("tid", ""))
        args = event.get("args")
        args = args if isinstance(args, dict) else {}
        if phase == "M":
            if event.get("name") == "process_name":
                process_names[pid] = str(args.get("name", ""))
            elif event.get("name") == "thread_name":
                thread_names[(pid, tid)] = str(args.get("name", ""))
        ts, dur = number(event.get("ts")), number(event.get("dur"))
        if phase == "X" and ts is not None and dur is not None and dur >= 0:
            origin = ts if origin is None else min(origin, ts)
    for category, counts in phases.items():
        show({"category": category, "ph_counts": dict(counts)})
    if origin is None:
        print("没有带有效时间的 X 事件")
        return

    print("\n=== 扫描 2/2：计算时间窗口 ===", flush=True)
    buckets = collections.defaultdict(list)
    stream_stats = collections.defaultdict(lambda: [0, 0.0, 0.0])
    windows = collections.defaultdict(list)
    for event in events(path):
        if event.get("ph") != "X":
            continue
        ts, dur = number(event.get("ts")), number(event.get("dur"))
        if ts is None or dur is None or dur < 0:
            continue
        interval = (ts, ts + dur)
        name = str(event.get("name", ""))
        pid, tid = str(event.get("pid", "")), str(event.get("tid", ""))
        process = process_names.get(pid, "")
        if process == "Ascend Hardware":
            stat = stream_stats[(pid, tid, name)]
            stat[0] += 1
            stat[1] += dur
            stat[2] = max(stat[2], dur)
            if "MEMCPY" in name.upper():
                buckets["device_memcpy"].append(interval)
            if "WAIT" in name.upper():
                buckets["device_wait_any_stream"].append(interval)
                windows["device_wait"].append((ts, ts + dur, pid, tid, name))
        if process == "Overlap Analysis" and name in ("Computing", "Free"):
            buckets[name].append(interval)
            if name == "Free":
                windows["Free"].append((ts, ts + dur, pid, tid, name))
        if process == "CANN" and "Synchronize" in name:
            buckets["host_sync_any_thread"].append(interval)
            windows["host_sync"].append((ts, ts + dur, pid, tid, name))
        if process == "Python" and (
            "npu_fx_compiler inference" in name or name == "vllm::moe_forward_shared"
        ):
            windows[name].append((ts, ts + dur, pid, tid, name))

    unions = {name: merge_intervals(values) for name, values in buckets.items()}
    copies = unions.get("device_memcpy", [])
    computing = unions.get("Computing", [])
    unions["copy_and_computing"] = intersect_intervals(copies, computing)

    print("\n=== 时间区间并集（不是各任务耗时之和）===")
    print("device_wait/host_sync 是任意流/线程上的等待覆盖，不是整个设备停顿。")
    for name, values in unions.items():
        show({"kind": name, "union_ms": span_ms(values)})
    if "Computing" in unions:
        show({"copy_without_Computing_ms": round(
            span_ms(copies) - span_ms(unions["copy_and_computing"]), 6),
            "note": "仅为时间交集分类；不等于可消除的延迟或关键路径"})

    print("\n=== 设备流上的 X 任务 Top 20 ===")
    for (pid, tid, name), (count, total, maximum) in sorted(
        stream_stats.items(), key=lambda item: item[1][1], reverse=True
    )[:20]:
        show({"stream": thread_names.get((pid, tid), tid), "name": name,
              "count": count, "sum_ms": round(total / 1000, 6),
              "max_ms": round(maximum / 1000, 6)})

    print("\n=== 关键块的位置及窗口内活动 ===")
    print("时间相对最早有效 X 事件；窗口交集不证明调用或依赖关系。")
    for kind, records in windows.items():
        ordered = sorted(records)
        if kind == "npu_fx_compiler inference":
            selected = ordered[:30]
        else:
            selected = sorted(set(ordered[:2] + sorted(
                records, key=lambda record: record[1] - record[0], reverse=True
            )[:5]))
        print("\nkind:", kind, "总次数:", len(records), "展示:", len(selected))
        for start, end, pid, tid, name in selected:
            row = {"name": name, "process": process_names.get(pid, pid),
                   "thread": thread_names.get((pid, tid), tid),
                   "start_ms": round((start - origin) / 1000, 6),
                   "end_ms": round((end - origin) / 1000, 6),
                   "duration_ms": round((end - start) / 1000, 6)}
            for label in ("device_memcpy", "Computing", "copy_and_computing",
                          "host_sync_any_thread"):
                row[label + "_ms"] = span_ms(intersect_intervals(
                    unions.get(label, []), [(start, end)]))
            show(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=pathlib.Path,
                        help="单次采集、单 rank 的 ascend_pt 目录")
    parser.add_argument("--details", action="store_true",
                        help="输出执行窗口、复制/计算重叠和补充 CSV 统计")
    args = parser.parse_args()
    if not args.profile_dir.is_dir():
        parser.error("目录不存在，请检查 profile_dir")
    if args.details:
        detailed_csv(args.profile_dir)
        detailed_trace(args.profile_dir)
    else:
        summarize_csv(args.profile_dir)
        summarize_trace(args.profile_dir)


if __name__ == "__main__":
    main()
