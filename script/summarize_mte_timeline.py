"""只读汇总原始 profile，区分设备算子与 CPU 提交；仅打印控制台。"""

import argparse
import math
import runpy
from pathlib import Path

MAX_INTERVALS = 500000
MAX_PROCESSES = 256
MAX_LABELS = 8192
MAX_LABEL_LENGTH = 1024
MAX_TRACES = 64
TOP_ROWS = 20


def number(event, key):
    value = event.get(key)
    try:
        if isinstance(value, bool):
            raise ValueError("boolean time")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite time")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid trace {key}: {value!r}; name={event.get('name')!r}") from exc
    return result


def add_stat(table, key, duration):
    if len(key) > MAX_LABEL_LENGTH:
        raise ValueError("事件名称过长，超过分析内存上限")
    if key not in table:
        if len(table) >= MAX_LABELS:
            raise ValueError("不同事件名称超过分析上限；请缩短采集")
        table[key] = [0, 0.0, 0.0]
    row = table[key]
    row[0] += 1
    row[1] += duration
    row[2] = max(row[2], duration)


def interval_union(intervals):
    if not intervals:
        return 0.0, 0.0
    intervals.sort()
    begin, end = intervals[0]
    first, total = begin, 0.0
    for left, right in intervals[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - begin
            begin, end = left, right
    return total + end - begin, end - first


def summarize(path, max_intervals=MAX_INTERVALS):
    """两次流式读取；设备区间、名称和进程数量都有硬上限。"""
    if max_intervals < 1:
        raise ValueError("max_intervals must be positive")
    read_events = runpy.run_path(str(Path(__file__).with_name("analyze_callback_gaps.py")))["events"]
    before = path.stat()
    processes = {}
    print(f"READ 1/2: {path}", flush=True)
    for event in read_events(path):
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        pid = str(event.get("pid"))
        if pid not in processes and len(processes) >= MAX_PROCESSES:
            raise ValueError("进程 metadata 数量超过分析上限")
        process_name = str(event.get("args", {}).get("name", ""))
        if len(pid) > MAX_LABEL_LENGTH or len(process_name) > MAX_LABEL_LENGTH:
            raise ValueError("进程 metadata 过长，超过分析内存上限")
        processes[pid] = process_name
    hardware = {pid for pid, name in processes.items() if "ascend hardware" in name.casefold()}
    intervals, names, categories, handoff, updates = [], {}, {}, {}, {}
    unsupported, negative = 0, 0
    print(f"READ 2/2: Ascend Hardware PIDs={sorted(hardware)}", flush=True)
    for event in read_events(path):
        name = str(event.get("name", ""))
        on_device = str(event.get("pid")) in hardware
        is_update = name.startswith("moe_offload/update_weights|")
        if not on_device and not is_update:
            continue
        if event.get("ph") != "X":
            if on_device and event.get("ph") in ("B", "E", "b", "e"):
                unsupported += 1
            continue
        start, duration = number(event, "ts"), number(event, "dur")
        if duration < 0:
            negative += 1
            continue
        if is_update and not on_device:
            fields = dict(part.split("=", 1) for part in name.split("|")[1:] if "=" in part)
            add_stat(updates, fields.get("kind", "unknown"), duration)
        if not on_device:
            continue
        if len(intervals) >= max_intervals:
            raise ValueError(f"设备区间超过上限 {max_intervals}；请缩短采集或显式增大 --max-intervals")
        intervals.append((start, start + duration))
        add_stat(names, name, duration)
        add_stat(categories, str(event.get("cat", "<missing>")), duration)
        for marker in ("host_sync_publish", "host_sync_wait"):
            if marker in name.lower():
                add_stat(handoff, marker, duration)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("读取过程中时间线仍在变化；请等导出完成后重试")
    busy, span = interval_union(intervals)
    result = dict(
        processes=processes,
        hardware_pids=sorted(hardware),
        device_events=len(intervals),
        busy_us=busy,
        span_us=span,
        names=names,
        categories=categories,
        handoff=handoff,
        updates=updates,
        unsupported=unsupported,
        negative=negative,
    )
    return result


def print_table(title, rows):
    print(title)
    if not rows:
        print("  无可识别事件")
    for name, (count, total, maximum) in sorted(rows.items(), key=lambda item: -item[1][1])[:TOP_ROWS]:
        print(
            f"  {name[:180]}: count={count}; sum_ms={total / 1000:.3f}; "
            f"mean_us={total / count:.3f}; max_us={maximum:.3f}"
        )


def report(result):
    print("时间单位：Chrome trace 微秒；sum_ms 为事件耗时相加，允许重叠，不能直接相加归因请求时延。")
    print_table("CPU 内层 update_weights（不是请求总时延）：", result["updates"])
    if not result["hardware_pids"]:
        print("DEVICE UNKNOWN：process_name 未识别到 Ascend Hardware；不把 CPU launch 名称猜作设备算子。")
        print("进程 metadata（最多显示 20 个）：")
        for pid, name in list(result["processes"].items())[:TOP_ROWS]:
            print(f"  pid={pid} name={name[:180]}")
        return
    print(
        f"DEVICE X events={result['device_events']}; span_ms={result['span_us'] / 1000:.3f}; "
        f"union_busy_ms={result['busy_us'] / 1000:.3f}; "
        f"uncovered_ms={(result['span_us'] - result['busy_us']) / 1000:.3f}"
    )
    print("union_busy 是任一设备轨道有 X 事件的时间并集，包含 AIV 轮询等待，不代表有效计算率。")
    print("span 是首个到最后一个设备事件的窗口，可能含多个请求及间隔；不能等同 HTTP 请求时延。")
    print_table("设备交接算子：", result["handoff"])
    print_table("设备 category：", result["categories"])
    print_table("设备名称按累计耗时排序（仅前 20 项；重叠不等于关键路径）：", result["names"])
    if result["unsupported"] or result["negative"]:
        print(f"WARNING: 未计入 B/E 类型设备事件={result['unsupported']}; 负时长事件={result['negative']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="profile 根目录或原始 trace_view.json")
    parser.add_argument("--max-intervals", type=int, default=MAX_INTERVALS)
    args = parser.parse_args()
    if args.max_intervals < 1:
        parser.error("--max-intervals 必须为正数")
    if args.profile.is_file():
        traces = [args.profile]
    else:
        traces = []
        for path in args.profile.rglob("trace_view.json"):
            if path.parent.name == "ASCEND_PROFILER_OUTPUT":
                if len(traces) >= MAX_TRACES:
                    parser.error("时间线文件超过 64 个，请指定更小的 profile 目录")
                traces.append(path)
    if not traces:
        parser.error("未找到原始 ASCEND_PROFILER_OUTPUT/trace_view.json")
    failed = 0
    for path in sorted(traces):
        try:
            print(f"\n=== {path} ===", flush=True)
            report(summarize(path, args.max_intervals))
        except (OSError, ValueError, TypeError) as exc:
            failed += 1
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
