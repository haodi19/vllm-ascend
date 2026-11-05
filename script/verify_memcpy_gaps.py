# SPDX-License-Identifier: Apache-2.0
"""只读复核指定设备轨道的MEMCPY间隔，输出可在MindStudio核对的边界。"""

import argparse
import collections
import json
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from inspect_moe_profile import MAX_TRACE_FILES, _gap_stats, _interval, copy_gap_boundaries, positive_gap_us
from offload_bubble_stats import activity_window_gaps, analyze_offload_batches, bounded_gap_stats

CASES = (
    ("callback_legacy", "python"),
    ("callback", "cpp"),
    ("mte_default", "cpp"),
    ("mte_combined", "cpp"),
)
MAX_BOUNDARY_EXAMPLES = 3
MAX_DISTRIBUTION_VALUES = 8
FIELD_ALIASES = {
    "taskid": "task_id",
    "streamid": "stream_id",
    "direction": "direction",
    "copydirection": "direction",
    "memcpykind": "direction",
    "memcpytype": "direction",
    "copytype": "direction",
    "size": "bytes",
    "bytes": "bytes",
    "copysize": "bytes",
    "datasize": "bytes",
    "sizebytes": "bytes",
    "copysizebytes": "bytes",
}
DIRECTIONS = {
    "H2D": "H2D",
    "D2H": "D2H",
    "D2D": "D2D",
    "H2H": "H2H",
    "HOSTTODEVICE": "H2D",
    "DEVICETOHOST": "D2H",
    "DEVICETODEVICE": "D2D",
    "HOSTTOHOST": "H2H",
    "ACL_MEMCPY_HOST_TO_DEVICE": "H2D",
    "ACL_MEMCPY_DEVICE_TO_HOST": "D2H",
    "ACL_MEMCPY_DEVICE_TO_DEVICE": "D2D",
    "ACL_MEMCPY_HOST_TO_HOST": "H2H",
}


def safe_copy_fields(args):
    """只输出已知方向及十进制数量/编号，不回显地址或任意参数。"""
    result = {}
    if not isinstance(args, dict):
        return result
    for key, value in args.items():
        normalized = "".join(char for char in str(key).lower() if char.isalnum())
        field = FIELD_ALIASES.get(normalized)
        if field == "direction" and isinstance(value, str):
            direction = DIRECTIONS.get(value.upper()) or DIRECTIONS.get(
                "".join(char for char in value.upper() if char.isalnum())
            )
            if direction:
                result[field] = direction
        elif field in ("task_id", "stream_id", "bytes"):
            if type(value) is int and value >= 0:
                result[field] = value
            elif isinstance(value, str) and value.isascii() and value.isdecimal():
                result[field] = int(value)
    return result


def read_copy_tracks(path):
    """只遍历一次JSON；EOF验证成功后才使用尾部元数据选择设备轨道。"""
    processes, names = {}, {}
    copies = collections.defaultdict(list)
    retained = 0
    for number, event in enumerate(events(path), 1):
        pid = "" if event.get("pid") is None else str(event["pid"])
        tid = "" if event.get("tid") is None else str(event["tid"])
        name, phase = str(event.get("name", "")), event.get("ph")
        args = event.get("args", {})
        if phase == "M" and isinstance(args, dict):
            if name == "process_name":
                value = str(args.get("name", ""))
                if pid in processes and processes[pid] != value:
                    raise ValueError("同一pid存在冲突的process_name元数据，拒绝猜测设备身份")
                processes[pid] = value
            elif name == "thread_name":
                value = str(args.get("name", ""))
                if (pid, tid) in names and names[pid, tid] != value:
                    raise ValueError("同一pid/tid存在冲突的thread_name元数据，拒绝猜测轨道身份")
                names[pid, tid] = value
        elif phase == "X" and "MEMCPY_ASYNC" in name.upper():
            start, end = _interval(event)
            copies[pid, tid].append(
                dict(
                    event_index=number,
                    start=start,
                    duration=float(event["dur"]),
                    end=end,
                    fields=safe_copy_fields(args),
                )
            )
            retained += 1
        if retained + len(processes) + len(names) > MAX_INTERVALS:
            raise ValueError("MEMCPY区间或元数据超过读取上限，未输出部分统计")
    hardware = {pid for pid, name in processes.items() if "ascend hardware" in name.lower()}
    return {key: values for key, values in copies.items() if key[0] in hardware}, names


def select_track(tracks, names, stream_name, pid=None):
    matches = [key for key in tracks if names.get(key) == stream_name and (pid is None or key[0] == pid)]
    if not matches:
        raise ValueError(f"指定设备轨道不存在：name={stream_name!r}, pid={pid!r}；不按tid或拷贝数量替代")
    if len(matches) != 1:
        raise ValueError(f"同名设备轨道不唯一：{matches}；请对单独trace使用--pid消歧，不合并计算")
    return matches[0]


def short_stats(stats):
    outliers = stats["outliers_us"]
    return {
        "raw": outliers["raw"],
        "filtered": outliers["filtered"],
        "removed": outliers["removed_count"],
        "rule": outliers["rule"],
        "reason": outliers["reason"],
        "bounds": outliers["bounds"],
    }


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def boundary_example(records, boundary, origin):
    left_index, right_index, gap = boundary
    left, right = records[left_index], records[right_index]

    def describe(record):
        return dict(
            event_index=record["event_index"],
            ts_us=record["start"],
            dur_us=record["duration"],
            end_us=record["end"],
            relative_ts_us=record["start"] - origin,
            **record["fields"],
        )

    return dict(
        left=describe(left),
        right=describe(right),
        subtraction_us=f"{right['start']!r} - {left['end']!r} = {gap!r}",
        gap_us=gap,
    )


def inspect_trace(path, stream_name, threshold, pid=None):
    print(f"TRACE={path}", flush=True)
    tracks, names = read_copy_tracks(path)
    for key in sorted(tracks):
        emit("COPY_TRACK", dict(pid=key[0], tid=key[1], name=names.get(key, ""), copies=len(tracks[key])))
    key = select_track(tracks, names, stream_name, pid)
    records = sorted(tracks[key], key=lambda item: (item["start"], item["end"], item["event_index"]))
    boundaries, overlaps = copy_gap_boundaries([(record["start"], record["end"]) for record in records])
    durations = [boundary[2] for boundary in boundaries]
    large = [boundary for boundary in boundaries if boundary[2] >= threshold]
    emit(
        "SELECTED",
        dict(
            pid=key[0],
            tid=key[1],
            name=names[key],
            relative_time_origin_us=records[0]["start"],
            copies=len(records),
            gaps=len(durations),
            zero_gaps=sum(value == 0 for value in durations),
            overlaps_excluded=overlaps,
            large_gaps=len(large),
            large_gap_threshold_us=threshold,
        ),
    )
    stats = _gap_stats(durations, threshold)
    emit("ALL_GAPS_US", short_stats(stats))
    emit("LARGE_GAPS_US", short_stats(stats["large_gaps"]))
    for field in ("direction", "bytes"):
        distribution = collections.Counter(record["fields"][field] for record in records if field in record["fields"])
        emit(
            "COPY_FIELDS",
            dict(
                field=field,
                recognized=sum(distribution.values()),
                missing=len(records) - sum(distribution.values()),
                most_common=distribution.most_common(MAX_DISTRIBUTION_VALUES),
                distinct_values=len(distribution),
            ),
        )
    origin = records[0]["start"]
    earliest = large[:MAX_BOUNDARY_EXAMPLES]
    for boundary in earliest:
        emit("FIRST_LARGE_GAP", boundary_example(records, boundary, origin))
    for boundary in sorted(large, key=lambda item: item[2], reverse=True)[:MAX_BOUNDARY_EXAMPLES]:
        if boundary not in earliest:
            emit("LONGEST_GAP", boundary_example(records, boundary, origin))
    print("PASS：完整JSON与指定轨道间隔复核完成；未归因为专家权重，未标注层或预取/按需方向。", flush=True)


def _nonnegative_int(value):
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def read_offload_header(path):
    """First streaming pass retains metadata/Host markers and copy counts only."""
    processes, names, callbacks, compilers = {}, {}, [], []
    copies = collections.Counter()
    for number, event in enumerate(events(path), 1):
        pid = "" if event.get("pid") is None else str(event["pid"])
        tid = "" if event.get("tid") is None else str(event["tid"])
        name, phase = str(event.get("name", "")), event.get("ph")
        args = event.get("args", {})
        if phase == "M" and isinstance(args, dict):
            if name == "process_name":
                value = str(args.get("name", ""))
                if pid in processes and processes[pid] != value:
                    raise ValueError("同一pid存在冲突的process_name元数据")
                processes[pid] = value
            elif name == "thread_name":
                value = str(args.get("name", ""))
                if (pid, tid) in names and names[pid, tid] != value:
                    raise ValueError("同一pid/tid存在冲突的thread_name元数据")
                names[pid, tid] = value
        elif phase == "X":
            if "MEMCPY_ASYNC" in name.upper():
                copies[pid, tid] += 1
            if name.startswith("moe_offload/host_callback|"):
                start, end = _interval(event)
                meta = fields(event)
                callbacks.append(
                    dict(
                        pid=pid,
                        tid=tid,
                        event_index=number,
                        start=start,
                        end=end,
                        call=_nonnegative_int(meta.get("call")),
                        layer=_nonnegative_int(meta.get("layer")),
                        kind=meta.get("kind") if meta.get("kind") in ("prefetch", "reactive") else "unknown",
                    )
                )
            elif "npu_fx_compiler_inference" in name.lower():
                start, end = _interval(event)
                compilers.append(dict(pid=pid, tid=tid, event_index=number, start=start, end=end))
        if len(processes) + len(names) + len(copies) + len(callbacks) + len(compilers) > MAX_INTERVALS:
            raise ValueError("Host标记或元数据超过读取上限")
    hardware = {pid for pid, name in processes.items() if "ascend hardware" in name.lower()}
    for marker in compilers:
        marker["process_name"] = processes.get(marker["pid"], "")
    return {key: value for key, value in copies.items() if key[0] in hardware}, names, callbacks, compilers


def read_device_tasks(path, key, retained_host_markers=0):
    """Second pass retains all X tasks only on the chosen hardware track."""
    tasks = []
    for number, event in enumerate(events(path), 1):
        if event.get("ph") != "X" or (str(event.get("pid")), str(event.get("tid"))) != key:
            continue
        start, end = _interval(event)
        name = str(event.get("name", "")).upper()
        if "MEMCPY_ASYNC" in name:
            kind = "copy"
        elif "TRANSPOSE" in name:
            kind = "transpose"
        elif any(token in name for token in ("EVENT", "WAIT", "NOTIFY")):
            kind = "event_or_wait"
        else:
            kind = "other_device_task"
        tasks.append(dict(start=start, end=end, kind=kind, event_index=number))
        if len(tasks) + retained_host_markers > MAX_INTERVALS:
            raise ValueError("选中设备流任务或Host标记超过读取上限")
    return tasks


def select_host_callbacks(callbacks, host_pid=None):
    pids = sorted({callback["pid"] for callback in callbacks})
    if host_pid is None:
        if len(pids) != 1 or not pids[0]:
            raise ValueError(f"Host回调进程不唯一或缺失：{pids}；请使用--host-pid，不跨进程猜测关联")
        host_pid = pids[0]
    selected = [callback for callback in callbacks if callback["pid"] == host_pid]
    if not selected:
        raise ValueError(f"指定Host进程没有完整host_callback：pid={host_pid!r}")
    return selected, host_pid


def compact_outliers(summary):
    def rounded(record):
        if record is None:
            return None
        return {key: round(value, 3) if isinstance(value, float) else value for key, value in record.items()}

    return dict(
        raw=rounded(summary["raw"]),
        filtered=rounded(summary["filtered"]),
        removed=summary["removed_count"],
        reason=summary["reason"],
        bounds=[round(value, 3) for value in summary["bounds"]] if summary["bounds"] is not None else None,
    )


def offload_pair_example(pair, origin):
    def callback(record):
        return {key: record[key] for key in ("pid", "tid", "call", "layer", "kind", "start", "end")}

    def task(record):
        return dict(
            event_index=record["event_index"],
            kind=record["kind"],
            ts_us=record["start"],
            end_us=record["end"],
            dur_us=record["end"] - record["start"],
            relative_ts_us=record["start"] - origin,
        )

    return dict(
        direction=pair["direction"],
        left_callback=callback(pair["left_callback"]),
        right_callback=callback(pair["right_callback"]),
        last_device_task=task(pair["left_task"]),
        next_first_device_task=task(pair["right_task"]),
        last_copy=task(pair["left_copy"]),
        next_first_copy=task(pair["right_copy"]),
        gap_us=pair["gap_us"],
        copy_only_gap_us=pair["copy_only_gap_us"],
        copy_only_excess_us=pair["copy_only_excess_us"],
        subtraction_us=f"{pair['right_task']['start']!r} - {pair['left_task']['end']!r} = {pair['gap_us']!r}",
    )


def inspect_offload_trace(path, stream_name, threshold, long_threshold, pid=None, host_pid=None):
    print(f"TRACE={path}\nREAD 1/2：元数据、回调窗口、编译推理标记及拷贝轨道计数", flush=True)
    before = path.stat()
    counts, names, callbacks, compilers = read_offload_header(path)
    for key in sorted(counts):
        emit("COPY_TRACK", dict(pid=key[0], tid=key[1], name=names.get(key, ""), copies=counts[key]))
    key = select_track(counts, names, stream_name, pid)
    print("READ 2/2：仅选中设备流的全部X任务，包含MEMCPY、Transpose及其他任务", flush=True)
    tasks = read_device_tasks(path, key, len(callbacks) + len(compilers))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("两遍读取期间trace发生变化，未输出间隔统计")
    emit(
        "TARGET_GAPS_US",
        dict(
            scope="full_capture",
            **activity_window_gaps(tasks, lower_us=threshold, long_us=long_threshold),
        ),
    )
    try:
        callbacks, host_pid = select_host_callbacks(callbacks, host_pid)
    except ValueError as exc:
        emit("CALLBACK_SCOPE_UNAVAILABLE", dict(reason=str(exc)))
        return False
    selected_compilers = [marker for marker in compilers if marker["pid"] == host_pid]
    window = min(callback["start"] for callback in callbacks), max(callback["end"] for callback in callbacks)
    emit(
        "TARGET_GAPS_US",
        dict(
            scope="callback_window",
            **activity_window_gaps(tasks, lower_us=threshold, long_us=long_threshold, window=window),
        ),
    )
    report = analyze_offload_batches(tasks, callbacks, lower_us=threshold, long_us=long_threshold)
    origin = min(task["start"] for task in tasks)
    emit(
        "SELECTED_OFFLOAD_TRACK",
        dict(
            pid=key[0],
            tid=key[1],
            name=names[key],
            host_pid=host_pid,
            relative_time_origin_us=origin,
            tasks=len(tasks),
            task_kinds=dict(collections.Counter(task["kind"] for task in tasks)),
            callbacks=len(callbacks),
            loaded_batches=len(report["batches"]),
            callbacks_without_matched_copy=len(callbacks) - len(report["batches"]),
        ),
    )
    emit("COPY_CALLBACK_WINDOWS", report["copy_phases"])
    emit("ACTIVITY_CALLBACK_WINDOWS", report["activity_phases"])
    emit("TASK_MATCH", report["task_matches"])
    emit("EXCLUDED_ADJACENT_PAIRS", report["excluded_pairs"])
    emit(
        "CALLBACK_WINDOW",
        dict(
            start_us=min(callback["start"] for callback in callbacks),
            end_us=max(callback["end"] for callback in callbacks),
        ),
    )
    compiler_examples = selected_compilers or compilers
    compiler_scope = "same_pid_as_callback" if selected_compilers else "trace_wide_unlinked"
    emit(
        "COMPILER_MARKERS",
        dict(
            all_host_count=len(compilers),
            selected_host_count=len(selected_compilers),
            example_scope=compiler_scope,
            first_start_us=min((marker["start"] for marker in compilers), default=None),
            last_end_us=max((marker["end"] for marker in compilers), default=None),
            max_duration_us=max((marker["end"] - marker["start"] for marker in compilers), default=None),
        ),
    )
    if compilers and not selected_compilers:
        print("编译推理标记与回调pid不同：仅显示全trace范围，未自动关联两类进程，也不参与批次配对。", flush=True)
    for marker in sorted(compiler_examples, key=lambda value: value["start"])[:MAX_BOUNDARY_EXAMPLES]:
        duration = marker["end"] - marker["start"]
        emit(
            "COMPILER_RANGE",
            dict(
                pid=marker["pid"],
                tid=marker["tid"],
                process_name=marker["process_name"],
                scope=compiler_scope,
                ts_us=marker["start"],
                end_us=marker["end"],
                dur_us=duration,
                dur_ms=duration / 1000.0,
                relative_ts_us=marker["start"] - origin,
            ),
        )
    for direction in report["directions"]:
        emit(
            "TARGET_DIRECTION_GAPS_US",
            dict(
                direction=direction,
                **bounded_gap_stats(
                    [pair["gap_us"] for pair in report["pairs"] if pair["direction"] == direction],
                    threshold,
                    long_threshold,
                ),
            ),
        )
    emit("ROUND_BOUNDARY_CANDIDATE_COUNT", len(report["round_pairs"]))
    for pair in [pair for pair in report["pairs"] if threshold <= pair["gap_us"] < long_threshold][
        :MAX_BOUNDARY_EXAMPLES
    ]:
        emit("NORMAL_PAIR_EXAMPLE", offload_pair_example(pair, origin))
    emit("OFFLOAD_STATUS", dict(partial=report["partial"], reasons=report["partial_reasons"]))
    print("仅按导出公共时间轴的唯一窗口包含关联，不是flow因果证明；窗口外拷贝不混入批次间隔。", flush=True)
    return not report["partial"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="四组实验根目录或单个trace_view.json文件")
    parser.add_argument("--stream-name", required=True, help="MindStudio中的完整显示名称，如Stream 56")
    parser.add_argument("--pid", help="单个trace内设备进程编号；用于同名轨道消歧")
    parser.add_argument(
        "--min-gap-us",
        "--large-gap-us",
        dest="large_gap_us",
        type=positive_gap_us,
        default=10.0,
        help="间隔下限（含边界），默认10μs；--large-gap-us为兼容别名",
    )
    parser.add_argument(
        "--offload-bubbles", action="store_true", help="按相邻Host回调关联加载批次，统计全设备任务之间的空闲"
    )
    parser.add_argument("--host-pid", help="加载回调所在Host进程编号；多Host进程时必须指定")
    parser.add_argument(
        "--long-gap-us", type=positive_gap_us, default=5000.0, help="空泡统计排除大于等于此值的间隔，默认5000μs"
    )
    args = parser.parse_args(argv)
    if args.offload_bubbles and args.long_gap_us <= args.large_gap_us:
        parser.error("--long-gap-us必须大于--large-gap-us")
    return args


def main(argv=None):
    args = parse_args(argv)
    print("只读现有trace；不导入Torch、不启动模型、不写文件。单位μs；按thread_name精确选流。", flush=True)
    if args.offload_bubbles:
        print("批次=唯一Host回调窗口内含MEMCPY的设备任务集合；含Transpose，空载回调不跳过。", flush=True)
        print(
            f"目标空泡仅保留 {args.large_gap_us:g}≤gap<{args.long_gap_us:g}μs；不再叠加IQR筛选。",
            flush=True,
        )
        print("同时统计全采集窗口（含Prefill）和首个回调开始至末个回调结束的连续范围。", flush=True)
        print("极短/极长间隔仅输出排除数；不认定外部噪声。设备流空闲不等于整卡空闲。", flush=True)
    else:
        print("旧拷贝复核模式：3×IQR仅作统计筛选，n<8或IQR=0保留原值；双窗口空泡请加--offload-bubbles。", flush=True)
        print("间隔=后一拷贝开始−此前覆盖区间最晚结束；重叠不计间隔，零间隔保留。", flush=True)
    root = args.root.resolve()
    groups = [("single_trace", [root])] if root.is_file() else []
    failures = passes = 0
    if not groups:
        for case, policy in CASES:
            directory = root / "profile" / case / policy
            traces = []
            for path in directory.rglob("trace_view.json"):
                traces.append(path)
                if len(traces) > MAX_TRACE_FILES:
                    break
            groups.append((case, sorted(traces)))
    for case, traces in groups:
        print(f"CASE={case}; traces={len(traces)}", flush=True)
        if not traces or len(traces) > MAX_TRACE_FILES:
            print("FAIL：缺少trace或超过文件数上限；其他组继续。", flush=True)
            failures += 1
            continue
        for path in traces:
            try:
                if args.offload_bubbles:
                    complete = inspect_offload_trace(
                        path, args.stream_name, args.large_gap_us, args.long_gap_us, args.pid, args.host_pid
                    )
                else:
                    inspect_trace(path, args.stream_name, args.large_gap_us, args.pid)
                    complete = True
            except (OSError, ValueError, TypeError) as exc:
                print(f"FAIL：{type(exc).__name__}: {exc}", flush=True)
                failures += 1
            else:
                passes += int(complete)
                failures += int(not complete)
    print(f"SUMMARY passed_traces={passes}; failed={failures}；不同轨道和文件未合并。", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
