# SPDX-License-Identifier: Apache-2.0
"""只读复核指定设备轨道的MEMCPY间隔，输出可在MindStudio核对的边界。"""

import argparse
import collections
import json
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events
from inspect_moe_profile import MAX_TRACE_FILES, _gap_stats, _interval, copy_gap_boundaries, positive_gap_us

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="四组实验根目录或单个trace_view.json文件")
    parser.add_argument("--stream-name", required=True, help="MindStudio中的完整显示名称，如Stream 56")
    parser.add_argument("--pid", help="单个trace内设备进程编号；用于同名轨道消歧")
    parser.add_argument("--large-gap-us", type=positive_gap_us, default=100.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("只读现有trace；不导入Torch、不启动模型、不写文件。单位μs；按thread_name精确选流。", flush=True)
    print("大间隔阈值不识别批次；3×IQR仅作统计筛选，n<8或IQR=0保留原值。", flush=True)
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
                inspect_trace(path, args.stream_name, args.large_gap_us, args.pid)
            except (OSError, ValueError, TypeError) as exc:
                print(f"FAIL：{type(exc).__name__}: {exc}", flush=True)
                failures += 1
            else:
                passes += 1
    print(f"SUMMARY passed_traces={passes}; failed={failures}；不同轨道和文件未合并。", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
