# SPDX-License-Identifier: Apache-2.0
"""Read saved profiles only; compare observed transfer service rates, not peak bandwidth.

DB units/relations: https://www.hiascend.com/doc_center/source/en/
CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html
TASK timestamps are ns; MEMCPY_INFO.size is bytes. Never use Host API duration.
"""

import argparse
import collections
import contextlib
import csv
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path

from analyze_callback_gaps import events
from moe_experiment_tables import number, table

MAX_RECORDS = 1_000_000
MAX_TABLE_ROWS = 24
DB_TIMEOUT_SECONDS = 60
EXPLICIT_BYTE_KEYS = frozenset(("bytes", "sizebytes", "copysizebytes", "datasizebytes"))
DIRECTION_KEYS = frozenset(("direction", "copydirection", "memcpykind", "memcpytype", "memcpyoperation"))
DIRECTIONS = {
    "h2d": "H2D",
    "hosttodevice": "H2D",
    "aclmemcpyhosttodevice": "H2D",
    "d2h": "D2H",
    "devicetohost": "D2H",
    "aclmemcpydevicetohost": "D2H",
    "d2d": "D2D",
    "devicetodevice": "D2D",
    "aclmemcpydevicetodevice": "D2D",
    "h2h": "H2H",
    "hosttohost": "H2H",
    "aclmemcpyhosttohost": "H2H",
    "hosttodeviceex": "H2D_EX",
    "devicetohostex": "D2H_EX",
    "addrdevicetodevice": "ADDR_D2D",
    "managedmemory": "MANAGED",
}


def normalized(value):
    return "".join(c for c in str(value).lower() if c.isalnum())


def direction(value):
    return DIRECTIONS.get(normalized(value)) if isinstance(value, str) else None


def positive_integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    return value if type(value) is int and value > 0 else None


def columns(db, name):
    return {row[1] for row in db.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')}


def add_sample(groups, missing, device, stream, kind, size, duration_ns):
    valid_lane = device is not None and stream is not None
    if not valid_lane:
        missing["missing_device_or_stream"] += 1
    size = positive_integer(size)
    if size is None:
        missing["missing_or_invalid_bytes"] += 1
    if kind is None:
        missing["missing_or_ambiguous_direction"] += 1
    valid_duration = type(duration_ns) in (int, float) and math.isfinite(duration_ns) and duration_ns > 0
    if not valid_duration:
        missing["nonpositive_or_invalid_duration"] += 1
    if size is not None and kind is not None and valid_duration and valid_lane:
        groups[str(device), str(stream), kind, size].append(duration_ns)


def read_database(path):
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("database has an active journal; use a completed capture")
    groups, missing = collections.defaultdict(list), collections.Counter()
    with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        deadline = time.monotonic() + DB_TIMEOUT_SECONDS
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        required = {
            "TASK": {"startNs", "endNs", "deviceId", "streamId", "globalTaskId"},
            "MEMCPY_INFO": {"globalTaskId", "size", "memcpyOperation"},
        }
        for name, fields in required.items():
            found = columns(db, name)
            print(f"DB_SCHEMA {name}: {sorted(found)}", flush=True)
            if not fields <= found:
                raise ValueError(f"unsupported or missing {name} schema")
        if db.execute("SELECT 1 FROM MEMCPY_INFO GROUP BY globalTaskId HAVING COUNT(*)>1 LIMIT 1").fetchone():
            raise ValueError("duplicate MEMCPY_INFO task IDs; refusing a multiplying join")
        maps = []
        for name, text_field in (("ENUM_MEMCPY_OPERATION", "name"), ("STRING_IDS", "value")):
            if {"id", text_field} <= columns(db, name):
                maps.append((name, text_field))
        operation_map = {}
        codes = db.execute("SELECT DISTINCT memcpyOperation FROM MEMCPY_INFO LIMIT 257").fetchall()
        if len(codes) > 256:
            raise ValueError("too many copy direction codes")
        for (code,) in codes:
            candidates = {direction(code)} - {None}
            for name, text_field in maps:
                for (value,) in db.execute(f'SELECT "{text_field}" FROM "{name}" WHERE id=?', (code,)):
                    candidate = direction(value)
                    if candidate:
                        candidates.add(candidate)
            operation_map[code] = next(iter(candidates)) if len(candidates) == 1 else None
        print("DB_DIRECTIONS " + json.dumps(operation_map, ensure_ascii=False), flush=True)
        missing["metadata_without_task"] = db.execute(
            "SELECT COUNT(*) FROM MEMCPY_INFO m WHERE NOT EXISTS "
            "(SELECT 1 FROM TASK t WHERE t.globalTaskId=m.globalTaskId)"
        ).fetchone()[0]
        seen = set()
        query = (
            "SELECT t.deviceId,t.streamId,t.globalTaskId,t.startNs,t.endNs,m.size,m.memcpyOperation "
            "FROM TASK t JOIN MEMCPY_INFO m ON t.globalTaskId=m.globalTaskId"
        )
        for count, row in enumerate(db.execute(query), 1):
            if count > MAX_RECORDS:
                raise ValueError("too many transfer records; no partial statistics")
            device, stream, task, start, end, size, operation = row
            identity = (device, stream, task, start, end)
            if identity in seen:
                raise ValueError("duplicate device transfer intervals; refusing double counting")
            seen.add(identity)
            duration = end - start if type(start) is int and type(end) is int else None
            add_sample(groups, missing, device, stream, operation_map.get(operation), size, duration)
        print(f"DEVICE_COPY_ROWS={len(seen)}; exclusions={dict(missing)}", flush=True)
    return groups


def read_json_trace(path):
    processes, lanes, copies, keys = {}, {}, [], collections.Counter()
    for event in events(path):
        pid, tid = str(event.get("pid")), str(event.get("tid"))
        args = event.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        name = str(event.get("name", ""))
        if event.get("ph") == "M" and name in ("process_name", "thread_name"):
            target, key = (processes, pid) if name == "process_name" else (lanes, (pid, tid))
            value = str(args.get("name", ""))
            if key in target and target[key] != value:
                raise ValueError("conflicting trace lane metadata")
            target[key] = value
        elif event.get("ph") == "X" and name.upper() == "MEMCPY_ASYNC":
            fields = {normalized(k): v for k, v in args.items()}
            sizes = {positive_integer(v) for k, v in fields.items() if k in EXPLICIT_BYTE_KEYS} - {None}
            directions = {direction(v) for k, v in fields.items() if k in DIRECTION_KEYS} - {None}
            copies.append(
                (
                    pid,
                    tid,
                    next(iter(directions)) if len(directions) == 1 else None,
                    next(iter(sizes)) if len(sizes) == 1 else None,
                    event.get("dur"),
                )
            )
            keys.update(str(k) for k in args)
        if len(copies) + len(processes) + len(lanes) > MAX_RECORDS:
            raise ValueError("too many trace records; no partial statistics")
    groups, missing = collections.defaultdict(list), collections.Counter()
    count = 0
    for pid, tid, kind, size, duration in copies:
        if "ascend hardware" not in processes.get(pid, "").lower():
            continue
        count += 1
        # Chrome trace durations are microseconds. Do not subtract epoch-sized float timestamps.
        duration_ns = duration * 1000 if type(duration) in (int, float) else None
        add_sample(groups, missing, pid, lanes.get((pid, tid), f"tid={tid}"), kind, size, duration_ns)
    print(f"JSON_DEVICE_COPY_ROWS={count}; exclusions={dict(missing)}; arg_keys={sorted(keys)[:24]}", flush=True)
    print("JSON仅接受明确标为bytes的字段；无单位Size不猜测为字节，未知数字方向不猜测枚举。", flush=True)
    return groups


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def summarize(groups):
    result = {}
    for key, durations in groups.items():
        total_ns = sum(durations)
        size = key[3]
        result[key] = dict(
            n=len(durations),
            total_bytes=size * len(durations),
            mean_us=statistics.mean(durations) / 1000,
            p50_us=statistics.median(durations) / 1000,
            p95_us=percentile(durations, 0.95) / 1000,
            gbps=size * len(durations) / total_ns,
        )
    return result


def inspect_capture(root, label):
    root = root.expanduser().resolve(strict=True)
    output = root if root.name == "ASCEND_PROFILER_OUTPUT" else root / "ASCEND_PROFILER_OUTPUT"
    if not output.is_dir():
        raise ValueError("provide one capture root containing ASCEND_PROFILER_OUTPUT, not a batch directory")
    print(f"\n===== {label} =====\nCAPTURE={root}", flush=True)
    manifest_path = output.parent / "aligned_profile_manifest.json"
    published = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = manifest.get("raw_capture_removed") is True
    print(f"ALIGNED_DIRECTORY_PUBLISHED={published}; 仅报告已有状态，本脚本不校准或改写时间。")
    mapping_path = output / "clock_alignment.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(
            f"CLOCK_METADATA status={mapping.get('status')}; device_scale={mapping.get('device_scale')}; "
            "元数据存在不等于DB已校正"
        )
    else:
        print("CLOCK=无同次校准记录；只使用设备任务自身耗时，不做Host/设备配对，不套用其他采集校准。")
    groups = {}
    databases = sorted(output.glob("ascend_pytorch_profiler*.db"))
    if len(databases) == 1:
        print(f"READ_DB={databases[0].name}", flush=True)
        try:
            groups = read_database(databases[0])
        except (ValueError, sqlite3.Error) as error:
            print(f"DB_UNAVAILABLE: {error}", flush=True)
    elif databases:
        print("DB_UNAVAILABLE: multiple profiler DBs; no automatic merge", flush=True)
    trace = output / "trace_view.json"
    if not groups and trace.is_file():
        print(f"READ_TRACE={trace.name}", flush=True)
        groups = read_json_trace(trace)
    values = summarize(groups)
    chosen = sorted(values, key=lambda key: (-values[key]["total_bytes"], key))[:MAX_TABLE_ROWS]
    print(
        table(
            ["设备ID/PID", "DB流ID/JSON流名", "方向", "字节/次", "次数", "均值μs", "P50μs", "P95μs", "有效GB/s"],
            [
                [
                    *key,
                    values[key]["n"],
                    number(values[key]["mean_us"]),
                    number(values[key]["p50_us"]),
                    number(values[key]["p95_us"]),
                    number(values[key]["gbps"]),
                ]
                for key in chosen
            ],
        ),
        flush=True,
    )
    print(f"VALID_GROUPS={len(values)}; printed={len(chosen)}; H2D_GROUPS={sum(k[2] == 'H2D' for k in values)}")
    print("统计范围仅为字段齐全的已关联设备任务；不能由此认定覆盖了全部权重传输。")
    if not values:
        print("BANDWIDTH_UNAVAILABLE：没有同时具备明确字节数、方向、正设备耗时的数据；不是带宽为0。")
        for path in sorted(output.glob("*.csv"))[:8]:
            with path.open(encoding="utf-8-sig", newline="") as source:
                header = next(csv.reader(source), [])
            print(f"CSV_HEADER {path.name}: {header[:24]}")
    return values


def compare(old, new):
    """Only unique per-direction/per-size lanes are comparable without user selection."""
    by_size = []
    for values in (old, new):
        grouped = collections.defaultdict(list)
        for key, stats in values.items():
            if key[2] == "H2D":
                grouped[key[3]].append((key, stats))
        by_size.append(grouped)
    rows = []
    for size in sorted(by_size[0].keys() & by_size[1].keys(), reverse=True):
        left, right = by_size[0][size], by_size[1][size]
        if len(left) != 1 or len(right) != 1:
            print(f"H2D_PAIR_UNRESOLVED bytes={size}; old_lanes={len(left)}; new_lanes={len(right)}")
            continue
        a, b = left[0][1], right[0][1]
        rows.append(
            [size, a["n"], b["n"], number(a["gbps"]), number(b["gbps"]), number(100 * (b["gbps"] / a["gbps"] - 1))]
        )
    print("\n同方向、同字节数H2D参考对照（全采集混合阶段；未证明权重批次/并行负载一致）：")
    print(table(["字节/次", "旧次数", "新次数", "旧GB/s", "新GB/s", "新相对旧变化%"], rows[:MAX_TABLE_ROWS]))
    if not rows:
        print("COMPARE_UNAVAILABLE：需先补足字段或确认流；不从显存容量、空泡或专家数量反推带宽。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True, help="One saved 128G PR capture root")
    parser.add_argument("--new", type=Path, required=True, help="One saved 84G PR capture root")
    args = parser.parse_args()
    print("只读已有profile：不导入Torch、不初始化NPU、不启动解析工具、不写文件。")
    print("统计全采集设备MEMCPY，按设备/流/方向/字节数分组；不含Host提交、等待、空泡或Transpose。")
    print("有效GB/s=总字节数/设备任务耗时之和；1GB=10^9字节。不是峰值带宽，不是跨流聚合吞吐，不做IQR。")
    values, failed = [], False
    for label, path in (("旧128G PR", args.old), ("新84G PR", args.new)):
        try:
            values.append(inspect_capture(path, label))
        except (OSError, ValueError, sqlite3.Error) as error:
            print(f"CAPTURE_FAILED {label}: {type(error).__name__}: {error}", flush=True)
            values.append({})
            failed = True
    compare(*values)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
