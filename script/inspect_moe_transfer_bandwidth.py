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
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import struct
import tempfile
import time
from pathlib import Path

from analyze_callback_gaps import events
from moe_experiment_tables import number, table

MAX_RECORDS = 1_000_000
MAX_TABLE_ROWS = 24
DB_TIMEOUT_SECONDS = 60
MAX_INVENTORY_DATABASES = 12
MAX_INVENTORY_ENTRIES = 20000
MAX_INVENTORY_TABLES = 12
MAX_SNAPSHOT_BYTES = 2 * 1024**3
SNAPSHOT_RESERVE_BYTES = 256 * 1024**2
COPY_CHUNK_BYTES = 1024**2
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


def duration_number(value):
    """Aligned JSON serializes precise durations as decimal strings, in microseconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def file_signature(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def copy_hash(source, destination=None):
    digest = hashlib.sha256()
    target = destination.open("wb") if destination is not None else contextlib.nullcontext()
    with source.open("rb") as incoming, target as outgoing:
        while chunk := incoming.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
            if outgoing is not None:
                outgoing.write(chunk)
    return digest.digest()


def wal_checksum(data, endian, state=(0, 0)):
    first, second = state
    for a, b in struct.iter_unpack(endian + "II", data):
        first = (first + a + second) & 0xFFFFFFFF
        second = (second + b + first) & 0xFFFFFFFF
    return first, second


def validate_wal(path):
    """Validate a frozen WAL, refusing malformed/stale tails instead of silently losing frames.

    Format: https://www.sqlite.org/fileformat2.html#wal_format
    This conservative reader can reject a valid reused WAL with stale trailing frames.
    """
    deadline = time.monotonic() + DB_TIMEOUT_SECONDS
    with path.open("rb") as source:
        header = source.read(32)
        if len(header) != 32:
            raise ValueError("incomplete WAL header")
        magic, version, page_size, _, salt1, salt2, c1, c2 = struct.unpack(">8I", header)
        if magic not in (0x377F0682, 0x377F0683) or version != 3007000:
            raise ValueError("unsupported WAL header")
        if page_size < 512 or page_size > 65536 or page_size & (page_size - 1):
            raise ValueError("invalid WAL page size")
        endian = "<" if magic == 0x377F0682 else ">"
        state = wal_checksum(header[:24], endian)
        if state != (c1, c2):
            raise ValueError("WAL header checksum mismatch")
        frames, commits = 0, 0
        while frame := source.read(24):
            data = source.read(page_size)
            if len(frame) != 24 or len(data) != page_size:
                raise ValueError("incomplete WAL frame")
            page, committed_pages, s1, s2, c1, c2 = struct.unpack(">6I", frame)
            state = wal_checksum(data, endian, wal_checksum(frame[:8], endian, state))
            if page == 0 or (s1, s2) != (salt1, salt2) or state != (c1, c2):
                raise ValueError("invalid or stale WAL frame; no partial snapshot used")
            frames += 1
            commits += committed_pages > 0
            if time.monotonic() > deadline:
                raise ValueError("WAL validation timed out")
    print(f"WAL_VALIDATED frames={frames}; commit_markers={commits}", flush=True)


@contextlib.contextmanager
def database_connection(path):
    """Read stable archives; SQLite can write only in a disposable copy when WAL exists."""
    wal, journal = Path(str(path) + "-wal"), Path(str(path) + "-journal")
    signatures = {p: file_signature(p) for p in (path, wal, journal)}
    if signatures[journal] and signatures[journal][0]:
        raise ValueError("nonempty rollback journal; activity/consistency unknown; original DB not opened")
    if not signatures[wal] or not signatures[wal][0]:
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            yield db
            if signatures != {p: file_signature(p) for p in signatures}:
                raise ValueError("source DB changed during reading; results discarded")
        return
    total = signatures[path][0] + signatures[wal][0]
    if total > MAX_SNAPSHOT_BYTES:
        raise ValueError("DB+WAL exceeds 2 GiB temporary snapshot limit")
    with tempfile.TemporaryDirectory(prefix="moe_profile_db_") as temporary:
        directory = Path(temporary)
        if shutil.disk_usage(directory).free < total + SNAPSHOT_RESERVE_BYTES:
            raise ValueError("insufficient temporary disk space for DB+WAL snapshot")
        copied = directory / path.name
        hashes = {p: copy_hash(p, directory / p.name) for p in (path, wal)}
        if signatures != {p: file_signature(p) for p in signatures} or any(
            copy_hash(p) != digest for p, digest in hashes.items()
        ):
            raise ValueError("source DB/WAL changed while copying; snapshot discarded")
        validate_wal(Path(str(copied) + "-wal"))
        # Do not use immutable here: SQLite must consume the copied WAL and rebuild its SHM index.
        with contextlib.closing(sqlite3.connect(str(copied), timeout=5)) as db:
            deadline = time.monotonic() + DB_TIMEOUT_SECONDS
            db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("copied DB+WAL failed quick_check")
            print(f"DB_SNAPSHOT=PASS; source={path.name}; DB+WAL bytes={total}; 原文件未修改", flush=True)
            yield db


def inventory_capture(output):
    """Bounded schema inventory, including raw CANN SQLite files; never parse raw binary data."""
    root = output.parent
    raw = sorted(p.name for p in root.glob("PROF_*") if p.is_dir())
    print(f"RAW_PROF_DIRECTORIES={raw[:8]}; count={len(raw)}")
    candidates, raw_files, visited = [], [], 0
    truncated = False
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not Path(folder, d).is_symlink())
        visited += len(files) + len(dirs)
        if visited > MAX_INVENTORY_ENTRIES:
            truncated = True
            break
        for name in sorted(files):
            path = Path(folder, name)
            if any(part.startswith("PROF_") for part in path.relative_to(root).parts) and len(raw_files) < 16:
                raw_files.append(str(path.relative_to(root)))
            if path.suffix.lower() not in (".db", ".sqlite", ".sqlite3") or path.is_symlink():
                continue
            candidates.append(path)
    # Prioritize raw runtime/task databases over unrelated analysis databases.
    candidates.sort(key=lambda p: (not any(k in p.name.lower() for k in ("runtime", "task", "memcpy")), str(p)))
    print(f"DB_INVENTORY found={len(candidates)}; scanned_entries={visited}; truncated={truncated}")
    print(f"RAW_FILE_EXAMPLES={raw_files}")
    for path in candidates[:MAX_INVENTORY_DATABASES]:
        print(f"DB_INVENTORY_FILE={path.relative_to(root)}")
        try:
            with database_connection(path) as db:
                deadline = time.monotonic() + DB_TIMEOUT_SECONDS
                db.set_progress_handler(lambda deadline=deadline: int(time.monotonic() > deadline), 10000)
                names = [
                    r[0]
                    for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 257")
                ]
                relevant = [n for n in names if any(k in n.lower() for k in ("memcpy", "memory", "task", "bandwidth"))]
                print(f"DB_TABLES first40={names[:40]}; count_at_most257={len(names)}")
                for name in relevant[:MAX_INVENTORY_TABLES]:
                    print(f"CANDIDATE_SCHEMA {name}: {sorted(columns(db, name))[:40]}")
        except (ValueError, OSError, sqlite3.Error) as error:
            print(f"DB_INVENTORY_UNAVAILABLE: {error}")
    print("以上仅目录/表结构，未将原始库中未知单位或未关联字段计入带宽；不启动离线解析。")


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
    groups, missing = collections.defaultdict(list), collections.Counter()
    with database_connection(path) as db:
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
    duration_types = collections.Counter()
    for pid, tid, kind, size, duration in copies:
        if "ascend hardware" not in processes.get(pid, "").lower():
            continue
        count += 1
        duration_types[type(duration).__name__] += 1
        # Chrome trace durations are microseconds. Do not subtract epoch-sized float timestamps.
        duration = duration_number(duration)
        duration_ns = duration * 1000 if duration is not None else None
        add_sample(groups, missing, pid, lanes.get((pid, tid), f"tid={tid}"), kind, size, duration_ns)
    print(f"JSON_DEVICE_COPY_ROWS={count}; exclusions={dict(missing)}; arg_keys={sorted(keys)[:24]}", flush=True)
    print(f"JSON_DURATION_TYPES={dict(duration_types)}; 数字字符串按JSON的us单位读取，不代表耗时无效。")
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


def inspect_capture(root, label, *, follow_source=True):
    root = root.expanduser().resolve(strict=True)
    output = root if root.name == "ASCEND_PROFILER_OUTPUT" else root / "ASCEND_PROFILER_OUTPUT"
    if not output.is_dir():
        raise ValueError("provide one capture root containing ASCEND_PROFILER_OUTPUT, not a batch directory")
    print(f"\n===== {label} =====\nCAPTURE={root}", flush=True)
    manifest_path = output.parent / "aligned_profile_manifest.json"
    published = False
    manifest_status = "missing"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = manifest.get("raw_capture_removed") is True
        manifest_status = manifest.get("status", "unknown")
    print(f"ALIGNMENT_MANIFEST={manifest_status}; RAW_CAPTURE_REPLACED={published}; 缺少标记不能判定未对齐。")
    mapping_path = output / "clock_alignment.json"
    if not mapping_path.is_file():
        mapping_path = output.parent / "clock_alignment.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(
            f"CLOCK_METADATA status={mapping.get('status')}; device_scale={mapping.get('device_scale')}; "
            "元数据存在不等于DB已校正"
        )
    else:
        print("CLOCK=当前采集根目录未找到校准记录，状态未知；只用设备自身耗时，不做Host配对或跨采集校准。")
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
        inventory_capture(output)
        source_name = manifest.get("source_capture")
        if follow_source and isinstance(source_name, str) and source_name:
            source = Path(source_name)
            print(f"RECORDED_SOURCE_CAPTURE={source}; exists={source.is_dir()}")
            if (source / "ASCEND_PROFILER_OUTPUT").is_dir() and source.resolve() != output.parent.resolve():
                if source.name != output.parent.name:
                    print("SOURCE_NOT_FOLLOWED: capture directory identities differ")
                else:
                    print(
                        "BANDWIDTH_SOURCE_FALLBACK=原始采集；仅比较设备拷贝自身速率，不做Host时间匹配，不合并重复数据。"
                    )
                    return inspect_capture(source, label + "/原始采集", follow_source=False)
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
    print("只读原始profile：不导入Torch、不初始化NPU、不启动解析工具。")
    print("带WAL的数据库在临时副本中读取并校验，结束自动清理；不修改或删除任何原始采集文件。")
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
