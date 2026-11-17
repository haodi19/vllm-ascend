# SPDX-License-Identifier: Apache-2.0
"""Inspect existing profiler clock metadata without importing Torch or parsing captures."""

import argparse
import ast
import configparser
import importlib.metadata
import json
import math
import os
import re
import sqlite3
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from analyze_callback_gaps import events

MAX_ROOTS = 16
MAX_ENTRIES = 10000
MAX_DEPTH = 10
MAX_FILES = 64
MAX_FILE_BYTES = 1024 * 1024
MAX_FIELDS = 100
MAX_SOURCE_LINES = 48
MAX_CANN_SOURCE_LINES = 280
MAX_METHOD_LINES = 40
MAX_LITERAL_CHARS = 160
MAX_MSTX_ROWS = 20000
DB_QUERY_TIMEOUT_SECONDS = 20
EXPORT_PROGRESS_SECONDS = 30
CALLBACK_PREFIX = "moe_offload/host_callback|"
MSTX_TABLES = ("MsprofTx", "MsprofTxEx")
SOURCE_CONSTANTS = frozenset(("HOST_DEFAULT_FREQ", "NANO_SECOND", "MICRO_SECOND", "FREQ_TO_MHz", "NS_TO_US", "USTONS"))
SAMPLE_INTERVAL_SECONDS = 0.5
PROFILER_INFO = re.compile(r"profiler_info(?:_\d+)?\.json$")
CANN_METADATA = re.compile(
    r"(?:(?:host|dev)_(?:start|end)[\w-]*\.(?:log|json|txt)(?:\.\d+)?|"
    r"(?:start|end)_info(?:\.\d+)?(?:\.json)?|info(?:\.\d+)?\.json(?:\.\d+)?)$"
)
CLOCK_KEYS = frozenset(
    (
        "freq",
        "frequency",
        "startcnt",
        "endcnt",
        "startmonotonic",
        "endmonotonic",
        "syscnt",
        "cntvct",
        "cntvctdiff",
        "counter",
        "devicesyscnt",
        "monotonictime",
        "hwtsfrequency",
        "aicfrequency",
        "clockfrequency",
        "sysclockfreq",
        "clockmonotonicraw",
        "collectiontimebegin",
        "collectiontimeend",
        "monotonictimeend",
        "begintime",
        "beginmonotonic",
        "endtime",
        "endmonotonicraw",
        "clockrealtime",
        "clockmonotonic",
        "syscntfreq",
        "timeoffset",
        "localtimediff",
    )
)
SENSITIVE_KEYS = frozenset(("env", "environ", "environment", "environmentvariables", "proxy", "credentials"))
VERSION_KEYS = frozenset(("torchnpuversion", "torchversion", "cannversion"))
SOURCE_FILES = (
    "torch_npu/profiler/analysis/_profiler_config.py",
    "torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py",
    "torch_npu/profiler/analysis/prof_view/_trace_view_parser.py",
    "torch_npu/profiler/analysis/prof_common_func/_constant.py",
)
SOURCE_KEYWORDS = re.compile(
    r"get_timestamp_from_syscnt|ratio =|timestamp =|config_freq =|config_offset =|config_start_cnt =|"
    r"self\._(?:freq|time_offset|start_cnt|localtime_diff) =|get_localtime_diff|begin_time|begin_monotonic|"
    r"CANN_BEGIN_TIME|CANN_BEGIN_MONOTONIC|FWK_END_TIME|FWK_END_MONOTONIC|"
    r"msprof_timeline_data =|_prune_trace_by_level\(msprof_timeline_data\)"
)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
NUMBER_WITH_UNIT = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]+)$")
UNITS = frozenset(("hz", "khz", "mhz", "ghz", "s", "ms", "us", "ns", "µs", "μs", "ticks", "cycles"))
UNIT_KEYS = frozenset(("unit", "units", "timeunit", "clockunit", "counterunit", "frequnit", "frequencyunit"))
CANN_ROOTS = (Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/cann-9.1.0"))
CANN_SOURCE_PREFIXES = ("tools/profiler/profiler_tool", "tools/profiler", "profiler_tool", "")
CANN_SOURCE_METHODS = (
    (
        "analysis/common_func/info_conf_reader.py",
        (
            "get_host_freq",
            "time_from_host_syscnt",
            "get_host_duration",
            "time_from_syscnt",
            "_load_dev_start_time",
            "_load_dev_start_path_line_by_line",
            "_load_host_start_time",
            "_check_monotonic_and_cnt",
            "get_freq",
            "_load_local_time_offset",
            "trans_into_local_time",
        ),
    ),
    ("analysis/viewer/msproftx_viewer.py", ("format_tx_timeline_data", "format_tx_ex_timeline_data")),
    ("analysis/profiling_bean/basic_info/host_start.py", ("__init__", "host_mon")),
    ("analysis/common_func/ms_constant/number_constant.py", ()),
)


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def entries(directory, budget):
    result = []
    with os.scandir(directory) as iterator:
        for entry in iterator:
            budget[0] += 1
            if budget[0] > MAX_ENTRIES:
                raise ValueError(f"directory scan exceeds {MAX_ENTRIES} entries")
            result.append(entry)
    return sorted(result, key=lambda entry: entry.name)


def discover_roots(inputs):
    """Find capture roots, stopping before traversing their potentially huge raw data."""
    roots, pending, visited, budget = set(), [], set(), [0]
    for value in inputs:
        path = Path(value).expanduser().resolve(strict=True)
        parent = next((p for p in (path, *path.parents) if p.name.endswith("_ascend_pt")), None)
        if parent is not None:
            roots.add(parent)
        elif path.is_dir():
            pending.append((path, 0))
        else:
            raise ValueError(f"no *_ascend_pt ancestor for file: {path}")
    while pending:
        path, depth = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        items = entries(path, budget)
        if path.name.endswith("_ascend_pt") or any(PROFILER_INFO.fullmatch(item.name) for item in items):
            roots.add(path)
        elif depth < MAX_DEPTH:
            pending.extend((Path(item.path), depth + 1) for item in items if item.is_dir(follow_symlinks=False))
        else:
            print(f"DISCOVERY_DEPTH_LIMIT={path}")
        if len(roots) > MAX_ROOTS:
            raise ValueError(f"more than {MAX_ROOTS} capture roots; provide narrower inputs")
    if len(roots) > MAX_ROOTS:
        raise ValueError(f"more than {MAX_ROOTS} capture roots; provide narrower inputs")
    if not roots:
        raise ValueError("no capture root found; no profile clock evidence was inspected")
    return sorted(roots)


def metadata_files(root):
    budget, found = [0], []
    for item in entries(root, budget):
        if PROFILER_INFO.fullmatch(item.name) and item.is_file(follow_symlinks=False):
            found.append(Path(item.path))
        elif item.name.startswith("PROF_") and item.is_dir(follow_symlinks=False):
            for subdir in entries(item.path, budget):
                if not re.fullmatch(r"host|device_\d+", subdir.name) or not subdir.is_dir(follow_symlinks=False):
                    continue
                found.extend(
                    Path(file.path)
                    for file in entries(subdir.path, budget)
                    if CANN_METADATA.fullmatch(file.name) and file.is_file(follow_symlinks=False)
                )
    if len(found) > MAX_FILES:
        raise ValueError(f"more than {MAX_FILES} metadata files; ambiguous capture")
    return sorted(found)


def read_small(path):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes")
    with path.open("r", encoding="utf-8-sig") as source:
        text = source.read(MAX_FILE_BYTES + 1)
    if len(text) > MAX_FILE_BYTES or "\x00" in text:
        raise ValueError("oversized or binary file")
    return text


def parse_metadata(path):
    text = read_small(path)
    if re.search(r"\.json(?:\.\d+)?$", path.name):
        return json.loads(text)
    if path.name.startswith(("start_info", "end_info")):
        return ast.literal_eval(text)
    if path.name.startswith(("dev_start", "dev_end")) and not text.lstrip().startswith("["):
        values = {}
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            key = key.strip()
            if separator and normalized(key) in CLOCK_KEYS | UNIT_KEYS:
                if key in values:
                    raise ValueError("duplicate device clock field")
                values[key] = value.strip()
        return values
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(text)
    return {section: dict(parser.items(section)) for section in parser.sections()}


def clock_fields(data):
    found, visited = [], 0

    def visit(value, path, depth):
        nonlocal visited
        visited += 1
        if depth > MAX_DEPTH or visited > MAX_ENTRIES:
            raise ValueError("metadata structure exceeds inspection limit")
        if isinstance(value, dict):
            for key, child in value.items():
                name = normalized(str(key))
                if name in SENSITIVE_KEYS or "proxy" in name or name.startswith("environment"):
                    continue
                child_path = f"{path}.{key}" if path else str(key)
                numeric = (
                    (isinstance(child, int) and not isinstance(child, bool))
                    or (isinstance(child, float) and math.isfinite(child))
                ) or (isinstance(child, str) and NUMBER.fullmatch(child.strip()) is not None)
                if name in CLOCK_KEYS and numeric:
                    found.append({"key": child_path, "value": child, "type": type(child).__name__})
                elif name in CLOCK_KEYS and isinstance(child, str):
                    match = NUMBER_WITH_UNIT.fullmatch(child.strip())
                    if re.fullmatch(r"CPU\[\d+\]\.Frequency", child_path, re.I):
                        # This exact diagnostic field must distinguish N/A from an unusual frequency format.
                        found.append(
                            {
                                "key": child_path,
                                "value": child[:MAX_LITERAL_CHARS],
                                "type": "frequency_literal",
                                "truncated": len(child) > MAX_LITERAL_CHARS,
                            }
                        )
                    elif match and match[2].lower() in UNITS:
                        found.append({"key": child_path, "value": child, "type": "unparsed_unit_string"})
                    else:
                        found.append({"key": child_path, "type": "unparsed_string", "value_omitted": True})
                elif name in UNIT_KEYS and isinstance(child, str) and child.strip().lower() in UNITS:
                    found.append({"key": child_path, "value": child, "type": "declared_unit"})
                elif name == "syscntenable" and isinstance(child, bool):
                    found.append({"key": child_path, "value": child, "type": "bool"})
                elif name in VERSION_KEYS and isinstance(child, str) and re.fullmatch(r"[\w.+-]{1,100}", child):
                    found.append({"key": child_path, "value": child, "type": "version"})
                elif isinstance(child, (dict, list)):
                    visit(child, child_path, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", depth + 1)

    visit(data, "", 0)
    if len(found) > MAX_FIELDS:
        found.sort(key=lambda row: not re.search(r"(?:^|\.)(?:CPU\[0\]|DeviceInfo(?:\[0\])?)\.", row["key"], re.I))
        omitted = len(found) - MAX_FIELDS
        found = found[:MAX_FIELDS] + [{"key": "<field_limit>", "type": "truncation", "omitted_count": omitted}]
    return found


def inspect_root(root):
    print(f"PROFILE_ROOT={root}")
    files = metadata_files(root)
    if not files:
        print("METADATA_MISSING: no recognized existing clock metadata files")
    for path in files:
        try:
            values = clock_fields(parse_metadata(path))
            print(
                "CLOCK_METADATA "
                + json.dumps({"file": str(path.relative_to(root)), "fields": values}, ensure_ascii=False)
            )
        except (OSError, ValueError, SyntaxError, configparser.Error) as error:
            # Do not echo parser exceptions: they may contain unrelated metadata values.
            print(f"METADATA_UNREADABLE file={path.relative_to(root)}; error_type={type(error).__name__}")
    print(f"METADATA_FILES={len(files)}; absent fields are unknown, not zero")


def inspect_installed_source():
    try:
        distribution = importlib.metadata.distribution("torch-npu")
    except importlib.metadata.PackageNotFoundError:
        print("INSTALLED_SOURCE=unavailable; torch-npu distribution metadata not found")
        return
    print(f"INSTALLED_TORCH_NPU={distribution.version}; current installation does not prove capture-time version")
    remaining = MAX_SOURCE_LINES
    for relative in SOURCE_FILES:
        path = Path(distribution.locate_file(relative))
        try:
            lines = read_small(path).splitlines()
        except (OSError, ValueError):
            print(f"SOURCE_MISSING={relative}")
            continue
        for index, line in enumerate(lines):
            if (
                remaining
                and SOURCE_KEYWORDS.search(line)
                and not re.search(r"environ|proxy|password|credential", line, re.I)
            ):
                print(f"CLOCK_SOURCE {relative}:{index + 1}: {line.strip()}")
                remaining -= 1
    print("SOURCE_SCOPE=selected conversion lines only; not executed, not a calibration")


def cann_source_paths(roots):
    """Probe fixed known installation layouts, never recursively walk the toolkit."""
    if len(roots) > MAX_ROOTS:
        raise ValueError(f"more than {MAX_ROOTS} CANN installation roots")
    found = {}
    for root in roots:
        root = Path(root).expanduser()
        for prefix in CANN_SOURCE_PREFIXES:
            for relative, methods in CANN_SOURCE_METHODS:
                path = (root / prefix / relative).resolve()
                if path.is_file():
                    found[path] = (relative, methods)
    return sorted(found.items())


def source_method_lines(text, names):
    """Return method bodies as source text; parsing AST does not execute the module."""
    lines, result = text.splitlines(), {}
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in SOURCE_CONSTANTS for target in node.targets
        ):
            result.setdefault("<clock_constants>", []).append(
                [(index, lines[index - 1]) for index in range(node.lineno, node.end_lineno + 1)]
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            excluded = set()
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    if isinstance(statement.value.value, str):
                        excluded.update(range(statement.lineno, statement.end_lineno + 1))
            selected = [
                (index, lines[index - 1])
                for index in range(node.lineno, node.end_lineno + 1)
                if index not in excluded and lines[index - 1].strip() and not lines[index - 1].lstrip().startswith("#")
            ]
            result.setdefault(node.name, []).append(selected)
    return result


def inspect_cann_source(roots):
    paths = cann_source_paths(roots)
    print(f"CANN_SOURCE_FILES={len(paths)}; installed source is not proof of the capture/export version")
    seen, remaining = set(), MAX_CANN_SOURCE_LINES
    for path, (relative, methods) in paths:
        seen.add(relative)
        try:
            blocks = source_method_lines(read_small(path), methods)
        except (OSError, ValueError, SyntaxError):
            print(f"CANN_SOURCE_UNREADABLE={path}")
            continue
        print(f"CANN_SOURCE_FILE={path}")
        for method in (("<clock_constants>",) if "<clock_constants>" in blocks else ()) + methods:
            if method not in blocks:
                print(f"CANN_METHOD_MISSING={method}")
            for block in blocks.get(method, []):
                count = min(len(block), MAX_METHOD_LINES, remaining)
                print(f"CANN_METHOD={method}; lines={len(block)}; shown={count}; truncated={count < len(block)}")
                for number, line in block[:count]:
                    if re.search(r"environ|proxy|password|credential", line, re.I):
                        print(f"  {number}: [sensitive source line omitted]")
                    else:
                        print(f"  {number}: {line}")
                remaining -= count
    for relative, _ in CANN_SOURCE_METHODS:
        if relative not in seen:
            print(f"CANN_SOURCE_MISSING={relative}")
    print("CANN_SOURCE_SCOPE=fixed paths and selected method bodies only; never imported or executed")


def mstx_db_paths(root):
    """Only inspect known, already parsed CANN SQLite locations."""
    return sorted(
        set(root.glob("PROF_*/host/sqlite/msproftx.db")) | set(root.glob("PROF_*/device_*/sqlite/msproftx.db"))
    )


def read_mstx_tables(path):
    # A live WAL could require shared-memory writes or make an immutable read stale.
    for suffix in ("-wal", "-journal"):
        journal = Path(str(path) + suffix)
        if journal.exists() and journal.stat().st_size:
            raise ValueError("active WAL/journal; use a completed export")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
        deadline = time.monotonic() + DB_QUERY_TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        tables = {}
        for table in MSTX_TABLES:
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if not {"message", "start_time", "end_time"}.issubset(columns):
                continue
            rows = connection.execute(
                f'SELECT message, start_time, end_time FROM "{table}" WHERE message LIKE ? LIMIT ?',
                (CALLBACK_PREFIX + "%", MAX_MSTX_ROWS + 1),
            ).fetchall()
            if len(rows) > MAX_MSTX_ROWS:
                raise ValueError("MSTX row limit exceeded; no partial fit")
            tables[table] = [row for row in rows if isinstance(row[0], str) and row[0].startswith(CALLBACK_PREFIX)]
        return tables
    finally:
        connection.close()


def unique_markers(rows):
    unique, repeated = {}, set()
    for message, start, end in rows:
        if message in unique:
            repeated.add(message)
        try:
            values = (Decimal(str(start)), Decimal(str(end)))
        except InvalidOperation as error:
            raise ValueError("non-numeric marker timestamp") from error
        if not all(value.is_finite() for value in values) or values[1] < values[0]:
            raise ValueError("invalid marker timestamp or duration")
        unique[message] = values
    return {key: value for key, value in unique.items() if key not in repeated}, len(repeated)


def compare_mstx_export(db_rows, trace_rows):
    raw, raw_duplicates = unique_markers(db_rows)
    exported, trace_duplicates = unique_markers(trace_rows)
    matches = sorted((exported[name][0], name) for name in raw.keys() & exported.keys())
    result = {
        "matched_unique": len(matches),
        "db_duplicate_names": raw_duplicates,
        "trace_duplicate_names": trace_duplicates,
        "db_unmatched": len(raw.keys() - exported.keys()),
    }
    if len(matches) < 2:
        return result
    first, last = matches[0][1], matches[-1][1]
    raw_delta = raw[last][0] - raw[first][0]
    trace_delta = exported[last][0] - exported[first][0]
    if raw_delta <= 0 or trace_delta <= 0:
        result["invalid_order"] = True
        return result
    # Units of the database are deliberately not assumed to be hardware counter ticks.
    ratio = raw_delta / trace_delta
    result["raw_units_per_trace_us"] = str(ratio)
    result["trace_span_us"] = str(trace_delta)
    result["if_raw_ns_export_span_change_us"] = str(trace_delta - raw_delta / 1000)
    residuals = [exported[name][0] - exported[first][0] - (raw[name][0] - raw[first][0]) / ratio for _, name in matches]
    result["endpoint_affine_residual_range_us"] = [str(min(residuals)), str(max(residuals))]
    result["samples"] = [
        {
            "name": name,
            "db_start": str(raw[name][0]),
            "db_end": str(raw[name][1]),
            "trace_start_us": str(exported[name][0]),
            "trace_end_us": str(exported[name][1]),
        }
        for name in dict.fromkeys(matches[index][1] for index in (0, len(matches) // 2, len(matches) - 1))
    ]
    return result


def inspect_mstx_export(root):
    paths = mstx_db_paths(root)
    if not paths:
        print("MSTX_EXPORT=unavailable; existing msproftx.db not found; no parser launched")
        return
    trace_path = root / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not trace_path.is_file():
        print("MSTX_EXPORT=unavailable; trace_view.json not found")
        return
    print(f"MSTX_EXPORT_READ={trace_path}; streaming existing JSON; may take time", flush=True)
    trace_rows = []
    last_progress = time.monotonic()
    for index, event in enumerate(events(trace_path), 1):
        if index % MAX_ENTRIES == 0 and time.monotonic() - last_progress >= EXPORT_PROGRESS_SECONDS:
            print(f"MSTX_EXPORT_PROGRESS events={index}; callbacks={len(trace_rows)}", flush=True)
            last_progress = time.monotonic()
        if event.get("ph") == "X" and str(event.get("name", "")).startswith(CALLBACK_PREFIX):
            start = Decimal(str(event["ts"]))
            trace_rows.append((event["name"], start, start + Decimal(str(event["dur"]))))
            if len(trace_rows) > MAX_MSTX_ROWS:
                raise ValueError("trace callback limit exceeded; no partial fit")
    for path in paths:
        try:
            tables = read_mstx_tables(path)
            print(f"MSTX_DB={path.relative_to(root)}; supported_tables={list(tables)}")
            for table, rows in tables.items():
                result = compare_mstx_export(rows, trace_rows)
                print("MSTX_EXPORT_MATCH " + json.dumps({"table": table, **result}, ensure_ascii=False))
        except (sqlite3.Error, ValueError, OSError) as error:
            print(f"MSTX_DB_UNREADABLE={path.relative_to(root)}; error_type={type(error).__name__}")
    print(
        "MSTX_EXPORT_SCOPE=unique full marker names only; database units not assumed; "
        "endpoint fit is a conversion check, not an independent physical-clock calibration; "
        "JSON float rounding can affect sub-microsecond residuals"
    )


def sample_host_clocks(seconds):
    names = ("CLOCK_MONOTONIC_RAW", "CLOCK_MONOTONIC", "CLOCK_REALTIME")
    if not all(hasattr(time, name) for name in names):
        print("HOST_CLOCK_SAMPLE=unsupported; requires Linux raw/monotonic/realtime clocks")
        return
    samples = []
    deadline = time.monotonic() + seconds
    while True:
        row = {}
        for name in names[1:]:
            before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            value = time.clock_gettime_ns(getattr(time, name))
            after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            row[name] = (value - (before + after) // 2, (before + after) // 2, after - before)
        samples.append(row)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(SAMPLE_INTERVAL_SECONDS, remaining))
    for name in names[1:]:
        first, last = samples[0][name], samples[-1][name]
        elapsed = (last[1] - first[1]) / 1e9
        delta_us = (last[0] - first[0]) / 1000
        print(
            "HOST_CLOCK_SAMPLE "
            + json.dumps(
                dict(
                    clock=name,
                    samples=len(samples),
                    first_offset_ns=first[0],
                    last_offset_ns=last[0],
                    elapsed_raw_seconds=elapsed,
                    offset_change_us=delta_us,
                    change_us_per_second=delta_us / elapsed if elapsed > 0 else None,
                    first_bracket_ns=first[2],
                    last_bracket_ns=last[2],
                    bracket_rate_uncertainty_us_per_second=(first[2] + last[2]) / 2000 / elapsed
                    if elapsed > 0
                    else None,
                    max_bracket_ns=max(row[name][2] for row in samples),
                )
            )
        )
    print("HOST_CLOCK_SCOPE=current terminal time; does not reconstruct capture or identify the MSTX clock")


def sample_seconds(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 30:
        raise argparse.ArgumentTypeError("seconds must be finite and between 0 and 30")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--sample-host-clocks-seconds", type=sample_seconds, default=0)
    parser.add_argument(
        "--audit-mstx-export",
        action="store_true",
        help="also read existing msproftx.db and trace_view.json; no offline exporter is launched",
    )
    parser.add_argument(
        "--cann-root", action="append", type=Path, help="CANN installation root; repeat for multiple roots"
    )
    args = parser.parse_args()
    print(
        "READ ONLY: existing small clock metadata and installed source; "
        "no Torch/model/NPU/parser processes, no file writes."
    )
    print("METADATA_VALUES=original values; no inferred units/frequencies or timestamp correction")
    print("SCRIPT_REVISION=profiler_clocks_v3; CPU Frequency literals preserved; units not inferred")
    if args.audit_mstx_export:
        print(
            "EXPORT_AUDIT=read existing SQLite/JSON only; match unique complete callback names; no timestamp correction"
        )
    try:
        roots = discover_roots(args.inputs)
        print(f"PROFILE_ROOT_COUNT={len(roots)}")
        for root in roots:
            inspect_root(root)
            if args.audit_mstx_export:
                inspect_mstx_export(root)
        inspect_installed_source()
        inspect_cann_source(args.cann_root or CANN_ROOTS)
        if args.sample_host_clocks_seconds:
            sample_host_clocks(args.sample_host_clocks_seconds)
    except (OSError, ValueError) as error:
        parser.exit(1, f"INSPECTION_FAILED: {error}\n")


if __name__ == "__main__":
    main()
