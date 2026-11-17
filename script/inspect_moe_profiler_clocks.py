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
import time
from pathlib import Path

MAX_ROOTS = 16
MAX_ENTRIES = 10000
MAX_DEPTH = 10
MAX_FILES = 64
MAX_FILE_BYTES = 1024 * 1024
MAX_FIELDS = 100
MAX_SOURCE_LINES = 48
MAX_CANN_SOURCE_LINES = 220
MAX_METHOD_LINES = 40
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
            "_load_host_start_time",
            "get_freq",
            "_load_local_time_offset",
            "trans_into_local_time",
        ),
    ),
    ("analysis/viewer/msproftx_viewer.py", ("format_tx_timeline_data", "format_tx_ex_timeline_data")),
    ("analysis/profiling_bean/basic_info/host_start.py", ("__init__", "host_mon")),
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
                    if match and match[2].lower() in UNITS:
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
        for method in methods:
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
        "--cann-root", action="append", type=Path, help="CANN installation root; repeat for multiple roots"
    )
    args = parser.parse_args()
    print(
        "READ ONLY: existing small clock metadata and installed source; "
        "no Torch/model/NPU/parser processes, no file writes."
    )
    print("METADATA_VALUES=original values; no inferred units/frequencies or timestamp correction")
    print("SCRIPT_REVISION=profiler_clocks_v2; unit-bearing strings remain unparsed; unknown strings are redacted")
    try:
        roots = discover_roots(args.inputs)
        print(f"PROFILE_ROOT_COUNT={len(roots)}")
        for root in roots:
            inspect_root(root)
        inspect_installed_source()
        inspect_cann_source(args.cann_root or CANN_ROOTS)
        if args.sample_host_clocks_seconds:
            sample_host_clocks(args.sample_host_clocks_seconds)
    except (OSError, ValueError) as error:
        parser.exit(1, f"INSPECTION_FAILED: {error}\n")


if __name__ == "__main__":
    main()
