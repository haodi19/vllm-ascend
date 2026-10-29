# SPDX-License-Identifier: Apache-2.0
"""从既有日志提取启动退出线索；不启动模型，不修改输入。"""

import argparse
import datetime as dt
import os
import re
import subprocess
import uuid
from collections import deque
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MAX_FILES = 12
API_RESERVED_FILES = 3
MAX_REPORT_LINES = 180
SAMPLES_PER_END = 8
MAX_FAULT_RECORDS = 24
MAX_LOG_TEXT = 1400
FAULT_KINDS = ("fatal", "error", "symptom")
FULL_TIME = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})[ T-](\d{2}:\d{2}:\d{2})(?!\d)")
SHORT_TIME = re.compile(r"(?<![\d-])(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})(?!\d)")
ENGINE_PID = re.compile(r"\bEngineCore\w*\s+pid=(\d+)")
API_PID = re.compile(r"\bAPIServer\s+pid=(\d+)")
TEXT_LOG_NAME = re.compile(r"\.log(?:[._-]\d+)*$", re.I)
FAULT = re.compile(
    r"\[ERROR\]|\[MOE-MTE FATAL\]|Traceback|Fatal Python|(?:Error|Exception):|"
    r"\b(?:fatal|fault|segfault|aborted?|failed|failure|terminate(?:d)?|killed|SIGKILL|SIGSEGV|SIGABRT)\b|"
    r"timed out|\b(?:wait|task|sync|synchronize|execution)\s+timeout\b|"
    r"\btimeout\s+(?:occurred|expired|error|failure)\b|out of memory|"
    r"\bexit(?:[_ ]?(?:code|status))?\s*[:=]?\s*-?[1-9]\d*\b|"
    r"\bexited\s+(?:with\s+)?(?:code\s*[:=]?\s*)?-?[1-9]\d*\b",
    re.I,
)
TIMEOUT_PARAMETER = re.compile(r"\btimeout(?:_ms)?\s*[:=]\s*[+-]?\d+(?:\.\d+)?\s*(?:ms|us|s)?\b", re.I)
FATAL = re.compile(
    r"\bfatal\b|segmentation fault|\bsegfault\b|"
    r"\b(?:SIGKILL|SIGSEGV|SIGABRT|SIGBUS|SIGILL|SIGFPE|SIGTERM|aborted|killed)\b|"
    r"terminate called|out of memory|"
    r"(?:received|killed by|(?:exited|terminated) (?:with|by))\s+signal\s*[:=]?\s*[1-9]\d*\b|"
    r"\bexit(?:[_ ]?(?:code|status))?\s*[:=]?\s*-?[1-9]\d*\b|"
    r"\bexited\s+(?:with\s+)?(?:code\s*[:=]?\s*)?-?[1-9]\d*\b",
    re.I,
)
ERROR_LEVEL = re.compile(r"\[(?:ERROR|CRITICAL)\]|\b(?:ERROR|CRITICAL)\s+\d{2}-\d{2}|(?:Error|Exception):")
RETURN_STATUS = re.compile(r"\b(?:result|ret|error_code|errcode)\s*[:=]\s*(-?(?:0x[0-9a-f]+|\d+))\b", re.I)


def fault_kind(line):
    # Numeric timeout fields describe a call/event configuration, including positive
    # deadlines. Removing only that field still preserves an ERROR on the same line.
    message = TIMEOUT_PARAMETER.sub("", line)
    if FATAL.search(message):
        return "fatal"
    if ERROR_LEVEL.search(message):
        return "error"
    if FAULT.search(message) or any(
        int(match[1], 16 if "x" in match[1].lower() else 10) != 0 for match in RETURN_STATUS.finditer(message)
    ):
        return "symptom"
    return None


def is_fault(line):
    return fault_kind(line) is not None


def timestamp(line, year=None):
    match = FULL_TIME.search(line)
    if match:
        value = " ".join(match.groups())
    elif year is not None and (match := SHORT_TIME.search(line)):
        value = f"{year}-{match[1]} {match[2]}"
    else:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def identity(lines):
    engine_records = [match[1] for line in lines if (match := ENGINE_PID.search(line))]
    api_records = [match[1] for line in lines if (match := API_PID.search(line))]
    engines = list(dict.fromkeys(engine_records))
    apis = list(dict.fromkeys(api_records))
    full_dates = [value for line in lines if (value := timestamp(line))]
    year = full_dates[-1].year if full_dates else None
    engine = engine_records[-1] if engine_records else None
    api = api_records[-1] if api_records else None
    engine_times = [
        value
        for line in lines
        if (match := ENGINE_PID.search(line)) and match[1] == engine and (value := timestamp(line, year))
    ]
    api_times = [
        value
        for line in lines
        if (match := API_PID.search(line)) and match[1] == api and is_fault(line) and (value := timestamp(line, year))
    ]
    last_engine = engine_times[-1] if engine_times else None
    last_api = api_times[-1] if api_times else None
    window = None
    if year is not None and last_engine is not None and last_api is not None and last_api >= last_engine:
        window = (last_engine - dt.timedelta(seconds=15), last_api + dt.timedelta(seconds=15))
    return engines, apis, engine, api, last_engine, last_api, window


def service_path(run_dir):
    root = run_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run_dir 必须是已有目录")
    for relative in ("consumer_join/mte/inference_python.log", "mte/inference_python.log"):
        candidate = root / relative
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("服务日志符号链接越出输入运行目录，拒绝读取")
            return resolved
    raise ValueError("找不到 consumer_join/mte/inference_python.log 或 mte/inference_python.log")


def context_indices(lines, indices, radius=2, limit=24):
    selected = set()
    for index in indices:
        selected.update(range(max(0, index - radius), min(len(lines), index + radius + 1)))
    return sorted(selected)[-limit:]


def service_excerpt(lines, engine, api):
    result, emitted = [], set()

    def add_context(title, indices):
        result.append(title)
        for index in indices:
            if index not in emitted:
                result.append(f"L{index + 1}: {lines[index]}")
                emitted.add(index)

    selected = [i for i, line in enumerate(lines) if (match := ENGINE_PID.search(line)) and match[1] == engine]
    selected = [i for i in selected if lines[i].partition(")")[2].strip()]
    api_faults = [
        i for i, line in enumerate(lines) if (match := API_PID.search(line)) and match[1] == api and is_fault(line)
    ]
    add_context(
        "SERVICE_API_FINAL_ERROR_CONTEXT（保留 Failed core proc(s)/退出状态文本）:",
        context_indices(lines, api_faults[-3:], radius=1, limit=8),
    )
    add_context("SERVICE_ENGINE_TAIL（最后 12 条非空记录，原文件行号）:", selected[-12:])
    faults = [i for i, line in enumerate(lines) if is_fault(line) and not API_PID.search(line)]
    add_context(
        "SERVICE_NON_API_FAULT_CONTEXT（含无进程前缀行；历史候选不能自动归因）:",
        context_indices(lines, faults),
    )
    api_trace = next(
        (
            i
            for i, line in enumerate(lines)
            if "Traceback" in line and (match := API_PID.search(line)) and match[1] == api
        ),
        None,
    )
    if api_trace is not None:
        add_context(
            "FIRST_API_TRACE_CONTEXT（前后 3 行）:",
            range(max(0, api_trace - 3), min(len(lines), api_trace + 4)),
        )
    return result


def pid_files(root, pids):
    if not pids:
        return [], ["NATIVE_SKIPPED: PID 集合为空"]
    if not root.is_dir():
        return [], [f"NATIVE_ROOT_UNAVAILABLE: {root}"]
    pid_pattern = re.compile(r"(?<!\d)(?:" + "|".join(re.escape(pid) for pid in pids) + r")(?!\d)")
    files, errors = [], []
    filtered = 0

    def onerror(error):
        errors.append(f"NATIVE_DIRECTORY_UNREADABLE: {error}")

    for directory, directories, names in os.walk(root, followlinks=False, onerror=onerror):
        if len(Path(directory).relative_to(root).parts) >= 5:
            directories[:] = []
        for name in names:
            path = Path(directory) / name
            if not pid_pattern.search(name):
                continue
            if not TEXT_LOG_NAME.search(name) or path.is_symlink():
                filtered += 1
                continue
            try:
                if path.is_file():
                    files.append((path.stat().st_mtime, path))
            except OSError as error:
                errors.append(f"NATIVE_FILE_UNREADABLE: {path}: {error}")
    errors.append(f"NATIVE_FILTERED_NON_TEXT_OR_LINK={filtered}; root={root}; 只读 .log 及数字轮转后缀")
    return [path for _, path in sorted(files, reverse=True)], errors


def select_files(files, engine, api):
    engine_pattern = re.compile(rf"(?<!\d){re.escape(engine or 'NO_ENGINE')}(?!\d)")
    engines = [path for path in files if engine_pattern.search(path.name)]
    apis = [path for path in files if path not in engines]
    engine_count = MAX_FILES - API_RESERVED_FILES if api else MAX_FILES
    selected = engines[:engine_count] + apis[:API_RESERVED_FILES]
    selected.extend(path for path in engines + apis if path not in selected)
    return selected[:MAX_FILES]


def scan_native(path, window):
    """Scan the entire initial file snapshot; retain bounded samples, not all text."""
    counts = dict.fromkeys(FAULT_KINDS, 0)
    samples = {kind: [] for kind in FAULT_KINDS}
    undated_counts = dict.fromkeys(FAULT_KINDS, 0)
    undated_samples = {kind: [] for kind in FAULT_KINDS}
    evidence = dict(
        size=0,
        bytes_read=0,
        lines=0,
        window_lines=0,
        first_time=None,
        last_time=None,
        counts=counts,
        undated_faults=0,
        truncated=0,
        parameter_only=0,
    )
    normal_tail = deque(maxlen=2)
    pending = None

    def display(line):
        if len(line) > MAX_LOG_TEXT:
            evidence["truncated"] += 1
            return line[:MAX_LOG_TEXT] + " [TEXT_TRUNCATED]"
        return line

    with path.open("rb") as source:
        evidence["size"] = source.seek(0, 2)
        source.seek(0)
        while evidence["bytes_read"] < evidence["size"]:
            raw = source.readline(evidence["size"] - evidence["bytes_read"])
            if not raw:
                break
            evidence["bytes_read"] += len(raw)
            evidence["lines"] += 1
            number = evidence["lines"]
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            value = timestamp(line, window[1].year if window else None)
            dated = FULL_TIME.search(line) or SHORT_TIME.search(line)
            if value is not None:
                evidence["first_time"] = min(evidence["first_time"] or value, value)
                evidence["last_time"] = max(evidence["last_time"] or value, value)
            if dated:
                pending = None  # Even an invalid/old date stops an attached stack.
            if value is None:
                if not dated and pending is not None and len(pending["context"]) < 2:
                    pending["context"].append((number, display(line)))
                kind = fault_kind(line) if not dated else None
                if kind is not None:
                    evidence["undated_faults"] += 1
                    undated_counts[kind] += 1
                    retained = undated_samples[kind]
                    if len(retained) == 2:
                        retained.pop()
                    retained.append(dict(line=number, text=display(line), kind=kind, context=[]))
                continue
            if window is None or not window[0] <= value <= window[1]:
                continue
            evidence["window_lines"] += 1
            kind = fault_kind(line)
            if kind is None:
                evidence["parameter_only"] += bool(TIMEOUT_PARAMETER.search(line))
                normal_tail.append((number, display(line)))
                continue
            counts[kind] += 1
            pending = dict(line=number, text=display(line), kind=kind, context=[])
            retained = samples[kind]
            if len(retained) == 2 * SAMPLES_PER_END:
                retained.pop(SAMPLES_PER_END)
            retained.append(pending)
        evidence["end_size"] = source.seek(0, 2)
    evidence["records"] = [record for kind in FAULT_KINDS for record in samples[kind]]
    evidence["normal_tail"] = list(normal_tail)
    evidence["undated_counts"] = undated_counts
    evidence["undated_samples"] = [record for kind in FAULT_KINDS for record in undated_samples[kind]]
    evidence["undated_records"] = [(record["line"], record["text"]) for record in evidence["undated_samples"]]
    return evidence


def kernel_excerpt(window, pids):
    if window is None:
        return ["KERNEL_SKIPPED: 缺少可验证的完整日期时间窗口"]
    try:
        result = subprocess.run(
            ["dmesg", "--time-format", "iso", "--color=never"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"KERNEL_UNAVAILABLE: {error}"]
    if result.returncode:
        return [f"KERNEL_UNAVAILABLE: rc={result.returncode}; {result.stderr.strip()[:400]}"]
    records = []
    for line in result.stdout.splitlines():
        value = timestamp(line)
        if value is None or not window[0] <= value <= window[1] or not is_fault(line):
            continue
        same_pid = any(re.search(rf"(?<!\d){re.escape(pid)}(?!\d)", line) for pid in pids)
        records.append(f"KERNEL_WINDOW_CANDIDATE pid_text_match={same_pid}: {line}")
    return [
        "KERNEL: 按日志本地时间过滤；PID 文本相同仍须核对 host/container namespace，不能直接归因。",
        *records[-10:],
        f"KERNEL_WINDOW_MATCHES={len(records)}; 未匹配不排除未记录、权限或时区差异。",
    ]


def collect(run_dir, log_roots=None):
    service = service_path(run_dir)
    lines = service.read_text(encoding="utf-8", errors="replace").splitlines()
    engines, apis, engine, api, last_engine, last_api, window = identity(lines)
    ambiguous = len(engines) > 1 or len(apis) > 1
    if ambiguous:
        window = None
    pids = [pid for pid in (engine, api) if pid]
    result = [
        "INSPECTOR_VERSION=2; mode=full_file_snapshot; timeout参数不作故障，fatal/error/symptom分别计数与排序。",
        f"READ_ONLY_INPUT={service}; service_lines={len(lines)}",
        f"ENGINE_PIDS={engines}; API_PIDS={apis}; selected_engine={engine}; selected_api={api}",
        f"AMBIGUOUS_PROCESS_SET={ambiguous}；多 Engine/API PID 时仅展示服务文本，不自动关联原生/kernel 日志。",
        f"LAST_ENGINE_TIME={last_engine}; LAST_API_ERROR_TIME={last_api}; WINDOW={window}",
        "TIME_BASIS=假定服务/native/kernel 日志使用相同主机本地时区；短日期年份仅取本文件完整时间戳。",
        "PROCESS_EXIT=NOT_ESTABLISHED；服务/API 的退出码不能代替 EngineCore 退出码。",
    ]
    roots = (
        log_roots
        if log_roots is not None
        else [Path.home() / "ascend/log", Path("/var/log/npu/slog"), Path("/usr/slog")]
    )
    files = []
    if not pids or window is None:
        result.append("NATIVE_SKIPPED: 缺少 PID 或可靠日期时间窗口，不把其他时间记录关联本轮")
    else:
        for root in roots:
            found, errors = pid_files(root, pids)
            files.extend(found)
            result.extend(errors[:2])
    files = list(dict.fromkeys(files))

    def modified(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0

    files.sort(key=modified, reverse=True)
    selected_files = select_files(files, engine, api)
    result.append(
        f"NATIVE_ROOTS={roots}; PID命名文件={len(files)}; 未扫描文件={max(0, len(files) - MAX_FILES)}; "
        f"最多 {MAX_FILES} 文件，从首字节扫描至打开时大小；目录深度≤5，不跟随链接、不读压缩日志。"
    )
    result.append(f"NATIVE_FILE_QUOTA=Engine 优先，API 预留 {API_RESERVED_FILES} 文件；空余配额可互用。")
    totals = dict.fromkeys(FAULT_KINDS, 0)
    collected = []
    for file_id, path in enumerate(selected_files, 1):
        try:
            evidence = scan_native(path, window)
        except OSError as error:
            result.append(f"NATIVE_UNREADABLE {path}: {error}")
            continue
        collected.append((file_id, evidence))
        for kind in FAULT_KINDS:
            totals[kind] += evidence["counts"][kind]
        result.append(
            f"NATIVE_LOG id={file_id}: {path}; snapshot_bytes={evidence['size']}; "
            f"scanned_bytes={evidence['bytes_read']}; end_size={evidence['end_size']}; "
            f"lines={evidence['lines']}; window_lines={evidence['window_lines']}; "
            f"counts={evidence['counts']}; parameter_only={evidence['parameter_only']}; "
            f"undated_faults={evidence['undated_faults']}; text_truncated={evidence['truncated']}; "
            f"time_range={evidence['first_time']}..{evidence['last_time']}"
        )
    # Classify across all files before spending the display budget. Otherwise an
    # earlier file's routine ERROR lines can hide another file's crash or exit.
    candidates = [
        (file_id, record, correlated)
        for file_id, evidence in collected
        for key, correlated in (("records", True), ("undated_samples", False))
        for record in evidence[key]
    ]
    candidates.sort(key=lambda item: FAULT_KINDS.index(item[1]["kind"]))
    shown = candidates[:MAX_FAULT_RECORDS]
    undated_total = sum(evidence["undated_faults"] for _, evidence in collected)
    result.append(
        f"NATIVE_WINDOW_COUNTS={totals}; "
        f"NATIVE_OMITTED_FAULT_RECORDS={sum(totals.values()) - sum(item[2] for item in shown)}; "
        f"NATIVE_UNDATED_TOTAL={undated_total}; "
        f"NATIVE_OMITTED_UNDATED_RECORDS={undated_total - sum(not item[2] for item in shown)}; "
        "计数仅为分类候选；非零result/ret也不自动证明失败。L均为原始行号。"
    )
    for file_id, record, correlated in shown:
        label = "NATIVE_FAULT_CANDIDATE" if correlated else "NATIVE_UNDATED_UNCORRELATED"
        result.append(f"{label} kind={record['kind']} file={file_id} L{record['line']}: {record['text']}")
        for number, line in record["context"]:
            result.append(f"NATIVE_ATTACHED_CONTEXT file={file_id} L{number}: {line}")
    for file_id, evidence in collected:
        for number, line in evidence["normal_tail"][-1:]:
            result.append(f"NATIVE_NORMAL_TAIL（仅最后普通记录）file={file_id} L{number}: {line}")
    result.extend(kernel_excerpt(window, pids))
    result.extend(service_excerpt(lines, engine, api))
    footer = [
        "CAUSE=NOT_ESTABLISHED；上述仅为时间/PID 范围内候选，普通 Submit/同步日志及退出码 0 不证明退出原因。",
        "边界：未恢复已退出 EngineCore 的 waitpid 状态；未读 cgroup 累计值、进程环境、tensor 或 coredump。",
        "原日志完整保留在绿区；未命中不排除轮转、未选文件、扫描中增长/缩短、日期缺失、时区或 namespace 差异。",
    ]
    omitted = max(0, len(result) - (MAX_REPORT_LINES - len(footer) - 1))
    result = result[: MAX_REPORT_LINES - len(footer) - 1]
    return "\n".join([*(line[:1400] for line in result), f"REPORT_OMITTED_LINES={omitted}", *footer]) + "\n"


def write_summary(report, output_parent):
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / f"mte_startup_inspect_{uuid.uuid4().hex}"
    output.mkdir(exist_ok=False)
    summary = output / "summary.txt"
    with summary.open("x", encoding="utf-8") as destination:
        destination.write(report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--log-root", type=Path, action="append", help="指定原生日志目录，可重复；替换默认目录")
    args = parser.parse_args()
    report = collect(args.run_dir, args.log_root)
    output = write_summary(report, REPOSITORY / "bench_results")
    print(report, end="")
    print(f"SUMMARY={output}")


if __name__ == "__main__":
    main()
