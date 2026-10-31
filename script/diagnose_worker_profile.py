# SPDX-License-Identifier: Apache-2.0
"""只读检查五组worker实验的profile原始数据、时间线和解析错误。"""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

from inspect_moe_profile import scan_trace

CASES = ("baseline", "allocated", "atomic", "spin", "baseline_repeat")
READ_BYTES = 1024 * 1024
MAX_LOGS = 6
MAX_LINES = 18
MAX_WORKERS = 8
MAX_LINE_CHARS = 1000
ERROR_WORDS = ("error", "exception", "traceback", "failed", "errno", "no space", "permission denied", "timeout")
LOG_PRIORITY = ("CANNTimeline", "ProfilingParser", "CANNExport", "CANNAnalyze", "Relation", "TraceView")


def excerpt(path):
    """Bound reads and output, retaining upstream errors rather than full logs."""
    size = path.stat().st_size
    with path.open("rb") as source:
        content = source.read(READ_BYTES)
        if size > READ_BYTES:
            source.seek(max(READ_BYTES, size - READ_BYTES))
            content += b"\n[... middle omitted ...]\n" + source.read(READ_BYTES)
    lines = content.decode("utf-8", errors="replace").splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if any(word in line.lower() for word in ERROR_WORDS) or any(
            marker in line for marker in ("/stop_profile", "cleanup_begin", "PROFILE INCOMPLETE")
        ):
            selected.update(range(max(0, index - 1), min(len(lines), index + 7)))
    indices = sorted(selected) if selected else list(range(max(0, len(lines) - 3), len(lines)))
    if len(indices) > MAX_LINES:
        indices = indices[: MAX_LINES // 2] + indices[-MAX_LINES // 2 :]
    return "\n".join(lines[index][:MAX_LINE_CHARS] for index in indices)


def raw_inventory(worker):
    """Presence and nonempty bytes are recovery prerequisites, not completeness proofs."""
    records = []
    for prof in sorted(worker.glob("PROF_*")):
        if not prof.is_dir():
            continue
        data_rows = []
        for data in sorted(prof.glob("*/data")):
            sizes = [path.stat().st_size for path in data.rglob("*") if path.is_file()]
            data_rows.append(dict(path=str(data.relative_to(worker)), files=len(sizes), bytes=sum(sizes)))
        records.append(
            dict(
                directory=prof.name,
                start_info=len(list(prof.rglob("start_info*"))),
                end_info=len(list(prof.rglob("end_info*"))),
                data=data_rows,
            )
        )
    framework = worker / "FRAMEWORK/torch.op_range"
    return dict(
        cann=records,
        framework_bytes=framework.stat().st_size if framework.is_file() else None,
        profiler_info=len(list(worker.glob("profiler_info*.json"))),
        profiler_metadata=(worker / "profiler_metadata.json").is_file(),
    )


def audit_worker(worker, *, check_trace=True):
    print(f"WORKER {worker}", flush=True)
    print("INPUTS " + json.dumps(raw_inventory(worker), ensure_ascii=False), flush=True)
    trace = worker / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    check = None
    if not trace.is_file():
        print("TRACE: missing")
    elif not check_trace:
        print(f"TRACE: bytes={trace.stat().st_size}; 本组沿用上轮检查，不重复读完整文件")
    else:
        print(f"TRACE: bytes={trace.stat().st_size}; 正在流式验证到文件末尾……", flush=True)
        try:
            check = scan_trace(trace, require_mte=True, large_gap_us=100.0)
            print(
                "TRACE_CHECK "
                + json.dumps(
                    {key: check[key] for key in ("json_complete", "usable", "issues", "host_callbacks")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if check["usable"]:
                for gap in check["host_gaps"]:
                    print("HOST_CALLBACK_GAP " + json.dumps(gap, ensure_ascii=False))
                print("基本可视化检查通过；不能据此否认解析器报错或证明所有关联完整。")
        except (OSError, ValueError, UnicodeError) as exc:
            print(f"TRACE_INVALID {type(exc).__name__}: {exc}")
    logs = sorted(worker.rglob("*.log"))
    logs.sort(
        key=lambda path: (
            next((index for index, name in enumerate(LOG_PRIORITY) if name in path.name), len(LOG_PRIORITY)),
            str(path),
        )
    )
    print(f"PARSER_LOGS count={len(logs)}; selected={min(len(logs), MAX_LOGS)}")
    for log in logs[:MAX_LOGS]:
        print(f"LOG {log.relative_to(worker)}")
        try:
            print(excerpt(log))
        except OSError as exc:
            print(f"READ_ERROR {exc}")
    return check


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, help="默认检查一组成功基线和所有未通过导出的组")
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    if not (root / "profile").is_dir():
        parser.error(f"没有profile目录：{root / 'profile'}")
    print("只读检查；不导入Torch、不初始化NPU、不启动模型或解析进程、不写文件。", flush=True)
    print(f"RUN={root}; 当前剩余空间={shutil.disk_usage(root).free / 1024**3:.2f} GiB（不是采集时快照）")
    for package in ("torch", "torch-npu"):
        try:
            print(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}=not installed")
    for case in args.cases or CASES:
        report_path = root / case / "profile_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = {}
        status = report.get("profile_status", "unknown")
        passed = status in ("complete", "complete_with_warnings")
        if args.cases is None and passed and case != "baseline":
            continue
        print(f"\nCASE {case}: recorded_status={status}", flush=True)
        print("RECORDED_ERRORS " + json.dumps(report.get("profile_errors", []), ensure_ascii=False))
        service_log = root / case / "inference_cpp.log"
        if service_log.is_file():
            print("SERVICE_EXCERPT\n" + excerpt(service_log))
        workers = sorted((root / "profile" / case / "cpp").rglob("*_ascend_pt"))
        if not workers:
            print("WORKER: missing")
        for worker in workers[:MAX_WORKERS]:
            try:
                audit_worker(worker, check_trace=not passed or args.cases is not None)
            except OSError as exc:
                print(f"READ_ERROR {worker}: {exc}")
        if len(workers) > MAX_WORKERS:
            print(f"WORKERS_OMITTED={len(workers) - MAX_WORKERS}")
    print("检查结束。原始数据和元信息齐全才具备离线重解析条件；存在文件不保证数据完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
