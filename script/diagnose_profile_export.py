"""只读打印 profile 解析失败的上下文，不启动解析、不修改采集、不创建日志。"""

import argparse
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

READ_BYTES = 1024 * 1024
MAX_LOGS = 24
MAX_LINES = 24
ERROR_WORDS = ("error", "exception", "traceback", "failed", "errno", "no space", "permission denied")


def inspect_inputs(root):
    """Print raw-data locations only; do not expose tensor or metadata contents."""
    workers = sorted(root.rglob("*_ascend_pt"))
    for worker in workers:
        print(f"\nWORKER {worker.relative_to(root)}")
        print("直接子项：" + ", ".join(p.name for p in sorted(worker.iterdir())))
        profs = sorted(worker.glob("PROF_*"))
        print(f"直接 PROF_* 目录数量={sum(p.is_dir() for p in profs)}")
        for pattern in ("start_info*", "end_info*", "profiler_info*.json", "profiler_metadata.json", "torch.op_range"):
            matches = sorted(worker.rglob(pattern))
            print(f"{pattern}: {len(matches)}")
            for path in matches[:12]:
                print(f"  {path.relative_to(worker)} bytes={path.stat().st_size}")
        for prof in profs:
            if not prof.is_dir():
                continue
            for data in sorted(prof.glob("*/data")):
                files = [p for p in data.rglob("*") if p.is_file()]
                print(
                    f"  RAW {data.relative_to(worker)} files={len(files)} bytes={sum(p.stat().st_size for p in files)}"
                )
    # Locate CANN directories placed outside the expected worker directory as well.
    print("\n本次实验目录内全部 PROF_* 位置：")
    for prof in sorted(root.rglob("PROF_*")):
        if prof.is_dir():
            print(prof.relative_to(root))


def excerpt(path):
    with path.open("rb") as source:
        head = source.read(READ_BYTES)
        if path.stat().st_size > READ_BYTES:
            source.seek(max(READ_BYTES, path.stat().st_size - READ_BYTES))
            head += b"\n[... middle omitted ...]\n" + source.read(READ_BYTES)
    lines = head.decode("utf-8", errors="replace").splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if any(word in line.lower() for word in ERROR_WORDS):
            selected.update(range(max(0, index - 2), min(len(lines), index + 9)))
    if not selected:
        selected.update(range(max(0, len(lines) - 4), len(lines)))
    ordered = sorted(selected)
    if len(ordered) > MAX_LINES:
        ordered = ordered[: MAX_LINES // 2] + ordered[-MAX_LINES // 2 :]
    return "\n".join(f"{index + 1}: {lines[index]}" for index in ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--inputs-only", action="store_true", help="仅核对原始采集结构，省略大量解析日志")
    parser.add_argument("--validate-callback", action="store_true", help="流式验证已有 callback 时间线，不重解析")
    args = parser.parse_args()
    for package in ("torch", "torch-npu"):
        try:
            print(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}=not installed")
    for root in args.roots:
        print(f"\n=== {root} ===")
        if not root.is_dir():
            print("目录不存在")
            continue
        print(f"磁盘剩余 GiB={shutil.disk_usage(root).free / (1024**3):.2f}")
        if args.inputs_only:
            inspect_inputs(root)
        for trace in sorted(root.rglob("trace_view.json")):
            with trace.open("rb") as source:
                head = source.read(32)
                source.seek(max(0, trace.stat().st_size - 32))
                tail = source.read(32)
            print(f"TRACE {trace.relative_to(root)} bytes={trace.stat().st_size}; head={head!r}; tail={tail!r}")
            if args.validate_callback and "callback" in trace.relative_to(root).parts:
                # Lazy import keeps the metadata-only mode standalone.
                from run_verified_cpp_tests import validate_profile_trace

                print("正在流式验证 callback 时间线，不写文件……", flush=True)
                try:
                    count, labels = validate_profile_trace(trace, require_mstx=True)
                    print(f"CALLBACK TRACE READABLE: events={count}; MSTX={labels}; 文件={trace}")
                except (ValueError, OSError) as exc:
                    print(f"CALLBACK TRACE INVALID: {exc}")
        if args.inputs_only:
            continue
        logs = sorted(root.rglob("*.log"))
        # Prioritize upstream parsers rather than printing every cascading failure.
        logs.sort(key=lambda p: (not any(s in p.name for s in ("CANNTimeline", "Relation", "CANNAnalyze")), str(p)))
        print(f"解析日志数量={len(logs)}，最多显示 {MAX_LOGS} 个")
        for log in logs[:MAX_LOGS]:
            print(f"\nLOG {log.relative_to(root)}")
            try:
                print(excerpt(log))
            except OSError as exc:
                print(f"读取失败：{exc}")
    if shutil.which("ps"):
        result = subprocess.run(["ps", "-eo", "pid,ppid,stat,etime,args"], capture_output=True, text=True, timeout=10)
        print("\n=== 相关存活进程（只读，不终止） ===")
        paths = [str(root.resolve()) for root in args.roots]
        for line in result.stdout.splitlines():
            if any(path in line for path in paths) and "diagnose_profile_export" not in line:
                print(line[:800])


if __name__ == "__main__":
    main()
