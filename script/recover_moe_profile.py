"""在新目录离线重新解析已有采集；保留原始文件，不启动模型。"""

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from profile_storage import GIB, PROFILE_RESERVE_GIB, require_raw_data, require_space
from run_verified_cpp_tests import validate_profile_trace


def run_parser(command, stream, timeout):
    """Isolate Linux parser children so a timeout cannot leave them running."""
    with subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
    ) as child:
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    source = args.profile_root.resolve()
    output = args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("输入与输出目录不能互相包含")
    workers = sorted(source.rglob("*_ascend_pt"))
    if not workers:
        parser.error("未找到 *_ascend_pt 采集目录")
    for worker in workers:
        require_raw_data(worker)
    copy_bytes = sum(
        entry.stat().st_size
        for worker in workers
        for entry in worker.rglob("*")
        if entry.is_file() and not {"ASCEND_PROFILER_OUTPUT", "logs"}.intersection(entry.relative_to(worker).parts)
    )
    require_space(output, minimum_gib=PROFILE_RESERVE_GIB + copy_bytes / GIB)
    output.mkdir(parents=True, exist_ok=False)
    failures = 0
    for worker in workers:
        relative = worker.relative_to(source)
        target = output / relative
        print(f"重新解析：{relative}", flush=True)
        shutil.copytree(worker, target, ignore=shutil.ignore_patterns("ASCEND_PROFILER_OUTPUT", "logs"))
        log = target.parent / f"{target.name}_reparse.log"
        command = [
            sys.executable,
            "-B",
            "-c",
            "import sys; from torch_npu.profiler.profiler import analyse; "
            "analyse(profiler_path=sys.argv[1], max_process_number=1)",
            str(target.parent),
        ]
        try:
            with log.open("w", encoding="utf-8") as stream:
                run_parser(command, stream, args.timeout)
            trace = target / "ASCEND_PROFILER_OUTPUT/trace_view.json"
            count, labels = validate_profile_trace(trace, require_mstx=True)
            print(f"可视化有效：events={count}; MSTX={labels}; 导入目录={target}", flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures += 1
            print(f"未恢复：{type(exc).__name__}: {exc}; 解析日志={log}", flush=True)
            if log.exists():
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-20:]), flush=True)
    print(f"离线解析结束：成功={len(workers) - failures}; 失败={failures}；原始采集保持不变。")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
