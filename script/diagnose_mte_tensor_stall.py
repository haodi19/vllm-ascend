"""原始量化 Tensor 拷贝复现与原生线程栈抓取；不用于性能对照。"""

import argparse
import contextlib
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"\[MOE-MTE NATIVE\] pid=(\d+) worker_tid=(\d+)")


def startup_excerpt(text):
    """Keep engine first-cause errors, not just the downstream API traceback."""
    lines = text.splitlines()
    if len(lines) <= 600:
        return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
    selected = set(range(max(0, len(lines) - 20), len(lines)))
    roots = [
        i for i, line in enumerate(lines) if "Traceback (most recent call last)" in line and "APIServer" not in line
    ]
    for index in roots[:3]:
        selected.update(range(max(0, index - 8), min(len(lines), index + 120)))
    errors = [
        i
        for i, line in enumerate(lines)
        if "APIServer" not in line and re.search(r"FAILED:|Error:|Exception:|\[MOE-MTE|\bERROR\b|\bFATAL\b", line)
    ]
    for index in errors[:24]:
        selected.update(range(max(0, index - 3), min(len(lines), index + 7)))
    return "\n".join(f"{i + 1}: {lines[i]}" for i in sorted(selected)[:600])


def read_failures(path, emit=print):
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.glob("run/*/inference_python.log")) or sorted(path.glob("*/inference_python.log"))
        if not paths and (path / "runner.log").is_file():
            paths = [path / "runner.log"]
    if not paths:
        raise ValueError(f"未找到服务日志或 runner.log: {path}")
    for log in paths:
        emit(f"READ-ONLY FIRST-CAUSE LOG: {log}")
        emit(startup_excerpt(log.read_text(encoding="utf-8", errors="replace")))


def selected_stacks(text, worker_tid):
    blocks = re.split(r"(?m)(?=^Thread \d+)", text)
    pattern = re.compile(r"host_sync|Session::poll|Synchronize|synchronize|aclrt|TaskQueue|Dequeue|Enqueue|gil", re.I)
    selected = [b for b in blocks if pattern.search(b) or (worker_tid and re.search(rf"LWP {worker_tid}\b", b))]
    # Keep the targeted worker first if its Python dummy thread was registered.
    selected.sort(key=lambda b: not (worker_tid and re.search(rf"LWP {worker_tid}\b", b)))
    return "\n".join(selected[:16]) if selected else text[:16000]


def descendant(pid, ancestor):
    """Only attach to an engine spawned by this invocation, never other jobs."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return False
        match = re.search(r"(?m)^PPid:\s*(\d+)", status)
        if not match:
            return False
        pid = int(match.group(1))
    return False


def capture(gdb, pid, tid, root, emit):
    command = [
        gdb,
        "--nx",
        "--batch",
        "-p",
        str(pid),
        "-ex",
        "set pagination off",
        "-ex",
        "set print frame-arguments none",
        "-ex",
        "thread apply all bt 20",
        "-ex",
        "detach",
    ]
    emit(f"NATIVE SNAPSHOT pid={pid} worker_tid={tid}; 最多 20 秒；暂停期间不能用于性能测量")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
        )
        text = result.stdout
        emit(f"gdb exit={result.returncode}")
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout or ""
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        emit("gdb 超时已终止；本次栈可能不完整")
    (root / f"native_{pid}.txt").write_text(text, encoding="utf-8")
    emit(selected_stacks(text, tid))
    if "#0" not in text:
        emit("NATIVE STACK UNAVAILABLE：检查 ptrace 权限或 gdb 输出，不能视为成功抓栈")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int)
    parser.add_argument("--read-log", type=Path, help="只读现有实验目录或日志，输出首个错误；不启动模型、不需要 gdb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", type=Path, help="可选：先从已有 MTE profile 分析纯设备时钟的交接间隔")
    args = parser.parse_args()
    if args.read_log:
        read_failures(args.read_log)
        if args.profile:
            from diagnose_moe_transfer_gaps import compare_callback_cycles

            compare_callback_cycles(args.profile)
        return 0
    if sys.platform != "linux" or args.npu is None or args.npu < 0:
        parser.error("请在 Linux 容器中指定物理 NPU 编号")
    gdb = shutil.which("gdb")
    if not gdb:
        parser.error("容器缺少 gdb，尚未启动模型。请先让管理员安装 gdb，再执行同一命令。")
    root = (args.output_dir or ROOT / "bench_results_tmp" / f"mte_native_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    run_dir = root / "run"
    command = [
        sys.executable,
        "-B",
        str(ROOT / "script/run_moe_host_sync.py"),
        "--npu",
        str(args.npu),
        "--backend",
        "mte",
        "--quant-copy",
        "tensor",
        "--diagnose-load-stages",
        "--requests",
        "1",
        "--timing-requests",
        "0",
        "--warmup-requests",
        "1",
        "--profile-timeout",
        "30",
        "--output-dir",
        str(run_dir),
    ]
    with (root / "diagnosis.txt").open("w", encoding="utf-8") as summary:

        def emit(text):
            print(text, flush=True)
            summary.write(text + "\n")
            summary.flush()

        emit(f"SUMMARY FILE: {root / 'diagnosis.txt'}")
        emit("复现 MTE + 原始 Tensor 拷贝。关闭重排/Storage 适配；仅诊断，不能作为性能结果。")
        if args.profile:
            # Lazy optional analyzer; no Torch/NPU imports in the supervisor.
            from diagnose_moe_transfer_gaps import load_trace, report_device_pipeline

            paths = [
                p
                for p in args.profile.rglob("trace_view.json")
                if "mte" in p.parts and p.parent.name == "ASCEND_PROFILER_OUTPUT"
            ]
            if not paths:
                emit("未找到已有 MTE trace_view.json；继续原始拷贝诊断")
            for path in paths:
                emit(f"DEVICE TRACE: {path}")
                with contextlib.redirect_stdout(io.StringIO()) as buffer:
                    _, _, device, _, hardware = load_trace(path)
                    if len(hardware) == 1:
                        report_device_pipeline(device)
                    else:
                        print("设备轨道不唯一，跳过设备配对")
                emit(buffer.getvalue())
        captured = set()
        with (root / "runner.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                last_progress = time.monotonic()
                while child.poll() is None:
                    for path in run_dir.glob("*/inference_python.log"):
                        with path.open("rb") as source:
                            source.seek(max(0, os.fstat(source.fileno()).st_size - 131072))
                            tail = source.read().decode(errors="replace")
                        for match in MARKER.finditer(tail):
                            pid, tid = map(int, match.groups())
                            if pid in captured:
                                continue
                            captured.add(pid)
                            if not descendant(pid, child.pid):
                                emit(f"拒绝附加 pid={pid}：不是本次启动进程的后代或已经退出")
                                continue
                            load_lines = [
                                line
                                for line in tail.splitlines()
                                if "[MOE-MTE LOAD]" in line or "active_callback=" in line
                            ]
                            emit("\n".join(load_lines[-18:]))
                            capture(gdb, pid, tid, root, emit)
                    if time.monotonic() - last_progress >= 30:
                        emit("WAIT: 本次复现仍在运行；原有启动/请求超时与服务清理继续生效")
                        last_progress = time.monotonic()
                    time.sleep(1)
                emit(f"RUNNER EXIT={child.returncode}; captured_engines={len(captured)}")
                text = (root / "runner.log").read_text(encoding="utf-8", errors="replace")
                emit("\n".join(text.splitlines()[-35:]))
                if child.returncode:
                    read_failures(root, emit)
                if not captured:
                    emit("没有触发抓栈：若推理通过，本轮未复现原阻塞；若失败，先查看上面的启动错误。")
                return child.returncode
            except BaseException:
                # Let the existing runner unwind its inference cleanup. Do not
                # kill device-wide processes or unrelated jobs on this NPU.
                child.send_signal(signal.SIGINT)
                raise


if __name__ == "__main__":
    raise SystemExit(main())
