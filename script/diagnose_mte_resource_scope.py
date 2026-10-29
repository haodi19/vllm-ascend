# SPDX-License-Identifier: Apache-2.0
"""合并验证 Torch 原生控核主流对照与增加 DSA 辅流限核，输出单份精简摘要。"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

from run_host_sync_stage1 import npu_number

CASE_TIMEOUT_SECONDS = 3000
PROGRESS_SECONDS = 15
CLEANUP_SECONDS = 45
REQUESTED_CUBE_CORES = 14
REQUESTED_ROUTING_CORES = 5
TRACE_CONTEXT_LINES = 40
LOG_LINE_CHARACTERS = 1500


def selected_cases(name):
    cases = (("native_main", 0), ("native_main_aux", REQUESTED_CUBE_CORES))
    return cases if name == "both" else tuple(case for case in cases if case[0] == name)


def host_memory_snapshot():
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return f"HOST_MEMORY unavailable: {exc}"
    return (
        "HOST_MEMORY "
        + "; ".join(line for line in lines if line.startswith(("MemTotal:", "MemAvailable:", "SwapFree:")))
        + "; 启动前快照，不代表容器限额或峰值需求"
    )


def failure_context(path):
    """Read bounded traceback context; preserve EngineCore errors before API fallout."""
    if not path.is_file():
        return f"LOG MISSING: {path}"
    engine_blocks, other_blocks = deque(maxlen=2), deque(maxlen=1)
    tail, errors = deque(maxlen=12), deque(maxlen=8)
    preceding = deque(maxlen=60)
    before_api = []
    active = None
    with path.open(encoding="utf-8", errors="replace") as source:
        for number, line in enumerate(source, 1):
            line = line.rstrip()[:LOG_LINE_CHARACTERS]
            record = f"{number}: {line}"
            tail.append(record)
            if "Traceback (most recent call last)" in line:
                if "APIServer" in line and not before_api:
                    before_api = list(preceding)
                active = []
                (engine_blocks if "EngineCore" in line else other_blocks).append(active)
            if active is not None and len(active) < TRACE_CONTEXT_LINES:
                active.append(record)
            if re.search(r"\b\w*(?:Error|Exception):|Killed|SIGKILL|SIGSEGV|out of memory", line, re.I):
                errors.append(record)
            preceding.append(record)
    blocks = engine_blocks or other_blocks
    sections = [f"READ ONLY LOG: {path}"]
    if not engine_blocks and before_api:
        sections.append("未发现 EngineCore Python 调用栈；首个 API 异常前最多60行（保留退出前阶段）：")
        sections.extend(before_api)
    elif blocks:
        sections.append("异常调用栈片段（优先 EngineCore；不自动判定根因）：")
        sections.extend(record for block in blocks for record in block)
    if errors:
        sections.append("末尾错误行：")
        sections.extend(errors)
    sections.append("日志尾部：")
    sections.extend(tail)
    return "\n".join(sections)


def kernel_failure_context():
    """Read kernel records only; unavailable records do not exclude external termination."""
    if sys.platform != "linux":
        return "KERNEL LOG: 非 Linux，跳过"
    try:
        result = subprocess.run(
            ["dmesg", "--ctime", "--color=never"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"KERNEL LOG UNAVAILABLE: {exc}"
    if result.returncode:
        return f"KERNEL LOG UNAVAILABLE: {result.stderr.strip()[:500]}"
    matches = deque(maxlen=16)
    for line in result.stdout.splitlines():
        if re.search(r"oom-kill|out of memory|killed process|segfault|NVRM|aicore.*error|device.*reset", line, re.I):
            matches.append(line[:LOG_LINE_CHARACTERS])
    return "KERNEL LOG 候选：需核对时间和 PID；容器 PID 可能不同，不能自动关联本次失败。\n" + (
        "\n".join(matches) if matches else "无匹配记录；不排除日志不可见、已轮转或外部终止。"
    )


def inspect_failure(output):
    case = output / "native_main_aux" if (output / "native_main_aux").is_dir() else output
    return (
        failure_context(case / "mte/inference_python.log")
        + "\n"
        + failure_context(case.parent / f"{case.name}.wrapper.log")
    )


def classify_report(text, exit_code):
    """Proceed only after the established scale wait, not arbitrary startup errors."""
    if exit_code == 0 and "RUNNER_EXIT=0" in text and "MOE_RESULT " in text:
        return "request_passed"
    actual_hcpre = []
    actual_routing = []
    actual_wait = []
    for line in text.splitlines():
        if line.startswith("GRAPH_KERNEL "):
            try:
                fields = json.loads(line[len("GRAPH_KERNEL ") :])
                name, blocks, _ = fields[-3:]
            except (ValueError, TypeError):
                continue
            if not isinstance(name, str):
                continue
            if "HcPre" in name:
                actual_hcpre.append(blocks)
            elif "MoeInitRouting" in name:
                actual_routing.append(blocks)
            elif "host_sync_wait" in name:
                actual_wait.append(blocks)
    if (
        "[MOE-MTE ACL_DETAIL]" in text
        and "ACL_WAIT_RETURN code=507046" in text
        and "w13_weight_scale" in text
        and actual_hcpre
        and all(blocks == REQUESTED_CUBE_CORES for blocks in actual_hcpre)
        and actual_routing
        and all(blocks == REQUESTED_ROUTING_CORES for blocks in actual_routing)
        and actual_wait
        and all(blocks == 1 for blocks in actual_wait)
    ):
        return "scale_wait_reproduced"
    return "inconclusive"


def concise_report(text):
    prefixes = (
        "COMPUTE_CUBE_REQUESTED=",
        "ROUTING_CORES_REQUESTED=",
        "DSA_CUBE_REQUESTED=",
        "NATIVE_STREAM_LIMITS=",
        "RESOURCE_API=",
        "RUNNER_EXIT=",
        "FAILED:",
        "MOE_RESULT ",
        "GRAPH_KERNEL ",
        "GRAPH_DUMP ",
        "MATCH ",
        "SNAPSHOT_MAPPING_",
    )
    markers = ("[MOE-MTE COMPUTE_LIMIT]", "[MOE-MTE DIAGNOSTIC BEGIN]", "[MOE-MTE FATAL]")
    lines = [
        line
        for line in text.splitlines()
        if line.startswith(prefixes)
        or any(marker in line for marker in markers)
        or (line.startswith("SNAPSHOT ") and "task_id=" in line)
    ]
    return "\n".join(dict.fromkeys(lines))


def stop_owned_process_group(process):
    """Interrupt this wrapper/runner group first so its inference cleanup can run."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.terminate()
    try:
        process.wait(timeout=CLEANUP_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=PROGRESS_SECONDS)


def run_case(root, output, npu, aux_cores):
    if output.exists():
        raise ValueError(f"拒绝覆盖已有实验目录：{output}")
    command = [
        sys.executable,
        "-B",
        str(root / "script/diagnose_mte_graph_mapping.py"),
        "--npu",
        str(npu),
        "--routing-cores",
        str(REQUESTED_ROUTING_CORES),
        "--compute-cube-cores",
        str(REQUESTED_CUBE_CORES),
        "--native-stream-limits",
        "--output-dir",
        str(output),
    ]
    if aux_cores:
        command += ["--dsa-cube-cores", str(aux_cores)]
    wrapper_log = output.parent / f"{output.name}.wrapper.log"
    with wrapper_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= CASE_TIMEOUT_SECONDS:
                    stop_owned_process_group(process)
                    return None, "总时限已到；已中断本轮包装器/运行器进程组。请检查本轮清理日志。"
                try:
                    code = process.wait(timeout=min(PROGRESS_SECONDS, CASE_TIMEOUT_SECONDS - elapsed))
                    break
                except subprocess.TimeoutExpired:
                    print(f"WAIT {output.name}: {time.monotonic() - started:.0f}s", flush=True)
        except BaseException:
            stop_owned_process_group(process)
            raise
    report = output / "graph_diagnosis.txt"
    if report.is_file():
        return code, report.read_text(encoding="utf-8", errors="replace")
    return code, "没有生成 graph_diagnosis.txt；包装器日志尾部：\n" + "\n".join(
        wrapper_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number)
    parser.add_argument(
        "--case",
        choices=("both", "native_main", "native_main_aux"),
        default="both",
        help="默认两组；可仅补跑此前未完成的一组",
    )
    inspect = parser.add_mutually_exclusive_group()
    inspect.add_argument("--inspect", type=Path, help="只读指定实验目录的启动错误；不加载模型、不写文件")
    inspect.add_argument("--inspect-latest", action="store_true", help="只读最近一次 mte_resource_scope 实验")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.inspect is not None or args.inspect_latest:
        if args.npu is not None or args.case != "both":
            parser.error("只读检查不接受 --npu/--case；不会重跑模型")
        output = args.inspect
        if args.inspect_latest:
            candidates = [path for path in (root / "bench_results").glob("mte_resource_scope_*") if path.is_dir()]
            if not candidates:
                parser.error("没有找到 mte_resource_scope 实验；可用 --inspect 指定目录")
            output = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        print("只读已有日志；不导入 Torch、不初始化 NPU、不启动模型、不写文件。")
        print(inspect_failure(output.resolve()))
        print(kernel_failure_context())
        return
    if args.npu is None:
        parser.error("运行实验必须指定 --npu；只读已有结果使用 --inspect-latest")
    output = root / "bench_results" / f"mte_resource_scope_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    print(f"SUMMARY FILE: {summary}", flush=True)
    with summary.open("w", encoding="utf-8") as dest:
        dest.write("纯 MTE 严格等待、原始权重拷贝。诊断对照，不作为性能或精度验收。\n")
        if args.case == "both":
            dest.write("先用 Torch 原生接口只限制主流；仅在同类 scale 等待复现后增加 DSA 辅流上限。\n")
        else:
            dest.write(f"单独补跑 {args.case}；需要结合此前主流对照判断，不自动认定两轮环境相同。\n")
        for name, aux_cores in selected_cases(args.case):
            dest.write(f"\nCASE {name}: main=14/28 routing=5 aux={'14/28' if aux_cores else 'unchanged'}\n")
            dest.write(host_memory_snapshot() + "\n")
            dest.flush()
            print(f"START {name}; 总时限={CASE_TIMEOUT_SECONDS}s", flush=True)
            code, text = run_case(root, output / name, args.npu, aux_cores)
            state = classify_report(text, code)
            dest.write(f"WRAPPER_EXIT={code}; RESULT={state}\n")
            dest.write((concise_report(text) or text) + "\n")
            dest.flush()
            print(f"DONE {name}: {state}", flush=True)
            if state == "request_passed":
                dest.write("本组请求通过，停止扩大改动；仍须检查输出正确性，不视为已证明根因。\n")
                break
            if state != "scale_wait_reproduced":
                dest.write(inspect_failure(output / name) + "\n")
                dest.write(kernel_failure_context() + "\n")
                dest.write("未形成有效的同类阻塞对照，停止后续组，避免重复加载模型。\n")
                break
    print(f"仅返回此文件内容：{summary}", flush=True)


if __name__ == "__main__":
    main()
