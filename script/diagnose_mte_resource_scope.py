# SPDX-License-Identifier: Apache-2.0
"""合并验证 Torch 原生控核主流对照与增加 DSA 辅流限核，输出单份精简摘要。"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from run_host_sync_stage1 import npu_number

CASE_TIMEOUT_SECONDS = 3000
PROGRESS_SECONDS = 15
CLEANUP_SECONDS = 45
REQUESTED_CUBE_CORES = 14
REQUESTED_ROUTING_CORES = 5


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
    parser.add_argument("--npu", type=npu_number, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_resource_scope_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    print(f"SUMMARY FILE: {summary}", flush=True)
    with summary.open("w", encoding="utf-8") as dest:
        dest.write("纯 MTE 严格等待、原始权重拷贝。诊断对照，不作为性能或精度验收。\n")
        dest.write("先用 Torch 原生接口只限制主流；仅在同类 scale 等待复现后增加 DSA 辅流上限。\n")
        for name, aux_cores in (("native_main", 0), ("native_main_aux", REQUESTED_CUBE_CORES)):
            dest.write(f"\nCASE {name}: main=14/28 routing=5 aux={'14/28' if aux_cores else 'unchanged'}\n")
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
                dest.write("未形成有效的同类阻塞对照，停止后续组，避免重复加载模型。\n")
                break
    print(f"仅返回此文件内容：{summary}", flush=True)


if __name__ == "__main__":
    main()
