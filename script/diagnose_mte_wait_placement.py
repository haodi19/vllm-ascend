# SPDX-License-Identifier: Apache-2.0
"""仅运行一次预取等待位置诊断，保留严格完成条件，汇总捕获结构与有限请求结果。"""

import argparse
import contextlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from diagnose_mte_resource_scope import (
    CASE_TIMEOUT_SECONDS,
    PROGRESS_SECONDS,
    concise_report,
    failure_context,
    host_memory_snapshot,
    kernel_failure_context,
    stop_owned_process_group,
)
from inspect_mte_graph_dump import event_candidates, identity
from run_host_sync_stage1 import npu_number
from summarize_mte_startup import startup_evidence

MAX_GRAPH_EXAMPLES = 2


def gated_wait_pairs(events):
    """Check unique event-name pairs in capture order; never infer execution times."""
    nodes = [node for node in events if isinstance(node, dict) and identity(node) is not None]
    keys = [identity(node) for node in nodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate model/stream/task identities; cannot establish capture order")
    pairs = []
    for model, stream in sorted({key[:2] for key in keys}):
        track = sorted((n for n in nodes if identity(n)[:2] == (model, stream)), key=identity)
        for index, wait in enumerate(track):
            if not wait.get("name", "").startswith("host_sync_wait") or index < 2:
                continue
            publish, gate_wait = track[index - 2 : index]
            if not publish.get("name", "").startswith("host_sync_publish"):
                continue
            if not re.fullmatch(r"EVENT_WAIT_\d+", gate_wait.get("name", "")):
                continue
            gate_records = event_candidates(nodes, gate_wait)
            if len(gate_records) != 1 or identity(gate_records[0])[1] == stream:
                continue
            if event_candidates(nodes, gate_records[0]) != [gate_wait]:
                continue
            following = track[index + 1 :]
            done_record = next((n for n in following if n.get("name", "").startswith("EVENT_RECORD_")), None)
            if done_record is None:
                continue
            between = following[: following.index(done_record)]
            # Mapping must remain after the strict wait. No additional host task
            # or wait may be silently interpreted as the mapping completion.
            if not between or not all(
                "MEMCPY" in str(n.get("args", {}).get("Task Type", "")).upper() or "MEMCPY" in n.get("name", "").upper()
                for n in between
            ):
                continue
            done_waits = event_candidates(nodes, done_record)
            if len(done_waits) != 1:
                continue
            if event_candidates(nodes, done_waits[0]) != [done_record]:
                continue
            gate_record, done_wait = gate_records[0], done_waits[0]
            if (
                identity(gate_record)[:2] != identity(done_wait)[:2]
                or identity(gate_record)[2] >= identity(done_wait)[2]
            ):
                continue
            pairs.append(
                {
                    "publish": identity(publish),
                    "gate_record": identity(gate_record),
                    "gate_wait": identity(gate_wait),
                    "strict_wait": identity(wait),
                    "mapping_tasks": [identity(n) for n in between],
                    "done_record": identity(done_record),
                    "done_wait": identity(done_wait),
                }
            )
    return pairs


def graph_evidence(run_dir):
    lines = []
    for path in sorted((run_dir / "mte/graph_debug").rglob("graph_dump*")):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            events = data.get("traceEvents", []) if isinstance(data, dict) else data
            if not isinstance(events, list):
                raise ValueError("not a trace event list")
            pairs = gated_wait_pairs(events)
            lines.append(f"GATE_GRAPH file={path.relative_to(run_dir)}; structural_pairs={len(pairs)}")
            for pair in pairs[:MAX_GRAPH_EXAMPLES]:
                lines.append("GATE_PAIR " + json.dumps(pair))
        except (OSError, ValueError, UnicodeError, TypeError, AttributeError) as exc:
            lines.append(f"GATE_GRAPH_UNVERIFIED {path.name}: {exc}")
    return lines or ["GATE_GRAPH_UNVERIFIED: 没有捕获图文件"]


def request_outcome(report, code):
    if code == 0 and "RUNNER_EXIT=0" in report and "MOE_RESULT " in report:
        return "finite_requests_passed"
    if any("w13_weight_scale" in line and "ACL_WAIT_RETURN code=507046" in line for line in report.splitlines()):
        return "scale_wait_reproduced"
    return "inconclusive"


def effective_configuration(text):
    """Retain only workload flags from the actual runner command, not shell prefixes."""
    commands = [
        line.split("Effective serve command: ", 1)[1]
        for line in text.splitlines()
        if line.startswith("Effective serve command: ")
    ]
    if len(commands) != 1:
        return "EFFECTIVE_CONFIG_UNAVAILABLE: 没有唯一的实际serve命令"
    try:
        words = shlex.split(commands[0])
    except ValueError as exc:
        return f"EFFECTIVE_CONFIG_UNAVAILABLE: {exc}"
    result = {}
    for flag in (
        "--model",
        "--speculative-config",
        "--compilation-config",
        "--additional-config",
        "--max-model-len",
        "--max-num-seqs",
    ):
        if flag in words and words.index(flag) + 1 < len(words):
            result[flag] = words[words.index(flag) + 1]
    if "serve" in words and words.index("serve") + 1 < len(words):
        result["model"] = words[words.index("serve") + 1]
    return "EFFECTIVE_CONFIG " + json.dumps(result, ensure_ascii=False)


def wrapper_cleanup_marker(process, log, reason):
    # Observation must not prevent the existing process-group cleanup.
    with contextlib.suppress(Exception):
        print(
            "[MOE-MTE RUNNER] "
            + json.dumps(
                dict(
                    event="wrapper_cleanup_begin",
                    wrapper_pid=process.pid,
                    reason=reason,
                    wall_time_ns=time.time_ns(),
                    monotonic_ns=time.monotonic_ns(),
                )
            ),
            file=log,
            flush=True,
        )


def run_once(root, run_dir, npu, trace_startup=False):
    command = [
        sys.executable,
        "-B",
        str(root / "script/diagnose_mte_graph_mapping.py"),
        "--npu",
        str(npu),
        "--prefetch-wait-at-join",
        "--output-dir",
        str(run_dir),
    ]
    if trace_startup:
        command.append("--trace-startup")
    with (run_dir.parent / "wrapper.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=(sys.platform == "linux")
        )
        started = time.monotonic()
        try:
            while time.monotonic() - started < CASE_TIMEOUT_SECONDS:
                try:
                    return process.wait(timeout=PROGRESS_SECONDS)
                except subprocess.TimeoutExpired:
                    print(f"WAIT consumer_join: {time.monotonic() - started:.0f}s", flush=True)
            if trace_startup:
                wrapper_cleanup_marker(process, log, "total_timeout")
            stop_owned_process_group(process)
            return None
        except BaseException:
            if trace_startup:
                wrapper_cleanup_marker(process, log, "interrupted")
            stop_owned_process_group(process)
            raise


def append_evidence(dest, run_dir, code, trace_startup=False):
    report_path = run_dir / "graph_diagnosis.txt"
    report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else ""
    state = request_outcome(report, code)
    dest.write(f"WRAPPER_EXIT={code}; RESULT={state}\n")
    dest.write(concise_report(report) + "\n")
    runner_summary = run_dir / "summary.txt"
    if runner_summary.is_file():
        dest.write(effective_configuration(runner_summary.read_text(encoding="utf-8", errors="replace")) + "\n")
    for marker, count in (("[MOE-MTE LOAD]", 8), ("[MOE-MTE PENDING]", 1), ("[MOE-MTE LAYOUT]", 1)):
        for line in [line for line in report.splitlines() if marker in line][-count:]:
            dest.write(line[:2000] + "\n")
    service_path = run_dir / "mte/inference_python.log"
    service = service_path.read_text(encoding="utf-8", errors="replace") if service_path.is_file() else ""
    if trace_startup:
        wrapper_path = run_dir.parent / "wrapper.log"
        wrapper = wrapper_path.read_text(encoding="utf-8", errors="replace") if wrapper_path.is_file() else ""
        dest.write("\n".join(startup_evidence(service, wrapper)) + "\n")
    placements = [line for line in service.splitlines() if "[MOE-MTE WAIT_PLACEMENT]" in line]
    # End-of-capture records prove that no split publication was left unjoined.
    end_records = [line for line in placements if '"phase": "capture_end"' in line]
    dest.write(f"WAIT_PLACEMENT_RECORDS={len(placements)}; CAPTURE_END_RECORDS={len(end_records)}\n")
    for line in placements[:2] + end_records[-6:] + placements[-2:]:
        dest.write(line[:2000] + "\n")
    dest.write("\n".join(graph_evidence(run_dir)) + "\n")
    if state == "inconclusive":
        if trace_startup and not service:
            dest.write(failure_context(runner_summary) + "\n")
        dest.write(failure_context(service_path) + "\n")
        dest.write(failure_context(run_dir.parent / "wrapper.log") + "\n")
        dest.write(kernel_failure_context() + "\n")
    dest.write("请求结果与捕获结构分开判读；图时间字段不是运行时PC/核占用。\n")
    dest.write("没有结构证据不等于没有干预；若结构不符，先检查本轮图，不重跑限核组。\n")
    dest.write("通过仅支持等待位置/重叠窗口参与触发，不证明资源死锁、输出精度或性能收益。\n")
    dest.write("仍失败时使用本轮快照，不套旧PID任务号；不自动追加其他实验。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", required=True, type=npu_number)
    parser.add_argument("--trace-startup", action="store_true", help="Trace this single run's startup and child exits")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("请在绿区 Linux NPU 容器运行")
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_wait_placement_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    print(f"SUMMARY FILE: {summary}", flush=True)
    with summary.open("w", encoding="utf-8") as dest:
        if args.trace_startup:
            dest.write("一次带启动观测的等待位置运行；阶段日志、崩溃栈与子进程退出状态，启动成功后继续原定请求。\n")
        dest.write("等待位置设置：早发布，预取严格等待与映射写回延后到主流消费者汇合点。\n")
        dest.write("不改核数、原始拷贝、worker策略和加载流；保留逐项诊断同步，不作性能比较。\n")
        dest.write("1次预热+3次有限请求；跳过profiler；不重跑原主流组或辅助流组。\n")
        dest.write(host_memory_snapshot() + "\n")
        dest.flush()
        code = run_once(root, output / "consumer_join", args.npu, args.trace_startup)
        append_evidence(dest, output / "consumer_join", code, args.trace_startup)
    print(f"仅返回检查脱敏后的文件内容：{summary}", flush=True)


if __name__ == "__main__":
    main()
