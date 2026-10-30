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

from diagnose_mte_graph_mapping import diagnostic_environment
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


def run_once(
    root, run_dir, npu, trace_startup=False, no_load_diagnostics=False, verify_callback=False, no_graph_dump=False
):
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
    if verify_callback:
        command = [
            sys.executable,
            "-B",
            str(root / "script/run_moe_host_sync.py"),
            "--npu",
            str(npu),
            "--backend",
            "both",
            "--verification-suite",
            "--diagnose-prefetch-wait-at-join",
            "--diagnose-graph-dump",
            "--skip-profile",
            "--requests",
            "3",
            "--warmup-requests",
            "1",
            "--profile-timeout",
            "30",
            "--runtime-source",
            "/home/l00992607/vllm-ascend",
            "--test-script",
            "/home/l00992607/vllm-ascend/script/test_graph2.sh",
            "--output-dir",
            str(run_dir),
        ]
    if trace_startup:
        command.append("--trace-startup")
    if no_graph_dump:
        command.remove("--diagnose-graph-dump")
        command.append("--no-graph-dump")
    if no_load_diagnostics:
        command.append("--no-load-diagnostics")
    with (run_dir.parent / "wrapper.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(sys.platform == "linux"),
            env=diagnostic_environment() if verify_callback else None,
        )
        started = time.monotonic()
        try:
            while time.monotonic() - started < CASE_TIMEOUT_SECONDS * (2 if verify_callback else 1):
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


def append_evidence(dest, run_dir, code, trace_startup=False, verify_callback=False, no_graph_dump=False):
    report_path = run_dir / "graph_diagnosis.txt"
    report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else ""
    runner_summary = run_dir / "summary.txt"
    runner_text = runner_summary.read_text(encoding="utf-8", errors="replace") if runner_summary.is_file() else ""
    state = request_outcome(report, code)
    if verify_callback:
        report = runner_text
        try:
            pairs = [
                json.loads(line.removeprefix("PAIR_RESULT "))
                for line in report.splitlines()
                if line.startswith("PAIR_RESULT ")
            ]
        except ValueError:
            pairs = []
        state = (
            "finite_output_match"
            if code == 0 and len(pairs) == 1 and isinstance(pairs[0], dict) and pairs[0].get("status") == "passed"
            else "inconclusive_or_mismatch"
        )
    dest.write(f"WRAPPER_EXIT={code}; RESULT={state}\n")
    dest.write(concise_report(report) + "\n")
    if verify_callback:
        for line in report.splitlines():
            if line.startswith("PAIR_RESULT "):
                dest.write(line + "\n")
        for backend in ("callback", "mte"):
            section = report.split(f"=== REAL MOE HOST BACKEND: {backend} ===", 1)
            body = section[1].split("=== REAL MOE HOST BACKEND:", 1)[0] if len(section) == 2 else ""
            dest.write(f"BACKEND={backend} " + effective_configuration(body) + "\n")
            if code != 0:
                log = run_dir / backend / "inference_python.log"
                if log.is_file():
                    dest.write(f"BACKEND={backend}\n" + failure_context(log) + "\n")
    elif runner_summary.is_file():
        dest.write(effective_configuration(runner_text) + "\n")
    for marker, count in (("[MOE-MTE LOAD]", 8), ("[MOE-MTE PENDING]", 1), ("[MOE-MTE LAYOUT]", 1)):
        for line in [line for line in report.splitlines() if marker in line][-count:]:
            dest.write(line[:2000] + "\n")
    service_path = run_dir / "mte/inference_python.log"
    service = service_path.read_text(encoding="utf-8", errors="replace") if service_path.is_file() else ""
    dest.write(f"LOAD_DIAGNOSTIC_RECORDS={sum('[MOE-MTE LOAD]' in line for line in service.splitlines())}\n")
    if trace_startup:
        wrapper_path = run_dir.parent / "wrapper.log"
        wrapper = wrapper_path.read_text(encoding="utf-8", errors="replace") if wrapper_path.is_file() else ""
        dest.write("\n".join(startup_evidence(service, wrapper)) + "\n")
    placements = [line for line in service.splitlines() if "[MOE-MTE WAIT_PLACEMENT]" in line]
    # End-of-capture records prove that no split publication was left unjoined.
    end_records = [line for line in placements if '"phase": "capture_end"' in line]
    dest.write(f"WAIT_PLACEMENT_RECORDS={len(placements)}; CAPTURE_END_RECORDS={len(end_records)}\n")
    for line in dict.fromkeys(placements[:2] + end_records[-6:] + placements[-2:]):
        dest.write(line[:2000] + "\n")
    if no_graph_dump:
        dump_records = sum("[MOE-MTE GRAPH_DUMP_" in line for line in service.splitlines())
        dest.write(f"GRAPH_DUMP_REQUESTED=False; GRAPH_DUMP_RECORDS={dump_records}\n")
        dest.write("GATE_GRAPH=NOT_REQUESTED；仅有CPU捕获登记与汇合检查，不能独立证明本轮设备图依赖结构。\n")
    else:
        dest.write("\n".join(graph_evidence(run_dir)) + "\n")
    if verify_callback:
        dest.write("输出对照与MTE捕获结构分开判读；仅覆盖三种固定输入，不证明logits精度、全部捕获尺寸或性能。\n")
    if state in ("inconclusive", "inconclusive_or_mismatch"):
        if trace_startup and not service:
            dest.write(failure_context(runner_summary) + "\n")
        dest.write(failure_context(service_path) + "\n")
        dest.write(failure_context(run_dir.parent / "wrapper.log") + "\n")
        if not verify_callback:
            dest.write(kernel_failure_context() + "\n")
    dest.write("请求结果与捕获结构分开判读；图时间字段不是运行时PC/核占用。\n")
    if not no_graph_dump:
        dest.write("没有结构证据不等于没有干预；若结构不符，先检查本轮图，不重跑限核组。\n")
    if not verify_callback:
        dest.write("通过仅支持等待位置/重叠窗口参与触发，不证明资源死锁、输出精度或性能收益。\n")
    dest.write("仍失败时使用本轮快照，不套旧PID任务号；不自动追加其他实验。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", required=True, type=npu_number)
    parser.add_argument("--trace-startup", action="store_true", help="Trace this single run's startup and child exits")
    parser.add_argument(
        "--no-graph-dump",
        action="store_true",
        help="Only with --verify-callback: disable graph debug, keep other settings",
    )
    parser.add_argument(
        "--verify-callback",
        action="store_true",
        help="Run callback then MTE on three diverse prompts; requires --no-load-diagnostics",
    )
    parser.add_argument(
        "--no-load-diagnostics",
        action="store_true",
        help="Remove only the per-load diagnostic path; keep strict protocol waits",
    )
    args = parser.parse_args()
    if args.verify_callback and not args.no_load_diagnostics:
        parser.error("--verify-callback requires --no-load-diagnostics")
    if args.no_graph_dump and not args.verify_callback:
        parser.error("--no-graph-dump requires --verify-callback")
    if sys.platform != "linux":
        parser.error("请在绿区 Linux NPU 容器运行")
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_wait_placement_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    print(f"SUMMARY FILE: {summary}", flush=True)
    with summary.open("w", encoding="utf-8") as dest:
        if args.trace_startup and not args.verify_callback:
            dest.write("一次带启动观测的等待位置运行；阶段日志、崩溃栈与子进程退出状态，启动成功后继续原定请求。\n")
        if args.verify_callback:
            dest.write("有限正确性对照：先callback、后MTE，各启动一次服务，前一服务清理后才启动后一服务。\n")
            dest.write("同一启动源配置；三种不同输入、最多32个生成token；比较逐请求文本摘要与usage，不作性能比较。\n")
        dest.write("MTE等待位置设置：早发布，预取严格等待与映射写回延后到主流消费者汇合点。\n")
        if args.no_graph_dump:
            dest.write("本组关闭NPUGraph图debug/转储；保留启动观察、Runtime日志、计时与MSTX，仍不是性能测量。\n")
        if args.verify_callback:
            dest.write("callback保留原报告回调及加载流；MTE保留当前SyncLaunchStream和worker策略。\n")
        dest.write("不改核数、原始拷贝、worker策略和加载流，不作性能比较。\n")
        dest.write(f"LOAD_DIAGNOSTICS_REQUESTED={not args.no_load_diagnostics}\n")
        dest.write(
            "从第一批恢复普通批量加载；保留加载流最终同步、严格generation wait和消费者join。\n"
            if args.no_load_diagnostics
            else "保留前8批逐项诊断同步。\n"
        )
        dest.write("每服务1次预热（首个输入）+3次有限请求；跳过profiler；不重跑原主流组或辅助流组。\n")
        dest.write(host_memory_snapshot() + "\n")
        dest.flush()
        code = run_once(
            root,
            output / "consumer_join",
            args.npu,
            args.trace_startup,
            args.no_load_diagnostics,
            args.verify_callback,
            args.no_graph_dump,
        )
        append_evidence(
            dest, output / "consumer_join", code, args.trace_startup, args.verify_callback, args.no_graph_dump
        )
    print(f"仅返回检查脱敏后的文件内容：{summary}", flush=True)


if __name__ == "__main__":
    main()
