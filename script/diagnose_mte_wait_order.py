# SPDX-License-Identifier: Apache-2.0
"""Compare two MTE wait orders while holding mapping behind the consumer gate."""

import argparse
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

from diagnose_mte_resource_scope import host_memory_snapshot
from diagnose_mte_wait_placement import append_evidence, graph_evidence, request_outcome, run_once
from run_host_sync_stage1 import npu_number

WORKER_STACK_LINES = 24
WORKER_STACK_LINE_CHARS = 600


def case_facts(run_dir, code, early):
    service_path = run_dir / "mte/inference_python.log"
    service = service_path.read_text(encoding="utf-8", errors="replace") if service_path.is_file() else ""
    report_path = run_dir / "graph_diagnosis.txt"
    report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else ""
    ends = []
    malformed = False
    for line in service.splitlines():
        if "[MOE-MTE WAIT_PLACEMENT] " in line:
            try:
                record = json.loads(line.split("[MOE-MTE WAIT_PLACEMENT] ", 1)[1])
                if record.get("phase") == "capture_end":
                    ends.append(record)
            except (ValueError, AttributeError):
                malformed = True
    valid_slots = all(
        isinstance(record.get("published_slots"), list) and all(type(slot) is int for slot in record["published_slots"])
        for record in ends
    )
    published = [slot for record in ends for slot in record["published_slots"]] if valid_slots else []
    ledger_ok = bool(ends) and valid_slots and not malformed and len(published) == len(set(published))
    ledger_ok = ledger_ok and all(
        record.get("published_slots") == record.get("joined_slots")
        and record.get("pending_waits") == 0
        and record.get("pending_layers") == []
        for record in ends
    )
    graph = graph_evidence(run_dir, early)
    counts = [int(match[1]) for line in graph if (match := re.search(r"structural_pairs=(\d+)", line))]
    structure_ok = ledger_ok and bool(published) and sum(counts) == len(published)
    structure_ok = structure_ok and len(counts) == len(ends) and not any("UNVERIFIED" in line for line in graph)

    # A watchdog snapshot is not an execution trace. Only inspect the active
    # worker's stack and pending prefetch slots from that same diagnostic block.
    blocks = service.split("[MOE-MTE DIAGNOSTIC BEGIN]")[1:]
    block = blocks[-1].split("[MOE-MTE DIAGNOSTIC END]", 1)[0] if blocks else ""
    active = re.search(r"active_callback=\((\d+),", block)
    worker_stack = []
    in_worker = False
    for line in block.splitlines():
        if "Python thread=" in line:
            in_worker = "active_worker=True" in line
        elif in_worker:
            worker_stack.append(line)
    pending = []
    pending_valid = False
    for line in block.splitlines():
        if "[MOE-MTE PENDING] " in line:
            try:
                entries = json.loads(line.split("[MOE-MTE PENDING] ", 1)[1])
                pending = [
                    entry
                    for entry in entries
                    if entry["slot"] in published
                    and entry["state"]["request_generation"] > entry["state"]["done_generation"]
                ]
                pending_valid = True
            except (ValueError, TypeError, KeyError):
                pending = []
                pending_valid = False
    active_prefetch_pending = bool(active) and int(active[1]) in {entry["slot"] for entry in pending}
    timed_out = {
        (int(slot), int(generation))
        for slot, generation in re.findall(r"\[MOE-MTE FATAL\] slot=(\d+) generation=(\d+) exceeded \d+s", service)
    }
    active_timed_out = bool(active) and any(
        entry["slot"] == int(active[1]) and (entry["slot"], entry["state"]["request_generation"]) in timed_out
        for entry in pending
    )
    return dict(
        request_outcome=request_outcome(report, code),
        expected_early_wait=early,
        capture_ledger_valid=bool(ledger_ok),
        structural_pairs=sum(counts),
        expected_prefetch_slots=len(published),
        expected_structure=bool(structure_ok),
        watchdog_snapshot_observed=bool(blocks),
        pending_snapshot_valid=pending_valid,
        pending_prefetch_count=len(pending) if pending_valid else None,
        active_prefetch_pending=active_prefetch_pending,
        worker_in_load_sync=any("_synchronize_h2d" in line for line in worker_stack),
        watchdog_timeout_for_active_prefetch=active_timed_out,
        active_prefetch_snapshot=next((entry for entry in pending if active and entry["slot"] == int(active[1])), None),
        worker_stack_tail=[line[:WORKER_STACK_LINE_CHARS] for line in worker_stack[-WORKER_STACK_LINES:]],
        worker_stack_omitted=max(0, len(worker_stack) - WORKER_STACK_LINES),
        worker_stack_truncated=sum(len(line) > WORKER_STACK_LINE_CHARS for line in worker_stack[-WORKER_STACK_LINES:]),
        evidence_boundary="capture topology and one Host snapshot; no kernel PC or resource-cycle proof",
    )


def run_comparison(root, output, npu, dest):
    facts = {}
    for case, early in (("late_wait", False), ("early_wait", True)):
        dest.write(f"\nCASE={case}; mapping_after_consumer_gate=True; strict_wait_before_gate={early}\n")
        dest.write(host_memory_snapshot() + "\n")
        dest.flush()
        case_root = output / case
        case_root.mkdir()
        run_dir = case_root / "run"
        code = run_once(root, run_dir, npu, trace_startup=True, no_load_diagnostics=True, wait_before_gate=early)
        append_evidence(dest, run_dir, code, trace_startup=True, wait_before_gate=early)
        runner_summary = run_dir / "summary.txt"
        if runner_summary.is_file():
            for line in runner_summary.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("kind="):
                    dest.write("CALLBACK_GROUP " + line + "\n")
        facts[case] = case_facts(run_dir, code, early)
        dest.write("WAIT_ORDER_FACTS " + json.dumps(dict(case=case, **facts[case])) + "\n")
        dest.flush()
        if not early and not (
            facts[case]["request_outcome"] == "finite_requests_passed" and facts[case]["expected_structure"]
        ):
            dest.write("BLOCKING_PROGRESS=CONTROL_NOT_ESTABLISHED；不启动提前wait组，先解释本轮对照失败或结构缺失。\n")
            return
    target = facts["early_wait"]
    if target["expected_structure"] and target["request_outcome"] == "finite_requests_passed":
        verdict = "EARLY_WAIT_LATE_MAPPING_PASSED"
    elif (
        target["expected_structure"]
        and target["active_prefetch_pending"]
        and target["worker_in_load_sync"]
        and target["watchdog_timeout_for_active_prefetch"]
    ):
        verdict = "EARLY_WAIT_LATE_MAPPING_PENDING_AT_LOAD_SYNC"
    else:
        verdict = "INCONCLUSIVE"
    dest.write(f"BLOCKING_PROGRESS={verdict}\n")
    dest.write("若提前wait组在预取加载同步处pending，支持提前等待重叠窗口参与；仍不能证明驻留资源或访存机制。\n")
    dest.write("若两组通过，只能说本轮允许提前wait的图未复现阻塞；实际重叠、映射时序和环境变化仍需区分。\n")
    dest.write("未得到具体未完成设备任务/PC时，不声称旧Transpose卡点已复现；不自动追加原路径或限核组。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", required=True, type=npu_number)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run inside the green Linux NPU container")
    root = Path(__file__).resolve().parents[1]
    output = root / "bench_results" / f"mte_wait_order_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    print(f"SUMMARY FILE: {summary}", flush=True)
    with summary.open("w", encoding="utf-8") as dest:
        dest.write("阻塞定位：两组都延后映射写回，只改变预取严格wait相对计算gate的位置。\n")
        dest.write("先跑延后wait对照，完成请求且图结构符合后，才跑提前wait；两个MTE服务顺序执行。\n")
        dest.write("沿用原短请求1次预热+3次请求、max_tokens=5；两组均开图debug以核对顺序，关闭逐项加载诊断。\n")
        dest.write("核数、原始拷贝、worker策略、加载流、严格完成条件相同；不做callback/性能对比。\n")
        run_comparison(root, output, args.npu, dest)
    print(f"仅返回检查脱敏后的文件内容：{summary}", flush=True)


if __name__ == "__main__":
    main()
