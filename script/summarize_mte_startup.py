# SPDX-License-Identifier: Apache-2.0
"""Summarize one startup trace without guessing a child status from the API exit."""

import json
import re

MARKERS = ("[MOE-MTE STARTUP]", "[MOE-MTE PROCESS]", "[MOE-MTE RUNNER]")
ENGINE_PID = re.compile(r"\bEngineCore\w*\s+pid=(\d+)")
STACK_HEADER = re.compile(r"Fatal Python error:|(?:Current thread|Thread) 0x[0-9a-f]+", re.I)
MAX_TRACE_RECORDS = 80
MAX_STACK_LINES = 80
MAX_STACK_BLOCK_LINES = 24
MAX_TEXT_CHARS = 1800


def integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def clipped(text):
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + " [TEXT_TRUNCATED]"


def trace_records(service_text, wrapper_text):
    records, seen = [], set()
    malformed = duplicates = 0
    for source, text in (("service", service_text), ("wrapper", wrapper_text)):
        for line in text.splitlines():
            for marker in MARKERS:
                if marker not in line:
                    continue
                try:
                    record = json.loads(line.split(marker, 1)[1].strip())
                    if not isinstance(record, dict):
                        raise ValueError("trace is not an object")
                except (ValueError, TypeError):
                    malformed += 1
                    break
                key = (marker, json.dumps(record, sort_keys=True))
                if key in seen:
                    duplicates += 1
                else:
                    seen.add(key)
                    records.append((marker, record, source))
                break
    return records, malformed, duplicates


def after_observed_markers(exit_record, markers):
    """None means timestamps cannot establish order; False is no preceding marker."""
    unknown = False
    for marker in markers:
        for clock in ("monotonic_ns", "wall_time_ns"):
            observed, requested = integer(exit_record.get(clock)), integer(marker.get(clock))
            if observed is not None and requested is not None:
                if observed >= requested:
                    return True
                break
        else:
            unknown = True
    return None if unknown else False


def stack_excerpt(service_text):
    lines = service_text.splitlines()
    headers = [index for index, line in enumerate(lines) if STACK_HEADER.search(line)]
    # Preserve the current thread even when many other Python threads are dumped first.
    priority = sorted(headers, key=lambda index: ("Current thread" not in lines[index], index))
    selected = set()
    total_candidates = set()
    for index in priority:
        block = [index]
        for following in range(index + 1, min(len(lines), index + MAX_STACK_BLOCK_LINES)):
            if not lines[following].strip() or STACK_HEADER.search(lines[following]):
                break
            block.append(following)
        total_candidates.update(block)
        for number in block:
            if len(selected) < MAX_STACK_LINES:
                selected.add(number)
    if not headers:
        return ["STARTUP_NATIVE_STACK=NOT_OBSERVED；未见崩溃栈不能排除原生故障或SIGKILL。"]
    result = [
        f"STARTUP_NATIVE_STACK_CANDIDATES={len(headers)}; shown_lines={len(selected)}; "
        f"omitted_candidate_lines={len(total_candidates - selected)}; 每段最多{MAX_STACK_BLOCK_LINES}行。",
        "栈摘录来自合并服务日志；未据此自动认定属于EngineCore或认定退出原因。",
    ]
    result.extend(f"STARTUP_STACK L{index + 1}: {clipped(lines[index])}" for index in sorted(selected))
    return result


def startup_evidence(service_text, wrapper_text=""):
    records, malformed, duplicates = trace_records(service_text, wrapper_text)
    stages = [record for marker, record, _ in records if marker == MARKERS[0]]
    processes = [record for marker, record, _ in records if marker == MARKERS[1]]
    runners = [record for marker, record, _ in records if marker == MARKERS[2]]
    engine_pids = {int(match[1]) for match in ENGINE_PID.finditer(service_text)}
    engine_pids.update(
        pid
        for record in stages
        if (pid := integer(record.get("pid"))) is not None
        and isinstance(record.get("phase"), str)
        and record["phase"].startswith(("runner.", "draft."))
    )
    engine_pids.update(
        pid
        for record in processes
        if record.get("event") == "start"
        and isinstance(record.get("name"), str)
        and record["name"].startswith("EngineCore")
        and (pid := integer(record.get("child_pid"))) is not None
    )
    observer_pids = {
        pid
        for record in processes
        if record.get("event") == "armed" and (pid := integer(record.get("observer_pid"))) is not None
    }
    armed_engines, pending, exceptions = set(), {}, []
    unmatched_end = invalid_stage = 0
    for record in stages:
        pid, tid = integer(record.get("pid")), integer(record.get("tid"))
        phase, event = record.get("phase"), record.get("event")
        if pid is None or not isinstance(phase, str) or event not in ("begin", "end", "exception"):
            invalid_stage += 1
            continue
        key = (pid, tid, phase)
        if event == "begin":
            pending.setdefault(key, []).append(record)
        elif pending.get(key):
            pending[key].pop()
            if event == "end" and phase.endswith(".faulthandler"):
                armed_engines.add(pid)
        else:
            unmatched_end += 1
        if event == "exception":
            exceptions.append((pid, phase, record.get("exception_type")))
    gaps = sum(record.get("event") == "observer_gap" for record in processes)
    verified = bool(observer_pids and engine_pids and engine_pids <= armed_engines)
    verified = verified and not (malformed or invalid_stage or unmatched_end or gaps)
    result = [
        f"STARTUP_TRACE_STATUS={'ARMED_EVIDENCE_PRESENT' if verified else 'UNVERIFIED'}; "
        f"stage_records={len(stages)}; process_records={len(processes)}; runner_records={len(runners)}; "
        f"malformed={malformed}; duplicates={duplicates}; invalid_stage={invalid_stage}; unmatched_end={unmatched_end}",
        f"STARTUP_ENGINE_PIDS={sorted(engine_pids)}; OBSERVER_ARMED_PIDS={sorted(observer_pids)}; "
        f"ENGINE_FAULTHANDLER_ARMED={sorted(armed_engines)}; OBSERVER_GAPS={gaps}",
    ]
    for pid in sorted(engine_pids):
        exits = [
            record
            for record in processes
            if record.get("event") == "exit"
            and integer(record.get("child_pid")) == pid
            and integer(record.get("returncode")) is not None
        ]
        if not exits:
            result.append(f"STARTUP_ENGINE_EXIT pid={pid}: NOT_OBSERVED；不使用API退出码替代。")
            continue
        for record in exits:
            observer = integer(record.get("observer_pid"))
            requests = [
                candidate
                for candidate in processes
                if candidate.get("event") == "signal_request"
                and integer(candidate.get("child_pid")) == pid
                and integer(candidate.get("observer_pid")) == observer
            ]
            cleanups = [
                candidate
                for candidate in runners
                if candidate.get("event") == "cleanup_begin"
                and observer is not None
                and integer(candidate.get("api_pid")) == observer
            ]
            wrapper_cleanups = [candidate for candidate in runners if candidate.get("event") == "wrapper_cleanup_begin"]
            result.append(
                "STARTUP_ENGINE_EXIT "
                + json.dumps(
                    dict(
                        pid=pid,
                        observer_pid=observer,
                        returncode=record["returncode"],
                        signal=record.get("signal"),
                        observed_after_parent_signal_request=after_observed_markers(record, requests),
                        observed_after_runner_cleanup=after_observed_markers(record, cleanups),
                        observed_after_run_wrapper_cleanup=after_observed_markers(record, wrapper_cleanups),
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    if not engine_pids:
        result.append("STARTUP_ENGINE_EXIT=NOT_OBSERVED；未识别EngineCore PID，不能将其他子进程状态当引擎状态。")
    open_phases = [(key, len(begins)) for key, begins in pending.items() if begins]
    result.append(f"STARTUP_OPEN_PHASES={len(open_phases)}; PHASE_EXCEPTIONS={len(exceptions)}")
    for (pid, tid, phase), count in open_phases[:MAX_TRACE_RECORDS]:
        result.append(f"STARTUP_OPEN pid={pid} tid={tid} phase={phase} unmatched_begin={count}")
    for pid, phase, exception in exceptions[:MAX_TRACE_RECORDS]:
        result.append(f"STARTUP_PHASE_EXCEPTION pid={pid} phase={phase} type={exception}")
    result.append(
        f"STARTUP_TRACE_RECORDS_SHOWN={min(len(records), MAX_TRACE_RECORDS)}; "
        f"OMITTED={max(0, len(records) - MAX_TRACE_RECORDS)}"
    )
    # Keep process/cleanup evidence even if a large model emits many phase records.
    priorities = sorted(range(len(records)), key=lambda index: records[index][0] == MARKERS[0])
    for index in sorted(priorities[:MAX_TRACE_RECORDS]):
        marker, record, source = records[index]
        result.append(
            f"TRACE source={source} {marker} {clipped(json.dumps(record, sort_keys=True, ensure_ascii=False))}"
        )
    result.extend(stack_excerpt(service_text))
    result.extend(
        (
            "退出时间是父进程取得状态的观测时刻，不是实际死亡时刻；清理请求之前/之后仅表示日志顺序。",
            "清理后的负退出码可能受清理影响；没有请求记录也不能排除直接os.kill或外部信号。",
            "未闭合阶段仅缩小最后可见CPU执行范围，不证明对应NPU操作是根因；观测启用不等于捕获全部终止。",
        )
    )
    return result
