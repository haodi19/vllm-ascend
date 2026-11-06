# SPDX-License-Identifier: Apache-2.0
"""流式请求TPOT计时及同次profile的Host间隔归零假设；不导入Torch。"""

import collections
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events
from inspect_moe_profile import _interval, copy_gap_boundaries
from moe_outlier_stats import outlier_summary

MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_CLOCK_DRIFT_US = 10000


def decode_sse(response, started_ns, *, clock=time.perf_counter_ns, timeout=300):
    """Timestamp each complete SSE event; chunks are not assumed to be tokens."""
    parts, data = [], []
    usage = None
    first = last = None
    chunks = 0
    finished = done = False
    while True:
        line = response.readline(MAX_SSE_LINE_BYTES + 1)
        now = clock()
        if (now - started_ns) / 1e9 > timeout:
            raise TimeoutError("streaming completion exceeded request timeout")
        if len(line) > MAX_SSE_LINE_BYTES:
            raise ValueError("SSE line exceeds size limit")
        if not line:
            break
        line = line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            if sum(len(part) for part in data) > MAX_SSE_LINE_BYTES:
                raise ValueError("SSE event exceeds size limit")
        elif not line and data:
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                done = True
                break
            record = json.loads(payload)
            if record.get("error"):
                raise ValueError("server returned an SSE error")
            if record.get("usage") is not None:
                usage = record["usage"]
            for choice in record.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ValueError("TPOT measurement requires one completion")
                text = choice.get("text", "")
                if not isinstance(text, str):
                    raise ValueError("invalid completion text")
                if text:
                    first = now if first is None else first
                    last = now
                    chunks += 1
                    parts.append(text)
                finished |= choice.get("finish_reason") is not None
    if not done or not finished or first is None:
        raise ValueError("incomplete SSE response: require text, finish_reason and [DONE]")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("stream must include authoritative completion_tokens; chunk count is not token count")
    latency = (last - started_ns) / 1000
    ttft = (first - started_ns) / 1000
    return dict(
        ttft_us=ttft,
        output_latency_us=latency,
        decode_us=latency - ttft,
        tpot_us=(latency - ttft) / (tokens - 1) if tokens > 1 else None,
        completion_tokens=tokens,
        content_chunks=chunks,
        usage=usage,
        digest=hashlib.sha256("".join(parts).encode()).hexdigest(),
    )


def streaming_completion(opener, port, payload, *, timeout):
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    wall_start = time.time_ns()
    start = time.perf_counter_ns()
    with opener.open(request, timeout=timeout) as response:
        result = decode_sse(response, start, timeout=timeout)
    end = time.perf_counter_ns()
    wall_end = time.time_ns()
    result.update(
        http_wall_us=(end - start) / 1000,
        request_start_epoch_us=wall_start / 1000,
        first_content_epoch_us=wall_start / 1000 + result["ttft_us"],
        last_content_epoch_us=wall_start / 1000 + result["output_latency_us"],
        request_end_epoch_us=wall_start / 1000 + (end - start) / 1000,
        wall_clock_drift_us=abs((wall_end - wall_start) - (end - start)) / 1000,
    )
    return result


def summarize_streams(samples):
    if not samples:
        raise ValueError("missing streaming TPOT samples; non-streaming history cannot recover TTFT")
    for sample in samples:
        for key in ("ttft_us", "output_latency_us", "decode_us", "http_wall_us"):
            value = sample[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid streaming timing: {key}")
    valid = [sample for sample in samples if sample["tpot_us"] is not None]
    return dict(
        requests=len(samples),
        valid_tpot=len(valid),
        undefined_single_token=len(samples) - len(valid),
        one_content_chunk=sum(sample["content_chunks"] == 1 for sample in samples),
        ttft_ms=outlier_summary([sample["ttft_us"] / 1000 for sample in samples]),
        tpot_ms=outlier_summary([sample["tpot_us"] / 1000 for sample in valid]),
    )


def read_host_intervals(path):
    before = path.stat()
    by_pid = collections.defaultdict(list)
    count = 0
    for event in events(path):
        if event.get("ph") == "X" and str(event.get("name", "")).startswith("moe_offload/host_callback|"):
            by_pid[str(event.get("pid"))].append(_interval(event))
            count += 1
            if count > MAX_INTERVALS:
                raise ValueError("too many Host callbacks")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("profile changed while reading")
    if len(by_pid) != 1 or count < 2:
        raise ValueError("miracle requires one unambiguous Host callback process and at least two callbacks")
    return sorted(next(iter(by_pid.values())))


def miracle_sample(sample, callbacks):
    """Subtract union gaps clipped to the SAME request's client decode window.

    Assumes exported Host epoch time aligns with the local client's wall clock,
    and every gap is entirely removable from the critical path. This is a
    counterfactual arithmetic estimate, not a measurement or proven lower bound.
    """
    if sample["tpot_us"] is None:
        raise ValueError("miracle requires at least two output tokens")
    if sample["wall_clock_drift_us"] > MAX_CLOCK_DRIFT_US:
        raise ValueError("wall clock drift prevents profile/client alignment")
    start, end = sample["request_start_epoch_us"], sample["request_end_epoch_us"]
    selected = sorted((a, b) for a, b in callbacks if start <= a and b <= end)
    if len(selected) < 2:
        raise ValueError("no matching callbacks inside request wall-time window; cannot assume clock alignment")
    if any(a < start < b or a < end < b for a, b in callbacks):
        raise ValueError("Host callback crosses request wall-time boundary")
    boundaries, _ = copy_gap_boundaries(selected)
    first, last = sample["first_content_epoch_us"], sample["last_content_epoch_us"]
    overlaps = [max(0, min(selected[right][0], last) - max(selected[left][1], first)) for left, right, _ in boundaries]
    removed = sum(overlaps)
    if removed > sample["decode_us"] + 1:
        raise ValueError("Host gap union exceeds measured decode duration")
    if not any(a < last and b > first for a, b in selected):
        raise ValueError("no Host callback overlaps post-first-token window; alignment is unverified")
    return dict(
        baseline="mte_combined; same profiled streaming request",
        measured_tpot_ms=sample["tpot_us"] / 1000,
        miracle_tpot_ms=max(0, sample["decode_us"] - removed) / (sample["completion_tokens"] - 1) / 1000,
        removed_host_gap_ms=removed / 1000,
        removable_ms_per_token=removed / (sample["completion_tokens"] - 1) / 1000,
        host_callbacks=len(selected),
        contributing_gaps=sum(value > 0 for value in overlaps),
        scope="all adjacent Host callback gaps clipped after first content; no duration/IQR/direction filtering",
        assumption=(
            "Host/client epoch aligned; all selected gaps removable from critical path; remaining work unchanged"
        ),
    )


def print_tpot_comparison(reports, root, labels):
    print("\nTPOT：首个非空文本块到最后文本块的时长/(completion_tokens-1)；MTP块数不作为token数。")
    print("实际TPOT来自流式请求；HTTP完整返回耗时另存http_wall_us；不从旧非流式日志推算。")
    print("one_content_chunk>0表示存在整段输出一次到达的请求，其TPOT无法反映内部逐token节奏。")
    for label in labels:
        report = reports[label]
        for scope, samples in (
            ("no_profile", report.get("request_window", {}).get("stream_samples", [])),
            ("with_profile", report.get("stream_samples", [])),
        ):
            summary = summarize_streams(samples)
            print("TPOT " + json.dumps(dict(case=label, scope=scope, **summary), ensure_ascii=False))
    combined = reports.get("mte_combined")
    if combined is None:
        return
    traces = combined.get("profile_trace_data", [])
    try:
        if len(traces) != 1 or not traces[0].get("usable") or not traces[0].get("json_complete"):
            raise ValueError("miracle requires one complete usable profile")
        # Use exactly the trace validated by the existing profile exporter.
        callbacks = read_host_intervals(Path(traces[0]["file"]))
        results = [miracle_sample(sample, callbacks) for sample in combined["stream_samples"]]
        (root / "tpot_miracle.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        for result in results:
            print("MIRACLE_ESTIMATE " + json.dumps(result, ensure_ascii=False))
        print("miracle仅与同次with_profile TPOT比较；不与20次无采集TPOT混算，不是实际运行的第五个版本。")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"MIRACLE_UNAVAILABLE: {exc}; 四组实测TPOT仍保留，未用其他窗口数据代填。")
