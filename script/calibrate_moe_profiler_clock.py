# SPDX-License-Identifier: Apache-2.0
"""Bounded eager/graph clock calibration, independent of MoE weights and handshakes."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import events
from inspect_moe_profiler_clocks import discover_roots, metadata_files, parse_metadata

SAMPLES = 61
INTERVAL_SECONDS = 0.2
CASE_TIMEOUT_SECONDS = 240
STARTUP_TIMEOUT_SECONDS = 60
PROGRESS_SECONDS = 30
WARMUP_REPLAYS = 10
TENSOR_ELEMENTS = 256
MARKER_PREFIX = "moe_clock_calibration|sample="


def affine_clock_bounds(measurements):
    """Bound D = scale * H + offset using submission/completion intervals.

    Each row is (host_submit, host_sync, device_start, device_end), in us.
    Solve every lower-offset <= upper-offset constraint. Unlike fitting midpoint
    timestamps, this retains Host submission and synchronization uncertainty.
    """
    if len(measurements) < 2:
        raise ValueError("need at least two clock measurements")
    for submit, sync, start, end in measurements:
        if any(not value.is_finite() for value in (submit, sync, start, end)) or submit > sync or start > end:
            raise ValueError("invalid affine clock measurement")
    lower, upper, impossible = Decimal(0), None, False
    for _, sync, _, end in measurements:
        for submit, _, start, _ in measurements:
            numerator, denominator = end - start, sync - submit
            if denominator > 0:
                lower = max(lower, numerator / denominator)
            elif denominator < 0:
                bound = numerator / denominator
                upper = bound if upper is None else min(upper, bound)
            elif numerator > 0:
                impossible = True
    possible = not impossible and upper is not None and upper > 0 and lower <= upper
    return {
        "n": len(measurements),
        "positive_affine_possible": possible,
        "rate_error_bounds_us_per_second": [
            str((lower - 1) * 10**6),
            None if upper is None else str((upper - 1) * 10**6),
        ],
        "basis": "all interval constraints; D=scale*H+offset; negative rate means device time advances slower than RAW",
        "scope": "conditional on one affine clock across capture; no original timestamps changed",
    }


def bracket_stats(bounds, samples):
    """Bounds on exported time minus a directly bracketed RAW timestamp, in us."""
    if len(bounds) != len(samples) or len(bounds) < 2:
        raise ValueError("need one bound per sample and at least two samples")
    if any(not lo.is_finite() or not hi.is_finite() or lo > hi for lo, hi in bounds):
        raise ValueError("invalid timestamp bracket")
    elapsed = Decimal(samples[-1]["start_before_ns"] - samples[0]["start_before_ns"]) / 10**9
    if elapsed <= 0:
        raise ValueError("non-increasing host clock")
    low = (bounds[-1][0] - bounds[0][1]) / elapsed
    high = (bounds[-1][1] - bounds[0][0]) / elapsed
    return {
        "n": len(bounds),
        "outside_bracket": sum(not lo <= 0 <= hi for lo, hi in bounds),
        "first_offset_bounds_us": [str(x) for x in bounds[0]],
        "last_offset_bounds_us": [str(x) for x in bounds[-1]],
        "drift_bounds_us_per_second": [str(low), str(high)],
        "max_bracket_width_us": str(max(hi - lo for lo, hi in bounds)),
        "max_distance_outside_us": str(max(max(lo, -hi, Decimal(0)) for lo, hi in bounds)),
    }


def analyze_trace(root, samples):
    if len(samples) < 2 or [row["sample"] for row in samples] != list(range(len(samples))):
        raise ValueError("missing or unordered Host samples")
    keys = ("start_before_ns", "start_after_ns", "submit_before_ns", "sync_after_ns", "end_before_ns", "end_after_ns")
    previous = -1
    for row in samples:
        times = [row[key] for key in keys]
        if any(not isinstance(value, int) for value in times) or times != sorted(times) or times[0] < previous:
            raise ValueError("invalid RAW measurement order")
        previous = times[-1]
    host_files = [p for p in metadata_files(root) if p.parent.name == "host" and p.name == "start_info"]
    if len(host_files) != 1:
        raise ValueError("need exactly one CANN host/start_info")
    info = parse_metadata(host_files[0])
    host_offset_ns = Decimal(str(info["collectionTimeBegin"])) * 1000 - Decimal(str(info["clockMonotonicRaw"]))
    markers, processes, candidates = {}, {}, []
    for event in events(root / "ASCEND_PROFILER_OUTPUT/trace_view.json"):
        name, pid = str(event.get("name", "")), str(event.get("pid"))
        if event.get("ph") == "M" and name == "process_name":
            processes[pid] = str(event.get("args", {}).get("name", ""))
        if event.get("ph") != "X":
            continue
        if name.startswith(MARKER_PREFIX):
            if name in markers:
                raise ValueError("duplicate calibration marker")
            markers[name] = event
        elif "add" in name.lower():
            candidates.append(event)
    if len(markers) != len(samples):
        raise ValueError(f"MSTX markers={len(markers)}; expected={len(samples)}")
    start_bounds, end_bounds = [], []
    for row in samples:
        marker = markers[f"{MARKER_PREFIX}{row['sample']}"]
        start = Decimal(str(marker["ts"])) * 1000 - host_offset_ns
        end = start + Decimal(str(marker["dur"])) * 1000
        start_bounds.append(((start - row["start_after_ns"]) / 1000, (start - row["start_before_ns"]) / 1000))
        end_bounds.append(((end - row["end_after_ns"]) / 1000, (end - row["end_before_ns"]) / 1000))
    result = {
        "host_epoch_offset_ns": str(host_offset_ns),
        "mstx_start_vs_raw": bracket_stats(start_bounds, samples),
        "mstx_end_vs_raw": bracket_stats(end_bounds, samples),
    }
    device = [e for e in candidates if processes.get(str(e.get("pid")), "").strip().lower() == "ascend hardware"]
    identities = {(str(e.get("pid")), str(e.get("tid")), e["name"]) for e in device}
    result["device_add_candidates"] = {"count": len(device), "identities": sorted(identities)}
    if len(device) == len(samples) and len(identities) == 1:
        # Exactly one known Add per synchronous replay: sequence mapping, never nearest-time matching.
        device.sort(key=lambda e: Decimal(str(e["ts"])))
        bounds, measurements = [], []
        origin = samples[0]["submit_before_ns"]
        for event, row in zip(device, samples):
            start = Decimal(str(event["ts"])) * 1000 - host_offset_ns
            end = start + Decimal(str(event["dur"])) * 1000
            # Device execution must follow submission and finish before synchronize returns.
            bounds.append(((end - row["sync_after_ns"]) / 1000, (start - row["submit_before_ns"]) / 1000))
            measurements.append(
                (
                    Decimal(row["submit_before_ns"] - origin) / 1000,
                    Decimal(row["sync_after_ns"] - origin) / 1000,
                    (start - origin) / 1000,
                    (end - origin) / 1000,
                )
            )
        result["device_offset_necessary_bounds_us"] = {
            "first": [str(x) for x in bounds[0]],
            "last": [str(x) for x in bounds[-1]],
            "zero_offset_violations": sum(not lo <= 0 <= hi for lo, hi in bounds),
            "constant_offset_intersection": [str(max(x[0] for x in bounds)), str(min(x[1] for x in bounds))],
            "constant_offset_possible": max(x[0] for x in bounds) <= min(x[1] for x in bounds),
            "inverted_bounds": sum(lo > hi for lo, hi in bounds),
            "basis": "exported minus measured RAW using the Host wall offset; one Add per synchronous replay",
        }
        result["device_affine_clock"] = affine_clock_bounds(measurements)
    else:
        result["device_mapping"] = "unproven; kernel count/name/stream did not identify exactly one Add per sample"
    # Event elapsed time gives an additional relative device clock, independently of JSON timestamps.
    if all("event_elapsed_us" in row for row in samples):
        first = samples[0]
        elapsed_bounds, measurements = [], []
        for row in samples:
            elapsed = Decimal(str(row["event_elapsed_us"]))
            measurements.append(
                (
                    Decimal(row["submit_before_ns"] - first["submit_before_ns"]) / 1000,
                    Decimal(row["sync_after_ns"] - first["submit_before_ns"]) / 1000,
                    elapsed,
                    elapsed,
                )
            )
            elapsed_bounds.append(
                (
                    (
                        Decimal(row["submit_before_ns"] - first["sync_after_ns"]) / 1000
                        - Decimal(str(row["event_elapsed_us"]))
                    ),
                    (
                        Decimal(row["sync_after_ns"] - first["submit_before_ns"]) / 1000
                        - Decimal(str(row["event_elapsed_us"]))
                    ),
                )
            )
        result["event_elapsed_vs_host_bounds_us"] = {
            "first": [str(x) for x in elapsed_bounds[0]],
            "last": [str(x) for x in elapsed_bounds[-1]],
            "violations": sum(not lo <= 0 <= hi for lo, hi in elapsed_bounds),
        }
        result["event_affine_clock"] = affine_clock_bounds(measurements)
    return result


def run_child(args):
    # Device libraries are imported only in the isolated Linux child.
    import torch
    import torch_npu

    torch.npu.set_device(0)
    ready, go = threading.Event(), threading.Event()
    records, errors = [], []

    def worker():
        try:
            torch.npu.set_device(0)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                tensor = torch.zeros(TENSOR_ELEMENTS, device="npu:0", dtype=torch.float32)
                for _ in range(WARMUP_REPLAYS):
                    tensor.add_(1.0)
            stream.synchronize()
            graph = None
            if args.child_mode == "graph":
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    tensor.add_(1.0)
                stream.synchronize()
                for _ in range(WARMUP_REPLAYS):
                    with torch.npu.stream(stream):
                        graph.replay()
                    stream.synchronize()
            baseline = tensor.cpu().clone()
            stamps = [torch.npu.Event(enable_timing=True) for _ in range(args.samples)]
            ready.set()
            if not go.wait(STARTUP_TIMEOUT_SECONDS):
                raise TimeoutError("profiler start did not release worker")

            def raw():
                return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

            for index in range(args.samples):
                row = {"sample": index, "start_before_ns": raw()}
                mark = torch_npu.npu.mstx.range_start(f"{MARKER_PREFIX}{index}")
                row["start_after_ns"] = raw()
                with torch.npu.stream(stream):
                    row["submit_before_ns"] = raw()
                    if graph is None:
                        tensor.add_(1.0)
                    else:
                        graph.replay()
                    stamps[index].record(stream)
                stream.synchronize()
                row["sync_after_ns"] = raw()
                row["end_before_ns"] = raw()
                torch_npu.npu.mstx.range_end(mark)
                row["end_after_ns"] = raw()
                records.append(row)
                if index + 1 < args.samples:
                    time.sleep(args.interval)
            for index, row in enumerate(records):
                row["event_elapsed_us"] = 0.0 if index == 0 else stamps[0].elapsed_time(stamps[index]) * 1000
            if not torch.equal(tensor.cpu(), baseline + args.samples):
                raise RuntimeError("calibration result mismatch")
        except BaseException as error:
            errors.append(error)
            ready.set()

    thread = threading.Thread(target=worker, name="moe-clock-calibration", daemon=True)
    thread.start()
    if not ready.wait(STARTUP_TIMEOUT_SECONDS):
        raise TimeoutError("worker initialization timed out")
    if errors:
        raise errors[0]
    profiler = torch_npu.profiler.profile(
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output_dir / "profile")),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, mstx=True
        ),
    )
    profiler.start()
    # Finish profiler preparation before releasing the already warmed-up worker.
    profiler.step()
    go.set()
    thread.join(STARTUP_TIMEOUT_SECONDS + args.samples * args.interval)
    if thread.is_alive():
        raise TimeoutError("worker computation timed out")
    # Preserve independent measurements even when export fails.
    (args.output_dir / "host_brackets.json").write_text(json.dumps(records), encoding="utf-8")
    # Complete the single active step and export before stopping in NONE state.
    profiler.step()
    profiler.stop()
    if errors:
        raise errors[0]
    print(f"WORKLOAD_PASS mode={args.child_mode}; device={torch.npu.get_device_name(0)}; samples={len(records)}")


def saved_case_result(case, expected_samples=None):
    roots = discover_roots([case / "profile"])
    if len(roots) != 1:
        raise ValueError("expected exactly one exported capture")
    samples = json.loads((case / "host_brackets.json").read_text(encoding="utf-8"))
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f"Host samples={len(samples)}; expected={expected_samples}")
    return roots[0], analyze_trace(roots[0], samples)


def run_parent(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "summary.txt").open("w", encoding="utf-8") as report:

        def emit(value):
            print(value, flush=True)
            report.write(value + "\n")
            report.flush()

        emit("时钟校准：仅小张量 Add，专用 Host 线程；eager 与 graph 顺序运行，无模型、无MoE握手。")
        emit("不修改原实验。Host时间以CLOCK_MONOTONIC_RAW夹住MSTX调用；设备以提交前和同步返回后约束。")
        emit(
            f"SUMMARY FILE: {args.output_dir / 'summary.txt'}; physical_npu={args.npu}; samples_per_case={args.samples}"
        )
        emit("偏移范围符号=导出坐标减RAW坐标（扣除Host固定纪元偏移）；不对原trace做校正。")
        emit("outside_bracket按零容差报告；亚微秒残差需结合导出量化判断，不自动认定时钟故障。")
        emit("全样本仿射约束：负速率表示设备时间相对RAW变慢；这是速率范围，不是已验证的修正值。")
        if args.reference_run:
            emit(f"REFERENCE_RUN={args.reference_run}; 只读取前次校准记录，不修改旧文件。")
            for mode in ("eager", "graph"):
                try:
                    _, result = saved_case_result(args.reference_run / mode)
                except Exception as error:
                    emit(f"REFERENCE_FAILED mode={mode}; {type(error).__name__}: {error}")
                    return 1
                emit("REFERENCE_CALIBRATION " + json.dumps({"mode": mode, **result}, ensure_ascii=False))
        for mode in ("eager", "graph"):
            case = args.output_dir / mode
            case.mkdir()
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.npu))
            command = [
                sys.executable,
                "-u",
                "-B",
                str(Path(__file__).resolve()),
                "--child-mode",
                mode,
                "--npu",
                str(args.npu),
                "--output-dir",
                str(case),
                "--samples",
                str(args.samples),
                "--interval",
                str(args.interval),
            ]
            with (case / "run.log").open("w", encoding="utf-8") as logfile:
                process = subprocess.Popen(
                    command, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
                )
                start = time.monotonic()
                while process.poll() is None:
                    try:
                        process.wait(timeout=PROGRESS_SECONDS)
                    except subprocess.TimeoutExpired:
                        emit(f"WAIT mode={mode}; seconds={time.monotonic() - start:.0f}; log={case / 'run.log'}")
                    if process.poll() is None and time.monotonic() - start >= CASE_TIMEOUT_SECONDS:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        emit(f"TIMEOUT mode={mode}; stopped this child process group")
                        break
            emit(f"CASE={mode}; exit={process.returncode}")
            lines = (case / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
            hints = [line for line in lines if "error" in line.lower() or "warning" in line.lower()]
            emit("EXPORT_LOG_HINTS " + json.dumps({"count": len(hints), "examples": hints[:3]}, ensure_ascii=False))
            if process.returncode:
                emit("\n".join(lines[-20:]))
                emit("停止后续组；本轮不形成有效对照。")
                return 1
            try:
                root, result = saved_case_result(case, args.samples)
            except Exception as error:
                emit(f"ANALYSIS_FAILED mode={mode}; {type(error).__name__}: {error}; 原记录保留。")
                return 1
            emit("CALIBRATION " + json.dumps({"mode": mode, **result}, ensure_ascii=False))
            emit(f"TRACE={root / 'ASCEND_PROFILER_OUTPUT/trace_view.json'}")
        emit("完成；这是当前环境的独立校准，不自动证明旧采集的根因或MoE协议正确。")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-run", type=Path, help="also analyze an existing calibration run into the new summary"
    )
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--interval", type=float, default=INTERVAL_SECONDS)
    parser.add_argument("--child-mode", choices=("eager", "graph"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if sys.platform != "linux":
        parser.error("NPU calibration requires Linux; analysis functions can be tested on CPU")
    if args.npu < 0 or not 10 <= args.samples <= 100 or not 0.1 <= args.interval <= 0.3:
        parser.error("require npu>=0, 10<=samples<=100, 0.1<=interval<=0.3")
    if args.child_mode:
        run_child(args)
    else:
        raise SystemExit(run_parent(args))


if __name__ == "__main__":
    main()
