# SPDX-License-Identifier: Apache-2.0
"""Isolated graph handoff benchmark. Run through run_host_sync_stage4.py."""

import argparse
import ctypes
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

from host_sync_device_probe import check_result


def summarize(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Expected finite non-negative timing samples")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def inspect_profile(path, backend, rounds):
    labels = {key: f"host_sync/{backend}/{key}" for key in ("callback_body", "replay_and_wait")}
    counts = dict.fromkeys(labels, 0)
    traces = list(path.rglob("trace_view.json"))
    for trace in traces:
        data = json.loads(trace.read_text(encoding="utf-8"))
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        for event in events:
            if event.get("ph") not in ("X", "B", "b"):
                continue
            for key, label in labels.items():
                if label == event.get("name"):
                    counts[key] += 1
    result = {
        "backend": backend,
        "trace_files": [str(p) for p in traces],
        "mstx_ranges": counts,
        "expected_ranges": rounds,
    }
    if not traces or any(count != rounds for count in counts.values()):
        raise RuntimeError("Profile content validation failed: " + json.dumps(result))
    return result


def register_benchmark_slots(session, slots, callback):
    """Satisfy the old device lease contract; publish only to the last slot."""

    def unexpected_callback():
        raise RuntimeError("Inactive benchmark slot received a device request")

    for slot in range(slots - 1):
        session.register_task(slot, unexpected_callback)
    session.register_task(slots - 1, callback)


def run(args):
    # Keep CLI and statistics checks independent of NPU libraries.
    import torch
    import torch_npu
    import vllm_ascend_host_sync as hs

    torch.npu.set_device(0)
    print(
        f"backend={args.backend}; device={torch.npu.get_device_name(0)}; torch_npu={torch_npu.__version__}", flush=True
    )
    stream = torch.npu.Stream()
    session = None
    state = None
    calls = 0
    timing_active = False
    profile_active = False
    callback_times = []

    def callback(_unused=None):
        nonlocal calls
        mark = torch_npu.npu.mstx.range_start(f"host_sync/{args.backend}/callback_body") if profile_active else None
        started = time.perf_counter_ns() if timing_active else 0
        calls += 1
        if timing_active:
            callback_times.append((time.perf_counter_ns() - started) / 1000)
        if mark is not None:
            torch_npu.npu.mstx.range_end(mark)

    if args.backend == "mte":
        library = ctypes.CDLL(str(args.library.resolve()))
        publish = library.host_sync_publish_launch
        publish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        publish.restype = None
        wait = library.host_sync_wait_launch
        wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        wait.restype = None
        state = torch.zeros(4, dtype=torch.int64, device="npu:0")
        torch.npu.synchronize()
        session = hs.Session(slots=args.slots, backend="acl", device_id=0)
        session.configure_polling(args.idle_mode, args.worker_timing)
        info = session.info()
        if info["protocol_version"] != 1 or info["slot_stride"] != 1024 or info["done_offset"] != 512:
            raise RuntimeError("Unsupported control-memory layout")
        active_slot = args.slots - 1
        register_benchmark_slots(session, args.slots, callback)
        if args.scan_mode == "active":
            session.configure_scan_slots([active_slot])
        session.start()
        alias = session.begin_device_access()
        alias += active_slot * info["slot_stride"]
        stream_pointer = int(stream.npu_stream)

        def submit():
            publish(stream_pointer, alias + info["request_offset"], state.data_ptr())
            wait(stream_pointer, alias + info["done_offset"], state.data_ptr(), args.max_polls)
    else:
        torch_npu.npu._subscribe_report(stream)

        def submit():
            torch_npu.npu._launch_host_func(stream, callback, None)

    def validate(expected):
        if calls != expected:
            raise RuntimeError(f"Callback mismatch: expected={expected}, actual={calls}")
        if state is not None:
            check_result(state.cpu().tolist(), expected, 1)

    # Warm up submission before capture, then verify capture itself did not run.
    submit()
    stream.synchronize()
    validate(1)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        submit()
    stream.synchronize()
    validate(1)
    expected = 1
    for _ in range(args.warmup):
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        expected += 1
        validate(expected)

    profiler = None
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        profiler = torch_npu.profiler.profile(
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=args.rounds, repeat=1),
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                mstx=True,
            ),
        )
        profiler.start()
        # Advance the profiler's preparation step before the measured workload.
        # The graph itself was already warmed up outside the profiler.
        profiler.step()
        profile_active = True

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record(stream)
    end.record(stream)
    end.synchronize()
    event_times, wall_times = [], []
    timing_active = profiler is None
    for sample in range(args.rounds):
        mark = torch_npu.npu.mstx.range_start(f"host_sync/{args.backend}/replay_and_wait") if profile_active else None
        start.record(stream)
        began = time.perf_counter_ns()
        with torch.npu.stream(stream):
            graph.replay()
        end.record(stream)
        end.synchronize()
        elapsed = (time.perf_counter_ns() - began) / 1000
        if mark is not None:
            torch_npu.npu.mstx.range_end(mark)
        # Event query and correctness D2H are outside the measured interval.
        event_times.append(start.elapsed_time(end) * 1000)
        wall_times.append(elapsed)
        expected += 1
        validate(expected)
        if profiler is not None:
            if sample == args.rounds - 1:
                print("PROFILE_WORKLOAD_PASS: all callbacks/results verified; starting scheduled export.", flush=True)
            profiler.step()
    timing_active = False
    profile_active = False
    if profiler is not None:
        profiler.stop()

    stream.synchronize()
    graph.reset()
    worker_stats = None
    if session is not None:
        session.end_device_access(stream_synchronized=True)
        session.stop()
        worker_stats = session.polling_stats()
        session.close()
    else:
        torch_npu.npu._unsubscribe_report(stream)

    if profiler is None:
        if len(callback_times) != args.rounds:
            raise RuntimeError("Callback sample count mismatch")
        print(
            "BENCH_RESULT "
            + json.dumps(
                {
                    "backend": args.backend,
                    "idle_mode": args.idle_mode,
                    "slots": args.slots,
                    "scan_mode": args.scan_mode,
                    "worker_timing": args.worker_timing,
                    "worker_stats": summarize_worker(worker_stats, 1 + args.warmup, args.rounds),
                    "rounds": args.rounds,
                    "warmup_replays": args.warmup,
                    "callback_count": calls,
                    "unit": "us",
                    "profiler_enabled": False,
                    "event_interval": summarize(event_times),
                    "replay_sync_host": summarize(wall_times),
                    "callback_body": summarize(callback_times),
                }
            ),
            flush=True,
        )
    else:
        print("PROFILE_RESULT " + json.dumps(inspect_profile(args.profile_dir, args.backend, args.rounds)), flush=True)


def summarize_worker(raw, warmup_generations, rounds):
    if raw is None or not raw["timing_enabled"]:
        return None
    if raw["task_samples_dropped"]:
        raise RuntimeError("Worker task samples overflowed; no valid measured statistics")
    selected = [row for row in raw["task_samples"] if row[1] > warmup_generations]
    if len(selected) != rounds:
        raise RuntimeError(f"Worker timing count mismatch: {len(selected)} != {rounds}")
    task = {
        name.removesuffix("_ns") + "_us": summarize([row[index] / 1000 for row in selected])
        for index, name in enumerate(raw["task_columns"])
        if index >= 2
    }
    idle = (
        {
            name.removesuffix("_ns") + "_us": summarize([row[index] / 1000 for row in raw["idle_samples"]])
            for index, name in enumerate(raw["idle_columns"])
        }
        if raw["idle_samples"]
        else {}
    )
    return dict(
        unit="us",
        task=task,
        idle=idle,
        idle_sample_stride=raw["idle_sample_stride"],
        idle_samples_dropped=raw["idle_samples_dropped"],
        idle_scope="session lifetime including startup, capture, warmup and validation; not request wake latency",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("callback", "mte"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-polls", type=int, default=100000)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--idle-mode", choices=("yield", "spin"), default="yield")
    parser.add_argument("--worker-timing", action="store_true")
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--scan-mode", choices=("all", "active"), default="all")
    args = parser.parse_args()
    if not 1 <= args.slots <= 4096:
        parser.error("slots: 1..4096")
    if not 1 <= args.rounds <= 10000 or not 1 <= args.warmup <= 1000 or not 1 <= args.max_polls <= 10000000:
        parser.error("rounds: 1..10000; warmup: 1..1000; max-polls: 1..10000000")
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        print("FAIL: isolated benchmark; no valid comparison for this run.", flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
