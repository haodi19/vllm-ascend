# SPDX-License-Identifier: Apache-2.0
"""Run real MoE inference with original callback or optional MTE dispatch."""

import argparse
import contextlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from profile_storage import require_space
from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, run_step
from run_host_sync_stage2 import target_cmake_args
from run_host_sync_stage3 import select_build_directory
from run_verified_cpp_tests import SummaryTee, inference, launch_spec


def configured_source(source, backend, library, diagnose_load=False, quant_copy="storage", prefetch_callback=False):
    start = re.search(r"(?m)^vllm\s+serve\s+", source)
    if start is None:
        raise ValueError("Missing vllm serve command")
    words = [word for word in shlex.split(source[start.start() :] + "\n", comments=True) if word.strip()]
    index = words.index("--additional-config") + 1
    config = json.loads(words[index])
    offload = config["expert_offload_config"]
    offload["host_sync_backend"] = backend
    offload["host_sync_library"] = str(library) if backend == "mte" else ""
    offload["host_sync_diagnose_load"] = bool(diagnose_load and backend == "mte")
    offload["host_sync_quant_copy"] = quant_copy if backend == "mte" else "storage"
    offload["host_sync_prefetch_callback"] = bool(prefetch_callback and backend == "mte")
    words[index] = json.dumps(config)
    return source[: start.start()] + shlex.join(words) + "\n"


def failure_excerpt(text):
    """Retain first-cause diagnostics even after HTTP shutdown floods the tail."""
    lines = text.splitlines()
    selected = set(range(max(0, len(lines) - 70), len(lines)))
    inside = False
    for index, line in enumerate(lines):
        if "[MOE-MTE DIAGNOSTIC BEGIN]" in line:
            inside = True
        if inside or "[MOE-MTE" in line:
            selected.add(index)
        if "[MOE-MTE DIAGNOSTIC END]" in line:
            inside = False
    # Startup failures precede final statistics and API-server tracebacks.
    # Preserve the first engine/native traceback windows, not only shutdown.
    roots = [
        index
        for index, line in enumerate(lines)
        if ("Traceback (most recent call last)" in line and "APIServer" not in line) or "MTE quant storage copy" in line
    ]
    for index in roots[:6]:
        selected.update(range(max(0, index - 8), min(len(lines), index + 120)))
    return "\n".join(lines[index] for index in sorted(selected))


def compare_weight_timings(reports):
    """Compare like-for-like transfer stages, separately from HTTP latency."""
    left, right = (reports[key]["weight_timing"] for key in ("callback", "mte"))
    if left["digests"] != right["digests"] or left["usage"] != right["usage"]:
        raise RuntimeError("CPU-timed weight comparison produced different outputs/usage")
    groups = [{(row["kind"], row["token_rows"]): row for row in report["groups"]} for report in (left, right)]
    print("\n=== WEIGHT STAGE COMPARISON: CPU timers only; profiler/MSTX OFF ===", flush=True)
    prefill = [report.get("prefill_load") for report in (left, right)]
    if all(row is not None for row in prefill):
        counts = [row["count"] for row in prefill]
        print(f"PREFILL full-expert loads: callback={counts[0]}; mte={counts[1]}", flush=True)
        if all(counts):
            means = [row["sum_us"] / row["count"] / 1000 for row in prefill]
            print(
                f"PREFILL ms/layer: callback={means[0]:.3f}; mte={means[1]:.3f}; "
                f"delta={means[1] - means[0]:+.3f}; count match={counts[0] == counts[1]}",
                flush=True,
            )
            if left["usage"]:
                totals = [row["sum_us"] / len(left["usage"]) / 1000 for row in prefill]
                print(
                    f"PREFILL accumulated ms/request: callback={totals[0]:.3f}; mte={totals[1]:.3f}; "
                    f"delta={totals[1] - totals[0]:+.3f} (separate from Decode below)",
                    flush=True,
                )
    print("Units: us per callback; synchronize is residual waiting, not the full DMA duration.", flush=True)
    for key in sorted(groups[0].keys() | groups[1].keys()):
        if any(key not in group for group in groups):
            print(f"kind={key[0]} rows={key[1]}: missing in one backend; not comparable", flush=True)
            continue
        before, after = (group[key] for group in groups)
        loads = [row["count"] * row["loads"]["mean"] for row in (before, after)]
        matched = before["count"] == after["count"] and abs(loads[0] - loads[1]) < 1e-6
        print(
            f"kind={key[0]} rows={key[1]} callbacks={before['count']}/{after['count']} "
            f"total_loaded_experts={loads[0]:.0f}/{loads[1]:.0f}; count/load match={matched}",
            flush=True,
        )
        print(f"{'metric':26} {'callback_us':>12} {'mte_us':>12} {'delta_us':>12} {'change':>10}", flush=True)
        for metric in ("load_prepare", "copy_submit", "scale_refresh", "load_submit", "synchronize", "load_and_wait"):
            values = [
                sum(row["metrics"][name]["mean"] for name in ("load_submit", "synchronize"))
                if metric == "load_and_wait"
                else row["metrics"][metric]["mean"]
                for row in (before, after)
            ]
            delta = values[1] - values[0]
            change = f"{100 * delta / values[0]:+.1f}%" if values[0] else "n/a"
            print(f"{metric:26} {values[0]:12.3f} {values[1]:12.3f} {delta:12.3f} {change:>10}", flush=True)
        if not matched:
            print("WARNING: workload counts differ; mean deltas alone do not establish a speedup.", flush=True)
    print("load_and_wait = load_submit + synchronize; do not add load_prepare/copy_submit again.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number)
    parser.add_argument("--read-log", type=Path, help="Print existing log diagnostics only; no build or model startup")
    parser.add_argument("--backend", choices=("both", "callback", "mte"), default="both")
    parser.add_argument(
        "--prefetch-handoff",
        choices=("mte", "callback"),
        default="mte",
        help="Diagnostic MTE ablation: change only prefetch handoff; keep demand and copy configuration",
    )
    parser.add_argument(
        "--quant-copy",
        choices=("storage", "tensor", "prepacked"),
        default="storage",
        help="MTE: storage packs per load; tensor uses original copy_; prepacked uses loader-prepared scale layout",
    )
    parser.add_argument(
        "--diagnose-load-stages",
        action="store_true",
        help="Serialize the first eight MTE load batches for diagnosis, not performance",
    )
    parser.add_argument("--test-script", type=Path, default=REPOSITORY_ROOT / "script/test_graph2.sh")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument(
        "--timing-requests",
        type=int,
        default=3,
        help="Weight-stage timing requests before profiling, with MSTX disabled; 0 skips this window",
    )
    parser.add_argument(
        "--latency-requests",
        type=int,
        default=0,
        help="Additional requests before profiling, with timer collection inactive",
    )
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--profile-timeout", type=int, default=120)
    parser.add_argument("--port", type=int)
    parser.add_argument("--timing-port", type=int, default=18155)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.read_log is not None:
        print(f"READ-ONLY SERVICE LOG: {args.read_log}", flush=True)
        print(failure_excerpt(args.read_log.read_text(encoding="utf-8", errors="replace")), flush=True)
        return 0
    if args.npu is None:
        parser.error("--npu is required for inference")
    if sys.platform != "linux":
        parser.error("Run inside the Linux NPU container")
    if min(args.requests, args.warmup_requests, args.startup_timeout, args.request_timeout, args.profile_timeout) < 1:
        parser.error("Counts/timeouts must be positive")
    if min(args.latency_requests, args.timing_requests) < 0:
        parser.error("--latency-requests and --timing-requests must be non-negative")
    root = (args.output_dir or REPOSITORY_ROOT / "bench_results" / f"moe_host_sync_{uuid4().hex[:8]}").resolve()
    require_space(root)
    require_space(REPOSITORY_ROOT, minimum_gib=2)
    root.mkdir(parents=True, exist_ok=False)
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as file,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, file)),
    ):
        print(f"SUMMARY FILE: {root / 'summary.txt'}", flush=True)
        try:
            return run(args, root)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            for log in root.glob("*/inference_python.log"):
                print(
                    f"SERVICE LOG TAIL: {log}\n" + failure_excerpt(log.read_text(encoding="utf-8", errors="replace")),
                    flush=True,
                )
            return 1


def run(args, root):
    backends = ("callback", "mte") if args.backend == "both" else (args.backend,)
    build = select_build_directory(None, False)
    library = build / "lib/libhost_sync_device.so"
    if "mte" in backends:
        commands = [
            (
                "build Host extension",
                [sys.executable, "-m", "pip", "install", "./csrc/host_sync", "--no-build-isolation", "--no-deps"],
            ),
            (
                "configure MTE kernels",
                [
                    "cmake",
                    "-S",
                    str(REPOSITORY_ROOT / "csrc/host_sync_device"),
                    "-B",
                    str(build),
                    "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest",
                    *target_cmake_args("ascend950pr_9579"),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
            ),
            ("build MTE kernels", ["cmake", "--build", str(build), "-j", "2"]),
        ]
        for label, command in commands:
            if not run_step(label, command, 600):
                return 1
    source = args.test_script.read_text(encoding="utf-8")
    _, command, _, _ = launch_spec(source, "python", args.npu, args.port, args.timing_port)
    port = int(command[command.index("--port") + 1])
    if not 1024 <= port <= 65535 or not 1024 <= args.timing_port <= 65535 or port == args.timing_port:
        raise ValueError("Use distinct serve/timing ports in 1024..65535")
    reports = {}
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    print(
        "REAL MOE: LRC=python in both groups; model/MTP/prefetch configuration inherited from test_graph2.sh",
        flush=True,
    )
    print(
        "MTE uses graph polling dispatch and SyncLaunchStream H2D; callback keeps its original load stream. "
        "Eager callback invocation is unchanged. Timers + profiler enabled.",
        flush=True,
    )
    for backend in backends:
        directory = root / backend
        directory.mkdir()
        run_args = SimpleNamespace(
            output_dir=directory,
            profile_run_dir=root / "profile" / backend,
            npu=args.npu,
            port=port,
            timing_port=args.timing_port,
            requests=args.requests,
            latency_requests=args.latency_requests,
            timing_requests=args.timing_requests,
            warmup_requests=args.warmup_requests,
            max_tokens=5,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            profile_timeout=args.profile_timeout,
            strict_profile=False,
            mstx=True,
        )
        print(f"\n=== REAL MOE HOST BACKEND: {backend} ===", flush=True)
        if args.diagnose_load_stages:
            print("DIAGNOSTIC LOAD BARRIERS ENABLED: timings are not a performance comparison", flush=True)
        print(f"QUANT COPY: {'original tensor' if backend == 'callback' else args.quant_copy}", flush=True)
        print(
            f"PREFETCH HANDOFF: {args.prefetch_handoff if backend == 'mte' else 'callback'}; DEMAND HANDOFF: {backend}",
            flush=True,
        )
        reports[backend] = inference(
            run_args,
            configured_source(
                source,
                backend,
                library,
                args.diagnose_load_stages,
                args.quant_copy,
                args.prefetch_handoff == "callback",
            ),
            "python",
            env,
        )
        if backend == "mte":
            log = (directory / "inference_python.log").read_text(encoding="utf-8", errors="replace")
            if args.diagnose_load_stages:
                print("\n".join(line for line in log.splitlines() if "[MOE-MTE LOAD]" in line), flush=True)
            if "[MOE-MTE] first real offload callback completed" not in log:
                raise RuntimeError("No evidence that real offload callbacks executed through MTE")
        print(
            "MOE_RESULT "
            + json.dumps(
                {
                    "host_backend": backend,
                    "request_mean_us": reports[backend]["request_us"],
                    "digests": reports[backend]["digests"],
                    "usage": reports[backend]["usage"],
                    "profile_status": reports[backend]["profile_status"],
                }
            ),
            flush=True,
        )
    if len(reports) == 2:
        match = reports["callback"]["digests"] == reports["mte"]["digests"]
        usage_match = reports["callback"]["usage"] == reports["mte"]["usage"]
        print(f"GENERATED TEXT MATCH: {match}; USAGE MATCH: {usage_match}", flush=True)
        print(
            "Diagnostic request mean us: " + json.dumps({key: value["request_us"] for key, value in reports.items()}),
            flush=True,
        )
        if not match or not usage_match:
            raise RuntimeError("Generated output differs; inspect before claiming correctness or speedup")
        if args.timing_requests:
            compare_weight_timings(reports)
        if args.latency_requests:
            left, right = (reports[key]["plain_latency"] for key in ("callback", "mte"))
            plain_match = left["digests"] == right["digests"] and left["usage"] == right["usage"]
            print(f"UNPROFILED OUTPUT/USAGE MATCH: {plain_match}", flush=True)
            before, after = left["request_ms"]["mean"], right["request_ms"]["mean"]
            print(
                f"UNPROFILED REQUEST MEAN ms: callback={before:.3f}; mte={after:.3f}; "
                f"change={(after / before - 1) * 100:.2f}% (negative=faster)",
                flush=True,
            )
            if not plain_match:
                raise RuntimeError("Unprofiled request outputs differ")
    print("REAL MOE RUN COMPLETE. Limited prompt coverage; diagnostic timings include profiler overhead.", flush=True)
    print(f"Profiles: {root / 'profile'}; return summary.txt text only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
