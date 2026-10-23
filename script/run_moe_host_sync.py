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


def configured_source(source, backend, library, diagnose_load=False, quant_copy="storage"):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number)
    parser.add_argument("--read-log", type=Path, help="Print existing log diagnostics only; no build or model startup")
    parser.add_argument("--backend", choices=("both", "callback", "mte"), default="both")
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
        reports[backend] = inference(
            run_args,
            configured_source(source, backend, library, args.diagnose_load_stages, args.quant_copy),
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
    print("REAL MOE RUN COMPLETE. Limited prompt coverage; diagnostic timings include profiler overhead.", flush=True)
    print(f"Profiles: {root / 'profile'}; return summary.txt text only.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
