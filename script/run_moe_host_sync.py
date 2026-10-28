# SPDX-License-Identifier: Apache-2.0
"""Run real MoE inference with original callback or optional MTE dispatch."""

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, run_step
from run_host_sync_stage2 import target_cmake_args
from run_host_sync_stage3 import select_build_directory
from run_verified_cpp_tests import SummaryTee, inference, launch_spec


def configured_source(
    source,
    backend,
    library,
    diagnose_load=False,
    scan_mode="all",
    idle_probe="locked",
    idle_mode="yield",
    diagnose_native_wait=False,
    diagnose_split_scale=False,
):
    start = re.search(r"(?m)^vllm\s+serve\s+", source)
    if start is None:
        raise ValueError("Missing vllm serve command")
    words = [word for word in shlex.split(source[start.start() :] + "\n", comments=True) if word.strip()]
    index = words.index("--additional-config") + 1
    config = json.loads(words[index])
    offload = config["expert_offload_config"]
    offload["host_sync_backend"] = backend
    offload["host_sync_library"] = str(library) if backend == "mte" else ""
    offload["host_sync_scan_mode"] = scan_mode if backend == "mte" else "all"
    offload["host_sync_idle_probe"] = idle_probe if backend == "mte" else "locked"
    offload["host_sync_idle_mode"] = idle_mode if backend == "mte" else "yield"
    if diagnose_load:
        offload["host_sync_diagnose_load"] = backend == "mte"
    if diagnose_native_wait or diagnose_split_scale:
        offload["host_sync_diagnose_load"] = backend == "mte"
        offload["host_sync_diagnose_native_wait"] = backend == "mte"
    if diagnose_split_scale:
        offload["host_sync_diagnose_split_scale"] = backend == "mte"
    words[index] = json.dumps(config)
    return source[: start.start()] + shlex.join(words) + "\n"


def failure_excerpt(text):
    """Bound copied output; full stacks and successful copies stay in service log."""
    lines = text.splitlines()
    selected = set()
    for marker, count in (
        ("[MOE-MTE LOAD]", 18),
        ("[MOE-MTE LAYOUT]", 1),
        ("STREAM_HANDLE_READY", 1),
        ("[MOE-MTE WORKER]", 1),
        ("[MOE-MTE DIAGNOSTIC BEGIN]", 1),
        ("active_callback=", 1),
        ("callback=_", 1),
        ("[MOE-MTE FATAL]", 1),
    ):
        selected.update([index for index, line in enumerate(lines) if marker in line][-count:])
    if not any("[MOE-MTE FATAL]" in line for line in lines):
        errors = [index for index, line in enumerate(lines) if "Error" in line or "ERROR" in line]
        selected.update(errors[:4] + errors[-4:])
        if errors:
            selected.update(range(max(0, errors[0] - 3), min(len(lines), errors[0] + 4)))
        else:
            selected.update(range(max(0, len(lines) - 8), len(lines)))
    return "\n".join(lines[index] for index in sorted(selected))


def prepare_runtime_artifacts(source, target):
    """Copy missing build artifacts only; never replace source code or existing files."""
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("runtime source must be a different, already working checkout")
    package = source / "vllm_ascend"
    custom = package / "_cann_ops_custom"
    if not (package / "_build_info.py").is_file() or not (custom / "vendors/custom_transformer").is_dir():
        raise ValueError("runtime source lacks _build_info.py or complete _cann_ops_custom package")
    files = [package / "_build_info.py"]
    if (package / "_version.py").is_file():
        files.append(package / "_version.py")
    files.extend(path for path in custom.rglob("*") if path.is_file())
    files.extend(path for path in package.rglob("*.so*") if path.is_file() and custom not in path.parents)
    copied = 0
    for path in files:
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1
    print(f"Runtime artifacts: copied missing files={copied}; existing files preserved; source={source}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, required=True)
    parser.add_argument("--backend", choices=("both", "callback", "mte"), default="both")
    parser.add_argument("--worker-scan", choices=("all", "allocated"), default="all")
    parser.add_argument("--worker-probe", choices=("locked", "atomic"), default="locked")
    parser.add_argument("--worker-idle", choices=("yield", "spin"), default="yield")
    parser.add_argument(
        "--diagnose-split-scale",
        action="store_true",
        help="Diagnostic only: split scale H2D and device layout copy, implies native wait diagnosis",
    )
    parser.add_argument(
        "--diagnose-native-wait",
        action="store_true",
        help="Diagnose raw ACL stream completion before torch synchronization; abort on ACL error",
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
    parser.add_argument(
        "--runtime-source", type=Path, help="Copy missing runtime build artifacts from an existing working checkout"
    )
    args = parser.parse_args()
    if args.diagnose_split_scale:
        args.diagnose_native_wait = True
    if args.diagnose_native_wait:
        args.diagnose_load_stages = True
    if sys.platform != "linux":
        parser.error("Run inside the Linux NPU container")
    if min(args.requests, args.warmup_requests, args.startup_timeout, args.request_timeout, args.profile_timeout) < 1:
        parser.error("Counts/timeouts must be positive")
    root = (args.output_dir or REPOSITORY_ROOT / "bench_results" / f"moe_host_sync_{uuid4().hex[:8]}").resolve()
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
    if args.runtime_source is not None:
        prepare_runtime_artifacts(args.runtime_source, REPOSITORY_ROOT)
    backends = ("callback", "mte") if args.backend == "both" else (args.backend,)
    build = select_build_directory(None, False)
    library = build / "lib/libhost_sync_device.so"
    if "mte" in backends:
        commands = [
            (
                "build Host extension",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "./csrc/host_sync",
                    "--no-build-isolation",
                    "--no-deps",
                    "--force-reinstall",
                ],
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
            ("CPU worker regression", [sys.executable, "-B", "tests/ut/expert_offload/test_worker_polling.py"]),
            ("MoE dispatch regression", [sys.executable, "-B", "tests/ut/expert_offload/test_moe_host_sync.py"]),
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
        print(
            f"Requested worker policy: scan={args.worker_scan}; probe={args.worker_probe}; "
            f"idle={args.worker_idle}; MTE only",
            flush=True,
        )
        if args.diagnose_load_stages:
            print("DIAGNOSTIC LOAD BARRIERS ENABLED: timings are not a performance comparison", flush=True)
        reports[backend] = inference(
            run_args,
            configured_source(
                source,
                backend,
                library,
                args.diagnose_load_stages,
                args.worker_scan,
                args.worker_probe,
                args.worker_idle,
                args.diagnose_native_wait,
                args.diagnose_split_scale,
            ),
            "python",
            env,
        )
        if backend == "mte":
            log = (directory / "inference_python.log").read_text(encoding="utf-8", errors="replace")
            expected_policy = (
                f"[MOE-MTE WORKER] scan={args.worker_scan} probe={args.worker_probe} idle={args.worker_idle}"
            )
            if expected_policy not in log:
                raise RuntimeError("Missing runtime evidence of requested MTE worker policies")
            print(expected_policy, flush=True)
            if args.diagnose_load_stages:
                print(f"DIAGNOSTIC EXCERPT (full log: {directory / 'inference_python.log'}):", flush=True)
                print(failure_excerpt(log), flush=True)
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
