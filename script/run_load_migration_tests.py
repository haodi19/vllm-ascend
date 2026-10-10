#!/usr/bin/env python3
"""Run the literal load_submit migration comparison; console output only."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def test_steps(python, skip_build):
    prefix = [python, "-u", "-B"]
    steps = []
    if not skip_build:
        steps.append(
            (
                "Build CPU extension",
                [*prefix, "-m", "pip", "install", "-v", "./csrc/lrc_policy", "--no-build-isolation", "--no-deps"],
            )
        )
    steps.append(("CPU parity tests", [*prefix, "tests/ut/expert_offload/test_load_submit_cpp.py"]))
    common = [
        *prefix,
        "bench_update_weights.py",
        "--mode",
        "npu",
        "--device",
        "0",
        "--backend",
        "cpp",
        "--h2d-backend",
        "torch",
        "--load-backend",
    ]
    for implementation in ("python", "cpp"):
        steps.append(
            (
                f"NPU byte validation: {implementation}",
                [*common, implementation, "--warmup", "40", "--iterations", "100", "--check-every", "1"],
            )
        )
    for index, implementation in enumerate(("python", "cpp", "cpp", "python"), start=1):
        steps.append((f"Performance {index}/4: {implementation}", [*common, implementation]))
    for implementation in ("python", "cpp"):
        steps.append((f"Detailed timing: {implementation}", [*common, implementation, "--detail-load"]))
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, default=4, help="Physical NPU ID (default: 4); child logical device is 0")
    parser.add_argument("--skip-build", action="store_true", help="Use the already rebuilt API 3 CPU extension")
    args = parser.parse_args(argv)
    if args.npu < 0:
        parser.error("npu must be >= 0")
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    steps = test_steps(sys.executable, args.skip_build)
    print("=== LOAD MIGRATION TEST BEGIN ===", flush=True)
    print(f"Physical NPU={args.npu}; logical device=0; steps={len(steps)}", flush=True)
    print("Fixed LRC=cpp, transport=torch. Only load-backend changes: python/cpp.", flush=True)
    print("No fused planning or ACL batching. Console only; no result/log files or Git operations.", flush=True)
    print("Building installs the CPU extension and creates normal build artifacts.", flush=True)
    for index, (label, command) in enumerate(steps, start=1):
        print(f"\n=== [{index}/{len(steps)}] {label} ===", flush=True)
        try:
            result = subprocess.run(command, cwd=repo, env=env, check=False)
        except OSError as exc:
            print(f"FAILED to start step {index}: {exc}. Remaining steps NOT run.", flush=True)
            return 1
        if result.returncode:
            print(f"FAILED step {index}: {label}; exit={result.returncode}. Remaining steps NOT run.", flush=True)
            return result.returncode if result.returncode > 0 else 1
        print(f"=== [{index}/{len(steps)}] PASS ===", flush=True)
    print("\n=== LOAD MIGRATION TEST COMPLETE: ALL STEPS PASSED ===", flush=True)
    print("Completion confirms checks ran; it does NOT imply a speedup.", flush=True)
    print("Return reviewed console headers, test summaries and timing tables; no files needed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; remaining steps NOT run.", flush=True)
        raise SystemExit(130) from None
