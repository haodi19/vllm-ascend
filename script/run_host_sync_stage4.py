# SPDX-License-Identifier: Apache-2.0
"""Compare isolated graph handoffs, then collect separate profiles."""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
from pathlib import Path
from uuid import uuid4

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, positive_number, run_step
from run_host_sync_stage2 import soc_version
from run_host_sync_stage3 import main as run_graph_regression
from run_host_sync_stage3 import select_build_directory


def compare_results(results):
    grouped = {name: [r for r in results if r["backend"] == name] for name in ("callback", "mte")}
    if not all(grouped.values()) or len(grouped["callback"]) != len(grouped["mte"]):
        raise ValueError("Comparison requires matching runs of both backends")
    if any(r["profiler_enabled"] or r["unit"] != "us" for r in results):
        raise ValueError("Profiled results or inconsistent timing units cannot enter comparison")
    if len({r["rounds"] for r in results}) != 1:
        raise ValueError("Comparison requires matching sample counts")
    compared = {}
    for metric in ("event_interval", "replay_sync_host", "callback_body"):
        means = {name: statistics.mean(r[metric]["mean"] for r in rows) for name, rows in grouped.items()}
        compared[metric] = {
            "callback_mean_us": means["callback"],
            "mte_mean_us": means["mte"],
            "mte_minus_callback_us": means["mte"] - means["callback"],
        }
    return compared


def capture_step(label, command, timeout, env):
    output = io.StringIO()
    print(f"Running {label}; deadline={timeout:g}s", flush=True)
    with contextlib.redirect_stdout(output):
        success = run_step(label, command, timeout, env=env, show_success_output=True)
    text = output.getvalue()
    print(text, end="", flush=True)
    return success, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, default=0)
    parser.add_argument("--only", choices=("all", "profile"), default="all")
    parser.add_argument("--soc-version", type=soc_version, default="ascend950pr_9579")
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-polls", type=int, default=100000)
    parser.add_argument("--profile-rounds", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--build-timeout", type=positive_number, default=600)
    parser.add_argument("--test-timeout", type=positive_number, default=120)
    parser.add_argument("--profile-timeout", type=positive_number, default=180)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("Stage 4 requires Linux + Ascend CANN")
    if not (
        1 <= args.rounds <= 10000
        and 1 <= args.warmup <= 1000
        and 1 <= args.repeats <= 10
        and 1 <= args.profile_rounds <= 100
        and 1 <= args.max_polls <= 10000000
    ):
        parser.error(
            "Invalid counts: rounds<=10000, warmup<=1000, repeats<=10, profile-rounds<=100, max-polls<=10000000"
        )
    build = select_build_directory(args.build_dir, args.skip_build)
    profile_root = (args.profile_dir or REPOSITORY_ROOT / "moe_profile/host_sync_stage4") / f"run_{uuid4().hex}"
    print("=== HOST SYNC STAGE 4: graph handoff comparison ===", flush=True)
    print(f"Profile root: {profile_root}", flush=True)
    regression_args = [
        "--npu",
        str(args.npu),
        "--soc-version",
        args.soc_version,
        "--cann-root",
        str(args.cann_root),
        "--build-dir",
        str(build),
        "--rounds",
        "10",
        "--max-polls",
        str(args.max_polls),
        "--build-timeout",
        str(args.build_timeout),
        "--test-timeout",
        str(args.test_timeout),
    ]
    if args.skip_build:
        regression_args.append("--skip-build")
    if run_graph_regression(regression_args) != 0:
        return 1
    if not run_step(
        "benchmark CPU checks",
        [sys.executable, "-B", str(REPOSITORY_ROOT / "tests/ut/expert_offload/test_host_sync_benchmark.py")],
        30,
        show_success_output=True,
    ):
        return 1
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)

    def command(backend, profiled):
        result = [
            sys.executable,
            "-u",
            "-B",
            str(REPOSITORY_ROOT / "script/host_sync_benchmark.py"),
            "--library",
            str(build / "lib/libhost_sync_device.so"),
            "--backend",
            backend,
            "--rounds",
            str(args.profile_rounds if profiled else args.rounds),
            "--warmup",
            str(args.warmup),
            "--max-polls",
            str(args.max_polls),
        ]
        if profiled:
            result += ["--profile-dir", str(profile_root / backend)]
        return result

    results = []
    for repeat in range(args.repeats if args.only == "all" else 0):
        order = ("callback", "mte") if repeat % 2 == 0 else ("mte", "callback")
        for backend in order:
            success, text = capture_step(
                f"timing {repeat + 1}/{args.repeats}: {backend}", command(backend, False), args.test_timeout, env
            )
            if not success:
                return 1
            records = [
                json.loads(line.removeprefix("BENCH_RESULT "))
                for line in text.splitlines()
                if line.startswith("BENCH_RESULT ")
            ]
            if len(records) != 1 or records[0]["backend"] != backend:
                raise RuntimeError("Missing or mismatched benchmark result")
            results.extend(records)
    if results:
        print("COMPARISON " + json.dumps(compare_results(results)), flush=True)
        print("These are complete mechanism measurements, not pure CPU/NPU synchronization overhead.", flush=True)
    profile_status = {}
    for backend in ("callback", "mte"):
        success, _ = capture_step(f"separate profile: {backend}", command(backend, True), args.profile_timeout, env)
        profile_status[backend] = "PASS" if success else "INCOMPLETE"
    print("\n=== STAGE 4 SUMMARY ===", flush=True)
    print("Unprofiled timing: " + ("PASS" if results else "NOT RUN (--only profile)"), flush=True)
    print("PROFILE STATUS " + json.dumps(profile_status), flush=True)
    print(f"Profile root: {profile_root}", flush=True)
    print("MoE integration / inference speedup / pure synchronization cost: NOT ESTABLISHED", flush=True)
    return 0 if all(v == "PASS" for v in profile_status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
