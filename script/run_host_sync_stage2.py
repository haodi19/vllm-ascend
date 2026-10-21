# SPDX-License-Identifier: Apache-2.0
"""Build isolated AIV handshake kernels and run bounded device correctness checks."""

import argparse
import os
import re
import sys
from pathlib import Path

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, positive_number, run_step


def soc_version(value):
    value = value.lower()
    if not re.fullmatch(r"ascend950(?:pr|dt)_[a-z0-9]+", value):
        raise argparse.ArgumentTypeError("Use a concrete SoC, e.g. ascend950pr_9579, not ascend950")
    return value


def target_cmake_args(soc):
    return [f"-DSOC_VERSION={soc_version(soc)}", "-DASCEND_COMPUTE_UNIT=ascend950"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, default=0)
    parser.add_argument("--soc-version", type=soc_version, default="ascend950pr_9579")
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--build-dir", type=Path, default=REPOSITORY_ROOT / "build/host_sync_device_stage2")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-polls", type=int, default=100000)
    parser.add_argument("--build-timeout", type=positive_number, default=600)
    parser.add_argument("--test-timeout", type=positive_number, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10000 or not 1 <= args.max_polls <= 10000000:
        parser.error("rounds must be 1..10000 and max-polls must be 1..10000000")
    if sys.platform != "linux":
        parser.error("Stage 2 build/device tests require Linux + Ascend CANN")
    print("=== HOST SYNC STAGE 2: isolated eager MTE handshake ===", flush=True)
    print("No MoE integration, OPP installation, graph replay or performance comparison.", flush=True)
    print(f"Build target: SOC_VERSION={args.soc_version}; ASCEND_COMPUTE_UNIT=ascend950", flush=True)
    prefix = [sys.executable, "-u", "-B"]
    stage1 = [*prefix, str(REPOSITORY_ROOT / "script/run_host_sync_stage1.py"), "--cpu-only"]
    if args.skip_build:
        stage1.append("--skip-build")
    if not run_step(
        "Host extension and CPU regression", stage1, args.build_timeout + args.test_timeout, show_success_output=True
    ):
        return 1
    if not run_step(
        "device result validation (CPU)",
        [*prefix, str(REPOSITORY_ROOT / "tests/ut/expert_offload/test_host_sync_device_probe.py")],
        args.test_timeout,
        show_success_output=True,
    ):
        return 1
    build = args.build_dir.resolve()
    if not args.skip_build:
        configure = [
            "cmake",
            "-S",
            str(REPOSITORY_ROOT / "csrc/host_sync_device"),
            "-B",
            str(build),
            f"-DASCEND_CANN_PACKAGE_PATH={args.cann_root.resolve()}",
            *target_cmake_args(args.soc_version),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if not run_step("configure isolated AscendC kernels", configure, args.build_timeout):
            return 1
        if not run_step(
            "build isolated AscendC kernels", ["cmake", "--build", str(build), "-j", "2"], args.build_timeout
        ):
            return 1
    library = build / "lib/libhost_sync_device.so"
    if not library.is_file():
        print(f"FAIL: kernel library missing: {library}", flush=True)
        return 1
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    print(f"Device subprocesses: physical NPU={args.npu}, logical NPU=0", flush=True)
    for case in ("normal", "callback-error", "poll-limit"):
        command = [
            *prefix,
            str(REPOSITORY_ROOT / "script/host_sync_device_probe.py"),
            "--library",
            str(library),
            "--case",
            case,
            "--rounds",
            str(args.rounds),
            "--max-polls",
            str(args.max_polls),
        ]
        if not run_step(f"NPU/CPU handshake: {case}", command, args.test_timeout, env=env, show_success_output=True):
            return 1
    print("\n=== STAGE 2 SUMMARY ===", flush=True)
    print("NPU MTE publish -> Host worker -> NPU MTE wait: PASS", flush=True)
    print("Callback failure and bounded polling: PASS", flush=True)
    print("Repeated device-generated generations (eager): PASS", flush=True)
    print("Graph replay / MoE / speedup / profile: NOT TESTED", flush=True)
    print("Return reviewed console output; no experiment log or result files are written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
