# SPDX-License-Identifier: Apache-2.0
"""Build and check graph dataflow with real D2H/H2D copies and synthetic weights."""

import argparse
import os
import sys

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, positive_number, run_step
from run_host_sync_stage3 import main as run_regression
from run_host_sync_stage3 import select_build_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--test-timeout", type=positive_number, default=120)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("Requires Linux + Ascend950 CANN")
    if not 1 <= args.rounds <= 10000 or not 8 <= args.elements <= 1048576 or args.elements % 8:
        parser.error("rounds: 1..10000; elements: multiple of 8 in 8..1048576")
    print("=== HOST SYNC STAGE 5: real-copy dataflow correctness ===", flush=True)
    build = select_build_directory(None, False)
    if (
        run_regression(
            [
                "--npu",
                str(args.npu),
                "--rounds",
                "10",
                "--build-dir",
                str(build),
                "--test-timeout",
                str(args.test_timeout),
            ]
        )
        != 0
    ):
        return 1
    if not run_step(
        "dataflow result checks (CPU)",
        [sys.executable, "-B", str(REPOSITORY_ROOT / "tests/ut/expert_offload/test_host_sync_dataflow.py")],
        30,
        show_success_output=True,
    ):
        return 1
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    for backend in ("callback", "mte"):
        command = [
            sys.executable,
            "-u",
            "-B",
            str(REPOSITORY_ROOT / "script/host_sync_dataflow.py"),
            "--library",
            str(build / "lib/libhost_sync_device.so"),
            "--backend",
            backend,
            "--rounds",
            str(args.rounds),
            "--elements",
            str(args.elements),
        ]
        if not run_step(f"graph dataflow: {backend}", command, args.test_timeout, env=env, show_success_output=True):
            return 1
    print("\n=== STAGE 5 SUMMARY ===", flush=True)
    print("D2H route -> Host selection -> separate-stream H2D+wait -> mapping H2D -> NPU consumer: PASS", flush=True)
    print("Changing input / expert choice / full weight validation each replay: PASS", flush=True)
    print("Actual MoE / Torch copy_ integration / multiple slots / performance: NOT TESTED", flush=True)
    print("Return reviewed console output; no log/result files written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
