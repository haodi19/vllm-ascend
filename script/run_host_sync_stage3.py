# SPDX-License-Identifier: Apache-2.0
"""Run eager regression followed by isolated graph handshake correctness checks."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, positive_number, run_step
from run_host_sync_stage2 import main as run_eager_regression
from run_host_sync_stage2 import soc_version


def select_build_directory(requested, skip_build):
    if requested is not None:
        return requested.resolve()
    if skip_build:
        return REPOSITORY_ROOT / "build/host_sync_device_stage2"
    # CANN's generated subprojects may reuse incompatible intermediate objects
    # on an incremental build. Keep all previous artifacts; use a fresh tree.
    return REPOSITORY_ROOT / "build/host_sync_device_stage3" / f"run_{uuid4().hex}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, default=0)
    parser.add_argument("--soc-version", type=soc_version, default="ascend950pr_9579")
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--build-dir", type=Path, help="Exact build directory; default: a fresh directory per run")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-polls", type=int, default=100000)
    parser.add_argument("--build-timeout", type=positive_number, default=600)
    parser.add_argument("--test-timeout", type=positive_number, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10000 or not 1 <= args.max_polls <= 10000000:
        parser.error("rounds must be 1..10000 and max-polls must be 1..10000000")
    if sys.platform != "linux":
        parser.error("Stage 3 requires Linux + Ascend CANN")
    print("=== HOST SYNC STAGE 3: graph capture/replay correctness ===", flush=True)
    args.build_dir = select_build_directory(args.build_dir, args.skip_build)
    print(f"Build directory: {args.build_dir}", flush=True)
    prefix = [sys.executable, "-u", "-B"]
    stage2 = [
        "--npu",
        str(args.npu),
        "--soc-version",
        args.soc_version,
        "--cann-root",
        str(args.cann_root),
        "--build-dir",
        str(args.build_dir),
        "--rounds",
        str(args.rounds),
        "--max-polls",
        str(args.max_polls),
        "--build-timeout",
        str(args.build_timeout),
        "--test-timeout",
        str(args.test_timeout),
    ]
    if args.skip_build:
        stage2.append("--skip-build")
    # Reuse orchestration directly so each subprocess keeps its own timeout and
    # progress remains visible, rather than hiding the entire stage in a pipe.
    if run_eager_regression(stage2) != 0:
        return 1
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    library = args.build_dir.resolve() / "lib/libhost_sync_device.so"
    for case in ("normal", "callback-error", "poll-limit"):
        command = [
            *prefix,
            str(REPOSITORY_ROOT / "script/host_sync_device_probe.py"),
            "--library",
            str(library),
            "--mode",
            "graph",
            "--case",
            case,
            "--rounds",
            str(args.rounds),
            "--max-polls",
            str(args.max_polls),
        ]
        if not run_step(f"graph handshake: {case}", command, args.test_timeout, env=env, show_success_output=True):
            return 1
    print("\n=== STAGE 3 SUMMARY ===", flush=True)
    print("Eager regression: PASS", flush=True)
    print("Graph capture without executing Host tasks: PASS", flush=True)
    print("Repeated replay with device-generated generations: PASS", flush=True)
    print("Graph callback error / bounded polling: PASS", flush=True)
    print("MoE / mixed Torch-native operations / performance / profile: NOT TESTED", flush=True)
    print("Return reviewed console output; no experiment log or result files are written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
