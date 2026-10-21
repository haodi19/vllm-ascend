# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and validate stage 1. Print a console summary; do not write logs."""

import argparse
import contextlib
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FAILURE_TAIL_LINES = 80

ACL_SMOKE = r"""
import json
import torch
import torch_npu
import vllm_ascend_host_sync as hs

torch.npu.set_device(0)
torch.npu.current_stream()
assert hs.API_VERSION == 1, hs.API_VERSION
print("device_name=" + str(torch.npu.get_device_name(0)), flush=True)
print("torch_version=" + str(torch.__version__), flush=True)
print("torch_npu_version=" + str(torch_npu.__version__), flush=True)
print("ACL test uses visible logical NPU 0; requests originate on Host.", flush=True)
calls = [0, 0]
def task(slot):
    def callback():
        calls[slot] += 1
    return callback

with hs.Session(slots=2, backend="acl", device_id=0) as session:
    info = session.info()
    assert info["backend"] == "acl", info
    assert info["host_address"] != 0, info
    assert info["device_address"] != 0, info
    print("ACL allocation: " + json.dumps({
        "backend": info["backend"],
        "slots": info["slots"],
        "slot_stride": info["slot_stride"],
        "allocation_bytes": info["allocation_bytes"],
        "host_address_nonzero": bool(info["host_address"]),
        "device_address_nonzero": bool(info["device_address"]),
    }), flush=True)
    for slot in range(2):
        session.register_task(slot, task(slot))
    session.start()
    for generation in range(1, 21):
        for slot in range(2):
            session.submit_for_test(slot, generation)
        for slot in range(2):
            result = session.wait(slot, generation)
            assert result["done_generation"] == generation, result
            assert result["status"] == 2, result
    assert calls == [20, 20], calls
    session.stop()
    assert session.info()["state"] == "stopped", session.info()
assert session.info()["state"] == "closed", session.info()
print("PASS: ACL registration, 40 HOST-SIMULATED requests, stop and release.", flush=True)
print("NOT TESTED: NPU MTE, CPU/NPU memory visibility, graph replay, performance.", flush=True)
"""


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def npu_number(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("NPU number must be non-negative")
    return number


def terminate_process(process):
    """Terminate only the subprocess group created by this script."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()


def run_step(label, command, timeout, env=None, show_success_output=False):
    print(f"\nSTART: {label}; timeout={timeout:g}s", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name == "posix"),
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process(process)
        output, _ = process.communicate(timeout=10)
        print("\n".join(output.splitlines()[-FAILURE_TAIL_LINES:]), flush=True)
        print(f"FAIL: {label}; timed out after {timeout:g}s.", flush=True)
        return False
    except BaseException:
        terminate_process(process)
        process.wait(timeout=10)
        raise
    elapsed = time.monotonic() - started
    if process.returncode:
        print("\n".join(output.splitlines()[-FAILURE_TAIL_LINES:]), flush=True)
        print(
            f"FAIL: {label}; exit={process.returncode}; elapsed={elapsed:.1f}s",
            flush=True,
        )
        return False
    if show_success_output:
        print(output.rstrip(), flush=True)
    print(f"PASS: {label}; elapsed={elapsed:.1f}s", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, default=0, help="Physical NPU ID; ACL child uses logical NPU 0.")
    parser.add_argument("--cpu-only", action="store_true", help="Skip ACL/device checks; do not report NPU validation.")
    parser.add_argument("--skip-build", action="store_true", help="Use an already installed extension.")
    parser.add_argument("--build-timeout", type=positive_number, default=600)
    parser.add_argument("--test-timeout", type=positive_number, default=120)
    parser.add_argument("--acl-timeout", type=positive_number, default=120)
    args = parser.parse_args()

    print("=== Host sync stage 1: native Host worker and registered control memory ===", flush=True)
    print("This is a functional test, not a CPU/NPU handshake or performance benchmark.", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    if not args.skip_build:
        if not run_step(
            "build prerequisites",
            [
                sys.executable,
                "-B",
                "-c",
                "import setuptools, wheel, pybind11; print('setuptools + wheel + pybind11 available')",
            ],
            timeout=30,
        ):
            print(
                "Build needs setuptools, wheel and pybind11 in this Python environment. "
                "Install missing build dependencies using your approved package source, then retry.",
                flush=True,
            )
            return 1
        if not run_step(
            "build/install standalone extension",
            [
                sys.executable,
                "-B",
                "-m",
                "pip",
                "install",
                str(REPOSITORY_ROOT / "csrc" / "host_sync"),
                "--no-build-isolation",
                "--no-deps",
                "--disable-pip-version-check",
            ],
            timeout=args.build_timeout,
        ):
            return 1
    else:
        print("SKIP: extension build (--skip-build).", flush=True)

    if not run_step(
        "CPU unit tests",
        [
            sys.executable,
            "-B",
            str(REPOSITORY_ROOT / "tests" / "ut" / "expert_offload" / "test_host_sync_session.py"),
        ],
        timeout=args.test_timeout,
        show_success_output=True,
    ):
        return 1

    if args.cpu_only:
        print("\nNOT TESTED: ACL registration and device address (--cpu-only).", flush=True)
    else:
        acl_env = os.environ.copy()
        acl_env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
        print(f"\nACL subprocess physical NPU: {args.npu}; logical NPU: 0", flush=True)
        if not run_step(
            "ACL registration and Host-simulated tasks",
            [sys.executable, "-B", "-c", ACL_SMOKE],
            timeout=args.acl_timeout,
            env=acl_env,
            show_success_output=True,
        ):
            return 1

    print("\n=== STAGE 1 SUMMARY ===", flush=True)
    print("CPU worker, generation protocol and lifecycle: PASS", flush=True)
    print(
        "ACL registration / Host-simulated requests: " + ("NOT TESTED" if args.cpu_only else "PASS"),
        flush=True,
    )
    print("NPU MTE / CPU-NPU visibility / graph replay: NOT IMPLEMENTED OR TESTED", flush=True)
    print("MoE integration / inference / performance / profile: NOT TESTED", flush=True)
    print("Copy this console output after reviewing it; no log or result file was written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
