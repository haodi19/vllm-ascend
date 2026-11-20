# SPDX-License-Identifier: Apache-2.0
"""Build and validate opt-in native MTE updates against the existing MTE path.

Only the vllm serve arguments are read from test_graph2.sh; its shell prelude is
not executed. Uses the caller's activated environment. No original file edits.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_host_sync_stage2 import target_cmake_args

ROOT = Path(__file__).resolve().parents[1]


def stop_process_group(proc):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=40)
    # A failed leader may leave compiler/engine children behind.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def service_command(source, case, library, port):
    match = re.search(r"(?m)^vllm\s+serve\s+", source)
    if match is None:
        raise ValueError("test script must contain vllm serve")
    words = [word for word in shlex.split(source[match.start() :], comments=True) if word.strip()]
    index = words.index("--additional-config") + 1
    config = json.loads(words[index])
    offload = config["expert_offload_config"]
    offload.update(
        host_update_backend=case,
        host_sync_backend="mte",
        host_sync_load_stream="ordinary",
        host_sync_scan_mode="allocated",
        host_sync_idle_probe="atomic",
        host_sync_idle_mode="yield",
        host_sync_library=str(library),
        cache_policy_backend="cpp",
        cache_policy_enabled=True,
        moe_offload_debug=False,
        callback_timing_port=0,
        callback_mstx_enabled=False,
        expert_trace_enabled=False,
        expert_prefetch_wait_timing=False,
        host_sync_diagnose_prefetch_wait_at_join=bool(offload.get("expert_prefetch_enabled")),
    )
    # Keep all other workload/cache/model settings from the original script.
    words[index] = json.dumps(config)
    words[words.index("--port") + 1] = str(port)
    if "--profiler-config" in words:
        index = words.index("--profiler-config")
        del words[index : index + 2]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", *words[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--port", type=int, default=8156)
    parser.add_argument("--test-script", type=Path, default=ROOT / "script/test_graph2.sh")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--library", type=Path, help="existing handshake library, required with --skip-build")
    args = parser.parse_args()
    if sys.platform != "linux" or args.npu < 0 or min(args.max_tokens, args.requests, args.warmup_requests) < 1:
        parser.error("requires Linux, a non-negative NPU and positive request/token counts")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.npu))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    summary = root / "summary.txt"

    def report(message):
        print(message, flush=True)
        with summary.open("a", encoding="utf-8") as out:
            out.write(message + "\n")

    def step(label, command, timeout=600):
        report(f"START {label}")
        with (root / "build.log").open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = proc.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f"{label} exit={code}; see {root / 'build.log'}")
            except BaseException:
                stop_process_group(proc)
                raise
        report(f"PASS {label}")

    def request(path, body=None, timeout=10):
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    try:
        report(
            "MTE Python update -> MTE native update; same cache/config; no profiler/MSTX. "
            "Native graph replay only; original eager/prefill. This is initial correctness/performance validation."
        )
        if not args.skip_build:
            for extension in ("lrc_policy", "host_sync"):
                step(
                    f"build {extension}",
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"./csrc/{extension}",
                        "--no-build-isolation",
                        "--no-deps",
                        "--force-reinstall",
                    ],
                )
            result = subprocess.run(
                [sys.executable, "-c", "import torch,torch_npu; print('SOC='+torch.npu.get_device_name(0).lower())"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            soc = next(line[4:] for line in result.stdout.splitlines() if line.startswith("SOC="))
            build = root / "device_build"
            step(
                "configure handshake",
                [
                    "cmake",
                    "-S",
                    str(ROOT / "csrc/host_sync_device"),
                    "-B",
                    str(build),
                    f"-DASCEND_CANN_PACKAGE_PATH={args.cann_root.resolve()}",
                    *target_cmake_args(soc),
                ],
            )
            step("build handshake", ["cmake", "--build", str(build), "-j", "2"])
            library = build / "lib/libhost_sync_device.so"
        else:
            if args.library is None:
                raise ValueError("--skip-build requires --library")
            library = args.library.resolve()
        if not library.is_file():
            raise FileNotFoundError(library)
        # Missing extensions/APIs are fatal, not unittest skips.
        step(
            "native API check",
            [
                sys.executable,
                "-c",
                "import _vllm_ascend_lrc as l; "
                "import vllm_ascend_host_sync as h; "
                "assert hasattr(l,'native_update_task'); assert hasattr(h.Session,'install_native_task')",
            ],
        )
        step("native CPU regression", [sys.executable, "-B", "tests/ut/expert_offload/test_native_update.py"])
        step("original worker regression", [sys.executable, "-B", "tests/ut/expert_offload/test_worker_polling.py"])
        step("NPU scale byte-layout parity", [sys.executable, "-B", "script/test_native_scale_copy.py"])
        reference = None
        means = {}
        source = args.test_script.read_text(encoding="utf-8")
        for case in ("python", "native"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", args.port))
            command = service_command(source, case, library, args.port)
            # This exact argv preserves the original model and MTP settings.
            (root / f"{case}_command.json").write_text(
                json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log_path = root / f"{case}.log"
            report(f"START service={case}; log={log_path}")
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    deadline = time.monotonic() + args.startup_timeout
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"{case} service exited: {proc.returncode}; see {log_path}")
                        try:
                            models = request("/v1/models")
                            break
                        except (OSError, ValueError, urllib.error.URLError):
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f"{case} startup timeout")
                            time.sleep(2)
                    outputs, samples = [], []
                    body = dict(
                        model=models["data"][0]["id"],
                        prompt="The future of AI is",
                        max_tokens=args.max_tokens,
                        temperature=0,
                        ignore_eos=True,
                    )
                    for i in range(args.warmup_requests + args.requests):
                        start = time.perf_counter()
                        result = request("/v1/completions", body, args.request_timeout)
                        elapsed = (time.perf_counter() - start) * 1000
                        if result["usage"]["completion_tokens"] != args.max_tokens:
                            raise RuntimeError("unexpected completion count")
                        if i >= args.warmup_requests:
                            samples.append(elapsed)
                            outputs.append((result["choices"][0]["text"], result["usage"]))
                        report(f"REQUEST {case} {i + 1}/{args.warmup_requests + args.requests}; ms={elapsed:.3f}")
                    if (
                        case == "native"
                        and "first native graph task completed without Python"
                        not in log_path.read_text(encoding="utf-8", errors="replace")
                    ):
                        raise RuntimeError("no confirmed native graph execution; refusing fallback-only comparison")
                    if reference is None:
                        reference = outputs
                    elif outputs != reference:
                        raise RuntimeError("native outputs/usage differ from original MTE; no speedup claim")
                    means[case] = statistics.mean(samples)
                    report(
                        "RESULT "
                        + json.dumps(
                            dict(
                                case=case,
                                requests=len(samples),
                                mean_ms=means[case],
                                samples_ms=samples,
                                output_match=True,
                                digest=hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                            )
                        )
                    )
                finally:
                    stop_process_group(proc)
            report(f"STOP service={case}")
        report(
            f"PASS finite output parity; E2E mean change={(means['native'] / means['python'] - 1) * 100:.3f}%; "
            "includes prefill, not decode-only TPOT; single service pair, not stability acceptance"
        )
    except Exception as exc:
        report(f"FAILED {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
