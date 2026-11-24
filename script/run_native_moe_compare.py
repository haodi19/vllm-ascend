# SPDX-License-Identifier: Apache-2.0
"""Build and validate opt-in native MTE updates against the existing MTE path.

Only the vllm serve arguments are read from test_graph2.sh; its shell prelude is
not executed. Uses the caller's activated environment. No original file edits.
"""

import argparse
import contextlib
import errno
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

from moe_tpot import streaming_completion, summarize_streams
from run_host_sync_stage2 import target_cmake_args

ROOT = Path(__file__).resolve().parents[1]
PORT_RELEASE_TIMEOUT = 10
PORT_POLL_INTERVAL = 0.2


def wait_port_free(port, timeout=0):
    """Allow closed TCP connections while rejecting a live listener/bound port."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.socket() as probe:
                probe.settimeout(0.5)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise OSError(errno.EADDRINUSE, "listener is still accepting connections")
            with socket.socket() as sock:
                # Match the server's bind policy: TIME_WAIT alone is not a
                # live service. SO_REUSEPORT is deliberately not enabled.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
            return
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Port {port} is unavailable after {timeout}s: {exc}; "
                    "next service not started. Check the listener or choose --port; no unrelated process was stopped"
                ) from exc
            time.sleep(PORT_POLL_INTERVAL)


def handshake_configure_command(build, cann_root, soc):
    return [
        "cmake",
        "-S",
        str(ROOT / "csrc/host_sync_device"),
        "-B",
        str(build),
        f"-DASCEND_CANN_PACKAGE_PATH={cann_root.resolve()}",
        *target_cmake_args(soc),
        "-DCMAKE_BUILD_TYPE=Release",
    ]


def fresh_handshake_build(parent):
    """Keep artifacts outside results, but never reuse AscendC merge objects."""
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = parent / f"build_{number:02d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            number += 1


def report_failed_step(log_path, start_offset, report):
    """Keep the last failing command visible without copying a whole build log."""
    with log_path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        log.seek(max(start_offset, log.tell() - 32768))
        tail = log.read().decode("utf-8", errors="replace").splitlines()[-60:]
    report("FAILED_STEP_LOG_BEGIN (last at most 60 lines / 32 KiB of this step)")
    for line in tail:
        report(line)
    report("FAILED_STEP_LOG_END")


def timing_row(case, timing, e2e_mean_ms):
    values = []
    for metric in ("decode_ms", "ttft_ms", "tpot_ms"):
        raw = timing[metric]["raw"]
        values.append("undefined" if raw is None else f"{raw['mean']:.3f}")
    return f"| {case} | {' | '.join(values)} | {e2e_mean_ms:.3f} |"


def report_service_failure(case, log_path, samples, completed, proc, exc, report):
    """Preserve failure evidence before cleanup adds secondary shutdown errors."""
    report(
        f"SERVICE_FAILURE case={case}; completed_requests_including_warmup={completed}; "
        f"completed_timed_requests={len(samples)}; service_exit={proc.poll()}; "
        f"error={type(exc).__name__}: {exc}"
    )
    partial = log_path.with_name(f"{case}_partial.json")
    partial.write_text(
        json.dumps(dict(case=case, status="incomplete", stream_samples=samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report(f"PARTIAL_SAMPLES={partial}; incomplete comparison, no speedup claim")
    report(f"SERVICE_LOG_BEFORE_CLEANUP={log_path}")
    report_failed_step(log_path, 0, report)


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
    # Match the established runner: test_graph2.sh may end in a continuation
    # backslash without a final newline. Keep quoted values/escapes intact.
    words = [word for word in shlex.split(source[match.start() :] + "\n", comments=True) if word.strip()]
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
    parser.add_argument("--skip-profile", action="store_true", help="Timing only; omit separate diagnostic services")
    parser.add_argument("--timing-port", type=int, default=18156)
    parser.add_argument("--profile-timeout", type=int, default=180)
    parser.add_argument(
        "--bubble-stream-name", default="Stream 56", help="Exact visualization stream name, never guessed"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build/native_moe_handshake",
        help="Persistent parent for fresh numbered handshake builds, independent of results",
    )
    args = parser.parse_args()
    if sys.platform != "linux" or args.npu < 0 or min(args.max_tokens, args.requests, args.warmup_requests) < 1:
        parser.error("requires Linux, a non-negative NPU and positive request/token counts")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.timing_port <= 65535 or args.timing_port == args.port or args.profile_timeout <= 0:
        parser.error("requires a separate valid timing port and a positive profile timeout")
    args.profile_requests = 1
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
        log_path = root / "build.log"
        start_offset = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open("a", encoding="utf-8") as out:
            out.write("STEP_COMMAND " + json.dumps(command) + "\n")
            out.flush()
            proc = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = proc.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f"{label} exit={code}; see {root / 'build.log'}")
            except BaseException:
                stop_process_group(proc)
                report_failed_step(log_path, start_offset, report)
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
            "MTE Python update -> MTE native update; same cache/config; timed requests have no profiler/MSTX. "
            "Native graph replay only; original eager/prefill. This is initial correctness/performance validation."
        )
        report(
            "PRIMARY=decode_ms (first to last content); TPOT=decode/(actual output tokens-1); "
            "HTTP E2E and TTFT retained. Five warmups by default; all samples in this summary."
        )
        build = args.build_dir.resolve()
        library = build / "lib/libhost_sync_device.so"
        if args.skip_build:
            if args.library is None:
                raise ValueError("--skip-build requires --library")
            library = args.library.resolve()
        # Parse both commands before compiling or touching the NPU. A syntax
        # error in the local launch script must not waste the hardware checks.
        source = args.test_script.read_text(encoding="utf-8")
        commands = {case: service_command(source, case, library, args.port) for case in ("python", "native")}
        report(f"PASS launch command parsing: {args.test_script}")
        wait_port_free(args.port)
        if not args.skip_profile:
            wait_port_free(args.timing_port)
        report(f"PASS initial port check: {args.port}")
        if not args.skip_build:
            # CANN merge_obj_text may rewrite intermediate .o files in place.
            # Reusing those objects has produced 'unknown file type' on rebuild.
            # Reserve a new directory; retain all previous artifacts/evidence.
            build = fresh_handshake_build(args.build_dir)
            library = build / "lib/libhost_sync_device.so"
            commands = {case: service_command(source, case, library, args.port) for case in ("python", "native")}
            report(f"HANDSHAKE_BUILD={build}; fresh directory; existing builds retained")
            report(f"HANDSHAKE_LIBRARY={library}; persistent path outside experiment results")
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
            step(
                "configure handshake",
                handshake_configure_command(build, args.cann_root, soc),
            )
            step("build handshake", ["cmake", "--build", str(build), "--verbose", "-j", "2"])
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
        step(
            "NPU native strict graph and GIL independence",
            [
                sys.executable,
                "-B",
                "script/test_native_update_graph.py",
                "--library",
                str(library),
                *([] if args.skip_profile else ["--native-profile"]),
            ],
            timeout=180,
        )
        reference = None
        means = {}
        timings = {}
        for case in ("python", "native"):
            wait_port_free(args.port)
            command = commands[case]
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
                samples, completed = [], 0
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
                        result = streaming_completion(opener, args.port, body, timeout=args.request_timeout)
                        elapsed = result["http_wall_us"] / 1000
                        if result["usage"]["completion_tokens"] != args.max_tokens:
                            raise RuntimeError("unexpected completion count")
                        if i >= args.warmup_requests:
                            samples.append(result)
                            outputs.append((result["digest"], result["usage"]))
                        completed += 1
                        report(
                            f"REQUEST {case} {i + 1}/{args.warmup_requests + args.requests}; "
                            f"e2e_ms={elapsed:.3f}; decode_ms={result['decode_us'] / 1000:.3f}"
                        )
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
                    means[case] = statistics.mean(sample["http_wall_us"] / 1000 for sample in samples)
                    timings[case] = summarize_streams(samples)
                    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                        if "[MOE-NATIVE]" in line:
                            report(line)
                    report(
                        "RESULT "
                        + json.dumps(
                            dict(
                                case=case,
                                requests=len(samples),
                                mean_ms=means[case],
                                stream_samples=samples,
                                timing=timings[case],
                                output_match=True,
                                digest=hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                            )
                        )
                    )
                except Exception as exc:
                    try:
                        report_service_failure(case, log_path, samples, completed, proc, exc, report)
                    except Exception as diagnostic_error:
                        report(f"FAILURE_DIAGNOSTICS_UNAVAILABLE {type(diagnostic_error).__name__}: {diagnostic_error}")
                    raise
                finally:
                    stop_process_group(proc)
            report(f"WAIT port release after service={case}; port={args.port}; timeout={PORT_RELEASE_TIMEOUT}s")
            wait_port_free(args.port, timeout=PORT_RELEASE_TIMEOUT)
            report(f"STOP service={case}; port={args.port} reusable")
        report("| Version | Decode mean ms (primary) | TTFT mean ms | TPOT mean ms/token | HTTP E2E mean ms |")
        report("| --- | ---: | ---: | ---: | ---: |")
        for case in ("python", "native"):
            report(timing_row(case, timings[case], means[case]))
        before = timings["python"]["decode_ms"]["raw"]["mean"]
        after = timings["native"]["decode_ms"]["raw"]["mean"]
        change = f"{(after / before - 1) * 100:.3f}%" if before > 0 else "undefined (single content chunk)"
        report(
            f"PASS finite output parity; decode mean change={change}; "
            f"E2E mean change={(means['native'] / means['python'] - 1) * 100:.3f}%; "
            "single service pair, not stability acceptance"
        )
        if not args.skip_profile:
            from native_moe_profile import profile_cases

            report(
                "PROFILE_PHASE: fresh Python/native services after timing; one request each after warmup; "
                "native C MSTX ranges, aligned visualization, Host gaps and loading bubbles"
            )
            profile_cases(args, commands, root, env, report, wait_port_free, stop_process_group, reference)
    except Exception as exc:
        report(f"FAILED {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
