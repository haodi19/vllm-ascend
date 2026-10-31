#!/usr/bin/env python3
"""Python vs verified C++ LRC: summary.txt, separate logs and inference profiles."""

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import runpy
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIN_PROFILE_FREE_GIB = 20
GIB = 1024**3
VERIFICATION_PROMPTS = (
    "Explain why the sky appears blue during the day in two sentences.",
    "计算17乘以23，并简要说明计算过程。",
    "Write a Python function that returns the sum of the squares of a list of integers.",
)
VERIFICATION_MAX_TOKENS = 32
METRICS = (
    "total",
    "observe",
    "choose_victims",
    "synchronize",
    "outer",
    "host_without_sync",
    "load_submit",
    "fused_plan",
    "load_prepare",
    "copy_submit",
    "scale_refresh",
    "load_other",
)


def ensure_profile_storage(path, *, stage):
    """Check actual output filesystem before building, starting a service or profiling."""
    directory = Path(path).resolve()
    while not directory.exists():
        directory = directory.parent
    free_gib = shutil.disk_usage(directory).free / GIB
    free_inodes = None
    if hasattr(os, "statvfs"):
        info = os.statvfs(directory)
        if info.f_files > 0:
            free_inodes = info.f_favail
    print(
        f"STORAGE: stage={stage}; path={directory}; free={free_gib:.2f} GiB; "
        f"required={MIN_PROFILE_FREE_GIB} GiB; available_inodes={free_inodes}",
        flush=True,
    )
    if free_gib < MIN_PROFILE_FREE_GIB or free_inodes == 0:
        raise RuntimeError(
            f"Insufficient profile storage at {directory}: free={free_gib:.2f} GiB, "
            f"required={MIN_PROFILE_FREE_GIB} GiB, available_inodes={free_inodes}. "
            "Stop comparison; completed results are retained. Free space or use a new output directory "
            "on a filesystem with sufficient space."
        )


class SummaryTee:
    """Flush each write so completed measurements survive a later stage failure."""

    def __init__(self, console, summary):
        self.console, self.summary = console, summary
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.summary.write(text)
            self.summary.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.summary.flush()


@contextlib.contextmanager
def progress(label, interval=15):
    begin = time.monotonic()
    done = threading.Event()

    def heartbeat():
        while not done.wait(interval):
            print(f"WAIT: {label}; elapsed={time.monotonic() - begin:.0f}s", flush=True)

    print(f"START: {label}", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    success = False
    try:
        yield
        success = True
    finally:
        done.set()
        thread.join()
        state = "DONE" if success else "FAILED"
        print(f"{state}: {label}; elapsed={time.monotonic() - begin:.1f}s", flush=True)


def launch_spec(source, backend, npu, port, timing_port, profile_dir=None, mstx=False):
    """Read literal serve options; retain environment setup, not shell side effects."""
    matches = list(re.finditer(r"(?m)^vllm\s+serve\s+", source))
    if len(matches) != 1:
        raise ValueError("Expected one literal vllm serve command in test_graph2.sh")
    start = matches[0].start()
    words = [w for w in shlex.split(source[start:] + "\n", comments=True) if w.strip()]

    def option(name, default=None):
        return words[words.index(name) + 1] if name in words else default

    def replace(name, value):
        if name in words:
            words[words.index(name) + 1] = str(value)
        else:
            words.extend((name, str(value)))

    for name in ("--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size", "--api-server-count"):
        if option(name, "1") != "1":
            raise ValueError("This comparison requires TP=DP=PP=API-server-count=1")
    config = json.loads(option("--additional-config", "{}"))
    offload = config.get("expert_offload_config", {})
    if not offload.get("expert_offload") or not offload.get("cache_policy_enabled") or offload.get("enable_multi_card"):
        raise ValueError("Launch script must enable single-card expert offload and LRC")
    if backend not in ("python", "cpp"):
        raise ValueError("Only original Python and verified cpp LRC are allowed")
    offload.update(
        cache_policy_backend=backend,
        h2d_backend="torch",
        callback_timing_port=timing_port,
    )
    if mstx:
        offload["callback_mstx_enabled"] = True
    else:
        offload.pop("callback_mstx_enabled", None)
    replace("--additional-config", json.dumps(config))
    if port is not None:
        replace("--port", port)
    if profile_dir is not None:
        profiler = json.loads(option("--profiler-config", "{}"))
        if profiler.get("profiler") != "torch" or not profiler.get("torch_profiler_dir"):
            raise ValueError("test_graph2.sh must configure the torch profiler and its output directory")
        profiler["torch_profiler_dir"] = str(profile_dir)
        replace("--profiler-config", json.dumps(profiler))
    # Keep launch settings, but do not delete logs or change host sysctl settings.
    prefix = [
        line
        for line in source[:start].splitlines()
        if line.strip().startswith(("export ", "unset ", "nic_name=", "local_ip="))
    ]
    if npu is not None:
        prefix.append(f"export ASCEND_RT_VISIBLE_DEVICES={npu}")
    prefix.append("export PYTHONDONTWRITEBYTECODE=1")
    prefix.append(f"export PYTHONPATH={shlex.quote(str(REPO))}:$PYTHONPATH")
    model_path = Path(words[2])
    if not model_path.is_absolute():
        model_path = REPO / model_path
    return "\n".join(prefix), words, option("--served-model-name", words[2]), model_path / "config.json"


def bash_command(prefix, words):
    return ["bash", "-c", prefix + "\nexec " + shlex.join(words)]


def startup_trace_command(command):
    """Change only the entry point; retain every serve argument and shell setting."""
    if command[:2] != ["vllm", "serve"]:
        raise ValueError("Startup tracing requires the validated vllm serve command")
    return [sys.executable, "-B", str(REPO / "script/mte_startup_bootstrap.py"), "--diagnostic", *command[1:]]


def startup_cleanup_marker(child, log, reason):
    record = dict(
        event="cleanup_begin",
        pid=os.getpid(),
        api_pid=child.pid,
        api_returncode=child.returncode,
        reason=reason,
        wall_time_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
    )
    line = "[MOE-MTE RUNNER] " + json.dumps(record)
    with contextlib.suppress(Exception):
        print(line, flush=True)
    if log is not None:
        with contextlib.suppress(Exception):
            print(line, file=log, flush=True)


def run_console(command, env, log_path):
    """Keep noisy child output in a log and only publish numeric timing rows."""
    means = {}
    with (
        log_path.open("w", encoding="utf-8") as log,
        progress(f"subprocess; log={log_path}"),
        subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as child,
    ):
        for line in child.stdout:
            log.write(line)
            log.flush()
            parts = line.split()
            if len(parts) == 7 and parts[0] in METRICS:
                with contextlib.suppress(ValueError):
                    means[parts[0]] = float(parts[1])
                    print(line, end="", flush=True)
            elif (
                parts
                and parts[0] == "metric"
                or line.startswith(
                    (
                        "Model dimensions:",
                        "torch=",
                        "mode=",
                        "backend=",
                        "experts=",
                        "detail_load=",
                        "load_backend=",
                        "manager_sha256=",
                        "policy_sha256=",
                        "transport_sha256=",
                        "debug=",
                        "h2d_backend=",
                        "warmup=",
                    )
                )
            ):
                print(line, end="", flush=True)
        if child.wait():
            raise RuntimeError(f"Stage failed (exit={child.returncode}); details in {log_path}")
    return means


def compare(label, python, cpp):
    print(f"\n=== COMPARISON: {label}; mean microseconds ===", flush=True)
    print(f"{'metric':26} {'python':>12} {'cpp':>12} {'cpp-python':>12} {'change':>10}", flush=True)
    for name in METRICS:
        if name not in python or name not in cpp:
            continue
        a, b = python[name], cpp[name]
        ratio = f"{100 * (b / a - 1):+.1f}%" if a else "n/a"
        print(f"{name:26} {a:12.3f} {b:12.3f} {b - a:12.3f} {ratio:>10}", flush=True)


def request(opener, port, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    status = (
        progress(f"POST {port}{path}; timeout={timeout}s")
        if payload is not None and path != "/v1/completions"
        else contextlib.nullcontext()
    )
    with status, opener.open(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def assert_port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied; stop that service yourself or choose another port") from exc


def stop_owned_service(child):
    """Only signal the process group created by this runner, never other services."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        child.poll()  # reap group leader so killpg does not see its zombie
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    print(f"Service group {child.pid} did not exit after SIGTERM; sending SIGKILL to this group only.", flush=True)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def wait_service_ports_free(args):
    deadline = time.monotonic() + 10
    while True:
        try:
            assert_port_free(args.port)
            assert_port_free(args.timing_port)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Owned service ports are still occupied after cleanup; next backend not started"
                ) from None
            time.sleep(0.2)


def wait_profile(directory, timeout):
    """Wait for exported device timelines before terminating the owned worker."""
    deadline = time.monotonic() + timeout
    next_update = 0
    while True:
        traces = list(directory.rglob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
        if traces and all(p.stat().st_size and (p.parent / "analyse.done").exists() for p in traces):
            print(f"PROFILE SAVED: {directory}; exported timelines={len(traces)}", flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Profile export not confirmed in {directory}; timelines={len(traces)}; "
                f"analyse.done={len(list(directory.rglob('analyse.done')))}; inspect profiler logs there"
            )
        if time.monotonic() >= next_update:
            print(
                f"WAIT: profile export; timelines={len(traces)}; "
                f"analyse.done={len(list(directory.rglob('analyse.done')))}; "
                f"remaining={max(0, deadline - time.monotonic()):.0f}s; directory={directory}",
                flush=True,
            )
            next_update = time.monotonic() + 15
        time.sleep(2)


def finish_profile(opener, args, profile_dir):
    """Bound stop/export waiting independently of already collected timings."""
    deadline = time.monotonic() + args.profile_timeout
    try:
        request(opener, args.port, "/stop_profile", {}, timeout=args.profile_timeout)
        wait_profile(profile_dir, max(0, deadline - time.monotonic()))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"PROFILE INCOMPLETE: {type(exc).__name__}: {exc}", flush=True)
        print(f"Existing profile files retained: {profile_dir}", flush=True)
        if getattr(args, "strict_profile", False):
            raise
        print("Timing data retained. Clean up owned service, then continue remaining backend tests.", flush=True)
        return "incomplete"
    return "complete"


def print_inference_report(backend, report, requests, *, profile_enabled=True):
    if report.get("mstx_enabled"):
        print(f"MSTX {backend}: enabled=True; marker_errors={report.get('mstx_errors', 0)}", flush=True)
    if report["errors"] or report["dropped"] or not report["count"]:
        raise RuntimeError(
            f"Incomplete callback timing: count={report['count']} errors={report['errors']} dropped={report['dropped']}"
        )
    print(f"INFERENCE {backend}: callbacks={report['count']}; errors=0; dropped=0", flush=True)
    for group in report["groups"]:
        print(
            f"\nkind={group['kind']} token_rows={group['token_rows']} callbacks={group['count']} "
            f"layers={group['layers']} loads_per_callback={group['loads']}",
            flush=True,
        )
        print(
            f"{'metric':26} {'mean_us':>10} {'p50_us':>10} {'p95_us':>10} {'p99_us':>10} {'min_us':>10} {'max_us':>10}",
            flush=True,
        )
        for name in METRICS:
            print(
                f"{name:26} "
                + " ".join(
                    f"{group['metrics'][name][key]:10.3f}" for key in ("mean", "p50", "p95", "p99", "min", "max")
                ),
                flush=True,
            )
        gap = group.get("gap_metrics", {})
        if gap:
            measured_sum = sum(item["mean"] for item in gap.values())
            print("GAP BREAKDOWN: ranked by mean; share of measured substeps, not the whole timeline gap", flush=True)
            for name, item in sorted(gap.items(), key=lambda pair: pair[1]["mean"], reverse=True):
                share = 100 * item["mean"] / measured_sum if measured_sum else 0
                print(
                    f"{name:26} mean_us={item['mean']:.3f} p95_us={item['p95']:.3f} "
                    f"max_us={item['max']:.3f} measured_share={share:.1f}%",
                    flush=True,
                )
    stats = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))["summarize"]
    instrumentation = "TIMERS + PROFILER" if profile_enabled else "TIMERS; NO PROFILER"
    print(f"Request wall time (ms, {instrumentation}): {stats([r[0] / 1000 for r in requests])}", flush=True)
    print(f"Request usage: {[r[1] for r in requests]}", flush=True)
    report["request_us"] = statistics.mean(r[0] for r in requests)
    report["request_stats_us"] = stats([r[0] for r in requests])
    report["usage"] = [r[1] for r in requests]
    report["digests"] = [r[2] for r in requests]
    return report


def inference(args, source, backend, env):
    output_dir = getattr(args, "output_dir", None)
    log_path = output_dir / f"inference_{backend}.log" if output_dir else None
    if log_path:
        print(f"Service log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") if log_path else contextlib.nullcontext() as log:
        return _inference(args, source, backend, env, log)


def completion_payloads(args, model):
    prompts = VERIFICATION_PROMPTS if getattr(args, "verification_suite", False) else ("The future of AI is",)
    return [
        dict(model=model, prompt=prompts[index % len(prompts)], max_tokens=args.max_tokens, temperature=0)
        for index in range(args.requests)
    ]


def _inference(args, source, backend, env, log):
    profile_enabled = not getattr(args, "skip_profile", False)
    profile_dir = args.profile_run_dir / backend
    prefix, command, model, _ = launch_spec(
        source, backend, args.npu, args.port, args.timing_port, profile_dir, mstx=getattr(args, "mstx", False)
    )
    if getattr(args, "trace_startup", False):
        command = startup_trace_command(command)
    assert_port_free(args.port)
    assert_port_free(args.timing_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    print(f"\n=== INFERENCE {backend}: fresh service; profile={profile_dir} ===", flush=True)
    print(f"Effective serve command: {shlex.join(command)}", flush=True)
    child = subprocess.Popen(
        bash_command(prefix, command), cwd=REPO, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT
    )
    try:
        with progress(f"{backend} service readiness; timeout={args.startup_timeout}s"):
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Service exited during startup: {child.returncode}")
                try:
                    request(opener, args.port, "/health", timeout=2)
                    status = request(opener, args.timing_port, "/status", timeout=2)
                    if status.get("backend") != backend or status.get("load_backend") != "python":
                        raise RuntimeError(f"Unexpected worker timing identity: {status}")
                    if getattr(args, "mstx", False) and status.get("mstx_enabled") is not True:
                        raise RuntimeError("Worker did not enable MSTX; check source/install paths")
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Service readiness timed out") from None
                    time.sleep(2)
        payloads = completion_payloads(args, model)
        request_specs = [
            dict(
                prompt_sha256=hashlib.sha256(payload["prompt"].encode()).hexdigest(),
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
            )
            for payload in payloads
        ]
        prompt_mode = "verification suite" if getattr(args, "verification_suite", False) else "identical prompt"
        print(
            f"Service ready. Warmup={args.warmup_requests}; measured requests={args.requests}; "
            f"max_tokens={args.max_tokens}; temperature=0; {prompt_mode}; sequential requests.",
            flush=True,
        )
        print("REQUEST_PLAN " + json.dumps(request_specs), flush=True)

        def completion(payload, phase="measured"):
            begin = time.perf_counter_ns()
            result = request(opener, args.port, "/v1/completions", payload, timeout=args.request_timeout)
            elapsed = (time.perf_counter_ns() - begin) / 1000
            if "choices" not in result or "usage" not in result:
                raise RuntimeError("Completion did not include choices/usage")
            if getattr(args, "verification_suite", False) and getattr(args, "output_dir", None):
                with (args.output_dir / "completion_records.jsonl").open("a", encoding="utf-8") as records:
                    records.write(
                        json.dumps(dict(phase=phase, payload=payload, response=result), ensure_ascii=False) + "\n"
                    )
            text = "".join(item.get("text", "") for item in result["choices"])
            return elapsed, result["usage"], hashlib.sha256(text.encode()).hexdigest()

        with progress(f"{backend} warmup: {args.warmup_requests} requests"):
            for index in range(args.warmup_requests):
                completion(payloads[index % len(payloads)], phase="warmup")
        request_window = None
        timing_count = getattr(args, "timing_requests", 0)
        if timing_count:
            if not profile_enabled or not getattr(args, "profile_compare", False):
                raise ValueError("Repeated pre-profile requests require the profile comparison preset")
            if request(opener, args.timing_port, "/status").get("active") is not False:
                raise RuntimeError("Repeated request window requires inactive callback timers")
            # Profiler has not started. Inactive callback wrappers skip both
            # per-stage clocks and MSTX ranges; wrapper checks still remain.
            with progress(f"{backend} no-profiler requests: {timing_count}"):
                samples = [
                    completion(payloads[index % len(payloads)], phase="no_profiler") for index in range(timing_count)
                ]
            if request(opener, args.timing_port, "/status").get("active") is not False:
                raise RuntimeError("Callback timers became active during repeated request window")
            summarize = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))["summarize"]
            durations = [sample[0] for sample in samples]
            request_window = dict(
                count=timing_count,
                instrumentation="profiler_not_started; callback_timers_inactive; wrappers_retained",
                samples_us=durations,
                stats_us=summarize(durations),
                stddev_us=statistics.pstdev(durations),
                request_specs=[request_specs[index % len(request_specs)] for index in range(timing_count)],
                digests=[sample[2] for sample in samples],
                usage=[sample[1] for sample in samples],
            )
            # Persist before profiling: export failures must not lose request measurements.
            (args.output_dir / "request_window.json").write_text(json.dumps(request_window), encoding="utf-8")
        untimed = None
        if getattr(args, "compare_join_cost", False):
            if profile_enabled or getattr(args, "mstx", False):
                raise ValueError("Join cost comparison must disable profiler and MSTX")
            # The collector is inactive after construction. No start/stop of
            # profiler or per-callback timers occurs in this first window.
            # Wrappers remain installed, so this is not an uninstrumented binary.
            if status.get("active") is not False or status.get("mstx_enabled") is not False:
                raise RuntimeError("Cost baseline requires inactive timers and disabled MSTX")
            with progress(f"{backend} timer-inactive requests: {args.requests}"):
                baseline_requests = [completion(payload, phase="timer_inactive") for payload in payloads]
            stats = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))["summarize"]
            untimed = dict(
                count=len(baseline_requests),
                request_us=statistics.mean(item[0] for item in baseline_requests),
                stats_us=stats([item[0] for item in baseline_requests]),
                digests=[item[2] for item in baseline_requests],
                usage=[item[1] for item in baseline_requests],
            )
        if profile_enabled:
            if getattr(args, "profile_compare", False):
                ensure_profile_storage(profile_dir, stage="before start_profile")
            request(opener, args.port, "/start_profile", {}, timeout=args.profile_timeout)
        requests = []
        try:
            request(opener, args.timing_port, "/start", {})
            with progress(f"{backend} measured inference: {args.requests} requests"):
                requests = [completion(payload) for payload in payloads]
        finally:
            try:
                report = request(opener, args.timing_port, "/stop", {}, timeout=40)
                if len(requests) == args.requests:
                    print_inference_report(backend, report, requests, profile_enabled=profile_enabled)
                    baseline = getattr(args, "python_baseline_report", None)
                    if backend == "cpp" and baseline is not None:
                        compare_inference_reports({"python": baseline, "cpp": report})
            finally:
                profile_status = finish_profile(opener, args, profile_dir) if profile_enabled else "skipped"
        report["profile_status"] = profile_status
        report["request_specs"] = request_specs
        if request_window is not None:
            report["request_window"] = request_window
        if untimed is not None:
            report["untimed"] = untimed
        if getattr(args, "mstx", False) and profile_enabled and not getattr(args, "profile_compare", False):
            report["mstx_trace_occurrences"] = inspect_mstx_trace(profile_dir)
            if profile_status == "complete":
                analyze = runpy.run_path(str(REPO / "script/analyze_callback_gaps.py"))["analyze"]
                for index, trace in enumerate(sorted(profile_dir.rglob("trace_view.json"))):
                    try:
                        analyze(trace, args.output_dir / f"trace_with_gaps_{backend}_{index}.json")
                    except (ValueError, OSError) as exc:
                        print(f"GAP ANALYSIS NOT VERIFIED: {exc}", flush=True)
        return report
    finally:
        if profile_enabled or getattr(args, "trace_startup", False) or getattr(args, "compare_join_cost", False):
            error_type = sys.exc_info()[0]
            with contextlib.suppress(Exception):
                startup_cleanup_marker(child, log, error_type.__name__ if error_type else "completed")
        with progress(f"{backend} service cleanup and port release; up to 50s"):
            stop_owned_service(child)
            wait_service_ports_free(args)


def inspect_mstx_trace(profile_dir):
    """Bounded-memory presence check, including partially exported large traces."""
    needle = b"moe_offload/"
    total = 0
    for path in sorted(profile_dir.rglob("trace_view.json")):
        count = 0
        tail = b""
        with path.open("rb") as trace:
            while chunk := trace.read(1024 * 1024):
                data = tail + chunk
                count += data.count(needle)
                tail = data[-(len(needle) - 1) :]
        print(f"MSTX TRACE: {path}; label_occurrences={count}", flush=True)
        total += count
    if not total:
        print("MSTX NOT VERIFIED: no moe_offload/ labels found in trace_view.json; inspect profiler logs.", flush=True)
    else:
        print(
            "MSTX labels found; open the profile in MindStudio to verify nesting and host/device alignment.", flush=True
        )
    return total


def read_python_baseline(path, source, args):
    """Recover only completed Python timing tables, never execute saved text."""
    text = path.read_text(encoding="utf-8")
    fingerprint = re.search(r"Requested NPU=(.+?);.*source_sha256=([0-9a-f]{64})", text)
    expected_npu = str(args.npu) if args.npu is not None else "inherit test_graph2.sh"
    if not fingerprint or fingerprint.groups() != (expected_npu, hashlib.sha256(source.encode()).hexdigest()):
        raise ValueError("Baseline NPU or test_graph2.sh hash differs; use the original device and launch script")
    section = text.split("=== INFERENCE python:", 1)
    if len(section) != 2:
        raise ValueError("No Python inference section in baseline")
    text = section[1].split("=== INFERENCE cpp:", 1)[0]
    settings = re.search(r"Warmup=(\d+); measured requests=(\d+); max_tokens=(\d+);", text)
    if not settings or tuple(map(int, settings.groups())) != (args.warmup_requests, args.requests, args.max_tokens):
        raise ValueError("Baseline request/warmup/max_tokens settings differ")
    count = re.search(r"INFERENCE python: callbacks=(\d+); errors=0; dropped=0", text)
    wall = re.search(r"Request wall time \(ms, TIMERS \+ PROFILER\): (.+)", text)
    usage = re.search(r"Request usage: (.+)", text)
    if not count or not wall or not usage:
        raise ValueError("Python baseline has incomplete or failed callback/request measurements")
    groups = []
    for block in text.split("\nkind=")[1:]:
        header = re.match(
            r"(prefetch|reactive) token_rows=(\d+) callbacks=(\d+) layers=(\[.*?\]) loads_per_callback=(\{[^\n]+\})",
            block,
        )
        if not header:
            raise ValueError("Unrecognized Python callback group")
        kind, rows, calls, layers, loads = header.groups()
        metrics = {}
        for line in block.splitlines()[1:]:
            parts = line.split()
            if len(parts) == 7 and parts[0] in METRICS:
                metrics[parts[0]] = dict(zip(("mean", "p50", "p95", "p99", "min", "max"), map(float, parts[1:])))
        if set(metrics) != set(METRICS):
            raise ValueError("Python callback timing table is incomplete")
        groups.append(
            dict(
                kind=kind,
                token_rows=int(rows),
                count=int(calls),
                layers=ast.literal_eval(layers),
                loads=ast.literal_eval(loads),
                metrics=metrics,
            )
        )
    if not groups or sum(g["count"] for g in groups) != int(count[1]):
        raise ValueError("Python callback counts do not match")
    report = dict(groups=groups, request_us=ast.literal_eval(wall[1])["mean"] * 1000, usage=ast.literal_eval(usage[1]))
    if len(report["usage"]) != args.requests:
        raise ValueError("Python request usage count does not match")
    print(f"PYTHON BASELINE: {path}; callbacks={count[1]}; saved means rounded to 0.001 us", flush=True)
    print("Resume: CPP inference only. Same launch-script hash/NPU/request settings; separate session.", flush=True)
    return report


def compare_inference_reports(reports):
    grouped = {b: {(g["kind"], g["token_rows"]): g for g in reports[b]["groups"]} for b in reports}
    for key in sorted(set(grouped["python"]) | set(grouped["cpp"])):
        if not all(key in grouped[b] for b in grouped):
            print(f"Unmatched callback group {key}: cannot compare", flush=True)
            continue
        a, b = grouped["python"][key], grouped["cpp"][key]
        print(
            f"Group {key}: counts python/cpp={a['count']}/{b['count']}; "
            f"mean loads={a['loads']['mean']:.3f}/{b['loads']['mean']:.3f}",
            flush=True,
        )
        compare(
            f"INFERENCE {key}",
            {k: a["metrics"][k]["mean"] for k in METRICS},
            {k: b["metrics"][k]["mean"] for k in METRICS},
        )
    print(
        f"Request mean ms: python={reports['python']['request_us'] / 1000:.3f} "
        f"cpp={reports['cpp']['request_us'] / 1000:.3f}",
        flush=True,
    )
    hashes_match = (
        reports["python"]["digests"] == reports["cpp"]["digests"]
        if "digests" in reports["python"]
        else "unavailable in saved baseline"
    )
    print(
        f"Request usage matches: {reports['python']['usage'] == reports['cpp']['usage']}; "
        f"generated-text hashes match: {hashes_match}",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, help="Override device visibility; default: inherit test_graph2.sh")
    parser.add_argument("--test-script", type=Path, default=REPO / "script/test_graph2.sh")
    parser.add_argument("--only", choices=("all", "micro", "inference"), default="all")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--mstx", action="store_true", help="Add host function ranges to inference profiles")
    parser.add_argument("--port", type=int, help="Default: inherit test_graph2.sh")
    parser.add_argument("--timing-port", type=int, default=18155)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument(
        "--profile-timeout", type=int, default=120, help="Stop/export waiting budget per backend; default 120 seconds"
    )
    parser.add_argument(
        "--strict-profile",
        action="store_true",
        help="Abort on incomplete profile instead of continuing with saved timings",
    )
    parser.add_argument("--output-dir", type=Path, help="New directory for summary.txt and detailed logs")
    parser.add_argument(
        "--python-baseline", type=Path, help="Resume with CPP inference only; compare saved Python summary"
    )
    args = parser.parse_args(argv)
    if args.mstx and (args.python_baseline is not None or args.only == "micro"):
        parser.error("--mstx requires fresh inference runs; cannot use --python-baseline or --only micro")
    if args.python_baseline is not None:
        args.only = "inference"
    if (args.npu is not None and args.npu < 0) or min(
        args.requests,
        args.warmup_requests,
        args.max_tokens,
        args.startup_timeout,
        args.request_timeout,
        args.profile_timeout,
    ) < 1:
        parser.error("NPU must be >=0; counts and timeouts must be positive")
    if os.name != "posix":
        parser.error("Execute in the Linux NPU container")
    args.output_dir = (
        args.output_dir
        or REPO / "bench_results" / (time.strftime("verified_cpp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    ).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (
        (args.output_dir / "summary.txt").open("w", encoding="utf-8") as summary,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, summary)),
    ):
        print(f"SUMMARY FILE: {args.output_dir / 'summary.txt'}", flush=True)
        try:
            run_tests(args)
        except KeyboardInterrupt:
            print("INTERRUPTED: partial results retained; owned service cleanup requested.", flush=True)
            raise SystemExit(130) from None
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}; partial results retained.", flush=True)
            raise SystemExit(1) from None


def run_tests(args):
    print(f"MSTX host ranges: {getattr(args, 'mstx', False)}; enabled ranges add diagnostic overhead", flush=True)
    source = args.test_script.read_text(encoding="utf-8")
    prefix, words, _, model_config = launch_spec(source, "python", args.npu, args.port, args.timing_port)
    args.port = int(words[words.index("--port") + 1]) if "--port" in words else 8000
    if not 1024 <= args.port <= 65535 or not 1024 <= args.timing_port <= 65535 or args.port == args.timing_port:
        raise ValueError("Use distinct ports in 1024..65535")
    if args.only != "micro":
        if "--profiler-config" not in words:
            raise ValueError("test_graph2.sh must contain --profiler-config")
        profiler = json.loads(words[words.index("--profiler-config") + 1])
        if profiler.get("profiler") != "torch" or not profiler.get("torch_profiler_dir"):
            raise ValueError("test_graph2.sh must configure the torch profiler output directory")
        base = Path(profiler["torch_profiler_dir"])
        if not base.is_absolute():
            base = REPO / base
        args.profile_run_dir = base / (time.strftime("verified_cpp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    if args.python_baseline is not None:
        args.python_baseline_report = read_python_baseline(args.python_baseline, source, args)
    env = os.environ.copy()
    if args.npu is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    py = [sys.executable, "-u", "-B"]
    print("=== VERIFIED CPP COMPARISON: only LRC observe/choose; all load logic stays Python ===", flush=True)
    print(
        f"Requested NPU={args.npu if args.npu is not None else 'inherit test_graph2.sh'}; "
        f"test_script={args.test_script}; source_sha256={hashlib.sha256(source.encode()).hexdigest()}",
        flush=True,
    )
    print("Detailed timers add overhead. No fused planner, ACL experiment or C++ load migration.", flush=True)
    print(
        "Inference inherits test_graph2.sh, including model/MTP/prefetch/debug/profiler settings.",
        flush=True,
    )
    print(
        "Timing summaries: summary.txt and console. Raw logs: output directory. "
        "Profiles: separate python/cpp directories.",
        flush=True,
    )
    print(
        "Inference times include profiler and timer overhead; micro times do not include profiler overhead.", flush=True
    )
    if args.only != "micro":
        assert_port_free(args.port)
        assert_port_free(args.timing_port)
    if not args.skip_build:
        run_console(
            [*py, "-m", "pip", "install", "-v", "./csrc/lrc_policy", "--no-build-isolation", "--no-deps"],
            env,
            args.output_dir / "build.log",
        )
    run_console([*py, "tests/ut/expert_offload/test_lrc_cpp.py", "--require-cpp"], env, args.output_dir / "parity.log")
    if getattr(args, "mstx", False):
        run_console([*py, "tests/ut/expert_offload/test_callback_mstx.py"], env, args.output_dir / "mstx_tests.log")
    if args.only != "inference":
        if not model_config.is_file():
            raise RuntimeError(f"Local model config is required to align micro dimensions: {model_config}")
        results = {"python": [], "cpp": []}
        for index, backend in enumerate(("python", "cpp", "cpp", "python"), 1):
            print(f"\n=== MICRO {backend}: original Python load_submit, detailed timing ===", flush=True)
            command = [
                *py,
                "bench_update_weights.py",
                "--mode",
                "npu",
                "--backend",
                backend,
                "--h2d-backend",
                "torch",
                "--load-backend",
                "python",
                "--detail-load",
                "--test-script",
                str(args.test_script.resolve()),
                "--model-config",
                str(model_config),
            ]
            means = run_console(bash_command(prefix, command), env, args.output_dir / f"micro_{index}_{backend}.log")
            if not all(k in means for k in METRICS):
                raise RuntimeError("Micro timing table is incomplete")
            results[backend].append(means)
        compare(
            "MICRO (two equal-sized runs/backend)",
            *({k: statistics.mean(r[k] for r in results[b]) for k in METRICS} for b in ("python", "cpp")),
        )
    if args.only != "micro":
        backends = ("cpp",) if args.python_baseline is not None else ("python", "cpp")
        reports = {b: inference(args, source, b, env) for b in backends}
        if args.python_baseline is None:
            compare_inference_reports(reports)
        print(f"PROFILE STATUS: { {b: r['profile_status'] for b, r in reports.items()} }", flush=True)
    print("=== VERIFIED CPP TIMING TEST COMPLETE; check PROFILE STATUS separately; not a speedup claim ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; owned inference service cleanup requested.", flush=True)
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FAILED: {exc}; remaining tests stopped.", flush=True)
        raise SystemExit(1) from None
