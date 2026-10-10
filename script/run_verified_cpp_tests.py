#!/usr/bin/env python3
"""Python vs verified C++ LRC: console timing tables and saved inference profiles."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import runpy
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
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


def launch_spec(source, backend, npu, port, timing_port, profile_dir=None):
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


def run_console(command, env):
    """Print child output as it arrives; retain only numeric metric means."""
    means = {}
    with subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            parts = line.split()
            if len(parts) == 7 and parts[0] in METRICS:
                with contextlib.suppress(ValueError):
                    means[parts[0]] = float(parts[1])
        if child.wait():
            raise RuntimeError(f"Stage failed (exit={child.returncode}); remaining tests stopped")
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
    with opener.open(req, timeout=timeout) as response:
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
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def wait_profile(directory, timeout):
    """Wait for exported device timelines before terminating the owned worker."""
    deadline = time.monotonic() + timeout
    while True:
        traces = list(directory.rglob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
        if traces and all(p.stat().st_size and (p.parent / "analyse.done").exists() for p in traces):
            print(f"PROFILE SAVED: {directory}; exported timelines={len(traces)}", flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Profile export not confirmed in {directory}; inspect profiler logs there")
        time.sleep(2)


def inference(args, source, backend, env):
    profile_dir = args.profile_run_dir / backend
    prefix, command, model, _ = launch_spec(source, backend, args.npu, args.port, args.timing_port, profile_dir)
    assert_port_free(args.port)
    assert_port_free(args.timing_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    print(f"\n=== INFERENCE {backend}: fresh service; profile={profile_dir} ===", flush=True)
    print(f"Effective serve command: {shlex.join(command)}", flush=True)
    child = subprocess.Popen(bash_command(prefix, command), cwd=REPO, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"Service exited during startup: {child.returncode}")
            try:
                request(opener, args.port, "/health", timeout=2)
                status = request(opener, args.timing_port, "/status", timeout=2)
                if status.get("backend") != backend or status.get("load_backend") != "python":
                    raise RuntimeError(f"Unexpected worker timing identity: {status}")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Service readiness timed out") from None
                time.sleep(2)
        payload = dict(model=model, prompt="The future of AI is", max_tokens=args.max_tokens, temperature=0)
        print(
            f"Service ready. Warmup={args.warmup_requests}; measured requests={args.requests}; "
            f"max_tokens={args.max_tokens}; temperature=0; identical prompt; sequential requests.",
            flush=True,
        )

        def completion():
            begin = time.perf_counter_ns()
            result = request(opener, args.port, "/v1/completions", payload, timeout=args.request_timeout)
            elapsed = (time.perf_counter_ns() - begin) / 1000
            if "choices" not in result or "usage" not in result:
                raise RuntimeError("Completion did not include choices/usage")
            text = "".join(item.get("text", "") for item in result["choices"])
            return elapsed, result["usage"], hashlib.sha256(text.encode()).hexdigest()

        for _ in range(args.warmup_requests):
            completion()
        request(opener, args.port, "/start_profile", {}, timeout=args.profile_timeout)
        try:
            request(opener, args.timing_port, "/start", {})
            requests = [completion() for _ in range(args.requests)]
        finally:
            try:
                report = request(opener, args.timing_port, "/stop", {}, timeout=40)
            finally:
                request(opener, args.port, "/stop_profile", {}, timeout=args.profile_timeout)
                wait_profile(profile_dir, args.profile_timeout)
        if report["errors"] or report["dropped"] or not report["count"]:
            raise RuntimeError(
                f"Incomplete callback timing: count={report['count']} "
                f"errors={report['errors']} dropped={report['dropped']}"
            )
        print(f"INFERENCE {backend}: callbacks={report['count']}; errors=0; dropped=0", flush=True)
        for group in report["groups"]:
            print(
                f"\nkind={group['kind']} token_rows={group['token_rows']} callbacks={group['count']} "
                f"layers={group['layers']} loads_per_callback={group['loads']}",
                flush=True,
            )
            print(
                f"{'metric':26} {'mean_us':>10} {'p50_us':>10} {'p95_us':>10} "
                f"{'p99_us':>10} {'min_us':>10} {'max_us':>10}",
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
        stats = runpy.run_path(str(REPO / "vllm_ascend/expert_offload/callback_timing.py"))["summarize"]
        print(f"Request wall time (ms, TIMERS + PROFILER): {stats([r[0] / 1000 for r in requests])}", flush=True)
        print(f"Request usage: {[r[1] for r in requests]}", flush=True)
        report["request_us"] = statistics.mean(r[0] for r in requests)
        report["usage"] = [r[1] for r in requests]
        report["digests"] = [r[2] for r in requests]
        return report
    finally:
        stop_owned_service(child)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, help="Override device visibility; default: inherit test_graph2.sh")
    parser.add_argument("--test-script", type=Path, default=REPO / "script/test_graph2.sh")
    parser.add_argument("--only", choices=("all", "micro", "inference"), default="all")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--port", type=int, help="Default: inherit test_graph2.sh")
    parser.add_argument("--timing-port", type=int, default=18155)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--profile-timeout", type=int, default=600)
    args = parser.parse_args(argv)
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
    source = args.test_script.read_text(encoding="utf-8")
    prefix, words, _, model_config = launch_spec(source, "python", args.npu, args.port, args.timing_port)
    args.port = int(words[words.index("--port") + 1]) if "--port" in words else 8000
    if not 1024 <= args.port <= 65535 or not 1024 <= args.timing_port <= 65535 or args.port == args.timing_port:
        parser.error("Use distinct ports in 1024..65535")
    if args.only != "micro":
        if "--profiler-config" not in words:
            parser.error("test_graph2.sh must contain --profiler-config")
        profiler = json.loads(words[words.index("--profiler-config") + 1])
        if profiler.get("profiler") != "torch" or not profiler.get("torch_profiler_dir"):
            parser.error("test_graph2.sh must configure the torch profiler output directory")
        base = Path(profiler["torch_profiler_dir"])
        if not base.is_absolute():
            base = REPO / base
        args.profile_run_dir = base / (time.strftime("verified_cpp_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    env = os.environ.copy()
    if args.npu is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    py = [sys.executable, "-u", "-B"]
    print("=== VERIFIED CPP COMPARISON: only LRC observe/choose; all load logic stays Python ===", flush=True)
    print("Detailed timers add overhead. No fused planner, ACL experiment or C++ load migration.", flush=True)
    print(
        "Inference inherits test_graph2.sh, including model/MTP/prefetch/debug/profiler settings.",
        flush=True,
    )
    print(
        "Timing summaries: console only. Profiles: separate python/cpp directories. No Git or host sysctl changes.",
        flush=True,
    )
    print(
        "Inference times include profiler and timer overhead; micro times do not include profiler overhead.", flush=True
    )
    if args.only != "micro":
        assert_port_free(args.port)
        assert_port_free(args.timing_port)
    if not args.skip_build:
        run_console([*py, "-m", "pip", "install", "-v", "./csrc/lrc_policy", "--no-build-isolation", "--no-deps"], env)
    run_console([*py, "tests/ut/expert_offload/test_lrc_cpp.py", "--require-cpp"], env)
    if args.only != "inference":
        if not model_config.is_file():
            raise RuntimeError(f"Local model config is required to align micro dimensions: {model_config}")
        results = {"python": [], "cpp": []}
        for backend in ("python", "cpp", "cpp", "python"):
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
            means = run_console(bash_command(prefix, command), env)
            if not all(k in means for k in METRICS):
                raise RuntimeError("Micro timing table is incomplete")
            results[backend].append(means)
        compare(
            "MICRO (two equal-sized runs/backend)",
            *({k: statistics.mean(r[k] for r in results[b]) for k in METRICS} for b in ("python", "cpp")),
        )
    if args.only != "micro":
        reports = {b: inference(args, source, b, env) for b in ("python", "cpp")}
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
        print(
            f"Request usage matches: {reports['python']['usage'] == reports['cpp']['usage']}; "
            f"generated-text hashes match: {reports['python']['digests'] == reports['cpp']['digests']}",
            flush=True,
        )
    print("=== VERIFIED CPP TEST COMPLETE; success is not a speedup claim ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; owned inference service cleanup requested.", flush=True)
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FAILED: {exc}; remaining tests stopped.", flush=True)
        raise SystemExit(1) from None
