# SPDX-License-Identifier: Apache-2.0
"""Background NUMA-gated token5/token128 runs with scoped failed-run cleanup."""

import argparse
import contextlib
import ctypes
import errno
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from moe_host_affinity import resolve_host_cpus
from summarize_native_moe_results import rebuild

if sys.platform == "linux":
    import fcntl

ROOT = Path(__file__).resolve().parents[1]
FREE_MB_THRESHOLD = 100000
POLL_SECONDS = 30
TOKEN_COUNTS = (5, 128)
OWNER_FILE = ".memory_retry_owner.json"
PR_SET_CHILD_SUBREAPER = 36
FAILURE_TAIL_BYTES = 16384


class ProcessCleanupError(Exception):
    """Cleanup could not be verified; preserve files and do not start a retry."""


def report(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def cpu_list(text):
    values = set()
    for part in text.strip().split(","):
        first, *last = part.split("-")
        values.update(range(int(first), int(last[0] if last else first) + 1))
    return values


def parse_nodes(text):
    nodes = {}
    for line in text.splitlines():
        free = re.fullmatch(r"node (\d+) free:\s*([0-9.]+) MB", line.strip())
        cpus = re.fullmatch(r"node (\d+) cpus:\s*([0-9 ]*)", line.strip())
        if free:
            nodes.setdefault(int(free[1]), {})["free_mb"] = float(free[2])
        if cpus:
            nodes.setdefault(int(cpus[1]), {})["cpus"] = set(map(int, cpus[2].split()))
    if not nodes or any("free_mb" not in node or "cpus" not in node for node in nodes.values()):
        raise ValueError("numactl -H did not provide complete node free-memory/CPU information")
    return nodes


def choose_node(nodes, allowed_mems, allowed_cpus):
    for node, info in sorted(nodes.items(), key=lambda item: (-item[1]["free_mb"], item[0])):
        cpus = sorted(info["cpus"] & allowed_cpus)
        if node not in allowed_mems or info["free_mb"] <= FREE_MB_THRESHOLD or not cpus:
            continue
        try:
            workers = resolve_host_cpus("auto", allowed=cpus)
        except ValueError as exc:
            report(f"NODE_SKIP node={node}; {exc}")
            continue
        return dict(node=node, free_mb=info["free_mb"], cpus=cpus, workers=workers)
    return None


def read_available_node():
    env = dict(os.environ, LC_ALL="C")
    result = subprocess.run(["numactl", "-H"], env=env, capture_output=True, text=True, timeout=15, check=True)
    nodes = parse_nodes(result.stdout)
    status = Path("/proc/self/status").read_text()
    match = re.search(r"(?m)^Mems_allowed_list:\s*(.+)$", status)
    if match is None:
        raise RuntimeError("Cannot read this process's allowed NUMA nodes")
    mems, cpus = cpu_list(match[1]), os.sched_getaffinity(0)
    report("MEMORY " + "; ".join(f"node{n}={v['free_mb']:.0f}MB" for n, v in sorted(nodes.items())))
    return choose_node(nodes, mems, cpus)


def verify_root(root, owner):
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise RuntimeError("Retry root changed; refusing cleanup")
    if json.loads((root / OWNER_FILE).read_text())["owner"] != owner:
        raise RuntimeError("Retry root ownership mismatch; refusing cleanup")


def clear_failed_token(root, owner, tokens):
    """Only this new run's exact token directory may be removed."""
    verify_root(root, owner)
    if tokens not in TOKEN_COUNTS:
        raise ValueError("Unexpected token directory")
    target = root / f"token{tokens}"
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or target.resolve(strict=True) != target or not target.is_dir():
        raise RuntimeError(f"Refusing non-directory/symlink cleanup: {target}")
    # rmtree does not follow symlinks. Refuse actual mounts, including descendants.
    for directory, children, _ in os.walk(target, followlinks=False):
        if os.path.ismount(directory) or any(os.path.ismount(Path(directory) / name) for name in children):
            raise RuntimeError(f"Refusing mounted directory cleanup: {directory}")
    shutil.rmtree(target)
    report(f"CLEARED {target}; other token results retained")


def save_failure_tail(root, tokens, reason):
    """Keep one bounded diagnostic file, not an accumulating set of profiles."""
    target = root / f"token{tokens}"
    lines = [f"token={tokens}; {reason}"]
    paths = [target / "run_details.log", *sorted(target.glob("*/inference.log"))]
    for path in paths:
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - FAILURE_TAIL_BYTES))
                lines.extend([f"FILE={path.name}", source.read().decode("utf-8", errors="replace")])
    (root / "last_failure.log").write_text("\n".join(lines), encoding="utf-8")


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]  # starttime: stat field 22, fields starts at field 3
    except (OSError, IndexError):
        return None


def descendants():
    """Only descendants of this supervisor, including adopted orphan services."""
    found = {}
    todo = [os.getpid()]
    while todo:
        parent = todo.pop()
        for path in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children = list(map(int, path.read_text().split()))
            except OSError:
                continue
            for child in children:
                if child not in found:
                    identity = process_identity(child)
                    if identity is not None:
                        found[child] = identity
                        todo.append(child)
    return found


def signal_owned(pid, identity, sig):
    if process_identity(pid) != identity:
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def reap_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def clean_owned_processes():
    """Run after the runner exits; never overlap a retry with its old service."""
    started = time.monotonic()
    announced = False
    while True:
        reap_children()
        children = descendants()
        if not children:
            return
        if not announced:
            report(f"CLEANUP owned descendants={sorted(children)}")
            announced = True
        sig = signal.SIGTERM if time.monotonic() - started < 10 else signal.SIGKILL
        for pid, identity in children.items():
            signal_owned(pid, identity, sig)
        # Includes uninterruptible tasks: no deletion/retry until they actually exit.
        if time.monotonic() - started >= 30:
            report(f"CLEANUP_WAIT owned descendants={sorted(children)}; next run not started")
            time.sleep(POLL_SECONDS)
        else:
            time.sleep(0.2)


def attempt_command(args, tokens, node):
    return [
        "numactl",
        "--membind=" + str(node["node"]),
        "--physcpubind=" + ",".join(map(str, node["cpus"])),
        sys.executable,
        "-u",
        "-B",
        str(ROOT / "script/run_native_moe_compare.py"),
        "--case",
        args.case,
        "--npu",
        str(args.npu),
        "--port",
        str(args.port),
        "--timing-port",
        str(args.timing_port),
        "--host-cpus",
        ",".join(map(str, node["workers"])),
        "--max-tokens",
        str(tokens),
        "--warmup-requests",
        "5",
        "--requests",
        "20",
        "--startup-timeout",
        "1800",
        "--request-timeout",
        "600",
        "--profile-timeout",
        "600",
        "--bubble-stream-name",
        "auto",
        "--output-dir",
        str(args.output_root / f"token{tokens}"),
    ]


def run_attempt(args, tokens, node, stop):
    process = subprocess.Popen(attempt_command(args, tokens, node), cwd=ROOT, start_new_session=True)
    try:
        while process.poll() is None:
            if stop.wait(1):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        return process.wait()
    finally:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            clean_owned_processes()
        except Exception as exc:
            raise ProcessCleanupError("Cannot confirm owned processes exited; files retained") from exc


def verify_success(root, tokens, case):
    directory = root / f"token{tokens}"
    status = json.loads((directory / case / "run_status.json").read_text())
    timing = json.loads((directory / case / "timing_result.json").read_text())
    samples = timing["stream_samples"]
    if (
        status["exit_code"] != 0
        or timing.get("output_match") is not True
        or len(samples) != 20
        or any(sample["completion_tokens"] != tokens for sample in samples)
        or not (directory / "summary.txt").is_file()
    ):
        raise RuntimeError("Runner exit=0 but saved success artifacts are incomplete")


def retry_loop(args, stop):
    root = args.output_root
    state = dict(completed=[], attempts={str(t): 0 for t in TOKEN_COUNTS}, status="waiting")
    for tokens in TOKEN_COUNTS:
        while not stop.is_set():
            verify_root(root, args.owner)
            state.update(status="waiting", max_tokens=tokens)
            (root / "retry_state.json").write_text(json.dumps(state), encoding="utf-8")
            try:
                node = read_available_node()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                report(f"MEMORY_CHECK_FAILED {exc}; retry in {POLL_SECONDS}s")
                stop.wait(POLL_SECONDS)
                continue
            if node is None:
                report(f"WAIT token{tokens}; need node free > {FREE_MB_THRESHOLD} MB; poll in {POLL_SECONDS}s")
                stop.wait(POLL_SECONDS)
                continue
            # No arbitrary existing results may be overwritten, even within this root.
            target = root / f"token{tokens}"
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Unexpected pre-existing attempt directory: {target}")
            state["attempts"][str(tokens)] += 1
            state.update(status="running", node=node["node"], host_cpus=node["workers"])
            (root / "retry_state.json").write_text(json.dumps(state), encoding="utf-8")
            report(
                f"START token{tokens}; attempt={state['attempts'][str(tokens)]}; "
                f"node={node['node']}; free={node['free_mb']:.0f}MB; host_cpus={node['workers']}"
            )
            try:
                code = run_attempt(args, tokens, node, stop)
                if stop.is_set():
                    report("STOPPED; partial results retained; no further retries")
                    return 130
                if code != 0:
                    raise RuntimeError(f"runner exit={code}")
                verify_success(root, tokens, args.case)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                report(f"ATTEMPT_FAILED token{tokens}: {exc}")
                # A full filesystem must not prevent reclaiming this failed attempt.
                with contextlib.suppress(OSError):
                    save_failure_tail(root, tokens, str(exc))
                clear_failed_token(root, args.owner, tokens)
                rebuild(root)
                stop.wait(POLL_SECONDS)
                continue
            state["completed"].append(tokens)
            state["status"] = "complete" if tokens == TOKEN_COUNTS[-1] else "waiting"
            (root / "retry_state.json").write_text(json.dumps(state), encoding="utf-8")
            rebuild(root)
            report(f"SUCCESS token{tokens}; retained={state['completed']}")
            break
        else:
            report("STOPPED while waiting for memory")
            return 130
    report(f"ALL_SUCCESS token5 + token128; SUMMARY={root / 'summary.txt'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int, default=6)
    parser.add_argument(
        "--case", choices=("python", "native", "native_full", "native_pipeline"), default="native_pipeline"
    )
    parser.add_argument("--port", type=int, default=8156)
    parser.add_argument("--timing-port", type=int, default=18156)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--owner", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux" or not shutil.which("numactl"):
        parser.error("Run inside the Linux container with numactl and the project env.sh active")
    if (
        args.npu < 0
        or not 1 <= args.port <= 65535
        or not 1 <= args.timing_port <= 65535
        or args.port == args.timing_port
    ):
        parser.error("Invalid NPU/ports")
    if args.output_root.is_symlink():
        parser.error("Output root must not be a symlink")
    args.output_root = args.output_root.resolve()
    if not args.worker:
        args.output_root.mkdir(parents=True, exist_ok=False)
        owner = uuid.uuid4().hex
        (args.output_root / OWNER_FILE).write_text(json.dumps(dict(owner=owner)), encoding="utf-8")
        command = [
            sys.executable,
            "-u",
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--owner",
            owner,
            "--npu",
            str(args.npu),
            "--case",
            args.case,
            "--port",
            str(args.port),
            "--timing-port",
            str(args.timing_port),
            "--output-root",
            str(args.output_root),
        ]
        with (args.output_root / "run.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (args.output_root / "background.pid").write_text(str(child.pid) + "\n")
        print(
            f"BACKGROUND_PID={child.pid}\nRESULTS={args.output_root}\n"
            f"LIVE: tail -f '{args.output_root / 'run.log'}'\nSTOP: kill -TERM {child.pid}",
            flush=True,
        )
        return 0
    verify_root(args.output_root, args.owner)
    with (args.output_root / ".memory_retry.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error or errno.EINVAL, "Cannot own orphan descendants; no experiment started")
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda _sig, _frame: stop.set())
        report(
            f"RETRY_BEGIN case={args.case}; physical_npu={args.npu}; free > {FREE_MB_THRESHOLD} MB; "
            f"token5 then token128; NUMA memory+CPU binding; poll={POLL_SECONDS}s"
        )
        return retry_loop(args, stop)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        report(f"RETRY_STOPPED {type(error).__name__}: {error}; no unrelated data removed")
        raise SystemExit(1) from error
