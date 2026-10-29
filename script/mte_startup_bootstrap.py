# SPDX-License-Identifier: Apache-2.0
"""Observe one diagnostic vLLM API process and its multiprocessing children."""

import argparse
import contextlib
import faulthandler
import importlib
import json
import multiprocessing
import multiprocessing.process
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

PREFIX = "[MOE-MTE PROCESS]"
EXIT_RECORDED_ATTRIBUTE = "_moe_startup_exit_recorded"
PROCESS_NAME_ATTRIBUTE = "_moe_startup_process_name"
MAX_SHEBANG_BYTES = 4096


def validate_vllm_interpreter():
    """Require the same Python executable as the original PATH console entry."""
    entrypoint = shutil.which("vllm")
    if entrypoint is None:
        raise RuntimeError("startup diagnostic cannot verify interpreter: PATH has no vllm entrypoint")
    entrypoint_path = Path(entrypoint)
    with entrypoint_path.open("rb") as source:
        first_line = source.readline(MAX_SHEBANG_BYTES + 1)
    if len(first_line) > MAX_SHEBANG_BYTES or not first_line.startswith(b"#!"):
        raise RuntimeError("startup diagnostic requires a recognized Python vllm console-script shebang")
    try:
        declaration = shlex.split(first_line[2:].decode("utf-8").strip())
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("startup diagnostic cannot parse the vllm interpreter shebang") from error
    if not declaration:
        raise RuntimeError("startup diagnostic found an empty vllm interpreter shebang")
    declared = declaration[0]
    if declared in ("/usr/bin/env", "/bin/env"):
        arguments = declaration[1:]
        if arguments[:1] == ["-S"]:
            arguments = arguments[1:]
        if len(arguments) != 1 or not arguments[0].startswith("python") or "/" in arguments[0]:
            raise RuntimeError("startup diagnostic supports only env python... without interpreter flags")
        declared = shutil.which(arguments[0])
        if declared is None:
            raise RuntimeError("startup diagnostic cannot resolve the env-declared Python interpreter")
    elif len(declaration) != 1 or not Path(declared).is_absolute() or not Path(declared).name.startswith("python"):
        raise RuntimeError("startup diagnostic requires an absolute Python shebang without interpreter flags")
    declared_path = Path(declared).absolute()
    expected = declared_path.resolve(strict=True)
    actual = Path(sys.executable).resolve(strict=True)
    if expected != actual:
        raise RuntimeError(f"startup diagnostic interpreter mismatch: vllm={expected}; bootstrap={actual}")
    # Virtual environments may symlink to the very same base executable while
    # selecting different site-packages. Equality of real paths alone is not enough.
    declared_prefix = declared_path.parent.parent
    current_prefix = Path(sys.prefix).resolve(strict=True)
    if (declared_prefix / "pyvenv.cfg").is_file():
        if declared_prefix.resolve(strict=True) != current_prefix:
            raise RuntimeError("startup diagnostic virtual-environment prefix does not match the vllm entrypoint")
    elif current_prefix != Path(sys.base_prefix).resolve(strict=True):
        raise RuntimeError("startup diagnostic cannot verify the vllm entrypoint's virtual-environment prefix")
    return {
        "python_executable": str(actual),
        "python_version": sys.version.split()[0],
        "python_prefix": str(current_prefix),
        "vllm_entrypoint": str(entrypoint_path.absolute()),
        "declared_interpreter": str(declared_path),
    }


def emit_record(record):
    """Logging must not change a child operation's return value or exception."""
    # A closed diagnostic output is not a reason to interrupt process cleanup.
    with contextlib.suppress(OSError, ValueError):
        print(PREFIX + " " + json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)


def popen_observation_points():
    """Find the implementation owners; POSIX spawn inherits fork's poll."""
    points = []
    methods = multiprocessing.get_all_start_methods()
    for method in methods:
        suffix = "spawn_win32" if os.name == "nt" and method == "spawn" else method
        if os.name != "nt" and method == "spawn":
            suffix = "spawn_posix"
        implementation = importlib.import_module(f"multiprocessing.popen_{suffix}").Popen
        names = ("poll", "wait") if os.name == "nt" else ("poll",)
        for name in names:
            owner = next(cls for cls in implementation.__mro__ if name in cls.__dict__)
            point = (owner, name)
            if point not in points:
                points.append(point)
    return methods, points


class ProcessObserver:
    """Observe existing calls only; never initiate polling, waiting or signaling."""

    def __init__(self, emit=emit_record):
        self.owner_pid = os.getpid()
        self.emit = emit
        self.restorations = []
        self.poll_wrappers = []

    def record(self, event, **fields):
        # Forked children inherit hooks, but this observer belongs to the API parent.
        if os.getpid() != self.owner_pid:
            return
        # Observers may still run during interpreter cleanup or through a custom
        # stderr proxy. Only diagnostic failures are isolated, never child calls.
        with contextlib.suppress(Exception):
            self.emit(
                dict(
                    event=event,
                    observer_pid=self.owner_pid,
                    wall_time_ns=time.time_ns(),
                    monotonic_ns=time.monotonic_ns(),
                    **fields,
                )
            )

    def observed_exit(self, popen, returncode):
        if os.getpid() != self.owner_pid or returncode is None or getattr(popen, EXIT_RECORDED_ATTRIBUTE, False):
            return
        setattr(popen, EXIT_RECORDED_ATTRIBUTE, True)
        signal_number = -returncode if returncode < 0 else None
        try:
            signal_name = signal.Signals(signal_number).name if signal_number is not None else None
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
        self.record(
            "exit",
            child_pid=popen.pid,
            name=getattr(popen, PROCESS_NAME_ATTRIBUTE, None),
            start_method=getattr(popen, "method", None),
            returncode=returncode,
            signal=signal_name,
            signal_number=signal_number,
        )

    def replace(self, owner, name, wrapper):
        original = getattr(owner, name)
        self.restorations.append((owner, name, original))
        setattr(owner, name, wrapper)

    def wrap_observation(self, original):
        def observe(popen, *args, **kwargs):
            result = original(popen, *args, **kwargs)
            self.observed_exit(popen, result)
            return result

        return observe

    def wrap_start(self, original):
        def start(process, *args, **kwargs):
            result = original(process, *args, **kwargs)
            if os.getpid() == self.owner_pid:
                popen = process._popen
                setattr(popen, PROCESS_NAME_ATTRIBUTE, process.name)
                self.record(
                    "start", child_pid=popen.pid, name=process.name, start_method=getattr(popen, "method", None)
                )
                if type(popen).poll not in self.poll_wrappers:
                    self.record(
                        "observer_gap",
                        child_pid=popen.pid,
                        reason="custom_popen_poll",
                        popen_type=f"{type(popen).__module__}.{type(popen).__qualname__}",
                    )
            return result

        return start

    def wrap_signal_request(self, original, method, signal_name):
        def request(process, *args, **kwargs):
            popen = process._popen
            self.record(
                "signal_request",
                child_pid=popen.pid if popen is not None else None,
                name=process.name,
                method=method,
                signal=signal_name,
            )
            return original(process, *args, **kwargs)

        return request

    def install(self, interpreter=None):
        if self.restorations:
            raise RuntimeError("process observer is already installed")
        methods, points = popen_observation_points()
        try:
            for owner, name in points:
                wrapper = self.wrap_observation(getattr(owner, name))
                self.replace(owner, name, wrapper)
                if name == "poll":
                    self.poll_wrappers.append(wrapper)
            base = multiprocessing.process.BaseProcess
            self.replace(base, "start", self.wrap_start(base.start))
            for method, signal_name in (("terminate", "SIGTERM"), ("kill", "SIGKILL")):
                self.replace(base, method, self.wrap_signal_request(getattr(base, method), method, signal_name))
        except BaseException:
            self.close()
            raise
        self.record(
            "armed",
            supported_start_methods=methods,
            observation_points=[f"{owner.__module__}.{owner.__qualname__}.{name}" for owner, name in points],
            scope="API-owned stdlib multiprocessing children; existing calls only",
            signal_scope="BaseProcess terminate/kill requests; direct os.kill and external signals are not observed",
            **(interpreter or {}),
        )
        return self

    def close(self):
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)
        self.restorations.clear()
        self.poll_wrappers.clear()


def run_vllm(arguments):
    # Keep the imported module side-effect free for spawn, tests and ordinary users.
    from vllm.entrypoints.cli.main import main as vllm_main

    original_argv = sys.argv
    sys.argv = ["vllm", *arguments]
    try:
        return vllm_main()
    finally:
        sys.argv = original_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true", required=True)
    parser.add_argument("vllm_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.vllm_arguments or args.vllm_arguments[0] != "serve":
        parser.error("this diagnostic entry supports only 'serve ...'")
    interpreter = validate_vllm_interpreter()
    faulthandler.enable(file=2, all_threads=True)
    observer = ProcessObserver().install(interpreter=interpreter)
    observer.record("faulthandler_armed", role="api_parent")
    # Do not restore hooks when main returns: multiprocessing may reap children
    # during interpreter atexit cleanup. This entry owns only this one API process.
    try:
        return run_vllm(args.vllm_arguments)
    except BaseException as error:
        observer.record("entry_exception", exception_type=type(error).__name__)
        raise


if __name__ == "__main__":
    sys.exit(main())
