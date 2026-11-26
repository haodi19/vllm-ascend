# SPDX-License-Identifier: Apache-2.0
"""Reuse validated native binaries independently of experiment result directories."""

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import sys
import sysconfig
from pathlib import Path

BUILD_COMPONENTS = ("lrc_policy", "host_sync", "host_sync_device")
INSTALLED_MODULES = ("_vllm_ascend_lrc", "_vllm_ascend_host_sync", "vllm_ascend_host_sync")
IGNORED_DIRECTORIES = frozenset(("build", "dist", "__pycache__", ".git"))
SOURCE_SUFFIXES = frozenset(
    (".py", ".toml", ".cfg", ".cpp", ".h", ".hpp", ".c", ".cc", ".cxx", ".cmake", ".txt", ".in")
)
CACHE_SCHEMA = 1
CACHE_FILENAME = "verified_build.json"


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_stamp(path):
    if not path:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        return dict(path=str(path), exists=False)
    info = path.stat()
    return dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns)


def build_environment(cann_root, soc, env):
    versions = {}
    for package in ("torch", "torch-npu", "pybind11", "setuptools", "pip", "cmake", "ninja"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cann_root = cann_root.resolve(strict=True)
    cann_files = (
        "version.info",
        "ascend_toolkit_install.info",
        "lib64/libascendcl.so",
        "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
        "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
        "ascendc_devkit/tikcpp/samples/cmake/ascendc.cmake",
        f"{platform.machine()}-linux/ccec_compiler/bin/bisheng",
        "compiler/ccec_compiler/bin/bisheng",
    )
    # These are existing compiler/toolchain variables, not new runtime switches.
    build_env = {
        key: env.get(key)
        for key in (
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
            "CMAKE_GENERATOR",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "LD_LIBRARY_PATH",
        )
    }
    tools = {name: file_stamp(shutil.which(name, path=env.get("PATH"))) for name in ("c++", "cmake", "ninja")}
    for key in ("CC", "CXX"):
        words = shlex.split(env.get(key, ""))
        tools[key] = file_stamp(shutil.which(words[0], path=env.get("PATH"))) if words else None
    return dict(
        python=file_stamp(sys.executable),
        prefix=sys.prefix,
        version=sys.version,
        abi=sysconfig.get_config_var("SOABI"),
        machine=platform.machine(),
        system=platform.system(),
        soc=soc,
        cann_root=str(cann_root),
        versions=versions,
        tools=tools,
        cann={name: file_stamp(cann_root / name) for name in cann_files},
        driver=file_stamp("/usr/local/Ascend/driver/version.info"),
        build_env=build_env,
    )


def build_fingerprint(repo, cann_root, soc, env):
    sources = {}
    for component in BUILD_COMPONENTS:
        directory = repo / "csrc" / component
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for parent, dirs, files in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in IGNORED_DIRECTORIES and not name.endswith(".egg-info")]
            for name in files:
                path = Path(parent) / name
                if path.suffix in SOURCE_SUFFIXES:
                    sources[path.relative_to(repo).as_posix()] = file_digest(path)
    for name in ("native_moe_build.py", "run_native_moe_compare.py", "run_host_sync_stage2.py"):
        sources[f"script/{name}"] = file_digest(repo / "script" / name)
    payload = dict(
        schema=CACHE_SCHEMA,
        repo=str(repo.resolve()),
        sources=sources,
        environment=build_environment(cann_root, soc, env),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def installed_artifacts(library):
    importlib.invalidate_caches()
    paths = {"handshake": library.resolve(strict=True)}
    for name in INSTALLED_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise FileNotFoundError(f"Installed native artifact missing: {name}")
        paths[name] = Path(spec.origin).resolve(strict=True)
    return {name: dict(path=str(path), sha256=file_digest(path)) for name, path in paths.items()}


def cached_library(parent, fingerprint):
    try:
        record = json.loads((parent / CACHE_FILENAME).read_text(encoding="utf-8"))
        if record["schema"] != CACHE_SCHEMA or record["fingerprint"] != fingerprint:
            return None
        library = Path(record["library"]).resolve(strict=True)
        library.relative_to(parent.resolve())
        if installed_artifacts(library) != record["artifacts"]:
            return None
        return library
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return None


def remember_build(parent, fingerprint, library):
    record = dict(
        schema=CACHE_SCHEMA,
        fingerprint=fingerprint,
        library=str(library.resolve()),
        artifacts=installed_artifacts(library),
    )
    temporary = parent / f".verified_build.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(parent / CACHE_FILENAME)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def build_lock(parent):
    # This runner is Linux-only; defer the platform-specific import for CPU tests.
    import fcntl

    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".build.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another native build/preflight is using this build directory; retry after it exits"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
