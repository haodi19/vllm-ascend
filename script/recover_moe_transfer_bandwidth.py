# SPDX-License-Identifier: Apache-2.0
"""Re-export a copied PROF capture with CANN, then inspect recovered transfer records.

Offline export syntax (Ascend 950 supported):
https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0021.html
No application/collection flags are passed. Raw source files are never export targets.
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from inspect_moe_transfer_bandwidth import inventory_capture, read_database, summarize
from moe_experiment_tables import number, table

COPY_CHUNK_BYTES = 1024**2
MAX_SOURCE_FILES = 200000
MIN_FREE_BYTES = 1024**3
EXPORT_TIMEOUT_SECONDS = 1800
HEARTBEAT_SECONDS = 30
MAX_SUMMARY_ROWS = 24
MAX_DATABASES = 8
LOG_TAIL_BYTES = 16384
FORBIDDEN_PATH_CHARS = "\n\f\r\b\t\v\x7f\"'\\%><|&$;`"


class Tee:
    def __init__(self, console, report):
        self.console = console
        self.report = report

    def write(self, text):
        self.console.write(text)
        self.report.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.report.flush()


def source_capture(path):
    path = path.expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("source must be a capture directory or a PROF_* directory")
    candidates = [path] if path.name.startswith("PROF_") else sorted(path.glob("PROF_*"))
    candidates = [p for p in candidates if p.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"expected one raw PROF_* directory, found {len(candidates)}")
    if candidates[0].is_symlink():
        raise ValueError("raw PROF directory must not be a symlink")
    return path, candidates[0]


def find_msprof(explicit):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [Path(shutil.which("msprof"))] if shutil.which("msprof") else []
        candidates += [
            Path("/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof"),
            Path("/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof"),
        ]
        for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
            if os.environ.get(key):
                candidates.append(Path(os.environ[key]) / "tools/profiler/bin/msprof")
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.absolute()
    raise ValueError("msprof not found; load the installed CANN environment or specify --msprof /path/to/msprof")


def fingerprint_tree(root):
    """Fingerprint regular raw files; reject links and incomplete source changes."""
    result = {}
    for folder, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if Path(folder, name).is_symlink():
                raise ValueError("symlink found in raw capture; refusing to follow external files")
        for name in sorted(files):
            path = Path(folder, name)
            if path.is_symlink() or not path.is_file():
                raise ValueError("raw capture contains non-regular files")
            digest = hashlib.sha256()
            before = path.stat()
            with path.open("rb") as source:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("source changed during fingerprinting")
            result[str(path.relative_to(root))] = {"bytes": after.st_size, "sha256": digest.hexdigest()}
            if len(result) > MAX_SOURCE_FILES:
                raise ValueError("too many raw files; refusing partial recovery")
    if not result:
        raise ValueError("empty raw capture")
    return result


def copy_capture(raw, destination):
    original = fingerprint_tree(raw)
    total = sum(record["bytes"] for record in original.values())
    required = max(MIN_FREE_BYTES, total * 4)
    if shutil.disk_usage(destination.parent).free < required:
        raise ValueError(f"insufficient recovery disk space; need at least {required / 1024**3:.2f} GiB")
    shutil.copytree(raw, destination)
    if fingerprint_tree(destination) != original or fingerprint_tree(raw) != original:
        raise ValueError("raw source/copy verification failed; export not started")
    print(f"COPY_VERIFIED files={len(original)}; bytes={total}; 原始采集不作为解析目标", flush=True)
    return original


def export_command(msprof, copied_raw):
    # text exports JSON/CSV and the aggregate DB, and retains intermediate SQLite data.
    if any(c in str(copied_raw) for c in FORBIDDEN_PATH_CHARS):
        raise ValueError("copied path contains characters unsupported by msprof")
    return [str(msprof), "--export=on", f"--output={copied_raw}", "--type=text", "--clear=off"]


def stop_export(process):
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def run_export(msprof, copied_raw, log, timeout):
    command = export_command(msprof, copied_raw)
    print("OFFLINE_EXPORT=" + json.dumps(command, ensure_ascii=False), flush=True)
    started = time.monotonic()
    with log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=copied_raw.parent,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("offline export timed out; copied data and log retained")
                try:
                    return process.wait(timeout=min(HEARTBEAT_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    print(f"WAIT 离线解析; seconds={time.monotonic() - started:.0f}; log={log}", flush=True)
        except BaseException:
            stop_export(process)
            raise


def log_tail(path):
    if path.is_file():
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - LOG_TAIL_BYTES))
            lines = source.read().decode("utf-8", errors="replace").splitlines()
        print("EXPORT_LOG_TAIL（最多12行）:")
        for line in lines[-12:]:
            print(line[:600])


def inspect_exports(copied_raw, output):
    databases = sorted(copied_raw.rglob("msprof_*.db"))
    print(f"EXPORTED_DATABASES={len(databases)}")
    if len(databases) > MAX_DATABASES:
        raise ValueError("too many exported databases; not automatically merging")
    recovered = []
    for path in databases:
        print(f"RECOVERED_DB={path}")
        try:
            values = summarize(read_database(path))
        except (ValueError, sqlite3.Error, OSError) as error:
            print(f"RECOVERED_DB_UNUSABLE: {error}")
            continue
        rows = []
        for key in sorted(values, key=lambda k: (-values[k]["total_bytes"], k)):
            stats = values[key]
            recovered.append(
                {
                    "database": str(path),
                    "device": key[0],
                    "stream": key[1],
                    "direction": key[2],
                    "bytes_per_copy": key[3],
                    **stats,
                }
            )
            rows.append(
                [
                    *key,
                    stats["n"],
                    number(stats["mean_us"]),
                    number(stats["p50_us"]),
                    number(stats["p95_us"]),
                    number(stats["gbps"]),
                ]
            )
        print(
            table(
                ["设备", "DB流", "方向", "字节/次", "次数", "均值us", "P50us", "P95us", "有效GB/s"],
                rows[:MAX_SUMMARY_ROWS],
            )
        )
    h2d_groups = sum(row["direction"] == "H2D" for row in recovered)
    status = "H2D_RECOVERED" if h2d_groups else "NO_USABLE_H2D_RECORDS"
    (output / "recovery_result.json").write_text(
        json.dumps({"status": status, "groups": recovered}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RECOVERY_STATUS={status}; H2D_GROUPS={h2d_groups}")
    if not h2d_groups:
        inventory_capture(copied_raw / "ASCEND_PROFILER_OUTPUT")
        print("本次重导出未恢复可计算H2D的数据；可能未采集所需字段或当前解析器不支持，不能伪造字节数。")
    print("GB/s=总字节数/设备拷贝耗时之和；不含Host、空泡或Transpose，不等于HBM峰值。")
    return 0 if h2d_groups else 2


def recover(source, output, msprof, timeout):
    capture, raw = source_capture(source)
    output = output.expanduser().resolve()
    if output == capture or capture in output.parents or output in capture.parents:
        raise ValueError("recovery output must be separate from the original capture")
    output.mkdir(parents=True, exist_ok=False)
    summary = output / "summary.txt"
    with summary.open("w", encoding="utf-8") as report, contextlib.redirect_stdout(Tee(sys.stdout, report)):
        print(f"SUMMARY_FILE={summary}")
        print(f"SOURCE={raw}; MSPROF={msprof}; timeout={timeout}s")
        print("复制原始采集后离线解析，不启动模型或采集任务；输出全部保留在新目录。")
        log = output / "export.log"
        copied_raw = output / raw.name
        try:
            original = copy_capture(raw, copied_raw)
            (output / "source_manifest.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
            previous = sorted(copied_raw.rglob("msprof_*.db"))
            if previous:
                raise ValueError("raw copy contains prior msprof exports; refusing to label existing DBs as recovered")
            code = run_export(msprof, copied_raw, log, timeout)
            print(f"EXPORT_EXIT={code}")
            if fingerprint_tree(raw) != original:
                raise ValueError("original capture changed during export; results cannot be trusted")
            print("SOURCE_UNCHANGED=True（全部原始文件SHA256复核）")
            log_tail(log)
            if code:
                print("RECOVERY_STATUS=EXPORT_FAILED; 不把解析失败解释为原数据没有字节数")
                return 1
            return inspect_exports(copied_raw, output)
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as error:
            print(f"RECOVERY_FAILED: {type(error).__name__}: {error}")
            log_tail(log)
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--msprof", type=Path)
    parser.add_argument("--timeout", type=int, default=EXPORT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        source = args.source.expanduser().resolve(strict=True)
        output = args.output_dir or source.parent / f"bandwidth_recovery_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        return recover(source, output, find_msprof(args.msprof), args.timeout)
    except (OSError, ValueError) as error:
        print(f"RECOVERY_FAILED: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
