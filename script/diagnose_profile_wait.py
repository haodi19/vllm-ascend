#!/usr/bin/env python3
"""Read-only snapshot of a verified comparison waiting for profile export."""

import argparse
import datetime
import re
import shutil
from collections import deque
from pathlib import Path


def tail(path, count):
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            return "".join(deque(stream, maxlen=count))
    except OSError as exc:
        return f"Cannot read {path}: {exc}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing this run's summary.txt")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    summary_path = root / "summary.txt"
    if not summary_path.is_file():
        parser.error(f"Missing {summary_path}")
    summary = summary_path.read_text(encoding="utf-8", errors="replace")
    print(f"=== Read-only snapshot: {datetime.datetime.now().isoformat(timespec='seconds')} ===")
    print(f"Run directory: {root}")
    print("\n=== Last summary lines ===")
    print(tail(summary_path, 12))
    matches = list(re.finditer(r"^=== INFERENCE (python|cpp): fresh service; profile=(.+) ===$", summary, re.MULTILINE))
    if not matches:
        print("No inference profile path recorded yet.")
        return
    backend, profile_text = matches[-1].groups()
    service_log = root / f"inference_{backend}.log"
    print(f"\n=== Active/last backend: {backend}; service log tail: {service_log} ===")
    print(tail(service_log, 60))
    profile = Path(profile_text)
    print(f"\n=== Profile directory: {profile} ===")
    if not profile.is_dir():
        print("Profile directory not present yet.")
        return
    free = shutil.disk_usage(profile).free / (1024**3)
    print(f"Free disk space: {free:.2f} GiB")
    # Read exported metadata and parser logs only; never open tensors or traces.
    files = []
    for pattern in ("trace_view.json", "analyse.done", "*.log", "*.csv", "*.db"):
        files.extend(profile.rglob(pattern))
    entries = []
    for path in files:
        try:
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            continue
    print(
        f"Timelines: {sum(p.name == 'trace_view.json' for _, _, p in entries)}; "
        f"analyse.done: {sum(p.name == 'analyse.done' for _, _, p in entries)}"
    )
    print("Most recently modified exports/logs (repeat command to check progress):")
    for modified, size, path in sorted(entries, reverse=True)[:20]:
        print(
            f"{datetime.datetime.fromtimestamp(modified).isoformat(timespec='seconds')} "
            f"bytes={size} {path.relative_to(profile)}"
        )
    logs = [entry for entry in sorted(entries, reverse=True) if entry[2].suffix == ".log"]
    for _, _, path in logs[:6]:
        print(f"\n=== Parser log tail: {path.relative_to(profile)} ===")
        print(tail(path, 12))
    print("=== Snapshot complete; no processes, profiles or files changed ===")


if __name__ == "__main__":
    main()
