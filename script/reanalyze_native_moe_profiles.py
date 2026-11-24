# SPDX-License-Identifier: Apache-2.0
"""Recompute Host/loading gaps from saved aligned captures, without running models."""

import argparse
import collections
import json
import re
from pathlib import Path

from export_moe_aligned_profile import aligned_trace_path
from native_moe_profile import analyze_profile, report_gap_table
from verify_memcpy_gaps import MAX_DISTRIBUTION_VALUES, read_copy_tracks


def saved_traces(runs):
    seen = set()
    for root in runs:
        for case in ("python", "native"):
            directory = root / "profile" / case
            if not directory.is_dir():
                continue
            if case in seen:
                raise ValueError(f"Duplicate {case} case; pass only one run per case")
            seen.add(case)
            paths = directory.rglob("trace_view.json")
            trace = next(paths, None)
            if trace is None or next(paths, None) is not None:
                raise ValueError(f"{case}: expected exactly one saved trace")
            yield root, case, aligned_trace_path(trace)
    if not seen:
        raise ValueError("No saved profile/python or profile/native found")


def list_streams(runs, report):
    report("READ ONLY: saved aligned trace inventory; no Torch/NPU/model, no writes, no automatic stream selection.")
    for root, case, trace in saved_traces(runs):
        report(f"SOURCE case={case}; run={root}; aligned_trace={trace}")
        tracks, names = read_copy_tracks(trace)
        if not tracks:
            raise ValueError(f"{case}: no hardware MEMCPY tracks")
        for key, copies in sorted(tracks.items()):
            directions = collections.Counter(item["fields"].get("direction", "unknown") for item in copies)
            sizes = collections.Counter(item["fields"].get("bytes") for item in copies)
            missing_bytes = sizes.pop(None, 0)
            report(
                "COPY_TRACK "
                + json.dumps(
                    dict(
                        case=case,
                        pid=key[0],
                        tid=key[1],
                        name=names.get(key, ""),
                        copies=len(copies),
                        directions=dict(directions),
                        common_bytes_counts=sizes.most_common(MAX_DISTRIBUTION_VALUES),
                        missing_bytes=missing_bytes,
                    ),
                    ensure_ascii=False,
                )
            )
    report(
        "INVENTORY_COMPLETE: names are visualization thread_name values, not tid. Counts alone do not identify weights."
    )


def saved_case(root, case):
    records = []
    physical_npus = set()
    with (root / "summary.txt").open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("RESULT "):
                row = json.loads(line[7:])
                if row.get("case") == case:
                    records.append(row)
            if line.startswith("PROFILE_ONLY:"):
                match = re.search(r"physical_npu=(\d+)", line)
                if match:
                    physical_npus.add(int(match[1]))
    if len(records) != 1 or not records[0].get("output_match"):
        raise ValueError(f"{case}: missing or ambiguous saved timing reference")
    reference = records[0].get("stream_samples", [])
    samples = json.loads((root / f"{case}_profile_samples.json").read_text(encoding="utf-8"))
    counters = json.loads((root / f"{case}_profile_counters.json").read_text(encoding="utf-8"))
    if not reference or not samples:
        raise ValueError(f"{case}: no completed saved requests")
    if any(
        (row["digest"], row["usage"]) != (reference[0]["digest"], reference[0]["usage"])
        for row in [*reference, *samples]
    ):
        raise ValueError(f"{case}: saved profile output differs from timing reference")
    if counters["mstx_errors"] or counters["errors"] or counters["dropped"] or counters["count"] <= 0:
        raise ValueError(f"{case}: incomplete saved Host counters")
    if len(physical_npus) > 1:
        raise ValueError(f"{case}: ambiguous physical NPU metadata")
    return counters["count"], next(iter(physical_npus), None), (reference[0]["digest"], reference[0]["usage"])


def reanalyze(runs, output, report, stream_name="auto"):
    profiles, devices = {}, {}
    expected_output = None
    for root, case, trace in saved_traces(runs):
        count, device, reference = saved_case(root, case)
        if expected_output is not None and reference != expected_output:
            raise ValueError("Python/native saved output references differ")
        expected_output = reference
        devices[case] = device
        report(f"SOURCE case={case}; physical_npu={device}; run={root}; aligned_trace={trace}")
        scan, details = analyze_profile(trace, case, count, stream_name, report)
        bubbles = [
            json.loads(line.split(" ", 1)[1])
            for line in details.splitlines()
            if line.startswith(("TARGET_GAPS_US ", "TARGET_DIRECTION_GAPS_US "))
        ]
        profiles[case] = dict(scan=scan, bubble_stats=bubbles)
        (output / f"{case}_profile_report.json").write_text(json.dumps(profiles[case]), encoding="utf-8")
        (output / f"{case}_bubble_details.log").write_text(details, encoding="utf-8")
        report(
            "PROFILE_CHECK "
            + json.dumps(
                dict(
                    case=case,
                    callbacks=scan["host_callbacks"],
                    nesting=scan["nesting"],
                    arithmetic=scan["host_time_arithmetic"],
                )
            )
        )
    if not profiles:
        raise ValueError("No saved profile/python or profile/native found")
    report_gap_table(profiles, report)
    report(
        "REANALYSIS_PASS "
        + json.dumps(
            dict(
                cases=list(profiles),
                physical_npus=devices,
                same_physical_npu=len(devices) == 2
                and None not in devices.values()
                and len(set(devices.values())) == 1,
            )
        )
    )
    report("Only saved profile statistics recomputed; no new decode/TPOT measurement or repeated-trial claim.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument(
        "--list-streams", action="store_true", help="Print saved hardware copy tracks without writing files"
    )
    parser.add_argument(
        "--bubble-stream-name",
        default="auto",
        help="Actual loading-stream marker; explicit name required for legacy profiles without markers",
    )
    args = parser.parse_args(argv)
    if args.list_streams:
        try:
            list_streams([run.resolve(strict=True) for run in args.runs], lambda line: print(line, flush=True))
        except Exception as exc:
            print(f"INVENTORY_FAILED {type(exc).__name__}: {exc}", flush=True)
            return 1
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def report(message):
        print(message, flush=True)
        with (output / "summary.txt").open("a", encoding="utf-8") as destination:
            destination.write(message + "\n")

    report(f"SUMMARY_FILE={output / 'summary.txt'}")
    report(
        "OFFLINE: read saved aligned profiles/counters/outputs; write only this new report directory. "
        "No Torch/NPU/model, no clock realignment, no original file changes. "
        "Host nesting/subtraction uses exact decimal arithmetic; no added tolerance."
    )
    try:
        reanalyze([run.resolve(strict=True) for run in args.runs], output, report, args.bubble_stream_name)
    except Exception as exc:
        report(f"REANALYSIS_FAILED {type(exc).__name__}: {exc}; completed partial output retained")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
