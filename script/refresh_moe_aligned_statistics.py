# SPDX-License-Identifier: Apache-2.0
"""Publish corrected existing captures and regenerate one summary, without model/NPU work."""

import argparse
import contextlib
import json
from pathlib import Path

from export_moe_aligned_profile import OUTPUT_NAME, publish_aligned_capture
from inspect_moe_profiler_clocks import discover_roots
from moe_experiment_tables import CASE_DESCRIPTIONS, write_tables
from moe_overlap_report import overlap_summary
from moe_tpot import print_tpot_comparison
from run_cpp_mte_profile_compare import loading_bubble_summary, print_comparison
from run_moe_host_sync import assess_profile_report


def refresh(run, stream):
    labels = [label for label in CASE_DESCRIPTIONS if (run / label / "profile_report.json").is_file()]
    if not labels:
        raise ValueError(f"no comparison reports: {run}")
    reports = {}
    for label in labels:
        print(f"ALIGN_AND_CHECK {run.name}/{label}", flush=True)
        report_path = run / label / "profile_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        roots = discover_roots([run / "profile" / label])
        if len(roots) != 1:
            raise ValueError(f"{label}: requires exactly one original/published capture")
        root = roots[0]
        mapping = json.loads((root / OUTPUT_NAME / "clock_alignment.json").read_text(encoding="utf-8"))
        report["aligned_profile"] = publish_aligned_capture(root, mapping)
        report["clock_alignment"] = json.loads(
            (root / OUTPUT_NAME / "clock_alignment.json").read_text(encoding="utf-8")
        )
        report["profile_clock_required"] = True
        logs = sorted((run / label).glob("inference_*.log"))
        if len(logs) != 1:
            raise ValueError(f"{label}: service log missing or ambiguous")
        assess_profile_report(
            report,
            logs[0].read_text(encoding="utf-8", errors="replace"),
            root,
            require_mte=report.get("host_backend") == "mte",
            retain_trace_data=True,
        )
        if not report.get("profile_trace_data") or not all(t.get("usable") for t in report["profile_trace_data"]):
            raise ValueError(f"{label}: corrected trace did not pass interval checks")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[label] = report
    # Preserve the previous summary if any recalculation fails. Publish only a complete replacement.
    pending = run / "summary_aligned.pending.txt"
    print(f"RECALCULATE {run}; progress/details={pending}", flush=True)
    with pending.open("w", encoding="utf-8") as output, contextlib.redirect_stdout(output):
        print("已有采集离线重统计：未重跑模型；Decode/TPOT沿用原请求测量，设备间隔全部来自校准后的profile。")
        print_comparison(reports, run, labels)
        bubbles = loading_bubble_summary(reports, run, labels, stream)
        print(bubbles)
        if "BUBBLES_UNAVAILABLE" in bubbles:
            raise ValueError("aligned bubbles unavailable; original summary not replaced")
        print_tpot_comparison(reports, run, labels)
        overlap = overlap_summary(reports, run, labels)
        print(write_tables(run, reports, labels, bubbles=bubbles, overlap=overlap))
    pending.replace(run / "summary.txt")
    print(f"SUMMARY FILE: {run / 'summary.txt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stream-name", default="Stream 56")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    children = [run / name for name in ("token5", "token128") if (run / name).is_dir()]
    for child in children or [run]:
        refresh(child, args.stream_name)
    if children:
        with (run / "summary.txt").open("w", encoding="utf-8") as output:
            for child in children:
                output.write(f"\n===== {child.name} =====\n")
                with (child / "summary.txt").open(encoding="utf-8") as source:
                    for line in source:
                        output.write(line)
        print(f"COMBINED SUMMARY: {run / 'summary.txt'}")


if __name__ == "__main__":
    main()
