# SPDX-License-Identifier: Apache-2.0
"""Explain saved overlap differences; no inference, trace reparsing or writes."""

import argparse
import collections
import json
import math
from pathlib import Path

from analyze_prefetch_compute_overlap import classify


def coordinates(row):
    if row.get("class") == "unmapped":
        return None
    duration = row.get("prefetch_duration_us")
    bounds = row.get("readiness_minus_prefetch_end_bounds_us")
    if duration is None or not isinstance(bounds, list) or len(bounds) != 2:
        return None
    duration, lower, upper = map(float, (duration, *bounds))
    if not all(math.isfinite(v) for v in (duration, lower, upper)) or duration < 0 or lower > upper:
        raise ValueError("invalid saved timing coordinates")
    delta = (lower + upper) / 2
    return dict(host=duration, compute=duration + delta, delta=delta, lower=lower, upper=upper)


def matched_rows(left, right):
    a, b = left["rows"], right["rows"]
    # Same layer occurrence in one recorded request, not a nearest-time match.
    # Equality of load counts is necessary here, but does not prove equal IDs.
    if not a or [str(r["layer"]) for r in a] != [str(r["layer"]) for r in b]:
        raise ValueError("layer sequence/count differs; refuse ordinal cross-configuration comparison")
    rejected, accepted = collections.Counter(), []
    kernel_pairs = collections.Counter()
    for index, (ra, rb) in enumerate(zip(a, b)):
        ca, cb = coordinates(ra), coordinates(rb)
        if ca is None or cb is None:
            rejected["boundary_unavailable"] += 1
            continue
        ka, kb = ra.get("last_compute_name"), rb.get("last_compute_name")
        kernel_pairs[str(ka), str(kb)] += 1
        if not ka or not kb or ka != kb:
            rejected["boundary_kernel_name_differs_or_missing"] += 1
            continue
        loads_a = [ra.get(k) for k in ("prefetch_loads", "reactive_loads")]
        loads_b = [rb.get(k) for k in ("prefetch_loads", "reactive_loads")]
        if any(v is None for v in loads_a + loads_b):
            rejected["loads_unknown"] += 1
            continue
        if loads_a != loads_b:
            rejected["load_counts_differ"] += 1
            continue
        accepted.append(dict(index=index, layer=ra["layer"], loads=loads_a[0], left=ca, right=cb))
    return accepted, rejected, kernel_pairs


def decompose(rows):
    if not rows:
        return None
    means = {
        side: {key: sum(r[side][key] for r in rows) / len(rows) for key in ("host", "compute", "delta")}
        for side in ("left", "right")
    }
    changes = {key: means["right"][key] - means["left"][key] for key in ("host", "compute", "delta")}
    return dict(n=len(rows), **means, changes=changes)


def transitions(rows, margin):
    counts = collections.Counter(
        (
            classify(r["left"]["lower"], r["left"]["upper"], 0, margin),
            classify(r["right"]["lower"], r["right"]["upper"], 0, margin),
        )
        for r in rows
    )
    return {f"{a}->{b}": n for (a, b), n in sorted(counts.items())}


def report_pair(results, left_name, right_name):
    left, right = results[left_name], results[right_name]
    rows, rejected, kernels = matched_rows(left, right)
    print(f"\nCOMPARE={left_name} -> {right_name}; input={len(left['rows'])}; matched={len(rows)}")
    print("EXCLUDED=" + json.dumps(rejected, ensure_ascii=False))
    print("BOUNDARY_KERNEL_PAIRS=" + json.dumps(kernels.most_common(5), ensure_ascii=False))
    print("TRACE_LEFT=" + left["trace"])
    print("TRACE_RIGHT=" + right["trace"])
    print("均值us；右组减左组。计算相对到达时间以各自预取Host开始为零，不是算子执行时长。")
    print(
        "| 样本范围 | n | 预取Host耗时 左/右 | 计算相对到达 左/右 | 计算减Host结束 左/右 | "
        "Host变化 | 计算变化 | 差值变化 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    margin = max(float(left["sensitivity_margin_us"]), float(right["sensitivity_margin_us"]))
    for label, subset in (
        ("共同有效", rows),
        ("预取零加载", [r for r in rows if r["loads"] == 0]),
        ("预取非零加载", [r for r in rows if r["loads"] > 0]),
    ):
        values = decompose(subset)
        if values is None:
            print(f"NO_SAMPLES={label}")
            continue
        pairs = [f"{values['left'][k]:.3f}/{values['right'][k]:.3f}" for k in ("host", "compute", "delta")]
        differences = [f"{values['changes'][k]:+.3f}" for k in ("host", "compute", "delta")]
        print("| " + " | ".join([label, str(len(subset)), *pairs, *differences]) + " |")
        print(f"TRANSITIONS {label}; common_margin_us={margin:.3f}: " + json.dumps(transitions(subset, margin)))
        print(f"TRANSITIONS {label}; common_margin_us=0: " + json.dumps(transitions(subset, 0)))
    examples = sorted(rows, key=lambda r: abs(r["right"]["delta"] - r["left"]["delta"]), reverse=True)[:5]
    print("LARGEST_DIFFERENCES=" + json.dumps(examples, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path, help="existing overlap_restat directory")
    args = parser.parse_args()
    root = args.analysis_dir.resolve(strict=True)
    paths = [root / token / "prefetch_overlap.json" for token in ("token5", "token128")]
    if not any(p.is_file() for p in paths):
        raise ValueError("no token5/token128 prefetch_overlap.json in analysis directory")
    print("只读已有逐样本JSON；不读大trace、不加载模型、不查询NPU、不写文件。")
    print("差值=计算相对到达时间-预取Host耗时；差值变化=计算变化-Host变化。")
    print("层序/次数、前后加载数量、边界算子名称相同才纳入配对；不等于专家ID或底层任务身份已证明相同。")
    print("比较使用同一批有效样本与相同判定余量。相对到达时间仍受预取启动时刻影响，不能独立证明设备变慢。")
    for path in paths:
        if not path.is_file():
            print(f"MISSING={path}")
            continue
        print(f"\nSOURCE={path}")
        results = json.loads(path.read_text(encoding="utf-8"))
        for left, right in (("callback", "mte_default"), ("mte_default", "mte_combined")):
            try:
                report_pair(results, left, right)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"UNAVAILABLE {left}->{right}: {exc}")


if __name__ == "__main__":
    main()
