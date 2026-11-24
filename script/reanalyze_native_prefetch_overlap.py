# SPDX-License-Identifier: Apache-2.0
"""Explain Python/native prefetch gaps from existing aligned model captures."""

import argparse
import collections
import contextlib
import json
from pathlib import Path

from analyze_moe_overlap_difference import coordinates, transitions
from analyze_prefetch_compute_overlap import analyze
from inspect_moe_profile import _stats
from moe_experiment_tables import number, table
from reanalyze_native_moe_profiles import saved_case, saved_traces


def comparison_rows(left, right):
    """Same layer occurrence and boundary kernel, without claiming equal experts."""
    a, b = left["rows"], right["rows"]
    identity = lambda r: (str(r["layer"]), r["prefetch_rows"], r["reactive_rows"])
    if not a or list(map(identity, a)) != list(map(identity, b)):
        raise ValueError("layer/occurrence/token-row sequences differ; cannot pair configurations")
    accepted, excluded = [], collections.Counter()
    for index, (ra, rb) in enumerate(zip(a, b)):
        ca, cb = coordinates(ra), coordinates(rb)
        if ca is None or cb is None:
            excluded["boundary_unavailable"] += 1
            continue
        if not ra.get("last_compute_name") or ra["last_compute_name"] != rb.get("last_compute_name"):
            excluded["boundary_kernel_differs"] += 1
            continue
        for row, values in ((ra, ca), (rb, cb)):
            values["gap"] = row["gap_us"]
            values["arrival_to_reactive"] = row["gap_us"] - values["delta"]
            anchor = row.get("previous_reactive_anchor_us")
            if anchor:
                values["anchored_prefetch_start"] = anchor["prefetch_start"]
                values["anchored_prefetch_end"] = anchor["prefetch_end"]
                values["anchored_compute"] = (anchor["compute_lower"] + anchor["compute_upper"]) / 2
        accepted.append(dict(index=index, layer=ra["layer"], left=ca, right=cb))
    return accepted, dict(excluded)


def render_host_times(results, report):
    report("HOST_FUNCTION_TIMES：全部Host任务（含首尾及零加载），不受早/晚分类筛选；阶段嵌套，不能相加。")
    report("每格为有效n/应有n；原始均值μs；3×IQR后n/均值μs。缺失标记不当作0，明细与原始统计保留在JSON。")
    rows = []
    for kind, label in (("all", "全部"), ("prefetch", "预取"), ("reactive", "按需")):
        for stage in ("host_callback", "update_weights", "load_submit", "copy_submit", "synchronize"):
            cells, means = [], []
            for case in ("python", "native"):
                if case not in results:
                    cells.append("—")
                    means.append(None)
                    continue
                result = results[case]
                stat = result["host_timings"][kind][stage]
                expected = result["host_timing_expected_counts"][kind]
                clean = stat["outliers_us"]["filtered"] or dict(n=0, mean=None)
                cells.append(
                    f"{stat['count']}/{expected}; {number(stat['mean_us'])}; {clean['n']}/{number(clean['mean'])}"
                )
                means.append(stat["mean_us"] if stat["count"] == expected else None)
            a, b = means
            change = number(100 * (b / a - 1)) + "%" if a and b is not None else "—"
            rows.append([label, stage, *cells, change])
    report(table(["范围", "Host函数/分项", "Python n/应有n; raw; IQR", "native n/应有n; raw; IQR", "原均值变化"], rows))


def render(results, report):
    classification, restats = [], []
    for case, result in results.items():
        report(
            f"CHECK {case}: callbacks={result['expected_callbacks']}; pairs={len(result['rows'])}; "
            f"margin_us={number(result['sensitivity_margin_us'])}; unmapped_reasons={result['reasons']}"
        )
        report(f"LOAD_EVIDENCE {case}: {result['load_issues']}")
        for mode, label, group_key in (
            ("direct", "对齐坐标直接判定", "direct_classification"),
            ("conservative", "保守判定", "classification"),
        ):
            groups = result[group_key]
            counts = [
                f"{groups[k]['count']}/{number(groups[k]['fraction_pct'])}%"
                for k in ("early", "late", "uncertain", "unmapped")
            ]
            classification.append(
                [
                    case,
                    label,
                    groups["all"]["count"],
                    *counts,
                    number(groups["early"]["mean_us"]),
                    number(groups["late"]["mean_us"]),
                ]
            )
            values = result[mode]["all_samples"]
            cells = [
                f"{values[k]['count']}/{number(values[k]['mean_us'])}"
                for k in ("original", "method1_drop_late", "method2_subtract_remaining_compute")
            ]
            original = values["original"]["mean_us"]
            changes = []
            for key in ("method1_drop_late", "method2_subtract_remaining_compute"):
                mean = values[key]["mean_us"]
                changes.append(number(100 * (1 - mean / original)) + "%" if original and mean is not None else "—")
            restats.append(
                [
                    case,
                    label,
                    *cells,
                    *changes,
                    f"{number(values['method2_aligned_boundary_low']['mean_us'])}～{number(values['method2_aligned_boundary_high']['mean_us'])}",
                    values["counts"].get("invalid_subtraction", 0),
                ]
            )
    render_host_times(results, report)
    report(
        table(
            [
                "组",
                "判定",
                "全部n",
                "计算早于预取结束",
                "计算晚于预取结束",
                "边界不确定",
                "无法识别",
                "早样本间隔均值μs",
                "晚样本间隔均值μs",
            ],
            classification,
        )
    )
    report(
        table(
            [
                "组",
                "判定",
                "原始n/均值μs",
                "方案1 n/均值μs",
                "方案2 n/均值μs",
                "方案1相对原始降低",
                "方案2相对原始降低",
                "方案2边界均值范围μs",
                "无法扣除n",
            ],
            restats,
        )
    )
    if set(results) != {"python", "native"}:
        report("COMPARISON_UNAVAILABLE: need both saved captures to explain the change between versions.")
        return
    if (
        results["python"]["physical_npu"] is None
        or results["python"]["physical_npu"] != results["native"]["physical_npu"]
    ):
        report("COMPARISON_UNAVAILABLE: physical NPU missing or different; per-case restats retained.")
        return
    try:
        pairs, excluded = comparison_rows(results["python"], results["native"])
    except ValueError as exc:
        report(f"COMPARISON_UNAVAILABLE: {exc}")
        return
    report(f"PAIRED_BOUNDARIES n={len(pairs)}; excluded={excluded}")
    report("按同层同次序和相同边界算子比较；未核验两组专家ID/加载数一致，不能作等工作量因果证明。")
    report("计算到达以各自预取开始为零；不是设备算子总耗时。另用上一按需Host结束作共同流程锚点检查预取启动变化。")
    metric_rows = []
    for key, label in (
        ("host", "预取Host耗时"),
        ("compute", "计算到达−预取开始"),
        ("delta", "计算到达−预取结束（有符号）"),
        ("arrival_to_reactive", "按需开始−计算到达"),
        ("gap", "预取→按需Host间隔"),
        ("anchored_prefetch_start", "预取开始−上一按需结束"),
        ("anchored_prefetch_end", "预取结束−上一按需结束"),
        ("anchored_compute", "计算到达−上一按需结束"),
    ):
        selected = [p for p in pairs if key in p["left"] and key in p["right"]]
        a, b = [_stats([p[side][key] for p in selected])["mean_us"] for side in ("left", "right")]
        metric_rows.append(
            [label, len(selected), number(a), number(b), number(b - a) if a is not None and b is not None else "—"]
        )
    report(table(["同一批配对样本", "n", "Python μs", "native μs", "native−Python μs"], metric_rows))
    report("分解恒等式：间隔变化=计算相对到达变化−预取Host耗时变化＋(按需开始−计算到达)变化。")
    for margin in (0, max(r["sensitivity_margin_us"] for r in results.values())):
        report("PAIRED_TRANSITIONS " + json.dumps(dict(margin_us=margin, transitions=transitions(pairs, margin))))


def reanalyze(runs, output, report):
    results, reference = {}, None
    with (output / "details.log").open("w", encoding="utf-8") as details:
        for root, case, trace in saved_traces(runs):
            count, device, expected = saved_case(root, case)
            if reference is not None and reference != expected:
                raise ValueError("Python/native saved output references differ")
            reference = expected
            counters = json.loads((root / f"{case}_profile_counters.json").read_text(encoding="utf-8"))
            if bool(counters.get("native")) != (case == "native"):
                raise ValueError(f"{case}: saved counters do not identify the expected update backend")
            report(f"START {case}: physical_npu={device}; trace={trace}")
            details.write(f"CASE={case}\n")
            try:
                with contextlib.redirect_stdout(details):
                    result = analyze(trace, profile_report=counters, update_backend=case, exact_relative=True)
                result.update(physical_npu=device, expected_callbacks=count)
                results[case] = result
                report(
                    f"DONE {case}: pairs={len(result['rows'])}; "
                    f"unmapped={result['classification']['unmapped']['count']}"
                )
            except (ValueError, KeyError, TypeError) as exc:
                # Preserve the other case and the reason; never present missing results as zero.
                report(f"CASE_UNAVAILABLE {case}: {type(exc).__name__}: {exc}")
    (output / "prefetch_overlap.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    render(results, report)
    if (
        not results
        or "native" not in results
        or results["native"]["classification"]["unmapped"]["count"] == len(results["native"]["rows"])
    ):
        raise ValueError("native compute boundary unavailable; no conclusion about exposed computation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="saved Python/native profile run directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def report(message):
        print(message, flush=True)
        with (output / "summary.txt").open("a", encoding="utf-8") as destination:
            destination.write(message + "\n")

    report(f"SUMMARY_FILE={output / 'summary.txt'}")
    report("只读已有对齐profile/计数器/输出；不导入Torch、不启动模型/NPU、不改原采集；仅写本次分析目录。")
    report("分析预取Host是否更早结束、计算是否暴露到间隔中；早/晚均指计算相对于预取Host结束。")
    report("方案1删除明确晚完成样本；方案2扣除其预取结束至计算到达的时间（边界区间中点）。")
    report("两方案全样本口径：早、不确定、无法识别样本均保留原间隔；不做IQR、不设时长截断。")
    report("直接判定不加余量；保守判定用同次校准包络宽度+1μs。边界范围不是完整时钟误差区间。")
    report("重统计相对原均值的降低是描述性结果，不是新测Decode/TPOT收益；native缺少逐专家计数时不猜测零加载。")
    try:
        reanalyze([p.resolve(strict=True) for p in args.runs], output, report)
    except Exception as exc:
        report(f"OVERLAP_FAILED {type(exc).__name__}: {exc}; completed output retained")
        return 1
    report("OVERLAP_COMPLETE; return this summary.txt, no model rerun needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
