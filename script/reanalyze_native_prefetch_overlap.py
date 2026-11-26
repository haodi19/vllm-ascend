# SPDX-License-Identifier: Apache-2.0
"""Explain Python/native prefetch gaps from existing aligned model captures."""

import argparse
import collections
import contextlib
import json
from pathlib import Path

from analyze_moe_overlap_difference import coordinates, transitions
from analyze_prefetch_compute_overlap import MAX_EXAMPLES, ORDER_US, analyze
from inspect_moe_profile import _stats
from moe_experiment_tables import number, table
from native_moe_artifacts import saved_case_file
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
    cases = tuple(c for c in ("python", "native", "native_full", "native_pipeline") if c in results)
    if "native_pipeline" in cases:
        report(
            "native_pipeline 的 host_callback/update_weights 仅含规划与入队；"
            "load_submit/copy_submit 在独立 DMA 线程，synchronize 不再执行。不能把 Host 缩短直接当作性能收益。"
        )
    rows = []
    for kind, label in (("all", "全部"), ("prefetch", "预取"), ("reactive", "按需")):
        for stage in ("host_callback", "update_weights", "load_submit", "copy_submit", "synchronize"):
            cells = []
            for case in cases:
                result = results[case]
                stat = result["host_timings"][kind][stage]
                expected = result["host_timing_expected_counts"][kind]
                clean = stat["outliers_us"]["filtered"] or dict(n=0, mean=None)
                cells.append(
                    f"{stat['count']}/{expected}; {number(stat['mean_us'])}; {clean['n']}/{number(clean['mean'])}"
                )
            rows.append([label, stage, *cells])
    report(table(["范围", "Host函数/分项", *(f"{c} n/应有n; raw; IQR" for c in cases)], rows))


def host_span_stats(result):
    rows = result["rows"]
    return {
        "prefetch_host": _stats([r["prefetch_duration_us"] for r in rows]),
        "gap": _stats([r["gap_us"] for r in rows]),
        "prefetch_start_to_reactive_start": _stats([r["prefetch_duration_us"] + r["gap_us"] for r in rows]),
    }


def coordinate_estimate_eligible(result):
    """Eligibility for sensitivity analysis, NEVER acceptance of failed order."""
    checks = result.get("mte_order_checks", [])
    if result.get("binding_error") or not checks or not any(not c["valid"] for c in checks):
        return False
    largest = max(max(c["publish_start_minus_host_start_us"], c["host_end_minus_wait_end_us"]) for c in checks)
    return largest <= result["sensitivity_margin_us"]


def render(results, report):
    classification, restats = [], []
    for case, result in results.items():
        report(
            f"CHECK {case}: callbacks={result['expected_callbacks']}; pairs={len(result['rows'])}; "
            f"margin_us={number(result['sensitivity_margin_us'])}; unmapped_reasons={result['reasons']}"
        )
        report(f"LOAD_EVIDENCE {case}: {result['load_issues']}")
        checks = result.get("mte_order_checks", [])
        failed = [item for item in checks if not item["valid"]]
        report(
            "PROTOCOL_ORDER "
            + json.dumps(
                dict(
                    case=case,
                    checked=len(checks),
                    failed=len(failed),
                    tolerance_us=ORDER_US,
                    binding_error=result.get("binding_error"),
                    max_publish_after_host_us=max(
                        (max(0, item["publish_start_minus_host_start_us"]) for item in checks), default=None
                    ),
                    max_wait_before_host_end_us=max(
                        (max(0, item["host_end_minus_wait_end_us"]) for item in checks), default=None
                    ),
                ),
                ensure_ascii=False,
            )
        )
        if failed:
            report(
                "ORDER_FAILURE_EXAMPLES "
                + json.dumps(
                    dict(
                        case=case,
                        examples=sorted(
                            failed,
                            key=lambda x: max(x["publish_start_minus_host_start_us"], x["host_end_minus_wait_end_us"]),
                            reverse=True,
                        )[:MAX_EXAMPLES],
                    ),
                    ensure_ascii=False,
                )
            )
        if result["classification"]["unmapped"]["count"]:
            report(
                f"PARTIAL_BOUNDARIES {case}: 无法识别样本保留原间隔；"
                "两方案只处理其余明确晚完成样本，不能视为全部计算等待已扣除。"
            )
        if "coordinate_estimate" in result:
            report(
                f"COORDINATE_ESTIMATE {case}: 下表使用同一对齐坐标的条件估算；完整流内配对仍成立，"
                "跨Host/设备顺序检查仍未通过。使用原校准余量区分早/晚/不确定；不修改时间戳，"
                "不利用wait拟合时钟。此估算不能证明实际握手顺序正确。"
            )
        view = result.get("coordinate_estimate", result)
        for mode, label, group_key in (
            ("direct", "对齐坐标直接判定", "direct_classification"),
            ("conservative", "保守判定", "classification"),
        ):
            groups = view[group_key]
            if "coordinate_estimate" in result:
                label += "（条件坐标估算）"
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
            values = view[mode]["all_samples"]
            cells = [
                f"{values[k]['count']}/{number(values[k]['mean_us'])}"
                for k in ("original", "method1_drop_late", "method2_subtract_remaining_compute")
            ]
            original = values["original"]["mean_us"]
            changes = []
            for key in ("method1_drop_late", "method2_subtract_remaining_compute"):
                mean = values[key]["mean_us"]
                changes.append(number(100 * (1 - mean / original)) + "%" if original and mean is not None else "—")
            boundary_range = (
                f"{number(values['method2_aligned_boundary_low']['mean_us'])}～"
                f"{number(values['method2_aligned_boundary_high']['mean_us'])}"
            )
            if groups["unmapped"]["count"] == groups["all"]["count"]:
                cells[1:] = ["—", "—"]
                changes = ["—", "—"]
                boundary_range = "—"
            restats.append(
                [
                    case,
                    label,
                    *cells,
                    *changes,
                    boundary_range,
                    values["counts"].get("invalid_subtraction", 0),
                ]
            )
    render_host_times(results, report)
    report("HOST_ONLY_SPAN：同一批预取→按需样本；只使用Host坐标，不依赖设备对齐；跨度=预取Host耗时+随后间隔。")
    report(
        table(
            ["组", "n", "预取Host均值μs", "随后间隔均值μs", "预取开始→按需开始均值μs"],
            [
                [case, len(result["rows"]), *(number(s["mean_us"]) for s in host_span_stats(result).values())]
                for case, result in results.items()
            ],
        )
    )
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
    for left, right in (
        ("python", "native"),
        ("python", "native_full"),
        ("native", "native_full"),
        ("native", "native_pipeline"),
        ("python", "native_pipeline"),
    ):
        if left in results and right in results:
            report(f"PAIR_COMPARISON {right} vs {left}")
            render_pair(results, report, left, right)
    if len(results) < 2:
        report("COMPARISON_UNAVAILABLE: need two saved captures to explain the change between versions.")


def render_pair(results, report, left_case, right_case):
    if (
        results[left_case]["physical_npu"] is None
        or results[left_case]["physical_npu"] != results[right_case]["physical_npu"]
    ):
        report("COMPARISON_UNAVAILABLE: physical NPU missing or different; per-case restats retained.")
        return
    try:
        views = {case: r.get("coordinate_estimate", r) for case, r in results.items()}
        pairs, excluded = comparison_rows(views[left_case], views[right_case])
    except ValueError as exc:
        report(f"COMPARISON_UNAVAILABLE: {exc}")
        return
    estimated = any("coordinate_estimate" in r for r in (results[left_case], results[right_case]))
    label = "PAIRED_COORDINATE_ESTIMATES" if estimated else "PAIRED_BOUNDARIES"
    report(f"{label} n={len(pairs)}; excluded={excluded}; conditional_on_aligned_coordinates={estimated}")
    if not pairs:
        report("COMPARISON_UNAVAILABLE: no valid paired compute boundaries; Host function timings retained.")
        return
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
    report(
        table(
            ["同一批配对样本", "n", f"{left_case} μs", f"{right_case} μs", f"{right_case}−{left_case} μs"],
            metric_rows,
        )
    )
    report("分解恒等式：间隔变化=计算相对到达变化−预取Host耗时变化＋(按需开始−计算到达)变化。")
    for margin in (0, max(r["sensitivity_margin_us"] for r in (results[left_case], results[right_case]))):
        report("PAIRED_TRANSITIONS " + json.dumps(dict(margin_us=margin, transitions=transitions(pairs, margin))))


def report_scope(report):
    report("分析预取Host是否更早结束、计算是否暴露到间隔中；早/晚均指计算相对于预取Host结束。")
    report("方案1删除明确晚完成样本；方案2扣除其预取结束至计算到达的时间（边界区间中点）。")
    report("两方案全样本口径：早、不确定、无法识别样本均保留原间隔；不做IQR、不设时长截断。")
    report("直接判定不加余量；保守判定用同次校准包络宽度+1μs。边界范围不是完整时钟误差区间。")
    report("重统计相对原均值的降低是描述性结果，不是新测Decode/TPOT收益；native缺少逐专家计数时不猜测零加载。")
    report("跨Host/设备顺序不符只阻止相关样本的计算分类；Host函数耗时独立统计，不平移时间戳、不放宽1μs检查。")
    report("LOAD_EVIDENCE只说明逐专家加载数能否核验；缺失加载后端元数据不等于Python加载路径运行失败。")
    report("顺序残差未超过原校准敏感性余量时，另列条件坐标估算；严格检查失败记录原样保留，不宣称顺序已验证。")


def reanalyze(runs, output, report):
    results, reference, requested_cases = {}, None, set()
    with (output / "prefetch_overlap_details.log").open("w", encoding="utf-8") as details:
        for root, case, trace in saved_traces(runs):
            requested_cases.add(case)
            count, device, expected = saved_case(root, case)
            if reference is not None and reference != expected:
                raise ValueError("Python/native saved output references differ")
            reference = expected
            counters = json.loads(saved_case_file(root, case, "profile_counters.json").read_text(encoding="utf-8"))
            if bool(counters.get("native")) != (case in ("native", "native_full", "native_pipeline")):
                raise ValueError(f"{case}: saved counters do not identify the expected update backend")
            report(f"START {case}: physical_npu={device}; trace={trace}")
            details.write(f"CASE={case}\n")
            try:
                with contextlib.redirect_stdout(details):
                    result = analyze(
                        trace,
                        profile_report=counters,
                        update_backend="python" if case == "python" else "native",
                        exact_relative=True,
                        retain_unmapped=True,
                    )
                result.update(physical_npu=device, expected_callbacks=count)
                result["host_only_span"] = host_span_stats(result)
                results[case] = result
                if coordinate_estimate_eligible(result):
                    # The failed strict result stays in result; no acceptance
                    # criterion or timestamps are changed by this second view.
                    try:
                        with contextlib.redirect_stdout(details):
                            estimate = analyze(
                                trace,
                                profile_report=counters,
                                update_backend="python" if case == "python" else "native",
                                exact_relative=True,
                                retain_unmapped=True,
                                estimate_order_mismatches=True,
                            )
                        result["coordinate_estimate"] = estimate
                    except (ValueError, KeyError, TypeError) as exc:
                        report(f"COORDINATE_ESTIMATE_UNAVAILABLE {case}: {exc}; strict results retained")
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
        or set(results) != requested_cases
        or any(
            r.get("coordinate_estimate", r)["classification"]["unmapped"]["count"] == len(r["rows"])
            for r in results.values()
        )
    ):
        raise ValueError("compute boundary unavailable for requested case(s); no complete overlap result")
    if any("coordinate_estimate" in r for r in results.values()):
        return "OVERLAP_ESTIMATED"
    partial = any(r["classification"]["unmapped"]["count"] for r in results.values())
    return "OVERLAP_PARTIAL" if partial else "OVERLAP_COMPLETE"


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
    report("SCRIPT_REVISION=native_overlap_v4_integrated_single_case")
    report("只读已有对齐profile/计数器/输出；不导入Torch、不启动模型/NPU、不改原采集；仅写本次分析目录。")
    report_scope(report)
    try:
        status = reanalyze([p.resolve(strict=True) for p in args.runs], output, report)
    except Exception as exc:
        report(f"OVERLAP_FAILED {type(exc).__name__}: {exc}; completed output retained")
        return 1
    report(f"{status}; return this summary.txt, no model rerun needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
