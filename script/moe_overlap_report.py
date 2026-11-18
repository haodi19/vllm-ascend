# SPDX-License-Identifier: Apache-2.0
"""Include same-capture overlap diagnostics in the normal experiment summary."""

import contextlib
import json
from pathlib import Path

from analyze_prefetch_compute_overlap import analyze
from moe_experiment_tables import VALID_INTERVAL_STATUSES, number, table
from run_verified_cpp_tests import progress


def overlap_summary(reports, root, labels):
    results, classification_rows, restat_rows = {}, [], []
    lines = [
        "\n## 预取→按需 Host间隔与计算推进\n",
        "仅对纯MTE消费者处严格等待的组识别计算边界；回调组保留原Host间隔，不套用MTE publish/wait配对规则。",
        "计算边界包含路由和依赖推进；预取结束取外层Host回调结束。"
        "保守判定使用校准包络宽度加1μs，直接判定不加余量；两者均保留。",
        "占比以全部同层相邻预取→按需样本为分母；零加载占比以相应晚完成样本为分母。未知加载量不当作零加载。",
    ]
    with (root / "overlap_details.log").open("w", encoding="utf-8") as details:
        for label in labels:
            report = reports.get(label, {})
            if report.get("host_backend") != "mte":
                continue
            try:
                traces = report.get("profile_trace_data", [])
                if report.get("profile_status") not in VALID_INTERVAL_STATUSES or len(traces) != 1:
                    raise ValueError("requires one complete checked capture")
                trace = traces[0]
                if not trace.get("usable") or not trace.get("json_complete"):
                    raise ValueError("incomplete or unusable trace")
                # This sibling belongs to the same capture. Never substitute raw timestamps.
                path = Path(trace["file"]).with_name("trace_view_aligned.json")
                details.write(f"CASE={label}\n")
                with progress(f"prefetch overlap {label}"), contextlib.redirect_stdout(details):
                    result = analyze(path)
                results[label] = result
                lines.append(
                    f"{label}：保守余量={number(result['sensitivity_margin_us'])}μs；"
                    f"加载证据检查={result['load_issues']}；边界未识别原因={result['reasons']}。"
                )
                for mode, description, key in (
                    ("conservative", "保守", "class"),
                    ("direct", "直接", "zero_margin_class"),
                ):
                    groups = result["classification" if mode == "conservative" else "direct_classification"]
                    late = [r for r in result["rows"] if r.get(key) == "late"]
                    zero = sum(r["prefetch_loads"] == 0 for r in late)
                    either = sum(r["prefetch_loads"] == 0 or r["reactive_loads"] == 0 for r in late)
                    unknown = sum(r["prefetch_loads"] is None for r in late)

                    def count_pct(kind, groups=groups):
                        group = groups[kind]
                        return f"{group['count']} / {number(group['fraction_pct'])}%"

                    def late_pct(count, late=late):
                        return f"{count} / {number(100 * count / len(late) if late else None)}%"

                    classification_rows.append(
                        [
                            label,
                            description,
                            groups["all"]["count"],
                            count_pct("early"),
                            count_pct("late"),
                            count_pct("uncertain"),
                            count_pct("unmapped"),
                            late_pct(zero),
                            late_pct(either),
                            unknown,
                        ]
                    )
                    for scope, values in result[mode].items():

                        def metric(name, values=values):
                            stat = values[name]
                            return f"{stat['count']} / {number(stat['mean_us'])} / {number(stat['p95_us'])}"

                        restat_rows.append(
                            [
                                label,
                                description,
                                scope,
                                metric("original"),
                                metric("method1_drop_late"),
                                metric("method2_subtract_remaining_compute"),
                                f"{number(values['method2_aligned_boundary_low']['mean_us'])}～"
                                f"{number(values['method2_aligned_boundary_high']['mean_us'])}",
                                values["counts"].get("invalid_subtraction", 0),
                            ]
                        )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results[label] = dict(status="unavailable", error=str(exc))
                lines.append(f"OVERLAP_UNAVAILABLE {label}: {exc}；原时延保留，缺失分析不记为0。")
    (root / "prefetch_overlap.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    lines += [
        "",
        table(
            [
                "组",
                "判定",
                "全部n",
                "早完成n/%",
                "晚完成n/%",
                "边界不确定n/%",
                "无法识别n/%",
                "晚完成中预取零加载n/%",
                "晚完成中任一端零加载n/%",
                "晚完成中预取加载未知n",
            ],
            classification_rows,
        ),
        "",
        "以下每格为n / 均值μs / P95μs，不做IQR或长尾截断。方案1去掉晚完成样本；"
        "方案2扣除预取结束至计算到达消费位置的时间（边界区间中点），不是累加算子忙碌时长。",
        "all_samples保留不确定/无法识别样本的原值；decidable_only只含确定早/晚样本。"
        "方案2无法合法扣除的样本保留原值并计数。边界范围不包含完整时钟误差。",
        "",
        table(
            [
                "组",
                "判定",
                "范围",
                "原间隔",
                "方案1：去掉晚完成",
                "方案2：扣除计算等待",
                "方案2边界均值范围μs",
                "无法扣除n",
            ],
            restat_rows,
        ),
        "",
    ]
    return "\n".join(lines)
