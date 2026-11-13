# SPDX-License-Identifier: Apache-2.0
"""Render measured comparison tables; missing data stays missing, never zero."""

import json
import math

from moe_outlier_stats import outlier_summary
from moe_tpot import summarize_streams

VALID_INTERVAL_STATUSES = ("complete", "complete_with_warnings", "recovered_intervals")
CASE_DESCRIPTIONS = {
    "callback_legacy": "早期回调配置：Python LRC，debug=true（当前代码复现配置）",
    "callback": "当前回调：C++ LRC，debug=false",
    "mte_default": "MTE默认：C++ LRC，all/locked/yield",
    "mte_combined": "MTE组合：C++ LRC，allocated/atomic/yield",
}


def number(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in values) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def bubble_records(text):
    case = None
    records = []
    for line in text.splitlines():
        if line.startswith("CASE="):
            case = line[5:].split(";", 1)[0]
        elif case and line.startswith(("TARGET_GAPS_US ", "TARGET_DIRECTION_GAPS_US ", "OFFLOAD_STATUS ")):
            kind, value = line.split(" ", 1)
            records.append((case, kind, json.loads(value)))
    return records


def render_tables(reports, labels, *, bubbles="", settings=None, failure=None):
    lines = ["# MoE对照实验结果", "", "## 配置与有效性", ""]
    for key, value in (settings or {}).items():
        lines.append(f"- {key}: {value}")
    if failure:
        lines.extend(["", f"本次未完成完整对照：{failure}。已有数据仅作记录，不据此比较收益。"])
    lines.extend(
        [
            "",
            table(
                ["组", "含义", "停止/导出状态", "时间线验证状态", "回调数"],
                [
                    [
                        label,
                        CASE_DESCRIPTIONS.get(label, label),
                        reports.get(label, {}).get(
                            "profile_export_status", reports.get(label, {}).get("profile_status", "缺失")
                        ),
                        reports.get(label, {}).get("profile_status", "缺失"),
                        reports.get(label, {}).get("count", "—"),
                    ]
                    for label in labels
                ],
            ),
            "",
            "recovered_intervals表示导出流程未确认结束，但已保存时间线通过独立完整性和回调覆盖检查。",
            "早期配置与其余组同时存在LRC和debug差异，不作单因素归因。各独立服务顺序运行，不自动认定统计显著。",
        ]
    )
    output_match = (
        bool(reports)
        and all(label in reports for label in labels)
        and all(
            reports[labels[0]].get(field)
            and all(reports[label].get(field) == reports[labels[0]].get(field) for label in labels)
            for field in ("request_specs", "digests", "usage")
        )
    )
    lines.extend(
        [
            "",
            f"采集请求参数、输出摘要及token用量一致：{output_match}（仅有限请求验证）。",
            "",
            table(
                ["组", "引擎清理边界唯一", "清理前EngineDeadError数"],
                [
                    [
                        label,
                        reports.get(label, {})
                        .get("engine_error_log_positions", {})
                        .get("cleanup_marker_unique", "未知"),
                        reports.get(label, {}).get("engine_error_log_positions", {}).get("before_cleanup", "未知"),
                    ]
                    for label in labels
                ],
            ),
        ]
    )
    rows = []
    for label in labels:
        report = reports.get(label, {})
        window = report.get("request_window", {})
        metrics = []
        if window.get("samples_us"):
            metrics.append(("端到端", outlier_summary([v / 1000 for v in window["samples_us"]])))
        streams = window.get("stream_samples", [])
        if streams:
            stream_summary = summarize_streams(streams)
            metrics.extend((name, stream_summary[key]) for name, key in (("TTFT", "ttft_ms"), ("TPOT", "tpot_ms")))
        for name in ("端到端", "TTFT", "TPOT"):
            stats = next((value for metric, value in metrics if metric == name), {})
            raw, clean = stats.get("raw") or {}, stats.get("filtered") or {}
            rows.append(
                [
                    label,
                    name,
                    raw.get("n", "—"),
                    number(raw.get("mean")),
                    number(clean.get("mean")),
                    number(raw.get("p50")),
                    number(raw.get("p95")),
                    stats.get("removed_count", "—"),
                ]
            )
    lines.extend(
        [
            "",
            "## 无profiler请求性能",
            "",
            table(["组", "指标/ms", "n", "raw均值", "filtered均值", "raw P50", "raw P95", "剔除数"], rows),
            "",
            "TPOT=(最后文本块到达−首个文本块到达)/(实际输出token数−1)，MTP块数不等于token数。",
            "filtered使用3×IQR，各指标独立筛选，分项不可相加；统计长尾不等于已证实的外部干扰。",
        ]
    )
    rows = []
    for label in labels:
        report = reports.get(label, {})
        traces = report.get("profile_trace_data", [])
        valid = report.get("profile_status") in VALID_INTERVAL_STATUSES
        for direction in ("prefetch->reactive", "reactive->prefetch"):
            gaps = (
                [
                    g
                    for t in traces
                    if t.get("usable") and t.get("json_complete")
                    for g in t.get("host_gaps", [])
                    if g.get("direction") == direction
                ]
                if valid
                else []
            )
            if len(traces) == 1 and len(gaps) == 1:
                stats = gaps[0].get("outliers_us", {})
                raw, clean = stats.get("raw") or {}, stats.get("filtered") or {}
                rows.append(
                    [
                        label,
                        direction,
                        raw.get("n", gaps[0].get("count", "—")),
                        number(raw.get("mean", gaps[0].get("mean_us"))),
                        number(clean.get("mean")),
                        number(raw.get("p50", gaps[0].get("p50_us"))),
                        stats.get("removed_count", "—"),
                    ]
                )
            else:
                rows.append([label, direction, "—", "—", "—", "—", "缺失或多时间线未合并"])
    lines.extend(
        [
            "",
            "## Host回调间隔",
            "",
            table(["组", "方向", "原始n", "raw均值/μs", "filtered均值/μs", "raw P50/μs", "剔除数"], rows),
            "",
            "前一回调结束至后一回调开始；不是纯同步指令耗时，不使用设备空泡10～5000μs筛选。",
        ]
    )
    records = bubble_records(bubbles)
    rows = []
    for label in labels:
        for scope in ("full_capture", "callback_window"):
            matches = [r for c, k, r in records if c == label and k == "TARGET_GAPS_US" and r["scope"] == scope]
            r = matches[0] if len(matches) == 1 else {}
            kept = r.get("kept") or {}
            rows.append(
                [
                    label,
                    scope,
                    kept.get("n", "—"),
                    number(kept.get("mean")),
                    number(kept.get("p50")),
                    number(kept.get("p95")),
                    number(kept["total"] / 1000) if "total" in kept else "—",
                    r.get("excluded_short", "—"),
                    r.get("excluded_long", "—"),
                ]
            )
    lines.extend(
        [
            "",
            "## 加载流空泡",
            "",
            "选定流全部设备任务取占用并集（含MEMCPY、Transpose）。仅保留10≤gap<5000μs，不再IQR。",
            "full_capture是全采集窗口（含Prefill）；callback_window是首末回调间连续范围。流空闲不等于整卡空闲。",
            "",
            table(["组", "范围", "保留n", "均值/μs", "P50/μs", "P95/μs", "累计/ms", "排除<10μs", "排除≥5ms"], rows),
        ]
    )
    rows = []
    for label in labels:
        states = [r for c, k, r in records if c == label and k == "OFFLOAD_STATUS"]
        state = (
            "子集：存在跨边界/歧义任务"
            if any(r.get("partial") for r in states)
            else "关联检查通过"
            if len(states) == 1
            else "未验证"
        )
        for direction in ("prefetch->reactive", "reactive->prefetch"):
            matches = [
                r
                for c, k, r in records
                if c == label and k == "TARGET_DIRECTION_GAPS_US" and r["direction"] == direction
            ]
            r = matches[0] if len(matches) == 1 else {}
            kept = r.get("kept") or {}
            rows.append(
                [label, direction, kept.get("n", "—"), number(kept.get("mean")), number(kept.get("p95")), state]
            )
    lines.extend(
        [
            "",
            "## 设备空泡方向子集",
            "",
            table(["组", "方向", "保留n", "均值/μs", "P95/μs", "关联状态"], rows),
            "",
            "只统计唯一关联的相邻加载批次；即使关联检查通过，也不包含空载回调及阈值外间隔。",
            "组间方向样本覆盖不同，不能把子集均值直接当成完整方向收益。均值减少也不保证累计空泡减少。",
            "",
        ]
    )
    return "\n".join(lines)


def write_tables(root, reports, labels, **kwargs):
    text = render_tables(reports, labels, **kwargs)
    miracle_path = root / "tpot_miracle.json"
    if miracle_path.is_file():
        try:
            results = json.loads(miracle_path.read_text(encoding="utf-8"))
            text += (
                "\n## Miracle假设（非实测）\n\n"
                + table(
                    ["同次profile TPOT/ms", "假设TPOT/ms", "减去的Host间隔/ms"],
                    [
                        [number(r["measured_tpot_ms"]), number(r["miracle_tpot_ms"]), number(r["removed_host_gap_ms"])]
                        for r in results
                    ],
                )
                + "\n\n假设全部Host间隔均可从关键路径移除；仅与同次profile请求比较，不能与无采集TPOT混算。\n"
            )
        except (OSError, ValueError, KeyError, TypeError):
            text += "\nMiracle数据缺失或无效，未补算替代值。\n"
    (root / "experiment_tables.md").write_text(text, encoding="utf-8")
    return text
