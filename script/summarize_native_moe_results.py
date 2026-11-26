# SPDX-License-Identifier: Apache-2.0
"""Rebuild concise native comparison summaries from saved results, without NPU work."""

import argparse
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

from moe_experiment_tables import number, table
from moe_tpot import summarize_streams
from native_moe_artifacts import CASES, saved_case_file, saved_timing_records
from reanalyze_native_moe_profiles import saved_case
from reanalyze_native_prefetch_overlap import render_host_times

SUMMARY_HEADER = "NATIVE_MOE_RESULT_SUMMARY_V1"
MAX_STATUS_CHARS = 240


def write_summary(root, text):
    """Archive a legacy console summary once; atomically replace only the report."""
    destination = root / "summary.txt"
    if destination.exists():
        with destination.open(encoding="utf-8", errors="replace") as source:
            first = source.readline()
            if first.rstrip() != SUMMARY_HEADER:
                with (root / "run_details.log").open("a", encoding="utf-8") as log:
                    log.write("\nLEGACY_SUMMARY_BEGIN\n" + first)
                    for line in source:
                        log.write(line)
                    log.write("\nLEGACY_SUMMARY_END\n")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as output:
        temporary = Path(output.name)
        output.write(text)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def timing_record(root, case):
    row = saved_timing_records(root, (case,)).get(case)
    if row is None:
        return None
    samples = row["stream_samples"]
    if row.get("output_match") is not True or not samples or len(samples) != row.get("requests"):
        raise ValueError("计时未完整完成或输出未通过核对")
    reference = samples[0]["digest"], samples[0]["usage"]
    for sample in samples:
        if (sample["digest"], sample["usage"]) != reference:
            raise ValueError("计时样本输出不一致")
        tokens = sample["completion_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise ValueError("实际 token 数无效")
        if sample["usage"]["completion_tokens"] != tokens:
            raise ValueError("token 数与 usage 不符")
        for key in ("decode_us", "ttft_us", "http_wall_us"):
            if not math.isfinite(sample[key]) or sample[key] < 0:
                raise ValueError("计时样本缺失或无效")
        expected = sample["decode_us"] / (tokens - 1) if tokens > 1 else None
        if expected is not None and not math.isclose(sample["tpot_us"], expected, abs_tol=1e-6):
            raise ValueError("TPOT 与 decode/token 数不符")
    return row


def gather(root):
    timings, profiles, overlaps, states = {}, {}, {}, []
    overlap_path = root / "prefetch_overlap.json"
    try:
        saved_overlap = read_json(overlap_path) if overlap_path.exists() else {}
        if not isinstance(saved_overlap, dict):
            raise ValueError("expected per-case object")
        overlap_error = None
    except (OSError, ValueError) as exc:
        saved_overlap, overlap_error = {}, f"重统计文件不可读：{exc}"
    for case in CASES:
        timing_state, profile_state, overlap_state = "缺失/未完成", "缺失/未完成", "缺失/未完成"
        notes = []
        try:
            row = timing_record(root, case)
            if row is not None:
                timings[case] = row
                timing_state = f"完成（{row['requests']} 次）"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            notes.append(f"计时：{exc}")
        path = saved_case_file(root, case, "profile_report.json")
        if path.exists():
            try:
                if case not in timings:
                    raise ValueError("缺少完整计时参考")
                count, _, _ = saved_case(root, case)
                profile = read_json(path)
                if profile.get("mapping", {}).get("status") != "calibration_validated":
                    raise ValueError("没有通过同次时钟校准的报告")
                if profile.get("aligned_profile", {}).get("status") != "exported":
                    raise ValueError("对齐目录未导出完成")
                profile["scan"]["host_gaps"]
                profile["bubble_stats"]
                profiles[case] = profile
                profile_state = "完成（独立采集）"
                if case in saved_overlap:
                    overlap = saved_overlap[case]
                    if overlap.get("expected_callbacks") != count:
                        raise ValueError("重统计回调数与本组不符")
                    if Path(overlap["trace"]).resolve() != Path(profile["mapping"]["aligned_trace"]).resolve():
                        raise ValueError("重统计与 profile 来源不符")
                    view = overlap.get("coordinate_estimate", overlap)
                    groups = view["classification"]
                    overlap_state = "已统计"
                    if "coordinate_estimate" in overlap:
                        overlap_state = "条件坐标估算（顺序检查未通过）"
                    elif groups["unmapped"]["count"]:
                        overlap_state = "部分/全部边界无法识别"
                    overlaps[case] = overlap
            except (OSError, ValueError, KeyError, TypeError) as exc:
                notes.append(f"profile/重统计：{exc}")
        if overlap_error:
            notes.append(overlap_error)
        status_path = saved_case_file(root, case, "run_status.json")
        if status_path.exists():
            try:
                status = read_json(status_path)
                if status.get("exit_code") != 0:
                    reason = str(status.get("error", "")).splitlines()
                    reason = reason[0][:MAX_STATUS_CHARS] if reason else ""
                    notes.append(f"本组所在任务退出={status.get('exit_code')}; {reason}")
            except (OSError, ValueError, TypeError) as exc:
                notes.append(f"任务状态不可读：{exc}")
        states.append([case, timing_state, profile_state, overlap_state, "; ".join(notes) or "—"])
    return timings, profiles, overlaps, states


def render_overlap(results, report):
    render_host_times(results, report)
    report("\n计算相对于预取 Host 结束的早/晚：晚完成可抬高预取→按需间隔。")
    report("方案1删除明确晚完成样本；方案2扣除其暴露计算时间（边界中点）。早、不确定、无法识别样本保留原值。")
    report("重统计不做 IQR；不是新的 Decode/TPOT 测量。保守判定含同次校准包络宽度+1μs。")
    classification, restats = [], []
    for case, result in results.items():
        view = result.get("coordinate_estimate", result)
        if "coordinate_estimate" in result:
            report(f"{case}：以下为条件坐标估算，跨 Host/设备顺序检查未通过；不能当作精确测量。")
        report(f"{case}：判定余量={number(result['sensitivity_margin_us'])}μs；专家加载证据={result['load_issues']}")
        for mode, label, key in (
            ("direct", "直接", "direct_classification"),
            ("conservative", "保守", "classification"),
        ):
            groups = view[key]
            classification.append(
                [
                    case,
                    label,
                    groups["all"]["count"],
                    *(
                        f"{groups[k]['count']}/{number(groups[k]['fraction_pct'])}%"
                        for k in ("early", "late", "uncertain", "unmapped")
                    ),
                    number(groups["early"]["mean_us"]),
                    number(groups["late"]["mean_us"]),
                ]
            )
            values = view[mode]["all_samples"]
            original = values["original"]["mean_us"]
            cells = [
                f"{values[k]['count']}/{number(values[k]['mean_us'])}"
                for k in ("original", "method1_drop_late", "method2_subtract_remaining_compute")
            ]
            changes = [
                number(100 * (1 - values[k]["mean_us"] / original)) + "%"
                if original and values[k]["mean_us"] is not None
                else "—"
                for k in ("method1_drop_late", "method2_subtract_remaining_compute")
            ]
            bounds = (
                f"{number(values['method2_aligned_boundary_low']['mean_us'])}～"
                f"{number(values['method2_aligned_boundary_high']['mean_us'])}"
            )
            if groups["unmapped"]["count"] == groups["all"]["count"]:
                cells[1:], changes, bounds = ["—", "—"], ["—", "—"], "—"
            restats.append([case, label, *cells, *changes, bounds, values["counts"].get("invalid_subtraction", 0)])
    report(
        table(
            [
                "组",
                "判定",
                "n",
                "计算早完成 n/%",
                "计算晚完成 n/%",
                "不确定 n/%",
                "无法识别 n/%",
                "早间隔均值μs",
                "晚间隔均值μs",
            ],
            classification,
        )
    )
    report(
        table(
            [
                "组",
                "判定",
                "原始 n/均值μs",
                "方案1 n/均值μs",
                "方案2 n/均值μs",
                "方案1降低",
                "方案2降低",
                "方案2边界范围μs",
                "无法扣除n",
            ],
            restats,
        )
    )


def render(root, data, cases=CASES):
    timings, profiles, overlaps, states = data
    if cases == CASES:
        pipeline_run = "native_pipeline" in timings or (root / "native_pipeline").exists()
        cases = tuple(c for c in CASES if c != "native_full" or c in timings) if pipeline_run else CASES[:3]
    lines = [
        SUMMARY_HEADER,
        f"RUN={root}",
        "python=Python MTE；native=原生 Host 更新；native_full=完整原生提交；native_pipeline=提前预测+独立 DMA 队列。",
        "仅汇总已保存结果，不重跑、不读大 trace。缺失/失败阶段不补零；已完成计时与失败的 profile 分开标注。",
        table(["组", "正式计时", "Host/加载流采集", "两种重统计", "状态说明"], [s for s in states if s[0] in cases]),
        "\n正式请求：profiler/MSTX 关闭；Decode=首个至最后内容，TPOT=Decode/(实际输出token−1)。"
        "单位ms，TPOT为ms/token。",
    ]
    if "native_pipeline" in cases:
        lines.append(
            "native_pipeline：Host 回调仅含规划+入队，DMA 提交另线程统计；空泡仅使用全采集窗口，"
            "异步 DMA 不按 Host 回调窗口判定方向。Host 缩短不等同于 Decode 收益。"
        )
    rows = []
    for case in cases:
        row = timings.get(case)
        if row is None:
            continue
        samples = row["stream_samples"]
        stats = summarize_streams(samples)
        if stats["one_content_chunk"]:
            lines.append(f"{case}：{stats['one_content_chunk']} 个请求仅一个内容块，TPOT不能反映内部逐token节奏。")
        for label, key in (("原始", "raw"), ("3×IQR", "filtered")):
            cells = [
                f"{stats[m][key]['n']}/{number(stats[m][key]['mean'])}" if stats[m][key] else "—"
                for m in ("decode_ms", "tpot_ms", "ttft_ms")
            ]
            rows.append(
                [
                    case,
                    label,
                    row["requests"],
                    str(sorted({s["completion_tokens"] for s in samples})),
                    *cells,
                    number(statistics.mean(s["http_wall_us"] / 1000 for s in samples)),
                ]
            )
    lines.append(
        table(
            [
                "组",
                "口径",
                "正式请求n",
                "实际token",
                "Decode n/均值（主数据）",
                "TPOT n/均值",
                "TTFT n/均值",
                "HTTP E2E原始均值",
            ],
            rows,
        )
    )
    lines.append("IQR逐指标独立筛选；E2E列保持原始均值。不同服务分开执行，表中并列不代表同次配对或稳定性验收。")
    selected = [timings[c] for c in cases if c in timings]
    if len(selected) > 1:
        references = {
            (r["stream_samples"][0]["digest"], json.dumps(r["stream_samples"][0]["usage"], sort_keys=True))
            for r in selected
        }
        lines.append(f"已完成组输出/usage一致={len(references) == 1}；只覆盖这些有限请求。")
    host_rows, bubble_rows = [], []
    for case in cases:
        profile = profiles.get(case)
        if profile is None:
            continue
        for row in profile["scan"]["host_gaps"]:
            clean = row["outliers_us"]["filtered"] or {}
            host_rows.append(
                [
                    case,
                    row["direction"],
                    row["count"],
                    number(row["mean_us"]),
                    clean.get("n", 0),
                    number(clean.get("mean")),
                ]
            )
        track = profile["scan"].get("loading_track")
        if track and track["loaded_batches"] == 0:
            lines.append(f"{case}：加载流未匹配到搬运批次，空泡不纳入。")
            continue
        for row in profile["bubble_stats"]:
            stat = row["kept"] or {}
            bubble_rows.append(
                [
                    case,
                    track.get("name", "—") if track else "见原报告",
                    row.get("scope", row.get("direction")),
                    stat.get("n", 0),
                    number(stat.get("mean")),
                ]
            )
    lines.extend(
        [
            "\nHost间隔：后一回调开始−前一回调结束，包含零加载回调。单位μs。",
            table(["组", "方向", "原始n", "原始均值", "IQR保留n", "IQR均值"], host_rows),
            "\n加载流空泡：同次对齐坐标，MEMCPY和Transpose等任务均算占用；10≤间隔<5000μs，不再做IQR。",
            table(["组", "加载流", "范围/方向", "n", "均值μs"], bubble_rows),
        ]
    )
    subset = {c: overlaps[c] for c in cases if c in overlaps}
    if subset:
        try:
            overlap_lines = []
            render_overlap(subset, overlap_lines.append)
            lines.extend(["\nHost函数耗时与预取→按需重统计", *overlap_lines])
        except (ValueError, KeyError, TypeError) as exc:
            lines.append(f"重统计表不可生成：{exc}；不使用不完整表格。")
    lines.append("\n编译、逐请求JSON及报错堆栈保留在 run_details.log / run.log 和各组日志中。")
    return "\n".join(lines) + "\n"


def rebuild(root):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    children = [root / f"token{tokens}" for tokens in (5, 128) if (root / f"token{tokens}").is_dir()]
    if children:
        sections = []
        for child in children:
            sections.append(f"\n===== {child.name} =====\n" + rebuild(child))
        text = SUMMARY_HEADER + f"\nRUN={root}\n" + "".join(sections)
        status = root / "status.txt"
        if status.exists():
            history = status.read_text(encoding="utf-8")
            codes = [int(code) for code in re.findall(r"\bexit=(\d+)", history)]
            text += f"\n任务退出记录（不覆盖上表各阶段有效性）：BATCH_EXIT={int(any(codes))}\n{history}"
        for tokens in (5, 128):
            if not (root / f"token{tokens}").is_dir():
                text += f"\ntoken{tokens}：未运行。\n"
    else:
        data = gather(root)
        text = render(root, data)
        for case in CASES:
            directory = root / case
            if directory.is_dir():
                write_summary(directory, render(root, data, (case,)))
    write_summary(root, text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    text = rebuild(args.run)
    print(text, end="")
    print(f"SUMMARY_FILE={args.run.resolve() / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
