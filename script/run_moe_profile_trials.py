# SPDX-License-Identifier: Apache-2.0
"""Repeat four-way profiles with fresh services and balanced execution positions."""

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

from moe_experiment_tables import CASE_DESCRIPTIONS, VALID_INTERVAL_STATUSES, bubble_records, number, table
from moe_outlier_stats import outlier_summary
from moe_tpot import summarize_streams
from run_cpp_mte_profile_compare import main as run_comparison
from run_moe_host_sync import CostProgressFile, handoff_profile_cases, parse_args

DEFAULT_TRIALS = 4
DEFAULT_PROFILE_TIMEOUT = 600
DIRECTIONS = ("prefetch->reactive", "reactive->prefetch")
LABELS = tuple(CASE_DESCRIPTIONS)


def load_trial(root, trial, exit_code):
    reports, problems = {}, []
    for label in LABELS:
        path = root / label / "profile_report.json"
        try:
            reports[label] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{label}: {type(exc).__name__}")
    try:
        bubbles = bubble_records((root / "summary.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        bubbles = []
        problems.append(f"bubbles: {type(exc).__name__}")
    return dict(trial=trial, exit_code=exit_code, reports=reports, bubbles=bubbles, problems=problems)


def matched_workload(reports):
    if any(label not in reports for label in LABELS):
        return False
    reference = reports[LABELS[0]]
    for report in reports.values():
        if report.get("errors") != 0 or report.get("dropped") != 0 or report.get("count", 0) <= 0:
            return False
        positions = report.get("engine_error_log_positions", {})
        if positions.get("cleanup_marker_unique") is not True or positions.get("before_cleanup") != 0:
            return False
        for field in ("request_specs", "digests", "usage"):
            if not reference.get(field) or report.get(field) != reference[field]:
                return False
        window, baseline = report.get("request_window", {}), reference.get("request_window", {})
        for field in ("count", "request_specs", "digests", "usage"):
            if not baseline.get(field) or window.get(field) != baseline[field]:
                return False
    return True


def observations(entry):
    """Keep individual capture statistics; never pool their unequal event counts."""
    rows = []
    comparable = entry["exit_code"] == 0 and matched_workload(entry["reports"])
    for label in LABELS:
        report = entry["reports"].get(label, {})

        def append(metric, stats, *, label=label, **extra):
            if stats.get("n", 0) > 0 and stats.get("mean") is not None:
                rows.append(
                    dict(trial=entry["trial"], case=label, metric=metric, comparable=comparable, **stats, **extra)
                )

        window = report.get("request_window", {})
        if window.get("samples_us"):
            result = outlier_summary([v / 1000 for v in window["samples_us"]])
            append("E2E/ms", result["raw"], filtered_mean=result["filtered"]["mean"])
        if window.get("stream_samples"):
            streams = summarize_streams(window["stream_samples"])
            for name, key in (("TTFT/ms", "ttft_ms"), ("TPOT/ms", "tpot_ms")):
                stats = streams[key]
                append(name, stats["raw"], filtered_mean=stats["filtered"].get("mean"))
        if report.get("profile_status") not in VALID_INTERVAL_STATUSES:
            continue
        traces = report.get("profile_trace_data", [])
        if len(traces) != 1 or not traces[0].get("usable") or not traces[0].get("json_complete"):
            continue
        for direction in DIRECTIONS:
            gaps = [g for g in traces[0].get("host_gaps", []) if g.get("direction") == direction]
            if len(gaps) == 1:
                stats = gaps[0].get("outliers_us", {})
                append(
                    f"Host {direction}/us",
                    stats.get("raw", {}),
                    filtered_mean=(stats.get("filtered") or {}).get("mean"),
                    removed=stats.get("removed_count"),
                )
        for scope in ("full_capture", "callback_window"):
            gaps = [
                r for c, k, r in entry["bubbles"] if c == label and k == "TARGET_GAPS_US" and r.get("scope") == scope
            ]
            if len(gaps) == 1:
                r = gaps[0]
                append(
                    f"Bubble {scope}/us",
                    r.get("kept") or {},
                    excluded_short=r.get("excluded_short"),
                    excluded_long=r.get("excluded_long"),
                )
        states = [r for c, k, r in entry["bubbles"] if c == label and k == "OFFLOAD_STATUS"]
        association = (
            "partial" if any(r.get("partial") for r in states) else "checked" if len(states) == 1 else "unknown"
        )
        for direction in DIRECTIONS:
            gaps = [
                r
                for c, k, r in entry["bubbles"]
                if c == label and k == "TARGET_DIRECTION_GAPS_US" and r.get("direction") == direction
            ]
            if len(gaps) == 1:
                append(f"Bubble subset {direction}/us", gaps[0].get("kept") or {}, association=association)
    return rows


def render_trials(entries, expected, settings):
    rows = [row for entry in entries for row in observations(entry)]
    reference = next((entry for entry in entries if matched_workload(entry["reports"])), None)
    compatible = {}
    for entry in entries:
        compatible[entry["trial"]] = bool(reference) and all(
            entry["reports"].get(label, {}).get(field) == reference["reports"][label].get(field)
            for label in LABELS
            for field in ("request_specs", "digests", "usage")
        )
    for row in rows:
        row["comparable"] = row["comparable"] and compatible[row["trial"]]
    lines = [
        "# MoE重复采集对照",
        "",
        json.dumps(settings, ensure_ascii=False),
        "",
        f"已结束 {len(entries)}/{expected} 轮；每轮四个独立服务，每服务预热后单独采一份profile。",
        "每四轮循环移位，四种配置各占一次执行位置；不是随机化试验，仍可能受服务器状态影响。",
        "",
    ]
    lines += [
        table(["配置", "含义"], CASE_DESCRIPTIONS.items()),
        "",
        "Host间隔=前一回调结束到后一回调开始；raw与3×IQR filtered并列。",
        "加载流对全部设备任务取占用并集（含MEMCPY、Transpose），仅保留10≤gap<5000μs，不再IQR。",
        "full_capture含Prefill；callback_window为首末卸载回调间连续范围。空泡不等于整卡空闲。",
        "独立重复数是profile份数，不是回调数；跨轮按每份profile等权，不删除表现较差的轮次。",
        "",
    ]
    status = []
    for entry in entries:
        for label in LABELS:
            report = entry["reports"].get(label, {})
            status.append(
                [
                    entry["trial"],
                    label,
                    entry["exit_code"],
                    matched_workload(entry["reports"]),
                    compatible[entry["trial"]],
                    report.get("profile_export_status", "—"),
                    report.get("profile_status", "missing"),
                    report.get("count", "—"),
                ]
            )
        for problem in entry["problems"]:
            lines += [f"读取问题：第{entry['trial']}轮 {problem}"]
    lines += [
        "## 有效性",
        "",
        table(
            ["轮", "配置", "exit", "同轮输出/测量检查", "与首个有效轮输出一致", "原导出状态", "时间线验证", "回调数"],
            status,
        ),
        "",
        "recovered_intervals表示停止/导出流程未确认结束，但时间线通过独立校验。缺失值不填0。",
        "失败或跨轮输出不一致保留已测数值，但不纳入跨轮/同轮差值汇总。",
        "",
    ]
    coverage = []
    for label in LABELS:
        for metric in [*(f"Host {d}/us" for d in DIRECTIONS), "Bubble full_capture/us", "Bubble callback_window/us"]:
            present = [r["trial"] for r in rows if r["case"] == label and r["metric"] == metric]
            missing = [entry["trial"] for entry in entries if entry["trial"] not in present]
            coverage.append([label, metric, f"{len(present)}/{expected}", ",".join(map(str, missing)) or "无"])
    lines += ["## 采集覆盖", "", table(["配置", "指标", "取得份数/计划", "已结束但缺失的轮次"], coverage), ""]
    for title, prefix in (("无采集请求性能", None), ("Host间隔逐次结果", "Host "), ("加载流空泡逐次结果", "Bubble ")):
        selected = [r for r in rows if (r["metric"].startswith(prefix) if prefix else r["metric"].endswith("/ms"))]
        lines += [
            f"## {title}",
            "",
            table(
                [
                    "轮",
                    "配置",
                    "指标/范围",
                    "n",
                    "raw/阈值内均值",
                    "filtered均值",
                    "P50",
                    "P95",
                    "累计/ms",
                    "剔除/短/长",
                    "方向关联",
                ],
                [
                    [
                        r["trial"],
                        r["case"],
                        r["metric"],
                        r["n"],
                        number(r["mean"]),
                        number(r.get("filtered_mean")),
                        number(r.get("p50")),
                        number(r.get("p95")),
                        number(r.get("total", 0) / 1000) if r["metric"].startswith("Bubble ") and "total" in r else "—",
                        f"{r.get('removed', '—')}/{r.get('excluded_short', '—')}/{r.get('excluded_long', '—')}",
                        r.get("association", "—"),
                    ]
                    for r in selected
                ],
            ),
            "",
        ]
    lines += ["方向关联partial仅是可唯一关联的子集，不能用其收益代替完整方向；不纳入下方跨轮收益表。", ""]
    aggregate = []
    metrics = sorted({r["metric"] for r in rows if "subset" not in r["metric"]})
    for metric in metrics:
        for label in LABELS:
            selected = [r for r in rows if r["case"] == label and r["metric"] == metric and r["comparable"]]
            if not selected:
                continue
            values = [r["mean"] for r in selected]
            aggregate.append(
                [
                    label,
                    metric,
                    len(values),
                    ", ".join(number(v) for v in values),
                    number(statistics.mean(values)),
                    number(min(values)),
                    number(max(values)),
                    number(statistics.stdev(values)) if len(values) > 1 else "—",
                ]
            )
    lines += [
        "## 各次均值的波动（raw/阈值内；每次等权）",
        "",
        table(["配置", "指标", "有效重复数", "逐次均值", "平均", "最小", "最大", "样本标准差"], aggregate),
        "",
    ]
    paired = []
    pairs = (
        ("callback_legacy", "callback"),
        ("callback", "mte_default"),
        ("mte_default", "mte_combined"),
        ("callback_legacy", "mte_combined"),
    )
    for metric in metrics:
        modes = ("mean", "total") if metric.startswith("Bubble ") else ("mean",)
        for mode in modes:
            for baseline, candidate in pairs:
                changes = []
                for entry in entries:
                    values = {
                        r["case"]: r.get(mode)
                        for r in rows
                        if r["trial"] == entry["trial"] and r["metric"] == metric and r["comparable"]
                    }
                    a, b = values.get(baseline), values.get(candidate)
                    if a is not None and b is not None and a > 0:
                        changes.append(100 * (b / a - 1))
                if changes:
                    paired.append(
                        [
                            f"{candidate} vs {baseline}",
                            metric + " " + mode,
                            len(changes),
                            ", ".join(f"{v:+.2f}%" for v in changes),
                            f"{statistics.mean(changes):+.2f}%",
                            f"{sum(v < 0 for v in changes)}/{len(changes)}",
                        ]
                    )
    lines += [
        "## 同轮配对变化（负数表示降低）",
        "",
        table(["对照", "指标", "有效配对数", "逐轮变化", "平均变化", "降低次数"], paired),
        "",
        "平均值与累计空泡分别比较；样本数变化时二者可不同向。这里只描述重复性，不作统计显著性或因果证明。",
        "",
    ]
    return "\n".join(lines)


def save_summary(root, entries, expected, settings):
    text = render_trials(entries, expected, settings)
    # Refresh after every completed trial, so an interrupted later trial retains prior results.
    (root / "summary.txt").write_text(text, encoding="utf-8")
    (root / "experiment_tables.md").write_text(text, encoding="utf-8")


def run_token_lengths(root, trials, remaining):
    """Keep token-length experiments separate, with one combined copyable report."""
    completed = []
    for tokens in (5, 128):
        directory = root / f"tokens{tokens}"
        code = 1
        try:
            code = main(
                [
                    *remaining,
                    "--trials",
                    str(trials),
                    "--output-dir",
                    str(directory),
                    "--max-tokens",
                    str(tokens),
                    *(["--ignore-eos"] if tokens == 128 else []),
                ]
            )
        finally:
            completed.append((tokens, code))
            sections = [
                "# 5-token与128-token重复实验",
                "",
                "两种长度串行执行，各自四组配置；不跨token长度合并统计。",
                "",
            ]
            for length, exit_code in completed:
                path = root / f"tokens{length}" / "summary.txt"
                sections += [
                    f"## max_tokens={length}; exit={exit_code}",
                    "",
                    path.read_text(encoding="utf-8") if path.is_file() else "未生成结果。",
                    "",
                ]
            text = "\n".join(sections)
            (root / "summary.txt").write_text(text, encoding="utf-8")
            (root / "experiment_tables.md").write_text(text, encoding="utf-8")
            print(f"TOKEN_SWEEP tokens={tokens} exit={code}; SUMMARY={root / 'summary.txt'}", flush=True)
        if code:
            return code
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, choices=range(2, 9))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--both-token-lengths",
        action="store_true",
        help="Run 5 tokens then 128 tokens (ignore_eos only for 128), each with its own trials",
    )
    own, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if own.both_token_lengths and any(
        arg == "--ignore-eos" or arg.split("=", 1)[0] == "--max-tokens" for arg in remaining
    ):
        parser.error("--both-token-lengths assigns --max-tokens and --ignore-eos automatically")
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    common = [
        "--include-legacy-callback",
        "--measure-tpot",
        "--cache-policy-backend",
        "cpp",
        "--bubble-stream-name",
        "Stream 56",
        "--profile-timeout",
        str(DEFAULT_PROFILE_TIMEOUT),
        *remaining,
    ]
    args = parse_args(common, profile_compare=True, handoff_compare=True)
    if own.output_dir is None:
        parser.error("--output-dir is required (new directory)")
    if args.requests != 1 or args.skip_profile or not args.bubble_stream_name or args.case_rotation:
        parser.error("requires --requests 1, profiles and bubble stream; rotations are assigned automatically")
    root = own.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if own.both_token_lengths:
        return run_token_lengths(root, own.trials, remaining)
    settings = dict(
        npu=args.npu,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        warmup_requests=args.warmup_requests,
        timing_requests=args.timing_requests,
        stream=args.bubble_stream_name,
        profile_timeout=args.profile_timeout,
    )
    entries = []
    save_summary(root, entries, own.trials, settings)
    for trial in range(1, own.trials + 1):
        directory = root / f"trial_{trial:02d}"
        rotation = (trial - 1) % len(LABELS)
        order = [c[0] for c in handoff_profile_cases(True, rotation)]
        print(f"TRIAL {trial}/{own.trials}: {' -> '.join(order)}", flush=True)
        exit_code = 1
        try:
            with (root / f"trial_{trial:02d}.log").open("w", encoding="utf-8") as log:
                progress = CostProgressFile(sys.stdout, log)
                with contextlib.redirect_stdout(progress), contextlib.redirect_stderr(log):
                    try:
                        exit_code = run_comparison(
                            [*common, "--case-rotation", str(rotation), "--output-dir", str(directory)],
                            handoff_compare=True,
                        )
                    finally:
                        progress.flush()
        finally:
            entries.append(load_trial(directory, trial, exit_code))
            save_summary(root, entries, own.trials, settings)
            print(f"TRIAL {trial} exit={exit_code}; SUMMARY={root / 'summary.txt'}", flush=True)
        if exit_code:
            print(f"停止后续轮次；已有数据保留。详情：{root / f'trial_{trial:02d}.log'}", flush=True)
            return exit_code
    print(f"完成；只需返回 {root / 'summary.txt'}，逐次profile保存在trial_*/profile。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
