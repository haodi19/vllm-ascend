# SPDX-License-Identifier: Apache-2.0
"""Run two services with verified C++ LRC and summarize existing profile intervals."""

import contextlib
import hashlib
import json
import math
import re
import sys
from uuid import uuid4

from mte_join_cost import LOAD_STATS
from run_moe_host_sync import CostProgressFile, parse_args, run
from run_verified_cpp_tests import REPO, SummaryTee

METRICS = (
    "total",
    "observe",
    "choose_victims",
    "outer",
    "host_without_sync",
    "load_prepare",
    "copy_submit",
    "scale_refresh",
    "load_submit",
    "synchronize",
)
MAX_GAP_ROWS = 24
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def display_numbers(value, key=""):
    """Round display only; keep full precision in the saved per-group report."""
    if isinstance(value, float):
        return round(value, 6 if key == "large_gap_fraction" else 3)
    if isinstance(value, dict):
        return {name: display_numbers(item, name) for name, item in value.items() if name != "scope"}
    if isinstance(value, list):
        return [display_numbers(item) for item in value]
    return value


def checked_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"无效测量值：{label}")
    return value


def checked_groups(report, backend):
    if report.get("units") != "us" or report.get("errors") != 0 or report.get("dropped") != 0:
        raise ValueError(f"{backend} 分项计时有错误、丢失或单位异常")
    if report.get("mstx_enabled") is not True:
        raise ValueError(f"{backend} 未启用 MSTX")
    groups = {}
    for group in report.get("groups", []):
        key = (group["kind"], group["token_rows"])
        if key in groups or key[0] not in ("prefetch", "reactive") or type(key[1]) is not int or key[1] <= 0:
            raise ValueError(f"{backend} 分组标识无效：{key}")
        if type(group["count"]) is not int or group["count"] <= 0 or not group.get("layers"):
            raise ValueError(f"{backend} 分组为空")
        for metric in METRICS:
            for stat in ("mean", "p95"):
                checked_number(group["metrics"][metric][stat], f"{backend}/{key}/{metric}/{stat}")
        for stat in LOAD_STATS:
            checked_number(group["loads"][stat], f"{backend}/{key}/loads/{stat}")
        groups[key] = group
    if not groups or sum(group["count"] for group in groups.values()) != report.get("count"):
        raise ValueError(f"{backend} 分组数量与总回调数不一致")
    fields = ("request_specs", "digests", "usage")
    if any(not isinstance(report.get(field), list) or not report[field] for field in fields):
        raise ValueError(f"{backend} 缺少请求计划、输出摘要或 token 用量")
    if len({len(report[field]) for field in fields}) != 1:
        raise ValueError(f"{backend} 请求与输出数量不一致")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or value == EMPTY_DIGEST
        for value in report["digests"]
    ):
        raise ValueError(f"{backend} 输出摘要为空或无效")
    if any(
        not isinstance(usage, dict)
        or type(usage.get("completion_tokens")) is not int
        or usage["completion_tokens"] <= 0
        for usage in report["usage"]
    ):
        raise ValueError(f"{backend} 没有有效生成 token")
    checked_number(report.get("request_us"), f"{backend}/request_mean")
    checked_number(report["request_stats_us"]["p95"], f"{backend}/request_p95")
    return groups


def print_profile(backend, report, root):
    print(f"PROFILE {backend}: status={report.get('profile_status')}; path={root / 'profile' / backend / 'cpp'}")
    for field in ("profile_errors", "profile_warnings"):
        items = report.get(field, [])
        if items:
            print(f"  {field} count={len(items)}: {str(items[0])[:300]}")
    traces = report.get("profile_trace_data", [])
    usable = [trace for trace in traces if trace.get("json_complete") is True and trace.get("usable") is True]
    if report.get("profile_status") not in ("complete", "complete_with_warnings") or not usable:
        print(f"GAPS {backend}: 无可验证的完整时间线；保留已取得的请求与 CPU 分项计时。")
        for check in report.get("profile_checks", [])[:2]:
            print(f"  TRACE_CHECK usable={check.get('usable')}; issues={check.get('issues', [])}")
        return False
    print(f"GAPS {backend}: usable_traces={len(usable)}/{len(traces)}；原 trace_view.json 可导入 MindStudio。")
    for trace in usable:
        print(f"TRACE {backend}: {trace['file']}")
        for label, field in (("HOST_CALLBACK_GAP", "host_gaps"), ("MEMCPY_GAP", "memcpy_gaps")):
            rows = trace.get(field, [])
            for row in rows[:MAX_GAP_ROWS]:
                print(label, backend, json.dumps(display_numbers(row), ensure_ascii=False, separators=(",", ":")))
            if len(rows) > MAX_GAP_ROWS:
                print(f"{label} {backend}: 省略 {len(rows) - MAX_GAP_ROWS} 条，完整数据在 profile_report.json。")
    return len(usable) == len(traces)


def print_comparison(reports, root):
    """Keep descriptive timings even when profile export fails; reject unequal outputs."""
    print("\n两组 LRC=cpp（observe/choose_victims 及配套状态管理）；权重准备/提交仍用 Python/Torch。")
    print("未开启融合规划、ACL 搬运或 CPU 重排。")
    print("计时含 profiler、MSTX 和内部计时开销；请求时延为辅助数据，不是无采集性能验收。")
    print("间隔覆盖整个采集窗口：Host 仅同进程相邻回调；MEMCPY 仅同设备同流，包含批次内/批次间。")
    print("大间隔阈值只按长度筛选，不自动识别同层预取→按需；MEMCPY 间隔不等于整卡空闲。")
    if set(reports) != {"callback", "mte"}:
        raise ValueError("两组未全部完成；已完成组的 profile_report.json 和原始 profile 保留")
    indexed = {backend: checked_groups(report, backend) for backend, report in reports.items()}
    profiles_ok = True
    for backend, report in reports.items():
        print(
            f"BACKEND {backend}: cache_policy={report.get('cache_policy_backend', 'not_recorded')}; "
            f"load={report.get('load_backend', 'not_recorded')}; h2d={report.get('h2d_backend', 'not_recorded')}"
        )
        print(
            f"REQUEST {backend}: count={len(report['digests'])} mean_ms={report['request_us'] / 1000:.3f} "
            f"p95_ms={report['request_stats_us']['p95'] / 1000:.3f}; "
            f"digest={','.join(dict.fromkeys(report['digests']))}; usage={json.dumps(report['usage'])}"
        )
        if len(report["digests"]) > 1:
            print(f"REQUEST_BOUNDARIES {backend}: 多个请求间停顿可能进入大间隔统计，不能当成层间空泡。")
        positions = report.get("engine_error_log_positions", {})
        print(f"ENGINE_ERRORS {backend}: {json.dumps(positions)}；只按日志顺序区分清理边界。")
        profiles_ok = print_profile(backend, report, root) and profiles_ok
    matching = {
        field: reports["callback"][field] == reports["mte"][field] for field in ("request_specs", "digests", "usage")
    }
    print("OUTPUT_MATCH", json.dumps(matching))
    if not all(matching.values()):
        raise ValueError("两组请求/输出不一致，比较失败；不据此给出性能收益")
    print("输出摘要与 token 用量一致，仅验证本轮有限请求一致性，不代表完整模型精度验收。")
    clean = all(
        report.get("engine_error_log_positions", {}).get("cleanup_marker_unique") is True
        and report["engine_error_log_positions"].get("before_cleanup") == 0
        for report in reports.values()
    )
    if not clean:
        print("清理前存在引擎错误或清理边界不明，以下仅列已取得的描述性数据，不作性能收益验收。")
    comparable = indexed["callback"].keys() == indexed["mte"].keys()
    for key in sorted(indexed["callback"].keys() | indexed["mte"].keys()):
        a, b = (indexed[backend].get(key) for backend in ("callback", "mte"))
        if a is None or b is None:
            comparable = False
            print(f"GROUP kind={key[0]} rows={key[1]}: 一组缺失，不能作同工作量比较")
            continue
        matched = (
            a["count"] == b["count"]
            and sorted(a["layers"]) == sorted(b["layers"])
            and all(math.isclose(a["loads"][stat], b["loads"][stat], rel_tol=1e-9, abs_tol=1e-9) for stat in LOAD_STATS)
        )
        comparable = comparable and matched
        print(
            f"GROUP kind={key[0]} rows={key[1]} callbacks={a['count']}/{b['count']} "
            f"experts={a['loads']['mean'] * a['count']:.1f}/{b['loads']['mean'] * b['count']:.1f} "
            f"workload_summary_match={matched}"
        )
        print("metric               callback_mean_us   mte_mean_us  delta_us  callback_p95_us  mte_p95_us")
        for metric in METRICS:
            first, second = a["metrics"][metric], b["metrics"][metric]
            print(
                f"{metric:20} {first['mean']:16.3f} {second['mean']:13.3f} "
                f"{second['mean'] - first['mean']:+9.3f} {first['p95']:16.3f} {second['p95']:11.3f}"
            )
        first, second = (
            sum(group["metrics"][name]["mean"] for name in ("load_submit", "synchronize")) for group in (a, b)
        )
        print(f"{'load_and_wait':20} {first:16.3f} {second:13.3f} {second - first:+9.3f}  p95不能分项相加")
    print(
        f"COMPARISON workload_summaries_match={comparable}; no_precleanup_engine_error={clean}; "
        f"both_profiles_usable={profiles_ok}"
    )
    print("分项时延与间隔是不同测量；上述统计不证明跨 CPU/NPU 时间线的因果关系。")
    return {
        "outputs_match": True,
        "workload_summaries_match": comparable,
        "no_precleanup_engine_error": clean,
        "both_profiles_usable": profiles_ok,
    }


def print_partial_reports(root):
    print("PARTIAL：完整对照未成功，以下仅保留各组已取得的数据，不能据此比较收益。")
    for backend in ("callback", "mte"):
        path = root / backend / "profile_report.json"
        if not path.is_file():
            print(f"PARTIAL {backend}: 未取得完成报告；service_log={root / backend / 'inference_cpp.log'}")
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            groups = checked_groups(report, backend)
            print(
                f"PARTIAL {backend}: count={report['count']}; request_mean_ms={report['request_us'] / 1000:.3f}; "
                f"profile={report.get('profile_status')}; cache_policy_backend={report.get('cache_policy_backend')}"
            )
            print(f"PARTIAL_ENGINE_ERRORS {backend}: {json.dumps(report.get('engine_error_log_positions', {}))}")
            print(f"PARTIAL_PROFILE {backend}: {root / 'profile' / backend / 'cpp'}")
            for key, group in sorted(groups.items()):
                metrics = {
                    name: round(group["metrics"][name]["mean"], 3)
                    for name in ("total", "observe", "choose_victims", "outer", "load_submit", "synchronize")
                }
                print(f"PARTIAL_METRICS {backend} kind={key[0]} rows={key[1]} mean_us={json.dumps(metrics)}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"PARTIAL {backend}: 报告无法验证（{type(exc).__name__}），保留于 {path}")


def main(argv=None):
    args = parse_args(
        ["--cache-policy-backend", "cpp", *(sys.argv[1:] if argv is None else argv)], profile_compare=True
    )
    root = (args.output_dir or REPO / "bench_results" / f"cpp_mte_profile_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as summary,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, summary)),
    ):
        print(f"SUMMARY FILE: {root / 'summary.txt'}", flush=True)
        print(
            f"配置：两组 cpp；MTE 严格 wait 延后到消费门控后；预热={args.warmup_requests}，"
            f"采集请求={args.requests}，max_tokens=5，大间隔阈值={args.large_gap_us:g} us。",
            flush=True,
        )
        print(
            f"详细日志：{root / 'run_details.log'}；停止采集最多等 {args.profile_timeout:g}s，失败继续另一组。",
            flush=True,
        )
        try:
            with (root / "run_details.log").open("w", encoding="utf-8") as details:
                progress = CostProgressFile(sys.stdout.console, details)
                with contextlib.redirect_stdout(progress):
                    try:
                        reports = run(args, root)
                    finally:
                        progress.flush()
            if not isinstance(reports, dict):
                raise RuntimeError(f"构建、回归或运行失败（exit={reports}），详见 run_details.log")
            print_comparison(reports, root)
            return 0
        except Exception as exc:
            print(
                f"FAILED: {type(exc).__name__}: {exc}; 已生成的报告/profile 保留，详情见 run_details.log。",
                flush=True,
            )
            print_partial_reports(root)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
