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
from run_moe_host_sync import (
    WORKER_PROFILE_CASES,
    CostProgressFile,
    failure_excerpt,
    parse_args,
    run,
    worker_profile_cases,
)
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


def print_comparison(reports, root, labels=("callback", "mte"), *, worker_cases=None):
    """Keep descriptive timings even when profile export fails; reject unequal outputs."""
    print(f"\n本轮{len(labels)}组 LRC=cpp（observe/choose_victims 及配套状态管理）；权重准备/提交仍用 Python/Torch。")
    print("未开启融合规划、ACL 搬运或 CPU 重排。")
    print("采集窗口的 REQUEST 和分项计时含 profiler、MSTX 开销；无采集整体时延单独列为 REQUEST_NO_PROFILE。")
    print("间隔覆盖整个采集窗口：Host 仅同进程相邻回调；MEMCPY 仅同设备同流，包含批次内/批次间。")
    print("大间隔阈值只按长度筛选，不自动识别同层预取→按需；MEMCPY 间隔不等于整卡空闲。")
    if set(reports) != set(labels):
        raise ValueError("对照组未全部完成；已完成组的 profile_report.json 和原始 profile 保留")
    windows = {label: reports[label].get("request_window") for label in labels}
    if any(windows.values()):
        if not all(windows.values()):
            raise ValueError("部分组缺少独立请求计时窗口，不能比较")
        print("独立整体时延：profiler尚未启动，回调分项计时未开启；包装检查仍在，不是零插桩二进制。")
        for label, window in windows.items():
            print_request_window(label, window)
        if any(
            windows[label][key] != windows[labels[0]][key]
            for label in labels[1:]
            for key in ("count", "request_specs", "digests", "usage")
        ):
            raise ValueError("独立请求窗口输出/请求配置不一致，不能据此比较收益")
        print("NO_PROFILE_OUTPUT_MATCH=True；以下百分比为请求均值变化，不是统计显著性检验。")
        for label in labels[1:]:
            value = windows[label]["stats_us"]["mean"]
            first = windows[labels[0]]["stats_us"]["mean"]
            last = windows[labels[-1]]["stats_us"]["mean"]
            print(
                f"NO_PROFILE_DELTA {label}: vs_first={100 * (value / first - 1):+.2f}%; "
                f"vs_last={100 * (value / last - 1):+.2f}%"
            )
    worker_cases = WORKER_PROFILE_CASES if worker_cases is None else worker_cases
    worker_compare = labels == tuple(case[0] for case in worker_cases)
    if worker_compare:
        for label, scan, probe, idle in worker_cases:
            report = reports[label]
            if (
                report.get("host_backend") != "mte"
                or report.get("load_stream_mode") != "ordinary"
                or report.get("cache_policy_backend") != "cpp"
                or report.get("worker_policy") != dict(scan=scan, probe=probe, idle=idle)
            ):
                raise ValueError(f"{label}: wrong worker experiment identity")
    if "ordinary" in labels:
        for label in labels:
            report = reports[label]
            if (report.get("host_backend"), report.get("load_stream_mode"), report.get("cache_policy_backend")) != (
                "callback" if label == "callback" else "mte",
                "ordinary" if label == "callback" else label,
                "cpp",
            ):
                raise ValueError(f"{label}: wrong MTE stream/policy identity")
    indexed = {backend: checked_groups(report, backend) for backend, report in reports.items()}
    profiles_ok = True
    for backend, report in reports.items():
        print(
            f"BACKEND {backend}: host={report.get('host_backend', backend)}; "
            f"load_stream={report.get('load_stream_mode', 'not_recorded')}; "
            f"cache_policy={report.get('cache_policy_backend', 'not_recorded')}; "
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
        if worker_compare:
            print(f"WORKER {backend}: {json.dumps(report['worker_policy'])}")
        profiles_ok = print_profile(backend, report, root) and profiles_ok
    matching = {
        field: all(reports[labels[0]][field] == reports[label][field] for label in labels[1:])
        for field in ("request_specs", "digests", "usage")
    }
    print("OUTPUT_MATCH", json.dumps(matching))
    if not all(matching.values()):
        raise ValueError("对照组请求/输出不一致，比较失败；不据此给出性能收益")
    print("输出摘要与 token 用量一致，仅验证本轮有限请求一致性，不代表完整模型精度验收。")
    clean = all(
        report.get("engine_error_log_positions", {}).get("cleanup_marker_unique") is True
        and report["engine_error_log_positions"].get("before_cleanup") == 0
        for report in reports.values()
    )
    if not clean:
        print("清理前存在引擎错误或清理边界不明，以下仅列已取得的描述性数据，不作性能收益验收。")
    comparable = all(indexed[labels[0]].keys() == indexed[label].keys() for label in labels[1:])
    for key in sorted(set().union(*(groups.keys() for groups in indexed.values()))):
        groups = [indexed[backend].get(key) for backend in labels]
        if any(group is None for group in groups):
            comparable = False
            print(f"GROUP kind={key[0]} rows={key[1]}: 一组缺失，不能作同工作量比较")
            continue
        a = groups[0]
        matched = all(
            a["count"] == b["count"]
            and sorted(a["layers"]) == sorted(b["layers"])
            and all(math.isclose(a["loads"][stat], b["loads"][stat], rel_tol=1e-9, abs_tol=1e-9) for stat in LOAD_STATS)
            for b in groups[1:]
        )
        comparable = comparable and matched
        print(
            f"GROUP kind={key[0]} rows={key[1]} order={','.join(labels)} "
            f"callbacks={'/'.join(str(group['count']) for group in groups)} "
            f"experts={'/'.join(format(group['loads']['mean'] * group['count'], '.1f') for group in groups)} "
            f"workload_summary_match={matched}"
        )
        if worker_compare and len(labels) > 3:
            print_worker_group_metrics(groups, labels)
            continue
        if len(labels) == 3:
            print_three_group_metrics(groups, labels)
            continue
        a, b = groups
        print(
            f"metric               {labels[0]}_mean_us  {labels[1]}_mean_us  delta_us  "
            f"{labels[0]}_p95_us  {labels[1]}_p95_us"
        )
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
        f"all_profiles_usable={profiles_ok}"
    )
    print("分项时延与间隔是不同测量；上述统计不证明跨 CPU/NPU 时间线的因果关系。")
    if worker_compare and labels == tuple(case[0] for case in WORKER_PROFILE_CASES):
        print("baseline_repeat与baseline配置相同；若两端基线波动接近候选收益，不能宣称优化稳定有效。")
    return {
        "outputs_match": True,
        "workload_summaries_match": comparable,
        "no_precleanup_engine_error": clean,
        "both_profiles_usable": profiles_ok,
    }


def print_three_group_metrics(groups, labels):
    print("metric               " + "  ".join(f"{label}(mean/p95_us)" for label in labels))
    for metric in METRICS:
        print(
            f"{metric:20} "
            + "  ".join(
                f"{group['metrics'][metric]['mean']:.3f}/{group['metrics'][metric]['p95']:.3f}" for group in groups
            )
        )
    totals = [sum(group["metrics"][metric]["mean"] for metric in ("load_submit", "synchronize")) for group in groups]
    print(f"{'load_and_wait':20} " + "  ".join(f"{value:.3f}" for value in totals) + "  仅均值，不累加p95")
    for metric in ("copy_submit", "load_submit", "synchronize", "load_and_wait", "total"):
        values = totals if metric == "load_and_wait" else [group["metrics"][metric]["mean"] for group in groups]
        print(
            f"DELTA {metric}: sync_launch-callback={values[1] - values[0]:+.3f} us; "
            f"ordinary-callback={values[2] - values[0]:+.3f} us; ordinary-sync_launch={values[2] - values[1]:+.3f} us"
        )


def print_worker_group_metrics(groups, labels):
    print("metric               " + "  ".join(f"{label}(mean/p95_us)" for label in labels))
    for metric in METRICS:
        print(
            f"{metric:20} "
            + "  ".join(
                f"{group['metrics'][metric]['mean']:.3f}/{group['metrics'][metric]['p95']:.3f}" for group in groups
            )
        )
    totals = [sum(group["metrics"][metric]["mean"] for metric in ("load_submit", "synchronize")) for group in groups]
    print(f"{'load_and_wait':20} " + "  ".join(f"{value:.3f}" for value in totals) + "  仅均值，不累加p95")
    for index, label in enumerate(labels[1:], 1):
        total_delta = groups[index]["metrics"]["total"]["mean"] - groups[0]["metrics"]["total"]["mean"]
        print(
            f"DELTA {label}: load_and_wait-first_baseline={totals[index] - totals[0]:+.3f} us; "
            f"load_and_wait-last_baseline={totals[index] - totals[-1]:+.3f} us; "
            f"total-first_baseline={total_delta:+.3f} us"
        )


def print_request_window(label, window):
    samples = window["samples_us"]
    if len(samples) != window["count"] or not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError(f"{label}: invalid no-profiler request samples")
    stats = window["stats_us"]
    middle = max(1, len(samples) // 2)
    first_half = sum(samples[:middle]) / middle
    second_half = sum(samples[middle:]) / len(samples[middle:]) if samples[middle:] else first_half
    print(
        f"REQUEST_NO_PROFILE {label}: n={len(samples)}; "
        + " ".join(f"{key}_ms={stats[key] / 1000:.3f}" for key in ("mean", "p50", "p95", "min", "max"))
        + f" stddev_ms={window['stddev_us'] / 1000:.3f}; "
        f"first_half_mean_ms={first_half / 1000:.3f}; second_half_mean_ms={second_half / 1000:.3f}"
    )
    print(f"REQUEST_SAMPLES_MS {label}: " + json.dumps([round(x / 1000, 3) for x in samples]))


def print_partial_reports(root, labels=("callback", "mte")):
    print("PARTIAL：完整对照未成功，以下仅保留各组已取得的数据，不能据此比较收益。")
    for backend in labels:
        window_path = root / backend / "request_window.json"
        if window_path.is_file():
            try:
                print_request_window(backend, json.loads(window_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"PARTIAL_REQUEST_WINDOW {backend}: {exc}")
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


def print_failure_logs(root, labels):
    for label in labels:
        path = root / label / "inference_cpp.log"
        if path.is_file() and not (root / label / "profile_report.json").exists():
            print(f"FAILED_CASE {label}; service_log={path}")
            with path.open("rb") as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 256 * 1024))
                tail = log.read().decode("utf-8", errors="replace")
            print(failure_excerpt(tail))


def print_atomic_trials(reports, trials):
    print("\n独立服务复验：负数表示 atomic 更快；逐轮比较，不把请求当作独立服务重复。")
    deltas = []
    for trial in range(1, trials + 1):
        suffix = "" if trial == 1 else f"_{trial}"
        baseline, atomic = (reports[f"{name}{suffix}"]["request_window"] for name in ("baseline", "atomic"))
        first, second = (window["stats_us"]["mean"] for window in (baseline, atomic))
        delta = 100 * (second / first - 1)
        deltas.append(delta)
        median_delta = 100 * (atomic["stats_us"]["p50"] / baseline["stats_us"]["p50"] - 1)
        order = "baseline->atomic" if trial % 2 else "atomic->baseline"
        print(
            f"ATOMIC_TRIAL {trial}: order={order}; n={baseline['count']}/{atomic['count']}; "
            f"baseline_ms={first / 1000:.3f}; atomic_ms={second / 1000:.3f}; "
            f"delta_ms={(second - first) / 1000:+.3f}; delta_pct={delta:+.3f}%; "
            f"median_delta_pct={median_delta:+.3f}%"
        )
    print(
        f"ATOMIC_TRIAL_SUMMARY trials={trials}; faster_pairs={sum(value < 0 for value in deltas)}; "
        f"mean_pair_delta_pct={sum(deltas) / len(deltas):+.3f}%; "
        f"range_pct=[{min(deltas):+.3f}, {max(deltas):+.3f}]"
    )
    print("该汇总是描述性结果；需同时查看顺序、组内波动及局部交接间隔，不自动宣称统计显著。")


def print_combination_trials(reports, trials):
    print("\n组合复验：负数表示候选更快；同时比较组合总效果及在 allocated 上添加 atomic 的增量。")
    for trial in range(1, trials + 1):
        suffix = "" if trial == 1 else f"_{trial}"
        order = "baseline->allocated->combined" if trial % 2 else "combined->allocated->baseline"
        for reference, candidate in (("baseline", "allocated"), ("baseline", "combined"), ("allocated", "combined")):
            a, b = (reports[f"{label}{suffix}"]["request_window"] for label in (reference, candidate))
            first, second = (window["stats_us"]["mean"] for window in (a, b))
            print(
                f"COMBINATION_TRIAL {trial}: order={order}; {reference}->{candidate}; n={a['count']}/{b['count']}; "
                f"reference_ms={first / 1000:.3f}; candidate_ms={second / 1000:.3f}; "
                f"delta_ms={(second - first) / 1000:+.3f}; delta_pct={100 * (second / first - 1):+.3f}%; "
                f"median_delta_pct={100 * (b['stats_us']['p50'] / a['stats_us']['p50'] - 1):+.3f}%"
            )
    print("局部收益不预设可叠加；结合各轮 Host 间隔和无采集请求判断，不以搬运分项代替交接效果。")


def main(argv=None, *, stream_compare=False, three_way=False, worker_compare=False):
    stream_compare = stream_compare or three_way
    args = parse_args(
        ["--cache-policy-backend", "cpp", *(sys.argv[1:] if argv is None else argv)],
        profile_compare=True,
        stream_compare=stream_compare,
        three_way=three_way,
        worker_compare=worker_compare,
    )
    labels = ("sync_launch", "ordinary") if stream_compare else ("callback", "mte")
    prefix = "mte_load_stream" if stream_compare else "cpp_mte_profile"
    if three_way:
        labels = ("callback", "sync_launch", "ordinary")
        prefix = "moe_three_way"
    if worker_compare:
        labels = tuple(case[0] for case in worker_profile_cases(args.atomic_trials, args.combination_trials))
        prefix = "moe_worker_profile"
    root = (args.output_dir or REPO / "bench_results" / f"{prefix}_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as summary,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, summary)),
    ):
        if worker_compare and args.combination_trials:
            print("MTE普通流＋cpp LRC；组合测试独立服务顺序：" + " → ".join(labels))
            print("baseline=all/locked/yield；allocated=allocated/locked/yield；combined=allocated/atomic/yield。")
            print("不绑核、不启用C++轮询内部计时；每组保存profile，默认策略不变。")
        elif worker_compare and args.atomic_trials:
            print("MTE普通流＋cpp LRC；独立服务顺序：" + " → ".join(labels))
            print("每轮仅改变 locked/atomic 预检查；scan=all、idle=yield。逐轮交换顺序，全部保存profile。")
        elif worker_compare:
            print("MTE普通流＋cpp LRC：baseline → allocated → atomic → spin → baseline_repeat。")
            print("三个候选各自只改一个worker开关；不绑核、不启用C++轮询内部计时；全部保存profile。")
        elif three_way:
            print("三组顺序运行：callback普通流 → MTE SyncLaunchStream → MTE普通流；全部cpp LRC。")
            print("两组MTE均使用消费者处严格等待；普通流只对齐提交方式，不替换交接机制或工作线程。")
        elif stream_compare:
            print("两组均为MTE晚等待＋cpp LRC；只改变加载流：sync_launch → ordinary。")
            print("普通流若请求超时则终止本轮，不跳过设备完成检查，不继续推理。")
        print(f"SUMMARY FILE: {root / 'summary.txt'}", flush=True)
        if args.timing_requests:
            print(
                f"整体时延另测每组{args.timing_requests}次请求，在profiler和回调计时开始之前执行；随后单独采profile。"
            )
        print(
            f"配置：各组 cpp；MTE 严格 wait 延后到消费门控后；预热={args.warmup_requests}，"
            f"采集请求={args.requests}，max_tokens=5，大间隔阈值={args.large_gap_us:g} us。",
            flush=True,
        )
        print(
            f"详细日志：{root / 'run_details.log'}；停止采集最多等 {args.profile_timeout:g}s，导出失败继续后续组。",
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
            if worker_compare and args.combination_trials:
                cases = worker_profile_cases(combination_trials=args.combination_trials)
                for trial in range(1, args.combination_trials + 1):
                    suffix = "" if trial == 1 else f"_{trial}"
                    group = tuple(f"{name}{suffix}" for name in ("baseline", "allocated", "combined"))
                    policies = tuple(next(case for case in cases if case[0] == label) for label in group)
                    print(f"\n=== 组合对照第 {trial} 轮；表格按 baseline、allocated、combined 排列 ===")
                    print_comparison({label: reports[label] for label in group}, root, group, worker_cases=policies)
                print_combination_trials(reports, args.combination_trials)
            elif worker_compare and args.atomic_trials:
                for trial in range(1, args.atomic_trials + 1):
                    suffix = "" if trial == 1 else f"_{trial}"
                    pair = (f"baseline{suffix}", f"atomic{suffix}")
                    policies = tuple((label, "all", probe, "yield") for label, probe in zip(pair, ("locked", "atomic")))
                    print(f"\n=== 独立对照第 {trial} 轮；表格统一按 baseline、atomic 排列 ===")
                    print_comparison({label: reports[label] for label in pair}, root, pair, worker_cases=policies)
                print_atomic_trials(reports, args.atomic_trials)
            else:
                print_comparison(reports, root, labels)
            return 0
        except Exception as exc:
            print(
                f"FAILED: {type(exc).__name__}: {exc}; 已生成的报告/profile 保留，详情见 run_details.log。",
                flush=True,
            )
            print_partial_reports(root, labels)
            if stream_compare or worker_compare:
                print_failure_logs(root, labels)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
