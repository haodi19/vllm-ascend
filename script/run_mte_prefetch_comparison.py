"""一次完成三组预取交接对照；汇总搬运计时、CPU 间隔与 profile 路径。"""

import argparse
import ast
import collections
import contextlib
import io
import json
import re
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from diagnose_moe_transfer_gaps import callback_cycles, load_trace
from diagnose_mte_tensor_stall import read_failures
from run_moe_host_sync import compare_weight_timings

ROOT = Path(__file__).resolve().parents[1]
CASES = (("callback", "callback", "mte"), ("mte", "mte", "mte"), ("hybrid", "mte", "callback"))
METRICS = (
    "to_prefetch_us",
    "prefetch_body_us",
    "to_reactive_us",
    "reactive_body_us",
    "to_reactive_start_us",
    "to_reactive_end_us",
)


def experiment_cases(original_copy):
    return (CASES[0], CASES[2]) if original_copy else CASES


def timing_from_log(text):
    """只恢复已完成的无 profiler 窗口；日志不含摘要时不声称输出一致。"""
    marker = "=== WEIGHT TIMING: CPU timers only; no active profiler/MSTX ==="
    if text.count(marker) != 1:
        raise ValueError("需要且只能有一个无 profiler 计时窗口")
    window = text.split(marker, 1)[1]
    end = re.search(r"^Request usage: (.+)$", window, re.M)
    if end is None:
        raise ValueError("计时窗口未完整结束")
    window = window[: end.end()]
    if not re.search(r"callbacks=\d+; errors=0; dropped=0", window):
        raise ValueError("计时样本存在错误或丢失")
    groups = {}
    current = None
    for line in window.splitlines():
        header = re.match(r"kind=(\w+) token_rows=(\d+) callbacks=(\d+).*loads_per_callback=(.+)", line)
        if header:
            kind, rows, count, loads = header.groups()
            current = dict(count=int(count), loads=ast.literal_eval(loads), metrics={})
            groups[(kind, int(rows))] = current
        elif current is not None:
            metric = re.fullmatch(r"(\w+)\s+([\d.eE+\-]+)(?:\s+[\d.eE+\-]+){5}\s*", line)
            if metric:
                current["metrics"][metric[1]] = float(metric[2])
    if not groups:
        raise ValueError("缺少分组计时")
    return groups, ast.literal_eval(end[1])


def partial_comparison(root, emit):
    groups = []
    usage = []
    for name in ("callback", "hybrid"):
        path = root / f"{name}_runner.log"
        rows, tokens = timing_from_log(path.read_text(encoding="utf-8", errors="replace"))
        groups.append(rows)
        usage.append(tokens)
    emit("只读恢复已完成计时；不启动模型、不写文件。日志不含输出摘要，未验证两组文本一致。")
    emit(f"token 用量一致={usage[0] == usage[1]}; 不包含 hybrid profile 间隔结果。")
    for key in sorted(groups[0].keys() & groups[1].keys()):
        left, right = (group[key] for group in groups)
        loads = [r["count"] * r["loads"]["mean"] for r in (left, right)]
        emit(
            f"kind={key[0]} rows={key[1]} callbacks={left['count']}/{right['count']} "
            f"experts={loads[0]:.0f}/{loads[1]:.0f}"
        )
        emit("metric callback_us hybrid_us delta_us")
        for metric in ("load_prepare", "copy_submit", "scale_refresh", "load_submit", "synchronize", "load_and_wait"):
            values = [
                r["metrics"]["load_submit"] + r["metrics"]["synchronize"]
                if metric == "load_and_wait"
                else r["metrics"][metric]
                for r in (left, right)
            ]
            emit(f"{metric} {values[0]:.3f} {values[1]:.3f} {values[1] - values[0]:+.3f}")
        if left["count"] != right["count"] or abs(loads[0] - loads[1]) > 1e-6:
            emit("加载量不同，不能直接把均值差归因为实现开销。")


def failure_summary(root, emit):
    """先保留测试进程的错误，避免把清理服务的异常误认成根因。"""
    found = False
    for name, _, _ in CASES:
        path = root / f"{name}_runner.log"
        if not path.is_file():
            continue
        found = True
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # run_moe_host_sync prints the runner exception before service excerpts.
        boundary = next((i for i, line in enumerate(lines) if line.startswith("SERVICE LOG TAIL:")), len(lines))
        runner = lines[:boundary]
        selected = set(range(max(0, len(runner) - 65), len(runner)))
        for i, line in enumerate(runner):
            if line.startswith(("FAILED", "Traceback")):
                selected.update(range(max(0, i - 3), min(len(runner), i + 20)))
        emit(f"RUNNER DIAGNOSTIC: {path}")
        emit("\n".join(runner[i] for i in sorted(selected)))
    if not found:
        raise ValueError(f"未找到 *_runner.log：{root}")


def check_outputs(before, after):
    for window in (None, "weight_timing"):
        left, right = (r if window is None else r.get(window) for r in (before, after))
        if left is None or right is None:
            raise ValueError("缺少计时窗口，不能比较")
        for key in ("digests", "usage"):
            if key not in left or key not in right or left[key] != right[key]:
                raise ValueError(f"输出不一致：window={window} field={key}；停止后续测试")


def compare_cycles(cases, emit):
    groups = {}
    for name, rows in cases.items():
        groups[name] = collections.defaultdict(list)
        for row in rows:
            groups[name][row["key"]].append(row)
    keys = set.intersection(*(set(group) for group in groups.values()))
    keys = {key for key in keys if len({len(group[key]) for group in groups.values()}) == 1}
    if not keys:
        emit("CPU CYCLES UNAVAILABLE: 没有层号/token 行数/样本数一致的三组样本")
        return
    selected = {name: [row for key in keys for row in group[key]] for name, group in groups.items()}
    emit(f"CPU CYCLES: samples/group={len(next(iter(selected.values())))}; units=us; 含 profiler 开销")
    names = list(selected)
    before = "mte" if "mte" in selected else "callback"
    emit("metric " + " ".join(names) + f" hybrid-minus-{before}")
    for metric in METRICS:
        means = {name: sum(r[metric] for r in rows) / len(rows) for name, rows in selected.items()}
        emit(
            f"{metric} " + " ".join(f"{means[name]:.3f}" for name in names) + f" {means['hybrid'] - means[before]:+.3f}"
        )
    emit("累计区间不与子项相加；专家加载量另见搬运表，不能仅凭间隔缩短宣布净收益。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=int)
    parser.add_argument("--read-partial", type=Path, help="只读汇总已有 callback/hybrid 日志中的完整 CPU 计时")
    parser.add_argument("--read-failure", type=Path, help="只读取已有三组对照目录，不启动模型或写文件")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-timeout", type=int, default=120)
    parser.add_argument(
        "--original-copy-final",
        action="store_true",
        help="最终两组验收：原回调与混合交接均使用原始 Tensor 拷贝，不开启逐任务同步",
    )
    args = parser.parse_args()
    if args.read_partial is not None:
        partial_comparison(args.read_partial, print)
        return 0
    if args.read_failure is not None:
        failure_summary(args.read_failure, print)
        return 0
    if sys.platform != "linux" or args.npu is None or args.npu < 0 or args.profile_timeout < 1:
        parser.error("请在 Linux 容器指定 NPU 编号及正数超时")
    root = (args.output_dir or ROOT / "bench_results_tmp" / f"prefetch_compare_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = experiment_cases(args.original_copy_final)
    quant_copy = "tensor" if args.original_copy_final else "prepacked"
    with (root / "summary.txt").open("w", encoding="utf-8") as summary:

        def emit(text):
            print(text, flush=True)
            summary.write(text + "\n")
            summary.flush()

        emit(f"SUMMARY FILE: {root / 'summary.txt'}")
        emit(f"最多启动 {len(cases)} 组，不自动重跑。callback=原回调；mte=全 MTE；hybrid=仅预取用原回调。")
        emit(f"MTE 拷贝={quant_copy}；关闭逐任务同步；配置继承 test_graph2.sh；每组预热1、计时3、profile1请求。")
        reports, cycles = {}, {}
        for name, backend, handoff in cases:
            directory = root / name
            log_path = root / f"{name}_runner.log"
            command = [
                sys.executable,
                "-B",
                str(ROOT / "script/run_moe_host_sync.py"),
                "--npu",
                str(args.npu),
                "--backend",
                backend,
                "--quant-copy",
                quant_copy,
                "--prefetch-handoff",
                handoff,
                "--timing-requests",
                "3",
                "--requests",
                "1",
                "--warmup-requests",
                "1",
                "--profile-timeout",
                str(args.profile_timeout),
                "--output-dir",
                str(directory),
            ]
            emit(f"START {name}; log={log_path}")
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            print(f"WAIT {name}；详细日志保存在绿区", flush=True)
                except BaseException:
                    child.send_signal(signal.SIGINT)
                    raise
            if child.returncode:
                emit(f"FAILED {name}: exit={child.returncode}; 停止后续组，不自动重跑")
                failure_summary(root, emit)
                try:
                    read_failures(directory, emit)
                except ValueError:
                    read_failures(log_path, emit)
                return 1
            report = json.loads((directory / backend / "result.json").read_text(encoding="utf-8"))
            if reports:
                try:
                    check_outputs(reports["callback"], report)
                except ValueError as exc:
                    emit(f"FAILED {name}: {exc}")
                    return 1
            reports[name] = report
            emit(f"PASS {name}: profile_status={report['profile_status']}; profiles={directory / 'profile'}")
            if name != "callback":
                with contextlib.redirect_stdout(io.StringIO()) as text:
                    compare_weight_timings({"callback": reports["callback"], "mte": report})
                emit(f"WEIGHT COMPARISON callback -> {name}\n{text.getvalue()}")
            if report["profile_status"] == "complete":
                paths = list((directory / "profile").rglob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
                if len(paths) == 1:
                    try:
                        callbacks, _, _, _, _ = load_trace(paths[0])
                        cycles[name] = callback_cycles(callbacks)
                    except (OSError, ValueError) as exc:
                        emit(f"PROFILE ANALYSIS FAILED {name}: {exc}")
        if "mte" in reports:
            with contextlib.redirect_stdout(io.StringIO()) as text:
                compare_weight_timings({"callback": reports["mte"], "mte": reports["hybrid"]})
            emit(
                "WEIGHT ABLATION full_mte -> hybrid (下表 callback 列代表 full_mte；mte 列代表 hybrid)\n"
                + text.getvalue()
            )
        if len(cycles) == len(cases):
            compare_cycles(cycles, emit)
        else:
            emit("CPU 间隔三组对照未完成；保留搬运数据，不把不完整 profile 算成功")
        emit("完成。请仅返回本 summary.txt；本轮不修改生产默认路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
