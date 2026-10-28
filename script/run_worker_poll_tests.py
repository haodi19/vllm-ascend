"""旧版 worker 第一阶段：yield/spin 与计时开关独立对照，不启动 MoE 模型。"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
from pathlib import Path
from uuid import uuid4

from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, run_step
from run_host_sync_stage2 import target_cmake_args
from run_host_sync_stage3 import select_build_directory
from run_verified_cpp_tests import SummaryTee


def cases(repeats, slots):
    for count in slots:
        for timing in (False, True):
            for repeat in range(repeats):
                modes = ("yield", "spin") if repeat % 2 == 0 else ("spin", "yield")
                for mode in modes:
                    yield count, timing, repeat, mode


def comparison(records):
    print("\n=== 对比：同一槽位数、同一计时开关；单位 us ===")
    for slots, timing in sorted({(r["slots"], r["worker_timing"]) for r in records}):
        selected = [r for r in records if (r["slots"], r["worker_timing"]) == (slots, timing)]
        print(f"slots={slots}; worker_timing={timing}")
        for metric in ("event_interval", "replay_sync_host"):
            for stat in ("mean", "p95", "max"):
                values = {
                    mode: statistics.mean(r[metric][stat] for r in selected if r["idle_mode"] == mode)
                    for mode in ("yield", "spin")
                }
                print(
                    f"{metric}.{stat}: yield={values['yield']:.3f} spin={values['spin']:.3f} "
                    f"delta={values['spin'] - values['yield']:+.3f}"
                )
    print("各轮分位数取平均，不是合并样本的分位数；事件区间含图执行，不能称为纯通知时延。")
    print("计时关闭组用于判断收益；开启组仅拆解开销。idle 样本含初始化/预热/校验间隙。")
    print("spin 的 idle_action 接近空计时开销；yield 前到获取 GIL 后不等于 NPU 发布到 CPU 发现。")
    print("本轮未改 mutex、扫描协议、绑核、权重搬运；未解决或测试真实 MoE scale 阻塞。")


def scan_cases(repeats, slots):
    for count in slots:
        for repeat in range(repeats):
            modes = ("all", "active") if repeat % 2 == 0 else ("active", "all")
            for mode in modes:
                yield count, False, repeat, "yield", mode


def scan_comparison(records):
    print("\n=== 扫描范围对照：yield 固定，内部计时关闭；单位 us ===")
    for slots in sorted({r["slots"] for r in records}):
        print(f"slots={slots}; active_slot={slots - 1}")
        for metric in ("event_interval", "replay_sync_host"):
            for stat in ("mean", "p50", "p95"):
                values = {
                    mode: statistics.mean(
                        r[metric][stat] for r in records if r["slots"] == slots and r["scan_mode"] == mode
                    )
                    for mode in ("all", "active")
                }
                print(
                    f"{metric}.{stat}: all={values['all']:.3f} active={values['active']:.3f} "
                    f"delta={values['active'] - values['all']:+.3f}"
                )
    print("分位数为各轮分位数的平均。仅验证声明槽位集合的独立交接，不代表实际模型收益或阻塞修复。")


def mutex_cases(repeats, slots):
    for count in slots:
        for repeat in range(repeats):
            for probe in ("locked", "atomic") if repeat % 2 == 0 else ("atomic", "locked"):
                yield count, False, repeat, "yield", "all", probe


def mutex_comparison(records):
    print("\n=== 空闲检查对照：全槽扫描、yield、内部计时关闭；单位 us ===")
    for slots in sorted({r["slots"] for r in records}):
        print(f"slots={slots}")
        for metric in ("event_interval", "replay_sync_host"):
            for stat in ("mean", "p50", "p95"):
                values = {
                    mode: statistics.mean(
                        r[metric][stat] for r in records if r["slots"] == slots and r["idle_probe"] == mode
                    )
                    for mode in ("locked", "atomic")
                }
                print(
                    f"{metric}.{stat}: locked={values['locked']:.3f} atomic={values['atomic']:.3f} "
                    f"delta={values['atomic'] - values['locked']:+.3f}"
                )
    print("只跳过无任务槽位的锁；任务领取、完成、管理路径仍加锁。分位数为各轮分位数平均，非模型收益。")


def affinity_cases(repeats, slots, cpu):
    for count in slots:
        for repeat in range(repeats):
            for selected in (-1, cpu) if repeat % 2 == 0 else (cpu, -1):
                yield count, False, repeat, "yield", "all", "locked", selected


def affinity_comparison(records):
    print("\n=== worker 绑核对照：全槽扫描、locked、yield、内部计时关闭；单位 us ===")
    for slots in sorted({r["slots"] for r in records}):
        print(f"slots={slots}")
        for metric in ("event_interval", "replay_sync_host"):
            for stat in ("mean", "p50", "p95"):
                values = {
                    bound: statistics.mean(
                        r[metric][stat] for r in records if r["slots"] == slots and (r["worker_cpu"] >= 0) == bound
                    )
                    for bound in (False, True)
                }
                print(
                    f"{metric}.{stat}: unbound={values[False]:.3f} bound={values[True]:.3f} "
                    f"delta={values[True] - values[False]:+.3f}"
                )
    print("仅绑定 worker；提交线程允许集合不变，仍可能与 worker 共享 CPU。分位数为各轮分位数平均。")


def combined_cases(repeats, slots):
    for count in slots:
        for repeat in range(repeats):
            pairs = [("all", "locked"), ("active", "locked"), ("all", "atomic"), ("active", "atomic")]
            for scan, probe in pairs if repeat % 2 == 0 else reversed(pairs):
                yield count, False, repeat, "yield", scan, probe, -1


def combined_comparison(records):
    print("\n=== 扫描/空闲锁组合：固定 yield、不绑核、关闭内部计时；单位 us ===")
    for slots in sorted({r["slots"] for r in records}):
        print(f"slots={slots}")
        for metric in ("event_interval", "replay_sync_host"):
            for stat in ("mean", "p50", "p95"):
                values = {
                    (scan, probe): statistics.mean(
                        r[metric][stat]
                        for r in records
                        if (r["slots"], r["scan_mode"], r["idle_probe"]) == (slots, scan, probe)
                    )
                    for scan in ("all", "active")
                    for probe in ("locked", "atomic")
                }
                print(f"{metric}.{stat}: " + " ".join(f"{s}/{p}={v:.3f}" for (s, p), v in values.items()))
                print(
                    f"  combined_minus_active={values['active', 'atomic'] - values['active', 'locked']:+.3f}; "
                    f"combined_minus_atomic={values['active', 'atomic'] - values['all', 'atomic']:+.3f}"
                )
    print("只检验两项收益是否重叠；单个活动槽位的独立图测试，不代表真实模型。分位数是各轮分位数平均。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=npu_number, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2048])
    parser.add_argument("--experiment", choices=("yield", "scan", "mutex", "affinity", "combined"), default="yield")
    parser.add_argument("--worker-cpu", type=int, help="affinity 实验指定 CPU；默认选允许集合中的最大编号")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("请在 Linux NPU 容器执行")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    worker_cpu = args.worker_cpu if args.worker_cpu is not None else allowed_cpus[-1]
    if args.experiment == "affinity" and (worker_cpu not in allowed_cpus or len(allowed_cpus) < 2):
        parser.error("affinity 需要至少两个允许 CPU，且指定编号必须在允许集合中")
    if args.worker_cpu is not None and args.experiment != "affinity":
        parser.error("--worker-cpu 仅用于 --experiment affinity")
    if not 1 <= args.rounds <= 4000 or not 1 <= args.repeats <= 4 or any(not 1 <= n <= 4096 for n in args.slots):
        parser.error("rounds=1..4000; repeats=1..4; slots=1..4096")
    root = (args.output_dir or REPOSITORY_ROOT / "bench_results" / f"worker_poll_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as summary,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, summary)),
    ):
        print(f"SUMMARY FILE: {root / 'summary.txt'}", flush=True)
        print("基线提交 942868dd；不加载模型、不采集 profiler。", flush=True)
        print(f"experiment={args.experiment}; 每项实验只切换自身选项，原始配置仍为默认。", flush=True)
        if args.experiment == "affinity":
            print(f"允许 CPU={allowed_cpus}; 选定 worker CPU={worker_cpu}；不保证空闲/独占或 NUMA 最优。")
        build = select_build_directory(None, False)
        py = [sys.executable, "-B"]
        steps = [
            (
                "build Host extension",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "./csrc/host_sync",
                    "--no-build-isolation",
                    "--no-deps",
                    "--force-reinstall",
                ],
                600,
            ),
            ("CPU lifecycle regression", [*py, "tests/ut/expert_offload/test_host_sync_session.py"], 120),
            ("CPU polling regression", [*py, "tests/ut/expert_offload/test_worker_polling.py"], 120),
            (
                "configure MTE kernels",
                [
                    "cmake",
                    "-S",
                    str(REPOSITORY_ROOT / "csrc/host_sync_device"),
                    "-B",
                    str(build),
                    "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest",
                    *target_cmake_args("ascend950pr_9579"),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                600,
            ),
            ("build MTE kernels", ["cmake", "--build", str(build), "-j", "2"], 600),
        ]
        for label, command, timeout in steps:
            if not run_step(label, command, timeout, show_success_output="regression" in label):
                return 1
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
        records = []
        configurations = (
            scan_cases(args.repeats, args.slots)
            if args.experiment == "scan"
            else ((*case, "all") for case in cases(args.repeats, args.slots))
        )
        configurations = (
            mutex_cases(args.repeats, args.slots)
            if args.experiment == "mutex"
            else ((*case, "locked") for case in configurations)
        )
        configurations = (
            affinity_cases(args.repeats, args.slots, worker_cpu)
            if args.experiment == "affinity"
            else ((*case, -1) for case in configurations)
        )
        if args.experiment == "combined":
            configurations = combined_cases(args.repeats, args.slots)
        for slots, timing, repeat, mode, scan_mode, idle_probe, cpu in configurations:
            label = (
                f"slots={slots} timing={timing} repeat={repeat + 1} mode={mode} "
                f"scan={scan_mode} probe={idle_probe} cpu={cpu}"
            )
            command = [
                *py,
                str(REPOSITORY_ROOT / "script/host_sync_benchmark.py"),
                "--backend",
                "mte",
                "--library",
                str(build / "lib/libhost_sync_device.so"),
                "--idle-mode",
                mode,
                "--slots",
                str(slots),
                "--scan-mode",
                scan_mode,
                "--idle-probe",
                idle_probe,
                "--worker-cpu",
                str(cpu),
                "--rounds",
                str(args.rounds),
                "--warmup",
                "20",
                "--max-polls",
                "1000000",
            ]
            if timing:
                command.append("--worker-timing")
            print(f"START {label}; timeout=120s", flush=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ok = run_step(label, command, 120, env=env, show_success_output=True)
            text = output.getvalue()
            (
                root / f"slots{slots}_{mode}_{scan_mode}_{idle_probe}_cpu{cpu}_timing{int(timing)}_r{repeat}.log"
            ).write_text(text, encoding="utf-8")
            if not ok:
                print(text, flush=True)
                print("停止后续组；失败样本不能纳入收益比较。", flush=True)
                return 1
            found = [
                json.loads(line[len("BENCH_RESULT ") :])
                for line in text.splitlines()
                if line.startswith("BENCH_RESULT ")
            ]
            if len(found) != 1:
                raise RuntimeError(f"缺少唯一 BENCH_RESULT: {label}")
            record = found[0]
            if record["worker_cpu"] != cpu or (cpu >= 0 and record["worker_affinity"] != [cpu]):
                raise RuntimeError(f"worker 绑核校验失败: {label}")
            if record["idle_probe"] != idle_probe:
                raise RuntimeError(f"空闲检查配置不匹配: {label}")
            if record["scan_mode"] != scan_mode:
                raise RuntimeError(f"扫描配置不匹配: {label}")
            if (record["slots"], record["worker_timing"], record["idle_mode"], record["rounds"]) != (
                slots,
                timing,
                mode,
                args.rounds,
            ):
                raise RuntimeError(f"结果配置不匹配: {label}")
            records.append(record)
            print("RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
        if args.experiment == "combined":
            combined_comparison(records)
        elif args.experiment == "affinity":
            affinity_comparison(records)
        elif args.experiment == "mutex":
            mutex_comparison(records)
        elif args.experiment == "scan":
            scan_comparison(records)
        else:
            comparison(records)
        print(f"完成；只需返回 {root / 'summary.txt'} 的内容。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
