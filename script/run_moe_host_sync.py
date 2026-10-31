# SPDX-License-Identifier: Apache-2.0
"""Run real MoE inference with original callback or optional MTE dispatch."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from inspect_moe_profile import positive_gap_us, scan_trace
from inspect_mte_runtime_task import inspect_run
from mte_join_cost import prepare_cost_source, print_cost_comparison
from run_host_sync_stage1 import REPOSITORY_ROOT, npu_number, run_step
from run_host_sync_stage2 import target_cmake_args
from run_host_sync_stage3 import select_build_directory
from run_verified_cpp_tests import (
    VERIFICATION_MAX_TOKENS,
    VERIFICATION_PROMPTS,
    SummaryTee,
    ensure_profile_storage,
    inference,
    launch_spec,
    progress,
)

MAX_ENGINE_ERROR_LINE_CHARS = 800

# Each candidate changes exactly one worker option; the final baseline exposes drift.
WORKER_PROFILE_CASES = (
    ("baseline", "all", "locked", "yield"),
    ("allocated", "allocated", "locked", "yield"),
    ("atomic", "all", "atomic", "yield"),
    ("spin", "all", "locked", "spin"),
    ("baseline_repeat", "all", "locked", "yield"),
)


def worker_profile_cases(atomic_trials=0):
    """Alternate pair order across fresh services; retain the original matrix."""
    if not atomic_trials:
        return WORKER_PROFILE_CASES
    cases = []
    for trial in range(1, atomic_trials + 1):
        suffix = "" if trial == 1 else f"_{trial}"
        pair = [
            (f"baseline{suffix}", "all", "locked", "yield"),
            (f"atomic{suffix}", "all", "atomic", "yield"),
        ]
        cases.extend(pair if trial % 2 else reversed(pair))
    return tuple(cases)


class CostProgressFile:
    """Keep full details locally; show only phase progress on the console."""

    def __init__(self, console, details):
        self.console, self.details = console, details
        self.pending = ""
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.details.write(text)
            self.details.flush()
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if line.startswith(("START:", "PASS:", "WAIT:", "DONE:", "FAILED:", "STORAGE:", "=== REAL MOE")):
                    self.console.write(line + "\n")
                    self.console.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.details.flush()
            self.console.flush()


def run_cost_comparison(args, root, console):
    print("成本对照：原回调 vs 消费者处严格等待的 MTE；原始权重拷贝，不限核。", flush=True)
    print("关闭 Runtime DEBUG、图 dump、逐项加载诊断、启动观测、MSTX 与 profiler。", flush=True)
    print(
        f"每组 {args.warmup_requests} 次预热 → {args.requests} 次内部计时关闭请求 → "
        f"{args.requests} 次分项计时请求；max_tokens=5。仅两个服务依次运行。",
        flush=True,
    )
    print(f"详细日志：{root / 'cost_details.log'}；本轮不采集 profile。", flush=True)
    with (
        (root / "cost_details.log").open("w", encoding="utf-8") as details,
        contextlib.redirect_stdout(CostProgressFile(console, details)),
    ):
        reports = run(args, root)
    if not isinstance(reports, dict):
        print("FAILED: 构建或前置检查未通过，未形成成本对照；查看 cost_details.log。", flush=True)
        return 1
    print_cost_comparison(reports)
    return 0


def configured_source(
    source,
    backend,
    library,
    diagnose_load=False,
    scan_mode="all",
    idle_probe="locked",
    idle_mode="yield",
    diagnose_native_wait=False,
    diagnose_split_scale=False,
    diagnose_scale_context=False,
    diagnose_prefetch_report_wait=False,
    diagnose_prefetch_poll_budget=0,
    diagnose_load_vector_cores=0,
    diagnose_graph_dump=False,
    diagnose_graph_dump_dir="moe_graph_debug",
    diagnose_routing_vector_cores=0,
    diagnose_compute_cube_cores=0,
    diagnose_dsa_cube_cores=0,
    diagnose_native_stream_limits=False,
    diagnose_prefetch_wait_at_join=False,
    no_load_diagnostics=False,
    no_graph_dump=False,
    diagnose_prefetch_wait_before_gate=False,
    mte_load_stream="ordinary",
):
    if mte_load_stream not in ("sync_launch", "ordinary"):
        raise ValueError("invalid MTE load stream")
    if no_graph_dump and diagnose_graph_dump:
        raise ValueError("no-graph-dump conflicts with diagnose-graph-dump")
    if no_load_diagnostics and any(
        (diagnose_load, diagnose_native_wait, diagnose_split_scale, diagnose_scale_context, diagnose_load_vector_cores)
    ):
        raise ValueError("no-load-diagnostics conflicts with per-load diagnosis")
    diagnose_split_scale = diagnose_split_scale or diagnose_scale_context
    diagnose_prefetch_report_wait = diagnose_prefetch_report_wait or bool(diagnose_prefetch_poll_budget)
    start = re.search(r"(?m)^vllm\s+serve\s+", source)
    if start is None:
        raise ValueError("Missing vllm serve command")
    words = [word for word in shlex.split(source[start.start() :] + "\n", comments=True) if word.strip()]
    index = words.index("--additional-config") + 1
    config = json.loads(words[index])
    offload = config["expert_offload_config"]
    offload["host_sync_backend"] = backend
    offload["host_sync_load_stream"] = mte_load_stream
    if diagnose_prefetch_wait_at_join or "host_sync_diagnose_prefetch_wait_at_join" in offload:
        offload["host_sync_diagnose_prefetch_wait_at_join"] = diagnose_prefetch_wait_at_join and backend == "mte"
    if diagnose_prefetch_wait_before_gate or "host_sync_diagnose_prefetch_wait_before_gate" in offload:
        offload["host_sync_diagnose_prefetch_wait_before_gate"] = (
            diagnose_prefetch_wait_before_gate and backend == "mte"
        )
    if diagnose_native_stream_limits:
        offload["host_sync_diagnose_native_stream_limits"] = backend == "mte"
    if diagnose_dsa_cube_cores:
        offload["host_sync_diagnose_dsa_cube_cores"] = diagnose_dsa_cube_cores if backend == "mte" else 0
    if diagnose_compute_cube_cores:
        offload["host_sync_diagnose_compute_cube_cores"] = diagnose_compute_cube_cores if backend == "mte" else 0
    if diagnose_routing_vector_cores:
        offload["host_sync_diagnose_routing_vector_cores"] = diagnose_routing_vector_cores if backend == "mte" else 0
    if diagnose_graph_dump:
        offload["host_sync_diagnose_graph_dump"] = backend == "mte"
        offload["host_sync_diagnose_graph_dump_dir"] = str(diagnose_graph_dump_dir)
    if diagnose_load_vector_cores:
        offload["host_sync_diagnose_load_vector_cores"] = diagnose_load_vector_cores if backend == "mte" else 0
    offload["host_sync_library"] = str(library) if backend == "mte" else ""
    offload["host_sync_scan_mode"] = scan_mode if backend == "mte" else "all"
    offload["host_sync_idle_probe"] = idle_probe if backend == "mte" else "locked"
    offload["host_sync_idle_mode"] = idle_mode if backend == "mte" else "yield"
    if diagnose_load:
        offload["host_sync_diagnose_load"] = backend == "mte"
    if diagnose_native_wait or diagnose_split_scale:
        offload["host_sync_diagnose_load"] = backend == "mte"
        offload["host_sync_diagnose_native_wait"] = backend == "mte"
    if diagnose_split_scale:
        offload["host_sync_diagnose_split_scale"] = backend == "mte"
    if diagnose_scale_context:
        offload["host_sync_diagnose_scale_context"] = backend == "mte"
    if diagnose_prefetch_report_wait:
        offload["host_sync_diagnose_prefetch_report_wait"] = backend == "mte"
    if diagnose_prefetch_poll_budget:
        offload["host_sync_diagnose_prefetch_poll_budget"] = diagnose_prefetch_poll_budget if backend == "mte" else 0
    if no_load_diagnostics:
        if any(
            offload.get(key)
            for key in (
                "host_sync_diagnose_split_scale",
                "host_sync_diagnose_scale_context",
                "host_sync_diagnose_load_vector_cores",
            )
        ):
            raise ValueError("no-load-diagnostics cannot inherit altered scale copies or load core limits")
        # Override inherited launch-script values explicitly, not just CLI defaults.
        offload["host_sync_diagnose_load"] = False
        offload["host_sync_diagnose_native_wait"] = False
    if no_graph_dump:
        offload["host_sync_diagnose_graph_dump"] = False
    words[index] = json.dumps(config)
    return source[: start.start()] + shlex.join(words) + "\n"


def failure_excerpt(text):
    """Bound copied output; full stacks and successful copies stay in service log."""
    lines = text.splitlines()
    selected = set()
    for marker, count in (
        ("[MOE-MTE LOAD]", 18),
        ("[MOE-MTE LAYOUT]", 1),
        ("STREAM_HANDLE_READY", 1),
        ("[MOE-MTE WORKER]", 1),
        ("[MOE-MTE VECTOR_LIMIT]", 1),
        ("[MOE-MTE GRAPH_DUMP_BEGIN]", 4),
        ("[MOE-MTE GRAPH_DUMP_END]", 4),
        ("[MOE-MTE DIAGNOSTIC BEGIN]", 1),
        ("active_callback=", 1),
        ("callback=_", 1),
        ("[MOE-MTE FATAL]", 1),
        ("[MOE-MTE STREAM_MAP]", 1),
        ("[MOE-MTE WAIT_CONTROL]", 1),
        ("[MOE-MTE WAIT_PLACEMENT]", 8),
        ("[MOE-MTE AIV_EXIT]", 8),
        ("[MOE-MTE AFTER_AIV_EXIT]", 8),
        ("[MOE-MTE EXIT_COUNTS]", 4),
        ("[MOE-MTE PENDING]", 1),
        ("[MOE-MTE ACL_DETAIL]", 1),
        ("[MOE-MTE SCALE_CONTEXT]", 1),
        ("[MOE-MTE PROBE_TARGET]", 2),
    ):
        selected.update([index for index, line in enumerate(lines) if marker in line][-count:])
    if not any("[MOE-MTE FATAL]" in line for line in lines):
        errors = [index for index, line in enumerate(lines) if "Error" in line or "ERROR" in line]
        selected.update(errors[:4] + errors[-4:])
        if errors:
            selected.update(range(max(0, errors[0] - 3), min(len(lines), errors[0] + 4)))
        else:
            selected.update(range(max(0, len(lines) - 8), len(lines)))
    return "\n".join(
        lines[index] if len(lines[index]) <= 2000 else lines[index][:2000] + " ... [full text in service log]"
        for index in sorted(selected)
    )


def assess_profile_report(
    report, service_log, profile_dir=None, *, require_mte=False, retain_trace_data=False, large_gap_us=None
):
    """Request success and an exported filename do not validate profiler output."""
    errors, warnings, checks, trace_data = [], [], [], []
    for line in service_log.splitlines():
        if "profiler.py" in line and "run failed" in line:
            errors.append(line.strip()[:500])
        elif "External init callback must run in same thread" in line:
            # Upstream Kineto logs this when client initialization uses another
            # thread. It does not itself establish an invalid Ascend timeline.
            # Retain the diagnostic, then inspect the exported trace separately.
            warnings.append(line.strip()[:500])
    if profile_dir is not None and report.get("profile_status") == "complete":
        traces = sorted(profile_dir.rglob("ASCEND_PROFILER_OUTPUT/trace_view.json"))
        if not traces:
            errors.append("No exported trace_view.json despite completed export")
        for path in traces:
            try:
                options = {} if large_gap_us is None else {"large_gap_us": large_gap_us}
                with progress(f"stream profile intervals: {path}") if retain_trace_data else contextlib.nullcontext():
                    check = scan_trace(path, require_mte=require_mte, **options)
                if retain_trace_data:
                    trace_data.append(dict(file=str(path), **check))
                checks.append(
                    dict(
                        file=str(path),
                        json_complete=check["json_complete"],
                        usable=check["usable"],
                        issues=check["issues"],
                    )
                )
            except (ValueError, OSError) as exc:
                errors.append(f"{path.name}: {exc}")
    if errors:
        report["profile_status"] = "invalid"
    elif report.get("profile_status") == "complete":
        if (checks and not all(check["usable"] for check in checks)) or report.get("mstx_trace_occurrences") == 0:
            report["profile_status"] = "unverified"
        elif warnings:
            report["profile_status"] = "complete_with_warnings" if checks else "unverified"
    report["profile_errors"] = list(dict.fromkeys(errors))[:4]
    report["profile_warnings"] = list(dict.fromkeys(warnings))[:4]
    report["profile_checks"] = checks
    if retain_trace_data:
        report["profile_trace_data"] = trace_data
    report["engine_error_seen"] = "EngineDeadError" in service_log
    report["engine_error_log_positions"] = engine_error_log_positions(service_log)
    return report


def engine_error_log_positions(service_log):
    """Describe merged-log position, not error timing or cause."""
    lines = service_log.splitlines()
    boundaries = []
    for index, line in enumerate(lines):
        if "[MOE-MTE RUNNER]" not in line:
            continue
        try:
            record = json.loads(line.split("[MOE-MTE RUNNER]", 1)[1].strip())
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("event") == "cleanup_begin":
            boundaries.append(index)
    errors = [index for index, line in enumerate(lines) if "EngineDeadError" in line]
    boundary = boundaries[0] if len(boundaries) == 1 else None
    return dict(
        cleanup_marker_unique=boundary is not None,
        before_cleanup=sum(index < boundary for index in errors) if boundary is not None else None,
        after_cleanup=sum(index > boundary for index in errors) if boundary is not None else None,
        total=len(errors),
        examples=[
            dict(
                line=index + 1,
                text=lines[index][:MAX_ENGINE_ERROR_LINE_CHARS],
                truncated=len(lines[index]) > MAX_ENGINE_ERROR_LINE_CHARS,
            )
            for index in (errors[:1] + errors[-1:] if len(errors) > 1 else errors)
        ],
        basis="merged log order only; not causal attribution",
    )


def compare_verification_reports(reports):
    """Reject missing/empty results before interpreting any output match."""
    for backend in ("callback", "mte"):
        report = reports[backend]
        if report.get("errors") != 0 or report.get("dropped") != 0 or report.get("count", 0) <= 0:
            raise ValueError(f"{backend}: missing or failed callback accounting")
        for field in ("request_specs", "digests", "usage"):
            if len(report.get(field, [])) != len(VERIFICATION_PROMPTS):
                raise ValueError(f"{backend}: incomplete {field}")
        if any(
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest == hashlib.sha256(b"").hexdigest()
            for digest in report["digests"]
        ):
            raise ValueError(f"{backend}: missing or empty generated text")
        if any(item.get("completion_tokens", 0) <= 0 for item in report["usage"]):
            raise ValueError(f"{backend}: empty completion usage")
        expected = [
            dict(
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                max_tokens=VERIFICATION_MAX_TOKENS,
                temperature=0,
            )
            for prompt in VERIFICATION_PROMPTS
        ]
        if report["request_specs"] != expected:
            raise ValueError(f"{backend}: request plan differs from the fixed suite")
    left, right = reports["callback"], reports["mte"]
    text_match = [a == b for a, b in zip(left["digests"], right["digests"])]
    usage_match = [a == b for a, b in zip(left["usage"], right["usage"])]
    result = dict(
        status="passed" if all(text_match + usage_match) else "mismatch",
        text_match=text_match,
        usage_match=usage_match,
        scope="three fixed prompts; exact text/usage only; no logits or performance claim",
    )
    print("PAIR_RESULT " + json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("Callback/MTE generated text or request usage differs")
    return result


def prepare_runtime_artifacts(source, target):
    """Copy missing build artifacts only; never replace source code or existing files."""
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("runtime source must be a different, already working checkout")
    package = source / "vllm_ascend"
    custom = package / "_cann_ops_custom"
    if not (package / "_build_info.py").is_file() or not (custom / "vendors/custom_transformer").is_dir():
        raise ValueError("runtime source lacks _build_info.py or complete _cann_ops_custom package")
    files = [package / "_build_info.py"]
    if (package / "_version.py").is_file():
        files.append(package / "_version.py")
    files.extend(path for path in custom.rglob("*") if path.is_file())
    files.extend(path for path in package.rglob("*.so*") if path.is_file() and custom not in path.parents)
    copied = 0
    for path in files:
        destination = target / path.relative_to(source)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1
    print(f"Runtime artifacts: copied missing files={copied}; existing files preserved; source={source}", flush=True)


def parse_args(argv=None, *, profile_compare=False, stream_compare=False, three_way=False, worker_compare=False):
    if worker_compare and (not profile_compare or stream_compare or three_way):
        raise ValueError("worker comparison requires its own profile preset")
    if three_way and not stream_compare:
        raise ValueError("three-way comparison requires stream comparison preset")
    if stream_compare and not profile_compare:
        raise ValueError("stream comparison requires profile comparison preset")
    parser = argparse.ArgumentParser(description=__doc__)
    if worker_compare:
        parser.add_argument(
            "--atomic-trials",
            type=int,
            default=0,
            choices=range(5),
            help="Fresh baseline/atomic pairs with alternating order (0: original matrix)",
        )
    parser.add_argument(
        "--compare-join-cost",
        action="store_true",
        help="Compare callback with late strict MTE wait; no profiler/debug; concise summary",
    )
    parser.add_argument("--npu", type=npu_number, required=True)
    parser.add_argument("--backend", choices=("both", "callback", "mte"), default="both")
    parser.add_argument(
        "--cache-policy-backend", choices=("python", "cpp"), default="python", help="LRC observe/choose implementation"
    )
    parser.add_argument("--worker-scan", choices=("all", "allocated"), default="all")
    parser.add_argument("--worker-probe", choices=("locked", "atomic"), default="locked")
    parser.add_argument("--worker-idle", choices=("yield", "spin"), default="yield")
    parser.add_argument(
        "--diagnose-prefetch-wait-at-join",
        action="store_true",
        help="Diagnostic: keep early publication, defer strict prefetch wait to consumer join",
    )
    parser.add_argument("--skip-profile", action="store_true", help="Run finite requests without starting profiler")
    parser.add_argument(
        "--diagnose-prefetch-wait-before-gate",
        action="store_true",
        help="Restore early strict wait while keeping mapping after the consumer gate",
    )
    parser.add_argument(
        "--verification-suite",
        action="store_true",
        help="Compare three fixed prompts with up to 32 output tokens, callback then MTE",
    )
    parser.add_argument(
        "--no-load-diagnostics",
        action="store_true",
        help="Use ordinary batched loads from the first batch; retain required completion waits",
    )
    parser.add_argument(
        "--trace-startup", action="store_true", help="Observe process exits for the wait-at-join diagnosis"
    )
    parser.add_argument(
        "--diagnose-prefetch-poll-budget",
        type=int,
        default=0,
        help="Bounded AIV followed by mandatory report completion guard; diagnostic only",
    )
    parser.add_argument(
        "--diagnose-prefetch-report-wait",
        action="store_true",
        help="Control experiment: replace prefetch AIV wait only; keep native worker and MTE publish",
    )
    parser.add_argument(
        "--diagnose-scale-context",
        action="store_true",
        help="Probe independent D2D targets in real worker; implies split-scale diagnosis",
    )
    parser.add_argument(
        "--diagnose-split-scale",
        action="store_true",
        help="Diagnostic only: split scale H2D and device layout copy, implies native wait diagnosis",
    )
    parser.add_argument(
        "--diagnose-native-wait",
        action="store_true",
        help="Diagnose raw ACL stream completion before torch synchronization; abort on ACL error",
    )
    parser.add_argument(
        "--diagnose-load-stages",
        action="store_true",
        help="Serialize the first eight MTE load batches for diagnosis, not performance",
    )
    parser.add_argument("--test-script", type=Path, default=REPOSITORY_ROOT / "script/test_graph2.sh")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--diagnose-graph-dump", action="store_true")
    parser.add_argument(
        "--no-graph-dump",
        action="store_true",
        help="Verification suite without NPUGraph debug APIs; strict handoffs remain enabled",
    )
    parser.add_argument("--diagnose-routing-vector-cores", type=int, default=0, choices=range(57))
    parser.add_argument("--diagnose-compute-cube-cores", type=int, default=0, choices=range(29))
    parser.add_argument("--diagnose-dsa-cube-cores", type=int, default=0, choices=range(29))
    parser.add_argument("--diagnose-native-stream-limits", action="store_true")
    parser.add_argument("--diagnose-load-vector-cores", type=int, default=0, choices=range(57))
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--profile-timeout", type=int, default=120)
    if profile_compare:
        parser.add_argument("--large-gap-us", type=positive_gap_us, default=100.0)
        parser.add_argument(
            "--timing-requests",
            type=int,
            default=0,
            help="Repeated requests before profiler/timers start; profile requests remain separate",
        )
    parser.add_argument("--port", type=int)
    parser.add_argument("--timing-port", type=int, default=18155)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--runtime-source", type=Path, help="Copy missing runtime build artifacts from an existing working checkout"
    )
    args = parser.parse_args(argv)
    args.profile_compare = profile_compare
    args.stream_compare = stream_compare
    args.three_way = three_way
    args.worker_compare = worker_compare
    if getattr(args, "timing_requests", 0) < 0:
        parser.error("--timing-requests must be nonnegative")
    if getattr(args, "atomic_trials", 0) and not args.timing_requests:
        parser.error("--atomic-trials requires positive --timing-requests")
    if args.requests is None:
        args.requests = 1 if profile_compare else (10 if args.compare_join_cost else 3)
    if args.warmup_requests is None:
        args.warmup_requests = 3 if args.compare_join_cost or profile_compare else 1
    if args.compare_join_cost or profile_compare:
        conflicts = [name for name, value in vars(args).items() if name.startswith("diagnose_") and value]
        if (
            conflicts
            or args.verification_suite
            or args.trace_startup
            or args.backend != "both"
            or (args.worker_scan, args.worker_probe, args.worker_idle) != ("all", "locked", "yield")
            or (profile_compare and (args.compare_join_cost or args.skip_profile or args.cache_policy_backend != "cpp"))
        ):
            parser.error("comparison requires both backends and original worker policy; no diagnosis switches")
        args.diagnose_prefetch_wait_at_join = True
        args.no_load_diagnostics = True
        args.no_graph_dump = True
        args.skip_profile = not profile_compare
    if args.diagnose_prefetch_wait_before_gate and not (
        args.backend == "mte"
        and args.diagnose_prefetch_wait_at_join
        and args.diagnose_graph_dump
        and args.no_load_diagnostics
        and not args.verification_suite
    ):
        parser.error("early-wait diagnosis requires MTE, join capture, graph dump and no load diagnostics")
    if args.no_graph_dump and (
        args.diagnose_graph_dump or not (args.verification_suite or args.compare_join_cost or profile_compare)
    ):
        parser.error("--no-graph-dump requires --verification-suite or cost mode; conflicts with graph dump")
    if args.trace_startup and (args.backend not in ("mte", "both") or not args.diagnose_prefetch_wait_at_join):
        parser.error("--trace-startup requires MTE and --diagnose-prefetch-wait-at-join")
    if args.no_load_diagnostics:
        if args.backend not in ("mte", "both") or not args.diagnose_prefetch_wait_at_join:
            parser.error("--no-load-diagnostics requires MTE and --diagnose-prefetch-wait-at-join")
        if any(
            (
                args.diagnose_load_stages,
                args.diagnose_native_wait,
                args.diagnose_split_scale,
                args.diagnose_scale_context,
                args.diagnose_load_vector_cores,
            )
        ):
            parser.error("--no-load-diagnostics conflicts with per-load diagnosis")
    if args.verification_suite and not (
        args.backend == "both"
        and args.skip_profile
        and args.no_load_diagnostics
        and args.diagnose_prefetch_wait_at_join
        and (args.diagnose_graph_dump or args.no_graph_dump)
        and args.requests == len(VERIFICATION_PROMPTS)
        and args.warmup_requests == 1
    ):
        parser.error(
            "verification suite requires both backends, join wait, an explicit graph debug choice, "
            "no load diagnostics, "
            "skip profile, 1 warmup and 3 requests"
        )
    if args.diagnose_load_vector_cores:
        args.diagnose_native_wait = True
    if not 0 <= args.diagnose_prefetch_poll_budget <= 10_000_000:
        parser.error("prefetch poll budget must be 0..10000000")
    if args.diagnose_prefetch_poll_budget:
        args.diagnose_prefetch_report_wait = True
    if args.diagnose_scale_context:
        args.diagnose_split_scale = True
    if args.diagnose_split_scale:
        args.diagnose_native_wait = True
    if args.diagnose_native_wait:
        args.diagnose_load_stages = True
    if sys.platform != "linux":
        parser.error("Run inside the Linux NPU container")
    if min(args.requests, args.warmup_requests, args.startup_timeout, args.request_timeout, args.profile_timeout) < 1:
        parser.error("Counts/timeouts must be positive")
    return args


def main():
    args = parse_args()
    root = (args.output_dir or REPOSITORY_ROOT / "bench_results" / f"moe_host_sync_{uuid4().hex[:8]}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    with (
        (root / "summary.txt").open("w", encoding="utf-8") as file,
        contextlib.redirect_stdout(SummaryTee(sys.stdout, file)),
    ):
        print(f"SUMMARY FILE: {root / 'summary.txt'}", flush=True)
        try:
            if args.compare_join_cost:
                return run_cost_comparison(args, root, sys.stdout.console)
            return run(args, root)
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            for log in root.glob(f"*/inference_{args.cache_policy_backend}.log"):
                print(
                    f"SERVICE LOG TAIL: {log}\n" + failure_excerpt(log.read_text(encoding="utf-8", errors="replace")),
                    flush=True,
                )
            if args.diagnose_native_wait:
                try:
                    inspect_run(root)
                except Exception as log_error:
                    print(f"Runtime log extraction unavailable: {log_error}", flush=True)
            return 1


def run(args, root):
    stream_compare = getattr(args, "stream_compare", False)
    worker_compare = getattr(args, "worker_compare", False)
    cache_policy_backend = getattr(args, "cache_policy_backend", "python")
    profile_compare = getattr(args, "profile_compare", False)
    if profile_compare:
        ensure_profile_storage(root, stage="before build")
    if cache_policy_backend not in ("python", "cpp"):
        raise ValueError("Only Python or verified C++ LRC is supported")
    if getattr(args, "trace_startup", False):
        for name in ("test_mte_startup_bootstrap.py", "test_startup_diagnostics.py"):
            if not run_step(
                "startup observation CPU regression", [sys.executable, "-B", f"tests/ut/expert_offload/{name}"], 120
            ):
                print(f"FAILED: startup observation preflight {name}; model not launched", flush=True)
                return 1
    if args.runtime_source is not None:
        prepare_runtime_artifacts(args.runtime_source, REPOSITORY_ROOT)
    if cache_policy_backend == "cpp":
        for label, command in (
            (
                "build verified C++ LRC extension",
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "./csrc/lrc_policy",
                    "--no-build-isolation",
                    "--no-deps",
                ],
            ),
            (
                "required C++ LRC parity",
                [sys.executable, "-B", "tests/ut/expert_offload/test_lrc_cpp.py", "--require-cpp"],
            ),
        ):
            if not run_step(label, command, 600):
                return 1
    backends = ("callback", "mte") if args.backend == "both" else (args.backend,)
    build = select_build_directory(None, False)
    library = build / "lib/libhost_sync_device.so"
    if "mte" in backends:
        commands = [
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
            ),
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
            ),
            ("build MTE kernels", ["cmake", "--build", str(build), "-j", "2"]),
            ("CPU worker regression", [sys.executable, "-B", "tests/ut/expert_offload/test_worker_polling.py"]),
            ("MoE dispatch regression", [sys.executable, "-B", "tests/ut/expert_offload/test_moe_host_sync.py"]),
        ]
        if args.diagnose_prefetch_wait_at_join:
            commands.append(
                (
                    "MoE deferred prefetch regression",
                    [sys.executable, "-B", "tests/ut/expert_offload/test_mte_prefetch_join.py"],
                )
            )
        for label, command in commands:
            if not run_step(label, command, 600):
                return 1
        if worker_compare and not run_step(
            "CPU Host lifecycle regression",
            [sys.executable, "-B", "tests/ut/expert_offload/test_host_sync_session.py"],
            120,
        ):
            return 1
    source = args.test_script.read_text(encoding="utf-8")
    cost_mode = getattr(args, "compare_join_cost", False)
    if cost_mode or profile_compare:
        source = prepare_cost_source(source)
    _, command, _, _ = launch_spec(source, cache_policy_backend, args.npu, args.port, args.timing_port)
    port = int(command[command.index("--port") + 1])
    if not 1024 <= port <= 65535 or not 1024 <= args.timing_port <= 65535 or port == args.timing_port:
        raise ValueError("Use distinct serve/timing ports in 1024..65535")
    reports = {}
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    print(
        f"REAL MOE: LRC={cache_policy_backend} in both groups; model/MTP/prefetch inherited from test_graph2.sh",
        flush=True,
    )
    print(
        "MTE uses graph polling dispatch; load stream selection is printed per case. "
        f"Eager callback invocation is unchanged. Timers enabled; profiler={not args.skip_profile}.",
        flush=True,
    )
    cases = (
        (("sync_launch", "mte", "sync_launch"), ("ordinary", "mte", "ordinary"))
        if stream_compare
        else tuple((backend, backend, "ordinary") for backend in backends)
    )
    if getattr(args, "three_way", False):
        cases = (("callback", "callback", "ordinary"), *cases)
    worker_policies = {
        label: (scan, probe, idle)
        for label, scan, probe, idle in worker_profile_cases(getattr(args, "atomic_trials", 0))
    }
    if worker_compare:
        cases = tuple((label, "mte", "ordinary") for label in worker_policies)
    for label, backend, load_stream_mode in cases:
        if profile_compare:
            ensure_profile_storage(root / "profile" / label / cache_policy_backend, stage=f"before service {label}")
        scan, probe, idle = (
            worker_policies[label] if worker_compare else (args.worker_scan, args.worker_probe, args.worker_idle)
        )
        directory = root / label
        directory.mkdir()
        run_args = SimpleNamespace(
            output_dir=directory,
            profile_run_dir=root / "profile" / label,
            npu=args.npu,
            port=port,
            timing_port=args.timing_port,
            requests=args.requests,
            timing_requests=getattr(args, "timing_requests", 0),
            warmup_requests=args.warmup_requests,
            max_tokens=VERIFICATION_MAX_TOKENS if getattr(args, "verification_suite", False) else 5,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            profile_timeout=args.profile_timeout,
            strict_profile=False,
            mstx=not cost_mode,
            skip_profile=args.skip_profile,
            compare_join_cost=cost_mode,
            profile_compare=profile_compare,
            trace_startup=getattr(args, "trace_startup", False) and backend == "mte",
            verification_suite=getattr(args, "verification_suite", False),
        )
        print(f"\n=== REAL MOE HOST BACKEND: {backend}; case={label}; load_stream={load_stream_mode} ===", flush=True)
        print(
            f"Requested worker policy: scan={scan}; probe={probe}; idle={idle}; MTE only",
            flush=True,
        )
        if args.diagnose_load_stages:
            print("DIAGNOSTIC LOAD BARRIERS ENABLED: timings are not a performance comparison", flush=True)
        reports[label] = inference(
            run_args,
            configured_source(
                source,
                backend,
                library,
                args.diagnose_load_stages,
                scan,
                probe,
                idle,
                args.diagnose_native_wait,
                args.diagnose_split_scale,
                args.diagnose_scale_context,
                args.diagnose_prefetch_report_wait,
                args.diagnose_prefetch_poll_budget,
                args.diagnose_load_vector_cores,
                args.diagnose_graph_dump,
                directory / "graph_debug",
                args.diagnose_routing_vector_cores,
                args.diagnose_compute_cube_cores,
                args.diagnose_dsa_cube_cores,
                args.diagnose_native_stream_limits,
                args.diagnose_prefetch_wait_at_join,
                getattr(args, "no_load_diagnostics", False),
                getattr(args, "no_graph_dump", False),
                getattr(args, "diagnose_prefetch_wait_before_gate", False),
                mte_load_stream=load_stream_mode,
            ),
            cache_policy_backend,
            env,
        )
        log = (directory / f"inference_{cache_policy_backend}.log").read_text(encoding="utf-8", errors="replace")
        if worker_compare and f"[MOE-MTE WORKER] scan={scan} probe={probe} idle={idle}" not in log:
            raise RuntimeError(f"{label}: missing runtime worker policy evidence")
        assess_profile_report(
            reports[label],
            log,
            root / "profile" / label / cache_policy_backend,
            require_mte=backend == "mte",
            retain_trace_data=profile_compare,
            large_gap_us=getattr(args, "large_gap_us", None) if profile_compare else None,
        )
        reports[label].update(
            host_backend=backend,
            cache_policy_backend=cache_policy_backend,
            load_backend="python",
            h2d_backend="torch",
            load_stream_mode=load_stream_mode if backend == "mte" else "ordinary",
            worker_policy=dict(scan=scan, probe=probe, idle=idle) if backend == "mte" else None,
        )
        if (stream_compare or worker_compare) and backend == "mte":
            expected_type = "SyncLaunchStream" if load_stream_mode == "sync_launch" else "Stream"
            expected = f"[MOE-MTE] weight load stream={expected_type}; mode={load_stream_mode}; H2D wait retained"
            if expected not in log:
                raise RuntimeError(f"Missing runtime load stream evidence: {expected}")
        if profile_compare:
            (directory / "profile_report.json").write_text(json.dumps(reports[label]), encoding="utf-8")
        if stream_compare or worker_compare:
            positions = reports[label]["engine_error_log_positions"]
            if not positions["cleanup_marker_unique"] or positions["before_cleanup"] != 0:
                raise RuntimeError(f"{label}: engine error before cleanup or missing cleanup boundary; stop comparison")
        if worker_compare:
            report = reports[label]
            if report.get("errors") != 0 or report.get("dropped") != 0 or report.get("count", 0) <= 0:
                raise RuntimeError(f"{label}: invalid callback measurements; stop comparison")
            baseline = reports["baseline"]
            if any(report.get(field) != baseline.get(field) for field in ("request_specs", "digests", "usage")):
                raise RuntimeError(f"{label}: output differs from baseline; stop comparison")
            if getattr(args, "timing_requests", 0):
                window, reference = report.get("request_window"), baseline.get("request_window")
                if (
                    not window
                    or not reference
                    or any(
                        window.get(field) != reference.get(field)
                        for field in ("count", "request_specs", "digests", "usage")
                    )
                ):
                    raise RuntimeError(f"{label}: repeated request outputs differ from baseline; stop comparison")
        if cost_mode:
            # Preserve each completed group before comparing results or starting
            # the next service, so a later failure does not lose measurements.
            (directory / "cost_report.json").write_text(json.dumps(reports[label]), encoding="utf-8")
            positions = reports[label]["engine_error_log_positions"]
            if not positions["cleanup_marker_unique"] or positions["before_cleanup"] != 0:
                raise RuntimeError(f"{backend}: missing cleanup boundary or request-phase EngineDeadError")
            if any(tag in log for tag in ("[MOE-MTE COMPUTE_LIMIT]", "[MOE-MTE WAIT_CONTROL]", "[MOE-MTE FATAL]")):
                raise RuntimeError(f"{backend}: unexpected diagnostic intervention or MTE failure")
        if backend == "mte":
            if getattr(args, "no_graph_dump", False) and (
                "[MOE-MTE GRAPH_DUMP_" in log or (directory / "graph_debug").exists()
            ):
                raise RuntimeError("Unexpected graph debug evidence despite --no-graph-dump")
            if getattr(args, "no_load_diagnostics", False) and "[MOE-MTE LOAD]" in log:
                raise RuntimeError("Per-load diagnostic records appeared despite --no-load-diagnostics")
            expected_policy = f"[MOE-MTE WORKER] scan={scan} probe={probe} idle={idle}"
            if expected_policy not in log:
                raise RuntimeError("Missing runtime evidence of requested MTE worker policies")
            print(expected_policy, flush=True)
            if args.diagnose_prefetch_report_wait:
                if "[MOE-MTE WAIT_CONTROL]" not in log:
                    raise RuntimeError("Missing evidence that prefetch report-wait control was used")
            if args.diagnose_load_stages or args.diagnose_prefetch_poll_budget:
                print(
                    f"DIAGNOSTIC EXCERPT (full log: {directory / f'inference_{cache_policy_backend}.log'}):", flush=True
                )
                print(failure_excerpt(log), flush=True)
            if "[MOE-MTE] first real offload callback completed" not in log:
                raise RuntimeError("No evidence that real offload callbacks executed through MTE")
        print(
            "MOE_RESULT "
            + json.dumps(
                {
                    "host_backend": backend,
                    "cache_policy_backend": cache_policy_backend,
                    "request_mean_us": reports[label]["request_us"],
                    "digests": reports[label]["digests"],
                    "usage": reports[label]["usage"],
                    "profile_status": reports[label]["profile_status"],
                    "profile_errors": reports[label]["profile_errors"],
                    "profile_warnings": reports[label]["profile_warnings"],
                    "profile_checks": reports[label]["profile_checks"],
                    "engine_error_seen": reports[label]["engine_error_seen"],
                    "engine_error_log_positions": reports[label]["engine_error_log_positions"],
                    "callbacks": reports[label]["count"],
                    "callback_errors": reports[label]["errors"],
                    "callback_dropped": reports[label]["dropped"],
                    "request_specs": reports[label]["request_specs"],
                }
            ),
            flush=True,
        )
    if cost_mode or profile_compare:
        return reports
    if getattr(args, "verification_suite", False):
        compare_verification_reports(reports)
    elif len(reports) == 2:
        match = reports["callback"]["digests"] == reports["mte"]["digests"]
        usage_match = reports["callback"]["usage"] == reports["mte"]["usage"]
        print(f"GENERATED TEXT MATCH: {match}; USAGE MATCH: {usage_match}", flush=True)
        print(
            "Diagnostic request mean us: " + json.dumps({key: value["request_us"] for key, value in reports.items()}),
            flush=True,
        )
        if not match or not usage_match:
            raise RuntimeError("Generated output differs; inspect before claiming correctness or speedup")
    print("REAL MOE RUN COMPLETE. Limited prompt coverage; diagnostic timings are not performance results.", flush=True)
    print(
        f"Profiler requested={not args.skip_profile}; artifacts: {root / 'profile'}; return summary.txt text only.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
