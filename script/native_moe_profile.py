# SPDX-License-Identifier: Apache-2.0
"""Separate profile services: native C MSTX, same-capture clock alignment and gaps."""

import contextlib
import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from align_moe_profile_clock import align_capture
from export_moe_aligned_profile import publish_aligned_capture
from inspect_moe_profile import scan_trace
from inspect_moe_profiler_clocks import discover_roots
from inspect_native_profile_nesting import audit_host_nesting
from moe_host_memory import after_service_memory, check_host_memory, log_memory
from moe_tpot import streaming_completion
from run_verified_cpp_tests import ensure_profile_storage, request, wait_profile
from verify_memcpy_gaps import inspect_offload_trace

MIN_BUBBLE_US = 10.0
MAX_BUBBLE_US = 5000.0
FAILURE_LOG_BYTES = 65536
FAILURE_LOG_TAIL_LINES = 20
FAILURE_LOG_CONTEXT = 2
FAILURE_LOG_MAX_MATCHES = 8


def verify_full_submit(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "[MOE-NATIVE] FULL_SUBMIT ready;" not in text or "[MOE-NATIVE] FULL_SUBMIT executed=" not in text:
        raise RuntimeError("native_full submission was not exercised; refusing fallback-only comparison")


def report_native_scale_memory(log_path, report):
    """Require the current allocation contract in both timing and profile services."""
    prefix = "[MOE-NATIVE] SCALE_MEMORY "
    records = [
        json.loads(line.split(prefix, 1)[1])
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if prefix in line
    ]
    if not records:
        raise RuntimeError("Native scale allocation evidence missing; verify the service uses current project sources")
    for record in records:
        if record.get("policy") != "shared_final_layout_v1" or record.get("duplicate_bytes") != 0:
            raise RuntimeError("Native service retained duplicate scales or used an unknown allocation policy")
        for key in ("layers", "reused_bytes", "replaced_bytes", "derived_bytes"):
            if type(record.get(key)) is not int or record[key] < (1 if key == "layers" else 0):
                raise ValueError(f"Invalid native scale memory field: {key}")
        report("NATIVE_SCALE_MEMORY " + json.dumps(record))
    return records


def report_profile_failure(case, log_path, error, report):
    """Keep the engine failure visible in summary.txt before cleanup adds noise."""
    report(f"PROFILE_SERVICE_FAILURE case={case}; error={type(error).__name__}: {error}; service_log={log_path}")
    with log_path.open("rb") as source:
        source.seek(0, 2)
        source.seek(max(0, source.tell() - FAILURE_LOG_BYTES))
        lines = source.read().decode("utf-8", errors="replace").splitlines()
    selected = set(range(max(0, len(lines) - FAILURE_LOG_TAIL_LINES), len(lines)))
    matches = 0
    for index, line in enumerate(lines):
        if any(
            token in line
            for token in (
                "OutOfMemoryError",
                "Memory_Allocation_Failure",
                "RuntimeError:",
                "ValueError:",
                "ImportError:",
                "error code is",
            )
        ):
            selected.update(
                range(max(0, index - FAILURE_LOG_CONTEXT), min(len(lines), index + FAILURE_LOG_CONTEXT + 1))
            )
            matches += 1
            if matches >= FAILURE_LOG_MAX_MATCHES:
                break
    report("PROFILE_SERVICE_LOG_EXCERPT_BEGIN (bounded tail and error context; full log retained)")
    for index in sorted(selected):
        report(lines[index])
    report("PROFILE_SERVICE_LOG_EXCERPT_END")


def profile_command(command, directory, timing_port):
    words = list(command)
    index = words.index("--additional-config") + 1
    config = json.loads(words[index])
    config["expert_offload_config"].update(callback_timing_port=timing_port, callback_mstx_enabled=True)
    words[index] = json.dumps(config)
    words.extend(
        [
            "--profiler-config",
            json.dumps(dict(profiler="torch", torch_profiler_dir=str(directory), torch_profiler_with_stack=False)),
        ]
    )
    return words


def analyze_profile(trace, case, expected_count, stream_name, report):
    scan = scan_trace(trace, require_mte=True)
    if not scan["usable"] or scan["host_callbacks"] != expected_count:
        report(
            "PROFILE_CHECK "
            + json.dumps(
                {key: scan[key] for key in ("json_complete", "host_callbacks", "updates", "nesting", "issues")}
            )
        )
        try:
            report("HOST_NESTING_AUDIT " + json.dumps(audit_host_nesting(trace), ensure_ascii=False))
        except Exception as exc:
            report(f"HOST_NESTING_AUDIT_FAILED {type(exc).__name__}: {exc}")
        raise RuntimeError(f"Incomplete profile: callbacks={scan['host_callbacks']}/{expected_count}; {scan['issues']}")
    for row in scan["host_gaps"]:
        report("HOST_CALLBACK_GAP " + case + " " + json.dumps(row, ensure_ascii=False))
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            complete = inspect_offload_trace(trace, stream_name, MIN_BUBBLE_US, MAX_BUBBLE_US)
    finally:
        # Stream numbering can change between captures. Preserve the inventory
        # even when exact-name selection raises, so saved profiles can be reused.
        text = capture.getvalue()
        for line in text.splitlines():
            if line.startswith(
                (
                    "COPY_TRACK ",
                    "LOADING_STREAM_SELECTION ",
                    "TARGET_GAPS_US ",
                    "TARGET_DIRECTION_GAPS_US ",
                    "SELECTED_OFFLOAD_TRACK ",
                    "OFFLOAD_STATUS ",
                    "CALLBACK_SCOPE_UNAVAILABLE ",
                    "EXCLUDED_ADJACENT_PAIRS ",
                )
            ):
                report(case + " " + line)
    if not complete:
        report(f"BUBBLES_PARTIAL {case}: full-window statistics retained; direction mapping may be incomplete")
    selected = [
        json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("SELECTED_OFFLOAD_TRACK ")
    ]
    scan["loading_track"] = selected[0] if len(selected) == 1 else None
    return scan, text


def report_gap_table(profiles, report):
    report("GAP_TABLE_US (Host: raw and 3xIQR; bubbles: only 10<=gap<5000us, no further IQR)")
    report("| case | metric | samples | mean_us | IQR kept | IQR mean_us |")
    report("| --- | --- | ---: | ---: | ---: | ---: |")
    for case, profile in profiles.items():
        for row in profile["scan"]["host_gaps"]:
            clean = row["outliers_us"]["filtered"]
            report(
                f"| {case} | Host {row['direction']} | {row['count']} | {row['mean_us']} | "
                f"{clean['n']} | {clean['mean']} |"
            )
        selected = profile["scan"].get("loading_track")
        if selected is not None and selected["loaded_batches"] == 0:
            report(
                f"BUBBLES_UNVERIFIED {case}: {selected['name']} has no matched loading batches; "
                "track diagnostics retained, excluded from loading-bubble comparison. "
                "Confirm the loading stream in this capture and reanalyze the saved profile."
            )
            continue
        for row in profile["bubble_stats"]:
            kept = row["kept"]
            scope = row.get("scope", row.get("direction"))
            count, mean = (0, "—") if kept is None else (kept["n"], kept["mean"])
            report(f"| {case} | Loading bubbles {scope} | {count} | {mean} | — | — |")


def report_prefetch_overlap(root, report):
    """Always append both restats to the run summary, including single-case runs."""
    # Offline discovery imports this module for the gap reader; defer to avoid
    # that circular import. Analysis runs only after all services have exited.
    from reanalyze_native_prefetch_overlap import reanalyze, report_scope

    report("PREFETCH_OVERLAP_BEGIN: saved aligned profiles; original gaps and both methods follow")
    report_scope(report)
    try:
        status = dict(status=reanalyze([root], root, report))
    except Exception as exc:
        status = dict(status="OVERLAP_FAILED", error=f"{type(exc).__name__}: {exc}")
        report("PROFILE_ANALYSIS_INCOMPLETE: overlap unavailable; measured timing/gaps and profiles retained")
    (root / "prefetch_overlap_status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    report("PREFETCH_OVERLAP_STATUS " + json.dumps(status, ensure_ascii=False))
    return status


def profile_cases(args, commands, root, env, report, wait_port_free, stop_process_group, expected_outputs):
    """No profiler/MSTX configuration is added to either timed service."""
    repo = Path(__file__).resolve().parents[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    profiles = {}
    for case, original in commands.items():
        directory = root / "profile" / case
        directory.mkdir(parents=True)
        ensure_profile_storage(
            directory, stage=f"before {case} profile service", min_free_gib=args.min_profile_free_gib
        )
        wait_port_free(args.port)
        wait_port_free(args.timing_port)
        check_host_memory(f"before profile/{case}", report)
        command = profile_command(original, directory, args.timing_port)
        log_path = root / f"{case}_profile.log"
        report(f"PROFILE_START {case}; separate diagnostic service; log={log_path}")
        (root / f"{case}_profile_command.json").write_text(json.dumps(command), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + args.startup_timeout
                next_update = time.monotonic() + 30
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f"{case} profile service exited: {child.returncode}; {log_path}")
                    try:
                        models = request(opener, args.port, "/v1/models", timeout=2)
                        status = request(opener, args.timing_port, "/status", timeout=2)
                        if not status.get("mstx_enabled"):
                            raise RuntimeError("MSTX control not enabled")
                        break
                    except (OSError, ValueError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{case} profile startup timeout; {log_path}")
                        if time.monotonic() >= next_update:
                            report(f"PROFILE_WAIT {case}: waiting for service; log={log_path}")
                            next_update = time.monotonic() + 30
                        time.sleep(2)
                body = dict(
                    model=models["data"][0]["id"],
                    prompt="The future of AI is",
                    max_tokens=args.max_tokens,
                    temperature=0,
                    ignore_eos=True,
                )
                for _ in range(args.warmup_requests):
                    streaming_completion(opener, args.port, body, timeout=args.request_timeout)
                request(opener, args.timing_port, "/clock/prepare", {}, timeout=args.profile_timeout)
                request(opener, args.port, "/start_profile", {}, timeout=args.profile_timeout)
                try:
                    before = request(opener, args.timing_port, "/clock/before", {}, timeout=args.profile_timeout)
                    (root / f"{case}_clock_samples.json").write_text(json.dumps(before), encoding="utf-8")
                    request(opener, args.timing_port, "/start", {})
                    try:
                        samples = [
                            streaming_completion(opener, args.port, body, timeout=args.request_timeout)
                            for _ in range(args.profile_requests)
                        ]
                    finally:
                        counters = request(opener, args.timing_port, "/stop", {}, timeout=40)
                    after = request(opener, args.timing_port, "/clock/after", {}, timeout=args.profile_timeout)
                    (root / f"{case}_clock_samples.json").write_text(json.dumps(after), encoding="utf-8")
                    (root / f"{case}_profile_samples.json").write_text(json.dumps(samples), encoding="utf-8")
                    (root / f"{case}_profile_counters.json").write_text(json.dumps(counters), encoding="utf-8")
                    report("LOADING_STREAM_RUNTIME " + case + " " + json.dumps(counters.get("loading_stream")))
                finally:
                    request(opener, args.port, "/stop_profile", {}, timeout=args.profile_timeout)
                wait_profile(directory, args.profile_timeout)
                if counters["mstx_errors"] or counters["errors"] or counters["dropped"] or not counters["count"]:
                    raise RuntimeError(f"Incomplete Host ranges: {counters}")
                if any((s["digest"], s["usage"]) != expected_outputs[0] for s in samples):
                    raise RuntimeError("Profile output differs from timed requests")
                if case in (
                    "native",
                    "native_full",
                ) and "first native graph task completed without Python" not in log_path.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    raise RuntimeError("Native profile did not execute native graph tasks")
                if case in ("native", "native_full"):
                    report_native_scale_memory(log_path, report)
                    if case == "native_full":
                        verify_full_submit(log_path)
            except Exception as error:
                try:
                    report_profile_failure(case, log_path, error, report)
                except Exception as exc:
                    report(f"PROFILE_FAILURE_EXCERPT_UNAVAILABLE {type(exc).__name__}: {exc}")
                try:
                    log_memory(f"failure profile/{case}/before_cleanup", report)
                except Exception as exc:
                    report(f"HOST_MEMORY_DIAGNOSTIC_FAILED {type(exc).__name__}: {exc}")
                raise
            finally:
                stop_process_group(child)
                after_service_memory(f"after profile/{case}", report, child.pid, drop_cache=case == "python")
        wait_port_free(args.port, timeout=10)
        wait_port_free(args.timing_port, timeout=10)
        captures = discover_roots([directory])
        if len(captures) != 1:
            raise RuntimeError(f"Expected one capture for {case}, got {len(captures)}")
        mapping = align_capture(captures[0], after["samples"])
        if mapping.get("status") != "calibration_validated":
            raise RuntimeError(f"Clock alignment failed: {mapping}")
        published = publish_aligned_capture(captures[0], mapping)
        if published.get("status") != "exported":
            raise RuntimeError(f"Aligned directory export failed: {published}")
        trace = captures[0] / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        mapping["aligned_trace"] = str(trace)
        report(f"ALIGNED_PROFILE {case}: {captures[0]}")
        report("CLOCK_ALIGNMENT " + case + " " + json.dumps(mapping))
        scan, details = analyze_profile(trace, case, counters["count"], args.bubble_stream_name, report)
        (root / f"{case}_bubble_details.log").write_text(details, encoding="utf-8")
        bubble_stats = [
            json.loads(line.split(" ", 1)[1])
            for line in details.splitlines()
            if line.startswith(("TARGET_GAPS_US ", "TARGET_DIRECTION_GAPS_US "))
        ]
        profiles[case] = dict(scan=scan, mapping=mapping, aligned_profile=published, bubble_stats=bubble_stats)
        (root / f"{case}_profile_report.json").write_text(json.dumps(profiles[case]), encoding="utf-8")
    report(f"PROFILE_COMPLETE cases={','.join(profiles)}; each profile output matched the saved timing reference")
    report("PROFILE_MEASUREMENT: Host gap = next task entry - previous task return; includes zero-load tasks.")
    report(
        "Loading bubbles: same-capture runtime loading stream (or explicit legacy name), "
        "all tasks including Transpose, 10<=gap<5000us."
    )
    report_gap_table(profiles, report)
    report_prefetch_overlap(root, report)
    return profiles
