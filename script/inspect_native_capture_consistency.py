# SPDX-License-Identifier: Apache-2.0
"""Audit saved output/configuration and unmatched MTE tasks without changing results."""

import argparse
import collections
import contextlib
import itertools
import json
import math
from decimal import Decimal
from pathlib import Path

from diagnose_mte_wait_timeline import read_trace, task_pairs
from export_moe_aligned_profile import aligned_trace_path
from native_moe_artifacts import CASES, saved_case_file, saved_timing_records

ORDER_US = 1.0
MAX_WINDOW_SHIFTS = 1024
MAX_EXAMPLES = 3
SERVICE_ONLY_OPTIONS = frozenset(("--port", "--profiler-config"))
SERVICE_ONLY_CONFIG = frozenset(("callback_timing_port", "callback_mstx_enabled", "host_sync_library"))


def emit(label, value):
    print(label + " " + json.dumps(value, ensure_ascii=False), flush=True)


def command_settings(command):
    """Keep all model/workload flags; separate expected instrumentation differences."""
    if not isinstance(command, list) or not all(isinstance(word, str) for word in command):
        raise ValueError("saved command is not an argv list")
    settings = {}

    def add(key, value):
        if key in settings:
            raise ValueError(f"duplicate saved command option: {key}")
        settings[key] = value

    def flatten(prefix, value):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                flatten(prefix + "." + key, child)
        else:
            add(prefix, value)

    positional = []
    index = 0
    while index < len(command):
        word = command[index]
        if not word.startswith("--"):
            positional.append(word)
            index += 1
            continue
        if "=" in word:
            key, value = word.split("=", 1)
            index += 1
        else:
            key, values = word, []
            index += 1
            while index < len(command) and not command[index].startswith("--"):
                values.append(command[index])
                index += 1
            value = True if not values else values[0] if len(values) == 1 else values
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                value = json.loads(value)
        flatten(key, value)
    add("positional", positional)
    return settings


def setting_category(key):
    if key == "--additional-config.expert_offload_config.host_update_backend":
        return "expected_case_selector"
    if key in SERVICE_ONLY_OPTIONS or any(key.startswith(option + ".") for option in SERVICE_ONLY_OPTIONS):
        return "service_or_profiler"
    if key.startswith("--additional-config.expert_offload_config.") and key.rsplit(".", 1)[-1] in SERVICE_ONLY_CONFIG:
        return "service_or_profiler"
    return "workload_or_other"


def command_differences(left, right):
    return [
        dict(key=key, category=setting_category(key), left=left.get(key), right=right.get(key))
        for key in sorted(left.keys() | right.keys())
        if (key in left, left.get(key)) != (key in right, right.get(key))
    ]


def signature(sample):
    return sample["digest"], json.dumps(sample["usage"], sort_keys=True)


def pair_identity(pair):
    return dict(publish_ids=pair[0].get("ids", {}), wait_ids=pair[1].get("ids", {}))


def inspect_window(pairs, hosts, margin):
    """Report contiguous ordinal hypotheses, never accept a new pairing or shift clocks.

    Coverage within the calibration envelope is only a diagnostic condition.
    It is not an independent proof of protocol correctness or of clock accuracy.
    """
    if not hosts or not pairs:
        return dict(status="empty", candidates=[])
    excess = len(pairs) - len(hosts)
    if excess < 0:
        return dict(status="fewer_device_pairs_than_host_markers", candidates=[])
    if excess + 1 > MAX_WINDOW_SHIFTS:
        return dict(status="too_many_candidate_windows", candidates=[], possible_offsets=excess + 1)
    candidates = []
    for offset in range(excess + 1):
        failures, largest = 0, 0.0
        for index, host in enumerate(hosts):
            publish, wait = pairs[offset + index]
            residual = max(publish["start"] - host["start"], host["end"] - wait["end"], 0.0)
            largest = max(largest, residual)
            failures += residual > ORDER_US
            if residual > margin:
                break
        else:
            prefix = pairs[:offset]
            suffix = pairs[offset + len(hosts) :]
            before = sum(wait["end"] < hosts[0]["start"] - margin for _, wait in prefix)
            after = sum(publish["start"] > hosts[-1]["end"] + margin for publish, _ in suffix)
            candidates.append(
                dict(
                    offset=offset,
                    matched=len(hosts),
                    unmatched_prefix=len(prefix),
                    unmatched_suffix=len(suffix),
                    prefix_clearly_before_host_window=before,
                    suffix_clearly_after_host_window=after,
                    unmatched_not_clearly_outside=len(prefix) + len(suffix) - before - after,
                    strict_order_failures=failures,
                    max_order_residual_us=largest,
                    first_host=dict(call=hosts[0]["context"]["call"], layer=hosts[0]["context"]["layer"]),
                    last_host=dict(call=hosts[-1]["context"]["call"], layer=hosts[-1]["context"]["layer"]),
                    first_pair=pair_identity(pairs[offset]),
                    last_pair=pair_identity(pairs[offset + len(hosts) - 1]),
                    first_suffix_publish_minus_host_end_us=(suffix[0][0]["start"] - hosts[-1]["end"])
                    if suffix
                    else None,
                    last_prefix_wait_minus_host_start_us=(prefix[-1][1]["end"] - hosts[0]["start"]) if prefix else None,
                )
            )
    return dict(
        status="diagnostic_only",
        candidate_count=len(candidates),
        candidates=candidates[:MAX_EXAMPLES],
        omitted_candidates=max(0, len(candidates) - MAX_EXAMPLES),
    )


def inspect_capture(root, case):
    directory = root / "profile" / case
    paths = directory.rglob("trace_view.json")
    path = next(paths, None)
    if path is None or next(paths, None) is not None:
        raise ValueError("expected exactly one saved trace")
    path = aligned_trace_path(path)
    mapping = json.loads(path.with_name("clock_alignment.json").read_text(encoding="utf-8"))
    if mapping.get("calibration_layout") != "model_request_brackets":
        raise ValueError("need same-request before/after clock calibration")
    margin = max(map(float, mapping["endpoint_mapping_envelope_width_us"])) + ORDER_US
    if not math.isfinite(margin) or margin < ORDER_US:
        raise ValueError("invalid calibration diagnostic margin")
    emit("SOURCE", dict(case=case, trace=str(path), diagnostic_margin_us=margin))
    data = read_trace(path, time_origin=Decimal(mapping["trace_origin_us"]))
    if data["issues"] or data["invalid_kinds"] or not data["all_hosts"]:
        raise ValueError(f"invalid Host markers: {data['issues']}")
    counters = json.loads(saved_case_file(root, case, "profile_counters.json").read_text(encoding="utf-8"))
    if any(counters.get(key) != 0 for key in ("mstx_errors", "errors", "dropped")):
        raise ValueError("saved Host collection reported errors/dropped records")
    if counters.get("count") != data["host_count"]:
        raise ValueError("saved Host counters disagree with trace")
    emit("HOST_COUNTS", dict(case=case, **{kind: len(hosts) for kind, hosts in data["groups"].items()}))
    for stream, tasks in sorted(data["tasks"].items()):
        pairs, reason = task_pairs(tasks, ORDER_US)
        emit("DEVICE_PAIRS", dict(case=case, stream=list(stream), tasks=len(tasks), pairs=len(pairs), sequence=reason))
        if reason != "ok":
            continue
        for kind, hosts in data["groups"].items():
            emit(
                "WINDOW_HYPOTHESIS",
                dict(case=case, stream=list(stream), kind=kind, **inspect_window(pairs, hosts, margin)),
            )


def inspect_run(root, cases, *, profile_cases=None):
    emit("RUN", str(root))
    records = saved_timing_records(root, cases)
    configurations = {}
    complete = True
    for case in cases:
        row = records.get(case)
        if row is None:
            emit("MISSING_RESULT", dict(case=case))
            complete = False
            continue
        samples = row.get("stream_samples", [])
        if not samples:
            raise ValueError(f"{case}: timing samples missing")
        signatures = collections.Counter(signature(sample) for sample in samples)
        emit(
            "OUTPUTS",
            dict(
                case=case,
                samples=len(samples),
                variants=[
                    dict(digest=digest, usage=json.loads(usage), requests=count)
                    for (digest, usage), count in signatures.items()
                ],
                content_chunks=dict(collections.Counter(sample.get("content_chunks") for sample in samples)),
            ),
        )
        profile_path = saved_case_file(root, case, "profile_samples.json")
        if profile_path.exists():
            profile_samples = json.loads(profile_path.read_text(encoding="utf-8"))
            emit(
                "PROFILE_OUTPUT_CHECK",
                dict(
                    case=case,
                    samples=len(profile_samples),
                    matches_timing=bool(profile_samples)
                    and len(signatures) == 1
                    and all(signature(sample) in signatures for sample in profile_samples),
                ),
            )
        try:
            command = json.loads(saved_case_file(root, case, "command.json").read_text(encoding="utf-8"))
            configurations[case] = command_settings(command)
        except (OSError, ValueError) as exc:
            complete = False
            emit("CONFIG_UNAVAILABLE", dict(case=case, error=str(exc)))
    for left, right in itertools.combinations(configurations, 2):
        differences = command_differences(configurations[left], configurations[right])
        emit(
            "COMMAND_DIFF",
            dict(
                left=left,
                right=right,
                differences=differences,
                scope="saved argv only; does not verify historical weights, binaries or environment",
            ),
        )
    for case in cases if profile_cases is None else profile_cases:
        try:
            inspect_capture(root, case)
        except (OSError, ValueError, KeyError) as exc:
            complete = False
            emit("CAPTURE_UNAVAILABLE", dict(case=case, error=f"{type(exc).__name__}: {exc}"))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved token5/token128 result directory")
    parser.add_argument("--profile-cases", nargs="+", choices=CASES, default=["native", "native_full"])
    args = parser.parse_args()
    print("只读已有结果、启动参数和对齐profile；不导入Torch、不启动模型、不写文件。", flush=True)
    print("审计三组保存的输出和参数；默认只检查 native/native_full 的设备任务窗口。", flush=True)
    print("候选窗口只用于定位多余任务，不修改配对验收、时间戳或原统计；margin不是完整时钟误差上限。", flush=True)
    try:
        complete = inspect_run(
            args.run.resolve(strict=True), list(CASES), profile_cases=list(dict.fromkeys(args.profile_cases))
        )
    except (OSError, ValueError, KeyError) as exc:
        emit("INSPECTION_FAILED", f"{type(exc).__name__}: {exc}")
        return 1
    print("INSPECTION_COMPLETE" if complete else "INSPECTION_PARTIAL: available evidence printed above", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
