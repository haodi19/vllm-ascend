# SPDX-License-Identifier: Apache-2.0
"""Create a separate aligned trace using probes from the SAME capture."""

import argparse
import collections
import json
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import _TraceReader, events
from calibrate_moe_profiler_clock import MARKER_PREFIX, affine_clock_bounds
from inspect_moe_profiler_clocks import discover_roots, metadata_files, parse_metadata

QUANTIZATION_US = Decimal("1")
MIN_SAMPLES_PER_PHASE = 8


def boundary_stats(measurements, scale=Decimal(1), offset=Decimal(0)):
    distances = [
        max(Decimal(0), submit - (start - offset) / scale, (end - offset) / scale - sync)
        for submit, sync, start, end in measurements
    ]
    return {
        "n": len(distances),
        "outside_bracket_zero_tolerance": sum(distance > 0 for distance in distances),
        "outside_bracket_1us_tolerance": sum(distance > QUANTIZATION_US for distance in distances),
        "max_distance_outside_us": str(max(distances, default=Decimal(0))),
    }


def fit_alignment(root, rows, *, require_model_phases=True):
    if [row["sample"] for row in rows] != list(range(len(rows))):
        raise ValueError("incomplete calibration sample sequence")
    if require_model_phases:
        phases = [row.get("phase") for row in rows]
        if any(phases.count(phase) < MIN_SAMPLES_PER_PHASE for phase in ("before", "after")):
            raise ValueError("need before/after probes from this capture")
        if phases != sorted(phases, key=lambda phase: {"before": 0, "after": 1}[phase]):
            raise ValueError("calibration phases out of order")
    elif len(rows) < 2 * MIN_SAMPLES_PER_PHASE:
        raise ValueError("need at least 16 standalone calibration samples")
    host_files = [p for p in metadata_files(root) if p.parent.name == "host" and p.name == "start_info"]
    if len(host_files) != 1:
        raise ValueError("ambiguous Host epoch metadata")
    info = parse_metadata(host_files[0])
    epoch = Decimal(str(info["collectionTimeBegin"])) - Decimal(str(info["clockMonotonicRaw"])) / 1000
    origin = rows[0]["submit_before_ns"]
    trace_origin = epoch + Decimal(origin) / 1000
    processes, markers, adds = {}, {}, collections.defaultdict(list)
    trace = root / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    for event in events(trace):
        pid, name = str(event.get("pid")), str(event.get("name", ""))
        if event.get("ph") == "M" and name == "process_name":
            processes[pid] = event.get("args", {}).get("name", "")
        if event.get("ph") != "X":
            continue
        if name.startswith(MARKER_PREFIX):
            if name in markers:
                raise ValueError("duplicate calibration marker")
            markers[name] = event
        elif "add" in name.lower():
            adds[(pid, str(event.get("tid")), name)].append(event)
    hardware = {pid for pid, name in processes.items() if str(name).strip().lower() == "ascend hardware"}
    if len(hardware) != 1:
        raise ValueError("only one identified device time domain is supported")
    lanes = [
        (identity, items) for identity, items in adds.items() if identity[0] in hardware and len(items) == len(rows)
    ]
    if len(markers) != len(rows) or len(lanes) != 1:
        raise ValueError("cannot uniquely identify calibration markers and dedicated Add stream")
    identity, kernels = lanes[0]
    kernels.sort(key=lambda event: Decimal(str(event["ts"])))
    measurements = []
    previous = -1
    keys = ("start_before_ns", "start_after_ns", "submit_before_ns", "sync_after_ns", "end_before_ns", "end_after_ns")
    for row, kernel in zip(rows, kernels):
        times = [row[key] for key in keys]
        if times != sorted(times) or times[0] < previous:
            raise ValueError("invalid RAW timestamp order")
        previous = times[-1]
        marker = markers[f"{MARKER_PREFIX}{row['sample']}"]
        mstart = Decimal(str(marker["ts"])) - epoch
        mend = mstart + Decimal(str(marker["dur"]))
        for stamp, left, right in (
            (mstart, "start_before_ns", "start_after_ns"),
            (mend, "end_before_ns", "end_after_ns"),
        ):
            if not Decimal(row[left]) / 1000 - QUANTIZATION_US <= stamp <= Decimal(row[right]) / 1000 + QUANTIZATION_US:
                raise ValueError("Host MSTX not consistent with direct RAW; refuse device-only correction")
        start = Decimal(str(kernel["ts"])) - trace_origin
        end = start + Decimal(str(kernel["dur"]))
        measurements.append(
            (
                Decimal(row["submit_before_ns"] - origin) / 1000,
                Decimal(row["sync_after_ns"] - origin) / 1000,
                start,
                end,
            )
        )
    # Fit even samples only; odd samples are independent acceptance checks.
    train = measurements[::2]
    rate = affine_clock_bounds(train)
    if not rate["positive_affine_possible"]:
        raise ValueError("one affine clock cannot explain training probes")
    low, high = map(Decimal, rate["rate_error_bounds_us_per_second"])
    scale = 1 + (low + high) / 2 / 10**6
    lower = max(end - scale * sync for submit, sync, start, end in train)
    upper = min(start - scale * submit for submit, sync, start, end in train)
    if lower > upper:
        raise ValueError("no common offset at selected rate")
    offset = (lower + upper) / 2
    validation_before = boundary_stats(measurements[1::2])
    validation_after = boundary_stats(measurements[1::2], scale, offset)
    if validation_after["outside_bracket_1us_tolerance"]:
        raise ValueError(f"held-out probes fail alignment; no corrected trace exported: {validation_after}")
    # Uncertainty envelope of all feasible training mappings, not just chosen midpoint.
    uncertainty = []
    for host in (measurements[0][0], measurements[-1][1]):
        scales = [1 + ppm / 10**6 for ppm in (low, high)]
        envelope_low = max(min(end + a * (host - sync) for a in scales) for submit, sync, start, end in train)
        envelope_high = min(max(start + a * (host - submit) for a in scales) for submit, sync, start, end in train)
        uncertainty.append(str(envelope_high - envelope_low))
    return {
        "status": "calibration_validated",
        "source": str(trace.resolve()),
        "trace_origin_us": str(trace_origin),
        "device_scale": str(scale),
        "device_offset_us": str(offset),
        "device_pids": sorted(hardware),
        "calibration_stream": identity,
        "training_samples": len(train),
        "held_out_samples": len(measurements[1::2]),
        "held_out_before": validation_before,
        "held_out_after": validation_after,
        "calibration_layout": "model_request_brackets" if require_model_phases else "standalone_calibration",
        "training_rate_bounds_us_per_second": [str(low), str(high)],
        "endpoint_mapping_envelope_width_us": uncertainty,
        "host_span_us": [str(measurements[0][0]), str(measurements[-1][1])],
        "quantization_tolerance_us": str(QUANTIZATION_US),
        "scope": "same-capture calibration; no MoE wait constraints used; not an exact physical clock recovery",
    }


def transform_event(event, mapping, counts):
    result = dict(event)
    if str(event.get("pid")) not in mapping["device_pids"] or event.get("ph") == "M" or "ts" not in event:
        return result
    scale, offset, origin = (Decimal(mapping[key]) for key in ("device_scale", "device_offset_us", "trace_origin_us"))
    stamp = (Decimal(str(event["ts"])) - origin - offset) / scale
    result["ts"] = str(origin + stamp)
    if "dur" in event:
        duration = Decimal(str(event["dur"]))
        if duration < 0:
            raise ValueError("negative duration in device trace")
        result["dur"] = str(duration / scale)
    counts["transformed_events"] += 1
    left, right = map(Decimal, mapping["host_span_us"])
    if not left <= stamp <= right:
        counts["outside_calibration_span"] += 1
    return result


def write_aligned_trace(source, destination, mapping):
    """Preserve every event and top-level field; transform device flow endpoints too."""
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("output must be a new file; original trace is never overwritten")
    temporary = destination.with_name(destination.name + ".partial")
    counts = collections.Counter()
    with source.open(encoding="utf-8-sig") as incoming, temporary.open("x", encoding="utf-8") as outgoing:
        reader = _TraceReader(incoming)

        def array():
            outgoing.write("[")
            separator = ""
            for event in reader.event_array():
                outgoing.write(separator + json.dumps(transform_event(event, mapping, counts), ensure_ascii=False))
                separator = ",\n"
            outgoing.write("]")

        if reader.peek() == "[":
            array()
        else:
            reader.take("{")
            outgoing.write("{")
            seen = set()
            while reader.peek() != "}":
                key = reader.value()
                if not isinstance(key, str) or key in seen:
                    raise ValueError("invalid or repeated root key")
                if seen:
                    outgoing.write(",")
                seen.add(key)
                reader.take(":")
                outgoing.write(json.dumps(key) + ":")
                if key == "traceEvents":
                    array()
                else:
                    outgoing.write(json.dumps(reader.value(), ensure_ascii=False))
                if reader.peek() != "}":
                    reader.take(",")
                    if reader.peek() == "}":
                        raise ValueError("trailing comma")
            reader.take("}")
            outgoing.write("}")
            if "traceEvents" not in seen:
                raise ValueError("missing traceEvents")
        if reader.peek():
            raise ValueError("trailing JSON content")
    # Exclusive destination: never replace a concurrently created output.
    try:
        destination.hardlink_to(temporary)
    except (NotImplementedError, OSError):
        # The files are in the same directory; avoid a silent overwrite on fallback.
        with destination.open("x", encoding="utf-8") as target, temporary.open(encoding="utf-8") as incoming:
            for block in iter(lambda: incoming.read(1024 * 1024), ""):
                target.write(block)
    temporary.unlink()
    return dict(counts)


def align_capture(root, samples, *, output_dir=None, require_model_phases=True):
    mapping = fit_alignment(root, samples, require_model_phases=require_model_phases)
    output = (output_dir if output_dir is not None else root / "ASCEND_PROFILER_OUTPUT") / "trace_view_aligned.json"
    mapping.update(write_aligned_trace(Path(mapping["source"]), output, mapping))
    mapping["aligned_trace"] = str(output)
    output.with_name("clock_alignment.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def align_saved_calibration(run, output_dir):
    """Use existing eager/graph captures and their own RAW records; no Torch/NPU."""
    output_dir.mkdir(parents=True, exist_ok=False)
    failed = False
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as summary:

        def emit(value):
            print(value, flush=True)
            summary.write(value + "\n")
            summary.flush()

        emit("离线验证可视化对齐：读取已有小张量校准，不导入Torch、不初始化NPU、不加载模型。")
        emit(f"SOURCE_RUN={run}; SUMMARY_FILE={output_dir / 'summary.txt'}")
        emit("偶数样本拟合，奇数样本独立检验；输出到新目录，原采集和原测量记录保持不变。")
        for mode in ("eager", "graph"):
            try:
                case = run / mode
                roots = discover_roots([case / "profile"])
                if len(roots) != 1:
                    raise ValueError("expected exactly one capture per case")
                samples = json.loads((case / "host_brackets.json").read_text(encoding="utf-8"))
                target = output_dir / mode
                target.mkdir()
                result = align_capture(roots[0], samples, output_dir=target, require_model_phases=False)
                emit("ALIGNMENT " + json.dumps({"mode": mode, **result}, ensure_ascii=False))
            except Exception as error:
                failed = True
                emit(f"ALIGNMENT_FAILED mode={mode}; {type(error).__name__}: {error}")
        emit(
            "PASS：对齐导出及留出样本检查通过；仅验证小张量采集，不代表真实MoE wait已验收。"
            if not failed
            else "FAIL：至少一组未通过；请返回本summary.txt，不重新加载模型。"
        )
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--calibration-run", type=Path, help="Offline check of an existing eager/graph calibration run")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New output directory for --calibration-run")
    args = parser.parse_args(argv)
    if args.calibration_run:
        if args.output_dir is None or args.samples is not None:
            parser.error("--calibration-run requires --output-dir and does not use --samples")
        return align_saved_calibration(args.calibration_run.resolve(strict=True), args.output_dir.resolve())
    if args.samples is None or args.output_dir is not None:
        parser.error("--capture requires --samples and does not use --output-dir")
    payload = json.loads(args.samples.read_text(encoding="utf-8"))
    print(json.dumps(align_capture(args.capture, payload["samples"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
