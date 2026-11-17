# SPDX-License-Identifier: Apache-2.0
"""Create a separate aligned trace using probes from the SAME model capture."""

import argparse
import collections
import json
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import _TraceReader, events
from calibrate_moe_profiler_clock import MARKER_PREFIX, affine_clock_bounds
from inspect_moe_profiler_clocks import metadata_files, parse_metadata

QUANTIZATION_US = Decimal("1")
MIN_SAMPLES_PER_PHASE = 8


def fit_alignment(root, rows):
    if [row["sample"] for row in rows] != list(range(len(rows))):
        raise ValueError("incomplete calibration sample sequence")
    phases = [row.get("phase") for row in rows]
    if any(phases.count(phase) < MIN_SAMPLES_PER_PHASE for phase in ("before", "after")):
        raise ValueError("need before/after probes from this capture")
    if phases != sorted(phases, key=lambda phase: {"before": 0, "after": 1}[phase]):
        raise ValueError("calibration phases out of order")
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
    for submit, sync, start, end in measurements[1::2]:
        if (start - offset) / scale < submit - QUANTIZATION_US or (end - offset) / scale > sync + QUANTIZATION_US:
            raise ValueError("held-out probes fail alignment; no corrected trace exported")
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


def align_capture(root, samples):
    mapping = fit_alignment(root, samples)
    output = root / "ASCEND_PROFILER_OUTPUT/trace_view_aligned.json"
    mapping.update(write_aligned_trace(Path(mapping["source"]), output, mapping))
    mapping["aligned_trace"] = str(output)
    output.with_name("clock_alignment.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.samples.read_text(encoding="utf-8"))
    print(json.dumps(align_capture(args.capture, payload["samples"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
