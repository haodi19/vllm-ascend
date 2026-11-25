# SPDX-License-Identifier: Apache-2.0
"""Read existing Host ranges and explain strict nesting failures; no model or writes."""

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from analyze_callback_gaps import MAX_INTERVALS, events, fields
from native_moe_artifacts import saved_case_file

MAX_EXAMPLES = 6
MAX_TRACES = 8
PREFIXES = (("moe_offload/host_callback|", "callback"), ("moe_offload/update_weights|", "update"))


def audit_host_nesting(trace):
    ranges = {"callback": {}, "update": {}}
    examples = []
    duplicates = missing = 0

    def note(value):
        if len(examples) < MAX_EXAMPLES:
            examples.append(value)

    # Preserve JSON numeric tokens as decimals; Decimal(str(float)) is too late.
    for event in events(Path(trace), parse_float=Decimal):
        name = str(event.get("name", ""))
        stage = next((stage for prefix, stage in PREFIXES if name.startswith(prefix)), None)
        if stage is None or event.get("ph") != "X":
            continue
        info = fields(event)
        pid = "" if event.get("pid") is None else str(event["pid"])
        tid = "" if event.get("tid") is None else str(event["tid"])
        call = info.get("call")
        try:
            start, duration = (Decimal(str(event[key])) for key in ("ts", "dur"))
        except (KeyError, InvalidOperation) as exc:
            raise ValueError(f"Invalid Host timestamp: {name}") from exc
        if not all(value.is_finite() and value >= 0 for value in (start, duration)):
            raise ValueError(f"Invalid Host interval: {name}")
        row = dict(name=name, tid=tid, ts=str(start), dur=str(duration), end=str(start + duration))
        if not pid or not call:
            missing += 1
            note(dict(reason="missing_pid_or_call", pid=pid, interval=row))
            continue
        key = (pid, call)
        if key in ranges[stage]:
            duplicates += 1
            note(dict(reason="duplicate_id", stage=stage, pid=pid, call=call, first=ranges[stage][key], duplicate=row))
            continue
        ranges[stage][key] = row
        if sum(len(group) for group in ranges.values()) > MAX_INTERVALS:
            raise ValueError("Too many Host ranges; audit stopped")

    callbacks, updates = ranges["callback"], ranges["update"]
    exact_invalid = float_invalid = float_only_invalid = wrong_thread = 0
    paired = callbacks.keys() & updates.keys()
    for pid, call in sorted(paired):
        outer, inner = callbacks[(pid, call)], updates[(pid, call)]
        outer_start, outer_end = Decimal(outer["ts"]), Decimal(outer["end"])
        inner_start, inner_end = Decimal(inner["ts"]), Decimal(inner["end"])
        same_thread = outer["tid"] == inner["tid"]
        exact_ok = same_thread and outer_start <= inner_start <= inner_end <= outer_end
        # Reproduce the legacy check including binary64 timestamp addition.
        ofs, ifs = float(outer["ts"]), float(inner["ts"])
        ofe, ife = ofs + float(outer["dur"]), ifs + float(inner["dur"])
        float_ok = same_thread and ofs <= ifs <= ife <= ofe
        wrong_thread += not same_thread
        exact_invalid += not exact_ok
        float_invalid += not float_ok
        float_only_invalid += exact_ok and not float_ok
        if not exact_ok or not float_ok:
            note(
                dict(
                    reason="nesting",
                    pid=pid,
                    call=call,
                    same_thread=same_thread,
                    exact_ok=exact_ok,
                    float_ok=float_ok,
                    callback=outer,
                    update=inner,
                    exact_start_overrun_us=str(max(Decimal(0), outer_start - inner_start)),
                    exact_end_overrun_us=str(max(Decimal(0), inner_end - outer_end)),
                    float_start_overrun_us=max(0.0, ofs - ifs),
                    float_end_overrun_us=max(0.0, ife - ofe),
                )
            )
    return dict(
        callbacks=len(callbacks),
        updates=len(updates),
        paired=len(paired),
        callback_only=len(callbacks.keys() - updates.keys()),
        update_only=len(updates.keys() - callbacks.keys()),
        duplicate_ids=duplicates,
        missing_ids=missing,
        wrong_thread=wrong_thread,
        invalid_float=float_invalid,
        invalid_exact=exact_invalid,
        float_only_invalid=float_only_invalid,
        examples=examples,
        scope="Exact exported decimals versus legacy float check; no tolerance or timestamp changes. "
        "Exact export violations do not alone prove runtime call ordering errors.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing native comparison result directory")
    args = parser.parse_args(argv)
    print("READ ONLY: existing trace; no Torch/NPU/model, no writes, no acceptance-rule changes.", flush=True)
    complete = True
    found = False
    for case in ("python", "native"):
        directory = args.run / "profile" / case
        traces = []
        for trace in directory.rglob("trace_view.json"):
            traces.append(trace)
            if len(traces) > MAX_TRACES:
                raise ValueError("Too many traces; pass a single experiment directory")
        for trace in sorted(traces):
            found = True
            print(f"TRACE case={case}; path={trace}", flush=True)
            try:
                audit = audit_host_nesting(trace)
                counters = saved_case_file(args.run, case, "profile_counters.json")
                expected = json.loads(counters.read_text(encoding="utf-8"))["count"] if counters.exists() else None
                print(
                    "HOST_NESTING_AUDIT "
                    + json.dumps(dict(case=case, expected_callbacks=expected, **audit), ensure_ascii=False),
                    flush=True,
                )
            except (OSError, ValueError, KeyError) as exc:
                complete = False
                print(f"AUDIT_FAILED {type(exc).__name__}: {exc}", flush=True)
    if not found:
        print("AUDIT_FAILED: no existing profile/*/trace_view.json found", flush=True)
    print("AUDIT_COMPLETE: diagnosis only; no profile repaired or performance result accepted.", flush=True)
    return 0 if complete and found else 1


if __name__ == "__main__":
    raise SystemExit(main())
