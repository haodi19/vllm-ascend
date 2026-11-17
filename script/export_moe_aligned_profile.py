# SPDX-License-Identifier: Apache-2.0
"""Export an importable profile copy, preserving DB lanes when a DB is present."""

import argparse
import collections
import contextlib
import json
import re
import shutil
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path

from analyze_callback_gaps import events
from calibrate_moe_profiler_clock import MARKER_PREFIX
from inspect_moe_profiler_clocks import discover_roots

OUTPUT_NAME = "ASCEND_PROFILER_OUTPUT"
DB_NAME = re.compile(r"ascend_pytorch_profiler(?:_\d+)?\.db$")
NS_PER_US = Decimal(1000)
MATCH_TOLERANCE_NS = Decimal(1000)
MIN_HOST_ANCHORS = 8
MAX_DIAGNOSTIC_LANES = 8


def trace_anchors(trace, mapping):
    markers, kernels = {}, []
    identity = tuple(map(str, mapping["calibration_stream"]))
    for event in events(trace):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        start = Decimal(str(event["ts"])) * NS_PER_US
        end = start + Decimal(str(event["dur"])) * NS_PER_US
        if name.startswith(MARKER_PREFIX):
            if name in markers:
                raise ValueError("duplicate Host calibration marker")
            markers[name] = (start, end)
        if (str(event.get("pid")), str(event.get("tid")), name) == identity:
            kernels.append((start, end))
    expected = mapping["training_samples"] + mapping["held_out_samples"]
    if len(markers) != expected or len(kernels) != expected:
        raise ValueError("trace calibration identities/counts disagree with saved alignment")
    return markers, sorted(kernels)


def columns(db, table):
    # Only fixed schema identifiers are used in SQL, never user-supplied SQL text.
    return {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}


def match_add_stream(db, kernels, mapping, epoch):
    """Match a complete DB lane against the already identified JSON probes.

    Model streams may run the same Add kernel as the calibration stream. A JSON
    tid is not assumed to be a DB streamId: require a unique full timestamp match,
    using the epoch determined from Host markers, without fitting another offset.
    """
    name = mapping["calibration_stream"][2]
    lanes = collections.defaultdict(list)
    for start, end, device, stream in db.execute(
        "SELECT t.startNs,t.endNs,t.deviceId,t.streamId FROM TASK t "
        "JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId "
        "JOIN STRING_IDS s ON s.id=c.name WHERE s.value=? ORDER BY t.startNs,t.endNs",
        (name,),
    ):
        lanes[(device, stream)].append((start, end))
    matches, candidates = [], []
    for (device, stream), pairs in lanes.items():
        candidate = {"device": device, "stream": stream, "count": len(pairs)}
        if len(pairs) == len(kernels):
            if any(start is None or end is None or start < 0 or end < start for start, end in pairs):
                candidate["invalid_intervals"] = True
            else:
                residual = max(
                    abs(Decimal(value) + epoch - stamp)
                    for pair, expected in zip(pairs, kernels)
                    for value, stamp in zip(pair, expected)
                )
                candidate["max_anchor_error_ns"] = str(residual)
                if stream is not None and residual <= MATCH_TOLERANCE_NS:
                    matches.append(candidate)
        candidates.append(candidate)
    if len(matches) != 1:
        # Include useful evidence in the normal export command, so a failed
        # identification does not require a separate read-only diagnostic script.
        candidates.sort(key=lambda item: (abs(item["count"] - len(kernels)), str(item["stream"])))
        details = {
            "json_identity": mapping["calibration_stream"],
            "expected_count": len(kernels),
            "same_name_streams": len(lanes),
            "timestamp_matched_streams": len(matches),
            "candidates": candidates[:MAX_DIAGNOSTIC_LANES],
        }
        if not lanes:
            details["available_add_names"] = db.execute(
                "SELECT s.value,COUNT(*) FROM TASK t "
                "JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId "
                "JOIN STRING_IDS s ON s.id=c.name WHERE LOWER(s.value) LIKE '%add%' "
                "GROUP BY s.value ORDER BY COUNT(*) DESC,s.value LIMIT ?",
                (MAX_DIAGNOSTIC_LANES,),
            ).fetchall()
        raise ValueError(
            "DB dedicated Add stream has no unique full match; counts or device timestamps differ: "
            + json.dumps(details, ensure_ascii=False)
        )
    return dict(matches[0], same_name_streams=len(lanes))


def match_database(db, markers, kernels, mapping):
    """Verify the DB epoch with Host anchors and device units with dedicated Adds."""
    required = {
        "TASK": {"startNs", "endNs", "deviceId", "streamId", "globalTaskId"},
        "COMPUTE_TASK_INFO": {"globalTaskId", "name"},
        "STRING_IDS": {"id", "value"},
        "MSTX_EVENTS": {"startNs", "endNs", "message"},
    }
    for table, fields in required.items():
        if not fields <= columns(db, table):
            raise ValueError(f"unsupported DB schema: {table} needs {sorted(fields)}")
    rows = db.execute(
        "SELECT s.value, m.startNs, m.endNs FROM MSTX_EVENTS m "
        "JOIN STRING_IDS s ON s.id=m.message WHERE s.value LIKE ?",
        (MARKER_PREFIX + "%",),
    ).fetchall()
    if len(rows) != len(markers) or len(rows) < MIN_HOST_ANCHORS or len({row[0] for row in rows}) != len(rows):
        raise ValueError("DB must contain all unique same-capture Host calibration markers")
    shifts = []
    for name, start, end in rows:
        if name not in markers:
            raise ValueError("DB/trace Host calibration marker names differ")
        shifts.extend((markers[name][0] - Decimal(start), markers[name][1] - Decimal(end)))
    epoch = sorted(shifts)[len(shifts) // 2]
    if max(abs(shift - epoch) for shift in shifts) > MATCH_TOLERANCE_NS:
        raise ValueError("DB Host timestamps do not share the JSON clock rate")
    device_ids = db.execute("SELECT DISTINCT deviceId FROM TASK").fetchall()
    if len(device_ids) != 1 or device_ids[0][0] is None:
        raise ValueError("only a single identified device DB is supported")
    selected = match_add_stream(db, kernels, mapping, epoch)
    return epoch, {
        "host_anchors": len(rows),
        "device_anchors": selected["count"],
        "max_anchor_error_ns": selected["max_anchor_error_ns"],
        "device_id": selected["device"],
        "calibration_stream_id": selected["stream"],
        "same_name_streams": selected["same_name_streams"],
    }


def convert_device_ns(value, mapping, db_epoch_ns):
    if value is None:
        return None
    origin = Decimal(mapping["trace_origin_us"]) * NS_PER_US - db_epoch_ns
    offset = Decimal(mapping["device_offset_us"]) * NS_PER_US
    result = origin + (Decimal(value) - origin - offset) / Decimal(mapping["device_scale"])
    integer = int(result.to_integral_value())
    if not 0 <= integer < 2**63:
        raise ValueError("converted timestamp outside SQLite signed integer range")
    return integer


def copy_aligned_database(source, destination, anchors, mapping, aligned_kernels):
    """Change task timestamps in an independent DB copy; IDs/Host tables stay intact.

    TASK.startNs/endNs are the Ascend Hardware timeline in MindStudio Insight's
    DbTraceDataBase. The raw JSON and DB must first agree at all calibration Adds.
    Other device domains are not inferred from the presence of a deviceId column.
    """
    for suffix in ("-wal", "-journal"):
        journal = Path(str(source) + suffix)
        if journal.exists() and journal.stat().st_size:
            raise ValueError("DB has an active journal; use a completed export")
    markers, kernels = anchors
    with contextlib.closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as incoming:
        epoch, checked = match_database(incoming, markers, kernels, mapping)
        # Communication uses additional domains which this single-card exporter
        # has not calibrated. Do not publish a partly corrected communication view.
        for table in ("COMMUNICATION_OP", "CCU", "DPU_TASK"):
            if columns(incoming, table) and incoming.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone():
                raise ValueError(f"unsupported populated device domain: {table}")
        with contextlib.closing(sqlite3.connect(destination)) as outgoing:
            incoming.backup(outgoing)
            outgoing.execute("PRAGMA journal_mode=DELETE")
            outgoing.create_function("aligned_ns", 1, lambda value: convert_device_ns(value, mapping, epoch))
            changed = {}
            with outgoing:
                for table in ("TASK", "OVERLAP_ANALYSIS"):
                    fields = columns(outgoing, table)
                    if not fields:
                        continue
                    if not {"startNs", "endNs"} <= fields:
                        raise ValueError(f"unsupported timestamp schema: {table}")
                    invalid = outgoing.execute(
                        f'SELECT 1 FROM "{table}" WHERE startNs IS NULL OR endNs IS NULL '
                        "OR startNs < 0 OR endNs < startNs LIMIT 1"
                    ).fetchone()
                    if invalid:
                        raise ValueError(f"invalid original device interval in {table}")
                    changed[table] = outgoing.execute(
                        f'UPDATE "{table}" SET startNs=aligned_ns(startNs),endNs=aligned_ns(endNs)'
                    ).rowcount
            if outgoing.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("corrected DB integrity check failed")
            # Verify that the stored, rounded ns coordinates implement the JSON mapping.
            _, after = match_database(outgoing, markers, aligned_kernels, mapping)
    return {
        "file": source.name,
        "changed_rows": changed,
        "trace_minus_db_epoch_ns": str(epoch),
        "before_check": checked,
        "after_check": after,
        "scope": "Ascend Hardware TASK and derived OVERLAP_ANALYSIS; Host MSTX/API unchanged",
    }


def export_capture(root, mapping, destination):
    """Publish a fresh standard *_ascend_pt directory, never rewrite the capture."""
    root, destination = root.resolve(strict=True), destination.resolve()
    if destination == root or root in destination.parents or destination in root.parents or destination.exists():
        raise ValueError("aligned directory must be new and outside the original capture")
    if not destination.name.endswith("_ascend_pt"):
        raise ValueError("aligned directory name must end in _ascend_pt")
    if mapping.get("status") != "calibration_validated" or mapping["held_out_after"]["outside_bracket_1us_tolerance"]:
        raise ValueError("same-capture alignment has not passed independent validation")
    output = root / OUTPUT_NAME
    trace, aligned = output / "trace_view.json", output / "trace_view_aligned.json"
    if not trace.is_file() or not aligned.is_file():
        raise ValueError("missing original/aligned trace pair")
    anchors = trace_anchors(trace, mapping)
    aligned_markers, aligned_kernels = trace_anchors(aligned, mapping)
    if anchors[0] != aligned_markers:
        raise ValueError("aligned JSON changed Host calibration markers")
    for original, corrected in zip(anchors[1], aligned_kernels):
        if any(
            abs(Decimal(convert_device_ns(old, mapping, Decimal(0))) - new) > 1 for old, new in zip(original, corrected)
        ):
            raise ValueError("aligned JSON does not implement its saved device mapping")
    databases = sorted(p for p in output.iterdir() if DB_NAME.fullmatch(p.name))
    if len(databases) > 1:
        raise ValueError("multiple profiler databases; no automatic DB selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed build remains outside the import path and is removed by TemporaryDirectory.
    with tempfile.TemporaryDirectory(prefix=".aligned_", dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        target = stage / OUTPUT_NAME
        target.mkdir(parents=True)
        copied, skipped = [], []
        for item in sorted(output.iterdir()):
            if item.name == "trace_view_aligned.json":
                shutil.copy2(item, target / "trace_view.json")
            elif (
                item.is_file()
                and item.suffix.lower() in (".csv", ".json")
                and item.name not in ("trace_view.json", "clock_alignment.json")
            ):
                # Auxiliary reports retain their original statistical values. They
                # are not advertised as recalculated cross-domain measurements.
                shutil.copy2(item, target / item.name)
                copied.append(item.name)
            else:
                skipped.append(item.name)
        database = None
        if databases:
            database = copy_aligned_database(
                databases[0], target / databases[0].name, anchors, mapping, aligned_kernels
            )
            skipped.remove(databases[0].name)
        for item in sorted(root.glob("profiler_*.json")):
            shutil.copy2(item, stage / item.name)
        report = {
            "status": "exported",
            "directory": str(destination),
            "source_capture": str(root),
            "format": "db" if database else "text",
            "database": database,
            "copied_auxiliary_reports": copied,
            "auxiliary_report_scope": "original statistics; not recalculated using corrected cross-domain overlap",
            "excluded_files": skipped,
            "raw_framework_and_cann_data_copied": False,
            "gui_import_verified": False,
        }
        (stage / "aligned_profile_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (stage / "clock_alignment.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        stage.rename(destination)
    return report


def export_run(run):
    roots = discover_roots([run / "profile"])
    if not roots:
        raise ValueError("no captures in RUN/profile")
    failed = False
    for root in roots:
        try:
            mapping = json.loads((root / OUTPUT_NAME / "clock_alignment.json").read_text(encoding="utf-8"))
            target = run / "profile_aligned" / root.relative_to(run / "profile")
            result = export_capture(root, mapping, target)
            print("ALIGNED_PROFILE " + json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as error:
            failed = True
            print(f"ALIGNED_PROFILE_FAILED {root}: {type(error).__name__}: {error}", flush=True)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing comparison run with validated alignment")
    args = parser.parse_args(argv)
    return export_run(args.run.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
