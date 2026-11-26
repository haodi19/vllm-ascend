# SPDX-License-Identifier: Apache-2.0
"""Case directories for new runs; read-only compatibility with flat reports."""

import json

CASES = ("python", "native", "native_full")


def saved_timing_records(root, cases=CASES):
    """Structured results take precedence over the old appended console log."""
    records = {}
    for case in cases:
        path = saved_case_file(root, case, "timing_result.json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or row.get("case") != case:
                raise ValueError(f"{case}: saved timing case mismatch")
            records[case] = row
    missing = set(cases) - records.keys()
    for name in ("run_details.log", "summary.txt"):
        path = root / name
        legacy = {}
        if not missing or not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if not line.startswith("RESULT "):
                    continue
                row = json.loads(line[7:])
                case = row["case"]
                if case not in CASES:
                    raise ValueError(f"Unknown timing case: {case}")
                if case not in missing:
                    continue
                if case in legacy:
                    raise ValueError(f"Duplicate timing result for {case}")
                legacy[case] = row
        records.update(legacy)
        missing -= legacy.keys()
    return records


def case_directory(root, case):
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def saved_case_file(root, case, filename):
    """Prefer the grouped artifact; never create directories while reading."""
    grouped = root / case / filename
    return grouped if grouped.exists() else root / f"{case}_{filename}"
