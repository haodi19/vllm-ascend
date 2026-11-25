# SPDX-License-Identifier: Apache-2.0
"""Case directories for new runs; read-only compatibility with flat reports."""


def case_directory(root, case):
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def saved_case_file(root, case, filename):
    """Prefer the grouped artifact; never create directories while reading."""
    grouped = root / case / filename
    return grouped if grouped.exists() else root / f"{case}_{filename}"
