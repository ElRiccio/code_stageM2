"""Generic CSV-writing helper: the tabular counterpart to
plotting.save_figure, kept separate from it since one writes figures and
the other writes tables (constraint: separate experiment code from
analysis/plotting code; a table is analysis output, not a figure, but the
two share nothing beyond "write to outdir/name" so they stay as two small
generic functions rather than one that overloads its own contract).
"""

from __future__ import annotations

import csv
import os


def save_csv(rows: list[dict], outdir: str, name: str) -> str:
    """Writes `rows` (a list of dicts sharing the same keys) as a CSV file
    at outdir/name, creating outdir if needed. Column order follows the
    first row's key order. Returns the full path written."""
    if not rows:
        raise ValueError("rows must be non-empty")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
