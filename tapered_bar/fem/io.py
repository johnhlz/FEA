"""CSV/JSON helpers."""
from __future__ import annotations

import csv
import json
import os
from typing import Iterable, Mapping


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(path: str, rows: Iterable[Mapping], fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames required for empty rows")
        fieldnames = list(rows[0].keys())
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_csv(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fnum(x, fmt: str = "{:.8e}"):
    """Format a number for CSV, returning '' for None/NaN."""
    try:
        if x is None:
            return ""
        if x != x:  # NaN
            return ""
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return str(x)


__all__ = ["ensure_dir", "save_csv", "load_csv", "save_json", "load_json", "fnum"]
