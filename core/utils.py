"""
utils.py
========
Small shared helpers: seeding, time formatting, CSV logging, fairness statistics, timers.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import statistics
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


# ------------------------------------------------------------------------------ seeding
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:      # pragma: no cover - torch always present in practice
        pass


# ------------------------------------------------------------------------------- clock
def hhmm(minutes: float) -> str:
    """Minutes since midnight -> 'HH:MM'."""
    minutes = float(minutes)
    h = int(minutes // 60) % 24
    m = int(round(minutes - 60 * (minutes // 60)))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def parse_clock(value: Any) -> float:
    """Accepts 'HH:MM', 'HH:MM:SS', datetime/time objects, or a raw number of minutes."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hasattr(value, "hour"):                     # datetime.time / datetime.datetime
        return float(value.hour * 60 + value.minute + getattr(value, "second", 0) / 60.0)
    text = str(value).strip()
    if not text:
        return 0.0
    if " " in text:                                # '2024-05-01 08:30:00'
        text = text.split(" ")[-1]
    if ":" in text:
        parts = text.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return float(h * 60 + m + s / 60.0)
    return float(text)


# ------------------------------------------------------------------------------ logging
def get_logger(name: str = "dfpdp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class CsvLogger:
    """Append-only CSV writer that creates the header from the first row."""

    def __init__(self, path: str, fieldnames: Optional[List[str]] = None):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = None
        self._writer = None
        if os.path.exists(path):
            os.remove(path)

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self.fieldnames = self.fieldnames or list(row.keys())
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def write_all(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
            self._writer = None


def write_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as fh:
            fh.write("")
        return path
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path


# ----------------------------------------------------------------------------- fairness
def fairness_stats(working_times: List[float]) -> Dict[str, float]:
    """[PAPER] fairness is reported, not rewarded (baseline)."""
    vals = [float(v) for v in working_times] or [0.0]
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals)
    return {
        "fairness_gap": max(vals) - min(vals),
        "fairness_std": std,
        "fairness_cv": (std / mean) if mean > 1e-9 else 0.0,
        "workload_mean": mean,
        "workload_max": max(vals),
        "workload_min": min(vals),
    }


def latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


# -------------------------------------------------------------------------------- timer
@contextmanager
def timer():
    """with timer() as t: ...   -> t['ms'] holds the elapsed wall-clock time."""
    box: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000.0
        box["s"] = box["ms"] / 1000.0


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default
