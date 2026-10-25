"""Hourly TP.HCM traffic-speed profile used by the prototype.

The temporal shape is taken from the HCMC_reference sheet supplied by the team.
The public prototype/reporting view intentionally uses a single TP.HCM profile so
the interface remains consistent with the current study scope.

This module is deliberately small and deterministic so the prototype can run
offline once it has been downloaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DAY_NAMES_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
REGION_LABELS = {
    "HCMC": "TP.HCM",
}

# Rows = hour 0..23, columns = Monday..Sunday.
HCMC_REFERENCE: List[List[float]] = [
    [26,26,25,26,26,26,25], [27,27,25,26,26,26,26],
    [27,27,26,27,27,27,27], [28,28,27,28,28,28,27],
    [28,27,27,27,27,28,28], [26,26,26,26,26,26,27],
    [22,22,22,23,23,25,26], [18,18,18,18,18,23,24],
    [17,17,17,17,17,21,23], [18,17,18,17,17,19,21],
    [19,18,18,18,18,18,21], [19,18,18,18,17,18,21],
    [20,19,19,19,18,18,21], [20,20,20,19,19,19,22],
    [19,19,19,19,18,19,21], [19,19,19,18,17,19,21],
    [18,17,17,17,16,19,20], [15,15,15,15,14,18,19],
    [15,15,15,15,14,17,18], [18,18,18,17,16,18,19],
    [20,20,20,20,18,18,19], [21,21,21,21,19,19,21],
    [23,22,22,22,21,20,22], [25,24,24,24,24,23,24],
]

HCMC_MEAN = 21.1
REGION_MEAN = {"HCMC": 21.1}
REGION_SCALE = {k: REGION_MEAN[k] / HCMC_MEAN for k in REGION_MEAN}


def region_matrix(region: str) -> List[List[float]]:
    region = region.upper()
    scale = REGION_SCALE.get(region, 1.0)
    return [[round(v * scale, 1) for v in row] for row in HCMC_REFERENCE]


def speed_kmh(region: str, absolute_minute: float, start_weekday: int = 0) -> float:
    """Return calibrated speed for absolute simulation minute.

    Day 0 is Monday by default.  For a multi-day scenario the weekday advances
    automatically and wraps after Sunday.
    """
    day_idx = max(0, int(float(absolute_minute) // 1440))
    minute_of_day = float(absolute_minute) % 1440
    hour = int(minute_of_day // 60) % 24
    weekday = (int(start_weekday) + day_idx) % 7
    return region_matrix(region)[hour][weekday]


def profile_for_day(region: str, day_index: int = 0, start_weekday: int = 0) -> List[Dict[str, float]]:
    weekday = (int(start_weekday) + int(day_index)) % 7
    matrix = region_matrix(region)
    return [{"hour": h, "speed_kmh": matrix[h][weekday]} for h in range(24)]


def traffic_payload(region: str, absolute_minute: float, start_weekday: int = 0) -> Dict:
    day_idx = max(0, int(float(absolute_minute) // 1440))
    weekday = (int(start_weekday) + day_idx) % 7
    hour = int((float(absolute_minute) % 1440) // 60) % 24
    current = speed_kmh(region, absolute_minute, start_weekday)
    profile = profile_for_day(region, day_idx, start_weekday)
    speeds = [x["speed_kmh"] for x in profile]
    return {
        "region": region,
        "region_label": REGION_LABELS.get(region, region),
        "day_index": day_idx,
        "weekday_index": weekday,
        "weekday_label": DAY_NAMES_VI[weekday],
        "hour": hour,
        "speed_kmh": current,
        "regional_mean_kmh": REGION_MEAN.get(region, HCMC_MEAN),
        "scale_factor": round(REGION_SCALE.get(region, 1.0), 6),
        "profile": profile,
        "day_min_kmh": min(speeds),
        "day_max_kmh": max(speeds),
        "source_note": "Hồ sơ tốc độ TP.HCM theo HCMC_reference của nhóm",
    }
