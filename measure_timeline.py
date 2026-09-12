"""Shared helpers for uniform and variable-length measure timelines."""
from __future__ import annotations

import bisect
import math


EPS = 1e-8


def uniform_measure_map(bars, beats_per_bar, time_signature="4/4"):
    bpb = float(beats_per_bar)
    return [
        {
            "bar": bar,
            "start_beat": round(bar * bpb, 10),
            "duration_beats": bpb,
            "time_signature": str(time_signature),
        }
        for bar in range(int(bars))
    ]


def resolved_measure_map(container):
    """Return a validated map, synthesizing the legacy uniform map if absent."""
    rows = container.get("measure_map") if isinstance(container, dict) else None
    if rows is None:
        rows = uniform_measure_map(
            int(container["bars"]), float(container["beats_per_bar"]),
            str(container.get("time_signature", "4/4")))
    if not isinstance(rows, list) or len(rows) != int(container["bars"]):
        raise ValueError("measure_map must contain exactly bars entries")
    out = []
    cursor = 0.0
    for index, row in enumerate(rows):
        start = float(row.get("start_beat", cursor))
        duration = float(row["duration_beats"])
        if (int(row.get("bar", index)) != index or not math.isfinite(start)
                or not math.isfinite(duration) or duration <= 0
                or not math.isclose(start, cursor, abs_tol=EPS)):
            raise ValueError(f"invalid measure_map entry {index}")
        time_signature = str(row.get("time_signature", container.get("time_signature", "4/4")))
        parts = time_signature.split("/", 1)
        if len(parts) != 2 or int(parts[0]) <= 0 or int(parts[1]) <= 0:
            raise ValueError(f"invalid time signature in measure_map entry {index}")
        out.append({
            "bar": index,
            "start_beat": round(start, 10),
            "duration_beats": duration,
            "time_signature": time_signature,
        })
        cursor = start + duration
    return out


def total_beats(measure_map):
    if not measure_map:
        return 0.0
    row = measure_map[-1]
    return float(row["start_beat"]) + float(row["duration_beats"])


def measure_index_at(measure_map, beat, *, end_in_previous=False):
    """Locate an absolute quarter-note beat in a half-open measure timeline."""
    if not measure_map:
        raise ValueError("empty measure_map")
    starts = [float(row["start_beat"]) for row in measure_map]
    value = float(beat) - (EPS if end_in_previous else 0.0)
    index = bisect.bisect_right(starts, value) - 1
    return max(0, min(len(measure_map) - 1, index))


def locate(measure_map, beat, *, end_in_previous=False):
    index = measure_index_at(measure_map, beat, end_in_previous=end_in_previous)
    return index, float(beat) - float(measure_map[index]["start_beat"])


def is_uniform(measure_map):
    if not measure_map:
        return True
    first = measure_map[0]
    return all(
        math.isclose(float(row["duration_beats"]), float(first["duration_beats"]), abs_tol=EPS)
        and str(row["time_signature"]) == str(first["time_signature"])
        for row in measure_map
    )
