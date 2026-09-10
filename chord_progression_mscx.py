#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export harmony_plan from a ScaleWeaver JSON as block chords in MSCX."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mscx_export


def _segments(bar: dict, beats_per_bar: float) -> list[dict]:
    rows = bar.get("chord_segments")
    if isinstance(rows, list) and rows:
        return rows
    # Compatibility with older files containing one chord directly on the bar.
    return [{
        "offset": 0.0,
        "duration": beats_per_bar,
        "voicing_pcs": bar.get("voicing_pcs"),
        "chord_id": bar.get("chord_id"),
        "chord_name": bar.get("chord_name"),
    }]


def unfold_voicing(pcs, edo: int, root_octave: int) -> tuple[int, ...]:
    """Preserve the first PC as root and unfold the remaining PCs above it."""
    values = tuple(int(pc) % edo for pc in pcs)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError(f"invalid chord pitch classes: {pcs!r}")
    root = values[0]
    root_step = root + int(root_octave) * edo
    return tuple(root_step + (pc - root) % edo for pc in values)


def make_chord_score(source: dict, root_octave: int = -1) -> dict:
    plan = source.get("harmony_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError('input JSON must contain a nonempty "harmony_plan" list')
    tuning = source.get("tuning")
    if not isinstance(tuning, dict) or str(tuning.get("system", "edo")).lower() != "edo":
        raise ValueError("input JSON must contain EDO tuning information")

    edo = int(tuning["edo"])
    beats_per_bar = float(source.get("beats_per_bar", 4.0))
    bars = int(source.get("bars", len(plan)))
    if bars < 1 or len(plan) < bars:
        raise ValueError("harmony_plan is shorter than the declared bar count")

    events = []
    for fallback_bar, bar in enumerate(plan[:bars]):
        bar_number = int(bar.get("bar", fallback_bar))
        for segment in _segments(bar, beats_per_bar):
            pcs = segment.get("voicing_pcs", segment.get("pcs"))
            if not isinstance(pcs, (list, tuple)) or not pcs:
                raise ValueError(f"bar {bar_number + 1} has a chord without voicing_pcs")
            offset = float(segment.get("offset", 0.0))
            duration = float(segment.get("duration", beats_per_bar))
            if offset < 0.0 or duration <= 0.0 or offset + duration > beats_per_bar + 1e-6:
                raise ValueError(f"bar {bar_number + 1} contains an invalid chord duration")
            start = bar_number * beats_per_bar + offset
            steps = unfold_voicing(pcs, edo, root_octave)
            # mscx_export merges equal-start/equal-duration events on one staff
            # into one MuseScore <Chord>, so emit one event per chord tone.
            for step in steps:
                events.append({
                    "start_beat": start,
                    "duration_beats": duration,
                    "step": step,
                    "role": "chords",
                    "harmony": segment.get("chord_id", ""),
                })

    result = {
        "format": "ScaleWeaverChordProgression/1",
        "seed": source.get("seed"),
        "tempo_bpm": float(source.get("tempo_bpm", 96.0)),
        "time_signature": str(source.get("time_signature", "4/4")),
        "beats_per_bar": beats_per_bar,
        "bars": bars,
        "tuning": copy.deepcopy(tuning),
        "voices": {"chords": events},
        "staff_order": ["chords"],
        "voice_layout": {"staff_order": ["chords"]},
    }
    return result


def export_file(input_json: Path, output_mscx: Path, root_octave: int, continuous: bool) -> None:
    source = json.loads(input_json.read_text(encoding="utf-8"))
    chord_score = make_chord_score(source, root_octave=root_octave)
    xml = mscx_export.generate(
        chord_score,
        staff_order=("chords",),
        layout_mode="system" if continuous else "page",
    )
    output_mscx.parent.mkdir(parents=True, exist_ok=True)
    output_mscx.write_text(xml, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_json", type=Path, help="ScaleWeaver output JSON")
    parser.add_argument("out_mscx", type=Path, nargs="?", help="output MSCX path")
    parser.add_argument(
        "--root-octave", type=int, default=-1,
        help="octave displacement of each chord root relative to the configured base note (default: -1)",
    )
    parser.add_argument("--continuous", action="store_true", help="use MuseScore continuous layout")
    args = parser.parse_args(argv)
    output = args.out_mscx or args.score_json.with_name(args.score_json.stem + "_chords.mscx")
    export_file(args.score_json, output, args.root_octave, args.continuous)
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
