"""Versioned intermediate representation between ScaleWeaver's two stages.

The front end owns the harmony progression, Lead rhythm and Lead pitches.  The
accompaniment stage treats all three as immutable input and creates the three
or four lower lines plus their annealed pitches.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from measure_timeline import resolved_measure_map, total_beats, uniform_measure_map


FORMAT = "ScaleWeaverFrontEndIR/1"
EPS = 1e-8


class FrontEndIRError(ValueError):
    """Raised when a front-end IR cannot be consumed safely."""


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FrontEndIRError(f"{label} must be finite")
    return value


def scale_identity(spec):
    """Return the tuning identity required to reject a mismatched scale."""
    return {
        "scale_id": str(spec.id),
        "scale_name": str(spec.name),
        "edo": int(spec.edo),
        "base_note": str(spec.base_note),
        "base_freq_hz": float(spec.base_freq_hz),
        "pcs": [int(x) for x in spec.pcs],
        "names": [str(x) for x in spec.names],
    }


def lead_rhythm_bars(lead, bars, beats_per_bar, measure_map=None,
                     time_signature="4/4"):
    """Expose the Lead timing skeleton independently of its pitch fields."""
    timeline = (uniform_measure_map(bars, beats_per_bar, time_signature)
                if measure_map is None else measure_map)
    starts = [float(row["start_beat"]) for row in timeline]
    rows = [{"bar": bar, "events": []} for bar in range(int(bars))]
    for event_index, event in enumerate(lead):
        start = float(event["start_beat"])
        import bisect
        bar = min(int(bars) - 1, max(0, bisect.bisect_right(starts, start + EPS) - 1))
        rows[bar]["events"].append({
            "event_index": int(event_index),
            "offset": round(start - starts[bar], 10),
            "duration_beats": float(event["duration_beats"]),
        })
    return rows


def build_ir(*, seed, bpm, time_signature, beats_per_bar, bars, spec,
             harmony_plan, lead, allow_sixteenth, lead_palette=None,
             motif_degrees=None, generator="melodyplan_joint",
             melody_plan=None, generator_metadata=None, measure_map=None):
    """Build a JSON-serializable IR from one front-end realization."""
    bars = int(bars)
    bpb = float(beats_per_bar)
    timeline = (uniform_measure_map(bars, bpb, time_signature)
                if measure_map is None else copy.deepcopy(measure_map))
    data = {
        "format": FORMAT,
        "seed": int(seed),
        "tempo_bpm": float(bpm),
        "time_signature": str(time_signature),
        "beats_per_bar": bpb,
        "bars": bars,
        "duration_seconds": total_beats(timeline) * 60.0 / float(bpm),
        "measure_map": timeline,
        "scale": scale_identity(spec),
        "generator": {
            "name": str(generator),
            "allow_sixteenth": bool(allow_sixteenth),
            "motif_degrees": None if motif_degrees is None else list(motif_degrees),
            "resolved_chord_progression": copy.deepcopy(
                spec.resolved_chord_progression()),
        },
        "harmony_plan": copy.deepcopy(harmony_plan),
        "lead": copy.deepcopy(lead),
        "lead_rhythm": {
            "representation": "per_bar_event_timing",
            "bars": lead_rhythm_bars(lead, bars, bpb, timeline, time_signature),
            "palette": copy.deepcopy(lead_palette),
        },
    }
    if generator_metadata:
        data["generator"].update(copy.deepcopy(generator_metadata))
    if melody_plan is not None:
        data["melody_plan"] = copy.deepcopy(melody_plan)
    validate_ir(data, spec=spec)
    return data


def _validate_scale(data, spec):
    expected = scale_identity(spec)
    actual = data.get("scale")
    if not isinstance(actual, dict):
        raise FrontEndIRError("scale must be an object")
    for key in ("scale_id", "edo", "pcs"):
        if actual.get(key) != expected[key]:
            raise FrontEndIRError(
                f"IR scale mismatch for {key}: {actual.get(key)!r} != {expected[key]!r}")
    if not math.isclose(float(actual.get("base_freq_hz", math.nan)),
                        expected["base_freq_hz"], rel_tol=0.0, abs_tol=1e-9):
        raise FrontEndIRError("IR base frequency does not match the active scale")


def _validate_harmony(plan, bars, measure_map, chord_ids):
    if not isinstance(plan, list) or len(plan) != bars:
        raise FrontEndIRError(f"harmony_plan must contain exactly {bars} bars")
    for bar, row in enumerate(plan):
        if not isinstance(row, dict) or int(row.get("bar", -1)) != bar:
            raise FrontEndIRError(f"harmony_plan[{bar}] has the wrong bar index")
        segments = row.get("chord_segments")
        if not isinstance(segments, list) or not segments:
            raise FrontEndIRError(f"harmony_plan[{bar}] has no chord segments")
        cursor = 0.0
        for index, segment in enumerate(sorted(segments, key=lambda x: float(x["offset"]))):
            offset = _finite(segment.get("offset"), f"harmony_plan[{bar}].offset")
            duration = _finite(segment.get("duration"), f"harmony_plan[{bar}].duration")
            if duration <= 0 or not math.isclose(offset, cursor, abs_tol=EPS):
                raise FrontEndIRError(
                    f"harmony_plan[{bar}] segment {index} leaves a gap or overlap")
            chord_id = segment.get("chord_id")
            if not isinstance(chord_id, str) or not chord_id:
                raise FrontEndIRError(f"harmony_plan[{bar}] segment {index} lacks chord_id")
            if chord_ids is not None and chord_id not in chord_ids:
                raise FrontEndIRError(f"IR chord {chord_id!r} is absent from the active scale")
            cursor = offset + duration
        expected_duration = float(measure_map[bar]["duration_beats"])
        if not math.isclose(cursor, expected_duration, abs_tol=EPS):
            raise FrontEndIRError(f"harmony_plan[{bar}] does not fill the bar")


def _validate_lead(lead, total):
    if not isinstance(lead, list) or not lead:
        raise FrontEndIRError("lead must be a non-empty event list")
    previous_start = -math.inf
    previous_end = -math.inf
    for index, event in enumerate(lead):
        if not isinstance(event, dict):
            raise FrontEndIRError(f"lead[{index}] must be an object")
        start = _finite(event.get("start_beat"), f"lead[{index}].start_beat")
        duration = _finite(event.get("duration_beats"), f"lead[{index}].duration_beats")
        step = event.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise FrontEndIRError(f"lead[{index}].step must be an integer")
        if duration <= 0 or start < -EPS or start + duration > total + EPS:
            raise FrontEndIRError(f"lead[{index}] lies outside the score")
        if start < previous_start - EPS:
            raise FrontEndIRError("lead events must be sorted by start_beat")
        if start < previous_end - EPS:
            raise FrontEndIRError(f"lead[{index}] overlaps the previous Lead event")
        previous_start = start
        previous_end = start + duration


def _validate_rhythm(data, lead, bars, bpb, measure_map):
    rhythm = data.get("lead_rhythm")
    if not isinstance(rhythm, dict):
        raise FrontEndIRError("lead_rhythm must be an object")
    expected = lead_rhythm_bars(
        lead, bars, bpb, measure_map, data.get("time_signature", "4/4"))
    if rhythm.get("bars") != expected:
        raise FrontEndIRError("lead_rhythm does not match the Lead event timing")


def validate_ir(data, *, spec=None, chord_ids=None):
    """Validate the public boundary without altering the supplied object."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise FrontEndIRError(f"format must be {FORMAT!r}")
    bars = data.get("bars")
    if isinstance(bars, bool) or not isinstance(bars, int) or bars <= 0:
        raise FrontEndIRError("bars must be a positive integer")
    bpb = _finite(data.get("beats_per_bar"), "beats_per_bar")
    bpm = _finite(data.get("tempo_bpm"), "tempo_bpm")
    if bpb <= 0 or bpm <= 0:
        raise FrontEndIRError("beats_per_bar and tempo_bpm must be positive")
    try:
        measure_map = resolved_measure_map(data)
    except (ValueError, TypeError, KeyError) as exc:
        raise FrontEndIRError(str(exc)) from exc
    if spec is not None:
        _validate_scale(data, spec)
        if chord_ids is None:
            try:
                import harmony_rhythm as hr
                chord_ids = set(hr.CHORDS)
            except Exception:
                chord_ids = None
    _validate_harmony(data.get("harmony_plan"), bars, measure_map, chord_ids)
    _validate_lead(data.get("lead"), total_beats(measure_map))
    _validate_rhythm(data, data["lead"], bars, bpb, measure_map)
    if "melody_plan" in data:
        from melody_plan import validate as validate_melody_plan
        plan = validate_melody_plan(data["melody_plan"])
        if int(plan["bars"]) != bars or not math.isclose(
                float(plan["beats_per_bar"]), bpb, abs_tol=EPS):
            raise FrontEndIRError("melody_plan duration does not match the front-end IR")
    return data


def content_sha256(data):
    """Stable fingerprint embedded in the final score, not enforced on edits."""
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_ir(path, *, spec=None, chord_ids=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_ir(data, spec=spec, chord_ids=chord_ids)


def save_ir(path, data, *, spec=None, chord_ids=None):
    path = Path(path)
    validate_ir(data, spec=spec, chord_ids=chord_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
