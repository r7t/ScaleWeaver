#!/usr/bin/env python3
"""Create a ScaleWeaver front-end IR from an MSCX rhythm template.

The reference supplies only barlines, time signatures, rests, attacks and
durations.  Source pitches, intervals and directions are deliberately excluded
from composition: target pitches are written from the configured scale under
its own background harmony field and native melodic priors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import harmony_rhythm as hr
import score_builder as sb
from frontend_ir import build_ir, save_ir
from measure_timeline import resolved_measure_map, total_beats
from mscx_roundtrip import attach_roundtrip


EPS = 1e-8
REFERENCE_FORMAT = "ScaleWeaverMSCXReferenceIR/1"
_DURATION_BEATS = {
    "longa": 16.0, "breve": 8.0, "whole": 4.0, "half": 2.0,
    "quarter": 1.0, "eighth": .5, "16th": .25, "32nd": .125,
    "64th": .0625, "128th": .03125,
}


def _fraction(text):
    return float(Fraction(str(text).strip()))


def _duration(element, active_modification=None):
    kind = element.findtext("durationType")
    if kind == "measure":
        explicit = element.findtext("duration")
        return 4.0 * _fraction(explicit) if explicit else None
    if kind not in _DURATION_BEATS:
        raise ValueError(f"unsupported MSCX durationType {kind!r}")
    value = _DURATION_BEATS[kind]
    dots = int(element.findtext("dots", "0"))
    value *= sum(.5 ** i for i in range(dots + 1))
    modification = element.find("TimeModification")
    if modification is not None:
        actual = int(modification.findtext("actualNotes", "1"))
        normal = int(modification.findtext("normalNotes", "1"))
        value *= normal / actual
    elif active_modification is not None:
        actual, normal = active_modification
        value *= normal / actual
    return value


def _measure_explicit_duration(measure):
    text = measure.get("len") or measure.findtext("len")
    return 4.0 * _fraction(text) if text else None


_OTTAVA_SEMITONES = {
    "8va": 12.0, "15ma": 24.0, "22ma": 36.0,
    "8vb": -12.0, "15mb": -24.0, "22mb": -36.0,
}


def _note_playback_midi(note, ottava_semitones=0.0):
    """Actual sounding pitch, including MuseScore playback-event offsets.

    ScaleWeaver notation may respell a note and compensate with
    Note/Events/Event/pitch.  Reading only Note/pitch + tuning therefore gives
    the wrong frequency for exactly the scores this importer is meant to use.
    """
    written = float(note.findtext("pitch"))
    tuning = float(note.findtext("tuning", "0")) / 100.0
    playback = note.findtext("./Events/Event/pitch")
    event_offset = float(playback) if playback is not None else 0.0
    return written + tuning + event_offset + float(ottava_semitones)


def _voice_events(voice, initial_ottava=0.0):
    """Return cursor-timed top notes from one MuseScore voice."""
    cursor = 0.0
    rows = []
    tuplet_stack = []
    ottava = float(initial_ottava)
    for child in voice:
        if child.tag == "Spanner" and child.get("type") == "Ottava":
            definition = child.find("Ottava")
            if definition is not None:
                ottava = _OTTAVA_SEMITONES.get(
                    definition.findtext("subtype", ""), ottava)
            elif child.find("prev") is not None:
                ottava = 0.0
            continue
        if child.tag == "Tuplet":
            tuplet_stack.append((int(child.findtext("actualNotes", "3")),
                                 int(child.findtext("normalNotes", "2"))))
            continue
        if child.tag == "endTuplet":
            if tuplet_stack:
                tuplet_stack.pop()
            continue
        if child.tag not in ("Chord", "Rest"):
            continue
        duration = _duration(child, tuplet_stack[-1] if tuplet_stack else None)
        if duration is None:
            continue
        if child.tag == "Chord" and child.find("grace") is None:
            notes = child.findall("Note")
            if notes:
                top = max(notes, key=lambda note: _note_playback_midi(note, ottava))
                actual_midi = _note_playback_midi(top, ottava)
                rows.append({
                    "offset": cursor,
                    "duration_beats": duration,
                    "source_pitch": int(top.findtext("pitch")),
                    "source_actual_midi": round(actual_midi, 9),
                    "source_actual_cents": round(actual_midi * 100.0, 6),
                    "tie_start": top.find("./Spanner[@type='Tie']/Tie") is not None,
                    "tie_stop": top.find("./Spanner[@type='Tie']/prev") is not None,
                })
        cursor += duration
    return rows, cursor, ottava


def _staff_candidate(staff):
    active_signature = (4, 4)
    measures = []
    notes = []
    absolute = 0.0
    active_ottava = 0.0
    for bar, measure in enumerate(staff.findall("Measure")):
        voices = measure.findall("voice")
        for voice in voices:
            time_sig = voice.find("TimeSig")
            if time_sig is not None:
                active_signature = (int(time_sig.findtext("sigN")),
                                    int(time_sig.findtext("sigD")))
                break
        parsed = [_voice_events(voice, active_ottava) for voice in voices]
        if parsed:
            active_ottava = parsed[0][2]
        # The melodic voice is normally voice 1.  If several exist, select the
        # one whose top-note line sits highest in this measure.
        nonempty = [(rows, extent) for rows, extent, _ in parsed if rows]
        if nonempty:
            rows, extent = max(
                nonempty,
                key=lambda item: statistics.fmean(
                    x["source_actual_midi"] for x in item[0]))
        else:
            rows, extent = ([], max((x[1] for x in parsed), default=0.0))
        nominal = 4.0 * active_signature[0] / active_signature[1]
        explicit = _measure_explicit_duration(measure)
        duration = explicit if explicit is not None else max(nominal, extent)
        measures.append({
            "bar": bar,
            "start_beat": round(absolute, 10),
            "duration_beats": round(duration, 10),
            "time_signature": f"{active_signature[0]}/{active_signature[1]}",
        })
        for row in rows:
            item = dict(row)
            item["bar"] = bar
            item["start_beat"] = round(absolute + float(row["offset"]), 10)
            notes.append(item)
        absolute += duration
    pitches = [row["source_actual_midi"] for row in notes]
    return {
        "staff_id": staff.get("id"), "measure_map": measures, "notes": notes,
        "mean_top_pitch": statistics.fmean(pitches) if pitches else -math.inf,
    }


def extract_reference(path, staff_id=None):
    root = ET.parse(path).getroot()
    score = root.find("Score")
    if score is None:
        raise ValueError("MSCX has no Score")
    candidates = [_staff_candidate(staff) for staff in score.findall("Staff")]
    candidates = [row for row in candidates if row["notes"]]
    if staff_id is not None:
        candidates = [row for row in candidates if str(row["staff_id"]) == str(staff_id)]
    if not candidates:
        raise ValueError("MSCX contains no pitched staff matching the request")
    selected = max(candidates, key=lambda row: (row["mean_top_pitch"], len(row["notes"])))

    metadata = {row.get("name"): row.text or "" for row in score.findall("metaTag")}
    selected["source_file"] = metadata.get("scaleweaverReferenceSource", "")
    selected["source_staff_id"] = metadata.get(
        "scaleweaverReferenceStaff", selected["staff_id"])

    # Join the two notated halves of a real tie.  Untied repeated notes remain
    # separate attacks and therefore remain part of the copied rhythm.
    merged = []
    for row in selected["notes"]:
        if (merged and row["tie_stop"] and merged[-1]["tie_start"]
                and abs(float(row["source_actual_cents"])
                        - float(merged[-1]["source_actual_cents"])) <= .1
                and math.isclose(merged[-1]["start_beat"] + merged[-1]["duration_beats"],
                                 row["start_beat"], abs_tol=1e-7)):
            merged[-1]["duration_beats"] = round(
                merged[-1]["duration_beats"] + row["duration_beats"], 10)
            merged[-1]["tie_start"] = row["tie_start"]
        else:
            merged.append(dict(row))
    selected["notes"] = merged
    return selected


def _analysis_source_direction(previous, current):
    """Descriptive source metadata; never consumed by the composer."""
    if previous is None:
        return 0
    delta = int(current) - int(previous)
    return 1 if delta > 0 else -1 if delta < 0 else 0


def _analysis_leap_class(source_delta):
    """Descriptive source metadata retained for inspection/round trips only."""
    value = abs(int(source_delta))
    if value == 0:
        return 0
    if value <= 2:
        return 1
    if value <= 4:
        return 2
    if value <= 7:
        return 3
    return 4


def build_reference_ir(path, staff_id=None):
    extracted = extract_reference(path, staff_id=staff_id)
    previous = None
    events = []
    for index, row in enumerate(extracted["notes"]):
        source_pitch = int(row["source_pitch"])
        delta = 0 if previous is None else source_pitch - previous
        item = {
            "event_index": index,
            "bar": int(row["bar"]),
            "start_beat": float(row["start_beat"]),
            "duration_beats": float(row["duration_beats"]),
            "source_pitch": source_pitch,
            "source_actual_midi": float(row.get("source_actual_midi", source_pitch)),
            "source_actual_cents": float(row.get(
                "source_actual_cents", 100.0 * source_pitch)),
            "direction_from_previous": _analysis_source_direction(previous, source_pitch),
            "source_interval_semitones": delta,
            "source_leap_class": _analysis_leap_class(delta),
        }
        events.append(item)
        previous = source_pitch
    rhythm_timeline = []
    cursor = 0.0
    for event in events:
        start = float(event["start_beat"])
        if start > cursor + EPS:
            rhythm_timeline.append({
                "kind": "rest", "start_beat": cursor,
                "duration_beats": round(start - cursor, 10),
            })
        rhythm_timeline.append({
            "kind": "note", "event_index": int(event["event_index"]),
            "start_beat": start,
            "duration_beats": float(event["duration_beats"]),
        })
        cursor = max(cursor, start + float(event["duration_beats"]))
    end = total_beats(extracted["measure_map"])
    if end > cursor + EPS:
        rhythm_timeline.append({
            "kind": "rest", "start_beat": cursor,
            "duration_beats": round(end - cursor, 10),
        })
    result = {
        "format": REFERENCE_FORMAT,
        # A filesystem basename is intentionally not semantic: it changes when
        # a canonical MSCX is written under a new name and would break IR
        # equality on the next pass.  Canonical ScaleWeaver MSCX may explicitly
        # preserve a provenance label through the two metadata tags above.
        "source_file": extracted.get("source_file", ""),
        "source_staff_id": extracted.get("source_staff_id", extracted["staff_id"]),
        "bars": len(extracted["measure_map"]),
        "total_beats": total_beats(extracted["measure_map"]),
        "measure_map": extracted["measure_map"],
        "highest_line": events,
        "rhythm_timeline": rhythm_timeline,
        "pitch_semantics": "source_midi_for_analysis_and_mscx_roundtrip_only",
    }
    return attach_roundtrip(result, Path(path).read_text(encoding="utf-8"))


def validate_reference_ir(data):
    if not isinstance(data, dict) or data.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"reference IR format must be {REFERENCE_FORMAT!r}")
    if int(data.get("bars", 0)) != len(data.get("measure_map", ())):
        raise ValueError("reference IR bars/measure_map mismatch")
    resolved_measure_map(data)
    if not data.get("highest_line"):
        raise ValueError("reference IR highest_line is empty")
    previous_end = -math.inf
    for index, row in enumerate(data["highest_line"]):
        start = float(row["start_beat"])
        duration = float(row["duration_beats"])
        if duration <= 0 or start < previous_end - EPS:
            raise ValueError(f"invalid or overlapping reference event {index}")
        previous_end = start + duration
    return data


def load_reference(path, staff_id=None):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return validate_reference_ir(json.loads(path.read_text(encoding="utf-8")))
    return build_reference_ir(path, staff_id=staff_id)


def _scaled_harmony_plan(bars, rng, measure_map):
    nominal = float(measure_map[0]["duration_beats"])
    generator_bpb = 4.0 if math.isclose(nominal, 4.0) else 3.0
    plan = hr.harmony_plan(bars, rng, generator_bpb)
    for bar, row in enumerate(plan):
        duration = float(measure_map[bar]["duration_beats"])
        scale = duration / generator_bpb
        for segment in row["chord_segments"]:
            segment["offset"] = round(float(segment["offset"]) * scale, 6)
            segment["duration"] = round(float(segment["duration"]) * scale, 6)
        # Force exact coverage after decimal rounding.
        segment = row["chord_segments"][-1]
        segment["duration"] = round(
            duration - float(segment["offset"]), 6)
        row["bar"] = bar
        row["beats_per_bar"] = duration
        row["start_beat"] = float(measure_map[bar]["start_beat"])
        row["time_signature"] = str(measure_map[bar]["time_signature"])
        hr._refresh_primary_from_first_segment(row)
    return plan


def _strong_attack(offset, time_signature):
    n, d = map(int, str(time_signature).split("/", 1))
    if abs(float(offset)) < 1e-7:
        return True
    return d == 8 and n >= 6 and abs(float(offset) - 1.5) < 1e-7


def _native_bar_guides(bars, seed):
    """Create a target-scale contour independent of all source pitches."""
    target_centre = sb.degree(int(round(sb.LEAD_REGISTER_TARGET_STEP)))
    rng = random.Random(int(seed) ^ 0x4E4154495645)
    guides = []
    for phrase_start in range(0, int(bars), 8):
        count = min(8, int(bars) - phrase_start)
        kind = rng.choices(
            ("arch", "rise", "fall", "valley"),
            weights=(.46, .20, .24, .10), k=1)[0]
        centre = target_centre + sb._lead_centre_jitter(rng)
        amplitude = rng.choice((2, 3, 3, 4))
        guides.extend(sb._lead_contour(kind, count, centre, amplitude))
    return tuple(guides)


def _rhythm_template_lead(reference, plan, measure_map, seed, beam_width=48):
    # The caller strips the reference to timing fields before entering this
    # function.  Keeping that boundary explicit prevents accidental future use
    # of source pitch, interval or direction as a hidden generation prior.
    rhythm_events = reference["events"]
    bar_guides = _native_bar_guides(len(measure_map), seed)
    rng = random.Random(int(seed) ^ 0x52485954484D)
    beam = [{"cost": 0.0, "pitches": [], "events": [], "recent": []}]
    for index, row in enumerate(rhythm_events):
        bar = int(row["bar"])
        measure = measure_map[bar]
        offset = float(row["start_beat"]) - float(measure["start_beat"])
        seg = hr.harmony_segment_at(plan[bar], offset)
        chord = hr.CHORDS[seg["chord_id"]]
        progress = max(0.0, min(1.0, offset / float(measure["duration_beats"])))
        here = float(bar_guides[bar])
        after = float(bar_guides[min(len(bar_guides) - 1, bar + 1)])
        guide = int(round(here * (1.0 - progress) + after * progress))
        native_direction = 1 if after > here else -1 if after < here else 0
        strong = _strong_attack(offset, measure["time_signature"])
        event_end = offset + float(row["duration_beats"])
        phrase_end = (bar % 8 == 7
                      and event_end >= float(measure["duration_beats"]) - 1e-7)
        boundary = index == 0 or index == len(rhythm_events) - 1 or phrase_end
        expanded = []
        for state in beam:
            prev = state["pitches"][-1] if state["pitches"] else None
            prev2 = state["pitches"][-2] if len(state["pitches"]) > 1 else None
            rows = []
            for cand in sb.POOLS["lead"]:
                components = sb.lead_candidate_cost_components(
                    prev, prev2, chord, strong, state["recent"], cand, guide,
                    native_direction, boundary, offset, row["duration_beats"],
                    measure["duration_beats"])
                if components is None:
                    continue
                rows.append((components["total"], int(cand)))
            for cost, cand in sorted(rows)[:8]:
                event = sb.event(
                    row["start_beat"], row["duration_beats"], cand, "lead", rng, seg,
                    structural=strong, motif="mscx_rhythm_template",
                    rhythm_template_event_index=index,
                    rhythm_template_source_bar=bar,
                )
                expanded.append({
                    "cost": state["cost"] + cost,
                    "pitches": state["pitches"] + [cand],
                    "events": state["events"] + [event],
                    "recent": (state["recent"] + [cand])[-32:],
                })
        if not expanded:
            raise sb.GenerationRejected(f"MSCX imitation failed at reference note {index}")
        expanded.sort(key=lambda state: state["cost"])
        beam = expanded[:int(beam_width)]
    best = beam[0]
    return best["events"], {
        "copied_rhythm_events": len(best["events"]),
        "source_pitch_fields_used": False,
        "source_direction_fields_used": False,
        "target_contour_source": "scaleweaver_native_seeded_contour",
        "beam_width": int(beam_width),
        "objective": best["cost"],
    }


def generate_frontend_ir(reference_mscx, *, seed=20260811, bpm=96.0,
                         spec=None, staff_id=None, beam_width=48):
    if spec is None:
        spec = hr.SCALE
    reference_ir = load_reference(reference_mscx, staff_id=staff_id)
    reference = {
        "staff_id": reference_ir["source_staff_id"],
        "measure_map": reference_ir["measure_map"],
        # Deliberately copy timing only.  This is the hard architectural
        # boundary that makes the result invariant to reference pitch edits.
        "events": [
            {key: row[key] for key in
             ("event_index", "bar", "start_beat", "duration_beats")}
            for row in reference_ir["highest_line"]
        ],
    }
    measure_map = reference["measure_map"]
    bars = len(measure_map)
    harmony = _scaled_harmony_plan(bars, random.Random(int(seed)), measure_map)
    lead, imitation = _rhythm_template_lead(
        reference, harmony, measure_map, seed, beam_width=beam_width)
    first_signature = str(measure_map[0]["time_signature"])
    first_bpb = float(measure_map[0]["duration_beats"])
    return build_ir(
        seed=seed, bpm=bpm, time_signature=first_signature,
        beats_per_bar=first_bpb, bars=bars, spec=spec,
        harmony_plan=harmony, lead=lead, allow_sixteenth=True,
        lead_palette={
            "source": "mscx_highest_line_rhythm_only",
            "reference_file": reference_ir["source_file"],
            "reference_staff_id": reference["staff_id"],
        },
        generator="mscx_rhythm_imitation", measure_map=measure_map,
        generator_metadata={
            "reference_file": reference_ir["source_file"],
            "reference_ir_format": REFERENCE_FORMAT,
            "reference_staff_id": reference["staff_id"],
            "reference_note_count": len(reference["events"]),
            "reference_total_beats": total_beats(measure_map),
            "imitation": imitation,
            "pitch_policy": (
                "target_scale_harmony_and_native_melodic_priors; "
                "source_pitch_interval_direction_ignored"),
        },
    )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_mscx")
    parser.add_argument("--scale", dest="scale_config", required=True)
    parser.add_argument("--rules", dest="rules_config")
    parser.add_argument("--style", dest="style_config")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bpm", type=float, default=96.0)
    parser.add_argument("--staff-id")
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--cse-dir")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--output", "-o", default="imitation_frontend_ir.json")
    return parser


def cli(argv=None):
    args = _build_parser().parse_args(argv)
    import main
    spec, _ = main._configure_adaptive_scale(
        args.scale_config, args.cse_dir, rules_config=args.rules_config,
        style_config=args.style_config, cse_workers=args.cse_workers)
    data = generate_frontend_ir(
        args.reference_mscx, seed=args.seed, bpm=args.bpm, spec=spec,
        staff_id=args.staff_id, beam_width=args.beam_width)
    save_ir(args.output, data, spec=spec)
    print("Generated imitation front-end IR", args.output)
    print("Reference staff", data["generator"]["reference_staff_id"],
          "Lead notes", len(data["lead"]), "Measures", data["bars"])
    return data


if __name__ == "__main__":
    cli()
