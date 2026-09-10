#!/usr/bin/env python3
"""Generate MelodyPlan/2 and jointly realize Lead pitch, rhythm and harmony."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random

import harmony_rhythm as hr
import score_builder as sb
from frontend_ir import build_ir, save_ir
from melody_plan import MODE_LAYERS, generate as generate_plan, resolve_melody_plan


EPS = 1e-7


def _clip_unit(row, half_index, bpb):
    half = float(bpb) / 2.0
    lo, hi = half_index * half, (half_index + 1) * half
    out = []
    for segment in row["chord_segments"]:
        a = max(lo, float(segment["offset"]))
        b = min(hi, float(segment["offset"]) + float(segment["duration"]))
        if b > a + EPS:
            item = copy.deepcopy(segment)
            item["offset"] = round(a - lo, 6)
            item["duration"] = round(b - a, 6)
            out.append(item)
    if not out:
        raise RuntimeError("harmony candidate does not fill a MelodyPlan unit")
    return out


def _unit_chord(segments, offset):
    for segment in reversed(segments):
        if float(offset) + EPS >= float(segment["offset"]):
            return segment
    return segments[0]


def _half_rhythm_cells(palette, bpb, allow_sixteenth):
    half = float(bpb) / 2.0
    cells = {}
    groups = ("basic", "fancy", "cadence", "sixteenth")
    for group in groups:
        if group == "sixteenth" and not allow_sixteenth:
            continue
        for _, durations in palette.get(group, ()):
            start = 0
            total = 0.0
            for index, duration in enumerate(durations):
                total += float(duration)
                if math.isclose(total, half, abs_tol=EPS):
                    cell = tuple(round(float(x), 6) for x in durations[start:index + 1])
                    cells.setdefault(cell, set()).add(group)
                    start = index + 1
                    total = 0.0
                elif total > half + EPS:
                    break
    cells.setdefault((round(half, 6),), set()).add("basic")
    return {row: frozenset(cells[row])
            for row in sorted(cells, key=lambda item: (len(item), item))}


def _onsets(durations):
    out, cursor = [], 0.0
    for duration in durations:
        out.append(round(cursor, 6))
        cursor += float(duration)
    return tuple(out)


def _rhythm_distance(a, b):
    aa, bb = _onsets(a), _onsets(b)
    overlap = min(len(aa), len(bb))
    displacement = sum(abs(aa[i] - bb[i]) for i in range(overlap))
    return abs(len(aa) - len(bb)) * 1.5 + displacement


def _fresh_rhythms(cells, target, limit, rng, dense_min_notes=5,
                   dense_candidate_probability=0.0,
                   fancy_frequency_multiplier=0.5,
                   syncopated_frequency_multiplier=0.5,
                   straight_flow_frequency_multiplier=1.0):
    """Weighted candidates near the target density, without forced replacement."""
    count = max(1, int(limit))
    available = list(cells)
    result = []
    while available and len(result) < count:
        weights = []
        for row in available:
            weight = math.exp(-1.35 * abs(len(row) - float(target)))
            weights.append(max(weight, 1e-12))
        fancy_flags = ["fancy" in cells[row] and "basic" not in cells[row]
                       for row in available]
        weights = hr.apply_category_frequency_multiplier(
            weights, fancy_flags, fancy_frequency_multiplier)
        weights = hr.apply_category_frequency_multiplier(
            weights, [hr.rhythm_is_syncopated(row) for row in available],
            syncopated_frequency_multiplier)
        straight_flow = [
            len(row) >= 3 and max(row) <= 1.0 + EPS
            and not hr.rhythm_is_syncopated(row)
            for row in available
        ]
        weights = hr.apply_category_frequency_multiplier(
            weights, straight_flow, straight_flow_frequency_multiplier)
        chosen = rng.choices(available, weights=weights, k=1)[0]
        result.append(chosen)
        available.remove(chosen)
    # Density is a candidate-bank property: it only ensures that the beam gets
    # to compare a dense option; it never forces the final result to be dense.
    dense = [row for row in cells if len(row) >= int(dense_min_notes)]
    dense_probability = max(0.0, min(1.0, float(dense_candidate_probability)))
    if dense and dense_probability > 0.0 and rng.random() < dense_probability:
        missing = [row for row in dense if row not in result]
        if missing:
            weights = [math.exp(-1.35 * abs(len(row) - float(target))) for row in missing]
            chosen = rng.choices(missing, weights=weights, k=1)[0]
            if len(result) < count:
                result.append(chosen)
            elif not any(len(row) >= int(dense_min_notes) for row in result):
                replace = min(range(len(result)), key=lambda i: (len(result[i]), i))
                result[replace] = chosen
    return result


def _copied_rhythms(source, cells, fidelity, limit):
    if fidelity == "exact":
        return [tuple(source)]
    ranked = sorted(cells, key=lambda row: (_rhythm_distance(source, row), len(row)))
    result = [tuple(source)]
    result.extend(row for row in ranked if tuple(row) != tuple(source))
    return result[:max(1, int(limit))]


def _melody_reprise_segments(actions):
    """Return maximal effective Melody-copy runs on the atomic timeline.

    MelodyPlan relations may cross any number of 0.5/1/2-bar material leaves.
    Leaf boundaries therefore must not participate in octave-shift fallback.
    A run ends only when the effective relation, requested octave, or advancing
    source timeline changes.
    """
    segments = {}
    segment_for_unit = {}
    cursor = 0
    while cursor < len(actions):
        first = actions[cursor]
        if "M" not in MODE_LAYERS[first["mode"]]:
            cursor += 1
            continue
        relation_id = first.get("relation_id")
        source_start = first.get("source_unit")
        octave_shift = int(first.get("octave_shift", 0))
        end = cursor + 1
        while end < len(actions):
            row = actions[end]
            if "M" not in MODE_LAYERS[row["mode"]]:
                break
            if row.get("relation_id") != relation_id:
                break
            if int(row.get("octave_shift", 0)) != octave_shift:
                break
            if row.get("source_unit") != source_start + (end - cursor):
                break
            end += 1
        segment_id = f"{relation_id or 'melody_copy'}@{cursor}:{end}"
        segment = {
            "id": segment_id,
            "relation_id": relation_id,
            "target_units": tuple(range(cursor, end)),
            "source_units": tuple(range(source_start, source_start + end - cursor)),
            "requested_octave_shift": octave_shift,
        }
        segments[segment_id] = segment
        for unit in range(cursor, end):
            segment_for_unit[unit] = segment_id
        cursor = end
    return segments, segment_for_unit


def _resolve_reprise_octave_shift(state, segment):
    """Resolve one legal register for an entire continuous reprise segment.

    Returning ``None`` rejects the current beam state.  This is important at
    reprise entrances: falling back from a requested octave shift to the
    original register is only safe when the original register also satisfies
    the hard Lead-jump rule.
    """
    requested = int(segment["requested_octave_shift"])
    source_pitches = []
    for source_unit in segment["source_units"]:
        source_pitches.extend(int(p) for p in state["materials"][source_unit]["pitches"])
    lead_pool = set(sb.POOLS["lead"])

    def legal(shift):
        shifted = tuple(p + int(shift) * int(hr.OCT) for p in source_pitches)
        if not shifted or not all(pitch in lead_pool for pitch in shifted):
            return False
        previous = state["prev"]
        for pitch in shifted:
            if not sb.lead_jump_ok(previous, pitch):
                return False
            previous = pitch
        return True

    if legal(requested):
        return requested
    if requested != 0 and legal(0):
        return 0
    return None


def _copy_pitches(source_steps, resolved_octave_shift):
    """Copy exactly in the segment-wide resolved register; never fold a leaf."""
    shift = int(resolved_octave_shift) * int(hr.OCT)
    return [int(raw) + shift for raw in source_steps]


def _next_reprise_entrance_is_reachable(state, unit, actions, segments,
                                         segment_for_unit):
    """Look one atomic unit ahead and protect a fixed Melody-copy entrance.

    The source material for a backward-pointing relation is complete by the
    time its target begins.  Checking after the preceding unit has been added
    lets the beam retain endings that can enter either the requested whole-
    segment octave register or its all-original-register fallback.
    """
    next_unit = int(unit) + 1
    if next_unit >= len(actions):
        return True
    action = actions[next_unit]
    if "M" not in MODE_LAYERS[action["mode"]]:
        return True
    segment_id = segment_for_unit.get(next_unit)
    if segment_id is None:
        return True
    segment = segments[segment_id]
    if next_unit != segment["target_units"][0]:
        return True
    return _resolve_reprise_octave_shift(state, segment) is not None


def _bar_role(section, unit):
    local_bar = max(0, int(unit // 2) - int(section["start_bar"]))
    return section["bar_roles"][min(local_bar, len(section["bar_roles"]) - 1)]


def _role_profile(cfg, role):
    return cfg["bar_role_profiles"][role]


def _role_direction(profile, unit):
    direction = int(profile["direction"])
    if profile["contour"] == "wave" and unit % 2:
        direction = -direction
    return direction


def _guide_degrees(section, unit, offsets, profile, bpb):
    """Continuous section-level register target at each actual onset."""
    degrees = [hr.degree(p) for p in sb.POOLS["lead"]]
    low, high = min(degrees), max(degrees)
    octave = max(1, len(hr.PCS))
    curve = section["register_curve"]
    section_start = float(section["start_bar"])

    def curve_at(bar_time):
        x = max(0.0, min(float(len(curve) - 1), bar_time - section_start))
        lo = int(math.floor(x))
        hi = min(len(curve) - 1, lo + 1)
        return float(curve[lo]) * (1.0 - (x - lo)) + float(curve[hi]) * (x - lo)

    guides = []
    for offset in offsets:
        bar_time = float(unit) * 0.5 + float(offset) / float(bpb)
        centre = (low + curve_at(bar_time) * (high - low) +
                  float(profile["register_offset_octaves"]) * octave)
        guides.append(max(low, min(high, int(round(centre)))))
    return guides


def _fresh_pitches(state, harmony, durations, section, unit, role_profile, rng,
                   motif_queue, allow_ties=False, tie_probability=0.0):
    prev, prev2 = state["prev"], state["prev2"]
    recent = list(state["recent"])
    offsets = _onsets(durations)
    guides = _guide_degrees(section, unit, offsets, role_profile, state["bpb"])
    direction = _role_direction(role_profile, unit)
    pitches, attacks = [], []
    offset = 0.0
    for index, duration in enumerate(durations):
        segment = _unit_chord(harmony, offset)
        chord = hr.CHORDS[segment["chord_id"]]
        strong = hr.is_strong_beat((unit % 2) * (state["bpb"] / 2.0) + offset,
                                   state["bpb"])
        continuation = (bool(allow_ties) and index == 0 and prev is not None and
                        rng.random() < float(tie_probability))
        motif_index = int(state.get("motif_cursor", 0)) + sum(attacks)
        if continuation:
            candidate = prev
        elif motif_queue and motif_index < len(motif_queue):
            candidate = sb._fit_motif_pitch(motif_queue[motif_index], prev, guides[index])
        else:
            candidate = sb.choose_lead_pitch(
                prev, prev2, chord, strong, recent, rng, guides[index],
                direction, unit % 2 == 1 and float(role_profile["cadence_root_weight"]) > 0,
                (unit % 2) * (state["bpb"] / 2.0) + offset,
                duration, state["bpb"])
        if candidate is None:
            candidate = sb.nearest_pc("lead", chord.foot % hr.OCT,
                                      hr.degree_pitch(guides[index]), prev)
        candidate = sb.fit_lead_pitch_to_jump(candidate, prev,
                                              hr.degree_pitch(guides[index]), True)
        pitches.append(int(candidate))
        attacks.append(not continuation)
        if not continuation:
            prev2, prev = prev, candidate
            recent.append(candidate)
        offset += float(duration)
    return (pitches, attacks,
            min(len(motif_queue), int(state.get("motif_cursor", 0)) + sum(attacks)))


def _unit_cost(state, harmony, durations, pitches, attacks, section, unit, role_profile, weights):
    bpb = state["bpb"]
    half = bpb / 2.0
    recent = list(state["recent"])
    prev = state["prev"]
    offsets = _onsets(durations)
    guide = _guide_degrees(section, unit, offsets, role_profile, bpb)
    target = (float(section["density_curve"][min(int((unit / 2) % 8),
                                                    len(section["density_curve"]) - 1)]) *
              float(role_profile["density_multiplier"]) *
              float(state["base_events_per_bar"]) / 2.0)
    density = abs(sum(bool(x) for x in attacks) - target) / max(1.0, target)
    register = background = strong_cost = 0.0
    voice = 0.0
    register_duration = background_duration = 0.0
    strong_count = voice_count = 0
    old = prev
    old2 = state["prev2"]
    for offset, duration, pitch, attack, guide_degree in zip(offsets, durations, pitches, attacks, guide):
        segment = _unit_chord(harmony, offset)
        chord = hr.CHORDS[segment["chord_id"]]
        beat = (unit % 2) * half + offset
        strong = hr.is_strong_beat(beat, bpb)
        components = sb.lead_candidate_cost_components(
            old, old2, chord, strong, recent, pitch, guide_degree,
            _role_direction(role_profile, unit),
            unit % 2 == 1 and float(role_profile["cadence_root_weight"]) > 0,
            beat, duration, bpb)
        if components is None:
            return math.inf, {key: math.inf for key in weights}
        # The pitch-selection and beam layers now share the same deterministic
        # primitives.  This layer only changes their temporal aggregation.
        register += float(duration) * (components["guide"] + components["register"])
        register_duration += float(duration)
        if attack:
            voice += sum(components[key] for key in (
                "interval", "leap_recovery", "landing", "sixteenth_flow", "direction",
                "ngram", "pc_frequency", "motion_diversity", "boundary"))
            voice_count += 1
        # A sustained note can cross a harmony segment.  Integrate the field
        # over the actual overlap instead of charging one value per onset.
        note_end = offset + float(duration)
        for background_segment in harmony:
            start = max(offset, float(background_segment["offset"]))
            end = min(note_end, float(background_segment["offset"]) +
                      float(background_segment["duration"]))
            if end > start + EPS:
                bg_chord = hr.CHORDS[background_segment["chord_id"]]
                overlap = end - start
                background += overlap * sb.cse_rt.lead_chord_field_cost(bg_chord, pitch, strong)
                background_duration += overlap
        if attack and strong and pitch % hr.OCT not in chord.pcs:
            strong_cost += 1.0
            strong_count += 1
        if attack:
            recent.append(pitch)
            old2, old = old, pitch
    register /= max(EPS, register_duration)
    voice /= max(1, voice_count)
    background /= max(EPS, background_duration)
    strong_cost /= max(1, strong_count)
    transition = 0.0
    last = hr.CHORDS.get(state["last_chord_id"])
    for segment in harmony:
        chord = hr.CHORDS[segment["chord_id"]]
        transition += -math.log(max(1e-15, hr._chord_transition_weight(last, chord)))
        last = chord
    transition /= max(1, len(harmony))
    cadence = 0.0
    role_cadence_weight = float(role_profile["cadence_root_weight"])
    if unit % 2 == 1 and role_cadence_weight > 0:
        last_chord = hr.CHORDS[harmony[-1]["chord_id"]]
        if not pitches or pitches[-1] % hr.OCT != last_chord.foot % hr.OCT:
            cadence += role_cadence_weight
    if unit % 16 == 15:
        cadence += max(0, len(pitches) - 2) * .35
    components = {
        "register": register, "density": density, "voice_leading": voice,
        "background": background, "strong_beat_chord_tone": strong_cost,
        "harmony_transition": transition, "cadence": cadence,
    }
    return sum(float(weights[key]) * value for key, value in components.items()), components


def _material_events(unit, material, rng, bpb, action, role, leaf):
    half = bpb / 2.0
    start = unit * half
    offset = 0.0
    output = []
    for duration, pitch, attack in zip(material["rhythm"], material["pitches"], material["attacks"]):
        segment = _unit_chord(material["harmony"], offset)
        beat_in_bar = (unit % 2) * half + offset
        output.append(sb.event(
            start + offset, duration, pitch, "lead", rng, segment,
            structural=hr.is_strong_beat(beat_in_bar, bpb),
            motif=f"melodyplan_{action['mode'].lower()}",
            section_repeat=action["mode"] != "FRESH",
            section_variation=action["mode"],
            ornamental_repeat=False,
            melodyplan_unit=unit,
            melodyplan_relation=action.get("relation_id"),
            melodyplan_source_unit=action.get("source_unit"),
            melodyplan_leaf=leaf["id"],
            bar_role=role,
            lead_continuation=not bool(attack),
        ))
        offset += float(duration)
    return output


def _merge_lead_continuations(events):
    """Collapse internal continue cells into one cross-unit Lead event."""
    merged = []
    for event in sorted(events, key=lambda row: float(row["start_beat"])):
        continuation = bool(event.pop("lead_continuation", False))
        if continuation:
            if not merged:
                raise sb.GenerationRejected("Lead continuation has no preceding attack")
            previous = merged[-1]
            adjacent = math.isclose(float(previous["start_beat"]) + float(previous["duration_beats"]),
                                    float(event["start_beat"]), abs_tol=EPS)
            if not adjacent or int(previous["step"]) != int(event["step"]):
                raise sb.GenerationRejected("Lead continuation does not match its preceding attack")
            previous["duration_beats"] = round(
                float(previous["duration_beats"]) + float(event["duration_beats"]), 6)
            previous["tie_continuations"] = int(previous.get("tie_continuations", 0)) + 1
            continue
        merged.append(event)
    return merged


def _apply_phrase_rests(events, plan, bpb, joint, rng):
    """Create optional breath gaps by shortening a phrase-final held note.

    Rests are deliberately restricted to answer/liquidation boundaries.  They
    are represented by event gaps in the IR, never by a fake pitch event.
    """
    if not bool(joint["allow_phrase_rests"]):
        return events
    boundaries = []
    for section in plan["sections"]:
        for local_bar, role in enumerate(section["bar_roles"]):
            if role in {"theme_answer", "liquidation"}:
                boundaries.append((float(section["start_bar"]) + local_bar + 1.0) * float(bpb))
    output = copy.deepcopy(events)
    allowed = tuple(sorted(float(x) for x in joint["phrase_rest_durations"] if float(x) > 0.0))
    for boundary in boundaries:
        if rng.random() >= float(joint["phrase_rest_probability"]):
            continue
        prior = next((event for event in reversed(output)
                      if math.isclose(float(event["start_beat"]) + float(event["duration_beats"]),
                                        boundary, abs_tol=EPS)), None)
        if prior is None:
            continue
        candidates = [duration for duration in allowed
                      if float(prior["duration_beats"]) - duration >= 0.125 - EPS]
        if not candidates:
            continue
        duration = rng.choice(candidates)
        prior["duration_beats"] = round(float(prior["duration_beats"]) - duration, 6)
        prior["phrase_rest_after"] = round(duration, 6)
    return output


def _merge_segments(segments):
    merged = []
    for segment in sorted(segments, key=lambda row: float(row["offset"])):
        item = copy.deepcopy(segment)
        if (merged and merged[-1]["chord_id"] == item["chord_id"] and
                math.isclose(float(merged[-1]["offset"]) + float(merged[-1]["duration"]),
                             float(item["offset"]), abs_tol=EPS)):
            merged[-1]["duration"] = round(float(merged[-1]["duration"]) +
                                                   float(item["duration"]), 6)
        else:
            merged.append(item)
    return merged


def _assemble_harmony(materials, banks, plan, bars, bpb, rng):
    rows = []
    half = bpb / 2.0
    for bar in range(bars):
        base = copy.deepcopy(banks[0][bar])
        segments = []
        for side in (0, 1):
            for segment in materials[bar * 2 + side]["harmony"]:
                item = copy.deepcopy(segment)
                item["offset"] = round(float(item["offset"]) + side * half, 6)
                segments.append(item)
        segments = _merge_segments(segments)
        # Every eight-bar phrase retains the original harmony generator's
        # mandatory closing anchor even after mixed-layer inheritance.
        if bar % 8 == 7 and segments[-1]["chord_id"] not in hr.ANCHOR_CHORD_IDS:
            segments[-1] = hr._force_anchor_segment(segments[-1], rng)
        base["bar"] = bar
        base["chord_segments"] = segments
        base["chords_in_bar"] = len(segments)
        base["melodyplan_section"] = plan["sections"][bar // 8]["id"]
        hr._refresh_primary_from_first_segment(base)
        rows.append(base)
    return rows


def generate_frontend_ir(seed=20260811, bars=48, bpm=96.0, motif_degrees=None,
                         time_signature="4/4", allow_sixteenth=True, spec=None):
    if spec is None:
        spec = hr.SCALE
    if type(bars) is not int or bars <= 0:
        raise ValueError("bars must be a positive integer")
    if not math.isfinite(float(bpm)) or float(bpm) <= 0:
        raise ValueError("bpm must be positive")
    ts, bpb = hr.normalize_time_signature(time_signature)
    cfg = resolve_melody_plan(spec.style.get("melody_plan"))
    if not cfg["enabled"]:
        from frontend_generate import generate_frontend_ir as generate_legacy_frontend
        return generate_legacy_frontend(
            seed, bars, bpm, motif_degrees, ts, allow_sixteenth, spec)
    plan = generate_plan(seed, bars, bpb, cfg)

    palette_rng = random.Random(int(seed) ^ 0x52485954484D)
    palette = hr.make_rhythm_palette(palette_rng, bpb)
    cells = _half_rhythm_cells(palette, bpb, allow_sixteenth)
    harmony_count = cfg["joint_generation"]["harmony_candidates"]
    banks = [hr.harmony_plan(bars, random.Random(int(seed) + 1000003 * i), bpb)
             for i in range(harmony_count)]
    if spec.resolved_chord_progression() is not None:
        banks = [banks[0]]

    initial = {
        "cost": 0.0, "materials": [], "lead": [], "recent": [],
        "prev": None, "prev2": None, "last_chord_id": None, "bpb": bpb,
        "base_events_per_bar": cfg["joint_generation"]["base_events_per_bar"],
        "octave_reprise_decisions": {}, "motif_cursor": 0,
        "components": {key: 0.0 for key in cfg["joint_generation"]["weights"]},
    }
    beam = [initial]
    motif_queue = sb.user_motif_steps(motif_degrees)
    section_by_unit = [plan["sections"][min(len(plan["sections"]) - 1, u // 16)]
                       for u in range(bars * 2)]
    joint = cfg["joint_generation"]
    ordered_units = [
        (leaf, unit, plan["unit_actions"][unit])
        for leaf in plan["leaf_actions"]
        for unit in leaf["unit_indices"]
    ]
    reprise_segments, reprise_segment_for_unit = _melody_reprise_segments(
        plan["unit_actions"])
    for leaf, unit, action in ordered_units:
        section = section_by_unit[unit]
        role = _bar_role(section, unit)
        role_profile = _role_profile(cfg, role)
        layers = MODE_LAYERS[action["mode"]]
        next_beam = []
        for state_index, state in enumerate(beam):
            source = (state["materials"][action["source_unit"]]
                      if action["source_unit"] is not None else None)
            if "H" in layers:
                harmony_choices = [copy.deepcopy(source["harmony"])]
            else:
                harmony_choices = [_clip_unit(bank[unit // 2], unit % 2, bpb)
                                   for bank in banks]
            density = section["density_curve"][min(7, (unit // 2) % 8)]
            target = (float(joint["base_events_per_bar"]) * float(density) *
                      float(role_profile["density_multiplier"]) / 2.0)
            if "R" in layers:
                rhythm_choices = _copied_rhythms(
                    source["rhythm"], cells, action["rhythm_fidelity"],
                    joint["rhythm_candidates"])
            else:
                rrng = random.Random(int(seed) ^ (unit * 1009 + state_index * 9176))
                rhythm_choices = _fresh_rhythms(cells, target,
                                                joint["rhythm_candidates"], rrng,
                                                joint["dense_half_min_notes"],
                                                joint["dense_half_candidate_probability"],
                                                joint["fancy_frequency_multiplier"],
                                                joint["syncopated_frequency_multiplier"],
                                                joint["straight_flow_frequency_multiplier"])
            for hi, harmony in enumerate(harmony_choices):
                for ri, rhythm in enumerate(rhythm_choices):
                    if "M" in layers:
                        segment_id = reprise_segment_for_unit[unit]
                        octave_decisions = state["octave_reprise_decisions"]
                        if segment_id in octave_decisions:
                            resolved_octave_shift = octave_decisions[segment_id]
                        else:
                            resolved_octave_shift = _resolve_reprise_octave_shift(
                                state, reprise_segments[segment_id])
                            if resolved_octave_shift is None:
                                continue
                            octave_decisions = dict(octave_decisions)
                            octave_decisions[segment_id] = resolved_octave_shift
                        copied = _copy_pitches(source["pitches"], resolved_octave_shift)
                        copied_attacks = list(source.get("attacks", (True,) * len(copied)))
                        # A copied continuation can only remain a tie when it
                        # still joins the preceding realised pitch/register.
                        if copied_attacks and not copied_attacks[0] and (
                                state["prev"] is None or copied[0] != state["prev"]):
                            copied_attacks[0] = True
                        pitch_choices = [(copied, copied_attacks, state["motif_cursor"])]
                    else:
                        octave_decisions = state["octave_reprise_decisions"]
                        pitch_choices = []
                        for mi in range(joint["melody_candidates"]):
                            crng = random.Random(int(seed) ^ 0x4A4F494E54 ^
                                                (unit * 1000003 + state_index * 65537 +
                                                 hi * 257 + ri * 31 + mi))
                            pitch_choices.append(_fresh_pitches(
                                state, harmony, rhythm, section, unit, role_profile,
                                crng, motif_queue,
                                bool(joint["allow_cross_unit_ties"]),
                                float(joint["cross_unit_tie_probability"])))
                    for mi, (pitches, attacks, motif_cursor) in enumerate(pitch_choices):
                        if len(pitches) != len(rhythm):
                            continue
                        crng = random.Random(int(seed) ^ (unit * 15485863 +
                                                         state_index * 8191 + hi * 509 +
                                                         ri * 67 + mi))
                        increment, components = _unit_cost(
                            state, harmony, rhythm, pitches, attacks, section, unit, role_profile,
                            joint["weights"])
                        noise = crng.uniform(0.0, float(joint["selection_noise"]))
                        material = {"harmony": copy.deepcopy(harmony),
                                    "rhythm": tuple(rhythm), "pitches": tuple(pitches),
                                    "attacks": tuple(bool(x) for x in attacks)}
                        actual_pitches = [pitch for pitch, attack in zip(pitches, attacks) if attack]
                        candidate = {
                            "cost": state["cost"] + increment + noise,
                            "materials": state["materials"] + [material],
                            "lead": state["lead"] + _material_events(
                                unit, material, crng, bpb, action, role, leaf),
                            "recent": (state["recent"] + actual_pitches)[-256:],
                            "prev": actual_pitches[-1] if actual_pitches else state["prev"],
                            "prev2": (actual_pitches[-2] if len(actual_pitches) > 1 else
                                      state["prev"] if actual_pitches else state["prev2"]),
                            "last_chord_id": harmony[-1]["chord_id"],
                            "bpb": bpb,
                            "base_events_per_bar": state["base_events_per_bar"],
                            "octave_reprise_decisions": octave_decisions,
                            "motif_cursor": motif_cursor,
                            "components": {key: state["components"][key] + components[key]
                                           for key in components},
                        }
                        if not _next_reprise_entrance_is_reachable(
                                candidate, unit, plan["unit_actions"],
                                reprise_segments, reprise_segment_for_unit):
                            continue
                        next_beam.append(candidate)
        if not next_beam:
            raise sb.GenerationRejected(
                f"MelodyPlan joint realization failed at leaf {leaf['id']} unit {unit}")
        next_beam.sort(key=lambda row: row["cost"])
        beam = next_beam[:joint["beam_width"]]

    best = beam[0]
    assembly_rng = random.Random(int(seed) ^ 0x415353454D424C59)
    harmony = _assemble_harmony(best["materials"], banks, plan, bars, bpb, assembly_rng)
    best_lead = _merge_lead_continuations(best["lead"])
    best_lead = _apply_phrase_rests(
        best_lead, plan, bpb, joint,
        random.Random(int(seed) ^ 0x425245415448))
    jumps = sb.lead_jump_errors(best_lead)
    if jumps:
        raise sb.GenerationRejected(f"MelodyPlan Lead exceeds jump limit: {jumps[:2]}")
    metadata = {
        "joint_objective": {
            "total": best["cost"], "components": best["components"],
            "weighted_components": {
                key: float(joint["weights"][key]) * best["components"][key]
                for key in best["components"]
            },
            "beam_width": joint["beam_width"],
            "harmony_candidate_banks": len(banks),
            "octave_reprise_decisions": best["octave_reprise_decisions"],
        }
    }
    return build_ir(
        seed=seed, bpm=bpm, time_signature=ts, beats_per_bar=bpb, bars=bars,
        spec=spec, harmony_plan=harmony, lead=best_lead,
        allow_sixteenth=allow_sixteenth,
        lead_palette=hr.rhythm_palette_metadata(palette),
        motif_degrees=motif_degrees, generator="melodyplan_joint",
        melody_plan=plan, generator_metadata=metadata,
    )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", dest="scale_config", required=True)
    parser.add_argument("--rules", dest="rules_config")
    parser.add_argument("--style", dest="style_config")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bars", type=int, default=48)
    parser.add_argument("--bpm", type=float, default=96.0)
    parser.add_argument("--time-signature", default="4/4")
    parser.add_argument("--chord-progression")
    parser.add_argument("--cse-dir")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--output", "-o", default="frontend_ir.json")
    parser.add_argument("--plan-output")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-sixteenth", dest="allow_sixteenth", action="store_true")
    group.add_argument("--no-sixteenth", dest="allow_sixteenth", action="store_false")
    parser.set_defaults(allow_sixteenth=True)
    return parser


def cli(argv=None):
    args = _build_parser().parse_args(argv)
    import main
    spec, _ = main._configure_adaptive_scale(
        args.scale_config, args.cse_dir, rules_config=args.rules_config,
        style_config=args.style_config, cse_workers=args.cse_workers,
        chord_progression=args.chord_progression)
    data = generate_frontend_ir(
        args.seed, args.bars, args.bpm, time_signature=args.time_signature,
        allow_sixteenth=args.allow_sixteenth, spec=spec)
    save_ir(args.output, data, spec=spec)
    if args.plan_output:
        with open(args.plan_output, "w", encoding="utf-8") as handle:
            json.dump(data["melody_plan"], handle, ensure_ascii=False, indent=2)
    print("Generated MelodyPlan front-end IR", args.output)
    return data


if __name__ == "__main__":
    cli()
