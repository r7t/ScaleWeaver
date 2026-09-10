#!/usr/bin/env python3
"""Read a ScaleWeaver front-end IR and generate/anneal the three lower voices."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random

import harmony_rhythm as hr
import score_builder as sb
from annealing_config import resolve_annealing
from frontend_ir import content_sha256, load_ir, validate_ir


FOUR_PART_VOICES = ("bass", "inner", "counter", "lead")
FIVE_PART_VOICES = ("bass", "inner", "inner2", "counter", "lead")


def _rescue_time_points(lead, plan, bars, bpb):
    """Smallest slices on which Lead and background harmony are constant."""
    points = {0.0, float(bars) * float(bpb)}
    for event in lead:
        points.add(float(event["start_beat"]))
        points.add(float(event["start_beat"]) + float(event["duration_beats"]))
    for bar, row in enumerate(plan[:bars]):
        base = bar * float(bpb)
        points.add(base)
        points.add(base + float(bpb))
        for segment in row["chord_segments"]:
            points.add(base + float(segment["offset"]))
            points.add(base + float(segment["offset"]) + float(segment["duration"]))
    return sorted(round(x, 6) for x in points)


def _active_step(events, start, end):
    mid = (float(start) + float(end)) * .5
    for event in events:
        if (float(event["start_beat"]) <= mid + 1e-9 and
                float(event["start_beat"]) + float(event["duration_beats"]) > mid - 1e-9):
            return int(event["step"])
    return None


def _temporary_event(start, duration, pitch):
    return {"start_beat": float(start), "duration_beats": float(duration),
            "step": int(pitch)}


def _rescue_voice_cost(voice, pitch, previous, recent, chord, metric, config):
    weights = config["objective_weights"]
    vl = config["voice_leading"]
    guide = sb.COUNTERPOINT_ROLE_CENTRE_DEGREE[voice]
    value = (weights["background"] * config["background_voice_weights"][voice]
             * metric.field(chord.id, int(pitch)))
    value += weights["voice_leading"] * (
        annealing_motion_cost(voice, previous, int(pitch), recent, config)
        + vl["register_weights"][voice] * abs(hr.degree(int(pitch)) - guide))
    value -= metric.pitch_class_reward(voice, int(pitch))
    return value


def annealing_motion_cost(voice, previous, current, recent, config):
    c = config["voice_leading"]
    cost = 0.0
    if previous is not None:
        jump = abs(hr.degree(int(current)) - hr.degree(int(previous)))
        cost += c["motion_weight"] * sb._lead_interval_cost(jump)
        cost += c["large_leap_cost"] * max(0, jump - sb.LEAD_PREFERRED_MAX_LEAP)
        if int(current) == int(previous):
            cost += c["repeat_cost"]
    cost += c["ngram_weights"][voice] * sb._lead_ngram_prior_cost(recent, current)
    if voice == "bass":
        cost += c["bass_ngram_weight"] * sb._bass_ngram_prior_cost(recent, current)
    return cost


def _merge_rescue_line(events):
    merged = []
    for event in events:
        if (merged and int(merged[-1]["step"]) == int(event["step"])
                and merged[-1].get("harmony") == event.get("harmony")
                and math.isclose(float(merged[-1]["start_beat"])
                                 + float(merged[-1]["duration_beats"]),
                                 float(event["start_beat"]), abs_tol=2e-6)):
            merged[-1]["duration_beats"] = round(
                float(merged[-1]["duration_beats"]) + float(event["duration_beats"]), 6)
        else:
            merged.append(event)
    return merged


def _best_rescue_sonority(pools, lead_existing, lead_step, lead_attacks,
                          start, duration, chord, previous, recent,
                          metric, config, rng):
    best = None
    checked = 0
    for counter in pools["counter"]:
        if math.isinf(sb._counterpoint_vertical_cost(
                "counter", counter, lead_existing, start, duration)):
            continue
        counter_event = _temporary_event(start, duration, counter)
        upper = {**lead_existing, "counter": [counter_event]}
        counter_cost = _rescue_voice_cost(
            "counter", counter, previous["counter"], recent["counter"],
            chord, metric, config)
        for inner in pools["inner"]:
            if math.isinf(sb._counterpoint_vertical_cost(
                    "inner", inner, upper, start, duration)):
                continue
            inner_event = _temporary_event(start, duration, inner)
            upper_inner = {**upper, "inner": [inner_event]}
            partial = counter_cost + _rescue_voice_cost(
                "inner", inner, previous["inner"], recent["inner"],
                chord, metric, config)
            for bass in pools["bass"]:
                checked += 1
                if math.isinf(sb._counterpoint_vertical_cost(
                        "bass", bass, upper_inner, start, duration)):
                    continue
                pitches = (bass, inner, counter)
                sounding = pitches + ((lead_step,) if lead_step is not None else ())
                value = partial + _rescue_voice_cost(
                    "bass", bass, previous["bass"], recent["bass"],
                    chord, metric, config)
                value += config["objective_weights"]["sounding"] * metric.sounding_cost(sounding)
                attacks = list(pitches)
                if lead_attacks:
                    attacks.append(lead_step)
                value += config["objective_weights"]["attack"] * metric.attack_cost(tuple(attacks))
                value += rng.uniform(0.0, 1e-7)
                if best is None or value < best[0]:
                    best = (value, bass, inner, counter)
    return best, checked


def _nominal_range_preflight(lead):
    """Detect fixed Lead pitches for which normal four-part spacing is impossible."""
    pools = {}
    for voice in ("counter", "inner", "bass"):
        lo, hi = sb.COUNTERPOINT_EFFECTIVE_RANGES[voice]
        pools[voice] = [int(p) for p in hr.POOLS[voice] if lo <= int(p) <= hi]

    def compatible(a, b):
        return a != b and not sb.hard_wolf(a, b) and not sb.step_second(a, b)

    for lead_step in sorted({int(event["step"]) for event in lead}):
        found = False
        for counter in pools["counter"]:
            if not counter < lead_step or not compatible(counter, lead_step):
                continue
            for inner in pools["inner"]:
                if not inner < counter or not all(
                        compatible(inner, p) for p in (counter, lead_step)):
                    continue
                if any(bass < inner and all(
                        compatible(bass, p) for p in (inner, counter, lead_step))
                       for bass in pools["bass"]):
                    found = True
                    break
            if found:
                break
        if not found:
            return False, lead_step
    return True, None


def _atomic_rescue_accompaniment(lead, plan, bars, bpb, seed, metric, config):
    """Guaranteed-local initializer used only after ordinary attempts fail.

    The three mutable voices are chosen jointly on intervals where Lead and
    harmony are constant.  Melodic jump limits are deliberately soft here;
    hard wolves, step-seconds and strict voice ordering remain inviolate.  If a
    fixed low Lead makes the nominal accompaniment ranges mathematically
    impossible, the rescue may extend lower voices downward by the configured
    number of octaves; the event is marked explicitly for diagnostics.
    """
    points = _rescue_time_points(lead, plan, bars, bpb)
    output = {"counter": [], "inner": [], "bass": []}
    recent = {voice: [] for voice in output}
    previous = {voice: None for voice in output}
    pools = {}
    expanded_pools = {}
    lower_octaves = config["initialization"]["atomic_rescue_lower_octaves"]
    for voice in output:
        lo, hi = sb.COUNTERPOINT_EFFECTIVE_RANGES[voice]
        pools[voice] = [int(p) for p in hr.POOLS[voice] if lo <= int(p) <= hi]
        configured_lo, configured_hi = hr.RANGES[voice]
        table_lo = min(metric.pitch_index[4])
        expanded_lo = max(table_lo, configured_lo - lower_octaves * hr.OCT)
        expanded_pools[voice] = list(hr.SCALE.make_pool(expanded_lo, configured_hi))
    rng = random.Random(int(seed) ^ 0x41544F4D4943)
    combinations_checked = 0
    expanded_range_slices = 0
    out_of_range_events = 0

    for slice_index, (start, end) in enumerate(zip(points, points[1:])):
        duration = float(end) - float(start)
        if duration <= 1e-8:
            continue
        bar = min(bars - 1, max(0, int((float(start) + 1e-8) // float(bpb))))
        offset = float(start) - bar * float(bpb)
        segment = hr.harmony_segment_at(plan[bar], offset)
        chord = hr.CHORDS[segment["chord_id"]]
        lead_step = _active_step(lead, start, end)
        lead_attacks = lead_step is not None and any(
            abs(float(e["start_beat"]) - float(start)) < 1e-7 for e in lead)
        lead_existing = {"lead": [_temporary_event(start, duration, lead_step)]} if lead_step is not None else {}
        best, checked = _best_rescue_sonority(
            pools, lead_existing, lead_step, lead_attacks, start, duration,
            chord, previous, recent, metric, config, rng)
        combinations_checked += checked
        used_expanded = False
        if best is None and lower_octaves:
            best, checked = _best_rescue_sonority(
                expanded_pools, lead_existing, lead_step, lead_attacks,
                start, duration, chord, previous, recent, metric, config, rng)
            combinations_checked += checked
            used_expanded = best is not None
            expanded_range_slices += int(used_expanded)

        if best is None:
            raise sb.GenerationRejected(
                f"atomic rescue found no hard-legal four-part sonority in slice "
                f"{slice_index} ({start:g}..{end:g})")
        _, bass, inner, counter = best
        for voice, pitch in (("bass", bass), ("inner", inner), ("counter", counter)):
            configured_lo, configured_hi = hr.RANGES[voice]
            outside = not configured_lo <= int(pitch) <= configured_hi
            out_of_range_events += int(outside)
            event = sb.event(
                start, duration, pitch, voice, rng, segment,
                surface="atomic_rescue", rhythm_pattern="adaptive_atomic",
                counterpoint=True, harmony_field_only=True,
                initialization_rescue=True,
                initialization_expanded_range=outside,
                initialization_expanded_range_slice=used_expanded)
            output[voice].append(event)
            recent[voice].append(pitch)
            previous[voice] = pitch

    for voice in output:
        output[voice] = _merge_rescue_line(output[voice])
    return output, {
        "time_slices": len(points) - 1,
        "combinations_checked": combinations_checked,
        "expanded_range_slices": expanded_range_slices,
        "out_of_range_events_before_merge": out_of_range_events,
        "events_after_merge": {voice: len(events) for voice, events in output.items()},
    }


def initialize_score_from_ir(frontend, metric, config, spec):
    """Create the greedy four-part starting point without changing the IR Lead."""
    validate_ir(frontend, spec=spec)
    seed = int(frontend["seed"])
    bars = int(frontend["bars"])
    bpm = float(frontend["tempo_bpm"])
    ts = str(frontend["time_signature"])
    bpb = float(frontend["beats_per_bar"])
    plan = copy.deepcopy(frontend["harmony_plan"])
    lead = copy.deepcopy(frontend["lead"])
    frozen_lead = copy.deepcopy(lead)
    ensemble_voices = tuple(spec.resolved_ensemble_voices())
    accompaniment_voices = tuple(v for v in ensemble_voices if v != "lead")

    lead_palette = copy.deepcopy(frontend["lead_rhythm"].get("palette"))
    palette_metadata = {
        "lead": lead_palette if isinstance(lead_palette, dict) else {
            "source": "front_end_ir",
            "bars": copy.deepcopy(frontend["lead_rhythm"]["bars"]),
        }
    }
    errors = []
    rescue_used = False
    rescue_statistics = None
    relaxed_jump_choices = 0
    expanded_range_choices = 0
    nominal_ranges_possible, blocking_lead_step = _nominal_range_preflight(lead)
    if not nominal_ranges_possible:
        errors.append(
            f"preflight: Lead step {blocking_lead_step} has no hard-legal four-part "
            "voicing inside nominal accompaniment ranges")
    try:
        import harmony_annealing as annealing
        # Expanded-range choices inside GreedyInitializer can preserve the
        # original accompaniment rhythms even when this diagnostic is false.
        ordinary_attempts = config["initialization"]["max_attempts"]
        for attempt in range(ordinary_attempts):
            accompaniment_seed = seed + attempt * config["initialization"]["seed_stride"]
            voices = {"lead": copy.deepcopy(lead)}
            attempt_palettes = dict(palette_metadata)
            greedy = annealing.GreedyInitializer(metric, config)
            sb._INITIAL_HARMONY_OBJECTIVE = greedy
            try:
                generation_order = [v for v in ("counter", "inner2", "inner", "bass")
                                    if v in accompaniment_voices]
                references = {"counter": "lead", "inner2": "counter",
                              "inner": "inner2" if "inner2" in accompaniment_voices else "counter",
                              "bass": "inner"}
                salts = {"counter": 0x2468ACE1, "inner2": 0x39D12F47,
                         "inner": 0x51A7E2C3, "bass": 0x6C39B5A7}
                for voice in generation_order:
                    salt, reference = salts[voice], references[voice]
                    rng = random.Random(accompaniment_seed ^ salt)
                    palette = hr.make_rhythm_palette(rng, bpb)
                    attempt_palettes[voice] = hr.rhythm_palette_metadata(palette)
                    voices[voice] = sb._generate_counterpoint_line(
                        voice, bars, plan, dict(voices), rng, bpb, palette, reference)
                palette_metadata = attempt_palettes
                relaxed_jump_choices = greedy.relaxed_jump_choices
                expanded_range_choices = greedy.expanded_range_choices
                break
            except sb.GenerationRejected as exc:
                errors.append(str(exc))
        else:
            tail = errors[-1] if errors else "no attempt was run"
            if not config["initialization"]["atomic_rescue"]:
                raise sb.GenerationRejected(
                    f"Cannot initialize fixed front-end IR after {len(errors)} "
                    f"accompaniment attempts: {tail}")
            accompaniment_seed = seed + ordinary_attempts * config["initialization"]["seed_stride"]
            if "inner2" in accompaniment_voices:
                raise sb.GenerationRejected(
                    "Five-part greedy initialization exhausted all attempts; "
                    "four-part atomic rescue cannot be used for a five-part layout")
            rescued, rescue_statistics = _atomic_rescue_accompaniment(
                lead, plan, bars, bpb, accompaniment_seed, metric, config)
            voices = {"lead": copy.deepcopy(lead), **rescued}
            palette_metadata = {
                **palette_metadata,
                "counter": {"source": "atomic_rescue"},
                "bass": {"source": "atomic_rescue"},
                "inner": {"source": "atomic_rescue"},
            }
            rescue_used = True
    finally:
        sb._INITIAL_HARMONY_OBJECTIVE = None

    if lead != frozen_lead or voices["lead"] != frozen_lead:
        raise AssertionError("Accompaniment initializer changed the IR Lead")
    voices = {voice: voices[voice] for voice in ensemble_voices}
    failures = sb.validate(voices)
    if any(failures):
        raise RuntimeError(f"Initializer validation failed: {failures}")

    ir_hash = content_sha256(frontend)
    return {
        "format": "ScaleWeaverScore/1",
        "seed": seed,
        "requested_seed": seed,
        "tempo_bpm": bpm,
        "time_signature": ts,
        "beats_per_bar": bpb,
        "bars": bars,
        "duration_seconds": float(frontend["duration_seconds"]),
        "configuration": hr.SCALE.to_json_dict(),
        "tuning": {
            "system": "edo",
            "edo": hr.OCT,
            "scale_id": hr.SCALE.id,
            "scale_name": hr.SCALE.name,
            "base_note": hr.SCALE.base_note,
            "base_freq_hz": hr.BASE_FREQ,
            "base_freq": hr.BASE_FREQ,
            "pcs": list(hr.PCS),
            "names": list(hr.NAMES),
            "pitch_classes": [
                {"name": name, "step": pc} for name, pc in zip(hr.NAMES, hr.PCS)
            ],
        },
        "frontend_ir": {
            "format": frontend["format"],
            "sha256": ir_hash,
            "generator": copy.deepcopy(frontend.get("generator", {})),
            "lead_notes": len(lead),
            "harmony_bars": len(plan),
            "immutable_during_accompaniment": ["lead", "harmony_plan", "lead_rhythm"],
        },
        "rhythmic_system": {
            "model": "front-end IR Lead; independently generated accompaniment rhythms; fixed during annealing",
            "allow_sixteenth": bool(frontend.get("generator", {}).get("allow_sixteenth", True)),
            "melodic_palettes": palette_metadata,
        },
        "harmonic_system": {
            "model": "front-end IR harmony plan + greedy legal accompaniment initialization + onset-group simulated annealing",
            "chords": {
                cid: {
                    "name": chord.name,
                    "pcs": list(chord.pcs),
                    "ratio": chord.ratio,
                    "function": chord.function,
                }
                for cid, chord in hr.CHORDS.items()
            },
        },
        "harmony_plan": plan,
        "voices": voices,
        "voice_layout": {
            "voice_order": list(ensemble_voices),
            "staff_order": list(reversed(ensemble_voices)),
            "three_part_counterpoint": len(ensemble_voices) == 3,
            "four_part_counterpoint": len(ensemble_voices) == 4,
            "five_part_counterpoint": len(ensemble_voices) == 5,
            "secondary_voice_mode": "counter",
            "secondary_voice_candidates": ["counter"],
            "pitch_regeneration_iterations": 0,
        },
        "initialization": {
            "mode": ("atomic_hard_legal_rescue" if rescue_used
                     else "greedy_from_frontend_ir"),
            "attempts": (ordinary_attempts
                         if rescue_used else attempt + 1),
            "accompaniment_seed": accompaniment_seed,
            "failures": errors,
            "relaxed_jump_choices": relaxed_jump_choices,
            "expanded_range_choices": expanded_range_choices,
            "atomic_rescue_used": rescue_used,
            "atomic_rescue_statistics": rescue_statistics,
            "nominal_range_preflight": {
                "possible": nominal_ranges_possible,
                "blocking_lead_step": blocking_lead_step,
            },
            "melody_seed": seed,
            "melody_generated_in_frontend": True,
            "frontend_ir_sha256": ir_hash,
        },
    }


def generate_from_ir(frontend, spec):
    """Run the unchanged accompaniment optimizer from an in-memory IR object."""
    import harmony_annealing as annealing

    validate_ir(frontend, spec=spec)
    config = resolve_annealing(spec.style.get("annealing"))
    metric = annealing.OptimizerMetric(spec, config)
    score = initialize_score_from_ir(frontend, metric, config, spec)
    score["initial_spectral_metrics"] = annealing.final_metrics(score, metric)
    score["initial_attack_limits"] = annealing.attack_diagnostics(score, metric)
    lead_hash = hashlib.sha256(
        json.dumps(score["voices"]["lead"], sort_keys=True).encode()
    ).hexdigest()
    score["annealing"] = (
        annealing.optimize(score, metric, config)
        if config["enabled"] else {"enabled": False}
    )
    if score["annealing"].get("enabled"):
        score["energy_sources"] = score["annealing"]["energy_sources"]
    score["melody_sha256"] = lead_hash

    hard, seconds, crossings = sb.validate(score["voices"])
    jumps = sb.lead_jump_errors(score["voices"])
    timing = sb.accompaniment_timing_errors(score["voices"], score["beats_per_bar"])
    if hard or seconds or crossings or jumps or timing:
        raise RuntimeError("Final generated score failed hard validation")
    score["validation"] = {
        "hard_wolves": len(hard),
        "step_seconds": len(seconds),
        "voice_crossings": len(crossings),
        "lead_jumps_over_configured_limit": len(jumps),
        "accompaniment_timing_errors": len(timing),
        "degree_cleanup_distances": 0,
        "step_cleanup_steps": 0,
        "exact_unison_overlaps": 0,
    }
    score["cleanup"] = {"enabled": False}
    score["unison_deduplication"] = {"enabled": False}
    score["diagnostics"] = {
        "lead_notes": len(score["voices"]["lead"]),
        "secondary_voice_mode": "counter",
    }
    score["final_spectral_metrics"] = annealing.final_metrics(score, metric)
    score["simultaneous_attack_cse_limits"] = annealing.attack_diagnostics(score, metric)
    return score


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="ScaleWeaverFrontEndIR/1 JSON")
    parser.add_argument("--scale", dest="scale_config", required=True)
    parser.add_argument("--rules", dest="rules_config")
    parser.add_argument("--style", dest="style_config")
    parser.add_argument("--cse-dir")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--output", "-o", default="adaptive_score.json")
    parser.add_argument("--no-statistics", dest="print_statistics",
                        action="store_false", default=True)
    parser.add_argument("--attack-analysis")
    return parser


def cli(argv=None):
    args = _build_parser().parse_args(argv)
    # main owns runtime configuration and final reporting; this module owns the
    # actual IR-to-four-part transformation.
    import main
    return main.save_score(
        filename=args.output,
        frontend_ir=args.input,
        scale_config=args.scale_config,
        rules_config=args.rules_config,
        style_config=args.style_config,
        cse_dir=args.cse_dir,
        cse_workers=args.cse_workers,
        attack_analysis_filename=args.attack_analysis,
        print_statistics=args.print_statistics,
    )


if __name__ == "__main__":
    cli()
