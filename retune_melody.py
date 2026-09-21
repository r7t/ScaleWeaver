#!/usr/bin/env python3
"""Copy an MSCX melody exactly and retune its inferred scale to a target scale."""
from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path

import harmony_rhythm as hr
import score_builder as sb
from frontend_ir import build_ir
from imitation_frontend import (_scaled_harmony_plan, _strong_attack,
                                load_reference)


PITCH_EQUIVALENCE_CENTS = 0.1


def _circular_distance(a, b):
    delta = abs(float(a) - float(b)) % 1200.0
    return min(delta, 1200.0 - delta)


def _cluster_linear(values, tolerance=PITCH_EQUIVALENCE_CENTS):
    groups = []
    for value in sorted(float(x) for x in values):
        # Complete-link clustering: every pair inside a group differs by at
        # most the stated threshold; a chain of 0.09c gaps cannot silently
        # merge endpoints that are farther than 0.1c apart.
        if groups and value - groups[-1][0] <= float(tolerance) + 1e-9:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _cluster_pitch_classes(values, tolerance=PITCH_EQUIVALENCE_CENTS):
    """Cluster modulo-octave cents, including the 0/1200 seam."""
    groups = _cluster_linear((float(x) % 1200.0 for x in values), tolerance)
    if len(groups) > 1 and (1200.0 - groups[-1][0] + groups[0][-1]
                            <= float(tolerance) + 1e-9):
        joined = groups[-1] + [x + 1200.0 for x in groups[0]]
        groups = [joined] + groups[1:-1]
    representatives = [sum(group) / len(group) % 1200.0 for group in groups]
    return tuple(sorted(representatives))


def _actual_cents(row):
    if "source_actual_cents" in row:
        return float(row["source_actual_cents"])
    if "source_actual_midi" in row:
        return 100.0 * float(row["source_actual_midi"])
    return 100.0 * float(row["source_pitch"])


def analyze_reference(reference_or_path, *, staff_id=None,
                      tolerance=PITCH_EQUIVALENCE_CENTS):
    reference = (load_reference(reference_or_path, staff_id=staff_id)
                 if not isinstance(reference_or_path, dict)
                 else reference_or_path)
    cents = [_actual_cents(row) for row in reference["highest_line"]]
    absolute_groups = _cluster_linear(cents, tolerance)
    pitch_classes = _cluster_pitch_classes(cents, tolerance)
    return {
        "reference": reference,
        "absolute_pitch_count": len(absolute_groups),
        "source_scale_note_count": len(pitch_classes),
        "source_pitch_classes_cents": pitch_classes,
        "equivalence_tolerance_cents": float(tolerance),
    }


def _rotated(values, start, period):
    values = tuple(float(x) for x in values)
    return tuple(values[(start + i) % len(values)]
                 + (period if start + i >= len(values) else 0.0)
                 for i in range(len(values)))


def best_scale_mapping(source_pitch_classes_cents, target_spec):
    """Minimum-RMS orientation-preserving injection into target pitch classes.

    A free global cent translation is removed before RMS evaluation, so the
    score measures scale-shape error rather than tonic/register placement.
    """
    source = tuple(sorted(float(x) % 1200.0
                          for x in source_pitch_classes_cents))
    target_steps = tuple(int(x) for x in target_spec.pcs)
    if len(target_steps) < len(source):
        raise ValueError(
            f"target scale has {len(target_steps)} notes per octave, fewer than "
            f"the inferred source scale's {len(source)}")
    target_cents = tuple(1200.0 * step / int(target_spec.edo)
                         for step in target_steps)
    best = None
    n = len(source)
    for source_rotation in range(n):
        source_order = _rotated(source, source_rotation, 1200.0)
        source_relative = tuple(x - source_order[0] for x in source_order)
        for subset_indices in itertools.combinations(range(len(target_steps)), n):
            subset_cents = tuple(target_cents[i] for i in subset_indices)
            subset_steps = tuple(target_steps[i] for i in subset_indices)
            for target_rotation in range(n):
                target_order = _rotated(subset_cents, target_rotation, 1200.0)
                target_relative = tuple(x - target_order[0] for x in target_order)
                residual = tuple(t - s for s, t in zip(
                    source_relative, target_relative))
                translation = sum(residual) / n
                errors = tuple(x - translation for x in residual)
                rms = math.sqrt(sum(x * x for x in errors) / n)
                maximum = max(abs(x) for x in errors)
                target_step_order = tuple(
                    subset_steps[(target_rotation + i) % n] for i in range(n))
                source_index_order = tuple((source_rotation + i) % n for i in range(n))
                mapping = dict(zip(source_index_order, target_step_order))
                key = (round(rms, 12), round(maximum, 12), subset_indices,
                       source_rotation, target_rotation)
                if best is None or key < best[0]:
                    best = (key, mapping, errors, translation)
    _, mapping, errors, translation = best
    return {
        "source_pc_to_target_step": mapping,
        "rms_mapping_error_cents": math.sqrt(
            sum(x * x for x in errors) / len(errors)),
        "maximum_mapping_error_cents": max(abs(x) for x in errors),
        "removed_global_translation_cents": translation,
    }


def _source_class_index(value, pitch_classes,
                        tolerance=PITCH_EQUIVALENCE_CENTS):
    distances = [_circular_distance(value % 1200.0, pc)
                 for pc in pitch_classes]
    index = min(range(len(distances)), key=distances.__getitem__)
    if distances[index] > float(tolerance) + 1e-7:
        raise ValueError(
            f"source pitch {value:.6f}c is {distances[index]:.6f}c from its "
            "nearest inferred scale degree")
    return index


def _target_base_midi_cents(spec):
    return 6900.0 + 1200.0 * math.log2(float(spec.base_freq_hz) / 440.0)


def retune_events(analysis, target_spec):
    pitch_classes = analysis["source_pitch_classes_cents"]
    mapping = best_scale_mapping(pitch_classes, target_spec)
    target_base = _target_base_midi_cents(target_spec)
    raw = []
    for row in analysis["reference"]["highest_line"]:
        cents = _actual_cents(row)
        source_index = _source_class_index(cents, pitch_classes)
        pc_step = int(mapping["source_pc_to_target_step"][source_index])
        octave = round((cents - (target_base + 1200.0 * pc_step / target_spec.edo))
                       / 1200.0)
        raw.append((row, pc_step + int(octave) * int(target_spec.edo),
                    source_index, pc_step, cents))

    # Each mapped class is placed in the octave nearest the source's actual
    # frequency.  Do not force it into the normal freshly-generated Lead pool:
    # this mode promises to copy the supplied melody and register, while the
    # fixed-Lead accompaniment initializer can expand lower voices as needed.
    octave_shift = 0
    shifted = [step for _, step, _, _, _ in raw]
    events = []
    for (row, _, source_index, pc_step, source_cents), step in zip(raw, shifted):
        events.append({
            "bar": int(row["bar"]),
            "start_beat": float(row["start_beat"]),
            "duration_beats": float(row["duration_beats"]),
            "step": int(step),
            "source_actual_cents": source_cents,
            "source_scale_degree_index": int(source_index),
            "target_pitch_class_step": int(pc_step),
        })
    mapping["register_octave_shift"] = int(octave_shift)
    return events, mapping


def generate_frontend_ir(reference_path, *, seed=20260811, bpm=96.0,
                         spec=None, staff_id=None):
    if spec is None:
        spec = hr.SCALE
    analysis = analyze_reference(reference_path, staff_id=staff_id)
    if len(spec.pcs) < analysis["source_scale_note_count"]:
        raise ValueError(
            f"target scale has {len(spec.pcs)} notes per octave, fewer than "
            f"the inferred source scale's {analysis['source_scale_note_count']}; "
            "generation aborted before accompaniment annealing")
    measure_map = analysis["reference"]["measure_map"]
    bars = len(measure_map)
    mapped, mapping = retune_events(analysis, spec)
    if spec.resolved_chord_progression():
        plan = _scaled_harmony_plan(bars, random.Random(int(seed)), measure_map)
        harmony_method = 'explicit_progression'
    else:
        from retune_harmony import infer_harmony
        plan = infer_harmony(mapped, measure_map)
        harmony_method = 'metrical_marginal_cse_v1'
    rng = random.Random(int(seed) ^ 0x524554554E45)
    lead = []
    for index, row in enumerate(mapped):
        measure = measure_map[row["bar"]]
        offset = row["start_beat"] - float(measure["start_beat"])
        segment = hr.harmony_segment_at(plan[row["bar"]], offset)
        lead.append(sb.event(
            row["start_beat"], row["duration_beats"], row["step"], "lead",
            rng, segment, structural=_strong_attack(
                offset, measure["time_signature"]),
            motif="mscx_exact_melody_retune", retune_source_event_index=index,
            retune_source_actual_cents=row["source_actual_cents"],
            retune_source_scale_degree_index=row["source_scale_degree_index"],
            retune_target_pitch_class_step=row["target_pitch_class_step"],
        ))
    jumps = sb.lead_jump_errors(lead)
    if jumps:
        raise ValueError(f"retuned fixed melody violates Lead jump limits: {jumps[:3]}")
    first_signature = str(measure_map[0]["time_signature"])
    first_bpb = float(measure_map[0]["duration_beats"])
    metadata = {
        "mode": "exact_melody_retune",
        "harmony_inference": harmony_method,
        "absolute_pitch_count": analysis["absolute_pitch_count"],
        "source_scale_note_count": analysis["source_scale_note_count"],
        "target_scale_note_count": len(spec.pcs),
        "source_pitch_classes_cents": list(
            analysis["source_pitch_classes_cents"]),
        "pitch_equivalence_tolerance_cents": PITCH_EQUIVALENCE_CENTS,
        "target_is_exact_source_scale_match": bool(
            len(spec.pcs) == analysis["source_scale_note_count"]
            and mapping["maximum_mapping_error_cents"]
            <= PITCH_EQUIVALENCE_CENTS),
        "mapping_rows": [
            {
                "source_pitch_class_cents": float(source_pc),
                "target_step": int(mapping["source_pc_to_target_step"][index]),
                "target_name": str(spec.names[
                    spec.pcs.index(mapping["source_pc_to_target_step"][index])]),
            }
            for index, source_pc in enumerate(
                analysis["source_pitch_classes_cents"])
        ],
        **mapping,
    }
    return build_ir(
        seed=seed, bpm=bpm, time_signature=first_signature,
        beats_per_bar=first_bpb, bars=bars, spec=spec,
        harmony_plan=plan, lead=lead, allow_sixteenth=True,
        lead_palette={"source": "mscx_exact_melody_retune"},
        generator="mscx_exact_melody_retune", measure_map=measure_map,
        generator_metadata={"retune": metadata},
    )


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_mscx")
    parser.add_argument("--scale", dest="scale_config", required=True)
    parser.add_argument("--rules", dest="rules_config")
    parser.add_argument("--style", dest="style_config")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bpm", type=float, default=96.0)
    parser.add_argument("--staff-id")
    parser.add_argument("--cse-dir")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--save-frontend-ir")
    parser.add_argument("--output", "-o", default="retuned_score.json")
    args = parser.parse_args(argv)

    # Cheap cardinality rejection deliberately precedes CSE loading/building.
    target = __import__("scale_config").load_scale(
        args.scale_config, rules=args.rules_config, style=args.style_config,
        require_composition=True)
    source = analyze_reference(args.reference_mscx, staff_id=args.staff_id)
    if len(target.pcs) < source["source_scale_note_count"]:
        raise SystemExit(
            f"Target has {len(target.pcs)} notes; source scale has "
            f"{source['source_scale_note_count']}. Nothing generated.")
    import main
    score = main.save_score(
        filename=args.output, seed=args.seed, bpm=args.bpm,
        scale_config=args.scale_config, rules_config=args.rules_config,
        style_config=args.style_config, cse_dir=args.cse_dir,
        cse_workers=args.cse_workers,
        retune_melody_source=args.reference_mscx,
        retune_staff_id=args.staff_id,
        frontend_ir_output=args.save_frontend_ir)
    info = score["frontend_ir"]["generator"]["retune"]
    print("Retune source: absolute pitches", info["absolute_pitch_count"],
          "scale notes", info["source_scale_note_count"])
    print("Retune mapping: target notes", info["target_scale_note_count"],
          "RMS", f'{info["rms_mapping_error_cents"]:.6f}c',
          "max", f'{info["maximum_mapping_error_cents"]:.6f}c')
    return score


if __name__ == "__main__":
    cli()
