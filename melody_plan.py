"""Hierarchical MelodyPlan/2 generator for ScaleWeaver's new front end.

Containment is a 48 -> 8 -> 4 -> 2 -> 1 bar tree.  Recurrence edges turn that
tree into a DAG.  Every edge carries one of six legal inheritance masks over
Melody pitch (M), Lead rhythm (R), and harmony progression (H).  The atomic
timeline remains a half-bar grid, while terminal material leaves may span a
half bar, one bar, or two bars.
"""
from __future__ import annotations

import copy
import math
import random


FORMAT = "MelodyPlan/2"
MODES = ("FRESH", "H", "R", "RH", "MR", "MRH")
MODE_LAYERS = {
    "FRESH": frozenset(),
    "H": frozenset("H"),
    "R": frozenset("R"),
    "RH": frozenset("RH"),
    "MR": frozenset("MR"),
    "MRH": frozenset("MRH"),
}


DEFAULTS = {
    "paired_phrases": True,
    "alternating_rhythm_probability": 0.0,
    "harmonic_rhythm_variation": 0.4,
    "enabled": True,
    "unit_bars": 0.5,
    "leaf_span_weights": {"0.5": 0.25, "1": 0.45, "2": 0.30},
    "fresh_root_count_weights": {"2": 0.65, "3": 0.30, "4": 0.05},
    "cross_phrase_form_weights": {
        "AABBAB": 0.34,
        "AAB_CBC": 0.27,
        "ABCABC": 0.21,
        "ABACBC": 0.14,
        "ABCDAC": 0.04,
    },
    "macro_pattern_weights": {
        "uniform": 0.12,
        "motif_over_harmony": 0.32,
        "rhythm_identity": 0.18,
        "reharmonized_reprise": 0.26,
        "fresh_tail": 0.12,
    },
    "relation_mode_weights": {
        "H": 0.05, "R": 0.07, "RH": 0.10, "MR": 0.30, "MRH": 0.48,
    },
    "copy_length_weights": {
        "0.5": 0.03, "1": 0.06, "1.5": 0.08, "2": 0.20,
        "2.5": 0.08, "3": 0.10, "3.5": 0.07, "4": 0.20,
        "5": 0.05, "6": 0.06, "7": 0.02, "8": 0.05,
    },
    "rhythm_exact_probability": 0.75,
    "octave_shift_weights": {"-1": 0.12, "0": 0.76, "1": 0.12},
    "internal_recurrence": {
        "eight_bar_probability": 0.92,
        "four_bar_probability": 0.32,
        "two_bar_probability": 0.48,
    },
    "four_bar_relation_mode_weights": {
        "R": 0.72,
        "MR": 0.20,
        "H": 0.05,
        "RH": 0.03,
    },
    "bar_role_profiles": {
        "theme_statement": {
            "contour": "rise", "direction": 1, "guide_amplitude_octaves": 0.16,
            "register_offset_octaves": -0.03, "density_multiplier": 1.0,
            "cadence_root_weight": 0.0,
        },
        "theme_answer": {
            "contour": "fall", "direction": -1, "guide_amplitude_octaves": 0.18,
            "register_offset_octaves": 0.02, "density_multiplier": 0.96,
            "cadence_root_weight": 0.18,
        },
        "development": {
            "contour": "wave", "direction": 1, "guide_amplitude_octaves": 0.24,
            "register_offset_octaves": 0.04, "density_multiplier": 1.08,
            "cadence_root_weight": 0.0,
        },
        "climax": {
            "contour": "rise", "direction": 1, "guide_amplitude_octaves": 0.26,
            "register_offset_octaves": 0.10, "density_multiplier": 1.12,
            "cadence_root_weight": 0.0,
        },
        "reprise": {
            "contour": "arch", "direction": 0, "guide_amplitude_octaves": 0.16,
            "register_offset_octaves": 0.0, "density_multiplier": 0.98,
            "cadence_root_weight": 0.08,
        },
        "continuation": {
            "contour": "wave", "direction": -1, "guide_amplitude_octaves": 0.18,
            "register_offset_octaves": -0.02, "density_multiplier": 0.92,
            "cadence_root_weight": 0.10,
        },
        "liquidation": {
            "contour": "fall", "direction": -1, "guide_amplitude_octaves": 0.14,
            "register_offset_octaves": -0.06, "density_multiplier": 0.72,
            "cadence_root_weight": 0.28,
        },
        "cadence": {
            "contour": "fall", "direction": -1, "guide_amplitude_octaves": 0.12,
            "register_offset_octaves": -0.09, "density_multiplier": 0.54,
            "cadence_root_weight": 1.0,
        },
    },
    "joint_generation": {
        "beam_width": 8,
        "harmony_candidates": 3,
        "rhythm_candidates": 3,
        "melody_candidates": 2,
        "base_events_per_bar": 7.75,
        "dense_half_min_notes": 5,
        "dense_half_candidate_probability": 0.0,
        "fancy_frequency_multiplier": 0.35,
        "syncopated_frequency_multiplier": 0.35,
        "straight_flow_frequency_multiplier": 1.2,
        "sixteenth_cell_weights": {"3+1": 1.0, "1+3": 1.0, "1+2+1": 1.0},
        "melodic_shape_weight": 0.0,
        "allow_cross_unit_ties": True,
        "cross_half_bar_tie_probability": 0.18,
        "cross_bar_tie_probability": 0.12,
        "max_tied_duration_beats": 3.0,
        "selection_noise": 0.035,
        "weights": {
            "register": 0.34,
            "density": 0.20,
            "voice_leading": 0.72,
            "background": 1.0,
            "strong_beat_chord_tone": 0.42,
            "harmony_transition": 0.16,
            "cadence": 0.55,
            "rhythm_style": 0.0,
        },
    },
}


REGISTER_PROFILES = {
    "central_climax": (0.48, 0.52, 0.62, 0.86, 0.60, 0.55, 0.45, 0.38),
    "late_climax": (0.43, 0.49, 0.56, 0.64, 0.84, 0.62, 0.48, 0.38),
    "double_wave": (0.48, 0.62, 0.53, 0.72, 0.55, 0.78, 0.48, 0.36),
    "descending_answer": (0.70, 0.62, 0.55, 0.76, 0.58, 0.50, 0.42, 0.34),
}

DENSITY_PROFILES = {
    "central_climax": (1.00, 0.98, 1.10, 1.08, 1.00, 0.96, 0.82, 0.55),
    "late_climax": (0.94, 1.00, 1.04, 1.08, 1.14, 0.96, 0.78, 0.52),
    "double_wave": (1.00, 1.08, 0.92, 1.10, 0.94, 1.06, 0.78, 0.52),
    "descending_answer": (1.04, 1.00, 0.98, 1.08, 0.94, 0.90, 0.76, 0.50),
}

BAR_ROLES = (
    "theme_statement", "theme_answer", "development", "climax",
    "reprise", "continuation", "liquidation", "cadence",
)


def _merge(default, supplied, path="melody_plan"):
    if supplied is None:
        return copy.deepcopy(default)
    if not isinstance(supplied, dict):
        raise ValueError(f"{path} must be an object")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"Unknown {path} keys: {sorted(unknown)}")
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        if isinstance(default[key], dict):
            result[key] = _merge(default[key], value, f"{path}.{key}")
        else:
            result[key] = value
    return result


def _validate_weights(row, path):
    if not row or any(not math.isfinite(float(v)) or float(v) < 0 for v in row.values()):
        raise ValueError(f"{path} must contain finite nonnegative weights")
    if math.fsum(float(v) for v in row.values()) <= 0:
        raise ValueError(f"{path} must have positive total weight")


def resolve_melody_plan(supplied=None):
    cfg = _merge(DEFAULTS, supplied)
    if not 0 <= float(cfg['alternating_rhythm_probability']) <= 1:
        raise ValueError('alternating_rhythm_probability must be in 0..1')
    if not isinstance(cfg['paired_phrases'], bool):
        raise ValueError('melody_plan.paired_phrases must be boolean')
    if not 0 <= float(cfg['harmonic_rhythm_variation']) <= 1:
        raise ValueError('melody_plan.harmonic_rhythm_variation must be in 0..1')
    if float(cfg["unit_bars"]) != 0.5:
        raise ValueError("melody_plan.unit_bars is currently fixed at 0.5")
    for key in ("leaf_span_weights", "fresh_root_count_weights", "cross_phrase_form_weights",
                "macro_pattern_weights", "four_bar_relation_mode_weights",
                "relation_mode_weights", "copy_length_weights", "octave_shift_weights"):
        _validate_weights(cfg[key], f"melody_plan.{key}")
    if set(cfg["leaf_span_weights"]) - {"0.5", "1", "2"}:
        raise ValueError("melody_plan.leaf_span_weights may contain only 0.5, 1, and 2")
    if set(cfg["relation_mode_weights"]) - (set(MODES) - {"FRESH"}):
        raise ValueError("melody_plan.relation_mode_weights contains an invalid mode")
    if set(cfg["four_bar_relation_mode_weights"]) - {"R", "MR", "H", "RH"}:
        raise ValueError("melody_plan.four_bar_relation_mode_weights contains an invalid mode")
    for name in cfg["cross_phrase_form_weights"]:
        compact = str(name).replace("_", "")
        if len(compact) != 6 or compact[0] != "A" or any(ch not in "ABCD" for ch in compact):
            raise ValueError("melody_plan.cross_phrase_form_weights keys must encode six A-D sections")
        seen = []
        for ch in compact:
            if ch not in seen:
                seen.append(ch)
        if seen != list("ABCD"[:len(seen)]) or not 2 <= len(seen) <= 4:
            raise ValueError("cross-phrase forms must introduce two to four families in A-B-C-D order")
    for key, value in cfg["internal_recurrence"].items():
        if not 0 <= float(value) <= 1:
            raise ValueError(f"melody_plan.internal_recurrence.{key} must be in 0..1")
    if not 0 <= float(cfg["rhythm_exact_probability"]) <= 1:
        raise ValueError("melody_plan.rhythm_exact_probability must be in 0..1")
    joint = cfg["joint_generation"]
    for key in ("beam_width", "harmony_candidates", "rhythm_candidates",
                "melody_candidates", "dense_half_min_notes"):
        if isinstance(joint[key], bool) or int(joint[key]) != joint[key] or int(joint[key]) <= 0:
            raise ValueError(f"melody_plan.joint_generation.{key} must be a positive integer")
        joint[key] = int(joint[key])
    if not math.isfinite(float(joint["selection_noise"])) or float(joint["selection_noise"]) < 0:
        raise ValueError("melody_plan.joint_generation.selection_noise must be nonnegative")
    if not math.isfinite(float(joint["base_events_per_bar"])) or float(joint["base_events_per_bar"]) <= 0:
        raise ValueError("melody_plan.joint_generation.base_events_per_bar must be positive")
    if not 0 <= float(joint["dense_half_candidate_probability"]) <= 1:
        raise ValueError("melody_plan.joint_generation.dense_half_candidate_probability must be in 0..1")
    if not isinstance(joint["allow_cross_unit_ties"], bool):
        raise ValueError("melody_plan.joint_generation.allow_cross_unit_ties must be boolean")
    for key in ("cross_half_bar_tie_probability", "cross_bar_tie_probability"):
        if not 0 <= float(joint[key]) <= 1:
            raise ValueError(f"melody_plan.joint_generation.{key} must be in 0..1")
    if (not math.isfinite(float(joint["max_tied_duration_beats"])) or
            float(joint["max_tied_duration_beats"]) <= 0):
        raise ValueError("melody_plan.joint_generation.max_tied_duration_beats must be positive")
    for key in ("fancy_frequency_multiplier", "syncopated_frequency_multiplier"):
        if not 0 <= float(joint[key]) <= 1:
            raise ValueError(f"melody_plan.joint_generation.{key} must be in 0..1")
    if not math.isfinite(float(joint["straight_flow_frequency_multiplier"])) or float(
            joint["straight_flow_frequency_multiplier"]) <= 0:
        raise ValueError(
            "melody_plan.joint_generation.straight_flow_frequency_multiplier must be positive")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in joint["weights"].values()):
        raise ValueError("melody_plan.joint_generation.weights must be finite and nonnegative")
    if not math.isfinite(float(joint['melodic_shape_weight'])) or float(joint['melodic_shape_weight']) < 0:
        raise ValueError('melodic_shape_weight must be finite and nonnegative')
    if any(not math.isfinite(float(v)) or not 0 < float(v) <= 4
           for v in joint['sixteenth_cell_weights'].values()):
        raise ValueError('sixteenth_cell_weights must be finite and in (0, 4]')
    for role in BAR_ROLES:
        profile = cfg["bar_role_profiles"].get(role)
        if not isinstance(profile, dict):
            raise ValueError(f"melody_plan.bar_role_profiles.{role} must be an object")
        if profile.get("contour") not in {"rise", "fall", "arch", "wave", "level"}:
            raise ValueError(f"melody_plan.bar_role_profiles.{role}.contour is invalid")
        direction = profile.get("direction")
        if isinstance(direction, bool) or int(direction) != direction or int(direction) not in (-1, 0, 1):
            raise ValueError(f"melody_plan.bar_role_profiles.{role}.direction must be -1, 0, or 1")
        for key in ("guide_amplitude_octaves", "register_offset_octaves",
                    "density_multiplier", "cadence_root_weight"):
            if not math.isfinite(float(profile.get(key, 0.0))):
                raise ValueError(f"melody_plan.bar_role_profiles.{role}.{key} must be finite")
        if float(profile["guide_amplitude_octaves"]) < 0 or float(profile["density_multiplier"]) <= 0 or float(profile["cadence_root_weight"]) < 0:
            raise ValueError(f"melody_plan.bar_role_profiles.{role} contains an invalid magnitude")
    return cfg


def _weighted_key(rng, row):
    keys = list(row)
    return rng.choices(keys, weights=[float(row[k]) for k in keys], k=1)[0]


def _hierarchy_nodes(start, span, section_id, parent=None):
    node_id = f"{section_id}:{start:g}+{span:g}"
    node = {
        "id": node_id, "section_id": section_id, "start_bar": start,
        "span_bars": span, "parent": parent, "children": [],
    }
    rows = [node]
    if span > 1.0 and abs(span / 2 - round(span / 2)) < 1e-9:
        half = span / 2
        left = _hierarchy_nodes(start, half, section_id, node_id)
        right = _hierarchy_nodes(start + half, half, section_id, node_id)
        node["children"] = [left[0]["id"], right[0]["id"]]
        rows.extend(left)
        rows.extend(right)
    return rows


def _manual_phrase_nodes(start, span, section_id):
    """One phrase parent with direct one-bar children; no binary sub-tree."""
    parent_id = f"{section_id}:{start:g}+{span:g}"
    parent = {"id": parent_id, "section_id": section_id,
              "start_bar": float(start), "span_bars": float(span),
              "parent": None, "children": []}
    rows = [parent]
    for offset in range(int(span)):
        child_id = f"{section_id}:bar:{start + offset:g}+1"
        parent["children"].append(child_id)
        rows.append({"id": child_id, "section_id": section_id,
                     "start_bar": float(start + offset), "span_bars": 1.0,
                     "parent": parent_id, "children": []})
    return rows


def _resample_phrase_profile(values, count):
    """Map the established 8-bar arc onto an arbitrary complete phrase."""
    if count <= 0:
        return []
    if count == 1:
        return [values[-1]]
    return [values[int(round(i * (len(values)-1) / (count-1)))]
            for i in range(count)]


def _one_bar_material_leaves(actions, sections):
    """Manual harmony exposes exactly one direct terminal node per bar."""
    leaves = []
    for bar in range(len(actions)//2):
        start = bar*2
        rows = actions[start:start+2]
        same = all((row["mode"],row.get("relation_id")) ==
                   (rows[0]["mode"],rows[0].get("relation_id")) for row in rows)
        section = next(s for s in sections
                       if s["start_bar"] <= bar < s["start_bar"]+s["span_bars"])
        leaves.append({
            "id": f"leaf_{bar:03d}", "index": bar,
            "start_unit": start, "unit_indices": [start,start+1],
            "start_bar": float(bar), "duration_bars": 1.0, "span_units": 2,
            "mode": rows[0]["mode"] if same else "MIXED",
            "source_start_unit": rows[0].get("source_unit") if same else None,
            "source_start_bar": rows[0].get("source_start_bar") if same else None,
            "relation_id": rows[0].get("relation_id") if same else None,
            "bar_roles": [section["bar_roles"][bar-int(section["start_bar"])]],
        })
    return leaves


def _sample_length(rng, cfg, maximum):
    choices = {k: v for k, v in cfg["copy_length_weights"].items()
               if 0.5 <= float(k) <= maximum + 1e-9}
    return float(_weighted_key(rng, choices))


def _sample_mode(rng, cfg):
    return _weighted_key(rng, cfg["relation_mode_weights"])


def _mode_without_harmony(mode):
    return {"H": "FRESH", "RH": "R", "MRH": "MR"}.get(mode, mode)


def _cross_phrase_families(rng, cfg, section_count):
    """Choose an explicit 48-bar family form such as AA'BB'A''B''.

    Apostrophes are represented by another occurrence of the same family
    letter.  Repeated families depend on their most recent occurrence, so a
    family can evolve instead of repeatedly cloning its first statement.
    """
    if section_count != 6:
        return None, None
    name = _weighted_key(rng, cfg["cross_phrase_form_weights"])
    return str(name), tuple(str(name).replace("_", ""))


def _partition_material_leaves(actions, sections, cfg, rng):
    """Tile the atomic half-bar grid with 0.5/1/2-bar material leaves.

    A longer leaf may not straddle an inheritance boundary: all of its atomic
    actions must share the same mode/relation and copied sources must advance
    in lockstep.  Two-bar leaves are aligned to the existing two-bar hierarchy,
    and one-bar leaves to bar boundaries.  This keeps dependencies unambiguous
    while giving the joint realizer phrase-sized terminal material.
    """
    total = len(actions)

    def compatible(start, span_units):
        if start + span_units > total:
            return False
        if span_units == 4 and start % 4:
            return False
        if span_units == 2 and start % 2:
            return False
        # Do not let a terminal material leaf cross an eight-bar section edge.
        if start // 16 != (start + span_units - 1) // 16:
            return False
        rows = actions[start:start + span_units]
        first = rows[0]
        fixed = (first["mode"], first.get("relation_id"),
                 first.get("rhythm_fidelity"), first.get("melody_transform"),
                 int(first.get("octave_shift", 0)))
        for offset, row in enumerate(rows):
            other = (row["mode"], row.get("relation_id"),
                     row.get("rhythm_fidelity"), row.get("melody_transform"),
                     int(row.get("octave_shift", 0)))
            if other != fixed:
                return False
            if first["mode"] == "FRESH":
                if row.get("source_unit") is not None:
                    return False
            elif row.get("source_unit") != first.get("source_unit") + offset:
                return False
        return True

    def roles_for(start, span_units):
        roles = []
        for unit in range(start, start + span_units):
            section = sections[min(len(sections) - 1, unit // 16)]
            local_bar = min(len(section["bar_roles"]) - 1,
                            max(0, unit // 2 - int(section["start_bar"])))
            role = section["bar_roles"][local_bar]
            if not roles or roles[-1] != role:
                roles.append(role)
        return roles

    leaves = []
    cursor = 0
    while cursor < total:
        candidates = []
        for span_bars, span_units in ((2.0, 4), (1.0, 2), (0.5, 1)):
            if compatible(cursor, span_units):
                candidates.append((span_bars, span_units))
        weights = [float(cfg["leaf_span_weights"].get(f"{span:g}", 0.0))
                   for span, _ in candidates]
        if math.fsum(weights) <= 0:
            span_bars, span_units = 0.5, 1
        else:
            span_bars, span_units = rng.choices(candidates, weights=weights, k=1)[0]
        atomic = actions[cursor:cursor + span_units]
        first = atomic[0]
        leaves.append({
            "id": f"leaf_{len(leaves):03d}",
            "index": len(leaves),
            "start_unit": cursor,
            "unit_indices": list(range(cursor, cursor + span_units)),
            "start_bar": cursor * 0.5,
            "duration_bars": span_bars,
            "span_units": span_units,
            "mode": first["mode"],
            "source_start_unit": first.get("source_unit"),
            "source_start_bar": first.get("source_start_bar"),
            "relation_id": first.get("relation_id"),
            "bar_roles": roles_for(cursor, span_units),
        })
        cursor += span_units
    return leaves


def generate(seed, bars=48, beats_per_bar=4, supplied=None,
             manual_progression_bars=None, planned_harmony=None):
    """Generate a deterministic hierarchical plan from one seed."""
    cfg = resolve_melody_plan(supplied)
    bars = int(bars)
    if bars <= 0:
        raise ValueError("bars must be positive")
    manual_progression = manual_progression_bars is not None
    phrase_bars = int(manual_progression_bars) if manual_progression else 8
    if phrase_bars <= 0:
        raise ValueError("manual progression phrase length must be positive")
    if manual_progression and bars % phrase_bars:
        raise ValueError(
            f"manual progression spans {phrase_bars} bars; --bars={bars} must be "
            "an exact multiple so every phrase is complete")
    rng = random.Random(int(seed) ^ 0x4D504C414E)
    unit_bars = 0.5
    unit_count = bars * 2
    actions = [{
        "unit": i, "start_bar": i * unit_bars, "duration_bars": unit_bars,
        "mode": "FRESH", "source_unit": None, "source_start_bar": None,
        "rhythm_fidelity": "fresh", "melody_transform": "fresh",
        "octave_shift": 0, "relation_id": None,
    } for i in range(unit_count)]
    relations = []

    def add_relation(target, source, length, mode, scope, *, only_fresh=False,
                     force_exact=None, priority=0):
        if mode == "FRESH" or source < 0 or target <= source or length <= 0:
            return None
        n = min(int(round(length / unit_bars)), unit_count - int(round(target / unit_bars)))
        source_unit = int(round(source / unit_bars))
        target_unit = int(round(target / unit_bars))
        n = min(n, target_unit - source_unit)
        if n <= 0:
            return None
        # A preplanned functional progression takes priority over H inheritance.
        # Keep H only when the whole copied span matches the planned harmony.
        if planned_harmony is not None and "H" in MODE_LAYERS[mode]:
            if any(planned_harmony[target_unit+i] != planned_harmony[source_unit+i]
                   for i in range(n)):
                mode = _mode_without_harmony(mode)
                if mode == "FRESH":
                    return None
        rid = f"rel_{len(relations):03d}"
        fidelity = ("exact" if "M" in MODE_LAYERS[mode] else
                    "exact" if force_exact is True else
                    "near_exact" if force_exact is False else
                    "exact" if rng.random() < cfg["rhythm_exact_probability"] else "near_exact")
        octave = 0
        transform = "fresh"
        if "M" in MODE_LAYERS[mode]:
            octave = int(_weighted_key(rng, cfg["octave_shift_weights"]))
            transform = "exact" if octave == 0 else "octave_shift"
        applied = []
        for offset in range(n):
            ti = target_unit + offset
            si = source_unit + offset
            if si >= ti or si < 0 or ti >= unit_count:
                continue
            if only_fresh and actions[ti]["mode"] != "FRESH":
                continue
            actions[ti].update({
                "mode": mode, "source_unit": si,
                "source_start_bar": si * unit_bars,
                "rhythm_fidelity": fidelity if "R" in MODE_LAYERS[mode] else "fresh",
                "melody_transform": transform,
                "octave_shift": octave,
                "relation_id": rid,
            })
            applied.append(ti)
        if not applied:
            return None
        row = {
            "id": rid, "scope": scope, "priority": int(priority),
            "source_start_bar": source, "target_start_bar": target,
            "requested_length_bars": length,
            "realized_length_bars": len(applied) * unit_bars,
            "mode": mode, "layers": sorted(MODE_LAYERS[mode]),
            "rhythm_fidelity": fidelity,
            "melody_transform": transform, "octave_shift": octave,
            "target_units": applied,
        }
        relations.append(row)
        return row

    section_count = (bars + phrase_bars - 1) // phrase_bars
    cross_phrase_form, cross_phrase_families = _cross_phrase_families(rng, cfg, section_count)
    if cross_phrase_families is not None:
        root_positions = {i for i,family in enumerate(cross_phrase_families)
                          if family not in cross_phrase_families[:i]}
    else:
        max_roots = min(4, section_count)
        min_roots = min(2, section_count)
        root_choices = {k: v for k, v in cfg["fresh_root_count_weights"].items()
                        if min_roots <= int(k) <= max_roots}
        root_count = int(_weighted_key(rng, root_choices)) if root_choices else section_count
        candidates = list(range(1, section_count))
        extra_roots = sorted(rng.sample(candidates, root_count - 1)) if root_count > 1 else []
        root_positions = {0, *extra_roots}
    family_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    family_for_section = {}
    source_for_section = {}
    sections = []
    nodes = []
    current_family = -1

    for section_index in range(section_count):
        start = section_index * phrase_bars
        span = min(phrase_bars, bars - start)
        is_root = section_index in root_positions
        if cross_phrase_families is not None:
            family = cross_phrase_families[section_index]
            previous = [i for i in range(section_index)
                        if cross_phrase_families[i] == family]
            source_index = previous[-1] if previous else None
        elif is_root:
            current_family += 1
            family = family_names[current_family]
            source_index = None
        else:
            choices = list(range(section_index))
            weights = [1.0 / (1.0 + 0.35 * (section_index - i - 1)) for i in choices]
            source_index = rng.choices(choices, weights=weights, k=1)[0]
            family = family_for_section[source_index]
        family_for_section[section_index] = family
        source_for_section[section_index] = source_index
        profile = rng.choice(tuple(REGISTER_PROFILES))
        section_id = f"section_{section_index + 1}"
        section = {
            "id": section_id, "index": section_index, "start_bar": start,
            "span_bars": span, "family": family, "fresh_root": is_root,
            "source_section": None if source_index is None else f"section_{source_index + 1}",
            "profile": profile,
            "register_curve": (_resample_phrase_profile(REGISTER_PROFILES[profile],span)
                               if manual_progression else list(REGISTER_PROFILES[profile][:span])),
            "density_curve": (_resample_phrase_profile(DENSITY_PROFILES[profile],span)
                              if manual_progression else list(DENSITY_PROFILES[profile][:span])),
            "bar_roles": (_resample_phrase_profile(BAR_ROLES,span)
                          if manual_progression else list(BAR_ROLES[:span])),
        }
        sections.append(section)
        if manual_progression:
            nodes.extend(_manual_phrase_nodes(float(start),float(span),section_id))
        elif span in (1, 2, 4, 8):
            nodes.extend(_hierarchy_nodes(float(start), float(span), section_id))

        if not is_root:
            source_start = source_index * phrase_bars
            pattern = _weighted_key(rng, cfg["macro_pattern_weights"])
            section["macro_pattern"] = pattern
            if pattern == "uniform":
                length = _sample_length(rng, cfg, span)
                add_relation(start, source_start, length, _sample_mode(rng, cfg),
                             "eight_bar_cross_section", priority=10)
            elif pattern == "motif_over_harmony":
                add_relation(start, source_start, span, "H",
                             "eight_bar_harmony_identity", force_exact=True, priority=10)
                motif_len = _sample_length(rng, cfg, min(4.0, span))
                add_relation(start, source_start, motif_len, "MRH",
                             "motif_complete_reprise", force_exact=True, priority=30)
            elif pattern == "rhythm_identity":
                add_relation(start, source_start, span, "R",
                             "eight_bar_rhythm_identity", priority=10)
                motif_len = _sample_length(rng, cfg, min(4.0, span))
                add_relation(start, source_start, motif_len, "MRH",
                             "motif_complete_reprise", force_exact=True, priority=30)
            elif pattern == "reharmonized_reprise":
                length = _sample_length(rng, cfg, span)
                add_relation(start, source_start, length, "MR",
                             "reharmonized_melody_reprise", force_exact=True, priority=20)
            else:
                maximum = max(0.5, span - 0.5)
                length = _sample_length(rng, cfg, maximum)
                add_relation(start, source_start, length, _sample_mode(rng, cfg),
                             "reprise_with_fresh_tail", priority=20)

        # Every externally fresh family can organize itself recursively.  The
        # characteristic bars 1-2 -> 5-6 recall is the strongest local edge.
        if is_root and span >= 5 and rng.random() < cfg["internal_recurrence"]["eight_bar_probability"]:
            length = _sample_length(rng, cfg, min(4.0, span - 4.0))
            add_relation(start + 4, start, length,
                         "R" if manual_progression else _sample_mode(rng, cfg),
                         "eight_bar_internal_reprise", only_fresh=True, priority=20)

        for block_start in range(start, start + span, 4):
            block_span = min(4, start + span - block_start)
            if block_span >= 4 and rng.random() < cfg["internal_recurrence"]["four_bar_probability"]:
                mode = ("R" if manual_progression else
                        _weighted_key(rng, cfg["four_bar_relation_mode_weights"]))
                # At this level pitch identity is motif-sized only: the first
                # half-bar may recur, while the remainder of the two-bar answer
                # is newly written.  Rhythm/harmony-only relations may span the
                # complete two-bar child.
                length = 0.5 if "M" in MODE_LAYERS[mode] else _sample_length(rng, cfg, 2.0)
                add_relation(block_start + 2, block_start, length, mode,
                             "four_bar_internal_relation", only_fresh=True, priority=40)
            for pair_start in range(block_start, block_start + block_span, 2):
                if pair_start + 2 <= start + span and rng.random() < cfg["internal_recurrence"]["two_bar_probability"]:
                    length = _sample_length(rng, cfg, 1.0)
                    # Sibling one-bar units may share rhythmic identity only.
                    # Melody and harmony are always regenerated independently.
                    add_relation(pair_start + 1, pair_start, length, "R",
                                 "two_bar_internal_relation", only_fresh=True, priority=50)

    # An internal H-copy may not replace the mandatory ending anchor of an
    # eight-bar section.  A corresponding cross-section ending may copy H
    # because its source is itself an anchored ending.
    for section in sections:
        final_unit = int(round((section["start_bar"] + section["span_bars"]) / unit_bars)) - 1
        if not 0 <= final_unit < unit_count:
            continue
        action = actions[final_unit]
        if "H" in MODE_LAYERS[action["mode"]]:
            source_unit = action["source_unit"]
            source_is_section_end = (source_unit is not None and
                                     (source_unit + 1) % (phrase_bars*2) == 0)
            if not source_is_section_end:
                action["mode"] = _mode_without_harmony(action["mode"])
                action["anchor_harmony_override"] = True
                if action["mode"] == "FRESH":
                    action.update({
                        "source_unit": None, "source_start_bar": None,
                        "rhythm_fidelity": "fresh", "melody_transform": "fresh",
                        "octave_shift": 0, "relation_id": None,
                    })

    if cfg['paired_phrases']:
        for section in sections:
            start = int(section['start_bar'])
            span = int(section['span_bars'])
            alternating = (random.Random(int(seed) ^ (start*1009) ^ 0x52454341).random()
                           < cfg['alternating_rhythm_probability'])
            # Backward edges at three nested scales; never cross a phrase edge.
            for bar in range(span):
                local = bar % 8
                source_bar = (0, 0, 0, 1, 0, 1, 2, 3)[local]
                role = ('statement', 'answer', 'develop', 'develop',
                        'recall', 'recall', 'liquidate', 'close')[local]
                answer_type = rng.choice(('opening', 'ending', 'sequence', 'rhythm'))
                sequence_shift = rng.choice((-2, -1, 1, 2))
                for half_index in range(2):
                    row = actions[(start + bar)*2 + half_index]
                    source = ((start + bar-local + source_bar)*2 + half_index
                              if local else None)
                    row['paired_phrase'] = {
                        'role': role, 'source_unit': source,
                        'closing': bar == span-1 and half_index == 1,
                        'answer_type': answer_type,
                        'sequence_shift': sequence_shift,
                        'half_index': half_index,
                        'rhythm_independent': alternating and local == 1,
                        'rhythm_recall': alternating and 2 <= local <= 5,
                    }
                    # Preserve intentional cross-section melody copies. Local
                    # copies become answers that can adapt to incoming harmony.
                    if row['source_unit'] is not None and row['source_unit'] >= start*2:
                        row.update(mode='FRESH', source_unit=None, source_start_bar=None,
                                   rhythm_fidelity='fresh', melody_transform='fresh',
                                   octave_shift=0, relation_id=None)
                    elif row['mode'] == 'R':
                        row.update(mode='FRESH', source_unit=None, source_start_bar=None,
                                   rhythm_fidelity='fresh', relation_id=None)
                    elif row['mode'] == 'RH':
                        row.update(mode='H', rhythm_fidelity='fresh')

    # Report effective inheritance after the paired-phrase overlay.
    for relation in relations:
        relation['target_units'] = [i for i in relation['target_units']
                                    if actions[i]['relation_id'] == relation['id']]
        relation['realized_length_bars'] = len(relation['target_units'])*unit_bars
        if relation['target_units']:
            mode = actions[relation['target_units'][0]]['mode']
            relation['mode'] = mode
            relation['layers'] = sorted(MODE_LAYERS[mode])
            if 'R' not in MODE_LAYERS[mode]:
                relation['rhythm_fidelity'] = 'fresh'
    relations = [r for r in relations if r['target_units']]

    leaves = (_one_bar_material_leaves(actions,sections) if manual_progression else
              _partition_material_leaves(actions, sections, cfg, rng))

    plan = {
        "format": FORMAT,
        "seed": int(seed),
        "bars": bars,
        "beats_per_bar": float(beats_per_bar),
        "unit_bars": unit_bars,
        "section_size_bars": phrase_bars,
        "manual_progression": manual_progression,
        "fresh_root_sections": sorted(root_positions),
        "fresh_root_count": len(root_positions),
        "material_families": sorted(set(family_for_section.values())),
        "cross_phrase_form": cross_phrase_form,
        "sections": sections,
        "hierarchy_nodes": nodes,
        "relations": relations,
        "unit_actions": actions,
        "leaf_actions": leaves,
        "inheritance_modes": {mode: sorted(MODE_LAYERS[mode]) for mode in MODES},
        "config": cfg,
    }
    return validate(plan)


def validate(plan):
    if not isinstance(plan, dict) or plan.get("format") != FORMAT:
        raise ValueError(f"MelodyPlan format must be {FORMAT}")
    bars = int(plan.get("bars", 0))
    actions = plan.get("unit_actions")
    leaves = plan.get("leaf_actions")
    if bars <= 0 or not isinstance(actions, list) or len(actions) != bars * 2:
        raise ValueError("MelodyPlan unit_actions must contain two units per bar")
    for index, action in enumerate(actions):
        mode = action.get("mode")
        if mode not in MODES:
            raise ValueError(f"Invalid inheritance mode at unit {index}: {mode}")
        source = action.get("source_unit")
        if mode == "FRESH":
            if source is not None:
                raise ValueError("FRESH action cannot have a source")
        elif not isinstance(source, int) or not 0 <= source < index:
            raise ValueError(f"Unit {index} must depend on an earlier source")
        if "M" in MODE_LAYERS[mode] and "R" not in MODE_LAYERS[mode]:
            raise ValueError("Melody inheritance requires rhythm inheritance")
    if not isinstance(leaves, list) or not leaves:
        raise ValueError("MelodyPlan leaf_actions must contain material leaves")
    covered = []
    expected_start = 0
    for leaf_index, leaf in enumerate(leaves):
        start = leaf.get("start_unit")
        span = leaf.get("span_units")
        units = leaf.get("unit_indices")
        if plan.get("manual_progression") and span != 2:
            raise ValueError("Manual progression material leaves must each span one bar")
        if start != expected_start or span not in (1, 2, 4):
            raise ValueError(f"Invalid material leaf {leaf_index}")
        if (span == 4 and start % 4 != 0) or (span == 2 and start % 2 != 0):
            raise ValueError(f"Material leaf {leaf_index} is not binary-grid aligned")
        expected_units = list(range(start, start + span))
        if units != expected_units:
            raise ValueError(f"Material leaf {leaf_index} has invalid unit coverage")
        covered.extend(units)
        expected_start += span
    if covered != list(range(bars * 2)):
        raise ValueError("MelodyPlan material leaves must cover every half-bar exactly once")
    return plan
