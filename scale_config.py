#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale configuration for the adaptive exact-EDO composition toolchain.

A scale is always a subset of one exact EDO.  The configuration also carries
scale-specific musical grammar that should *not* be hard-coded in the generic
composer: melodic-contour behaviour, vertical wolf intervals, and post-pass
cleanup distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re
import copy
import warnings



DEFINITION_FIELDS = frozenset(('format', 'id', 'name', 'edo', 'pcs', 'names', 'base_note', 'base_freq_hz'))
STYLE_FIELDS = frozenset(('format', 'scale_id', 'prime_rewards', 'ngram_file',
                         'lead_pc_target_distribution', 'cse_weights',
                         'attack_cse_percentile_limits', 'annealing', 'voice_pc_rewards',
                         'melody_plan'))
CSE_WEIGHT_DEFAULTS = {
    'CSE_2D_A': 1.0,
    'CSE_2D_B': 0.0,
    'CSE_2D_C': 0.0,  # CCSE coefficient; no additive offset
    'LEAD_CHORD_FIELD_CSE_STRONG_WEIGHT': 2.0,
    'LEAD_CHORD_FIELD_CSE_WEAK_WEIGHT': 1.0,
    'LEAD_CHORD_FIELD_CSE_CENTER': 0.0,
    'COUNTERPOINT_BACKGROUND_CSE_WEIGHT': {'counter': .58, 'inner': .72, 'bass': .62},
    'COUNTERPOINT_ACTUAL_CSE_WEIGHT': {'counter': .90, 'inner': 1.05, 'bass': .82},
    'COUNTERPOINT_ATTACK_CSE_VIEW_WEIGHTS': {'sounding': 1.5, 'simultaneous': 1.0, 'future_attack': .5},
    'COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT': 6.0,
    'COUNTERPOINT_RELATIVE_PURITY_WEIGHT': 2.0,
    'COUNTERPOINT_EXTRA_OCTAVE_FAMILY_PAIR_COST': 4.60,
    'COUNTERPOINT_PARTIAL_OCTAVE_FAMILY_PAIR_COST': 3.10,
}


# Removed policies are accepted only for migration and never enter scoring.
# This explicit registry prevents old profiles from silently restoring them.
REMOVED_GENERATOR_FIELDS = frozenset((
    'tonal_core_pcs','tonal_core_shares','tonal_core_memory','tonal_core_prior',
    'tonal_core_response','tonal_core_target','lead_core_pc_targets',
    'lead_core_freq_prior','lead_core_pc_gain','lead_core_total_gain',
    'lead_core_count_gain','lead_core_base_bonus','lead_pc_target_normalize',
    'pc_selection_reward',
))
REMOVED_HARMONY_FIELDS = frozenset((
    'melody_7_colour_by_pc','chord_pc_log_reward','chord_presence_bonuses',
    'prime_onset_cost','prime_release_cost','prime_memory_cost',
    'chord_tone_7_colour_continuity_cost','established_colour_release_extra',
    'prime_class_prior',
))

def _strip_removed_policies(rules):
    metadata=rules.get('metadata',{})
    removed=[]
    for section,keys in (('generator',REMOVED_GENERATOR_FIELDS),
                         ('harmony_style',REMOVED_HARMONY_FIELDS)):
        values=metadata.get(section,{})
        for key in sorted(keys & values.keys()):
            values.pop(key)
            removed.append(f'metadata.{section}.{key}')
    if removed:
        warnings.warn('Removed legacy scoring fields ignored: '+', '.join(removed)+
                      '. Use style.lead_pc_target_distribution and style.prime_rewards.',
                      UserWarning,stacklevel=3)


def resolved_cse_weights(style):
    supplied = style.get('cse_weights', {})
    unknown = set(supplied) - set(CSE_WEIGHT_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown style CSE weights: {sorted(unknown)}')
    result = copy.deepcopy(CSE_WEIGHT_DEFAULTS)
    for key, raw in supplied.items():
        if isinstance(result[key], dict):
            if not isinstance(raw, dict) or set(raw) - set(result[key]):
                raise ValueError(f'Invalid voice/view keys in cse_weights.{key}')
            result[key].update({k: float(v) for k, v in raw.items()})
        else:
            result[key] = float(raw)
    for key, value in result.items():
        values = value.values() if isinstance(value, dict) else (value,)
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f'Non-finite cse_weights.{key}')
    return result


def resolved_attack_cse_percentile_limits(style):
    """Return enabled simultaneous-attack blend limits by cardinality.

    Six fractions select limits by total attack cardinality and subset
    cardinality. ``fallback_slack`` is also a fraction. All refer to the
    *active* ``a*CSE + b*SE + c*CCSE`` distributions; omitting the whole field
    disables the feature for backward compatibility.
    """
    if 'attack_cse_percentile_limits' not in style:
        return {}
    supplied = style.get('attack_cse_percentile_limits')
    if not isinstance(supplied, dict):
        raise ValueError('attack_cse_percentile_limits must be an object')
    expected = {
        'two_note_attack': {'dyad'},
        'three_note_attack': {'dyad_subsets','triad'},
        'four_note_attack': {'dyad_subsets','triad_subsets','tetrad'},
    }
    if set(supplied) != set(expected) | {'fallback_slack'}:
        raise ValueError('attack_cse_percentile_limits requires three attack groups and fallback_slack')
    result = {'limits':{},'fallback_slack':float(supplied['fallback_slack'])}
    card_map = {
        'two_note_attack':(2,{'dyad':2}),
        'three_note_attack':(3,{'dyad_subsets':2,'triad':3}),
        'four_note_attack':(4,{'dyad_subsets':2,'triad_subsets':3,'tetrad':4}),
    }
    for name,(attack_card,subset_map) in card_map.items():
        group = supplied[name]
        if not isinstance(group,dict) or set(group) != expected[name]:
            raise ValueError(f'Invalid attack CSE limit group: {name}')
        result['limits'][attack_card] = {
            subset_card:float(group[key]) for key,subset_card in subset_map.items()
        }
    values = [result['fallback_slack']] + [
        value for group in result['limits'].values() for value in group.values()
    ]
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise ValueError('attack CSE percentiles and fallback_slack must be finite fractions in [0, 1]')
    return result

def resolved_voice_pc_rewards(style, pitch_classes):
    """Signed, octave-equivalent accompaniment pitch-class preferences.

    Positive values are rewards, negative values are penalties. Lead has no
    interface here: its existing target-distribution generator remains in charge.
    """
    raw=style.get('voice_pc_rewards',{})
    # inner2 is the additional middle voice in five-part writing.  Keep lead
    # deliberately absent: its distribution is controlled by the melody plan.
    voices=('bass','inner','inner2','counter');allowed=set(map(int,pitch_classes))
    if not isinstance(raw,dict) or set(raw)-set(voices):
        raise ValueError('voice_pc_rewards supports only bass, inner, inner2 and counter')
    result={v:{} for v in voices}
    for voice,rows in raw.items():
        if not isinstance(rows,dict):raise ValueError(f'voice_pc_rewards.{voice} must be an object')
        for key,value in rows.items():
            try:pc=int(str(key))
            except (TypeError,ValueError):raise ValueError(f'Invalid pitch class in voice_pc_rewards.{voice}: {key}')
            if pc not in allowed or str(pc) in result[voice]:
                raise ValueError(f'Invalid or duplicate pitch class in voice_pc_rewards.{voice}: {key}')
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'voice_pc_rewards.{voice}.{key} must be a finite number')
            result[voice][str(pc)]=float(value)
    return result


def _slug(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("_.-")
    return text or "scale"


def _tuple_ints(value) -> tuple[int, ...]:
    return tuple(int(x) for x in (value or ()))


@dataclass(frozen=True)
class ScaleSpec:
    id: str
    name: str
    edo: int
    pcs: tuple[int, ...]
    names: tuple[str, ...]
    base_freq_hz: float
    base_note: str
    voice_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Vertical hard-wolf collection.  Values are EDO-step interval classes:
    # abs(a-b) % edo == value is forbidden.
    forbidden_intervals: tuple[int, ...] = ()

    # Exact absolute step distances; no octave reduction or inversion folding.
    absolute_wolf_intervals: tuple[int, ...] = ()

    # Cleanup degree distances count scale positions, independently of EDO steps.
    cleanup_degree_distances: tuple[int, ...] = ()
    cleanup_step_distances: tuple[int, ...] = ()
    cleanup_passes: int = 8
    cleanup_max_shift_degrees: dict[str, int] = field(default_factory=dict)
    # Soft generation-time penalty for configured non-wolf special distances.
    # These distances are *not* hard legality rules: generation may keep them
    # when musically necessary, and the final cleanup pass is best-effort only.
    cleanup_soft_penalty: float = 0.72

    # Melodic-direction grammar.  Kept as a compact user-editable dictionary so
    # later scales can tune contour strengths without changing Python code.
    melody: dict[str, Any] = field(default_factory=dict)

    # Sparse scale-specific n-gram file. Relative paths are resolved against the
    # style JSON directory by load_scale(). Other scales may point to an empty
    # file. The file is intentionally NOT part of the CSE signature.
    ngram_file: str | None = None

    # Functional-harmony pitch partitions. ``harmony`` contains two orthogonal
    # partitions: pitch_roles={tonic, dominant, subdominant, other} and
    # stability={stable, unstable}. Missing data receives a conservative
    # conventional auto-partition, but explicit configs are recommended.
    harmony: dict[str, Any] = field(default_factory=dict)

    # Private origin directory used only for resolving relative auxiliary files.
    config_dir: str | None = field(default=None, repr=False, compare=False)

    cse_wide_bounds: tuple[int, int] | None = None
    cse_inner_bounds: tuple[int, int] | None = None
    cse_five_bounds: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    style_dir: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        edo = int(self.edo)
        if edo < 3:
            raise ValueError("edo must be >= 3")
        pcs = tuple(int(x) for x in self.pcs)
        if len(pcs) < 3:
            raise ValueError("scale must contain at least three pitch classes")
        if len(set(pcs)) != len(pcs):
            raise ValueError("scale pitch classes must be unique")
        if tuple(sorted(pcs)) != pcs:
            raise ValueError("scale pitch classes must be strictly ascending")
        if pcs[0] != 0:
            raise ValueError("scale must include 0 as its first pitch class")
        if any(x < 0 or x >= edo for x in pcs):
            raise ValueError("every pitch class must satisfy 0 <= pc < edo")
        if len(self.names) != len(pcs):
            raise ValueError("names and pcs must have the same length")
        if not (float(self.base_freq_hz) > 0 and math.isfinite(float(self.base_freq_hz))):
            raise ValueError("base_freq_hz must be a positive finite number")
        for x in self.forbidden_intervals:
            if not 0 < abs(int(x)) < edo:
                raise ValueError("forbidden_intervals must be non-zero interval classes smaller than edo")
        for x in self.absolute_wolf_intervals:
            if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                raise ValueError('wolf_absolute_intervals must contain positive integer step distances')
        for x in self.cleanup_degree_distances:
            if int(x) <= 0:
                raise ValueError("cleanup_degree_distances must be positive")
        for x in self.cleanup_step_distances:
            if int(x) <= 0:
                raise ValueError("cleanup_step_distances must be positive")
        if int(self.cleanup_passes) < 0:
            raise ValueError("cleanup_passes must be >= 0")
        if not (math.isfinite(float(self.cleanup_soft_penalty)) and float(self.cleanup_soft_penalty) >= 0.0):
            raise ValueError("cleanup_soft_penalty must be a finite non-negative number")
        # Force validation of the two functional-harmony partitions now, so
        # malformed scale configs fail before any expensive CSE generation.
        self.resolved_harmony()
        self.resolved_chord_codes()
        self.resolved_chord_progression()

    def resolved_chord_codes(self) -> dict[str, dict[str, str]]:
        """Return optional one-character aliases for named frozen chords."""
        raw = self.harmony.get('chord_codes')
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError('harmony.chord_codes must be an object')
        out: dict[str, dict[str, str]] = {}
        for code, value in raw.items():
            code = str(code)
            if len(code) != 1 or code not in '0123456789abcdefghijklmnopqrstuvwxyz':
                raise ValueError('harmony.chord_codes keys must be one lowercase letter or digit')
            if isinstance(value, str):
                item = {'chord_id': value}
            elif isinstance(value, dict):
                if set(value) - {'chord_id', 'name', 'ratio'}:
                    raise ValueError(f'harmony.chord_codes.{code} has unsupported fields')
                item = {str(k): str(v) for k, v in value.items()}
            else:
                raise ValueError(f'harmony.chord_codes.{code} must be a chord id or object')
            if not item.get('chord_id'):
                raise ValueError(f'harmony.chord_codes.{code}.chord_id is required')
            out[code] = item
        return out

    def resolved_chord_progression(self):
        """Optional repeating scale-degree or named-chord progression."""
        raw = self.harmony.get('chord_progression')
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) - {'degrees', 'codes', 'bars_per_chord'}:
            raise ValueError('harmony.chord_progression requires degrees or codes and optional bars_per_chord')
        if ('degrees' in raw) == ('codes' in raw):
            raise ValueError('chord_progression must contain exactly one of degrees or codes')
        duration = raw.get('bars_per_chord', 1)
        if type(duration) is not int or duration < 1:
            raise ValueError('chord_progression.bars_per_chord must be a positive integer')
        if 'codes' in raw:
            codes = raw.get('codes')
            definitions = self.resolved_chord_codes()
            if not isinstance(codes, (list, tuple)) or not codes:
                raise ValueError('chord_progression.codes must be a nonempty list')
            if any(type(code) is not str or len(code) != 1 or code not in definitions for code in codes):
                raise ValueError('chord_progression.codes contains an undefined chord code')
            return {'codes': list(codes), 'bars_per_chord': duration}
        degrees = raw.get('degrees')
        if not isinstance(degrees, (list, tuple)) or not degrees:
            raise ValueError('chord_progression.degrees must be a nonempty integer list')
        if any(type(d) is not int or not 1 <= d <= self.note_count for d in degrees):
            raise ValueError(f'chord_progression degrees must be integers in 1..{self.note_count}')
        for d in degrees:
            pcs = tuple(self.pcs[(d-1+i) % self.note_count] for i in (0, 2, 4))
            if len(set(pcs)) != 3:
                raise ValueError('This scale has too few degrees for stacked-third triads')
            if any(self.is_octave_equivalent_wolf(a, b)
                   for i, a in enumerate(pcs) for b in pcs[i+1:]):
                raise ValueError(f'chord_progression degree {d} conflicts with configured octave-equivalent wolves')
        return {'degrees': list(degrees), 'bars_per_chord': duration}

    @property
    def note_count(self) -> int:
        return len(self.pcs)

    @cached_property
    def pc_name(self) -> dict[int, str]:
        return dict(zip(self.pcs, self.names))

    @cached_property
    def pc_index(self) -> dict[int, int]:
        return {int(pc): i for i, pc in enumerate(self.pcs)}

    @property
    def safe_id(self) -> str:
        return _slug(self.id)

    @cached_property
    def wolf_interval_classes(self) -> frozenset[int]:
        return frozenset(abs(int(x)) % self.edo for x in self.forbidden_intervals)

    def is_octave_equivalent_wolf(self, a: int, b: int) -> bool:
        return abs(int(a) - int(b)) % self.edo in self.wolf_interval_classes

    def is_hard_wolf(self, a: int, b: int) -> bool:
        distance = abs(int(a) - int(b))
        return (distance in self.absolute_wolf_intervals
                or self.is_octave_equivalent_wolf(a, b))

    def resolved_cleanup_max_shift_degrees(self) -> dict[str, int]:
        defaults = {"lead": 4, "counter": 5, "inner": 6, "bass": 6}
        for voice, value in (self.cleanup_max_shift_degrees or {}).items():
            defaults[str(voice)] = max(0, int(value))
        return defaults

    def resolved_melody(self) -> dict[str, Any]:
        # Values are in scale degrees, so the same grammar scales naturally from
        # seven notes to ten notes without pretending the EDO itself is diatonic.
        n = self.note_count
        defaults: dict[str, Any] = {
            "enabled": True,
            "phrase_bars": 8,
            "phrase_shapes": {"arch": .44, "rise": .22, "fall": .22, "valley": .12},
            "bar_shapes": {"arch": .40, "rise": .25, "fall": .25, "valley": .10},
            "phrase_amplitudes": [max(2, round(n * .22)), max(2, round(n * .30)), max(3, round(n * .38))],
            "bar_amplitudes": [max(1, round(n * .16)), max(2, round(n * .24)), max(2, round(n * .32))],
            "guide_weight": .34,
            "direction_reward": .16,
            "direction_penalty": .20,
            "leap_recovery_threshold": max(3, round(n * .45)),
            "leap_recovery_reward": .45,
            "same_direction_large_leap_penalty": .34,
            "centre_jitter_degrees": max(1, round(n * .12)),
        }
        user = dict(self.melody or {})
        for k, v in user.items():
            if k in ("phrase_shapes", "bar_shapes") and isinstance(v, dict):
                merged = dict(defaults[k]); merged.update({str(a): float(b) for a, b in v.items()})
                defaults[k] = merged
            else:
                defaults[k] = v
        return defaults

    def resolved_ngram_path(self) -> Path | None:
        raw = self.ngram_file
        if raw is None or not str(raw).strip():
            return None
        p = Path(str(raw)).expanduser()
        if not p.is_absolute() and self.style_dir:
            p = Path(self.style_dir) / p
        return p.resolve()

    def resolved_harmony(self) -> dict[str, Any]:
        """Return and validate functional pitch-role/stability partitions.

        The two partitions are orthogonal: a pitch can simultaneously be, for
        example, ``dominant`` and ``stable``.  Within each partition, however,
        every scale pitch class must occur exactly once.
        """
        pcs = tuple(int(x) for x in self.pcs)
        pcset = frozenset(pcs)
        user = dict(self.harmony or {})
        roles_user = dict(user.get("pitch_roles") or {})
        stab_user = dict(user.get("stability") or {})

        def nearest(target_steps: float, banned=()):
            banned = set(int(x) for x in banned)
            rows = [pc for pc in pcs if pc not in banned]
            if not rows:
                return None
            return min(rows, key=lambda pc: min((pc-target_steps) % self.edo, (target_steps-pc) % self.edo))

        tonic_default = 0
        dom_default = nearest(self.edo * math.log2(3/2), {tonic_default})
        sub_default = nearest(self.edo * math.log2(4/3), {tonic_default, dom_default})
        roles = {
            "tonic": tuple(int(x) for x in roles_user.get("tonic", (tonic_default,))),
            "dominant": tuple(int(x) for x in roles_user.get("dominant", (() if dom_default is None else (dom_default,)))),
            "subdominant": tuple(int(x) for x in roles_user.get("subdominant", (() if sub_default is None else (sub_default,)))),
        }
        roles["other"] = tuple(int(x) for x in roles_user.get(
            "other", tuple(pc for pc in pcs if pc not in set(roles["tonic"] + roles["dominant"] + roles["subdominant"]))))

        if len(roles["tonic"]) != 1:
            raise ValueError("harmony.pitch_roles.tonic must contain exactly one pitch class")
        role_flat = [pc for key in ("tonic", "dominant", "subdominant", "other") for pc in roles[key]]
        if frozenset(role_flat) != pcset or len(role_flat) != len(set(role_flat)):
            raise ValueError("harmony.pitch_roles must partition every scale pitch class exactly once")

        if stab_user:
            stable = tuple(int(x) for x in stab_user.get("stable", ()))
            unstable = tuple(int(x) for x in stab_user.get("unstable", ()))
        else:
            # Conventional auto-default: tonic, dominant, and the closest 5/4
            # scale pitch are stable; everything else is unstable.
            mediant = nearest(self.edo * math.log2(5/4), ())
            stable_set = {roles["tonic"][0]}
            stable_set.update(roles["dominant"])
            if mediant is not None:
                stable_set.add(mediant)
            stable = tuple(pc for pc in pcs if pc in stable_set)
            unstable = tuple(pc for pc in pcs if pc not in stable_set)
        stab_flat = list(stable) + list(unstable)
        if frozenset(stab_flat) != pcset or len(stab_flat) != len(set(stab_flat)):
            raise ValueError("harmony.stability must partition every scale pitch class exactly once")

        return {
            "pitch_roles": {k: tuple(roles[k]) for k in ("tonic", "dominant", "subdominant", "other")},
            "stability": {"stable": tuple(stable), "unstable": tuple(unstable)},
        }

    @property
    def signature(self) -> str:
        payload = {
            "edo": self.edo,
            "pcs": self.pcs,
            "names": self.names,
            "base_freq_hz": self.base_freq_hz,
            "voice_ranges": self.resolved_voice_ranges(),
            # CSE data depends on tuning + pitch domains only.  Musical rules
            # (wolves, cleanup, melodic contour) may be edited without forcing
            # an expensive CSE rebuild.
            "cse_wide_bounds": self.resolved_cse_wide_bounds(),
            "cse_inner_bounds": self.resolved_cse_inner_bounds(),
            "cse_five_bounds": self.resolved_cse_five_bounds(),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def resolved_voice_ranges(self) -> dict[str, tuple[int, int]]:
        e = self.edo
        # Generic defaults preserve the original four-part register geometry:
        # Bass has a genuinely low band, Inner starts about one octave below
        # the reference, Counter starts clearly above Inner's floor, and Lead
        # has the broadest upper register.  Keeping Counter's lower bound above
        # Inner's lower bound is important because Bass generation explicitly
        # leaves a future slot for Inner.
        defaults = {
            "bass": (-round(1.80 * e), -max(1, round(0.68 * e))),
            "inner": (-e, round(1.32 * e)),
            "counter": (-round(0.58 * e), round(2.10 * e)),
            # Keep the Lead floor safely above Counter's default floor.  The
            # extra gap (rather than merely +1 EDO step) leaves room after the
            # scale's second/wolf intervals are removed.
            "lead": (-round(0.30 * e), round(2.22 * e)),
        }
        out = dict(defaults)
        for voice, bounds in (self.voice_ranges or {}).items():
            if len(bounds) != 2:
                raise ValueError(f"voice_ranges[{voice!r}] must be [lo, hi]")
            lo, hi = map(int, bounds)
            if lo > hi:
                raise ValueError(f"voice_ranges[{voice!r}] has lo > hi")
            out[str(voice)] = (lo, hi)
        return out

    def resolved_ensemble_voices(self) -> tuple[str, ...]:
        # Five-part configurations opt in merely by declaring an inner2 range;
        # old four-part JSON remains byte-for-byte compatible.
        if 'inner2' in (self.voice_ranges or {}):
            return ("bass", "inner", "inner2", "counter", "lead")
        return ("bass", "inner", "counter", "lead")

    def resolved_cse_wide_bounds(self) -> tuple[int, int]:
        if self.cse_wide_bounds is not None:
            return tuple(map(int, self.cse_wide_bounds))
        ranges = self.resolved_voice_ranges()
        lo = min(v[0] for v in ranges.values())
        hi = max(v[1] for v in ranges.values())
        return min(lo, -3 * self.edo), max(hi, 3 * self.edo)

    def resolved_cse_inner_bounds(self) -> tuple[int, int]:
        if self.cse_inner_bounds is not None:
            return tuple(map(int, self.cse_inner_bounds))
        ranges = self.resolved_voice_ranges()
        return ranges["inner"][0], max(ranges["inner"][1], ranges["lead"][1])

    def resolved_cse_five_bounds(self) -> tuple[int, int]:
        """Absolute domain for true five-note sounding/background fields."""
        if self.cse_five_bounds is not None:
            return tuple(map(int, self.cse_five_bounds))
        return (-round(2.2 * self.edo), round(2.2 * self.edo))

    def degree_pitch(self, degree: int) -> int:
        o, i = divmod(int(degree), self.note_count)
        return o * self.edo + self.pcs[i]

    def degree(self, step: int) -> int:
        o, pc = divmod(int(step), self.edo)
        try:
            i = self.pc_index[pc]
        except ValueError as exc:
            raise ValueError(f"step {step} is not a pitch of scale {self.id}") from exc
        except KeyError as exc:
            raise ValueError(f"step {step} is not a pitch of scale {self.id}") from exc
        return o * len(self.pcs) + i

    def is_scale_pitch(self, step: int) -> bool:
        return int(step) % self.edo in self.pc_index

    def freq(self, step: int) -> float:
        return float(self.base_freq_hz) * 2.0 ** (float(step) / self.edo)

    def pitch_name(self, step: int) -> str:
        o, pc = divmod(int(step), self.edo)
        name = self.pc_name.get(pc, str(step))
        if o == 0:
            return name
        if o == 1:
            return "高" + name
        if o == -1:
            return "低" + name
        return (f"高{o}" if o > 1 else f"低{-o}") + name

    def make_pool(self, lo: int, hi: int) -> tuple[int, ...]:
        lo, hi = int(lo), int(hi)
        o0 = math.floor(lo / self.edo) - 1
        o1 = math.ceil(hi / self.edo) + 1
        return tuple(
            o * self.edo + pc
            for o in range(o0, o1 + 1)
            for pc in self.pcs
            if lo <= o * self.edo + pc <= hi
        )

    def tuning_cents(self, include_octave: bool = True) -> tuple[float, ...]:
        vals = tuple(1200.0 * pc / self.edo for pc in self.pcs)
        return vals + ((1200.0,) if include_octave else ())

    def resolved_lead_pc_targets(self) -> dict[int, float]:
        weights=self.style.get('lead_pc_target_distribution',{})
        if not weights:
            return {}
        parsed={int(k):float(v) for k,v in weights.items()}
        total=math.fsum(parsed.values())
        if not math.isfinite(total) or total<=0:
            raise ValueError('lead_pc_target_distribution must have finite positive total weight')
        return {pc:parsed.get(pc,0.0)/total for pc in self.pcs}

    def definition_dict(self) -> dict[str, Any]:
        return {'format': 'ScaleDefinition/1', 'id': self.id, 'name': self.name,
                'edo': self.edo, 'pcs': list(self.pcs), 'names': list(self.names),
                'base_note': self.base_note, 'base_freq_hz': self.base_freq_hz}

    def to_json_dict(self) -> dict[str, Any]:
        """Portable content snapshot for results; origins remain runtime-only."""
        return {'format': 'ScaleConfiguration/1', 'definition': self.definition_dict(),
                'rules': copy.deepcopy(self.rules), 'style': copy.deepcopy(self.style)}

    def runtime_snapshot(self) -> dict[str, Any]:
        """Self-contained worker snapshot with auxiliary paths already resolved."""
        snapshot = self.to_json_dict()
        snapshot['style']['ngram_file'] = str(self.resolved_ngram_path()) if self.ngram_file else None
        return snapshot


def _load_flat(data: dict, config_dir=None, style_dir=None, rules=None, style=None) -> ScaleSpec:
    edo = int(data["edo"])
    pcs = tuple(int(x) for x in data["pcs"])
    names = tuple(str(x) for x in data.get("names") or [str(i + 1) for i in range(len(pcs))])
    voice_ranges = {str(k): tuple(map(int, v)) for k, v in (data.get("voice_ranges") or {}).items()}
    wide = data.get("cse_wide_bounds")
    inner = data.get("cse_inner_bounds")
    five = data.get("cse_five_bounds")

    # Accept both the simple top-level fields and a nested cleanup object.
    cleanup = dict(data.get("cleanup") or {})
    degree_cleanup = data.get("cleanup_degree_distances", cleanup.get("degree_distances", ()))
    step_cleanup = data.get("cleanup_step_distances", cleanup.get("step_distances", ()))
    cleanup_passes = data.get("cleanup_passes", cleanup.get("passes", 8))
    cleanup_shifts = data.get("cleanup_max_shift_degrees", cleanup.get("max_shift_degrees", {}))
    cleanup_soft_penalty = data.get("cleanup_soft_penalty", cleanup.get("soft_penalty", 0.72))

    # "wolf_intervals" is accepted as a clearer alias, but old
    # forbidden_intervals configs continue to work.
    wolves = data.get("wolf_intervals", data.get("forbidden_intervals", ()))

    return ScaleSpec(
        id=str(data.get("id") or f"edo{edo}_{'-'.join(map(str, pcs))}"),
        name=str(data.get("name") or data.get("id") or f"{edo}-EDO subset"),
        edo=edo,
        pcs=pcs,
        names=names,
        base_freq_hz=float(data["base_freq_hz"]),
        base_note=str(data["base_note"]),
        voice_ranges=voice_ranges,
        forbidden_intervals=_tuple_ints(wolves),
        absolute_wolf_intervals=tuple(data.get("wolf_absolute_intervals", ())),
        cleanup_degree_distances=_tuple_ints(degree_cleanup),
        cleanup_step_distances=_tuple_ints(step_cleanup),
        cleanup_passes=int(cleanup_passes),
        cleanup_max_shift_degrees={str(k): int(v) for k, v in (cleanup_shifts or {}).items()},
        cleanup_soft_penalty=float(cleanup_soft_penalty),
        melody=dict(data.get("melody") or {}),
        ngram_file=(None if data.get("ngram_file") is None else str(data.get("ngram_file"))),
        harmony=dict(data.get("harmony") or {}),
        config_dir=config_dir,
        cse_wide_bounds=(tuple(map(int, wide)) if wide is not None else None),
        cse_inner_bounds=(tuple(map(int, inner)) if inner is not None else None),
        cse_five_bounds=(tuple(map(int, five)) if five is not None else None),
        metadata=dict(data.get("metadata") or {}),
        rules=copy.deepcopy(rules or {}), style=copy.deepcopy(style or {}), style_dir=style_dir,
    )


def _read_json(source, expected_format):
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        origin = str(path.parent)
    elif isinstance(source, dict):
        data, origin = copy.deepcopy(source), None
    else:
        raise TypeError(f'{expected_format} must be a JSON path or object')
    if not isinstance(data, dict) or data.get('format') != expected_format:
        raise ValueError(f'Expected {expected_format}, got {data.get("format") if isinstance(data, dict) else type(data).__name__}')
    return data, origin


def load_scale(path_or_dict, *, rules=None, style=None, require_composition=False) -> ScaleSpec:
    """Load a definition and sibling .rules/.style JSONs, or explicit profiles.

    A definition contains no composition references. Relative auxiliary files
    belong to the profile that declares them. Worker snapshots are content-based.
    """
    if isinstance(path_or_dict, ScaleSpec):
        if rules is not None or style is not None:
            return load_scale(path_or_dict.definition_dict(), rules=rules if rules is not None else path_or_dict.rules,
                              style=style if style is not None else path_or_dict.runtime_snapshot()['style'])
        return path_or_dict
    if isinstance(path_or_dict, dict) and path_or_dict.get('format') == 'ScaleConfiguration/1':
        return load_scale(path_or_dict['definition'], rules=path_or_dict['rules'] if rules is None else rules,
                          style=path_or_dict['style'] if style is None else style)
    definition, config_dir = _read_json(path_or_dict, 'ScaleDefinition/1')
    unknown = set(definition) - DEFINITION_FIELDS
    missing = DEFINITION_FIELDS - set(definition)
    if unknown or missing:
        raise ValueError(f'ScaleDefinition fields: unknown={sorted(unknown)}, missing={sorted(missing)}')
    sid = definition['id']
    if config_dir:
        path = Path(path_or_dict)
        stem = path.stem.removesuffix('.scale')
        if rules is None:
            sibling = Path(config_dir) / f'{stem}.rules.json'
            if require_composition or sibling.is_file(): rules = sibling
        if style is None:
            sibling = Path(config_dir) / f'{stem}.style.json'
            if require_composition or sibling.is_file(): style = sibling
    rule_data, rule_dir = _read_json(rules or {'format':'CompositionRules/1','scale_id':sid}, 'CompositionRules/1')
    style_data, style_dir = _read_json(style or {'format':'CompositionStyle/1','scale_id':sid}, 'CompositionStyle/1')
    allowed_rules = {'format','scale_id','voice_ranges','wolf_intervals','forbidden_intervals','wolf_absolute_intervals',
                     'cleanup','cleanup_degree_distances','cleanup_step_distances','cleanup_passes',
                     'cleanup_max_shift_degrees','cleanup_soft_penalty','melody','harmony',
                     'cse_wide_bounds','cse_inner_bounds','cse_five_bounds','metadata','runtime','voicebank','se_parameters','spectral_parameters'}
    for data, allowed in ((rule_data, allowed_rules),(style_data, STYLE_FIELDS)):
        if data.get('scale_id') != sid:
            raise ValueError(f'{data["format"]}.scale_id must match {sid!r}')
        if set(data) - allowed:
            raise ValueError(f'Unknown {data["format"]} fields: {sorted(set(data)-allowed)}')
    _strip_removed_policies(rule_data)
    generator = rule_data.get('metadata', {}).get('generator', {})
    misplaced = set(generator) & {'prime_rewards','lead_pc_target_distribution','ngram_file'}
    if misplaced:
        raise ValueError(f'Style fields found in rules.metadata.generator: {sorted(misplaced)}')
    for key,default in (('lead_pc_target_prior',18.0),('lead_pc_target_gain',4.2),('lead_pc_count_gain',.66)):
        value=float(generator.get(key,default))
        if not math.isfinite(value) or value<0 or (key=='lead_pc_target_prior' and value==0):
            raise ValueError(f'{key} must be finite and '+('positive' if key=='lead_pc_target_prior' else 'nonnegative'))
    pcset = set(definition['pcs'])
    style_data['voice_pc_rewards']=resolved_voice_pc_rewards(style_data,pcset)
    rewards = style_data.get('prime_rewards', {})
    if set(map(str,rewards)) - {'2','3','5','7','11','13','17'}:
        raise ValueError('prime_rewards supports only 2,3,5,7,11,13,17')
    if any(not math.isfinite(float(v)) for v in rewards.values()):
        raise ValueError('prime_rewards must be finite')
    targets = style_data.get('lead_pc_target_distribution', {})
    if any(int(k) not in pcset or not math.isfinite(float(v)) or float(v)<0 for k,v in targets.items()):
        raise ValueError('lead_pc_target_distribution requires scale pitch classes and finite nonnegative weights')
    if len({int(k) for k in targets}) != len(targets):
        raise ValueError('Duplicate numeric keys in lead_pc_target_distribution')
    if targets and (not math.isfinite(sum(map(float,targets.values()))) or sum(map(float, targets.values())) <= 0):
        raise ValueError('lead_pc_target_distribution must have positive total weight')
    resolved_cse_weights(style_data)
    from annealing_config import resolve_annealing
    resolve_annealing(style_data.get("annealing"))
    from melody_plan import resolve_melody_plan
    resolve_melody_plan(style_data.get("melody_plan"))
    attack_limits = resolved_attack_cse_percentile_limits(style_data)
    raw_ngram = style_data.get('ngram_file')
    if raw_ngram is not None and not isinstance(raw_ngram, str):
        raise ValueError('ngram_file must be a filename or null')
    flat = {**definition, **{k:v for k,v in rule_data.items() if k not in ('format','scale_id')}}
    flat['ngram_file'] = raw_ngram
    runtime = rule_data.get('runtime', {})
    from spectral_model import resolve_parameters
    resolve_parameters(rule_data.get('spectral_parameters'))
    if 'se_parameters' in rule_data:
        from se_model import resolve_parameters as legacy_se_parameters
        legacy_se_parameters(rule_data['se_parameters'])
        warnings.warn('rules.se_parameters is retired and ignored by the unified model; '
                      'use rules.spectral_parameters (fixed sigma_hz, default 1.0)',
                      UserWarning, stacklevel=2)
    runtime_keys = {'COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS'}
    if set(runtime) & {'CSE_2D_A','CSE_2D_B','CSE_2D_C'}:
        raise ValueError('Move CSE_2D_A/B/C from rules.runtime to style.cse_weights')
    if set(runtime)-runtime_keys: raise ValueError(f'Unknown runtime rules: {sorted(set(runtime)-runtime_keys)}')
    if any(not math.isfinite(float(v)) for v in runtime.values()): raise ValueError('Runtime values must be finite')
    count=runtime.get('COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS',1)
    if float(count) != int(count) or int(count)<0: raise ValueError('Maximum octave-family pairs must be a nonnegative integer')
    style_data['prime_rewards']={str(k):float(v) for k,v in rewards.items()}
    style_data['lead_pc_target_distribution']={str(k):float(v) for k,v in targets.items()}
    return _load_flat(flat, config_dir, style_dir or config_dir, rule_data, style_data)


def tuning_text(spec: ScaleSpec) -> str:
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 5e-10:
            return f"{int(round(x))}c"
        return f"{x:.5f}".rstrip("0").rstrip(".") + "c"

    head = f"{spec.base_note}: {spec.base_freq_hz:g}"
    cents = " ".join(fmt(x) for x in spec.tuning_cents(include_octave=True))
    return head + "\n" + cents + "\n"


def write_tuning_file_if_missing(spec: ScaleSpec, directory: str | Path = ".", filename: str | None = None) -> tuple[Path, bool]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def safe_display_name(text):
        text = re.sub(r'[<>:"/\\|?*]+', "_", str(text)).strip(" .")
        return text or spec.safe_id

    path = directory / (filename or f"{safe_display_name(spec.name)}.txt")
    if path.exists():
        return path, False
    path.write_text(tuning_text(spec), encoding="utf-8")
    return path, True
