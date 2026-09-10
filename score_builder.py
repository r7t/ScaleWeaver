"""ScaleWeaver score builder: melodic/rhythmic search and four-line counterpoint.

CSE lookup and CSE-derived scoring live in ``adaptive_cse_runtime.py``.
The Lead keeps n-grams, contour, motif recall and hard jump rules; bar-sequence
and chord-arpeggio devices were intentionally removed in this compact revision.
After the first pass, fixed-rhythm iterative regeneration repeatedly resamples
all four pitch lines against the other voices' realised vertical context.
"""
from __future__ import annotations
import math
import random
from functools import lru_cache
from math import gcd

import harmony_rhythm as hr
import adaptive_cse_runtime as cse_rt
from ngram_table import load_ngram_table, EMPTY_NGRAM_TABLE

# ---------------------------------------------------------------------------
# User-facing prime-colour rewards.
#
# Positive = reward (lower candidate cost), negative = penalty.  These are the
# ONLY knobs normal score generation reads.  They are passed explicitly into
# cse_rt, so editing this score_builder cannot accidentally target an inactive
# runtime module under another filename.
# ---------------------------------------------------------------------------
PRIME_2_REWARD = 0.0
PRIME_3_REWARD = 0
PRIME_5_REWARD = 0
PRIME_7_REWARD = 0
PRIME_11_REWARD = 0.0
PRIME_13_REWARD = 0.0
PRIME_17_REWARD = 0.0
PRIME_REWARD_DEFAULTS = {2:0.0,3:0.0,5:0.0,7:0.0,11:0.0,13:0.0,17:0.0}

# ---------------------------------------------------------------------------
# Iterative fixed-rhythm pitch regeneration.
#
# Pass 0 creates the rhythm/timing skeleton exactly once.  Every later pass
# keeps every start/duration unchanged and regenerates pitches only.
#
# A full iteration is a Gauss-Seidel-style sweep:
#     Lead -> Counter -> Bass -> Inner
# Earlier voices in the same sweep are immediately visible to later voices;
# not-yet-regenerated voices remain available from the preceding iteration.
# Thus every regenerated voice always sees the other three sounding lines.
# ---------------------------------------------------------------------------
PITCH_REGEN_ITERATIONS = 0
PITCH_REGEN_MAX_RETRIES = 4
PITCH_REGEN_LEAD_ACTUAL_CSE_WEIGHT = 1.0
PITCH_REGEN_GUIDE_JITTER_DEGREES = 1

def _active_prime_rewards():
    return {
        2: float(PRIME_2_REWARD), 3: float(PRIME_3_REWARD),
        5: float(PRIME_5_REWARD), 7: float(PRIME_7_REWARD),
        11: float(PRIME_11_REWARD), 13: float(PRIME_13_REWARD),
        17: float(PRIME_17_REWARD),
    }

class GenerationRejected(RuntimeError):
    """Expected candidate-level failure: abandon this seed without treating it as a program bug."""

from harmony_rhythm import (
    BASE_FREQ, CHORDS, HARD_WOLF, NAMES, OCT, PALETTE_COLOR_FAMILIES, PCS,
    POOLS, SECOND_MAX_STEP, VELOCITY, degree,
    degree_pitch, freq, harmony_plan, harmony_segment_at, is_strong_beat,
    make_rhythm_palette, normalize_time_signature, pitch_name, rhythm_for_bar,
    rhythm_palette_metadata, validate_rhythm_bar,
)

absolute_cse_entry = cse_rt.absolute_cse_entry
absolute_vertical_cse_interval = cse_rt.absolute_vertical_cse_interval
attack_aware_vertical_cse_interval = cse_rt.attack_aware_vertical_cse_interval
lead_chord_field_cse = cse_rt.lead_chord_field_cse
four_part_relative_purity_summary = cse_rt.four_part_relative_purity_summary
chord_cse_score = cse_rt.chord_cse_score
vertical_cse_score = cse_rt.vertical_cse_score
COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT = cse_rt.COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT
COUNTERPOINT_RELATIVE_PURITY_WEIGHT = cse_rt.COUNTERPOINT_RELATIVE_PURITY_WEIGHT
COUNTERPOINT_EFFECTIVE_RANGES = cse_rt.COUNTERPOINT_EFFECTIVE_RANGES

LEAD_SECTION_BASE_CENTRES = {}
LEAD_SHAPE_KINDS = ('arch', 'rise', 'fall', 'valley')
LEAD_PHRASE_SHAPE_WEIGHTS = (.46, .22, .22, .10)
LEAD_BAR_SHAPE_WEIGHTS = (.42, .25, .25, .08)
LEAD_PHRASE_AMPLITUDES = (2, 3, 3, 4)
LEAD_BAR_AMPLITUDES = (2, 2, 3, 3)
LEAD_CENTRE_JITTER_DEGREES = 1
MOTIF_DEGREE_ORIGIN = 0
RATIO_SMOOTH_PRIMES = (2, 3, 5, 7, 11)
CLEANUP_DEGREE_DISTANCES = ()
CLEANUP_STEP_DISTANCES = ()
CLEANUP_SOFT_PENALTY = 0.0
SCALE_NAME = 'scale'
SCALE_ID = 'scale'
BASE_NOTE = ''

def configure_scale(spec):
    """Rebind scale-dependent globals without changing generation call order."""
    global BASE_FREQ, CHORDS, HARD_WOLF, NAMES, OCT, PALETTE_COLOR_FAMILIES, PCS, POOLS, SECOND_MAX_STEP, VELOCITY
    global LEAD_MAX_JUMP_DEGREES, LEAD_MAX_ADJACENT_DEGREES, LEAD_DISTANCE_WEIGHTS
    global LEAD_PC_TARGETS, LEAD_PC_TARGET_PRIOR, LEAD_PC_TARGET_GAIN, LEAD_PC_COUNT_GAIN
    global COUNTERPOINT_MAX_JUMP_DEGREES, COUNTERPOINT_ROLE_CENTRE_DEGREE
    global COUNTERPOINT_EFFECTIVE_RANGES, LEAD_SECTION_BASE_CENTRES, MOTIF_DEGREE_ORIGIN, RATIO_SMOOTH_PRIMES, _NGRAM_TABLE
    global LEAD_PHRASE_SHAPE_WEIGHTS, LEAD_BAR_SHAPE_WEIGHTS
    global LEAD_PHRASE_AMPLITUDES, LEAD_BAR_AMPLITUDES, LEAD_CENTRE_JITTER_DEGREES
    global LEAD_NORMAL_FLOW_MAX, LEAD_MEDIUM_LEAP_MAX
    global COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT, COUNTERPOINT_RELATIVE_PURITY_WEIGHT
    global LEAD_STRONG_RECOVERY_THRESHOLD, LEAD_PREFERRED_MAX_LEAP, LEAD_SIXTEENTH_FLOW_BONUS
    global LEAD_LEAP_TARGET_RATE, LEAD_LEAP_MIN_DEGREES, LEAD_LEAP_BALANCE_GAIN
    global LEAD_LARGE_LEAP_TARGET_RATE, LEAD_LARGE_LEAP_MIN_DEGREES
    global LEAD_LARGE_LEAP_BALANCE_GAIN, LEAD_LEAP_BALANCE_WINDOW
    global LEAD_LEAP_BALANCE_PRIOR, LEAD_STRUCTURAL_LEAP_MULTIPLIER
    global LEAD_SHORT_NOTE_LEAP_MULTIPLIER
    global CLEANUP_DEGREE_DISTANCES, CLEANUP_STEP_DISTANCES, CLEANUP_SOFT_PENALTY
    global SCALE_NAME, SCALE_ID, BASE_NOTE
    global PRIME_2_REWARD, PRIME_3_REWARD, PRIME_5_REWARD, PRIME_7_REWARD, PRIME_11_REWARD, PRIME_13_REWARD, PRIME_17_REWARD
    global LEAD_REGISTER_TARGET_STEP, LEAD_REGISTER_HALF_SPAN, LEAD_REGISTER_POINT_WEIGHT
    global LEAD_REGISTER_HIGH_MULTIPLIER, LEAD_REGISTER_LOW_MULTIPLIER, LEAD_REGISTER_POWER
    global LEAD_REGISTER_CENTROID_FEEDBACK, LEAD_REGISTER_CENTROID_WINDOW
    BASE_FREQ=float(hr.BASE_FREQ); CHORDS=hr.CHORDS; HARD_WOLF=hr.HARD_WOLF; NAMES=hr.NAMES; OCT=hr.OCT
    PALETTE_COLOR_FAMILIES=hr.PALETTE_COLOR_FAMILIES; PCS=hr.PCS; POOLS=hr.POOLS; SECOND_MAX_STEP=hr.SECOND_MAX_STEP; VELOCITY=hr.VELOCITY
    meta=(spec.metadata or {}).get('generator', {})
    LEAD_MAX_JUMP_DEGREES=int(meta.get('lead_max_jump_degrees', spec.note_count)); LEAD_MAX_ADJACENT_DEGREES=LEAD_MAX_JUMP_DEGREES
    motion = meta.get('lead_motion', {})
    dw = meta.get('lead_distance_weights')
    LEAD_DISTANCE_WEIGHTS = ({int(k):float(v) for k,v in dw.items()} if dw else
                            _scaled_lead_distance_weights(LEAD_MAX_JUMP_DEGREES, spec.note_count, motion))
    if set(LEAD_DISTANCE_WEIGHTS) != set(range(LEAD_MAX_JUMP_DEGREES+1)) or any(not math.isfinite(v) or v <= 0 for v in LEAD_DISTANCE_WEIGHTS.values()):
        raise ValueError('lead_distance_weights must give a positive weight for every distance 0..lead_max_jump_degrees')
    def threshold(key, default):
        return max(1, int(math.floor(float(motion.get(key,default))*spec.note_count+.5)))
    LEAD_NORMAL_FLOW_MAX=threshold('normal_flow_octaves', .3)
    LEAD_MEDIUM_LEAP_MAX=max(LEAD_NORMAL_FLOW_MAX,threshold('medium_leap_octaves',.5))
    LEAD_STRONG_RECOVERY_THRESHOLD=max(LEAD_NORMAL_FLOW_MAX,threshold('strong_recovery_octaves',.5))
    LEAD_PREFERRED_MAX_LEAP=max(LEAD_NORMAL_FLOW_MAX,threshold('preferred_max_leap_octaves',.5))
    LEAD_SIXTEENTH_FLOW_BONUS=_scaled_sixteenth_flow_bonus(spec.note_count,motion)
    LEAD_LEAP_TARGET_RATE=float(motion.get('leap_target_rate',0.))
    LEAD_LEAP_MIN_DEGREES=threshold('leap_min_octaves',.4)
    LEAD_LEAP_BALANCE_GAIN=float(motion.get('leap_balance_gain',0.))
    LEAD_LARGE_LEAP_TARGET_RATE=float(motion.get('large_leap_target_rate',0.))
    LEAD_LARGE_LEAP_MIN_DEGREES=max(
        LEAD_LEAP_MIN_DEGREES,threshold('large_leap_min_octaves',.7))
    LEAD_LARGE_LEAP_BALANCE_GAIN=float(motion.get('large_leap_balance_gain',0.))
    LEAD_LEAP_BALANCE_WINDOW=max(1,int(motion.get('leap_balance_window',24)))
    LEAD_LEAP_BALANCE_PRIOR=max(0.,float(motion.get('leap_balance_prior',8.)))
    LEAD_STRUCTURAL_LEAP_MULTIPLIER=max(
        0.,float(motion.get('structural_leap_multiplier',1.25)))
    LEAD_SHORT_NOTE_LEAP_MULTIPLIER=max(
        0.,float(motion.get('short_note_leap_multiplier',.25)))
    if (not 0.<=LEAD_LEAP_TARGET_RATE<=1. or
            not 0.<=LEAD_LARGE_LEAP_TARGET_RATE<=LEAD_LEAP_TARGET_RATE or
            LEAD_LEAP_BALANCE_GAIN<0. or LEAD_LARGE_LEAP_BALANCE_GAIN<0. or
            not all(math.isfinite(x) for x in (
                LEAD_LEAP_TARGET_RATE,LEAD_LARGE_LEAP_TARGET_RATE,
                LEAD_LEAP_BALANCE_GAIN,LEAD_LARGE_LEAP_BALANCE_GAIN,
                LEAD_LEAP_BALANCE_PRIOR,LEAD_STRUCTURAL_LEAP_MULTIPLIER,
                LEAD_SHORT_NOTE_LEAP_MULTIPLIER))):
        raise ValueError('lead_motion leap-balance parameters are invalid')
    COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT=cse_rt.COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT
    COUNTERPOINT_RELATIVE_PURITY_WEIGHT=cse_rt.COUNTERPOINT_RELATIVE_PURITY_WEIGHT
    # One normalized target for the entire scale. Unlisted pitches have zero
    # soft target; an empty map disables feedback. No extra core or base bonus.
    LEAD_PC_TARGETS = spec.resolved_lead_pc_targets()
    LEAD_PC_TARGET_PRIOR=float(meta.get('lead_pc_target_prior',18.0))
    LEAD_PC_TARGET_GAIN=float(meta.get('lead_pc_target_gain',4.2))
    LEAD_PC_COUNT_GAIN=float(meta.get('lead_pc_count_gain',.66))
    default_jumps={'counter':max(1,spec.note_count-1),
                   'inner2':max(1,spec.note_count-2),
                   'inner':max(1,spec.note_count-2),
                   'bass':max(1,spec.note_count-3)}
    default_centres={'counter':2,
                     'inner2':-max(1,spec.note_count//4),
                     'inner':-max(1,spec.note_count//2),
                     'bass':-max(2,2*spec.note_count-3)}
    COUNTERPOINT_MAX_JUMP_DEGREES=default_jumps | {
        str(k):int(v) for k,v in meta.get('counterpoint_max_jump_degrees', {}).items()}
    COUNTERPOINT_ROLE_CENTRE_DEGREE=default_centres | {
        str(k):int(v) for k,v in meta.get('counterpoint_role_centre_degree', {}).items()}
    COUNTERPOINT_EFFECTIVE_RANGES=getattr(cse_rt,'COUNTERPOINT_EFFECTIVE_RANGES', {k:tuple(v) for k,v in hr.RANGES.items() if k in ('counter','inner','bass')})

    # Explicit section centres override centres derived from the active register.
    mel = spec.resolved_melody()
    configured_centres = meta.get('lead_section_base_centres')
    if configured_centres:
        LEAD_SECTION_BASE_CENTRES = {
            str(k): tuple(int(x) for x in v)
            for k, v in configured_centres.items()
        }
    else:
        lead_pool = list(POOLS.get('lead', ()))
        if not lead_pool:
            raise RuntimeError('Lead pool is empty while configuring Lead section centres')
        lead_lo, lead_hi = map(float, hr.RANGES['lead'])
        register_target = float(mel.get('register_target_step', 0.5 * (lead_lo + lead_hi)))
        centre_pitch = min(lead_pool, key=lambda p: abs(float(p) - register_target))
        centre_degree = int(spec.degree(int(centre_pitch)))
        lead_degrees = [int(spec.degree(int(p))) for p in lead_pool]
        degree_lo, degree_hi = min(lead_degrees), max(lead_degrees)
        spacing = max(1, int(round(float(spec.note_count) / 5.0)))

        def clamp_degree(x):
            return max(degree_lo, min(degree_hi, int(x)))

        LEAD_SECTION_BASE_CENTRES = {
            'A': tuple(clamp_degree(x) for x in (
                centre_degree - 2 * spacing,
                centre_degree - spacing,
                centre_degree,
                centre_degree + spacing,
            )),
            'B': tuple(clamp_degree(x) for x in (
                centre_degree - spacing,
                centre_degree,
                centre_degree + spacing,
                centre_degree + 2 * spacing,
            )),
        }

    def _shape_weights(config_key, defaults):
        raw = mel.get(config_key, {}) or {}
        vals = tuple(max(0.0, float(raw.get(k, d)))
                     for k, d in zip(LEAD_SHAPE_KINDS, defaults))
        return vals if sum(vals) > 0.0 else tuple(defaults)

    LEAD_PHRASE_SHAPE_WEIGHTS = _shape_weights(
        'phrase_shapes', (.46, .22, .22, .10))
    LEAD_BAR_SHAPE_WEIGHTS = _shape_weights(
        'bar_shapes', (.42, .25, .25, .08))

    phrase_amps = tuple(max(0, int(x)) for x in mel.get(
        'phrase_amplitudes', (2, 3, 3, 4)))
    bar_amps = tuple(max(0, int(x)) for x in mel.get(
        'bar_amplitudes', (2, 2, 3, 3)))
    LEAD_PHRASE_AMPLITUDES = phrase_amps or (2, 3, 3, 4)
    LEAD_BAR_AMPLITUDES = bar_amps or (2, 2, 3, 3)
    LEAD_CENTRE_JITTER_DEGREES = max(
        0, int(mel.get('centre_jitter_degrees', 1)))

    MOTIF_DEGREE_ORIGIN=int(meta.get('motif_degree_origin', 4*spec.note_count))
    RATIO_SMOOTH_PRIMES=tuple(int(x) for x in meta.get('ratio_smooth_primes',(2,3,5,7,11)))
    prime_rewards = spec.style.get('prime_rewards', {}) or {}
    PRIME_2_REWARD=float(prime_rewards.get('2', prime_rewards.get(2, PRIME_REWARD_DEFAULTS[2])))
    PRIME_3_REWARD=float(prime_rewards.get('3', prime_rewards.get(3, PRIME_REWARD_DEFAULTS[3])))
    PRIME_5_REWARD=float(prime_rewards.get('5', prime_rewards.get(5, PRIME_REWARD_DEFAULTS[5])))
    PRIME_7_REWARD=float(prime_rewards.get('7', prime_rewards.get(7, PRIME_REWARD_DEFAULTS[7])))
    PRIME_11_REWARD=float(prime_rewards.get('11', prime_rewards.get(11, PRIME_REWARD_DEFAULTS[11])))
    PRIME_13_REWARD=float(prime_rewards.get('13', prime_rewards.get(13, PRIME_REWARD_DEFAULTS[13])))
    PRIME_17_REWARD=float(prime_rewards.get('17', prime_rewards.get(17, PRIME_REWARD_DEFAULTS[17])))
    CLEANUP_DEGREE_DISTANCES=tuple(int(x) for x in spec.cleanup_degree_distances)
    CLEANUP_STEP_DISTANCES=tuple(int(x) for x in spec.cleanup_step_distances)
    CLEANUP_SOFT_PENALTY=float(spec.cleanup_soft_penalty)
    SCALE_NAME=str(spec.name); SCALE_ID=str(spec.id); BASE_NOTE=str(spec.base_note)

    # Lead register control.  The target is the physical midpoint of the configured
    # Lead range in EDO-step/log-frequency space, not the midpoint pitch class.
    # A static asymmetric edge potential counteracts CSE's tendency to open the
    # texture upward; a proportional rolling-centroid controller keeps the long-
    # term Lead mean centred instead of simply pushing the whole melody downward.
    lead_lo, lead_hi = map(float, hr.RANGES['lead'])
    LEAD_REGISTER_TARGET_STEP = float(mel.get('register_target_step', 0.5*(lead_lo+lead_hi)))
    LEAD_REGISTER_HALF_SPAN = max(1.0, 0.5*(lead_hi-lead_lo))
    LEAD_REGISTER_POINT_WEIGHT = float(mel.get('register_point_weight', 0.55))
    LEAD_REGISTER_HIGH_MULTIPLIER = float(mel.get('register_high_multiplier', 2.60))
    LEAD_REGISTER_LOW_MULTIPLIER = float(mel.get('register_low_multiplier', 0.65))
    LEAD_REGISTER_POWER = max(1.0, float(mel.get('register_power', 2.20)))
    LEAD_REGISTER_CENTROID_FEEDBACK = float(mel.get('register_centroid_feedback', 1.20))
    LEAD_REGISTER_CENTROID_WINDOW = max(1, int(mel.get('register_centroid_window', 32)))

    _NGRAM_TABLE=load_ngram_table(spec.resolved_ngram_path(), edo=spec.edo, allowed_pcs=spec.pcs)
    ratio_cost.cache_clear()
    _lead_interval_cost.cache_clear()

# Lead register potential.  Defaults are overwritten by configure_scale().
LEAD_REGISTER_TARGET_STEP = 0.0
LEAD_REGISTER_HALF_SPAN = 1.0
LEAD_REGISTER_POINT_WEIGHT = 0.55
LEAD_REGISTER_HIGH_MULTIPLIER = 2.60
LEAD_REGISTER_LOW_MULTIPLIER = 0.65
LEAD_REGISTER_POWER = 2.20
LEAD_REGISTER_CENTROID_FEEDBACK = 1.20
LEAD_REGISTER_CENTROID_WINDOW = 32

def _lead_register_cost(cand, recent, recent_mean_step=None):
    """Asymmetric edge penalty + rolling-centroid feedback for Lead.

    ``x`` is the candidate's signed position in the configured Lead range, with
    x=0 at the target centre and roughly +/-1 at the two range edges.  High
    notes are intentionally more expensive than equally distant low notes.

    The second term is a proportional controller: if the recent Lead centroid
    has drifted high, high candidates become more expensive and low candidates
    receive a small compensating reward; if it has drifted low, the sign reverses.
    Thus the *mean* is pulled toward the target centre rather than toward the
    lower half of the range despite the asymmetric point penalty.
    """
    span = max(1e-9, float(LEAD_REGISTER_HALF_SPAN))
    x = (float(cand) - float(LEAD_REGISTER_TARGET_STEP)) / span
    side = LEAD_REGISTER_HIGH_MULTIPLIER if x >= 0.0 else LEAD_REGISTER_LOW_MULTIPLIER
    point = float(LEAD_REGISTER_POINT_WEIGHT) * side * (abs(x) ** float(LEAD_REGISTER_POWER))

    if not recent or LEAD_REGISTER_CENTROID_FEEDBACK == 0.0:
        return point
    if recent_mean_step is None:
        window = recent[-int(LEAD_REGISTER_CENTROID_WINDOW):]
        mean_step = sum(float(p) for p in window) / len(window)
    else:
        mean_step = float(recent_mean_step)
    mean_err = (mean_step - float(LEAD_REGISTER_TARGET_STEP)) / span
    feedback = float(LEAD_REGISTER_CENTROID_FEEDBACK) * mean_err * x
    return point + feedback

LEAD_MAX_JUMP_DEGREES = 0
LEAD_MAX_ADJACENT_DEGREES = LEAD_MAX_JUMP_DEGREES
FORBIDDEN_ACCOMP_DURATIONS = ()

LEAD_PRIMARY_CHORD_REWARD = .42
LEAD_PRIMARY_NONCHORD_COST = .22
LEAD_INTEGER_CHORD_REWARD = .20
LEAD_INTEGER_NONCHORD_COST = .08
LEAD_LONG_CHORD_REWARD = .28
LEAD_LONG_NONCHORD_COST = .18
LEAD_SIXTEENTH_FLOW_BONUS = {1:.24, 2:.20, 3:.13}
LEAD_SIXTEENTH_LARGE_LEAP_COST = .22

LEAD_PC_TARGETS = {}
LEAD_PC_TARGET_PRIOR = 18.0
LEAD_PC_TARGET_GAIN = 4.2
LEAD_PC_COUNT_GAIN = .66
LEAD_LONG_FOOT_REWARD = .20
LEAD_BOUNDARY_FOOT_REWARD = .20

_NGRAM_TABLE = EMPTY_NGRAM_TABLE

def _lead_ngram_prior_cost(recent, cand):
    # A 2..5-gram can inspect at most the previous four symbols.  Converting
    # the complete melody prefix here made beam generation accidentally
    # quadratic in score length.
    pcs = tuple(int(x) % OCT for x in recent[-4:])
    return -float(_NGRAM_TABLE.accumulated_bias('lead', pcs, int(cand) % OCT, 2, 5))

def _bass_ngram_prior_cost(recent, cand):
    pcs = tuple(int(x) % OCT for x in recent[-4:])
    return -float(_NGRAM_TABLE.accumulated_bias('bass', pcs, int(cand) % OCT, 2, 5))

def lead_jump_degrees(a, b):
    """Absolute ScaleWeaver degree distance between two Lead pitches."""
    return abs(degree(int(b)) - degree(int(a)))

def lead_jump_ok(prev, cand, max_degrees=None):
    """Public hard guard: adjacent Lead notes may span at most one scale octave."""
    if max_degrees is None:
        max_degrees = LEAD_MAX_JUMP_DEGREES
    return prev is None or lead_jump_degrees(prev, cand) <= int(max_degrees)

def fit_lead_pitch_to_jump(cand, prev, target=None, preserve_pc=True):
    """Octave-fit a Lead candidate to the <= ten-distance rule when possible."""
    cand = int(cand)
    if lead_jump_ok(prev, cand):
        return cand
    legal = [p for p in POOLS['lead'] if lead_jump_ok(prev, p)]
    if not legal:
        return cand
    same_pc = [p for p in legal if p % OCT == cand % OCT]
    choices = same_pc if preserve_pc and same_pc else legal
    tgt = cand if target is None else int(target)
    return min(choices, key=lambda p: (abs(degree(p) - degree(tgt)), abs(p - tgt), abs(p - cand)))

def event(start, dur, p, voice, rng, harmony, **extra):
    base_voice = 'counter' if str(voice).startswith('counter') else voice
    e = {
        'start_beat': round(float(start), 6), 'duration_beats': round(float(dur), 6),
        'step': int(p), 'name': pitch_name(p), 'freq': freq(p),
        'velocity': round(VELOCITY[base_voice] * rng.uniform(.94, 1.05), 4),
        'phase': round(rng.uniform(0, 2 * math.pi), 6),
        'harmony': harmony['chord_id'], 'role': voice,
    }
    e.update(extra)
    return e

def nearest_pc(voice, pc, around, prev=None):
    xs = [p for p in POOLS[voice] if p % OCT == pc]
    if prev is not None:
        legal = [p for p in xs if lead_jump_ok(prev, p)]
        if legal:
            xs = legal
    return min(xs, key=lambda p: abs(p - around)) if xs else None

def _sample_low_cost(scored, rng, top=5, temperature=.72):
    """Compatibility sampler shared by Counter/other accompaniment code."""
    scored.sort()
    top_items = scored[:top]
    if not top_items:
        return None
    if len(top_items) == 1 or rng.random() < .52:
        return top_items[0][1]
    best = top_items[0][0]
    weights = [math.exp(-(s - best) / temperature) for s, _ in top_items]
    return rng.choices([pitch for _, pitch in top_items], weights=weights, k=1)[0]

def _scaled_lead_distance_weights(max_jump, note_count, motion):
    """An octave-fraction prior, evaluated on the active scale degree grid."""
    flat=float(motion.get('distance_flat_octaves', .2))
    decay=float(motion.get('distance_decay_per_octave',5.5))
    minimum=float(motion.get('distance_min_weight',.01))
    out={0:float(motion.get('repeated_note_weight',.3))}
    for d in range(1,max_jump+1):
        out[d]=max(minimum, math.exp(-decay*max(0.0,d/note_count-flat)))
    return out


def _scaled_sixteenth_flow_bonus(note_count, motion):
    profile=motion.get('sixteenth_flow_profile', [[.1,.24],[.2,.20],[.3,.13]])
    if not profile: return {}
    profile=sorted((float(x),float(y)) for x,y in profile)
    out={}
    for d in range(1,max(1,int(math.floor(profile[-1][0]*note_count+.5)))+1):
        x=d/note_count
        if x<=profile[0][0]: value=profile[0][1]
        elif x>profile[-1][0]: continue
        else:
            for (a,ya),(b,yb) in zip(profile,profile[1:]):
                if a<x<=b:
                    value=ya+(x-a)/(b-a)*(yb-ya);break
        if value>0:out[d]=value
    return out

LEAD_DISTANCE_WEIGHTS = {}
LEAD_NORMAL_FLOW_MAX = 0
LEAD_MEDIUM_LEAP_MAX = 0
LEAD_STRONG_RECOVERY_THRESHOLD = 0
LEAD_MEDIUM_RECOVERY_BONUS = .16
LEAD_LEAP_RECOVERY_BONUS = .62
LEAD_PREFERRED_MAX_LEAP = 0
LEAD_LEAP_TARGET_RATE = 0.0
LEAD_LEAP_MIN_DEGREES = 1
LEAD_LEAP_BALANCE_GAIN = 0.0
LEAD_LARGE_LEAP_TARGET_RATE = 0.0
LEAD_LARGE_LEAP_MIN_DEGREES = 1
LEAD_LARGE_LEAP_BALANCE_GAIN = 0.0
LEAD_LEAP_BALANCE_WINDOW = 24
LEAD_LEAP_BALANCE_PRIOR = 8.0
LEAD_STRUCTURAL_LEAP_MULTIPLIER = 1.25
LEAD_SHORT_NOTE_LEAP_MULTIPLIER = .25
LEAD_GUIDE_WEIGHT = .175
LEAD_PREDICTABLE_GREEDY = .70
LEAD_PREDICTABLE_TOP = 3
LEAD_PREDICTABLE_TEMPERATURE = .42
LEAD_MOTIF_REPEAT_RATE = .54
LEAD_SECTION_MOTIF_COUNT = 2
LEAD_SECTION_PRIMARY_REUSE_RATE = 1.00
LEAD_SECTION_SECONDARY_REUSE_RATE = .58
LEAD_SECTION_WEAK_VARIATION_RATE = .28
LEAD_SECTION_MACRO_REUSE_RATE = .72

LEAD_ANCHOR_BAR_RATE = .76
LEAD_ANCHOR_APPEARANCES = (3, 4, 5)
LEAD_ANCHOR_APPEARANCE_WEIGHTS = (.58, .30, .12)
LEAD_ANCHOR_WEAK_VARIATION_RATE = .18

LEAD_FANCY_RHYTHM_RATE = .16
LEAD_SYNCOPATED_RHYTHM_MULTIPLIER = .35
COUNTERPOINT_FANCY_RHYTHM_RATE = {'counter': .28, 'inner2': .25, 'inner': .22, 'bass': .05}
COUNTERPOINT_MODE_WEIGHTS = {'counter': 1.0}
COUNTERPOINT_MAX_JUMP_DEGREES = {}
COUNTERPOINT_NGRAM_SCALE = {'counter': .82, 'inner2': .77, 'inner': .72, 'bass': .62}
COUNTERPOINT_CONTRARY_BONUS = .12
COUNTERPOINT_PARALLEL_MOTION_COST = .15
COUNTERPOINT_REPEAT_COST = .52
COUNTERPOINT_REGISTER_COST = {'counter': .020, 'inner2': .022, 'inner': .024, 'bass': .020}
COUNTERPOINT_ROLE_CENTRE_DEGREE = {}

def _choose_melodic_rhythm(bar, rng, bpb, palette, *, cadence=False,
                            allow_sixteenth=False, fancy_rate=.45,
                            syncopation_multiplier=.5):
    """Choose a bar rhythm while keeping fancy non-division cells active.

    Sixteenths remain an explicit optional layer.  With them disabled, fancy
    dotted/asymmetric cells compete directly with the basic half-bar grammar,
    and the plain quarter-note pattern is slightly *downweighted* rather than
    treated as the default.
    """
    dense_rate = hr.LEAD_SIXTEENTH_CADENCE_RATE if cadence else hr.LEAD_SIXTEENTH_RATE
    if allow_sixteenth and palette.get('sixteenth') and rng.random() < dense_rate:
        return tuple(float(x) for x in rng.choice(palette['sixteenth'])[1])
    if cadence and palette.get('cadence') and rng.random() < .70:
        return tuple(float(x) for x in rng.choice(palette['cadence'])[1])
    if palette.get('fancy') and rng.random() < float(fancy_rate):
        return tuple(float(x) for x in rng.choice(palette['fancy'])[1])
    choices = palette.get('basic') or ()
    if not choices:
        return tuple(float(x) for x in rhythm_for_bar(bar, rng, bpb, palette))
    weights = []
    for name, durations in choices:
        weight = .72 if name.endswith('_2222') or name.endswith('_222') else 1.0
        weights.append(weight)
    weights = hr.apply_category_frequency_multiplier(
        weights,
        [hr.rhythm_is_syncopated(durations) for _name, durations in choices],
        syncopation_multiplier,
    )
    return tuple(float(x) for x in rng.choices(choices, weights=weights, k=1)[0][1])

def _lead_pc_frequency_cost(recent, pc, pc_counts=None):
    """Soft pitch-class feedback using only the configured normalized target."""
    if not LEAD_PC_TARGETS:
        return 0.0
    pc = int(pc) % OCT
    target = LEAD_PC_TARGETS.get(pc, 0.0)
    n = len(recent)
    prior = LEAD_PC_TARGET_PRIOR
    count = (sum(int(p) % OCT == pc for p in recent)
             if pc_counts is None else int(pc_counts.get(pc, 0)))
    deficit = target - (count + prior * target) / (n + prior)
    count_deficit = target * n - count
    return (-LEAD_PC_TARGET_GAIN * max(-.08, min(.08, deficit))
            -LEAD_PC_COUNT_GAIN * max(-.55, min(.55, count_deficit / math.sqrt(n + prior))))

def pitch_class_prior_metadata():
    return {'enabled': bool(LEAD_PC_TARGETS),
            'source': 'style.lead_pc_target_distribution',
            'pc_targets': {str(pc): target for pc,target in LEAD_PC_TARGETS.items()},
            'prior_notes': LEAD_PC_TARGET_PRIOR,
            'frequency_gain': LEAD_PC_TARGET_GAIN, 'count_gain': LEAD_PC_COUNT_GAIN,
            'principle': 'Normalized full-scale soft feedback; unlisted pitches have zero target, not a hard ban; no extra core/group or static pitch bonus.'}

@lru_cache(maxsize=128)
def _lead_interval_cost(ad):
    """Negative log of the ScaleWeaver-specific melodic-distance prior.

    1/2/3-distance motion is intentionally broad and nearly flat; 4/5-distance
    is a moderate leap, while 6+ falls off rapidly.
    """
    ad = max(0, min(LEAD_MAX_JUMP_DEGREES, int(ad)))
    return -math.log(max(1e-9, LEAD_DISTANCE_WEIGHTS[ad]))


def _lead_motion_diversity_cost(recent, cand, strong=False, duration=1.0):
    """Sliding soft targets for deliberate leaps in the fixed Lead grammar.

    The targets are disabled unless a scale opts in through ``lead_motion``.
    A deficit rewards a qualifying leap and gives non-leaps a much smaller
    balancing cost; an excess reverses those signs.  This preserves variation
    instead of turning every eligible onset into a jump.
    """
    if not recent or (LEAD_LEAP_TARGET_RATE<=0. and
                      LEAD_LARGE_LEAP_TARGET_RATE<=0.):
        return 0.0
    window=list(recent[-(LEAD_LEAP_BALANCE_WINDOW+1):])
    intervals=[abs(degree(b)-degree(a)) for a,b in zip(window,window[1:])]
    candidate_distance=abs(degree(int(cand))-degree(int(recent[-1])))

    def component(target,threshold,gain):
        if target<=0. or gain<=0.:
            return 0.0
        count=sum(distance>=threshold for distance in intervals)
        observed=(count+LEAD_LEAP_BALANCE_PRIOR*target)/max(
            1e-9,len(intervals)+LEAD_LEAP_BALANCE_PRIOR)
        deficit=target-observed
        if candidate_distance>=threshold:
            return -gain*deficit
        return gain*deficit*target/max(1e-9,1.-target)

    cost=(component(LEAD_LEAP_TARGET_RATE,LEAD_LEAP_MIN_DEGREES,
                    LEAD_LEAP_BALANCE_GAIN)+
          component(LEAD_LARGE_LEAP_TARGET_RATE,LEAD_LARGE_LEAP_MIN_DEGREES,
                    LEAD_LARGE_LEAP_BALANCE_GAIN))
    if strong:
        cost*=LEAD_STRUCTURAL_LEAP_MULTIPLIER
    if float(duration)<=.250001:
        cost*=LEAD_SHORT_NOTE_LEAP_MULTIPLIER
    return cost

def _sample_predictable(scored, rng):
    scored.sort(key=lambda x: x[0])
    xs = scored[:LEAD_PREDICTABLE_TOP]
    if not xs:
        return None
    if len(xs) == 1 or rng.random() < LEAD_PREDICTABLE_GREEDY:
        return xs[0][1]
    best = xs[0][0]
    weights = [math.exp(-(s - best) / LEAD_PREDICTABLE_TEMPERATURE) for s, _ in xs]
    return rng.choices([p for _, p in xs], weights=weights, k=1)[0]

def _lead_centre_jitter(rng):
    """Scale-aware section-centre jitter with a centre-biased distribution."""
    j = max(0, int(LEAD_CENTRE_JITTER_DEGREES))
    if j <= 0:
        return 0
    if j == 1:
        # Preserve the historical ScaleWeaver distribution exactly.
        return rng.choice((-1, 0, 0, 0, 1))
    values = tuple(range(-j, j + 1))
    weights = tuple((j + 1) - abs(x) for x in values)
    return int(rng.choices(values, weights=weights, k=1)[0])


def _lead_contour(kind, n, centre, amplitude):
    if n <= 1:
        return (int(round(centre)),)
    out = []
    for i in range(n):
        x = i / (n - 1)
        if kind == 'rise':
            y = -1 + 2*x
        elif kind == 'fall':
            y = 1 - 2*x
        elif kind == 'valley':
            y = 2*abs(x - .5) - 1
        else:                                        # arch
            y = 1 - 2*abs(x - .5)
        out.append(int(round(centre + amplitude*y)))
    return tuple(out)

def _lead_metric_class(beat_in_bar, beats_per_bar):
    if beat_in_bar is None:
        return 'offbeat'
    loc = round(float(beat_in_bar) % float(beats_per_bar), 6)
    if int(beats_per_bar) == 4:
        for x, name in ((0.0, 'beat1'), (1.0, 'beat2'), (2.0, 'beat3'), (3.0, 'beat4')):
            if abs(loc - x) < 1e-6:
                return name
        return 'offbeat'
    if abs(loc) < 1e-6:
        return 'beat1'
    return 'integer' if abs(loc - round(loc)) < 1e-6 else 'offbeat'

def _lead_landing_cost(pc, chord, beat_in_bar, duration, beats_per_bar):
    """Soft metrical/harmonic landing cost inferred from the two reference leads."""
    pc = int(pc) % OCT
    dur = max(.0, float(duration if duration is not None else 1.0))
    cls = _lead_metric_class(beat_in_bar, beats_per_bar)
    cost = 0.0

    if cls in ('beat1', 'beat3'):
        chord_reward = LEAD_PRIMARY_CHORD_REWARD
        nonchord_cost = LEAD_PRIMARY_NONCHORD_COST
    elif cls in ('beat2', 'beat4', 'integer'):
        chord_reward = LEAD_INTEGER_CHORD_REWARD
        nonchord_cost = LEAD_INTEGER_NONCHORD_COST
    else:
        chord_reward = nonchord_cost = 0.0

    if pc in chord.pcs:
        cost -= chord_reward
    else:
        cost += nonchord_cost

    longness = max(0.0, min(1.0, (dur - 1.0) / 1.0))
    if longness:
        if pc in chord.pcs:
            cost -= LEAD_LONG_CHORD_REWARD * longness
        else:
            cost += LEAD_LONG_NONCHORD_COST * longness
        if pc == chord.foot % OCT:
            cost -= LEAD_LONG_FOOT_REWARD * longness
    return cost

def lead_candidate_cost_components(prev, prev2, chord, strong, recent, cand,
                                    guide_degree, direction=0, boundary=False,
                                    beat_in_bar=None, duration=1.0,
                                    beats_per_bar=4):
    """Deterministic Lead scoring shared by candidate sampling and beam ranking.

    The returned values are *unweighted primitive costs*.  Callers decide how
    to aggregate them across a unit, but cannot accidentally use a different
    definition of interval, n-gram, landing, or background cost.
    """
    cand = int(cand)
    d, pc = degree(cand), cand % OCT
    centre = int(guide_degree)
    prev_d = degree(prev) if prev is not None else None
    old = degree(prev) - degree(prev2) if prev is not None and prev2 is not None else 0
    if prev is not None:
        dd = d - prev_d
        ad = abs(dd)
        if ad > LEAD_MAX_JUMP_DEGREES:
            return None
    else:
        dd = ad = 0
    pc_counts = {}
    for pitch in recent:
        prior_pc = int(pitch) % OCT
        pc_counts[prior_pc] = pc_counts.get(prior_pc, 0) + 1
    recent_mean_step = None
    if recent and LEAD_REGISTER_CENTROID_FEEDBACK != 0.0:
        window = recent[-int(LEAD_REGISTER_CENTROID_WINDOW):]
        recent_mean_step = sum(float(p) for p in window) / len(window)

    out = {
        "guide": LEAD_GUIDE_WEIGHT * abs(d - centre),
        "register": _lead_register_cost(cand, recent, recent_mean_step),
        "interval": 0.0,
        "leap_recovery": 0.0,
        "landing": _lead_landing_cost(pc, chord, beat_in_bar, duration, beats_per_bar),
        "sixteenth_flow": 0.0,
        "direction": 0.0,
        "ngram": _lead_ngram_prior_cost(recent, cand),
        "pc_frequency": _lead_pc_frequency_cost(recent, pc, pc_counts),
        "motion_diversity": _lead_motion_diversity_cost(recent, cand, strong, duration),
        "background": cse_rt.lead_chord_field_cost(chord, cand, strong),
        "boundary": 0.0,
    }
    if prev is not None:
        out["interval"] = _lead_interval_cost(ad)
        if ad > LEAD_PREFERRED_MAX_LEAP:
            out["interval"] += .42 * (ad - LEAD_PREFERRED_MAX_LEAP)
        old_ad = abs(old)
        if LEAD_NORMAL_FLOW_MAX < old_ad <= LEAD_MEDIUM_LEAP_MAX:
            if dd * old < 0 and 1 <= ad <= LEAD_NORMAL_FLOW_MAX:
                out["leap_recovery"] -= LEAD_MEDIUM_RECOVERY_BONUS
            elif dd * old > 0 and ad > LEAD_NORMAL_FLOW_MAX:
                out["leap_recovery"] += .12
        elif old_ad > LEAD_STRONG_RECOVERY_THRESHOLD:
            if dd * old < 0 and 1 <= ad <= LEAD_NORMAL_FLOW_MAX:
                out["leap_recovery"] -= LEAD_LEAP_RECOVERY_BONUS
            elif dd * old > 0 and ad >= LEAD_NORMAL_FLOW_MAX:
                out["leap_recovery"] += .40
        if float(duration) <= .250001:
            if ad in LEAD_SIXTEENTH_FLOW_BONUS:
                out["sixteenth_flow"] -= LEAD_SIXTEENTH_FLOW_BONUS[ad]
            elif ad > LEAD_NORMAL_FLOW_MAX:
                out["sixteenth_flow"] += LEAD_SIXTEENTH_LARGE_LEAP_COST * (ad - LEAD_NORMAL_FLOW_MAX)
        if direction and dd:
            out["direction"] = -.10 if dd * direction > 0 else .12
    if boundary and pc == chord.foot % OCT:
        out["boundary"] = -LEAD_BOUNDARY_FOOT_REWARD
    out["total"] = sum(out.values())
    return out


def choose_lead_pitch(prev, prev2, chord, strong, recent, rng, guide_degree,
                      direction=0, boundary=False, beat_in_bar=None,
                      duration=1.0, beats_per_bar=4):
    scored = []
    for cand in POOLS['lead']:
        components = lead_candidate_cost_components(
            prev, prev2, chord, strong, recent, cand, guide_degree, direction,
            boundary, beat_in_bar, duration, beats_per_bar)
        if components is not None:
            scored.append((components["total"] + rng.uniform(0.0, .055), cand))
    cand = _sample_predictable(scored, rng)
    if cand is not None:
        return cand
    return nearest_pc('lead', chord.foot % OCT, degree_pitch(int(guide_degree)), prev)

def user_motif_steps(motif_degrees):
    return [degree_pitch(int(x) - MOTIF_DEGREE_ORIGIN) for x in motif_degrees] if motif_degrees else []

def _fit_motif_pitch(raw, prev, guide_degree):
    pc = raw % OCT
    candidates = [p for p in POOLS['lead'] if p % OCT == pc]
    if prev is not None:
        legal = [p for p in candidates if lead_jump_ok(prev, p)]
        if legal:
            candidates = legal
    return min(candidates, key=lambda p: abs(degree(p) - int(guide_degree)) +
               (.25 * abs(degree(p) - degree(prev)) if prev is not None else 0))

def _phrase_direction(guides, i):
    if i + 1 < len(guides):
        x = guides[i + 1] - guides[i]
    elif i:
        x = guides[i] - guides[i-1]
    else:
        x = 0
    return 1 if x > 0 else -1 if x < 0 else 0

def _nearest_degree_pitch(target_degree, prev=None):
    candidates = list(POOLS['lead'])
    if prev is not None:
        legal = [p for p in candidates if lead_jump_ok(prev, p)]
        if legal:
            candidates = legal
    return min(candidates, key=lambda p: abs(degree(p) - int(target_degree)))

def _harmony_bar_similar(h1, h2, bpb):
    """True when two full bar-level harmony paths are motif-compatible.

    All change-points from both bars are merged.  In every resulting time slice
    the two active chords must share at least two ScaleWeaver pitch classes.  This
    is stricter than comparing only the first chord and therefore also handles
    multi-chord bars safely.
    """
    points = {0.0, float(bpb)}
    for h in (h1, h2):
        for seg in h.get('chord_segments', ()):
            points.add(round(float(seg['offset']), 6))
            points.add(round(float(seg['offset']) + float(seg['duration']), 6))
    points = sorted(x for x in points if -1e-9 <= x <= float(bpb) + 1e-9)
    for a, b in zip(points, points[1:]):
        if b - a <= 1e-8:
            continue
        mid = (a + b) * .5
        sa, sb = harmony_segment_at(h1, mid), harmony_segment_at(h2, mid)
        ca, cb = CHORDS[sa['chord_id']], CHORDS[sb['chord_id']]
        if len(set(ca.pcs) & set(cb.pcs)) < 2:
            return False
    return True

def _section_motif_indices(phrase_len, forbidden, rng):
    """Pick two memorable bars from different halves when possible."""
    usable = [i for i in range(max(0, phrase_len - 1)) if i not in forbidden]
    if not usable:
        return []
    left = [i for i in usable if i < phrase_len // 2]
    right = [i for i in usable if i >= phrase_len // 2]
    chosen = []
    if left:
        chosen.append(rng.choice(left))
    if right and len(chosen) < LEAD_SECTION_MOTIF_COUNT:
        chosen.append(rng.choice(right))
    rest = [i for i in usable if i not in chosen]
    while rest and len(chosen) < LEAD_SECTION_MOTIF_COUNT:
        x = rng.choice(rest); chosen.append(x); rest.remove(x)
    return sorted(chosen)

def _replay_bar_template(template, bar, plan, rng, bpb, prev, prev2, recent,
                         tag, section_repeat=False, vary_weak=False, source_bar=None):
    out, copied = [], []
    for item in template:
        off, dur = float(item['offset']), float(item['dur'])
        seg = harmony_segment_at(plan[bar], off)
        chord = CHORDS[seg['chord_id']]
        strong = is_strong_beat(off, bpb)
        source_pitch = int(item['step'])
        cand = fit_lead_pitch_to_jump(source_pitch, prev, source_pitch, True)
        if vary_weak and not strong and rng.random() < LEAD_SECTION_WEAK_VARIATION_RATE:
            trial = _nearest_degree_pitch(degree(source_pitch) + rng.choice((-1, 1)), prev)
            if trial is not None and lead_jump_ok(prev, trial):
                cand = trial
        if strong and lead_chord_field_cse(chord, cand) > 2.76:
            alt = choose_lead_pitch(prev, prev2, chord, True, recent, rng,
                                    degree(source_pitch), 0, False, off, dur, bpb)
            if alt is not None:
                cand = alt
        cand = fit_lead_pitch_to_jump(cand, prev, source_pitch, True)
        out.append(event(bar*bpb + off, dur, cand, 'lead', rng, seg,
                         structural=strong, motif=tag, section_repeat=bool(section_repeat),
                         section_variation=tag, ornamental_repeat=(prev is not None and cand == prev),
                         motif_source_bar=source_bar))
        copied.append({'offset': round(off, 6), 'dur': dur, 'step': int(cand)})
        prev2, prev = prev, cand
        recent.append(cand)
    return out, copied, prev, prev2

def _plan_lead_anchor_bars(plan, bars, bpb, rng):
    """Plan one ritornello-like one-bar anchor for the whole score.

    The source is chosen from the first phrase, then only later non-cadence bars
    with a harmony-compatible full-bar trajectory are eligible.  This creates
    recognition at bar scale without forcing an AABA whole-section copy.
    """
    if bars < 16 or rng.random() >= LEAD_ANCHOR_BAR_RATE:
        return None
    source_candidates = [i for i in range(min(7, bars))
                         if plan[i].get('chords_in_bar', 1) <= 2]
    rng.shuffle(source_candidates)
    for source in source_candidates:
        later = [j for j in range(max(8, source + 2), bars)
                 if j % 8 != 7 and _harmony_bar_similar(plan[source], plan[j], bpb)]
        if len(later) < 2:
            continue
        desired = rng.choices(LEAD_ANCHOR_APPEARANCES,
                              weights=LEAD_ANCHOR_APPEARANCE_WEIGHTS, k=1)[0]
        target_count = min(len(later), max(2, desired - 1))
        chosen = []
        for j in later:
            if not chosen or j - chosen[-1] >= 4:
                chosen.append(j)
            if len(chosen) >= target_count:
                break
        if len(chosen) < target_count:
            rest = [j for j in later if j not in chosen]
            rng.shuffle(rest)
            chosen += rest[:target_count-len(chosen)]
        chosen = tuple(sorted(chosen[:target_count]))
        return {'source': int(source), 'targets': chosen,
                'appearances': 1 + len(chosen)}
    return None

def generate_lead(bars, plan, rng, motif_degrees=None, beats_per_bar=4,
                  rhythm_palette=None, allow_sixteenth=False):
    palette = rhythm_palette or make_rhythm_palette(rng, beats_per_bar)
    motif_queue = user_motif_steps(motif_degrees)
    events, recent, section_memory = [], [], {}
    prev = prev2 = None
    anchor_plan = _plan_lead_anchor_bars(plan, bars, beats_per_bar, rng)
    anchor_template = None
    phrase_start = 0
    while phrase_start < bars:
        phrase_len = min(8, bars - phrase_start)
        section = plan[phrase_start].get('section', 'A')
        memory = section_memory.get(section)
        anchor_source = anchor_plan['source'] if anchor_plan else None
        anchor_targets = set(anchor_plan['targets']) if anchor_plan else set()
        anchor_local = {b - phrase_start for b in ({anchor_source} | anchor_targets)
                        if b is not None and phrase_start <= b < phrase_start + phrase_len}
        if memory:
            base_centre = memory['base_centre'] + _lead_centre_jitter(rng)
            macro_kind = (memory['macro_kind'] if rng.random() < LEAD_SECTION_MACRO_REUSE_RATE
                          else rng.choices(LEAD_SHAPE_KINDS,
                                           LEAD_PHRASE_SHAPE_WEIGHTS, k=1)[0])
        else:
            base_centre = rng.choice(LEAD_SECTION_BASE_CENTRES.get(section, LEAD_SECTION_BASE_CENTRES['A']))
            macro_kind = rng.choices(LEAD_SHAPE_KINDS,
                                     LEAD_PHRASE_SHAPE_WEIGHTS, k=1)[0]
        bar_centres = _lead_contour(
            macro_kind, phrase_len, base_centre, rng.choice(LEAD_PHRASE_AMPLITUDES))
        rhythms = [_choose_melodic_rhythm(
            phrase_start + j, rng, beats_per_bar, palette, cadence=(j == phrase_len - 1),
            allow_sixteenth=allow_sixteenth, fancy_rate=LEAD_FANCY_RHYTHM_RATE,
            syncopation_multiplier=LEAD_SYNCOPATED_RHYTHM_MULTIPLIER)
            for j in range(phrase_len)]
        recall_indices = set()
        if memory:
            eligible = [idx for idx, info in memory['motifs'].items()
                        if idx < phrase_len and _harmony_bar_similar(
                            plan[info['source_bar']], plan[phrase_start + idx], beats_per_bar)]
            if eligible:
                if rng.random() < LEAD_SECTION_PRIMARY_REUSE_RATE:
                    recall_indices.add(eligible[0])
                recall_indices.update(idx for idx in eligible[1:]
                                      if rng.random() < LEAD_SECTION_SECONDARY_REUSE_RATE)
                if not recall_indices:
                    recall_indices.add(eligible[0])
                recall_indices.difference_update(anchor_local)
                for idx in recall_indices:
                    rhythms[idx] = tuple(x['dur'] for x in memory['motifs'][idx]['template'])
        repeat_pair = None
        if phrase_len >= 4 and rng.random() < LEAD_MOTIF_REPEAT_RATE:
            pairs = []
            for source in range(phrase_len - 2):
                if source in recall_indices or source in anchor_local:
                    continue
                for target in range(source + 2, phrase_len):
                    if target in recall_indices or target in anchor_local:
                        continue
                    if _harmony_bar_similar(plan[phrase_start + source],
                                             plan[phrase_start + target], beats_per_bar):
                        pairs.append((source, target))
            if pairs:
                nonfinal = [x for x in pairs if x[1] != phrase_len - 1]
                repeat_pair = rng.choice(nonfinal or pairs)
                rhythms[repeat_pair[1]] = rhythms[repeat_pair[0]]
        bar_templates = {}
        for j in range(phrase_len):
            bar = phrase_start + j
            if anchor_plan and bar in anchor_targets and anchor_template:
                replayed, template, prev, prev2 = _replay_bar_template(
                    anchor_template, bar, plan, rng, beats_per_bar, prev, prev2, recent,
                    'ritornello_anchor_repeat', vary_weak=rng.random() < LEAD_ANCHOR_WEAK_VARIATION_RATE,
                    source_bar=anchor_source)
                for e in replayed:
                    e['anchor_bar_group'] = f'anchor:{anchor_source}'
                    e['anchor_bar_position'] = 'repeat'
                events.extend(replayed); bar_templates[j] = template
                continue
            if memory and j in recall_indices:
                info = memory['motifs'][j]
                replayed, template, prev, prev2 = _replay_bar_template(
                    info['template'], bar, plan, rng, beats_per_bar, prev, prev2, recent,
                    f'section_{section}_family_repeat', True, True, info['source_bar'])
                events.extend(replayed); bar_templates[j] = template
                continue
            if repeat_pair is not None and j == repeat_pair[1]:
                replayed, template, prev, prev2 = _replay_bar_template(
                    bar_templates.get(repeat_pair[0], ()), bar, plan, rng, beats_per_bar,
                    prev, prev2, recent, 'single_motif_repeat', source_bar=phrase_start + repeat_pair[0])
                events.extend(replayed); bar_templates[j] = template
                continue
            rhythm = rhythms[j]
            local_kind = rng.choices(LEAD_SHAPE_KINDS,
                                     LEAD_BAR_SHAPE_WEIGHTS, k=1)[0]
            guides = _lead_contour(
                local_kind, len(rhythm), bar_centres[j], rng.choice(LEAD_BAR_AMPLITUDES))
            off, template = 0.0, []
            for i, dur in enumerate(rhythm):
                seg = harmony_segment_at(plan[bar], off)
                chord = CHORDS[seg['chord_id']]
                strong = is_strong_beat(off, beats_per_bar)
                boundary = (j == 0 and i == 0) or (j == phrase_len - 1 and i == len(rhythm) - 1)
                guide = guides[i]
                if motif_queue:
                    cand = _fit_motif_pitch(motif_queue.pop(0), prev, guide)
                    tag = 'user'
                else:
                    cand = choose_lead_pitch(prev, prev2, chord, strong, recent, rng, guide,
                                             _phrase_direction(guides, i), boundary,
                                             off, dur, beats_per_bar)
                    tag = f'phrase_{macro_kind}_{local_kind}'
                if cand is None:
                    cand = nearest_pc('lead', chord.foot % OCT, degree_pitch(guide), prev)
                cand = fit_lead_pitch_to_jump(cand, prev, degree_pitch(guide), True)
                events.append(event(bar*beats_per_bar + off, dur, cand, 'lead', rng, seg,
                                    structural=strong, motif=tag, section_repeat=False,
                                    section_variation='fresh_phrase', ornamental_repeat=False))
                template.append({'offset': round(off, 6), 'dur': float(dur), 'step': int(cand)})
                prev2, prev = prev, cand
                recent.append(cand)
                off = round(off + float(dur), 6)
            bar_templates[j] = template
            if anchor_plan and bar == anchor_source:
                anchor_template = [dict(x) for x in template]
                for e in events[-len(template):]:
                    e['anchor_bar_group'] = f'anchor:{anchor_source}'
                    e['anchor_bar_position'] = 'source'
        if memory is None:
            motif_indices = _section_motif_indices(phrase_len, set(repeat_pair or ()) | set(anchor_local), rng)
            section_memory[section] = {
                'base_centre': base_centre, 'macro_kind': macro_kind,
                'motifs': {idx: {'source_bar': phrase_start + idx,
                                 'template': [dict(x) for x in bar_templates[idx]]}
                           for idx in motif_indices if idx in bar_templates}}
        phrase_start += phrase_len
    return events

def smooth_allowed(n):
    for q in RATIO_SMOOTH_PRIMES:
        while n > 1 and n % q == 0: n //= q
    return n == 1

@lru_cache(None)
def ratio_cost(step_diff):
    s = abs(step_diff) % OCT
    if not s: return 0.0
    target, best = 2 ** (s / OCT), 99.0
    for q in range(1, 17):
        for p in range(q, 2 * q + 1):
            if gcd(p, q) == 1 and smooth_allowed(p) and smooth_allowed(q):
                best = min(best, .045 * abs(1200 * math.log2(p / q / target)) + .32 * math.log2(max(2, p*q)))
    return best

def hard_wolf(a, b): return hr.hard_wolf(a, b)

def step_second(a, b):
    d = abs(a - b)
    return 0 < d <= SECOND_MAX_STEP

def overlaps(events, start, end):
    return [e for e in events if e['start_beat'] < end - 1e-9 and e['start_beat'] + e['duration_beats'] > start + 1e-9]

def _special_distance_soft_cost(cand, existing, start, dur):
    """Duration-weighted soft cost for configured non-wolf special intervals.

    This is deliberately generic: it knows only scale-degree distances and
    exact EDO-step distances supplied by ScaleSpec.  Wolves remain hard guards
    elsewhere.  A special interval may survive when no better legal candidate
    exists; the final cleanup pass is therefore best-effort rather than a
    validation requirement.
    """
    weight=float(CLEANUP_SOFT_PENALTY)
    if weight <= 0 or (not CLEANUP_DEGREE_DISTANCES and not CLEANUP_STEP_DISTANCES):
        return 0.0
    start=float(start); end=start+float(dur)
    if end <= start + 1e-12:
        return 0.0
    total=0.0
    for xs in existing.values():
        for e in overlaps(xs,start,end):
            a=max(start,float(e['start_beat']))
            b=min(end,float(e['start_beat'])+float(e['duration_beats']))
            if b <= a + 1e-9:
                continue
            q=int(e['step'])
            bad_degree=(abs(degree(int(cand))-degree(q)) in CLEANUP_DEGREE_DISTANCES)
            bad_step=(abs(int(cand)-q) in CLEANUP_STEP_DISTANCES)
            if bad_degree or bad_step:
                total += (b-a)/(end-start)
    return weight*total

def vertical_cost(cand, existing, start, dur):
    end = start + dur
    for e in overlaps(existing.get('lead', []), start, end):
        if cand >= e['step']:
            return math.inf
    cost = 0.0
    for xs in existing.values():
        for e in overlaps(xs, start, end):
            if hard_wolf(cand, e['step']) or step_second(cand, e['step']):
                return math.inf
            cost += .22 * ratio_cost(cand - e['step'])
    return cost

_VOICE_RANK = {'bass': 0, 'inner': 1, 'inner2': 2, 'counter': 3, 'lead': 4}

def _melodic_rhythm_name(ds):
    return 'mel_' + '_'.join(str(round(float(x), 3)).replace('.', 'p') for x in ds)

def _recent_motion(events, start):
    """Return the most recent onset-to-onset ScaleWeaver-degree motion before start."""
    xs = [e for e in events if float(e['start_beat']) <= float(start) + 1e-9]
    if len(xs) < 2:
        return 0
    xs = sorted(xs, key=lambda e: (float(e['start_beat']), float(e['duration_beats'])))
    onset = []
    for e in xs:
        if onset and abs(float(e['start_beat']) - float(onset[-1]['start_beat'])) < 1e-9:
            onset[-1] = e
        else:
            onset.append(e)
    if len(onset) < 2:
        return 0
    return degree(int(onset[-1]['step'])) - degree(int(onset[-2]['step']))

def _active_reference_step(existing, reference_voice, start, dur):
    if not reference_voice:
        return None
    xs = overlaps(existing.get(reference_voice, []), float(start), float(start) + float(dur))
    if not xs:
        return None
    mid = float(start) + float(dur) * .5
    return int(min(xs, key=lambda e: abs((float(e['start_beat']) + .5*float(e['duration_beats'])) - mid))['step'])

def _counterpoint_vertical_cost(voice, cand, existing, start, dur):
    """Hard registral ordering plus the project's ordinary vertical legality."""
    base = vertical_cost(cand, existing, start, dur)
    if math.isinf(base):
        return base
    rank = _VOICE_RANK[voice]
    end = float(start) + float(dur)
    for other_voice, xs in existing.items():
        if other_voice not in _VOICE_RANK:
            continue
        other_rank = _VOICE_RANK[other_voice]
        if other_rank == rank:
            continue
        for e in overlaps(xs, start, end):
            q = int(e['step'])
            if int(cand) == q:
                return math.inf
            if rank < other_rank and int(cand) >= q:
                return math.inf
            if rank > other_rank and int(cand) <= q:
                return math.inf
    return base

def _bass_leaves_inner_slot(cand, existing, start, dur, inner_pool=None):
    """Look ahead one layer so sparse Bass choices do not box Inner out."""
    end = float(start) + float(dur)
    upper_events = [(v, e) for v, xs in existing.items() if v in _VOICE_RANK and _VOICE_RANK[v] > 0
                    for e in overlaps(xs, start, end)]
    points = {float(start), end}
    for _, e in upper_events:
        points.add(max(float(start), float(e['start_beat'])))
        points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    if inner_pool is None:
        lo, hi = COUNTERPOINT_EFFECTIVE_RANGES['inner']
        inner_pool = [p for p in POOLS['inner'] if lo <= int(p) <= hi]
    for a, b in zip(sorted(points), sorted(points)[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        uppers = [int(e['step']) for _, e in upper_events
                  if float(e['start_beat']) < mid + 1e-9
                  and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
        if not uppers:
            continue
        ceiling = min(uppers)
        found = False
        for p in inner_pool:
            if not int(cand) < int(p) < int(ceiling):
                continue
            if any(hard_wolf(p, q) or step_second(p, q) for q in [int(cand), *uppers]):
                continue
            found = True
            break
        if not found:
            return False
    return True

def _counterpoint_interval_cost(ad, max_jump):
    ad = int(ad)
    if ad > int(max_jump):
        return math.inf
    c = _lead_interval_cost(min(ad, LEAD_MAX_JUMP_DEGREES))
    if ad > LEAD_PREFERRED_MAX_LEAP:
        c += .10 * (ad - LEAD_PREFERRED_MAX_LEAP)
    return c


# Bound only during the new greedy initializer; melody code is untouched.
_INITIAL_HARMONY_OBJECTIVE = None

def _choose_counterpoint_pitch(voice, prev, prev2, recent, chord, existing,
                               start, dur, rng, guide_degree, reference_voice,
                               prev_vertical_distance=None):
    """Choose one note of an independent contrapuntal line.

    The harmony plan is *not* a chord-tone whitelist.  It contributes only a
    virtual full-chord CSE background field.  Horizontal melody, actual vertical
    legality/CSE, registral ordering and contrary-motion preferences do the rest.
    """
    if _INITIAL_HARMONY_OBJECTIVE is not None:
        return _INITIAL_HARMONY_OBJECTIVE.choose_pitch(
            voice, prev, prev2, recent, chord, existing, start, dur, guide_degree)
    pool_voice = 'counter' if voice.startswith('counter') else voice
    max_jump = COUNTERPOINT_MAX_JUMP_DEGREES['counter' if voice.startswith('counter') else voice]
    ngram_scale = COUNTERPOINT_NGRAM_SCALE['counter' if voice.startswith('counter') else voice]
    reg_w = COUNTERPOINT_REGISTER_COST['counter' if voice.startswith('counter') else voice]
    prev_d = degree(prev) if prev is not None else None
    old = degree(prev) - degree(prev2) if prev is not None and prev2 is not None else 0
    ref_motion = _recent_motion(existing.get(reference_voice, []), start) if reference_voice else 0
    ref_step = _active_reference_step(existing, reference_voice, start, dur)
    scored = []
    lo, hi = COUNTERPOINT_EFFECTIVE_RANGES['counter' if voice.startswith('counter') else voice]
    for cand in POOLS[pool_voice]:
        if not lo <= int(cand) <= hi:
            continue
        vc = _counterpoint_vertical_cost(voice, cand, existing, start, dur)
        if math.isinf(vc):
            continue
        if (voice == 'bass' and not existing.get('inner')
                and not _bass_leaves_inner_slot(cand, existing, start, dur)):
            continue

        d = degree(cand)
        dd = d - prev_d if prev_d is not None else 0
        ad = abs(dd)
        ic = _counterpoint_interval_cost(ad, max_jump) if prev is not None else 0.0
        if math.isinf(ic):
            continue
        cost = vc + ic + reg_w * abs(d - int(guide_degree))
        cost += _special_distance_soft_cost(cand, existing, start, dur)

        if prev is not None and int(cand) == int(prev):
            cost += COUNTERPOINT_REPEAT_COST
        cost += ngram_scale * _lead_ngram_prior_cost(recent, cand)
        if voice == 'bass':
            cost += _bass_ngram_prior_cost(recent, cand)
        if prev is not None and dd and ref_motion:
            if dd * ref_motion < 0:
                cost -= COUNTERPOINT_CONTRARY_BONUS
            elif dd * ref_motion > 0:
                cost += COUNTERPOINT_PARALLEL_MOTION_COST

        if ref_step is not None:
            vert = abs(degree(int(ref_step)) - d)
            if prev_vertical_distance is not None and vert == int(
                    prev_vertical_distance) and dd and ref_motion and dd * ref_motion > 0:
                cost += .12

        if prev is not None and old:
            old_ad = abs(old)
            if (old_ad > LEAD_STRONG_RECOVERY_THRESHOLD
                    and dd * old < 0 and 1 <= ad <= LEAD_NORMAL_FLOW_MAX):
                cost -= .18
            elif (old_ad > LEAD_STRONG_RECOVERY_THRESHOLD
                    and dd * old > 0 and ad >= LEAD_NORMAL_FLOW_MAX):
                cost += .24

        for cse_cost in cse_rt.counterpoint_post_melodic_cost_components(
                voice, cand, chord, existing, start, dur,
                prime_rewards=_active_prime_rewards()):
            cost += cse_cost

        attack_limit = cse_rt.simultaneous_attack_soft_limit(
            cand, existing, start)
        scored.append((cost + rng.uniform(0.0, .10), cand, attack_limit))

    scored = cse_rt.apply_simultaneous_attack_soft_limits(scored)
    p = _sample_low_cost(scored, rng, top=6, temperature=.70)
    if p is None:
        raise GenerationRejected(f'no legal contrapuntal pitch for {voice} at beat {start}')
    return p


def _line_phrase_centres(voice, bars, rng):
    key = 'counter' if voice.startswith('counter') else voice
    base = COUNTERPOINT_ROLE_CENTRE_DEGREE[key]
    out = []
    for phrase_start in range(0, bars, 8):
        n = min(8, bars - phrase_start)
        kind = rng.choices(('arch', 'rise', 'fall', 'valley'), (.42, .22, .24, .12), k=1)[0]
        centre = base + rng.choice((-2, -1, 0, 0, 1, 2))
        amp = rng.choice((2, 3, 3, 4))
        out.extend(_lead_contour(kind, n, centre, amp))
    return out

def _choose_sparse_bass_rhythm(rng, bpb, cadence=False):
    """Very low attack density while keeping Bass continuously sounding."""
    if int(bpb) == 4:
        patterns = [
            ((4.0,), .14), ((2.0, 2.0), .40),
            ((3.0, 1.0), .14), ((1.0, 3.0), .14),
            ((1.5, 2.5), .09), ((2.5, 1.5), .09),
        ]
    else:
        patterns = [
            ((3.0,), .14), ((1.5, 1.5), .34),
            ((2.0, 1.0), .17), ((1.0, 2.0), .17),
            ((.5, 2.5), .09), ((2.5, .5), .09),
        ]
    if cadence:
        patterns = [(ds, w * (1.7 if len(ds) == 1 else 1.0)) for ds, w in patterns]
    return tuple(float(x) for x in rng.choices(
        [ds for ds, _ in patterns], weights=[w for _, w in patterns], k=1)[0])

def _generate_counterpoint_line(voice, bars, plan, existing, rng, bpb,
                                rhythm_palette, reference_voice):
    """Generate one complete monophonic melodic line spanning every bar."""
    events, recent = [], []
    prev = prev2 = None
    prev_vertical_distance = None
    centres = _line_phrase_centres(voice, bars, rng)
    fancy_key = 'counter' if voice.startswith('counter') else voice
    fancy_rate = COUNTERPOINT_FANCY_RHYTHM_RATE.get(
        fancy_key, COUNTERPOINT_FANCY_RHYTHM_RATE['inner'])

    for bar, h in enumerate(plan[:bars]):
        cadence = (bar % 8 == 7) or (bar == bars - 1)
        if voice == 'bass':
            ds = _choose_sparse_bass_rhythm(rng, bpb, cadence=cadence)
        else:
            ds = _choose_melodic_rhythm(bar, rng, bpb, rhythm_palette,
                                        cadence=cadence, allow_sixteenth=False,
                                        fancy_rate=fancy_rate)
        validate_rhythm_bar(ds, bpb, f'{voice}_melodic')
        off = 0.0
        pattern_name = _melodic_rhythm_name(ds)
        for dur in ds:
            seg = harmony_segment_at(h, off)
            chord = CHORDS[seg['chord_id']]
            start = bar * bpb + off
            p = _choose_counterpoint_pitch(
                voice, prev, prev2, recent, chord, existing, start, dur, rng,
                centres[bar], reference_voice, prev_vertical_distance,
            )
            ref_step = _active_reference_step(existing, reference_voice, start, dur)
            if ref_step is not None:
                prev_vertical_distance = abs(degree(ref_step) - degree(p))
            e = event(start, dur, p, voice, rng, seg,
                      surface='counterpoint_line', rhythm_pattern=pattern_name,
                      counterpoint=True, harmony_field_only=True)
            events.append(e)
            recent.append(p)
            prev2, prev = prev, p
            off = round(off + float(dur), 6)
    return events

def generate_counter_voice(bars, plan, lead, rng, bpb, rhythm_palette, mode=None):
    """Generate the sole independent Counter line.

    Fixed Lead-minus-3 / Lead-minus-7 score-wide parallel modes have been
    removed.  Ordinary contrary-motion and repeated-vertical-distance penalties
    inside _choose_counterpoint_pitch remain as contrapuntal preferences.
    """
    line = _generate_counterpoint_line(
        'counter', bars, plan, {'lead': lead}, rng, bpb, rhythm_palette, 'lead')
    return 'counter', line

def generate_inner(bars, plan, existing, rng, bpb, rhythm_palette):
    secondary = 'counter' if existing.get('counter') else 'lead'
    return _generate_counterpoint_line('inner', bars, plan, existing, rng, bpb,
                                       rhythm_palette, secondary)

def generate_bass(bars, plan, existing, rng, bpb, rhythm_palette):
    reference = 'inner' if existing.get('inner') else ('counter' if existing.get('counter') else 'lead')
    return _generate_counterpoint_line('bass', bars, plan, existing, rng, bpb,
                                       rhythm_palette, reference)


def _clone_event_with_pitch(template, pitch):
    """Copy an event while changing pitch and absolutely nothing rhythmic.

    start_beat, duration_beats, phase, velocity, motif/rhythm metadata and all
    other structural fields remain byte-for-byte equivalent Python values.
    """
    out = dict(template)
    p = int(pitch)
    out['step'] = p
    out['name'] = pitch_name(p)
    out['freq'] = freq(p)
    return out


def _iterative_lead_pitch(prev, prev2, recent, chord, strong, existing,
                          start, dur, rng, guide_degree, direction=0,
                          boundary=False, beat_in_bar=None, beats_per_bar=4):
    """Lead candidate search used only by fixed-rhythm regeneration.

    This is deliberately the Lead grammar rather than the generic accompaniment
    grammar: distance prior, recovery, n-grams, tonic-core frequency control,
    metrical landing and virtual chord field are all retained.  The new part is
    hard legality + attack-aware realised CSE against the other three voices.
    """
    scored = []
    prev_d = degree(prev) if prev is not None else None
    old = degree(prev) - degree(prev2) if prev is not None and prev2 is not None else 0
    centre = int(guide_degree)

    for cand in POOLS['lead']:
        d, pc = degree(cand), cand % OCT

        vc = _counterpoint_vertical_cost('lead', cand, existing, start, dur)
        if math.isinf(vc):
            continue

        if prev is not None:
            dd = d - prev_d
            ad = abs(dd)
            if ad > LEAD_MAX_JUMP_DEGREES:
                continue
        else:
            dd = ad = 0

        cost = vc + LEAD_GUIDE_WEIGHT * abs(d - centre)
        cost += _lead_register_cost(cand, recent)
        cost += _special_distance_soft_cost(cand, existing, start, dur)

        if prev is not None:
            cost += _lead_interval_cost(ad)
            if ad > LEAD_PREFERRED_MAX_LEAP:
                cost += .42 * (ad - LEAD_PREFERRED_MAX_LEAP)
            old_ad = abs(old)
            if LEAD_NORMAL_FLOW_MAX < old_ad <= LEAD_MEDIUM_LEAP_MAX:
                if dd * old < 0 and 1 <= ad <= LEAD_NORMAL_FLOW_MAX:
                    cost -= LEAD_MEDIUM_RECOVERY_BONUS
                elif dd * old > 0 and ad > LEAD_NORMAL_FLOW_MAX:
                    cost += .12
            elif old_ad > LEAD_STRONG_RECOVERY_THRESHOLD:
                if dd * old < 0 and 1 <= ad <= LEAD_NORMAL_FLOW_MAX:
                    cost -= LEAD_LEAP_RECOVERY_BONUS
                elif dd * old > 0 and ad >= LEAD_NORMAL_FLOW_MAX:
                    cost += .40

        cost += _lead_landing_cost(
            pc, chord, beat_in_bar, dur, beats_per_bar)

        if prev is not None and float(dur) <= .250001:
            if ad in LEAD_SIXTEENTH_FLOW_BONUS:
                cost -= LEAD_SIXTEENTH_FLOW_BONUS[ad]
            elif ad > LEAD_NORMAL_FLOW_MAX:
                cost += LEAD_SIXTEENTH_LARGE_LEAP_COST * (ad - LEAD_NORMAL_FLOW_MAX)

        if prev is not None and direction and dd:
            cost += -.10 if dd * direction > 0 else .12

        cost += _lead_ngram_prior_cost(recent, cand)
        cost += _lead_pc_frequency_cost(recent, pc)
        cost += _lead_motion_diversity_cost(recent, cand, strong, dur)
        cost += cse_rt.lead_chord_field_cost(chord, cand, strong)

        actual, coverage, _ = attack_aware_vertical_cse_interval(
            cand, existing, start, dur)
        if actual is not None and coverage > 0:
            cost += (hr._CSE_STRENGTH
                     * float(PITCH_REGEN_LEAD_ACTUAL_CSE_WEIGHT)
                     * float(actual))

        # Keep prime-colour rewards active for the regenerated Lead too when the
        # runtime exposes the same attack-aware helper used by accompaniment.
        prime_cost_fn = getattr(cse_rt, 'attack_aware_prime_reward_cost', None)
        if prime_cost_fn is not None:
            cost += prime_cost_fn(
                cand, existing, start, dur,
                prime_rewards=_active_prime_rewards())

        if boundary and pc == chord.foot % OCT:
            cost -= LEAD_BOUNDARY_FOOT_REWARD

        cost += rng.uniform(0.0, .055)
        attack_limit = cse_rt.simultaneous_attack_soft_limit(
            cand, existing, start)
        scored.append((cost, cand, attack_limit))

    scored = cse_rt.apply_simultaneous_attack_soft_limits(scored)
    cand = _sample_predictable(scored, rng)
    if cand is None:
        raise GenerationRejected(
            f'no legal iterative Lead pitch at beat {start}')
    return cand


def _regenerate_lead_fixed_rhythm(template, previous_line, plan, existing,
                                  rng, bpb):
    """Regenerate every Lead pitch while preserving its complete timing skeleton."""
    out, recent = [], []
    prev = prev2 = None
    n = len(template)

    for i, base in enumerate(template):
        start = float(base['start_beat'])
        dur = float(base['duration_beats'])
        bar = min(len(plan) - 1, int((start + 1e-9) // bpb))
        off = start - bar * bpb
        seg = harmony_segment_at(plan[bar], off)
        chord = CHORDS[seg['chord_id']]
        strong = is_strong_beat(off, bpb)

        template_step = int(base['step'])
        jitter = int(PITCH_REGEN_GUIDE_JITTER_DEGREES)
        guide = degree(template_step) + (rng.randint(-jitter, jitter) if jitter > 0 else 0)

        if i + 1 < len(template):
            next_old = degree(int(template[i + 1]['step']))
            this_old = degree(template_step)
            direction = 1 if next_old > this_old else -1 if next_old < this_old else 0
        else:
            direction = 0

        phrase_pos = bar % 8
        boundary = (
            (phrase_pos == 0 and abs(off) < 1e-9)
            or (phrase_pos == 7 and abs((off + dur) - bpb) < 1e-9)
            or i == 0 or i == n - 1
        )

        p = _iterative_lead_pitch(
            prev, prev2, recent, chord, strong, existing,
            start, dur, rng, guide, direction, boundary, off, bpb)

        out.append(_clone_event_with_pitch(base, p))
        recent.append(p)
        prev2, prev = prev, p

    return out


def _regenerate_counterpoint_fixed_rhythm(voice, template, previous_line, plan,
                                          existing, rng, bpb, reference_voice):
    """Regenerate one accompaniment line on an immutable onset/duration skeleton."""
    events, recent = [], []
    prev = prev2 = None
    prev_vertical_distance = None

    for i, base in enumerate(template):
        start = float(base['start_beat'])
        dur = float(base['duration_beats'])
        bar = min(len(plan) - 1, int((start + 1e-9) // bpb))
        off = start - bar * bpb
        seg = harmony_segment_at(plan[bar], off)
        chord = CHORDS[seg['chord_id']]

        template_step = int(base['step'])
        jitter = int(PITCH_REGEN_GUIDE_JITTER_DEGREES)
        guide = degree(template_step) + (rng.randint(-jitter, jitter) if jitter > 0 else 0)

        p = _choose_counterpoint_pitch(
            voice, prev, prev2, recent, chord, existing,
            start, dur, rng, guide, reference_voice, prev_vertical_distance)

        ref_step = _active_reference_step(
            existing, reference_voice, start, dur)
        if ref_step is not None:
            prev_vertical_distance = abs(degree(ref_step) - degree(p))

        events.append(_clone_event_with_pitch(base, p))
        recent.append(p)
        prev2, prev = prev, p

    return events


def _pitch_iteration_rng(seed, iteration, attempt, voice):
    salts = {
        'lead': 0x19A4C3D7,
        'counter': 0x2B7E51A9,
        'bass': 0x63D91F25,
        'inner': 0x51C8A73B,
    }
    x = (int(seed) & 0xFFFFFFFFFFFFFFFF)
    x ^= (int(iteration) + 1) * 0x9E3779B185EBCA87
    x ^= (int(attempt) + 1) * 0xC2B2AE3D27D4EB4F
    x ^= salts[voice]
    return random.Random(x & 0xFFFFFFFFFFFFFFFF)


def _fixed_rhythm_signature(voices):
    return {
        voice: tuple(
            (float(e['start_beat']), float(e['duration_beats']))
            for e in xs)
        for voice, xs in voices.items()
    }


def _regenerate_all_pitches_iteratively(voices, plan, seed, bpb,
                                        iterations=PITCH_REGEN_ITERATIONS):
    """Run full fixed-rhythm pitch sweeps over all four voices.

    The immutable templates are captured before the first iteration.  At every
    sweep, Lead sees the preceding Counter/Bass/Inner; Counter then sees the new
    Lead; Bass sees new Lead+Counter and the preceding Inner; Inner finally sees
    all three newly regenerated voices.  On the next sweep every voice therefore
    receives feedback from the complete preceding four-part result.
    """
    iterations = max(0, int(iterations))
    if iterations == 0:
        return voices, []

    templates = {
        voice: [dict(e) for e in xs]
        for voice, xs in voices.items()
    }
    fixed_signature = _fixed_rhythm_signature(templates)
    current = {
        voice: [dict(e) for e in xs]
        for voice, xs in voices.items()
    }
    history = []

    for iteration in range(iterations):
        previous = {
            voice: [dict(e) for e in xs]
            for voice, xs in current.items()
        }

        success = None
        last_error = None
        for attempt in range(max(1, int(PITCH_REGEN_MAX_RETRIES))):
            trial = {
                voice: [dict(e) for e in xs]
                for voice, xs in previous.items()
            }
            try:
                # 1) Lead: actual Counter/Bass/Inner CSE + Lead grammar.
                lead_existing = {v: xs for v, xs in trial.items() if v != 'lead'}
                trial['lead'] = _regenerate_lead_fixed_rhythm(
                    templates['lead'], previous['lead'], plan, lead_existing,
                    _pitch_iteration_rng(seed, iteration, attempt, 'lead'), bpb)

                # 2) Independent Counter only.
                counter_existing = {v: xs for v, xs in trial.items() if v != 'counter'}
                trial['counter'] = _regenerate_counterpoint_fixed_rhythm(
                    'counter', templates['counter'], previous['counter'], plan,
                    counter_existing,
                    _pitch_iteration_rng(seed, iteration, attempt, 'counter'),
                    bpb, 'lead')

                # 3) Bass.  Existing Inner from the preceding sweep is a real
                # sounding constraint, so the old "leave a future Inner slot"
                # lookahead is skipped automatically.
                bass_existing = {v: xs for v, xs in trial.items() if v != 'bass'}
                bass_reference = 'inner' if bass_existing.get('inner') else 'counter'
                trial['bass'] = _regenerate_counterpoint_fixed_rhythm(
                    'bass', templates['bass'], previous['bass'], plan,
                    bass_existing,
                    _pitch_iteration_rng(seed, iteration, attempt, 'bass'),
                    bpb, bass_reference)

                # 4) Inner sees the three newly regenerated lines.
                inner_existing = {v: xs for v, xs in trial.items() if v != 'inner'}
                trial['inner'] = _regenerate_counterpoint_fixed_rhythm(
                    'inner', templates['inner'], previous['inner'], plan,
                    inner_existing,
                    _pitch_iteration_rng(seed, iteration, attempt, 'inner'),
                    bpb, 'counter')

                # Exact rhythm immutability check.
                if _fixed_rhythm_signature(trial) != fixed_signature:
                    raise RuntimeError('iterative pitch pass changed rhythm skeleton')

                hard, seconds, crossings = validate(trial)
                timing = accompaniment_timing_errors(trial, bpb)
                jumps = lead_jump_errors(trial)
                if hard or seconds or crossings or timing or jumps:
                    raise GenerationRejected(
                        f'iteration {iteration+1} validation failed '
                        f'hard={hard[:1]} seconds={seconds[:1]} '
                        f'crossings={crossings[:1]} timing={timing[:1]} '
                        f'jumps={jumps[:1]}')

                success = trial
                history.append({
                    'iteration': iteration + 1,
                    'attempt': attempt + 1,
                    'lead_notes': len(trial.get('lead', ())),
                    'counter_notes': len(trial.get('counter', ())),
                    'bass_notes': len(trial.get('bass', ())),
                    'inner_notes': len(trial.get('inner', ())),
                    'pitch_changes_from_previous': {
                        voice: sum(
                            int(a['step']) != int(b['step'])
                            for a, b in zip(previous.get(voice, ()), trial.get(voice, ()))
                        )
                        for voice in ('lead', 'counter', 'bass', 'inner')
                    },
                })
                break
            except GenerationRejected as exc:
                last_error = exc

        if success is None:
            raise GenerationRejected(
                f'pitch regeneration iteration {iteration+1} failed after '
                f'{PITCH_REGEN_MAX_RETRIES} attempts: {last_error}')
        current = success

    return current, history


def accompaniment_timing_errors(voices, bpb):
    errors=[]; eps=2e-6
    for voice, xs in voices.items():
        for e in xs:
            bar=int((e['start_beat']+eps)//bpb)
            # The fixed Lead IR may legally sustain across half-bar and bar
            # boundaries.  Generated accompaniment voices remain bar-local.
            if voice != 'lead' and e['start_beat']+e['duration_beats']>(bar+1)*bpb+eps:
                errors.append((voice,'bar_overrun',bar,e['start_beat'],e['duration_beats']))
    lead=sorted(voices.get('lead', []), key=lambda e:e['start_beat'])
    if any(a['start_beat']+a['duration_beats']>b['start_beat']+eps for a,b in zip(lead,lead[1:])):
        errors.append(('lead','overlap'))
    for voice in ('bass','inner','counter'):
        by_bar={}
        for e in voices.get(voice,[]): by_bar.setdefault(int((e['start_beat']+eps)//bpb),[]).append(e)
        for bar,xs in by_bar.items():
            xs=sorted(xs,key=lambda e:e['start_beat'])
            if any(a['start_beat']+a['duration_beats']>b['start_beat']+eps for a,b in zip(xs,xs[1:])):
                errors.append((voice,'overlap',bar))
    return errors

def validate(voices):
    names=list(voices); hard=[]; seconds=[]; crossings=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for e in voices[a]:
                for f in overlaps(voices[b],e['start_beat'],e['start_beat']+e['duration_beats']):
                    if hard_wolf(e['step'],f['step']):
                        hard.append((a,b,e['start_beat'],e['name'],f['name']))
                    if step_second(e['step'],f['step']):
                        seconds.append((a,b,e['start_beat'],e['name'],f['name']))
                    ra, rb = _VOICE_RANK.get(a), _VOICE_RANK.get(b)
                    if ra is not None and rb is not None and ra != rb:
                        if (ra < rb and int(e['step']) >= int(f['step'])) or (ra > rb and int(e['step']) <= int(f['step'])):
                            crossings.append((a,b,e['start_beat'],e['name'],f['name']))
    return hard,seconds,crossings

def lead_jump_errors(voices_or_lead):
    """Return adjacent Lead jumps above the hard ten-distance limit.

    Compatibility: callers may pass either the whole ``voices`` dictionary
    (as main.py does) or the Lead event list directly (as generate_once does).
    """
    lead = voices_or_lead.get('lead', []) if isinstance(voices_or_lead, dict) else voices_or_lead
    return [(a['start_beat'], b['start_beat'], abs(degree(b['step']) - degree(a['step'])))
            for a, b in zip(lead, lead[1:])
            if abs(degree(b['step']) - degree(a['step'])) > LEAD_MAX_JUMP_DEGREES]

def diagnostics(voices, plan, bpb):
    lead = voices.get('lead', [])
    segments = [seg for h in plan for seg in h['chord_segments']]
    total_harmony = sum(float(seg['duration']) for seg in segments) or 1.0
    cse_chord = sum(chord_cse_score(CHORDS[seg['chord_id']]) * float(seg['duration'])
                    for seg in segments) / total_harmony
    lead_cse = []
    for e in lead:
        bar = min(len(plan) - 1, int(e['start_beat'] // bpb))
        chord = CHORDS[harmony_segment_at(plan[bar], e['start_beat'] - bar*bpb)['chord_id']]
        lead_cse.append(vertical_cse_score(chord, e['step']))
    absolute = []
    for voice, xs in voices.items():
        for e in xs:
            others = {v: [x for x in ys if x is not e] for v, ys in voices.items()}
            value, coverage = absolute_vertical_cse_interval(
                e['step'], others, float(e['start_beat']), float(e['duration_beats']))
            if value is not None and coverage > 0:
                absolute.append((value, float(e['duration_beats']) * coverage))
    absolute_mean = (sum(v*w for v, w in absolute) / sum(w for _, w in absolute)) if absolute else None
    chord_usage = {cid: 0 for cid in CHORDS}
    function_counts = {f: 0 for f in 'TSCD'}
    for seg in segments:
        chord_usage[seg['chord_id']] += 1
        function_counts[seg['function']] += 1
    used = {cid for cid, n in chord_usage.items() if n}
    moves = [abs(degree(b['step']) - degree(a['step'])) for a, b in zip(lead, lead[1:])]
    same_pc = sum(a['step'] % OCT == b['step'] % OCT for a, b in zip(lead, lead[1:]))
    aba = sum(a['step'] % OCT == c['step'] % OCT for a, c in zip(lead, lead[2:]))
    secondary = 'counter' if voices.get('counter') else None
    lead_mean_step = (sum(float(e['step']) for e in lead) / len(lead)) if lead else None
    lead_register_offset = (lead_mean_step - LEAD_REGISTER_TARGET_STEP) if lead_mean_step is not None else None
    return {
        'lead_notes': len(lead),
        'lead_max_jump_degrees': max(moves, default=0),
        'lead_same_pc_rate': round(same_pc / max(1, len(lead)-1), 4),
        'lead_aba_pc_rate': round(aba / max(1, len(lead)-2), 4),
        'lead_register_target_step': round(float(LEAD_REGISTER_TARGET_STEP), 4),
        'lead_register_mean_step': round(float(lead_mean_step), 4) if lead_mean_step is not None else None,
        'lead_register_mean_offset_steps': round(float(lead_register_offset), 4) if lead_register_offset is not None else None,
        'lead_register_high_multiplier': float(LEAD_REGISTER_HIGH_MULTIPLIER),
        'lead_register_low_multiplier': float(LEAD_REGISTER_LOW_MULTIPLIER),
        'lead_section_base_centres': {k: list(v) for k, v in LEAD_SECTION_BASE_CENTRES.items()},
        'lead_phrase_amplitudes': list(LEAD_PHRASE_AMPLITUDES),
        'lead_bar_amplitudes': list(LEAD_BAR_AMPLITUDES),
        'lead_centre_jitter_degrees': int(LEAD_CENTRE_JITTER_DEGREES),
        'scale_note_count': len(PCS),
        'lead_normal_flow_max_degrees': int(LEAD_NORMAL_FLOW_MAX),
        'lead_medium_leap_max_degrees': int(LEAD_MEDIUM_LEAP_MAX),
        'lead_preferred_max_leap_degrees': int(LEAD_PREFERRED_MAX_LEAP),
        'lead_strong_recovery_threshold_degrees': int(LEAD_STRONG_RECOVERY_THRESHOLD),
        'bass_notes_per_bar': round(len(voices.get('bass', [])) / max(1, len(plan)), 4),
        'secondary_voice_mode': secondary,
        'prime_rewards': _active_prime_rewards(),
        'prime_reward_runtime': getattr(cse_rt, '__file__', None),
        'function_counts': function_counts,
        'chord_usage': {cid: n for cid, n in chord_usage.items() if n},
        'palette_color_family_count': sum(bool(set(ids) & used) for ids in PALETTE_COLOR_FAMILIES.values()),
        'cse_chord_mean': round(cse_chord, 5),
        'cse_lead_vertical_raw_mean': round(sum(lead_cse) / max(1, len(lead_cse)), 5),
        'cse_actual_register_raw_mean': round(absolute_mean, 5) if absolute_mean is not None else None,
        'cse_lead_vertical_percentile_mean': round(sum(lead_cse) / max(1, len(lead_cse)), 5),
        'cse_absolute_vertical_percentile_mean': round(absolute_mean, 5) if absolute_mean is not None else None,
    }

def generate_once(seed, bars, bpm, motif_degrees, time_signature, allow_sixteenth=False,
                  pitch_regen_iterations=None):
    ts, bpb = normalize_time_signature(time_signature)
    harmony_rng = random.Random(seed)
    plan = harmony_plan(bars, harmony_rng, bpb)

    lead_rng = random.Random(int(seed) ^ 0x13579BDF)
    lead_palette = make_rhythm_palette(lead_rng, bpb)
    lead = generate_lead(bars, plan, lead_rng, motif_degrees, bpb, lead_palette,
                         allow_sixteenth=allow_sixteenth)
    voices = {'lead': lead}

    counter_rng = random.Random(int(seed) ^ 0x2468ACE1)
    counter_palette = make_rhythm_palette(counter_rng, bpb)
    counter_mode, counter_line = generate_counter_voice(
        bars, plan, lead, counter_rng, bpb, counter_palette)
    voices['counter'] = counter_line

    bass_rng = random.Random(int(seed) ^ 0x6C39B5A7)
    bass_palette = make_rhythm_palette(bass_rng, bpb)
    voices['bass'] = generate_bass(bars, plan, dict(voices), bass_rng, bpb, bass_palette)

    inner_rng = random.Random(int(seed) ^ 0x51A7E2C3)
    inner_palette = make_rhythm_palette(inner_rng, bpb)
    voices['inner'] = generate_inner(bars, plan, dict(voices), inner_rng, bpb, inner_palette)

    # Rhythm has now been chosen exactly once for all four voices.  Every
    # following pass changes pitch fields only.
    active_pitch_regen_iterations = (
        int(PITCH_REGEN_ITERATIONS)
        if pitch_regen_iterations is None
        else max(0, int(pitch_regen_iterations))
    )
    voices, pitch_iteration_history = _regenerate_all_pitches_iteratively(
        voices, plan, seed, bpb, active_pitch_regen_iterations)

    order = ('bass', 'inner', 'counter', 'lead')
    voices = {k: voices[k] for k in order if k in voices}
    hard, seconds, crossings = validate(voices)
    timing = accompaniment_timing_errors(voices, bpb)
    jumps = lead_jump_errors(lead)
    if hard or seconds or crossings or timing or jumps:
        raise GenerationRejected(
            f'generation validation failed hard={hard[:2]} seconds={seconds[:2]} '
            f'crossings={crossings[:2]} timing={timing[:2]} jumps={jumps[:2]}')

    palettes = {
        'lead': rhythm_palette_metadata(lead_palette),
        'counter': rhythm_palette_metadata(counter_palette),
        'inner': rhythm_palette_metadata(inner_palette),
        'bass': rhythm_palette_metadata(bass_palette),
    }
    return {
        'format': 'ScaleWeaverScore/1', 'seed': seed, 'tempo_bpm': bpm,
        'time_signature': ts, 'beats_per_bar': bpb, 'bars': bars,
        'duration_seconds': bars * bpb * 60 / bpm,
        'configuration': hr.SCALE.to_json_dict(),
        'tuning': {'system': 'edo', 'edo': OCT,
                   'scale_id': SCALE_ID, 'scale_name': SCALE_NAME,
                   'base_note': BASE_NOTE,
                   'base_freq_hz': BASE_FREQ, 'base_freq': BASE_FREQ,
                   'pcs': list(PCS), 'names': list(NAMES),
                   'pitch_classes': [{'name': n, 'step': p} for n, p in zip(NAMES, PCS)]},
        'rhythmic_system': {
            'model': 'four independent melodic lines with motif recall; fixed-rhythm iterative pitch regeneration; bar-sequence and chord-arpeggio devices disabled',
            'allow_sixteenth': bool(allow_sixteenth),
            'fancy_rhythm_first_class': True,
            'melodic_palettes': palettes,
        },
        'harmonic_system': {
            'model': 'four-line iterative counterpoint; fixed rhythms, regenerated pitches; realised other-voice CSE plus virtual harmony field; fixed parallel-3/7 modes removed',
            'chords': {
                cid: {'name': c.name, 'pcs': list(c.pcs), 'ratio': c.ratio, 'function': c.function}
                for cid, c in CHORDS.items()
            },
        },
        'harmony_plan': plan,
        'voice_layout': {
            'four_part_counterpoint': True,
        'relative_purity_enabled': True,
            'secondary_voice_mode': 'counter',
            'secondary_voice_candidates': ['counter'],
            'secondary_voice_mutually_exclusive': False,
            'fixed_parallel_three_seven_removed': True,
            'pitch_regeneration_iterations': int(active_pitch_regen_iterations),
        },
        'voices': voices,
        'validation': {
            'hard_wolves': 0, 'step_seconds': 0, 'voice_crossings': 0,
            'accompaniment_timing_errors': 0, 'lead_jumps_over_configured_limit': 0,
        },
        'diagnostics': {
            **diagnostics(voices, plan, bpb),
            'pitch_regeneration_iterations': int(active_pitch_regen_iterations),
            'pitch_regeneration_history': pitch_iteration_history,
            'fixed_rhythm_during_regeneration': True,
            'fixed_parallel_three_seven_removed': True,
        },
    }
