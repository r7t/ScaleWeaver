"""ScaleWeaver score builder: melodic search and multi-voice counterpoint.

CSE lookup and CSE-derived scoring live in ``adaptive_cse_runtime.py``.
The Lead keeps n-grams, contour, motif recall and hard jump rules; bar-sequence
and chord-arpeggio devices were intentionally removed in this compact revision.
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
    _lead_shape_statistics.cache_clear()
    _lead_history_counts.cache_clear()
    _lead_history_intervals.cache_clear()

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

@lru_cache(maxsize=8192)
def _lead_shape_statistics(pitches):
    from melody_character import shape_statistics
    return shape_statistics(pitches,degree)

@lru_cache(maxsize=8192)
def _lead_history_counts(pitches):
    counts={}
    for p in pitches:
        pc=int(p)%OCT
        counts[pc]=counts.get(pc,0)+1
    return counts

@lru_cache(maxsize=8192)
def _lead_history_intervals(pitches):
    degrees=[degree(p) for p in pitches]
    return tuple(abs(b-a) for a,b in zip(degrees,degrees[1:]))

def _lead_reference_shape_cost(recent, cand, duration=1.):
    weight = float(hr.SCALE.style.get('melody_plan', {}).get('joint_generation', {}).get('melodic_shape_weight', 0.))
    if not weight:
        return 0.
    from melody_character import shape_cost
    statistics=_lead_shape_statistics(tuple(recent[-25:])) if recent else None
    return weight * shape_cost(recent, cand, degree, duration,statistics)

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
    intervals=_lead_history_intervals(tuple(recent[-(LEAD_LEAP_BALANCE_WINDOW+1):]))
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


def _lead_phrase_contour(kind, n, centre, amplitude):
    """Cadence-aware macro contour for one complete phrase.

    A short rising phrase otherwise spends every pre-cadential bar climbing
    and leaves all of the descent to the final tonic.  For phrases of at most
    six bars, move the peak to the third-last bar and use the last two bars as
    a gradual release.  Longer and non-rising contours retain their original
    shape.
    """
    if kind != 'rise' or n > 6 or n < 4:
        return _lead_contour(kind, n, centre, amplitude)
    peak = n - 3
    out = []
    for i in range(n):
        if i <= peak:
            y = -1.0 + 2.0 * i / peak
        else:
            y = 1.0 - 1.4 * (i - peak) / (n - 1 - peak)
        out.append(int(round(centre + amplitude * y)))
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
    """Return deterministic Lead costs for sampling and beam search."""
    cand = int(cand)
    d, pc = degree(cand), cand % OCT
    prev_d = degree(prev) if prev is not None else None
    old = degree(prev) - degree(prev2) if prev is not None and prev2 is not None else 0
    centre = int(guide_degree)
    if prev is not None:
        dd = d - prev_d
        ad = abs(dd)
        if ad > LEAD_MAX_JUMP_DEGREES:
            return None
    else:
        dd = ad = 0
    pc_counts = _lead_history_counts(tuple(recent))
    recent_mean_step = None
    if recent and LEAD_REGISTER_CENTROID_FEEDBACK != 0.0:
        window = recent[-int(LEAD_REGISTER_CENTROID_WINDOW):]
        recent_mean_step = sum(float(p) for p in window) / len(window)

    cost = LEAD_GUIDE_WEIGHT * abs(d - centre)
    cost += _lead_register_cost(cand, recent, recent_mean_step)
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
    cost += _lead_landing_cost(pc, chord, beat_in_bar, duration, beats_per_bar)
    if prev is not None and float(duration) <= .250001:
        if ad in LEAD_SIXTEENTH_FLOW_BONUS:
            cost -= LEAD_SIXTEENTH_FLOW_BONUS[ad]
        elif ad > LEAD_NORMAL_FLOW_MAX:
            cost += LEAD_SIXTEENTH_LARGE_LEAP_COST * (ad - LEAD_NORMAL_FLOW_MAX)
    if prev is not None and direction and dd:
        cost += -.10 if dd * direction > 0 else .12
    cost += _lead_ngram_prior_cost(recent, cand)
    cost += _lead_pc_frequency_cost(recent, pc, pc_counts)
    cost += _lead_motion_diversity_cost(recent, cand, strong, duration)
    cost += _lead_reference_shape_cost(recent, cand, duration)
    cost += cse_rt.lead_chord_field_cost(chord, cand, strong)
    if boundary and pc == chord.foot % OCT:
        cost -= LEAD_BOUNDARY_FOOT_REWARD
    return {'total':cost}


def choose_lead_pitch(prev, prev2, chord, strong, recent, rng, guide_degree,
                      direction=0, boundary=False, beat_in_bar=None,
                      duration=1.0, beats_per_bar=4):
    scored = []
    for cand in POOLS['lead']:
        components = lead_candidate_cost_components(
            prev, prev2, chord, strong, recent, cand, guide_degree, direction,
            boundary, beat_in_bar, duration, beats_per_bar)
        if components is not None:
            scored.append((components['total'] + rng.uniform(0.0, .055), cand))
    cand = _sample_predictable(scored, rng)
    if cand is not None:
        return cand
    return nearest_pc('lead', chord.foot % OCT, degree_pitch(centre), prev)

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


def _line_phrase_centres(voice, bars, rng, plan=None):
    key = 'counter' if voice.startswith('counter') else voice
    base = COUNTERPOINT_ROLE_CENTRE_DEGREE[key]
    out = []
    phrase_start = 0
    while phrase_start < bars:
        phrase_bars = int(plan[phrase_start].get('phrase_bars', 8)) if plan else 8
        n = min(phrase_bars, bars - phrase_start)
        kind = rng.choices(('arch', 'rise', 'fall', 'valley'), (.42, .22, .24, .12), k=1)[0]
        centre = base + rng.choice((-2, -1, 0, 0, 1, 2))
        amp = rng.choice((2, 3, 3, 4))
        out.extend(_lead_phrase_contour(kind, n, centre, amp))
        phrase_start += n
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
    centres = _line_phrase_centres(voice, bars, rng, plan)
    fancy_key = 'counter' if voice.startswith('counter') else voice
    fancy_rate = COUNTERPOINT_FANCY_RHYTHM_RATE.get(
        fancy_key, COUNTERPOINT_FANCY_RHYTHM_RATE['inner'])

    for bar, h in enumerate(plan[:bars]):
        phrase_bars = int(h.get('phrase_bars', 8))
        cadence = ((bar + 1) % phrase_bars == 0) or (bar == bars - 1)
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



def accompaniment_timing_errors(voices, bpb):
    """Return accompaniment timing violations."""
    errors=[]; eps=2e-6
    accompaniment = tuple(voice for voice in voices if voice != 'lead')
    for voice in accompaniment:
        xs = voices[voice]
        for e in xs:
            bar=int((e['start_beat']+eps)//bpb)
            if e['start_beat']+e['duration_beats']>(bar+1)*bpb+eps:
                errors.append((voice,'bar_overrun',bar,e['start_beat'],e['duration_beats']))
    for voice in accompaniment:
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

def lead_jump_errors(lead):
    """Return adjacent Lead jumps above the configured hard limit."""
    return [(a['start_beat'], b['start_beat'], abs(degree(b['step']) - degree(a['step'])))
            for a, b in zip(lead, lead[1:])
            if abs(degree(b['step']) - degree(a['step'])) > LEAD_MAX_JUMP_DEGREES]
