"""Runtime spectral evaluator used by score generation.

This module owns every CSE lookup adapter and every CSE-derived weighting term
used by score generation: virtual chord-field CSE, actual-register CSE,
attack-aware CSE, relative-purity floors, optimistic four-part look-ahead,
octave-family anti-collapse costs, and the soft 11-limit risk cost.

It depends only on ``harmony_rhythm`` and event dictionaries.  It deliberately
does not import ``score_builder`` so the dependency direction stays one-way:

    score_builder -> adaptive_cse_runtime -> harmony_rhythm

"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations

import harmony_rhythm as hr
from harmony_rhythm import CHORDS, OCT, POOLS
from scale_config import (resolved_cse_weights,
                          resolved_attack_cse_percentile_limits)
from spectral_model import resolve_parameters
from spectral_bundle import _bundle_covers_spec_data

# ---------------------------------------------------------------------------
# Active spectral weights, rebound from the selected style at configuration time.
# ---------------------------------------------------------------------------
globals().update(resolved_cse_weights({}))
ATTACK_CSE_PERCENTILE_LIMITS = {}

# ---------------------------------------------------------------------------
# Unified spectral blend: A*CSE + B*SE + C*CCSE. C is NOT an offset.
# mixed_entry()[1] is the true midrank percentile of this complete blend.
# ---------------------------------------------------------------------------
_SPECTRAL_PARAMETERS = resolve_parameters()


def _blend_raw_cse_se(raw_cse, raw_se, raw_ccse):
    return (float(CSE_2D_A)*float(raw_cse) + float(CSE_2D_B)*float(raw_se)
            + float(CSE_2D_C)*float(raw_ccse))


ATTACK_JI_PRIMES = (2, 3, 5, 7, 11, 13, 17)
_ACTIVE_JI_TABLE = None
_ACTIVE_JI_TABLE_PATH = None

COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS = 1

# The look-ahead completion search needs exactly the same usable registers as
# the caller.  Keeping the single canonical copy here prevents CSE projection
# and actual counterpoint selection from silently diverging.
COUNTERPOINT_EFFECTIVE_RANGES = {}


def configure_scale(spec, ji_table):
    """Rebind scale globals and attach the scale-specific canonical JI map."""
    global CHORDS, OCT, POOLS, COUNTERPOINT_EFFECTIVE_RANGES, _VOICE_RANK
    global _ACTIVE_JI_TABLE, _ACTIVE_JI_TABLE_PATH
    global _SPECTRAL_PARAMETERS, ATTACK_CSE_PERCENTILE_LIMITS
    _SPECTRAL_PARAMETERS = resolve_parameters(spec.rules.get('spectral_parameters'))
    globals().update(resolved_cse_weights(spec.style))
    ATTACK_CSE_PERCENTILE_LIMITS = resolved_attack_cse_percentile_limits(spec.style)
    runtime = spec.rules.get('runtime', {})
    for key, default in {'COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS':1}.items():
        value = float(runtime.get(key, default))
        if not math.isfinite(value): raise ValueError(f'Non-finite runtime.{key}')
        globals()[key] = int(value) if isinstance(default,int) else value
    CHORDS = hr.CHORDS
    OCT = hr.OCT
    POOLS = hr.POOLS
    rows=((spec.metadata or {}).get('generator', {}) or {}).get('counterpoint_effective_ranges')
    if rows:
        COUNTERPOINT_EFFECTIVE_RANGES={str(k):tuple(map(int,v)) for k,v in rows.items()}
    else:
        r=spec.resolved_voice_ranges(); COUNTERPOINT_EFFECTIVE_RANGES={k:tuple(r[k]) for k in spec.resolved_ensemble_voices()[:-1]}
    _VOICE_RANK={voice:i for i,voice in enumerate(spec.resolved_ensemble_voices())}
    if ji_table is None or not hasattr(ji_table, 'lookup') or not hasattr(ji_table, 'data'):
        raise TypeError('configure_scale requires a loaded JI table')
    _ACTIVE_JI_TABLE = ji_table
    _ACTIVE_JI_TABLE_PATH = ji_table.path
    if hr.CSE is None or not _bundle_covers_spec_data(hr.CSE.data, spec):
        raise ValueError('Configure a unified SE/CSE/CCSE bundle covering the active pitch domains first')
    # Clear all context-dependent memoized values, including completion search.
    # Switching scale or style in one interpreter must equal a fresh process.
    for value in tuple(globals().values()):
        if callable(value) and hasattr(value, 'cache_clear'):
            value.cache_clear()



_VOICE_RANK = {
    'bass': 0, 'inner': 1, 'counter': 2,
    'lead': 3,
}


def overlaps(events, start, end):
    """Local event-overlap helper; avoids importing score_builder."""
    return [e for e in events
            if e['start_beat'] < end - 1e-9
            and e['start_beat'] + e['duration_beats'] > start + 1e-9]


def _prime_reward_weights(values):
    return {p:float(values.get(p, values.get(str(p), 0.0)))
            for p in ATTACK_JI_PRIMES}


def _prime_rewards_enabled(values):
    return any(abs(x) > 1e-15 for x in _prime_reward_weights(values).values())


def _prime_bit(prime):
    return 1 << ATTACK_JI_PRIMES.index(int(prime))


def _integer_prime_mask(n):
    """Prime-presence mask of one positive integer, limited to 2..17."""
    n = abs(int(n))
    mask = 0
    for p in ATTACK_JI_PRIMES:
        if n and n % p == 0:
            mask |= _prime_bit(p)
    return mask


def _ratio_prime_mask(ratio):
    """Prime mask of a reduced ratio string such as '21/20'."""
    text = str(ratio)
    if '/' in text:
        a, b = text.split('/', 1)
        return _integer_prime_mask(int(a)) | _integer_prime_mask(int(b))
    return _integer_prime_mask(int(text))


@lru_cache(maxsize=1)
def _load_attack_ji_dyad_index():
    """Return dyad/group prime masks from the active scale-specific JI table."""
    if _ACTIVE_JI_TABLE is None:
        raise RuntimeError('canonical JI table has not been configured for the active scale')
    data = _ACTIVE_JI_TABLE.data
    if (int(data.get('prime_limit', -1)) != 17
            or data.get('format') != ji_ratio.FORMAT
            or int(data.get('edo', -1)) != int(OCT)):
        raise ValueError(f'incompatible adaptive attack JI table: {_ACTIVE_JI_TABLE_PATH}')

    dyads = {}
    for residue_s, row in data.get('dyad_residue', {}).items():
        residue = int(residue_s)
        ratio = str(row['ratio'])
        dyads[residue] = {'ratio': ratio, 'prime_mask': int(_ratio_prime_mask(ratio))}

    absolute_dyads = {}
    for key, item in data.get('2', {}).items():
        parts = tuple(int(x) for x in key.split(','))
        if len(parts) != 2 or parts[0] != 0:
            continue
        mask = 0
        for n in item.get('integers', ()):
            mask |= _integer_prime_mask(int(n))
        absolute_dyads[int(parts[1])] = {
            'prime_mask': int(mask),
            'ratio': str(item.get('ratios_from_lowest', ['1/1', ''])[1]),
        }

    group_masks = {3: {}, 4: {}, 5: {}}
    for cardinality in (3, 4, 5):
        for key, item in data.get(str(cardinality), {}).items():
            mask = 0
            for n in item.get('integers', ()):
                mask |= _integer_prime_mask(int(n))
            group_masks[cardinality][key] = int(mask)
    return {
        'path': str(_ACTIVE_JI_TABLE_PATH),
        'dyads': dyads,
        'absolute_dyads': absolute_dyads,
        'group_masks': group_masks,
    }


@lru_cache(maxsize=2048)
def interval_prime_mask(step_difference):
    """Prime mask of one absolute ScaleWeaver interval from the attack-JI table.

    ``step_difference`` may be compound.  A nonzero octave component contributes
    prime 2, while the residue uses the JSON-derived canonical dyad ratio.
    """
    diff = abs(int(step_difference))
    if diff == 0:
        return 0
    index = _load_attack_ji_dyad_index()
    exact = index.get('absolute_dyads', {}).get(diff)
    if exact is not None:
        return int(exact['prime_mask'])

    octaves, residue = divmod(diff, int(OCT))
    mask = _prime_bit(2) if octaves else 0
    if residue:
        item = index['dyads'].get(int(residue))
        if item is None:
            raise KeyError(
                f'no patent-val canonical dyad ratio for {residue} / {OCT} octave')
        mask |= int(item['prime_mask'])
    return int(mask)


def _sonority_prime_mask(steps):
    """Prime-presence mask of the whole actual sonority.

    For 3/4/5 notes this is the prime content of the JSON table's canonical
    gcd-reduced integer tuple, exactly matching the attack-analysis notion of
    "this ratio contains prime p".  Dyads use the JSON-derived pairwise ratio.
    If a rare wide-register 3/4/5-note voicing is outside the precomputed top-level
    key range, the union of its JSON-derived dyad masks is used as a conservative
    fallback.  One note has no interval and therefore mask zero.
    """
    xs = tuple(sorted(int(x) for x in steps))
    n = len(xs)
    if n <= 1:
        return 0
    if n == 2:
        return interval_prime_mask(xs[1] - xs[0])
    if n in (3, 4, 5):
        offsets = tuple(x - xs[0] for x in xs)
        key = ','.join(str(x) for x in offsets)
        item = _load_attack_ji_dyad_index()['group_masks'][n].get(key)
        if item is not None:
            return int(item)
    mask = 0
    for a, b in combinations(xs, 2):
        mask |= interval_prime_mask(abs(int(a) - int(b)))
    return int(mask)


def _sonority_prime_reward(steps, prime_rewards):
    """Sum each configured prime reward once when the whole ratio contains it."""
    mask = _sonority_prime_mask(steps)
    weights = _prime_reward_weights(prime_rewards)
    return sum(weights[p] for p in ATTACK_JI_PRIMES
               if mask & _prime_bit(p))


def _sounding_prime_reward_interval(cand, existing, start, dur, prime_rewards):
    """Duration-weighted prime reward of all notes actually sounding with cand."""
    cand = int(cand)
    start = float(start)
    end = start + float(dur)
    ovs = [e for xs in existing.values() for e in overlaps(xs, start, end)]
    points = {start, end}
    for e in ovs:
        points.add(max(start, float(e['start_beat'])))
        points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    weighted = total = 0.0
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        active = [int(e['step']) for e in ovs
                  if float(e['start_beat']) < mid + 1e-9
                  and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
        span = b - a
        weighted += span * _sonority_prime_reward((cand, *active), prime_rewards)
        total += span
    return weighted / total if total > 1e-9 else 0.0


def attack_aware_prime_reward_interval(cand, existing, start, dur, prime_rewards):
    """Prime-colour analogue of ``attack_aware_vertical_cse_interval``.

    The same three real contexts are used:
      - sounding: all notes actually sounding throughout candidate duration,
      - simultaneous: only notes that truly attack with the candidate,
      - future_attack: full sounding set immediately after each later attack.

    Each context scores prime PRESENCE in its whole canonical ratio: every prime
    can contribute its configured reward at most once per view/context.  The three
    views use ``COUNTERPOINT_ATTACK_CSE_VIEW_WEIGHTS`` so prime colour follows
    exactly the same attack emphasis as CSE.  Positive reward is returned here;
    callers subtract it from candidate cost.
    """
    if not _prime_rewards_enabled(prime_rewards):
        return 0.0, {'views': {}, 'contexts': None}

    cand = int(cand)
    sounding = _sounding_prime_reward_interval(cand, existing, start, dur, prime_rewards)
    contexts = _attack_cse_contexts(cand, existing, start, dur)

    # A lone true attack contains no interval and therefore contributes 0 rather
    # than disappearing from the simultaneous view.
    simultaneous_group = contexts['simultaneous'] or (cand,)
    simultaneous = _sonority_prime_reward(simultaneous_group, prime_rewards)

    future_values = [_sonority_prime_reward(group, prime_rewards)
                     for group in contexts['future_attack']]
    future_attack = (sum(future_values) / len(future_values)
                     if future_values else None)

    raw = {
        'sounding': sounding,
        'simultaneous': simultaneous,
        'future_attack': future_attack,
    }
    weighted = total_weight = 0.0
    used = {}
    for key, value in raw.items():
        if value is None:
            continue
        weight = float(COUNTERPOINT_ATTACK_CSE_VIEW_WEIGHTS.get(key, 1.0))
        if weight <= 0:
            continue
        weighted += weight * float(value)
        total_weight += weight
        used[key] = float(value)
    combined = weighted / total_weight if total_weight > 0 else 0.0
    return combined, {
        'views': used,
        'contexts': contexts,
        'future_attack_values': tuple(future_values),
        'weights': _prime_reward_weights(prime_rewards),
        'ji_table': _load_attack_ji_dyad_index()['path'],
    }


def attack_aware_prime_reward_cost(cand, existing, start, dur, prime_rewards):
    """Convert positive prime reward into a negative candidate cost.

    This must never silently short-circuit to zero when nonzero prime rewards
    are configured.  ``attack_aware_prime_reward_interval`` returns a reward,
    while the score builder minimizes cost, hence the minus sign.
    """
    reward, _details = attack_aware_prime_reward_interval(
        cand, existing, start, dur, prime_rewards
    )
    return -float(reward)


def hard_wolf(a, b):
    return hr.hard_wolf(a, b)


def step_second(a, b):
    d = abs(int(a) - int(b))
    return 0 < d <= hr.SECOND_MAX_STEP


def _effective_pool(voice):
    key = 'counter' if str(voice).startswith('counter') else str(voice)
    lo, hi = COUNTERPOINT_EFFECTIVE_RANGES[key]
    return tuple(int(p) for p in POOLS[key] if lo <= int(p) <= hi)

def _projected_quartet_legal(quartet):
    bass, inner, counter, lead = map(int, quartet)
    if not (bass < inner < counter < lead):
        return False
    for a, b in combinations((bass, inner, counter, lead), 2):
        if hard_wolf(a, b) or step_second(a, b):
            return False
    return True

@lru_cache(None)
def _three_note_relative_floor(steps):
    """Lowest raw 2-note CSE after deleting one tone from a 3-note sonority.

    Higher is better: if one deletion leaves an extremely pure dyad, the third
    tone is comparatively weakly integrated and the floor becomes low.
    """
    xs = tuple(sorted(int(x) for x in steps))
    if len(xs) != 3:
        return None
    vals = []
    for i in range(3):
        sub = xs[:i] + xs[i + 1:]
        raw = _absolute_cse_score(sub, default=None)
        if raw is not None:
            vals.append(float(raw))
    return min(vals) if len(vals) == 3 else None

@lru_cache(None)
def _four_note_relative_floor(steps):
    """Lowest raw 3-note CSE after deleting one tone from a quartet.

    This is the relative-purity quantity requested by the generator: a quartet
    is better integrated when *every* one-tone deletion leaves a comparatively
    high-CSE triad.  A chord such as 7:14:21:48 therefore receives little reward
    when deleting 48 exposes a much purer 7:14:21 core.
    """
    xs = tuple(sorted(int(x) for x in steps))
    if len(xs) != 4:
        return None
    vals = []
    for i in range(4):
        sub = xs[:i] + xs[i + 1:]
        raw = _absolute_cse_score(sub, default=None)
        if raw is not None:
            vals.append(float(raw))
    return min(vals) if len(vals) == 4 else None

def _quartet_objective(raw, quartet, raw_weight, relative_weight, octave_weight):
    rel = _four_note_relative_floor(tuple(quartet))
    relative_term = 0.0 if rel is None else -float(relative_weight) * float(rel)
    return (float(raw_weight) * float(raw)
            + relative_term
            + float(octave_weight) * max(0, _octave_family_pair_count(quartet) - 1))

def _lookahead_quartet_legal(quartet, max_octave_family_pairs=None):
    """Legality for optimistic four-part CSE completion.

    The realised generator keeps octave-family use as a *soft* musical rule, but
    the look-ahead lower bound must not collapse into a 1:2:4:8-like solution.
    Otherwise a huge raw-CSE weight would steer Counter/Bass toward a fantasy
    completion that the anti-doubling objective later tries to undo.  Restrict
    the projected optimum to at most one octave-family pair (2/1, 4/1 or
    8/1) per simultaneous quartet.  This is a constraint on the *forecasted
    optimum only*; the realised score still uses the editable soft penalty.
    """
    if max_octave_family_pairs is None:
        max_octave_family_pairs = COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS
    return (_projected_quartet_legal(quartet)
            and _octave_family_pair_count(quartet) <= int(max_octave_family_pairs))

@lru_cache(None)
def _best_completion_counter_lead(counter, lead, raw_weight, relative_weight, octave_weight, max_oct_pairs):
    """Best legal future quartet objective for a fixed Counter+Lead pair."""
    counter, lead = int(counter), int(lead)
    best = math.inf
    for bass in _effective_pool('bass'):
        if bass >= counter:
            continue
        for inner in _effective_pool('inner'):
            q = (bass, inner, counter, lead)
            if not _lookahead_quartet_legal(q, max_oct_pairs):
                continue
            raw = _absolute_cse_score(q, default=None)
            if raw is None:
                continue
            best = min(best, _quartet_objective(raw, q, raw_weight, relative_weight, octave_weight))
    return None if math.isinf(best) else best

@lru_cache(None)
def _best_completion_bass_counter_lead(bass, counter, lead, raw_weight, relative_weight, octave_weight, max_oct_pairs):
    """Best legal future quartet objective for fixed Bass+Counter+Lead."""
    bass, counter, lead = int(bass), int(counter), int(lead)
    best = math.inf
    for inner in _effective_pool('inner'):
        q = (bass, inner, counter, lead)
        if not _lookahead_quartet_legal(q, max_oct_pairs):
            continue
        raw = _absolute_cse_score(q, default=None)
        if raw is None:
            continue
        best = min(best, _quartet_objective(raw, q, raw_weight, relative_weight, octave_weight))
    return None if math.isinf(best) else best

def _projected_four_part_objective_interval(voice, cand, existing, start, dur):
    """Duration-weighted future quartet objective for Counter/Bass choice.

    This is deliberately a lower-bound/look-ahead score, not a fake realised
    quartet.  Missing lower parts are enumerated inside their effective ranges.
    The exact same raw CSE and >1-octave-family soft penalty used by Inner are
    used in the projection, so increasing the public weight changes earlier
    accompaniment decisions as well.
    """
    role = 'counter' if str(voice).startswith('counter') else str(voice)
    if role not in ('counter', 'bass'):
        return None, 0.0
    end = float(start) + float(dur)
    relevant = {v: xs for v, xs in existing.items() if v in _VOICE_RANK}
    points = {float(start), end}
    for xs in relevant.values():
        for e in overlaps(xs, start, end):
            points.add(max(float(start), float(e['start_beat'])))
            points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    weighted = covered = 0.0
    rw = float(COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT)
    relw = float(COUNTERPOINT_RELATIVE_PURITY_WEIGHT)
    ow = float(COUNTERPOINT_EXTRA_OCTAVE_FAMILY_PAIR_COST)
    mop = int(COUNTERPOINT_LOOKAHEAD_MAX_OCTAVE_FAMILY_PAIRS)
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        active = {}
        for v, xs in relevant.items():
            act = [e for e in xs if float(e['start_beat']) < mid + 1e-9
                   and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
            if act:
                active[_VOICE_RANK[v]] = int(act[0]['step'])
        span = b - a
        projected = None
        if role == 'counter':
            lead = active.get(_VOICE_RANK['lead'])
            if lead is not None:
                projected = _best_completion_counter_lead(int(cand), lead, rw, relw, ow, mop)
        else:  # bass
            lead = active.get(_VOICE_RANK['lead'])
            counter = active.get(_VOICE_RANK['counter'])
            if lead is not None and counter is not None:
                projected = _best_completion_bass_counter_lead(int(cand), counter, lead, rw, relw, ow, mop)
        if projected is None:
            continue
        weighted += span * float(projected)
        covered += span
    if not covered:
        return None, 0.0
    return weighted / covered, covered / max(1e-9, float(dur))

@lru_cache(None)
def _lead_chord_field_cse_cached(chord_id, cand):
    chord = CHORDS[chord_id]
    background = tuple(hr.chord_reference_steps(chord))
    return _absolute_cse_score(background + (int(cand),), default=2.4)

def lead_chord_field_cse(chord, cand):
    """Raw corrected CSE of candidate Lead + the complete current chord field."""
    return _lead_chord_field_cse_cached(chord.id, int(cand))

@lru_cache(None)
def _absolute_cse_cached(steps, blend_a, blend_b, blend_c):
    """Cached unified lookup keyed by sonority and all three coefficients."""
    xs = tuple(sorted(map(int, steps)))
    n = len(xs)
    if hr.CSE is None or not 2 <= n <= 5: return None

    def lookup(pitches):
        return hr.CSE.mixed_entry(pitches, blend_a, blend_b, blend_c,
                                 table_key='inner5' if n == 5 else f'bass{n}')

    item = lookup(xs)
    if item is not None or n != 5: return item
    # Existing five-note background-field projection; every metric uses the
    # same projected register. Actual sounding 2..4-tone scores are never shifted.
    for shift_oct in range(-4, 5):
        if shift_oct == 0: continue
        item = lookup(tuple(x+shift_oct*OCT for x in xs))
        if item is not None: return item
    return None


def absolute_cse_entry(steps):
    return _absolute_cse_cached(tuple(sorted(map(int, steps))),
                                float(CSE_2D_A), float(CSE_2D_B), float(CSE_2D_C))


def simultaneous_attack_soft_limit(cand, existing, start, eps=1e-9):
    """Inspect the six blend-percentile standards for the resulting attack.

    Only exact-onset attacks participate; sustained notes are deliberately not
    included. The applicable dyad/triad/tetrad subsets depend on whether the
    complete attack has two, three, or four notes. This function never declares
    a pitch illegal.
    """
    config = ATTACK_CSE_PERCENTILE_LIMITS
    if not config:
        return {'enabled':False,'violations':0,'max_excess':0.0,
                'sum_excess':0.0,'checks':()}
    others = []
    for events in existing.values():
        for event in events:
            if abs(float(event['start_beat'])-float(start)) <= float(eps):
                others.append(int(event['step']))
    group = (int(cand),)+tuple(others)
    attack_card = len(group)
    subset_limits = config['limits'].get(attack_card,{})
    checks = []
    for card,limit in sorted(subset_limits.items()):
        for combo in combinations(group,card):
            steps = tuple(sorted(combo))
            item = absolute_cse_entry(steps)
            if item is None:
                continue
            percentile = float(item[1])
            checks.append({'attack_cardinality':attack_card,
                           'subset_cardinality':card,'steps':steps,
                           'raw':float(item[0]),'percentile':percentile,
                           'limit':float(limit),
                           'excess':max(0.0,percentile-float(limit))})
    excesses = [row['excess'] for row in checks if row['excess'] > 1e-12]
    return {
        'enabled':True,
        'violations':len(excesses),
        'max_excess':max(excesses,default=0.0),
        'sum_excess':sum(excesses),
        'checks':tuple(checks),
    }


def apply_simultaneous_attack_soft_limits(scored, fallback_slack=None):
    """Apply a relaxable percentile gate without ever emptying candidates.

    ``scored`` rows are ``(ordinary_cost, pitch, limit_diagnostics)``.  If at
    least one candidate respects every configured limit, only such candidates
    enter stochastic selection. Otherwise the lowest-excess layer plus the
    configured percentile slack is retained, with a smooth cost inside it.
    Thus the limit is strict when feasible, soft when unavoidable, and cannot
    itself cause generation failure.
    """
    if not scored:
        return []
    safe = [row for row in scored if int(row[2].get('violations',0)) == 0]
    if safe:
        return [(float(cost),pitch) for cost,pitch,_ in safe]
    if fallback_slack is None:
        fallback_slack = (ATTACK_CSE_PERCENTILE_LIMITS or {}).get(
            'fallback_slack',0.10)
    minimum = min(float(diag.get('max_excess',0.0)) for _,_,diag in scored)
    ceiling = minimum + max(0.0,float(fallback_slack)) + 1e-12
    out = []
    for cost,pitch,diag in scored:
        max_excess = float(diag.get('max_excess',0.0))
        if max_excess > ceiling:
            continue
        # Musical costs still decide among near-minimal relaxations, while a
        # steep continuous term favours the smallest total overshoot.
        excess_cost = 40.0*(max_excess-minimum) + 12.0*float(diag.get('sum_excess',0.0))
        out.append((float(cost)+excess_cost,pitch))
    # Defensive fallback: numerical surprises must never erase a non-empty set.
    return out or [(float(cost),pitch) for cost,pitch,_ in scored]


def simultaneous_attack_limit_summary(voices, example_limit=12):
    """Audit every dyad/triad/tetrad inside every exact-onset attack group."""
    config = ATTACK_CSE_PERCENTILE_LIMITS
    limits = config.get('limits',{}) if config else {}
    names = {
        (2,2):'two_note_attack.dyad',
        (3,2):'three_note_attack.dyad_subsets',
        (4,2):'four_note_attack.dyad_subsets',
        (3,3):'three_note_attack.triad',
        (4,3):'four_note_attack.triad_subsets',
        (4,4):'four_note_attack.tetrad',
    }
    summary = {
        'enabled':bool(limits),
        'metric':'a*CSE + b*SE + c*CCSE',
        'weights':{'a':float(CSE_2D_A),'b':float(CSE_2D_B),'c':float(CSE_2D_C)},
        'percentile_limits':{
            names[(attack_card,subset_card)]:float(limit)
            for attack_card,rows in sorted(limits.items())
            for subset_card,limit in sorted(rows.items())
        },
        'fallback_slack':float(config.get('fallback_slack',0.10)) if config else None,
        'soft_policy':'prefer fully compliant candidates; if none exist, retain the minimum-excess layer within configured fallback_slack',
        'by_standard':{},
        'violations':[],
    }
    if not limits:
        return summary
    attacks = {}
    for voice,events in voices.items():
        for event in events:
            key = round(float(event['start_beat']),9)
            attacks.setdefault(key,[]).append((str(voice),int(event['step'])))
    for attack_card,subset_limits in sorted(limits.items()):
        for card,limit in sorted(subset_limits.items()):
            standard = names[(attack_card,card)]
            limit = float(limit)
            checked = exceeded = 0
            max_percentile = max_excess = 0.0
            for beat,group in sorted(attacks.items()):
                if len(group) != attack_card:
                    continue
                for subset in combinations(group,card):
                    item = absolute_cse_entry(tuple(step for _,step in subset))
                    if item is None:
                        continue
                    checked += 1
                    percentile = float(item[1])
                    excess = max(0.0,percentile-limit)
                    max_percentile = max(max_percentile,percentile)
                    max_excess = max(max_excess,excess)
                    if excess > 1e-12:
                        exceeded += 1
                        if len(summary['violations']) < int(example_limit):
                            summary['violations'].append({
                                'beat':beat,'attack_cardinality':attack_card,
                                'subset_cardinality':card,
                                'voices':[voice for voice,_ in subset],
                                'steps':[step for _,step in subset],
                                'raw':float(item[0]),'percentile':percentile,
                                'limit':limit,'excess':excess,
                            })
            summary['by_standard'][standard] = {
                'checked':checked,'exceeded':exceeded,
                'max_percentile':max_percentile if checked else None,
                'max_excess':max_excess if checked else None,
            }
    summary['total_checked'] = sum(row['checked'] for row in summary['by_standard'].values())
    summary['total_exceeded'] = sum(row['exceeded'] for row in summary['by_standard'].values())
    return summary


def _absolute_cse_score(steps, default=None):
    item = absolute_cse_entry(steps)
    return float(item[0]) if item is not None else default

def _target_cse(cand, others):
    others = list(map(int, others))
    for n in range(min(5, len(others)+1), 1, -1):
        vals = []
        for combo in set(combinations(others, n-1)):
            item = absolute_cse_entry((int(cand),) + combo)
            if item is not None: vals.append(float(item[0]))
        if vals: return sum(vals) / len(vals)
    return None

def _group_cse(group, others):
    """Raw corrected CSE for a simultaneous candidate group in real register.

    Every pitch in ``group`` is forced into the lookup.  When the surrounding
    texture contains more notes than the 2..5-note table supports, we average
    the largest scorable target-containing subsets.  Repeated pitches are kept
    because the CSE table is multiset-aware.
    """
    group = tuple(map(int, group))
    others = list(map(int, others))
    if not 1 <= len(group) <= 5:
        return None
    max_other = min(len(others), 5 - len(group))
    min_other = 0 if len(group) >= 2 else 1
    for r in range(max_other, min_other - 1, -1):
        vals = []
        seen = set()
        for combo in combinations(others, r):
            key = tuple(sorted(group + tuple(combo)))
            if key in seen or not 2 <= len(key) <= 5:
                continue
            seen.add(key)
            item = absolute_cse_entry(key)
            if item is not None:
                vals.append(float(item[0]))
        if vals:
            return sum(vals) / len(vals)
    return None

def _vertical_group_cse(group, existing, start, dur):
    """Duration-weighted CSE of a block voicing against changing other voices."""
    end = start + dur
    ovs = [e for xs in existing.values() for e in overlaps(xs, start, end)]
    points = {start, end}
    for e in ovs:
        points.add(max(start, float(e['start_beat'])))
        points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    weighted = covered = 0.0
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = (a + b) / 2
        others = [
            e['step'] for e in ovs
            if e['start_beat'] < mid + 1e-9
            and e['start_beat'] + e['duration_beats'] > mid - 1e-9
        ]
        value = _group_cse(group, others)
        if value is not None:
            weighted += (b - a) * value
            covered += b - a
    return (weighted / covered if covered else None, covered / max(1e-9, dur))

def absolute_vertical_cse_interval(cand, existing, start, dur):
    end = start + dur
    ovs = [e for xs in existing.values() for e in overlaps(xs, start, end)]
    points = sorted({start, end, *[max(start, e['start_beat']) for e in ovs], *[min(end, e['start_beat']+e['duration_beats']) for e in ovs]})
    weighted = covered = 0.0
    for a, b in zip(points, points[1:]):
        mid = (a+b)/2
        others = [e['step'] for e in ovs if e['start_beat'] < mid + 1e-9 and e['start_beat']+e['duration_beats'] > mid - 1e-9]
        v = _target_cse(cand, others)
        if v is not None: weighted += (b-a)*v; covered += b-a
    return (weighted/covered if covered else None, covered/max(1e-9, dur))

def _attack_cse_contexts(cand, existing, start, dur):
    """Return the concrete attack-sensitive vertical contexts for ``cand``.

    The candidate itself attacks at ``start`` and sustains until ``start+dur``.

    ``simultaneous`` contains the candidate plus only notes whose own onset is
    exactly ``start``.  Already-sustaining notes are intentionally excluded.

    ``future_attack`` contains one full sounding-set context for every distinct
    onset strictly inside the candidate's sustain.  At such a later onset,
    already-sustaining notes ARE included.  Therefore, in the user's example

        Lead: 4 + 1 + 2 + 1
        Accompaniment: 2 + 2 + 2 + 2
        Bass: 8

    the third accompaniment note sees:
        - ordinary sounding CSE: Bass + Lead2 + candidate (then Lead3 later),
        - simultaneous CSE: Lead2 + candidate,
        - future-attack CSE: Bass + Lead3 + candidate.
    """
    cand = int(cand)
    start = float(start)
    end = start + float(dur)
    eps = 1e-9

    events = []
    for xs in existing.values():
        for e in overlaps(xs, start, end):
            events.append(e)

    simultaneous_others = [
        int(e['step']) for e in events
        if abs(float(e['start_beat']) - start) <= eps
    ]
    simultaneous = ((cand,) + tuple(simultaneous_others)
                    if simultaneous_others else ())

    future_times = sorted({
        float(e['start_beat']) for e in events
        if start + eps < float(e['start_beat']) < end - eps
    })
    future = []
    for t in future_times:
        active = [
            int(e['step']) for e in events
            if float(e['start_beat']) <= t + eps
            and float(e['start_beat']) + float(e['duration_beats']) > t + eps
        ]
        if active:
            future.append((cand,) + tuple(active))
    return {
        'simultaneous': simultaneous,
        'future_attack': tuple(future),
        'future_attack_times': tuple(future_times),
    }

def attack_aware_vertical_cse_interval(cand, existing, start, dur):
    """Blend sounding, simultaneous-attack and later-attack CSE views.

    The old duration-weighted sounding CSE is retained as one view.  Two
    attack-specific views are added without changing the overall numeric scale:
    available view scores are averaged using
    ``COUNTERPOINT_ATTACK_CSE_VIEW_WEIGHTS``.

    Returns ``(combined, coverage, details)``.  ``details`` is intentionally
    explicit so diagnostics/tests can verify exactly which attack contexts were
    scored.
    """
    sounding, coverage = absolute_vertical_cse_interval(
        cand, existing, start, dur)
    contexts = _attack_cse_contexts(cand, existing, start, dur)

    simultaneous = None
    if contexts['simultaneous']:
        simultaneous = _target_cse(
            int(cand), contexts['simultaneous'][1:])

    future_values = []
    for group in contexts['future_attack']:
        value = _target_cse(int(cand), group[1:])
        if value is not None:
            future_values.append(float(value))
    future_attack = (sum(future_values) / len(future_values)
                     if future_values else None)

    raw = {
        'sounding': sounding,
        'simultaneous': simultaneous,
        'future_attack': future_attack,
    }
    weighted = total_weight = 0.0
    used = {}
    for key, value in raw.items():
        if value is None:
            continue
        weight = float(COUNTERPOINT_ATTACK_CSE_VIEW_WEIGHTS.get(key, 1.0))
        if weight <= 0:
            continue
        weighted += weight * float(value)
        total_weight += weight
        used[key] = float(value)

    combined = weighted / total_weight if total_weight > 0 else None
    details = {
        'views': used,
        'contexts': contexts,
        'future_attack_values': tuple(future_values),
    }
    return combined, coverage, details

def _octave_family_pair_count(steps):
    xs = [int(x) for x in steps]
    return sum(abs(a - b) in (OCT, 2 * OCT, 3 * OCT)
               for a, b in combinations(xs, 2))

def _partial_octave_family_excess_interval(cand, existing, start, dur):
    """Soft anti-collapse cost for any currently realised vertical slice."""
    end = float(start) + float(dur)
    ovs = [e for v, xs in existing.items() if v in _VOICE_RANK
           for e in overlaps(xs, start, end)]
    points = {float(start), end}
    for e in ovs:
        points.add(max(float(start), float(e['start_beat'])))
        points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    weighted = total = 0.0
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        xs = [int(cand)] + [int(e['step']) for e in ovs
                            if float(e['start_beat']) < mid + 1e-9
                            and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
        span = b - a
        weighted += span * max(0, _octave_family_pair_count(xs) - 1)
        total += span
    return weighted / max(1e-9, total)

def _four_part_raw_cse_interval(voice, cand, existing, start, dur):
    """Duration-weighted quartet raw CSE and deletion-floor relative purity.

    Returns (raw4, relative_floor3, coverage, octave_excess).  Lower raw4 is
    better; higher relative_floor3 is better.
    """
    end = float(start) + float(dur)
    relevant = {v: xs for v, xs in existing.items() if v in _VOICE_RANK}
    points = {float(start), end}
    for xs in relevant.values():
        for e in overlaps(xs, start, end):
            points.add(max(float(start), float(e['start_beat'])))
            points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    raw_weighted = relative_weighted = covered = octave_excess_weighted = 0.0
    relative_covered = 0.0
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        active_by_rank = {}
        ok = True
        for v, xs in relevant.items():
            act = [e for e in xs if float(e['start_beat']) < mid + 1e-9
                   and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
            if not act:
                continue
            r = _VOICE_RANK[v]
            if r in active_by_rank:
                ok = False
                break
            active_by_rank[r] = int(act[0]['step'])
        active_by_rank[_VOICE_RANK[voice]] = int(cand)
        if not ok or set(active_by_rank) != {0, 1, 2, 3}:
            continue
        quartet = tuple(active_by_rank[r] for r in (0, 1, 2, 3))
        raw = _absolute_cse_score(quartet, default=None)
        if raw is None:
            continue
        rel = _four_note_relative_floor(tuple(quartet))
        span = b - a
        raw_weighted += span * float(raw)
        covered += span
        if rel is not None:
            relative_weighted += span * float(rel)
            relative_covered += span
        octave_excess_weighted += span * max(0, _octave_family_pair_count(quartet) - 1)
    if not covered:
        return None, None, 0.0, 0.0
    rel_mean = relative_weighted / relative_covered if relative_covered > 1e-9 else None
    return (raw_weighted / covered, rel_mean,
            covered / max(1e-9, float(dur)), octave_excess_weighted / covered)

def _three_part_relative_floor_interval(voice, cand, existing, start, dur):
    """Relative purity of realised 3-part slices using the full-range 2-note table.

    This is primarily active while Bass is chosen (Lead+Counter already exist).
    Higher is better, so callers subtract it from the candidate cost.
    """
    end = float(start) + float(dur)
    relevant = {v: xs for v, xs in existing.items() if v in _VOICE_RANK}
    points = {float(start), end}
    for xs in relevant.values():
        for e in overlaps(xs, start, end):
            points.add(max(float(start), float(e['start_beat'])))
            points.add(min(end, float(e['start_beat']) + float(e['duration_beats'])))
    points = sorted(points)
    weighted = covered = 0.0
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = .5 * (a + b)
        by_rank = {}
        ok = True
        for v, xs in relevant.items():
            act = [e for e in xs if float(e['start_beat']) < mid + 1e-9
                   and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9]
            if not act:
                continue
            r = _VOICE_RANK[v]
            if r in by_rank:
                ok = False
                break
            by_rank[r] = int(act[0]['step'])
        by_rank[_VOICE_RANK[voice]] = int(cand)
        if not ok or len(by_rank) != 3:
            continue
        triple = tuple(by_rank[r] for r in sorted(by_rank))
        rel = _three_note_relative_floor(tuple(triple))
        if rel is None:
            continue
        span = b - a
        weighted += span * float(rel)
        covered += span
    return (weighted / covered, covered / max(1e-9, float(dur))) if covered else (None, 0.0)

# ---------------------------------------------------------------------------
# Weighted-cost API used by score_builder.  Weight arithmetic lives here so
# score_builder is responsible only for non-CSE musical rules and candidate
# selection.
# ---------------------------------------------------------------------------

def lead_chord_field_cost(chord, cand, strong):
    cse = lead_chord_field_cse(chord, cand)
    field_w = (LEAD_CHORD_FIELD_CSE_STRONG_WEIGHT
               if strong else LEAD_CHORD_FIELD_CSE_WEAK_WEIGHT)
    return hr._CSE_STRENGTH * field_w * (cse - LEAD_CHORD_FIELD_CSE_CENTER)



def counterpoint_post_melodic_cost_components(voice, cand, chord, existing,
                                                start, dur, prime_rewards):
    """Return weighted spectral costs after the melodic terms."""
    key = 'counter' if str(voice).startswith('counter') else str(voice)
    bg_w = COUNTERPOINT_BACKGROUND_CSE_WEIGHT[key]
    actual_w = COUNTERPOINT_ACTUAL_CSE_WEIGHT[key]
    out = []

    # Virtual full-chord background field.
    field = lead_chord_field_cse(chord, cand)
    duration_scale = .78 + .28 * min(1.0, max(0.0, float(dur) - .5) / 1.5)
    out.append(hr._CSE_STRENGTH * bg_w * duration_scale * field)

    # Actual sounding + simultaneous attack + later attack views.
    actual, coverage, _attack_details = attack_aware_vertical_cse_interval(
        cand, existing, start, dur)
    if actual is not None and coverage > 0:
        out.append(hr._CSE_STRENGTH * actual_w * actual)

    # Same prime-colour reward as the fixed-melody harmonizer path.  This is
    # based only on real sounding/attack contexts, never on the virtual chord
    # field or optimistic missing-voice completion.
    prime_cost = attack_aware_prime_reward_cost(cand, existing, start, dur, prime_rewards)
    if prime_cost:
        out.append(prime_cost)

    # Soft anti-collapse penalty.
    partial_octave_excess = _partial_octave_family_excess_interval(
        cand, existing, start, dur)
    out.append(hr._CSE_STRENGTH * COUNTERPOINT_PARTIAL_OCTAVE_FAMILY_PAIR_COST
               * partial_octave_excess)

    # Four-part optimistic look-ahead for Counter/Bass.
    projected_q, projected_cov = _projected_four_part_objective_interval(
        voice, cand, existing, start, dur)
    if projected_q is not None and projected_cov > 0:
        out.append(hr._CSE_STRENGTH * projected_q)

    # Three-part deletion-floor relative purity.
    triple_rel, triple_cov = _three_part_relative_floor_interval(
        voice, cand, existing, start, dur)
    if triple_rel is not None and triple_cov > 0:
        out.append(-hr._CSE_STRENGTH * COUNTERPOINT_RELATIVE_PURITY_WEIGHT
                   * triple_rel)

    # Exact realised quartet terms when all four voices are present.
    quartet_raw, quartet_rel, quartet_cov, octave_excess = _four_part_raw_cse_interval(
        voice, cand, existing, start, dur)
    if quartet_raw is not None and quartet_cov > 0:
        out.append(hr._CSE_STRENGTH * COUNTERPOINT_FOUR_PART_RAW_CSE_WEIGHT
                   * quartet_raw)
        if quartet_rel is not None:
            out.append(-hr._CSE_STRENGTH * COUNTERPOINT_RELATIVE_PURITY_WEIGHT
                       * quartet_rel)
        out.append(hr._CSE_STRENGTH * COUNTERPOINT_EXTRA_OCTAVE_FAMILY_PAIR_COST
                   * octave_excess)

    return tuple(out)
