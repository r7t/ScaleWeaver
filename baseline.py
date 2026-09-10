#!/usr/bin/env python3
"""Lead-conditioned four-part CSE baselines for the ScaleWeaver generator.

Two baselines are computed against the dedicated absolute Bass-to-Lead four-note
raw-CSE table:

1) Enumerated optimum (local vertical lower bound)
   Keep every realised Lead pitch and its duration.  For each Lead pitch,
   enumerate every hard-legal Bass/Inner/secondary triple.  Split legal quartets
   into those containing at least one exact 2/1 pair (one octave) and those with no
   exact 2/1 pair, take the minimum raw CSE in each class, and linearly mix the two
   class means using the real score's duration-weighted exact-2/1 slice rate.

2) Random lower-voice reference
   Keep the same realised Lead pitches/durations, randomly select legal
   Bass/Inner/secondary triples from the same two exact-2/1 classes, and use the
   same linear mixture.  Multiple Monte-Carlo samples estimate the random mean.

The resulting values give a useful bracket for the realised score.  Because lower
raw CSE is better, ``bounded_cse_percentile_low_is_good`` is 0% at the enumerated
local optimum and 100% at the random reference.  Its complement
``fraction_of_random_to_optimal_gap_closed_percent`` is 100% at the optimum and
0% at random.

The random reference is an empirical reference, not a mathematical worst-case
upper bound.  The enumerated optimum *is* a local vertical lower bound under the
same Lead, per-voice ranges, strict voice ordering, hard-wolf and step-second
rules, but deliberately ignores horizontal counterpoint continuity.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from itertools import combinations, product
from pathlib import Path

import harmony_rhythm as hr
import score_builder as sb

VOICE_RANK = {'bass': 0, 'inner': 1, 'counter': 2, 'lead': 3}


def four_voice_vertical_segments(voices):
    """Return (start, end, quartet) slices containing one note at each rank.

    Quartet order is always Bass, Inner, secondary voice, Lead.  The secondary
    the secondary voice is the independent counter line.
    """
    ranked = {}
    for voice, xs in voices.items():
        rank = VOICE_RANK.get(voice)
        if rank is not None and xs:
            ranked.setdefault(rank, []).append((voice, xs))

    points = set()
    for groups in ranked.values():
        for _, xs in groups:
            for e in xs:
                points.add(float(e['start_beat']))
                points.add(float(e['start_beat']) + float(e['duration_beats']))
    points = sorted(points)

    rows = []
    for a, b in zip(points, points[1:]):
        if b <= a + 1e-9:
            continue
        mid = 0.5 * (a + b)
        quartet = []
        ok = True
        for rank in (0, 1, 2, 3):
            active = []
            for _, xs in ranked.get(rank, []):
                active.extend(
                    e for e in xs
                    if float(e['start_beat']) < mid + 1e-9
                    and float(e['start_beat']) + float(e['duration_beats']) > mid - 1e-9
                )
            if len(active) != 1:
                ok = False
                break
            quartet.append(int(active[0]['step']))
        if ok:
            rows.append((a, b, tuple(quartet)))
    return rows


def exact_2_1_pair_count(steps):
    xs = tuple(int(x) for x in steps)
    return sum(abs(a - b) == hr.OCT for a, b in combinations(xs, 2))


def octave_family_pair_counts(steps):
    xs = tuple(int(x) for x in steps)
    return {
        '2/1': sum(abs(a - b) == hr.OCT for a, b in combinations(xs, 2)),
        '4/1': sum(abs(a - b) == 2 * hr.OCT for a, b in combinations(xs, 2)),
        '8/1': sum(abs(a - b) == 3 * hr.OCT for a, b in combinations(xs, 2)),
    }


def has_exact_2_1(steps):
    return exact_2_1_pair_count(steps) > 0


def four_part_raw_cse_summary(voices, include_segments=False):
    """Duration-weighted realised four-part raw CSE plus exact 2/1-count diagnostics."""
    rows = four_voice_vertical_segments(voices)
    weighted = total = 0.0
    pair_duration = {'2/1': 0.0, '4/1': 0.0, '8/1': 0.0}
    exact_count_duration = {}
    extra_octave = 0.0
    scored = 0

    for a, b, quartet in rows:
        dur = b - a
        raw = hr.cse4_score(quartet, default=None)
        if raw is None:
            continue
        scored += 1
        weighted += dur * float(raw)
        total += dur
        counts = octave_family_pair_counts(quartet)
        k = int(counts['2/1'])
        exact_count_duration[k] = exact_count_duration.get(k, 0.0) + dur
        for name in pair_duration:
            pair_duration[name] += dur * counts[name]
        extra_octave += dur * max(0, sum(counts.values()) - 1)

    pair_rates = {
        name: pair_duration[name] / max(1e-9, total * 6.0)
        for name in ('2/1', '4/1', '8/1')
    }
    pair_rates['octave_family_total'] = sum(pair_rates.values())
    exact_count_weights = {
        str(k): dur / max(1e-9, total)
        for k, dur in sorted(exact_count_duration.items())
    }
    result = {
        'mean_raw_cse': weighted / total if total else None,
        'scored_duration': total,
        'scored_slices': scored,
        # Distribution requested for the baselines: k means exactly k of the six
        # voice pairs are an exact octave-step 2/1 relation.
        'exact_2_1_pair_count_duration_weights': exact_count_weights,
        'exact_2_1_pair_count_duration': {
            str(k): dur for k, dur in sorted(exact_count_duration.items())
        },
        'octave_pair_rates': pair_rates,
        'extra_octave_pairs_per_slice_time': extra_octave / max(1e-9, total),
    }
    if include_segments:
        result['segments'] = rows
    return result


def _voice_pool(voice):
    key = 'counter' if voice.startswith('counter') else voice
    fallback = key
    eff = getattr(sb, 'COUNTERPOINT_EFFECTIVE_RANGES', {})
    lo, hi = eff.get(key, hr.RANGES[fallback])
    return tuple(int(p) for p in hr.POOLS[fallback] if lo <= int(p) <= hi)


def baseline_pools():
    return {
        'bass': _voice_pool('bass'),
        'inner': _voice_pool('inner'),
        'counter': _voice_pool('counter'),
    }


def quartet_hard_legal(q):
    """Vertical legality shared by both baselines.

    We retain strict voice order and the generator's hard wolf / adjacent-degree
    exclusions. Horizontal continuity is intentionally omitted: the enumerated
    result is meant to be a vertical lower bound.
    """
    q = tuple(int(x) for x in q)
    if len(q) != 4 or not (q[0] < q[1] < q[2] < q[3]):
        return False
    for a, b in combinations(q, 2):
        if sb.hard_wolf(a, b) or sb.step_second(a, b):
            return False
    return hr.cse4_score(q, default=None) is not None


def enumerate_lower_triples_for_lead(lead_step, pools=None):
    """Enumerate legal lower triples for a fixed Lead, grouped by exact 2/1-pair count.

    The returned dict maps k -> [(raw_cse, quartet), ...], where k is the exact
    number of the six vertical voice pairs satisfying |delta step| == hr.OCT.
    Thus k=0, k=1, k=2, ... are distinct strata; they are never merged into a
    generic "contains an octave" bucket.
    """
    lead_step = int(lead_step)
    pools = pools or baseline_pools()
    by_count = {}
    for bass, inner, counter in product(
        pools['bass'], pools['inner'], pools['counter']
    ):
        q = (bass, inner, counter, lead_step)
        if not quartet_hard_legal(q):
            continue
        raw = float(hr.cse4_score(q))
        k = exact_2_1_pair_count(q)
        by_count.setdefault(k, []).append((raw, q))
    return by_count


def _prepare_enumeration(rows):
    pools = baseline_pools()
    cache = {}
    for _, _, q in rows:
        lead = int(q[3])
        if lead not in cache:
            cache[lead] = enumerate_lower_triples_for_lead(lead, pools)
    return cache


def _actual_joint_weights(rows):
    """Duration weights for the exact realised (Lead pitch, exact-2/1-count) strata."""
    joint = {}
    by_count = {}
    total = 0.0
    for a, b, q in rows:
        dur = b - a
        lead = int(q[3])
        k = exact_2_1_pair_count(q)
        joint[(lead, k)] = joint.get((lead, k), 0.0) + dur
        by_count[k] = by_count.get(k, 0.0) + dur
        total += dur
    return joint, by_count, total


def enumerated_optimum_baseline(voices, prepared=None):
    """Lead-conditioned enumerated local minimum with exact k-pair matching.

    For every realised vertical slice, keep its Lead and the exact number k of
    pure 2/1 pairs. Among all hard-legal lower triples with that same Lead and
    exactly the same k, take the minimum raw CSE. Duration-weighting the resulting
    minima is equivalent to a linear mixture whose k=0,1,2,... weights are exactly
    those of the real score, while also preserving the real Lead x k correlation.
    """
    real = four_part_raw_cse_summary(voices, include_segments=True)
    rows = real['segments']
    cache = prepared or _prepare_enumeration(rows)
    joint, by_count, total = _actual_joint_weights(rows)

    weighted = 0.0
    missing = []
    best_examples = {}
    conditioned_sums = {}
    conditioned_weights = {}

    for (lead, k), dur in sorted(joint.items()):
        vals = cache.get(lead, {}).get(k, ())
        if not vals:
            missing.append({'lead_step': lead, 'exact_2_1_pair_count': k, 'duration': dur})
            continue
        best_raw, best_q = min(vals, key=lambda x: x[0])
        weighted += dur * best_raw
        conditioned_sums[k] = conditioned_sums.get(k, 0.0) + dur * best_raw
        conditioned_weights[k] = conditioned_weights.get(k, 0.0) + dur
        best_examples[f'{lead}:k{k}'] = {
            'raw_cse': best_raw,
            'quartet': list(best_q),
            'real_duration_weight': dur / max(1e-9, total),
        }

    mixed = weighted / total if total and not missing else None
    conditioned = {
        str(k): conditioned_sums[k] / conditioned_weights[k]
        for k in sorted(conditioned_sums)
    }
    count_weights = {
        str(k): dur / max(1e-9, total) for k, dur in sorted(by_count.items())
    }
    enum_counts = {
        str(lead): {str(k): len(vals) for k, vals in sorted(groups.items())}
        for lead, groups in sorted(cache.items())
    }
    return {
        'method': (
            'fixed realised Lead and exact realised 2/1-pair count k for every vertical slice; '
            'enumerate every hard-legal Bass/Inner/secondary triple in the same (Lead,k) stratum; '
            'use the minimum raw CSE; duration-weight across real slices. Therefore k=0,1,2,... '
            'weights, and their correlation with Lead pitch, exactly match the real score'
        ),
        'real_exact_2_1_pair_count_duration_weights': count_weights,
        'conditioned_mean_min_raw_cse_by_exact_pair_count': conditioned,
        'mixed_mean_raw_cse': mixed,
        'enumerated_legal_counts_by_lead_and_exact_pair_count': enum_counts,
        'best_quartet_examples_by_lead_and_exact_pair_count': best_examples,
        'missing_exact_strata': missing,
        'horizontal_constraints_ignored': True,
    }


def random_lower_voice_baseline(voices, seed, samples=8, prepared=None):
    """Random lower-voice reference with exact (Lead,k) matching to the real score."""
    real = four_part_raw_cse_summary(voices, include_segments=True)
    rows = real['segments']
    cache = prepared or _prepare_enumeration(rows)
    joint, by_count, total = _actual_joint_weights(rows)
    count_weights = {
        str(k): dur / max(1e-9, total) for k, dur in sorted(by_count.items())
    }
    results = []

    for j in range(max(1, int(samples))):
        rng = random.Random(int(seed) ^ (0x9E3779B9 + j * 0x85EBCA6B))
        weighted = 0.0
        conditioned_sums = {}
        conditioned_weights = {}
        missing = []

        # Sample independently for every realised vertical slice, but only from
        # legal quartets sharing that slice's exact Lead and exact k count.
        for a, b, q in rows:
            dur = b - a
            lead = int(q[3])
            k = exact_2_1_pair_count(q)
            vals = cache.get(lead, {}).get(k, ())
            if not vals:
                missing.append({'lead_step': lead, 'exact_2_1_pair_count': k, 'duration': dur})
                continue
            raw, _ = rng.choice(vals)
            weighted += dur * raw
            conditioned_sums[k] = conditioned_sums.get(k, 0.0) + dur * raw
            conditioned_weights[k] = conditioned_weights.get(k, 0.0) + dur

        mixed = weighted / total if total and not missing else None
        conditioned = {
            str(k): conditioned_sums[k] / conditioned_weights[k]
            for k in sorted(conditioned_sums)
        }
        results.append({
            'index': j,
            'conditioned_mean_raw_cse_by_exact_pair_count': conditioned,
            'mixed_mean_raw_cse': mixed,
            'missing_exact_strata': missing,
        })

    vals = [x['mixed_mean_raw_cse'] for x in results if x['mixed_mean_raw_cse'] is not None]
    mean = sum(vals) / len(vals) if vals else None
    std = None
    if vals:
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
    return {
        'method': (
            'fixed realised Lead and exact realised 2/1-pair count k for every vertical slice; '
            'randomly choose a hard-legal Bass/Inner/secondary triple from the same (Lead,k) '
            'stratum. Thus the random reference has exactly the real score\'s k=0,1,2,... '
            'duration distribution and Lead x k distribution'
        ),
        'real_exact_2_1_pair_count_duration_weights': count_weights,
        'samples': results,
        'baseline_mean_raw_cse': mean,
        'baseline_std_raw_cse': std,
    }


def evaluate_baselines(voices, seed, samples=8):
    """Compute realised CSE, exact-k enumerated lower bound, random reference and bracket."""
    real = four_part_raw_cse_summary(voices, include_segments=True)
    rows = real['segments']
    prepared = _prepare_enumeration(rows)
    optimum = enumerated_optimum_baseline(voices, prepared=prepared)
    random_ref = random_lower_voice_baseline(
        voices, seed=seed, samples=samples, prepared=prepared
    )

    actual = real['mean_raw_cse']
    lower = optimum['mixed_mean_raw_cse']
    upper = random_ref['baseline_mean_raw_cse']
    low_good = None
    gap_closed = None
    low_good_clamped = None
    gap_closed_clamped = None
    if actual is not None and lower is not None and upper is not None and upper > lower + 1e-12:
        low_good = 100.0 * (actual - lower) / (upper - lower)
        gap_closed = 100.0 * (upper - actual) / (upper - lower)
        low_good_clamped = min(100.0, max(0.0, low_good))
        gap_closed_clamped = min(100.0, max(0.0, gap_closed))

    public_real = {k: v for k, v in real.items() if k != 'segments'}
    return {
        'metric': 'raw corrected four-note CSE; lower is better; table percentile is not used',
        'lead_conditioning': 'all baselines keep every realised Lead pitch and vertical slice duration',
        'exact_2_1_mixture_rule': (
            'stratify by the exact number k of octave-step 2/1 pairs among the six voice pairs; '
            'for every realised slice preserve both its Lead and its exact k. This exactly '
            'preserves the real score\'s duration weights for k=0,1,2,... rather than merely '
            'matching whether any 2/1 is present'
        ),
        'real_score': public_real,
        'enumerated_optimum_lower_bound': optimum,
        'random_lower_voice_reference': random_ref,
        'bracket': {
            'enumerated_lower_bound_raw_cse': lower,
            'actual_raw_cse': actual,
            'random_reference_raw_cse': upper,
            'bounded_cse_percentile_low_is_good': low_good,
            'bounded_cse_percentile_low_is_good_clamped_0_100': low_good_clamped,
            'fraction_of_random_to_optimal_gap_closed_percent': gap_closed,
            'fraction_of_random_to_optimal_gap_closed_percent_clamped_0_100': gap_closed_clamped,
            'interpretation': (
                '0% bounded percentile = exact-k enumerated local optimum; 100% = exact-k '
                'matched random lower-voice reference. Gap-closed percentage is complementary.'
            ),
            'random_is_empirical_reference_not_strict_worst_case': True,
        },
    }

def main():
    ap = argparse.ArgumentParser(description='Compute Lead-conditioned ScaleWeaver four-part CSE baselines.')
    ap.add_argument('score_json', help='generated ScaleWeaver score JSON')
    ap.add_argument('-o', '--output', help='optional output JSON for baseline results')
    ap.add_argument('--samples', type=int, default=8, help='Monte-Carlo random baseline samples')
    ap.add_argument('--cse-dir', default=None, help='Directory containing the active scale CSE bundle')
    args = ap.parse_args()

    score = json.loads(Path(args.score_json).read_text(encoding='utf-8'))
    import main as composer
    configuration = score.get('configuration')
    if configuration is None:
        raise ValueError('Baseline requires a score with its three-layer configuration snapshot')
    ngram = configuration['style'].get('ngram_file')
    if ngram and not Path(ngram).is_absolute():
        configuration['style']['ngram_file'] = str((Path(args.score_json).resolve().parent/ngram).resolve())
    composer._configure_adaptive_scale(configuration, args.cse_dir)

    result = evaluate_baselines(
        score['voices'],
        seed=int(score.get('selected_candidate_seed', score.get('seed', 0))),
        samples=args.samples,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
        print('wrote', args.output)
    else:
        print(text)


if __name__ == '__main__':
    main()
