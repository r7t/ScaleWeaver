"""Infer harmony from a fixed, already retuned melody in the active scale."""
from __future__ import annotations

import math
import statistics

import harmony_rhythm as hr


def metrical_weight(offset, signature):
    """Quarter-note units; downbeat > secondary strong beat > beat > offbeat."""
    n, d = map(int, signature.split('/'))
    unit = 4. / d
    if abs(offset) < 1e-7:
        return 4.
    if d == 8 and n >= 6 and n % 3 == 0:
        return 3. if abs(offset / (3 * unit) - round(offset / (3 * unit))) < 1e-7 else 1.
    if n >= 4 and n % 2 == 0 and abs(offset - n * unit / 2) < 1e-7:
        return 3.
    return 1.5 if abs(offset / unit - round(offset / unit)) < 1e-7 else 1.


def harmonic_boundaries(measure):
    n, d = map(int, measure['time_signature'].split('/'))
    duration = float(measure['duration_beats'])
    stride = n * 2. / d if n >= 4 and n % 2 == 0 else duration
    # Compound beats are three eighths = 1.5 quarter-note beats.
    if d == 8 and n >= 6 and n % 3 == 0:
        stride = 12. / d
    return [0.] + [i * stride for i in range(1, math.ceil(duration / stride))] + [duration]


def marginal_cse(chord, melody_step, cache):
    """CSE(C union m) - CSE(C), in the SAME register and raw metric.

    Evaluate a compact complete chord below the melody, including octave
    doubling of chord tones. Missing cache entries are exposed
    as None, never replaced with a fabricated dissonance value.
    """
    key = (chord.id, int(melody_step))
    if key in cache:
        return cache[key]
    ref = hr.chord_reference_steps(chord)
    shift = math.floor((melody_step - max(ref)) / hr.OCT) * hr.OCT
    background = tuple(p + shift for p in ref)
    if melody_step in background:
        cache[key] = 0.
        return 0.
    result = None
    if hr.CSE is not None and len(background) < 5:
        base = hr.CSE.entry(background, metric='cse')
        added = hr.CSE.entry(tuple(sorted(background + (int(melody_step),))), metric='cse')
        if base is not None and added is not None:
            result = float(added[0] - base[0])
    cache[key] = result
    return result


def segment_evidence(chord, events, measure, offset, duration, cache):
    start = float(measure['start_beat']) + offset
    end = start + duration
    mass = outside = delta_sum = available = missing = 0.
    for note in events:
        a = max(start, note['start_beat'])
        b = min(end, note['start_beat'] + note['duration_beats'])
        if b <= a + 1e-9:
            continue
        # A tie crossing a harmonic boundary contributes its sounding duration
        # without acquiring a new downbeat attack bonus.
        attack = note['start_beat'] >= start - 1e-7
        weight = (metrical_weight(note['start_beat'] - measure['start_beat'],
                                  measure['time_signature']) if attack else 1.)
        weight *= math.sqrt(b - a)
        mass += weight
        outside += weight * (note['step'] % hr.OCT not in chord.pcs)
        delta = marginal_cse(chord, note['step'], cache)
        if delta is None:
            missing += weight
        else:
            delta_sum += weight * delta
            available += weight
    return dict(nonchord_fraction=outside / mass if mass else 0.,
                marginal_cse=delta_sum / available if available else 0.,
                cse_coverage=available / mass if mass else 0.,
                evidence_weight=mass, missing_cse_weight=missing)


def infer_harmony(events, measure_map):
    """Viterbi across metrical harmonic slots; never reads annotated chords."""
    chords = sorted(hr.CHORDS.values(), key=lambda c: c.id)
    cache, slots = {}, []
    for bar, measure in enumerate(measure_map):
        boundaries = harmonic_boundaries(measure)
        for a, b in zip(boundaries, boundaries[1:]):
            evidence = [segment_evidence(c, events, measure, a, b-a, cache) for c in chords]
            # Extra chord tones trivially improve coverage; charge a small
            # complexity cost so adding every melody pitch is not a shortcut.
            neutral = statistics.median([e['marginal_cse'] for e in evidence
                                         if e['cse_coverage'] > 0.] or [0.])
            costs = [e['nonchord_fraction'] + 1.5 * (e['marginal_cse'] * e['cse_coverage']
                     + neutral * (1.-e['cse_coverage']))
                     + .25 * max(0, len(c.pcs)-3) for c, e in zip(chords, evidence)]
            slots.append((bar, a, b-a, evidence, costs))
    # Weak transition prior: common tones/function can disambiguate similar
    # melodic fits, but cannot impose a randomly sampled route on the melody.
    transitions = {}
    bases = {c.id: hr._chord_base_weight(c) for c in chords}
    def transition(i, j):
        if (i, j) not in transitions:
            transitions[i, j] = -.06 * math.log(max(1e-15,
                hr._chord_transition_weight(chords[i], chords[j]) / bases[chords[j].id]))
        return transitions[i, j]
    costs = {}
    back = []
    for index, (_, _, _, _, local) in enumerate(slots):
        parents, updated = {}, {}
        # Bound dense arbitrary-scale vocabularies; TianGan's 32 chords all
        # remain in the search. Selection uses melody evidence, not annotations.
        candidates = sorted(range(len(chords)), key=lambda j: (local[j], chords[j].id))[:64]
        for j in candidates:
            parent = min(costs, key=lambda i: (costs[i] + transition(i, j), i)) if index else 0
            parents[j] = parent
            updated[j] = local[j] + (costs[parent] + transition(parent, j) if index else 0.)
        costs = updated
        back.append(parents)
    chosen = min(costs, key=lambda j: (costs[j], j))
    route = []
    for parents in reversed(back):
        route.append(chosen)
        chosen = parents[chosen]
    route.reverse()
    plan = [dict(bar=i, beats_per_bar=float(m['duration_beats']),
                 start_beat=float(m['start_beat']), time_signature=m['time_signature'],
                 harmony_model='melody_inference', chord_segments=[])
            for i, m in enumerate(measure_map)]
    for (bar, offset, duration, evidence, local), j in zip(slots, route):
        chord = chords[j]
        segment = hr._natural_segment(chord, offset, duration, chord.function)
        segment.update(hr.THREE_TONE_INFO.get(chord.id, {}))
        segment['melody_inference'] = dict(
            **evidence[j], local_cost=local[j],
            local_rank=sorted(range(len(chords)), key=lambda i: (local[i], chords[i].id)).index(j)+1,
            alternatives=[dict(chord_id=chords[i].id, chord_name=chords[i].name,
                               local_cost=local[i], **evidence[i])
                          for i in sorted(range(len(chords)), key=lambda i: local[i])[:5]])
        plan[bar]['chord_segments'].append(segment)
    for row in plan:
        hr._refresh_primary_from_first_segment(row)
    return plan
