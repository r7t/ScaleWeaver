"""Structural melody targets and nested rhythmic answers, before ornamentation."""
import math

import harmony_rhythm as hr
import score_builder as sb


def cadence_skeleton(previous, harmony, half, grid):
    """Choose a nearby stable ending, with a short approach when needed."""
    chord = hr.CHORDS[harmony[-1]['chord_id']]
    pool = [p for p in sb.POOLS['lead'] if p % hr.OCT in chord.pcs]
    reference = previous if previous is not None else pool[len(pool)//2]
    from harmonic_realization import tonic_pc
    tonic = tonic_pc(hr)
    target = min(pool, key=lambda p: (
        abs(hr.degree(p)-hr.degree(reference))
        - (.75 if hr.NATURAL_SCALE_MODE and p % hr.OCT == tonic else 0), p))
    if abs(hr.degree(target)-hr.degree(reference)) <= 1:
        return [{'offset': 0., 'step': int(target)}]
    approach = [p for p in sb.POOLS['lead']
                if abs(hr.degree(p)-hr.degree(target)) == 1
                and sb.lead_jump_ok(previous, p)]
    if approach and grid < half:
        bridge = min(approach, key=lambda p: abs(hr.degree(p)-hr.degree(reference)))
        return [{'offset': 0., 'step': int(bridge)},
                {'offset': grid, 'step': int(target)}]
    legal = [p for p in pool if sb.lead_jump_ok(previous, p)]
    target = min(legal or [reference], key=lambda p: abs(p-reference))
    return [{'offset': 0., 'step': int(target)}]


def make_skeleton(state, harmony, guides, relation, source, allow_sixteenth):
    half = state['bpb']/2
    grid = .25 if allow_sixteenth else .5
    if relation['closing']:
        return cadence_skeleton(state['prev'], harmony, half, grid)
    middle = max(grid, math.floor(half/2/grid)*grid)
    offsets = [0.] if relation['closing'] or middle >= half else [0., middle]
    previous = state['prev']
    anchors = []
    source_anchors = source.get('skeleton', []) if source else []
    for index, offset in enumerate(offsets):
        segment = next(s for s in reversed(harmony) if s['offset'] <= offset+1e-7)
        chord = hr.CHORDS[segment['chord_id']]
        guide = guides[min(index, len(guides)-1)]
        kind = relation.get('answer_type', 'sequence')
        linked = (kind == 'sequence' or
                  kind == 'opening' and relation.get('half_index', 0) == 0 or
                  kind == 'ending' and relation.get('half_index', 0) == 1)
        if source_anchors and linked:
            origin = source_anchors[min(index, len(source_anchors)-1)]['step']
            guide = hr.degree(origin) + relation.get('sequence_shift', 1)
        pool = [p for p in sb.POOLS['lead']
                if p % hr.OCT in chord.pcs and sb.lead_jump_ok(previous, p)]
        # A hard jump/range limit wins over structural chord membership.
        if not pool:
            pool = [p for p in sb.POOLS['lead'] if sb.lead_jump_ok(previous, p)]
        pitch = min(pool, key=lambda p: (abs(hr.degree(p)-guide)
                                        + ((.25 if source_anchors and linked else .65)
                                           *abs(hr.degree(p)-hr.degree(previous))
                                           if previous is not None else 0),
                                        abs(p-previous) if previous is not None else 0, p))
        anchors.append({'offset': offset, 'step': int(pitch)})
        previous = pitch
    return anchors


def expand_rhythms(candidates, cells, skeleton, source, relation, limit, cell_weights=None):
    def onsets(row):
        cursor = 0.
        out = []
        for duration in row:
            out.append(round(cursor, 6))
            cursor += duration
        return set(out)

    required = {round(a['offset'], 6) for a in skeleton}
    half = sum(candidates[0])
    fallback = tuple(b-a for a,b in zip(
        sorted(required), sorted(required)[1:]+[half]))
    available = list(dict.fromkeys([*candidates, *cells, fallback]))
    legal = [r for r in available if required <= onsets(r)]
    if relation['closing']:
        return [fallback]
    if source is None:
        preferred = [r for r in candidates if r in legal]
        return (preferred + [r for r in legal if r not in preferred])[:limit]
    original = tuple(source['rhythm'])
    if relation.get('rhythm_recall') and original in legal:
        return [original]
    src = onsets(original)
    target = len(original)
    if relation['role'] == 'develop':
        target += 1
    elif relation['role'] == 'liquidate':
        target = max(len(required), target-1)
        shorter = [r for r in legal if len(r) <= target]
        legal = shorter or legal
    from melody_character import rhythm_preference
    legal.sort(key=lambda r: (len(src ^ onsets(r)) + .8*abs(len(r)-target)
                             - .3*math.log(rhythm_preference(r, cell_weights)), r))
    kind = relation.get('answer_type', 'rhythm')
    linked = (kind in {'rhythm', 'sequence'} or
              kind == 'opening' and relation.get('half_index', 0) == 0 or
              kind == 'ending' and relation.get('half_index', 0) == 1)
    if relation['role'] in {'answer', 'recall'} and original in legal and linked:
        # Exact rhythmic identity makes the smallest answer audible.
        return [original]
    if relation['role'] == 'answer' and not linked:
        varied = [r for r in legal if r != original]
        return (varied or legal)[:limit]
    return legal[:limit]


def structural_pitch(candidate, previous, offset, skeleton, recent=()):
    anchor = next((a for a in skeleton if abs(a['offset']-offset) < 1e-7), None)
    if anchor:
        return anchor['step']
    upcoming = next((a for a in skeleton if a['offset'] > offset), None)
    pool = [p for p in sb.POOLS['lead'] if sb.lead_jump_ok(previous, p)
            and (upcoming is None or sb.lead_jump_ok(p, upcoming['step']))]
    def cost(p):
        value = .45*abs(hr.degree(p)-hr.degree(candidate))
        if previous is not None:
            value += .25*max(0, abs(hr.degree(p)-hr.degree(previous))-2)
            if p == previous:
                value += 1.5 if len(recent) >= 2 and recent[-2] == p else .3
        if len(recent) >= 2 and p == recent[-2] and p != previous:
            value += 1.0  # discourage immediate A-B-A ornament loops
        if upcoming and previous is not None:
            start, end = hr.degree(previous), hr.degree(upcoming['step'])
            degree = hr.degree(p)
            value += .5*max(0, min(start,end)-degree, degree-max(start,end))
        return value, abs(p-candidate), p
    return min(pool, key=cost)
