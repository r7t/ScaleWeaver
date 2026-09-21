"""Optional 3-tone phrase planner. Legacy harmony is left intact.

Functional affinities borrow tonal syntax, not twelve-tone pitch rounding.
Every reference interval is mapped through the active EDO's patent val.
"""
import math
from collections import defaultdict

from ji_ratio import patent_integer_value
from three_tone import classify_three_tone, tonic_relation


def resolve_config(harmony):
    raw = harmony.get('three_tone_progression', {})
    if not isinstance(raw, dict):
        raise ValueError('harmony.three_tone_progression must be an object')
    # Old special_weight is accepted as an alias for external-group weight.
    raw = dict(raw)
    if 'special_weight' in raw:
        value = raw.pop('special_weight')
        if 'outside_scale_weight' in raw and raw['outside_scale_weight'] != value:
            raise ValueError('special_weight conflicts with outside_scale_weight')
        raw.setdefault('outside_scale_weight', value)
    defaults = dict(enabled=False, max_integer=64, outside_scale_weight=.25,
                    implied_weight=.8, resolution_weight=1.2)
    if set(raw) - set(defaults):
        raise ValueError('unknown three_tone_progression option')
    cfg = {**defaults, **raw}
    if type(cfg['enabled']) is not bool:
        raise ValueError('three_tone_progression.enabled must be boolean')
    if type(cfg['max_integer']) is not int or cfg['max_integer'] < 3:
        raise ValueError('three_tone_progression.max_integer must be an integer >= 3')
    for key in ('outside_scale_weight', 'implied_weight', 'resolution_weight'):
        if (type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or
                not 0 <= cfg[key] <= (4 if key == 'resolution_weight' else 1)):
            raise ValueError('three_tone_progression.%s is out of range' % key)
    return cfg


def build_annotations(spec, chords, ji_table, cfg):
    """Resolve ratios without preferring a classification outcome or function."""
    out = {}
    tonic = int(spec.resolved_harmony()['pitch_roles']['tonic'][0])
    for cid, chord in chords.items():
        candidates = []
        if chord.ratio:
            candidates.append((chord.pcs, chord.integers, 0., 'chord_vocabulary'))
        elif ji_table is not None:
            for foot in chord.pcs:
                steps = tuple(sorted(foot + (p-foot) % spec.edo for p in chord.pcs))
                item = ji_table.lookup(steps)
                if item and item.get('integers'):
                    ns = tuple(item['integers'])
                    divisor = math.gcd(*ns)
                    ns = tuple(n // divisor for n in ns)
                    candidates.append((steps, ns, float(item.get('rms_error_cents', 0.)), 'ji_table'))
        if not candidates:
            out[cid] = dict(three_tone_pc=None, three_tone_status='unclassified',
                            three_tone_reason='missing_ratio', three_tone_ratio=None,
                            three_tone_ratio_source=None, three_tone_relation='unclassified',
                            three_tone_engine_version=2, three_tone_eligible=False,
                            three_tone_exclusion_reason='missing_ratio')
            continue
        steps, ns, rms, source = min(candidates, key=lambda x:
                                     (max(x[1]), sum(x[1]), x[2], x[1], x[0]))
        tone = classify_three_tone(steps, ns, spec.edo, spec.pcs)
        row = dict(three_tone_pc=tone.pc, three_tone_status=tone.status,
                   three_tone_reason=tone.reason, three_tone_inferred_pc=tone.inferred_pc,
                   three_tone_ratio=list(ns), three_tone_ratio_pcs=[p % spec.edo for p in steps],
                   three_tone_ratio_source=source, three_tone_rms_error_cents=rms,
                   three_tone_relation=tonic_relation(tone, tonic, spec.edo),
                   three_tone_engine_version=2, three_tone_in_chord=tone.status=='present',
                   three_tone_in_scale=tone.status in ('present','implied'),
                   three_tone_explicit_harmonic=tone.explicit_harmonic,
                   three_tone_eligible=tone.pc is not None,
                   three_tone_exclusion_reason=None if tone.pc is not None else tone.reason)
        if max(ns) > cfg['max_integer']:
            row.update(three_tone_eligible=False, three_tone_exclusion_reason='ratio_complexity_limit')
        if tone.pc is not None:
            row.update(function_profile(tone.pc, tone.status, tonic, spec.edo))
        out[cid] = row
    return out


def _distance(a, b, edo):
    d = (a-b) % edo
    return min(d, edo-d) * 12. / edo


def affinities(pc, tonic, edo):
    """Heuristic major/minor and chromatic colours, separate from classification.

    The fifth is the strongest T reference. C and F-like anchors prepare;
    D and B-like anchors approach. b2/b6 colours permit modal mixture.
    These are compositional priors, not acoustically established functions.
    """
    templates = {
        'T': ((3, 2, 1.), (5, 4, .22)),
        'S': ((1, 1, .85), (4, 3, 1.), (8, 5, .65), (16, 15, .4)),
        'D': ((9, 8, 1.), (15, 8, .65), (45, 32, .45), (16, 15, .35)),
    }
    return {fn: max(weight * math.exp(-2 * _distance(
        pc, (tonic + patent_integer_value(n, edo)-patent_integer_value(d, edo)) % edo, edo)**2)
        for n, d, weight in refs) for fn, refs in templates.items()}


def function_profile(pc, status, tonic, edo):
    """Separate tonal direction from whether the reference is actually heard."""
    weights = affinities(pc, tonic, edo)
    function = max(weights, key=weights.get)
    if weights[function] < .2:
        function = 'C'
    support = {'present':1., 'implied':.75, 'outside_scale':.45}[status]
    tension = {'present':0., 'implied':.2, 'outside_scale':.55}[status]
    tension += .45 * (1.-weights['T'])
    return dict(three_tone_function=function, three_tone_function_weights=weights,
                three_tone_tonic_strength=support * weights['T'],
                three_tone_tension=tension, three_tone_relative_pc=(pc-tonic)%edo)


def resolution_score(previous_pc, previous_tension, pc, status, tonic, edo):
    """Fifth-chain direction plus release from implicit/chromatic references."""
    fifth = patent_integer_value(3, edo) % edo
    circle = math.exp(-2 * _distance(pc, (previous_pc-fifth)%edo, edo)**2)
    profile = function_profile(pc, status, tonic, edo)
    release = max(0., previous_tension-profile['three_tone_tension'])
    return .8 * circle + release * (.5 + profile['three_tone_tonic_strength'])


def make_section(hr, rng, bpb, bars):
    """Actual-length phrase: establish, prolong, prepare, approach, resolve."""
    cfg, annotations = hr.THREE_TONE_CONFIG, hr.THREE_TONE_INFO
    tonic = int(hr.SCALE.resolved_harmony()['pitch_roles']['tonic'][0])
    fifth = patent_integer_value(3, hr.OCT) % hr.OCT
    anchor = (tonic + fifth) % hr.OCT
    groups = defaultdict(list)
    for cid, row in annotations.items():
        if (row['three_tone_eligible'] and
                not (row['three_tone_status']=='outside_scale' and cfg['outside_scale_weight']==0)):
            groups[row['three_tone_pc']].append(cid)
    if not groups:
        raise ValueError('three-tone harmony requires reliable local ratios; use legacy mode or supply a JI table')
    endings = [cid for cid in groups.get(anchor, ()) if tonic in hr.CHORDS[cid].pcs]
    fallback = not endings
    if fallback:
        endings = [cid for ids in groups.values() for cid in ids if tonic in hr.CHORDS[cid].pcs]
    if not endings:
        raise ValueError('three-tone harmony has no classified chord containing the tonic')
    # Prefer a sounded reference, then an in-scale one, before external fallback.
    availability = {'present':0, 'implied':1, 'outside_scale':2}
    best = min(availability[annotations[cid]['three_tone_status']] for cid in endings)
    endings = [cid for cid in endings if availability[annotations[cid]['three_tone_status']]==best]
    fallback = fallback or best == 2
    # Prefer compact tonic triads over extensions at the actual cadence.
    triads = [cid for cid in endings if len(hr.CHORDS[cid].pcs) == 3]
    endings = triads or endings
    roles = (['T'] if bars == 1 else ['D', 'T'] if bars == 2 else
             ['T'] + ['S' if i >= (bars-2)/2 else 'T' for i in range(1, bars-2)] + ['D', 'T'])
    group_affinity = {pc: affinities(pc, tonic, hr.OCT) for pc in groups}
    out, prev, prev_group = [], None, None
    for i, target in enumerate(roles):
        boundary = i == bars-1 or (i == 0 and bars > 2)
        if boundary:
            pool = endings
        else:
            weighted_groups = []
            for pc in groups:
                score = 3.5 * group_affinity[pc][target]
                status_weights = {'present':1., 'implied':cfg['implied_weight'],
                                  'outside_scale':cfg['outside_scale_weight']}
                prior = sum(status_weights[annotations[cid]['three_tone_status']]
                            for cid in groups[pc]) / len(groups[pc])
                if prior <= 0:
                    continue
                score += math.log(prior)
                if prev_group is not None:
                    score += cfg['resolution_weight'] * sum(resolution_score(
                        prev_group, annotations[prev.id]['three_tone_tension'], pc,
                        annotations[cid]['three_tone_status'], tonic, hr.OCT)
                        for cid in groups[pc]) / len(groups[pc])
                    score += (.35 if pc == prev_group and i < bars/2 else 0.)
                    score -= (.8 if pc == prev_group and i >= bars/2 else 0.)
                if i == bars-2:
                    score -= 2. if pc == anchor else 0.
                    score += .5 * group_affinity[pc]['S']  # plagal/modal alternative
                weighted_groups.append((pc, math.exp(score)))
            if not weighted_groups:
                raise ValueError('three-tone availability weights disable every available group')
            group = hr.weighted_choice(weighted_groups, rng)
            pool = groups[group]
        weighted = []
        for cid in pool:
            c = hr.CHORDS[cid]
            score = .7 * c.stability - .08 * float(hr.chord_cse_score(c))
            if boundary and annotations[cid]['three_tone_status'] == 'present':
                score += .7
            if prev is not None:
                score += cfg['resolution_weight'] * resolution_score(
                    prev_group, annotations[prev.id]['three_tone_tension'],
                    annotations[cid]['three_tone_pc'], annotations[cid]['three_tone_status'],
                    tonic, hr.OCT)
                score += .2 * len(set(prev.pcs) & set(c.pcs))
                motion = sum(min(_distance(p, q, hr.OCT) for q in c.pcs) for p in prev.pcs) / len(prev.pcs)
                score -= .25 * motion
                if prev.id == cid:
                    score -= 1.5
            if cid in hr.NATURAL_TRIAD_INFO:
                score += 1.4
                if hr.NATURAL_TRIAD_INFO[cid]['quality'] == 'minor' and not boundary:
                    score += .6
            weighted.append((cid, math.exp(max(-60., min(60., score)))))
        cid = hr.weighted_choice(weighted, rng)
        c, row = hr.CHORDS[cid], annotations[cid]
        pc = row['three_tone_pc']
        function = row['three_tone_function']
        seg = hr._natural_segment(c, 0., bpb, target)
        seg.update(row)
        seg.update(function=function, legacy_function=c.function,
                   harmony_model='three_tone', three_tone_cadence_fallback=fallback,
                   phrase_role=('cadence' if i == bars-1 else 'establish' if boundary else
                                'approach' if i == bars-2 else 'develop'))
        degree = hr._natural_degree(c) if hr.SCALE.note_count == 7 else None
        if degree is not None:
            seg.update(root_degree=degree, root_pc=hr.PCS[degree-1])
        if boundary:
            seg['root_pc'] = tonic
        out.append([seg])
        prev, prev_group = c, pc
    return out
