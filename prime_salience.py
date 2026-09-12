"""Register-sensitive exposure of twelve basic 7-odd-limit intervals."""
import math
from functools import lru_cache

INTERVAL_RATIOS = {
    '3/2': 3/2, '4/3': 4/3, '5/4': 5/4, '6/5': 6/5,
    '7/6': 7/6, '8/7': 8/7, '8/5': 8/5, '5/3': 5/3,
    '7/5': 7/5, '7/4': 7/4, '10/7': 10/7, '12/7': 12/7,
}
TEMPLATES = {
    '5': ((5/4, 1.0), (6/5, 1.0), (8/5, 1.0), (5/3, 1.0),
          (7/5, .7), (10/7, .7)),
    '7': ((7/6, 1.0), (12/7, 1.0), (8/7, 1.0), (7/4, 1.0),
          (7/5, .9), (10/7, .9)),
}


class PrimeSalience:
    def __init__(self, edo, config):
        self.edo = edo
        self.config = dict(config)
        self.enabled = (any(config[k] for k in ('weight_5', 'weight_7', 'joint_weight'))
                        or any(config['interval_rewards'].values()))
        self.templates = {p: [(1200*math.log2(r), w) for r, w in rows]
                          for p, rows in TEMPLATES.items()}
        self.interval_cents = tuple((name, 1200*math.log2(ratio))
                                    for name, ratio in INTERVAL_RATIOS.items())
        self._features = lru_cache(maxsize=100000)(self._features)
        self._interval_features = lru_cache(maxsize=100000)(self._interval_features)

    def _fusion(self, cents):
        c = self.config
        harmonic = []
        for x in cents[1:]:
            h = round(2**(x/1200))
            harmonic.append(math.exp(-.5*((x-1200*math.log2(h))/c['tolerance_cents'])**2)
                            if 1 <= h <= 16 else 0.0)
        return min(harmonic) if len(cents) >= 4 else 0.0

    def _edge_exposure(self, distance, voice_gap):
        c = self.config
        compound_octaves = max(0, int(distance//1200))
        return (c['compound_decay_per_octave']**compound_octaves /
                (1+c['intervening_voice_discount']*voice_gap))

    def interval_features(self, pitches):
        xs = sorted(pitches)
        if len(xs) < 2:
            return {name: 0.0 for name in INTERVAL_RATIOS}
        values = self._interval_features(tuple(x-xs[0] for x in xs))
        return dict(zip(INTERVAL_RATIOS, values))

    def _interval_features(self, xs):
        cents = [1200*x/self.edo for x in xs]
        fusion_factor = 1-self.config['fusion_discount']*self._fusion(cents)
        values = []
        for _, target in self.interval_cents:
            best = 0.0
            for i, a in enumerate(cents):
                for j in range(i+1, len(cents)):
                    distance = cents[j]-a
                    residue = distance % 1200
                    match = math.exp(-.5*((residue-target)/self.config['tolerance_cents'])**2)
                    best = max(best, match*self._edge_exposure(distance, j-i-1))
            values.append(best*fusion_factor)
        return tuple(values)

    def features(self, pitches):
        xs = sorted(pitches)
        if len(xs) < 2:
            return (0.0, 0.0, 0.0)
        return self._features(tuple(x-xs[0] for x in xs))

    def _features(self, xs):
        c = self.config
        cents = [1200*x/self.edo for x in xs]
        fusion = self._fusion(cents)
        scores = []
        for prime in ('5', '7'):
            edges = []
            for i, a in enumerate(cents):
                for j in range(i+1, len(cents)):
                    distance = cents[j]-a
                    residue = distance % 1200
                    match = max(w*math.exp(-.5*((residue-t)/c['tolerance_cents'])**2)
                                for t, w in self.templates[prime])
                    edges.append(match*self._edge_exposure(distance, j-i-1))
            scores.append(max(edges, default=0.0)*(1-c['fusion_discount']*fusion))
        return scores[0], scores[1], fusion

    def reward(self, pitches, context):
        if not self.enabled:
            return 0.0
        s5, s7, _ = self.features(pitches)
        c = self.config
        directed = sum(c['interval_rewards'][name]*value
                       for name, value in self.interval_features(pitches).items())
        return c[context+'_weight']*(directed+c['weight_5']*s5+c['weight_7']*s7
                                     +c['joint_weight']*min(s5, s7))

    def diagnostics(self, score):
        events = [e for rows in score['voices'].values() for e in rows]
        points = sorted({t for e in events for t in
                         (e['start_beat'], e['start_beat']+e['duration_beats'])})
        rows = []
        for a, b in zip(points, points[1:]):
            xs = [e['step'] for e in events
                  if e['start_beat'] <= (a+b)/2 < e['start_beat']+e['duration_beats']]
            if len(xs) >= 2:
                rows.append((b-a, self.features(xs), self.interval_features(xs)))
        groups = {}
        for e in events:
            groups.setdefault(round(e['start_beat'], 6), []).append(e['step'])

        def mean(items):
            total = sum(w for w, _, _ in items)
            base = {'count': len(items), 'weight': total, **{
                name: sum(w*f[i] for w, f, _ in items)/total if total else None
                for i, name in enumerate(('salience_5', 'salience_7', 'fusion'))}}
            base['interval_salience'] = {
                name: sum(w*f[name] for w, _, f in items)/total if total else None
                for name in INTERVAL_RATIOS}
            return base

        attack_rows = [(1, self.features(xs), self.interval_features(xs))
                       for xs in groups.values() if len(xs) >= 2]
        return {'definition': 'twelve_interval_exposure_v2; heuristic, not measured perception',
                'sounding_duration_weighted': mean(rows),
                'attack_equal_onset': mean(attack_rows)}
