import copy
import math
import unittest
from prime_salience import PrimeSalience
from annealing_config import resolve_annealing


class SalienceTests(unittest.TestCase):
    def metric(self, **weights):
        c = resolve_annealing({'prime_salience': weights})['prime_salience']
        return PrimeSalience(72, c)

    def test_specific_intervals_and_detuning(self):
        m = self.metric(weight_5=1, weight_7=1)
        self.assertGreater(m.features([0, 23])[0], .96)
        self.assertGreater(m.features([0, 16])[1], .98)
        self.assertLess(max(m.features([0, 42])[:2]), .001)
        self.assertLess(m.features([0, 20])[0], m.features([0, 23])[0])

    def test_twelve_directed_channels(self):
        for name, ratio in __import__('prime_salience').INTERVAL_RATIOS.items():
            step = round(72*math.log2(ratio))
            m = self.metric(interval_rewards={name: 1})
            self.assertGreater(m.interval_features([0, step])[name], .9, name)
            self.assertGreater(m.reward([0, step], 'sounding'), .9, name)

    def test_fusion_and_octave_exposure(self):
        m = self.metric(weight_5=1, weight_7=1)
        fused = [round(72*math.log2(r)) for r in (1, 2, 3, 5, 7)]
        self.assertGreater(m.features(fused)[2], .95)
        self.assertLess(m.features(fused)[0], m.features([0, 23])[0])
        self.assertLess(m.features(fused)[1], m.features([0, 16])[1])
        self.assertLess(m.features([0, 160])[1], m.features([0, 16])[1])
        direct = m.interval_features([0, 23])['5/4']
        compound = m.interval_features([0, 95])['5/4']
        self.assertAlmostEqual(compound/direct, .65, places=6)
        separated = m.interval_features([0, 10, 23])['5/4']
        self.assertLess(separated, direct)
        self.assertEqual(m.features(fused), m.features([x-72 for x in fused]))

    def test_disabled_and_balanced_bonus(self):
        self.assertEqual(self.metric().reward([0, 23], 'sounding'), 0)
        m = self.metric(joint_weight=1)
        self.assertGreater(m.reward([0, 16, 23], 'attack'), .5)
        self.assertLess(m.reward([0, 23], 'attack'), .001)

    def test_real_optimizer_cost_hooks(self):
        from harmony_annealing import OptimizerMetric
        m = OptimizerMetric.__new__(OptimizerMetric)
        m._corrected_cached = lambda xs: 1.5
        m.corrected = lambda xs: 1.5
        m.cse_loss = lambda context, value: value
        m.prime_reward = lambda xs: 0
        m.attack_config = {'cardinality_weights': {'2': 1}}
        m.prime_salience = self.metric(weight_5=1)
        self.assertLess(m._sounding_cost_uncached((0, 23)), .54)
        self.assertLess(m._attack_cost_uncached((0, 23)), .54)
        m.prime_salience = self.metric()
        self.assertEqual(m._sounding_cost_uncached((0, 23)), 1.5)

    def test_validation_and_diagnostics(self):
        for row in ({'tolerance_cents': 0}, {'fusion_discount': 1.1}, {'weight_5': -1},
                    {'compound_decay_per_octave': 1.1}, {'interval_rewards': {'9/7': 1}}):
            with self.assertRaises(ValueError):
                resolve_annealing({'prime_salience': row})
        score = {'voices': {'bass': [{'step': 0, 'start_beat': 0, 'duration_beats': 2}],
                            'lead': [{'step': 23, 'start_beat': 0, 'duration_beats': 1},
                                     {'step': 16, 'start_beat': 1, 'duration_beats': 1}]}}
        before = copy.deepcopy(score)
        d = self.metric(weight_5=1).diagnostics(score)
        self.assertEqual(score, before)
        self.assertEqual(d['sounding_duration_weighted']['weight'], 2)
        self.assertEqual(d['attack_equal_onset']['count'], 1)

    def test_sparse_reward_matches_full_feature_equation(self):
        import random
        rng = random.Random(42)
        for weights in ({}, {'interval_rewards': {'7/6': .1, '7/5': .1}},
                        {'weight_5': .3, 'weight_7': .2, 'joint_weight': .4,
                         'interval_rewards': {'5/4': .2, '7/4': .1}}):
            m = self.metric(**weights)
            c = m.config
            for _ in range(100):
                pitches = [rng.randrange(-100, 150) for _ in range(rng.randrange(7))]
                s5, s7, _ = m.features(pitches)
                directed = sum(c['interval_rewards'][k]*v
                               for k, v in m.interval_features(pitches).items())
                for context in ('attack', 'sounding'):
                    expected = c[context+'_weight']*(directed+c['weight_5']*s5
                               +c['weight_7']*s7+c['joint_weight']*min(s5, s7))
                    self.assertEqual(m.reward(pitches, context), expected)


if __name__ == '__main__':
    unittest.main()
