import random
import unittest

from melody_character import shape_cost, shape_statistics, sixteenth_cells, rhythm_preference, INTERVAL_TARGETS
from melody_plan import resolve_melody_plan
from joint_frontend_generate import _fresh_rhythms


class MelodyCharacterTests(unittest.TestCase):
    def test_history_summary_preserves_original_shape_score(self):
        rng = random.Random(626)
        for _ in range(300):
            recent = [rng.randrange(-20, 21) for _ in range(rng.randrange(1, 60))]
            candidate = rng.randrange(-20, 21)
            duration = rng.choice((.25, .5, 1.))
            pitches = recent[-25:]
            intervals = [b-a for a, b in zip(pitches, pitches[1:])]
            jump = candidate-pitches[-1]
            category = min(4, abs(jump))
            target = INTERVAL_TARGETS[category]
            expected = (sum(min(4, abs(d)) == category for d in intervals)
                        +8*target)/(len(intervals)+8)-target
            if jump == 0:
                expected += .18
            if intervals and jump and intervals[-1]:
                pairs = [(a, b) for a, b in zip(intervals, intervals[1:]) if a and b]
                turn_rate = (sum(a*b < 0 for a, b in pairs)+8*.43)/(len(pairs)+8)
                expected += .35*(turn_rate-.43)*(int(jump*intervals[-1] < 0)-.43)
            expected *= .3 if duration <= .250001 else 1.
            self.assertEqual(shape_cost(recent, candidate, int, duration,
                             shape_statistics(recent, int)), expected)
        self.assertEqual(shape_cost([], 0, int), 0.)

    def test_exact_sixteenth_units_and_no_eighth_note_confusion(self):
        self.assertEqual(sixteenth_cells((.75,.25,.25,.75,.25,.5,.25)),
                         {'3+1':2,'1+3':1,'1+2+1':1})
        self.assertEqual(sixteenth_cells((1.5,.5,.5,1.5)),
                         {'3+1':0,'1+3':0,'1+2+1':0})

    def test_weighted_sampling_changes_orientation_without_removing_cells(self):
        cells = {(.75,.25):{'sixteenth'}, (.25,.75):{'sixteenth'},
                 (.25,.5,.25):{'sixteenth'}, (.5,.5):{'basic'}}
        prefs = {'3+1':1.5,'1+3':.6,'1+2+1':.9}
        rng = random.Random(20260914)
        counts = {r:0 for r in cells}
        for _ in range(5000):
            row = _fresh_rhythms(cells, 2.5, 1, rng,
                fancy_frequency_multiplier=1., syncopated_frequency_multiplier=1.,
                sixteenth_cell_weights=prefs)[0]
            counts[row] += 1
        self.assertGreater(counts[(.75,.25)], counts[(.25,.75)]*1.7)
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertAlmostEqual(rhythm_preference((.25,.5,.25), prefs), .9)

    def test_neutral_defaults_do_not_change_random_choices(self):
        cells = {(.75,.25):{'sixteenth'}, (.25,.75):{'sixteenth'}, (.5,.5):{'basic'}}
        a = _fresh_rhythms(cells, 2, 3, random.Random(42))
        b = _fresh_rhythms(cells, 2, 3, random.Random(42),
                          sixteenth_cell_weights={'3+1':1.,'1+3':1.,'1+2+1':1.})
        self.assertEqual(a, b)

    def test_shape_feedback_favours_underrepresented_small_motion(self):
        recent = list(range(0, 60, 5))
        self.assertLess(shape_cost(recent, recent[-1]+1, int),
                        shape_cost(recent, recent[-1]+5, int))
        self.assertAlmostEqual(shape_cost(recent, recent[-1]+1, int, .25),
                               .3*shape_cost(recent, recent[-1]+1, int, .5))
        self.assertGreater(shape_cost([0], 0, int), shape_cost([0], 1, int))

    def test_invalid_preferences_rejected(self):
        for key, value in [('melodic_shape_weight', -1), ('melodic_shape_weight', float('nan')),
                           ('sixteenth_cell_weights', {'3+1':0}),
                           ('sixteenth_cell_weights', {'1+3':float('inf')})]:
            with self.assertRaises(ValueError):
                resolve_melody_plan({'joint_generation':{key:value}})

    def test_alternating_recall_reuses_generated_complex_rhythm(self):
        from melody_plan import generate
        from phrase_skeleton import expand_rhythms
        plan = generate(42, 8, supplied={'alternating_rhythm_probability':1.})
        actions = plan['unit_actions']
        self.assertTrue(actions[2]['paired_phrase']['rhythm_independent'])
        self.assertTrue(actions[4]['paired_phrase']['rhythm_recall'])
        self.assertEqual(actions[4]['paired_phrase']['source_unit'], 0)
        self.assertEqual(actions[6]['paired_phrase']['source_unit'], 2)
        rhythm = (.75,.25,.5,.25,.25)
        source = {'rhythm':rhythm}
        choices = expand_rhythms([(.5,)*4], {rhythm:None},
                                [{'offset':0.},{'offset':1.}], source,
                                actions[4]['paired_phrase'], 3)
        self.assertEqual(choices, [rhythm])
        # Closure has priority even if it falls on a recurring slot.
        closing = {**actions[4]['paired_phrase'], 'closing':True}
        self.assertEqual(expand_rhythms([(.5,)*4], {rhythm:None},
                         [{'offset':0.}], source, closing, 3), [(2.,)])


if __name__ == '__main__':
    unittest.main()
