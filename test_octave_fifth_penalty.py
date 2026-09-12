import itertools
import math
import unittest

from annealing_config import resolve_annealing
from harmony_annealing import (Energy, _octave_fifth_four_patterns,
                               octave_family_count,
                               octave_fifth_four_subset,
                               octave_fifth_three_subset_count)


class OctaveFifthPenaltyTests(unittest.TestCase):
    def test_three_subset_examples(self):
        steps=lambda ns:tuple(round(72*math.log2(n)) for n in ns)
        self.assertEqual(octave_fifth_three_subset_count(steps((1,2,3,5,7)),72),1)
        self.assertEqual(octave_family_count(steps((1,2,3,5,7)),72),2)
        self.assertEqual(octave_fifth_three_subset_count(steps((2,3,4,5,10)),72),1)
        self.assertEqual(octave_family_count(steps((2,3,4,5,10)),72),3)
        self.assertEqual(octave_fifth_three_subset_count(steps((3,4,8)),72),0)
        self.assertEqual(octave_family_count(steps((3,4,8)),72),1)

    def test_only_adjacent_members_of_an_octave_family_count(self):
        steps=lambda ns:tuple(round(72*math.log2(n)) for n in ns)
        # Complete octave chains have n-1 edges, not all n-choose-2 pairs.
        self.assertEqual(octave_family_count(steps((1,2,4,8,16)),72),4)
        # 1:2:8 contributes two adjacent family edges; 5:10 contributes one.
        self.assertEqual(octave_family_count(steps((1,2,5,8,10)),72),3)
        self.assertEqual(octave_family_count(steps((1,2,4)),72),2)
        self.assertEqual(octave_family_count(steps((1,4)),72),1)
        self.assertEqual(octave_family_count(steps((1,16)),72),1)

    def test_both_direct_triad_shapes_and_transposition(self):
        self.assertEqual(octave_fifth_three_subset_count((11,83,125),72),1)  # 1:2:3
        self.assertEqual(octave_fifth_three_subset_count((11,53,83),72),1)   # 2:3:4

    def test_only_adjacent_attack_or_sounding_notes_form_virtual_octaves(self):
        steps=lambda ns:tuple(round(72*math.log2(n)) for n in ns)
        self.assertEqual(octave_fifth_three_subset_count(steps((1,2,3,7)),72),1)
        self.assertEqual(octave_fifth_three_subset_count(steps((2,3,4,5)),72),1)
        # These contain matching non-adjacent subsets 2:4:6 and 4:6:8,
        # respectively, but an intervening sounding/attacking note hides them.
        self.assertEqual(octave_fifth_three_subset_count(steps((2,4,5,6)),72),0)
        self.assertEqual(octave_fifth_three_subset_count(steps((4,5,6,8)),72),0)

    def test_overlapping_adjacent_windows_count_separately(self):
        steps=lambda ns:tuple(round(72*math.log2(n)) for n in ns)
        self.assertEqual(octave_fifth_three_subset_count(steps((1,2,3,4)),72),2)
        self.assertEqual(octave_family_count(steps((1,2,3,4)),72),4)

    def test_exact_harmonic_subsets(self):
        harmonics=tuple(round(72*math.log2(n)) for n in (1,2,3,4,6,8,12,16))
        for row in itertools.combinations(harmonics,4):
            five=tuple(x+11 for x in row)+(400,)
            self.assertTrue(octave_fifth_four_subset(tuple(sorted(five)),72),row)

    def test_nonmatching_and_nonfive(self):
        self.assertFalse(octave_fifth_four_subset((0,72,114,167,239),72))
        self.assertFalse(octave_fifth_four_subset((0,72,114,144),72))

    def test_config_default_and_override(self):
        self.assertEqual(resolve_annealing()['sounding_octave_fifth_subset_cost'],0)
        self.assertEqual(resolve_annealing({'sounding_octave_fifth_subset_cost':1.25})
                         ['sounding_octave_fifth_subset_cost'],1.25)

    def test_sounding_and_attack_use_same_penalty(self):
        energy=Energy.__new__(Energy)
        energy.pitches=[0,72,114,144,300]
        energy.config={'sounding_octave_family_pair_allowance':{'5':2},
                       'sounding_octave_excess_cost':1.0,
                       'sounding_octave_fifth_subset_cost':1.0}
        class Metric:
            edo=72
            sounding_cost=staticmethod(lambda xs: 0.0)
            attack_cost=staticmethod(lambda xs: 0.0)
        energy.metric=Metric()
        # Two adjacent octave-family edges + both 1:2:3 and 2:3:4 triples,
        # with an allowance of two, plus the existing four-note-subset cost.
        self.assertEqual(energy.evaluate(('sounding',(0,1,2,3,4),1.0,None)),3.0)
        self.assertEqual(energy.evaluate(('attack',(0,1,2,3,4),1.0,None)),3.0)

    def test_virtual_octaves_apply_to_three_four_five_attack_and_sounding(self):
        energy=Energy.__new__(Energy)
        energy.pitches=[0,72,114,167,202]
        energy.config={'sounding_octave_family_pair_allowance':
                       {'2':1,'3':1,'4':1,'5':2},
                       'sounding_octave_excess_cost':1.0,
                       'sounding_octave_fifth_subset_cost':0.0}
        class Metric:
            edo=72
            sounding_cost=staticmethod(lambda xs: 0.0)
            attack_cost=staticmethod(lambda xs: 0.0)
        energy.metric=Metric()
        for n in (3,4):
            ids=tuple(range(n))
            self.assertEqual(energy.evaluate(('sounding',ids,1.0,None)),1.0)
            self.assertEqual(energy.evaluate(('attack',ids,1.0,None)),1.0)
        ids=tuple(range(5))
        self.assertEqual(energy.evaluate(('sounding',ids,1.0,None)),0.0)
        self.assertEqual(energy.evaluate(('attack',ids,1.0,None)),0.0)


if __name__=='__main__':unittest.main()
