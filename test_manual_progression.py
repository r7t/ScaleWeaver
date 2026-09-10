"""Contract tests: no cache builds or audio required."""
from pathlib import Path
import random
import unittest
import harmony_rhythm as hr
from scale_config import load_scale

class ConstantBundle:
    def mixed_entry(self, steps, *weights):
        return (1.0, 0.5)

class ManualProgressionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).parent / 'scales'
        self.spec = load_scale(self.folder/'major_31.json', require_composition=True)

    def configured(self, progression):
        snapshot = self.spec.runtime_snapshot()
        snapshot['rules']['harmony']['chord_progression'] = progression
        spec = load_scale(snapshot)
        hr.configure_scale(spec, ConstantBundle())
        return spec

    def test_exact_cycles_roots_and_partial_ending(self):
        hr.configure_scale(self.spec, ConstantBundle())
        rng = random.Random(31)
        state = rng.getstate()
        plan = hr.harmony_plan(17, rng, 4)
        self.assertEqual(rng.getstate(), state)
        self.assertEqual([h['root_degree'] for h in plan], [4,5,3,6,2,5,1]*2+[4,5,3])
        self.assertEqual(plan[0]['voicing_pcs'], [13,23,0])
        self.assertEqual(plan[1]['voicing_pcs'], [18,28,5])
        self.assertEqual(plan[-1]['voicing_pcs'], [10,18,28])
        for h in plan:
            chord = hr.chord_at(plan, h['bar']*4+3.9, 4)
            self.assertEqual(chord.foot, h['root_pc'])
            self.assertEqual(h['chords_in_bar'], 1)
            self.assertEqual(h['chord_segments'][0]['duration'], 4.)

    def test_three_four_and_two_bars_per_chord(self):
        self.configured({'degrees':[4,5,1], 'bars_per_chord':2})
        plan = hr.harmony_plan(8, random.Random(0), 3)
        self.assertEqual([h['root_degree'] for h in plan], [4,4,5,5,1,1,4,4])
        self.assertTrue(all(h['chord_segments'][0]['duration']==3 for h in plan))

    def test_snapshot_roundtrip(self):
        restored = load_scale(self.spec.runtime_snapshot())
        self.assertEqual(restored.resolved_chord_progression(), self.spec.resolved_chord_progression())
        self.assertEqual(restored.resolved_ngram_path(), self.spec.resolved_ngram_path())

    def test_auto_after_manual_has_no_state_leak(self):
        snapshot = self.spec.runtime_snapshot()
        snapshot['rules']['harmony'].pop('chord_progression')
        automatic = load_scale(snapshot)
        hr.configure_scale(automatic, ConstantBundle())
        before = hr.harmony_plan(16, random.Random(31), 4)
        hr.configure_scale(self.spec, ConstantBundle())
        hr.harmony_plan(14, random.Random(31), 4)
        hr.configure_scale(automatic, ConstantBundle())
        after = hr.harmony_plan(16, random.Random(31), 4)
        self.assertEqual(before, after)
        self.assertFalse(any(cid.startswith('MANUAL_') for cid in hr.CHORDS))
        self.assertEqual(after[0]['function'], 'T')
        self.assertEqual(after[-1]['end_function'], 'T')

    def test_bad_inputs_fail_during_load(self):
        for value in ({'degrees':[]}, {'degrees':[0]}, {'degrees':[8]},
                      {'degrees':[True]}, {'degrees':[1.5]}, {'degrees':'4536251'},
                      {'degrees':[4],'bars_per_chord':0},
                      {'degrees':[4],'bars_per_chord':1.5}, {'degree':[4]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.configured(value)

    def test_does_not_bypass_octave_equivalent_wolves(self):
        snapshot = self.spec.runtime_snapshot()
        snapshot['rules']['wolf_intervals']=[13]
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            load_scale(snapshot)

    def test_ngram_exact_degree_mapping(self):
        mapping=dict(zip([0,29,55,71,100,126,155],self.spec.pcs))
        def rows(path):
            return [line.split('\t') for line in path.read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
        old=rows(self.folder/'ji_major_171.ngram.txt')
        new=rows(self.folder/'major_31.ngram.txt')
        self.assertEqual(len(old),len(new))
        for a,b in zip(old,new):
            self.assertEqual((a[0],a[1],a[3]),(b[0],b[1],b[3]))
            self.assertEqual([mapping[int(x)] for x in a[2].split(',')],
                             [int(x) for x in b[2].split(',')])

if __name__=='__main__':
    unittest.main()
