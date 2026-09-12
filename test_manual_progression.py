"""Contract tests: no cache builds or audio required."""
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import harmony_rhythm as hr
import score_builder as sb
from joint_frontend_generate import generate_frontend_ir
from melody_plan import MODE_LAYERS, generate as generate_melody_plan
from scale_config import load_scale

class ConstantBundle:
    def mixed_entry(self, steps, *weights):
        return (1.0, 0.5)

class ManualProgressionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).parent / 'scales'
        automatic = load_scale(self.folder/'major_31.json', require_composition=True)
        snapshot = automatic.runtime_snapshot()
        snapshot['rules']['harmony']['chord_progression'] = {
            'degrees': [4, 5, 3, 6, 2, 5, 1],
            'bars_per_chord': 1,
        }
        self.spec = load_scale(snapshot)

    def configured(self, progression):
        snapshot = self.spec.runtime_snapshot()
        snapshot['rules']['harmony']['chord_progression'] = progression
        spec = load_scale(snapshot)
        hr.configure_scale(spec, ConstantBundle())
        return spec

    def test_natural_scales_default_to_automatic_harmony(self):
        for name in ('major_12', 'major_19', 'major_22', 'major_31', 'minor_22'):
            with self.subTest(scale=name):
                spec = load_scale(self.folder/f'{name}.json', require_composition=True)
                self.assertIsNone(spec.resolved_chord_progression())

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

    def test_manual_melodyplan_phrase_and_direct_bar_children(self):
        plan=generate_melody_plan(31,bars=14,manual_progression_bars=7)
        self.assertTrue(plan['manual_progression'])
        self.assertEqual(plan['section_size_bars'],7)
        self.assertEqual([s['span_bars'] for s in plan['sections']],[7,7])
        self.assertEqual(len(plan['leaf_actions']),14)
        self.assertTrue(all(x['span_units']==2 and x['duration_bars']==1
                            for x in plan['leaf_actions']))
        nodes={n['id']:n for n in plan['hierarchy_nodes']}
        roots=[n for n in nodes.values() if n['parent'] is None]
        self.assertEqual(len(roots),2)
        self.assertTrue(all(len(n['children'])==7 for n in roots))
        self.assertTrue(all(not nodes[c]['children'] for n in roots for c in n['children']))

    def test_manual_internal_relations_are_rhythm_only(self):
        cfg={
            'cross_phrase_form_weights':
                {'AABBAB':1,'AAB_CBC':0,'ABCABC':0,'ABACBC':0,'ABCDAC':0},
            'macro_pattern_weights':
                {'uniform':0,'motif_over_harmony':0,'rhythm_identity':0,
                 'reharmonized_reprise':1,'fresh_tail':0},
            'copy_length_weights':
                {'0.5':0,'1':1,'1.5':0,'2':0,'2.5':0,'3':0,'3.5':0,
                 '4':0,'5':0,'6':0,'7':10,'8':0},
            'internal_recurrence':
                {'eight_bar_probability':1,'four_bar_probability':1,
                 'two_bar_probability':1},
        }
        plan=generate_melody_plan(31,bars=42,supplied=cfg,
                                  manual_progression_bars=7)
        internal=[r for r in plan['relations'] if 'internal' in r['scope']]
        cross=[r for r in plan['relations'] if 'cross' in r['scope'] or
               'reprise' in r['scope'] and 'internal' not in r['scope']]
        self.assertTrue(internal)
        self.assertTrue(all(r['mode']=='R' and 'M' not in r['layers'] for r in internal))
        self.assertTrue(any('M' in MODE_LAYERS[r['mode']] for r in cross))
        for relation in cross:
            if 'M' in MODE_LAYERS[relation['mode']]:
                for target in relation['target_units']:
                    action=plan['unit_actions'][target]
                    self.assertEqual(target % 14,action['source_unit'] % 14)

    def test_manual_melodyplan_rejects_partial_phrase(self):
        with self.assertRaisesRegex(ValueError,'exact multiple'):
            generate_melody_plan(31,bars=48,manual_progression_bars=7)

    def test_five_bar_rise_reserves_two_bar_cadential_descent(self):
        contour = sb._lead_phrase_contour('rise', 5, 0, 4)
        self.assertEqual(contour.index(max(contour)), 2)
        self.assertGreater(contour[2], contour[3])
        self.assertGreater(contour[3], contour[4])

    def test_joint_frontend_uses_manual_phrase_grid(self):
        hr.configure_scale(self.spec,ConstantBundle())
        sb.configure_scale(self.spec)
        with patch.object(sb.cse_rt,'lead_chord_field_cost',lambda *args:0.0):
            ir=generate_frontend_ir(seed=31,bars=14,spec=self.spec)
        plan=ir['melody_plan']
        self.assertEqual(plan['section_size_bars'],7)
        self.assertEqual(len(plan['leaf_actions']),14)
        self.assertEqual([h['phrase_index'] for h in ir['harmony_plan']],
                         [0]*7+[1]*7)
        self.assertEqual([h['bar_in_phrase'] for h in ir['harmony_plan']],
                         list(range(7))*2)
        self.assertTrue(all(r['mode']=='R' for r in plan['relations']
                            if 'internal' in r['scope']))

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
