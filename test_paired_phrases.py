"""Nested answers must survive realization, meter changes and harmony changes."""
from pathlib import Path
import unittest
from unittest.mock import patch

import harmony_rhythm as hr
import score_builder as sb
from melody_plan import generate, resolve_melody_plan
from joint_frontend_generate import generate_frontend_ir, _onsets
from scale_config import load_scale
from test_lead_ties import ConstantBundle
from phrase_skeleton import cadence_skeleton


class PairedPhraseTests(unittest.TestCase):
    def test_answer_variants_share_a_bar_transform(self):
        types = set()
        for seed in range(12):
            actions = generate(seed, 8)['unit_actions']
            a, b = (actions[i]['paired_phrase'] for i in (2, 3))
            self.assertEqual(a['answer_type'], b['answer_type'])
            self.assertEqual(a['sequence_shift'], b['sequence_shift'])
            types.add(a['answer_type'])
        self.assertEqual(types, {'opening', 'ending', 'sequence', 'rhythm'})

    def test_cadence_and_ornaments_across_scales(self):
        from test_natural_progression import CanonicalJI, MajorBiasedBundle
        folder = Path(__file__).parent/'scales'
        for name in ('major_31', 'tiangan_72'):
            spec = load_scale(folder/f'{name}.json', require_composition=True)
            hr.configure_scale(spec, MajorBiasedBundle() if name == 'major_31'
                               else ConstantBundle(),
                               CanonicalJI() if name == 'major_31' else None)
            sb.configure_scale(spec)
            chord = next(iter(hr.CHORDS.values()))
            harmony = [{'chord_id': chord.id, 'offset': 0., 'duration': 2.}]
            for previous in sb.POOLS['lead']:
                anchors = cadence_skeleton(previous, harmony, 2., .25)
                pitches = [previous]+[a['step'] for a in anchors]
                self.assertTrue(all(sb.lead_jump_ok(a,b) for a,b in zip(pitches,pitches[1:])))
                self.assertIn(anchors[-1]['step'] % hr.OCT, chord.pcs)
                if len(anchors) == 2:
                    self.assertEqual(abs(hr.degree(pitches[-1])-hr.degree(pitches[-2])),1)

    def test_harmonic_rhythm_preserves_order_and_duration(self):
        import random
        folder = Path(__file__).parent/'scales'
        for name in ('major_31', 'tiangan_72'):
            spec = load_scale(folder/f'{name}.json', require_composition=True)
            hr.configure_scale(spec, ConstantBundle())
            ids = list(hr.CHORDS)[:3]
            template = [[{'chord_id': cid, 'offset': 0., 'duration': 4.}]
                        for cid in ids]
            result = hr.vary_harmonic_rhythm(template, random.Random(19), 4.)
            self.assertTrue(any(len(bar) > 1 for bar in result))
            sequence = []
            for bar in result:
                self.assertAlmostEqual(sum(s['duration'] for s in bar), 4.)
                offset = 0.
                for segment in bar:
                    self.assertAlmostEqual(segment['offset'], offset)
                    offset += segment['duration']
                    if not sequence or sequence[-1] != segment['chord_id']:
                        sequence.append(segment['chord_id'])
            self.assertEqual(sequence, ids)

    def test_backward_nested_edges_and_partial_phrases(self):
        for bars in (1, 2, 3, 5, 8, 13, 48):
            plan = generate(17, bars)
            for action in plan['unit_actions']:
                source = action['paired_phrase']['source_unit']
                if source is not None:
                    self.assertLess(source, action['unit'])
                    self.assertEqual(source//16, action['unit']//16)
            if bars >= 8:
                self.assertEqual([a['paired_phrase']['source_unit']
                                  for a in plan['unit_actions'][:16:2]],
                                 [None, 0, 0, 2, 0, 2, 4, 6])

    def test_switch_and_validation(self):
        self.assertNotIn('paired_phrase', generate(1, 8, supplied={
            'paired_phrases': False})['unit_actions'][0])
        with self.assertRaises(ValueError):
            resolve_melody_plan({'paired_phrases': 'yes'})

    def test_manual_phrase_edges_stay_local(self):
        for span in (5, 7):
            plan = generate(31, span*2, manual_progression_bars=span)
            for action in plan['unit_actions']:
                source = action['paired_phrase']['source_unit']
                if source is not None:
                    self.assertEqual(source//(span*2), action['unit']//(span*2))
            for end in (span*2-1, span*4-1):
                self.assertTrue(plan['unit_actions'][end]['paired_phrase']['closing'])

    def test_realized_answers_anchors_and_closure(self):
        folder = Path(__file__).parent/'scales'
        spec = load_scale(folder/'tiangan_72.json',
                          rules=folder/'tiangan_72_5.rules.json',
                          style=folder/'tiangan_72_5_norm.style.json',
                          require_composition=True)
        hr.configure_scale(spec, ConstantBundle())
        sb.configure_scale(spec)
        for meter, sixteenth in [('4/4', True), ('3/4', False), ('4/4', False)]:
            with self.subTest(meter=meter):
                with patch.object(sb.cse_rt, 'lead_chord_field_cost', return_value=0.):
                    ir = generate_frontend_ir(17, 8, time_signature=meter,
                                             allow_sixteenth=sixteenth, spec=spec)
                plan = ir['melody_plan']
                materials = plan['realized_materials']
                for i in (2, 3, 8, 9, 10, 11):
                    relation = plan['unit_actions'][i]['paired_phrase']
                    source = relation['source_unit']
                    kind = relation['answer_type']
                    linked = (kind in {'rhythm', 'sequence'} or
                              kind == 'opening' and i % 2 == 0 or
                              kind == 'ending' and i % 2 == 1)
                    if (linked or relation.get('rhythm_recall')) and not relation.get('rhythm_independent'):
                        self.assertEqual(materials[i]['rhythm'], materials[source]['rhythm'])
                self.assertLessEqual(len(materials[-1]['rhythm']), 2)
                for material in materials:
                    self.assertTrue(material['skeleton'])
                    for anchor in material['skeleton']:
                        self.assertIn(round(anchor['offset'], 6), _onsets(material['rhythm']))
                        time = material['unit']*ir['beats_per_bar']/2+anchor['offset']
                        event = next(e for e in ir['lead'] if
                                     e['start_beat'] <= time+1e-7 and
                                     e['start_beat']+e['duration_beats'] > time+1e-7)
                        self.assertEqual(event['step'], anchor['step'])
                self.assertFalse(sb.lead_jump_errors(ir['lead']))


if __name__ == '__main__':
    unittest.main()
