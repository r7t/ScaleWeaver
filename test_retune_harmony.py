import unittest
from unittest.mock import patch

import harmony_rhythm as hr
import score_builder as sb
from scale_config import load_scale
from retune_harmony import (metrical_weight, harmonic_boundaries, marginal_cse,
                            segment_evidence, infer_harmony)


class RetuneHarmonyTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_scale('scales/tiangan_72.json')
        hr.configure_scale(self.spec, None)
        self.measure = dict(bar=0, start_beat=0., duration_beats=4., time_signature='4/4')

    def test_metric_hierarchy_and_meter_boundaries(self):
        self.assertEqual([metrical_weight(t, '4/4') for t in (0, 2, 1, .5)], [4, 3, 1.5, 1])
        for signature, duration, expected in [('4/4', 4, [0, 2, 4]),
                ('3/4', 3, [0, 3]), ('6/8', 3, [0, 1.5, 3]),
                ('9/8', 4.5, [0, 1.5, 3, 4.5]), ('4/4', 1, [0, 1])]:
            self.assertEqual(harmonic_boundaries(dict(time_signature=signature,
                                                     duration_beats=duration)), expected)

    def test_marginal_uses_raw_difference_and_same_background(self):
        chord = hr.CHORDS['P3_024']
        class Bundle:
            def entry(self, steps, **kwargs):
                return (len(steps) ** 2, .5)
        with patch.object(hr, 'CSE', Bundle()):
            self.assertEqual(marginal_cse(chord, 79, {}), 7.)
        with patch.object(hr, 'CSE', None):
            self.assertIsNone(marginal_cse(chord, 79, {}))

    def test_sustained_note_contributes_to_both_harmonic_slots(self):
        chord = hr.CHORDS['P3_024']
        note = dict(start_beat=1., duration_beats=2., step=79)
        a = segment_evidence(chord, [note], self.measure, 0, 2, {})
        b = segment_evidence(chord, [note], self.measure, 2, 2, {})
        self.assertEqual(a['nonchord_fraction'], 1.)
        self.assertEqual(b['nonchord_fraction'], 1.)
        self.assertEqual(a['evidence_weight'], 1.5)
        self.assertEqual(b['evidence_weight'], 1.)
        self.assertEqual(b['cse_coverage'], 0.)

    def test_melody_changes_chord_choice_without_changing_notes(self):
        import copy
        def notes(pcs):
            return [dict(start_beat=i*.5, duration_beats=.5, step=72+pcs[i%3]) for i in range(8)]
        a, b = notes((0, 16, 30)), notes((16, 35, 58))
        original = copy.deepcopy(a)
        pa, pb = infer_harmony(a, [self.measure]), infer_harmony(b, [self.measure])
        self.assertEqual(a, original)
        self.assertEqual(set(pa[0]['voicing_pcs']), {0, 16, 30})
        self.assertEqual(set(pb[0]['voicing_pcs']), {16, 35, 58})
        self.assertEqual(pa, infer_harmony(a, [self.measure]))

    def test_rest_measure_has_complete_harmony(self):
        plan = infer_harmony([], [self.measure])
        self.assertEqual(sum(s['duration'] for s in plan[0]['chord_segments']), 4.)
        self.assertTrue(all(s['melody_inference']['evidence_weight']==0
                            for s in plan[0]['chord_segments']))

    def test_marginal_cse_breaks_equal_coverage_tie(self):
        a, b = hr.CHORDS['P3_024'], hr.CHORDS['P3_036']
        events = [dict(start_beat=0., duration_beats=4., step=72)]
        def evidence(chord, *args):
            return dict(nonchord_fraction=0., marginal_cse=0. if chord.id==b.id else 1.,
                        cse_coverage=1., evidence_weight=4., missing_cse_weight=0.)
        with patch.object(hr, 'CHORDS', {a.id:a, b.id:b}), \
             patch('retune_harmony.segment_evidence', side_effect=evidence):
            plan = infer_harmony(events, [self.measure])
        self.assertTrue(all(s['chord_id']==b.id for s in plan[0]['chord_segments']))

    def test_automatic_retune_uses_inference_without_random_harmony(self):
        import retune_melody as rt
        sb.configure_scale(self.spec)
        mapped = [dict(bar=0, start_beat=i*.5, duration_beats=.5, step=72+(0,16,30)[i%3],
                       source_actual_cents=6000., source_scale_degree_index=0,
                       target_pitch_class_step=(0,16,30)[i%3]) for i in range(8)]
        analysis = dict(reference={'measure_map':[self.measure]},
                        source_scale_note_count=1, absolute_pitch_count=1,
                        source_pitch_classes_cents=(0.,))
        mapping = dict(source_pc_to_target_step={0:0}, maximum_mapping_error_cents=0.)
        with patch.object(rt, 'analyze_reference', return_value=analysis), \
             patch.object(rt, 'retune_events', return_value=(mapped, mapping)), \
             patch.object(rt, '_scaled_harmony_plan', side_effect=AssertionError('random harmony')):
            ir = rt.generate_frontend_ir('unused.mscx', spec=self.spec)
        self.assertEqual(ir['generator']['retune']['harmony_inference'], 'metrical_marginal_cse_v1')
        self.assertEqual([(e['step'],e['start_beat'],e['duration_beats']) for e in ir['lead']],
                         [(e['step'],e['start_beat'],e['duration_beats']) for e in mapped])

    def test_similarity_accepts_related_nonidentical_chords(self):
        from analyze_retune_harmony import harmonic_similarity
        a, b = hr.CHORDS['P3_024'], hr.CHORDS['P3_026']
        result = harmonic_similarity(dict(pcs=a.pcs), dict(voicing_pcs=b.pcs, chord_id=b.id))
        self.assertEqual(result['shared_tones'], 2)
        self.assertAlmostEqual(result['pitch_dice'], 2./3.)
        self.assertIsNone(result['same_three_tone'])

    def test_explicit_progression_bypasses_inference_and_preserves_retuned_events(self):
        import retune_melody as rt
        snapshot = self.spec.runtime_snapshot()
        snapshot['rules']['harmony']['chord_codes'] = {'x': 'P3_024'}
        snapshot['rules']['harmony']['chord_progression'] = {'codes': ['x'], 'bars_per_chord': 1}
        spec = load_scale(snapshot)
        hr.configure_scale(spec, None)
        sb.configure_scale(spec)
        mapped = [dict(bar=0, start_beat=0., duration_beats=4., step=72,
                       source_actual_cents=6000., source_scale_degree_index=0,
                       target_pitch_class_step=0)]
        analysis = dict(reference={'measure_map':[self.measure]},
                        source_scale_note_count=1, absolute_pitch_count=1,
                        source_pitch_classes_cents=(0.,))
        mapping = dict(source_pc_to_target_step={0:0}, maximum_mapping_error_cents=0.)
        with patch.object(rt, 'analyze_reference', return_value=analysis), \
             patch.object(rt, 'retune_events', return_value=(mapped, mapping)), \
             patch('retune_harmony.infer_harmony', side_effect=AssertionError('must preserve manual plan')):
            ir = rt.generate_frontend_ir('unused.mscx', spec=spec)
        self.assertEqual(ir['generator']['retune']['harmony_inference'], 'explicit_progression')
        self.assertEqual(ir['lead'][0]['step'], 72)
        self.assertEqual(ir['lead'][0]['duration_beats'], 4.)
        self.assertEqual(ir['harmony_plan'][0]['chord_id'], 'P3_024')


if __name__ == '__main__':
    unittest.main()
