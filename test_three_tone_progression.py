"""Integration contracts for the opt-in planner, without cache generation."""
import hashlib
import json
import random
import unittest
from unittest.mock import patch

import harmony_rhythm as hr
import score_builder as sb
from scale_config import load_scale
from test_manual_progression import ConstantBundle
from test_natural_progression import CanonicalJI, MajorBiasedBundle
from three_tone_progression import resolve_config, function_profile, resolution_score


def configure(name='tiangan_72', enabled=True, progression=None):
    spec = load_scale('scales/'+name+'.json', require_composition=True)
    snapshot = spec.runtime_snapshot()
    snapshot['rules']['harmony']['three_tone_progression'] = {'enabled':enabled}
    if progression:
        snapshot['rules']['harmony']['chord_progression'] = progression
    spec = load_scale(snapshot)
    bundle, ji = ((MajorBiasedBundle(), CanonicalJI()) if name == 'major_31'
                  else (ConstantBundle(), None))
    hr.configure_scale(spec, bundle, ji)
    return spec


class ThreeToneProgressionTests(unittest.TestCase):
    def test_legacy_output_is_unchanged_and_configuration_resets(self):
        expected = {'tiangan_72':'fcbcc072cd15dab9caa69dc59e7435881e96a88d8568b7e51134b86dd7d728f0',
                    'major_31':'2f9fdcaeb998d283f0a9f6846436213eb704de448187c4f771c9d3dd126e2599'}
        for name, digest in expected.items():
            configure(name)
            configure(name, enabled=False)
            plan = hr.harmony_plan(13, random.Random(42), 4)
            self.assertEqual(hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), digest)
            self.assertEqual(hr.THREE_TONE_INFO, {})

    def test_phrase_lengths_meters_and_endings(self):
        for name, anchor in [('tiangan_72',42), ('major_31',18)]:
            configure(name)
            for bars in (1,2,3,4,7,8,9,13,17):
                for bpb in (3,4):
                    for seed in range(4):
                        plan = hr.harmony_plan(bars,random.Random(seed),bpb)
                        self.assertEqual(len(plan), bars)
                        for i, row in enumerate(plan):
                            self.assertEqual(row['harmony_model'],'three_tone')
                            self.assertAlmostEqual(sum(s['duration'] for s in row['chord_segments']),bpb)
                            self.assertEqual(row['three_tone_pc'],row['chord_segments'][0]['three_tone_pc'])
                            if (i+1) % 8 == 0 or i == bars-1:
                                last = row['chord_segments'][-1]
                                self.assertEqual(last['three_tone_pc'],anchor)
                                self.assertIn(0,last['voicing_pcs'])
                                self.assertEqual(last['root_pc'],0)
                                self.assertFalse(last['three_tone_cadence_fallback'])

    def test_classification_and_deterministic_planning(self):
        configure()
        self.assertEqual(hr.THREE_TONE_INFO['P3_057']['three_tone_reason'],'completion_outside_scale')
        self.assertEqual(hr.THREE_TONE_INFO['P3_038']['three_tone_status'],'implied')
        self.assertEqual(hr.harmony_plan(17,random.Random(5),4),hr.harmony_plan(17,random.Random(5),4))
        configure('major_31')
        ii = next(c for c in hr.CHORDS.values() if set(c.pcs)=={5,13,23})
        self.assertEqual(hr.THREE_TONE_INFO[ii.id]['three_tone_pc'],13)

    def test_manual_progression_still_wins(self):
        configure(progression={'codes':['j','0','a'], 'bars_per_chord':1})
        rng = random.Random(1)
        state = rng.getstate()
        plan = hr.harmony_plan(3,rng,4)
        self.assertEqual(rng.getstate(),state)
        self.assertEqual([r['chord_code'] for r in plan],['j','0','a'])
        self.assertEqual(plan[0]['three_tone_status'],'outside_scale')
        self.assertEqual(plan[0]['three_tone_pc'],19)
        self.assertEqual(plan[0]['function'],hr.CHORDS['P3_057'].function)

    def test_missing_ratios_and_invalid_config_are_explicit(self):
        spec = configure('major_31')
        hr.configure_scale(spec,ConstantBundle())
        with self.assertRaisesRegex(ValueError,'reliable local ratios'):
            hr.harmony_plan(8,random.Random(1),4)
        for bad in ({'enabled':'true'},{'special_weight':float('nan')},{'max_integer':1}):
            with self.assertRaises(ValueError):
                resolve_config({'three_tone_progression':bad})

    def test_joint_frontend_preserves_new_route(self):
        from joint_frontend_generate import generate_frontend_ir
        spec = configure()
        sb.configure_scale(spec)
        expected = hr.harmony_plan(4,random.Random(48),4)
        with patch.object(sb.cse_rt,'lead_chord_field_cost',lambda *args:0.):
            ir = generate_frontend_ir(seed=48,bars=4,spec=spec)
        def route(plan):
            return [[(s['chord_id'],s['offset'],s['duration'],s['three_tone_pc'])
                     for s in row['chord_segments']] for row in plan]
        self.assertEqual(route(ir['harmony_plan']),route(expected))

    def test_new_cadence_survives_empty_legacy_anchor_set(self):
        from joint_frontend_generate import generate_frontend_ir
        from harmonic_realization import is_anchor, musical_root
        spec = configure()
        sb.configure_scale(spec)
        with patch.object(hr,'ANCHOR_CHORD_IDS',()), patch.object(
                sb.cse_rt,'lead_chord_field_cost',lambda *args:0.):
            ir = generate_frontend_ir(seed=3,bars=2,spec=spec)
            final = ir['harmony_plan'][-1]['chord_segments'][-1]
            self.assertEqual(final['three_tone_pc'],42)
            self.assertTrue(is_anchor(hr,final))
            self.assertEqual(musical_root(hr,final),0)

    def test_cli_exposes_both_models(self):
        from main import _build_cli_parser
        parser = _build_cli_parser()
        for model in ('legacy','three-tone'):
            args = parser.parse_args(['--scale','scales/tiangan_72.json','--harmony-model',model])
            self.assertEqual(args.harmony_model,model)

    def test_mode_tonic_and_missing_fifth_group(self):
        for name, tonic, dominant, subdominant, expected, fallback in (
                ('major_31',18,5,0,5,False),
                ('tiangan_72',42,7,0,None,True)):
            spec = configure(name)
            snapshot = spec.runtime_snapshot()
            snapshot['rules']['harmony']['pitch_roles'] = {
                'tonic':[tonic], 'dominant':[dominant], 'subdominant':[subdominant],
                'other':[pc for pc in spec.pcs if pc not in (tonic,dominant,subdominant)]}
            spec = load_scale(snapshot)
            hr.configure_scale(spec,ConstantBundle(),CanonicalJI() if name=='major_31' else None)
            final = hr.harmony_plan(3,random.Random(4),4)[-1]['chord_segments'][-1]
            self.assertEqual(final['root_pc'],tonic)
            self.assertEqual(final['three_tone_cadence_fallback'],fallback)
            if not fallback:
                self.assertEqual(final['three_tone_pc'],expected)

    def test_external_group_is_available_to_normal_functional_planning(self):
        configure()
        original = hr.weighted_choice
        def select_external_when_available(items, rng):
            items = list(items)
            if any(value == 19 for value, weight in items):
                return 19
            return original(items, rng)
        with patch.object(hr,'weighted_choice',side_effect=select_external_when_available):
            plan = hr.harmony_plan(8,random.Random(4),4)
        external = [s for row in plan for s in row['chord_segments']
                    if s['three_tone_status']=='outside_scale']
        self.assertTrue(external)
        for s in external:
            self.assertEqual(s['chord_id'],'P3_057')
            self.assertEqual(s['three_tone_pc'],19)
            self.assertEqual(s['three_tone_engine_version'],2)
            self.assertIn('three_tone_function_weights',s)
            self.assertNotIn(19,s['voicing_pcs'])
        self.assertEqual(plan[-1]['chord_segments'][-1]['three_tone_status'],'present')

    def test_function_axes_support_and_resolution(self):
        for pc, fn in ((42,'T'),(0,'S'),(12,'D')):
            profile=function_profile(pc,'present',0,72)
            self.assertEqual(profile['three_tone_function'],fn)
        strengths=[function_profile(42,status,0,72)['three_tone_tonic_strength']
                   for status in ('present','implied','outside_scale')]
        self.assertGreater(strengths[0],strengths[1])
        self.assertGreater(strengths[1],strengths[2])
        self.assertGreater(resolution_score(12,.8,42,'present',0,72),
                           resolution_score(12,.8,12,'present',0,72))
        # External 19 also participates in the fifth chain, towards 49.
        self.assertGreater(resolution_score(19,.8,49,'present',0,72),
                           resolution_score(19,.8,19,'outside_scale',0,72))

    def test_complexity_filter_preserves_classification(self):
        spec=configure()
        snapshot=spec.runtime_snapshot()
        snapshot['rules']['harmony']['three_tone_progression']['max_integer']=3
        hr.configure_scale(load_scale(snapshot),ConstantBundle())
        row=hr.THREE_TONE_INFO['P3_057']
        self.assertEqual((row['three_tone_pc'],row['three_tone_status']),(19,'outside_scale'))
        self.assertFalse(row['three_tone_eligible'])
        self.assertEqual(row['three_tone_exclusion_reason'],'ratio_complexity_limit')

    def test_old_config_alias_and_external_disable(self):
        self.assertEqual(resolve_config({'three_tone_progression':{'special_weight':.1}})
                         ['outside_scale_weight'],.1)
        configure()
        hr.THREE_TONE_CONFIG['outside_scale_weight']=0.
        plan=hr.harmony_plan(24,random.Random(1),4)
        self.assertFalse(any(s['three_tone_status']=='outside_scale'
                             for r in plan for s in r['chord_segments']))


if __name__ == '__main__':
    unittest.main()
