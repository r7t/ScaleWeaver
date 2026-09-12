"""Progression contracts independent of spectral cache availability."""
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import harmony_rhythm as hr
import score_builder as sb
from joint_frontend_generate import generate_frontend_ir
from melody_plan import MODE_LAYERS
from scale_config import load_scale


class CanonicalJI:
    def lookup(self, steps):
        if len(steps) != 3:
            return None
        candidates = []
        for shape in hr._NATURAL_MAJOR_SHAPES | hr._NATURAL_MINOR_SHAPES:
            errors = [(s-steps[0])*1200/31 - 1200*math.log2(n/shape[0])
                      for s,n in zip(steps,shape)]
            rms = math.sqrt(sum(e*e for e in errors)/3)
            candidates.append((rms,shape))
        rms,shape = min(candidates)
        return {'integers':shape,'rms_error_cents':rms} if rms < 10 else None


class MajorBiasedBundle:
    def mixed_entry(self, steps, *weights):
        pcs = {s % 31 for s in steps}
        minor = pcs in ({5,13,23},{0,10,23},{10,18,28})
        return (20.0 if minor else 1.0, .5)


class NaturalProgressionTests(unittest.TestCase):
    def setUp(self):
        self.scales = Path(__file__).parent/'scales'
        self.spec = load_scale(self.scales/'major_31.json', require_composition=True)
        hr.configure_scale(self.spec, MajorBiasedBundle(), CanonicalJI())

    def test_meantone_scope_excludes_ji_and_non_meantone(self):
        for name, expected in [('major_12',True),('major_19',True),('major_31',True),
                               ('ji_major_171',False),('major_22',False),('minor_22',False)]:
            with self.subTest(scale=name):
                spec = load_scale(self.scales/f'{name}.json',require_composition=True)
                self.assertEqual(hr._is_meantone_diatonic(spec),expected)
                # Even many apparent JI triads cannot activate the route for JI.
                if not expected:
                    with patch.object(hr,'_detect_natural_scale_triads',return_value={i:{} for i in range(6)}):
                        hr.configure_scale(spec,MajorBiasedBundle())
                        self.assertFalse(hr.NATURAL_SCALE_MODE)

    def check_route(self, plan, bars):
        segments = [s for row in plan for s in row['chord_segments']]
        functions = [s['function'] for s in segments]
        self.assertNotIn(('D','S'),list(zip(functions,functions[1:])))
        self.assertEqual(segments[0]['root_degree'],1)
        self.assertEqual(segments[-1]['root_degree'],1)
        for row in plan:
            self.assertAlmostEqual(sum(s['duration'] for s in row['chord_segments']),row['beats_per_bar'])
        if bars >= 8:
            minor_time = sum(s['duration'] for s in segments if s['natural_quality']=='minor')
            self.assertGreaterEqual(minor_time,2*(bars//8)*plan[0]['beats_per_bar'])
            degrees = [s['root_degree'] for s in segments]
            compact = [d for i,d in enumerate(degrees) if not i or d!=degrees[i-1]]
            self.assertIn([2,5,1],[compact[i:i+3] for i in range(len(compact)-2)])

    def test_complete_and_partial_routes_with_costly_minor_chords(self):
        self.assertTrue(hr.NATURAL_SCALE_MODE)
        for bars in (1,2,3,4,5,6,7,8,13,14,17,48):
            for seed in range(12):
                for bpb in (3,4):
                    with self.subTest(bars=bars,seed=seed,bpb=bpb):
                        plan=hr.harmony_plan(bars,random.Random(seed),bpb)
                        self.check_route(plan,bars)

    def test_joint_search_preserves_route_and_reports_inheritance_honestly(self):
        sb.configure_scale(self.spec)
        for seed in (1,48):
            expected=hr.harmony_plan(48,random.Random(seed),4)
            with patch.object(sb.cse_rt,'lead_chord_field_cost',lambda *args:0.0):
                ir=generate_frontend_ir(seed=seed,bars=48,spec=self.spec)
            self.check_route(ir['harmony_plan'],48)
            self.assertEqual([[s['chord_id'] for s in r['chord_segments']] for r in ir['harmony_plan']],
                             [[s['chord_id'] for s in r['chord_segments']] for r in expected])
            for action in ir['melody_plan']['unit_actions']:
                if 'H' in MODE_LAYERS[action['mode']]:
                    target=expected[action['unit']//2]['chord_id']
                    source=expected[action['source_unit']//2]['chord_id']
                    self.assertEqual(target,source)


if __name__ == '__main__':
    unittest.main()
