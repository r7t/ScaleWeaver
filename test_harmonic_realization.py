import unittest
from types import SimpleNamespace
from unittest.mock import patch
from annealing_config import resolve_annealing
from harmonic_realization import realization_cost, musical_root
from harmony_annealing import Energy
from joint_frontend_generate import _cadence_pitches


class RealizationTests(unittest.TestCase):
    def setUp(self):
        self.c = resolve_annealing()['harmonic_realization']
        self.voices = ('bass','inner','counter','lead')

    def cost(self, xs):
        return realization_cost(xs,self.voices,(0,4,7),0,12,self.c)

    def test_accompaniment_owns_chord_even_with_nonchord_lead(self):
        self.assertEqual(self.cost((0,4,7,14)),0)
        self.assertGreater(self.cost((0,7,12,16)),0)
        self.assertGreater(self.cost((5,9,12,16)),self.cost((0,4,7,14)))

    def test_joint_coverage_rewards_filling_lead_complement(self):
        c = dict(self.c,accompaniment_coverage=0,accompaniment_membership=0,bass_root=0)
        cost=lambda xs:realization_cost(xs,self.voices,(0,4,7),0,12,c)
        self.assertLess(cost((0,7,12,16)),cost((0,12,24,16)))

    def test_root_not_acoustic_foot(self):
        hr=SimpleNamespace(OCT=12,PCS=(0,2,4,5,7,9,11),NATURAL_SCALE_MODE=True,
                           SCALE=SimpleNamespace(metadata={}),
                           CHORDS={'x':SimpleNamespace(id='x',pcs=(0,4,7),foot=7)})
        self.assertEqual(musical_root(hr,{'chord_id':'x'}),0)
        self.assertEqual(musical_root(hr,{'chord_id':'x','root_pc':7}),7)

    def test_tetrad_capacity(self):
        self.assertEqual(realization_cost((0,4,7,10),self.voices,(0,4,7,10),0,12,self.c),0)

    def test_factor_evaluation_and_reporting(self):
        e=Energy.__new__(Energy);e.config=resolve_annealing();e.metric=SimpleNamespace(edo=12)
        e.pitches=[0,4,7,14]
        f=('harmonic_realization',(0,1,2,3),2,(self.voices,(0,4,7),0))
        e.factors=[f];e.values=[e.evaluate(f)]
        self.assertEqual(e.energy_sources()['harmonic_realization_energy'],0)
        e.pitches[0]=5
        self.assertGreater(e.evaluate(f),0)
        e.restore(e.pitches)
        e.incident=[{0},{0},{0},set()]
        before=e.total
        trial=e.trial({0:0})
        self.assertLess(trial[0],0)
        self.assertEqual(e.pitches[0],5)
        e.commit({0:0},trial)
        self.assertAlmostEqual(e.total,before+trial[0])
        incremental=e.total
        e.restore(e.pitches)
        self.assertAlmostEqual(incremental,e.total)

    def test_cadence_only_and_legal_tonic(self):
        import joint_frontend_generate as jf
        spec=SimpleNamespace(style={},resolved_harmony=lambda:{'pitch_roles':{'tonic':[0]}})
        with patch.object(jf.hr,'SCALE',spec),patch.object(jf.hr,'OCT',12), \
             patch.object(jf.sb,'POOLS',{'lead':list(range(12,25))}), \
             patch.object(jf.sb,'lead_jump_ok',lambda a,b:a is None or abs(a-b)<=7):
            self.assertEqual(_cadence_pitches({'prev':16},[19,21],14),[19,21])
            result=_cadence_pitches({'prev':16},[19,21],15)
            self.assertEqual(result[-1]%12,0)
            self.assertLessEqual(abs(result[-1]-result[-2]),7)

    def test_harmony_boundary_splits_sustained_factor(self):
        import harmony_annealing as ha
        e=Energy.__new__(Energy)
        e.config=resolve_annealing();e.metric=SimpleNamespace(edo=12,max_cardinality=4)
        e.starts=[0]*4;e.ends=[4]*4;e.voice=list(self.voices)
        e.fixed={3};e.incident=[set() for _ in range(4)];e.factors=[]
        e.onsets={0:[0,1,2,3]};e.accompaniment=[];e.by_voice={}
        e.score={'bars':1,'beats_per_bar':4,'harmony_plan':[
            {'chord_segments':[{'chord_id':'a','offset':0,'duration':2},
                               {'chord_id':'b','offset':2,'duration':2}]}]}
        hr=SimpleNamespace(OCT=12,PCS=(0,2,4,5,7,9,11),NATURAL_SCALE_MODE=True,
            SCALE=SimpleNamespace(metadata={}),ANCHOR_CHORD_IDS=['a'],
            CHORDS={k:SimpleNamespace(id=k,pcs=pcs,foot=foot)
                    for k,pcs,foot in [('a',(0,4,7),7),('b',(5,9,0),0)]})
        with patch.object(ha,'hr',hr):e._build_factors()
        factors=[f for f in e.factors if f[0]=='harmonic_realization']
        self.assertEqual(len(factors),2)
        self.assertEqual([f[3][2] for f in factors],[0,5])
        self.assertEqual([f[2] for f in factors],[5.0,1.0])
        for index,f in enumerate(e.factors):
            if f[0]=='harmonic_realization':self.assertIn(index,e.incident[0])
        self.assertEqual(e.incident[3],set())

if __name__=='__main__':unittest.main()
