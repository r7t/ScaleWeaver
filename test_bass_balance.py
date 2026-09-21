import unittest
from types import SimpleNamespace
from annealing_config import resolve_annealing
from harmony_annealing import Energy, bass_distribution


class BassBalanceTests(unittest.TestCase):
    def energy(self, pitches, durations):
        e=Energy.__new__(Energy)
        e.metric=SimpleNamespace(edo=72)
        e.pitches=list(pitches)
        e.factors=[('bass_balance',tuple(range(len(pitches))),12.0,
                    (tuple(durations),sum(durations),.25))]
        e.incident=[{0} for _ in pitches]
        e.restore(e.pitches)
        return e

    def test_duration_not_attack_count_and_octave_invariance(self):
        a=self.energy([0,7,30,42],[7,1,1,1])
        b=self.energy([72,7,30,42],[7,1,1,1])
        c=self.energy([0,7,30,42],[1,1,1,1])
        self.assertAlmostEqual(a.total,12*.45**2)
        self.assertEqual(a.total,b.total)
        self.assertEqual(c.total,0)

    def test_trial_commit_restore_and_reporting(self):
        e=self.energy([0,72,7,30],[1,1,1,1])
        before=e.total
        trial=e.trial({1:42})
        self.assertEqual(e.pitches,[0,72,7,30])
        self.assertLess(trial[0],0)
        e.commit({1:42},trial)
        self.assertAlmostEqual(e.total,before+trial[0])
        incremental=e.total
        e.restore(e.pitches)
        self.assertAlmostEqual(e.total,incremental)
        self.assertAlmostEqual(sum(e.energy_sources().values()),e.total)

    def test_disabled_by_default_and_invalid_ceiling(self):
        self.assertEqual(resolve_annealing()['bass_balance']['weight'],0)
        for x in (0,1.1,float('nan')):
            with self.assertRaises(ValueError):
                resolve_annealing({'bass_balance':{'max_share':x}})

    def test_distribution_merges_octaves(self):
        s={'voices':{'bass':[{'step':p,'duration_beats':d}
                            for p,d in [(0,2),(72,1),(7,1)]]}}
        r=bass_distribution(s,72)
        self.assertEqual(r['duration_shares'],{'0':.75,'7':.25})
        self.assertEqual(r['attack_counts'],{'0':2,'7':1})


if __name__=='__main__':
    unittest.main()
