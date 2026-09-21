"""Optimized proposals must preserve candidate order and random consumption."""
import random
import unittest
from unittest.mock import patch

import harmony_annealing as ha
from annealing_config import resolve_annealing


def scalar_proposal(e,rng,search):
    group=rng.choice(e.groups)
    sizes=[s for s in range(1,len(group)+1) if str(s) in search['move_size_weights']]
    weights=[search['move_size_weights'][str(s)] for s in sizes]
    if not sum(weights):return None
    k=rng.choices(sizes,weights)[0]
    selected=rng.sample(group,k);pending=set(selected);moves={}
    local=rng.random()<search['local_probability']
    for i in selected:
        domain=e.local_domains[i].get(e.pitches[i],e.domains[i]) if local else e.domains[i]
        options=[]
        for p in domain:
            if any(j not in pending and moves.get(j,e.pitches[j]) not in
                   e.compatible[i,j][p] for j in e.neighbors[i]):continue
            limit=ha.sb.COUNTERPOINT_MAX_JUMP_DEGREES[e.voice[i]]
            if any(abs(e.pitch_degree[p]-e.pitch_degree[e.pitches[j]])>limit
                   for j in e.horizontal[i]):continue
            options.append(p)
        if not options:return None
        moves[i]=rng.choice(options);pending.remove(i)
    return {i:p for i,p in moves.items() if p!=e.pitches[i]} or None


class OptimizerPerformanceTests(unittest.TestCase):
    def test_bound_factors_follow_restored_pitch_list(self):
        from types import SimpleNamespace
        e=ha.Energy.__new__(ha.Energy)
        e.metric=SimpleNamespace(edo=72)
        e._vertical_value=lambda kind,xs,data:sum(xs)**2
        e._motion_value=lambda voice,xs:sum(abs(b-a) for a,b in zip(xs,xs[1:]))
        factors=[(kind,(0,),.3,{p:p*p for p in range(10)})
                 for kind in ('background','register','pitch_class_reward')]
        factors += [(kind,(0,1,2),.7,None)
                    for kind in ('sounding','attack','harmonic_realization')]
        factors += [('voice_leading',ids,.2,'bass') for ids in ((0,),(0,1,2))]
        factors += [('bass_balance',(0,1,2),12.,((1.,2.,1.),4.,.25))]
        bound=[e._compile_factor(f) for f in factors]
        for pitches in ([1,2,3],[3,3,2],[0,9,1]):
            e.pitches=list(pitches)
            self.assertEqual([f(e.pitches) for f in bound],
                             [e.evaluate(f) for f in factors])

    def test_bitmasks_match_scalar_including_rng_state(self):
        e=ha.Energy.__new__(ha.Energy)
        e.pitches=[0,2,4,1,3,5,8]
        e.voice=['bass','inner','counter']*2+['lead']
        e.domains={i:tuple(range(-3,9)) for i in range(6)}
        e.pitch_degree={p:p for p in range(-3,9)}
        e.groups=[(0,1,2),(3,4,5)]
        e.neighbors=[[j for j in range(7) if j!=i and
                      (j==6 or j//3==i//3)] for i in range(6)]+[list(range(6))]
        e.horizontal=[[i+3 if i<3 else i-3] for i in range(6)]+[[]]
        e.compatible={}
        for i in range(6):
            for j in e.neighbors[i]:
                other=e.domains.get(j,(e.pitches[j],))
                e.compatible[i,j]={p:frozenset(q for q in other if
                    (p<q if i%3<j%3 or j==6 else p>q) and abs(p-q)!=2)
                    for p in e.domains[i]}
        e.local_domains={i:{p:tuple(q for q in d if abs(q-p)<=3) for p in d}
                         for i,d in e.domains.items()}
        with patch.dict(ha.sb.COUNTERPOINT_MAX_JUMP_DEGREES,
                        {'bass':4,'inner':4,'counter':4}):
            e._build_proposal_masks()
            a=random.Random(42);b=random.Random(42)
            search=resolve_annealing()['search']
            for _ in range(2000):
                expected=scalar_proposal(e,a,search)
                actual=e.proposal(b,search)
                self.assertEqual(expected,actual)
                self.assertEqual(a.getstate(),b.getstate())
                for i,p in (actual or {}).items():e.pitches[i]=p


if __name__=='__main__':unittest.main()
