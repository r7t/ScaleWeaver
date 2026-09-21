"""Fixed-melody, fixed-rhythm whole-score simulated annealing.

No whole-score best-of, no Lead moves, no pitch-class diversity objective.
Optimizer sonority values use a configurable mean of the full raw value and
the worst raw subsets at every smaller supported cardinality.
"""
from __future__ import annotations
import bisect
import copy
import itertools
import math
import random
import time
from functools import lru_cache

import numpy as np
import harmony_rhythm as hr
import score_builder as sb
from annealing_config import resolve_annealing
from prime_salience import PrimeSalience
from harmonic_realization import musical_root, is_anchor, realization_cost
from scale_config import resolved_voice_pc_rewards
from subset_cse_cache import ensure_cache as ensure_subset_cse_cache

EPS=1e-8


@lru_cache(maxsize=None)
def _octave_fifth_four_patterns(octave):
    harmonic_steps=tuple(round(octave*math.log2(n)) for n in (1,2,3,4,6,8,12,16))
    return frozenset(tuple(x-row[0] for x in row)
                     for row in itertools.combinations(harmonic_steps,4))


@lru_cache(maxsize=500000)
def octave_fifth_four_subset(xs,octave):
    """Whether any four actual notes form the requested 3-limit harmonic subset."""
    if len(xs)!=5:return False
    patterns=_octave_fifth_four_patterns(octave)
    for row in itertools.combinations(sorted(xs),4):
        if tuple(x-row[0] for x in row) in patterns:return True
    return False


@lru_cache(maxsize=None)
def _octave_fifth_three_patterns(octave):
    """Direct 1:2:3 and 2:3:4 patterns in the active EDO."""
    return frozenset({
        (0,round(octave*math.log2(2)),round(octave*math.log2(3))),
        (0,round(octave*math.log2(3/2)),round(octave*math.log2(2))),
    })


@lru_cache(maxsize=500000)
def octave_fifth_three_subset_count(xs,octave):
    """Count adjacent triples that are direct 1:2:3 or 2:3:4 structures.

    The comparison uses absolute register distances after a common
    transposition.  Other 3-limit triples such as 3:4:8 therefore do not
    match.  After sorting the notes present in this attack/sounding factor,
    only consecutive three-note windows are inspected.  Intervening voices
    therefore hide a non-adjacent 1:2:3 or 2:3:4 subset.  Overlapping matching
    windows each contribute one virtual octave to the anti-collapse count.
    """
    if not 3<=len(xs)<=5:return 0
    patterns=_octave_fifth_three_patterns(octave)
    ordered=tuple(sorted(xs))
    return sum(tuple(x-row[0] for x in row) in patterns
               for row in zip(ordered,ordered[1:],ordered[2:]))


@lru_cache(maxsize=500000)
def octave_family_count(xs,octave):
    """Exact octave pairs plus virtual octaves from 1:2:3 / 2:3:4 triples."""
    # Within one octave-equivalence class, count only adjacent occupied
    # registers.  Thus 1:2:4 has the two edges 1--2 and 2--4, not the
    # redundant transitive pair 1--4.  This also handles gaps wider than the
    # old hard-coded three-octave range (for example a bare 1:16 pair).
    families={}
    for pitch in xs:
        families.setdefault(pitch%octave,set()).add(pitch)
    exact=sum(max(0,len(registers)-1) for registers in families.values())
    return exact+octave_fifth_three_subset_count(tuple(sorted(xs)),octave)


class PrimeReward:
    """Instance-local weights and canonical JI masks; one reward per prime.

    Snapshot the runtime's canonical index, including its wide-register
    dyad-union fallback, so a later scale/style load cannot
    change an existing optimizer's reward. Zero weights require no JI work.
    """
    def __init__(self,spec,scale=1.0):
        self.weights=tuple(scale*float(spec.style.get('prime_rewards',{}).get(str(p),0.0))
                           for p in sb.cse_rt.ATTACK_JI_PRIMES)
        self.enabled=any(self.weights);self.edo=spec.edo
        self.index=sb.cse_rt._load_attack_ji_dyad_index() if self.enabled else None
        self._reward=lru_cache(maxsize=100000)(self._reward)

    def _dyad_mask(self,diff):
        if not diff:return 0
        exact=self.index['absolute_dyads'].get(diff)
        if exact is not None:return exact['prime_mask']
        octaves,residue=divmod(diff,self.edo)
        return (1 if octaves else 0) | (self.index['dyads'][residue]['prime_mask'] if residue else 0)

    def _reward(self,offsets):
        mask=self.index['group_masks'].get(len(offsets),{}).get(','.join(map(str,offsets)))
        if mask is None:
            mask=0
            for a,b in itertools.combinations(offsets,2):mask|=self._dyad_mask(b-a)
        return sum(w for i,w in enumerate(self.weights) if mask & (1<<i))

    def __call__(self,xs):
        if not self.enabled or len(xs)<2:return 0.0
        xs=sorted(xs)
        return self._reward(tuple(p-xs[0] for p in xs))


class OptimizerMetric:
    def __init__(self,spec,config):
        self.spec=spec
        self.bundle=hr.CSE
        self.weights=tuple(float(spec.style.get('cse_weights',{}).get(k,d)) for k,d in
                           [('CSE_2D_A',1),('CSE_2D_B',0),('CSE_2D_C',0)])
        self.hierarchical_weights=config['worst_subset_cse_weights']
        self.rank_terms={}
        self.pitch_index={}
        self.max_cardinality=len(spec.resolved_ensemble_voices())
        if self.max_cardinality not in (3,4,5):
            raise ValueError('Harmony annealing supports exactly 3, 4 or 5 voices')
        # Rank metadata is tiny. The actual hierarchy values and ordered
        # distributions are read from a persistent, weight-specific mmap.
        for n in range(2,6):
            key='inner5' if n==5 else f'bass{n}'
            t=self.bundle.tables[key]
            steps=t['steps'];self.pitch_index[n]={int(p):i for i,p in enumerate(steps)}
            self.rank_terms[n]=np.array([[math.comb(a+i,i+1) for a in range(len(steps))]
                                          for i in range(n)],dtype=np.int64)
        self.subset_cache=ensure_subset_cse_cache(
            self.bundle,self.weights,self.hierarchical_weights)
        self.corrected_arrays=self.subset_cache.values
        ranges=spec.resolved_voice_ranges()
        active_voices=spec.resolved_ensemble_voices()
        self.five_nominal_bounds=(min(ranges[v][0] for v in active_voices),
                                  max(ranges[v][1] for v in active_voices))
        self.five_oob_base=float(np.max(self.corrected_arrays[5]))
        self.five_oob_cost_per_octave=5.0
        self.attack_config=config['attack']
        self.loss_polynomials=config['cse_loss_polynomials']
        self.edo=spec.edo
        self.pc_rewards={v:{int(pc):reward for pc,reward in rows.items()}
                         for v,rows in resolved_voice_pc_rewards(spec.style,spec.pcs).items()}
        self.prime_reward=PrimeReward(spec,config['prime_salience']['presence_scale'])
        self.prime_salience=PrimeSalience(spec.edo,config['prime_salience'])
        # Annealing revisits the same small set of sonorities many thousands
        # of times. Keep the entire rank -> derived-value -> loss chain hot.
        self._rank_cached=lru_cache(maxsize=500000)(self._rank_uncached)
        self._corrected_cached=lru_cache(maxsize=500000)(self._corrected_uncached)
        self._sounding_cost_cached=lru_cache(maxsize=500000)(self._sounding_cost_uncached)
        self._attack_cost_cached=lru_cache(maxsize=250000)(self._attack_cost_uncached)
        self.field=lru_cache(maxsize=100000)(self.field)

    def rank(self,xs):
        return self._rank_cached(tuple(sorted(map(int,xs))))

    def _rank_uncached(self,xs):
        n=len(xs)
        try:idx=[self.pitch_index[n][p] for p in xs]
        except KeyError as exc:raise ValueError(f'Optimizer pitch outside table domain: {xs}') from exc
        return sum(int(self.rank_terms[n][i,a]) for i,a in enumerate(idx))

    def raw(self,xs):
        if len(xs)<2:return 0.0
        n=len(xs);rank=self.rank(xs)
        key='inner5' if n==5 else f'bass{n}'
        offset=int(self.bundle.tables[key]['offset_bytes'])+16*rank
        value=0.0
        for metric,weight in zip(('cse','se','ccse'),self.weights):
            if weight:
                value+=float(weight)*float(np.frombuffer(
                    self.bundle._maps[metric],dtype='<f8',count=1,offset=offset)[0])
        return value

    def corrected(self,xs):
        if len(xs)<2:return 0.0
        return self._corrected_cached(tuple(sorted(map(int,xs))))

    def _corrected_uncached(self,xs):
        if len(xs)==5 and any(p not in self.pitch_index[5] for p in xs):
            lo,hi=self.five_nominal_bounds
            excess_steps=sum(max(0,lo-p)+max(0,p-hi) for p in xs)
            return self.five_oob_base+self.five_oob_cost_per_octave*excess_steps/self.edo
        return float(self.corrected_arrays[len(xs)][self._rank_cached(xs)])

    def cse_loss(self,context,value):
        """Apply a configurable quartic to the hierarchical sonority value.

        These coefficients are independent of the spectral A/B/C blend.
        Raw SE/CSE/CCSE reporting remains unchanged.
        """
        p=self.loss_polynomials[context]
        return ((((p['a4']*value+p['a3'])*value+p['a2'])*value
                 +p['a1'])*value+p['a0'])

    def sounding_cost(self,xs):
        if len(xs)<2:return 0.0
        return self._sounding_cost_cached(tuple(sorted(map(int,xs))))

    def _sounding_cost_uncached(self,xs):
        return (self.cse_loss('sounding',self._corrected_cached(xs))-self.prime_reward(xs)
                -self.prime_salience.reward(xs,'sounding'))

    def pitch_class_reward(self,voice,pitch):
        return self.pc_rewards[voice].get(int(pitch)%self.edo,0.0)

    def attack_cost(self,xs):
        n=len(xs)
        if n<2:return 0.0
        return self._attack_cost_cached(tuple(sorted(map(int,xs))))

    def _attack_cost_uncached(self,xs):
        n=len(xs)
        weight=self.attack_config['cardinality_weights'].get(str(n),0.0)
        if not weight:return 0.0
        context={2:'two_note_attack',3:'three_note_attack',4:'four_note_attack',5:'five_note_attack'}[n]
        return weight*(self.cse_loss(context,self.corrected(xs))-self.prime_reward(xs)
                       -self.prime_salience.reward(xs,'attack'))

    def field(self,chord_id,pitch):
        background=tuple(hr.chord_reference_steps(hr.CHORDS[chord_id]))
        # A tetrad background plus the actual candidate is a true five-note
        # field.  It therefore uses inner5 and the pentad correction, rather
        # than averaging four-note projections.
        return self.corrected(background+(pitch,))


def motion_cost(voice,previous,current,recent,config):
    c=config['voice_leading'];cost=0.0
    if previous is not None:
        jump=abs(hr.degree(current)-hr.degree(previous))
        cost+=c['motion_weight']*sb._lead_interval_cost(jump)
        cost+=c['large_leap_cost']*max(0,jump-sb.LEAD_PREFERRED_MAX_LEAP)
        if current==previous:cost+=c['repeat_cost']
    cost+=c['ngram_weights'][voice]*sb._lead_ngram_prior_cost(recent,current)
    if voice=='bass':cost+=c['bass_ngram_weight']*sb._bass_ngram_prior_cost(recent,current)
    return cost


class GreedyInitializer:
    def __init__(self,metric,config):
        self.metric=metric;self.config=config
        self.relaxed_jump_choices=0
        self.expanded_range_choices=0

    def choose_pitch(self,voice,prev,prev2,recent,chord,existing,start,dur,guide):
        c=self.config;w=c['objective_weights'];scored=[];relaxed=[];expanded=[]
        lo,hi=sb.COUNTERPOINT_EFFECTIVE_RANGES[voice]
        lower_octaves=c['initialization']['atomic_rescue_lower_octaves']
        expanded_lo=max(min(self.metric.pitch_index[self.metric.max_cardinality]),hr.RANGES[voice][0]-lower_octaves*hr.OCT)
        candidate_pool=sorted(set(hr.POOLS[voice]) |
                              set(hr.SCALE.make_pool(expanded_lo,hr.RANGES[voice][1])))
        end=start+dur
        ovs=[e for events in existing.values() for e in sb.overlaps(events,start,end)]
        points=sorted({start,end,*[max(start,e['start_beat']) for e in ovs],
                       *[min(end,e['start_beat']+e['duration_beats']) for e in ovs]})
        contexts=[]
        for a,b in zip(points,points[1:]):
            mid=(a+b)/2
            others=[e['step'] for e in ovs if e['start_beat']<=mid<e['start_beat']+e['duration_beats']]
            if others:contexts.append((b-a,others))
        attack=[e['step'] for e in ovs if abs(e['start_beat']-start)<EPS]
        for p in candidate_pool:
            nominal=lo<=p<=hi and p in hr.POOLS[voice]
            if math.isinf(sb._counterpoint_vertical_cost(voice,p,existing,start,dur)):continue
            jump_excess=0
            if prev is not None:
                jump_excess=max(0,abs(hr.degree(p)-hr.degree(prev))-
                                sb.COUNTERPOINT_MAX_JUMP_DEGREES[voice])
            sounding=sum(span*self.metric.sounding_cost(tuple(others)+(p,)) for span,others in contexts)/max(dur,EPS)
            value=w['sounding']*sounding
            if attack:value+=w['attack']*self.metric.attack_cost(tuple(attack)+(p,))
            value+=w['background']*c['background_voice_weights'][voice]*self.metric.field(chord.id,p)
            value+=w['voice_leading']*(motion_cost(voice,prev,p,recent,c)
                    +c['voice_leading']['register_weights'][voice]*abs(hr.degree(p)-guide))
            value-=self.metric.pitch_class_reward(voice,p)
            row=(value+c['initialization']['relaxed_jump_extra_cost']*jump_excess,p)
            if not nominal:
                step_distance=max(0,lo-int(p))/max(1,hr.OCT)
                expanded.append((row[0]+c['initialization']['expanded_range_extra_cost']*(1+step_distance),p))
            elif jump_excess:
                relaxed.append(row)
            else:
                scored.append(row)
        if scored:return min(scored)[1]
        if relaxed and c['initialization']['relax_melodic_jumps']:
            self.relaxed_jump_choices+=1
            return min(relaxed)[1]
        if expanded:
            self.expanded_range_choices+=1
            return min(expanded)[1]
        raise sb.GenerationRejected(f'greedy initializer: no vertically legal {voice} pitch at {start}')


class Energy:
    """Sparse factors; each proposal updates only its incident factors."""
    def __init__(self,score,metric,config):
        self.score=score;self.metric=metric;self.config=config
        self.voices=tuple(score.get('voice_layout',{}).get('voice_order') or metric.spec.resolved_ensemble_voices())
        self.rank={v:i for i,v in enumerate(self.voices)}
        self.accompaniment=self.voices[:-1]
        self.events=[];self.voice=[];self.by_voice={};self.pitches=[]
        for v in self.voices:
            ids=[]
            for e in score['voices'][v]:
                ids.append(len(self.events));self.events.append(e);self.voice.append(v);self.pitches.append(int(e['step']))
            self.by_voice[v]=ids
        self.mutable=[i for i,v in enumerate(self.voice) if v!='lead']
        self.fixed=set(self.by_voice['lead'])
        self.starts=[e['start_beat'] for e in self.events]
        self.ends=[e['start_beat']+e['duration_beats'] for e in self.events]
        self.neighbors=[[] for _ in self.events];self.horizontal=[[] for _ in self.events]
        self._interval_legal_cache={}
        self.factors=[];self.incident=[set() for _ in self.events]
        self.onsets={}
        for i,t in enumerate(self.starts):self.onsets.setdefault(round(t,6),[]).append(i)
        self.groups=[tuple(i for i in ids if i not in self.fixed) for ids in self.onsets.values()]
        self.groups=sorted((g for g in self.groups if g),key=lambda g:self.starts[g[0]])
        # Voices are monophonic and time-sorted.  Find cross-voice overlaps by
        # binary-searching the other voice's start times instead of testing all
        # O(event_count^2) pairs.
        voice_starts={v:[self.starts[i] for i in ids] for v,ids in self.by_voice.items()}
        for va,vb in itertools.combinations(self.voices,2):
            b_ids=self.by_voice[vb];b_starts=voice_starts[vb]
            for i in self.by_voice[va]:
                stop=bisect.bisect_left(b_starts,self.ends[i]-EPS)
                for j in b_ids[:stop]:
                    if self.ends[j]>self.starts[i]+EPS:
                        self.neighbors[i].append(j);self.neighbors[j].append(i)
        for v,ids in self.by_voice.items():
            for a,b in zip(ids,ids[1:]):self.horizontal[a].append(b);self.horizontal[b].append(a)
        self.domains={}
        for i in self.mutable:
            v=self.voice[i];lo,hi=sb.COUNTERPOINT_EFFECTIVE_RANGES[v]
            domain={p for p in hr.POOLS[v] if lo<=p<=hi and all(
                self.pair_legal(i,p,j,self.pitches[j]) for j in self.neighbors[i] if j in self.fixed)}
            # A rescue initializer may place a rare note below the nominal
            # range to remain under a fixed low Lead.  Keep that legal starting
            # value in its own domain so energy construction and subsequent
            # moves are defined; all ordinary proposal targets remain present.
            initial=self.pitches[i]
            if all(self.pair_legal(i,initial,j,self.pitches[j])
                   for j in self.neighbors[i] if j in self.fixed):
                domain.add(initial)
            self.domains[i]=tuple(sorted(domain))
        all_pitches=set(self.pitches)
        for domain in self.domains.values():all_pitches.update(domain)
        self.pitch_degree={p:hr.degree(p) for p in all_pitches}
        radius=int(config['search']['local_radius_degrees'])
        self.local_domains={
            i:{current:tuple(p for p in domain
                             if abs(self.pitch_degree[p]-self.pitch_degree[current])<=radius)
               for current in domain}
            for i,domain in self.domains.items()
        }
        # Directed compatibility tables turn the inner proposal loop's wolf,
        # second and voice-order functions into one hash-set membership test.
        self.compatible={}
        for i in self.mutable:
            for j in self.neighbors[i]:
                other=self.domains[j] if j in self.domains else (self.pitches[j],)
                self.compatible[(i,j)]={
                    p:frozenset(q for q in other if self.pair_legal(i,p,j,q))
                    for p in self.domains[i]
                }
        self._motion_value=lru_cache(maxsize=500000)(self._motion_value_uncached)
        self._build_factors()
        self.values=[self.evaluate(f) for f in self.factors]
        self.total=math.fsum(self.values)

    def add(self,kind,ids,scale,data=None):
        f=len(self.factors);self.factors.append((kind,tuple(ids),float(scale),data))
        for i in ids:
            if i not in self.fixed:self.incident[i].add(f)

    def _build_factors(self):
        w=self.config['objective_weights'];duration=self.score['bars']*self.score['beats_per_bar']
        balance = self.config['bass_balance']
        bass_ids = self.by_voice.get('bass', [])
        if balance['weight'] and bass_ids:
            durations = tuple(self.events[i]['duration_beats'] for i in bass_ids)
            self.add('bass_balance', bass_ids, balance['weight'],
                     (durations, sum(durations), balance['max_share']))
        points=sorted(set(self.starts+self.ends))
        self.slices=[]
        starts_at={};ends_at={}
        for i,(start,end) in enumerate(zip(self.starts,self.ends)):
            starts_at.setdefault(start,[]).append(i);ends_at.setdefault(end,[]).append(i)
        active=set()
        for a,b in zip(points,points[1:]):
            if b-a<EPS:continue
            for i in ends_at.get(a,()):active.discard(i)
            active.update(starts_at.get(a,()))
            ids=tuple(sorted(active))
            if len(ids)>=2:
                self.slices.append((a,b,ids));self.add('sounding',ids,w['sounding']*(b-a)/duration)
        attacks=[ids for ids in self.onsets.values() if 2<=len(ids)<=self.metric.max_cardinality]
        for ids in attacks:self.add('attack',ids,w['attack']/max(1,len(attacks)))
        realization = self.config['harmonic_realization']
        if realization['enabled'] and realization['weight']:
            # Intersect sounding slices with harmonic boundaries: a sustained
            # note must support BOTH harmonies if the plan changes underneath.
            cursor = 0.0
            for bar_index, bar in enumerate(self.score['harmony_plan']):
                bpb = float(bar.get('beats_per_bar',self.score['beats_per_bar']))
                for segment in bar['chord_segments']:
                    start = cursor + segment['offset']
                    end = start + segment['duration']
                    anchor = is_anchor(hr,segment)
                    multiplier = realization['anchor_multiplier'] if anchor else 1.0
                    cadence = (int(bar.get('bar_in_phrase',bar_index % 8)) ==
                               int(bar.get('phrase_bars',8))-1)
                    if anchor and cadence: multiplier *= realization['cadence_multiplier']
                    if anchor and bar_index == len(self.score['harmony_plan'])-1:
                        multiplier = max(multiplier,realization['final_multiplier'])
                    for a,b,ids in self.slices:
                        span = min(b,end)-max(a,start)
                        if span <= EPS: continue
                        data = (tuple(self.voice[i] for i in ids),
                                tuple(hr.CHORDS[segment['chord_id']].pcs),
                                musical_root(hr,segment))
                        self.add('harmonic_realization',ids,
                                 realization['weight']*multiplier*span/duration,data)
                cursor += bpb
        transitions=sum(max(0,len(self.by_voice[v])-1) for v in self.accompaniment)
        mutable_count=max(1,len(self.accompaniment))
        for v in self.accompaniment:
            ids=self.by_voice[v]
            for pos,i in enumerate(ids):
                e=self.events[i];start=self.starts[i];end=self.ends[i];bpb=self.score['beats_per_bar']
                # Evaluate the complete background over the note's duration,
                # including any harmony-segment changes inside it.
                field_rows=[]
                for bar in range(int(start//bpb),min(len(self.score['harmony_plan']),int((end-EPS)//bpb)+1)):
                    for seg in self.score['harmony_plan'][bar]['chord_segments']:
                        a=max(start,bar*bpb+seg['offset']);b=min(end,bar*bpb+seg['offset']+seg['duration'])
                        if b>a:field_rows.append((seg['chord_id'],b-a))
                table={p:sum(span*self.metric.field(cid,p) for cid,span in field_rows)
                       for p in self.domains[i]}
                self.add('background',(i,),w['background']*self.config['background_voice_weights'][v]/(mutable_count*duration),table)
                hist=ids[max(0,pos-4):pos+1]
                self.add('voice_leading',hist,w['voice_leading']/max(1,transitions),v)
                reg={p:self.config['voice_leading']['register_weights'][v]
                     *abs(hr.degree(p)-sb.COUNTERPOINT_ROLE_CENTRE_DEGREE[v]) for p in self.domains[i]}
                self.add('register',(i,),w['voice_leading']*e['duration_beats']/(mutable_count*duration),reg)
                if any(self.metric.pc_rewards[v].values()):
                    # Each voice contributes its mean event reward, independently
                    # of event count/duration and of the four existing weights.
                    rewards={p:-self.metric.pitch_class_reward(v,p) for p in self.domains[i]}
                    self.add('pitch_class_reward',(i,),1.0/len(ids),rewards)

    def evaluate(self,factor):
        kind,ids,scale,data=factor;xs=tuple(self.pitches[i] for i in ids)
        if kind=='sounding':
            allowance=int(self.config['sounding_octave_family_pair_allowance'].get(str(len(xs)),1))
            value=(self.metric.sounding_cost(xs)
                   + self.config['sounding_octave_excess_cost']
                   * max(0,octave_family_count(tuple(sorted(xs)),self.metric.edo)-allowance)
                   + self.config['sounding_octave_fifth_subset_cost']
                   * octave_fifth_four_subset(tuple(sorted(xs)),self.metric.edo))
        elif kind=='attack':
            value=(self.metric.attack_cost(xs)
                   + (self.config['sounding_octave_excess_cost']
                      * max(0,octave_family_count(tuple(sorted(xs)),self.metric.edo)
                            - int(self.config['sounding_octave_family_pair_allowance']
                                  .get(str(len(xs)),1)))
                      if 3<=len(xs)<=5 else 0.0)
                   + self.config['sounding_octave_fifth_subset_cost']
                   * octave_fifth_four_subset(tuple(sorted(xs)),self.metric.edo))
        elif kind=='bass_balance':
            durations,total,ceiling=data
            shares={}
            for pitch,duration in zip(xs,durations):
                pc=pitch % self.metric.edo
                shares[pc]=shares.get(pc,0.0)+duration/total
            value=sum(max(0.0,share-ceiling)**2 for share in shares.values())
        elif kind=='harmonic_realization':
            voices,pcs,root=data
            value=realization_cost(xs,voices,pcs,root,self.metric.edo,
                                   self.config['harmonic_realization'])
        elif kind in ('background','register','pitch_class_reward'):value=data[xs[0]]
        else:value=self._motion_value(data,xs)
        return scale*value

    def _motion_value_uncached(self,voice,xs):
        return motion_cost(voice,xs[-2] if len(xs)>1 else None,xs[-1],xs[:-1],self.config)

    def pair_legal(self,i,p,j,q):
        if self.rank[self.voice[i]]<self.rank[self.voice[j]]:
            if p>=q:return False
            lower,upper=p,q
        elif p<=q:return False
        else:lower,upper=q,p
        key=(int(lower),int(upper))
        result=self._interval_legal_cache.get(key)
        if result is None:
            result=not hr.hard_wolf(lower,upper) and not sb.step_second(lower,upper)
            self._interval_legal_cache[key]=result
        return result

    def legal(self,moves):
        for i,p in moves.items():
            if i in self.fixed or p not in self.domains[i]:return False
            for j in self.neighbors[i]:
                if moves.get(j,self.pitches[j]) not in self.compatible[(i,j)][p]:return False
            for j in self.horizontal[i]:
                if abs(self.pitch_degree[p]-self.pitch_degree[moves.get(j,self.pitches[j])])>sb.COUNTERPOINT_MAX_JUMP_DEGREES[self.voice[i]]:return False
        return True

    def proposal(self,rng,search):
        group=rng.choice(self.groups)
        sizes=[s for s in range(1,len(group)+1) if str(s) in search['move_size_weights']]
        weights=[search['move_size_weights'][str(s)] for s in sizes]
        if not sum(weights):return None
        k=rng.choices(sizes,weights)[0]
        selected=rng.sample(group,k);pending=set(selected);moves={}
        local=rng.random()<search['local_probability']
        for i in selected:
            options=[]
            candidate_domain=(self.local_domains[i].get(self.pitches[i],self.domains[i])
                              if local else self.domains[i])
            for p in candidate_domain:
                invalid=False
                for j in self.neighbors[i]:
                    if (j not in pending and
                            moves.get(j,self.pitches[j]) not in self.compatible[(i,j)][p]):
                        invalid=True;break
                if invalid:continue
                max_jump=sb.COUNTERPOINT_MAX_JUMP_DEGREES[self.voice[i]]
                for j in self.horizontal[i]:
                    if abs(self.pitch_degree[p]-self.pitch_degree[self.pitches[j]])>max_jump:
                        invalid=True;break
                if invalid:continue
                options.append(p)
            if not options:return None
            moves[i]=rng.choice(options);pending.remove(i)
        moves={i:p for i,p in moves.items() if p!=self.pitches[i]}
        # Each selected vertical pair is checked when the later member leaves
        # pending; simultaneous groups cannot contain two events of one
        # monophonic voice, so horizontal checks need no second pass.
        return moves or None

    def trial(self,moves):
        affected=sorted(set().union(*(self.incident[i] for i in moves)))
        old={i:self.pitches[i] for i in moves}
        for i,p in moves.items():self.pitches[i]=p
        values=[self.evaluate(self.factors[f]) for f in affected]
        for i,p in old.items():self.pitches[i]=p
        delta=math.fsum(v-self.values[f] for f,v in zip(affected,values))
        return delta,affected,values

    def commit(self,moves,trial):
        delta,affected,values=trial
        for i,p in moves.items():self.pitches[i]=p
        for f,v in zip(affected,values):self.values[f]=v
        self.total+=delta

    def restore(self,pitches):
        self.pitches=list(pitches);self.values=[self.evaluate(f) for f in self.factors];self.total=math.fsum(self.values)

    def components(self):
        out=dict.fromkeys(('attack','sounding','background','voice_leading','pitch_class_reward','harmonic_realization','bass_balance'),0.0)
        for f,v in zip(self.factors,self.values):out['voice_leading' if f[0]=='register' else f[0]]+=v
        return out

    def energy_sources(self):
        """Four exhaustive reporting buckets; does not affect optimization."""
        parts=self.components()
        return {
            'harmonic_realization_energy':parts['harmonic_realization'],
            'background_chord_field_energy':parts['background'],
            'attack_energy':parts['attack'],
            'sounding_energy':parts['sounding'],
            # Register and voice-specific pitch-class terms describe how each
            # individual line is written, so they belong to this fourth bucket.
            'voice_leading_energy':parts['voice_leading']+parts['pitch_class_reward']+parts['bass_balance'],
        }

    def export(self):
        for i in self.mutable:
            e=self.events[i];p=self.pitches[i]
            e.update(step=p,name=hr.pitch_name(p),freq=hr.freq(p))


def bass_distribution(score, edo):
    """Attack and duration shares are different for a sustained bass line."""
    events=score['voices'].get('bass', [])
    counts={};durations={}
    for event in events:
        pc=str(int(event['step']) % edo)
        counts[pc]=counts.get(pc,0)+1
        durations[pc]=durations.get(pc,0.0)+event['duration_beats']
    total=sum(durations.values())
    shares={pc:d/total for pc,d in durations.items()} if total else {}
    return dict(attack_counts=counts, duration_beats=durations,
                duration_shares=shares, max_duration_share=max(shares.values(),default=0.0),
                effective_pitch_classes=1/sum(s*s for s in shares.values()) if shares else 0.0)


def optimize(score,metric,config):
    started=time.perf_counter();energy=Energy(score,metric,config)
    initial_bass=bass_distribution(score,metric.edo)
    initial=energy.total;initial_components=energy.components();best=initial;best_pitches=list(energy.pitches)
    initial_sources=energy.energy_sources()
    initial_salience=metric.prime_salience.diagnostics(score) if metric.prime_salience.enabled else None
    initial_pitches=list(energy.pitches);fixed_events=copy.deepcopy(score['voices']['lead'])
    rhythm={v:[(e['start_beat'],e['duration_beats']) for e in xs] for v,xs in score['voices'].items()}
    rng=random.Random(int(score['seed'])^0xA66EA11);search=config['search'];steps=search['steps']
    accepted=proposed=uphill=0;counts={1:0,2:0,3:0};history=[]
    for step in range(steps):
        temp=search['start_temperature']*(search['end_temperature']/search['start_temperature'])**(step/max(1,steps-1))
        moves=energy.proposal(rng,search)
        if moves:
            proposed+=1;counts[len(moves)]+=1;trial=energy.trial(moves);delta=trial[0]
            if delta<=0 or rng.random()<math.exp(-delta/temp):
                energy.commit(moves,trial);accepted+=1;uphill+=int(delta>0)
                if energy.total<best-1e-12:best=energy.total;best_pitches=list(energy.pitches)
        if search['progress_every'] and (step+1)%search['progress_every']==0:
            row={'step':step+1,'best':best,'current':energy.total,'temperature':temp};history.append(row)
            print(f'Annealing {step+1}/{steps}: best={best:.6f} current={energy.total:.6f}',flush=True)
    energy.restore(best_pitches)
    # Deterministic downhill finish on the best visited state.
    quenched=0
    for _ in range(search['quench_sweeps']):
        order=list(energy.mutable);rng.shuffle(order);changed=0
        for i in order:
            winner=None;improvement=-1e-12
            for p in energy.domains[i]:
                if p==energy.pitches[i] or not energy.legal({i:p}):continue
                trial=energy.trial({i:p})
                if trial[0]<improvement:improvement=trial[0];winner=(p,trial)
            if winner:energy.commit({i:winner[0]},winner[1]);changed+=1
        quenched+=changed
        if not changed:break
    incremental=energy.total;energy.restore(energy.pitches)
    if not math.isclose(incremental,energy.total,rel_tol=1e-10,abs_tol=1e-9):raise AssertionError('Incremental energy drift')
    if energy.total>initial+1e-9:raise AssertionError('Annealing failed to preserve best state')
    energy.export()
    if score['voices']['lead']!=fixed_events:raise AssertionError('Annealing modified melody')
    if rhythm!={v:[(e['start_beat'],e['duration_beats']) for e in xs] for v,xs in score['voices'].items()}:raise AssertionError('Annealing changed rhythm')
    final_sources=energy.energy_sources()
    salience_report={'initial':initial_salience,
                     'final':metric.prime_salience.diagnostics(score)} if metric.prime_salience.enabled else None
    if salience_report:
        for context in ('sounding_duration_weighted','attack_equal_onset'):
            print('Prime salience '+context+': '+str(salience_report['initial'][context])
                  +' -> '+str(salience_report['final'][context]),flush=True)
    energy_sources={
        'initial':initial_sources,
        'final':final_sources,
        'delta':{key:final_sources[key]-initial_sources[key] for key in initial_sources},
        'initial_sum':math.fsum(initial_sources.values()),
        'final_sum':math.fsum(final_sources.values()),
    }
    return {'enabled':True,'method':'whole_score_onset_group_annealing','config':copy.deepcopy(config),
            'bass_distribution':{'initial':initial_bass,'final':bass_distribution(score,metric.edo)},
            'subset_cse_cache_manifest':str(metric.subset_cache.manifest_path),
            'initial_energy':initial,'final_energy':energy.total,'initial_components':initial_components,
            'final_components':energy.components(),'energy_sources':energy_sources,
            'prime_salience':salience_report,
            'steps':steps,'proposed_moves':proposed,
            'accepted_moves':accepted,'accepted_uphill_moves':uphill,'move_sizes':counts,
            'quench_moves':quenched,'changed_notes':sum(a!=b for a,b in zip(initial_pitches,energy.pitches)),
            'lead_unchanged':True,'rhythm_unchanged':True,'history':history,
            'elapsed_seconds':time.perf_counter()-started}


def final_metrics(score,metric):
    """Only unadjusted SE/CSE/CCSE/blend values: duration vs onset means."""
    voices=score['voices'];events=[e for xs in voices.values() for e in xs]
    def values(steps):
        key='inner5' if len(steps)==5 else f'bass{len(steps)}'
        row=metric.bundle.metrics_entry(tuple(steps),table_key=key)
        if row is None:raise ValueError(f'Final metrics missing sonority: {steps}')
        d={m:float(row[m][0]) for m in ('cse','ccse','se')}
        d['blending']=sum(w*d[m] for w,m in zip(metric.weights,('cse','se','ccse')))
        return d
    points=sorted({t for e in events for t in (e['start_beat'],e['start_beat']+e['duration_beats'])})
    rows=[]
    for a,b in zip(points,points[1:]):
        mid=(a+b)/2;active=[e['step'] for e in events if e['start_beat']<=mid<e['start_beat']+e['duration_beats']]
        if len(active)==metric.max_cardinality:rows.append((b-a,values(active)))
    def summarize(rows,kind):
        weight=sum(w for w,_ in rows)
        return {'aggregation':kind,'count':len(rows),'weight':weight,
                **{m:sum(w*d[m] for w,d in rows)/weight if weight else None for m in ('cse','ccse','se','blending')}}
    sounding_key=f'{metric.max_cardinality}_note_sounding'
    out={sounding_key:summarize(rows,'duration_weighted_mean'),
         'definition':'baseline-corrected spectral entropies, without subset subtraction',
         'blending_weights':dict(zip(('a','b','c'),metric.weights))}
    if metric.max_cardinality==4:
        out['four_note_sounding']=out[sounding_key]
    groups={}
    for e in events:groups.setdefault(round(e['start_beat'],6),[]).append(e['step'])
    for n in range(2,metric.max_cardinality+1):
        rows=[(1,values(xs)) for xs in groups.values() if len(xs)==n]
        out[f'{n}_note_simultaneous_attack']=summarize(rows,'equal_onset_mean_exact_cardinality')
    return out
