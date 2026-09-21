#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified exact-EDO harmony/rhythm engine.

The rhythm grammar and random-call order are inherited from the 2026-08-30
TianGan generator. Scale-specific pitch data live in ScaleSpec. The canonical
72-EDO TianGan vocabulary retains its frozen 32 chords. Seven-note scales with
an equal-fifth meantone chain and at least five canonical-JI major/minor triads
use a separate degree progression route with ii-V-I cadences.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import combinations
import math, random, sys
from pathlib import Path
from typing import Iterable

from scale_config import ScaleSpec
import copy
from spectral_bundle import SpectralBundle

# Import-safe defaults; configure_scale() binds the active definition and rules.
SCALE: ScaleSpec | None = None
CSE: SpectralBundle | None = None
OCT=0
PCS=()
NAMES=()
PC_NAME=dict(zip(PCS,NAMES))
BASE_FREQ=0.0
RANGES={}
POOLS={}
VELOCITY={'bass':0.6,'inner':0.6,'inner2':0.6,'counter':0.6,'lead':0.6}
HARD_WOLF=-1
SECOND_MAX_STEP=0
_CSE_STRENGTH=1.15
_CSE_HARMONY_GAIN=7.0
CSE_CHORD_BLEND=.68
CSE_CHORD_SOFTMAX_BASE=5.5
FORM_TEMPLATE=('A','A','B','B','A','B')

@dataclass(frozen=True)
class Chord:
    id:str
    name:str
    pcs:tuple[int,...]
    ratio:str
    foot:int
    stability:float
    integers:tuple[int,...]
    has3:bool
    has5:bool
    has7:bool
    chord_tone_7_colour_mean:float
    max_integer:int
    function_code:str='C'
    @property
    def function(self): return self.function_code
    @property
    def melodic_7_colour_strength(self): return float(self.chord_tone_7_colour_mean)
    @property
    def prime_support(self): return tuple(p for p,yes in ((3,self.has3),(5,self.has5),(7,self.has7)) if yes)
    @property
    def prime_class(self): return f"3{'+' if self.has3 else '-'}5{'+' if self.has5 else '-'}7{'+' if self.has7 else '-'}"

_CANONICAL_TIANGAN_CHORD_ROWS=(
('P3_024', '戊甲丙', (30, 0, 16), '4:6:7', 30, 0.5534591498116744, (4, 6, 7), True, False, True, 1.3333333333333333, 7, 'S'),
('P3_026', '甲丙庚', (0, 16, 42), '6:7:9', 0, 0.5300587012617293, (6, 7, 9), True, False, True, 1.6666666666666667, 9, 'T'),
('P3_028', '甲丙壬', (0, 16, 58), '12:14:21', 0, 0.4639273663636983, (12, 14, 21), True, False, True, 2.6666666666666665, 21, 'T'),
('P3_036', '庚甲丁', (42, 0, 23), '3:4:5', 42, 0.6022557437539208, (3, 4, 5), True, True, False, 1.0, 5, 'T'),
('P3_038', '甲丁壬', (0, 23, 58), '4:5:7', 0, 0.5534591498116744, (4, 5, 7), False, True, True, 2.0, 7, 'T'),
('P3_046', '甲戊庚', (0, 30, 42), '6:8:9', 0, 0.564275651940643, (6, 8, 9), True, False, False, 0.3333333333333333, 9, 'S'),
('P3_047', '戊辛甲', (30, 49, 0), '10:12:15', 30, 0.49773135763520054, (10, 12, 15), True, True, False, 0.0, 15, 'S'),
('P3_057', '辛甲己', (49, 0, 35), '4:5:7', 49, 0.5534591498116744, (4, 5, 7), False, True, True, 1.3333333333333333, 7, 'T'),
('P3_068', '甲庚壬', (0, 42, 58), '4:6:7', 0, 0.5534591498116744, (4, 6, 7), True, False, True, 1.6666666666666667, 7, 'T'),
('P3_137', '乙丁辛', (7, 23, 49), '6:7:9', 7, 0.5300587012617293, (6, 7, 9), True, False, True, 0.6666666666666666, 9, 'D'),
('P3_139', '乙丁癸', (7, 23, 65), '12:14:21', 7, 0.4639273663636983, (12, 14, 21), True, False, True, 1.6666666666666667, 21, 'D'),
('P3_147', '辛乙戊', (49, 7, 30), '3:4:5', 49, 0.6022557437539208, (3, 4, 5), True, True, False, 0.0, 5, 'S'),
('P3_149', '乙戊癸', (7, 30, 65), '4:5:7', 7, 0.5534591498116744, (4, 5, 7), False, True, True, 1.0, 7, 'C'),
('P3_179', '乙辛癸', (7, 49, 65), '4:6:7', 7, 0.5534591498116744, (4, 6, 7), True, False, True, 1.0, 7, 'C'),
('P3_246', '丙戊庚', (16, 30, 42), '7:8:9', 16, 0.5300587012617293, (7, 8, 9), True, False, True, 1.6666666666666667, 9, 'S'),
('P3_258', '丙己壬', (16, 35, 58), '10:12:15', 16, 0.49773135763520054, (10, 12, 15), True, True, False, 4.0, 15, 'C'),
('P3_259', '己癸丙', (35, 65, 16), '3:4:5', 35, 0.6022557437539208, (3, 4, 5), True, True, False, 3.6666666666666665, 5, 'C'),
('P3_268', '丙庚壬', (16, 42, 58), '14:18:21', 16, 0.4639273663636983, (14, 18, 21), True, False, True, 3.0, 21, 'C'),
('P3_359', '癸丁己', (65, 23, 35), '6:8:9', 65, 0.564275651940643, (6, 8, 9), True, False, False, 3.0, 9, 'D'),
('P3_368', '丁庚壬', (23, 42, 58), '5:6:7', 23, 0.5314574775365284, (5, 6, 7), True, True, True, 2.3333333333333335, 7, 'T'),
('P3_369', '丁庚癸', (23, 42, 65), '10:12:15', 23, 0.49773135763520054, (10, 12, 15), True, True, False, 2.0, 15, 'T'),
('P3_379', '丁辛癸', (23, 49, 65), '14:18:21', 23, 0.4639273663636983, (14, 18, 21), True, False, True, 1.6666666666666667, 21, 'D'),
('P3_479', '戊辛癸', (30, 49, 65), '5:6:7', 30, 0.5314574775365284, (5, 6, 7), True, True, True, 1.0, 7, 'C'),
('P3_579', '辛癸己', (49, 65, 35), '12:14:21', 49, 0.4639273663636983, (12, 14, 21), True, False, True, 2.3333333333333335, 21, 'C'),
('P4_0147', '乙戊辛甲', (7, 30, 49, 0), '8:10:12:15', 7, 0.49773135763520054, (8, 10, 12, 15), True, True, False, 0.0, 15, 'S'),
('P4_0246', '甲丙戊庚', (0, 16, 30, 42), '6:7:8:9', 0, 0.5300587012617293, (6, 7, 8, 9), True, False, True, 1.25, 9, 'S'),
('P4_0268', '甲丙庚壬', (0, 16, 42, 58), '12:14:18:21', 0, 0.4639273663636983, (12, 14, 18, 21), True, False, True, 2.25, 21, 'T'),
('P4_0368', '甲丁庚壬', (0, 23, 42, 58), '4:5:6:7', 0, 0.5314574775365284, (4, 5, 6, 7), True, True, True, 1.75, 7, 'T'),
('P4_0369', '甲丁庚癸', (0, 23, 42, 65), '8:10:12:15', 0, 0.49773135763520054, (8, 10, 12, 15), True, True, False, 1.5, 15, 'T'),
('P4_1379', '乙丁辛癸', (7, 23, 49, 65), '12:14:18:21', 7, 0.4639273663636983, (12, 14, 18, 21), True, False, True, 1.25, 21, 'D'),
('P4_1479', '乙戊辛癸', (7, 30, 49, 65), '4:5:6:7', 7, 0.5314574775365284, (4, 5, 6, 7), True, True, True, 0.75, 7, 'C'),
('P4_2589', '癸丙己壬', (65, 16, 35, 58), '8:10:12:15', 65, 0.49773135763520054, (8, 10, 12, 15), True, True, False, 3.75, 15, 'C'),
)

CHORDS:dict[str,Chord]={}
FUNCTION_CHORD_FAMILIES={f:frozenset() for f in 'TSDC'}
PALETTE_COLOR_FAMILIES={}
ANCHOR_CHORD_ID=None
ANCHOR_CHORD_IDS=()
NATURAL_SCALE_MODE=False
NATURAL_TRIAD_INFO={}
THREE_TONE_CONFIG={'enabled':False}
THREE_TONE_INFO={}

_NATURAL_MAJOR_SHAPES=frozenset(((3,4,5),(4,5,6),(5,6,8)))
_NATURAL_MINOR_SHAPES=frozenset(((10,12,15),(12,15,20),(15,20,24)))

# Scale-independent harmonic-function and voice-leading preferences.
FUNCTION_CLASS_PRIOR={'T':.30,'S':.25,'D':.25,'C':.20}
FUNCTION_TRANSITION_LOG_BIAS={
'T':{'T':-.18,'S':+.78,'D':-.22,'C':-.06},
'S':{'T':-.30,'S':-.16,'D':+.82,'C':-.05},
'D':{'T':+.92,'S':-.34,'D':-.18,'C':-.04},
'C':{'T':+.24,'S':+.12,'D':+.08,'C':-.14},}
COMMON_TONE_REWARD=.13
REPEAT_FACTOR=.30
CARDINALITY_BASE_WEIGHT={3:.70,4:.30}
_FUNCTION_TRANSITION_DEFAULTS=copy.deepcopy(FUNCTION_TRANSITION_LOG_BIAS)

def configure(cse_strength=.7,cse_harmony_gain=1.0):
    global _CSE_STRENGTH,_CSE_HARMONY_GAIN
    _CSE_STRENGTH=max(0.,float(cse_strength)); _CSE_HARMONY_GAIN=max(0.,float(cse_harmony_gain))


def _meta(path, default):
    cur=(SCALE.metadata if SCALE is not None else {}) or {}
    for key in path:
        if not isinstance(cur,dict) or key not in cur: return default
        cur=cur[key]
    return cur


def configure_scale(spec:ScaleSpec,bundle:SpectralBundle,ji_table=None):
    global SCALE,CSE,OCT,PCS,NAMES,PC_NAME,BASE_FREQ,RANGES,POOLS,VELOCITY,HARD_WOLF,SECOND_MAX_STEP
    global CHORDS,FUNCTION_CHORD_FAMILIES,PALETTE_COLOR_FAMILIES,ANCHOR_CHORD_ID,ANCHOR_CHORD_IDS
    global NATURAL_SCALE_MODE,NATURAL_TRIAD_INFO
    global THREE_TONE_CONFIG,THREE_TONE_INFO
    global FUNCTION_CLASS_PRIOR,FUNCTION_TRANSITION_LOG_BIAS
    SCALE,CSE=spec,bundle; OCT=int(spec.edo); PCS=tuple(spec.pcs); NAMES=tuple(spec.names); PC_NAME=dict(zip(PCS,NAMES))
    BASE_FREQ=float(spec.base_freq_hz); RANGES=spec.resolved_voice_ranges(); POOLS={v:list(spec.make_pool(*b)) for v,b in RANGES.items()}
    gen=_meta(('generator',),{})
    VELOCITY={**{'bass':.6,'inner':.6,'inner2':.6,'counter':.6,'lead':.6}, **{str(k):float(v) for k,v in gen.get('velocity',{}).items()}}
    wolves=tuple(spec.wolf_intervals); HARD_WOLF=int(wolves[0]) if wolves else -1
    SECOND_MAX_STEP=int(gen.get('second_max_step', max(1, round(OCT/spec.note_count))))
    hstyle=_meta(('harmony_style',),{})
    FUNCTION_CLASS_PRIOR={**{'T':.30,'S':.25,'D':.25,'C':.20}, **{str(k):float(v) for k,v in hstyle.get('function_class_prior',{}).items()}}
    fb=hstyle.get('function_transition_log_bias');
    FUNCTION_TRANSITION_LOG_BIAS=copy.deepcopy(_FUNCTION_TRANSITION_DEFAULTS)
    if fb:
        for a,row in fb.items(): FUNCTION_TRANSITION_LOG_BIAS[a].update({str(b):float(v) for b,v in row.items()})
    CHORDS=_build_chords(spec)
    NATURAL_TRIAD_INFO=_detect_natural_scale_triads(spec,ji_table)
    min_hits=int(((spec.harmony or {}).get('natural_scale_progression') or {}).get('min_triad_count',5))
    NATURAL_SCALE_MODE=(_is_meantone_diatonic(spec) and len(NATURAL_TRIAD_INFO)>=max(5,min_hits))
    if not NATURAL_SCALE_MODE:
        NATURAL_TRIAD_INFO={}
    if NATURAL_SCALE_MODE:
        for cid, chord in tuple(CHORDS.items()):
            degree = _natural_degree(chord)
            if degree is not None:
                CHORDS[cid] = replace(chord, function_code={1:"T", 2:"S", 3:"T", 4:"S", 5:"D", 6:"T", 7:"D"}[degree])
    _add_manual_progression_chords(spec)
    from three_tone_progression import resolve_config, build_annotations
    THREE_TONE_CONFIG=resolve_config(spec.harmony or {})
    THREE_TONE_INFO=(build_annotations(spec,CHORDS,ji_table,THREE_TONE_CONFIG)
                     if THREE_TONE_CONFIG['enabled'] else {})
    FUNCTION_CHORD_FAMILIES={f:frozenset(cid for cid,c in CHORDS.items() if c.function==f) for f in 'TSDC'}
    PALETTE_COLOR_FAMILIES={}
    for cid,c in CHORDS.items(): PALETTE_COLOR_FAMILIES.setdefault(c.prime_class,set()).add(cid)
    PALETTE_COLOR_FAMILIES={k:frozenset(v) for k,v in PALETTE_COLOR_FAMILIES.items()}
    ANCHOR_CHORD_IDS=_anchor_ids(spec)
    ANCHOR_CHORD_ID=ANCHOR_CHORD_IDS[0] if ANCHOR_CHORD_IDS else None
    if not ANCHOR_CHORD_IDS and not THREE_TONE_CONFIG['enabled']:
        raise RuntimeError('no eligible T-function anchor chords found')


def _build_chords(spec):
    if spec.edo == 72 and tuple(spec.pcs) == (0,7,16,23,30,35,42,49,58,65):
        out={}
        for row in _CANONICAL_TIANGAN_CHORD_ROWS:
            cid,name,pcs,ratio,foot,stability,integers,h3,h5,h7,m7,mx,fn=row
            out[cid]=Chord(cid,name,tuple(pcs),ratio,int(foot),float(stability),tuple(integers),bool(h3),bool(h5),bool(h7),float(m7),int(mx),fn)
        return out
    out={}
    stable=set(spec.resolved_harmony()['stability']['stable'])
    for k in (3,4):
        for combo in combinations(spec.pcs,k):
            # A pitch-class set has no fixed register. Exact-distance wolves
            # are enforced on sounding voicings, not used to erase its vocabulary.
            if any(spec.is_octave_equivalent_wolf(a,b) for a,b in combinations(combo,2)): continue
            candidates=[]
            for foot in combo:
                ref=tuple(sorted(foot+(pc-foot)%spec.edo for pc in combo)); item=objective_entry(ref)
                if item is not None: candidates.append((float(item[0]),float(item[1]),int(foot),ref))
            if not candidates: continue
            raw,pct,foot,ref=min(candidates)
            fn=functional_class_for_pcs(combo)
            cid=f'G{k}_'+''.join(f'{spec.pcs.index(pc):02d}' for pc in combo)
            name=''.join(spec.pc_name[pc] for pc in combo)
            st=max(0.,1.-pct)+.18*sum(pc in stable for pc in combo)/k
            # Generic scales intentionally do not invent TianGan-specific prime-colour semantics.
            out[cid]=Chord(cid,name,tuple(combo),'',foot,st,tuple(range(1,k+1)),False,False,False,0.0,k,fn)
    return out


def _add_manual_progression_chords(spec):
    progression = spec.resolved_chord_progression()
    if progression is None:
        return
    if 'codes' in progression:
        definitions = spec.resolved_chord_codes()
        missing = sorted({definitions[code]['chord_id'] for code in progression['codes']} - set(CHORDS))
        if missing:
            raise ValueError(f'Unknown chord ids in manual chord-code progression: {missing}')
        return
    for d in dict.fromkeys(progression['degrees']):
        pcs = tuple(spec.pcs[(d-1+i) % spec.note_count] for i in (0, 2, 4))
        root = pcs[0]
        ref = tuple(sorted(root + (pc-root) % spec.edo for pc in pcs))
        item = objective_entry(ref)
        if item is None:
            raise ValueError(f'No spectral entry for manual chord degree {d}: {ref}')
        cid = f'MANUAL_TRIAD_{d}'
        CHORDS[cid] = Chord(
            cid, '-'.join(spec.pc_name[pc] for pc in pcs), pcs, '', root,
            max(0., 1.-float(item[1])), (1, 2, 3), False, False, False,
            0.0, 3, functional_class_for_pcs(pcs))


def functional_class_for_pcs(pcs):
    """Generic functional classification for non-frozen chord vocabularies."""
    ps=frozenset(int(x)%OCT for x in pcs)
    h=SCALE.resolved_harmony(); roles=h['pitch_roles']; unstable=set(h['stability']['unstable'])
    tonic=int(roles['tonic'][0]); dom=set(roles['dominant']); sub=set(roles['subdominant'])
    if ps & sub and not ps & dom: return 'S'
    if tonic in ps and (ps & dom or not ps & sub): return 'T'
    if ps & dom and ps & unstable and tonic not in ps: return 'D'
    return 'C'


def _anchor_ids(spec):
    hstyle=(spec.metadata or {}).get('harmony_style',{}); a=hstyle.get('anchor',{})
    cardinality=int(a.get('cardinality',3)); req_all=set(map(int,a.get('required_all',(spec.resolved_harmony()['pitch_roles']['tonic'][0],))))
    req_any=set(map(int,a.get('required_any',spec.resolved_harmony()['pitch_roles']['dominant'])))
    ids=[]
    for cid,c in CHORDS.items():
        ps=set(c.pcs)
        if len(c.pcs)==cardinality and c.function=='T' and req_all.issubset(ps) and (not req_any or ps & req_any): ids.append(cid)
    if NATURAL_SCALE_MODE:
        natural_ids=tuple(cid for cid in ids if cid in NATURAL_TRIAD_INFO)
        if natural_ids:
            return natural_ids
    return tuple(ids)


def _reduce_integer_shape(values):
    values=tuple(int(x) for x in values)
    if not values or any(x<=0 for x in values):
        return ()
    divisor=values[0]
    for value in values[1:]:
        divisor=math.gcd(divisor,value)
    return tuple(value//max(1,divisor) for value in values)


def _detect_natural_scale_triads(spec,ji_table):
    """Find conventional major/minor triads by the canonical JI fit.

    A seven-note scale is considered diatonic only after at least five distinct
    pitch-class triads fit one of the three inversions of 4:5:6 or 10:12:15.
    Every possible chord foot is tested; the generic CSE-selected foot must not
    hide an otherwise conventional inversion.
    """
    if int(spec.note_count)!=7 or ji_table is None:
        return {}
    hits={}
    for cid,chord in CHORDS.items():
        if len(chord.pcs)!=3:
            continue
        candidates=[]
        for foot in chord.pcs:
            ref=tuple(sorted(int(foot)+(int(pc)-int(foot))%int(spec.edo)
                             for pc in chord.pcs))
            item=ji_table.lookup(ref)
            if not item:
                continue
            shape=_reduce_integer_shape(item.get('integers',()))
            if shape in _NATURAL_MAJOR_SHAPES:
                quality='major'
            elif shape in _NATURAL_MINOR_SHAPES:
                quality='minor'
            else:
                continue
            candidates.append((float(item.get('rms_error_cents',math.inf)),
                               max(shape),shape,int(foot),quality))
        if candidates:
            rms,_,shape,foot,quality=min(candidates)
            hits[cid]={'quality':quality,'shape':shape,'foot':foot,'rms_error_cents':rms}
    return hits


def degree_pitch(d:int)->int: return SCALE.degree_pitch(int(d))
def degree(p:int)->int: return SCALE.degree(int(p))
def pitch_name(p:int)->str: return SCALE.pitch_name(int(p))
def freq(p:int)->float: return SCALE.freq(int(p))
def make_pool(voice:str): return list(POOLS[voice])

def hard_wolf(a,b):
    if SCALE is None: return False
    return SCALE.is_hard_wolf(a,b)


def _entry(steps,default=None,prefer_inner=False):
    if CSE is None: return default
    return CSE.entry(tuple(sorted(map(int,steps))),prefer_inner=prefer_inner) or default

def cse_entry(steps): return _entry(steps,None,False)
def cse4_entry(steps): return _entry(steps,None,False) if len(tuple(steps))==4 else None
def cse4_score(steps,default=None):
    x=cse4_entry(tuple(steps)); return float(x[0]) if x is not None else default

def chord_reference_steps(chord:Chord):
    foot=chord.foot%OCT
    return tuple(sorted({foot+(pc-foot)%OCT for pc in chord.pcs}))
def objective_entry(steps):
    """Use the configured blend for harmony choices; raw APIs remain pure CSE."""
    if CSE is None: return cse_entry(steps)
    weights=SCALE.style.get('cse_weights',{})
    return CSE.mixed_entry(steps, weights.get('CSE_2D_A',1.),
                          weights.get('CSE_2D_B',0.), weights.get('CSE_2D_C',0.))

def chord_cse_score(chord):
    row=objective_entry(chord_reference_steps(chord))
    return row[0] if row is not None else 2.0

def _chord_base_weight(chord):
    """Function/cardinality prior and CSE, without pitch or prime-family quotas."""
    function_size=max(1,len(FUNCTION_CHORD_FAMILIES[chord.function]))
    w=FUNCTION_CLASS_PRIOR[chord.function]/function_size
    w*=CARDINALITY_BASE_WEIGHT[len(chord.pcs)]
    cse=chord_cse_score(chord); gain=_CSE_STRENGTH*_CSE_HARMONY_GAIN
    # Negative entropy blends must retain their ordering instead of all
    # collapsing to zero. Preserve the original curve on nonnegative costs.
    cost=.10*gain*cse
    w*=1./(1.+cost) if cost>=0 else 1.-cost
    if NATURAL_SCALE_MODE and chord.id in NATURAL_TRIAD_INFO:
        cfg=_natural_progression_config()
        w*=cfg['triad_weight']
    return max(w,1e-15)

def _chord_transition_weight(prev,cand,continuity_scale=1.0):
    """Use harmonic function, common tones and repetition; CSE scores sonority."""
    w=_chord_base_weight(cand); scale=max(0.,float(continuity_scale))
    if prev is not None:
        w*=math.exp(scale*FUNCTION_TRANSITION_LOG_BIAS[prev.function][cand.function])
        overlap=len(set(prev.pcs)&set(cand.pcs))
        w*=math.exp(scale*COMMON_TONE_REWARD*overlap)
        if prev.id==cand.id: w*=REPEAT_FACTOR
    return max(w,1e-15)

def weighted_choice(items,rng):
    vals,weights=zip(*items); return rng.choices(vals,weights=weights,k=1)[0]
def choose_chord(prev_cid,rng,continuity_scale=1.0):
    prev=CHORDS.get(prev_cid) if prev_cid else None
    dist=[(cid,_chord_transition_weight(prev,chord,continuity_scale)) for cid,chord in CHORDS.items()]
    return weighted_choice(dist,rng)

def section_chord_counts(rng,bpb):
    counts=[1]*8; movable=list(range(1,8)); n_two=(3 if rng.random()<(.74 if bpb==4 else .62) else 2)
    for i in rng.sample(movable,n_two): counts[i]=2
    if bpb==4:
        dense=[i for i in range(1,7) if counts[i]==1]
        if dense and rng.random()<.58: counts[rng.choice(dense)]=3
        dense=[i for i in range(1,7) if counts[i]==1]
        if dense and rng.random()<.18: counts[rng.choice(dense)]=4
    else:
        dense=[i for i in range(1,7) if counts[i]==1]
        if dense and rng.random()<.42: counts[rng.choice(dense)]=3
    counts[7]=min(counts[7],2); return counts

def choose_chord_durations(rng,bpb,n,phrase_final=False):
    if n==1:return (float(bpb),)
    if bpb==4:
        if n==2:
            if phrase_final:return (2.,2.)
            return weighted_choice((((2.,2.),.72),((3.,1.),.14),((1.,3.),.14)),rng)
        if n==3:return weighted_choice((((1.,1.,2.),.42),((1.,2.,1.),.29),((2.,1.,1.),.29)),rng)
        return (1.,1.,1.,1.)
    if n==2:
        if phrase_final:return (1.5,1.5)
        return weighted_choice((((1.5,1.5),.58),((2.,1.),.21),((1.,2.),.21)),rng)
    return (1.,1.,1.)

def _choose_anchor_chord_id(rng): return rng.choice(ANCHOR_CHORD_IDS)

def _is_canonical_tiangan_scale(spec):
    return (spec is not None and int(spec.edo)==72 and
            tuple(spec.pcs)==(0,7,16,23,30,35,42,49,58,65))


def _natural_progression_config():
    """Resolve the automatic seven-note major/minor progression defaults."""
    raw=dict((SCALE.harmony or {}).get('natural_scale_progression') or {})
    return {
        'enabled':bool(raw.get('enabled',True)),
        'min_triad_count':max(5,int(raw.get('min_triad_count',5))),
        # Retained for the joint objective's conventional-triad prior.
        'triad_weight':max(0.,float(raw.get('triad_weight',24.0))),
    }


# All routes prepare the dominant through ii or IV and resolve it to I.
# vi and iii provide tonic-family minor colour without D -> S retrogression.
_NATURAL_DEGREE_PATHS=(
    ((1,6,2,5,1,2,5,1), 1.0),
    ((1,3,6,4,2,2,5,1), .8),
    ((1,4,2,5,1,2,5,1), .7),
    ((1,6,2,2,5,5,1,1), .6),
)


def _natural_degree(chord):
    for i in range(len(PCS)):
        if set(chord.pcs)=={PCS[(i+j)%len(PCS)] for j in (0,2,4)}:
            return i+1
    return None


def _is_meantone_diatonic(spec):
    """Require a diatonic chain of six identical fifths; excludes JI wolves."""
    if spec.note_count != 7:
        return False
    pcs=set(spec.pcs)
    fifth=round(spec.edo * math.log2(1.5))
    # Meantone identifies four fifths with a major third plus two octaves.
    # An equal-fifth chain alone also admits non-meantone systems (e.g. 22).
    if 4*fifth-2*spec.edo != round(spec.edo * math.log2(5/4)):
        return False
    return any({(start+i*fifth)%spec.edo for i in range(7)}==pcs
               for start in pcs)


def _natural_segment(c,offset,duration,target):
    info=NATURAL_TRIAD_INFO.get(c.id)
    return {
        'offset':round(float(offset),6),'duration':round(float(duration),6),
        'chord_id':c.id,'chord_name':c.name,'ratio':c.ratio,
        'function':c.function,'stability':c.stability,
        'voicing_pcs':list(c.pcs),'prime_support':list(c.prime_support),
        'prime_class':c.prime_class,'has3':bool(c.has3),'has5':bool(c.has5),
        'has7':bool(c.has7),
        'chord_tone_7_colour_mean':round(c.chord_tone_7_colour_mean,6),
        'melodic_7_colour_strength':round(c.melodic_7_colour_strength,6),
        'max_integer':c.max_integer,'target_function':target,
        'natural_triad':bool(info),
        'natural_quality':(info or {}).get('quality'),
        'natural_ji_shape':list((info or {}).get('shape',())),
    }


def make_natural_section_harmony(section,rng,bpb,force_first_bar_tonic=False,bars=8):
    """Plan complete degree routes before the spectral/voicing search."""
    path=list(weighted_choice(_NATURAL_DEGREE_PATHS,rng)[:bars])
    # Truncated phrases still prepare and resolve a cadence.
    if bars < 8:
        path=([1,3,6,4][:bars-3]+[2,5,1] if bars>=3
              else [5,1][-bars:])
    if force_first_bar_tonic:
        path[0]=1
    by_degree={_natural_degree(c):c for c in CHORDS.values()
               if _natural_degree(c) is not None}
    missing=set(path)-set(by_degree)
    if missing:
        raise ValueError(f'meantone progression lacks legal degree triads: {sorted(missing)}')
    output=[]
    for degree in path:
        c=by_degree[degree]
        segment=_natural_segment(c,0.,bpb,c.function)
        segment.update(root_degree=degree,root_pc=PCS[degree-1])
        output.append([segment])
    return output

def _tiangan_progression_config():
    """Resolve the TianGan-only phrase curve from the rules JSON."""
    raw=dict((SCALE.harmony or {}).get('tiangan_progression') or {})
    order=tuple(map(int,raw.get('pitch_stability_order',(0,30,42,16,49,7,58,23,65,35))))
    if len(order)!=10 or set(order)!=set(SCALE.pcs):
        raise ValueError('tiangan_progression.pitch_stability_order must list all ten pitch classes once')
    denominator=max(1,len(order)-1)
    note_stability={pc:(len(order)-1-index)/denominator for index,pc in enumerate(order)}
    functions=('T','S','D')
    labels_raw=raw.get('pitch_function_weights') or {
        '0':{'T':1.0}, '7':{'S':1.0,'D':.35}, '16':{'T':.35},
        '23':{'D':1.0}, '30':{'S':1.0,'T':.35}, '35':{},
        '42':{'T':1.0,'D':.35}, '49':{'S':.35},
        '58':{'T':.35}, '65':{'D':.35},
    }
    function_weights={}
    for pc in SCALE.pcs:
        row={str(k):float(v) for k,v in (labels_raw.get(str(pc),labels_raw.get(pc,{})) or {}).items()}
        if set(row)-set(functions) or any(v<0. for v in row.values()):
            raise ValueError(f'tiangan_progression.pitch_function_weights.{pc} is invalid')
        function_weights[pc]={fn:row.get(fn,0.) for fn in functions}
    stability_curve=tuple(map(float,raw.get('stability_curve',(.68,.58,.46,.30,.22,.38,.58,.82))))
    if len(stability_curve)!=8 or any(not 0.<=x<=1. for x in stability_curve):
        raise ValueError('tiangan_progression.stability_curve must contain eight values in 0..1')
    start={**{'T':.76,'S':.19,'D':.05},
           **{str(k):float(v) for k,v in (raw.get('function_start_weights') or {}).items()}}
    default_transitions={
        'T':{'T':.18,'S':.62,'D':.20},
        'S':{'T':.10,'S':.18,'D':.72},
        'D':{'T':.62,'S':.15,'D':.23},
    }
    transition_raw=raw.get('function_transition_weights') or {}
    transitions={fn:{**default_transitions[fn],
                     **{str(k):float(v) for k,v in (transition_raw.get(fn) or {}).items()}}
                 for fn in functions}
    penultimate={**{'T':.17,'S':.15,'D':.68},
                 **{str(k):float(v) for k,v in (raw.get('penultimate_function_weights') or {}).items()}}
    final_function=str(raw.get('final_function','T'))
    for label,row in [('function_start_weights',start),
                      ('penultimate_function_weights',penultimate),*transitions.items()]:
        if set(row)!=set(functions) or any(v<0. for v in row.values()) or sum(row.values())<=0.:
            raise ValueError(f'tiangan_progression.{label} must give nonnegative T/S/D weights')
    if final_function not in functions:
        raise ValueError('tiangan_progression.final_function must be T, S or D')
    weights={
        'function_affinity':4.5, 'stability_curve':12.0, 'cse':.10,
        'off_path_penalty':6.0, 'out_of_key_penalty':1.5,
        'common_tone':.18, 'repeat_factor':.22,
    }
    weights.update({str(k):float(v) for k,v in (raw.get('weights') or {}).items()})
    prevalence={}
    for key,item in (raw.get('chord_pitch_class_prevalence') or {}).items():
        pc=int(key)
        if pc not in SCALE.pcs or not isinstance(item,dict):
            raise ValueError(f'tiangan_progression.chord_pitch_class_prevalence.{key} is invalid')
        target=float(item.get('target_share',0.))
        base=float(item.get('base_log_weight_boost',0.))
        feedback=float(item.get('feedback_gain',0.))
        prior=float(item.get('prior_duration',0.))
        excluded=frozenset(map(str,item.get('excluded_boost_chord_ids',())))
        if (not 0.<=target<=1. or not math.isfinite(base) or
                not math.isfinite(feedback) or feedback<0. or
                not math.isfinite(prior) or prior<0.):
            raise ValueError(f'tiangan_progression.chord_pitch_class_prevalence.{key} has invalid weights')
        prevalence[pc]={'target_share':target,'base_log_weight_boost':base,
                        'feedback_gain':feedback,'prior_duration':prior,
                        'excluded_boost_chord_ids':excluded}
    return {
        'enabled':bool(raw.get('enabled',True)), 'note_stability':note_stability,
        'functions':functions, 'function_weights':function_weights,
        'out_of_key_pcs':frozenset(map(int,raw.get('out_of_key_pitch_classes',(35,)))),
        'function_start_weights':start,
        'function_transition_weights':transitions,
        'penultimate_function_weights':penultimate,
        'final_function':final_function,
        'stability_curve':stability_curve, 'weights':weights,
        'chord_pitch_class_prevalence':prevalence,
    }

def _sample_tiangan_function_path(rng,cfg):
    """Biased Markov path: directional on average, different every phrase."""
    def pick(row):
        return weighted_choice([(fn,float(row[fn])) for fn in cfg['functions']],rng)
    path=[pick(cfg['function_start_weights'])]
    for bar in range(1,7):
        row=dict(cfg['function_transition_weights'][path[-1]])
        if bar==6:
            # Approach the cadence mostly from D, while retaining alternatives.
            row={fn:row[fn]*cfg['penultimate_function_weights'][fn] for fn in cfg['functions']}
        path.append(pick(row))
    path.append(cfg['final_function'])
    return tuple(path)

def _tiangan_chord_stability(chord,cfg):
    return sum(cfg['note_stability'][pc] for pc in chord.pcs)/len(chord.pcs)

def _tiangan_function_affinity(chord,target,cfg):
    return sum(cfg['function_weights'][pc][target] for pc in chord.pcs)/len(chord.pcs)

def _tiangan_chord_weight(prev,cand,target,target_stability,cfg,prevalence_state=None):
    """TianGan progression energy: side direction plus ranked-note stability."""
    weights=cfg['weights']
    stability=_tiangan_chord_stability(cand,cfg)
    affinity=_tiangan_function_affinity(cand,target,cfg)
    cse=float(chord_cse_score(cand))
    exponent=(weights['function_affinity']*affinity
              -weights['stability_curve']*(stability-target_stability)**2
              -weights['cse']*cse)
    if affinity<=0.:
        exponent-=weights['off_path_penalty']
    exponent-=weights['out_of_key_penalty']*sum(
        pc in cfg['out_of_key_pcs'] for pc in cand.pcs)/len(cand.pcs)
    if prevalence_state is not None:
        elapsed=float(prevalence_state.get('duration',0.))
        hits=prevalence_state.get('hits',{})
        for pc,item in cfg['chord_pitch_class_prevalence'].items():
            if pc not in cand.pcs or cand.id in item['excluded_boost_chord_ids']:
                continue
            prior=item['prior_duration']
            observed=(float(hits.get(pc,0.))+prior*item['target_share'])/max(1e-9,elapsed+prior)
            exponent+=(item['base_log_weight_boost']+
                       item['feedback_gain']*(item['target_share']-observed))
    # Keep the existing 3/4-note prior, but do not reuse the generic TSDC
    # transition matrix: TianGan has its own T->S->D->T trajectory.
    w=CARDINALITY_BASE_WEIGHT[len(cand.pcs)]*math.exp(max(-60.,min(60.,exponent)))
    if prev is not None:
        overlap=len(set(prev.pcs)&set(cand.pcs))
        w*=math.exp(weights['common_tone']*overlap)
        if prev.id==cand.id:
            w*=weights['repeat_factor']
    return max(w,1e-15)

def _choose_tiangan_chord(prev_cid,rng,target,target_stability,cfg,prevalence_state=None):
    prev=CHORDS.get(prev_cid) if prev_cid else None
    return weighted_choice([
        (cid,_tiangan_chord_weight(prev,chord,target,target_stability,cfg,prevalence_state))
        for cid,chord in CHORDS.items()
    ],rng)

def make_tiangan_section_harmony(section,rng,bpb,force_first_bar_tonic=False,
                                 prevalence_state=None):
    """Eight-bar TianGan phrase: medium/high -> low -> highest anchor."""
    cfg=_tiangan_progression_config()
    if prevalence_state is None:
        prevalence_state={'duration':0.,'hits':{}}
    function_path=_sample_tiangan_function_path(rng,cfg)
    bars_out=[]; prev_cid=None; count_profile=section_chord_counts(rng,bpb)
    duration_plan=[choose_chord_durations(rng,bpb,n,i==7) for i,n in enumerate(count_profile)]
    for i,durations in enumerate(duration_plan):
        target=function_path[i]; target_stability=cfg['stability_curve'][i]
        phrase_final=i==7; segments=[]; offset=0.
        for j,dur in enumerate(durations):
            forced_opening=force_first_bar_tonic and i==0 and j==0
            forced_final=phrase_final and j==len(durations)-1
            if forced_opening or forced_final:
                cid=_choose_anchor_chord_id(rng)
            else:
                cid=_choose_tiangan_chord(prev_cid,rng,target,target_stability,cfg,
                                          prevalence_state)
            c=CHORDS[cid]
            seg={'offset':round(offset,6),'duration':round(float(dur),6),'chord_id':cid,
                 'chord_name':c.name,'ratio':c.ratio,'function':c.function,
                 'stability':c.stability,'voicing_pcs':list(c.pcs),
                 'prime_support':list(c.prime_support),'prime_class':c.prime_class,
                 'has3':bool(c.has3),'has5':bool(c.has5),'has7':bool(c.has7),
                 'chord_tone_7_colour_mean':round(c.chord_tone_7_colour_mean,6),
                 'melodic_7_colour_strength':round(c.melodic_7_colour_strength,6),
                 'max_integer':c.max_integer,'target_function':target,
                 'tiangan_stability_score':round(_tiangan_chord_stability(c,cfg),6),
                 'target_stability':round(target_stability,6)}
            segments.append(seg); offset+=float(dur); prev_cid=cid
            prevalence_state['duration']=float(prevalence_state.get('duration',0.))+float(dur)
            hits=prevalence_state.setdefault('hits',{})
            for pc in cfg['chord_pitch_class_prevalence']:
                if pc in c.pcs:
                    hits[pc]=float(hits.get(pc,0.))+float(dur)
        if abs(offset-float(bpb))>1e-6:
            raise RuntimeError(f'harmony bar duration {offset} != {bpb}')
        bars_out.append(segments)
    return bars_out

def make_section_harmony(section,rng,bpb,force_first_bar_tonic=False):
    bars_out=[]; prev_cid=None; count_profile=section_chord_counts(rng,bpb)
    duration_plan=[choose_chord_durations(rng,bpb,n,i==7) for i,n in enumerate(count_profile)]
    for i,durations in enumerate(duration_plan):
        phrase_final=i==7; segments=[]; offset=0.
        for j,dur in enumerate(durations):
            forced_opening=force_first_bar_tonic and i==0 and j==0; forced_final=phrase_final and j==len(durations)-1
            if forced_opening or forced_final: cid=_choose_anchor_chord_id(rng)
            else:
                boundary_relief=.72 if (j==0 and i in (0,4)) else 1.; cid=choose_chord(prev_cid,rng,boundary_relief)
            c=CHORDS[cid]; seg={'offset':round(offset,6),'duration':round(float(dur),6),'chord_id':cid,'chord_name':c.name,'ratio':c.ratio,'function':c.function,'stability':c.stability,'voicing_pcs':list(c.pcs),'prime_support':list(c.prime_support),'prime_class':c.prime_class,'has3':bool(c.has3),'has5':bool(c.has5),'has7':bool(c.has7),'chord_tone_7_colour_mean':round(c.chord_tone_7_colour_mean,6),'melodic_7_colour_strength':round(c.melodic_7_colour_strength,6),'max_integer':c.max_integer}
            segments.append(seg); offset+=float(dur); prev_cid=cid
        if abs(offset-float(bpb))>1e-6: raise RuntimeError(f'harmony bar duration {offset} != {bpb}')
        bars_out.append(segments)
    return bars_out

def harmony_segment_at(h,beat_in_bar):
    beat=min(max(0.,float(beat_in_bar)),float(h['beats_per_bar'])-1e-9)
    for seg in reversed(h['chord_segments']):
        if beat+1e-9>=seg['offset']:return seg
    return h['chord_segments'][0]
def harmony_remaining(h,beat_in_bar):
    seg=harmony_segment_at(h,beat_in_bar); return max(0.,seg['offset']+seg['duration']-float(beat_in_bar))
def _force_anchor_segment(seg,rng):
    c=CHORDS[_choose_anchor_chord_id(rng)]; out=dict(seg); out.update({'chord_id':c.id,'chord_name':c.name,'ratio':c.ratio,'function':c.function,'stability':c.stability,'voicing_pcs':list(c.pcs),'prime_support':list(c.prime_support),'prime_class':c.prime_class,'has3':bool(c.has3),'has5':bool(c.has5),'has7':bool(c.has7),'chord_tone_7_colour_mean':round(c.chord_tone_7_colour_mean,6),'melodic_7_colour_strength':round(c.melodic_7_colour_strength,6),'max_integer':c.max_integer}); return out
def _refresh_primary_from_first_segment(h):
    seg=h['chord_segments'][0]; keys=('chord_id','chord_name','ratio','function','stability','voicing_pcs','prime_support','prime_class','has3','has5','has7','chord_tone_7_colour_mean','melodic_7_colour_strength','max_integer')
    for key in keys:
        if key in seg:h[key]=list(seg[key]) if key in ('voicing_pcs','prime_support') else seg[key]
    if 'three_tone_status' in seg:
        h.update({k:v for k,v in seg.items() if k.startswith('three_tone_') or
                  k in ('legacy_function','phrase_role','target_function')})
    h['end_function']=h['chord_segments'][-1]['function']; h['end_prime_class']=h['chord_segments'][-1]['prime_class']
def vary_harmonic_rhythm(template, rng, bpb):
    """Move existing chord boundaries by half a bar without reordering chords.

    Applies to every automatic scale; explicit manual progressions bypass it.
    Existing interior changes are preserved. Duration floors protect the
    planned minor colour instead of letting shorter chords erase it.
    """
    from melody_plan import resolve_melody_plan
    probability = float(resolve_melody_plan(SCALE.style.get('melody_plan'))[
        'harmonic_rhythm_variation'])
    if not probability or len(template) < 2:
        return template
    segments, boundaries = [], [0.]
    for bar in template:
        for segment in bar:
            segments.append(dict(segment))
            boundaries.append(boundaries[-1]+float(segment['duration']))
    minor = [s.get('natural_quality') == 'minor' for s in segments]
    def minor_duration(points):
        return sum(b-a for a,b,m in zip(points, points[1:], minor) if m)
    floor = min(minor_duration(boundaries), 2*bpb)
    choices = [i for i in range(1, len(segments))
               if abs(boundaries[i]/bpb-round(boundaries[i]/bpb)) < 1e-7
               and segments[i-1]['chord_id'] != segments[i]['chord_id']]
    rng.shuffle(choices)
    moved = False
    for position, i in enumerate(choices):
        if rng.random() > probability and (moved or position < len(choices)-1):
            continue
        directions = [-1, 1]
        rng.shuffle(directions)
        for direction in directions:
            trial = list(boundaries)
            trial[i] += direction*bpb/2
            if (trial[i]-trial[i-1] < bpb/2-1e-7 or
                    trial[i+1]-trial[i] < bpb/2-1e-7 or
                    minor_duration(trial) < floor-1e-7):
                continue
            boundaries = trial
            moved = True
            break
    result = []
    for bar in range(len(template)):
        rows = []
        for segment, a, b in zip(segments, boundaries, boundaries[1:]):
            lo, hi = max(a, bar*bpb), min(b, (bar+1)*bpb)
            if hi-lo <= 1e-7:
                continue
            row = dict(segment, offset=round(lo-bar*bpb, 6), duration=round(hi-lo, 6))
            if rows and rows[-1]['chord_id'] == row['chord_id']:
                rows[-1]['duration'] = round(rows[-1]['duration']+row['duration'], 6)
            else:
                rows.append(row)
        result.append(rows)
    return result


def harmony_plan(bars,rng,bpb):
    bars=int(bars)
    if bars<=0:raise ValueError('bars must be positive')
    progression = SCALE.resolved_chord_progression()
    if progression is not None:
        return _manual_harmony_plan(bars, bpb, progression)
    three_tone_mode=THREE_TONE_CONFIG['enabled']
    tiangan_cfg=_tiangan_progression_config() if _is_canonical_tiangan_scale(SCALE) else None
    tiangan_mode=bool(tiangan_cfg and tiangan_cfg['enabled'])
    natural_cfg=_natural_progression_config() if NATURAL_SCALE_MODE else None
    natural_mode=bool(natural_cfg and natural_cfg['enabled'])
    templates={}; occurrences={'A':0,'B':0}; out=[]; phrase_count=(bars+7)//8
    prevalence_state={'duration':0.,'hits':{}}
    for ph in range(phrase_count):
        section=FORM_TEMPLATE[ph%len(FORM_TEMPLATE)]
        if three_tone_mode:
            from three_tone_progression import make_section
            template=make_section(sys.modules[__name__],rng,bpb,min(8,bars-len(out)))
        elif tiangan_mode:
            # TianGan deliberately redraws every phrase instead of copying the
            # A/B templates; the seed remains deterministic, but phrases vary.
            template=make_tiangan_section_harmony(
                section,rng,bpb,force_first_bar_tonic=len(out)==0,
                prevalence_state=prevalence_state)
        elif natural_mode:
            length=min(8,bars-len(out))
            key=(section,length)
            if key not in templates:
                templates[key]=make_natural_section_harmony(
                    section,rng,bpb,force_first_bar_tonic=len(out)==0,bars=length)
            template=templates[key]
        else:
            if section not in templates:
                templates[section]=make_section_harmony(section,rng,bpb,force_first_bar_tonic=len(out)==0)
            template=templates[section]
        template = vary_harmonic_rhythm(template, rng, bpb)
        occ=occurrences[section]; occurrences[section]+=1
        for i,template_segments in enumerate(template):
            if len(out)>=bars:break
            segments=[dict(seg) for seg in template_segments]; primary=segments[0]; absolute_bar=len(out)
            row={'bar':absolute_bar,'phrase_index':ph,'bar_in_phrase':i,'subphrase_index':absolute_bar//4,'subphrase_in_phrase':i//4,'bar_in_subphrase':i%4,'section':section,'section_occurrence':occ,'beats_per_bar':bpb,'chord_id':primary['chord_id'],'chord_name':primary['chord_name'],'ratio':primary['ratio'],'function':primary['function'],'stability':primary['stability'],'voicing_pcs':list(primary['voicing_pcs']),'prime_support':list(primary['prime_support']),'prime_class':primary['prime_class'],'has3':primary['has3'],'has5':primary['has5'],'has7':primary['has7'],'chord_tone_7_colour_mean':primary['chord_tone_7_colour_mean'],'melodic_7_colour_strength':primary['melodic_7_colour_strength'],'max_integer':primary['max_integer'],'chords_in_bar':len(segments),'end_function':segments[-1]['function'],'end_prime_class':segments[-1]['prime_class'],'chord_segments':segments}
            if three_tone_mode:
                row.update({k:v for k,v in primary.items()
                            if k.startswith('three_tone_') or k in
                            ('harmony_model','legacy_function','phrase_role','target_function',
                             'natural_triad','natural_quality','natural_ji_shape','root_degree','root_pc')})
            elif tiangan_mode:
                row.update({'harmony_model':'tiangan_ranked_TSDT',
                            'target_function':primary['target_function'],
                            'tiangan_stability_score':primary['tiangan_stability_score'],
                            'target_stability':primary['target_stability']})
            elif natural_mode:
                row.update({'harmony_model':'natural_scale_TSDT',
                            'target_function':primary['target_function'],
                            'natural_triad':primary['natural_triad'],
                            'natural_quality':primary['natural_quality'],
                            'natural_ji_shape':list(primary['natural_ji_shape']),
                            'natural_triad_count':len(NATURAL_TRIAD_INFO)})
            out.append(row)
    if out and not three_tone_mode and not tiangan_mode and not natural_mode:
        out[0]['chord_segments'][0]=_force_anchor_segment(out[0]['chord_segments'][0],rng); _refresh_primary_from_first_segment(out[0])
        out[-1]['chord_segments'][-1]=_force_anchor_segment(out[-1]['chord_segments'][-1],rng); _refresh_primary_from_first_segment(out[-1])
    return out

def _manual_harmony_plan(bars, bpb, progression):
    """Fixed background chords; never overwrite the opening or closing chord."""
    codes = progression.get('codes')
    degrees = progression.get('degrees')
    sequence = codes if codes is not None else degrees
    definitions = SCALE.resolved_chord_codes() if codes is not None else None
    span = progression['bars_per_chord']
    phrase_bars = span * len(sequence)
    out = []
    for bar in range(bars):
        position = (bar // span) % len(sequence)
        token = sequence[position]
        if codes is not None:
            c = CHORDS[definitions[token]['chord_id']]
            d = None
        else:
            d = token
            c = CHORDS[f'MANUAL_TRIAD_{d}']
        seg = {
            'offset': 0.0, 'duration': float(bpb), 'chord_id': c.id,
            'chord_name': c.name, 'ratio': c.ratio, 'function': c.function,
            'stability': c.stability, 'voicing_pcs': list(c.pcs),
            'prime_support': list(c.prime_support), 'prime_class': c.prime_class,
            'has3': c.has3, 'has5': c.has5, 'has7': c.has7,
            'chord_tone_7_colour_mean': c.chord_tone_7_colour_mean,
            'melodic_7_colour_strength': c.melodic_7_colour_strength,
            'max_integer': c.max_integer, 'root_pc': c.foot,
        }
        if d is not None:
            seg['root_degree'] = d
        else:
            seg['chord_code'] = token
        if THREE_TONE_CONFIG['enabled']:
            seg.update(THREE_TONE_INFO[c.id])
        ph, local = divmod(bar, phrase_bars)
        section = FORM_TEMPLATE[ph % len(FORM_TEMPLATE)]
        h = {
            'bar': bar, 'phrase_index': ph, 'bar_in_phrase': local,
            'phrase_bars': phrase_bars,
            'subphrase_index': bar, 'subphrase_in_phrase': local,
            'bar_in_subphrase': 0, 'section': section,
            'section_occurrence': sum(FORM_TEMPLATE[i % len(FORM_TEMPLATE)] == section for i in range(ph)),
            'beats_per_bar': bpb, 'chords_in_bar': 1, 'chord_segments': [seg],
            'manual_progression': True, 'progression_index': position,
            'progression_cycle': bar // (span*len(sequence)),
            'root_pc': c.foot,
        }
        if d is not None:
            h['root_degree'] = d
        else:
            h['chord_code'] = token
        _refresh_primary_from_first_segment(h)
        if THREE_TONE_CONFIG['enabled']:
            h.update(THREE_TONE_INFO[c.id])
        out.append(h)
    return out

def chord_at(plan,absolute_beat,beats_per_bar):
    bar=min(len(plan)-1,max(0,int(float(absolute_beat)//float(beats_per_bar)))); local=float(absolute_beat)-bar*float(beats_per_bar); return CHORDS[harmony_segment_at(plan[bar],local)['chord_id']]

# Rhythm grammar
SUPPORTED_TIME_SIGNATURES = {'4/4': 4, '3/4': 3}

# Rhythm grammar. In 4/4, digit strings use eighth-note units:
# 2222 = four quarter notes; each 22 half-bar may be replaced by
# 211 / 121 / 112 / 1111 / 31 / 13. This yields 49 coherent basic patterns.
HALF_BAR_CELLS_44 = ('22', '211', '121', '112', '1111', '31', '13')
BASIC_RHYTHM_CODES_44 = tuple(a + b for a in HALF_BAR_CELLS_44 for b in HALF_BAR_CELLS_44)
FANCY_RHYTHM_CODES_44 = (
    '2321', '2132', '1232', '2312', '332', '21212',
    '3212', '3221', '3311', '1223', '2123', '1133',
    '233', '323', '11123', '11312', '12212', '21311',
)

# 3/4 uses the same song-level palette idea.
BASIC_RHYTHM_CODES_34 = (
    '222', '2112', '1212', '1122', '2211', '2121', '1221',
    '3111', '1311', '1131', '1113', '111111',
)
FANCY_RHYTHM_CODES_34 = ('321', '231', '312', '213', '132', '123', '33', '2121')

# Lead-only compressed rhythm layer.  These rates make sixteenth-note motion
# common without forcing it into every cadence.  No tuplets/triplets exist in
# either the lead or accompaniment rhythm palettes.
LEAD_SIXTEENTH_RATE = 0.62
LEAD_SIXTEENTH_CADENCE_RATE = 0.32
LEAD_SIXTEENTH_PALETTE_SIZE = 12
LEAD_FANCY_RHYTHM_RATE = 0.16
LEAD_SYNCOPATED_BASIC_WEIGHT = 0.35

# Lower accompaniment (Inner/Bass) deliberately has a slower rhythmic layer.
# In 4/4, adding the half-bar cell '4' means the two quarter-note slots '22'
# may fuse into one half note.  Pairing it with every ordinary half-bar cell
# yields front-half, back-half, and whole-bar long-value variants without
# changing the Lead/Counter rhythm grammar.
LOWER_HALF_BAR_CELLS_44 = ('4',) + HALF_BAR_CELLS_44
LOWER_LONG_RHYTHM_CODES_44 = tuple(
    a + b for a in LOWER_HALF_BAR_CELLS_44 for b in LOWER_HALF_BAR_CELLS_44
    if '4' in (a, b)
)
# 3/4 has no literal two-beat half-bar, so use the analogous ordinary patterns:
# a two-beat half note at the front or back, with the remaining beat either
# intact or split into two eighth notes.
LOWER_LONG_RHYTHM_CODES_34 = ('42', '411', '24', '114')
# Very slow anchors used only by Bass/Inner.  They make the lower layer sustain
# instead of competing with Lead/Counter articulation.
LOWER_WHOLE_RHYTHMS = {
    4: ('acc_whole_8', (4.0,)),
    3: ('acc_whole34_6', (3.0,)),
}

CADENCE_RHYTHMS = {
    4: (
        ('cad_2224', (1, 1, 2)),
        ('cad_2114', (1, .5, .5, 2)),
        ('cad_1124', (.5, .5, 1, 2)),
        ('cad_1214', (.5, 1, .5, 2)),
        ('cad_314', (1.5, .5, 2)),
    ),
    3: (
        ('cad_24', (1, 2)),
        ('cad_114', (.5, .5, 2)),
        ('cad_213', (1, .5, 1.5)),
        ('cad_123', (.5, 1, 1.5)),
    ),
}

# Accompaniment rhythm vocabulary deliberately reuses the same large grammar as
# the lead.  A song first chooses a small shared palette, then each lower voice
# receives only a subset of it.  Thus the global library is broad, while one
# piece speaks a coherent rhythmic "dialect" instead of becoming a grab bag.

def _rhythm_slots(durations):
    """Return non-overlapping slots whose rounded boundaries close exactly.

    The score format stores beat positions to 6 decimals.  Quantising cumulative
    boundaries first and taking differences guarantees that every rhythm closes
    exactly at the bar line, including the new compressed sixteenth-note layer.
    """
    boundaries = [0.0]
    for dur in durations:
        boundaries.append(boundaries[-1] + float(dur))
    q = [round(x, 6) for x in boundaries]
    return tuple(((q[i], round(q[i + 1] - q[i], 6)) for i in range(len(durations))))

def _rhythm_attacks(durations, max_duration=None):
    """Convert a rhythm skeleton into non-overlapping accompaniment attacks."""
    out = []
    for beat, slot in _rhythm_slots(durations):
        note_dur = slot if max_duration is None else min(slot, float(max_duration))
        out.append((beat, round(max(0.08, note_dur), 6)))
    return tuple(out)

def _semi_broken_attacks(durations):
    out = []
    for i, (beat, slot) in enumerate(_rhythm_slots(durations)):
        strongish = abs(beat - round(beat)) < 1e-06 and (i == 0 or i % 3 == 0)
        kind = 'dyad' if strongish or i % 4 == 0 else 'single'
        out.append((beat, slot, kind))
    return tuple(out)

def _has_triplet_value(durations):
    """Return whether a generated palette accidentally contains a tuplet."""
    vals = (1 / 6, 1 / 3, 2 / 3)
    return any(any(abs(float(d) - v) < 1e-06 for v in vals) for d in durations)

def rhythm_is_syncopated(durations):
    """Recognise the palette's weak-beat/cross-beat rhythmic cells.

    Regular half and whole notes are not labelled syncopated.  The detector
    covers 121/31/13-style cells and their fancy combinations.
    """
    cursor = 0.0
    for raw in durations:
        duration = float(raw)
        fractional_start = abs(cursor - round(cursor)) > 1e-06
        crosses_integer = math.floor(cursor + 1e-06) < math.floor(cursor + duration - 1e-06)
        fractional_long_value = duration > .5 + 1e-06 and abs(duration - round(duration)) > 1e-06
        if fractional_long_value or (fractional_start and crosses_integer):
            return True
        cursor += duration
    return False

def apply_category_frequency_multiplier(weights, flags, multiplier):
    """Rescale a category so its probability, not merely its raw weight, changes.

    With multiplier=.5 the category's probability under the supplied weights
    is halved, regardless of how many category members are in the palette.
    """
    values = [max(0.0, float(weight)) for weight in weights]
    marked = math.fsum(weight for weight, flag in zip(values, flags) if flag)
    other = math.fsum(weight for weight, flag in zip(values, flags) if not flag)
    if marked <= 0.0 or other <= 0.0:
        return values
    probability = marked / (marked + other)
    target = max(0.0, min(1.0 - 1e-12, float(multiplier) * probability))
    factor = target * other / (marked * (1.0 - target))
    return [weight * factor if flag else weight
            for weight, flag in zip(values, flags)]

def make_accompaniment_rhythm_palette(rng, bpb):
    """Choose a coherent accompaniment palette without tuplets.

    Bass remains strongly long-valued, Inner keeps long anchors but is mildly
    denser, and Counter retains the agile ordinary vocabulary.
    """
    if bpb == 4:
        basic_all = list(_named_digit_patterns(BASIC_RHYTHM_CODES_44, 'acc_basic'))
        base = next(x for x in basic_all if x[0] == 'acc_basic_2222')
        shared_basic = [base] + rng.sample([x for x in basic_all if x != base], 6)
        long_all = list(_named_digit_patterns(LOWER_LONG_RHYTHM_CODES_44, 'acc_long'))
        half_anchor = next(x for x in long_all if x[0] == 'acc_long_44')
        shared_long = [half_anchor] + rng.sample([x for x in long_all if x != half_anchor], 4)
        plain_fancy_all = list(_named_digit_patterns(FANCY_RHYTHM_CODES_44, 'acc_fancy'))
        shared_plain_fancy = rng.sample(plain_fancy_all, 3)
    else:
        basic_all = list(_named_digit_patterns(BASIC_RHYTHM_CODES_34, 'acc_basic34'))
        base = next(x for x in basic_all if x[0] == 'acc_basic34_222')
        shared_basic = [base] + rng.sample([x for x in basic_all if x != base], 5)
        long_all = list(_named_digit_patterns(LOWER_LONG_RHYTHM_CODES_34, 'acc_long34'))
        anchors = [next(x for x in long_all if x[0] == 'acc_long34_42'),
                   next(x for x in long_all if x[0] == 'acc_long34_24')]
        shared_long = anchors + rng.sample([x for x in long_all if x not in anchors], 2)
        plain_fancy_all = list(_named_digit_patterns(FANCY_RHYTHM_CODES_34, 'acc_fancy34'))
        shared_plain_fancy = rng.sample(plain_fancy_all, 2)
    whole = LOWER_WHOLE_RHYTHMS[bpb]

    def lower_subset(n_basic=2, n_long=4, n_fancy=1):
        basics = [whole, base]
        basics += rng.sample([x for x in shared_basic if x != base], min(n_basic, len(shared_basic) - 1))
        basics += rng.sample(shared_long, min(n_long, len(shared_long)))
        basics = list(dict.fromkeys(basics))
        fancy = rng.sample(shared_plain_fancy, min(n_fancy, len(shared_plain_fancy)))
        assert not any(_has_triplet_value(ds) for _, ds in basics + fancy)
        return {'basic': basics, 'fancy': fancy}

    def inner_subset():
        # Keep the old lower-layer long-value vocabulary, but admit a little
        # more quarter/eighth motion.  Increased chord-change splitting already
        # makes Inner denser, so the rhythm palette itself remains restrained.
        basics = [whole, base]
        basics += rng.sample([x for x in shared_basic if x != base], min(1, len(shared_basic) - 1))
        basics += rng.sample(shared_long, min(4, len(shared_long)))
        basics = list(dict.fromkeys(basics))
        fancy = rng.sample(shared_plain_fancy, min(1, len(shared_plain_fancy)))
        assert not any(_has_triplet_value(ds) for _, ds in basics + fancy)
        return {'basic': basics, 'fancy': fancy}

    def counter_subset(n_basic, n_plain_fancy):
        basics = [base] + rng.sample([x for x in shared_basic if x != base], min(n_basic - 1, len(shared_basic) - 1))
        fancy = rng.sample(shared_plain_fancy, min(n_plain_fancy, len(shared_plain_fancy)))
        return {'basic': basics, 'fancy': fancy}

    return {
        'shared': {'basic': shared_basic, 'lower_long': shared_long, 'plain_fancy': shared_plain_fancy},
        'bass': lower_subset(2, 4, 1),
        'inner': inner_subset(),
        'counter': counter_subset(6, 2),
    }

def choose_accompaniment_rhythm(palette, voice, rng, fancy_rate):
    vp = palette[voice]
    if vp['fancy'] and rng.random() < fancy_rate:
        return rng.choice(vp['fancy'])
    if voice == 'counter':
        weights = [2.2 if name.endswith('_2222') or name.endswith('_222') else 1.0
                   for name, _ in vp['basic']]
        return rng.choices(vp['basic'], weights=weights, k=1)[0]
    if voice == 'inner':
        # Roughly 20% denser than the old Inner after harmony-boundary splitting:
        # whole/half-note anchors remain common, but quarter-rich patterns are
        # no longer as strongly suppressed.
        weights = []
        for _, ds in vp['basic']:
            ds = tuple(float(x) for x in ds)
            eighths = sum(abs(x - .5) < 1e-6 for x in ds)
            if len(ds) == 1:
                w = 4.2
            elif len(ds) == 2 and min(ds) >= 1.0 - 1e-6:
                w = 7.0
            elif len(ds) == 3:
                w = 2.4
            else:
                w = .75
            w *= .48 ** eighths
            weights.append(max(.03, w))
        return rng.choices(vp['basic'], weights=weights, k=1)[0]

    # Lower-layer density model: half notes and whole-bar sustains dominate;
    # eighth-note-rich patterns are explicitly suppressed. Inner is even more
    # conservative than Bass because chordal attacks contain multiple notes.
    weights = []
    for _, ds in vp['basic']:
        ds = tuple(float(x) for x in ds)
        eighths = sum(abs(x - .5) < 1e-6 for x in ds)
        halves = sum(abs(x - 2.0) < 1e-6 for x in ds)
        total = sum(ds)
        if len(ds) == 1:
            # 3/4 needs an even stronger dotted-half anchor; otherwise its
            # natural 2+1 patterns still articulate too often.
            if abs(total - 3.0) < 1e-6:
                w = 18.0 if voice == 'bass' else 26.0
            else:
                w = 4.0 if voice == 'bass' else 8.0
        elif len(ds) == 2 and min(ds) >= 1.0 - 1e-6:
            if abs(total - 3.0) < 1e-6:
                w = 4.5 if voice == 'bass' else 3.5
            else:
                w = 7.0 if voice == 'bass' else 6.0
        else:
            w = 1.5 if voice == 'bass' else .8
        w *= (1.7 ** halves) * ((.28 if voice == 'bass' else .18) ** eighths)
        w *= (.78 if voice == 'bass' else .62) ** max(0, len(ds) - 2)
        weights.append(max(.01, w))
    return rng.choices(vp['basic'], weights=weights, k=1)[0]

def validate_rhythm_bar(durations, bpb, name='rhythm'):
    total = sum((float(x) for x in durations))
    if abs(total - float(bpb)) > 1e-06:
        raise RuntimeError(f'{name} totals {total} beats, expected {bpb}')

def accompaniment_rhythm_palette_metadata(palette):

    def encode(group):
        return {kind: [{'name': name, 'durations': [round(float(x), 6) for x in ds]} for name, ds in items] for kind, items in group.items()}
    return {voice: encode(group) for voice, group in palette.items()}

def _digit_rhythm(code):
    return tuple((int(ch) / 2 for ch in code))

def _named_digit_patterns(codes, prefix):
    return tuple(((f'{prefix}_{code}', _digit_rhythm(code)) for code in codes))

def _scaled_half_pattern(name, durations):
    return name, tuple(round(float(d) * 0.5, 6) for d in durations)

def _lead_sixteenth_patterns_44(rng, count=LEAD_SIXTEENTH_PALETTE_SIZE):
    """Build 4/4 sixteenth patterns one half-bar at a time.

    A dense half compresses one existing 4-beat pattern into 2 beats.  A plain
    half is one of the seven native 2-beat cells (22/211/121/112/1111/31/13).
    The palette deliberately contains front-dense, back-dense and both-dense
    layouts, so a bar need not carry sixteenths in both halves.
    """
    whole = list(_named_digit_patterns(BASIC_RHYTHM_CODES_44, 'dense_src_basic'))
    whole += list(_named_digit_patterns(FANCY_RHYTHM_CODES_44, 'dense_src_fancy'))
    # A dense source must actually yield at least one 0.25-beat note after
    # compression.  This avoids counting plain eighth-note halves as
    # "sixteenth" texture just because they came through the dense branch.
    whole = [(n, ds) for n, ds in whole if any(abs(float(d) - .5) < 1e-9 for d in ds)]
    plain = [(f'half_{code}', _digit_rhythm(code)) for code in HALF_BAR_CELLS_44]

    def dense_half():
        name, ds = rng.choice(whole)
        return name, tuple(round(float(d) * .5, 6) for d in ds)

    def plain_half():
        return rng.choice(plain)

    out = []
    layouts = ('front', 'back', 'both')
    for i in range(count):
        layout = layouts[i % len(layouts)]
        if layout == 'front':
            n1, d1 = dense_half(); n2, d2 = plain_half()
        elif layout == 'back':
            n1, d1 = plain_half(); n2, d2 = dense_half()
        else:
            n1, d1 = dense_half(); n2, d2 = dense_half()
        ds = tuple(d1 + d2)
        validate_rhythm_bar(ds, 4, f'lead16_44_{i}')
        if not any(abs(float(d) - .25) < 1e-9 for d in ds):
            raise RuntimeError('4/4 sixteenth pattern lacks a 0.25-beat note')
        out.append((f'lead16_44_{layout}_{i}_{n1}_{n2}', ds))
    rng.shuffle(out)
    return out

def _lead_sixteenth_patterns_34(rng, count=LEAD_SIXTEENTH_PALETTE_SIZE):
    """Build 3/4 dense patterns from three halved 2-beat basic cells."""
    cells = [(code, _digit_rhythm(code)) for code in HALF_BAR_CELLS_44]
    out = []
    for i in range(count):
        chosen = [rng.choice(cells) for _ in range(3)]
        ds = tuple(round(float(d) * 0.5, 6) for _, cell in chosen for d in cell)
        validate_rhythm_bar(ds, 3, f'lead16_34_{i}')
        out.append((f"lead16_34_{i}_{'_'.join(code for code, _ in chosen)}", ds))
    return out

def make_rhythm_palette(rng, bpb):
    if bpb == 4:
        basic_all = list(_named_digit_patterns(BASIC_RHYTHM_CODES_44, 'basic'))
        base = next(x for x in basic_all if x[0] == 'basic_2222')
        basic = [base] + rng.sample([x for x in basic_all if x != base], 5)
        fancy_all = list(_named_digit_patterns(FANCY_RHYTHM_CODES_44, 'fancy'))
        fancy = rng.sample(fancy_all, min(6, len(fancy_all)))
        cadence = rng.sample(list(CADENCE_RHYTHMS[4]), 2)
        sixteenth = _lead_sixteenth_patterns_44(rng)
    else:
        basic_all = list(_named_digit_patterns(BASIC_RHYTHM_CODES_34, 'basic34'))
        base = next(x for x in basic_all if x[0] == 'basic34_222')
        basic = [base] + rng.sample([x for x in basic_all if x != base], 4)
        fancy_all = list(_named_digit_patterns(FANCY_RHYTHM_CODES_34, 'fancy34'))
        fancy = rng.sample(fancy_all, min(5, len(fancy_all)))
        cadence = rng.sample(list(CADENCE_RHYTHMS[3]), 2)
        sixteenth = _lead_sixteenth_patterns_34(rng)
    return {'basic': basic, 'fancy': fancy, 'cadence': cadence, 'sixteenth': sixteenth}


def rhythm_palette_metadata(palette):
    return {k: [{'name': name, 'durations': [round(float(x), 6) for x in ds]} for name, ds in items] for k, items in palette.items()}

def normalize_time_signature(ts):
    if isinstance(ts, (tuple, list)):
        ts = f'{int(ts[0])}/{int(ts[1])}'
    ts = str(ts)
    if ts not in SUPPORTED_TIME_SIGNATURES:
        raise ValueError('time_signature must be 4/4 or 3/4')
    return (ts, SUPPORTED_TIME_SIGNATURES[ts])

def is_strong_beat(beat, beats_per_bar):
    return abs(beat) < 1e-06 or (beats_per_bar == 4 and abs(beat - 2) < 1e-06)

def rhythm_for_bar(bar, rng, bpb, palette):
    phrase_final = bar % 8 == 7
    dense_rate = LEAD_SIXTEENTH_CADENCE_RATE if phrase_final else LEAD_SIXTEENTH_RATE
    if palette.get('sixteenth') and rng.random() < dense_rate:
        _, rhythm = rng.choice(palette['sixteenth'])
    elif phrase_final:
        _, rhythm = rng.choice(palette['cadence'])
    elif rng.random() < LEAD_FANCY_RHYTHM_RATE:
        _, rhythm = rng.choice(palette['fancy'])
    else:
        choices = palette['basic']
        # Do not let plain quarter-note division dominate once sixteenths are off.
        weights = []
        for name, durations in choices:
            weight = 0.75 if name.endswith('_2222') or name.endswith('_222') else 1.0
            weights.append(weight)
        weights = apply_category_frequency_multiplier(
            weights,
            [rhythm_is_syncopated(durations) for _name, durations in choices],
            LEAD_SYNCOPATED_BASIC_WEIGHT,
        )
        _, rhythm = rng.choices(choices, weights=weights, k=1)[0]
    return list(rhythm)
