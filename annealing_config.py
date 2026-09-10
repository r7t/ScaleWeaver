"""JSON-only controls for fixed-melody whole-score harmony optimization."""
import copy
import math

DEFAULTS = {
    'enabled': True,
    # Deprecated compatibility input. It is retained so older style files
    # still load, but the optimizer now uses worst_subset_cse_weights below.
    'subset_correction': {'triad': 1/3, 'tetrad': 1/3, 'pentad': 1/3},
    'worst_subset_cse_weights': {
        'triad': {'full': 1.0, 'worst_dyad': 1.0},
        'tetrad': {'full': 1.0, 'worst_dyad': 1.0, 'worst_triad': 1.0},
        'pentad': {
            'full': 1.0,
            'worst_dyad': 1.0,
            'worst_triad': 1.0,
            'worst_tetrad': 1.0,
        },
    },
    'objective_weights': {'attack': 2.0, 'sounding': 6.0, 'background': 2.0, 'voice_leading': 0.5},
    'cse_loss_polynomials': {
        'two_note_attack': {'a4': 0.0, 'a3': 0.0, 'a2': 1.0, 'a1': -2.0, 'a0': 1.0},
        'three_note_attack': {'a4': 0.0, 'a3': 0.0, 'a2': 1.0, 'a1': -1.0, 'a0': 0.0},
        'four_note_attack': {'a4': 0.0, 'a3': 0.0, 'a2': 1.0, 'a1': 0.0, 'a0': 0.0},
        'five_note_attack': {'a4': 0.0, 'a3': 0.0, 'a2': 1.0, 'a1': 0.0, 'a0': 0.0},
        'sounding': {'a4': 0.0, 'a3': 0.0, 'a2': 1.0, 'a1': -1.0, 'a0': 0.0},
    },
    'attack': {'cardinality_weights': {'2': 1.0, '3': 1.0, '4': 1.0, '5': 1.0}},
    'background_voice_weights': {'counter': 0.58, 'inner2': 0.65, 'inner': 0.72, 'bass': 0.62},
    'voice_leading': {
        'motion_weight': 1.0, 'repeat_cost': 2.0,
        'large_leap_cost': 0.10,
        'ngram_weights': {'counter': 0.82, 'inner2': 0.77, 'inner': 0.72, 'bass': 0.62},
        'bass_ngram_weight': 1.0,
        'register_weights': {'counter': 0.020, 'inner2': 0.022, 'inner': 0.024, 'bass': 0.020},
    },
    'sounding_octave_excess_cost': 1.0,
    # Five voices may use more exact-octave-family reinforcement before the
    # anti-collapse penalty starts. Other texture sizes retain the old limit.
    'sounding_octave_family_pair_allowance': {'2': 1, '3': 1, '4': 1, '5': 3},
    'initialization': {
        'max_attempts': 24,
        'seed_stride': 1000003,
        'relax_melodic_jumps': True,
        'relaxed_jump_extra_cost': 12.0,
        'atomic_rescue': True,
        'atomic_rescue_lower_octaves': 1,
        'rescue_beam_width': 384,
        'expanded_range_extra_cost': 20.0,
    },
    'search': {
        'steps': 200000, 'start_temperature': 0.004, 'end_temperature': 0.00001,
        'move_size_weights': {'1': 0.60, '2': 0.30, '3': 0.10},
        'local_probability': 0.8, 'local_radius_degrees': 3,
        'quench_sweeps': 2, 'progress_every': 25000,
    },
}


def resolve_annealing(supplied=None):
    supplied=copy.deepcopy({} if supplied is None else supplied)
    # Accept old quadratic style rows and promote them to the quartic schema.
    # Obsolete percentile penalty weights are ignored deliberately.
    rows=supplied.get('cse_loss_polynomials',{})
    for name,row in rows.items():
        if isinstance(row,dict) and set(row)=={'a','b','c'}:
            rows[name]={'a4':0.0,'a3':0.0,
                        'a2':row['a'],'a1':row['b'],'a0':row['c']}
    attack=supplied.get('attack')
    if isinstance(attack,dict):
        attack.pop('percentile_excess_weight',None)
        attack.pop('percentile_excess_squared_weight',None)

    def merge(default, raw, path):
        if isinstance(default, dict):
            if not isinstance(raw, dict) or set(raw)-set(default):
                raise ValueError(f'Invalid annealing object/keys at {path}')
            return {k:merge(v,raw[k],path+'.'+k) if k in raw else copy.deepcopy(v)
                    for k,v in default.items()}
        if isinstance(default, bool):
            if type(raw) is not bool: raise ValueError(f'{path} must be boolean')
            return raw
        if path.startswith('annealing.cse_loss_polynomials.'):
            if isinstance(raw,bool) or not isinstance(raw,(int,float)) or not math.isfinite(raw):
                raise ValueError(f'{path} must be finite')
            return float(raw)
        if isinstance(default,int):
            if type(raw) is not int or raw < 0: raise ValueError(f'{path} must be a nonnegative integer')
        elif isinstance(raw,bool) or not isinstance(raw,(int,float)) or not math.isfinite(raw) or raw < 0:
            raise ValueError(f'{path} must be finite and nonnegative')
        return raw
    result=merge(DEFAULTS,supplied,'annealing')
    if any(v>1 for v in result['subset_correction'].values()):
        raise ValueError('subset correction must be in [0,1]')
    for name,weights in result['worst_subset_cse_weights'].items():
        if sum(float(v) for v in weights.values()) <= 0:
            raise ValueError(f'annealing.worst_subset_cse_weights.{name} must have positive total weight')
    s=result['search']
    if s['start_temperature']<=0 or s['end_temperature']<=0 or s['end_temperature']>s['start_temperature']:
        raise ValueError('require 0 < end_temperature <= start_temperature')
    if not 0<=s['local_probability']<=1 or sum(s['move_size_weights'].values())<=0:
        raise ValueError('invalid proposal probabilities')
    if not result['initialization']['max_attempts'] or not result['initialization']['seed_stride']:
        raise ValueError('initialization attempts/stride must be positive')
    return result
