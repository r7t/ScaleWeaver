"""Paired re-annealing of saved scores with the current TianGan daily style.

Both arms start from identical pitches, melody, rhythm and harmony. This is
an ablation of the balance term, not a reconstruction of historical runs.
"""
import argparse
import copy
import json
from pathlib import Path

import main
from annealing_config import resolve_annealing
from harmony_annealing import OptimizerMetric, optimize, bass_distribution


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores',nargs='+')
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--weight',type=float,default=12.0)
    parser.add_argument('--cse-dir',default='CSE_cache')
    parser.add_argument('--output',default='output/bass_balance_comparison.json')
    args=parser.parse_args()
    spec,_=main._configure_adaptive_scale(
        'scales/tiangan_72.json',args.cse_dir,
        rules_config='scales/tiangan_72_5.rules.json',
        style_config='scales/tiangan_72_5_norm.style.json')
    raw=copy.deepcopy(spec.style['annealing'])
    raw['search'].update(steps=args.steps,progress_every=10000,quench_sweeps=1)
    raw['bass_balance']['weight']=args.weight
    cfg=resolve_annealing(raw)
    metric=OptimizerMetric(spec,cfg)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in args.scores:
        source=json.loads(Path(path).read_text(encoding='utf-8'))
        definition=source['configuration']['definition']
        if definition['edo']!=spec.edo or set(definition['pcs'])!=set(spec.pcs):
            raise ValueError('Comparison requires a TianGan 72-EDO score')
        for weight in (0.0,args.weight):
            score=copy.deepcopy(source);config=copy.deepcopy(cfg)
            config['bass_balance']['weight']=weight
            report=optimize(score,metric,config)
            results.append(dict(source=str(path),seed=score['seed'],weight=weight,
                steps=args.steps,initial=bass_distribution(source,spec.edo),
                final=bass_distribution(score,spec.edo),
                components=report['final_components'],
                lead_unchanged=report['lead_unchanged'],
                rhythm_unchanged=report['rhythm_unchanged']))
            out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    run()
