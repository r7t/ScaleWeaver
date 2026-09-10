#!/usr/bin/env python3
"""Build adaptive SE0 binary tables directly, or compile precomputed JSON tables."""
import argparse
import multiprocessing as mp
from pathlib import Path
from scale_config import load_scale
from se_precompute import add_model_arguments, argument_parameters, table_filename
from sebundle import build_bundle, compile_json_bundle, ensure_bundle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(ap)
    ap.add_argument('--out-dir',default='.')
    ap.add_argument('--from-json',metavar='DIRECTORY',help='Compile the four JSON tables in this directory')
    for k in (2,3,4,5): ap.add_argument(f'--table{k}',help='Explicit source JSON (supply all four)')
    ap.add_argument('--workers',type=int)
    ap.add_argument('--chunk-size',type=int,default=256)
    ap.add_argument('--progress-every',type=int,default=5000)
    ap.add_argument('--force',action='store_true')
    args = ap.parse_args(argv)
    spec = load_scale(args.scale,rules=args.rules,style=args.style)
    parameters = argument_parameters(args,spec)
    explicit = {k:getattr(args,f'table{k}') for k in (2,3,4,5)}
    if args.from_json or any(explicit.values()):
        if not args.from_json and not all(explicit.values()): ap.error('Supply all of --table2/3/4/5, or --from-json')
        paths = {k:Path(explicit[k]) if explicit[k] else Path(args.from_json)/table_filename(spec,k) for k in explicit}
        manifest,_ = compile_json_bundle(spec,paths,args.out_dir,parameters=parameters)
    elif args.force:
        manifest,_ = build_bundle(spec,args.out_dir,parameters=parameters,workers=args.workers,
                                  chunk_size=args.chunk_size,progress_every=args.progress_every)
    else:
        manifest = ensure_bundle(spec,args.out_dir,parameters=parameters,workers=args.workers,
                                 chunk_size=args.chunk_size,progress_every=args.progress_every)
    print(manifest)


if __name__ == '__main__':
    mp.freeze_support()
    main()
