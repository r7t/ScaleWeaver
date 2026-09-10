#!/usr/bin/env python3
"""Parallel SE0 JSON precomputation for any ScaleDefinition/1 EDO subset."""
from __future__ import annotations
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from scale_config import load_scale
from se_model import (cpp_buffer_size, note_spectrum_se0, entropy_se0,
                      score_from_spectra, spectrum_for, resolve_parameters)

_WORKER_NOTE_SPECS = None
_WORKER_SINGLE_ENTROPY = None
_WORKER_ZERO = None


def _init_worker(note_specs, single_entropy, n_bins):
    global _WORKER_NOTE_SPECS, _WORKER_SINGLE_ENTROPY, _WORKER_ZERO
    _WORKER_NOTE_SPECS = note_specs
    _WORKER_SINGLE_ENTROPY = single_entropy
    _WORKER_ZERO = np.zeros(n_bins, dtype=np.float64)


def _score_chunk(chunk):
    return [score_from_spectra(combo, _WORKER_NOTE_SPECS,
                               _WORKER_SINGLE_ENTROPY, _WORKER_ZERO)
            for combo in chunk]


def _batched(iterable, size):
    it = iter(iterable)
    while True:
        batch = tuple(itertools.islice(it, size))
        if not batch:
            return
        yield batch


def _stats(values: np.ndarray) -> dict:
    return {
        'count': int(values.size),
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(np.median(values)),
    }


def _percentiles(values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile, identical to the current CSE JSON convention."""
    sorted_values = np.sort(values)
    lo = np.searchsorted(sorted_values, values, side='left')
    hi = np.searchsorted(sorted_values, values, side='right')
    rank = 0.5 * (lo + hi - 1)
    return rank / max(1, len(values) - 1)


def _stream_table_json(path: Path, cardinality: int, steps: tuple[int, ...],
                       values: np.ndarray, percentiles: np.ndarray,
                       single_entropy: dict[int, float], spec, parameters, n_bins):
    bounds = [int(steps[0]), int(steps[-1])]
    metadata = {
        'format': 'AdaptiveSE/absolute-combinations-1',
        'scale': spec.definition_dict(),
        'description': (
            f'Every absolute {cardinality}-note scale combination-with-repetition '
            f'in pitch pool {bounds[0]}..{bounds[1]}. '
            'Entry value=[raw_se_without_convolution, percentile-within-this-cardinality].'
        ),
        'tuning': {
            'edo': spec.edo,
            'pcs': list(spec.pcs),
            'names': list(spec.names),
            'bounds': bounds,
            'cse_steps': list(steps),  # retained key for drop-in native builder compatibility
            'se_steps': list(steps),
        },
        'parameters': {
            **parameters,
            'spectrum_buffer_size': n_bins,
            'convolution': 'none',
            'score': 'mixture_entropy - mean(single_note_entropy)',
            'entropy_probability_threshold': 1e-12,
            'cardinality': int(cardinality),
            'pool_semantics': 'all scale pitches inside the absolute-step bounds',
        },
        'single_note_entropy': {
            str(step): round(float(single_entropy[step]), 12) for step in steps
        },
        'stats_by_size': {str(cardinality): _stats(values)},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        f.write('{')
        first = True
        for key, val in metadata.items():
            if not first:
                f.write(',')
            first = False
            f.write(json.dumps(key, ensure_ascii=False))
            f.write(':')
            f.write(json.dumps(val, ensure_ascii=False, separators=(',', ':')))
        f.write(',"entries":{')
        combo_iter = itertools.combinations_with_replacement(steps, cardinality)
        for i, (combo, se, pct) in enumerate(zip(combo_iter, values, percentiles)):
            if i:
                f.write(',')
            key = ','.join(map(str, combo))
            f.write(json.dumps(key))
            f.write(':[')
            f.write(format(round(float(se), 12), '.12g'))
            f.write(',')
            f.write(format(round(float(pct), 9), '.9g'))
            f.write(']')
        f.write('}}')


def score_cardinality(cardinality: int, steps: tuple[int, ...], pool,
                      note_specs, single_entropy, n_bins: int,
                      workers: int, chunk_size: int, progress_every: int) -> np.ndarray:
    count = math.comb(len(steps) + cardinality - 1, cardinality)
    values = np.empty(count, dtype=np.float64)
    cursor = 0
    t0 = time.time()
    combo_iter = itertools.combinations_with_replacement(steps, cardinality)

    def consume(rows):
        nonlocal cursor
        arr = np.asarray(rows, dtype=np.float64)
        values[cursor:cursor + len(arr)] = arr
        cursor += len(arr)
        crossed = (progress_every and
                   (cursor == count or cursor // progress_every !=
                    (cursor - len(arr)) // progress_every))
        if crossed:
            elapsed = max(1e-9, time.time() - t0)
            rate = cursor / elapsed
            remain = (count - cursor) / max(1e-9, rate)
            print(f'k={cardinality}: {cursor:,}/{count:,}  {100*cursor/count:6.2f}%  '
                  f'{rate:,.1f} sonorities/s  ETA {remain/60:.1f} min', flush=True)

    if workers <= 1:
        zero = np.zeros(n_bins, dtype=np.float64)
        for chunk in _batched(combo_iter, chunk_size):
            consume([score_from_spectra(c, note_specs, single_entropy, zero)
                     for c in chunk])
    else:
        chunks = _batched(combo_iter, chunk_size)
        for rows in pool.imap(_score_chunk, chunks, chunksize=1):
            consume(rows)

    if cursor != count:
        raise RuntimeError(f'k={cardinality}: expected {count} scores, got {cursor}')
    return values


def resolve_domains(spec, wide_bounds=None, inner_bounds=None):
    wide = tuple(wide_bounds or spec.resolved_cse_wide_bounds())
    inner = tuple(inner_bounds or spec.resolved_cse_inner_bounds())
    if wide[0] > wide[1] or inner[0] > inner[1]:
        raise ValueError('SE bounds must have min <= max')
    # Match CSE's expansion when its inner domain extends beyond the wide one.
    wide = (min(wide[0], inner[0]), max(wide[1], inner[1]))
    wide_steps, inner_steps = spec.make_pool(*wide), spec.make_pool(*inner)
    if not wide_steps or not inner_steps: raise ValueError('Empty SE pitch pool')
    return wide_steps, inner_steps


def iter_tables(spec, *, parameters=None, workers=None, chunk_size=256,
                progress_every=5000, wide_bounds=None, inner_bounds=None,
                start_method=None):
    """Yield (cardinality, pitch pool, raw values, percentiles, single entropies)."""
    spec = load_scale(spec)
    p = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
    wide, inner = resolve_domains(spec, wide_bounds, inner_bounds)
    if chunk_size < 1: raise ValueError('chunk_size must be positive')
    if progress_every < 0: raise ValueError('progress_every must be nonnegative')
    workers = max(1, int(workers if workers is not None else min(8, os.cpu_count() or 1)))
    n_bins = cpp_buffer_size(p['max_freq_hz'], p['resolution_hz'])
    specs = {s:spectrum_for(s, spec.edo, p) for s in wide}
    singles = {s:entropy_se0(v) for s,v in specs.items()}
    ctx = mp.get_context(start_method or ('spawn' if os.name == 'nt' else 'fork'))
    pool = None
    try:
        if workers > 1:
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(specs, singles, n_bins))
        for card, steps in ((2,wide),(3,wide),(4,wide),(5,inner)):
            values = score_cardinality(card, steps, pool, specs, singles, n_bins,
                                       workers, chunk_size, progress_every)
            yield card, steps, values, _percentiles(values), singles
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def table_filename(spec, cardinality):
    domain = 'inner' if cardinality == 5 else 'wide'
    return f'{spec.safe_id}_se_{domain}_{cardinality}note.json'


def add_model_arguments(ap):
    ap.add_argument('--scale', required=True, help='ScaleDefinition/1 JSON')
    ap.add_argument('--rules', help='CompositionRules/1 override')
    ap.add_argument('--style', help='CompositionStyle/1 override')
    ap.add_argument('--sigma0', type=float)
    ap.add_argument('--q', type=float)
    ap.add_argument('--base-freq', type=float, help='SE model reference frequency (default 100 Hz)')
    ap.add_argument('--max-freq', type=float)
    ap.add_argument('--resolution', type=float)


def argument_parameters(args, spec):
    p = dict(spec.rules.get('se_parameters', {}))
    for key, attr in (('sigma0_hz_at_100hz','sigma0'),('q_per_100hz','q'),
                      ('model_base_freq_hz','base_freq'),('max_freq_hz','max_freq'),
                      ('resolution_hz','resolution')):
        value = getattr(args, attr)
        if value is not None: p[key] = value
    return resolve_parameters(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(ap)
    for name in ('full-min','full-max','five-min','five-max'):
        ap.add_argument('--'+name, type=int)
    ap.add_argument('--out-dir', default='.')
    for k in (2,3,4,5): ap.add_argument(f'--output{k}')
    ap.add_argument('--workers', type=int)
    ap.add_argument('--chunk-size', type=int, default=256)
    ap.add_argument('--progress-every', type=int, default=5000)
    args = ap.parse_args(argv)
    spec = load_scale(args.scale, rules=args.rules, style=args.style)
    parameters = argument_parameters(args, spec)
    wide = list(spec.resolved_cse_wide_bounds())
    inner = list(spec.resolved_cse_inner_bounds())
    for bounds, lo, hi in ((wide,args.full_min,args.full_max),(inner,args.five_min,args.five_max)):
        if lo is not None: bounds[0] = lo
        if hi is not None: bounds[1] = hi
    n_bins = cpp_buffer_size(parameters['max_freq_hz'], parameters['resolution_hz'])
    for card, steps, values, pct, singles in iter_tables(spec, parameters=parameters,
            workers=args.workers, chunk_size=args.chunk_size, progress_every=args.progress_every,
            wide_bounds=wide, inner_bounds=inner):
        path = Path(args.out_dir) / (getattr(args,f'output{card}') or table_filename(spec,card))
        _stream_table_json(path, card, steps, values, pct, singles, spec, parameters, n_bins)
        print(f'wrote {path}: {len(values):,} rows', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()

