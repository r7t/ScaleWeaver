#!/usr/bin/env python3
"""Build SE, CSE and CCSE native tables together for any configured EDO subset."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import struct
import tempfile
import time
from contextlib import ExitStack

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from scale_config import load_scale
from spectral_model import (METRICS, ALGORITHM, resolve_parameters, fft_size,
                            prepare_notes, score_from_notes)

FORMAT = 'AdaptiveSpectral/native-colex-f64-1'
TABLE_KEYS = ('bass2','inner2','bass3','inner3','bass4','inner4','inner5')


def colex_rank(indices):
    return sum(math.comb(int(a)+i, i+1) for i, a in enumerate(indices))


def resolve_domains(spec):
    wide, inner = spec.resolved_cse_wide_bounds(), spec.resolved_cse_inner_bounds()
    five = spec.resolved_cse_five_bounds()
    wide = (min(wide[0], inner[0], five[0]), max(wide[1], inner[1], five[1]))
    wide, inner = spec.make_pool(*wide), spec.make_pool(*inner)
    if not wide or not inner: raise ValueError('Empty spectral pitch domain')
    return tuple(wide), tuple(inner)


def resolve_five_domain(spec):
    five = tuple(spec.make_pool(*spec.resolved_cse_five_bounds()))
    if not five: raise ValueError('Empty five-note spectral pitch domain')
    return five


def numerical_signature(spec, parameters=None):
    spec = load_scale(spec)
    p = resolve_parameters(spec.rules.get('spectral_parameters') if parameters is None else parameters)
    wide, inner = resolve_domains(spec); five = resolve_five_domain(spec)
    payload = dict(algorithm=ALGORITHM, edo=spec.edo, pcs=spec.pcs,
                   wide=wide, inner=inner, five=five, parameters=p)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def bundle_paths(spec, directory='.', parameters=None):
    spec = load_scale(spec)
    stem = f'{spec.safe_id}_spectral_{numerical_signature(spec, parameters)}'
    d = Path(directory)
    return d/f'{stem}_manifest.json', {m:d/f'{stem}_{m}_f64.bin' for m in METRICS}


def _same_parameters(left, right):
    """Compare normalized numerical model parameters, not cache filenames."""
    return resolve_parameters(left) == resolve_parameters(right)


def _bundle_covers_spec_data(data, spec, parameters=None):
    """Whether a completed bundle can safely serve the requested score.

    Spectral tables are valid on a *superset* of the requested pitch pools.
    Requiring the cached table's filename/signature to equal the current
    nominal domain needlessly rebuilt large pentad tables after a range or
    rescue-policy edit.
    """
    spec = load_scale(spec)
    stored_scale = data.get('scale', {})
    same_scale = (
        str(stored_scale.get('id')) == str(spec.id)
        and int(stored_scale.get('edo', -1)) == int(spec.edo)
        and tuple(map(int, stored_scale.get('pcs', ()))) == tuple(spec.pcs)
        and float(stored_scale.get('base_freq_hz', float('nan'))) == float(spec.base_freq_hz)
    )
    if (data.get('format') != FORMAT or data.get('algorithm') != ALGORITHM
            or not same_scale):
        return False
    try:
        wide, inner = resolve_domains(spec); five = resolve_five_domain(spec)
        tables = data['tables']
        for key in TABLE_KEYS:
            required = five if key == 'inner5' else wide if key.startswith('bass') else inner
            if not set(required).issubset(set(map(int, tables[key]['steps']))):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def percentiles(values):
    ordered = np.sort(values)
    lo = np.searchsorted(ordered, values, side='left')
    hi = np.searchsorted(ordered, values, side='right')
    return (lo+hi-1)*0.5/max(1, len(values)-1)


_NOTES = None


def _init_worker(notes):
    global _NOTES
    _NOTES = notes


def _score_chunk(chunk):
    return [score_from_notes(c, *_NOTES) for c in chunk]


def _chunks(combos, size):
    it = iter(combos)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk: return
        yield chunk


def _table_values(card, steps, notes, pool, chunk_size, progress_every):
    count = math.comb(len(steps)+card-1, card)
    values = np.empty((count, 3), dtype=np.float64)
    chunks = _chunks(itertools.combinations_with_replacement(steps, card), chunk_size)
    rows = (pool.imap(_score_chunk, chunks, chunksize=1) if pool else
            ([score_from_notes(c, *notes) for c in batch] for batch in chunks))
    cursor = 0
    started = time.monotonic()
    for batch in rows:
        old = cursor
        values[cursor:cursor+len(batch)] = batch
        cursor += len(batch)
        if progress_every and (cursor == count or cursor//progress_every != old//progress_every):
            rate = cursor/max(1e-9, time.monotonic()-started)
            print(f'SE/CSE/CCSE k={card}: {cursor:,}/{count:,} ({rate:,.0f} chords/s)', flush=True)
    if cursor != count or not np.isfinite(values).all():
        raise ValueError('Incomplete/non-finite spectral table')
    colex = np.empty_like(values)
    for i, combo in enumerate(itertools.combinations_with_replacement(range(len(steps)), card)):
        colex[colex_rank(combo)] = values[i]
    return colex


def build_bundle(spec, directory='.', *, parameters=None, workers=None,
                 chunk_size=256, progress_every=5000, start_method=None):
    spec = load_scale(spec)
    p = resolve_parameters(spec.rules.get('spectral_parameters') if parameters is None else parameters)
    if chunk_size < 1 or progress_every < 0: raise ValueError('Invalid progress/chunk size')
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    if workers < 1: raise ValueError('workers must be positive')
    wide, inner = resolve_domains(spec); five = resolve_five_domain(spec)
    manifest, binaries = bundle_paths(spec, directory, p)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    print(f'Unified spectral bundle: {spec.id}, {spec.edo}-EDO; pools {len(wide)}/{len(inner)}', flush=True)
    # These arrays are built only once and shared by all cardinalities/metrics.
    notes = prepare_notes(wide, spec.edo, p)
    pool = None
    tables, offset = {}, 0
    try:
        if workers > 1:
            ctx = mp.get_context(start_method or ('spawn' if os.name == 'nt' else 'fork'))
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(notes,))
        # Final manifest appears only after all three complete files exist.
        with tempfile.TemporaryDirectory(dir=manifest.parent, prefix='.spectral-') as tmp:
            tmp = Path(tmp)
            with ExitStack() as stack:
                files = {m:stack.enter_context((tmp/binaries[m].name).open('wb')) for m in METRICS}
                for card in (2,3,4,5):
                    steps = five if card == 5 else wide
                    values = _table_values(card, steps, notes, pool, chunk_size, progress_every)
                    rows = np.empty((len(values),3,2), dtype='<f8')
                    rows[:,:,0] = values
                    for j in range(3): rows[:,j,1] = percentiles(values[:,j])
                    domains = [(f'inner{card}',five)] if card == 5 else [(f'bass{card}',wide),(f'inner{card}',inner)]
                    for key, target in domains:
                        if target == steps:
                            selected = rows
                        else:
                            idx = {s:i for i,s in enumerate(steps)}
                            selected = np.empty((math.comb(len(target)+card-1,card),3,2),dtype='<f8')
                            for combo in itertools.combinations_with_replacement(range(len(target)),card):
                                selected[colex_rank(combo)] = rows[colex_rank(tuple(idx[target[i]] for i in combo))]
                        for j,m in enumerate(METRICS):
                            payload = selected[:,j,:].tobytes()
                            if files[m].write(payload) != len(payload): raise OSError('Incomplete spectral write')
                        tables[key] = dict(cardinality=card, steps=list(target), count=len(selected),
                                           offset_values=offset, offset_bytes=offset*8)
                        offset += len(selected)*2
            data = dict(format=FORMAT, algorithm=ALGORITHM, scale=spec.definition_dict(),
                        scale_signature=spec.signature, numerical_signature=numerical_signature(spec,p),
                        parameters=p, fft_size=fft_size(p['max_freq_hz'],p['resolution_hz']),
                        metrics=list(METRICS), binary_files={m:v.name for m,v in binaries.items()},
                        endianness='little', dtype='float64', row_width=2,
                        row_semantics=['baseline_corrected_entropy','metric_midrank_percentile'],
                        score='H(mixture) - mean(H(single_note)); powers SE=1,CSE=2,CCSE=4',
                        pitch_reference='absolute scale step 0 = model_base_freq_hz; no chord renormalization',
                        single_note_raw_entropies={str(s):dict(zip(METRICS,map(float,h))) for s,h in notes[2].items()},
                        tables=tables, total_values=offset, total_bytes=offset*8)
            for m in METRICS:
                path = tmp/binaries[m].name
                if path.stat().st_size != offset*8: raise OSError('Spectral file size mismatch')
                os.replace(path,binaries[m])
            staged = tmp/manifest.name
            staged.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            os.replace(staged,manifest)
    finally:
        if pool:
            pool.terminate()
            pool.join()
    return manifest, binaries


def ensure_bundle(spec, directory='.', **kwargs):
    spec = load_scale(spec)
    p = kwargs.get('parameters')
    manifest, _ = bundle_paths(spec,directory,p)
    try:
        with SpectralBundle(manifest,spec,p,allow_superset=True) as bundle:
            if _bundle_covers_spec_data(bundle.data, spec, p):
                return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # A precomputed 5-note table commonly has a wider register than a later
    # four-part run.  Reuse it when it covers the active pools instead of
    # rebuilding merely because its domain-derived filename differs.
    directory = Path(directory)
    choices = []
    for candidate in directory.glob(f'{spec.safe_id}_spectral_*_manifest.json'):
        try:
            with SpectralBundle(candidate, spec, p, allow_superset=True) as bundle:
                if _bundle_covers_spec_data(bundle.data, spec, p):
                    choices.append((int(bundle.data.get('total_bytes', 1 << 62)), candidate))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if choices:
        return min(choices, key=lambda row: row[0])[1]
    build_bundle(spec,directory,**kwargs)
    return manifest


class SpectralBundle:
    """Three read-only metric maps with seven cardinality/domain tables.

    ``entry`` and ``score`` read one metric; ``mixed_entry`` computes the
    configured raw blend and ranks it against the complete blended table.
    """
    def __init__(self, manifest_path, spec=None, parameters=None, *, allow_superset=False):
        self.manifest_path = Path(manifest_path).resolve()
        self._files, self._maps = {}, {}
        # Lazily materialised sorted distributions for the active linear blend.
        # A blend percentile is not a linear blend of the three metric
        # percentiles, so it must be ranked against every item in the same table.
        self._mixed_sorted = {}
        try:
            self.data = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            d = self.data
            if d.get('format') != FORMAT or d.get('algorithm') != ALGORITHM:
                raise ValueError('Not a unified spectral bundle; old SE/CSE caches cannot be reused')
            if d.get('row_width') != 2 or d.get('dtype') != 'float64' or d.get('endianness') != 'little':
                raise ValueError('Invalid spectral row layout')
            if (spec is not None and not allow_superset
                    and d.get('numerical_signature') != numerical_signature(spec,parameters)):
                raise ValueError('Spectral parameters or pitch domains mismatch')
            self.spec = load_scale(d['scale'])
            self.tables = d['tables']
            if set(self.tables) != set(TABLE_KEYS): raise ValueError('Incomplete spectral table set')
            self._indices = {}
            offset = 0
            for key in TABLE_KEYS:
                t = self.tables[key]
                steps = tuple(t['steps'])
                card = int(key[-1])
                count = math.comb(len(steps)+card-1,card)
                if (not steps or steps != tuple(sorted(set(steps))) or t['cardinality'] != card
                        or t['count'] != count or t['offset_bytes'] != offset*8 or t['offset_values'] != offset):
                    raise ValueError(f'Invalid spectral table layout: {key}')
                self._indices[key] = {s:i for i,s in enumerate(steps)}
                offset += count*2
            if d['total_bytes'] != offset*8 or d['total_values'] != offset:
                raise ValueError('Invalid spectral bundle size')
            actual = dict(algorithm=ALGORITHM, edo=self.spec.edo, pcs=self.spec.pcs,
                          wide=tuple(self.tables['bass2']['steps']),
                          inner=tuple(self.tables['inner5']['steps']),
                          five=tuple(self.tables['inner5']['steps']),
                          parameters=resolve_parameters(d['parameters']))
            actual['inner'] = tuple(self.tables['inner2']['steps'])
            signature = hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()[:24]
            if signature != d.get('numerical_signature'):
                raise ValueError('Spectral manifest signature does not match its contents')
            if spec is not None:
                active_spec=load_scale(spec);wide, inner = resolve_domains(active_spec);five=resolve_five_domain(active_spec)
                for key,t in self.tables.items():
                    expected=(five if key == 'inner5' else wide if key.startswith('bass') else inner)
                    actual_steps = tuple(t['steps'])
                    if not allow_superset and actual_steps != expected:
                        raise ValueError('Spectral manifest domain mismatch')
            for m in METRICS:
                name = d['binary_files'][m]
                if Path(name).name != name: raise ValueError('Binary filename must be a basename')
                path = self.manifest_path.parent/name
                if path.stat().st_size != d['total_bytes']: raise ValueError(f'Truncated {m} file')
                self._files[m] = path.open('rb')
                self._maps[m] = mmap.mmap(self._files[m].fileno(),0,access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    def close(self):
        self._mixed_sorted.clear()
        for value in self._maps.values(): value.close()
        for value in self._files.values(): value.close()
        self._maps.clear(); self._files.clear()

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

    def _location(self, steps, prefer_inner=False, table_key=None):
        xs = tuple(sorted(map(int,steps)))
        card = len(xs)
        if not 2 <= card <= 5: return None
        keys = ([table_key] if table_key else
                [f'inner{card}',f'bass{card}'] if prefer_inner else [f'bass{card}',f'inner{card}'])
        for key in keys:
            t = self.tables.get(key)
            if t is None or t['cardinality'] != card: continue
            idx = self._indices[key]
            if any(x not in idx for x in xs): continue
            rank = colex_rank(tuple(idx[x] for x in xs))
            return key, t['offset_bytes']+16*rank
        return None

    def _address(self, steps, prefer_inner=False, table_key=None):
        location = self._location(steps, prefer_inner, table_key)
        return None if location is None else location[1]

    def entry(self, steps, *, metric='cse', prefer_inner=False, table_key=None):
        if metric not in METRICS: raise ValueError(f'Unknown metric: {metric}')
        address = self._address(steps,prefer_inner,table_key)
        return None if address is None else struct.unpack_from('<dd',self._maps[metric],address)

    def metrics_entry(self, steps, *, prefer_inner=False, table_key=None):
        address = self._address(steps,prefer_inner,table_key)
        if address is None: return None
        return {m:struct.unpack_from('<dd',self._maps[m],address) for m in METRICS}

    def mixed_entry(self, steps, a=1., b=0., c=0., *, prefer_inner=False, table_key=None):
        location = self._location(steps,prefer_inner,table_key)
        if location is None: return None
        key, address = location
        row = {m:struct.unpack_from('<dd',self._maps[m],address) for m in METRICS}
        a,b,c = map(float,(a,b,c))
        raw = a*row['cse'][0]+b*row['se'][0]+c*row['ccse'][0]
        if (a,b,c) == (1.0,0.0,0.0):
            return raw,row['cse'][1]

        cache_key = (key,a,b,c)
        ordered = self._mixed_sorted.get(cache_key)
        if ordered is None:
            table = self.tables[key]
            count = int(table['count'])
            offset = int(table['offset_bytes'])
            raws = {}
            for metric in METRICS:
                rows = np.frombuffer(self._maps[metric],dtype='<f8',
                                     count=count*2,offset=offset)
                raws[metric] = rows[0::2]
            ordered = np.sort(a*raws['cse']+b*raws['se']+c*raws['ccse'])
            self._mixed_sorted[cache_key] = ordered
        lo = int(np.searchsorted(ordered,raw,side='left'))
        hi = int(np.searchsorted(ordered,raw,side='right'))
        percentile = (lo+hi-1)*0.5/max(1,len(ordered)-1)
        return raw,float(percentile)

    def score(self, steps, default=None, *, metric='cse', prefer_inner=False):
        row = self.entry(steps,metric=metric,prefer_inner=prefer_inner)
        return row[0] if row else default

    def percentile(self, steps, default=.5, *, metric='cse', prefer_inner=False):
        row = self.entry(steps,metric=metric,prefer_inner=prefer_inner)
        return row[1] if row else default


def export_json(manifest_path, directory=None):
    """Optional human-readable SE/CSE/CCSE files, without holding entries in RAM."""
    paths = {}
    with SpectralBundle(manifest_path) as bundle:
        directory = Path(directory or bundle.manifest_path.parent)
        directory.mkdir(parents=True,exist_ok=True)
        for metric in METRICS:
            path = directory/(bundle.manifest_path.stem.removesuffix('_manifest')+f'_{metric}.json')
            with path.open('w',encoding='utf-8') as f:
                header = {'format':'AdaptiveSpectral/JSON-1','metric':metric,
                          'numerical_signature':bundle.data['numerical_signature'],
                          'parameters':bundle.data['parameters'],'scale':bundle.data['scale'],
                          'single_note_raw_entropy':{s:h[metric] for s,h in bundle.data['single_note_raw_entropies'].items()}}
                f.write(json.dumps(header,ensure_ascii=False)[:-1]+',"tables":{')
                for i,key in enumerate(TABLE_KEYS):
                    t = bundle.tables[key]
                    if i: f.write(',')
                    f.write(json.dumps(key)+':{"entries":{')
                    for j,combo in enumerate(itertools.combinations_with_replacement(t['steps'],t['cardinality'])):
                        if j: f.write(',')
                        f.write(json.dumps(','.join(map(str,combo)))+':'+json.dumps(bundle.entry(combo,metric=metric,table_key=key)))
                    f.write('}}')
                f.write('}}\n')
            paths[metric] = path
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scale_json',nargs='?')
    ap.add_argument('--scale')
    ap.add_argument('--rules'); ap.add_argument('--style')
    ap.add_argument('--out-dir',default='.')
    ap.add_argument('--workers',type=int)
    ap.add_argument('--chunk-size',type=int,default=256)
    ap.add_argument('--progress-every',type=int,default=5000)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--export-json',action='store_true')
    a = ap.parse_args(argv)
    if not (a.scale or a.scale_json): ap.error('supply --scale or scale_json')
    spec = load_scale(a.scale or a.scale_json,rules=a.rules,style=a.style)
    kwargs = dict(workers=a.workers,chunk_size=a.chunk_size,progress_every=a.progress_every)
    manifest = (build_bundle(spec,a.out_dir,**kwargs)[0] if a.force else ensure_bundle(spec,a.out_dir,**kwargs))
    print(manifest)
    if a.export_json:
        for path in export_json(manifest).values(): print(path)


if __name__ == '__main__':
    mp.freeze_support()
    main()
