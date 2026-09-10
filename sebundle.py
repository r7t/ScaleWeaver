"""Adaptive SE0 native colex bundles; direct parallel build or JSON compilation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from csebundle import colex_rank
from scale_config import load_scale
from se_model import resolve_parameters
from se_precompute import iter_tables, resolve_domains

FORMAT = 'AdaptiveSE/native-colex-f64-1'
ALGORITHM = 'SE0/frequency-scaled-sigma-area-compensation-1'
TABLE_KEYS = ('bass2','inner2','bass3','inner3','bass4','inner4','inner5')


def numerical_signature(spec, parameters=None, domains=None):
    spec = load_scale(spec)
    parameters = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
    wide, inner = domains or resolve_domains(spec)
    payload = {'algorithm':ALGORITHM, 'edo':spec.edo, 'pcs':list(spec.pcs),
               'wide_steps':list(wide), 'inner_steps':list(inner), 'parameters':parameters}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()[:24]


def bundle_paths(spec, directory='.', parameters=None):
    spec = load_scale(spec)
    directory = Path(directory)
    signature = numerical_signature(spec, parameters)
    return (directory/f'{spec.safe_id}_se_native_manifest.json',
            directory/f'{spec.safe_id}_se_{signature}_f64.bin')


def _colex_rows(values, percentiles, steps, card):
    rows = np.empty((len(values),2), dtype='<f8')
    for i, combo in enumerate(itertools.combinations_with_replacement(range(len(steps)),card)):
        rows[colex_rank(combo)] = values[i], percentiles[i]
    return rows


def _extract(rows, source, target, card):
    out = np.empty((math.comb(len(target)+card-1,card),2), dtype='<f8')
    source_idx = {s:i for i,s in enumerate(source)}
    for indices in itertools.combinations_with_replacement(range(len(target)),card):
        src = tuple(source_idx[target[i]] for i in indices)
        out[colex_rank(indices)] = rows[colex_rank(src)]
    return out


def compile_dense(entries, source_steps, target_steps, cardinality):
    """Validate every JSON key and produce the original colex [raw, percentile] layout."""
    source = tuple(source_steps)
    target = tuple(target_steps)
    if not target or source != tuple(sorted(set(source))) or target != tuple(sorted(set(target))):
        raise ValueError('SE pitch domains must be nonempty, sorted and unique')
    if not set(target).issubset(source): raise ValueError('SE target domain is not in source')
    expected = math.comb(len(source)+cardinality-1,cardinality)
    if len(entries) != expected: raise ValueError(f'SE entries: expected {expected}, got {len(entries)}')
    rows = np.full((math.comb(len(target)+cardinality-1,cardinality),2),np.nan,dtype='<f8')
    idx = {s:i for i,s in enumerate(target)}
    source_set = set(source)
    seen = set()
    for key, item in entries.items():
        pitches = tuple(int(x) for x in key.split(','))
        if len(pitches) != cardinality or pitches != tuple(sorted(pitches)) or not set(pitches).issubset(source_set):
            raise ValueError(f'Invalid SE combination: {key}')
        if pitches in seen: raise ValueError(f'Duplicate SE combination: {key}')
        seen.add(pitches)
        if len(item) != 2 or not all(math.isfinite(float(v)) for v in item) or not 0 <= float(item[1]) <= 1:
            raise ValueError(f'Invalid SE row: {key}')
        if all(s in idx for s in pitches): rows[colex_rank(tuple(idx[s] for s in pitches))] = item
    if not np.isfinite(rows).all(): raise ValueError('Incomplete SE native table')
    return rows


def _write_bundle(spec, directory, parameters, source_tables):
    wide, inner = resolve_domains(spec)
    manifest_path, binary_path = bundle_paths(spec,directory,parameters)
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    tables = {}
    offset = 0
    temporary = []
    try:
        # Start precompute workers before opening the output file, so forked
        # workers cannot inherit the bundle writer's file descriptor.
        first = next(source_tables)
        with tempfile.NamedTemporaryFile(dir=manifest_path.parent,suffix='.bin.tmp',delete=False) as bf:
            temporary.append(Path(bf.name))
            cards = []
            for card, steps, rows in itertools.chain((first,),source_tables):
                cards.append(card)
                expected_steps = inner if card == 5 else wide
                if tuple(steps) != expected_steps: raise ValueError(f'SE k={card}: domain differs from scale rules')
                count = math.comb(len(steps)+card-1,card)
                if rows.shape != (count,2) or not np.isfinite(rows).all(): raise ValueError('Invalid SE rows')
                domains = [(f'inner{card}',inner)] if card == 5 else [(f'bass{card}',wide),(f'inner{card}',inner)]
                for key, target in domains:
                    data = rows if target == tuple(steps) else _extract(rows,steps,target,card)
                    payload = np.asarray(data,dtype='<f8').tobytes()
                    if bf.write(payload) != len(payload): raise OSError('Incomplete SE binary write')
                    bf.flush()
                    tables[key] = {'cardinality':card,'steps':list(target),'count':len(data),
                                   'offset_values':offset,'offset_bytes':offset*8}
                    offset += data.size
            if cards != [2,3,4,5]: raise ValueError('SE bundle requires cards 2,3,4,5 in order')
        if temporary[0].stat().st_size != offset*8:
            raise OSError('SE binary size mismatch before publishing bundle')
        manifest = {'format':FORMAT,'algorithm':ALGORITHM,'scale':spec.definition_dict(),
                    'numerical_signature':numerical_signature(spec,parameters),
                    'parameters':parameters,'binary_file':binary_path.name,
                    'endianness':'little','dtype':'float64','row_width':2,
                    'value_semantics':['raw_se_without_convolution','percentile'],
                    'tables':tables,'total_values':offset,'total_bytes':offset*8}
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=manifest_path.parent,
                                         suffix='.json.tmp',delete=False) as mf:
            temporary.append(Path(mf.name))
            json.dump(manifest,mf,ensure_ascii=False,indent=2)
            mf.write('\n')
        os.replace(temporary[0],binary_path)
        os.replace(temporary[1],manifest_path)
    finally:
        for path in temporary: path.unlink(missing_ok=True)
        if hasattr(source_tables,'close'): source_tables.close()
    print(f'SE bundle complete: {manifest_path.name}',flush=True)
    return manifest_path,binary_path


def build_bundle(spec, directory='.', *, parameters=None, workers=None, chunk_size=256,
                 progress_every=5000, start_method=None):
    spec = load_scale(spec)
    parameters = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
    print(f'Building SE0 bundle: {spec.name}, {spec.edo}-EDO, {spec.note_count} notes/octave',flush=True)
    def sources():
        iterator = iter_tables(spec,parameters=parameters,workers=workers,chunk_size=chunk_size,
                               progress_every=progress_every,start_method=start_method)
        try:
            for card,steps,values,pct,_ in iterator:
                yield card,steps,_colex_rows(values,pct,steps,card)
        finally: iterator.close()
    return _write_bundle(spec,directory,parameters,sources())


def compile_json_bundle(spec, paths, directory='.', *, parameters=None):
    spec = load_scale(spec)
    parameters = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
    wide,inner = resolve_domains(spec)
    def sources():
        for card in (2,3,4,5):
            d = json.loads(Path(paths[card]).read_text(encoding='utf-8'))
            if d.get('format') != 'AdaptiveSE/absolute-combinations-1': raise ValueError('Unsupported SE JSON format; regenerate with se_precompute.py')
            tuning = d.get('tuning',{})
            if tuning.get('edo') != spec.edo or tuning.get('pcs') != list(spec.pcs): raise ValueError('SE JSON tuning mismatch')
            definition = d.get('scale',{})
            if definition.get('edo') != spec.edo or definition.get('pcs') != list(spec.pcs): raise ValueError('SE JSON scale mismatch')
            actual = d.get('parameters',{})
            if actual.get('cardinality') != card or actual.get('convolution') != 'none': raise ValueError('SE JSON algorithm/cardinality mismatch')
            if any(actual.get(k) != v for k,v in parameters.items()): raise ValueError('SE JSON model parameters mismatch')
            steps = tuple(tuning.get('se_steps',()))
            target = inner if card == 5 else wide
            if steps != target: raise ValueError('SE JSON domain differs from scale rules; use the same --rules')
            yield card,steps,compile_dense(d['entries'],steps,target,card)
    return _write_bundle(spec,directory,parameters,sources())


def ensure_bundle(spec, directory='.', *, parameters=None, **build_kwargs):
    spec = load_scale(spec)
    parameters = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
    manifest_path,_ = bundle_paths(spec,directory,parameters)
    try:
        with SEBundle(manifest_path,spec,parameters): pass
    except (OSError,ValueError,KeyError,TypeError,OverflowError):
        build_bundle(spec,directory,parameters=parameters,**build_kwargs)
    return manifest_path


class SEBundle:
    def __init__(self, manifest_path, spec=None, parameters=None):
        self._fh = self._mm = None
        self.manifest_path = Path(manifest_path).resolve()
        self.data = d = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if d.get('format') != FORMAT or d.get('algorithm') != ALGORITHM: raise ValueError('Unsupported SE bundle')
        if (d.get('endianness'),d.get('dtype'),d.get('row_width')) != ('little','float64',2): raise ValueError('Unsupported SE binary layout')
        self.parameters = resolve_parameters(d['parameters'])
        self.tables = d['tables']
        if set(self.tables) != set(TABLE_KEYS): raise ValueError('Incomplete SE bundle')
        self._indices = {}
        offset = 0
        for key in TABLE_KEYS:
            t = self.tables[key]
            card = int(key[-1]); steps = tuple(t['steps'])
            if not steps or any(type(s) is not int for s in steps) or steps != tuple(sorted(set(steps))): raise ValueError('Invalid SE steps')
            count = math.comb(len(steps)+card-1,card)
            if (t['cardinality'],t['count'],t['offset_values'],t['offset_bytes']) != (card,count,offset,offset*8): raise ValueError('Invalid SE table layout')
            self._indices[key] = {s:i for i,s in enumerate(steps)}
            offset += count*2
        if d.get('total_values') != offset or d.get('total_bytes') != offset*8: raise ValueError('Invalid SE bundle size')
        stored_spec = load_scale(d['scale'])
        wide = tuple(self.tables['bass2']['steps']); inner = tuple(self.tables['inner5']['steps'])
        if not set(inner).issubset(wide) or any(not stored_spec.is_scale_pitch(s) for s in wide): raise ValueError('Invalid SE pitch domain')
        for key,t in self.tables.items():
            if tuple(t['steps']) != (inner if key.startswith('inner') else wide): raise ValueError('Inconsistent SE domains')
        if d['numerical_signature'] != numerical_signature(stored_spec,self.parameters,(wide,inner)): raise ValueError('Invalid SE numerical signature')
        if spec is not None:
            spec = load_scale(spec)
            expected = resolve_parameters(spec.rules.get('se_parameters') if parameters is None else parameters)
            if d['numerical_signature'] != numerical_signature(spec,expected): raise ValueError('SE bundle tuning, domain or model is stale')
        self.bin_path = self.manifest_path.parent / d['binary_file']
        if self.bin_path.stat().st_size != offset*8: raise ValueError('Truncated SE binary')
        try:
            self._fh = self.bin_path.open('rb')
            self._mm = mmap.mmap(self._fh.fileno(),0,access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    def entry(self, steps, table_key):
        ss = tuple(sorted(map(int,steps)))
        t = self.tables.get(table_key)
        if t is None or len(ss) != t['cardinality']: return None
        idx = self._indices[table_key]
        if any(s not in idx for s in ss): return None
        rank = colex_rank(tuple(idx[s] for s in ss))
        return struct.unpack_from('<dd',self._mm,t['offset_bytes']+rank*16)

    def close(self):
        if self._mm is not None: self._mm.close(); self._mm = None
        if self._fh is not None: self._fh.close(); self._fh = None

    def __enter__(self): return self
    def __exit__(self,*exc): self.close()
