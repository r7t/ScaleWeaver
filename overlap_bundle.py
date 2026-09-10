#!/usr/bin/env python3
"""Native O1/O2 precomputation and lookup for the phase-cancellation overlap objective."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

from scale_config import load_scale, resolved_overlap_objective
from spectral_bundle import TABLE_KEYS, colex_rank, resolve_domains
from spectral_model import resolve_parameters

FORMAT = "AdaptiveOverlap/native-colex-f64-1"
ALGORITHM = "PhaseCancellationEnergyOverlap/original-spectrum/root-normalized/O1-O2/v2"
METRICS = ("o1", "o2")
SOURCE = Path(__file__).with_name("overlap_precompute.cpp")
EXECUTABLE = Path(__file__).with_name("overlap_precompute_native")


def overlap_parameters(spec):
    """Use exactly the original SE/CSE/CCSE spectrum requested by the user."""
    p = resolve_parameters(spec.rules.get("spectral_parameters"))
    return {
        "sigma_hz": p["sigma_hz"],
        "q_per_100hz": p["q_per_100hz"],
        "model_base_freq_hz": p["model_base_freq_hz"],
        "max_freq_hz": p["max_freq_hz"],
        "resolution_hz": p["resolution_hz"],
        "phase_cancel_epsilon": 1e-15,
    }


def numerical_signature(spec):
    spec = load_scale(spec)
    wide, inner = resolve_domains(spec)
    payload = {
        "algorithm": ALGORITHM, "edo": spec.edo, "pcs": spec.pcs,
        "wide": wide, "inner": inner, "parameters": overlap_parameters(spec),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def bundle_paths(spec, directory="."):
    spec = load_scale(spec)
    stem = f"{spec.safe_id}_overlap_{numerical_signature(spec)}"
    directory = Path(directory)
    return directory / f"{stem}_manifest.json", {
        metric: directory / f"{stem}_{metric}_f64.bin" for metric in METRICS
    }


def _compile_backend(force=False):
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if force or not EXECUTABLE.is_file() or EXECUTABLE.stat().st_mtime < SOURCE.stat().st_mtime:
        command = [
            os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-march=native",
            "-fopenmp", "-DNDEBUG", str(SOURCE), "-o", str(EXECUTABLE),
        ]
        subprocess.run(command, check=True)
    return EXECUTABLE


def _reverse_percentiles(values):
    """Badness percentile: 0 is best/highest overlap, 1 is worst/lowest."""
    ordered = np.sort(values)
    lo = np.searchsorted(ordered, values, side="left")
    hi = np.searchsorted(ordered, values, side="right")
    return 1.0 - (lo + hi - 1) * 0.5 / max(1, len(values) - 1)


def _select_domain(values, source_steps, target_steps, cardinality):
    if source_steps == target_steps:
        return np.asarray(values)
    source_index = {step: i for i, step in enumerate(source_steps)}
    count = math.comb(len(target_steps) + cardinality - 1, cardinality)
    ranks = np.empty(count, dtype=np.int64)
    for combo in itertools.combinations_with_replacement(range(len(target_steps)), cardinality):
        rank = colex_rank(combo)
        source_combo = tuple(source_index[target_steps[i]] for i in combo)
        ranks[rank] = colex_rank(source_combo)
    return np.asarray(values)[ranks]


def build_bundle(spec, directory=".", *, workers=32, progress_every=5000,
                 force_compile=False):
    spec = load_scale(spec)
    workers = int(workers)
    if workers < 1 or progress_every < 0:
        raise ValueError("workers must be positive and progress_every nonnegative")
    wide, inner = resolve_domains(spec)
    manifest, binaries = bundle_paths(spec, directory)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    parameters = overlap_parameters(spec)
    executable = _compile_backend(force_compile)
    source_layout = [(2, wide), (3, wide), (4, wide), (5, inner)]
    raw_count = sum(math.comb(len(steps) + card - 1, card) for card, steps in source_layout)
    print(f"O1/O2 overlap bundle: {spec.id}, {spec.edo}-EDO; "
          f"pools {len(wide)}/{len(inner)}; workers={workers}", flush=True)

    with tempfile.TemporaryDirectory(dir=manifest.parent, prefix=".overlap-") as temp_name:
        temp = Path(temp_name)
        raw_paths = {metric: temp / f"raw_{metric}.bin" for metric in METRICS}
        command = [
            str(executable), "--edo", str(spec.edo),
            "--wide", ",".join(map(str, wide)), "--inner", ",".join(map(str, inner)),
            "--workers", str(workers), "--progress-every", str(progress_every),
            "--sigma", str(parameters["sigma_hz"]), "--q", str(parameters["q_per_100hz"]),
            "--base-freq", str(parameters["model_base_freq_hz"]),
            "--max-freq", str(parameters["max_freq_hz"]),
            "--resolution", str(parameters["resolution_hz"]),
            "--o1", str(raw_paths["o1"]), "--o2", str(raw_paths["o2"]),
        ]
        subprocess.run(command, check=True)
        expected_raw_bytes = raw_count * 8
        for path in raw_paths.values():
            if path.stat().st_size != expected_raw_bytes:
                raise OSError(f"Incomplete native overlap output: {path}")
        raw_maps = {metric: np.memmap(path, dtype="<f8", mode="r")
                    for metric, path in raw_paths.items()}
        source_slices = {}
        cursor = 0
        for card, steps in source_layout:
            count = math.comb(len(steps) + card - 1, card)
            source_slices[card] = (cursor, cursor + count, steps)
            cursor += count

        staged_paths = {metric: temp / binaries[metric].name for metric in METRICS}
        files = {metric: path.open("wb") for metric, path in staged_paths.items()}
        tables, offset_values = {}, 0
        try:
            for key in TABLE_KEYS:
                card = int(key[-1])
                target = wide if key.startswith("bass") else inner
                start, end, source_steps = source_slices[card]
                selected = {
                    metric: _select_domain(raw_maps[metric][start:end], source_steps, target, card)
                    for metric in METRICS
                }
                count = len(selected["o1"])
                for metric in METRICS:
                    rows = np.empty((count, 2), dtype="<f8")
                    rows[:, 0] = selected[metric]
                    rows[:, 1] = _reverse_percentiles(selected[metric])
                    payload = rows.tobytes()
                    if files[metric].write(payload) != len(payload):
                        raise OSError("Incomplete overlap table write")
                tables[key] = {
                    "cardinality": card, "steps": list(target), "count": count,
                    "offset_values": offset_values, "offset_bytes": offset_values * 8,
                }
                offset_values += count * 2
        finally:
            for file in files.values(): file.close()
            raw_maps.clear()

        data = {
            "format": FORMAT, "algorithm": ALGORITHM,
            "scale": spec.definition_dict(), "scale_signature": spec.signature,
            "numerical_signature": numerical_signature(spec), "parameters": parameters,
            "metrics": list(METRICS),
            "binary_files": {metric: path.name for metric, path in binaries.items()},
            "endianness": "little", "dtype": "float64", "row_width": 2,
            "row_semantics": ["overlap", "reverse_midrank_badness_percentile"],
            "score": "k * O1**o1_exponent * O2**o2_exponent; higher overlap is better",
            "pitch_reference": "lowest chord tone is transposed to model_base_freq_hz before spectrum construction; intervals are not octave-folded",
            "workers_requested": workers, "tables": tables,
            "total_values": offset_values, "total_bytes": offset_values * 8,
        }
        for metric, staged in staged_paths.items():
            if staged.stat().st_size != data["total_bytes"]:
                raise OSError(f"Overlap file size mismatch: {metric}")
            os.replace(staged, binaries[metric])
        staged_manifest = temp / manifest.name
        staged_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged_manifest, manifest)
    return manifest, binaries


def ensure_bundle(spec, directory=".", **kwargs):
    spec = load_scale(spec)
    manifest, _ = bundle_paths(spec, directory)
    try:
        with OverlapBundle(manifest, spec): pass
    except (OSError, ValueError, KeyError, TypeError):
        build_bundle(spec, directory, **kwargs)
    return manifest


class OverlapBundle:
    def __init__(self, manifest_path, spec=None):
        self.manifest_path = Path(manifest_path).resolve()
        self._files, self._maps, self._objective_sorted, self._composition_sorted = {}, {}, {}, {}
        try:
            self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            d = self.data
            if d.get("format") != FORMAT or d.get("algorithm") != ALGORITHM:
                raise ValueError("Not a compatible O1/O2 bundle")
            if d.get("row_width") != 2 or d.get("dtype") != "float64" or d.get("endianness") != "little":
                raise ValueError("Invalid overlap row layout")
            if spec is not None and d.get("numerical_signature") != numerical_signature(spec):
                raise ValueError("Overlap parameters or pitch domains mismatch")
            self.spec = load_scale(d["scale"])
            self.tables = d["tables"]
            if set(self.tables) != set(TABLE_KEYS):
                raise ValueError("Incomplete overlap table set")
            self._indices = {}
            offset = 0
            for key in TABLE_KEYS:
                table = self.tables[key]
                steps = tuple(table["steps"])
                card = int(key[-1])
                count = math.comb(len(steps) + card - 1, card)
                if (not steps or steps != tuple(sorted(set(steps))) or
                    table["cardinality"] != card or table["count"] != count or
                    table["offset_values"] != offset or table["offset_bytes"] != offset * 8):
                    raise ValueError(f"Invalid overlap table layout: {key}")
                self._indices[key] = {step: i for i, step in enumerate(steps)}
                offset += count * 2
            if d["total_values"] != offset or d["total_bytes"] != offset * 8:
                raise ValueError("Invalid overlap bundle size")
            for metric in METRICS:
                name = d["binary_files"][metric]
                if Path(name).name != name:
                    raise ValueError("Overlap binary filename must be a basename")
                path = self.manifest_path.parent / name
                if path.stat().st_size != d["total_bytes"]:
                    raise ValueError(f"Truncated {metric} overlap file")
                self._files[metric] = path.open("rb")
                self._maps[metric] = mmap.mmap(self._files[metric].fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    def close(self):
        self._objective_sorted.clear()
        self._composition_sorted.clear()
        for value in self._maps.values(): value.close()
        for value in self._files.values(): value.close()
        self._maps.clear(); self._files.clear()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def _location(self, steps, prefer_inner=False, table_key=None):
        pitches = tuple(sorted(map(int, steps)))
        card = len(pitches)
        if not 2 <= card <= 5: return None
        keys = ([table_key] if table_key else
                [f"inner{card}", f"bass{card}"] if prefer_inner else
                [f"bass{card}", f"inner{card}"])
        for key in keys:
            table = self.tables.get(key)
            if table is None or table["cardinality"] != card: continue
            index = self._indices[key]
            if any(pitch not in index for pitch in pitches): continue
            rank = colex_rank(tuple(index[pitch] for pitch in pitches))
            return key, table["offset_bytes"] + 16 * rank
        return None

    def entry(self, steps, *, metric="o1", prefer_inner=False, table_key=None):
        if metric not in METRICS: raise ValueError(f"Unknown overlap metric: {metric}")
        location = self._location(steps, prefer_inner, table_key)
        return None if location is None else struct.unpack_from("<dd", self._maps[metric], location[1])

    def metrics_entry(self, steps, *, prefer_inner=False, table_key=None):
        location = self._location(steps, prefer_inner, table_key)
        if location is None: return None
        return {metric: struct.unpack_from("<dd", self._maps[metric], location[1]) for metric in METRICS}

    def objective_array(self, table_key, k=1.0, o1_exponent=1.0, o2_exponent=2.0):
        table = self.tables[table_key]
        count, offset = int(table["count"]), int(table["offset_bytes"])
        arrays = {}
        for metric in METRICS:
            rows = np.frombuffer(self._maps[metric], dtype="<f8", count=count * 2, offset=offset)
            arrays[metric] = rows[0::2]
        return float(k) * np.power(arrays["o1"], float(o1_exponent)) * np.power(arrays["o2"], float(o2_exponent))

    def objective_entry(self, steps, k=1.0, o1_exponent=1.0, o2_exponent=2.0,
                        *, prefer_inner=False, table_key=None):
        location = self._location(steps, prefer_inner, table_key)
        if location is None: return None
        key, address = location
        o1 = struct.unpack_from("<d", self._maps["o1"], address)[0]
        o2 = struct.unpack_from("<d", self._maps["o2"], address)[0]
        k, a, b = float(k), float(o1_exponent), float(o2_exponent)
        score = k * o1 ** a * o2 ** b
        cache_key = (key, k, a, b)
        ordered = self._objective_sorted.get(cache_key)
        if ordered is None:
            ordered = np.sort(self.objective_array(key, k, a, b))
            self._objective_sorted[cache_key] = ordered
        lo = int(np.searchsorted(ordered, score, side="left"))
        hi = int(np.searchsorted(ordered, score, side="right"))
        rank = (lo + hi - 1) * 0.5 / max(1, len(ordered) - 1)
        badness = rank if k < 0.0 else 1.0 - rank
        return float(score), float(badness)

    def composition_array(self, table_key, spectral_bundle, *, k, o1_exponent,
                          o2_exponent, cse_a, cse_b, cse_c):
        """Lower-is-better signed overlap plus the restored spectral blend."""
        value = self.objective_array(table_key,k,o1_exponent,o2_exponent).copy()
        value += float(cse_a) * spectral_bundle.metric_array(table_key,'cse')
        value += float(cse_b) * spectral_bundle.metric_array(table_key,'se')
        value += float(cse_c) * spectral_bundle.metric_array(table_key,'ccse')
        return value

    def composition_entry(self, steps, spectral_bundle, *, k, o1_exponent,
                          o2_exponent, cse_a, cse_b, cse_c,
                          prefer_inner=False, table_key=None):
        location=self._location(steps,prefer_inner,table_key)
        if location is None:return None
        key,address=location
        o1=struct.unpack_from('<d',self._maps['o1'],address)[0]
        o2=struct.unpack_from('<d',self._maps['o2'],address)[0]
        spectral=spectral_bundle.metrics_entry(steps,table_key=key)
        if spectral is None:return None
        values=tuple(map(float,(k,o1_exponent,o2_exponent,cse_a,cse_b,cse_c)))
        k,a,b,ca,cb,cc=values
        score=k*o1**a*o2**b+ca*spectral['cse'][0]+cb*spectral['se'][0]+cc*spectral['ccse'][0]
        cache_key=(key,)+values
        ordered=self._composition_sorted.get(cache_key)
        if ordered is None:
            ordered=np.sort(self.composition_array(
                key,spectral_bundle,k=k,o1_exponent=a,o2_exponent=b,
                cse_a=ca,cse_b=cb,cse_c=cc))
            self._composition_sorted[cache_key]=ordered
        lo=int(np.searchsorted(ordered,score,side='left'))
        hi=int(np.searchsorted(ordered,score,side='right'))
        # The combined expression is an explicit low-is-good composition cost.
        badness=(lo+hi-1)*0.5/max(1,len(ordered)-1)
        return float(score),float(badness)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scale_json", nargs="?")
    parser.add_argument("--scale"); parser.add_argument("--rules"); parser.add_argument("--style")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args(argv)
    if not (args.scale or args.scale_json): parser.error("supply --scale or scale_json")
    spec = load_scale(args.scale or args.scale_json, rules=args.rules, style=args.style)
    kwargs = {"workers": args.workers, "progress_every": args.progress_every,
              "force_compile": args.force_compile}
    manifest = (build_bundle(spec, args.out_dir, **kwargs)[0] if args.force
                else ensure_bundle(spec, args.out_dir, **kwargs))
    print(manifest)


if __name__ == "__main__":
    main()
