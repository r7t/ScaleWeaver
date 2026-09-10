"""Persistent optimizer CSE derived from an existing spectral bundle.

The expensive spectral SE/CSE/CCSE files are immutable inputs.  This module
builds a much cheaper secondary table containing the active linear metric blend
averaged with its worst raw subsets.  Changing only hierarchy weights therefore
rebuilds this derived table and never recomputes spectra.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

FORMAT = "ScaleWeaverWorstSubsetCSECache/1"
CARDINALITY_NAMES = {3: "triad", 4: "tetrad", 5: "pentad"}
SUBSET_WEIGHT_NAMES = {2: "worst_dyad", 3: "worst_triad", 4: "worst_tetrad"}
SOURCE_TABLES = {2: "bass2", 3: "bass3", 4: "bass4", 5: "inner5"}
METRICS = ("cse", "se", "ccse")


def _canonical_weights(weights):
    return {
        name: {key: float(row[key]) for key in sorted(row)}
        for name, row in sorted(weights.items())
    }


def cache_signature(bundle, blend_weights, hierarchy_weights):
    table_layout = {
        str(n): {
            "source": SOURCE_TABLES[n],
            "steps": list(map(int, bundle.tables[SOURCE_TABLES[n]]["steps"])),
            "count": int(bundle.tables[SOURCE_TABLES[n]]["count"]),
            "offset_bytes": int(bundle.tables[SOURCE_TABLES[n]]["offset_bytes"]),
        }
        for n in range(2, 6)
    }
    payload = {
        "format": FORMAT,
        "source_numerical_signature": bundle.data["numerical_signature"],
        "blend_weights": list(map(float, blend_weights)),
        "worst_subset_cse_weights": _canonical_weights(hierarchy_weights),
        "source_tables": table_layout,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24], payload


def cache_paths(bundle, blend_weights, hierarchy_weights):
    signature, payload = cache_signature(bundle, blend_weights, hierarchy_weights)
    stem = bundle.manifest_path.stem.removesuffix("_manifest")
    stem = f"{stem}_worst_subset_{signature}"
    directory = bundle.manifest_path.parent
    return directory / f"{stem}_manifest.json", directory / f"{stem}_f64.bin", payload


def _raw_blend_arrays(bundle, blend_weights):
    arrays = {}
    for n in range(2, 6):
        table = bundle.tables[SOURCE_TABLES[n]]
        count = int(table["count"])
        offset = int(table["offset_bytes"])
        values = np.zeros(count, dtype=np.float64)
        for metric, weight in zip(METRICS, blend_weights):
            weight = float(weight)
            if not weight:
                continue
            source = np.frombuffer(
                bundle._maps[metric], dtype="<f8", count=count * 2, offset=offset
            )
            values += weight * source[::2]
            del source
        arrays[n] = values
    return arrays


def _rank_terms(cardinality, pool_size):
    return np.array(
        [[math.comb(a + i, i + 1) for a in range(pool_size)]
         for i in range(cardinality)],
        dtype=np.int64,
    )


def _fill_hierarchical_table(target, n, steps, raw, hierarchy_weights,
                             pitch_indices, rank_terms, chunk_size):
    weights = hierarchy_weights[CARDINALITY_NAMES[n]]
    full_weight = float(weights["full"])
    total_weight = full_weight + sum(
        float(weights[SUBSET_WEIGHT_NAMES[k]]) for k in range(2, n)
    )
    if total_weight <= 0:
        raise ValueError(f"nonpositive hierarchical CSE weight total for {n} notes")

    iterator = itertools.combinations_with_replacement(range(len(steps)), n)
    completed = 0
    while True:
        batch = tuple(itertools.islice(iterator, int(chunk_size)))
        if not batch:
            break
        indices = np.asarray(batch, dtype=np.int16 if len(steps) < 32768 else np.int32)
        ranks = sum(rank_terms[n][i, indices[:, i]] for i in range(n))
        aggregate = full_weight * raw[n][ranks]

        for k in range(2, n):
            weight = float(weights[SUBSET_WEIGHT_NAMES[k]])
            if weight <= 0:
                continue
            index_map = np.asarray(
                [pitch_indices[k][int(p)] for p in steps], dtype=np.int32
            )
            worst = np.full(len(indices), -np.inf, dtype=np.float64)
            for kept in itertools.combinations(range(n), k):
                subset_indices = index_map[indices[:, kept]]
                subset_ranks = sum(
                    rank_terms[k][i, subset_indices[:, i]] for i in range(k)
                )
                np.maximum(worst, raw[k][subset_ranks], out=worst)
            aggregate += weight * worst

        target[ranks] = aggregate / total_weight
        completed += len(indices)
        if completed % max(int(chunk_size), 1) == 0:
            print(f"  derived worst-subset CSE k={n}: {completed:,}/{len(target):,}", flush=True)
    if completed != len(target):
        raise RuntimeError(f"derived CSE table k={n} incomplete: {completed}/{len(target)}")


def build_cache(bundle, blend_weights, hierarchy_weights, *, chunk_size=131072):
    manifest_path, binary_path, signature_payload = cache_paths(
        bundle, blend_weights, hierarchy_weights
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _raw_blend_arrays(bundle, blend_weights)
    steps = {
        n: tuple(map(int, bundle.tables[SOURCE_TABLES[n]]["steps"]))
        for n in range(2, 6)
    }
    pitch_indices = {n: {p: i for i, p in enumerate(steps[n])} for n in range(2, 6)}
    rank_terms = {n: _rank_terms(n, len(steps[n])) for n in range(2, 6)}

    sections = {"values": {}, "ordered": {}}
    cursor = 0
    for n in range(2, 6):
        count = len(raw[n])
        sections["values"][str(n)] = {"offset_values": cursor, "count": count}
        cursor += count
    for n in range(2, 5):
        count = len(raw[n])
        sections["ordered"][str(n)] = {"offset_values": cursor, "count": count}
        cursor += count

    fd, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=".worst-subset-", suffix=".bin"
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    mmap = None
    try:
        mmap = np.memmap(temporary_path, dtype="<f8", mode="w+", shape=(cursor,))
        for n in range(2, 6):
            row = sections["values"][str(n)]
            target = mmap[row["offset_values"]:row["offset_values"] + row["count"]]
            if n == 2:
                target[:] = raw[n]
            else:
                _fill_hierarchical_table(
                    target, n, steps[n], raw, hierarchy_weights,
                    pitch_indices, rank_terms, chunk_size,
                )
            # A memmap slice owns another reference to the same Windows file
            # mapping.  Do not leave the final loop's view alive at replace().
            del target
        for n in range(2, 5):
            source_row = sections["values"][str(n)]
            ordered_row = sections["ordered"][str(n)]
            source = mmap[source_row["offset_values"]:
                          source_row["offset_values"] + source_row["count"]]
            ordered = mmap[ordered_row["offset_values"]:
                           ordered_row["offset_values"] + ordered_row["count"]]
            ordered[:] = np.sort(source)
            del source, ordered
        mmap.flush()
        # del mmap alone does not reliably close numpy.memmap's OS handle on
        # Windows.  Explicitly close it before moving the temporary file.
        mmap._mmap.close()
        del mmap
        mmap = None
        expected_bytes = cursor * 8
        if temporary_path.stat().st_size != expected_bytes:
            raise OSError("derived CSE binary has an invalid size")
        # Cache filenames are content-addressed by both metric and subset
        # weights.  An existing file of the expected size is therefore the
        # same immutable cache, possibly created or mmap'ed by another process.
        # Reuse it instead of trying to replace a Windows-open target.
        if binary_path.is_file() and binary_path.stat().st_size == expected_bytes:
            temporary_path.unlink()
        else:
            try:
                os.replace(temporary_path, binary_path)
            except PermissionError:
                # Another builder may have won the race after the check above.
                if not (binary_path.is_file()
                        and binary_path.stat().st_size == expected_bytes):
                    raise
                temporary_path.unlink()

        data = {
            **signature_payload,
            "cache_signature": cache_signature(
                bundle, blend_weights, hierarchy_weights
            )[0],
            "binary_file": binary_path.name,
            "dtype": "float64",
            "endianness": "little",
            "aggregation": "weighted mean of full raw blend and maximum raw blend at each subset cardinality",
            "sections": sections,
            "total_values": cursor,
            "total_bytes": expected_bytes,
        }
        with tempfile.NamedTemporaryFile(
            dir=manifest_path.parent, prefix=".worst-subset-", suffix=".json",
            mode="w", encoding="utf-8", delete=False,
        ) as staged:
            json.dump(data, staged, ensure_ascii=False, indent=2)
            staged.write("\n")
            staged_path = Path(staged.name)
        os.replace(staged_path, manifest_path)
        print(f"Derived worst-subset CSE cache complete: {manifest_path.name}", flush=True)
        return manifest_path
    finally:
        if mmap is not None:
            try:
                mmap.flush()
            except (BufferError, OSError, ValueError):
                pass
            try:
                mmap._mmap.close()
            except (AttributeError, BufferError, OSError, ValueError):
                pass
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except PermissionError:
                # Preserve the original build exception.  On Windows, delayed
                # cleanup is preferable to masking it with a second exception.
                pass


def _compatible(data, binary_path, expected_payload):
    try:
        return (
            data.get("format") == FORMAT
            and all(data.get(key) == value for key, value in expected_payload.items())
            and binary_path.name == data.get("binary_file")
            and binary_path.is_file()
            and binary_path.stat().st_size == int(data.get("total_bytes", -1))
        )
    except (OSError, TypeError, ValueError):
        return False


class SubsetCSECache:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.binary_path = self.manifest_path.parent / self.data["binary_file"]
        self.values = {}
        self.ordered = {}
        for section_name, destination in (("values", self.values), ("ordered", self.ordered)):
            for n_text, row in self.data["sections"][section_name].items():
                destination[int(n_text)] = np.memmap(
                    self.binary_path,
                    dtype="<f8",
                    mode="r",
                    offset=int(row["offset_values"]) * 8,
                    shape=(int(row["count"]),),
                )


def ensure_cache(bundle, blend_weights, hierarchy_weights, *, chunk_size=131072):
    manifest_path, binary_path, expected_payload = cache_paths(
        bundle, blend_weights, hierarchy_weights
    )
    good = False
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            good = _compatible(data, binary_path, expected_payload)
        except (OSError, json.JSONDecodeError):
            good = False
    if not good:
        print("No matching worst-subset CSE cache; deriving it from the existing spectral bundle.", flush=True)
        build_cache(
            bundle, blend_weights, hierarchy_weights, chunk_size=chunk_size
        )
    return SubsetCSECache(manifest_path)
