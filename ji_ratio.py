#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patent-val canonical rational mappings for exact-EDO subset scales.

The active musical scale is an exact EDO subset.  Every 2/3/4/5-note sonority is
interpreted using the EDO's 17-limit patent val rather than by nearest-cents
fitting.

Rules
-----
* prime limit: 17 (2, 3, 5, 7, 11, 13, 17 only)
* mapping: exact mapping under the EDO's patent val
* no fixed integer limit
* admissibility: RMS tuning error <= one tenth of an EDO step plus 3.5 cents
  (including the 0-cent root error)
* primary objective: minimum largest integer after gcd reduction
* deterministic tie-break: lower integer sum, lower RMS error, lower maximum
  absolute error, then lexicographically smaller integer tuple

The search is unbounded in integer size: smooth integers are generated in
ascending order and the first maximum-integer level at which a valid mapping is
found is necessarily globally optimal for the primary objective.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from scale_config import ScaleSpec, load_scale

PRIME_LIMIT = 17
PRIMES = (2, 3, 5, 7, 11, 13, 17)
RMS_ERROR_STEP_FRACTION = 0.1
RMS_ERROR_OFFSET_CENTS = 3.5
CARDINALITIES = (2, 3, 4, 5)
FORMAT = "AdaptiveAttackCanonicalJI/5-patent-val"
OBJECTIVE = "exact patent-val mapping; minimum gcd-reduced maximum integer"


class PatentMappingError(RuntimeError):
    pass


def rms_error_limit_cents(edo: int) -> float:
    """Return 10% of one EDO step plus the fixed 3.5-cent allowance."""
    edo = int(edo)
    if edo <= 0:
        raise ValueError("edo must be positive")
    return 120.0 / float(edo) + RMS_ERROR_OFFSET_CENTS


def patent_val(edo: int) -> tuple[int, ...]:
    """Return the 17-limit patent val <N, round(N log2 3), ..., round(N log2 17)>."""
    edo = int(edo)
    return tuple(int(round(edo * math.log2(p))) for p in PRIMES)


@lru_cache(maxsize=None)
def _integer_factor_exponents(n: int) -> tuple[int, ...]:
    n = int(n)
    if n < 1:
        raise ValueError("integer must be positive")
    m = n
    exps = []
    for p in PRIMES:
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        exps.append(e)
    if m != 1:
        raise ValueError(f"{n} is not {PRIME_LIMIT}-limit smooth")
    return tuple(exps)


@lru_cache(maxsize=None)
def patent_integer_value(n: int, edo: int) -> int:
    exps = _integer_factor_exponents(int(n))
    val = patent_val(int(edo))
    return int(sum(e * v for e, v in zip(exps, val)))


def _smooth_integer_stream():
    """Yield all positive 17-smooth integers in strictly increasing order."""
    heap = [1]
    seen = {1}
    while True:
        x = heapq.heappop(heap)
        yield int(x)
        for p in PRIMES:
            y = int(x * p)
            if y not in seen:
                seen.add(y)
                heapq.heappush(heap, y)


def _gcd_many(xs: Iterable[int]) -> int:
    g = 0
    for x in xs:
        g = math.gcd(g, int(x))
    return max(1, g)


def _ratio_errors(edo: int, steps: tuple[int, ...], integers: tuple[int, ...]):
    root = int(integers[0])
    errors = [
        1200.0 * math.log2(float(n) / float(root))
        - 1200.0 * float(s) / float(edo)
        for s, n in zip(steps, integers)
    ]
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_abs = max(abs(e) for e in errors)
    return errors, float(rms), float(max_abs)


def _make_result(edo: int, steps: tuple[int, ...], integers: tuple[int, ...], *, source="precomputed") -> dict:
    cardinality = len(steps)
    root = integers[0]
    errors, rms, max_abs = _ratio_errors(edo, steps, integers)

    ratios = []
    for n in integers:
        g = math.gcd(root, n)
        ratios.append(f"{n // g}/{root // g}")

    pairwise = []
    for i in range(cardinality):
        for j in range(i + 1, cardinality):
            g = math.gcd(integers[i], integers[j])
            pairwise.append({
                "lower_index": i,
                "upper_index": j,
                "ratio": f"{integers[j] // g}/{integers[i] // g}",
                "error_cents": round(float(errors[j] - errors[i]), 6),
            })

    return {
        "cardinality": cardinality,
        "integers": list(map(int, integers)),
        "ratio_string": ":".join(str(int(x)) for x in integers),
        "ratios_from_lowest": ratios,
        "pairwise_ratios": pairwise,
        "error_cents_by_voice": [round(float(x), 6) for x in errors],
        "rms_error_cents": round(float(rms), 6),
        "max_abs_error_cents": round(float(max_abs), 6),
        "max_integer": int(max(integers)),
        "patent_steps": list(map(int, steps)),
        "source": source,
    }


def ji_bounds(spec: ScaleSpec) -> tuple[int, int]:
    ranges = spec.resolved_voice_ranges()
    return min(v[0] for v in ranges.values()), max(v[1] for v in ranges.values())


def ji_signature(spec: ScaleSpec) -> str:
    lo, hi = ji_bounds(spec)
    payload = {
        "edo": int(spec.edo),
        "pcs": list(map(int, spec.pcs)),
        "absolute_step_span": [int(lo), int(hi)],
        "prime_limit": PRIME_LIMIT,
        "patent_val": list(patent_val(spec.edo)),
        "max_rms_error_cents": rms_error_limit_cents(spec.edo),
        "objective": OBJECTIVE,
        "cardinalities": list(CARDINALITIES),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def table_path(spec: ScaleSpec, directory: str | Path = ".") -> Path:
    spec = load_scale(spec)
    return Path(directory).expanduser().resolve() / (
        f"{spec.safe_id}_attack_ji_17limit_patent_rms_dynamic_{ji_signature(spec)}.json"
    )


def possible_interval_tuples(spec: ScaleSpec, cardinality: int) -> tuple[tuple[int, ...], ...]:
    lo, hi = ji_bounds(spec)
    positions = spec.make_pool(lo, hi)
    out = set()
    for xs in itertools.combinations(positions, int(cardinality)):
        root = xs[0]
        out.add(tuple(int(x - root) for x in xs))
    return tuple(sorted(out))


def _validate_target_reachability(edo: int, targets: Iterable[tuple[int, ...]]):
    """Fail fast only when patent-val arithmetic makes an exact step impossible."""
    g = _gcd_many(patent_val(int(edo)))
    if g <= 1:
        return
    bad = sorted({int(s) for steps in targets for s in steps if int(s) % g != 0})
    if bad:
        raise PatentMappingError(
            f"{edo}-EDO 17-limit patent val has gcd {g}; exact patent mapping "
            f"cannot represent step offsets such as {bad[:8]!r}"
        )


def _fit_targets_patent(
    edo: int,
    targets: Iterable[tuple[int, ...]],
    cardinality: int,
    max_rms_error_cents: float | None = None,
):
    """Fit many targets with an unbounded minimum-max-integer patent search.

    Search proceeds by increasing 17-smooth maximum integer ``m``.  For a
    target ``(0,s1,...,sk)`` and fixed ``m``, exact patent mapping immediately
    fixes the root's patent value to ``V(m)-sk``; intermediate notes are then
    looked up in patent-value buckets.  This avoids enumerating the enormous
    set of irrelevant smooth-integer combinations.

    Once a target is found at maximum integer ``m``, every smaller possible
    maximum has already been exhausted, so the primary objective is globally
    solved.  All candidates at that same ``m`` are compared before resolving
    the target.
    """
    cardinality = int(cardinality)
    if max_rms_error_cents is None:
        max_rms_error_cents = rms_error_limit_cents(edo)
    targets = tuple(sorted(set(tuple(map(int, x)) for x in targets)))
    if not targets:
        return {}, 1
    if cardinality not in CARDINALITIES:
        raise ValueError("cardinality must be 2, 3, 4, or 5")
    for steps in targets:
        if len(steps) != cardinality or steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError(f"invalid strictly increasing target tuple: {steps!r}")

    _validate_target_reachability(edo, targets)
    unresolved = set(targets)
    resolved: dict[tuple[int, ...], tuple[int, ...]] = {}

    smooth_before: list[int] = []
    values: dict[int, int] = {}
    logs: dict[int, float] = {}
    buckets: dict[int, list[int]] = {}
    stream = _smooth_integer_stream()
    last_max = 1

    while unresolved:
        m = next(stream)
        vm = patent_integer_value(m, edo)
        values[m] = vm
        logs[m] = math.log2(m)
        last_max = m

        found_this_max: dict[tuple[int, ...], tuple[tuple, tuple[int, ...]]] = {}
        if len(smooth_before) >= cardinality - 1:
            # Every candidate at this layer has largest integer exactly m.
            # Iterate targets and use patent-value buckets to recover only
            # roots/intermediate integers that can map EXACTLY to that target.
            for steps in tuple(unresolved):
                root_patent = vm - int(steps[-1])
                roots = buckets.get(root_patent, ())
                if not roots:
                    continue

                for root in roots:
                    if root >= m:
                        continue
                    base_log = logs[root]

                    if cardinality == 2:
                        middle_products = ((),)
                    else:
                        middle_lists = []
                        possible = True
                        for step in steps[1:-1]:
                            vals = tuple(
                                n for n in buckets.get(values[root] + int(step), ())
                                if root < n < m
                            )
                            if not vals:
                                possible = False
                                break
                            middle_lists.append(vals)
                        if not possible:
                            continue
                        middle_products = itertools.product(*middle_lists)

                    for middle in middle_products:
                        ints = (root,) + tuple(middle) + (m,)
                        if any(a >= b for a, b in zip(ints, ints[1:])):
                            continue
                        if _gcd_many(ints) != 1:
                            continue
                        # Exact patent mapping is guaranteed by bucket lookup,
                        # but keep this assertion-like guard against logic drift.
                        mapped = (0,) + tuple(values[n] - values[root] for n in ints[1:])
                        if mapped != steps:
                            continue

                        target_cents = [1200.0 * s / float(edo) for s in steps]
                        errors = [
                            1200.0 * (logs[n] - base_log) - t
                            for n, t in zip(ints, target_cents)
                        ]
                        rms = math.sqrt(sum(e * e for e in errors) / cardinality)
                        if rms > float(max_rms_error_cents) + 1e-12:
                            continue
                        max_abs = max(abs(e) for e in errors)
                        key = (
                            sum(ints),
                            round(float(rms), 12),
                            round(float(max_abs), 12),
                            ints,
                        )
                        prev = found_this_max.get(steps)
                        if prev is None or key < prev[0]:
                            found_this_max[steps] = (key, ints)

            # Resolve only after every candidate sharing this maximum integer
            # has been tested, preserving deterministic tie-breaking.
            for steps, (_, ints) in found_this_max.items():
                resolved[steps] = ints
                unresolved.discard(steps)

        smooth_before.append(m)
        buckets.setdefault(vm, []).append(m)

        if m in (16, 32, 64, 128, 256, 512, 1024) and unresolved:
            print(
                f"    patent search k={cardinality}: max_integer={m}, "
                f"unresolved={len(unresolved):,}",
                flush=True,
            )

    out = {
        ",".join(map(str, steps)): _make_result(
            edo, steps, resolved[steps], source="patent-val-minmax"
        )
        for steps in targets
    }
    return out, int(last_max)

def _patent_table(spec: ScaleSpec, cardinality: int):
    targets = possible_interval_tuples(spec, cardinality)
    return _fit_targets_patent(spec.edo, targets, cardinality)


def _dyad_residue_table(spec: ScaleSpec) -> dict[str, dict]:
    residues = sorted({
        (int(b) - int(a)) % int(spec.edo)
        for a in spec.pcs for b in spec.pcs
        if int(a) != int(b)
    })
    targets = tuple((0, int(r)) for r in residues if int(r) != 0)
    rows, _ = _fit_targets_patent(spec.edo, targets, 2)
    out = {}
    for residue in residues:
        if residue == 0:
            continue
        item = rows.get(f"0,{residue}")
        if item is None:
            continue
        den, num = map(int, item["integers"])
        g = math.gcd(num, den)
        num //= g
        den //= g
        out[str(residue)] = {
            "numerator": int(num),
            "denominator": int(den),
            "ratio": f"{num}/{den}",
            "error_cents": round(
                1200.0 * math.log2(num / den)
                - 1200.0 * residue / float(spec.edo),
                6,
            ),
            "rms_error_cents": item["rms_error_cents"],
            "max_integer": item["max_integer"],
        }
    return out


def build_table(spec: ScaleSpec, directory: str | Path = ".") -> Path:
    spec = load_scale(spec)
    path = table_path(spec, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = ji_bounds(spec)
    pval = patent_val(spec.edo)
    rms_limit = rms_error_limit_cents(spec.edo)
    print(
        f"Building patent-val attack JI table for {spec.name}: {spec.edo}-EDO, "
        f"17-prime-limit, no integer limit, RMS<={rms_limit:g}c, "
        f"patent={pval}, span {lo}..{hi}",
        flush=True,
    )

    tables = {}
    entry_counts = {}
    search_maxima = {}
    for card in CARDINALITIES:
        rows, max_seen = _patent_table(spec, card)
        tables[str(card)] = rows
        entry_counts[str(card)] = len(rows)
        search_maxima[str(card)] = int(max_seen)
        print(
            f"  patent JI k={card}: {len(rows):,} interval tuples; "
            f"largest search integer used={max_seen}",
            flush=True,
        )

    dyads = _dyad_residue_table(spec)
    print(f"  patent JI octave-reduced dyad residues: {len(dyads):,}", flush=True)

    payload = {
        "format": FORMAT,
        "scale_signature": ji_signature(spec),
        "scale_id": spec.id,
        "scale_name": spec.name,
        "edo": int(spec.edo),
        "pitch_classes": list(map(int, spec.pcs)),
        "absolute_step_span": [int(lo), int(hi)],
        "prime_limit": PRIME_LIMIT,
        "primes": list(PRIMES),
        "patent_val": list(map(int, pval)),
        "integer_limit": None,
        "max_rms_error_cents": float(rms_limit),
        "objective": OBJECTIVE,
        "cardinalities": list(CARDINALITIES),
        "tie_break": [
            "sum_reduced_integers",
            "rms_error_cents",
            "max_abs_error_cents",
            "lexicographic_reduced_tuple",
        ],
        "cardinality_entries": entry_counts,
        "search_largest_integer_used": search_maxima,
        "dyad_source": "direct 17-limit patent-val fit, minimum maximum integer, RMS threshold",
        "dyad_residue": dyads,
        **tables,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Patent-val attack JI table complete: {path.name} "
        f"({path.stat().st_size/1024/1024:.2f} MiB)",
        flush=True,
    )
    return path


def _compatible(data: dict, spec: ScaleSpec) -> bool:
    lo, hi = ji_bounds(spec)
    return (
        data.get("format") == FORMAT
        and data.get("scale_signature") == ji_signature(spec)
        and int(data.get("edo", -1)) == int(spec.edo)
        and tuple(int(x) for x in data.get("pitch_classes", ())) == tuple(spec.pcs)
        and tuple(int(x) for x in data.get("absolute_step_span", ())) == (int(lo), int(hi))
        and int(data.get("prime_limit", -1)) == PRIME_LIMIT
        and data.get("integer_limit", "not-null") is None
        and abs(float(data.get("max_rms_error_cents", -1.0))
                - rms_error_limit_cents(spec.edo)) < 1e-12
        and tuple(int(x) for x in data.get("patent_val", ())) == patent_val(spec.edo)
        and tuple(int(x) for x in data.get("cardinalities", ())) == CARDINALITIES
    )


def ensure_table(spec: ScaleSpec, directory: str | Path = ".") -> Path:
    spec = load_scale(spec)
    path = table_path(spec, directory)
    good = False
    if path.is_file():
        try:
            good = _compatible(json.loads(path.read_text(encoding="utf-8")), spec)
        except Exception:
            good = False
    if not good:
        print(
            "No matching patent-val JI table found; generating it automatically on this first run.",
            flush=True,
        )
        path = build_table(spec, directory)
    return path


def ensure_loaded_table(spec: ScaleSpec, directory: str | Path = "."):
    """Return one validated JITable without parsing a large JSON twice."""
    spec = load_scale(spec)
    path = table_path(spec, directory)
    if path.is_file():
        try:
            return JITable(path, spec)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    print(
        "No matching patent-val JI table found; generating it automatically on this first run.",
        flush=True,
    )
    return JITable(build_table(spec, directory), spec)


class JITable:
    def __init__(self, path: str | Path, spec: ScaleSpec | None = None):
        self.path = Path(path).expanduser().resolve()
        # The generated JI JSON can be tens of megabytes.  Parsing it on every
        # composition dominated both front- and back-end timings, so retain a
        # local binary decode cache tied to the JSON's exact size/mtime.  The
        # JSON remains authoritative and changing it invalidates this cache.
        stat = self.path.stat()
        cache_path = self.path.with_suffix(self.path.suffix + ".decoded.pkl")
        stamp = (int(stat.st_size), int(stat.st_mtime_ns))
        self.data = None
        try:
            with cache_path.open("rb") as fh:
                cached_stamp, cached_data = pickle.load(fh)
            if tuple(cached_stamp) == stamp:
                self.data = cached_data
        except (OSError, EOFError, ValueError, TypeError, pickle.PickleError):
            pass
        if self.data is None:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, dir=cache_path.parent,
                    prefix=".ji-decoded-", suffix=".pkl"
                ) as fh:
                    temporary = Path(fh.name)
                    pickle.dump((stamp, self.data), fh, protocol=pickle.HIGHEST_PROTOCOL)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(temporary, cache_path)
            except OSError:
                if temporary is not None:
                    try: temporary.unlink()
                    except OSError: pass
        if self.data.get("format") != FORMAT:
            raise ValueError(f"unsupported JI table format: {self.data.get('format')!r}")
        self.edo = int(self.data["edo"])
        self.prime_limit = int(self.data["prime_limit"])
        self.integer_limit = None
        self.max_rms_error_cents = float(self.data["max_rms_error_cents"])
        self.patent_val = tuple(int(x) for x in self.data["patent_val"])
        if spec is not None and not _compatible(self.data, load_scale(spec)):
            raise ValueError("JI table does not match the active scale")

    def lookup(self, interval_steps: Iterable[int]) -> dict | None:
        steps = tuple(int(x) for x in interval_steps)
        if len(steps) not in CARDINALITIES:
            return None
        if not steps:
            return None
        # Normalize an arbitrary absolute tuple to offsets from its lowest tone.
        root_step = min(steps)
        offsets = tuple(sorted(int(x - root_step) for x in steps))
        key = ",".join(map(str, offsets))
        item = self.data.get(str(len(offsets)), {}).get(key)
        if item is not None:
            return json.loads(json.dumps(item, ensure_ascii=False))

        # For a wide dyad outside the precomputed voice span, octave-extend the
        # patent-derived residue mapping. This remains exact under the patent val
        # because prime 2 maps to exactly one EDO octave.
        if len(offsets) == 2:
            diff = int(offsets[1])
            octaves, residue = divmod(diff, self.edo)
            if residue:
                row = self.data.get("dyad_residue", {}).get(str(residue))
                if row is None:
                    return canonical_fallback(self.edo, offsets)
                num = int(row["numerator"]) * (2 ** octaves)
                den = int(row["denominator"])
            else:
                num = 2 ** octaves
                den = 1
            g = math.gcd(num, den)
            return _make_result(
                self.edo,
                offsets,
                (den // g, num // g),
                source="precomputed-json-dyad-octave-extension",
            )
        return canonical_fallback(self.edo, offsets)


@lru_cache(maxsize=32768)
def _canonical_fallback_cached(edo: int, steps: tuple[int, ...]):
    rows, _ = _fit_targets_patent(
        int(edo),
        (tuple(map(int, steps)),),
        len(steps),
        rms_error_limit_cents(int(edo)),
    )
    return rows.get(",".join(map(str, steps)))


def canonical_fallback(edo: int, interval_steps: Iterable[int]) -> dict | None:
    """Unbounded patent-val fallback for unusual out-of-table inputs."""
    steps = tuple(int(x) for x in interval_steps)
    card = len(steps)
    if card not in CARDINALITIES or not steps:
        return None
    if steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:])):
        # Repeated/unison tuples are intentionally not invented here; ordinary
        # generated attack tables contain distinct sounding pitches.
        return None
    item = _canonical_fallback_cached(int(edo), steps)
    return json.loads(json.dumps(item, ensure_ascii=False)) if item is not None else None


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            "Precompute adaptive canonical attack JI table using the active "
            "EDO's 17-limit patent val, no integer limit, "
            "RMS<=10% of one EDO step + 3.5c"
        )
    )
    ap.add_argument("scale_json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    spec = load_scale(args.scale_json)
    path = table_path(spec, args.out_dir)
    if args.force:
        path.unlink(missing_ok=True)
    print(ensure_table(spec, args.out_dir))


if __name__ == "__main__":
    main()
