#!/usr/bin/env python3
"""Prebuild the weight-specific worst-subset optimizer CSE cache."""
from __future__ import annotations

import argparse

from annealing_config import resolve_annealing
from csebundle import CSEBundle, ensure_bundle
from scale_config import load_scale
from subset_cse_cache import ensure_cache


def main():
    parser = argparse.ArgumentParser(
        description="Derive optimizer subset CSE without recomputing SE/CSE/CCSE spectra"
    )
    parser.add_argument("scale_json")
    parser.add_argument("--rules")
    parser.add_argument("--style")
    parser.add_argument("--cache-dir", default=".")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--chunk-size", type=int, default=131072)
    args = parser.parse_args()

    spec = load_scale(
        args.scale_json,
        rules=args.rules,
        style=args.style,
        require_composition=True,
    )
    primary_manifest = ensure_bundle(
        spec, args.cache_dir, workers=args.cse_workers
    )
    config = resolve_annealing(spec.style.get("annealing"))
    cse = spec.style.get("cse_weights", {})
    blend_weights = tuple(
        float(cse.get(key, default))
        for key, default in (
            ("CSE_2D_A", 1.0),
            ("CSE_2D_B", 0.0),
            ("CSE_2D_C", 0.0),
        )
    )
    with CSEBundle(primary_manifest, spec) as bundle:
        cache = ensure_cache(
            bundle,
            blend_weights,
            config["worst_subset_cse_weights"],
            chunk_size=args.chunk_size,
        )
        print(cache.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
