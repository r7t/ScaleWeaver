#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create default Lead anti-ABAB n-gram files for scale JSON files.

Scans the supplied directory (current directory by default). For each JSON:
- requires top-level "edo" and "pcs" to be treated as a scale config;
- reads "ngram_file" from the companion style profile;
- otherwise uses "<json_stem>.ngram.txt";
- never overwrites an existing n-gram file;
- creates, for every ordered pair of scale pitch classes A, B:
    lead 4 A,B,A,B -0.4
    lead 5 A,B,A,B,A -1.2

A and B are allowed to be equal.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from scale_config import load_scale


FOUR_GRAM_PENALTY = -0.4
FIVE_GRAM_PENALTY = -1.2


def load_scale_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] {path.name}: cannot read JSON ({exc})")
        return None

    if not isinstance(data, dict) or data.get("format") != "ScaleDefinition/1":
        print(f"[skip] {path.name}: not a scale JSON")
        return None

    pcs = data.get("pcs")
    if not isinstance(pcs, list) or not pcs:
        print(f"[skip] {path.name}: invalid or empty pcs")
        return None

    try:
        pcs = [int(x) for x in pcs]
    except (TypeError, ValueError):
        print(f"[skip] {path.name}: pcs must be integers")
        return None

    if len(set(pcs)) != len(pcs):
        print(f"[skip] {path.name}: duplicate pcs")
        return None

    return data, pcs


def ngram_path_for(json_path: Path, data: dict) -> Path:
    spec = load_scale(json_path)
    configured = spec.resolved_ngram_path()
    if configured is not None: return configured
    return json_path.with_name(json_path.stem.removesuffix('.scale') + '.ngram.txt')


def build_ngram_text(pcs: list[int]) -> str:
    lines = [
        "# Auto-generated default anti-repetition n-gram table.",
        "# Format: role  n  pc0,pc1,...  bias",
        "# For every ordered pair A,B:",
        f"#   Lead ABAB  -> {FOUR_GRAM_PENALTY}",
        f"#   Lead ABABA -> {FIVE_GRAM_PENALTY}",
        "",
    ]

    for a in pcs:
        for b in pcs:
            lines.append(
                f"lead 4 {a},{b},{a},{b} {FOUR_GRAM_PENALTY}"
            )

    for a in pcs:
        for b in pcs:
            lines.append(
                f"lead 5 {a},{b},{a},{b},{a} {FIVE_GRAM_PENALTY}"
            )

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".",
                        help="directory containing ScaleDefinition JSON files")
    args = parser.parse_args(argv)
    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    json_files = sorted(root.glob("*.json"))

    if not json_files:
        print("No JSON files found in current directory.")
        return 0

    created = 0
    existed = 0
    skipped = 0

    for json_path in json_files:
        loaded = load_scale_json(json_path)
        if loaded is None:
            skipped += 1
            continue

        data, pcs = loaded
        out_path = ngram_path_for(json_path, data)

        if out_path.exists():
            print(f"[exists] {json_path.name} -> {out_path.name}")
            existed += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(build_ngram_text(pcs), encoding="utf-8")

        n = len(pcs)
        print(
            f"[created] {json_path.name} -> {out_path.name} "
            f"({n*n} four-grams + {n*n} five-grams)"
        )
        created += 1

    print(
        f"\nDone: created={created}, existing={existed}, skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
