#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse scale-specific n-gram bias table loader.

Text format (one non-zero entry per line):
    <role> <n> <pc0,pc1,...> <bias>

``role`` is normally ``lead`` or ``bass``.  Bias follows the historical
ScaleWeaver convention: positive values are rewards and negative values are
penalties; score builders convert it to cost by subtracting the accumulated
bias.  Blank lines and ``#`` comments are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NGramTable:
    path: Path | None
    entries: dict[tuple[str, int, tuple[int, ...]], float]

    def bias(self, role: str, gram: tuple[int, ...]) -> float:
        gram = tuple(int(x) for x in gram)
        return float(self.entries.get((str(role), len(gram), gram), 0.0))

    def accumulated_bias(self, role: str, recent_pcs, cand_pc: int,
                         min_n: int = 2, max_n: int = 5) -> float:
        max_history = max(0, int(max_n) - 1)
        recent = tuple(int(x) for x in recent_pcs[-max_history:])
        cand = int(cand_pc)
        role = str(role)
        total = 0.0
        for n in range(int(min_n), int(max_n) + 1):
            if len(recent) >= n - 1:
                gram = recent[-(n - 1):] + (cand,)
                total += self.entries.get((role, n, gram), 0.0)
        return float(total)

    @property
    def nonzero_count(self) -> int:
        return len(self.entries)


EMPTY_NGRAM_TABLE = NGramTable(None, {})


def load_ngram_table(path: str | Path | None, *, edo: int | None = None,
                     allowed_pcs=None) -> NGramTable:
    if path is None or str(path).strip() == "":
        return EMPTY_NGRAM_TABLE
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"n-gram table not found: {p}")
    allowed = None if allowed_pcs is None else frozenset(int(x) for x in allowed_pcs)
    entries: dict[tuple[str, int, tuple[int, ...]], float] = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{p}:{lineno}: expected 4 fields: role n pcs bias")
        role, n_s, pcs_s, bias_s = parts
        n = int(n_s)
        gram = tuple(int(x) for x in pcs_s.split(",") if x != "")
        bias = float(bias_s)
        if n < 2 or n > 5:
            raise ValueError(f"{p}:{lineno}: n must be 2..5")
        if len(gram) != n:
            raise ValueError(f"{p}:{lineno}: n={n} but gram has {len(gram)} pcs")
        if bias == 0.0:
            raise ValueError(f"{p}:{lineno}: sparse table must not store zero entries")
        if edo is not None and any(not 0 <= x < int(edo) for x in gram):
            raise ValueError(f"{p}:{lineno}: pitch class outside 0..{int(edo)-1}")
        if allowed is not None and any(x not in allowed for x in gram):
            raise ValueError(f"{p}:{lineno}: gram contains pitch class outside the scale")
        key = (role, n, gram)
        if key in entries:
            raise ValueError(f"{p}:{lineno}: duplicate n-gram entry {key}")
        entries[key] = bias
    return NGramTable(p.resolve(), entries)
