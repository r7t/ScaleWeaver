#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输出五音 sounding 的完整比例和五个 delete-one 四音子集比例。

用法：python print_attack_ratios.py SCORE_attacks.json
每行六列：完整五音比例，随后是依次删除由低到高第 0..4 个音所得的
五个独立 canonical 四音比例。末尾按完整五音比例统计质因子出现率。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMES = (2, 3, 5, 7, 11, 13, 17)


def _ratio_and_integers(canonical):
    if not isinstance(canonical, dict):
        return None, None
    integers = canonical.get("integers")
    parsed = None
    if isinstance(integers, list) and integers:
        try:
            parsed = tuple(int(x) for x in integers)
        except (TypeError, ValueError):
            parsed = None
    ratio = canonical.get("ratio_string")
    if not isinstance(ratio, str) or not ratio.strip():
        if not parsed:
            return None, None
        ratio = ":".join(str(x) for x in parsed)
    ratio = ratio.strip()
    if parsed is None:
        try:
            parsed = tuple(int(x.strip()) for x in ratio.split(":"))
        except (TypeError, ValueError):
            return None, None
    return ratio, parsed


def iter_six_ratio_records(data):
    attacks = data.get("attacks", [])
    if not isinstance(attacks, list):
        raise ValueError("JSON 中的 'attacks' 必须是列表")
    for attack in attacks:
        if not isinstance(attack, dict) or int(attack.get("sounding_note_count", 0)) != 5:
            continue
        full_ratio, full_ints = _ratio_and_integers(attack.get("canonical_ji"))
        if full_ratio is None or len(full_ints) != 5:
            continue
        subsets = attack.get("canonical_ji_four_note_subsets")
        if not isinstance(subsets, list) or len(subsets) != 5:
            continue
        by_deleted = {}
        for row in subsets:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("deleted_index"))
            except (TypeError, ValueError):
                continue
            ratio, integers = _ratio_and_integers(row.get("canonical_ji"))
            if ratio is not None and len(integers) == 4:
                by_deleted[idx] = (ratio, integers)
        if any(i not in by_deleted for i in range(5)):
            continue
        subset_ratios = tuple(by_deleted[i][0] for i in range(5))
        subset_ints = tuple(by_deleted[i][1] for i in range(5))
        yield (full_ratio, *subset_ratios), full_ints, subset_ints


# 兼容曾经导入旧函数名的外部脚本。
iter_five_ratio_records = iter_six_ratio_records


def prime_factor_coverage(records):
    """按完整 canonical 五整数比例统计每个质因子的出现率。"""
    counts = {p: 0 for p in PRIMES}
    total = 0
    for _ratios, full_integers, _subset_integer_sets in records:
        total += 1
        for p in PRIMES:
            if any(n % p == 0 for n in full_integers):
                counts[p] += 1
    return total, counts


def main():
    ap = argparse.ArgumentParser(
        description="输出完整五音 canonical 比例及五个 delete-one 四音子集比例"
    )
    ap.add_argument("json_file", type=Path, help="生成器输出的 *_attacks.json")
    args = ap.parse_args()
    if not args.json_file.is_file():
        raise SystemExit(f"文件不存在：{args.json_file}")
    try:
        data = json.loads(args.json_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON 解析失败：{exc}") from exc

    records = list(iter_six_ratio_records(data))
    for six_ratios, _full_ints, _subset_ints in records:
        print(six_ratios[0])
        #print("\t".join(six_ratios))

    print()
    print("Prime factor coverage (full five-note canonical ratio):")
    total, counts = prime_factor_coverage(records)
    if total == 0:
        print("没有可统计的五音 sounding canonical 构拟。")
        return 0
    for p in PRIMES:
        pct = 100.0 * counts[p] / total
        print(f"{p:<2} {pct:6.2f}%  ({counts[p]}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
