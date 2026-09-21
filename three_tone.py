"""Local-ratio 3-tone classification, independent of progression heuristics.

Pitch/integer pairs must describe one supplied rational interpretation under
the EDO patent val. This module never treats a chord's arbitrary placeholder
integers as a rational interpretation. Completion keeps that interpretation;
the inferred pitch is retained even outside the scale and need not be played.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd

from ji_ratio import patent_integer_value


@dataclass(frozen=True)
class ThreeTone:
    status: str
    pc: int | None
    inferred_pc: int | None
    reason: str
    explicit_harmonic: bool = False

    def to_dict(self):
        return dict(asdict(self), in_chord=self.status == 'present',
                    in_scale=self.status in ('present', 'implied'))


def classify_three_tone(pcs, integers, edo, scale_pcs):
    """Return present / implied / outside_scale for a valid local ratio.

    Unclassified denotes unavailable/invalid interpretation, not a fourth
    harmonic class. Pitch membership is evaluated after the val mapping;
    tempered aliases count as present while explicit_harmonic distinguishes
    an actual integer of the form 3 * 2**k.

    GCD reduction followed by removal of powers of two makes direct detection
    invariant under common scaling, inversion, and octave doubling. The patent
    val maps ratios by their prime factors, not by rounding each interval.
    """
    if type(edo) is not int or edo <= 0:
        raise ValueError('edo must be a positive integer')
    pcs, integers = tuple(pcs), tuple(integers)
    if not pcs or len(pcs) != len(integers):
        raise ValueError('one integer is required for each pitch')
    if any(type(n) is not int or n <= 0 for n in integers):
        raise ValueError('ratio integers must be positive integers')
    if any(type(pc) is not int for pc in pcs):
        raise ValueError('pitches must be integer EDO steps')
    scale = {pc % edo for pc in scale_pcs}
    if not {pc % edo for pc in pcs} <= scale:
        raise ValueError('chord pitches must belong to the scale')
    common = reduce(gcd, integers)
    integers = tuple(n // common for n in integers)
    try:
        origins = {(pc - patent_integer_value(n, edo)) % edo
                   for pc, n in zip(pcs, integers)}
    except ValueError:
        return ThreeTone('unclassified', None, None, 'unsupported_prime')
    if len(origins) != 1:
        return ThreeTone('unclassified', None, None, 'inconsistent_patent_mapping')
    candidate = (next(iter(origins)) + patent_integer_value(3, edo)) % edo
    direct = {pc % edo for pc, n in zip(pcs, integers)
              if n // (n & -n) == 3}
    if direct:
        return ThreeTone('present', candidate, candidate, 'explicit_3_times_power_of_2', True)
    if candidate not in scale:
        return ThreeTone('outside_scale', candidate, candidate, 'completion_outside_scale')
    if candidate in {pc % edo for pc in pcs}:
        return ThreeTone('present', candidate, candidate, 'tempered_alias')
    return ThreeTone('implied', candidate, candidate, 'one_note_completion')


def tonic_relation(three_tone, tonic, edo):
    """Only identify unambiguous reference positions, not a full T/S/D theory."""
    if three_tone.pc is None:
        return 'unclassified'
    delta = (three_tone.pc - tonic) % edo
    fifth = patent_integer_value(3, edo) % edo
    if delta == fifth:
        return 'tonic_reference'
    if delta == 0:
        return 'subdominant_reference'
    if delta == (2 * fifth) % edo:
        return 'dominant_reference'
    if delta == (-fifth) % edo:
        return 'subdominant_extension'
    return 'other_colour'


def tiangan_audit():
    from harmony_rhythm import _CANONICAL_TIANGAN_CHORD_ROWS
    pcs = (0, 7, 16, 23, 30, 35, 42, 49, 58, 65)
    names = dict(zip(pcs, '甲乙丙丁戊己庚辛壬癸'))
    result = []
    for row in _CANONICAL_TIANGAN_CHORD_ROWS:
        tone = classify_three_tone(row[2], row[6], 72, pcs)
        result.append(dict(chord_id=row[0], chord_name=row[1], ratio=row[3],
                           three_tone=tone.to_dict(),
                           three_tone_name=names.get(tone.pc, '第%s步' % tone.pc),
                           tonic_relation=tonic_relation(tone, 0, 72)))
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(tiangan_audit(), ensure_ascii=False, indent=2))
