#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export an adaptive exact-EDO-subset score JSON to MuseScore 3 (.mscx).

Every scale degree is written one diatonic staff position above the previous
one. The staff uses max(5, floor(note_count/2)+1) lines, so the original
ten-tone ScaleWeaver scale naturally remains a six-line staff while seven-note
scales use the ordinary five-line height. Playback tuning is computed directly
from freq = base_freq * 2**(step/edo); 186ed6 is not used.

Alongside the MSCX, a <scale name>.txt tuning file is created in the output
directory only when that file does not already exist.  Every staff also chooses
a readable octave display (8va/15ma/22ma or 8vb/15mb/22mb) automatically;
this is notation-only and never changes the exact EDO sounding pitch.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# Runtime tuning / notation configuration (filled from score["tuning"])
# ---------------------------------------------------------------------------
OCT_STEPS = 0
BASE_FREQ = 0.0
BASE_NOTE = ""
SCALE_NAME = "Adaptive scale"
SCALE_ID = "adaptive_scale"
PCS = (0,)
NAMES = ("1",)
PC_NAME = {0: "1"}
STAFF_LINES = 5


# diatonic letters per degree index, cycling E F G A B C D ...
NAT = ((4, 18), (5, 13), (7, 15), (9, 17), (11, 19), (0, 14), (2, 16))

# Staff-wide ottava choices.  They change notation only: note_xml subtracts the
# ottava from the written pitch while MuseScore adds it back during playback,
# so the exact EDO sounding pitch remains unchanged.
REGISTERS = (("none", 0, 0, "", 0),)

def configure_from_score(score):
    global OCT_STEPS, BASE_FREQ, BASE_NOTE, SCALE_NAME, SCALE_ID, NAT
    global PCS, NAMES, PC_NAME, STAFF_LINES, REGISTERS
    t = score.get("tuning", {})
    if str(t.get("system", "edo")).lower() != "edo":
        raise ValueError("adaptive exporter requires tuning.system='edo'")
    edo = int(t["edo"])
    pcs = tuple(int(x) for x in t.get("pcs", ()))
    names = tuple(str(x) for x in t.get("names", ()))
    if not pcs:
        rows = t.get("pitch_classes", ())
        pcs = tuple(int(r["step"]) for r in rows)
        names = tuple(str(r.get("name", i + 1)) for i, r in enumerate(rows))
    if len(pcs) < 3 or len(pcs) != len(names):
        raise ValueError("score tuning must contain matching pcs/names with at least 3 notes")
    if pcs[0] != 0 or tuple(sorted(pcs)) != pcs or any(x < 0 or x >= edo for x in pcs):
        raise ValueError("score pcs must be a sorted exact-EDO subset starting at 0")
    OCT_STEPS = edo
    PCS, NAMES = pcs, names
    PC_NAME = dict(zip(PCS, NAMES))
    BASE_FREQ = float(t["base_freq_hz"])
    BASE_NOTE = str(t["base_note"])
    match = re.fullmatch(r'([A-Ga-g])(-?\d+)', BASE_NOTE)
    if match is None: raise ValueError('base_note must be a natural note such as E4')
    letters=('C','D','E','F','G','A','B')
    anchor=letters.index(match[1].upper())
    natural=((0,14),(2,16),(4,18),(5,13),(7,15),(9,17),(11,19))
    NAT=natural[anchor:]+natural[:anchor]

    SCALE_NAME = str(t.get("scale_name", f"{edo}-EDO subset"))
    SCALE_ID = str(t.get("scale_id", f"edo{edo}"))
    STAFF_LINES = max(5, len(PCS) // 2 + 1)
    text = f"{BASE_NOTE}: {BASE_FREQ:g} Hz; exact {OCT_STEPS}-EDO; steps: " + " ".join(map(str, PCS))

    # One EDO-subset octave contains len(PCS) *notation degrees*, whereas a
    # conventional 8va semitone shift moves a natural-note spelling by seven
    # staff degrees. Compensate the difference using the configured note count.
    # This keeps the visual interval topology invariant under automatic ottava.
    extra_per_oct = 7 - len(PCS)
    def reg(subtype, semitone_shift, sounding_centre, label):
        octaves = int(semitone_shift // 12)
        diatonic = octaves * extra_per_oct
        return (subtype, semitone_shift, sounding_centre, label, diatonic)

    # centre_step is a *sounding* EDO-step centre.  compute_registers chooses
    # the nearest octave band for each actual voice, giving readable staves
    # without touching tuning.  MuseScore 3 supports these six ottava subtypes.
    REGISTERS = (
        reg("22ma",  36,  3 * OCT_STEPS, text + "; display 22ma"),
        reg("15ma",  24,  2 * OCT_STEPS, text + "; display 15ma"),
        reg("8va",   12,      OCT_STEPS, text + "; display 8va"),
        reg("none",   0,              0, text),
        reg("8vb",  -12,     -OCT_STEPS, text + "; display 8vb"),
        reg("15mb", -24, -2 * OCT_STEPS, text + "; display 15mb"),
        reg("22mb", -36, -3 * OCT_STEPS, text + "; display 22mb"),
    )

# Known voice families only affect display names and the default page order.
# The actual voice set is discovered dynamically from score["voices"].
VOICE_NAMES = {
    'bass': 'Bass', 'inner': 'Inner', 'counter': 'Counter', 'lead': 'Lead',
}
VOICE_SHORT = {
    'bass': 'Bs.', 'inner': 'In.', 'counter': 'Ct.', 'lead': 'Ld.',
}


def _voice_base_and_number(voice):
    """Return (base_name, optional numeric suffix), e.g. counter2 -> (counter, 2)."""
    m = re.fullmatch(r'(.+?)(\d+)?', str(voice))
    base = m.group(1) if m else str(voice)
    number = int(m.group(2)) if m and m.group(2) else None
    return base, number


def voice_labels(voice):
    """Human-readable long/short labels for both known and arbitrary voices."""
    voice = str(voice)
    base, number = _voice_base_and_number(voice)
    if base in VOICE_NAMES:
        long_name = VOICE_NAMES[base]
        short_name = VOICE_SHORT[base]
        if number is not None:
            long_name += f' {number}'
            short_name = short_name.rstrip('.') + f'{number}.'
        return long_name, short_name

    # Custom keys such as "alto_counter2" become "Alto Counter 2".
    long_name = base.replace('_', ' ').replace('-', ' ').strip().title() or voice
    if number is not None:
        long_name += f' {number}'
    words = [w for w in re.split(r'[^A-Za-z0-9]+', base) if w]
    if words:
        initials = ''.join(w[0].upper() for w in words[:3])
        short_name = initials + (str(number) if number is not None else '') + '.'
    else:
        short_name = (voice[:3] or 'V').title() + '.'
    return long_name, short_name


def discover_staff_order(score):
    """Return all score voice keys in a deterministic top-to-bottom staff order.

    An explicit score['staff_order'] or score['voice_layout']['staff_order'] is
    honoured first (unknown/missing names are ignored); any remaining voices are
    appended using the default family order:
        lead* -> counter* -> custom voices -> inner* -> bass*
    Numeric suffixes are ordered naturally for any custom voice names.
    """
    voices = score.get('voices')
    if not isinstance(voices, dict) or not voices:
        raise ValueError('score["voices"] must be a non-empty object/dict')
    keys = list(voices.keys())
    key_set = set(keys)

    explicit = score.get('staff_order')
    if explicit is None:
        explicit = score.get('voice_layout', {}).get('staff_order') if isinstance(score.get('voice_layout'), dict) else None
    ordered = []
    if isinstance(explicit, (list, tuple)):
        for v in explicit:
            if v in key_set and v not in ordered:
                ordered.append(v)

    insertion = {v: i for i, v in enumerate(keys)}

    def sort_key(v):
        base, number = _voice_base_and_number(v)
        n = 1 if number is None else number
        if base == 'lead':
            return (0, n, insertion[v])
        if base == 'counter':
            return (1, n, insertion[v])
        if base == 'inner':
            return (3, n, insertion[v])
        if base == 'bass':
            return (4, n, insertion[v])
        return (2, insertion[v], 0)

    ordered.extend(sorted((v for v in keys if v not in ordered), key=sort_key))
    return tuple(ordered)


def degree_index(step: int) -> int:
    """Scale-degree index for the current arbitrary EDO subset."""
    o, pc = divmod(int(step), OCT_STEPS)
    return o * len(PCS) + PCS.index(pc)


def central_notation(d: int):
    """Written 12-TET pitch (midi) + natural tpc of degree d in the central
    register, i.e. the configured base_note.  This is the register-independent staff position.
    """
    i = d % 7
    semi, tpc = NAT[i]
    base = re.fullmatch(r'([A-Ga-g])(-?\d+)', BASE_NOTE)
    anchor = ('C','D','E','F','G','A','B').index(base[1].upper())
    octv = int(base[2]) + (d + anchor) // 7
    return 12 * (octv + 1) + semi, tpc


# semitone pitch-class -> natural tpc
SEMI_TPC = {semi: tpc for semi, tpc in NAT}

# semitones to move to the next / previous natural letter from a natural class
_NEXT_LETTER = {0: 2, 2: 2, 4: 1, 5: 2, 7: 2, 9: 2, 11: 1}
_PREV_LETTER = {0: 1, 2: 2, 4: 2, 5: 1, 7: 2, 9: 2, 11: 2}


def shift_natural(midi, n):
    """Move a written MIDI pitch up/down by `n` natural letter steps, staying
    on natural letters (octaves wrap at B<->C).  `n` is the per-register
    diatonic shift (8vb=+3, 15mb=+6, 22mb=+9, 8va=-3, 15ma=-6, 22ma=-9)."""
    while n > 0:
        midi += _NEXT_LETTER[midi % 12]
        n -= 1
    while n < 0:
        midi -= _PREV_LETTER[midi % 12]
        n += 1
    return midi


def freq_exact_edo(step) -> float:
    return BASE_FREQ * 2.0 ** (float(step) / OCT_STEPS)


def tuning_cents(step) -> float:
    """Cents offset from the central-register written 12-TET pitch to the
    exact EDO frequency.  Register-independent by design: the written
    pitch (central_notation) plus the staff ottava always lands back on the
    central position, so this single value keeps playback exact in any
    register.
    """
    midi, _ = central_notation(degree_index(step))
    f12 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    return 1200.0 * math.log2(freq_exact_edo(step) / f12)


def choose_register(avg_step) -> tuple:
    """Pick a staff-wide ottava from the voice's real sounding centre."""
    return min(REGISTERS, key=lambda r: (abs(float(avg_step) - float(r[2])), abs(r[1])))


# ---------------------------------------------------------------------------
# Duration encoding (mscx)
# ---------------------------------------------------------------------------
NOTE_VALUES = (
    ('whole', 4.0), ('half', 2.0), ('quarter', 1.0), ('eighth', 0.5),
    ('16th', 0.25), ('32nd', 0.125), ('64th', 0.0625),
)
# triplet durations (beats) -> (durationType,)
TRIPLETS = {
    2.0 / 3.0: ('quarter',),
    1.0 / 3.0: ('eighth',),
    1.0 / 6.0: ('16th',),
}


def _duration_value(duration_type, dots=0):
    """Duration in quarter-note beats for a non-tuplet MSCX note."""
    base = dict(NOTE_VALUES).get(duration_type)
    if base is None:
        raise ValueError(f"unknown MSCX duration type {duration_type!r}")
    return float(base) * (2.0 - 0.5 ** int(dots))


def duration_pieces(d):
    """Return [(durationType, dots, is_triplet), ...] summing to d beats."""
    d = round(float(d), 6)
    for tv, (t,) in TRIPLETS.items():
        if abs(d - tv) < 1e-6:
            return [(t, 0, True)]
    out = []
    rem = d
    while rem > 1e-6:
        best = None
        for t, v in NOTE_VALUES:
            for dots in (0, 1, 2):
                val = v * (2 - 0.5 ** dots)
                if val <= rem + 1e-6 and (best is None or val > best[0]):
                    best = (val, t, dots)
        if best is None:
            raise ValueError(f'cannot represent duration {d} beats')
        out.append((best[1], best[2], False))
        rem = round(rem - best[0], 6)
    return out


# ---------------------------------------------------------------------------
# XML building
# ---------------------------------------------------------------------------
def _i(n):
    return '  ' * n


def _tie_spanner_xml(direction, location):
    """Return MuseScore 3's per-Note Tie spanner.

    MSCX does not use MusicXML's ``<tie type=.../>`` tag.  A tie is a
    Spanner attached to both Notes, and its endpoint is expressed relative to
    the current Note.  ``location`` is ``('measures', n)`` for a bar crossing
    or ``('fractions', ticks)`` for a local duration-piece split.
    """
    if direction not in ("next", "prev"):
        raise ValueError("tie direction must be next or prev")
    kind, amount = location
    if kind not in ("measures", "fractions") or int(amount) == 0:
        raise ValueError("invalid MSCX tie location")
    s = f'{_i(4)}<Spanner type="Tie">\n'
    if direction == "next":
        s += f'{_i(5)}<Tie>\n{_i(5)}</Tie>\n'
    s += f'{_i(5)}<{direction}>\n'
    s += f'{_i(6)}<location>\n'
    s += f'{_i(7)}<{kind}>{int(amount)}</{kind}>\n'
    s += f'{_i(6)}</location>\n'
    s += f'{_i(5)}</{direction}>\n'
    s += f'{_i(4)}</Spanner>\n'
    return s


def note_xml(step, tie_next=None, tie_prev=None, shift=0, diatonic=0):
    # Written pitch = central notation shifted by the register's octave offset,
    # then by the register's diatonic-letter shift so the notation sits on a
    # natural spelling near the actual pitch (small <tuning>).  The tuning is
    # reduced by the same interval, so playback (written + ottava + tuning)
    # stays exact in every register.
    midi, _tpc = central_notation(degree_index(step))
    current = midi - shift
    written = shift_natural(current, diatonic)
    tpc = SEMI_TPC[written % 12]
    # written - current is in semitones; subtract the same interval in cents
    # so playback (written + ottava + tuning) is unchanged by the notation shift.
    tun = tuning_cents(step) - (written - current) * 100.0
    s = f'{_i(3)}<Note>\n'
    s += f'{_i(4)}<pitch>{written}</pitch>\n'
    s += f'{_i(4)}<tpc>{tpc}</tpc>\n'
    s += f'{_i(4)}<tuning>{tun:.4f}</tuning>\n'
    if tie_next is not None:
        s += _tie_spanner_xml("next", tie_next)
    if tie_prev is not None:
        s += _tie_spanner_xml("prev", tie_prev)
    s += f'{_i(3)}</Note>\n'
    return s


def chord_piece_xml(t, dots, steps, shift, diatonic=0, tie_next=None, tie_prev=None):
    s = f'{_i(2)}<Chord>\n'
    if dots:
        s += f'{_i(3)}<dots>{dots}</dots>\n'
    s += f'{_i(3)}<durationType>{t}</durationType>\n'
    for step in steps:
        s += note_xml(step, tie_next, tie_prev, shift, diatonic)
    s += f'{_i(2)}</Chord>\n'
    return s


def chord_xml(dur, steps, shift=0, diatonic=0, tie_start=False, tie_stop=False):
    """One event of `dur` beats at `steps`.  Emits tied Chords when the
    duration is composite.  Triplets are NOT emitted here: the caller
    (build_bar) groups consecutive triplet chords into a <Tuplet>.  A bare
    <TimeModification> chord makes MuseScore count each triplet as a full
    note (the reported 5/4 bars), so we omit it and rely on the Tuplet.
    """
    pieces = duration_pieces(dur)
    n = len(pieces)
    out = []
    previous_ticks = None
    for k, (t, dots, _trip) in enumerate(pieces):
        # A duration decomposition introduces local ties.  A tie attached to
        # the input event itself crosses the enclosing bar, so its endpoint is
        # one Measure away.  MuseScore's fraction unit is its 480-tick quarter.
        piece_beats = _duration_value(t, dots)
        next_location = (('measures', 1) if tie_start and k == n - 1 else
                         ('fractions', round(piece_beats * 480)) if k < n - 1 else None)
        prev_location = (('measures', -1) if tie_stop and k == 0 else
                         ('fractions', -previous_ticks) if k > 0 else None)
        out.append(chord_piece_xml(t, dots, steps, shift, diatonic,
                                   next_location, prev_location))
        previous_ticks = round(piece_beats * 480)
    return ''.join(out)


def rest_piece_xml(t, dots=0):
    """One rest inside an already-declared tuplet or other explicit group."""
    s = f'{_i(2)}<Rest>\n'
    if dots:
        s += f'{_i(3)}<dots>{dots}</dots>\n'
    s += f'{_i(3)}<durationType>{t}</durationType>\n'
    s += f'{_i(2)}</Rest>\n'
    return s


def rest_xml(dur):
    pieces = duration_pieces(dur)
    out = []
    for t, dots, trip in pieces:
        s = f'{_i(2)}<Rest>\n'
        if dots:
            s += f'{_i(3)}<dots>{dots}</dots>\n'
        if trip:
            s += f'{_i(3)}<TimeModification>\n'
            s += f'{_i(4)}<actualNotes>3</actualNotes>\n'
            s += f'{_i(4)}<normalNotes>2</normalNotes>\n'
            s += f'{_i(3)}</TimeModification>\n'
        s += f'{_i(3)}<durationType>{t}</durationType>\n'
        s += f'{_i(2)}</Rest>\n'
        out.append(s)
    return ''.join(out)


def measure_rest_xml(bpb, sig_d):
    return (
        f'{_i(2)}<Rest>\n'
        f'{_i(3)}<durationType>measure</durationType>\n'
        f'{_i(3)}<duration>{bpb}/{sig_d}</duration>\n'
        f'{_i(2)}</Rest>\n'
    )


def build_bar(events, bar, bpb, shift=0, diatonic=0):
    """Build one measure, preserving complete 3:2 tuplets even through rests.

    Earlier versions only grouped consecutive *chords*.  A legal triplet pattern
    such as note-rest-note was therefore exported as two isolated triplet notes
    plus a rest, which MuseScore interpreted with the wrong measure duration.
    Here rests participate in the same three-item tuplet group.
    """
    grouped = {}
    for e in events:
        beat = round(float(e['start_beat']) - bar * bpb, 6)
        dur = round(float(e['duration_beats']), 6)
        grouped.setdefault((beat, dur, bool(e.get('tie_start')), bool(e.get('tie_stop'))), []).append(int(e['step']))

    if not grouped:
        return measure_rest_xml(bpb, 4)

    items = []          # (kind, dur, steps); kind in {'rest', 'chord'}
    cursor = 0.0
    for (beat, dur, tie_start, tie_stop), steps in sorted(grouped.items()):
        if beat > cursor + 1e-6:
            items.append(('rest', round(beat - cursor, 6), None))
        items.append(('chord', dur, (steps, tie_start, tie_stop)))
        cursor = beat + dur
    if bpb > cursor + 1e-6:
        items.append(('rest', round(bpb - cursor, 6), None))

    def triplet_piece(item):
        pieces = duration_pieces(item[1])
        return pieces[0] if len(pieces) == 1 and pieces[0][2] else None

    out = []
    i = 0
    while i < len(items):
        tp = triplet_piece(items[i])
        if tp is not None:
            # A simple explicit 3:2 group always consists of three equal notated
            # values.  Include rests as members, so note-rest-note stays one tuplet.
            group = items[i:i + 3]
            gps = [triplet_piece(x) for x in group]
            if len(group) == 3 and all(g is not None and g[0] == tp[0] and g[1] == tp[1] for g in gps):
                t, dots, _ = tp
                out.append(f'{_i(1)}<Tuplet>\n')
                out.append(f'{_i(2)}<normalNotes>2</normalNotes>\n')
                out.append(f'{_i(2)}<actualNotes>3</actualNotes>\n')
                out.append(f'{_i(2)}<baseNote>{t}</baseNote>\n')
                out.append(f'{_i(2)}<Number>\n')
                out.append(f'{_i(3)}<style>Tuplet</style>\n')
                out.append(f'{_i(3)}<text>3</text>\n')
                out.append(f'{_i(2)}</Number>\n')
                out.append(f'{_i(1)}</Tuplet>\n')
                for kind, _dur, steps in group:
                    if kind == 'rest':
                        out.append(rest_piece_xml(t, dots))
                    else:
                        pitches, tie_start, tie_stop = steps
                        out.append(chord_piece_xml(
                            t, dots, pitches, shift, diatonic,
                            ('measures', 1) if tie_start else None,
                            ('measures', -1) if tie_stop else None))
                out.append(f'{_i(1)}<endTuplet/>\n')
                i += 3
                continue

        kind, dur, steps = items[i]
        if kind == 'rest':
            out.append(rest_xml(dur))
        else:
            pitches, tie_start, tie_stop = steps
            out.append(chord_xml(dur, pitches, shift, diatonic, tie_start, tie_stop))
        i += 1
    return ''.join(out)


# ---------------------------------------------------------------------------
# Header fragments copied from the reference file so MuseScore 3.6 accepts it
# ---------------------------------------------------------------------------
ORDER_XML = '''  <Order id="orchestral" customized="1">
    <name>Orchestral</name>
    <instrument id="piano">
      <family id="keyboards">Keyboards</family>
      </instrument>
    <section id="woodwind" brackets="true" showSystemMarkings="true" barLineSpan="true" thinBrackets="true">
      <family>flutes</family>
      <family>oboes</family>
      <family>clarinets</family>
      <family>saxophones</family>
      <family>bassoons</family>
      <unsorted group="woodwinds"/>
      </section>
    <section id="brass" brackets="true" showSystemMarkings="false" barLineSpan="true" thinBrackets="true">
      <family>horns</family>
      <family>trumpets</family>
      <family>cornets</family>
      <family>flugelhorns</family>
      <family>trombones</family>
      <family>tubas</family>
      </section>
    <section id="timpani" brackets="true" showSystemMarkings="false" barLineSpan="true" thinBrackets="true">
      <family>timpani</family>
      </section>
    <section id="percussion" brackets="true" showSystemMarkings="false" barLineSpan="true" thinBrackets="true">
      <family>keyboard-percussion</family>
      <family>drums</family>
      <family>unpitched-metal-percussion</family>
      <family>unpitched-wooden-percussion</family>
      <family>other-percussion</family>
      </section>
    <family>keyboards</family>
    <family>harps</family>
    <family>organs</family>
    <family>synths</family>
    <section id="plucked-strings" brackets="true" showSystemMarkings="false" barLineSpan="true" thinBrackets="true">
      <family>plucked-strings</family>
      </section>
    <soloists/>
    <section id="voices" brackets="true" showSystemMarkings="false" barLineSpan="false" thinBrackets="true">
      <family>voices</family>
      </section>
    <section id="strings" brackets="true" showSystemMarkings="true" barLineSpan="true" thinBrackets="true">
      <family>orchestral-strings</family>
      </section>
    <unsorted/>
    </Order>
'''

META_TAGS = '''  <metaTag name="arranger"></metaTag>
  <metaTag name="composer"></metaTag>
  <metaTag name="copyright"></metaTag>
  <metaTag name="creationDate">{date}</metaTag>
  <metaTag name="lyricist"></metaTag>
  <metaTag name="movementNumber"></metaTag>
  <metaTag name="movementTitle"></metaTag>
  <metaTag name="platform">Linux</metaTag>
  <metaTag name="poet"></metaTag>
  <metaTag name="source"></metaTag>
  <metaTag name="translator"></metaTag>
  <metaTag name="workNumber"></metaTag>
  <metaTag name="workTitle">{title}</metaTag>
'''


def part_xml(staff_id, voice, register):
    subtype, shift, centre, text, diatonic = register
    name, short = voice_labels(voice)
    name = escape(name)
    short = escape(short)
    reg_note = '' if subtype == 'none' else f' ({subtype})'
    return f'''  <Part>
    <Staff id="{staff_id}">
      <StaffType group="pitched">
        <name>stdNormal</name>
        <lines>{STAFF_LINES}</lines>
        </StaffType>
      <bracket type="1" span="1" col="0"/>
      <barLineSpan>1</barLineSpan>
      </Staff>
    <trackName>{name}{reg_note}</trackName>
    <Instrument id="piano">
      <longName>{name}{reg_note}</longName>
      <shortName>{short}</shortName>
      <trackName>{name}{reg_note}</trackName>
      <minPitchP>21</minPitchP>
      <maxPitchP>108</maxPitchP>
      <minPitchA>21</minPitchA>
      <maxPitchA>108</maxPitchA>
      <instrumentId>keyboard.piano</instrumentId>
      <Channel>
        <program value="0"/>
        <synti>Fluid</synti>
        </Channel>
      </Instrument>
    </Part>
'''


def _frac(n, d):
    """Reduced fraction in MuseScore's explicit 'num/den' form (1/1, 3/4, ...).
    MuseScore requires the denominator for Ottava spanner locations: '1' is not
    accepted, so a whole-measure offset must be written '1/1'."""
    from math import gcd
    g = gcd(int(n), int(d))
    return f'{int(n) // g}/{int(d) // g}'


def first_measure_header(score, register, bpb, sig_n, sig_d, staff_id, with_tempo):
    subtype, shift, centre, text, diatonic = register
    s = ''
    s += f'{_i(1)}<Clef>\n'
    s += f'{_i(2)}<concertClefType>C2</concertClefType>\n'
    s += f'{_i(2)}<transposingClefType>C2</transposingClefType>\n'
    s += f'{_i(1)}</Clef>\n'
    s += f'{_i(1)}<KeySig>\n'
    s += f'{_i(2)}<accidental>0</accidental>\n'
    s += f'{_i(1)}</KeySig>\n'
    s += f'{_i(1)}<TimeSig>\n'
    s += f'{_i(2)}<sigN>{sig_n}</sigN>\n'
    s += f'{_i(2)}<sigD>{sig_d}</sigD>\n'
    s += f'{_i(1)}</TimeSig>\n'
    if staff_id == 1:
        s += f'{_i(1)}<SystemText>\n'
        s += f'{_i(2)}<text><b><font size="11"></font><font face="Noto Serif CJK SC Light"></font>{escape(SCALE_NAME)}</b></text>\n'
        s += f'{_i(1)}</SystemText>\n'
    s += f'{_i(1)}<StaffText>\n'
    s += f'{_i(2)}<placement>below</placement>\n'
    s += f'{_i(2)}<text><b><font face="Noto Serif CJK SC Light"/>{text}</b></text>\n'
    s += f'{_i(1)}</StaffText>\n'
    if staff_id == 1 and with_tempo:
        bpm = float(score['tempo_bpm'])
        s += f'{_i(1)}<Tempo>\n'
        s += f'{_i(2)}<tempo>{bpm / 60.0:.9f}</tempo>\n'
        s += f'{_i(2)}<followText>1</followText>\n'
        s += f'{_i(2)}<text>♩ = {int(bpm)}</text>\n'
        s += f'{_i(1)}</Tempo>\n'
    if subtype != 'none':
        bars = int(score['bars'])
        frac = _frac(bpb, 4)
        s += f'{_i(1)}<Spanner type="Ottava">\n'
        s += f'{_i(2)}<Ottava>\n'
        s += f'{_i(3)}<subtype>{subtype}</subtype>\n'
        s += f'{_i(2)}</Ottava>\n'
        s += f'{_i(2)}<next>\n'
        s += f'{_i(3)}<location>\n'
        s += f'{_i(4)}<measures>{bars - 1}</measures>\n'
        s += f'{_i(4)}<fractions>{frac}</fractions>\n'
        s += f'{_i(3)}</location>\n'
        s += f'{_i(2)}</next>\n'
        s += f'{_i(1)}</Spanner>\n'
    return s


def last_measure_footer(register, score, bpb):
    subtype, shift, centre, text, diatonic = register
    if subtype == 'none':
        return ''
    bars = int(score['bars'])
    frac = _frac(bpb, 4)
    s = f'{_i(1)}<Spanner type="Ottava">\n'
    s += f'{_i(2)}<prev>\n'
    s += f'{_i(3)}<location>\n'
    s += f'{_i(4)}<measures>{-(bars - 1)}</measures>\n'
    s += f'{_i(4)}<fractions>-{frac}</fractions>\n'
    s += f'{_i(3)}</location>\n'
    s += f'{_i(2)}</prev>\n'
    s += f'{_i(1)}</Spanner>\n'
    return s


def staff_music_xml(score, voice, events, register, staff_id, bpb, sig_n, sig_d):
    bars = int(score['bars'])
    shift = register[1]
    diatonic = register[4]
    by_bar = {}
    # MuseScore measures cannot contain an event extending past their own end.
    # Split a sustained IR event into tied notated fragments while preserving
    # one sounding pitch throughout the cross-bar hold.
    for e in events:
        start = float(e['start_beat'])
        end = start + float(e['duration_beats'])
        cursor = start
        while cursor < end - 1e-8:
            bar = int(cursor // bpb)
            bar_end = min(end, (bar + 1) * bpb)
            fragment = dict(e)
            fragment['start_beat'] = round(cursor, 6)
            fragment['duration_beats'] = round(bar_end - cursor, 6)
            fragment['tie_stop'] = cursor > start + 1e-8
            fragment['tie_start'] = bar_end < end - 1e-8
            by_bar.setdefault(bar, []).append(fragment)
            cursor = bar_end
    out = [f'<Staff id="{staff_id}">\n']
    for bar in range(bars):
        out.append(f'{_i(0)}<Measure>\n')
        out.append(f'{_i(1)}<voice>\n')
        if bar == 0:
            out.append(first_measure_header(
                score, register, bpb, sig_n, sig_d, staff_id, with_tempo=True))
        out.append(build_bar(by_bar.get(bar, []), bar, bpb, shift, diatonic))
        if bar == bars - 1:
            out.append(last_measure_footer(register, score, bpb))
        out.append(f'{_i(1)}</voice>\n')
        out.append(f'{_i(0)}</Measure>\n')
    out.append('</Staff>\n')
    return ''.join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def compute_registers(score, staff_order=None) -> dict:
    staff_order = tuple(staff_order or discover_staff_order(score))
    regs = {}
    for voice in staff_order:
        evs = score['voices'].get(voice, [])
        if not isinstance(evs, list):
            raise ValueError(f'score["voices"][{voice!r}] must be a list of events')
        if not evs:
            regs[voice] = REGISTERS[0]
            continue
        avg = sum(float(e['step']) for e in evs) / len(evs)
        regs[voice] = choose_register(avg)
    return regs


def generate(score, regs=None, staff_order=None, layout_mode='page') -> str:
    configure_from_score(score)
    staff_order = tuple(staff_order or discover_staff_order(score))
    if layout_mode not in ('page', 'system'):
        raise ValueError(f'unsupported MuseScore layout mode: {layout_mode!r}')
    bpb = float(score['beats_per_bar'])
    ts = str(score.get('time_signature', '4/4')).split('/', 1)
    sig_n, sig_d = int(ts[0]), int(ts[1])
    if regs is None:
        regs = compute_registers(score, staff_order)

    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.append('<museScore version="3.02">\n')
    out.append('  <programVersion>3.6.2</programVersion>\n')
    out.append('  <programRevision>5a2037e</programRevision>\n')
    out.append('  <Score>\n')
    out.append(f'    <layoutMode>{layout_mode}</layoutMode>\n')
    out.append('    <LayerTag id="0" tag="default"></LayerTag>\n')
    out.append('    <currentLayer>0</currentLayer>\n')
    out.append('    <Division>480</Division>\n')
    out.append('    <Style>\n')
    out.append('      <enableVerticalSpread>1</enableVerticalSpread>\n')
    # Explicit A4 page geometry makes MuseScore perform ordinary line wrapping
    # and page breaks instead of treating the score as one endless vertical canvas.
    # Values are inches, matching MuseScore 3.x .mscx style serialization.
    out.append('      <pageWidth>8.26772</pageWidth>\n')
    out.append('      <pageHeight>11.6929</pageHeight>\n')
    out.append('      <pagePrintableWidth>7.26772</pagePrintableWidth>\n')
    out.append('      <pageEvenLeftMargin>0.5</pageEvenLeftMargin>\n')
    out.append('      <pageOddLeftMargin>0.5</pageOddLeftMargin>\n')
    out.append('      <pageEvenTopMargin>0.5</pageEvenTopMargin>\n')
    out.append('      <pageOddTopMargin>0.5</pageOddTopMargin>\n')
    out.append('      <pageEvenBottomMargin>0.5</pageEvenBottomMargin>\n')
    out.append('      <pageOddBottomMargin>0.5</pageOddBottomMargin>\n')
    out.append('      <useStandardNoteNames>0</useStandardNoteNames>\n')
    out.append('      <Spatium>1.75</Spatium>\n')
    out.append('      </Style>\n')
    out.append('    <showInvisible>1</showInvisible>\n')
    out.append('    <showUnprintable>1</showUnprintable>\n')
    out.append('    <showFrames>1</showFrames>\n')
    out.append('    <showMargins>0</showMargins>\n')
    # The score header is the scale's actual configured display name only.
    visible_title = SCALE_NAME
    out.append(META_TAGS.format(
        date=date.today().isoformat(),
        title=visible_title))
    out.append(ORDER_XML)
    for i, voice in enumerate(staff_order, start=1):
        out.append(part_xml(i, voice, regs[voice]))
    for i, voice in enumerate(staff_order, start=1):
        out.append(staff_music_xml(
            score, voice, score['voices'].get(voice, []), regs[voice], i,
            bpb, sig_n, sig_d))
    out.append('  </Score>\n')
    out.append('</museScore>\n')
    return ''.join(out)


def _safe_filename(text):
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(text)).strip(" .")
    return text or "scale"

def _fmt_cent(x):
    if abs(x - round(x)) < 5e-10:
        return f"{int(round(x))}c"
    return f"{x:.5f}".rstrip("0").rstrip(".") + "c"

def write_tuning_file_if_missing(score, directory):
    configure_from_score(score)
    directory = Path(directory)
    path = directory / f"{_safe_filename(SCALE_NAME)}.txt"
    if path.exists():
        return path, False
    cents = [1200.0 * pc / OCT_STEPS for pc in PCS] + [1200.0]
    text = f"{BASE_NOTE}:{BASE_FREQ:g}\n" + " ".join(_fmt_cent(x) for x in cents) + "\n"
    path.write_text(text, encoding="utf-8")
    return path, True

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('score_json')
    ap.add_argument('out_mscx', nargs='?', default='adaptive_score.mscx')
    ap.add_argument(
        '--continuous', action='store_true',
        help='Use MuseScore continuous system layout instead of normal paged layout.'
    )
    args = ap.parse_args(argv)

    with open(args.score_json, encoding='utf-8') as f:
        score = json.load(f)
    configure_from_score(score)

    staff_order = discover_staff_order(score)
    regs = compute_registers(score, staff_order)
    layout_mode = 'system' if args.continuous else 'page'
    xml = generate(score, regs, staff_order, layout_mode=layout_mode)
    out_path = Path(args.out_mscx)
    out_path.write_text(xml, encoding='utf-8')
    tuning_path, created = write_tuning_file_if_missing(score, out_path.parent)
    print('Generated', args.out_mscx)
    print(('Generated tuning ' if created else 'Reused tuning ') + str(tuning_path))
    print(f'  voices={len(staff_order)} order={list(staff_order)}')
    print(f'  layout={layout_mode}')
    for i, voice in enumerate(staff_order, start=1):
        subtype, shift, centre, text, diatonic = regs[voice]
        print(f'  staff-{i} {voice}: '
              f'register={subtype} centre_step={centre} '
              f'diatonic={diatonic:+d} text="{text}"')
    return 0


if __name__ == '__main__':
    main()
