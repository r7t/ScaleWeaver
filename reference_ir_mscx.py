"""Canonical MSCX serialization for ScaleWeaverMSCXReferenceIR/1."""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

from imitation_frontend import REFERENCE_FORMAT, validate_reference_ir
from measure_timeline import locate
from mscx_export import _duration_value, _frac, _tie_spanner_xml, duration_pieces
from mscx_roundtrip import canonicalize_mscx, reusable_roundtrip_xml


_TPC = {0: 14, 1: 21, 2: 16, 3: 23, 4: 18, 5: 13,
        6: 20, 7: 15, 8: 22, 9: 17, 10: 24, 11: 19}


def _note(pitch, tie_next=None, tie_prev=None):
    text = "      <Note>\n"
    text += f"        <pitch>{int(pitch)}</pitch>\n"
    text += f"        <tpc>{_TPC[int(pitch) % 12]}</tpc>\n"
    if tie_next is not None:
        text += _tie_spanner_xml("next", tie_next)
    if tie_prev is not None:
        text += _tie_spanner_xml("prev", tie_prev)
    text += "      </Note>\n"
    return text


def _time_modification():
    return (
        "      <TimeModification>\n"
        "        <actualNotes>3</actualNotes>\n"
        "        <normalNotes>2</normalNotes>\n"
        "      </TimeModification>\n")


def _chord_piece(kind, dots, pitch, tie_next=None, tie_prev=None, triplet=False):
    text = "    <Chord>\n"
    if dots:
        text += f"      <dots>{int(dots)}</dots>\n"
    if triplet:
        text += _time_modification()
    text += f"      <durationType>{kind}</durationType>\n"
    text += _note(pitch, tie_next, tie_prev)
    text += "    </Chord>\n"
    return text


def _chord(duration, pitch, tie_start=False, tie_stop=False):
    pieces = duration_pieces(float(duration))
    out = []
    previous_ticks = None
    for index, (kind, dots, triplet) in enumerate(pieces):
        beats = _duration_value(kind, dots)
        tie_next = (("measures", 1) if tie_start and index == len(pieces) - 1
                    else ("fractions", round(beats * 480)) if index < len(pieces) - 1
                    else None)
        tie_prev = (("measures", -1) if tie_stop and index == 0
                    else ("fractions", -previous_ticks) if index > 0 else None)
        out.append(_chord_piece(
            kind, dots, pitch, tie_next, tie_prev, triplet=triplet))
        previous_ticks = round(beats * 480)
    return "".join(out)


def _rest_piece(kind, dots, triplet=False):
    text = "    <Rest>\n"
    if dots:
        text += f"      <dots>{int(dots)}</dots>\n"
    if triplet:
        text += _time_modification()
    text += f"      <durationType>{kind}</durationType>\n"
    text += "    </Rest>\n"
    return text


def _rest(duration):
    out = []
    for kind, dots, triplet in duration_pieces(float(duration)):
        out.append(_rest_piece(kind, dots, triplet=triplet))
    return "".join(out)


def _measure_music(events, start, duration):
    cursor = 0.0
    out = []
    for event in sorted(events, key=lambda row: float(row["start_beat"])):
        offset = float(event["start_beat"]) - float(start)
        if offset > cursor + 1e-8:
            out.append(_rest(offset - cursor))
        out.append(_chord(
            event["duration_beats"], event["source_pitch"],
            bool(event.get("tie_start")), bool(event.get("tie_stop"))))
        cursor = offset + float(event["duration_beats"])
    if duration > cursor + 1e-8:
        out.append(_rest(duration - cursor))
    if not out:
        ticks = round(duration * 480)
        out.append(
            "    <Rest>\n"
            "      <durationType>measure</durationType>\n"
            f"      <duration>{_frac(ticks, 1920)}</duration>\n"
            "    </Rest>\n")
    return "".join(out)


def _fragments(data):
    measure_map = data["measure_map"]
    by_bar = {}
    for event in data["highest_line"]:
        start = float(event["start_beat"])
        end = start + float(event["duration_beats"])
        cursor = start
        while cursor < end - 1e-8:
            bar, _ = locate(measure_map, cursor + 1e-9)
            measure = measure_map[bar]
            measure_end = (float(measure["start_beat"])
                           + float(measure["duration_beats"]))
            fragment_end = min(end, measure_end)
            by_bar.setdefault(bar, []).append({
                "start_beat": round(cursor, 10),
                "duration_beats": round(fragment_end - cursor, 10),
                "source_pitch": int(event["source_pitch"]),
                "tie_stop": cursor > start + 1e-8,
                "tie_start": fragment_end < end - 1e-8,
            })
            cursor = fragment_end
    return by_bar


def generate(data):
    validate_reference_ir(data)
    preserved = reusable_roundtrip_xml(data)
    if preserved is not None:
        return preserved
    measure_map = data["measure_map"]
    by_bar = _fragments(data)
    source_file = escape(str(data.get("source_file", "")))
    source_staff = escape(str(data.get("source_staff_id", "1")))
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<museScore version="3.02">\n',
        '  <programVersion>3.6.3</programVersion>\n',
        '  <programRevision>scaleweaver-reference-ir-1</programRevision>\n',
        '  <Score>\n',
        '    <Division>480</Division>\n',
        f'    <metaTag name="scaleweaverReferenceSource">{source_file}</metaTag>\n',
        f'    <metaTag name="scaleweaverReferenceStaff">{source_staff}</metaTag>\n',
        '    <Part>\n',
        '      <Staff id="1"><StaffType group="pitched"><name>stdNormal</name><lines>5</lines></StaffType></Staff>\n',
        '      <trackName>Reference highest line</trackName>\n',
        '      <Instrument id="piano"><trackName>Reference highest line</trackName><instrumentId>keyboard.piano</instrumentId><Channel><program value="0"/></Channel></Instrument>\n',
        '    </Part>\n',
        '    <Staff id="1">\n',
    ]
    previous_signature = None
    for bar, measure in enumerate(measure_map):
        duration = float(measure["duration_beats"])
        signature = str(measure["time_signature"])
        n, d = map(int, signature.split("/", 1))
        nominal = 4.0 * n / d
        length = ("" if math.isclose(duration, nominal, abs_tol=1e-8)
                  else f' len="{_frac(round(duration * 480), 1920)}"')
        out.append(f'      <Measure{length}>\n')
        out.append('        <voice>\n')
        if signature != previous_signature:
            out.extend([
                '          <TimeSig>\n',
                f'            <sigN>{n}</sigN>\n',
                f'            <sigD>{d}</sigD>\n',
                '          </TimeSig>\n',
            ])
        music = _measure_music(
            by_bar.get(bar, ()), measure["start_beat"], duration)
        out.append("\n".join("      " + line if line else line
                             for line in music.rstrip("\n").split("\n")) + "\n")
        out.append('        </voice>\n')
        out.append('      </Measure>\n')
        previous_signature = signature
    out.extend(['    </Staff>\n', '  </Score>\n', '</museScore>\n'])
    return canonicalize_mscx("".join(out))
