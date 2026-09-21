"""Compare inferred harmony with explicitly notated MSCX chords, without training.

The harmony staff is read only AFTER inference. Playback offsets, tuning and
ottavas are interpreted by the same parser used for melody import.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import harmony_rhythm as hr
from imitation_frontend import _voice_events, _scaled_harmony_plan
from retune_melody import analyze_reference, retune_events, _source_class_index
from retune_harmony import infer_harmony


def harmonic_similarity(reference, candidate):
    """Separate pitch overlap and 3-tone colour; no claim of optimal labels."""
    left, right = set(reference['pcs']), set(candidate['voicing_pcs'])
    a = reference.get('three_tone', {})
    b = hr.THREE_TONE_INFO.get(candidate['chord_id'], {})
    wa, wb = a.get('three_tone_function_weights'), b.get('three_tone_function_weights')
    cosine = None
    if wa and wb:
        keys = set(wa) | set(wb)
        norm = math.sqrt(sum(wa.get(k, 0.)**2 for k in keys) * sum(wb.get(k, 0.)**2 for k in keys))
        cosine = sum(wa.get(k, 0.)*wb.get(k, 0.) for k in keys)/norm if norm else None
    return dict(shared_tones=len(left & right),
                pitch_dice=2.*len(left & right)/max(1, len(left)+len(right)),
                same_three_tone=(a['three_tone_pc']==b['three_tone_pc']
                    if a.get('three_tone_pc') is not None and b.get('three_tone_pc') is not None else None),
                function_cosine=cosine)


def annotated_chords(path, staff_id, bars, analysis, mapping):
    staff = ET.parse(path).getroot().find('./Score/Staff[@id="%s"]' % staff_id)
    if staff is None:
        raise ValueError('annotation staff not found')
    result = []
    ottavas = {}
    for bar, measure in enumerate(staff.findall('Measure')[:bars]):
        for voice_index, voice in enumerate(measure.findall('voice')):
            initial = ottavas.get(voice_index, 0.)
            count = max((len(c.findall('Note')) for c in voice.findall('Chord')), default=0)
            groups = {}
            for note_index in range(count):
                projected = copy.deepcopy(voice)
                for chord in projected.findall('Chord'):
                    for i, note in enumerate(chord.findall('Note')):
                        if i != note_index:
                            chord.remove(note)
                events, _, _ = _voice_events(projected, initial)
                for event in events:
                    source_index = _source_class_index(event['source_actual_cents'],
                        analysis['source_pitch_classes_cents'])
                    pc = mapping['source_pc_to_target_step'][source_index]
                    groups.setdefault((event['offset'], event['duration_beats']), set()).add(pc)
            _, _, ottavas[voice_index] = _voice_events(voice, initial)
            for (offset, duration), pcs in sorted(groups.items()):
                chord = next((c for c in hr.CHORDS.values() if set(c.pcs) == pcs), None)
                result.append(dict(bar=bar, offset=offset, duration=duration,
                                   pcs=sorted(pcs), chord_id=chord.id if chord else None,
                                   chord_name=chord.name if chord else 'outside_vocabulary',
                                   three_tone=hr.THREE_TONE_INFO.get(chord.id, {}) if chord else {}))
    return result


def evaluate(path, spec, *, melody_staff='1', harmony_staff='3', bars=8, seed=42):
    analysis = analyze_reference(path, staff_id=melody_staff)
    mapped, mapping = retune_events(analysis, spec)
    measure_map = analysis['reference']['measure_map'][:bars]
    events = [e for e in mapped if e['bar'] < bars]
    predicted = infer_harmony(events, measure_map)
    baseline = _scaled_harmony_plan(bars, random.Random(seed), measure_map)
    truth = annotated_chords(path, harmony_staff, bars, analysis, mapping)
    rows = []
    for row in truth:
        bar, offset = row['bar'], row['offset']
        guess = hr.harmony_segment_at(predicted[bar], offset)
        old = hr.harmony_segment_at(baseline[bar], offset)
        alternatives = guess['melody_inference']['alternatives']
        end = offset + row['duration']
        sounding = [dict(offset=e['start_beat']-measure_map[bar]['start_beat'],
                         duration=e['duration_beats'], name=hr.PC_NAME[e['step'] % hr.OCT])
                    for e in events if e['bar'] == bar and
                    offset <= e['start_beat']-measure_map[bar]['start_beat'] < end]
        rows.append(dict(**row, melody=sounding, predicted=guess['chord_name'],
                         baseline=old['chord_name'], match=set(row['pcs'])==set(guess['voicing_pcs']),
                         baseline_match=set(row['pcs'])==set(old['voicing_pcs']),
                         similarity=harmonic_similarity(row, guess),
                         baseline_similarity=harmonic_similarity(row, old),
                         top5_match=any(a['chord_id']==row['chord_id'] for a in alternatives),
                         evidence=guess['melody_inference']))
    similarity_summary = {}
    for source in ('similarity', 'baseline_similarity'):
        summary = {}
        for metric in ('pitch_dice', 'same_three_tone', 'function_cosine'):
            values = [r[source][metric] for r in rows if r[source][metric] is not None]
            summary[metric] = dict(mean=sum(values)/len(values) if values else None, count=len(values))
        similarity_summary[source] = summary
    return dict(reference=str(path), annotated_segments=len(rows), similarity_summary=similarity_summary,
                exact_matches=sum(r['match'] for r in rows),
                baseline_matches=sum(r['baseline_match'] for r in rows),
                top5_matches=sum(r['top5_match'] for r in rows), baseline_seed=seed,
                evaluation='same-piece diagnostic, not held-out generalization accuracy', rows=rows)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference')
    p.add_argument('--scale', required=True)
    p.add_argument('--rules')
    p.add_argument('--style')
    p.add_argument('--cse-dir', default='CSE_cache')
    p.add_argument('--harmony-model', choices=('legacy', 'three-tone'), default='three-tone')
    p.add_argument('--melody-staff', default='1')
    p.add_argument('--harmony-staff', default='3')
    p.add_argument('--bars', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import main
    spec, _ = main._configure_adaptive_scale(args.scale, args.cse_dir,
        rules_config=args.rules, style_config=args.style, harmony_model=args.harmony_model)
    report = evaluate(args.reference, spec, melody_staff=args.melody_staff,
                      harmony_staff=args.harmony_staff, bars=args.bars, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print({k:v for k,v in report.items() if k != 'rows'})
    for r in report['rows']:
        print(r['bar']+1, r['offset'], r['chord_name'], '->', r['predicted'], r['match'])


if __name__ == '__main__':
    cli()
