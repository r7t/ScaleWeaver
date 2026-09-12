"""ScaleWeaver composition pipeline and command-line entry point."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from functools import lru_cache
from itertools import combinations
import harmony_rhythm as hr
import score_builder as sb
import ji_ratio
from scale_config import load_scale
from spectral_bundle import (SpectralBundle, ensure_bundle,
                             _bundle_covers_spec_data)

DEFAULT_SCALE_CONFIG=None
_ACTIVE_SCALE_SPEC=None
_ACTIVE_CSE_BUNDLE=None
_ACTIVE_CSE_MANIFEST=None
_ACTIVE_JI_TABLE=None
_ACTIVE_JI_TABLE_PATH=None

def _configure_adaptive_scale(scale_config=None, cse_dir=None, *, rules_config=None, style_config=None, cse_strength=1.15, cse_harmony_gain=1.0, cse_workers=None, chord_progression=None):
    """Configure the single unified generator for one exact-EDO scale."""
    global _ACTIVE_SCALE_SPEC, _ACTIVE_CSE_BUNDLE, _ACTIVE_CSE_MANIFEST, _ACTIVE_JI_TABLE, _ACTIVE_JI_TABLE_PATH
    scale_path = DEFAULT_SCALE_CONFIG if scale_config is None else scale_config
    if scale_path is None: raise ValueError("A scale definition is required; pass --scale")
    spec = load_scale(scale_path, rules=rules_config, style=style_config, require_composition=True)
    if chord_progression is not None:
        from dataclasses import replace
        import copy
        harmony = copy.deepcopy(spec.harmony)
        text = str(chord_progression).strip()
        if text.lower() == 'auto':
            harmony.pop('chord_progression', None)
        else:
            # A scale may define one-character named chord aliases. Otherwise
            # retain the generic compact-degree grammar used by older configs.
            import re
            definitions = spec.resolved_chord_codes()
            coded_text = text.lower()
            coded_tokens = (re.split(r'[,\s]+', coded_text)
                            if re.search(r'[,\s]', coded_text) else list(coded_text))
            if definitions and coded_tokens and all(token in definitions for token in coded_tokens):
                harmony['chord_progression'] = {'codes': coded_tokens, 'bars_per_chord': 1}
                coded_tokens = None
            if coded_tokens is None:
                pass
            elif not text or not re.fullmatch(r'[0-9,\s]+', text):
                raise ValueError('chord_progression must use defined chord codes, scale degrees, or auto')
            else:
                tokens = re.split(r'[,\s]+', text) if re.search(r'[,\s]', text) else list(text)
                if any(not token for token in tokens):
                    raise ValueError('chord_progression contains an empty item')
                harmony['chord_progression'] = {'degrees': [int(d) for d in tokens], 'bars_per_chord': 1}
        rules = copy.deepcopy(spec.rules)
        rules['harmony'] = harmony
        spec = replace(spec, harmony=harmony, rules=rules)
    if cse_dir is None:
        if isinstance(scale_path, (str, Path)):
            p=Path(scale_path)
            cse_dir = (p.parent / 'cse_cache') if p.suffix else Path('.') / 'cse_cache'
        else:
            cse_dir = Path('.') / 'cse_cache'
    # A changed domain may rebuild the same filename: close its old mmap first.
    if (_ACTIVE_CSE_BUNDLE is not None
            and not _bundle_covers_spec_data(_ACTIVE_CSE_BUNDLE.data, spec)):
        _ACTIVE_CSE_BUNDLE.close()
        _ACTIVE_CSE_BUNDLE = None
        _ACTIVE_CSE_MANIFEST = None
    manifest = ensure_bundle(spec, cse_dir, workers=cse_workers)
    # Close a previous mmap only when genuinely switching manifests.
    if _ACTIVE_CSE_BUNDLE is None or Path(_ACTIVE_CSE_MANIFEST).resolve() != Path(manifest).resolve():
        if _ACTIVE_CSE_BUNDLE is not None:
            try: _ACTIVE_CSE_BUNDLE.close()
            except Exception: pass
        _ACTIVE_CSE_BUNDLE = SpectralBundle(manifest, spec, allow_superset=True)
        _ACTIVE_CSE_MANIFEST = Path(manifest)
    ji_dir = Path(cse_dir) / 'ji_cache'
    _ACTIVE_JI_TABLE = ji_ratio.ensure_loaded_table(spec, ji_dir)
    _ACTIVE_JI_TABLE_PATH = _ACTIVE_JI_TABLE.path
    hr.configure_scale(spec, _ACTIVE_CSE_BUNDLE, _ACTIVE_JI_TABLE)
    # Share the already parsed (potentially very large) JI table with the
    # runtime. Passing its path used to parse the same JSON a second time.
    sb.cse_rt.configure_scale(spec, _ACTIVE_JI_TABLE)
    sb.configure_scale(spec)
    hr.configure(cse_strength, cse_harmony_gain)
    for value in tuple(globals().values()):
        if callable(value) and getattr(value, "__module__", None) == __name__ and hasattr(value,"cache_clear"):
            value.cache_clear()
    _ACTIVE_SCALE_SPEC = spec
    return spec, Path(manifest)

def _voice_pitch_class_ratios(voices):
    """Return per-voice ScaleWeaver pitch-class ratios by note-event count."""
    rows = {}
    for voice, events in voices.items():
        counts = {name: 0 for name in hr.NAMES}
        total = 0
        for e in events:
            if 'step' not in e:
                continue
            pc = int(e['step']) % hr.OCT
            name = hr.PC_NAME.get(pc)
            if name is None:
                continue
            counts[name] += 1
            total += 1
        rows[voice] = {
            name: (counts[name] / total if total else 0.0)
            for name in hr.NAMES
        }
    return rows





# ---------------------------------------------------------------------------
# Attack-instant interval / canonical just-intonation export
# ---------------------------------------------------------------------------
ATTACK_ANALYSIS_PRIME_LIMIT = 17
ATTACK_ANALYSIS_EPS = 1e-9
ATTACK_ANALYSIS_PRIMES = (2, 3, 5, 7, 11, 13, 17)

def _load_attack_ji_precomputed_table():
    """Return the active scale-specific canonical 17-limit patent-val JI table."""
    if _ACTIVE_JI_TABLE is None:
        return None
    return _ACTIVE_JI_TABLE.data


def _require_attack_ji_precomputed_table():
    table = _load_attack_ji_precomputed_table()
    if table is None:
        raise RuntimeError('canonical JI table has not been configured for the active scale')
    return table






def _precomputed_attack_ji_lookup(interval_steps):
    if _ACTIVE_JI_TABLE is None:
        return None
    return _ACTIVE_JI_TABLE.lookup(tuple(int(x) for x in interval_steps))




@lru_cache(maxsize=32768)
def _canonical_integer_ji_from_step_intervals(interval_steps):
    """Return the canonical patent-val JI model for a 2/3/4/5-note sonority.

    The active mapping has no fixed integer limit: it requires exact mapping
    under the EDO's 17-limit patent val, using the configured dynamic RMS
    threshold, then minimizes the
    gcd-reduced largest integer.
    """
    steps = tuple(int(x) for x in interval_steps)
    if len(steps) not in (2, 3, 4, 5):
        raise ValueError(f'expected 2, 3, 4, or 5 offsets, got {steps!r}')
    if not steps or steps[0] != 0 or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError(f'expected strictly increasing offsets beginning at 0, got {steps!r}')
    precomputed = _precomputed_attack_ji_lookup(steps)
    if precomputed is not None:
        return precomputed

    edo = int(hr.OCT)
    return ji_ratio.canonical_fallback(edo, steps)


def _canonical_three_note_subsets(steps):
    """Fit the four delete-one-note three-note subsets independently.

    ``steps`` must contain the four sounding absolute ScaleWeaver steps in low-to-high
    order.  Subset order is deterministic:
      delete index 0, delete index 1, delete index 2, delete index 3.
    """
    xs = tuple(int(x) for x in steps)
    if len(xs) != 4:
        raise ValueError(f'expected four absolute steps, got {xs!r}')

    out = []
    for deleted_index in range(4):
        kept_indices = tuple(i for i in range(4) if i != deleted_index)
        subset_abs = tuple(xs[i] for i in kept_indices)
        root = subset_abs[0]
        subset_intervals = tuple(x - root for x in subset_abs)
        canonical = _canonical_integer_ji_from_step_intervals(subset_intervals)
        out.append({
            'deleted_index': deleted_index,
            'kept_indices': list(kept_indices),
            'absolute_steps': list(subset_abs),
            'interval_steps_from_lowest': list(subset_intervals),
            'canonical_ji': canonical,
        })
    return out


def _canonical_four_note_subsets(steps):
    """Fit the five delete-one four-note subsets of a five-note sonority."""
    xs = tuple(int(x) for x in steps)
    if len(xs) != 5:
        raise ValueError(f'expected five absolute steps, got {xs!r}')

    out = []
    for deleted_index in range(5):
        kept_indices = tuple(i for i in range(5) if i != deleted_index)
        subset_abs = tuple(xs[i] for i in kept_indices)
        root = subset_abs[0]
        subset_intervals = tuple(x - root for x in subset_abs)
        canonical = _canonical_integer_ji_from_step_intervals(subset_intervals)
        out.append({
            'deleted_index': deleted_index,
            'kept_indices': list(kept_indices),
            'absolute_steps': list(subset_abs),
            'interval_steps_from_lowest': list(subset_intervals),
            'canonical_ji': canonical,
        })
    return out

def _pitched_events_with_voice(voices):
    rows = []
    for voice, events in voices.items():
        for e in events:
            if 'step' not in e:
                continue
            start = float(e['start_beat'])
            end = start + float(e['duration_beats'])
            if end <= start + ATTACK_ANALYSIS_EPS:
                continue
            rows.append((str(voice), e, start, end))
    return rows


def _ratio_prime_presence(integers):
    """Prime-presence flags for one gcd-reduced canonical integer ratio."""
    xs = tuple(abs(int(x)) for x in (integers or ()))
    return {
        str(p): any(x > 0 and x % p == 0 for x in xs)
        for p in ATTACK_ANALYSIS_PRIMES
    }





def _prime_flags_from_fit(fit):
    """Prime flags for one canonical gcd-reduced integer fit."""
    ints = tuple(fit.get('integers', ())) if isinstance(fit, dict) else ()
    return _ratio_prime_presence(ints) if ints else None


def _prime_flags_any_three_note_subset(subsets):
    """Union prime presence over all four independently fitted 3-note subsets.

    A four-note sonority is considered analyzable only when all four delete-one
    subsets have valid canonical fits.  This exactly restores the historical
    'ANY of four three-note subsets' definition and prevents a full four-integer
    fit from injecting a high prime that none of the triadic views requires.
    """
    rows = list(subsets or ())
    if len(rows) != 4:
        return None
    flags = {str(p): False for p in ATTACK_ANALYSIS_PRIMES}
    for row in rows:
        fit = row.get('canonical_ji') if isinstance(row, dict) else None
        subflags = _prime_flags_from_fit(fit)
        if subflags is None:
            return None
        for p, yes in subflags.items():
            flags[p] = flags[p] or bool(yes)
    return flags


def _simultaneous_prime_flags(row):
    """Prime flags for a true simultaneous attack.

    Dyads and triads use their own canonical fit. Four-note simultaneous attacks
    use the same ANY-of-four-triad-subsets definition as the sounding-quartet
    statistic, so cardinality 4 cannot be distorted by a different full-quartet
    fitting convention.
    """
    card = int(row.get('attacking_note_count', 0))
    if card == 4:
        return _prime_flags_any_three_note_subset(
            row.get('simultaneous_attack_three_note_subsets')
        )
    fit = row.get('simultaneous_attack_canonical_ji')
    return _prime_flags_from_fit(fit)




def _prime_presence_summary_from_attacks(attacks):
    """Prime coverage using one consistent canonical definition.

    * Five notes SOUNDING: use the genuine canonical five-integer JI fit.
    * Four notes SOUNDING: historical definition = a prime is present when it
      occurs in ANY of the four independently fitted delete-one 3-note subsets.
    * True simultaneous attacks: dyads/triads use their own canonical fit;
      four-note attacks again use ANY of four independently fitted triad subsets.
    * Single-note attacks are excluded from simultaneous prime statistics.
    """
    four_counts = {str(p): 0 for p in ATTACK_ANALYSIS_PRIMES}
    five_counts = {str(p): 0 for p in ATTACK_ANALYSIS_PRIMES}
    three_counts = {str(p): 0 for p in ATTACK_ANALYSIS_PRIMES}
    attack_counts = {str(p): 0 for p in ATTACK_ANALYSIS_PRIMES}
    by_card = {
        k: {
            'total': 0,
            'missing': 0,
            'counts': {str(p): 0 for p in ATTACK_ANALYSIS_PRIMES},
        }
        for k in (2, 3, 4, 5)
    }
    four_total = four_missing = 0
    five_total = five_missing = 0
    three_total = three_missing = 0
    attack_total = attack_missing = 0

    for row in attacks:
        if int(row.get('sounding_note_count', 0)) == 3:
            three_total += 1
            flags = _prime_flags_from_fit(row.get('canonical_ji'))
            if flags is None:
                three_missing += 1
            else:
                for p, yes in flags.items():
                    three_counts[p] += int(bool(yes))

        if int(row.get('sounding_note_count', 0)) == 4:
            four_total += 1
            flags = _prime_flags_any_three_note_subset(
                row.get('canonical_ji_three_note_subsets')
            )
            if flags is None:
                four_missing += 1
            else:
                for p, yes in flags.items():
                    four_counts[p] += int(bool(yes))

        if int(row.get('sounding_note_count', 0)) == 5:
            five_total += 1
            flags = _prime_flags_from_fit(row.get('canonical_ji'))
            if flags is None:
                five_missing += 1
            else:
                for p, yes in flags.items():
                    five_counts[p] += int(bool(yes))

        attack_card = int(row.get('attacking_note_count', 0))
        if attack_card >= 2:
            attack_total += 1
            if attack_card in by_card:
                by_card[attack_card]['total'] += 1

            flags = _simultaneous_prime_flags(row)
            if flags is None:
                attack_missing += 1
                if attack_card in by_card:
                    by_card[attack_card]['missing'] += 1
            else:
                for p, yes in flags.items():
                    attack_counts[p] += int(bool(yes))
                    if attack_card in by_card:
                        by_card[attack_card]['counts'][p] += int(bool(yes))

    def pack(total, missing, counts, definition):
        analyzed = max(0, total - missing)
        return {
            'eligible_attack_count': total,
            'analyzed_attack_count': analyzed,
            'missing_ratio_count': missing,
            'prime_presence_ratio': {
                p: (counts[p] / analyzed if analyzed else 0.0)
                for p in map(str, ATTACK_ANALYSIS_PRIMES)
            },
            'prime_presence_count': counts,
            'definition': definition,
        }

    result = {
        'three_part_sounding_at_attack': pack(
            three_total, three_missing, three_counts,
            'Exactly three notes sounding; prime presence in the canonical three-integer JI fit.'
        ),
        'five_part_sounding_at_attack': pack(
            five_total, five_missing, five_counts,
            'Exactly five notes sounding; prime presence in the canonical five-integer JI fit.'
        ),
        'four_part_sounding_at_attack': pack(
            four_total, four_missing, four_counts,
            'Exactly four notes sounding; prime present iff ANY of the four independently fitted delete-one 3-note subsets contains it.'
        ),
        'simultaneous_multi_note_attack': pack(
            attack_total, attack_missing, attack_counts,
            'At least two notes actually attacking together; dyad/triad/five-note attacks use their own fit, four-note attacks use ANY of four independently fitted triad subsets.'
        ),
    }
    result['simultaneous_attack_by_cardinality'] = {
        str(k): pack(
            row['total'], row['missing'], row['counts'],
            (f'True simultaneous {k}-note attacks; '
             + ('ANY of four independently fitted triad subsets.' if k == 4
                else 'prime presence in the canonical attack fit.'))
        )
        for k, row in by_card.items()
    }
    return result

def _lead_directed_step_statistics(voices):
    """Signed adjacent Lead motion in absolute active-EDO steps."""
    lead = sorted(
        (e for e in voices.get('lead', ()) if 'step' in e),
        key=lambda e: (float(e.get('start_beat', 0.0)), float(e.get('duration_beats', 0.0)))
    )
    counts = {}
    for a, b in zip(lead, lead[1:]):
        d = int(b['step']) - int(a['step'])
        counts[d] = counts.get(d, 0) + 1
    total = sum(counts.values())
    ordered = sorted(counts.items())
    return {
        'transition_count': total,
        'by_directed_edo_steps': {
            (f'{step:+d}' if step else '0'): {
                'count': count,
                'ratio': count / total if total else 0.0,
            }
            for step, count in ordered
        },
    }


def _compact_output_statistics(score, attack_analysis=None):
    """Small user-facing statistics block; does not affect generation/search.

    Every configured scale now owns an automatically generated canonical
    17-prime-limit patent-val JI map, so prime/JI attack statistics are
    available for arbitrary exact-EDO subset scales.
    """
    attack_analysis = attack_analysis or {}
    return {
        'lead_directed_steps': _lead_directed_step_statistics(score.get('voices', {})),
        'attack_prime_presence': attack_analysis.get('prime_statistics', {}),
    }


def _format_prime_ratios(summary):
    ratios = summary.get('prime_presence_ratio', {}) if isinstance(summary, dict) else {}
    return ' '.join(
        f'{p}:{100.0 * float(ratios.get(str(p), 0.0)):5.1f}%'
        for p in ATTACK_ANALYSIS_PRIMES
    )


def _print_compact_statistics(score, attack_analysis=None):
    attack_analysis = attack_analysis or {}
    stats = score.get('statistics', {})
    print('Voice pitch-class ratios (event count)')
    ratios = _voice_pitch_class_ratios(score.get('voices', {}))
    for voice, row in ratios.items():
        text = ' '.join(
            f'{name}:{100.0 * row[name]:5.1f}%'
            for name in hr.NAMES
        )
        print(f'  {voice:<8} {text}')

    bars = max(1, int(score.get('bars', 1)))
    bass_notes = len(score.get('voices', {}).get('bass', []))
    print(
        f'Bass density: {bass_notes / bars:.3f} notes/bar '
        f'({bass_notes}/{bars})'
    )
    lead = stats.get('lead_directed_steps', {})
    step_rows = lead.get('by_directed_edo_steps', {})
    step_text = ' '.join(
        f'{step}:{100.0 * float(info["ratio"]):.1f}%'
        for step, info in step_rows.items()
    ) or '(none)'
    print(f'Lead directed {int(hr.OCT)}-EDO steps (n={lead.get("transition_count", 0)}): {step_text}')

    prime = attack_analysis.get('prime_statistics', {})
    if prime:
        five = prime.get('five_part_sounding_at_attack', {})
        four = prime.get('four_part_sounding_at_attack', {})
        three = prime.get('three_part_sounding_at_attack', {})
        multi = prime.get('simultaneous_multi_note_attack', {})
        if int(five.get('eligible_attack_count', 0)) > 0:
            print(
                f'Five-part sounding attack prime ratios (n={five.get("analyzed_attack_count", 0)}): '
                + _format_prime_ratios(five)
            )
        if int(four.get('eligible_attack_count', 0)) > 0:
            print(
                f'Four-part sounding attack prime ratios (n={four.get("analyzed_attack_count", 0)}): '
                + _format_prime_ratios(four)
            )
        if int(three.get('eligible_attack_count', 0)) > 0:
            print(
                f'Three-part sounding attack prime ratios (n={three.get("analyzed_attack_count", 0)}): '
                + _format_prime_ratios(three)
            )
        print(
            f'True simultaneous multi-attack prime ratios (n={multi.get("analyzed_attack_count", 0)}): '
            + _format_prime_ratios(multi)
        )
        by_card = prime.get('simultaneous_attack_by_cardinality', {})
        for k in ('2', '3', '4', '5'):
            row = by_card.get(k, {})
            if int(row.get('eligible_attack_count', 0)) > 0:
                print(
                    f'  simultaneous {k}-note (n={row.get("analyzed_attack_count", 0)}): '
                    + _format_prime_ratios(row)
                )
    else:
        print('Prime attack ratios: unavailable (attack-JI table not loaded)')

    print('Final spectral metrics (no subset subtraction):')
    voice_count=len(score.get('voices',{}))
    keys=[f'{voice_count}_note_sounding']+[f'{n}_note_simultaneous_attack' for n in range(2,voice_count+1)]
    for key in keys:
        row=score['final_spectral_metrics'][key]
        fields=' '.join(f'{m}={row[m]:.6f}' if row[m] is not None else f'{m}=NA'
                        for m in ('cse','ccse','se','blending'))
        print(f'  {key}: {fields} n={row["count"]}')
    ann=score.get('annealing',{})
    if ann.get('enabled'):
        print(f'Annealing: energy={ann["initial_energy"]:.6f}->{ann["final_energy"]:.6f} '
              f'changed_notes={ann["changed_notes"]} lead_unchanged={ann["lead_unchanged"]}')
        sources=ann.get('energy_sources',{})
        if sources:
            print('Energy sources (initial -> final; delta):')
            for key in sources['initial']:
                before=sources['initial'][key];after=sources['final'][key];delta=sources['delta'][key]
                print(f'  {key}: {before:.6f} -> {after:.6f}; {delta:+.6f}')


def build_attack_interval_analysis(score):
    """Describe attack instants at which exactly five pitches are sounding.

    A simultaneous multi-voice attack contributes one row.  At each attack we
    inspect the half-open sounding set [start, end), so a note ending exactly on
    that attack is not incorrectly retained. Other texture cardinalities are
    intentionally omitted from the attack JSON.
    """
    # Do not silently fall back to a different JI fitter: that changes prime
    # coverage statistics while leaving the output format looking identical.
    _require_attack_ji_precomputed_table()
    voices = score.get('voices', {})
    rows = _pitched_events_with_voice(voices)
    attack_beats = sorted({round(start, 9) for _, _, start, _ in rows})
    attacks = []
    statistics_attacks = []

    for beat in attack_beats:
        attacking = [
            (voice, e) for voice, e, start, _ in rows
            if abs(start - beat) <= ATTACK_ANALYSIS_EPS
        ]
        attacking.sort(key=lambda x: (int(x[1]['step']), x[0]))
        attacking_steps = [int(e['step']) for _, e in attacking]
        simultaneous_attack_fit = None
        simultaneous_attack_subsets = None
        if 2 <= len(attacking_steps) <= 5:
            sim_root = attacking_steps[0]
            sim_intervals = tuple(x - sim_root for x in attacking_steps)
            simultaneous_attack_fit = _canonical_integer_ji_from_step_intervals(sim_intervals)
            if len(attacking_steps) == 4:
                simultaneous_attack_subsets = _canonical_three_note_subsets(
                    tuple(attacking_steps)
                )
        simultaneous_attack_ratio = (
            simultaneous_attack_fit.get('ratio_string')
            if isinstance(simultaneous_attack_fit, dict) else
            ('1' if len(attacking_steps) == 1 else None)
        )
        sounding = [
            (voice, e) for voice, e, start, end in rows
            if start <= beat + ATTACK_ANALYSIS_EPS and end > beat + ATTACK_ANALYSIS_EPS
        ]
        sounding.sort(key=lambda x: (int(x[1]['step']), x[0]))
        notes = []
        for voice, e in sounding:
            step = int(e['step'])
            notes.append({
                'voice': voice,
                'step': step,
                'name': str(e.get('name') or hr.pitch_name(step)),
                'freq_hz': round(float(e.get('freq', hr.freq(step))), 9),
            })

        steps = [n['step'] for n in notes]
        interval_from_lowest = [x - steps[0] for x in steps] if steps else []
        adjacent = [b - a for a, b in zip(steps, steps[1:])]
        pairwise = [
            {'lower_index': i, 'upper_index': j, 'steps': steps[j] - steps[i]}
            for i in range(len(steps)) for j in range(i + 1, len(steps))
        ]

        canonical = None
        canonical_subsets = None
        canonical_note = None
        if len(steps) == 3:
            canonical = _canonical_integer_ji_from_step_intervals(
                tuple(interval_from_lowest)
            )
            if canonical is None:
                canonical_note = (
                    'no legal three-integer construction found under the configured limits'
                )
        elif len(steps) == 4:
            canonical = _canonical_integer_ji_from_step_intervals(
                tuple(interval_from_lowest)
            )
            canonical_subsets = _canonical_three_note_subsets(tuple(steps))
            if canonical is None:
                canonical_note = (
                    'no legal four-integer construction found under the configured limits'
                )
        elif len(steps) == 5:
            canonical = _canonical_integer_ji_from_step_intervals(
                tuple(interval_from_lowest)
            )
            canonical_subsets = _canonical_four_note_subsets(tuple(steps))
            if canonical is None:
                canonical_note = (
                    'no legal five-integer construction found under the configured limits'
                )

        analysis_row = {
            'attack_index': len(attacks) if len(steps) == 5 else None,
            'beat': beat,
            'bar_index_1based': int(beat // float(score.get('beats_per_bar', 4.0))) + 1,
            'beat_in_bar': round(beat % float(score.get('beats_per_bar', 4.0)), 9),
            'attacking_voices': sorted({voice for voice, _ in attacking}),
            'attacking_note_count': len(attacking_steps),
            'attacking_steps_low_to_high': list(attacking_steps),
            'simultaneous_attack_ratio': simultaneous_attack_ratio,
            'simultaneous_attack_canonical_ji': simultaneous_attack_fit,
            'simultaneous_attack_three_note_subsets': simultaneous_attack_subsets,
            'sounding_note_count': len(notes),
            'notes_low_to_high': notes,
            'interval_steps_from_lowest': interval_from_lowest,
            'adjacent_interval_steps': adjacent,
            'pairwise_intervals': pairwise,
            'canonical_ji': canonical,
            'canonical_ji_three_note_subsets': (
                canonical_subsets if len(steps) == 4 else None
            ),
            'canonical_ji_four_note_subsets': (
                canonical_subsets if len(steps) == 5 else None
            ),
            'canonical_ji_note': canonical_note,
        }
        statistics_attacks.append(analysis_row)
        if len(steps) == 5:
            attacks.append(analysis_row)

    result = {
        'format': 'ScaleWeaverAttackIntervals/1',
        'description': (
            'Only distinct pitched attack instants with exactly five sounding pitches. '
            'The sounding sonority and the notes that actually attack at that instant are analyzed separately.'
        ),
        'tuning': {
            'edo': int(hr.OCT),
            'base_freq_hz': float(hr.BASE_FREQ),
            'pitch_classes': list(getattr(hr, 'PCS', ())),
            'names': list(getattr(hr, 'NAMES', ())),
        },
        'canonical_ji_method': {
            'prime_limit': ATTACK_ANALYSIS_PRIME_LIMIT,
            'precomputed_table': str(_ACTIVE_JI_TABLE_PATH),
            'precomputed_table_loaded': _load_attack_ji_precomputed_table() is not None,
            'precomputed_table_loaded_path': (
                (_load_attack_ji_precomputed_table() or {}).get('_loaded_path')
            ),
            'two_note_source': 'direct 17-limit patent-val mapping; minimum maximum integer under the dynamic RMS threshold',
            'target': 'active-EDO interval steps relative to the lowest pitch',
            'objective': 'exact 17-limit patent-val mapping; RMS<=10% of one EDO step + 3.5c; minimum gcd-reduced maximum integer; no fixed integer limit',
            'tie_break': 'after minimum maximum integer: lower integer sum, lower RMS, lower max absolute error, then lexicographic tuple',
            'simultaneous_attack_ratio': 'single attacking note is literal 1; 2/3/4/5-note attacks use the same canonical fitter',
        },
        'source_attack_instant_count': len(attack_beats),
        'attack_count': len(attacks),
        'five_note_sounding_attack_count': len(attacks),
        'multi_note_simultaneous_attack_count': sum(a['attacking_note_count'] >= 2 for a in attacks),
        'simultaneous_attack_cardinality_count': {
            str(k): sum(a['attacking_note_count'] == k for a in attacks)
            for k in sorted({a['attacking_note_count'] for a in attacks})
        },
        'attacks': attacks,
    }
    # Prime statistics intentionally use every attack instant, while the JSON
    # detail table remains restricted to five-note sounding rows.
    result['prime_statistics'] = _prime_presence_summary_from_attacks(statistics_attacks)
    return result


def _default_attack_analysis_path(score_filename):
    p = Path(score_filename)
    suffix = p.suffix or '.json'
    return p.with_name(p.stem + '_attacks' + suffix)





def _ir_progression_argument(frontend):
    """Recreate a CLI progression override needed by generic manual chords."""
    progression=frontend.get('generator',{}).get('resolved_chord_progression')
    if not progression:return None
    if 'codes' in progression:return ','.join(str(x) for x in progression['codes'])
    if 'degrees' in progression:return ','.join(str(int(x)) for x in progression['degrees'])
    return None


def generate_score(seed=20260811,bars=48,bpm=96,motif_degrees=None,time_signature='4/4',
                   cse_strength=1.15,cse_harmony_gain=1.0,allow_sixteenth=True,
                   scale_config=None,cse_dir=None,cse_workers=None,rules_config=None,style_config=None,
                   chord_progression=None,frontend_ir=None,frontend_ir_output=None,
                   imitation_reference=None,imitation_staff_id=None,
                   imitation_beam_width=48,retune_melody_source=None,
                   retune_staff_id=None):
    sources = [frontend_ir is not None, imitation_reference is not None,
               retune_melody_source is not None]
    if sum(sources) > 1:
        raise ValueError('frontend_ir, imitation_reference and retune_melody_source are mutually exclusive')
    frontend=None
    if frontend_ir is not None:
        from frontend_ir import load_ir,validate_ir
        frontend=(validate_ir(frontend_ir) if isinstance(frontend_ir,dict)
                  else load_ir(frontend_ir))
        if chord_progression is not None:
            raise ValueError('chord_progression cannot replace the harmony plan stored in frontend_ir')
        chord_progression=_ir_progression_argument(frontend)
    spec,manifest=_configure_adaptive_scale(scale_config,cse_dir,rules_config=rules_config,
        style_config=style_config,cse_strength=cse_strength,cse_harmony_gain=cse_harmony_gain,
        cse_workers=cse_workers,chord_progression=chord_progression)
    from frontend_ir import save_ir,validate_ir
    if frontend is None:
        if not math.isfinite(bpm) or bpm<=0:raise ValueError('bpm must be positive')
        if imitation_reference is not None:
            from imitation_frontend import generate_frontend_ir
            frontend=generate_frontend_ir(
                imitation_reference,seed=int(seed),bpm=bpm,spec=spec,
                staff_id=imitation_staff_id,beam_width=imitation_beam_width)
        elif retune_melody_source is not None:
            from retune_melody import generate_frontend_ir
            frontend=generate_frontend_ir(
                retune_melody_source,seed=int(seed),bpm=bpm,spec=spec,
                staff_id=retune_staff_id)
        else:
            from joint_frontend_generate import generate_frontend_ir
            ts,_=hr.normalize_time_signature(time_signature)
            if type(bars) is not int or bars<=0:
                raise ValueError('bars must be a positive integer')
            frontend=generate_frontend_ir(
                int(seed),bars,bpm,motif_degrees,ts,allow_sixteenth,spec)
    else:
        validate_ir(frontend,spec=spec)
    if frontend_ir_output is not None:
        save_ir(frontend_ir_output,frontend,spec=spec)
    from accompaniment_generate import generate_from_ir
    score=generate_from_ir(frontend,spec)
    resolved_annealing=score.get('annealing',{}).get('config',{})
    score['cse_system']={'table_file':str(manifest),'metric':'a*CSE+b*SE+c*CCSE',
        'optimizer':('weighted mean of full raw blend and the worst raw subset '
                     'at every smaller cardinality; no recursive aggregation'),
        'optimizer_cache_file':score.get('annealing',{}).get('subset_cse_cache_manifest'),
        'worst_subset_cse_weights':resolved_annealing.get(
            'worst_subset_cse_weights',
            spec.style.get('annealing',{}).get('worst_subset_cse_weights'))}
    return score


def save_score(filename='adaptive_score.json',attack_analysis_filename=None,
               print_statistics=True,auto_attack_analysis=True,**kwargs):
    score=generate_score(**kwargs)
    path=Path(filename);path.parent.mkdir(parents=True,exist_ok=True)
    attack_analysis=None
    if attack_analysis_filename is not None or auto_attack_analysis:
        attack_path=Path(attack_analysis_filename) if attack_analysis_filename else _default_attack_analysis_path(filename)
        attack_path.parent.mkdir(parents=True,exist_ok=True)
        attack_analysis=build_attack_interval_analysis(score)
        attack_path.write_text(json.dumps(attack_analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshot=_ACTIVE_SCALE_SPEC.to_json_dict()
    ngram=_ACTIVE_SCALE_SPEC.resolved_ngram_path()
    if ngram is not None:snapshot['style']['ngram_file']=os.path.relpath(ngram,path.resolve().parent)
    score['configuration']=snapshot
    score['statistics']=_compact_output_statistics(score,attack_analysis)
    path.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated',filename)
    print('Melody seed',score['seed'],'Validation',score['validation'])
    if print_statistics:_print_compact_statistics(score,attack_analysis)
    return score


def _build_cli_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scale',dest='scale_config',required=True)
    ap.add_argument('--rules',dest='rules_config');ap.add_argument('--style',dest='style_config')
    ap.add_argument('--seed',type=int,default=20260811);ap.add_argument('--bars',type=int,default=48)
    ap.add_argument('--bpm',type=float,default=96.0);ap.add_argument('--time-signature',default='4/4')
    ap.add_argument('--chord-progression',help='fixed degree/code loop; auto restores automatic harmony')
    ap.add_argument('--cse-dir');ap.add_argument('--cse-workers',type=int)
    ap.add_argument('--output','-o',default='adaptive_score.json')
    ap.add_argument('--frontend-ir',
                    help='read ScaleWeaverFrontEndIR/1 and skip Lead/harmony generation')
    ap.add_argument('--save-frontend-ir',dest='frontend_ir_output',
                    help='write the generated/read front-end IR before accompaniment')
    source=ap.add_mutually_exclusive_group()
    source.add_argument('--imitate-rhythm','--imitate-mscx','--imitate-reference',
                        dest='imitation_reference',
                        help='MSCX or reference IR used as a rhythm/barline template')
    source.add_argument('--retune-melody',dest='retune_melody_source',
                        help='copy and minimum-RMS retune the complete source melody')
    ap.add_argument('--imitation-staff-id')
    ap.add_argument('--imitation-beam-width',type=int,default=48)
    ap.add_argument('--retune-staff-id')
    g=ap.add_mutually_exclusive_group()
    g.add_argument('--allow-sixteenth',dest='allow_sixteenth',action='store_true')
    g.add_argument('--no-sixteenth',dest='allow_sixteenth',action='store_false')
    ap.set_defaults(allow_sixteenth=True)
    ap.add_argument('--no-statistics',dest='print_statistics',action='store_false',default=True)
    ap.add_argument('--attack-analysis',default=None)
    return ap


def cli(argv=None):
    a=vars(_build_cli_parser().parse_args(argv));filename=a.pop('output')
    attack=a.pop('attack_analysis')
    return save_score(filename=filename,attack_analysis_filename=attack,**a)

if __name__=='__main__':
    cli()
