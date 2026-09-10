#!/usr/bin/env python3
"""Convert an arbitrary 12-EDO MSCX score to a configured ScaleWeaver scale.

The source score is parsed into a pitch-only JSON IR.  Rhythm, onset, duration,
staff, voice, chord membership and the number of notes are immutable.  Notes
are mapped to the nearest target-scale pitch and then globally optimized by
simulated annealing.  Every pitch may move, but only inside a 300-cent window
around its nearest mapped pitch.  Tied notes move as one variable.

For sonorities larger than four notes, the objective is the arithmetic mean of
all four-note subset costs.  This deliberately avoids evaluating a native
five-or-more-note CSE.  Wolf intervals are hard constraints.  Notes outside the
precomputed absolute-pitch domain are octave-folded before optimization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import harmony_annealing
import harmony_rhythm as hr
import main as composer
from annealing_config import resolve_annealing


EPS = 1e-8
DURATION_BEATS = {
    "long": 16.0,
    "longa": 16.0,
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
    "256th": 0.015625,
}
TPC_SHARP = {
    0: 14, 1: 21, 2: 16, 3: 23, 4: 18, 5: 13,
    6: 20, 7: 15, 8: 22, 9: 17, 10: 24, 11: 19,
}


def _fraction_beats(text: str | None, default: float = 0.0) -> float:
    if not text:
        return float(default)
    try:
        return float(Fraction(text.strip()) * 4)
    except (ValueError, ZeroDivisionError):
        return float(default)


def _event_duration(element: ET.Element, measure_beats: float,
                    tuplets: dict[str, float]) -> float:
    explicit = element.findtext("duration")
    if explicit:
        duration = _fraction_beats(explicit, measure_beats)
    else:
        kind = (element.findtext("durationType") or "").strip()
        duration = measure_beats if kind == "measure" else DURATION_BEATS.get(kind, 0.0)
        dots = int(element.findtext("dots") or 0)
        if dots:
            duration *= 2.0 - 0.5 ** dots
    tuplet_id = element.findtext("Tuplet")
    if tuplet_id:
        duration *= tuplets.get(tuplet_id.strip(), 1.0)
    return float(duration)


def _is_grace(chord: ET.Element) -> bool:
    tags = {child.tag.lower() for child in chord}
    return any(tag.startswith("grace") or tag in {"acciaccatura", "appoggiatura"}
               for tag in tags)


def _tie_flags(note: ET.Element) -> tuple[bool, bool]:
    start = any(x.get("type") == "start" for x in note.findall("tie"))
    stop = any(x.get("type") == "stop" for x in note.findall("tie"))
    for spanner in note.findall("Spanner"):
        if spanner.get("type") != "Tie":
            continue
        start = start or spanner.find("Tie") is not None or spanner.find("next") is not None
        stop = stop or spanner.find("prev") is not None
    return bool(start), bool(stop)


def _actual_staves(score: ET.Element) -> list[ET.Element]:
    return [staff for staff in score.findall("Staff") if staff.find("Measure") is not None]


def _measure_grid(staves: list[ET.Element]) -> tuple[list[float], list[float]]:
    count = max((len(staff.findall("Measure")) for staff in staves), default=0)
    starts, lengths = [], []
    cursor = 0.0
    numerator, denominator = 4, 4
    all_measures = [staff.findall("Measure") for staff in staves]
    for index in range(count):
        candidates = [rows[index] for rows in all_measures if index < len(rows)]
        for measure in candidates:
            timesig = measure.find(".//TimeSig")
            if timesig is not None:
                numerator = int(timesig.findtext("sigN") or numerator)
                denominator = int(timesig.findtext("sigD") or denominator)
                break
        normal = numerator * 4.0 / denominator
        explicit = next((m.get("len") for m in candidates if m.get("len")), None)
        length = _fraction_beats(explicit, normal) if explicit else normal
        starts.append(cursor)
        lengths.append(length)
        cursor += length
    return starts, lengths


def parse_mscx(path: str | Path) -> tuple[ET.ElementTree, dict, list[ET.Element]]:
    path = Path(path).expanduser().resolve()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    score = root.find("Score")
    if score is None:
        raise ValueError("MSCX has no Score element")
    division = int(score.findtext("Division") or 480)
    staves = _actual_staves(score)
    if not staves:
        raise ValueError("MSCX contains no notated staff measures")
    measure_starts, measure_lengths = _measure_grid(staves)

    notes: list[dict] = []
    xml_notes: list[ET.Element] = []
    chord_serial = 0
    for staff_order, staff in enumerate(staves):
        staff_id = str(staff.get("id") or staff_order + 1)
        for measure_index, measure in enumerate(staff.findall("Measure")):
            measure_start = measure_starts[measure_index]
            measure_beats = measure_lengths[measure_index]
            for voice_index, voice in enumerate(measure.findall("voice")):
                tuplets: dict[str, float] = {}
                for definition in voice.findall("Tuplet"):
                    tuplet_id = definition.get("id")
                    if not tuplet_id:
                        continue
                    actual = int(definition.findtext("actualNotes") or 1)
                    normal = int(definition.findtext("normalNotes") or 1)
                    tuplets[tuplet_id] = normal / max(1, actual)

                cursor = measure_start
                for child in list(voice):
                    if child.tag == "tick":
                        try:
                            cursor = float(child.text or 0) / division
                        except ValueError:
                            pass
                        continue
                    if child.tag == "location":
                        cursor += _fraction_beats(child.findtext("fractions"), 0.0)
                        cursor += float(child.findtext("measures") or 0) * measure_beats
                        continue
                    if child.tag not in {"Chord", "Rest"}:
                        continue
                    duration = 0.0 if child.tag == "Chord" and _is_grace(child) else _event_duration(
                        child, measure_beats, tuplets
                    )
                    if child.tag == "Chord":
                        chord_notes = child.findall("Note")
                        for note_index, note in enumerate(chord_notes):
                            pitch_text = note.findtext("pitch")
                            if pitch_text is None:
                                continue
                            midi = int(float(pitch_text))
                            tuning = float(note.findtext("tuning") or 0.0)
                            tie_start, tie_stop = _tie_flags(note)
                            event_id = len(notes)
                            notes.append({
                                "id": event_id,
                                "xml_note_index": len(xml_notes),
                                "staff_id": staff_id,
                                "staff_order": staff_order,
                                "measure_index": measure_index,
                                "voice_index": voice_index,
                                "chord_serial": chord_serial,
                                "chord_note_index": note_index,
                                "start_beats": round(cursor, 9),
                                "duration_beats": round(duration, 9),
                                "end_beats": round(cursor + duration, 9),
                                "midi_pitch": midi,
                                "tpc": int(note.findtext("tpc") or TPC_SHARP[midi % 12]),
                                "tuning_cents": tuning,
                                "source_frequency_hz": 440.0 * 2.0 ** ((midi - 69 + tuning / 100.0) / 12.0),
                                "tie_start": tie_start,
                                "tie_stop": tie_stop,
                                "attacks": not tie_stop,
                                "grace": duration <= EPS,
                            })
                            xml_notes.append(note)
                        chord_serial += 1
                    cursor += duration

    if not notes:
        raise ValueError("MSCX contains no pitched notes")
    source_bytes = path.read_bytes()
    ir = {
        "format": "MSCXPitchIR/1",
        "source_file": path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_tuning": "12-EDO note pitch plus optional MSCX tuning cents",
        "division": division,
        "staff_count": len(staves),
        "measure_count": len(measure_starts),
        "duration_beats": max((n["end_beats"] for n in notes), default=0.0),
        "note_count": len(notes),
        "notes": notes,
    }
    return tree, ir, xml_notes


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def tie_groups(notes: list[dict]) -> tuple[list[list[int]], list[int]]:
    dsu = DisjointSet(len(notes))
    lanes: dict[tuple, list[int]] = defaultdict(list)
    for note in notes:
        key = (note["staff_id"], note["voice_index"], note["midi_pitch"],
               round(note["tuning_cents"], 6))
        lanes[key].append(note["id"])
    for ids in lanes.values():
        ids.sort(key=lambda i: (notes[i]["start_beats"], notes[i]["end_beats"]))
        ending: dict[float, list[int]] = defaultdict(list)
        for i in ids:
            note = notes[i]
            key = round(note["start_beats"], 8)
            if note["tie_stop"]:
                candidates = ending.get(key, [])
                previous = next((j for j in reversed(candidates) if notes[j]["tie_start"]), None)
                if previous is not None:
                    dsu.union(previous, i)
            ending[round(note["end_beats"], 8)].append(i)
    rows: dict[int, list[int]] = defaultdict(list)
    for i in range(len(notes)):
        rows[dsu.find(i)].append(i)
    groups = list(rows.values())
    event_to_variable = [0] * len(notes)
    for variable, members in enumerate(groups):
        for event in members:
            event_to_variable[event] = variable
    return groups, event_to_variable


def nearest_scale_step(frequency: float, spec, lo: int, hi: int) -> tuple[int, int]:
    ideal = spec.edo * math.log2(float(frequency) / float(spec.base_freq_hz))
    octave = math.floor(ideal / spec.edo)
    candidates = [pc + (octave + shift) * spec.edo
                  for pc in spec.pcs for shift in range(-2, 3)]
    raw = min(candidates, key=lambda p: (abs(p - ideal), abs(p)))
    folded = raw
    while folded < lo:
        folded += spec.edo
    while folded > hi:
        folded -= spec.edo
    if not lo <= folded <= hi:
        legal = spec.make_pool(lo, hi)
        folded = min(legal, key=lambda p: abs(p - raw))
    return int(raw), int(folded)


@dataclass(frozen=True)
class Factor:
    kind: str
    variables: tuple[int, ...]
    scale: float
    data: object = None


class ImportedScoreOptimizer:
    def __init__(self, ir: dict, spec, metric: harmony_annealing.OptimizerMetric,
                 *, seed: int, max_move_cents: float, steps: int | None,
                 start_temperature: float | None, end_temperature: float | None,
                 displacement_weight: float, interval_weight: float,
                 quench_sweeps: int | None):
        self.ir = ir
        self.notes = ir["notes"]
        self.spec = spec
        self.metric = metric
        self.rng = random.Random(int(seed))
        self.seed = int(seed)
        self.max_move_cents = float(max_move_cents)
        self.groups, self.event_to_variable = tie_groups(self.notes)
        self.config = resolve_annealing(spec.style.get("annealing"))
        search = copy.deepcopy(self.config["search"])
        if steps is not None:
            search["steps"] = int(steps)
        if start_temperature is not None:
            search["start_temperature"] = float(start_temperature)
        if end_temperature is not None:
            search["end_temperature"] = float(end_temperature)
        if quench_sweeps is not None:
            search["quench_sweeps"] = int(quench_sweeps)
        self.search = search
        self.displacement_weight = float(displacement_weight)
        self.interval_weight = float(interval_weight)

        common = sorted(set(metric.pitch_index[2]) & set(metric.pitch_index[3])
                        & set(metric.pitch_index[4]))
        self.table_lo, self.table_hi = common[0], common[-1]
        pool = tuple(p for p in spec.make_pool(self.table_lo, self.table_hi) if p in set(common))
        self.nearest: list[int] = []
        self.raw_nearest: list[int] = []
        self.domains: list[tuple[int, ...]] = []
        self.folded_variables = 0
        for members in self.groups:
            reference = self.notes[members[0]]
            raw, folded = nearest_scale_step(reference["source_frequency_hz"], spec,
                                             self.table_lo, self.table_hi)
            if raw != folded:
                self.folded_variables += 1
            domain = tuple(p for p in pool
                           if abs(p - folded) * 1200.0 / spec.edo <= self.max_move_cents + EPS)
            if folded not in domain:
                domain = tuple(sorted(set(domain + (folded,))))
            self.raw_nearest.append(raw)
            self.nearest.append(folded)
            self.domains.append(domain)
        self.pitches = list(self.nearest)

        self.slices = self._sounding_slices()
        self.attacks = self._attack_groups()
        self.neighbors = [set() for _ in self.groups]
        for _a, _b, variables in self.slices:
            unique = tuple(sorted(set(variables)))
            for x, y in itertools.combinations(unique, 2):
                self.neighbors[x].add(y)
                self.neighbors[y].add(x)
        self._make_wolf_free_initial()
        self.initial_pitches = list(self.pitches)

        self.factors: list[Factor] = []
        self.incident = [set() for _ in self.groups]
        self._build_factors()
        self.values = [self.evaluate(factor) for factor in self.factors]
        self.total = math.fsum(self.values)

    def _sounding_slices(self) -> list[tuple[float, float, tuple[int, ...]]]:
        starts: dict[float, list[int]] = defaultdict(list)
        ends: dict[float, list[int]] = defaultdict(list)
        for note in self.notes:
            if note["duration_beats"] <= EPS:
                continue
            starts[note["start_beats"]].append(note["id"])
            ends[note["end_beats"]].append(note["id"])
        points = sorted(set(starts) | set(ends))
        active: set[int] = set()
        slices = []
        for pos, point in enumerate(points[:-1]):
            for event in ends.get(point, ()):
                active.discard(event)
            active.update(starts.get(point, ()))
            nxt = points[pos + 1]
            if nxt - point > EPS and active:
                variables = tuple(self.event_to_variable[e] for e in sorted(active))
                slices.append((point, nxt, variables))
        return slices

    def _attack_groups(self) -> list[tuple[int, ...]]:
        groups: dict[float, list[int]] = defaultdict(list)
        for note in self.notes:
            if note["attacks"] and not note["grace"]:
                groups[note["start_beats"]].append(self.event_to_variable[note["id"]])
        return [tuple(rows) for _, rows in sorted(groups.items()) if len(rows) >= 2]

    def _conflict_count(self, variable: int, pitch: int) -> int:
        return sum(hr.hard_wolf(pitch, self.pitches[other]) for other in self.neighbors[variable])

    def _wolf_edges(self) -> list[tuple[int, int]]:
        return [(i, j) for i, rows in enumerate(self.neighbors) for j in rows
                if i < j and hr.hard_wolf(self.pitches[i], self.pitches[j])]

    def _make_wolf_free_initial(self) -> None:
        if not self._wolf_edges():
            return
        attempts = max(2000, 30 * len(self.groups))
        for restart in range(8):
            if restart:
                self.pitches = [self.rng.choice(domain) for domain in self.domains]
            for _ in range(attempts):
                bad = self._wolf_edges()
                if not bad:
                    return
                counts: dict[int, int] = defaultdict(int)
                for a, b in bad:
                    counts[a] += 1
                    counts[b] += 1
                worst = max(counts.values())
                choices = [v for v, count in counts.items() if count == worst]
                variable = self.rng.choice(choices)
                old = self.pitches[variable]
                scored = []
                for candidate in self.domains[variable]:
                    conflicts = self._conflict_count(variable, candidate)
                    distance = abs(candidate - self.nearest[variable])
                    scored.append((conflicts, distance, self.rng.random(), candidate))
                self.pitches[variable] = min(scored)[-1]
                if self.pitches[variable] == old and len(self.domains[variable]) > 1:
                    alternatives = [p for p in self.domains[variable] if p != old]
                    self.pitches[variable] = self.rng.choice(alternatives)
        raise RuntimeError("No wolf-free mapping exists inside the configured quarter-octave domains")

    def add_factor(self, factor: Factor) -> None:
        index = len(self.factors)
        self.factors.append(factor)
        for variable in set(factor.variables):
            self.incident[variable].add(index)

    def _build_factors(self) -> None:
        weights = self.config["objective_weights"]
        duration = max(float(self.ir["duration_beats"]), EPS)
        for start, end, variables in self.slices:
            if len(variables) >= 2:
                self.add_factor(Factor("sounding", variables,
                                       weights["sounding"] * (end - start) / duration))
        if self.attacks:
            scale = weights["attack"] / len(self.attacks)
            for variables in self.attacks:
                self.add_factor(Factor("attack", variables, scale))
        if self.groups:
            scale = weights["voice_leading"] * self.displacement_weight / len(self.groups)
            for variable in range(len(self.groups)):
                self.add_factor(Factor("displacement", (variable,), scale,
                                       self.nearest[variable]))

        lanes: dict[tuple, list[int]] = defaultdict(list)
        for note in self.notes:
            if note["tie_stop"]:
                continue
            key = (note["staff_id"], note["voice_index"], note["chord_note_index"])
            lanes[key].append(note["id"])
        transitions = []
        for rows in lanes.values():
            rows.sort(key=lambda event: self.notes[event]["start_beats"])
            variables = [self.event_to_variable[event] for event in rows]
            transitions.extend((a, b) for a, b in zip(variables, variables[1:]) if a != b)
        if transitions:
            scale = weights["voice_leading"] * self.interval_weight / len(transitions)
            for a, b in transitions:
                expected = self.nearest[b] - self.nearest[a]
                self.add_factor(Factor("interval", (a, b), scale, expected))

    @lru_cache(maxsize=300000)
    def _sounding_cost(self, pitches: tuple[int, ...]) -> float:
        pitches = tuple(sorted(pitches))
        if len(pitches) < 2:
            return 0.0
        if len(pitches) <= 4:
            return self.metric.sounding_cost(pitches)
        # Replace the unavailable native >4-note blend with one scalar: the
        # mean corrected blend of every four-note subset.  Apply the sounding
        # loss shape once to that mean, exactly as if it were the true CSE.
        values = [self.metric.corrected(tuple(subset))
                  for subset in itertools.combinations(pitches, 4)]
        proxy = math.fsum(values) / len(values)
        return (self.metric.cse_loss("sounding", proxy)
                - self.metric.prime_reward(pitches))

    @lru_cache(maxsize=200000)
    def _attack_cost(self, pitches: tuple[int, ...]) -> float:
        pitches = tuple(sorted(pitches))
        if len(pitches) < 2:
            return 0.0
        if len(pitches) <= 4:
            return self.metric.attack_cost(pitches)
        values = [self.metric.attack_cost(tuple(subset))
                  for subset in itertools.combinations(pitches, 4)]
        return math.fsum(values) / len(values)

    def evaluate(self, factor: Factor) -> float:
        pitches = tuple(self.pitches[v] for v in factor.variables)
        if factor.kind == "sounding":
            value = self._sounding_cost(tuple(sorted(pitches)))
        elif factor.kind == "attack":
            value = self._attack_cost(tuple(sorted(pitches)))
        elif factor.kind == "displacement":
            cents = (pitches[0] - int(factor.data)) * 1200.0 / self.spec.edo
            value = (cents / self.max_move_cents) ** 2
        elif factor.kind == "interval":
            error = (pitches[1] - pitches[0] - int(factor.data)) * 1200.0 / self.spec.edo
            value = (error / self.max_move_cents) ** 2
        else:
            raise ValueError(f"unknown factor: {factor.kind}")
        return factor.scale * value

    def legal(self, moves: dict[int, int]) -> bool:
        for variable, pitch in moves.items():
            if pitch not in self.domains[variable]:
                return False
            for other in self.neighbors[variable]:
                if hr.hard_wolf(pitch, moves.get(other, self.pitches[other])):
                    return False
        return True

    def _proposal(self) -> dict[int, int] | None:
        if self.attacks and self.rng.random() < 0.7:
            group = tuple(sorted(set(self.rng.choice(self.attacks))))
            max_size = min(3, len(group))
            size = self.rng.choices(range(1, max_size + 1),
                                    weights=(0.68, 0.24, 0.08)[:max_size])[0]
            variables = self.rng.sample(group, size)
        else:
            variables = [self.rng.randrange(len(self.groups))]
        moves = {}
        local = self.rng.random() < self.search["local_probability"]
        radius = int(self.search["local_radius_degrees"])
        for variable in variables:
            current = self.pitches[variable]
            options = list(self.domains[variable])
            if local:
                current_degree = hr.degree(current)
                options = [p for p in options if abs(hr.degree(p) - current_degree) <= radius]
            options = [p for p in options if p != current]
            if not options:
                continue
            moves[variable] = self.rng.choice(options)
        return moves if moves and self.legal(moves) else None

    def _try(self, moves: dict[int, int], temperature: float) -> bool:
        affected = set()
        for variable in moves:
            affected.update(self.incident[variable])
        before = math.fsum(self.values[index] for index in affected)
        old = {variable: self.pitches[variable] for variable in moves}
        for variable, pitch in moves.items():
            self.pitches[variable] = pitch
        after_values = {index: self.evaluate(self.factors[index]) for index in affected}
        after = math.fsum(after_values.values())
        delta = after - before
        accept = delta <= 0.0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-15))
        if accept:
            for index, value in after_values.items():
                self.values[index] = value
            self.total += delta
        else:
            for variable, pitch in old.items():
                self.pitches[variable] = pitch
        return accept

    def energy_sources(self) -> dict[str, float]:
        rows: dict[str, list[float]] = defaultdict(list)
        for factor, value in zip(self.factors, self.values):
            rows[factor.kind].append(value)
        return {kind: math.fsum(values) for kind, values in sorted(rows.items())} | {
            "total": math.fsum(self.values)
        }

    def optimize(self) -> dict:
        steps = int(self.search["steps"])
        start_t = float(self.search["start_temperature"])
        end_t = float(self.search["end_temperature"])
        progress = int(self.search["progress_every"])
        initial_energy = self.total
        initial_sources = self.energy_sources()
        best_energy = self.total
        best = list(self.pitches)
        accepted = attempted = 0
        started = time.time()
        for step in range(steps):
            fraction = step / max(1, steps - 1)
            temperature = start_t * (end_t / start_t) ** fraction
            moves = self._proposal()
            if moves is None:
                continue
            attempted += 1
            if self._try(moves, temperature):
                accepted += 1
                if self.total < best_energy:
                    best_energy = self.total
                    best = list(self.pitches)
            if progress and (step + 1) % progress == 0:
                print(f"anneal {step + 1}/{steps}: energy={self.total:.8f}, "
                      f"best={best_energy:.8f}, accepted={accepted/max(1, attempted):.1%}",
                      flush=True)
        self.pitches = best
        self.values = [self.evaluate(factor) for factor in self.factors]
        self.total = math.fsum(self.values)

        quench_changes = 0
        for _ in range(int(self.search["quench_sweeps"])):
            changed = 0
            order = list(range(len(self.groups)))
            self.rng.shuffle(order)
            for variable in order:
                current = self.pitches[variable]
                best_pitch = current
                best_delta = 0.0
                affected = set(self.incident[variable])
                before = math.fsum(self.values[index] for index in affected)
                for pitch in self.domains[variable]:
                    if pitch == current or not self.legal({variable: pitch}):
                        continue
                    self.pitches[variable] = pitch
                    after = math.fsum(self.evaluate(self.factors[index]) for index in affected)
                    delta = after - before
                    if delta < best_delta:
                        best_delta, best_pitch = delta, pitch
                self.pitches[variable] = current
                if best_pitch != current:
                    self._try({variable: best_pitch}, 0.0)
                    changed += 1
            quench_changes += changed
            if not changed:
                break
        return {
            "seed": self.seed,
            "steps": steps,
            "accepted_moves": accepted,
            "attempted_moves": attempted,
            "acceptance_ratio": accepted / max(1, attempted),
            "initial_energy": initial_energy,
            "best_energy_before_quench": best_energy,
            "final_energy": self.total,
            "initial_energy_sources": initial_sources,
            "final_energy_sources": self.energy_sources(),
            "quench_changes": quench_changes,
            "elapsed_seconds": time.time() - started,
        }

    def event_steps(self) -> list[int]:
        return [self.pitches[self.event_to_variable[note["id"]]] for note in self.notes]

    def validation(self) -> dict:
        wolves = []
        for start, end, variables in self.slices:
            pitches = [self.pitches[v] for v in variables]
            for i, j in itertools.combinations(range(len(pitches)), 2):
                if hr.hard_wolf(pitches[i], pitches[j]):
                    wolves.append({"start": start, "end": end,
                                  "steps": [pitches[i], pitches[j]]})
        displacement = [abs(p - q) * 1200.0 / self.spec.edo
                        for p, q in zip(self.pitches, self.nearest)]
        return {
            "wolf_interval_count": len(wolves),
            "wolf_intervals": wolves[:20],
            "max_annealing_movement_cents": max(displacement, default=0.0),
            "movement_limit_cents": self.max_move_cents,
            "pitch_domain_violations": sum(
                pitch not in domain for pitch, domain in zip(self.pitches, self.domains)
            ),
            "source_note_count": len(self.notes),
            "output_note_count": len(self.event_steps()),
            "octave_folded_tie_groups": self.folded_variables,
        }

    def _raw_subset_metrics(self, pitches: tuple[int, ...]) -> dict[str, float]:
        pitches = tuple(sorted(pitches))
        if len(pitches) < 2:
            return {"se": 0.0, "cse": 0.0, "ccse": 0.0, "blending": 0.0}
        subsets = (pitches,) if len(pitches) <= 4 else itertools.combinations(pitches, 4)
        rows = []
        for subset in subsets:
            values = self.metric.bundle.metrics_entry(tuple(subset), table_key=f"bass{len(subset)}")
            if values is None:
                continue
            rows.append({name: values[name][0] for name in ("se", "cse", "ccse")})
        if not rows:
            raise RuntimeError(f"No spectral entry for pitches {pitches}")
        result = {name: math.fsum(row[name] for row in rows) / len(rows)
                  for name in ("se", "cse", "ccse")}
        a, b, c = self.metric.weights
        result["blending"] = a * result["cse"] + b * result["se"] + c * result["ccse"]
        return result

    def spectral_summary(self) -> dict:
        totals = {name: 0.0 for name in ("se", "cse", "ccse", "blending")}
        weight = 0.0
        count = 0
        for start, end, variables in self.slices:
            if len(variables) < 2:
                continue
            row = self._raw_subset_metrics(tuple(self.pitches[v] for v in variables))
            span = end - start
            for name in totals:
                totals[name] += span * row[name]
            weight += span
            count += 1
        return {
            "aggregation": "duration-weighted; >4 notes use mean of all four-note subsets",
            "slice_count": count,
            "weight_beats": weight,
            **{name: totals[name] / max(weight, EPS) for name in totals},
            "blend_weights": {
                "CSE_2D_A": self.metric.weights[0],
                "CSE_2D_B": self.metric.weights[1],
                "CSE_2D_C": self.metric.weights[2],
            },
        }


def step_notation(step: int, spec) -> tuple[int, int, float, float]:
    frequency = spec.base_freq_hz * 2.0 ** (float(step) / spec.edo)
    midi_float = 69.0 + 12.0 * math.log2(frequency / 440.0)
    midi = int(round(midi_float))
    midi = min(127, max(0, midi))
    tuning = 100.0 * (midi_float - midi)
    return midi, TPC_SHARP[midi % 12], tuning, frequency


def rewrite_mscx(tree: ET.ElementTree, xml_notes: list[ET.Element], steps: list[int],
                 spec, output: str | Path) -> None:
    if len(xml_notes) != len(steps):
        raise ValueError("XML note count changed before export")
    for note, step in zip(xml_notes, steps):
        midi, tpc, tuning, _frequency = step_notation(step, spec)
        pitch_node = note.find("pitch")
        if pitch_node is None:
            pitch_node = ET.SubElement(note, "pitch")
        pitch_node.text = str(midi)
        tpc_node = note.find("tpc")
        if tpc_node is None:
            tpc_node = ET.SubElement(note, "tpc")
        tpc_node.text = str(tpc)
        tuning_node = note.find("tuning")
        if tuning_node is None:
            tuning_node = ET.SubElement(note, "tuning")
        tuning_node.text = f"{tuning:.6f}"
        for accidental in list(note.findall("Accidental")):
            note.remove(accidental)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="UTF-8", xml_declaration=True, short_empty_elements=True)


def converted_payload(ir: dict, optimizer: ImportedScoreOptimizer, spec,
                      annealing_result: dict) -> dict:
    steps = optimizer.event_steps()
    notes = []
    for source, step in zip(ir["notes"], steps):
        variable = optimizer.event_to_variable[source["id"]]
        midi, tpc, tuning, frequency = step_notation(step, spec)
        notes.append({
            **source,
            "tie_group": variable,
            "nearest_target_step": optimizer.nearest[variable],
            "initial_wolf_free_step": optimizer.initial_pitches[variable],
            "target_step": step,
            "target_pitch_class": step % spec.edo,
            "target_name": spec.names[spec.pcs.index(step % spec.edo)],
            "target_frequency_hz": frequency,
            "output_midi_pitch": midi,
            "output_tpc": tpc,
            "output_tuning_cents": tuning,
            "annealing_movement_cents": (
                step - optimizer.nearest[variable]
            ) * 1200.0 / spec.edo,
        })
    return {
        "format": "MSCXScaleConversion/1",
        "source": {key: ir[key] for key in (
            "source_file", "source_sha256", "division", "staff_count",
            "measure_count", "duration_beats", "note_count"
        )},
        "target_scale": {
            "id": spec.id,
            "name": spec.name,
            "edo": spec.edo,
            "pcs": list(spec.pcs),
            "names": list(spec.names),
            "base_note": spec.base_note,
            "base_freq_hz": spec.base_freq_hz,
        },
        "conversion": {
            "nearest_mapping": "absolute-frequency nearest target-scale pitch",
            "out_of_range_policy": "whole-octave folding into common bass2/3/4 CSE domain",
            "max_annealing_movement_cents": optimizer.max_move_cents,
            "ties_share_one_pitch_variable": True,
            "note_addition_or_deletion": False,
            "polyphony_over_four": "mean of all four-note subset CSE objectives",
        },
        "annealing": annealing_result,
        "spectral_metrics": optimizer.spectral_summary(),
        "validation": optimizer.validation(),
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mscx")
    parser.add_argument("--scale", required=True, help="target ScaleDefinition JSON")
    parser.add_argument("--rules", help="target CompositionRules JSON")
    parser.add_argument("--style", help="target CompositionStyle JSON")
    parser.add_argument("--cse-dir", help="spectral/JI cache directory; defaults to scale directory")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", help="output basename; default: <input>_<scale-id>")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--start-temperature", type=float)
    parser.add_argument("--end-temperature", type=float)
    parser.add_argument("--quench-sweeps", type=int)
    parser.add_argument("--max-move-cents", type=float, default=300.0)
    parser.add_argument("--displacement-weight", type=float, default=0.12)
    parser.add_argument("--interval-weight", type=float, default=0.35)
    parser.add_argument("--workers", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_move_cents <= 0 or args.max_move_cents > 300.0 + EPS:
        raise ValueError("--max-move-cents must be in (0, 300]")
    source_path = Path(args.input_mscx).expanduser().resolve()
    scale_path = Path(args.scale).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree, ir, xml_notes = parse_mscx(source_path)
    spec, _manifest = composer._configure_adaptive_scale(
        scale_path,
        Path(args.cse_dir).expanduser().resolve() if args.cse_dir else scale_path.parent,
        rules_config=args.rules,
        style_config=args.style,
        cse_workers=args.workers,
    )
    config = resolve_annealing(spec.style.get("annealing"))
    metric = harmony_annealing.OptimizerMetric(spec, config)
    optimizer = ImportedScoreOptimizer(
        ir, spec, metric,
        seed=args.seed,
        max_move_cents=args.max_move_cents,
        steps=args.steps,
        start_temperature=args.start_temperature,
        end_temperature=args.end_temperature,
        displacement_weight=args.displacement_weight,
        interval_weight=args.interval_weight,
        quench_sweeps=args.quench_sweeps,
    )

    prefix = args.prefix or f"{source_path.stem}_{spec.safe_id}"
    source_ir_path = output_dir / f"{prefix}.source_ir.json"
    converted_json_path = output_dir / f"{prefix}.converted.json"
    converted_mscx_path = output_dir / f"{prefix}.converted.mscx"
    source_ir_path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
    result = optimizer.optimize()
    payload = converted_payload(ir, optimizer, spec, result)
    validation = payload["validation"]
    if (validation["wolf_interval_count"] or validation["pitch_domain_violations"]
            or validation["source_note_count"] != validation["output_note_count"]
            or validation["max_annealing_movement_cents"] > args.max_move_cents + 1e-6):
        raise RuntimeError(f"conversion validation failed: {validation}")
    converted_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rewrite_mscx(tree, xml_notes, optimizer.event_steps(), spec, converted_mscx_path)
    print(f"Source IR: {source_ir_path}")
    print(f"Converted JSON: {converted_json_path}")
    print(f"Converted MSCX: {converted_mscx_path}")
    print(f"Validation: {validation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
