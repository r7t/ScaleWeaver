#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScaleWeaver SoundFont2/FluidSynth music synthesizer.

Handles per-note microtonal pitch bends and optional SF2/raw percussion.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from raw_synth import DEFAULT_SR, OUTPUT_PEAK, score_bpm, synth_percussion

DEFAULT_SF2_MAP: dict[str, Any] = {
    "default": {"bank": 0, "program": 0, "gain": 1.0},
    "roles": {
        # General MIDI program numbers are ZERO-BASED here.
        "bass":    {"bank": 0, "program": 32, "gain": 1.00},  # Acoustic Bass
        "inner":   {"bank": 0, "program": 48, "gain": 1.00},  # String Ensemble 1
        "counter":  {"bank": 0, "program": 71, "gain": 1.00},  # Clarinet
        "lead":    {"bank": 0, "program": 73, "gain": 1.00},  # Flute
    },
    "instruments": {
        "contrabass":   {"bank": 0, "program": 43, "gain": 1.00},
        "plucked_bass": {"bank": 0, "program": 32, "gain": 1.00},
        "cello_pad":    {"bank": 0, "program": 48, "gain": 0.82},
        "warm_keys":    {"bank": 0, "program": 4,  "gain": 0.90},
        "glass_keys":   {"bank": 0, "program": 8,  "gain": 0.85},
        "clarinet_like": {"bank": 0, "program": 71, "gain": 0.95},
        "flute_like":   {"bank": 0, "program": 73, "gain": 0.95},
        "warm_reed":    {"bank": 0, "program": 68, "gain": 0.92},
        "bright_reed":  {"bank": 0, "program": 64, "gain": 0.95},
        "flute_bell":   {"bank": 0, "program": 8,  "gain": 0.90},
    },
}

GM_DRUM_KEYS = {
    "frame_drum": 45,  # Low Tom
    "wood_click": 76,  # Hi Wood Block
    "shaker": 70,      # Maracas
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_sf2_map(path: str | Path | None) -> dict:
    mapping = DEFAULT_SF2_MAP
    if path is None:
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        raise ValueError("SF2 map must be a JSON object.")
    return _deep_merge(mapping, user)


def _normalize_preset(spec: dict | None) -> dict:
    spec = dict(spec or {})
    if "program_1based" in spec and "program" not in spec:
        spec["program"] = int(spec["program_1based"]) - 1
    return {
        "bank": int(spec.get("bank", 0)),
        "program": int(spec.get("program", 0)),
        "gain": float(spec.get("gain", 1.0)),
    }


def resolve_sf2_preset(
    event: dict,
    mapping: dict,
    force_bank: int | None,
    force_program: int | None,
) -> dict:
    if isinstance(event.get("sf2"), dict):
        return _normalize_preset(event["sf2"])
    if force_program is not None:
        return {"bank": int(force_bank or 0), "program": int(force_program), "gain": 1.0}
    inst = event.get("instrument")
    if inst and inst in mapping.get("instruments", {}):
        return _normalize_preset(mapping["instruments"][inst])
    role = event.get("role")
    if role and role in mapping.get("roles", {}):
        return _normalize_preset(mapping["roles"][role])
    return _normalize_preset(mapping.get("default", {}))


class FluidSynthC:
    """Small ctypes wrapper around the subset of libfluidsynth used here."""

    def __init__(
        self,
        sf2_path: str | Path,
        sr: int,
        midi_channels: int = 128,
        synth_gain: float = 0.55,
        reverb: bool = False,
        chorus: bool = False,
        lib_path: str | None = None,
    ) -> None:
        self.sr = int(sr)
        self.midi_channels = int(midi_channels)
        self.lib = self._load_lib(lib_path)
        self._bind()

        self.settings = self.lib.new_fluid_settings()
        if not self.settings:
            raise RuntimeError("new_fluid_settings() failed")
        self._setnum("synth.sample-rate", float(sr))
        self._setint("synth.midi-channels", int(midi_channels))
        self._setnum("synth.gain", float(synth_gain))
        self._setint("synth.reverb.active", 1 if reverb else 0)

        if reverb:
            self._setnum("synth.reverb.room-size", 0.18)
            self._setnum("synth.reverb.damp", 0.45)
            self._setnum("synth.reverb.width", 0.8)
            self._setnum("synth.reverb.level", 0.18)
        self._setint("synth.chorus.active", 1 if chorus else 0)

        self.synth = self.lib.new_fluid_synth(self.settings)
        if not self.synth:
            self.close()
            raise RuntimeError("new_fluid_synth() failed")

        sf2_path = Path(sf2_path)
        if not sf2_path.is_file():
            self.close()
            raise FileNotFoundError(sf2_path)
        self.sfid = int(self.lib.fluid_synth_sfload(
            self.synth, os.fsencode(str(sf2_path)), 1
        ))
        if self.sfid < 0:
            self.close()
            raise RuntimeError(f"FluidSynth could not load SoundFont: {sf2_path}")

    @staticmethod
    def _load_lib(explicit: str | None):
        candidates = [
            explicit,
            os.environ.get("FLUIDSYNTH_LIB"),
            ctypes.util.find_library("fluidsynth"),
            "libfluidsynth.so.3",
            "libfluidsynth.so",
            "libfluidsynth.dylib",
            "fluidsynth.dll",
            "libfluidsynth-3.dll",
            r"E:\fluidsynth-v2.6.0-win10-x64-cpp11\bin\libfluidsynth-3.dll"
        ]
        errors = []
        for p in candidates:
            if not p:
                continue
            try:
                return C.CDLL(p)
            except OSError as exc:
                errors.append(f"{p}: {exc}")
        msg = "\n".join(errors[-4:])
        raise RuntimeError(
            "libfluidsynth was not found. Install FluidSynth/libfluidsynth, "
            "or pass --fluidsynth-lib PATH / set FLUIDSYNTH_LIB.\n" + msg
        )

    def _bind(self) -> None:
        L = self.lib
        L.new_fluid_settings.restype = C.c_void_p
        L.delete_fluid_settings.argtypes = [C.c_void_p]
        L.fluid_settings_setnum.argtypes = [C.c_void_p, C.c_char_p, C.c_double]
        L.fluid_settings_setnum.restype = C.c_int
        L.fluid_settings_setint.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.fluid_settings_setint.restype = C.c_int

        L.new_fluid_synth.argtypes = [C.c_void_p]
        L.new_fluid_synth.restype = C.c_void_p
        L.delete_fluid_synth.argtypes = [C.c_void_p]

        L.fluid_synth_sfload.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        L.fluid_synth_sfload.restype = C.c_int
        L.fluid_synth_program_select.argtypes = [
            C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int
        ]
        L.fluid_synth_program_select.restype = C.c_int
        L.fluid_synth_noteon.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int]
        L.fluid_synth_noteon.restype = C.c_int
        L.fluid_synth_noteoff.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.fluid_synth_noteoff.restype = C.c_int
        L.fluid_synth_pitch_bend.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.fluid_synth_pitch_bend.restype = C.c_int
        L.fluid_synth_pitch_wheel_sens.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.fluid_synth_pitch_wheel_sens.restype = C.c_int
        L.fluid_synth_write_float.argtypes = [
            C.c_void_p, C.c_int,
            C.c_void_p, C.c_int, C.c_int,
            C.c_void_p, C.c_int, C.c_int,
        ]
        L.fluid_synth_write_float.restype = C.c_int

    def _setnum(self, name: str, value: float) -> None:
        rc = self.lib.fluid_settings_setnum(self.settings, name.encode(), float(value))
        if rc != 0:
            raise RuntimeError(f"FluidSynth setting rejected: {name}={value}")

    def _setint(self, name: str, value: int) -> None:
        rc = self.lib.fluid_settings_setint(self.settings, name.encode(), int(value))
        if rc != 0:
            # Some builds omit optional settings; only sample-rate/channel count are
            # structurally required.  Reverb/chorus settings are allowed to fail.
            if name in {"synth.sample-rate", "synth.midi-channels"}:
                raise RuntimeError(f"FluidSynth setting rejected: {name}={value}")

    def program_select(self, chan: int, bank: int, program: int) -> int:
        return int(self.lib.fluid_synth_program_select(
            self.synth, int(chan), self.sfid, int(bank), int(program)
        ))

    def set_bend(self, chan: int, bend14: int, range_semitones: int) -> None:
        if self.lib.fluid_synth_pitch_wheel_sens(
            self.synth, int(chan), int(range_semitones)
        ) != 0:
            raise RuntimeError(f"Could not set pitch-bend range on channel {chan}")
        if self.lib.fluid_synth_pitch_bend(
            self.synth, int(chan), int(bend14)
        ) != 0:
            raise RuntimeError(f"Could not set pitch bend on channel {chan}")

    def noteon(self, chan: int, key: int, vel: int) -> None:
        if self.lib.fluid_synth_noteon(
            self.synth, int(chan), int(key), int(vel)
        ) != 0:
            raise RuntimeError(f"FluidSynth noteon failed: ch={chan} key={key}")

    def noteoff(self, chan: int, key: int) -> None:
        self.lib.fluid_synth_noteoff(self.synth, int(chan), int(key))

    def render(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        out = np.empty((n, 2), dtype=np.float32)
        # Use temporary contiguous mono buffers; interleaved column slices are not
        # contiguous and therefore cannot be passed as increment=1.
        left = np.empty(n, dtype=np.float32)
        right = np.empty(n, dtype=np.float32)
        rc = self.lib.fluid_synth_write_float(
            self.synth, int(n),
            left.ctypes.data_as(C.c_void_p), 0, 1,
            right.ctypes.data_as(C.c_void_p), 0, 1,
        )
        if rc != 0:
            raise RuntimeError("fluid_synth_write_float() failed")
        out[:, 0] = left
        out[:, 1] = right
        return out.astype(np.float64, copy=False)

    def close(self) -> None:
        synth = getattr(self, "synth", None)
        settings = getattr(self, "settings", None)
        if synth:
            self.lib.delete_fluid_synth(synth)
            self.synth = None
        if settings:
            self.lib.delete_fluid_settings(settings)
            self.settings = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@dataclass
class _ChannelState:
    active_event: int | None = None
    key: int | None = None
    free_after_sample: int = 0


class MicrotonalChannelAllocator:
    """One independently bent MIDI channel per active ScaleWeaver note."""

    def __init__(self, synth: FluidSynthC, release_hold_samples: int):
        self.synth = synth
        # MIDI channel 10 (index 9) of every 16-channel group may be treated as a
        # percussion channel by GM-aware SoundFonts, so keep those out of the
        # pitched-note pool.
        self.channels = [
            ch for ch in range(synth.midi_channels) if (ch % 16) != 9
        ]
        self.state = {ch: _ChannelState() for ch in self.channels}
        self.event_channel: dict[int, int] = {}
        self.release_hold_samples = int(release_hold_samples)
        self.peak_active = 0

    def allocate(self, event_id: int, now_sample: int) -> int:
        free = [
            ch for ch in self.channels
            if self.state[ch].active_event is None
            and self.state[ch].free_after_sample <= now_sample
        ]
        if not free:
            # Prefer a released channel whose protected tail ends soonest.  This
            # should be extremely rare with the default 128 FluidSynth channels.
            released = [
                ch for ch in self.channels
                if self.state[ch].active_event is None
            ]
            if released:
                ch = min(released, key=lambda c: self.state[c].free_after_sample)
            else:
                raise RuntimeError(
                    "SF2 microtonal channel pool exhausted. Increase --sf2-channels."
                )
        else:
            ch = free[0]
        self.state[ch].active_event = event_id
        self.event_channel[event_id] = ch
        self.peak_active = max(
            self.peak_active,
            sum(s.active_event is not None for s in self.state.values())
        )
        return ch

    def release(self, event_id: int, now_sample: int) -> tuple[int, int] | None:
        ch = self.event_channel.pop(event_id, None)
        if ch is None:
            return None
        st = self.state[ch]
        key = st.key
        st.active_event = None
        st.key = None
        st.free_after_sample = now_sample + self.release_hold_samples
        return ch, key if key is not None else -1


def freq_to_midi_bend(freq: float, bend_range_semitones: float) -> tuple[int, int, float]:
    if not (freq > 0.0 and math.isfinite(freq)):
        raise ValueError(f"Invalid event frequency: {freq}")
    midi_float = 69.0 + 12.0 * math.log2(freq / 440.0)
    key = int(round(midi_float))
    if not 0 <= key <= 127:
        raise ValueError(
            f"Frequency {freq:.6f} Hz maps outside MIDI 0..127 (nearest key {key})."
        )
    delta = midi_float - key;
    if abs(delta) > bend_range_semitones + 1e-12:
        raise ValueError(
            f"Pitch offset {delta:.4f} semitones exceeds bend range +/-{bend_range_semitones}."
        )
    # FluidSynth uses 0..16383 with center 8192.
    bend = int(round(8192 + (delta / bend_range_semitones) * 8192))
    bend = max(0, min(16383, bend))
    cents = delta * 100.0
    return key, bend, cents

def _all_score_events(score: dict) -> list[dict]:
    voices = score.get("voices", {})
    if not voices:
        raise ValueError("Score JSON has no 'voices' dictionary.")
    out = []
    for voice_name, events in voices.items():
        for e in events:
            x = dict(e)
            x.setdefault("role", voice_name)
            out.append(x)
    return out


def _mix_raw_percussion(
    stereo: np.ndarray,
    events: list[dict],
    sec_per_beat: float,
    sr: int,
) -> None:
    for e in events:
        if "freq" in e and e.get("role") != "percussion":
            continue
        start_sec = float(e["start_beat"]) * sec_per_beat
        dur_sec = float(e["duration_beats"]) * sec_per_beat
        x = synth_percussion(e, dur_sec, sr)
        i0 = int(round(start_sec * sr))
        i1 = min(len(stereo), i0 + len(x))
        if i1 > i0:
            stereo[i0:i1, 0] += x[:i1-i0]
            stereo[i0:i1, 1] += x[:i1-i0]


def render_score_sf2(
    score: dict,
    sf2_path: str | Path,
    sr: int = DEFAULT_SR,
    sf2_map_path: str | Path | None = None,
    force_bank: int | None = None,
    force_program: int | None = None,
    synth_gain: float = 0.55,
    velocity_scale: float = 1.0,
    bend_range_semitones: int = 2,
    midi_channels: int = 128,
    release_hold_sec: float = 0.8,
    tail_sec: float = 1.5,
    reverb: bool = False,
    chorus: bool = False,
    percussion_backend: str = "raw",
    mono: bool = False,
    fluidsynth_lib: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    bpm = score_bpm(score)
    sec_per_beat = 60.0 / bpm
    events = _all_score_events(score)
    pitched = [e for e in events if "freq" in e and e.get("role") != "percussion"]
    percussion = [e for e in events if e not in pitched]
    mapping = load_sf2_map(sf2_map_path)

    max_end_beat = max(
        (float(e["start_beat"]) + float(e["duration_beats"]) for e in events),
        default=0.0,
    )
    total_sec = max_end_beat * sec_per_beat + max(.05, float(tail_sec))
    total_samples = max(1, int(math.ceil(total_sec * sr)))
    mix = np.zeros((total_samples, 2), dtype=np.float64)

    # action tuple: (sample, priority, kind, event_id, event)
    # note-off priority 0 before note-on priority 1 at the same sample.
    actions = []
    max_abs_cents = 0.0
    for eid, e in enumerate(pitched):
        s0 = int(round(float(e["start_beat"]) * sec_per_beat * sr))
        s1 = int(round(
            (float(e["start_beat"]) + float(e["duration_beats"]))
            * sec_per_beat * sr
        ))
        key, bend, cents = freq_to_midi_bend(float(e["freq"]), bend_range_semitones)
        max_abs_cents = max(max_abs_cents, abs(cents))
        meta = dict(e)
        meta["_midi_key"] = key
        meta["_bend"] = bend
        meta["_cents"] = cents
        actions.append((s0, 1, "on", eid, meta))
        actions.append((max(s0 + 1, s1), 0, "off", eid, meta))

    # SF2 GM drums are deliberately optional.  Raw percussion remains the default
    # because the score's frame_drum/wood_click/shaker semantics predate GM kits.
    drum_events = []
    if percussion_backend == "sf2":
        for j, e in enumerate(percussion):
            s0 = int(round(float(e["start_beat"]) * sec_per_beat * sr))
            inst = e.get("instrument", "wood_click")
            key = int(e.get("sf2_drum_key", GM_DRUM_KEYS.get(inst, 76)))
            drum_events.append((s0, key, e))

    actions.sort(key=lambda x: (x[0], x[1]))
    release_hold_samples = int(round(max(0.0, release_hold_sec) * sr))

    with FluidSynthC(
        sf2_path=sf2_path,
        sr=sr,
        midi_channels=midi_channels,
        synth_gain=synth_gain,
        reverb=reverb,
        chorus=chorus,
        lib_path=fluidsynth_lib,
    ) as fs:
        alloc = MicrotonalChannelAllocator(fs, release_hold_samples)

        # Configure the dedicated GM percussion channels if requested.
        drum_channels = [ch for ch in range(midi_channels) if (ch % 16) == 9]
        drum_channel = drum_channels[0] if drum_channels else None
        if percussion_backend == "sf2":
            if drum_channel is None:
                raise RuntimeError("No GM percussion channel is available.")
            # GM drum kit: bank 128, program 0 is conventional.  If the SoundFont
            # does not provide it, FluidSynth may fall back internally.
            fs.program_select(drum_channel, 128, 0)

        # Merge drum note-ons into the chronological action stream.  Drum notes
        # are one-shot; we do not need pitch bend or a dedicated noteoff schedule.
        for k, (s0, key, e) in enumerate(drum_events):
            actions.append((s0, 2, "drum", -(k + 1), {**e, "_midi_key": key}))
        actions.sort(key=lambda x: (x[0], x[1]))

        cur = 0
        chunk = 4096
        for sample, _prio, kind, eid, e in actions:
            sample = min(max(0, sample), total_samples)
            while cur < sample:
                n = min(chunk, sample - cur)
                mix[cur:cur+n] += fs.render(n)
                cur += n

            if kind == "off":
                released = alloc.release(eid, sample)
                if released is not None:
                    ch, key = released
                    if key >= 0:
                        fs.noteoff(ch, key)
                continue

            if kind == "drum":
                vel = int(round(127 * float(e.get("velocity", .3)) * velocity_scale))
                vel = max(1, min(127, vel))
                fs.noteon(drum_channel, int(e["_midi_key"]), vel)
                continue

            # Pitched note-on.
            ch = alloc.allocate(eid, sample)
            preset = resolve_sf2_preset(e, mapping, force_bank, force_program)
            rc = fs.program_select(ch, preset["bank"], preset["program"])
            if rc != 0:
                # Be robust to incomplete SoundFonts: try bank 0 / program 0.
                rc2 = fs.program_select(ch, 0, 0)
                if rc2 != 0:
                    raise RuntimeError(
                        f"Preset not found and fallback failed: bank={preset['bank']} "
                        f"program={preset['program']}"
                    )
            fs.set_bend(ch, int(e["_bend"]), int(bend_range_semitones))
            vel_float = float(e.get("velocity", .7)) * preset["gain"] * velocity_scale
            vel = max(1, min(127, int(round(127 * vel_float))))
            key = int(e["_midi_key"])
            alloc.state[ch].key = key
            fs.noteon(ch, key, vel)

        while cur < total_samples:
            n = min(chunk, total_samples - cur)
            mix[cur:cur+n] += fs.render(n)
            cur += n

        if verbose:
            print(
                f"SF2: {len(pitched)} pitched events, peak independent channels "
                f"{alloc.peak_active}/{len(alloc.channels)}, max nearest-MIDI offset "
                f"{max_abs_cents:.3f} cents"
            )

    if percussion_backend == "raw":
        _mix_raw_percussion(mix, percussion, sec_per_beat, sr)

    # Keep SF2 timbre intact: no v7 tanh waveshaper.  Peak-normalize only.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-12:
        mix *= OUTPUT_PEAK / peak

    if mono:
        return np.mean(mix, axis=1)
    return mix
