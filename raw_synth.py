#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScaleWeaver raw waveform synthesizer.

Contains the original native waveform/instrument synthesis path plus raw percussion.
It has no dependency on FluidSynth or the neural voice layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_SR = 44100
MAX_PARTIALS = 12
DECAY_SCALE = 1.0
NYQUIST_MARGIN = 0.48
OUTPUT_PEAK = 0.95
DEFAULT_TAIL_SEC = 0.055

@dataclass(frozen=True)
class InstrumentProfile:
    waveform: str
    clip_level: float = 0.5
    gain: float = 1.0
    attack: float = 0.006
    release: float = 0.045
    decay_mul: float = 1.0
    fast_mult: float = 4.6
    fast_base: float = 0.18
    fast_growth: float = 0.43
    slow_harmonic: float = 0.105
    fast_harmonic: float = 0.16
    outer_tilt: float = 0.0
    pitch_decay_exp: float = 0.20


INSTRUMENTS = {
    "contrabass": InstrumentProfile(
        "full_wave_rectified", gain=.90, attack=.006, release=.060,
        decay_mul=.72, fast_mult=3.8, fast_base=.12, fast_growth=.32,
        outer_tilt=.10, pitch_decay_exp=.14),
    "plucked_bass": InstrumentProfile(
        "full_wave_rectified", gain=.92, attack=.0025, release=.050,
        decay_mul=1.15, fast_mult=5.0, fast_base=.30, fast_growth=.42,
        outer_tilt=-.04, pitch_decay_exp=.17),
    "cello_pad": InstrumentProfile(
        "clipped_sine", clip_level=.58, gain=.86, attack=.014, release=.075,
        decay_mul=.62, fast_mult=3.7, fast_base=.11, fast_growth=.28,
        outer_tilt=.10, pitch_decay_exp=.15),
    "warm_keys": InstrumentProfile(
        "full_wave_rectified", gain=.84, attack=.0040, release=.050,
        decay_mul=.94, fast_mult=4.5, fast_base=.19, fast_growth=.38,
        outer_tilt=.02, pitch_decay_exp=.18),
    "glass_keys": InstrumentProfile(
        "clipped_sine", clip_level=.38, gain=.78, attack=.0025, release=.080,
        decay_mul=1.05, fast_mult=5.0, fast_base=.25, fast_growth=.42,
        outer_tilt=-.06, pitch_decay_exp=.19),
    "clarinet_like": InstrumentProfile(
        "clipped_sine", clip_level=.48, gain=.90, attack=.006, release=.042,
        decay_mul=.70, fast_mult=4.0, fast_base=.13, fast_growth=.30,
        outer_tilt=.05, pitch_decay_exp=.15),
    "flute_like": InstrumentProfile(
        "sine", gain=.88, attack=.010, release=.065,
        decay_mul=.56, fast_mult=3.6, fast_base=.09, fast_growth=.24,
        outer_tilt=.12, pitch_decay_exp=.13),
    "warm_reed": InstrumentProfile(
        "clipped_sine", clip_level=.55, gain=.98, attack=.005, release=.045,
        decay_mul=.76, fast_mult=4.1, fast_base=.15, fast_growth=.33,
        outer_tilt=.00, pitch_decay_exp=.16),
    "bright_reed": InstrumentProfile(
        "clipped_sine", clip_level=.40, gain=.94, attack=.0035, release=.040,
        decay_mul=.82, fast_mult=4.3, fast_base=.18, fast_growth=.36,
        outer_tilt=-.07, pitch_decay_exp=.17),
    "flute_bell": InstrumentProfile(
        "triangle", gain=.84, attack=.0025, release=.090,
        decay_mul=.98, fast_mult=4.8, fast_base=.23, fast_growth=.40,
        outer_tilt=-.06, pitch_decay_exp=.18),
}
DEFAULT_PROFILE = InstrumentProfile(
    "clipped_sine", clip_level=.50, gain=.92, attack=.005, release=.045,
    decay_mul=.78, outer_tilt=0.0, pitch_decay_exp=.17,
)


def score_bpm(score: dict) -> float:
    if "tempo_bpm" in score:
        return float(score["tempo_bpm"])
    return float(score.get("meta", {}).get("bpm", 96.0))

def raw_wave(theta: np.ndarray, profile: InstrumentProfile) -> np.ndarray:
    kind = profile.waveform
    if kind == "sine":
        return np.sin(theta)
    if kind == "clipped_sine":
        return np.clip(np.sin(theta), -profile.clip_level, profile.clip_level)
    if kind == "full_wave_rectified":
        return np.abs(np.sin(.5 * theta)) - (2.0 / math.pi)
    if kind == "triangle":
        return (2.0 / math.pi) * np.arcsin(np.sin(theta))
    if kind == "soft_square":
        drive = 2.2
        return np.tanh(drive * np.sin(theta)) / math.tanh(drive)
    raise ValueError(f"unknown waveform: {kind}")


def dual_decay_envelope(
    t: np.ndarray,
    freq: float,
    n: int,
    profile: InstrumentProfile,
    decay_scale: float,
    reference_freq: float,
) -> np.ndarray:
    pitch_factor = max(.70, (freq / reference_freq) ** profile.pitch_decay_exp)
    harmonic_slow = 1.0 + profile.slow_harmonic * (n - 1) ** .82
    r_slow = .6084 * decay_scale * profile.decay_mul * pitch_factor * harmonic_slow
    harmonic_fast = 1.0 + profile.fast_harmonic * (n - 1) ** .90
    r_fast = profile.fast_mult * r_slow * harmonic_fast
    a_fast = profile.fast_base + profile.fast_growth * (1.0 - math.exp(-(n - 1) / 3.2))
    a_fast = min(.78, max(.04, a_fast))
    return a_fast * np.exp(-r_fast * t) + (1.0 - a_fast) * np.exp(-r_slow * t)


def raised_cosine_gate(
    n_samples: int,
    sr: int,
    attack_sec: float,
    release_sec: float,
) -> np.ndarray:
    env = np.ones(n_samples, dtype=np.float64)
    if n_samples <= 1:
        return env
    na = min(n_samples, max(1, int(round(attack_sec * sr))))
    nr = min(n_samples, max(1, int(round(release_sec * sr))))
    if na > 1:
        x = np.linspace(0.0, math.pi, na, endpoint=True)
        env[:na] *= .5 - .5 * np.cos(x)
    if nr > 1:
        x = np.linspace(0.0, math.pi, nr, endpoint=True)
        env[-nr:] *= .5 + .5 * np.cos(x)
    return env


def synth_note(
    freq: float,
    duration_sec: float,
    velocity: float,
    event_phase: float,
    sr: int,
    decay_scale: float,
    max_partials: int,
    instrument: str = "bright_reed",
    reference_freq: float | None = None,
) -> np.ndarray:
    profile = INSTRUMENTS.get(instrument, DEFAULT_PROFILE)
    tail = max(DEFAULT_TAIL_SEC, profile.release * .72)
    render_dur = max(.001, duration_sec + tail)
    n_samples = max(1, int(round(render_dur * sr)))
    t = np.arange(n_samples, dtype=np.float64) / sr
    signal = np.zeros(n_samples, dtype=np.float64)

    q = 2.0 ** (freq / 300.0)
    outer_limit = NYQUIST_MARGIN * sr
    used_weight = 0.0

    for n in range(1, max_partials + 1):
        outer_freq = n * freq
        if outer_freq >= outer_limit:
            break
        amp = q ** (-n) * n ** (-profile.outer_tilt)
        if amp < 1e-5:
            break
        theta = 2.0 * math.pi * outer_freq * t + n * event_phase
        raw = raw_wave(theta, profile)
        env = dual_decay_envelope(t, freq, n, profile, decay_scale, reference_freq or freq)
        signal += amp * env * raw
        used_weight += abs(amp)

    if used_weight > 1e-12:
        signal /= used_weight

    signal *= float(velocity) * profile.gain
    attack = min(profile.attack, max(.0015, duration_sec * .08))
    release = min(profile.release + tail, max(.012, duration_sec * .16 + tail))
    signal *= raised_cosine_gate(n_samples, sr, attack, release)
    return signal


def synth_percussion(event: dict, duration_sec: float, sr: int) -> np.ndarray:
    instrument = event.get("instrument", "wood_click")
    dur = max(.035, duration_sec)
    if instrument == "frame_drum":
        dur = max(dur, .24)
    elif instrument == "shaker":
        dur = max(dur, .09)
    n = max(1, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    vel = float(event.get("velocity", .3))
    seed = int(round(float(event.get("start_beat", 0.0)) * 10000)) + {
        "frame_drum": 17, "wood_click": 29, "shaker": 43
    }.get(instrument, 59)
    rng = np.random.default_rng(seed)

    if instrument == "frame_drum":
        f0, f1 = 82.0, 48.0
        k = (f1 - f0) / max(dur, 1e-6)
        phase = 2 * math.pi * (f0 * t + .5 * k * t * t)
        x = np.sin(phase) * np.exp(-11.5 * t)
        x += .08 * rng.normal(size=n) * np.exp(-44 * t)
    elif instrument == "shaker":
        noise = rng.normal(size=n)
        x = np.concatenate(([0.0], np.diff(noise))) * np.exp(-30 * t)
    else:
        noise = rng.normal(size=n)
        x = .48 * noise * np.exp(-52 * t) + .52 * np.sin(2 * math.pi * 1450 * t) * np.exp(-40 * t)

    fade_n = min(n, max(2, int(.0015 * sr)))
    if fade_n > 1:
        x[:fade_n] *= np.linspace(0.0, 1.0, fade_n)
    peak = float(np.max(np.abs(x))) if len(x) else 1.0
    if peak > 1e-12:
        x /= peak
    return x * vel * .70


def render_score_raw(
    score: dict,
    sr: int = DEFAULT_SR,
    decay_scale: float = DECAY_SCALE,
    max_partials: int = MAX_PARTIALS,
) -> np.ndarray:
    """The original v7 rendering path, kept behavior-compatible."""
    bpm = score_bpm(score)
    sec_per_beat = 60.0 / bpm
    voices = score.get("voices", {})
    if not voices:
        raise ValueError("Score JSON has no 'voices' dictionary.")

    max_end_beat = max(
        (float(e["start_beat"]) + float(e["duration_beats"])
         for events in voices.values() for e in events),
        default=0.0,
    )
    total_sec = max_end_beat * sec_per_beat + .18
    mix = np.zeros(max(1, int(math.ceil(total_sec * sr))), dtype=np.float64)

    for events in voices.values():
        for e in events:
            start_sec = float(e["start_beat"]) * sec_per_beat
            dur_sec = float(e["duration_beats"]) * sec_per_beat
            if "freq" not in e or e.get("role") == "percussion":
                note = synth_percussion(e, dur_sec, sr)
            else:
                note = synth_note(
                    freq=float(e["freq"]),
                    duration_sec=dur_sec,
                    velocity=float(e.get("velocity", .7)),
                    event_phase=float(e.get("phase", 0.0)),
                    sr=sr,
                    decay_scale=decay_scale,
                    max_partials=max_partials,
                    instrument=e.get("instrument", "bright_reed"),
                    reference_freq=float(score["tuning"]["base_freq_hz"]),
                )

            i0 = int(round(start_sec * sr))
            i1 = min(len(mix), i0 + len(note))
            if i1 > i0:
                mix[i0:i1] += note[:i1 - i0]

    mix = np.tanh(1.10 * mix)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 1e-12:
        mix *= OUTPUT_PEAK / peak
    return mix
