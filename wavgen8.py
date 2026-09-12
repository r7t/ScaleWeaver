#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ScaleWeaver WAV renderer — modular raw/SF2 + exact-F0 singing voicebank.

Modules in the same directory:
  raw_synth.py        native waveform backend
  sf2_synth.py        SoundFont2/FluidSynth backend
  voicebank_synth.py  mnemonic singing backend using pre-rendered exact-F0 WAVs
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import wave
from pathlib import Path

import numpy as np

from raw_synth import DEFAULT_SR, MAX_PARTIALS, DECAY_SCALE, render_score_raw
from sf2_synth import render_score_sf2
from voicebank_synth import (
    DEFAULT_SING_GAIN,
    DEFAULT_VOICE_OCTAVES,
    DEFAULT_ATTACK_OFFSET_SEC,
    DEFAULT_BANK_DIR,
    mix_lead_voicebank,
)


def quantize_frequency_to_edo(freq: float, edo: int, base_freq: float) -> float:
    freq = float(freq)
    base_freq = float(base_freq)
    edo = int(edo)
    if not (freq > 0.0 and math.isfinite(freq)):
        raise ValueError(f"Invalid frequency for quantization: {freq}")
    if not (base_freq > 0.0 and math.isfinite(base_freq)):
        raise ValueError(f"Invalid quantization base frequency: {base_freq}")
    if edo <= 0:
        raise ValueError(f"EDO must be a positive integer, got {edo}")
    step = int(round(edo * math.log2(freq / base_freq)))
    return base_freq * (2.0 ** (step / edo))


def quantize_score_to_edo(score: dict, edo: int | None, base_freq: float | None = None):
    if edo is None:
        return score, None
    edo = int(edo)
    if edo <= 0:
        raise ValueError(f"EDO must be a positive integer, got {edo}")
    if base_freq is None: base_freq=float(score["tuning"]["base_freq_hz"])
    out = copy.deepcopy(score)
    shifts_cents: list[float] = []
    for events in out.get("voices", {}).values():
        for event in events:
            if "freq" not in event:
                continue
            old = float(event["freq"])
            new = quantize_frequency_to_edo(old, edo, base_freq)
            event["freq"] = new
            shifts_cents.append(1200.0 * math.log2(new / old))
    stats = {
        "edo": edo,
        "count": len(shifts_cents),
        "max_abs_cents": max((abs(x) for x in shifts_cents), default=0.0),
        "mean_abs_cents": (sum(abs(x) for x in shifts_cents) / len(shifts_cents)) if shifts_cents else 0.0,
    }
    return out, stats


def write_wav(path: str | Path, audio: np.ndarray, sr: int = DEFAULT_SR) -> None:
    x = np.asarray(audio)
    if x.ndim == 1:
        channels = 1
    elif x.ndim == 2 and x.shape[1] in (1, 2):
        channels = int(x.shape[1])
    else:
        raise ValueError(f"Audio must be mono or stereo, got shape {x.shape}")
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(Path(path)), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def render_file(
    score_path: str | Path,
    wav_path: str | Path,
    *,
    backend: str = "raw",
    sr: int = DEFAULT_SR,
    decay_scale: float = DECAY_SCALE,
    max_partials: int = MAX_PARTIALS,
    quantize_edo: int | None = None,
    quantize_base_freq: float | None = None,
    sing_lead: bool = False,
    sing_gain: float = DEFAULT_SING_GAIN,
    sing_octaves: int = DEFAULT_VOICE_OCTAVES,
    sing_attack_offset_sec: float = DEFAULT_ATTACK_OFFSET_SEC,
    voicebank_dir: str | Path = DEFAULT_BANK_DIR,
    voicebank_missing: str = "error",
    **sf2_kwargs,
) -> None:
    with open(score_path, "r", encoding="utf-8") as f:
        score = json.load(f)

    if quantize_base_freq is None: quantize_base_freq=float(score["tuning"]["base_freq_hz"])
    score, quant_stats = quantize_score_to_edo(score, quantize_edo, quantize_base_freq)
    verbose = bool(sf2_kwargs.get("verbose", True))
    if quant_stats is not None and verbose:
        print(
            f"Tuning: {quant_stats['edo']}-EDO anchored at {quantize_base_freq:.6f} Hz; "
            f"quantized {quant_stats['count']} events, mean |shift| "
            f"{quant_stats['mean_abs_cents']:.3f} cents, max |shift| "
            f"{quant_stats['max_abs_cents']:.3f} cents"
        )

    if backend == "raw":
        audio = render_score_raw(score, sr, decay_scale, max_partials)
    elif backend == "sf2":
        if not sf2_kwargs.get("sf2_path"):
            raise ValueError("--backend sf2 requires --sf2 PATH")
        audio = render_score_sf2(score, sr=sr, **sf2_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if sing_lead:
        audio = mix_lead_voicebank(
            audio,
            score,
            sr,
            bank_dir=voicebank_dir,
            gain=sing_gain,
            octave_shift=sing_octaves,
            attack_offset_sec=sing_attack_offset_sec,
            missing=voicebank_missing,
            verbose=verbose,
        )

    write_wav(wav_path, audio, sr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ScaleWeaver JSON -> WAV renderer (raw/SF2 + exact-F0 singing voicebank)"
    )
    ap.add_argument("score_json")
    ap.add_argument("output_wav")
    ap.add_argument("--backend", choices=("raw", "sf2"), default="raw")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    ap.add_argument("--quantize-edo", type=int, default=None, metavar="N")
    ap.add_argument("--quantize-base", dest="quantize_base_freq", type=float, default=None)

    # Singing overlay.
    ap.add_argument("--sing-lead", action="store_true",
                    help="Overlay ScaleWeaver mnemonic singing on lead notes.")
    ap.add_argument("--sing-gain", type=float, default=DEFAULT_SING_GAIN)
    ap.add_argument("--sing-octaves", type=int, default=DEFAULT_VOICE_OCTAVES)
    ap.add_argument("--sing-attack-offset", dest="sing_attack_offset_sec", type=float,
                    default=DEFAULT_ATTACK_OFFSET_SEC,
                    help=f"Start voice before instrumental attack (default {DEFAULT_ATTACK_OFFSET_SEC:.3f} s).")
    ap.add_argument("--voicebank-dir", default=str(DEFAULT_BANK_DIR),
                    help="Directory containing voicebank manifest.json and configured exact-F0 WAVs.")
    ap.add_argument("--voicebank-missing", choices=("error", "skip"), default="error",
                    help="How to handle a missing sample required by the score.")

    # Raw backend.
    ap.add_argument("--decay-scale", type=float, default=DECAY_SCALE)
    ap.add_argument("--partials", type=int, default=MAX_PARTIALS)

    # SF2 backend.
    ap.add_argument("--sf2", dest="sf2_path")
    ap.add_argument("--sf2-map", dest="sf2_map_path")
    ap.add_argument("--sf2-bank", type=int, default=None)
    ap.add_argument("--sf2-program", type=int, default=None)
    ap.add_argument("--sf2-gain", dest="synth_gain", type=float, default=.55)
    ap.add_argument("--sf2-velocity-scale", dest="velocity_scale", type=float, default=1.0)
    ap.add_argument("--sf2-bend-range", dest="bend_range_semitones", type=int, default=2)
    ap.add_argument("--sf2-channels", dest="midi_channels", type=int, default=128)
    ap.add_argument("--sf2-release-hold", dest="release_hold_sec", type=float, default=.8)
    ap.add_argument("--sf2-tail", dest="tail_sec", type=float, default=1.5)
    ap.add_argument("--sf2-percussion", dest="percussion_backend", choices=("raw", "sf2", "none"), default="raw")
    ap.add_argument("--sf2-mono", dest="mono", action="store_true")
    ap.add_argument("--sf2-reverb", dest="reverb", action="store_true")
    ap.add_argument("--sf2-chorus", dest="chorus", action="store_true")
    ap.set_defaults(reverb=False, chorus=False)
    ap.add_argument("--fluidsynth-lib", default=None)
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    ap.set_defaults(verbose=True)

    args = ap.parse_args()
    render_file(
        args.score_json,
        args.output_wav,
        backend=args.backend,
        sr=args.sample_rate,
        decay_scale=args.decay_scale,
        max_partials=args.partials,
        quantize_edo=args.quantize_edo,
        quantize_base_freq=args.quantize_base_freq,
        sing_lead=args.sing_lead,
        sing_gain=args.sing_gain,
        sing_octaves=args.sing_octaves,
        sing_attack_offset_sec=args.sing_attack_offset_sec,
        voicebank_dir=args.voicebank_dir,
        voicebank_missing=args.voicebank_missing,
        sf2_path=args.sf2_path,
        sf2_map_path=args.sf2_map_path,
        force_bank=args.sf2_bank,
        force_program=args.sf2_program,
        synth_gain=args.synth_gain,
        velocity_scale=args.velocity_scale,
        bend_range_semitones=args.bend_range_semitones,
        midi_channels=args.midi_channels,
        release_hold_sec=args.release_hold_sec,
        tail_sec=args.tail_sec,
        reverb=args.reverb,
        chorus=args.chorus,
        percussion_backend=args.percussion_backend,
        mono=args.mono,
        fluidsynth_lib=args.fluidsynth_lib,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
