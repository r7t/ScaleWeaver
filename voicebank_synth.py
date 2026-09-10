#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-F0 singing samples for an explicitly configured EDO-subset scale.

Initialize with --scale DEFINITION --init DIR. The profile supplies pitches,
register and optional reference syllables. Playback validates target frequencies
against score events and never retunes samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scale_config import load_scale

_SCALE = None
LEAD_STEP_RANGE = None
REFERENCE_TEXT = {}
DEFAULT_VOICE_OCTAVES = -1
DEFAULT_SING_GAIN = 0.30
DEFAULT_ATTACK_OFFSET_SEC = 0.050
DEFAULT_RELEASE_FADE_SEC = 0.035
DEFAULT_BANK_DIR = Path(__file__).resolve().parent / 'voicebank'


def configure_scale(scale, *, lead_range=None, reference_text=None):
    global _SCALE, LEAD_STEP_RANGE, REFERENCE_TEXT
    _SCALE = load_scale(scale)
    LEAD_STEP_RANGE = tuple(lead_range or _SCALE.resolved_voice_ranges()['lead'])
    REFERENCE_TEXT = dict(reference_text if reference_text is not None else
                          _SCALE.rules.get('voicebank', {}).get('reference_text', {}))


def _require_scale():
    if _SCALE is None: raise RuntimeError('A scale definition is required for voicebank initialization')
    return _SCALE


@dataclass(frozen=True)
class VoicebankEntry:
    step: int
    name: str
    filename: str
    target_f0_hz: float
    gain: float = 1.0
    # Optional per-sample adjustment. Positive means start even earlier.
    attack_adjust_ms: float = 0.0


def configured_lead_steps() -> tuple[int, ...]:
    return _require_scale().make_pool(*LEAD_STEP_RANGE)


def step_name(step: int) -> str:
    spec = _require_scale()
    return spec.pc_name[int(step) % spec.edo]


def lead_freq_from_step(step: int) -> float:
    return _require_scale().freq(step)


def voice_freq_from_step(step: int, octave_shift: int = DEFAULT_VOICE_OCTAVES) -> float:
    return lead_freq_from_step(step) * (2.0 ** int(octave_shift))


def default_filename(step: int, name: str | None = None) -> str:
    name = name or step_name(step)
    return f"step_{int(step):+04d}_{name}.wav"


def build_manifest_dict(octave_shift: int = DEFAULT_VOICE_OCTAVES) -> dict[str, Any]:
    entries = []
    for step in configured_lead_steps():
        name = step_name(step)
        entries.append({
            "step": step,
            "name": name,
            "reference_text": REFERENCE_TEXT.get(name,name),
            "file": default_filename(step, name),
            "lead_f0_hz": round(lead_freq_from_step(step), 9),
            "target_f0_hz": round(voice_freq_from_step(step, octave_shift), 9),
            "gain": 1.0,
            "attack_adjust_ms": 0.0,
        })
    return {
        "format": "ScaleVoiceBank/1",
        "scale": _require_scale().definition_dict(),
        "description": (
            "Pre-rendered singing samples at exact ScaleWeaver F0. Runtime playback does not retune or time-stretch."
        ),
        "tuning": {
            "edo": _require_scale().edo,
            "base_note": _require_scale().base_note,
            "base_freq_hz": _require_scale().base_freq_hz,
            "lead_step_range": list(LEAD_STEP_RANGE),
            "pcs": list(_require_scale().pcs),
            "voice_octave_shift": int(octave_shift),
        },
        "reference_text": REFERENCE_TEXT,
        "entries": entries,
    }


def init_voicebank(bank_dir: str | Path, *, octave_shift: int = DEFAULT_VOICE_OCTAVES, force: bool = False, scale=None) -> Path:
    if scale is not None: configure_scale(scale)
    _require_scale()
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bank_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass --force to overwrite template files")

    manifest = build_manifest_dict(octave_shift)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    plan_path = bank_dir / "render_plan.csv"
    with plan_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "name", "reference_text", "lead_f0_hz", "target_f0_hz",
            "suggested_duration_sec", "filename",
        ])
        for e in manifest["entries"]:
            # 0.75 s is long enough for a clear consonant, stable vowel, and natural release.
            # wavgen8 will *not* force this to fill longer notes.
            w.writerow([
                e["step"], e["name"], e["reference_text"],
                f"{e['lead_f0_hz']:.6f}", f"{e['target_f0_hz']:.6f}",
                "0.75", e["file"],
            ])

    readme = bank_dir / "README_voicebank.txt"
    readme.write_text(
        'Generate each WAV listed in render_plan.csv at target_f0_hz.\n'
        'Use the configured reference text; retain the listed filenames.\n'
        'Samples play at their original frequency without pitch shifting.\n', encoding='utf-8')
    return manifest_path


def load_voicebank(bank_dir: str | Path) -> tuple[dict[int, VoicebankEntry], dict[str, Any]]:
    bank_dir = Path(bank_dir)
    manifest_path = bank_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Voicebank manifest not found: {manifest_path}\n"
            f"Create a template first: python voicebank_synth.py --init {bank_dir}"
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get('format') != 'ScaleVoiceBank/1':
        raise ValueError(f'Unsupported voicebank format: {data.get("format")!r}')
    tuning=data['tuning']
    configure_scale(data['scale'], lead_range=tuning['lead_step_range'], reference_text=data.get('reference_text',{}))
    if int(tuning['edo']) != _SCALE.edo or tuple(tuning['pcs']) != _SCALE.pcs or not math.isclose(float(tuning['base_freq_hz']), _SCALE.base_freq_hz,rel_tol=1e-10):
        raise ValueError('Voicebank tuning differs from its scale definition')

    entries: dict[int, VoicebankEntry] = {}
    for raw in data.get("entries", []):
        step = int(raw["step"])
        name = str(raw.get("name") or step_name(step))
        expected_name = step_name(step)
        if name != expected_name:
            raise ValueError(f"Voicebank entry step {step}: name {name!r} != expected {expected_name!r}")
        entry = VoicebankEntry(
            step=step,
            name=name,
            filename=str(raw.get("file") or default_filename(step, name)),
            target_f0_hz=float(raw.get("target_f0_hz", voice_freq_from_step(step, int(tuning.get("voice_octave_shift", -1))))),
            gain=float(raw.get("gain", 1.0)),
            attack_adjust_ms=float(raw.get("attack_adjust_ms", 0.0)),
        )
        expected_f0 = voice_freq_from_step(step, int(tuning['voice_octave_shift']))
        if not math.isfinite(entry.target_f0_hz) or not math.isclose(entry.target_f0_hz,expected_f0,rel_tol=1e-8):
            raise ValueError(f'Voicebank step {step} target frequency differs from scale definition')
        if step in entries: raise ValueError(f'Duplicate voicebank step {step}')
        entries[step] = entry
    return entries, data


def _load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Load WAV/other libsndfile audio to mono float64 and sample-rate convert.

    soundfile is preferred because singing synthesizers commonly export 24-bit
    PCM or float WAV.  A minimal 16-bit PCM fallback keeps the module usable in
    a bare Python environment.
    """
    try:
        import soundfile as sf  # type: ignore
        y, sr = sf.read(str(path), always_2d=True, dtype="float64")
        y = np.mean(y, axis=1)
        sr = int(sr)
    except ImportError:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise RuntimeError(
                    f"{path.name}: install soundfile for non-16-bit WAV support: pip install soundfile"
                )
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            if channels > 1:
                arr = arr.reshape(-1, channels).mean(axis=1)
            y = arr

    if sr <= 0 or y.size == 0:
        return np.zeros(0, dtype=np.float64)
    if sr != int(target_sr):
        # Sample-rate conversion only.  Duration and pitch are preserved because
        # the new sample count follows the SR ratio; this is not musical retuning.
        n_out = max(1, int(round(y.size * float(target_sr) / sr)))
        old_x = np.linspace(0.0, 1.0, y.size, endpoint=False)
        new_x = np.linspace(0.0, 1.0, n_out, endpoint=False)
        y = np.interp(new_x, old_x, y)
    return np.asarray(y, dtype=np.float64)


def _fade_edges(y: np.ndarray, sr: int, attack_sec: float = .0015, release_sec: float = DEFAULT_RELEASE_FADE_SEC) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    if y.size <= 1:
        return y
    na = min(y.size, max(1, int(round(attack_sec * sr))))
    nr = min(y.size, max(1, int(round(release_sec * sr))))
    if na > 1:
        x = np.linspace(0.0, math.pi, na)
        y[:na] *= .5 - .5 * np.cos(x)
    if nr > 1:
        x = np.linspace(0.0, math.pi, nr)
        y[-nr:] *= .5 + .5 * np.cos(x)
    return y


def _truncate_naturally(y: np.ndarray, max_samples: int, sr: int, release_sec: float) -> np.ndarray:
    """Never stretch/fill.  Only shorten a source if the score note is shorter."""
    if max_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    if y.size <= max_samples:
        return _fade_edges(y, sr, attack_sec=.0015, release_sec=min(.012, release_sec))
    out = np.asarray(y[:max_samples], dtype=np.float64).copy()
    # A smooth release makes a forced short-note cutoff unobtrusive.
    nr = min(out.size, max(1, int(round(release_sec * sr))))
    if nr > 1:
        x = np.linspace(0.0, math.pi, nr)
        out[-nr:] *= .5 + .5 * np.cos(x)
    if out.size:
        out[-1] = 0.0
    return out


def _ensure_audio_shape(base_audio: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(base_audio, dtype=np.float64)
    if x.ndim == 1:
        return x.copy(), 1
    if x.ndim == 2 and x.shape[1] in (1, 2):
        return x.copy(), x.shape[1]
    raise ValueError(f"music audio must be mono or stereo, got {x.shape}")


def mix_lead_voicebank(
    base_audio: np.ndarray,
    score: dict,
    sr: int,
    *,
    bank_dir: str | Path = DEFAULT_BANK_DIR,
    gain: float = DEFAULT_SING_GAIN,
    octave_shift: int = DEFAULT_VOICE_OCTAVES,
    attack_offset_sec: float = DEFAULT_ATTACK_OFFSET_SEC,
    release_fade_sec: float = DEFAULT_RELEASE_FADE_SEC,
    missing: str = "error",
    verbose: bool = True,
) -> np.ndarray:
    """Mix pre-rendered exact-F0 ScaleWeaver singing samples onto the lead.

    No musical pitch shifting or time stretching is performed.  This is the key
    design difference from the old Neural-TTS/Praat path.
    """
    if missing not in {"error", "skip"}:
        raise ValueError("missing must be 'error' or 'skip'")
    entries, manifest = load_voicebank(bank_dir)
    bank_dir = Path(bank_dir)
    bank_oct = int(manifest.get("tuning", {}).get("voice_octave_shift", DEFAULT_VOICE_OCTAVES))
    if int(octave_shift) != bank_oct:
        raise ValueError(
            f"This voicebank was rendered for octave_shift={bank_oct}, but --sing-octaves={octave_shift}. "
            "To preserve naturalness this backend does not retune samples; render another bank for that octave."
        )

    lead = list(score.get("voices", {}).get("lead", ()))
    if not lead:
        return np.asarray(base_audio, dtype=np.float64)

    for e in lead:
        if 'step' in e and 'freq' in e and int(e['step']) in entries:
            target=float(e['freq'])*2**int(octave_shift)
            if not math.isclose(target, entries[int(e['step'])].target_f0_hz, rel_tol=1e-8):
                raise ValueError('Score tuning does not match the exact-F0 voicebank')
    required_steps = sorted({int(e["step"]) for e in lead if "step" in e and "freq" in e})
    absent_manifest = [s for s in required_steps if s not in entries]
    absent_files = [s for s in required_steps if s in entries and not (bank_dir / entries[s].filename).is_file()]
    problems = absent_manifest + absent_files
    if problems and missing == "error":
        lines = []
        for s in problems:
            name = step_name(s) if _SCALE.is_scale_pitch(s) else "?"
            fn = entries[s].filename if s in entries else default_filename(s, name)
            lines.append(f"  step {s:+d} {name}: {fn}")
        raise FileNotFoundError(
            "Voicebank is missing samples required by this score:\n" + "\n".join(lines) +
            f"\nBank: {bank_dir}"
        )

    out, channels = _ensure_audio_shape(base_audio)
    bpm = float(score.get("tempo_bpm", score.get("meta", {}).get("bpm", 96.0)))
    sec_per_beat = 60.0 / bpm
    sample_cache: dict[int, np.ndarray] = {}
    rendered = 0
    skipped = 0

    for e in lead:
        if "freq" not in e or "step" not in e:
            continue
        step = int(e["step"])
        entry = entries.get(step)
        if entry is None:
            skipped += 1
            continue
        path = bank_dir / entry.filename
        if not path.is_file():
            skipped += 1
            continue

        if step not in sample_cache:
            sample_cache[step] = _load_audio_mono(path, sr)
        source = sample_cache[step]
        if source.size == 0:
            skipped += 1
            continue

        dur_sec = max(0.0, float(e.get("duration_beats", 0.0)) * sec_per_beat)
        # Because the consonant starts before the piano note, let a short sample
        # occupy note duration + pre-attack.  For long notes the source ends on
        # its own; no padding and no looping are ever used.
        pre = max(0.0, float(attack_offset_sec) + entry.attack_adjust_ms / 1000.0)
        max_voice_sec = max(.06, dur_sec + pre)
        y = _truncate_naturally(source, int(round(max_voice_sec * sr)), sr, release_fade_sec)

        # Preserve the source singer's own dynamics; only apply entry/global gain
        # and a mild score-velocity factor so syllables remain intelligible.
        vel = max(0.0, min(1.5, float(e.get("velocity", .7))))
        amp = float(gain) * float(entry.gain) * (0.78 + 0.22 * vel)
        y = y * amp

        start_sec = float(e.get("start_beat", 0.0)) * sec_per_beat - pre
        i0 = int(round(start_sec * sr))
        src0 = 0
        if i0 < 0:
            src0 = -i0
            i0 = 0
        if src0 >= y.size or i0 >= len(out):
            continue
        n = min(y.size - src0, len(out) - i0)
        if n <= 0:
            continue
        yy = y[src0:src0+n]
        if channels == 1:
            out[i0:i0+n] += yy
        else:
            out[i0:i0+n, 0] += yy
            out[i0:i0+n, 1] += yy
        rendered += 1

    # Keep the instrumental mix intact as much as possible.  Only prevent final
    # clipping; do not globally re-normalize every successful render.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > .995:
        out *= .995 / peak

    if verbose:
        print(
            f"Voicebank singing: {rendered} lead notes, {len(sample_cache)} source WAVs used, "
            f"{skipped} skipped; bank={bank_dir}; no pitch shift / no time stretch"
        )
    return out


def check_voicebank(bank_dir: str | Path, *, verify_f0: bool = False) -> int:
    entries, manifest = load_voicebank(bank_dir)
    bank_dir = Path(bank_dir)
    missing = []
    bad = []
    for step in configured_lead_steps():
        e = entries.get(step)
        if e is None or not (bank_dir / e.filename).is_file():
            missing.append(step)
            continue
        if verify_f0:
            try:
                import parselmouth  # type: ignore
                snd = parselmouth.Sound(str(bank_dir / e.filename))
                pitch = snd.to_pitch_ac(
                    time_step=0.0,
                    pitch_floor=max(55.0, e.target_f0_hz * .55),
                    pitch_ceiling=min(1200.0, e.target_f0_hz * 1.8),
                )
                vals = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
                vals = vals[vals > 0]
                if vals.size:
                    med = float(np.median(vals))
                    cents = 1200.0 * math.log2(med / e.target_f0_hz)
                    if abs(cents) > 20.0:
                        bad.append((step, med, e.target_f0_hz, cents))
            except ImportError:
                print("F0 verification skipped: pip install praat-parselmouth")
                verify_f0 = False
    print(f"Voicebank: {bank_dir}")
    print(f"Required entries: {len(configured_lead_steps())}; missing WAVs: {len(missing)}")
    if missing:
        for s in missing:
            print(f"  MISSING step {s:+d} {step_name(s)} -> {default_filename(s)}")
    if bad:
        print(f"Large F0 mismatches (>20 cents): {len(bad)}")
        for s, med, target, cents in bad:
            print(f"  step {s:+d} {step_name(s)}: measured {med:.2f} Hz, target {target:.2f} Hz, {cents:+.1f} cents")
    return 1 if missing or bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize/check the scale-specific exact-F0 singing voicebank")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", metavar="DIR", help="Create manifest.json + render_plan.csv template")
    g.add_argument("--check", metavar="DIR", help="Check that all configured WAV files exist")
    ap.add_argument("--octaves", type=int, default=DEFAULT_VOICE_OCTAVES,
                    help=f"Voice octave shift relative to lead when initializing (default {DEFAULT_VOICE_OCTAVES})")
    ap.add_argument("--force", action="store_true", help="Overwrite manifest/template files on --init")
    ap.add_argument("--verify-f0", action="store_true", help="Also estimate each sample F0 with praat-parselmouth")
    ap.add_argument("--scale",help="ScaleDefinition/1 JSON required for --init")
    args = ap.parse_args()
    if args.init:
        if not args.scale: ap.error("--init requires --scale")
        p = init_voicebank(args.init, octave_shift=args.octaves, force=args.force, scale=args.scale)
        print(f"Created: {p}")
        print(f"Fill the WAV files listed in: {Path(args.init) / 'render_plan.csv'}")
        return
    raise SystemExit(check_voicebank(args.check, verify_f0=args.verify_f0))


if __name__ == "__main__":
    main()
