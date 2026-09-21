#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command adaptive-scale JSON + instrumental WAV + MSCX pipeline.

Place this entry point beside the latest ScaleWeaver modules and wavgen8.py.
Set the three JSON paths below, then run python piano.py <seed>.
Event frequencies already encode the target tuning; do not re-quantize them.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MAIN_PY = HERE / "main.py"
MSCX_PY = HERE / "mscx_export.py"
WAVGEN_PY = HERE / "wavgen8.py"
SF2_FILE = HERE / "piano.sf2"

# 日常使用只需修改这三个文件，然后运行 python piano.py <种子>。
# 相对路径以 piano.py 所在目录为基准，也可以填写绝对路径。
SCALE_DEFINITION_FILE = "scales/tiangan_72.json"
COMPOSITION_RULES_FILE = "scales/tiangan_72_5.rules.json"
STYLE_PARAMETERS_FILE = "scales/tiangan_72_5_norm.style.json"
# SCALE_DEFINITION_FILE = "scales/major_31.json"
# COMPOSITION_RULES_FILE = "scales/major_31.rules.json"
# STYLE_PARAMETERS_FILE = "scales/major_31.style.json"
# SCALE_DEFINITION_FILE = "scales/major_12.json"
# COMPOSITION_RULES_FILE = "scales/major_12.rules.json"
# STYLE_PARAMETERS_FILE = "scales/major_12.style.json"
# SCALE_DEFINITION_FILE = "scales/ji_major_171.json"
# COMPOSITION_RULES_FILE = "scales/ji_major_171.rules.json"
# STYLE_PARAMETERS_FILE = "scales/ji_major_171.style.json"

_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}


def safe_seed_filename(text, digest=None):
    """Keep a textual seed readable while making it a portable filename."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text).rstrip(' .')
    if not cleaned or cleaned in {'.', '..'}:
        cleaned = 'seed'
    if cleaned.split('.')[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = '_' + cleaned
    digest = digest or hashlib.sha256(text.encode('utf-8')).digest()
    if cleaned != text:
        cleaned += '-' + digest.hex()[:8]
    if len(cleaned.encode('utf-8')) > 120:
        prefix = cleaned.encode('utf-8')[:107].decode('utf-8', 'ignore').rstrip(' .')
        cleaned = prefix + '-' + digest.hex()[:8]
    return cleaned


def resolve_seed(value):
    """Return the real integer seed, output stem, and original seed text."""
    text = str(value).strip()
    if not text:
        raise ValueError("seed 不能为空")
    try:
        numeric = int(text)
    except ValueError:
        digest = hashlib.sha256(
            b"ScaleWeaver/piano-seed/v1\0" + text.encode('utf-8')).digest()
        real_seed = int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)
        return real_seed, safe_seed_filename(text, digest), text
    return numeric, str(numeric), text

def configured_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else HERE / path).resolve()


def resolve_config_paths(args):
    """CLI overrides are relative to the caller; globals are relative to this script."""
    scale = Path(args.scale).expanduser().resolve() if args.scale else configured_path(SCALE_DEFINITION_FILE)
    stem = scale.stem.removesuffix('.scale')
    rules = (Path(args.rules).expanduser().resolve() if args.rules else
             scale.with_name(stem+'.rules.json') if args.scale else configured_path(COMPOSITION_RULES_FILE))
    style = (Path(args.style).expanduser().resolve() if args.style else
             scale.with_name(stem+'.style.json') if args.scale else configured_path(STYLE_PARAMETERS_FILE))
    return scale,rules,style


def run(cmd, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=HERE, check=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed",nargs="?",default=None,
                    help="integer seed, or arbitrary text mapped through SHA-256")
    ap.add_argument("--scale",help="ScaleDefinition/1 JSON (default: top-of-file configuration)")
    ap.add_argument("--rules",help="CompositionRules/1 JSON (optional sibling override)")
    ap.add_argument("--style",help="CompositionStyle/1 JSON (optional sibling override)")
    ap.add_argument("--chord-progression",help="Loop degree triads, e.g. 4536251; auto disables the JSON progression")
    ap.add_argument("--output-dir",default=str(HERE)+"/output")
    ap.add_argument("--cse-dir",default=str(HERE)+"/CSE_cache")
    ap.add_argument("--bars",type=int,default=48)
    ap.add_argument("--bpm",type=float,default=96.0)
    ap.add_argument("--time-signature",default="4/4")
    ap.add_argument("--imitate-rhythm","--imitate-mscx","--imitate-reference",
                    dest="imitation_reference",
                    help="MSCX or reference IR used only as a rhythm/barline template")
    ap.add_argument("--imitation-staff-id")
    ap.add_argument("--imitation-beam-width",type=int,default=48)
    ap.add_argument("--retune-melody",
                    help="copy and minimum-RMS retune the complete source melody")
    ap.add_argument("--retune-staff-id")
    ap.add_argument("--save-frontend-ir",action="store_true")
    ap.add_argument('--harmony-model',choices=('legacy','three-tone'),default='three-tone')
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--allow-sixteenth",dest="allow_sixteenth",action="store_true")
    g.add_argument("--no-sixteenth",dest="allow_sixteenth",action="store_false")
    ap.set_defaults(allow_sixteenth=True)
    ap.add_argument("--cse-workers",type=int,default=None)
    ap.add_argument("--wavgen",default=str(WAVGEN_PY),help="WAV renderer script")
    ap.add_argument("--sf2",default=str(SF2_FILE),help="SoundFont file")
    ap.add_argument("--no-wav",action="store_true",help="Skip WAV rendering")
    ap.add_argument("--continuous",action="store_true")
    args=ap.parse_args(argv)

    seed_input=args.seed
    if seed_input is None:
        seed_input=input("请输入 seed（例如 626 或 always-with-me）：")
    try:
        seed,output_stem,seed_text=resolve_seed(seed_input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scale,rules,style=resolve_config_paths(args)
    from scale_config import load_scale
    try:
        load_scale(scale,rules=rules,style=style,require_composition=True)
    except (OSError,ValueError) as exc:
        raise SystemExit(f"Invalid scale configuration: {exc}") from exc
    for p in (MAIN_PY,MSCX_PY):
        if not p.is_file(): raise SystemExit(f"Missing required file: {p}")

    if not args.no_wav:
        wavgen = Path(args.wavgen).expanduser().resolve()
        sf2 = Path(args.sf2).expanduser().resolve()
        if not wavgen.is_file(): raise SystemExit(f"WAV renderer not found: {wavgen}")
        if not sf2.is_file(): raise SystemExit(f"SoundFont not found: {sf2}")

    out_dir=Path(args.output_dir).expanduser().resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    json_path=out_dir/f"{output_stem}.json"
    wav_path=out_dir/f"{output_stem}.wav"
    mscx_path=out_dir/f"{output_stem}.mscx"

    cmd=[sys.executable,str(MAIN_PY),"--scale",str(scale),"--seed",str(seed),
         "--bars",str(args.bars),"--bpm",str(args.bpm),"--time-signature",args.time_signature,
         "--cse-dir",str(Path(args.cse_dir).expanduser().resolve()),"--output",str(json_path),
         '--harmony-model',args.harmony_model]
    cmd += ["--rules",str(rules),"--style",str(style)]
    if args.imitation_reference is not None:
        cmd += ["--imitate-rhythm",str(Path(args.imitation_reference).expanduser().resolve()),
                "--imitation-beam-width",str(args.imitation_beam_width)]
        if args.imitation_staff_id is not None:
            cmd += ["--imitation-staff-id",str(args.imitation_staff_id)]
    if args.retune_melody is not None:
        cmd += ["--retune-melody",str(Path(args.retune_melody).expanduser().resolve())]
        if args.retune_staff_id is not None:
            cmd += ["--retune-staff-id",str(args.retune_staff_id)]
    if args.save_frontend_ir:
        cmd += ["--save-frontend-ir",str(out_dir/f"{output_stem}.frontend.json")]
    if args.chord_progression is not None:
        cmd += ["--chord-progression",args.chord_progression]
    if args.allow_sixteenth:
        cmd.append("--allow-sixteenth")
    else:
        cmd.append("--no-sixteenth")
    if args.cse_workers is not None: cmd += ["--cse-workers",str(args.cse_workers)]
    run(cmd,f"[1/3] Compose exact-EDO JSON: {json_path.name}")

    if not args.no_wav:
        # Render the event frequencies directly. No pitch quantization or vocals.
        run([
            sys.executable,str(wavgen),str(json_path),str(wav_path),
            "--backend","sf2","--sf2",str(sf2),"--sf2-bank","0",
            "--sf2-program","0","--sf2-reverb"
        ],f"[2/3] Render instrumental WAV: {wav_path.name}")
    else:
        print("\n[2/3] WAV skipped (--no-wav)")

    mcmd=[sys.executable,str(MSCX_PY),str(json_path),str(mscx_path)]
    if args.continuous: mcmd.append("--continuous")
    run(mcmd,f"[3/3] Export exact-EDO MSCX + tuning file: {mscx_path.name}")

    print("\n"+"="*72)
    print("Done")
    print("="*72)
    print("seed input:",seed_text)
    print("real seed :",seed)
    print("scale:",scale.name)
    print("rules:",rules.name)
    print("style:",style.name)
    print("JSON :",json_path.name)
    if not args.no_wav: print("WAV  :",wav_path.name,"(instrumental only)")
    print("MSCX :",mscx_path.name)


if __name__=="__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
