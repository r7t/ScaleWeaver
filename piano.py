#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command adaptive-scale JSON + instrumental WAV + MSCX pipeline.

Place this entry point beside the latest ScaleWeaver modules and wavgen8.py.
Set the three JSON paths below, then run python piano.py <seed>.
Event frequencies already encode the target tuning; do not re-quantize them.
"""
from __future__ import annotations

import argparse
from pathlib import Path
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
STYLE_PARAMETERS_FILE = "scales/tiangan_72_5.style.json"


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
    ap.add_argument("seed",nargs="?",type=int,default=None)
    ap.add_argument("--scale",help="ScaleDefinition/1 JSON (default: top-of-file configuration)")
    ap.add_argument("--rules",help="CompositionRules/1 JSON (optional sibling override)")
    ap.add_argument("--style",help="CompositionStyle/1 JSON (optional sibling override)")
    ap.add_argument("--chord-progression",help="Loop degree triads, e.g. 4536251; auto disables the JSON progression")
    ap.add_argument("--output-dir",default=str(HERE)+"/output")
    ap.add_argument("--cse-dir",default=str(HERE)+"/CSE_cache")
    ap.add_argument("--bars",type=int,default=48)
    ap.add_argument("--bpm",type=float,default=96.0)
    ap.add_argument("--time-signature",default="4/4")
    ap.add_argument("--best-of",type=int,default=1,help="Deprecated: writer now uses whole-score annealing")
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--allow-sixteenth",dest="allow_sixteenth",action="store_true")
    g.add_argument("--no-sixteenth",dest="allow_sixteenth",action="store_false")
    ap.set_defaults(allow_sixteenth=True)
    cg=ap.add_mutually_exclusive_group()
    cg.add_argument("--cleanup",dest="cleanup",action="store_true",
                    help="Enable optional post-generation pitch cleanup")
    cg.add_argument("--no-cleanup",dest="cleanup",action="store_false")
    ap.set_defaults(cleanup=False)
    ap.add_argument("--cse-workers",type=int,default=None)
    ap.add_argument("--wavgen",default=str(WAVGEN_PY),help="WAV renderer script")
    ap.add_argument("--sf2",default=str(SF2_FILE),help="SoundFont file")
    ap.add_argument("--no-wav",action="store_true",help="Skip WAV rendering")
    ap.add_argument("--continuous",action="store_true")
    args=ap.parse_args(argv)

    seed=args.seed
    if seed is None:
        try: seed=int(input("请输入 seed（例如 626）：").strip())
        except ValueError: raise SystemExit("seed 必须是整数")

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
    json_path=out_dir/f"{seed}.json"
    wav_path=out_dir/f"{seed}.wav"
    mscx_path=out_dir/f"{seed}.mscx"

    cmd=[sys.executable,str(MAIN_PY),"--scale",str(scale),"--seed",str(seed),
         "--bars",str(args.bars),"--bpm",str(args.bpm),"--time-signature",args.time_signature,
         "--best-of",str(args.best_of),"--cse-dir",str(Path(args.cse_dir).expanduser().resolve()),"--output",str(json_path),"--melodyplan-frontend"] #,"--chord-progression","0j1g9aqd"]
    cmd += ["--rules",str(rules),"--style",str(style)]
    if args.chord_progression is not None:
        cmd += ["--chord-progression",args.chord_progression]
    if args.allow_sixteenth:
        cmd.append("--allow-sixteenth")
    else:
        cmd.append("--no-sixteenth")
    cmd.append("--cleanup" if args.cleanup else "--no-cleanup")
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
    print("seed :",seed)
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
