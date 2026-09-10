#!/usr/bin/env python3
"""Generate ScaleWeaver's harmony progression and Lead as a front-end IR."""
from __future__ import annotations

import argparse
import math
import random

import harmony_rhythm as hr
import score_builder as sb
from frontend_ir import build_ir, save_ir


def generate_frontend_ir(seed=20260811, bars=48, bpm=96.0, motif_degrees=None,
                         time_signature="4/4", allow_sixteenth=True, spec=None):
    """Run the unchanged legacy front end and stop before accompaniment."""
    if spec is None:
        spec = hr.SCALE
    if type(bars) is not int or bars <= 0:
        raise ValueError("bars must be a positive integer")
    if not math.isfinite(float(bpm)) or float(bpm) <= 0:
        raise ValueError("bpm must be positive")
    ts, bpb = hr.normalize_time_signature(time_signature)

    harmony_rng = random.Random(int(seed))
    plan = hr.harmony_plan(bars, harmony_rng, bpb)

    lead_rng = random.Random(int(seed) ^ 0x13579BDF)
    lead_palette = hr.make_rhythm_palette(lead_rng, bpb)
    lead = sb.generate_lead(
        bars, plan, lead_rng, motif_degrees, bpb, lead_palette,
        allow_sixteenth=allow_sixteenth,
    )
    jumps = sb.lead_jump_errors(lead)
    if jumps:
        raise sb.GenerationRejected(
            f"front-end Lead exceeds the configured jump limit: {jumps[:2]}")

    return build_ir(
        seed=seed,
        bpm=bpm,
        time_signature=ts,
        beats_per_bar=bpb,
        bars=bars,
        spec=spec,
        harmony_plan=plan,
        lead=lead,
        allow_sixteenth=allow_sixteenth,
        lead_palette=hr.rhythm_palette_metadata(lead_palette),
        motif_degrees=motif_degrees,
        generator="legacy_frontend",
    )


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", dest="scale_config", required=True)
    parser.add_argument("--rules", dest="rules_config")
    parser.add_argument("--style", dest="style_config")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--bars", type=int, default=48)
    parser.add_argument("--bpm", type=float, default=96.0)
    parser.add_argument("--time-signature", default="4/4")
    parser.add_argument("--chord-progression",
                        help="fixed degree/code loop; auto restores automatic harmony")
    parser.add_argument("--cse-dir")
    parser.add_argument("--cse-workers", type=int)
    parser.add_argument("--output", "-o", default="frontend_ir.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allow-sixteenth", dest="allow_sixteenth", action="store_true")
    group.add_argument("--no-sixteenth", dest="allow_sixteenth", action="store_false")
    parser.set_defaults(allow_sixteenth=True)
    return parser


def cli(argv=None):
    args = _build_parser().parse_args(argv)
    # Runtime setup remains shared with the full generator so the legacy front
    # end sees exactly the same scale, chord catalogue and CSE tables.
    import main
    spec, _ = main._configure_adaptive_scale(
        args.scale_config,
        args.cse_dir,
        rules_config=args.rules_config,
        style_config=args.style_config,
        cse_workers=args.cse_workers,
        chord_progression=args.chord_progression,
    )
    data = generate_frontend_ir(
        seed=args.seed,
        bars=args.bars,
        bpm=args.bpm,
        time_signature=args.time_signature,
        allow_sixteenth=args.allow_sixteenth,
        spec=spec,
    )
    path = save_ir(args.output, data, spec=spec)
    print("Generated front-end IR", path)
    print("Lead notes", len(data["lead"]), "Harmony bars", len(data["harmony_plan"]))
    return data


if __name__ == "__main__":
    cli()
