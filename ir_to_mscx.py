#!/usr/bin/env python3
"""Convert ScaleWeaver reference/front-end/score IR JSON to MSCX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _frontend_score(data):
    scale = data["scale"]
    return {
        "format": "ScaleWeaverScore/1",
        "bars": int(data["bars"]),
        "beats_per_bar": float(data["beats_per_bar"]),
        "time_signature": str(data["time_signature"]),
        "tempo_bpm": float(data["tempo_bpm"]),
        "measure_map": data.get("measure_map"),
        "tuning": {
            "system": "edo", "edo": int(scale["edo"]),
            "scale_id": scale["scale_id"], "scale_name": scale["scale_name"],
            "base_note": scale["base_note"],
            "base_freq_hz": float(scale["base_freq_hz"]),
            "pcs": list(scale["pcs"]), "names": list(scale["names"]),
        },
        "voices": {"lead": data["lead"]},
        "voice_layout": {"staff_order": ["lead"]},
    }


def convert(data):
    kind = data.get("format")
    if kind == "ScaleWeaverMSCXReferenceIR/1":
        from reference_ir_mscx import generate
        return generate(data)
    if kind == "ScaleWeaverFrontEndIR/1":
        from mscx_export import generate
        from mscx_roundtrip import canonicalize_mscx
        return canonicalize_mscx(generate(_frontend_score(data)))
    if kind == "ScaleWeaverScore/1":
        from mscx_export import generate
        from mscx_roundtrip import canonicalize_mscx
        return canonicalize_mscx(generate(data))
    raise ValueError(f"unsupported IR format {kind!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json")
    parser.add_argument("output_mscx", nargs="?", default="ir_score.mscx")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    text = convert(data)
    path = Path(args.output_mscx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("Generated", path)
    return text


if __name__ == "__main__":
    main()
