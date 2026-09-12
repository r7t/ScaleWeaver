#!/usr/bin/env python3
"""Convert an MSCX score to a reusable ScaleWeaver imitation reference IR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation_frontend import build_reference_ir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mscx")
    parser.add_argument("output_json", nargs="?", default="mscx_reference_ir.json")
    parser.add_argument("--staff-id")
    args = parser.parse_args(argv)
    data = build_reference_ir(args.input_mscx, staff_id=args.staff_id)
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated MSCX reference IR", path)
    print("Reference staff", data["source_staff_id"],
          "Lead notes", len(data["highest_line"]), "Measures", data["bars"])
    return data


if __name__ == "__main__":
    main()
