#!/usr/bin/env python3
"""Verify ScaleWeaver's canonical MSCX/IR fixed-point invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from imitation_frontend import build_reference_ir
from ir_to_mscx import convert


def _json_bytes(data):
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def check_mscx(path):
    ir1 = build_reference_ir(path)
    mscx2 = convert(ir1)
    with tempfile.TemporaryDirectory(prefix="scaleweaver-roundtrip-") as directory:
        root = Path(directory)
        p2 = root / "canonical2.mscx"
        p2.write_text(mscx2, encoding="utf-8")
        ir2 = build_reference_ir(p2)
        mscx3 = convert(ir2)
    same_ir = _json_bytes(ir1) == _json_bytes(ir2)
    same_mscx = mscx2.encode("utf-8") == mscx3.encode("utf-8")
    return same_ir, same_mscx, _digest(_json_bytes(ir2)), _digest(mscx2)


def check_ir(path):
    ir1 = json.loads(Path(path).read_text(encoding="utf-8"))
    mscx1 = convert(ir1)
    with tempfile.TemporaryDirectory(prefix="scaleweaver-roundtrip-") as directory:
        root = Path(directory)
        p1 = root / "canonical1.mscx"
        p2 = root / "canonical2.mscx"
        p1.write_text(mscx1, encoding="utf-8")
        ir2 = build_reference_ir(p1)
        mscx2 = convert(ir2)
        p2.write_text(mscx2, encoding="utf-8")
        ir3 = build_reference_ir(p2)
    same_mscx = mscx1.encode("utf-8") == mscx2.encode("utf-8")
    same_ir = _json_bytes(ir2) == _json_bytes(ir3)
    return same_ir, same_mscx, _digest(_json_bytes(ir2)), _digest(mscx1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MSCX or ScaleWeaver IR JSON")
    args = parser.parse_args(argv)
    path = Path(args.input)
    result = check_ir(path) if path.suffix.lower() == ".json" else check_mscx(path)
    same_ir, same_mscx, ir_hash, mscx_hash = result
    print("IR fixed point:", "PASS" if same_ir else "FAIL", ir_hash)
    print("MSCX fixed point:", "PASS" if same_mscx else "FAIL", mscx_hash)
    if not same_ir or not same_mscx:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
