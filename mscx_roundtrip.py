"""Canonical XML and semantic guards for stable MSCX/IR round trips."""
from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET


ROUNDTRIP_FORMAT = "ScaleWeaverCanonicalMSCX/1"


def canonicalize_mscx(text):
    """Return deterministic, parseable MSCX XML.

    Canonicalization deliberately normalizes formatting and attribute quoting;
    notation semantics and the complete XML tree are retained.
    """
    root = ET.fromstring(text)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def semantic_payload(data):
    payload = copy.deepcopy(data)
    payload.pop("roundtrip_mscx", None)
    return payload


def semantic_sha256(data):
    encoded = json.dumps(
        semantic_payload(data), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_roundtrip(data, mscx_text):
    data["roundtrip_mscx"] = {
        "format": ROUNDTRIP_FORMAT,
        "semantic_sha256": semantic_sha256(data),
        "xml": canonicalize_mscx(mscx_text),
    }
    return data


def reusable_roundtrip_xml(data):
    block = data.get("roundtrip_mscx")
    if not isinstance(block, dict) or block.get("format") != ROUNDTRIP_FORMAT:
        return None
    if block.get("semantic_sha256") != semantic_sha256(data):
        return None
    xml = block.get("xml")
    if not isinstance(xml, str):
        return None
    # Validate even trusted JSON: malformed embedded XML must never be emitted.
    return canonicalize_mscx(xml)
