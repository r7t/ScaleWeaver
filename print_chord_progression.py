#!/usr/bin/env python3
"""Print a readable chord progression from a ScaleWeaver JSON result.

Examples:
    python print_chord_progression.py 6333.json
    python print_chord_progression.py 6333.json --details --output chords.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _harmony_bars(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept both complete annealing output and standalone frontend IR."""
    bars = document.get("harmony_plan")
    if isinstance(bars, list):
        return bars
    frontend = document.get("frontend_ir")
    if isinstance(frontend, dict):
        bars = frontend.get("harmony_bars")
        if isinstance(bars, list):
            return bars
    raise ValueError("未找到 harmony_plan 或 frontend_ir.harmony_bars")


def _number(value: Any) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _segment_label(segment: dict[str, Any], beats_per_bar: float, details: bool) -> str:
    offset = float(segment.get("offset", 0.0))
    duration = float(segment.get("duration", beats_per_bar))
    begin, end = offset + 1.0, offset + duration
    placement = "" if offset == 0.0 and duration == beats_per_bar else f"{_number(begin)}–{_number(end)}拍 "
    name = segment.get("chord_name") or segment.get("chord_id") or "<未命名和弦>"
    if not details:
        return f"{placement}{name}"
    fields = []
    if segment.get("function"):
        fields.append(str(segment["function"]))
    if segment.get("ratio"):
        fields.append(str(segment["ratio"]))
    if segment.get("chord_id"):
        fields.append(str(segment["chord_id"]))
    suffix = f" [{' / '.join(fields)}]" if fields else ""
    return f"{placement}{name}{suffix}"


def format_progression(document: dict[str, Any], details: bool = False) -> str:
    beats_per_bar = float(document.get("beats_per_bar", 4.0))
    lines: list[str] = []
    for index, bar in enumerate(_harmony_bars(document), start=1):
        actual_index = int(bar.get("bar", index - 1)) + 1
        segments = bar.get("chord_segments")
        if not isinstance(segments, list) or not segments:
            segments = [bar]
        labels = [_segment_label(segment, beats_per_bar, details) for segment in segments]
        section = bar.get("section")
        section_text = f" ({section})" if section else ""
        lines.append(f"{actual_index:>3}{section_text}: " + " | ".join(labels))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="输出 ScaleWeaver JSON 的逐小节和弦进行")
    parser.add_argument("input", type=Path, help="ScaleWeaver 输出 JSON")
    parser.add_argument("--details", action="store_true", help="同时输出功能、比例和 chord_id")
    parser.add_argument("-o", "--output", type=Path, help="写入文本文件；省略时输出到终端")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    output = format_progression(document, details=args.details)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"已写入 {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
