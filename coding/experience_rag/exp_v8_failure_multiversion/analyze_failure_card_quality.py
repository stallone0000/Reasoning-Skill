#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline quality analysis for failure-card files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from card_quality import analyze_cards, split_cards_by_quality


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def parse_input_arg(values: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        parsed.append((label.strip(), Path(raw_path).resolve()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure-card quality offline")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeated label=path input. Example: --input v5=cards_v5_diagnostic.jsonl",
    )
    parser.add_argument(
        "--write_filtered_dir",
        default="",
        help="Optional directory to write <label>_accepted.jsonl and <label>_rejected.jsonl",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    inputs = parse_input_arg(args.input)
    if not inputs:
        raise RuntimeError("Provide at least one --input label=path")

    summary: Dict[str, dict] = {}
    filtered_dir = Path(args.write_filtered_dir).resolve() if args.write_filtered_dir else None
    if filtered_dir:
        filtered_dir.mkdir(parents=True, exist_ok=True)
    for label, path in inputs:
        cards = list(iter_jsonl(path))
        accepted, rejected = split_cards_by_quality(cards)
        summary[label] = {
            "path": str(path),
            **analyze_cards(cards),
        }
        if filtered_dir:
            accepted_path = filtered_dir / f"{label}_accepted.jsonl"
            rejected_path = filtered_dir / f"{label}_rejected.jsonl"
            with accepted_path.open("w", encoding="utf-8") as f:
                for row in accepted:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with rejected_path.open("w", encoding="utf-8") as f:
                for row in rejected:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary[label]["accepted_path"] = str(accepted_path)
            summary[label]["rejected_path"] = str(rejected_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
