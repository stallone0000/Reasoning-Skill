#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare multiple exp_v8 run directories at the variant level."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from card_quality import analyze_cards


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
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


def parse_run_arg(values: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got: {item}")
        label, raw_path = item.split("=", 1)
        parsed.append((label.strip(), Path(raw_path).resolve()))
    return parsed


def summarize_variant(variant_dir: Path) -> Dict[str, object]:
    results = list(iter_jsonl(variant_dir / "task_results.jsonl"))
    cards = list(iter_jsonl(variant_dir / "cards.jsonl"))
    report = analyze_cards(cards) if cards else {}
    error_counter = Counter(row.get("error") or "unknown" for row in results if not row.get("ok"))

    asym_total = sum(1 for row in results if row.get("failure_origin") == "asymmetric_with_counterexample")
    asym_ok = sum(
        1
        for row in results
        if row.get("failure_origin") == "asymmetric_with_counterexample" and row.get("ok")
    )
    return {
        "tasks_total": len(results),
        "ok": sum(1 for row in results if row.get("ok")),
        "failed": sum(1 for row in results if not row.get("ok")),
        "ok_rate": round(sum(1 for row in results if row.get("ok")) / max(len(results), 1), 4),
        "cards_total": len(cards),
        "avg_attempts": round(
            sum(int(row.get("attempts", 0) or 0) for row in results) / max(len(results), 1),
            2,
        ),
        "asym_total": asym_total,
        "asym_ok": asym_ok,
        "asym_ok_rate": round(asym_ok / max(asym_total, 1), 4) if asym_total else None,
        "top_errors": error_counter.most_common(5),
        "quality": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple failure-card run directories")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Repeated label=run_dir input. Example: --run old=results/pilot60_api_20260319",
    )
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    runs = parse_run_arg(args.run)
    if not runs:
        raise RuntimeError("Provide at least one --run label=dir")

    summary: Dict[str, dict] = {}
    for label, run_dir in runs:
        variants = sorted(path.name for path in run_dir.iterdir() if path.is_dir())
        summary[label] = {
            "run_dir": str(run_dir),
            "variants": {variant: summarize_variant(run_dir / variant) for variant in variants},
        }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
