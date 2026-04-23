#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from common import load_jsonl


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def numeric(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def summarize_condition(items: list[dict[str, Any]]) -> dict[str, Any]:
    success = [item for item in items if item.get("status") == "success"]
    question_ids = sorted({str(item["question_id"]) for item in items})
    repeat_ids = sorted({int(item.get("repeat_idx", 0)) for item in items})
    attempts = len(question_ids) * max(len(repeat_ids), 1)

    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_repeat[int(item.get("repeat_idx", 0))].append(item)

    repeat_accuracies: list[float] = []
    for repeat_idx in repeat_ids or [0]:
        repeat_items = by_repeat[repeat_idx]
        correct = sum(int(item.get("is_correct") or 0) for item in repeat_items if item.get("status") == "success")
        repeat_accuracies.append(correct / max(len(question_ids), 1))

    question_votes: dict[str, list[int]] = defaultdict(list)
    for item in items:
        question_votes[str(item["question_id"])].append(int(item.get("is_correct") or 0))
    majority_correct = sum(1 for votes in question_votes.values() if mean(votes) > 0.5)

    sample = items[0]
    return {
        "benchmark": sample.get("benchmark", ""),
        "model": sample.get("model_label", ""),
        "mode": sample.get("mode", ""),
        "questions": len(question_ids),
        "repeats": len(repeat_ids) or 1,
        "attempts": attempts,
        "success_rate_pct": round(len(success) / max(attempts, 1) * 100, 2),
        "mean_accuracy_pct": round(mean_or_zero(repeat_accuracies) * 100, 2),
        "accuracy_std_pct": round(std_or_zero(repeat_accuracies) * 100, 2),
        "majority_vote_accuracy_pct": round(majority_correct / max(len(question_ids), 1) * 100, 2),
        "avg_prompt_tokens": round(mean_or_zero([numeric(item, "prompt_tokens") for item in success]), 1),
        "avg_completion_tokens": round(mean_or_zero([numeric(item, "completion_tokens") for item in success]), 1),
        "avg_total_tokens": round(mean_or_zero([numeric(item, "total_tokens") for item in success]), 1),
        "avg_cost": round(mean_or_zero([numeric(item, "cost", "cost_yuan") for item in success]), 8),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark condition and direct-vs-TRS comparison tables.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(args.results_root).glob("*/*.jsonl")):
        rows = load_jsonl(path)
        if not rows:
            continue
        sample = rows[0]
        key = (str(sample.get("benchmark", "")), str(sample.get("model_label", "")), str(sample.get("mode", "")))
        grouped[key].extend(rows)

    condition_rows = [summarize_condition(items) for _, items in sorted(grouped.items()) if items]

    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in condition_rows:
        by_pair[(row["benchmark"], row["model"])][row["mode"]] = row

    comparison_rows: list[dict[str, Any]] = []
    for (benchmark, model), modes in sorted(by_pair.items()):
        direct = modes.get("direct")
        trs = modes.get("trs")
        if not direct or not trs:
            continue
        comparison_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "direct_accuracy_pct": direct["mean_accuracy_pct"],
                "trs_accuracy_pct": trs["mean_accuracy_pct"],
                "accuracy_delta_pct": round(trs["mean_accuracy_pct"] - direct["mean_accuracy_pct"], 2),
                "direct_total_tokens": direct["avg_total_tokens"],
                "trs_total_tokens": trs["avg_total_tokens"],
                "total_tokens_delta": round(trs["avg_total_tokens"] - direct["avg_total_tokens"], 1),
                "direct_cost": direct["avg_cost"],
                "trs_cost": trs["avg_cost"],
                "cost_delta": round(trs["avg_cost"] - direct["avg_cost"], 8),
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "benchmark_conditions.csv", condition_rows)
    write_markdown(output_dir / "benchmark_conditions.md", condition_rows)
    write_csv(output_dir / "benchmark_comparisons.csv", comparison_rows)
    write_markdown(output_dir / "benchmark_comparisons.md", comparison_rows)
    summary = {"conditions": condition_rows, "comparisons": comparison_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
