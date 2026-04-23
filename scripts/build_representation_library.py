#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from common import estimate_tokens, load_jsonl, write_jsonl


def trim_to_token_budget(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    return text[: max(1, budget * 4)].rstrip()


def trace_from_item(item: dict[str, Any]) -> str:
    trace = (item.get("reasoning_trace") or "").strip()
    if trace:
        return trace
    return "\n\n".join(
        part.strip()
        for part in [item.get("model_think", ""), item.get("model_response", "")]
        if str(part).strip()
    )


def build_raw_example(item: dict[str, Any]) -> str:
    return (
        "Retrieved solved example\n"
        f"Problem:\n{item.get('question', '').strip()}\n\n"
        f"Solution:\n{(item.get('model_response') or item.get('answer') or '').strip()}\n\n"
        f"Official answer:\n{item.get('answer', '').strip()}"
    ).strip()


def build_raw_cot(item: dict[str, Any], budget: int) -> str:
    prefix = (
        "Retrieved prior reasoning trace\n"
        f"Problem:\n{item.get('question', '').strip()}\n\n"
        "Reasoning:\n"
    )
    suffix = f"\n\nFinal answer:\n{item.get('answer', '').strip()}"
    reserved = estimate_tokens(prefix + suffix)
    available = max(16, budget - reserved)
    return f"{prefix}{trim_to_token_budget(trace_from_item(item), available)}{suffix}".strip()


def normalize_record(item: dict[str, Any], text: str, representation_type: str, budget_tokens: int) -> dict[str, Any]:
    return {
        "question_id": item.get("question_id") or item.get("source_question_id"),
        "source_question_id": item.get("question_id") or item.get("source_question_id"),
        "question": item.get("question", ""),
        "topic": item.get("topic", ""),
        "keywords": item.get("keywords", ""),
        "heuristic": text,
        "representation_type": representation_type,
        "source_budget_tokens": budget_tokens,
        "status": item.get("status", "success"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize skill, summary, raw-example, or raw-CoT records into one retrieval library schema."
    )
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument(
        "--mode",
        choices=["structured", "free_summary", "raw_example", "raw_cot"],
        required=True,
    )
    parser.add_argument("--summary-field", default="free_summary")
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    out: list[dict[str, Any]] = []
    for item in rows:
        structured = (item.get("heuristic") or "").strip()
        budget = max(24, estimate_tokens(structured)) if structured else 256

        if args.mode == "structured":
            text = structured
        elif args.mode == "free_summary":
            text = trim_to_token_budget((item.get(args.summary_field) or "").strip(), budget)
        elif args.mode == "raw_example":
            text = trim_to_token_budget(build_raw_example(item), budget)
        else:
            text = build_raw_cot(item, budget)

        if not text:
            continue
        out.append(normalize_record(item, text, args.mode, budget))

    write_jsonl(args.output_file, out)
    print(f"wrote {len(out)} retrieval records -> {args.output_file}")


if __name__ == "__main__":
    main()
