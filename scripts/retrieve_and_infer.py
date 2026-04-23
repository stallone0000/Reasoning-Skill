#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import aiohttp

from common import (
    BM25,
    answers_match,
    build_retrieval_document,
    call_chat_completion,
    extract_final_answer,
    load_jsonl,
    now_iso,
    render_prompt,
    response_texts,
    usage_counts,
    write_jsonl,
)


class RepresentationBank:
    def __init__(self, library_file: str) -> None:
        self.records = [
            row for row in load_jsonl(library_file) if preferred_skill_text(row).strip()
        ]
        self.documents = [build_retrieval_document(row) for row in self.records]
        self.bm25 = BM25(self.documents)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        hits = []
        for idx, score in self.bm25.top_k(query, top_k):
            row = self.records[idx]
            hits.append(
                {
                    "record": row,
                    "score": score,
                    "question_id": row.get("question_id") or row.get("source_question_id"),
                    "heuristic": preferred_skill_text(row),
                }
            )
        return hits


def preferred_skill_text(row: dict[str, Any]) -> str:
    text = row.get("heuristic") or row.get("skill_text") or row.get("inject_text") or row.get("free_summary") or ""
    if text:
        return str(text)
    fields = []
    for label, name in [
        ("Trigger", "trigger"),
        ("Do", "do"),
        ("Avoid", "avoid"),
        ("Check", "check"),
        ("Risk", "risk"),
        ("Complexity", "complexity"),
    ]:
        value = row.get(name)
        if isinstance(value, list) and value:
            fields.append(f"{label}: " + "; ".join(str(item) for item in value))
        elif value:
            fields.append(f"{label}: {value}")
    return "\n".join(fields)


def mock_inference_response(item: dict[str, Any], hints: str) -> str:
    answer = item.get("answer") or "UNKNOWN"
    hint_line = "Used retrieved hints." if hints.strip() else "Solved directly."
    return f"{hint_line}\nFinal answer: {answer}"


async def process_one(
    session: aiohttp.ClientSession | None,
    item: dict[str, Any],
    *,
    bank: RepresentationBank,
    prompt_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hits = bank.search(item.get("question", ""), args.top_k)
    hints = "\n\n---\n\n".join(hit["heuristic"] for hit in hits)
    prompt = render_prompt(prompt_template, problem=item.get("question", ""), hints=hints)

    if args.mock:
        response_text = mock_inference_response(item, hints)
        predicted = extract_final_answer(response_text)
        return {
            **item,
            "model_label": "mock-model",
            "gen_model": "mock-model",
            "mode": "trs",
            "raw_model_response": response_text,
            "predicted_answer": predicted,
            "is_correct": int(answers_match(predicted, item.get("answer", ""))) if item.get("answer") else None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "retrieved_count": len(hits),
            "retrieved_question_ids": [hit["question_id"] for hit in hits],
            "retrieved_scores": [round(hit["score"], 6) for hit in hits],
            "retrieved_preview": [hit["heuristic"] for hit in hits],
            "status": "success",
            "error": "",
            "timestamp": now_iso(),
        }

    assert session is not None
    for retry_attempt in range(args.max_retries + 1):
        result = await call_chat_completion(
            session,
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_tag="trs_retrieve_infer",
        )
        if result["status"] != "success":
            if retry_attempt < args.max_retries:
                await asyncio.sleep(min(20.0, 2.0**retry_attempt))
                continue
            return {
                **item,
                "model_label": args.model_label,
                "gen_model": args.model,
                "mode": "trs",
                "status": "error",
                "error": result["error"],
                "retry_attempt": retry_attempt,
                "timestamp": now_iso(),
            }

        payload = result["response"]
        reasoning_text, response_text = response_texts(payload)
        predicted = extract_final_answer(response_text)
        usage = usage_counts(payload, prompt_text=prompt, response_text=reasoning_text + "\n" + response_text)
        completion_cost = usage["completion_tokens"] / 1_000_000 * args.output_price
        prompt_cost = usage["prompt_tokens"] / 1_000_000 * args.input_price
        return {
            **item,
            "model_label": args.model_label,
            "gen_model": args.model,
            "mode": "trs",
            "raw_reasoning": reasoning_text,
            "raw_model_response": response_text,
            "predicted_answer": predicted,
            "is_correct": int(answers_match(predicted, item.get("answer", ""))) if item.get("answer") else None,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost": prompt_cost + completion_cost,
            "retrieved_count": len(hits),
            "retrieved_question_ids": [hit["question_id"] for hit in hits],
            "retrieved_scores": [round(hit["score"], 6) for hit in hits],
            "retrieved_preview": [hit["heuristic"] for hit in hits],
            "status": "success",
            "error": "",
            "retry_attempt": retry_attempt,
            "timestamp": now_iso(),
        }

    raise RuntimeError("unreachable")


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    bank = RepresentationBank(args.library_file)
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def guarded(session: aiohttp.ClientSession | None, item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_one(session, item, bank=bank, prompt_template=prompt_template, args=args)

    if args.mock:
        out = [await guarded(None, item) for item in rows]
    else:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        connector = aiohttp.TCPConnector(limit=max(1, args.max_concurrent * 2))
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    print(f"wrote {len(out)} TRS predictions -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve skill cards and run TRS inference.")
    parser.add_argument("input_file")
    parser.add_argument("library_file")
    parser.add_argument("prompt_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default=os.environ.get("TRS_MODEL", ""))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--input-price", type=float, default=0.0, help="Cost per 1M prompt tokens.")
    parser.add_argument("--output-price", type=float, default=0.0, help="Cost per 1M completion tokens.")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if not args.model_label:
        args.model_label = args.model or "mock-model"
    if not args.mock and not args.model:
        parser.error("--model or TRS_MODEL is required unless --mock is used")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
