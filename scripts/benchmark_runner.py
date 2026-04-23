#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Any

import aiohttp

from common import (
    answers_match,
    call_chat_completion,
    extract_final_answer,
    load_jsonl,
    now_iso,
    render_prompt,
    response_texts,
    usage_counts,
    write_jsonl,
)
from retrieve_and_infer import RepresentationBank, mock_inference_response


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECT_PROMPT = ROOT / "prompts" / "direct_prompt.txt"
DEFAULT_TRS_PROMPT = ROOT / "prompts" / "trs_prompt.txt"


VERIFY_PROMPT = """You are an answer checker for a benign academic benchmark.
Judge only whether the candidate's final answer matches the official answer.
Respond with exactly CORRECT or INCORRECT.

Problem:
{question}

Official answer:
{gold}

Candidate answer:
{candidate}
""".strip()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "run"


def benchmark_name(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "benchmark"
    return str(rows[0].get("benchmark") or "benchmark")


async def verify_candidate(
    session: aiohttp.ClientSession | None,
    *,
    question: str,
    gold: str,
    candidate: str,
    args: argparse.Namespace,
) -> tuple[str, str, int | None]:
    if not gold:
        return "UNKNOWN", "", None
    if args.verify == "none":
        return "SKIPPED", "", None
    if args.verify == "exact" or args.mock:
        verdict = "CORRECT" if answers_match(candidate, gold) else "INCORRECT"
        return verdict, verdict, int(verdict == "CORRECT")

    assert session is not None
    prompt = VERIFY_PROMPT.format(question=question, gold=gold, candidate=candidate)
    result = await call_chat_completion(
        session,
        prompt,
        model=args.verify_model,
        temperature=0.0,
        max_tokens=32,
        timeout_tag="trs_verifier",
    )
    if result["status"] != "success":
        return "UNKNOWN", result["error"], None
    _, content = response_texts(result["response"])
    first = content.splitlines()[0].strip().upper() if content else "UNKNOWN"
    if "INCORRECT" in first:
        return "INCORRECT", content, 0
    if "CORRECT" in first:
        return "CORRECT", content, 1
    return "UNKNOWN", content, None


async def run_generation(
    session: aiohttp.ClientSession | None,
    *,
    item: dict[str, Any],
    repeat_idx: int,
    mode: str,
    bank: RepresentationBank | None,
    direct_template: str,
    trs_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    retrieved: list[dict[str, Any]] = []
    hints = ""
    if mode == "trs":
        if bank is None:
            raise ValueError("TRS mode requires --library-file")
        retrieved = bank.search(item.get("question", ""), args.top_k)
        hints = "\n\n---\n\n".join(hit["heuristic"] for hit in retrieved)
        prompt = render_prompt(trs_template, problem=item.get("question", ""), hints=hints)
    else:
        prompt = render_prompt(direct_template, problem=item.get("question", ""))

    if args.mock:
        response_text = mock_inference_response(item, hints if mode == "trs" else "")
        reasoning_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        retry_attempt = 0
    else:
        assert session is not None
        response_text = ""
        reasoning_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        retry_attempt = 0
        for retry_attempt in range(args.max_retries + 1):
            result = await call_chat_completion(
                session,
                prompt,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_tag="trs_benchmark_runner",
            )
            if result["status"] != "success":
                if retry_attempt < args.max_retries:
                    await asyncio.sleep(min(20.0, 2.0**retry_attempt))
                    continue
                return {
                    **item,
                    "repeat_idx": repeat_idx,
                    "record_id": f"{item['question_id']}__repeat_{repeat_idx}",
                    "model_label": args.model_label,
                    "gen_model": args.model,
                    "mode": mode,
                    "status": "error",
                    "error": result["error"],
                    "retry_attempt": retry_attempt,
                    "timestamp": now_iso(),
                }
            payload = result["response"]
            reasoning_text, response_text = response_texts(payload)
            usage = usage_counts(payload, prompt_text=prompt, response_text=reasoning_text + "\n" + response_text)
            break

    predicted = extract_final_answer(response_text)
    verdict, raw_verifier, is_correct = await verify_candidate(
        session,
        question=item.get("question", ""),
        gold=item.get("answer", ""),
        candidate=predicted or response_text,
        args=args,
    )
    cost = (
        usage["prompt_tokens"] / 1_000_000 * args.input_price
        + usage["completion_tokens"] / 1_000_000 * args.output_price
    )
    return {
        **item,
        "repeat_idx": repeat_idx,
        "record_id": f"{item['question_id']}__repeat_{repeat_idx}",
        "model_label": args.model_label,
        "gen_model": args.model if not args.mock else "mock-model",
        "mode": mode,
        "raw_reasoning": reasoning_text,
        "raw_model_response": response_text,
        "predicted_answer": predicted,
        "verifier_model": args.verify_model if args.verify == "llm" else args.verify,
        "gpt_verify": verdict,
        "raw_verifier_response": raw_verifier,
        "is_correct": is_correct,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost": cost,
        "cost_yuan": cost,
        "retrieved_count": len(retrieved),
        "retrieved_question_ids": [hit["question_id"] for hit in retrieved],
        "retrieved_scores": [round(hit["score"], 6) for hit in retrieved],
        "retrieved_preview": [hit["heuristic"] for hit in retrieved],
        "status": "success",
        "error": "",
        "retry_attempt": retry_attempt,
        "timestamp": now_iso(),
    }


async def run_condition(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    bank: RepresentationBank | None,
    direct_template: str,
    trs_template: str,
    output_file: Path,
    args: argparse.Namespace,
) -> None:
    tasks: list[tuple[dict[str, Any], int]] = []
    for row in rows:
        for repeat_idx in range(args.repeats):
            tasks.append((row, repeat_idx))

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def guarded(session: aiohttp.ClientSession | None, item: dict[str, Any], repeat_idx: int) -> dict[str, Any]:
        async with semaphore:
            return await run_generation(
                session,
                item=item,
                repeat_idx=repeat_idx,
                mode=mode,
                bank=bank,
                direct_template=direct_template,
                trs_template=trs_template,
                args=args,
            )

    if args.mock:
        out = [await guarded(None, item, repeat_idx) for item, repeat_idx in tasks]
    else:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        connector = aiohttp.TCPConnector(limit=max(1, args.max_concurrent * 2))
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            out = await asyncio.gather(*(guarded(session, item, repeat_idx) for item, repeat_idx in tasks))

    write_jsonl(output_file, out)
    print(f"wrote {len(out)} {mode} records -> {output_file}")


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no rows found in {args.input_file}")

    direct_template = Path(args.direct_prompt_file).read_text(encoding="utf-8")
    trs_template = Path(args.trs_prompt_file).read_text(encoding="utf-8")
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    bank = RepresentationBank(args.library_file) if "trs" in modes else None

    bench = benchmark_name(rows)
    out_dir = Path(args.results_root) / bench
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if mode not in {"direct", "trs"}:
            raise ValueError(f"unknown mode: {mode}")
        output_file = out_dir / f"{slugify(args.model_label)}_{mode}.jsonl"
        await run_condition(
            rows,
            mode=mode,
            bank=bank,
            direct_template=direct_template,
            trs_template=trs_template,
            output_file=output_file,
            args=args,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct and TRS benchmark conditions.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--library-file", default=str(ROOT / "data" / "sample_skill_cards.jsonl"))
    parser.add_argument("--results-root", default=str(ROOT / "results" / "benchmark"))
    parser.add_argument("--direct-prompt-file", default=str(DEFAULT_DIRECT_PROMPT))
    parser.add_argument("--trs-prompt-file", default=str(DEFAULT_TRS_PROMPT))
    parser.add_argument("--modes", default="direct,trs")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=os.environ.get("TRS_MODEL", ""))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--verify", choices=["exact", "llm", "none"], default="exact")
    parser.add_argument("--verify-model", default=os.environ.get("TRS_VERIFY_MODEL", ""))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--input-price", type=float, default=0.0)
    parser.add_argument("--output-price", type=float, default=0.0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if not args.model_label:
        args.model_label = args.model or "mock-model"
    if not args.mock and not args.model:
        parser.error("--model or TRS_MODEL is required unless --mock is used")
    if args.verify == "llm" and not args.verify_model:
        parser.error("--verify-model or TRS_VERIFY_MODEL is required for --verify llm")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
