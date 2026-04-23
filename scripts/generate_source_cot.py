#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import aiohttp

from common import (
    build_reasoning_trace,
    call_chat_completion,
    load_jsonl,
    mock_source_response,
    now_iso,
    render_prompt,
    response_texts,
    usage_counts,
    write_jsonl,
)


async def process_one(
    session: aiohttp.ClientSession | None,
    item: dict[str, Any],
    *,
    prompt_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = render_prompt(prompt_template, problem=item.get("question", ""))

    if args.mock:
        model_think, model_response = mock_source_response(item.get("question", ""), item.get("answer", ""))
        return {
            **item,
            "source_model": "mock-source",
            "model_think": model_think,
            "model_response": model_response,
            "reasoning_trace": build_reasoning_trace(model_think, model_response),
            "source_prompt_tokens": 0,
            "source_completion_tokens": 0,
            "source_total_tokens": 0,
            "status": "success",
            "error": "",
            "retry_attempt": 0,
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
            timeout_tag="trs_source_cot",
        )
        if result["status"] != "success":
            if retry_attempt < args.max_retries:
                await asyncio.sleep(min(20.0, 2.0**retry_attempt))
                continue
            return {
                **item,
                "source_model": args.model,
                "model_think": "",
                "model_response": "",
                "reasoning_trace": "",
                "status": "error",
                "error": result["error"],
                "retry_attempt": retry_attempt,
                "timestamp": now_iso(),
            }

        payload = result["response"]
        model_think, model_response = response_texts(payload)
        reasoning_trace = build_reasoning_trace(model_think, model_response)
        if not reasoning_trace and retry_attempt < args.max_retries:
            await asyncio.sleep(min(20.0, 2.0**retry_attempt))
            continue

        usage = usage_counts(payload, prompt_text=prompt, response_text=reasoning_trace)
        return {
            **item,
            "source_model": args.model,
            "model_think": model_think,
            "model_response": model_response,
            "reasoning_trace": reasoning_trace,
            "source_prompt_tokens": usage["prompt_tokens"],
            "source_completion_tokens": usage["completion_tokens"],
            "source_total_tokens": usage["total_tokens"],
            "status": "success" if reasoning_trace else "error",
            "error": "" if reasoning_trace else "empty reasoning trace",
            "retry_attempt": retry_attempt,
            "timestamp": now_iso(),
        }

    raise RuntimeError("unreachable")


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def guarded(session: aiohttp.ClientSession | None, item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_one(session, item, prompt_template=prompt_template, args=args)

    if args.mock:
        out = [await guarded(None, item) for item in rows]
    else:
        timeout = aiohttp.ClientTimeout(total=args.timeout)
        connector = aiohttp.TCPConnector(limit=max(1, args.max_concurrent * 2))
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            out = await asyncio.gather(*(guarded(session, item) for item in rows))

    write_jsonl(args.output_file, out)
    print(f"wrote {len(out)} source reasoning records -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw source reasoning traces from source problems.")
    parser.add_argument("input_file")
    parser.add_argument("prompt_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default=os.environ.get("TRS_SOURCE_MODEL") or os.environ.get("TRS_MODEL") or "")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mock", action="store_true", help="Run offline with deterministic toy responses.")
    args = parser.parse_args()

    if not args.mock and not args.model:
        parser.error("--model or TRS_SOURCE_MODEL is required unless --mock is used")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
