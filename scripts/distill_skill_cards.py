#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import aiohttp

from common import (
    call_chat_completion,
    extract_xml_content,
    load_jsonl,
    mock_skill_response,
    now_iso,
    render_prompt,
    response_texts,
    usage_counts,
    write_jsonl,
)


def trace_from_item(item: dict[str, Any]) -> str:
    trace = (item.get("reasoning_trace") or "").strip()
    if trace:
        return trace
    think = (item.get("model_think") or "").strip()
    response = (item.get("model_response") or "").strip()
    return "\n\n".join(part for part in [think, response] if part)


async def process_one(
    session: aiohttp.ClientSession | None,
    item: dict[str, Any],
    *,
    prompt_template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trace = trace_from_item(item)
    prompt = render_prompt(
        prompt_template,
        problem=item.get("question", ""),
        trace=trace,
        answer=item.get("answer", ""),
    )

    if args.mock:
        content = mock_skill_response(item.get("question", ""), trace, item.get("answer", ""))
        return {
            **item,
            "distill_model": "mock-distill",
            "heuristic": extract_xml_content(content, "learned_heuristic"),
            "keywords": extract_xml_content(content, "retrieval_keywords"),
            "raw_skill_response": content,
            "distill_prompt_tokens": 0,
            "distill_completion_tokens": 0,
            "distill_total_tokens": 0,
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
            timeout_tag="trs_skill_distill",
        )
        if result["status"] != "success":
            if retry_attempt < args.max_retries:
                await asyncio.sleep(min(20.0, 2.0**retry_attempt))
                continue
            return {
                **item,
                "distill_model": args.model,
                "heuristic": "",
                "keywords": "",
                "status": "error",
                "error": result["error"],
                "retry_attempt": retry_attempt,
                "timestamp": now_iso(),
            }

        payload = result["response"]
        _, content = response_texts(payload)
        heuristic = extract_xml_content(content, "learned_heuristic")
        keywords = extract_xml_content(content, "retrieval_keywords")
        if (not heuristic or not keywords) and retry_attempt < args.max_retries:
            await asyncio.sleep(min(20.0, 2.0**retry_attempt))
            continue

        usage = usage_counts(payload, prompt_text=prompt, response_text=content)
        return {
            **item,
            "distill_model": args.model,
            "heuristic": heuristic,
            "keywords": keywords,
            "raw_skill_response": content,
            "distill_prompt_tokens": usage["prompt_tokens"],
            "distill_completion_tokens": usage["completion_tokens"],
            "distill_total_tokens": usage["total_tokens"],
            "status": "success" if heuristic and keywords else "error",
            "error": "" if heuristic and keywords else "missing learned_heuristic or retrieval_keywords",
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
    print(f"wrote {len(out)} skill cards -> {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill source reasoning traces into retrieval skill cards.")
    parser.add_argument("input_file")
    parser.add_argument("prompt_file")
    parser.add_argument("output_file")
    parser.add_argument("--model", default=os.environ.get("TRS_DISTILL_MODEL") or os.environ.get("TRS_MODEL") or "")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mock", action="store_true", help="Run offline with deterministic toy skill cards.")
    args = parser.parse_args()

    if not args.mock and not args.model:
        parser.error("--model or TRS_DISTILL_MODEL is required unless --mock is used")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
