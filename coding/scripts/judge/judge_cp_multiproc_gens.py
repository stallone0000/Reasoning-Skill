#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge_cp_multiproc_gens.py

Multiprocess wrapper over judge_cp.py for gens JSONL input:
  {"unique_key": "...", "llm_output": "..."}
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Iterator, List

import judge_cp


CASES_IDX: Dict[str, dict] = {}
CFG: judge_cp.JudgeConfig = judge_cp.JudgeConfig()
LLM_FIELD = "llm_output"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                yield json.loads(line_s)
            except Exception:
                continue


def init_worker(cases_idx: Dict[str, dict], cfg_kwargs: dict, llm_field: str) -> None:
    global CASES_IDX, CFG, LLM_FIELD
    CASES_IDX = cases_idx
    CFG = judge_cp.JudgeConfig(**cfg_kwargs)
    LLM_FIELD = llm_field


def judge_worker(gen: dict) -> dict:
    uk = gen.get("unique_key")
    if not uk:
        return {"unique_key": "", "status": "MISSING_KEY", "lang": None}
    prob = CASES_IDX.get(uk)
    if not prob:
        return {"unique_key": uk, "status": "NO_CASES", "lang": None}
    llm_output = gen.get(LLM_FIELD, "")
    try:
        return judge_cp.judge_one(prob, llm_output, CFG)
    except Exception as e:
        return {"unique_key": uk, "status": "JUDGE_ERROR", "error": str(e)[:2000], "lang": None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Multiprocess judging for gens jsonl")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm_field", default="llm_output")
    ap.add_argument("--include_private", action="store_true")
    ap.add_argument("--float_tol", type=float, default=1e-6)
    ap.add_argument("--default_timeout", type=float, default=10.0)
    ap.add_argument("--default_mem_mb", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cases_path = Path(args.cases).resolve()
    gens_path = Path(args.gens).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading cases index from {cases_path} ...", flush=True)
    cases_idx = judge_cp.load_cases_index(str(cases_path))
    print(f"  cases_idx={len(cases_idx)}", flush=True)

    gens: List[dict] = list(iter_jsonl(gens_path))
    print(f"Loaded gens={len(gens)} from {gens_path}", flush=True)

    cfg_kwargs = {
        "float_tol": args.float_tol,
        "include_private": args.include_private,
        "default_timeout_s": args.default_timeout,
        "default_mem_mb": args.default_mem_mb,
    }

    t0 = time.time()
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        with mp.get_context("fork").Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(cases_idx, cfg_kwargs, args.llm_field),
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(judge_worker, gens, chunksize=1), 1):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if i % 10 == 0 or i == len(gens):
                    elapsed = max(1e-6, time.time() - t0)
                    rate = i / elapsed
                    print(f"[{i}/{len(gens)}] rate={rate:.2f} item/s", flush=True)
                    if i % 50 == 0:
                        f.flush()

    elapsed = time.time() - t0
    print(f"Done: wrote={written} to {out_path} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

