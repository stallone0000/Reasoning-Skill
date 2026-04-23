#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4b_inference_v2.py

Run RAG inference experiments with v2 improvements:
  - twostage-5:     Two-stage retrieval (question→source_questions→cards) + threshold + dynamic_k
  - hybrid_v2-5:    Improved hybrid (old hybrid + threshold + dynamic_k)
  - bm25-5:         BM25 (backward compatible, with deduped v2 cards)
  - hybrid-5:       Classic hybrid (backward compatible, with deduped v2 cards)
  - baseline:       No RAG

Usage:
  python step4b_inference_v2.py \
    --experiment twostage-5 \
    --model gemini \
    --n_problems 1000 \
    --workers 8

  python step4b_inference_v2.py \
    --experiment hybrid_v2-5 \
    --model doubao \
    --threshold 0.5 \
    --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import numpy as np

RM_ROOT = None
for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "LATEST_STATUS.md").exists() and (_candidate / "README.md").exists():
        RM_ROOT = _candidate
        break
if RM_ROOT is None:
    raise RuntimeError("Cannot locate reasoning_memory root")
if str(RM_ROOT) not in sys.path:
    sys.path.insert(0, str(RM_ROOT))

from rm_runtime import DEFAULT_SAMPLING_PAYLOAD, QihooChatClient, resolve_api_key

# ── Constants ──
MODEL_MAP = {
    "gemini": "cloudsway/gemini-3-flash-preview",
    "doubao": "volcengine/doubao-seed-2-0-pro",
    "gptoss": "qiniu/gpt-oss-120b",
}

API_URL = "http://api.zhinao.qihoo.net/v1/chat/completions"
API_HOST = "api.360.cn"
DEFAULT_CASES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "questions" / "nemotron_cp_cases_34799_v1.jsonl"
)

CP_INSTRUCTION = "Solve this competitive programming problem using Python or C++."

CLIENT: Optional[QihooChatClient] = None


# ── API call (same as step4) ──
def call_api_normal(
    user_prompt: str,
    model: str,
    api_key: str,
    timeout: int = 900,
    max_retries: int = 6,
) -> Optional[Dict]:
    del api_key
    if CLIENT is None:
        raise RuntimeError("Qihoo chat client is not initialized")

    result = CLIENT.chat(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=timeout,
        max_retries=max_retries,
        base_payload=DEFAULT_SAMPLING_PAYLOAD,
    )
    if result is None:
        return None
    return {
        "content": result.content,
        "usage": result.usage,
    }


# ── Prompts ──
def make_baseline_prompt(question: str) -> str:
    return f"{CP_INSTRUCTION}\n\nProblem:\n{question}"

def make_rag_prompt(question: str, experiences_block: str) -> str:
    return (
        f"[EXPERIENCES]\n{experiences_block}\n[/EXPERIENCES]\n\n"
        f"{CP_INSTRUCTION}\n\nProblem:\n{question}"
    )

def make_rag_prompt_v2(question: str, experiences_block: str) -> str:
    """v2 prompt with clearer instructions about when to use hints."""
    return (
        f"[RELEVANT EXPERIENCE HINTS]\n"
        f"Below are experience hints from solving similar problems. "
        f"Use a hint ONLY if its trigger clearly matches this problem; "
        f"IGNORE hints that seem irrelevant.\n\n"
        f"{experiences_block}\n"
        f"[/RELEVANT EXPERIENCE HINTS]\n\n"
        f"{CP_INSTRUCTION}\n\nProblem:\n{question}"
    )


# ── Checkpoint ──
def load_checkpoint(path: Path) -> Dict[str, dict]:
    done = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("unique_key")
                    if key:
                        done[key] = rec
                except Exception:
                    pass
    return done

def append_checkpoint(path: Path, rec: dict, lock: threading.Lock):
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Step4b: v2 Inference on holdout set")
    
    parser.add_argument("--experiment", required=True,
                        choices=["baseline", "bm25-5", "hybrid-5",
                                 "twostage-5", "hybrid_v2-5"],
                        help="Experiment configuration")
    parser.add_argument("--model", required=True, choices=["gemini", "doubao", "gptoss"])
    parser.add_argument("--n_problems", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    
    # v2 specific
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Relevance threshold (skip RAG if max_sim < threshold)")
    parser.add_argument("--dynamic_k", action="store_true", default=True,
                        help="Use dynamic top_k based on relevance")
    parser.add_argument("--no_dynamic_k", action="store_true",
                        help="Disable dynamic top_k")
    parser.add_argument("--prompt_version", choices=["v1", "v2"], default="v2",
                        help="Prompt version: v1=original, v2=improved instructions")
    
    # Paths
    parser.add_argument("--data_dir", default="data_v2")
    parser.add_argument("--questions_path", default="data_v2/holdout_questions.jsonl")
    parser.add_argument("--output_dir", default="results_v2")
    parser.add_argument("--embed_label", default="bge-m3")
    
    # API
    parser.add_argument("--api_key", default=None)
    
    # Judge
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--cases_path",
                        default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--judge_timeout", type=float, default=10.0)
    parser.add_argument("--judge_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.no_dynamic_k:
        args.dynamic_k = False
    
    try:
        api_key = resolve_api_key(args.api_key, required=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    global CLIENT
    CLIENT = QihooChatClient(api_key=api_key, api_url=API_URL, api_host=API_HOST)
    
    model_name = MODEL_MAP[args.model]
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_tag = f"{args.experiment}_{args.model}"
    gens_path = output_dir / f"gens_{exp_tag}.jsonl"
    
    print(f"Experiment: {args.experiment} (v2)")
    print(f"Model: {model_name}")
    print(f"Threshold: {args.threshold}, Dynamic_k: {args.dynamic_k}")
    print(f"Prompt: {args.prompt_version}")
    print(f"Data dir: {data_dir}")
    print(f"Output: {gens_path}")
    
    # ── Load questions ──
    questions_path = (base_dir / args.questions_path).resolve()
    if not questions_path.exists():
        # Fallback to old data dir
        questions_path = base_dir / "data" / "holdout_questions.jsonl"
    
    questions = []
    with questions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                questions.append(json.loads(line_s))
    questions = questions[:args.n_problems]
    print(f"Loaded {len(questions)} questions")
    
    # ── Load retriever ──
    retriever = None
    q_vecs = None
    
    if args.experiment != "baseline":
        sys.path.insert(0, str(base_dir))
        from retrieval import TwoStageRetriever, format_experiences_block
        
        retriever = TwoStageRetriever.load(
            data_dir=str(data_dir),
            embed_label=args.embed_label,
            load_bm25=True,
        )
        
        # Load pre-computed query vectors
        q_vecs_path = data_dir / f"holdout_q_vecs_{args.embed_label}.npy"
        q_keys_path = data_dir / "holdout_q_keys.json"
        
        if q_vecs_path.exists() and q_keys_path.exists():
            q_all = np.load(q_vecs_path)
            with q_keys_path.open() as f:
                q_keys = json.load(f)
            key2idx = {k: i for i, k in enumerate(q_keys)}
            idxs = [key2idx[q["unique_key"]] for q in questions if q["unique_key"] in key2idx]
            if len(idxs) == len(questions):
                q_vecs = q_all[idxs]
                print(f"  Pre-computed query vectors: {q_vecs.shape}")
            else:
                print("[WARN] q_keys mismatch, no precomputed vectors")
        
        # Determine strategy
        exp_parts = args.experiment.split("-")
        strategy = exp_parts[0]  # twostage, hybrid_v2, bm25, hybrid
        print(f"  Strategy: {strategy}")
    
    # ── Load checkpoint ──
    done = load_checkpoint(gens_path)
    todo = [q for q in questions if q["unique_key"] not in done]
    print(f"Checkpoint: {len(done)} done, {len(todo)} remaining")
    
    if not todo:
        print("All done!")
    else:
        ckpt_lock = threading.Lock()
        stats = Counter()
        skip_stats = Counter()
        t_start = time.time()
        
        key_to_idx = {q["unique_key"]: i for i, q in enumerate(questions)}
        
        def process_one(idx: int, q: dict) -> dict:
            key = q["unique_key"]
            question_text = q["question"]
            
            if args.experiment == "baseline":
                prompt = make_baseline_prompt(question_text)
                retrieved_card_ids = []
                retrieval_meta = {}
            else:
                q_vec_i = q_vecs[idx] if q_vecs is not None else None
                
                results, meta = retriever.retrieve(
                    question_text,
                    strategy=strategy,
                    top_k=args.top_k,
                    q_vec=q_vec_i,
                    relevance_threshold=args.threshold,
                    dynamic_k=args.dynamic_k,
                )
                retrieval_meta = meta
                
                if not results or meta.get("skipped"):
                    # Below threshold — fall back to baseline (no RAG)
                    prompt = make_baseline_prompt(question_text)
                    retrieved_card_ids = []
                    skip_stats["skipped"] += 1
                else:
                    experiences_block = format_experiences_block(results, include_meta=True)
                    if args.prompt_version == "v2":
                        prompt = make_rag_prompt_v2(question_text, experiences_block)
                    else:
                        prompt = make_rag_prompt(question_text, experiences_block)
                    retrieved_card_ids = [c["card_id"] for c in results]
            
            # Call API
            t0 = time.time()
            result = call_api_normal(prompt, model_name, api_key)
            elapsed = time.time() - t0
            
            if result is None:
                rec = {
                    "unique_key": key,
                    "experiment": args.experiment,
                    "model": model_name,
                    "status": "API_FAIL",
                    "elapsed_s": round(elapsed, 2),
                    "retrieved_cards": retrieved_card_ids,
                    "retrieval_meta": retrieval_meta,
                }
                stats["api_fail"] += 1
            else:
                rec = {
                    "unique_key": key,
                    "experiment": args.experiment,
                    "model": model_name,
                    "status": "OK",
                    "llm_output": result["content"],
                    "usage": result["usage"],
                    "elapsed_s": round(elapsed, 2),
                    "retrieved_cards": retrieved_card_ids,
                    "retrieval_meta": retrieval_meta,
                    "prompt_len": len(prompt),
                }
                stats["ok"] += 1
            
            append_checkpoint(gens_path, rec, ckpt_lock)
            
            total_done = stats["ok"] + stats["api_fail"]
            if total_done % 10 == 0:
                elapsed_total = time.time() - t_start
                rate = total_done / elapsed_total * 60
                tokens_info = ""
                if result and "usage" in result:
                    tokens_info = f", ct={result['usage'].get('completion_tokens', '?')}"
                skip_info = f", skipped={skip_stats['skipped']}" if skip_stats["skipped"] else ""
                print(f"  [{total_done}/{len(todo)}] {key[:20]} → {rec['status']} "
                      f"({elapsed:.1f}s{tokens_info}{skip_info}) [{rate:.1f} q/min]")
            
            return rec
        
        print(f"\nStarting inference with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for q in todo:
                full_idx = key_to_idx[q["unique_key"]]
                fut = pool.submit(process_one, full_idx, q)
                futures[fut] = q["unique_key"]
            
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [WORKER ERROR] {futures[fut]}: {e}")
                    stats["worker_error"] += 1
        
        elapsed_total = time.time() - t_start
        print(f"\nInference complete: {dict(stats)}")
        print(f"Skipped (below threshold): {skip_stats['skipped']}")
        print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    
    # ── Judge ──
    if not args.skip_judge:
        print(f"\n{'='*60}")
        print("Judging results...")
        print(f"{'='*60}")
        
        all_gens = load_checkpoint(gens_path)
        ok_gens = {k: v for k, v in all_gens.items() if v.get("status") == "OK"}
        judge_path = output_dir / f"judge_{exp_tag}.jsonl"
        judge_done = load_checkpoint(judge_path) if judge_path.exists() else {}
        judge_todo = [k for k in ok_gens if k not in judge_done]
        
        print(f"  OK gens: {len(ok_gens)}, judged: {len(judge_done)}, todo: {len(judge_todo)}")
        
        if judge_todo:
            tmp_gens = output_dir / f"_tmp_gens_{exp_tag}.jsonl"
            tmp_judge = output_dir / f"_tmp_judge_{exp_tag}.jsonl"
            with tmp_gens.open("w") as f:
                for key in judge_todo:
                    gen = ok_gens[key]
                    f.write(json.dumps({
                        "unique_key": key,
                        "llm_output": gen["llm_output"],
                    }, ensure_ascii=False) + "\n")
            
            judge_script = str(Path(__file__).resolve().parent.parent / "scripts" / "judge" / "judge_cp_multiproc_gens.py")
            
            import subprocess
            cmd = [
                sys.executable, judge_script,
                "--cases", str(args.cases_path),
                "--gens", str(tmp_gens),
                "--out", str(tmp_judge),
                "--default_timeout", str(args.judge_timeout),
                "--workers", str(args.judge_workers),
            ]
            print(f"  Running judge...")
            proc = subprocess.run(cmd, text=True)
            tmp_gens.unlink(missing_ok=True)
            if proc.returncode != 0:
                print(f"  Judge failed with code={proc.returncode}")
                tmp_judge.unlink(missing_ok=True)
            else:
                new_judge = load_checkpoint(tmp_judge) if tmp_judge.exists() else {}
                merged = dict(judge_done)
                merged.update(new_judge)
                with judge_path.open("w", encoding="utf-8") as f:
                    for key, rec in merged.items():
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tmp_judge.unlink(missing_ok=True)
        
        # Report (fail-fast if judge incomplete)
        judge_results = load_checkpoint(judge_path) if judge_path.exists() else {}
        missing_judge = set(ok_gens.keys()) - set(judge_results.keys())
        if missing_judge:
            print(f"  ERROR: Judge incomplete: {len(missing_judge)} keys missing (have {len(judge_results)}, need {len(ok_gens)}). Refusing to write report.")
            sys.exit(1)
        status_counter = Counter()
        cts = []
        for key, gen in ok_gens.items():
            jr = judge_results.get(key, {})
            status_counter[jr.get("status", "NOT_JUDGED")] += 1
            ct = gen.get("usage", {}).get("completion_tokens")
            if ct:
                cts.append(ct)
        
        n_total = len(ok_gens)
        n_ac = status_counter.get("AC", 0)
        ac_rate = n_ac / n_total if n_total > 0 else 0
        avg_ct = np.mean(cts) if cts else 0
        
        # Count how many used RAG vs fallback baseline
        n_rag = sum(1 for v in ok_gens.values() if v.get("retrieved_cards"))
        n_fallback = n_total - n_rag
        
        report = {
            "experiment": args.experiment,
            "model": model_name,
            "n_problems": n_total,
            "n_ac": n_ac,
            "ac_rate": round(ac_rate, 4),
            "status_distribution": dict(status_counter),
            "avg_completion_tokens": round(float(avg_ct)),
            "n_rag_applied": n_rag,
            "n_fallback_baseline": n_fallback,
            "threshold": args.threshold,
            "dynamic_k": args.dynamic_k,
        }
        
        report_path = output_dir / f"report_{exp_tag}.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n  ┌─ Report: {exp_tag} ──────────────────────")
        print(f"  │ Problems:     {n_total}")
        print(f"  │ AC Rate:      {ac_rate:.1%} ({n_ac}/{n_total})")
        print(f"  │ Status:       {dict(status_counter)}")
        print(f"  │ Avg CT:       {avg_ct:.0f}")
        print(f"  │ RAG applied:  {n_rag} ({n_rag/n_total:.1%})")
        print(f"  │ Fallback:     {n_fallback} ({n_fallback/n_total:.1%})")
        print(f"  └────────────────────────────────────────────")


if __name__ == "__main__":
    main()
