#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_inference_and_judge.py

Run Baseline + RAG inference experiments on the 1k holdout set.

Experiments:
  - baseline:         No experiences, normal call
  - bm25-5:           BM25-only retrieval, top-5
  - embed_m3-5:       Embedding (bge-m3) only, top-5
  - embed_large-5:    Embedding (bge-large-en-v1.5) only, top-5
  - hybrid_m3-5:      Hybrid BM25+bge-m3, top-5
  - hybrid_large-5:   Hybrid BM25+bge-large-en-v1.5, top-5
  - random-5:         Random experience cards, top-5 (deterministic by key)

Models:
  - cloudsway/gemini-3-flash-preview
  - volcengine/doubao-seed-2-0-pro

Usage:
  python step4_inference_and_judge.py \
    --experiment baseline \
    --model gemini \
    --n_problems 50 \
    --workers 8

  python step4_inference_and_judge.py \
    --experiment hybrid_m3-5 \
    --model doubao \
    --n_problems 1000 \
    --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

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

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

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

# CP instruction — must be in every prompt (same as baselines)
CP_INSTRUCTION = "Solve this competitive programming problem using Python or C++."

CLIENT: Optional[QihooChatClient] = None

# ──────────────────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────────────────

def call_api_normal(
    user_prompt: str,
    model: str,
    api_key: str,
    timeout: int = 900,
    max_retries: int = 6,
) -> Optional[Dict]:
    """Normal API call for inference.
    
    Key decisions:
      - NO assistant prompt prefill
      - Doubao: thinking enabled server-side by default
      - 所有模型统一: temperature=1.0, top_p=1.0, top_k=0 (与 baseline 对齐)
      - gpt-oss-120b: 额外设 max_tokens=32768 (避免 8192 默认截断)
      - 其他模型不设 max_tokens (避免成为限制)
      - Only take content, NOT reasoning_content (judge sees code answer only)
    
    Returns dict with 'content', 'usage' or None on failure.
    """
    del api_key  # resolved once in main; the shared client owns it now
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


# ──────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────

def make_baseline_prompt(question: str) -> str:
    """Baseline: CP instruction + question (same as run_cp_baselines.py direct)"""
    return f"{CP_INSTRUCTION}\n\nProblem:\n{question}"


def make_rag_prompt(question: str, experiences_block: str) -> str:
    """RAG: experiences + CP instruction + question"""
    return (
        f"[EXPERIENCES]\n{experiences_block}\n[/EXPERIENCES]\n\n"
        f"{CP_INSTRUCTION}\n\nProblem:\n{question}"
    )


# ──────────────────────────────────────────────────────────
# Checkpoint logic
# ──────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: Path) -> Dict[str, dict]:
    """Load completed results from checkpoint JSONL."""
    done = {}
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
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


def append_checkpoint(ckpt_path: Path, rec: dict, lock: threading.Lock):
    """Thread-safe append to checkpoint JSONL."""
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        with ckpt_path.open("a", encoding="utf-8") as f:
            f.write(line)


def load_questions_jsonl(path: Path) -> List[dict]:
    questions: List[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("unique_key"):
                questions.append(rec)
    return questions


def load_eval_keys(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"eval_keys_path not found: {path}")

    keys: List[str] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                except Exception:
                    rec = line_s
                if isinstance(rec, str):
                    k = rec.strip()
                elif isinstance(rec, dict):
                    k = str(rec.get("unique_key") or rec.get("key") or "").strip()
                else:
                    k = ""
                if k:
                    keys.append(k)
    else:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            keys = [str(x).strip() for x in obj if str(x).strip()]
        elif isinstance(obj, dict):
            for cand in ("eval_keys_300", "eval_keys", "keys", "holdout_keys"):
                arr = obj.get(cand)
                if isinstance(arr, list):
                    keys = [str(x).strip() for x in arr if str(x).strip()]
                    break
    # de-dup while preserving order
    return list(dict.fromkeys(keys))


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step4: Inference + Judge on holdout set")
    
    parser.add_argument("--experiment", required=True,
                        choices=["baseline", "bm25-5", "embed_m3-5", "embed_large-5",
                                 "hybrid_m3-5", "hybrid_large-5", "random-5"],
                        help="Experiment configuration")
    parser.add_argument("--model", required=True, choices=["gemini", "doubao", "gptoss"],
                        help="Model to use")
    parser.add_argument("--n_problems", type=int, default=1000,
                        help="Number of holdout problems to run")
    parser.add_argument(
        "--eval_keys_path",
        default="",
        help="Optional JSON/JSONL key list. If set, evaluate exactly these keys (not front-N).",
    )
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent API workers")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of experience cards to retrieve (for non-baseline)")
    
    # Paths
    parser.add_argument("--data_dir", default="data",
                        help="Data directory with indexes and cards")
    parser.add_argument(
        "--questions_path",
        default="data/holdout_questions.jsonl",
        help="Question JSONL path. Must contain unique_key/question fields.",
    )
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--cases_path",
                        default=str(DEFAULT_CASES_PATH),
                        help="Path to judge cases JSONL")
    
    # API
    parser.add_argument("--api_key", default=None,
                        help="API key (or set env API_KEY)")
    parser.add_argument("--device", default="cuda:1",
                        help="Device for embedding model")
    
    # Judge
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip judge step (only do inference)")
    parser.add_argument("--judge_timeout", type=float, default=10.0,
                        help="Per-test timeout for judge (seconds)")
    parser.add_argument("--judge_workers", type=int, default=4,
                        help="Number of judge worker processes")
    
    args = parser.parse_args()
    
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
    judge_path = output_dir / f"judge_{exp_tag}.jsonl"
    report_path = output_dir / f"report_{exp_tag}.json"
    
    print(f"Experiment: {args.experiment}")
    print(f"Model: {model_name}")
    print(f"Output: {gens_path}")
    
    # ── Load questions ──
    questions_path = (base_dir / args.questions_path).resolve()
    all_questions = load_questions_jsonl(questions_path)
    print(f"Loaded questions: {len(all_questions)} from {questions_path}")

    eval_keys: List[str] = []
    if args.eval_keys_path:
        eval_keys_path = (base_dir / args.eval_keys_path).resolve()
        eval_keys = load_eval_keys(eval_keys_path)
        q_by_key = {q["unique_key"]: q for q in all_questions}
        questions = [q_by_key[k] for k in eval_keys if k in q_by_key]
        missing = [k for k in eval_keys if k not in q_by_key]
        if missing:
            print(f"[WARN] eval keys missing in questions file: {len(missing)}")
        if args.n_problems > 0 and args.n_problems < len(questions):
            questions = questions[:args.n_problems]
            print(f"Eval keys capped by n_problems={args.n_problems}")
        print(f"Eval mode=fixed_keys ({len(questions)} keys), path={eval_keys_path}")
    else:
        questions = all_questions[:args.n_problems]
        print(f"Eval mode=front_n, n={len(questions)}")
    
    # ── Load retrieval components (if not baseline) ──
    retriever = None
    q_vecs = None
    
    if args.experiment != "baseline":
        import pickle
        import numpy as np
        sys.path.insert(0, str(base_dir))
        from retrieval import ExperienceRetriever, format_experiences_block, FastBM25
        
        # Parse experiment config
        # Formats: bm25-5, embed_m3-5, embed_large-5, hybrid_m3-5, hybrid_large-5, random-5
        parts = args.experiment.split("-")
        strategy_part = parts[0]  # bm25, embed_m3, embed_large, hybrid_m3, hybrid_large
        
        if strategy_part == "bm25":
            strategy = "bm25"
            embed_label = None
        elif strategy_part == "embed_m3":
            strategy = "embed"
            embed_label = "bge-m3"
        elif strategy_part == "embed_large":
            strategy = "embed"
            embed_label = "bge-large-en-v1.5"
        elif strategy_part == "hybrid_m3":
            strategy = "hybrid"
            embed_label = "bge-m3"
        elif strategy_part == "hybrid_large":
            strategy = "hybrid"
            embed_label = "bge-large-en-v1.5"
        elif strategy_part == "random":
            strategy = "random"
            embed_label = None
        else:
            raise ValueError(f"Unknown strategy: {strategy_part}")
        
        # Load cards
        print("Loading cards...")
        cards = []
        with (data_dir / "cards_meta.jsonl").open() as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    cards.append(json.loads(line_s))
        print(f"  {len(cards)} cards")
        
        # Load BM25
        bm25 = None
        if strategy in ("bm25", "hybrid"):
            fast_bm25_path = data_dir / "fast_bm25_index.pkl"
            bm25_path = data_dir / "bm25_index.pkl"
            if fast_bm25_path.exists():
                print(f"Loading FastBM25...")
                with fast_bm25_path.open("rb") as f:
                    bm25 = pickle.load(f)
            elif bm25_path.exists():
                print(f"Loading rank_bm25...")
                with bm25_path.open("rb") as f:
                    bm25 = pickle.load(f)
        
        # Load FAISS
        import faiss
        faiss_index = None
        if embed_label and strategy in ("embed", "hybrid"):
            faiss_path = data_dir / f"faiss_{embed_label}.index"
            if faiss_path.exists():
                print(f"Loading FAISS ({embed_label})...")
                faiss_index = faiss.read_index(str(faiss_path))
            
            # Load pre-computed query vectors
            q_vecs_path = data_dir / f"holdout_q_vecs_{embed_label}.npy"
            if q_vecs_path.exists():
                q_all = np.load(q_vecs_path)
                if eval_keys:
                    q_keys_path = data_dir / "holdout_q_keys.json"
                    if q_keys_path.exists():
                        with q_keys_path.open("r", encoding="utf-8") as f:
                            q_keys = json.load(f)
                        key2idx = {k: i for i, k in enumerate(q_keys)}
                        idxs = [key2idx[q["unique_key"]] for q in questions if q["unique_key"] in key2idx]
                        if len(idxs) != len(questions):
                            print("[WARN] holdout_q_keys mismatch, fallback to no precomputed query vectors.")
                            q_vecs = None
                        else:
                            q_vecs = q_all[idxs]
                    else:
                        print("[WARN] holdout_q_keys.json missing, fallback to no precomputed query vectors.")
                        q_vecs = None
                else:
                    q_vecs = q_all[:len(questions)]
                if q_vecs is not None:
                    print(f"  Pre-computed query vectors: {q_vecs.shape}")
        
        retriever = ExperienceRetriever(
            cards=cards,
            bm25_index=bm25,
            faiss_index=faiss_index,
            embed_model=None,
            doc_vecs=None,
        )
        print(f"  Retriever ready: strategy={strategy}")
    
    # ── Load checkpoint ──
    done = load_checkpoint(gens_path)
    print(f"Checkpoint: {len(done)} already completed")
    
    todo = [q for q in questions if q["unique_key"] not in done]
    print(f"Remaining: {len(todo)} problems")
    
    if not todo:
        print("All problems already completed!")
    else:
        # ── Run inference ──
        ckpt_lock = threading.Lock()
        stats = Counter()
        t_start = time.time()
        
        def process_one(idx: int, q: dict) -> dict:
            key = q["unique_key"]
            question_text = q["question"]
            
            # Build prompt
            if args.experiment == "baseline":
                prompt = make_baseline_prompt(question_text)
                retrieved_card_ids = []
            else:
                q_vec_i = q_vecs[idx] if q_vecs is not None else None
                results = retriever.retrieve(
                    question_text,
                    strategy=strategy,
                    top_k=args.top_k,
                    q_vec=q_vec_i,
                    random_seed=key,
                )
                experiences_block = format_experiences_block(results, include_meta=True)
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
                    tokens_info = f", tokens={result['usage'].get('completion_tokens', '?')}"
                print(f"  [{total_done}/{len(todo)}] {key} → {rec['status']} "
                      f"({elapsed:.1f}s{tokens_info}) [{rate:.1f} q/min]")
            
            return rec
        
        # We need to map each question to its index in the full questions list for q_vec lookup
        key_to_full_idx = {q["unique_key"]: i for i, q in enumerate(questions)}
        
        print(f"\nStarting inference with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for q in todo:
                full_idx = key_to_full_idx[q["unique_key"]]
                fut = pool.submit(process_one, full_idx, q)
                futures[fut] = q["unique_key"]
            
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    key = futures[fut]
                    print(f"  [WORKER ERROR] {key}: {e}")
                    stats["worker_error"] += 1
        
        elapsed_total = time.time() - t_start
        print(f"\nInference complete: {dict(stats)}")
        print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    
    # ── Judge ──
    if not args.skip_judge:
        print(f"\n{'='*60}")
        print(f"  Judging results...")
        print(f"{'='*60}")
        
        # Reload all gens
        all_gens = load_checkpoint(gens_path)
        ok_gens = {k: v for k, v in all_gens.items() if v.get("status") == "OK"}
        print(f"  Total gens: {len(all_gens)}, OK: {len(ok_gens)}")
        
        # Check if judge already done
        judge_done = {}
        if judge_path.exists():
            judge_done = load_checkpoint(judge_path)
        print(f"  Judge already done: {len(judge_done)}")
        
        judge_todo_keys = [k for k in ok_gens if k not in judge_done]
        
        if judge_todo_keys:
            # Write a temporary gens file for judge_cp.py
            tmp_gens = output_dir / f"_tmp_gens_{exp_tag}.jsonl"
            with tmp_gens.open("w") as f:
                for key in judge_todo_keys:
                    gen = ok_gens[key]
                    f.write(json.dumps({
                        "unique_key": key,
                        "llm_output": gen["llm_output"],
                    }, ensure_ascii=False) + "\n")
            
            judge_script = str(Path(__file__).resolve().parent.parent / "scripts" / "judge" / "judge_cp_multiproc_gens.py")
            # IMPORTANT:
            # judge_cp_multiproc_gens.py writes a full output file (overwrite semantics).
            # So we must write to a temp judge file, then merge with existing judge_done.
            tmp_judge = output_dir / f"_tmp_judge_{exp_tag}.jsonl"
            
            import subprocess
            cmd = [
                sys.executable, judge_script,
                "--cases", str(args.cases_path),
                "--gens", str(tmp_gens),
                "--out", str(tmp_judge),
                "--default_timeout", str(args.judge_timeout),
                "--workers", str(args.judge_workers),
            ]
            print(f"  Running judge: {' '.join(cmd[-8:])}")
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                print(f"  Judge failed with code={proc.returncode}")
            else:
                # Merge old judge_done + new batch results to avoid losing past lines.
                new_judge = load_checkpoint(tmp_judge) if tmp_judge.exists() else {}
                merged = dict(judge_done)
                merged.update(new_judge)
                with judge_path.open("w", encoding="utf-8") as f:
                    for key, rec in merged.items():
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  Judge completed")
            
            # Clean up temp file
            tmp_gens.unlink(missing_ok=True)
            tmp_judge.unlink(missing_ok=True)
        
        # ── Generate report ──
        print(f"\n  Generating report...")
        judge_results = load_checkpoint(judge_path)
        missing_judge = set(ok_gens.keys()) - set(judge_results.keys())
        if missing_judge:
            print(f"  ERROR: Judge incomplete: {len(missing_judge)} keys missing from judge results (have {len(judge_results)}, need {len(ok_gens)}). Refusing to write report.")
            sys.exit(1)
        
        status_counter = Counter()
        completion_tokens_list = []
        total_tokens_list = []
        
        for key, gen in ok_gens.items():
            jr = judge_results.get(key, {})
            status = jr.get("status", "NOT_JUDGED")
            status_counter[status] += 1
            
            usage = gen.get("usage", {})
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
            if ct is not None:
                completion_tokens_list.append(ct)
            if tt is not None:
                total_tokens_list.append(tt)
        
        import numpy as np
        
        n_total = len(ok_gens)
        n_ac = status_counter.get("AC", 0)
        ac_rate = n_ac / n_total if n_total > 0 else 0
        
        avg_ct = float(np.mean(completion_tokens_list)) if completion_tokens_list else 0
        avg_tt = float(np.mean(total_tokens_list)) if total_tokens_list else 0
        median_ct = float(np.median(completion_tokens_list)) if completion_tokens_list else 0
        
        report = {
            "experiment": args.experiment,
            "model": model_name,
            "n_problems": n_total,
            "n_ac": n_ac,
            "ac_rate": round(ac_rate, 4),
            "status_distribution": dict(status_counter),
            "avg_completion_tokens": round(avg_ct),
            "median_completion_tokens": round(median_ct),
            "avg_total_tokens": round(avg_tt),
            "tokens_per_ac": round(sum(total_tokens_list) / n_ac) if n_ac > 0 else None,
            "n_api_fail": len(all_gens) - len(ok_gens),
        }
        
        with report_path.open("w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n  ┌─ Report: {exp_tag} ─────────────────")
        print(f"  │ Problems:     {n_total}")
        print(f"  │ AC Rate:      {ac_rate:.1%} ({n_ac}/{n_total})")
        print(f"  │ Status:       {dict(status_counter)}")
        print(f"  │ Avg Compl.Tokens:   {avg_ct:.0f}")
        print(f"  │ Median Compl.Tokens: {median_ct:.0f}")
        print(f"  │ Avg Total Tokens:    {avg_tt:.0f}")
        if n_ac > 0:
            print(f"  │ Tokens/AC:    {sum(total_tokens_list)/n_ac:.0f}")
        print(f"  └────────────────────────────────────")
        print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
