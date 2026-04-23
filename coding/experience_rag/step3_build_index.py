#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_build_index.py

P3: Build BM25 + Embedding indexes from experience cards for retrieval.

Steps:
  1) Load cards_raw.jsonl, filter schema-drift, build index_text
  2) Load split_keys.json to exclude holdout keys (no data leakage)
  3) Build BM25 index (rank_bm25) → save as pickle
  4) Build Embedding index (faiss) with configurable model → save as .faiss + metadata
  5) Also encode holdout questions for later retrieval testing

Embedding models supported (for ablation):
  - BAAI/bge-m3           (local, 1024 dim, multilingual, strong retrieval)
  - BAAI/bge-large-en-v1.5 (download needed, 1024 dim, English-focused)

Usage:
  python step3_build_index.py --embed_model /path/to/BAAI/bge-m3
  python step3_build_index.py --embed_model BAAI/bge-large-en-v1.5 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

STANDARD_TYPES = {"success", "failure", "contrast", "meta"}

# Tokenizer for BM25 — captures algorithm keywords, operators, numbers
BM25_TOKEN_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_+\-]*"   # identifiers/keywords
    r"|\d+(?:\^\d+)?"               # numbers like 10^5
    r"|[+\-*/<>=!]+"                # operators
)


# ──────────────────────────────────────────────────────────
# Data loading & cleaning
# ──────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_index_text(card: dict) -> str:
    """Build the text used for both BM25 and embedding indexing.
    
    Includes trigger/tags/do/avoid/check — the core actionable fields.
    Excludes detours/risk to avoid narrative noise polluting retrieval signal.
    """
    trigger = card.get("trigger", [])
    tags = card.get("tags", [])
    do = card.get("do", [])
    avoid = card.get("avoid", [])
    check = card.get("check", [])

    parts = []
    if trigger:
        parts.append("TRIGGER: " + " | ".join(trigger))
    if tags:
        parts.append("TAGS: " + " ".join(tags))
    if do:
        parts.append("DO: " + " ; ".join(do))
    if avoid:
        parts.append("AVOID: " + " ; ".join(avoid))
    if check:
        parts.append("CHECK: " + " ; ".join(check))
    return "\n".join(parts)


def build_inject_text(card: dict) -> str:
    """Build the short text that gets injected into the prompt at inference time."""
    trig = " | ".join(card.get("trigger", [])[:2])
    do = "; ".join(card.get("do", [])[:3])
    avoid = "; ".join(card.get("avoid", [])[:2])
    check = "; ".join(card.get("check", [])[:2])
    parts = []
    if trig:
        parts.append(f"If {trig}: ")
    if do:
        parts.append(f"Do: {do}. ")
    if avoid:
        parts.append(f"Avoid: {avoid}. ")
    if check:
        parts.append(f"Check: {check}.")
    return "".join(parts).strip()


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize for BM25. Lowercased, captures identifiers/keywords/numbers/operators."""
    return BM25_TOKEN_RE.findall((text or "").lower())


def _load_holdout_questions(flash_input_path: Path, holdout_keys: Set[str]) -> Dict[str, str]:
    """Stream-load the large Flash JSON array to extract holdout question texts.
    
    The file is a JSON array of objects with 'unique_key' and 'question' fields.
    We stream it to avoid loading 1.4G entirely into memory.
    """
    import ijson
    
    holdout_questions: Dict[str, str] = {}
    
    try:
        with flash_input_path.open("rb") as f:
            for rec in ijson.items(f, "item"):
                key = rec.get("unique_key")
                if key in holdout_keys:
                    question = rec.get("question", "")
                    if question:
                        holdout_questions[key] = question
                if len(holdout_questions) >= len(holdout_keys):
                    break
    except ImportError:
        # Fallback: load entire JSON (uses more memory but works without ijson)
        print("  NOTE: ijson not installed, loading full JSON into memory ...")
        with flash_input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            key = rec.get("unique_key")
            if key in holdout_keys:
                question = rec.get("question", "")
                if question:
                    holdout_questions[key] = question
        del data
    
    return holdout_questions


def load_and_clean_cards(
    cards_path: Path,
    exclude_keys: Set[str],
) -> Tuple[List[dict], dict]:
    """Load cards, filter schema drift & holdout, build index_text/inject_text.
    
    Returns:
        cards: list of cleaned card dicts (with index_text, inject_text added)
        stats: dict of cleaning statistics
    """
    stats = Counter()
    cards = []
    
    for row in iter_jsonl(cards_path):
        stats["total_raw"] += 1
        
        card_type = row.get("type", "")
        if card_type not in STANDARD_TYPES:
            stats["filtered_schema_drift"] += 1
            continue
        
        problem_id = row.get("problem_id", "")
        if problem_id in exclude_keys:
            stats["filtered_holdout"] += 1
            continue
        
        # Validate required fields
        trigger = row.get("trigger", [])
        if not trigger or not isinstance(trigger, list):
            stats["filtered_no_trigger"] += 1
            continue
        
        index_text = build_index_text(row)
        inject_text = build_inject_text(row)
        
        if len(index_text.strip()) < 20:
            stats["filtered_too_short"] += 1
            continue
        
        card = {
            "card_id": row.get("card_id", ""),
            "problem_id": problem_id,
            "type": card_type,
            "category": row.get("category", ""),
            "trigger": trigger,
            "tags": row.get("tags", []),
            "do": row.get("do", []),
            "avoid": row.get("avoid", []),
            "check": row.get("check", []),
            "detours": row.get("detours", []),
            "risk": row.get("risk", ""),
            "complexity": row.get("complexity", ""),
            "index_text": index_text,
            "inject_text": inject_text,
        }
        cards.append(card)
        stats["kept"] += 1
        stats[f"type_{card_type}"] += 1
    
    return cards, dict(stats)


# ──────────────────────────────────────────────────────────
# BM25 Index
# ──────────────────────────────────────────────────────────

def build_bm25_index(cards: List[dict]) -> Any:
    """Build BM25Okapi index from card index_texts."""
    from rank_bm25 import BM25Okapi
    
    corpus = [bm25_tokenize(c["index_text"]) for c in cards]
    bm25 = BM25Okapi(corpus)
    return bm25


def build_fast_bm25_index(cards: List[dict]):
    """Build FastBM25 (scipy sparse matrix) index — orders of magnitude faster than rank_bm25."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retrieval import FastBM25
    
    corpus = [bm25_tokenize(c["index_text"]) for c in cards]
    bm25 = FastBM25(k1=1.5, b=0.75).fit(corpus)
    return bm25


# ──────────────────────────────────────────────────────────
# Embedding Index
# ──────────────────────────────────────────────────────────

def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 128,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts with a SentenceTransformer model."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
    )
    return vecs.astype("float32")


def build_faiss_index(vectors: np.ndarray):
    """Build a FAISS IndexFlatIP (inner product = cosine on normalized vectors)."""
    import faiss
    
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P3: Build BM25 + Embedding indexes for experience retrieval")
    
    # Input paths
    parser.add_argument(
        "--cards_raw",
        default="../experience_summary_outputs_full_v2_nomax/cards_raw.jsonl",
        help="Path to cards_raw.jsonl",
    )
    parser.add_argument(
        "--split_keys",
        default="../experience_summary_outputs_full_v2_nomax/split_keys.json",
        help="Path to split_keys.json (for holdout exclusion)",
    )
    
    # Embedding model
    parser.add_argument(
        "--embed_model",
        default="/home/jovyan/zhaoguangxiang-data/model/BAAI/bge-m3",
        help="SentenceTransformer model name or path",
    )
    parser.add_argument(
        "--embed_label",
        default="",
        help="Label for this embedding model (used in output filenames). Auto-detected if empty.",
    )
    parser.add_argument("--device", default="cuda:1", help="Device for embedding model")
    parser.add_argument("--batch_size", type=int, default=128, help="Encoding batch size")
    
    # Output
    parser.add_argument("--output_dir", default="data", help="Output directory for indexes")
    
    # Options
    parser.add_argument("--skip_bm25", action="store_true", help="Skip BM25 index building")
    parser.add_argument("--skip_embed", action="store_true", help="Skip embedding index building")
    parser.add_argument("--encode_questions", action="store_true", default=True,
                        help="Also encode holdout question texts for retrieval")
    parser.add_argument(
        "--flash_input",
        default="/home/jovyan/zhaoguangxiang-data/zhaoguangxiang-research/research/reasoning_memory/nemotron_cp_unique_questions_34729_withimages_flash.json",
        help="Flash input file for loading question texts (large JSON array)",
    )
    
    args = parser.parse_args()
    
    base = Path(__file__).resolve().parent
    cards_path = (base / args.cards_raw).resolve()
    split_path = (base / args.split_keys).resolve()
    output_dir = (base / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load split keys ──
    print("Loading split keys ...")
    with split_path.open() as f:
        split_data = json.load(f)
    holdout_keys = set(split_data.get("holdout_keys", []))
    train_keys = set(split_data.get("train_keys", []))
    print(f"  train={len(train_keys)}, holdout={len(holdout_keys)}")
    
    # ── Load & clean cards ──
    print(f"Loading and cleaning cards from {cards_path} ...")
    cards, clean_stats = load_and_clean_cards(cards_path, exclude_keys=holdout_keys)
    print(f"  Cleaning stats: {json.dumps(clean_stats, indent=2)}")
    print(f"  Final card count: {len(cards)}")
    
    # ── Save cleaned card metadata (lightweight, no vectors) ──
    cards_meta_path = output_dir / "cards_meta.jsonl"
    print(f"Saving card metadata to {cards_meta_path} ...")
    with cards_meta_path.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    
    # ── Derive embed_label ──
    if args.embed_label:
        embed_label = args.embed_label
    else:
        model_name = args.embed_model.rstrip("/").split("/")[-1]
        embed_label = model_name.replace(" ", "_").lower()
    print(f"Embed label: {embed_label}")
    
    # ── Build BM25 indexes ──
    if not args.skip_bm25:
        # Build FastBM25 (scipy sparse, fast query)
        print("\n=== Building FastBM25 index (scipy sparse) ===")
        t0 = time.time()
        fast_bm25 = build_fast_bm25_index(cards)
        elapsed = time.time() - t0
        print(f"  Built in {elapsed:.1f}s, corpus size={fast_bm25.corpus_size}, vocab size={len(fast_bm25.vocab)}")
        
        fast_bm25_path = output_dir / "fast_bm25_index.pkl"
        with fast_bm25_path.open("wb") as f:
            pickle.dump(fast_bm25, f)
        print(f"  Saved to {fast_bm25_path} ({fast_bm25_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Also build rank_bm25 (for comparison, can be slow)
        print("\n=== Building rank_bm25 BM25Okapi index (legacy) ===")
        t0 = time.time()
        bm25 = build_bm25_index(cards)
        elapsed = time.time() - t0
        print(f"  Built in {elapsed:.1f}s, corpus size={bm25.corpus_size}")
        
        bm25_path = output_dir / "bm25_index.pkl"
        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        print(f"  Saved to {bm25_path} ({bm25_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # ── Build Embedding index ──
    if not args.skip_embed:
        print(f"\n=== Building Embedding index with {args.embed_model} ===")
        from sentence_transformers import SentenceTransformer
        
        t0 = time.time()
        embed_model = SentenceTransformer(args.embed_model, device=args.device)
        print(f"  Model loaded in {time.time() - t0:.1f}s")
        print(f"  Embedding dim: {embed_model.get_sentence_embedding_dimension()}")
        print(f"  Max seq length: {embed_model.max_seq_length}")
        
        # Encode card index_texts
        print(f"  Encoding {len(cards)} card index_texts ...")
        t0 = time.time()
        index_texts = [c["index_text"] for c in cards]
        doc_vecs = encode_texts(embed_model, index_texts, batch_size=args.batch_size)
        print(f"  Encoded in {time.time() - t0:.1f}s, shape={doc_vecs.shape}")
        
        # Build FAISS index
        import faiss
        faiss_index = build_faiss_index(doc_vecs)
        
        # Save
        faiss_path = output_dir / f"faiss_{embed_label}.index"
        faiss.write_index(faiss_index, str(faiss_path))
        print(f"  FAISS index saved to {faiss_path} ({faiss_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save raw vectors too (for potential later analysis)
        vecs_path = output_dir / f"doc_vecs_{embed_label}.npy"
        np.save(vecs_path, doc_vecs)
        print(f"  Doc vectors saved to {vecs_path}")
        
        # ── Encode holdout questions ──
        if args.encode_questions:
            print("\n=== Encoding holdout questions ===")
            # Load question texts from flash_input (large JSON array)
            flash_input_path = Path(args.flash_input) if os.path.isabs(args.flash_input) else (base / args.flash_input).resolve()
            
            if flash_input_path.exists():
                print(f"  Loading question texts from {flash_input_path} ...")
                t0 = time.time()
                
                holdout_questions = _load_holdout_questions(flash_input_path, holdout_keys)
                print(f"  Loaded {len(holdout_questions)} holdout questions in {time.time() - t0:.1f}s")
                
                # Save holdout questions
                holdout_q_path = output_dir / "holdout_questions.jsonl"
                with holdout_q_path.open("w", encoding="utf-8") as f:
                    for key in sorted(holdout_questions.keys()):
                        f.write(json.dumps({"unique_key": key, "question": holdout_questions[key]}, ensure_ascii=False) + "\n")
                print(f"  Saved to {holdout_q_path}")
                
                # Encode holdout questions
                print(f"  Encoding {len(holdout_questions)} holdout questions ...")
                q_keys = sorted(holdout_questions.keys())
                q_texts = [holdout_questions[k] for k in q_keys]
                t0 = time.time()
                q_vecs = encode_texts(embed_model, q_texts, batch_size=args.batch_size)
                print(f"  Encoded in {time.time() - t0:.1f}s, shape={q_vecs.shape}")
                
                q_vecs_path = output_dir / f"holdout_q_vecs_{embed_label}.npy"
                np.save(q_vecs_path, q_vecs)
                
                q_keys_path = output_dir / "holdout_q_keys.json"
                with q_keys_path.open("w") as f:
                    json.dump(q_keys, f)
                print(f"  Saved query vectors and keys")
            else:
                print(f"  WARNING: flash_input not found at {flash_input_path}, skipping question encoding")
        
        # Cleanup GPU memory
        del embed_model, doc_vecs
        import torch
        torch.cuda.empty_cache()
    
    # ── Save build report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cards_source": str(cards_path),
        "total_cards": len(cards),
        "clean_stats": clean_stats,
        "embed_model": args.embed_model,
        "embed_label": embed_label,
        "device": args.device,
        "output_dir": str(output_dir),
        "files": {
            "cards_meta": str(cards_meta_path),
            "bm25_index": str(output_dir / "bm25_index.pkl") if not args.skip_bm25 else None,
            "faiss_index": str(output_dir / f"faiss_{embed_label}.index") if not args.skip_embed else None,
            "doc_vecs": str(output_dir / f"doc_vecs_{embed_label}.npy") if not args.skip_embed else None,
        },
    }
    report_path = output_dir / f"build_index_report_{embed_label}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Done! Report: {report_path}")


if __name__ == "__main__":
    main()
