#!/usr/bin/env python3
"""
build_v5_index.py — Build index with success + contrast + V5 diagnostic cards.

Merges:
1. Original success cards (unchanged)
2. Original contrast cards (unchanged)
3. V5 edge_fix and wrong_approach cards (replacing old failure cards)

Quality filters: inject_text < 250 chars, non-empty type-specific fields.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # experience_rag/
sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_+\-]*|\d+(?:\^\d+)?|[+\-*/<>=!]+")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    pass


def build_index_text(card: dict) -> str:
    """Build searchable text for BM25 + embedding retrieval."""
    parts = []
    trigger = card.get("trigger", [])
    tags = card.get("tags", [])
    do = card.get("do", [])
    avoid = card.get("avoid", [])
    check = card.get("check", [])

    if trigger:
        parts.append("TRIGGER: " + " | ".join(trigger))
    if tags:
        parts.append("TAGS: " + " ".join(tags))

    card_type = card.get("type", "")
    if card_type == "edge_fix":
        ep = card.get("edge_pattern", "")
        fh = card.get("fix_hint", "")
        if ep:
            parts.append("EDGE: " + ep)
        if fh:
            parts.append("FIX: " + fh)
    elif card_type == "wrong_approach":
        wa = card.get("wrong_approach", "")
        cd = card.get("correct_direction", "")
        if wa:
            parts.append("WRONG: " + wa)
        if cd:
            parts.append("CORRECT: " + cd)

    if do:
        parts.append("DO: " + " ; ".join(do))
    if avoid:
        parts.append("AVOID: " + " ; ".join(avoid))
    if check:
        parts.append("CHECK: " + " ; ".join(check))

    return "\n".join(parts)


def build_inject_text(card: dict) -> str:
    """Build short text injected into inference prompt."""
    card_type = card.get("type", "").lower()
    trigger = card.get("trigger", [])[:2]
    trig_str = " | ".join(trigger)

    if card_type == "edge_fix":
        edge = card.get("edge_pattern", "")
        fix = card.get("fix_hint", "")
        parts = []
        if trig_str:
            parts.append(f"EDGE CASE if {trig_str}:")
        if edge:
            parts.append(edge + ".")
        if fix:
            parts.append(f"Fix: {fix}.")
        return " ".join(parts).strip()

    elif card_type == "wrong_approach":
        wrong = card.get("wrong_approach", "")
        correct = card.get("correct_direction", "")
        parts = []
        if trig_str:
            parts.append(f"AVOID if {trig_str}:")
        if wrong:
            parts.append(f"{wrong} fails.")
        if correct:
            parts.append(f"Try: {correct}.")
        return " ".join(parts).strip()

    else:
        # Original format for success/contrast cards
        do = card.get("do", [])[:3]
        avoid = card.get("avoid", [])[:2]
        check = card.get("check", [])[:2]
        parts = []
        if trig_str:
            parts.append(f"If {trig_str}:")
        if do:
            parts.append(f"Do: {'; '.join(do)}.")
        if avoid:
            parts.append(f"Avoid: {'; '.join(avoid)}.")
        if check:
            parts.append(f"Check: {'; '.join(check)}.")
        return " ".join(parts).strip()


def bm25_tokenize(text: str) -> list:
    return TOKEN_RE.findall((text or "").lower())


def is_quality_v5_card(card: dict) -> bool:
    """Check if a V5 card meets quality thresholds."""
    ctype = card.get("type", "")
    if ctype == "edge_fix":
        ep = card.get("edge_pattern", "")
        fh = card.get("fix_hint", "")
        return len(ep) >= 5 and len(fh) >= 5
    elif ctype == "wrong_approach":
        wa = card.get("wrong_approach", "")
        return len(wa) >= 5
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5_cards", type=str,
                        default=str(Path(__file__).parent / "cards_v5_diagnostic.jsonl"))
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).parent / "data_v5"))
    parser.add_argument("--embed_model_path", type=str,
                        default="/home/jovyan/zhaoguangxiang-data/model/BAAI/bge-m3")
    parser.add_argument("--embed_label", type=str, default="bge-m3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_inject_len", type=int, default=250,
                        help="Max inject_text length (filter out longer)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── load existing cards (success + contrast) ────────────────────
    print("[1/6] Loading existing cards ...")
    orig_cards_path = ROOT / "data" / "cards_meta.jsonl"
    orig_cards = list(iter_jsonl(orig_cards_path))
    print(f"  Original cards_meta.jsonl: {len(orig_cards)}")

    success_cards = [c for c in orig_cards if c.get("type") == "success"]
    contrast_cards = [c for c in orig_cards if c.get("type") == "contrast"]
    print(f"  Success: {len(success_cards)}, Contrast: {len(contrast_cards)}")

    # Load holdout keys to exclude
    holdout_keys = set()
    holdout_path = ROOT / "data" / "holdout_q_keys.json"
    if holdout_path.exists():
        holdout_keys = set(json.loads(holdout_path.read_text()))
    print(f"  Holdout keys: {len(holdout_keys)}")

    # ─── load V5 cards ───────────────────────────────────────────────
    print("[2/6] Loading V5 diagnostic cards ...")
    v5_cards_path = Path(args.v5_cards)
    if not v5_cards_path.exists():
        print(f"  ERROR: V5 cards not found at {v5_cards_path}")
        sys.exit(1)

    v5_cards_raw = list(iter_jsonl(v5_cards_path))
    print(f"  V5 cards loaded: {len(v5_cards_raw)}")

    # Filter V5 cards
    v5_cards = []
    v5_holdout = 0
    v5_quality_filtered = 0
    for c in v5_cards_raw:
        if c.get("problem_id", "") in holdout_keys:
            v5_holdout += 1
            continue
        if not c.get("trigger"):
            continue
        if not is_quality_v5_card(c):
            v5_quality_filtered += 1
            continue
        v5_cards.append(c)

    print(f"  V5 after filtering: {len(v5_cards)} "
          f"(holdout={v5_holdout}, quality={v5_quality_filtered})")

    # ─── merge ───────────────────────────────────────────────────────
    print("[3/6] Merging cards ...")
    merged = []

    for c in success_cards:
        if c.get("problem_id", "") in holdout_keys:
            continue
        merged.append(c)
    n_success = len(merged)

    for c in contrast_cards:
        if c.get("problem_id", "") in holdout_keys:
            continue
        merged.append(c)
    n_contrast = len(merged) - n_success

    for c in v5_cards:
        merged.append(c)
    n_v5 = len(merged) - n_success - n_contrast

    print(f"  Merged: success={n_success}, contrast={n_contrast}, "
          f"v5={n_v5}, total={len(merged)}")

    # ─── build index_text and inject_text ────────────────────────────
    print("[4/6] Building index_text and inject_text ...")
    for c in merged:
        c["index_text"] = build_index_text(c)
        c["inject_text"] = build_inject_text(c)

    # Filter: index_text >= 20 chars, inject_text <= max_inject_len
    before = len(merged)
    merged = [c for c in merged if len(c.get("index_text", "")) >= 20]
    inject_filtered = 0
    if args.max_inject_len > 0:
        new_merged = []
        for c in merged:
            if c.get("type") in ("edge_fix", "wrong_approach"):
                if len(c.get("inject_text", "")) > args.max_inject_len:
                    inject_filtered += 1
                    continue
            new_merged.append(c)
        merged = new_merged

    print(f"  After filters: {len(merged)} (dropped {before - len(merged)}, "
          f"inject_len_filtered={inject_filtered})")

    # Report inject_text stats by type
    for ctype in ("success", "contrast", "edge_fix", "wrong_approach"):
        texts = [c["inject_text"] for c in merged if c.get("type") == ctype]
        if texts:
            lengths = [len(t) for t in texts]
            print(f"  {ctype}: n={len(texts)}, inject_text mean={sum(lengths)/len(lengths):.0f}, "
                  f"median={sorted(lengths)[len(lengths)//2]}")

    # Write cards_meta.jsonl
    cards_meta_path = output_dir / "cards_meta.jsonl"
    with open(cards_meta_path, "w") as f:
        for c in merged:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Written: {cards_meta_path}")

    # ─── build BM25 index ────────────────────────────────────────────
    print("[5/6] Building BM25 index ...")
    from retrieval import FastBM25

    corpus = [bm25_tokenize(c["index_text"]) for c in merged]
    bm25 = FastBM25(k1=1.5, b=0.75).fit(corpus)
    bm25_path = output_dir / "fast_bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 index: {bm25.corpus_size} docs")

    try:
        from rank_bm25 import BM25Okapi
        bm25_okapi = BM25Okapi(corpus)
        with open(output_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_okapi, f)
    except ImportError:
        print("  rank_bm25 not installed, skipping BM25Okapi")

    # ─── build embedding index ───────────────────────────────────────
    print("[6/6] Building embedding index ...")
    from sentence_transformers import SentenceTransformer
    import faiss

    embed_model = SentenceTransformer(args.embed_model_path, device=args.device)
    doc_texts = [c["index_text"] for c in merged]
    doc_vecs = embed_model.encode(doc_texts, normalize_embeddings=True,
                                   batch_size=128, show_progress_bar=True)
    doc_vecs = doc_vecs.astype("float32")

    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    faiss_path = output_dir / f"faiss_{args.embed_label}.index"
    faiss.write_index(index, str(faiss_path))
    np.save(output_dir / f"doc_vecs_{args.embed_label}.npy", doc_vecs)

    # Encode holdout questions
    holdout_q_path = ROOT / "data" / "holdout_questions.jsonl"
    if holdout_q_path.exists():
        q_keys = []
        q_texts = []
        for rec in iter_jsonl(holdout_q_path):
            q_keys.append(rec["unique_key"])
            q_texts.append(rec.get("question", ""))

        q_vecs = embed_model.encode(q_texts, normalize_embeddings=True,
                                     batch_size=128, show_progress_bar=True)
        q_vecs = q_vecs.astype("float32")
        np.save(output_dir / f"holdout_q_vecs_{args.embed_label}.npy", q_vecs)
        with open(output_dir / "holdout_q_keys.json", "w") as f:
            json.dump(q_keys, f)
        print(f"  Holdout questions encoded: {len(q_keys)}")

    # Copy holdout files
    import shutil
    for fname in ("holdout_questions.jsonl", "holdout_categories.json"):
        src = ROOT / "data" / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Report
    type_dist = Counter(c.get("type", "?") for c in merged)
    report = {
        "total_cards": len(merged),
        "type_distribution": dict(type_dist),
        "bm25_docs": bm25.corpus_size,
        "faiss_docs": index.ntotal,
        "embed_dim": doc_vecs.shape[1],
        "embed_model": args.embed_label,
    }
    with open(output_dir / f"build_index_report_{args.embed_label}.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Index Built ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
