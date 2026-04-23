#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval.py

Experience retrieval module supporting multiple retrieval strategies (for ablation):
  - bm25-only:   BM25 keyword retrieval
  - embed-only:  Dense embedding retrieval (FAISS)
  - hybrid:      BM25 + Embedding with RRF fusion

Plus:
  - MMR diversity selection
  - Soft failure-type mixing
  - Card rendering for injection

Usage:
    from retrieval import ExperienceRetriever
    
    retriever = ExperienceRetriever.load("data/", embed_label="bge-m3")
    results = retriever.retrieve(question_text, strategy="hybrid", top_k=5)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────
# Tokenization (shared with step3)
# ──────────────────────────────────────────────────────────

BM25_TOKEN_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_+\-]*"
    r"|\d+(?:\^\d+)?"
    r"|[+\-*/<>=!]+"
)

SIM_TOKEN_RE = re.compile(r"[a-z0-9_+\-]+|\d+|\^|\*")


def bm25_tokenize(text: str) -> List[str]:
    return BM25_TOKEN_RE.findall((text or "").lower())


# ──────────────────────────────────────────────────────────
# Fast BM25 via scipy sparse matrix (replaces rank_bm25)
# ──────────────────────────────────────────────────────────

class FastBM25:
    """BM25-Okapi implemented with scipy sparse matrix for fast query scoring.
    
    Build: O(N * avg_doc_len) — one-time cost.
    Query: O(|query_tokens| * avg_nnz_per_term) — typically <5ms for 44k docs.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.tf_norm: Optional[Any] = None  # scipy.sparse.csr_matrix
        self.n_docs: int = 0
        self.avgdl: float = 0.0
    
    def fit(self, tokenized_corpus: List[List[str]]) -> "FastBM25":
        """Build BM25 index from pre-tokenized corpus."""
        from scipy.sparse import csr_matrix
        
        N = len(tokenized_corpus)
        self.n_docs = N
        
        # Build vocabulary
        vocab: Dict[str, int] = {}
        for doc in tokenized_corpus:
            for token in doc:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        V = len(vocab)
        
        # Compute document lengths
        doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(doc_lens.mean()) if N > 0 else 1.0
        
        # Build TF matrix (docs × vocab) as sparse
        rows, cols, data = [], [], []
        df = np.zeros(V, dtype=np.int32)  # document frequency
        
        for doc_idx, doc in enumerate(tokenized_corpus):
            tf = Counter(doc)
            for token, count in tf.items():
                tid = vocab[token]
                rows.append(doc_idx)
                cols.append(tid)
                # Pre-compute BM25 normalized TF:
                # tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                dl = doc_lens[doc_idx]
                denom = count + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                data.append(count * (self.k1 + 1.0) / denom)
                df[tid] += 1
        
        # (vocab × docs) in CSR format — row access by term_id is O(1)
        self.tf_norm = csr_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)),
            shape=(N, V),
        ).T.tocsr()
        
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        
        return self
    
    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Score all documents for a query. Returns (n_docs,) array."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for token in set(query_tokens):  # deduplicate query tokens
            tid = self.vocab.get(token)
            if tid is None:
                continue
            # tf_norm is (vocab × docs) CSC — row tid gives us the normalized TF for all docs
            row = self.tf_norm[tid]  # sparse row
            scores[row.indices] += self.idf[tid] * row.data
        return scores
    
    def get_top_k(self, query_tokens: List[str], top_k: int = 50) -> List[int]:
        """Return top-k doc indices sorted by BM25 score (descending)."""
        scores = self.get_scores(query_tokens)
        if top_k >= self.n_docs:
            idx = np.argsort(-scores)
        else:
            # Partial sort for efficiency
            idx = np.argpartition(-scores, top_k)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        return [int(i) for i in idx if scores[i] > 0]
    
    @property
    def corpus_size(self) -> int:
        return self.n_docs


def sim_tokens(text: str) -> set:
    return set(SIM_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dense_top_k(
    q_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    """Top-k cosine/IP retrieval using precomputed dense vectors without FAISS."""
    if q_vec.ndim > 1:
        q = q_vec.reshape(-1)
    else:
        q = q_vec

    scores = doc_vecs @ q.astype("float32")
    n = len(scores)
    if n == 0:
        return [], []

    k = min(top_k, n)
    if k <= 0:
        return [], []

    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    kept = [int(i) for i in idx if scores[i] > 0]
    kept_scores = [float(scores[i]) for i in kept]
    return kept, kept_scores


# ──────────────────────────────────────────────────────────
# RRF Fusion
# ──────────────────────────────────────────────────────────

def rrf_fuse(rank_lists: List[List[int]], k: int = 60, top_k: int = 80) -> List[int]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    score: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + r + 1)
    fused = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]


# ──────────────────────────────────────────────────────────
# MMR Diversity Selection
# ──────────────────────────────────────────────────────────

def mmr_select(
    candidates: List[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[dict]:
    """Maximal Marginal Relevance selection.
    
    Each candidate dict must have: rank (int), inject_text (str).
    """
    if len(candidates) <= top_k:
        return candidates

    selected = [candidates[0]]
    sel_tok = [sim_tokens(candidates[0]["inject_text"])]
    remaining = list(candidates[1:])

    while len(selected) < top_k and remaining:
        best_i, best_score = 0, -1e18
        for i, cand in enumerate(remaining):
            relevance = 1.0 / (cand["rank"] + 1)
            cand_tok = sim_tokens(cand["inject_text"])
            max_sim = max(jaccard(cand_tok, t) for t in sel_tok)
            score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if score > best_score:
                best_score, best_i = score, i
        picked = remaining.pop(best_i)
        selected.append(picked)
        sel_tok.append(sim_tokens(picked["inject_text"]))

    return selected


# ──────────────────────────────────────────────────────────
# Soft Failure Mixing
# ──────────────────────────────────────────────────────────

def ensure_failure_mix_soft(
    selected: List[dict],
    all_candidates: List[dict],
    min_failure: int = 1,
    rank_threshold: int = 15,
) -> List[dict]:
    """Ensure at least one non-success card if available and relevant enough."""
    non_success = sum(1 for s in selected if s["type"] != "success")
    if non_success >= min_failure:
        return selected

    selected_ids = {s["card_id"] for s in selected}
    pool = [c for c in all_candidates if c["type"] != "success" and c["card_id"] not in selected_ids]
    if not pool:
        return selected

    best = min(pool, key=lambda x: x["rank"])
    if best["rank"] > rank_threshold:
        return selected

    for i in range(len(selected) - 1, -1, -1):
        if selected[i]["type"] == "success":
            selected[i] = best
            break
    return selected


# ──────────────────────────────────────────────────────────
# Card Rendering
# ──────────────────────────────────────────────────────────

_DEFAULT_META = (
    "(META) Use a hint only if its trigger matches; "
    "ignore irrelevant hints; output only final code with minimal explanation."
)
META_CARD_TEXT = os.environ.get("RAG_META_TEXT", _DEFAULT_META)


def render_card_for_injection(card: dict) -> str:
    """Render a card into compact injection text."""
    return card.get("inject_text", "")


def format_experiences_block(
    retrieved_cards: List[dict],
    include_meta: bool = True,
) -> str:
    """Format the [EXPERIENCES] block for prompt injection."""
    lines = []
    if include_meta:
        lines.append(f"- {META_CARD_TEXT}")
    for c in retrieved_cards:
        text = render_card_for_injection(c)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# ExperienceRetriever
# ──────────────────────────────────────────────────────────

class ExperienceRetriever:
    """Unified retrieval interface supporting BM25, Embedding, and Hybrid strategies."""
    
    def __init__(
        self,
        cards: List[dict],
        bm25_index: Optional[Any] = None,
        faiss_index: Optional[Any] = None,
        embed_model: Optional[Any] = None,
        doc_vecs: Optional[np.ndarray] = None,
    ):
        self.cards = cards
        self.bm25 = bm25_index
        self.faiss_index = faiss_index
        self.embed_model = embed_model
        self.doc_vecs = doc_vecs
        
        # Pre-build card_id → index mapping
        self._id_to_idx = {c["card_id"]: i for i, c in enumerate(cards)}
    
    @classmethod
    def load(
        cls,
        data_dir: str,
        embed_label: str = "bge-m3",
        embed_model_path: Optional[str] = None,
        device: str = "cuda:1",
        load_bm25: bool = True,
        load_embed: bool = True,
    ) -> "ExperienceRetriever":
        """Load pre-built indexes from data directory.
        
        Args:
            data_dir: Path to the data/ directory containing indexes
            embed_label: Label of the embedding model (matches build step)
            embed_model_path: Path to SentenceTransformer model (for query encoding at retrieval time)
            device: Device for embedding model
            load_bm25: Whether to load BM25 index
            load_embed: Whether to load embedding index
        """
        data_path = Path(data_dir)
        
        # Load cards metadata
        cards_meta_path = data_path / "cards_meta.jsonl"
        print(f"Loading cards from {cards_meta_path} ...")
        cards = []
        with cards_meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cards.append(json.loads(line))
        print(f"  Loaded {len(cards)} cards")
        
        # Load BM25 (prefer FastBM25 if available)
        bm25 = None
        if load_bm25:
            fast_bm25_path = data_path / "fast_bm25_index.pkl"
            bm25_path = data_path / "bm25_index.pkl"
            if fast_bm25_path.exists():
                print(f"Loading FastBM25 index from {fast_bm25_path} ...")
                with fast_bm25_path.open("rb") as f:
                    bm25 = pickle.load(f)
                print(f"  FastBM25 corpus size: {bm25.corpus_size}, vocab size: {len(bm25.vocab)}")
            elif bm25_path.exists():
                print(f"Loading rank_bm25 index from {bm25_path} ...")
                with bm25_path.open("rb") as f:
                    bm25 = pickle.load(f)
                print(f"  BM25 corpus size: {bm25.corpus_size}")
            else:
                print(f"  WARNING: No BM25 index found in {data_path}")
        
        # Load Embedding
        faiss_index = None
        embed_model = None
        doc_vecs = None
        if load_embed:
            try:
                import faiss as faiss_lib
            except ModuleNotFoundError:
                faiss_lib = None
                print("  WARNING: faiss not installed, will fall back to numpy dense search when possible")
            
            faiss_path = data_path / f"faiss_{embed_label}.index"
            if faiss_path.exists():
                if faiss_lib is not None:
                    print(f"Loading FAISS index from {faiss_path} ...")
                    faiss_index = faiss_lib.read_index(str(faiss_path))
                    print(f"  FAISS index: {faiss_index.ntotal} vectors, dim={faiss_index.d}")
                else:
                    print(f"  NOTE: skipping FAISS load from {faiss_path} because faiss is unavailable")
            else:
                print(f"  WARNING: FAISS index not found at {faiss_path}")
            
            vecs_path = data_path / f"doc_vecs_{embed_label}.npy"
            if vecs_path.exists():
                doc_vecs = np.load(vecs_path)
                print(f"  Doc vectors loaded: {doc_vecs.shape}")
            
            # Load embedding model for query encoding
            if embed_model_path:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model from {embed_model_path} ...")
                embed_model = SentenceTransformer(embed_model_path, device=device)
                print(f"  Loaded, dim={embed_model.get_sentence_embedding_dimension()}")
        
        return cls(
            cards=cards,
            bm25_index=bm25,
            faiss_index=faiss_index,
            embed_model=embed_model,
            doc_vecs=doc_vecs,
        )
    
    def _bm25_retrieve(self, query: str, top_k: int = 50) -> List[int]:
        """BM25 retrieval, returns list of card indices sorted by score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded")
        tokens = bm25_tokenize(query)
        if isinstance(self.bm25, FastBM25):
            return self.bm25.get_top_k(tokens, top_k=top_k)
        else:
            # Fallback for rank_bm25.BM25Okapi
            scores = self.bm25.get_scores(tokens)
            idx = np.argsort(-scores)[:top_k]
            return [int(i) for i in idx if scores[i] > 0]
    
    def _embed_retrieve(self, query: str, top_k: int = 50) -> List[int]:
        """Embedding retrieval via FAISS, returns list of card indices."""
        if self.embed_model is None:
            raise RuntimeError("Embedding model not loaded")
        q_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_vec, top_k)
            return [int(i) for i in indices[0] if i >= 0]
        if self.doc_vecs is not None:
            indices, _ = dense_top_k(q_vec, self.doc_vecs, top_k)
            return indices
        raise RuntimeError("Embedding index/doc_vecs not loaded")
    
    def _embed_retrieve_precomputed(self, q_vec: np.ndarray, top_k: int = 50) -> List[int]:
        """Embedding retrieval with pre-computed query vector."""
        if self.faiss_index is not None:
            if q_vec.ndim == 1:
                q_vec = q_vec.reshape(1, -1)
            scores, indices = self.faiss_index.search(q_vec.astype("float32"), top_k)
            return [int(i) for i in indices[0] if i >= 0]
        if self.doc_vecs is not None:
            indices, _ = dense_top_k(q_vec, self.doc_vecs, top_k)
            return indices
        raise RuntimeError("FAISS/doc_vecs not loaded")
    
    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        bm25_candidates: int = 50,
        embed_candidates: int = 50,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        failure_mix: bool = True,
        failure_rank_threshold: int = 15,
        q_vec: Optional[np.ndarray] = None,
        random_seed: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve top-k experience cards for a given question.
        
        Args:
            query: Question text
            strategy: "bm25", "embed", "hybrid", "random"
            top_k: Number of cards to return
            bm25_candidates: Initial BM25 candidates before fusion
            embed_candidates: Initial embedding candidates before fusion
            rrf_k: RRF constant (higher = less weight to top ranks)
            mmr_lambda: MMR trade-off (higher = more relevance, less diversity)
            failure_mix: Whether to softly mix in failure/contrast cards
            failure_rank_threshold: Max rank for soft failure mixing
            q_vec: Pre-computed query embedding (avoids re-encoding)
        
        Returns:
            List of card dicts with added 'rank' field
        """
        rank_lists = []

        # Random retrieval baseline: deterministic by (random_seed, query)
        if strategy == "random":
            n = min(top_k, len(self.cards))
            if n <= 0:
                return []
            seed_material = f"{random_seed or ''}|{query or ''}"
            seed_int = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed_int)
            picked_idx = rng.sample(range(len(self.cards)), n)
            picked = []
            for rank, card_idx in enumerate(picked_idx):
                card = dict(self.cards[card_idx])
                card["rank"] = rank
                card["_card_idx"] = card_idx
                picked.append(card)
            return picked
        
        if strategy in ("bm25", "hybrid"):
            if self.bm25 is not None:
                bm25_ranks = self._bm25_retrieve(query, top_k=bm25_candidates)
                rank_lists.append(bm25_ranks)
        
        if strategy in ("embed", "hybrid"):
            if q_vec is not None:
                embed_ranks = self._embed_retrieve_precomputed(q_vec, top_k=embed_candidates)
            elif self.embed_model is not None and self.faiss_index is not None:
                embed_ranks = self._embed_retrieve(query, top_k=embed_candidates)
            else:
                embed_ranks = []
            if embed_ranks:
                rank_lists.append(embed_ranks)
        
        if not rank_lists:
            return []
        
        # Fuse if multiple lists, otherwise use the single list
        if len(rank_lists) > 1:
            fused = rrf_fuse(rank_lists, k=rrf_k, top_k=max(bm25_candidates, embed_candidates))
        else:
            fused = rank_lists[0]
        
        # Build candidate dicts with rank info
        candidates = []
        for rank, card_idx in enumerate(fused):
            if card_idx < 0 or card_idx >= len(self.cards):
                continue
            card = dict(self.cards[card_idx])
            card["rank"] = rank
            card["_card_idx"] = card_idx
            candidates.append(card)
        
        if not candidates:
            return []
        
        # MMR diversity selection
        selected = mmr_select(candidates, top_k=top_k, lambda_param=mmr_lambda)
        
        # Soft failure mixing
        if failure_mix:
            selected = ensure_failure_mix_soft(
                selected, candidates,
                min_failure=1,
                rank_threshold=failure_rank_threshold,
            )
        
        return selected
    
    def retrieve_batch(
        self,
        queries: List[str],
        strategy: str = "hybrid",
        top_k: int = 5,
        q_vecs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[List[dict]]:
        """Batch retrieval for multiple queries."""
        results = []
        for i, query in enumerate(queries):
            q_vec = q_vecs[i] if q_vecs is not None else None
            result = self.retrieve(query, strategy=strategy, top_k=top_k, q_vec=q_vec, **kwargs)
            results.append(result)
        return results


# ──────────────────────────────────────────────────────────
# TwoStageRetriever — improved retrieval v2
# ──────────────────────────────────────────────────────────

class TwoStageRetriever:
    """Two-stage retrieval: question→similar_questions→cards.
    
    Stage 1: FAISS on source question texts → find similar training problems
    Stage 2: Gather cards from those problems → BM25 rerank on card text → MMR select
    
    Key improvements over ExperienceRetriever:
    - Relevance threshold: skip injection if max similarity too low
    - Dynamic top_k: fewer cards when relevance is marginal
    - Source-question-based retrieval avoids domain mismatch
    """
    
    def __init__(
        self,
        cards: List[dict],
        bm25_index: Optional[Any] = None,
        faiss_card_index: Optional[Any] = None,
        faiss_src_q_index: Optional[Any] = None,
        src_q_keys: Optional[List[str]] = None,
        pid_to_card_indices: Optional[Dict[str, List[int]]] = None,
        doc_vecs: Optional[np.ndarray] = None,
    ):
        self.cards = cards
        self.bm25 = bm25_index
        self.faiss_card_index = faiss_card_index
        self.faiss_src_q_index = faiss_src_q_index
        self.src_q_keys = src_q_keys or []
        self.pid_to_card_indices = pid_to_card_indices or {}
        self.doc_vecs = doc_vecs
    
    @classmethod
    def load(
        cls,
        data_dir: str,
        embed_label: str = "bge-m3",
        load_bm25: bool = True,
    ) -> "TwoStageRetriever":
        """Load v2 indexes from data_v2 directory."""
        data_path = Path(data_dir)
        
        # Load cards
        cards_meta_path = data_path / "cards_meta.jsonl"
        print(f"Loading cards from {cards_meta_path} ...")
        cards = []
        with cards_meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cards.append(json.loads(line))
        print(f"  Loaded {len(cards)} cards")
        
        # Load BM25
        bm25 = None
        if load_bm25:
            fast_bm25_path = data_path / "fast_bm25_index.pkl"
            if fast_bm25_path.exists():
                print(f"Loading FastBM25 from {fast_bm25_path} ...")
                with fast_bm25_path.open("rb") as f:
                    bm25 = pickle.load(f)
                print(f"  FastBM25 corpus: {bm25.corpus_size}, vocab: {len(bm25.vocab)}")
        
        # Load FAISS card index
        try:
            import faiss as faiss_lib
        except ModuleNotFoundError:
            faiss_lib = None
            print("  WARNING: faiss not installed, will fall back to numpy dense search where supported")
        faiss_card_index = None
        faiss_card_path = data_path / f"faiss_{embed_label}.index"
        if faiss_card_path.exists():
            if faiss_lib is not None:
                print(f"Loading card FAISS from {faiss_card_path} ...")
                faiss_card_index = faiss_lib.read_index(str(faiss_card_path))
                print(f"  Card FAISS: {faiss_card_index.ntotal} vectors")
            else:
                print(f"  NOTE: skipping card FAISS load from {faiss_card_path}")
        
        # Load FAISS source question index
        faiss_src_q_index = None
        src_faiss_path = data_path / f"faiss_src_questions_{embed_label}.index"
        if src_faiss_path.exists():
            if faiss_lib is not None:
                print(f"Loading source question FAISS from {src_faiss_path} ...")
                faiss_src_q_index = faiss_lib.read_index(str(src_faiss_path))
                print(f"  Source question FAISS: {faiss_src_q_index.ntotal} vectors")
            else:
                print(f"  NOTE: skipping source question FAISS load from {src_faiss_path}")
        
        # Load source question keys
        src_q_keys = []
        src_q_keys_path = data_path / "src_q_keys.json"
        if src_q_keys_path.exists():
            with src_q_keys_path.open() as f:
                src_q_keys = json.load(f)
        
        # Load pid_to_card_indices
        pid_to_card_indices = {}
        pid_map_path = data_path / "pid_to_card_indices.json"
        if pid_map_path.exists():
            with pid_map_path.open() as f:
                pid_to_card_indices = json.load(f)
        
        # Load doc vectors
        doc_vecs = None
        vecs_path = data_path / f"doc_vecs_{embed_label}.npy"
        if vecs_path.exists():
            doc_vecs = np.load(vecs_path)
            print(f"  Doc vectors: {doc_vecs.shape}")
        
        return cls(
            cards=cards,
            bm25_index=bm25,
            faiss_card_index=faiss_card_index,
            faiss_src_q_index=faiss_src_q_index,
            src_q_keys=src_q_keys,
            pid_to_card_indices=pid_to_card_indices,
            doc_vecs=doc_vecs,
        )
    
    def _retrieve_by_source_questions(
        self,
        q_vec: np.ndarray,
        top_q: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """Stage 1: Find similar source questions, return card indices + similarity scores.
        
        Returns:
            card_indices: list of card indices from matched problems
            max_sim: highest cosine similarity to any source question
        """
        if self.faiss_src_q_index is None:
            return [], []
        
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        
        scores, indices = self.faiss_src_q_index.search(q_vec.astype("float32"), top_q)
        
        card_indices = []
        sims = []
        seen_cards = set()
        
        for rank_idx in range(len(indices[0])):
            idx = int(indices[0][rank_idx])
            sim = float(scores[0][rank_idx])
            if idx < 0 or idx >= len(self.src_q_keys):
                continue
            
            pid = self.src_q_keys[idx]
            card_idxs = self.pid_to_card_indices.get(pid, [])
            for ci in card_idxs:
                if ci not in seen_cards:
                    card_indices.append(ci)
                    sims.append(sim)
                    seen_cards.add(ci)
        
        return card_indices, sims
    
    def retrieve(
        self,
        query: str,
        strategy: str = "twostage",
        top_k: int = 5,
        q_vec: Optional[np.ndarray] = None,
        relevance_threshold: float = 0.5,
        dynamic_k: bool = True,
        top_q: int = 20,
        bm25_candidates: int = 50,
        embed_candidates: int = 50,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        failure_mix: bool = True,
        failure_rank_threshold: int = 15,
        random_seed: Optional[str] = None,
    ) -> Tuple[List[dict], dict]:
        """Retrieve experience cards with v2 improvements.
        
        Strategies:
          - "twostage": question→source_questions→cards + BM25 rerank
          - "hybrid_v2": same as old hybrid but with threshold + dynamic_k
          - "bm25": BM25 only (with threshold)
          - "embed": Embedding only (with threshold)
          - "hybrid": Classic hybrid (backward compatible, no threshold)
        
        Returns:
            (cards, meta) where meta contains retrieval diagnostics
        """
        meta = {
            "strategy": strategy,
            "max_sim": 0.0,
            "effective_k": top_k,
            "n_candidates": 0,
            "skipped": False,
        }
        
        # ── Two-stage retrieval ──
        if strategy == "twostage":
            if q_vec is None:
                meta["skipped"] = True
                meta["skip_reason"] = "no_q_vec"
                return [], meta
            
            # Stage 1: Find similar source questions
            card_indices, sims = self._retrieve_by_source_questions(q_vec, top_q=top_q)
            
            if not card_indices:
                meta["skipped"] = True
                meta["skip_reason"] = "no_source_match"
                return [], meta
            
            max_sim = max(sims) if sims else 0.0
            meta["max_sim"] = max_sim
            meta["n_candidates"] = len(card_indices)
            
            # Relevance threshold: skip if too dissimilar
            if max_sim < relevance_threshold:
                meta["skipped"] = True
                meta["skip_reason"] = f"below_threshold ({max_sim:.3f} < {relevance_threshold})"
                return [], meta
            
            # Dynamic top_k based on relevance
            if dynamic_k:
                if max_sim >= 0.85:
                    effective_k = top_k       # High relevance: full top_k
                elif max_sim >= 0.7:
                    effective_k = max(3, top_k - 1)  # Medium: slightly fewer
                elif max_sim >= 0.6:
                    effective_k = max(2, top_k - 2)  # Low: fewer
                else:
                    effective_k = max(1, top_k - 3)  # Very low: minimal
                meta["effective_k"] = effective_k
            else:
                effective_k = top_k
            
            # Stage 2: BM25 rerank within candidate cards
            candidates = []
            if self.bm25 is not None:
                tokens = bm25_tokenize(query)
                bm25_scores = self.bm25.get_scores(tokens)
                scored = [(ci, bm25_scores[ci]) for ci in card_indices]
                scored.sort(key=lambda x: x[1], reverse=True)
            else:
                scored = [(ci, 1.0 / (i + 1)) for i, ci in enumerate(card_indices)]
            
            for rank, (ci, score) in enumerate(scored):
                card = dict(self.cards[ci])
                card["rank"] = rank
                card["_card_idx"] = ci
                card["_score"] = score
                candidates.append(card)
            
            # MMR diversity
            selected = mmr_select(candidates, top_k=effective_k, lambda_param=mmr_lambda)
            
            # Failure mix
            if failure_mix:
                selected = ensure_failure_mix_soft(
                    selected, candidates,
                    min_failure=1, rank_threshold=failure_rank_threshold,
                )
            
            meta["n_selected"] = len(selected)
            return selected, meta
        
        # ── hybrid_v2: old hybrid + threshold + dynamic_k ──
        if strategy == "hybrid_v2":
            rank_lists = []
            max_sim = 0.0
            
            if self.bm25 is not None:
                bm25_ranks = self.bm25.get_top_k(bm25_tokenize(query), top_k=bm25_candidates)
                rank_lists.append(bm25_ranks)
            
            if self.faiss_card_index is not None and q_vec is not None:
                qv = q_vec.reshape(1, -1).astype("float32") if q_vec.ndim == 1 else q_vec.astype("float32")
                scores, indices = self.faiss_card_index.search(qv, embed_candidates)
                embed_ranks = [int(i) for i in indices[0] if i >= 0]
                if scores[0][0] > 0:
                    max_sim = float(scores[0][0])
                rank_lists.append(embed_ranks)
            elif self.doc_vecs is not None and q_vec is not None:
                embed_ranks, embed_scores = dense_top_k(q_vec, self.doc_vecs, embed_candidates)
                if embed_scores and embed_scores[0] > 0:
                    max_sim = float(embed_scores[0])
                rank_lists.append(embed_ranks)
            
            meta["max_sim"] = max_sim
            
            if not rank_lists:
                meta["skipped"] = True
                return [], meta
            
            # Threshold check
            if max_sim < relevance_threshold:
                meta["skipped"] = True
                meta["skip_reason"] = f"below_threshold ({max_sim:.3f} < {relevance_threshold})"
                return [], meta
            
            # Dynamic k
            if dynamic_k:
                if max_sim >= 0.85:
                    effective_k = top_k
                elif max_sim >= 0.7:
                    effective_k = max(3, top_k - 1)
                elif max_sim >= 0.6:
                    effective_k = max(2, top_k - 2)
                else:
                    effective_k = max(1, top_k - 3)
                meta["effective_k"] = effective_k
            else:
                effective_k = top_k
            
            if len(rank_lists) > 1:
                fused = rrf_fuse(rank_lists, k=rrf_k, top_k=max(bm25_candidates, embed_candidates))
            else:
                fused = rank_lists[0]
            
            candidates = []
            for rank, card_idx in enumerate(fused):
                if card_idx < 0 or card_idx >= len(self.cards):
                    continue
                card = dict(self.cards[card_idx])
                card["rank"] = rank
                card["_card_idx"] = card_idx
                candidates.append(card)
            
            meta["n_candidates"] = len(candidates)
            
            if not candidates:
                meta["skipped"] = True
                return [], meta
            
            selected = mmr_select(candidates, top_k=effective_k, lambda_param=mmr_lambda)
            if failure_mix:
                selected = ensure_failure_mix_soft(selected, candidates, min_failure=1, rank_threshold=failure_rank_threshold)
            
            meta["n_selected"] = len(selected)
            return selected, meta
        
        # ── Backward-compatible strategies (bm25, embed, hybrid) ──
        # These are kept for comparison, without threshold/dynamic_k
        rank_lists = []
        
        if strategy in ("bm25", "hybrid"):
            if self.bm25 is not None:
                bm25_ranks = self.bm25.get_top_k(bm25_tokenize(query), top_k=bm25_candidates)
                rank_lists.append(bm25_ranks)
        
        if strategy in ("embed", "hybrid"):
            if self.faiss_card_index is not None and q_vec is not None:
                qv = q_vec.reshape(1, -1).astype("float32") if q_vec.ndim == 1 else q_vec.astype("float32")
                scores, indices = self.faiss_card_index.search(qv, embed_candidates)
                embed_ranks = [int(i) for i in indices[0] if i >= 0]
                rank_lists.append(embed_ranks)
        
        if not rank_lists:
            return [], meta
        
        if len(rank_lists) > 1:
            fused = rrf_fuse(rank_lists, k=rrf_k, top_k=max(bm25_candidates, embed_candidates))
        else:
            fused = rank_lists[0]
        
        candidates = []
        for rank, card_idx in enumerate(fused):
            if card_idx < 0 or card_idx >= len(self.cards):
                continue
            card = dict(self.cards[card_idx])
            card["rank"] = rank
            card["_card_idx"] = card_idx
            candidates.append(card)
        
        if not candidates:
            return [], meta
        
        selected = mmr_select(candidates, top_k=top_k, lambda_param=mmr_lambda)
        if failure_mix:
            selected = ensure_failure_mix_soft(selected, candidates, min_failure=1, rank_threshold=failure_rank_threshold)
        
        meta["n_selected"] = len(selected)
        return selected, meta


# ──────────────────────────────────────────────────────────
# CLI for quick testing
# ──────────────────────────────────────────────────────────

def main():
    """Quick test: retrieve experiences for a sample question."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test experience retrieval")
    parser.add_argument("--data_dir", default="data", help="Data directory with indexes")
    parser.add_argument("--embed_label", default="bge-m3")
    parser.add_argument("--embed_model", default="/home/jovyan/zhaoguangxiang-data/model/BAAI/bge-m3")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--strategy", default="hybrid", choices=["bm25", "embed", "hybrid"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None, help="Test query (if not given, uses first holdout question)")
    args = parser.parse_args()
    
    retriever = ExperienceRetriever.load(
        data_dir=args.data_dir,
        embed_label=args.embed_label,
        embed_model_path=args.embed_model,
        device=args.device,
    )
    
    # Get test query
    if args.query:
        query = args.query
    else:
        holdout_q_path = Path(args.data_dir) / "holdout_questions.jsonl"
        if holdout_q_path.exists():
            with holdout_q_path.open() as f:
                first = json.loads(f.readline())
                query = first["question"]
                print(f"\nUsing holdout question: {first['unique_key']}")
                print(f"Question preview: {query[:200]}...\n")
        else:
            query = "Given an array of n integers, find the longest increasing subsequence. Constraints: n ≤ 10^5."
            print(f"\nUsing default test query.\n")
    
    # Test all strategies
    for strategy in ["bm25", "embed", "hybrid"]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}, top_k={args.top_k}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(query, strategy=strategy, top_k=args.top_k)
        
        for i, card in enumerate(results):
            print(f"\n  [{i+1}] rank={card['rank']}, type={card['type']}, card_id={card['card_id'][:40]}")
            print(f"      trigger: {card.get('trigger', [])[:2]}")
            print(f"      inject:  {card.get('inject_text', '')[:120]}...")
        
        # Show the full experiences block
        block = format_experiences_block(results, include_meta=True)
        tokens_est = len(block) / 4
        print(f"\n  → Experiences block: {len(block)} chars (~{tokens_est:.0f} tokens)")


if __name__ == "__main__":
    main()
