#!/usr/bin/env python3
"""Per-question-type retrieval analysis: BM25 vs base dense vs FT dense.

Reproduces the 5,000-query test-split methodology used by
``evaluate_bm25_baseline.py`` (same length filters, same line order, same
record-ID split function), then buckets by ``question_type`` and reports
Recall@{1,5,10}, MRR@10, NDCG@10 for each of: BM25, base E5-base-v2,
fine-tuned E5-base-v2.

Output: paper_results/diagnostics/retrieval_by_qtype.json

Runs on Mac (MPS) or CUDA in ~10 min for the full 5k slice.

Usage:
    cd /Users/asad/Projects/Thesis/MS-GPT
    source .venv/bin/activate
    python scripts/evaluate_retrieval_by_qtype.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("retrieval_by_qtype")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# enriched.jsonl is the consolidated dataset with question_type / answer_style /
# split fields populated. consolidated_qa.jsonl pre-dates the enrichment pass
# and lacks question_type, so we use enriched.jsonl as the canonical source.
DEFAULT_DATA = PROJECT_ROOT.parent / "paper_results" / "dataset" / "enriched.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "paper_results" / "diagnostics" / "retrieval_by_qtype.json"
DEFAULT_BASE_E5 = "intfloat/e5-base-v2"
DEFAULT_FT_E5 = PROJECT_ROOT / "models" / "fine_tuned_embeddings_e5_base_v2" / "final_model"


# ---------------------------------------------------------------------------
# Inlined from src/embedding_trainers/data_utils.py & evaluate_bm25_baseline.py
# Keeping these identical so the per-type slice matches the published 5k slice.
# ---------------------------------------------------------------------------

def get_split(record_id: str, train_ratio: float = 0.85, val_ratio: float = 0.10) -> str:
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    t = int(train_ratio * 100)
    v = t + int(val_ratio * 100)
    if h < t:
        return "train"
    if h < v:
        return "val"
    return "test"


def clean_answer(text: str, max_len: int = 512) -> str:
    if not text:
        return ""
    text = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", text)
    text = re.sub(r"\(\d+(?:[,\-–]\s*\d+)*\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"doi:\s*\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"10\.\d{4,}/\S+", "", text)
    text = re.sub(
        r"(?:Fig\.|Figure|Table|Supplementary\s+(?:Fig|Table|Material))\s*\d+[A-Za-z]?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
    return text.strip()


def clean_question(text: str, max_len: int = 256) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
        if not text.endswith("?"):
            text = text + "?"
    return text.strip()


# ---------------------------------------------------------------------------
# Sampling — identical filter chain as evaluate_bm25_baseline.py
# ---------------------------------------------------------------------------

def load_test_samples(jsonl_path: Path, sample_size: int = 5000) -> dict:
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qtypes: dict[str, str] = {}
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if count >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in rec or "answer" not in rec:
                continue
            rid = rec.get("id", str(line_num))
            if get_split(rid) != "test":
                continue
            q = clean_question(rec["question"].strip())
            a = clean_answer(rec["answer"].strip())
            if not (10 <= len(q) <= 256):
                continue
            if not (20 <= len(a) <= 512):
                continue
            qid = f"q_{rid}"
            cid = f"c_{rid}"
            queries[qid] = q
            corpus[cid] = a
            qtypes[qid] = (rec.get("question_type") or "unknown").lower()
            count += 1
    logger.info("Loaded %d test queries with question_type", count)
    return {"queries": queries, "corpus": corpus, "qtypes": qtypes}


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def score_bm25(queries: dict, corpus: dict) -> dict[str, list[int]]:
    """Return {qid: [doc_indices ranked by score]} so we can compute Recall@k."""
    from rank_bm25 import BM25Okapi

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    tokenised = [tokenize(corpus[c]) for c in corpus_ids]
    bm25 = BM25Okapi(tokenised)
    rankings: dict[str, list[int]] = {}
    for i, qid in enumerate(query_ids):
        if (i + 1) % 1000 == 0:
            logger.info("  BM25 %d / %d", i + 1, len(query_ids))
        scores = bm25.get_scores(tokenize(queries[qid]))
        rankings[qid] = np.argsort(-scores)[:50].tolist()
    return rankings, corpus_ids


# ---------------------------------------------------------------------------
# Per-type metric aggregation
# ---------------------------------------------------------------------------

def per_type_metrics(rankings: dict, corpus_ids: list, qtypes: dict) -> dict:
    by_type: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "ndcg": 0.0,
    })
    corpus_pos = {cid: i for i, cid in enumerate(corpus_ids)}
    qid_to_relevant_idx: dict[str, int] = {}
    for qid in rankings:
        cid = f"c_{qid[2:]}"
        if cid in corpus_pos:
            qid_to_relevant_idx[qid] = corpus_pos[cid]

    for qid, ranking in rankings.items():
        rel_idx = qid_to_relevant_idx.get(qid)
        if rel_idx is None:
            continue
        t = qtypes.get(qid, "unknown")
        bucket = by_type[t]
        bucket["n"] += 1
        try:
            rank = ranking.index(rel_idx) + 1
        except ValueError:
            rank = 999
        if rank <= 1:
            bucket["r1"] += 1
        if rank <= 5:
            bucket["r5"] += 1
        if rank <= 10:
            bucket["r10"] += 1
        if rank <= 10:
            bucket["mrr"] += 1.0 / rank
            bucket["ndcg"] += 1.0 / float(np.log2(rank + 1))

    out: dict[str, dict] = {}
    overall = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "ndcg": 0.0}
    for t, b in by_type.items():
        n = b["n"]
        out[t] = {
            "n": n,
            "recall@1": b["r1"] / n if n else 0.0,
            "recall@5": b["r5"] / n if n else 0.0,
            "recall@10": b["r10"] / n if n else 0.0,
            "mrr@10": b["mrr"] / n if n else 0.0,
            "ndcg@10": b["ndcg"] / n if n else 0.0,
        }
        for k in ("n", "r1", "r5", "r10", "mrr", "ndcg"):
            overall[k] += b[k]
    n = overall["n"]
    out["__overall__"] = {
        "n": n,
        "recall@1": overall["r1"] / n if n else 0.0,
        "recall@5": overall["r5"] / n if n else 0.0,
        "recall@10": overall["r10"] / n if n else 0.0,
        "mrr@10": overall["mrr"] / n if n else 0.0,
        "ndcg@10": overall["ndcg"] / n if n else 0.0,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="consolidated_qa.jsonl (same source as the BM25 baseline)")
    ap.add_argument("--sample-size", type=int, default=5000)
    ap.add_argument("--base-e5", default=DEFAULT_BASE_E5,
                    help="Base E5-base-v2 model id (HuggingFace)")
    ap.add_argument("--ft-e5", default=str(DEFAULT_FT_E5),
                    help="Local fine-tuned E5-base-v2 directory")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--skip-base", action="store_true",
                    help="Skip the base-encoder pass (use cached results)")
    ap.add_argument("--skip-ft", action="store_true",
                    help="Skip the fine-tuned-encoder pass")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="Force a torch device. MPS sometimes produces NaN/Inf in matmul "
                         "for base E5 — use 'cpu' for cleanest numbers.")
    ap.add_argument("--merge", default=None,
                    help="Path to an existing retrieval_by_qtype.json to merge into "
                         "(useful when only re-running base or ft)")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    samples = load_test_samples(data_path, args.sample_size)
    queries = samples["queries"]
    corpus = samples["corpus"]
    qtypes = samples["qtypes"]

    qtype_counts: dict[str, int] = defaultdict(int)
    for t in qtypes.values():
        qtype_counts[t] += 1
    logger.info("Question-type counts (n=%d): %s", len(queries), dict(qtype_counts))

    results: dict[str, dict] = {
        "config": {
            "data": str(data_path),
            "sample_size": args.sample_size,
            "n_queries": len(queries),
            "n_corpus": len(corpus),
            "qtype_counts": dict(qtype_counts),
        }
    }

    logger.info("=== BM25 ===")
    t0 = time.time()
    bm25_rank, corpus_ids = score_bm25(queries, corpus)
    logger.info("BM25 done in %.1fs", time.time() - t0)
    results["bm25"] = per_type_metrics(bm25_rank, corpus_ids, qtypes)

    if not args.skip_base:
        logger.info("=== Base E5-base-v2 ===")
        t0 = time.time()
        rank, cids = dense_score(args.base_e5, queries, corpus, args.device)
        results["base_e5"] = per_type_metrics(rank, cids, qtypes)
        logger.info("Base E5 done in %.1fs", time.time() - t0)
    if not args.skip_ft:
        logger.info("=== Fine-tuned E5-base-v2 ===")
        t0 = time.time()
        rank, cids = dense_score(args.ft_e5, queries, corpus, args.device)
        results["ft_e5"] = per_type_metrics(rank, cids, qtypes)
        logger.info("FT E5 done in %.1fs", time.time() - t0)

    if args.merge and Path(args.merge).exists():
        prev = json.loads(Path(args.merge).read_text())
        for k, v in prev.items():
            results.setdefault(k, v)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)

    # Print a compact human-readable summary.
    print()
    print("Recall@10 per question type (n shown in parens):")
    print(f"{'type':<14}{'n':>6}  {'BM25':>8}  {'Base E5':>8}  {'FT E5':>8}  {'+FT (pp)':>10}")
    for t in sorted(results["bm25"].keys()):
        b = results["bm25"][t]["recall@10"]
        be = results.get("base_e5", {}).get(t, {}).get("recall@10", float("nan"))
        ft = results.get("ft_e5", {}).get(t, {}).get("recall@10", float("nan"))
        n = results["bm25"][t]["n"]
        delta = (ft - be) * 100 if not (np.isnan(be) or np.isnan(ft)) else float("nan")
        print(f"{t:<14}{n:>6}  {b:>8.3f}  {be:>8.3f}  {ft:>8.3f}  {delta:>10.1f}")


def dense_score(model_name_or_path, queries: dict, corpus: dict,
                device_override: str = "auto") -> tuple[dict, list]:
    """Encode + cosine-rank.

    Matches the methodology of src/embedding_trainers/evaluators.py:
    - No E5 prefixes (the published evaluator skips them)
    - Manual L2 normalisation after encoding (avoids the in-encode path
      that sometimes produces NaN/Inf on MPS for base E5)
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device_override and device_override != "auto":
        device = device_override
    else:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    logger.info("Loading %s on %s", model_name_or_path, device)
    model = SentenceTransformer(str(model_name_or_path), device=device)

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    q_texts = [queries[q] for q in query_ids]
    c_texts = [corpus[c] for c in corpus_ids]

    logger.info("Encoding %d queries...", len(q_texts))
    q_emb = model.encode(q_texts, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True)
    logger.info("Encoding %d corpus docs...", len(c_texts))
    c_emb = model.encode(c_texts, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True)

    # Replace any NaN/Inf rows (rare on MPS) with zero vectors so they
    # silently lose every comparison — they can't sneak to the top.
    q_emb = np.nan_to_num(q_emb, nan=0.0, posinf=0.0, neginf=0.0)
    c_emb = np.nan_to_num(c_emb, nan=0.0, posinf=0.0, neginf=0.0)
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    c_norm = np.linalg.norm(c_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.where(q_norm > 0, q_norm, 1.0)
    c_emb = c_emb / np.where(c_norm > 0, c_norm, 1.0)
    sims = q_emb @ c_emb.T

    rankings: dict[str, list[int]] = {}
    for i, qid in enumerate(query_ids):
        rankings[qid] = np.argsort(-sims[i])[:50].tolist()
    return rankings, corpus_ids


if __name__ == "__main__":
    main()
