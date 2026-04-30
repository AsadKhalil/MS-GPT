#!/usr/bin/env python3
"""Zero-shot RAG-QA baseline on the MSQA-Bench test set using the OpenAI API.

Addresses reviewer concern: no zero-shot strong-LLM baseline. Uses the same
1000-query test subsample, the same chat format
(``src.llm_trainers.data_utils.format_chat_messages``), and the same RAG
system prompt as the fine-tuned eval, so numbers are directly comparable
to the QLoRA-fine-tuned LLMs.

Computes: ROUGE-1/2/L (rouge-score), token-F1/precision/recall, exact match,
length stats. BERTScore and NLI faithfulness require torch and are deferred
— predictions are saved as JSONL so the GPU box can re-score them later.

Resumable: completed records (by ``annotation_id``) are skipped on re-run.

Usage:
    source .venv/bin/activate
    python scripts/eval_zero_shot_openai.py \\
        --model gpt-4o-mini-2024-07-18 \\
        --sample-size 1000

Cost: ~$0.20 for gpt-4o-mini @ 1000 queries; ~$3 for gpt-4o.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zero_shot")


SYSTEM_PROMPT_RAG = (
    "You are a scientific assistant. Answer the question accurately "
    "based on the provided context."
)
SYSTEM_PROMPT_DIRECT = (
    "You are a scientific assistant with deep expertise in mass spectrometry "
    "and related analytical techniques. Answer the question accurately and concisely."
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def truncate_context(context: str, max_length: int = 1536) -> str:
    if not context or len(context) <= max_length:
        return context
    truncated = context[:max_length]
    last_period = truncated.rfind(". ")
    if last_period > max_length * 0.5:
        return truncated[: last_period + 1]
    return truncated.rsplit(" ", 1)[0]


def build_messages(record: dict, max_context_len: int = 1536):
    question = (record.get("question") or "").strip()
    context = (record.get("context") or "").strip()
    if context:
        ctx_t = truncate_context(context, max_context_len)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": f"Context: {ctx_t}\n\nQuestion: {question}"},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": question},
    ]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def token_f1(pred: str, ref: str):
    p_toks = tokenize(pred)
    r_toks = tokenize(ref)
    if not p_toks or not r_toks:
        return (0.0, 0.0, 0.0)
    common = Counter(p_toks) & Counter(r_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return (0.0, 0.0, 0.0)
    precision = n_common / len(p_toks)
    recall = n_common / len(r_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-jsonl", type=Path,
                   default=Path("paper_results/dataset/splits/test.jsonl"))
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--output-name", default=None,
                   help="Directory tag under paper_results/model_results/llms/. Default: <model>_zeroshot.")
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-context-chars", type=int, default=1536)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many records (debug).")
    args = p.parse_args(argv)

    load_env_file(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        return 1

    try:
        from openai import OpenAI
        from rouge_score import rouge_scorer
    except Exception as e:
        logger.error("Missing dependency (%s). Install: pip install openai rouge-score", e)
        return 1

    if not args.test_jsonl.exists():
        logger.error("Test JSONL not found: %s", args.test_jsonl)
        return 1

    out_tag = args.output_name or f"{args.model.replace('/', '_').replace('-', '_')}_zeroshot"
    out_dir = Path("paper_results/model_results/llms") / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"predictions_test_n{args.sample_size}.jsonl"
    metrics_path = out_dir / f"eval_results_test_n{args.sample_size}.json"

    # Load records: deterministic first-N (matches eval_llms_1k.py)
    logger.info("Loading test records from %s", args.test_jsonl)
    records = []
    with args.test_jsonl.open() as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= args.sample_size:
                break
    if args.limit is not None:
        records = records[: args.limit]
    logger.info("Loaded %d test records", len(records))

    # Resume: skip completed predictions
    done_ids = set()
    if preds_path.exists():
        with preds_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        logger.info("Resuming: %d records already predicted", len(done_ids))

    client = OpenAI()

    t0 = time.time()
    n_ok = n_err = 0
    with preds_path.open("a") as f:
        for i, rec in enumerate(records, 1):
            rid = rec.get("id") or f"row_{i}"
            if rid in done_ids:
                continue
            messages = build_messages(rec, args.max_context_chars)
            for attempt in range(1, 4):
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_output_tokens,
                    )
                    pred = (resp.choices[0].message.content or "").strip()
                    f.write(json.dumps({
                        "id": rid,
                        "question": rec.get("question", ""),
                        "reference": rec.get("answer", ""),
                        "context": rec.get("context", ""),
                        "has_context": bool(rec.get("context")),
                        "prediction": pred,
                        "model": args.model,
                    }) + "\n")
                    f.flush()
                    n_ok += 1
                    if i % 25 == 0 or i <= 5:
                        elapsed = time.time() - t0
                        rate = n_ok / max(elapsed, 1e-3)
                        eta = (len(records) - i) / max(rate, 1e-6)
                        logger.info("[%4d/%d] %s pred_chars=%d rate=%.1f/s eta=%.0fs",
                                    i, len(records), rid, len(pred), rate, eta)
                    break
                except Exception as e:
                    if attempt == 3:
                        n_err += 1
                        logger.warning("[%d] %s: ERROR %s", i, rid, e)
                    else:
                        time.sleep(2 ** attempt)

    logger.info("Generation done. ok=%d err=%d elapsed=%.1fs", n_ok, n_err, time.time() - t0)

    # Score
    logger.info("Computing ROUGE / token-F1 / exact match...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1s, r2s, rls = [], [], []
    f1s, ps, rs = [], [], []
    ems = []
    pred_lens, ref_lens = [], []
    n = 0
    with preds_path.open() as f:
        for line in f:
            r = json.loads(line)
            pred = r["prediction"]
            ref = r["reference"]
            sc = scorer.score(ref, pred)
            r1s.append(sc["rouge1"].fmeasure)
            r2s.append(sc["rouge2"].fmeasure)
            rls.append(sc["rougeL"].fmeasure)
            pr, rc, f1 = token_f1(pred, ref)
            ps.append(pr); rs.append(rc); f1s.append(f1)
            ems.append(1 if normalize(pred) == normalize(ref) else 0)
            pred_lens.append(len(pred))
            ref_lens.append(len(ref))
            n += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "model_name": args.model,
        "config": "zero_shot",
        "num_samples": n,
        "rouge1": mean(r1s),
        "rouge2": mean(r2s),
        "rougeL": mean(rls),
        "token_precision": mean(ps),
        "token_recall": mean(rs),
        "token_f1": mean(f1s),
        "exact_match": mean(ems),
        "avg_pred_length": mean(pred_lens),
        "avg_ref_length": mean(ref_lens),
        "predictions_jsonl": str(preds_path),
        "note": (
            "BERTScore and NLI-faithfulness deferred — predictions JSONL is the "
            "input for a follow-up torch-based scoring pass on the GPU box."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", metrics_path)

    print("\n=== Zero-shot results ===")
    print(f"  model     : {args.model}")
    print(f"  n         : {n}")
    print(f"  ROUGE-1   : {metrics['rouge1']:.3f}")
    print(f"  ROUGE-2   : {metrics['rouge2']:.3f}")
    print(f"  ROUGE-L   : {metrics['rougeL']:.3f}")
    print(f"  Token-F1  : {metrics['token_f1']:.3f}")
    print(f"  EM        : {metrics['exact_match']:.3f}")
    print(f"  pred_len  : {metrics['avg_pred_length']:.0f} chars")
    print(f"  ref_len   : {metrics['avg_ref_length']:.0f} chars")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
