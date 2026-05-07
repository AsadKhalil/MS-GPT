#!/usr/bin/env python3
"""Run the end-to-end 2x2 RAG ablation for MSQA.

This evaluates four conditions:

  1. base embedding + base LLM
  2. base embedding + fine-tuned LLM adapter
  3. fine-tuned embedding + base LLM
  4. fine-tuned embedding + fine-tuned LLM adapter

Unlike scripts/run_rag_zeroshot_eval.py, this script does real retrieval:
it indexes the evaluation corpus, retrieves top-k passages for each question,
feeds those retrieved passages to the generator, and scores the generated
answer against the reference answer and retrieved evidence.

Recommended quick run:

  python3 scripts/run_rag_ablation.py \
    --sample-size 200 \
    --llm-backend local \
    --load-4bit \
    --output-dir paper_results/rag_ablation_mistral_n200

Recommended thesis run:

  python3 scripts/run_rag_ablation.py \
    --sample-size 1000 \
    --llm-backend local \
    --load-4bit \
    --output-dir paper_results/rag_ablation_mistral_n1000

For a vLLM/OpenAI-compatible server, use --llm-backend openai and provide
served model names for --served-base-llm and --served-ft-llm.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import gc
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


LOG = logging.getLogger("rag_ablation")

DEFAULT_BASE_EMBEDDING = "intfloat/e5-base-v2"
DEFAULT_BASE_LLM = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MODEL_ROOT = "models"
DEFAULT_FT_EMBEDDING_NAME = "fine_tuned_embeddings_e5_base_v2"
DEFAULT_LLM_RUN_NAME = "mistral_7b_v0.3"
SYSTEM_PROMPT_RAG = (
    "You are a scientific assistant. Answer the question accurately "
    "based on the provided context."
)
SYSTEM_PROMPT_DIRECT = (
    "You are a scientific assistant with deep expertise in mass spectrometry "
    "and related analytical techniques. Answer the question accurately and concisely."
)


@dataclass
class RetrievedRecord:
    id: str
    question: str
    answer: str
    question_type: str
    relevant_context_id: str
    retrieved_context: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]


@dataclass
class CellResult:
    condition: str
    embedding_label: str
    llm_label: str
    num_samples: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    token_f1: float
    exact_match: float
    faithfulness_score: float
    answer_quality_score: float
    avg_tokens_per_second: float
    wall_time_s: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", default="paper_results/dataset/splits/test.jsonl")
    p.add_argument("--split", default="test", help="Use records whose split field matches this value.")
    p.add_argument("--ignore-split-field", action="store_true")
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--sample-mode", choices=["stratified", "first", "random"], default="stratified")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieval-k", type=int, default=10, help="Retrieve this many passages for metrics.")
    p.add_argument("--prompt-k", type=int, default=5, help="Pass this many retrieved passages to the LLM.")
    p.add_argument("--max-context-chars", type=int, default=3000)
    p.add_argument("--max-passage-chars", type=int, default=900)
    p.add_argument(
        "--model-root",
        default=DEFAULT_MODEL_ROOT,
        help=(
            "Directory containing fine_tuned_embeddings_* and fine_tuned_llms/. "
            "Use this when models live outside the repo."
        ),
    )
    p.add_argument("--base-embedding", default=DEFAULT_BASE_EMBEDDING)
    p.add_argument(
        "--ft-embedding-name",
        default=DEFAULT_FT_EMBEDDING_NAME,
        help="Fine-tuned embedding directory name under --model-root.",
    )
    p.add_argument(
        "--ft-embedding",
        default=None,
        help="Explicit fine-tuned embedding path. Overrides --model-root/--ft-embedding-name.",
    )
    p.add_argument("--embedding-device", default="auto")
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument("--base-llm", default=DEFAULT_BASE_LLM)
    p.add_argument(
        "--llm-run-name",
        default=DEFAULT_LLM_RUN_NAME,
        help="LLM run directory name under --model-root/fine_tuned_llms/.",
    )
    p.add_argument(
        "--ft-adapter",
        default=None,
        help="Explicit LoRA adapter path. Overrides --model-root/--llm-run-name.",
    )
    p.add_argument("--llm-backend", choices=["local", "openai"], default="local")
    p.add_argument("--served-base-llm", default=None, help="OpenAI/vLLM model name for the base LLM.")
    p.add_argument("--served-ft-llm", default=None, help="OpenAI/vLLM model name for the adapter/merged LLM.")
    p.add_argument("--openai-base-url", default="http://localhost:8000/v1")
    p.add_argument("--openai-api-key", default="EMPTY")
    p.add_argument("--openai-concurrency", type=int, default=8)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--conditions",
        default="base_base,base_ft,ft_base,ft_ft",
        help="Comma-separated subset of: base_base,base_ft,ft_base,ft_ft",
    )
    p.add_argument("--output-dir", default="paper_results/rag_ablation")
    p.add_argument("--resume", action="store_true", help="Reuse cached retrieval and prediction JSONL files.")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-faithfulness", action="store_true")
    return p.parse_args()


def resolve_model_paths(args: argparse.Namespace) -> None:
    model_root = Path(args.model_root)
    if args.ft_embedding is None:
        args.ft_embedding = str(model_root / args.ft_embedding_name / "final_model")
    if args.ft_adapter is None:
        args.ft_adapter = str(model_root / "fine_tuned_llms" / args.llm_run_name / "final_adapter")


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not args.ignore_split_field and rec.get("split") and rec.get("split") != args.split:
                continue
            if not rec.get("question") or not rec.get("answer") or not rec.get("context"):
                continue
            records.append(rec)
    if not records:
        raise ValueError(f"No usable records loaded from {args.jsonl}")
    return records


def sample_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.sample_size <= 0 or args.sample_size >= len(records):
        return records
    rng = random.Random(args.seed)
    if args.sample_mode == "first":
        return records[: args.sample_size]
    if args.sample_mode == "random":
        picked = records[:]
        rng.shuffle(picked)
        return picked[: args.sample_size]

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_type.setdefault(rec.get("question_type") or "unknown", []).append(rec)
    for group in by_type.values():
        rng.shuffle(group)

    sampled: List[Dict[str, Any]] = []
    total = len(records)
    remainders: List[Tuple[float, str, int]] = []
    for qtype, group in sorted(by_type.items()):
        raw = args.sample_size * len(group) / total
        take = min(len(group), int(math.floor(raw)))
        sampled.extend(group[:take])
        remainders.append((raw - take, qtype, take))

    missing = args.sample_size - len(sampled)
    for _, qtype, already in sorted(remainders, reverse=True):
        if missing <= 0:
            break
        group = by_type[qtype]
        if already < len(group):
            sampled.append(group[already])
            missing -= 1

    rng.shuffle(sampled)
    return sampled[: args.sample_size]


def build_corpus(records: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    for i, rec in enumerate(records):
        cid = str(rec.get("context_id") or f"ctx_{i}")
        context = str(rec.get("context") or "").strip()
        if context:
            corpus[cid] = context
    return corpus


def embedding_device(explicit: str) -> str:
    if explicit and explicit.lower() != "auto":
        return explicit
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def retrieve_contexts(
    embedding_label: str,
    model_name_or_path: str,
    corpus: Dict[str, str],
    query_records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    cache_path: Path,
) -> List[RetrievedRecord]:
    if args.resume and cache_path.exists():
        LOG.info("Loading cached retrieval: %s", cache_path)
        out = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                out.append(RetrievedRecord(**json.loads(line)))
        return out

    from paper.evaluation.retrieval_baselines import EmbeddingRetriever

    retriever = EmbeddingRetriever(
        model_name_or_path=model_name_or_path,
        device=embedding_device(args.embedding_device),
        batch_size=args.embed_batch_size,
    )
    retriever.index(corpus)
    queries = {str(r.get("id")): str(r["question"]) for r in query_records}
    all_results = retriever.retrieve_batch(queries, top_k=max(args.retrieval_k, args.prompt_k))

    retrieved_records: List[RetrievedRecord] = []
    for rec in query_records:
        qid = str(rec.get("id"))
        hits = all_results[qid]
        prompt_context = format_retrieved_context(hits[: args.prompt_k], corpus, args)
        retrieved_records.append(
            RetrievedRecord(
                id=qid,
                question=str(rec["question"]),
                answer=str(rec["answer"]),
                question_type=str(rec.get("question_type") or "unknown"),
                relevant_context_id=str(rec.get("context_id")),
                retrieved_context=prompt_context,
                retrieved_ids=[doc_id for doc_id, _ in hits],
                retrieved_scores=[score for _, score in hits],
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for item in retrieved_records:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    LOG.info("Wrote %s retrieval cache: %s", embedding_label, cache_path)
    return retrieved_records


def format_retrieved_context(
    hits: Sequence[Tuple[str, float]],
    corpus: Dict[str, str],
    args: argparse.Namespace,
) -> str:
    chunks: List[str] = []
    total = 0
    for rank, (doc_id, score) in enumerate(hits, 1):
        passage = corpus.get(doc_id, "").strip().replace("\n", " ")
        if len(passage) > args.max_passage_chars:
            passage = passage[: args.max_passage_chars].rsplit(" ", 1)[0]
        block = f"[Passage {rank}; score={score:.4f}]\n{passage}"
        if total + len(block) > args.max_context_chars:
            remaining = args.max_context_chars - total
            if remaining > 200:
                chunks.append(block[:remaining].rsplit(" ", 1)[0])
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks)


def retrieval_metrics(retrieved_records: Sequence[RetrievedRecord]) -> Dict[str, float]:
    r1 = r5 = r10 = mrr = ndcg = 0.0
    n = len(retrieved_records)
    for rec in retrieved_records:
        relevant = rec.relevant_context_id
        ids = rec.retrieved_ids
        if relevant in ids[:1]:
            r1 += 1.0
        if relevant in ids[:5]:
            r5 += 1.0
        if relevant in ids[:10]:
            r10 += 1.0
        if relevant in ids[:10]:
            rank = ids.index(relevant) + 1
            mrr += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
    denom = max(n, 1)
    return {
        "recall_at_1": r1 / denom,
        "recall_at_5": r5 / denom,
        "recall_at_10": r10 / denom,
        "mrr_at_10": mrr / denom,
        "ndcg_at_10": ndcg / denom,
    }


def load_local_model(base_llm: str, adapter: Optional[str], load_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_llm, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    LOG.info("Loading local LLM: %s adapter=%s 4bit=%s", base_llm, adapter, load_4bit)
    model = AutoModelForCausalLM.from_pretrained(base_llm, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def format_messages(question: str, context: Optional[str]) -> List[Dict[str, str]]:
    if context:
        return [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": question},
    ]


def render_prompt(tokenizer, rec: RetrievedRecord) -> str:
    messages = format_messages(rec.question, rec.retrieved_context)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"System: {messages[0]['content']}\n\n"
        f"User: {messages[1]['content']}\n\n"
        "Assistant:"
    )


def generate_local(
    model,
    tokenizer,
    records: Sequence[RetrievedRecord],
    args: argparse.Namespace,
) -> Tuple[List[str], List[float]]:
    import torch

    device = next(model.parameters()).device
    predictions: List[str] = []
    tps: List[float] = []
    for i, rec in enumerate(records, 1):
        prompt = render_prompt(tokenizer, rec)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=max(args.temperature, 1e-7),
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        tps.append(len(new_tokens) / elapsed if elapsed > 0 else 0.0)
        if i % 10 == 0:
            LOG.info("Generated %d/%d (%.1f tok/s)", i, len(records), tps[-1])
    return predictions, tps


def generate_openai(
    served_model: str,
    records: Sequence[RetrievedRecord],
    args: argparse.Namespace,
) -> Tuple[List[str], List[float]]:
    if args.openai_concurrency <= 1:
        predictions: List[str] = []
        tps: List[float] = []
        for i, rec in enumerate(records, 1):
            text, speed = openai_completion(served_model, rec, args)
            predictions.append(text)
            tps.append(speed)
            if i % 10 == 0:
                LOG.info("Generated %d/%d via OpenAI backend (%.1f tok/s)", i, len(records), speed)
        return predictions, tps

    predictions = [""] * len(records)
    tps = [0.0] * len(records)
    with ThreadPoolExecutor(max_workers=args.openai_concurrency) as pool:
        futures = {
            pool.submit(openai_completion, served_model, rec, args): i
            for i, rec in enumerate(records)
        }
        for done, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            text, speed = future.result()
            predictions[idx] = text
            tps[idx] = speed
            if done % 10 == 0:
                LOG.info("Generated %d/%d via OpenAI backend", done, len(records))
    return predictions, tps


def openai_completion(
    served_model: str,
    rec: RetrievedRecord,
    args: argparse.Namespace,
) -> Tuple[str, float]:
    url = args.openai_base_url.rstrip("/") + "/chat/completions"
    messages = format_messages(rec.question, rec.retrieved_context)
    body = {
        "model": served_model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.openai_api_key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed for {url}: {exc}") from exc
    elapsed = time.perf_counter() - started
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or max(1, len(text.split()))
    speed = float(completion_tokens) / elapsed if elapsed > 0 else 0.0
    return text, speed


def load_cached_predictions(path: Path) -> Optional[Tuple[List[str], List[float]]]:
    if not path.exists():
        return None
    predictions: List[str] = []
    tps: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            predictions.append(item["prediction"])
            tps.append(float(item.get("tokens_per_second", 0.0)))
    return predictions, tps


def write_predictions(path: Path, records: Sequence[RetrievedRecord], predictions: Sequence[str], tps: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec, pred, speed in zip(records, predictions, tps):
            item = asdict(rec)
            item["prediction"] = pred
            item["tokens_per_second"] = speed
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def score_condition(
    condition: str,
    embedding_label: str,
    llm_label: str,
    records: Sequence[RetrievedRecord],
    predictions: Sequence[str],
    tps: Sequence[float],
    args: argparse.Namespace,
) -> CellResult:
    from src.llm_trainers.evaluators import (
        compute_bertscore,
        compute_exact_match,
        compute_faithfulness,
        compute_rouge,
        compute_token_f1,
    )

    started = time.perf_counter()
    references = [r.answer for r in records]
    contexts = [r.retrieved_context for r in records]
    rouge = compute_rouge(list(predictions), references)
    token = compute_token_f1(list(predictions), references)
    exact = compute_exact_match(list(predictions), references)
    bert_f1 = 0.0
    if not args.no_bertscore:
        try:
            bert = compute_bertscore(list(predictions), references, device=embedding_device("auto"))
            bert_f1 = bert["bertscore_f1"]
        except Exception as exc:
            LOG.warning("BERTScore failed for %s: %s", condition, exc)
    faith = 0.0
    if not args.no_faithfulness:
        try:
            faith = compute_faithfulness(list(predictions), contexts, device=embedding_device("auto"))
        except Exception as exc:
            LOG.warning("Faithfulness failed for %s: %s", condition, exc)
    retr = retrieval_metrics(records)
    return CellResult(
        condition=condition,
        embedding_label=embedding_label,
        llm_label=llm_label,
        num_samples=len(records),
        recall_at_1=retr["recall_at_1"],
        recall_at_5=retr["recall_at_5"],
        recall_at_10=retr["recall_at_10"],
        mrr_at_10=retr["mrr_at_10"],
        ndcg_at_10=retr["ndcg_at_10"],
        rouge1=rouge.get("rouge1", 0.0),
        rouge2=rouge.get("rouge2", 0.0),
        rougeL=rouge.get("rougeL", 0.0),
        bertscore_f1=bert_f1,
        token_f1=token["token_f1"],
        exact_match=exact,
        faithfulness_score=faith,
        answer_quality_score=(bert_f1 + faith) / 2.0 if bert_f1 or faith else 0.0,
        avg_tokens_per_second=sum(tps) / max(len(tps), 1),
        wall_time_s=time.perf_counter() - started,
    )


def latex_table(results: Sequence[CellResult]) -> str:
    order = ["base_base", "base_ft", "ft_base", "ft_ft"]
    by_name = {r.condition: r for r in results}
    lines = []
    for name in order:
        if name not in by_name:
            continue
        r = by_name[name]
        emb = "Base" if r.embedding_label == "base" else "Fine-tuned"
        llm = "Base" if r.llm_label == "base" else "Fine-tuned"
        lines.append(
            f"{emb} & {llm} & {r.recall_at_5:.3f} & {r.bertscore_f1:.3f} & "
            f"{r.faithfulness_score:.3f} & {r.answer_quality_score:.3f} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    resolve_model_paths(args)
    if args.prompt_k > args.retrieval_k:
        raise ValueError("--prompt-k cannot exceed --retrieval-k")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = {c.strip() for c in args.conditions.split(",") if c.strip()}
    valid = {"base_base", "base_ft", "ft_base", "ft_ft"}
    unknown = conditions - valid
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    all_records = load_records(args)
    sample = sample_records(all_records, args)
    corpus = build_corpus(all_records)
    LOG.info("Loaded %d records; sampled %d queries; corpus has %d contexts", len(all_records), len(sample), len(corpus))

    embeddings_needed = set()
    if {"base_base", "base_ft"} & conditions:
        embeddings_needed.add("base")
    if {"ft_base", "ft_ft"} & conditions:
        embeddings_needed.add("ft")

    retrieved: Dict[str, List[RetrievedRecord]] = {}
    if "base" in embeddings_needed:
        retrieved["base"] = retrieve_contexts(
            "base",
            args.base_embedding,
            corpus,
            sample,
            args,
            out_dir / "retrieval_base.jsonl",
        )
    if "ft" in embeddings_needed:
        retrieved["ft"] = retrieve_contexts(
            "ft",
            args.ft_embedding,
            corpus,
            sample,
            args,
            out_dir / "retrieval_ft.jsonl",
        )

    results: List[CellResult] = []
    llm_jobs = [
        ("base", None, [c for c in ["base_base", "ft_base"] if c in conditions]),
        ("ft", args.ft_adapter, [c for c in ["base_ft", "ft_ft"] if c in conditions]),
    ]

    for llm_label, adapter, cell_names in llm_jobs:
        if not cell_names:
            continue
        local_model = tokenizer = None
        if args.llm_backend == "local":
            local_model, tokenizer = load_local_model(args.base_llm, adapter, args.load_4bit)
        served_model = args.served_base_llm if llm_label == "base" else args.served_ft_llm
        if args.llm_backend == "openai" and not served_model:
            raise ValueError(f"--served-{llm_label}-llm is required for --llm-backend openai")

        for condition in cell_names:
            embedding_label = "base" if condition.startswith("base_") else "ft"
            recs = retrieved[embedding_label]
            pred_path = out_dir / f"predictions_{condition}.jsonl"
            cached = load_cached_predictions(pred_path) if args.resume else None
            if cached:
                predictions, tps = cached
                LOG.info("Loaded cached predictions for %s: %s", condition, pred_path)
            elif args.llm_backend == "local":
                predictions, tps = generate_local(local_model, tokenizer, recs, args)
                write_predictions(pred_path, recs, predictions, tps)
            else:
                predictions, tps = generate_openai(str(served_model), recs, args)
                write_predictions(pred_path, recs, predictions, tps)

            result = score_condition(condition, embedding_label, llm_label, recs, predictions, tps, args)
            results.append(result)
            with (out_dir / f"metrics_{condition}.json").open("w", encoding="utf-8") as f:
                json.dump(asdict(result), f, indent=2)
            LOG.info("%s: R@5=%.3f BERTScore=%.3f Faith=%.3f Score=%.3f",
                     condition, result.recall_at_5, result.bertscore_f1,
                     result.faithfulness_score, result.answer_quality_score)

        del local_model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    merged_results = {r.condition: r for r in results}
    for name in valid:
        metrics_path = out_dir / f"metrics_{name}.json"
        if name not in merged_results and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                merged_results[name] = CellResult(**json.load(f))
    results = list(merged_results.values())

    payload = {
        "args": vars(args),
        "results": [asdict(r) for r in sorted(results, key=lambda r: r.condition)],
        "latex_rows": latex_table(results),
    }
    with (out_dir / "rag_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with (out_dir / "rag_ablation_table_rows.tex").open("w", encoding="utf-8") as f:
        f.write(payload["latex_rows"] + "\n")
    LOG.info("Wrote summary: %s", out_dir / "rag_ablation_summary.json")
    LOG.info("LaTeX rows:\n%s", payload["latex_rows"])


if __name__ == "__main__":
    main()
