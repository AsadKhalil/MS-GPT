#!/usr/bin/env python3
"""Compute lightweight submission diagnostics for MSQA-Bench.

The script intentionally uses only the Python standard library so it can run
on a review machine without the training stack. It produces:

* split-leakage checks and train/test answer-overlap diagnostics
* audit calibration by rule-based quality-score bucket
* release license/status counts
* available zero-shot closed-model generation results
* a CSV template for manual question-type validation, if no gold labels exist
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "paper_results/dataset/splits/train.jsonl"
VAL_PATH = ROOT / "paper_results/dataset/splits/val.jsonl"
TEST_PATH = ROOT / "paper_results/dataset/splits/test.jsonl"
GOLD_SAMPLE = ROOT / "paper_results/annotation/gold_set_sample.csv"
GOLD_ADJUDICATED = ROOT / "paper_results/annotation/gold_set_adjudicated.csv"
LICENSE_AUDIT = ROOT / "paper_results/neurips_release/license_audit.csv"
OUT_DIR = ROOT / "paper_results/diagnostics"
OUT_JSON = OUT_DIR / "submission_diagnostics.json"
QUESTION_TYPE_TEMPLATE = ROOT / "paper_results/annotation/question_type_validation_template.csv"


TOKEN_RE = re.compile(r"[a-z0-9]+")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def ngram_hashes(tokens: list[str], n: int = 5) -> set[int]:
    if len(tokens) < n:
        return set()
    return {
        zlib.crc32(" ".join(tokens[i : i + n]).encode("utf-8"))
        for i in range(len(tokens) - n + 1)
    }


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))


def compute_leakage() -> dict[str, Any]:
    split_paths = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    docs: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    train_answers: set[str] = set()
    train_questions: set[str] = set()
    train_answer_5grams: set[int] = set()

    for split, path in split_paths.items():
        doc_ids: set[str] = set()
        count = 0
        for row in iter_jsonl(path):
            count += 1
            doc_id = row.get("doc_id") or row.get("document_hash") or row.get("file_name")
            if doc_id:
                doc_ids.add(str(doc_id))
            if split == "train":
                answer_norm = normalize(row.get("answer", ""))
                question_norm = normalize(row.get("question", ""))
                if answer_norm:
                    train_answers.add(answer_norm)
                    train_answer_5grams.update(ngram_hashes(answer_norm.split(), n=5))
                if question_norm:
                    train_questions.add(question_norm)
        docs[split] = doc_ids
        split_counts[split] = count

    test_answer_total = 0
    test_answer_len5_total = 0
    exact_answer_overlap = 0
    exact_answer_overlap_len5 = 0
    exact_question_overlap = 0
    answer_5gram_fractions: list[float] = []

    for row in iter_jsonl(TEST_PATH):
        answer_norm = normalize(row.get("answer", ""))
        question_norm = normalize(row.get("question", ""))
        answer_tokens = answer_norm.split()
        if answer_norm:
            test_answer_total += 1
            if answer_norm in train_answers:
                exact_answer_overlap += 1
            if len(answer_tokens) >= 5:
                test_answer_len5_total += 1
                if answer_norm in train_answers:
                    exact_answer_overlap_len5 += 1
                grams = ngram_hashes(answer_tokens, n=5)
                if grams:
                    answer_5gram_fractions.append(
                        sum(1 for gram in grams if gram in train_answer_5grams) / len(grams)
                    )
        if question_norm and question_norm in train_questions:
            exact_question_overlap += 1

    doc_overlap = {
        "train_val": len(docs["train"] & docs["val"]),
        "train_test": len(docs["train"] & docs["test"]),
        "val_test": len(docs["val"] & docs["test"]),
    }

    return {
        "split_counts": split_counts,
        "unique_docs": {split: len(ids) for split, ids in docs.items()},
        "doc_overlap": doc_overlap,
        "exact_test_answer_in_train_pct": pct(exact_answer_overlap, test_answer_total),
        "exact_test_answer_in_train": exact_answer_overlap,
        "test_answer_total": test_answer_total,
        "exact_test_answer_len5_in_train_pct": pct(
            exact_answer_overlap_len5, test_answer_len5_total
        ),
        "exact_test_answer_len5_in_train": exact_answer_overlap_len5,
        "test_answer_len5_total": test_answer_len5_total,
        "exact_test_question_in_train_pct": pct(exact_question_overlap, split_counts["test"]),
        "exact_test_question_in_train": exact_question_overlap,
        "answer_5gram_overlap": {
            "n": len(answer_5gram_fractions),
            "mean": statistics.mean(answer_5gram_fractions) if answer_5gram_fractions else 0.0,
            "median": statistics.median(answer_5gram_fractions) if answer_5gram_fractions else 0.0,
            "p90": percentile(answer_5gram_fractions, 0.90),
            "p95": percentile(answer_5gram_fractions, 0.95),
            "share_ge_0_8_pct": pct(
                sum(1 for value in answer_5gram_fractions if value >= 0.8),
                len(answer_5gram_fractions),
            ),
            "share_eq_1_0_pct": pct(
                sum(1 for value in answer_5gram_fractions if value == 1.0),
                len(answer_5gram_fractions),
            ),
        },
    }


def load_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["annotation_id"]: row for row in reader}


def quality_bucket(score: float) -> str:
    if score < 0.60:
        return "<0.60"
    if score < 0.70:
        return "0.60-0.69"
    if score < 0.80:
        return "0.70-0.79"
    return ">=0.80"


def compute_quality_calibration() -> dict[str, Any]:
    sample = load_csv_by_id(GOLD_SAMPLE)
    adjudicated = load_csv_by_id(GOLD_ADJUDICATED)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores: list[float] = []
    full_correct: list[float] = []
    usable_correct: list[float] = []
    usable_evidence: list[float] = []

    for ann_id, row in adjudicated.items():
        if ann_id not in sample:
            continue
        try:
            score = float(sample[ann_id]["quality_score"])
        except (KeyError, TypeError, ValueError):
            continue
        answer = row["answer_correct__final"]
        evidence = row["evidence_support__final"]
        item = {
            "score": score,
            "answer_full": answer == "Yes",
            "answer_usable": answer in {"Yes", "Partial"},
            "evidence_usable": evidence in {"Yes", "Partial"},
        }
        buckets[quality_bucket(score)].append(item)
        scores.append(score)
        full_correct.append(1.0 if item["answer_full"] else 0.0)
        usable_correct.append(1.0 if item["answer_usable"] else 0.0)
        usable_evidence.append(1.0 if item["evidence_usable"] else 0.0)

    bucket_rows = []
    for label in ["<0.60", "0.60-0.69", "0.70-0.79", ">=0.80"]:
        rows = buckets.get(label, [])
        if not rows:
            continue
        bucket_rows.append(
            {
                "bucket": label,
                "n": len(rows),
                "mean_quality_score": statistics.mean(row["score"] for row in rows),
                "answer_full_pct": pct(sum(row["answer_full"] for row in rows), len(rows)),
                "answer_yes_or_partial_pct": pct(
                    sum(row["answer_usable"] for row in rows), len(rows)
                ),
                "evidence_yes_or_partial_pct": pct(
                    sum(row["evidence_usable"] for row in rows), len(rows)
                ),
            }
        )

    return {
        "n": len(scores),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "spearman": {
            "answer_full": spearman(scores, full_correct),
            "answer_yes_or_partial": spearman(scores, usable_correct),
            "evidence_yes_or_partial": spearman(scores, usable_evidence),
        },
        "buckets": bucket_rows,
    }


def compute_license_summary() -> dict[str, Any]:
    record_counts: Counter[tuple[str, str]] = Counter()
    doc_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_counts: Counter[tuple[str, str, str]] = Counter()
    with LICENSE_AUDIT.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            license_label = row.get("license") or "missing"
            status = row.get("redistribution_status") or "missing"
            key = (license_label, status)
            record_counts[key] += 1
            doc_id = row.get("document_hash")
            if doc_id:
                doc_sets[key].add(doc_id)
            split_counts[(row.get("split") or "missing", license_label, status)] += 1

    rows = []
    for (license_label, status), count in sorted(record_counts.items()):
        rows.append(
            {
                "license": license_label,
                "redistribution_status": status,
                "records": count,
                "documents": len(doc_sets[(license_label, status)]),
            }
        )
    return {
        "records_by_license_and_status": rows,
        "split_counts": {
            f"{split}|{license_label}|{status}": count
            for (split, license_label, status), count in sorted(split_counts.items())
        },
    }


def load_zero_shot_results() -> list[dict[str, Any]]:
    paths = [
        ROOT / "paper_results/model_results/llms/gpt4o_zeroshot/eval_results_test_n1000.json",
        ROOT
        / "paper_results/model_results/llms/gpt4o_mini_zeroshot/eval_results_test_n1000.json",
    ]
    rows = []
    for path in paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            rows.append(
                {
                    "model_name": data.get("model_name"),
                    "num_samples": data.get("num_samples"),
                    "rouge1": data.get("rouge1"),
                    "rougeL": data.get("rougeL"),
                    "token_f1": data.get("token_f1"),
                    "exact_match": data.get("exact_match"),
                    "avg_pred_length": data.get("avg_pred_length"),
                    "note": data.get("note"),
                }
            )
    return rows


def write_question_type_template() -> dict[str, Any]:
    if not GOLD_SAMPLE.exists():
        return {"written": False, "reason": "gold sample missing"}
    if QUESTION_TYPE_TEMPLATE.exists():
        return {"written": False, "path": str(QUESTION_TYPE_TEMPLATE), "reason": "already exists"}

    with GOLD_SAMPLE.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)

    fieldnames = [
        "annotation_id",
        "question",
        "answer",
        "context",
        "predicted_question_type",
        "gold_question_type",
        "notes",
    ]
    with QUESTION_TYPE_TEMPLATE.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "annotation_id": row["annotation_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "context": row["context"],
                    "predicted_question_type": row["question_type"],
                    "gold_question_type": "",
                    "notes": "",
                }
            )

    return {"written": True, "path": str(QUESTION_TYPE_TEMPLATE), "rows": len(rows)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "leakage": compute_leakage(),
        "quality_calibration": compute_quality_calibration(),
        "license_summary": compute_license_summary(),
        "zero_shot_generation": load_zero_shot_results(),
        "question_type_validation_template": write_question_type_template(),
    }
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
