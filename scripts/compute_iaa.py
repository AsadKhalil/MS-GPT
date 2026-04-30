#!/usr/bin/env python3
"""Inter-annotator agreement for the MSQA-Bench gold-set audit.

Supports 2+ annotators. For each rubric dimension reports:
  - pairwise Cohen's kappa (with 95% bootstrap CI)
  - Fleiss's kappa over all annotators (≥3 annotators only)
  - raw % agreement (all annotators identical)
  - per-annotator label distribution
  - confusion matrices (pairwise)

Also writes an adjudication CSV that flags every row with disagreement,
so a follow-up adjudication round can resolve them.

Usage:
    python scripts/compute_iaa.py \\
        --annotator paper_results/annotation/gold_set_annotated.csv:asad \\
        --annotator paper_results/annotation/gold_set_attila.csv:attila \\
        --annotator paper_results/annotation/gold_set_llm_judge.csv:gpt-4o \\
        --output paper_results/annotation/iaa_report.json \\
        --adjudication-csv paper_results/annotation/gold_adjudication.csv \\
        --markdown paper_results/annotation/iaa_table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RUBRIC_FIELDS: Tuple[str, ...] = (
    "answer_correct",
    "evidence_support",
    "evidence_quality",
    "question_clarity",
)


def load_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["annotation_id"]: r for r in rows if r.get("annotation_id", "").strip()}


def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    n = len(pairs)
    labels = sorted({l for pair in pairs for l in pair})
    obs = sum(1 for a, b in pairs if a == b) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    expected = sum((counts_a[l] / n) * (counts_b[l] / n) for l in labels)
    if expected >= 1.0:
        return 1.0
    return (obs - expected) / (1 - expected)


def fleiss_kappa(matrix: List[List[int]]) -> float:
    """matrix[i][k] = number of annotators who assigned subject i to category k.
    Assumes the same number of annotators per subject."""
    if not matrix:
        return float("nan")
    N = len(matrix)
    K = len(matrix[0])
    n = sum(matrix[0])
    if n < 2 or N == 0:
        return float("nan")
    # Per-category proportion across all subjects
    p = [sum(matrix[i][k] for i in range(N)) / (N * n) for k in range(K)]
    P_e = sum(pk * pk for pk in p)
    # Per-subject agreement
    P_i = [
        (sum(matrix[i][k] ** 2 for k in range(K)) - n) / (n * (n - 1))
        for i in range(N)
    ]
    P_bar = sum(P_i) / N
    if 1 - P_e == 0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def bootstrap_ci(
    fn,
    items: Sequence,
    n_iter: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on a statistic over a sequence of items."""
    if len(items) < 2:
        return (float("nan"), float("nan"))
    import random
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_iter):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        try:
            v = fn(sample)
            if not math.isnan(v):
                samples.append(v)
        except Exception:
            continue
    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return (lo, hi)


def confusion_matrix(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        cm[a][b] += 1
    return {k: dict(v) for k, v in cm.items()}


def write_adjudication(
    annotators: Dict[str, Dict[str, Dict[str, str]]],
    common_ids: Sequence[str],
    output: Path,
) -> int:
    names = list(annotators.keys())
    base = next(iter(annotators.values()))
    fieldnames = ["annotation_id", "question_type", "question", "answer"]
    for f in RUBRIC_FIELDS:
        for n in names:
            fieldnames.append(f"{f}__{n}")
        fieldnames.append(f"{f}__final")
    n_disagreements = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for aid in common_ids:
            row = base.get(aid, {})
            out = {
                "annotation_id": aid,
                "question_type": row.get("question_type", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            }
            disagreed = False
            for field in RUBRIC_FIELDS:
                labels = []
                for n in names:
                    val = (annotators[n].get(aid, {}).get(field) or "").strip()
                    out[f"{field}__{n}"] = val
                    if val:
                        labels.append(val)
                unique = set(labels)
                if len(unique) == 1 and len(labels) == len(names):
                    out[f"{field}__final"] = labels[0]
                else:
                    out[f"{field}__final"] = ""
                    if len(unique) > 1:
                        disagreed = True
            if disagreed:
                n_disagreements += 1
            w.writerow(out)
    return n_disagreements


def render_markdown(report: dict) -> str:
    lines: List[str] = []
    pairs = report["pair_order"]
    names = report["annotators"]
    lines.append("## Inter-annotator agreement\n")
    lines.append(f"_Annotators: {', '.join(names)} | n_common = {report['n_common']}_\n")
    header = ["Dimension"] + [f"κ ({a}↔{b})" for a, b in pairs]
    if len(names) >= 3:
        header.append("Fleiss κ")
    header.append("Raw %")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for dim in RUBRIC_FIELDS:
        row = [dim]
        for a, b in pairs:
            d = report["pairwise_cohen_kappa"][dim][f"{a}__{b}"]
            row.append(f"{d['kappa']:.2f} [{d['ci95_lo']:.2f},{d['ci95_hi']:.2f}] (n={d['n']})")
        if len(names) >= 3:
            d = report["fleiss_kappa"][dim]
            row.append(f"{d['kappa']:.2f} [{d['ci95_lo']:.2f},{d['ci95_hi']:.2f}]")
        row.append(f"{report['raw_agreement_pct'][dim]['pct']:.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    if len(names) >= 3:
        lines.append("\n_Mean Fleiss κ across dimensions:_ "
                     f"**{report['summary']['mean_fleiss_kappa']:.3f}**")
    lines.append(f"\n_Mean pairwise Cohen κ:_ **{report['summary']['mean_pairwise_cohen']:.3f}**")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator", action="append", required=True,
                   help="path:label, e.g. asad.csv:asad. Repeat for ≥2.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adjudication-csv", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=500)
    args = p.parse_args(argv)

    if len(args.annotator) < 2:
        p.error("Need ≥2 --annotator entries.")

    annotators: Dict[str, Dict[str, Dict[str, str]]] = {}
    for spec in args.annotator:
        path_str, _, label = spec.rpartition(":")
        if not path_str or not label:
            p.error(f"Bad --annotator spec: {spec!r} (expected path:label)")
        annotators[label] = load_csv(Path(path_str))

    names = list(annotators.keys())
    common_ids = sorted(set.intersection(*[set(a.keys()) for a in annotators.values()]))
    print(f"Annotators: {names}", file=sys.stderr)
    print(f"Common annotation_ids across all annotators: {len(common_ids)}", file=sys.stderr)

    pair_order = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]

    report: Dict = {
        "annotators": names,
        "n_common": len(common_ids),
        "pair_order": pair_order,
        "pairwise_cohen_kappa": {},
        "fleiss_kappa": {},
        "raw_agreement_pct": {},
        "label_distributions": {},
        "confusion_matrices": {},
        "summary": {},
    }

    pairwise_means: List[float] = []
    fleiss_means: List[float] = []
    for dim in RUBRIC_FIELDS:
        valid_ids = [
            aid for aid in common_ids
            if all((annotators[n].get(aid, {}).get(dim) or "").strip() for n in names)
        ]
        report["pairwise_cohen_kappa"][dim] = {}
        report["confusion_matrices"][dim] = {}
        for a, b in pair_order:
            pairs = [
                ((annotators[a][aid][dim] or "").strip(),
                 (annotators[b][aid][dim] or "").strip())
                for aid in valid_ids
            ]
            score = cohen_kappa(pairs)
            ci_lo, ci_hi = bootstrap_ci(cohen_kappa, pairs, n_iter=args.bootstrap)
            report["pairwise_cohen_kappa"][dim][f"{a}__{b}"] = {
                "kappa": float(score),
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
                "n": len(pairs),
            }
            report["confusion_matrices"][dim][f"{a}__{b}"] = confusion_matrix(pairs)
            if not math.isnan(score):
                pairwise_means.append(score)

        if len(names) >= 3:
            categories = sorted({
                (annotators[n][aid][dim] or "").strip()
                for n in names for aid in valid_ids
            })
            cat_idx = {c: i for i, c in enumerate(categories)}
            matrix = []
            for aid in valid_ids:
                counts = [0] * len(categories)
                for n in names:
                    counts[cat_idx[(annotators[n][aid][dim] or "").strip()]] += 1
                matrix.append(counts)
            score = fleiss_kappa(matrix)
            def _fk(m):
                return fleiss_kappa(list(m))
            ci_lo, ci_hi = bootstrap_ci(_fk, matrix, n_iter=args.bootstrap)
            report["fleiss_kappa"][dim] = {
                "kappa": float(score),
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
                "n": len(valid_ids),
            }
            if not math.isnan(score):
                fleiss_means.append(score)

        n_agree = sum(
            1 for aid in valid_ids
            if len({(annotators[n][aid][dim] or "").strip() for n in names}) == 1
        )
        report["raw_agreement_pct"][dim] = {
            "n_agree": n_agree,
            "n_total": len(valid_ids),
            "pct": (100.0 * n_agree / len(valid_ids)) if valid_ids else 0.0,
        }
        report["label_distributions"][dim] = {
            n: dict(Counter(
                (annotators[n][aid][dim] or "").strip() for aid in valid_ids
            ))
            for n in names
        }

    report["summary"] = {
        "mean_pairwise_cohen": (sum(pairwise_means) / len(pairwise_means)) if pairwise_means else float("nan"),
    }
    if fleiss_means:
        report["summary"]["mean_fleiss_kappa"] = sum(fleiss_means) / len(fleiss_means)

    if args.adjudication_csv is not None:
        report["summary"]["disagreements_written"] = write_adjudication(
            annotators, common_ids, args.adjudication_csv,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}", file=sys.stderr)

    md = render_markdown(report)
    print("\n" + md)
    if args.markdown is not None:
        args.markdown.write_text(md, encoding="utf-8")
        print(f"Wrote {args.markdown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
