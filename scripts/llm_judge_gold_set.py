#!/usr/bin/env python3
"""LLM-judge pass for the MSQA-Bench gold-set audit.

Calls an OpenAI structured-output model (GPT-4o by default) to label each
record on the same four rubric dimensions used by the human annotators:
answer_correct, evidence_support, evidence_quality, question_clarity.

The output schema matches the human annotation CSV so it can be plugged
directly into ``scripts/compute_iaa.py`` as a third annotator.

Resumable: if the output CSV already exists, completed annotation_ids are
skipped on a re-run.

Usage:
    python scripts/llm_judge_gold_set.py \
        --input  paper_results/annotation/gold_set_for_llm_judge.csv \
        --output paper_results/annotation/gold_set_llm_judge.csv \
        --model  gpt-4o-2024-08-06

Requires: OPENAI_API_KEY in the environment (or in ./.env).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


RUBRIC_PROMPT = """\
You are an expert annotator for a question-answering benchmark in mass
spectrometry research. Score each record on four independent rubric
dimensions. Return only the structured fields requested.

Score each dimension INDEPENDENTLY. A failure on one dimension must not
contaminate the others.

answer_correct ∈ {Yes, Partial, No}
  Yes      — the answer correctly resolves the question given the context.
  Partial  — on-topic and partially correct, but omits a material qualifier,
             includes one minor error, hedges an extractable fact, or is
             truncated mid-sentence with the relevant information present.
  No       — factually wrong, unrelated, or contradicted by the context.

evidence_support ∈ {Yes, Partial, No}
  Yes      — the cited passage directly supports the answer; no extra
             inference required.
  Partial  — the passage is topically relevant and makes the answer
             plausible, but a reader must combine it with outside knowledge
             to fully justify the answer.
  No       — the passage does not support the answer, or is unrelated.

question_clarity ∈ {Good, Ambiguous, Bad}
  Good      — unambiguous and well-formed in isolation.
  Ambiguous — well-formed but admits multiple reasonable interpretations
              without context (e.g. "the experiment" with no referent).
  Bad       — malformed, ungrammatical, or refers to figures / tables /
              sections that do not exist for the reader.

evidence_quality ∈ {Good, Weak, Missing}
  Good    — on-topic, coherent prose, free of extraction artefacts.
  Weak    — on-topic but shows extraction artefacts (broken hyphenation,
            table fragments, references-section intrusion, header/footer
            noise, mixed languages).
  Missing — empty, unrelated, or unrecoverable.

Decision rules:
- "The text does not provide a reason" answers to causal questions are
  valid abstentions: score answer_correct=Yes if the context truly does
  not contain a reason.
- Truncated but informative answers: Partial, not No.
- Hedge words near numeric answers ("approximately", "around") are not a
  penalty unless the question demands precision.
- Reference-section intrusion or DOIs in the context make evidence_quality
  Weak, not Missing, as long as on-topic prose remains.
"""


FEW_SHOT = """\
Example 1.
Question: Which soybean variety has the highest soymetide content according to Table 1?
Answer: Tokachi with 597 ± 59 µg/g
Context: Table 1. Determination of Soymetide in Different Soybean Varieties... Tokachi Japan 597 ± 59 ...
Question type: factual
Verdicts: answer_correct=Yes, evidence_support=Yes, question_clarity=Good, evidence_quality=Good

Example 2.
Question: How does the text describe the process of handling data?
Answer: Encoding, decoding, and managing data structures
Context: [methods paragraph about chromatography software]
Question type: method
Verdicts: answer_correct=No (non-specific generic answer), evidence_support=No, question_clarity=Ambiguous (no referent), evidence_quality=Good

Example 3.
Question: Why is a logarithmic scale used in Figure 1?
Answer: The text does not provide a reason for the use of a logarithmic scale.
Context: Figure 1 caption with no rationale given.
Question type: causal
Verdicts: answer_correct=Yes (valid abstention), evidence_support=Yes, question_clarity=Good, evidence_quality=Good
"""


class Verdict(BaseModel):
    answer_correct: Literal["Yes", "Partial", "No"]
    answer_correct_reason: str = Field(
        description="One sentence justifying the answer_correct label."
    )
    evidence_support: Literal["Yes", "Partial", "No"]
    evidence_support_reason: str = Field(
        description="One sentence justifying the evidence_support label."
    )
    question_clarity: Literal["Good", "Ambiguous", "Bad"]
    question_clarity_reason: str = Field(
        description="One sentence justifying the question_clarity label."
    )
    evidence_quality: Literal["Good", "Weak", "Missing"]
    evidence_quality_reason: str = Field(
        description="One sentence justifying the evidence_quality label."
    )


OUTPUT_FIELDS = [
    "annotation_id",
    "question_type",
    "answer_correct",
    "evidence_support",
    "evidence_quality",
    "question_clarity",
    "annotator_id",
    "rationale_answer",
    "rationale_evidence_support",
    "rationale_question_clarity",
    "rationale_evidence_quality",
    "notes",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(newline="", encoding="utf-8") as f:
        return {r["annotation_id"] for r in csv.DictReader(f) if r.get("annotation_id")}


def judge_one(client: OpenAI, model: str, row: dict) -> Verdict:
    user_msg = (
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}\n"
        f"Context: {row['context']}\n"
        f"Question type (machine label, do not let it bias you): {row.get('question_type','unknown')}\n\n"
        "Score this record on the four rubric dimensions."
    )
    resp = client.beta.chat.completions.parse(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": RUBRIC_PROMPT + "\n\n" + FEW_SHOT},
            {"role": "user", "content": user_msg},
        ],
        response_format=Verdict,
    )
    return resp.choices[0].message.parsed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("paper_results/annotation/gold_set_for_llm_judge.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("paper_results/annotation/gold_set_llm_judge.csv"))
    p.add_argument("--model", default="gpt-4o-2024-08-06")
    p.add_argument("--limit", type=int, default=None,
                   help="Max records to process this run (debug).")
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args(argv)

    load_env_file(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    with args.input.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = already_done(args.output)
    todo = [r for r in rows if r["annotation_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Total: {len(rows)} | already done: {len(done)} | this run: {len(todo)}",
          file=sys.stderr)
    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    client = OpenAI()

    t0 = time.time()
    n_ok = n_err = 0
    with args.output.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
        if write_header:
            w.writeheader()
        for i, row in enumerate(todo, 1):
            for attempt in range(1, args.max_retries + 1):
                try:
                    v = judge_one(client, args.model, row)
                    w.writerow({
                        "annotation_id": row["annotation_id"],
                        "question_type": row.get("question_type", ""),
                        "answer_correct": v.answer_correct,
                        "evidence_support": v.evidence_support,
                        "evidence_quality": v.evidence_quality,
                        "question_clarity": v.question_clarity,
                        "annotator_id": args.model,
                        "rationale_answer": v.answer_correct_reason,
                        "rationale_evidence_support": v.evidence_support_reason,
                        "rationale_question_clarity": v.question_clarity_reason,
                        "rationale_evidence_quality": v.evidence_quality_reason,
                        "notes": "",
                    })
                    f.flush()
                    n_ok += 1
                    print(
                        f"[{i:3d}/{len(todo)}] {row['annotation_id']}: "
                        f"correct={v.answer_correct} support={v.evidence_support} "
                        f"clarity={v.question_clarity} quality={v.evidence_quality}",
                        file=sys.stderr,
                    )
                    break
                except Exception as e:
                    if attempt == args.max_retries:
                        n_err += 1
                        print(f"[{i:3d}/{len(todo)}] {row['annotation_id']}: ERROR {e}",
                              file=sys.stderr)
                    else:
                        time.sleep(2 ** attempt)

    dt = time.time() - t0
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={dt:.1f}s "
          f"({dt/max(n_ok,1):.2f}s/record)",
          file=sys.stderr)
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
