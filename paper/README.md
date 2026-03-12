# Paper — MSQA-Bench

This package implements **MSQA-Bench**, a benchmark for evaluating AI systems on mass spectrometry question-answering tasks. It covers the full lifecycle: dataset curation, human annotation, retrieval/RAG evaluation, and results reporting.

## Directory Structure

```
paper/
├── annotation/          # Human evaluation tools
│   └── gold_set_sampler.py
├── dataset/             # Dataset engineering pipeline
│   ├── metadata_extractor.py
│   ├── quality_filters.py
│   ├── question_classifier.py
│   ├── schema_enricher.py
│   └── split_generator.py
├── evaluation/          # Benchmark evaluation framework
│   ├── rag_pipeline.py
│   ├── retrieval_baselines.py
│   └── faithfulness_metrics.py
└── figures/             # Paper tables and visualizations
    └── generate_tables.py
```

## Modules

### 1. Dataset (`dataset/`)

Transforms raw QA pairs into benchmark-quality records through a multi-stage pipeline:

- **`metadata_extractor.py`** — Extracts document metadata (DOI, PMID, arXiv ID, title, year, venue, license) from source text using regex patterns and optional CrossRef API enrichment.
- **`quality_filters.py`** — Filters and deduplicates QA pairs. Computes quality metrics including answer-context overlap, question specificity, answer completeness, and MinHash-based deduplication (LSH).
- **`question_classifier.py`** — Rule-based classification of questions into 7 types: factual, method, definition, comparison, numeric, causal, and unknown. Includes MS-domain-specific indicators.
- **`schema_enricher.py`** — Enriches each QA record with metadata, evidence spans, question type, answer style (extractive/abstractive), quality score, and split assignment. Produces the final `EnrichedQARecord` schema. Includes a `to_public_dict()` method that omits copyrighted text for safe release.
- **`split_generator.py`** — Deterministic document-level train/val/test splitting (85/10/5) using MD5 hashing. Ensures all QA pairs from the same paper stay in the same split to prevent data leakage.

### 2. Annotation (`annotation/`)

- **`gold_set_sampler.py`** — Stratified sampling of QA pairs for human evaluation. Exports annotation-ready CSV/JSONL with fields for answer correctness, evidence support, evidence quality, and question clarity. Computes inter-annotator agreement (Cohen's Kappa).

### 3. Evaluation (`evaluation/`)

- **`retrieval_baselines.py`** — Implements BM25 (sparse) and embedding-based (dense) retrieval baselines. Evaluates with standard IR metrics: Recall@k, MRR, NDCG, and MAP. Supports both base models (e.g., all-MiniLM-L6-v2) and fine-tuned models.
- **`rag_pipeline.py`** — Full RAG pipeline with three modes:
  - **standard** — retrieve then generate
  - **cite** — forces inline citations in answers
  - **abstain** — allows the model to decline answering when unsure

  Uses a vLLM-served model (default: Qwen2.5-14B-Instruct-AWQ) via OpenAI-compatible API.
- **`faithfulness_metrics.py`** — Detects hallucinations and measures citation quality. Computes unsupported claim rate, citation precision/recall, and an overall faithfulness score. Supports optional NLI-based verification (roberta-large-mnli).

### 4. Figures (`figures/`)

- **`generate_tables.py`** — Generates LaTeX tables for the paper: dataset statistics, question type distribution, retrieval results comparison, and faithfulness metrics.

## Pipeline Flow

```
Consolidated QA (data/qa_outputs/)
  │
  ├─► metadata_extractor    → DOI, PMID, venue, license
  ├─► quality_filters        → filter low-quality, deduplicate
  ├─► question_classifier    → factual / method / definition / ...
  ├─► split_generator        → train / val / test (document-level)
  │
  └─► schema_enricher        → combines all above into EnrichedQARecord
        │
        ├─► gold_set_sampler  → sample for human annotation
        ├─► retrieval_baselines + rag_pipeline → benchmark evaluation
        ├─► faithfulness_metrics → hallucination analysis
        └─► generate_tables   → LaTeX tables for paper
```

## Key Design Decisions

- **Document-level splits** prevent data leakage — all QA pairs from the same paper stay together.
- **Deterministic MD5 hashing** ensures reproducible splits regardless of processing order.
- **Quality scoring** weights: overlap (0.3), specificity (0.25), completeness (0.2), length (0.2).
- **Citation tracking** in RAG maps generated claims to retrieved passages for faithfulness analysis.
- **Public-safe export** via `to_public_dict()` omits copyrighted context, retaining only citation offsets.
