# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MS-GPT is a pipeline for processing mass spectrometry research PDFs into fine-tuned AI models. The workflow: PDF ingestion → text extraction → Q&A generation → model fine-tuning (embeddings + LLMs).

## Environment Setup

```bash
source .venv/bin/activate
pip install -r config/requirements.txt
```

Python 3.10, virtual environment at `.venv/`. GPU workloads use CUDA (typically `CUDA_VISIBLE_DEVICES=2`).

## Key Commands

### PDF Text Extraction
```bash
# Fast batch extraction (recommended for large-scale)
python src/vision_extractors/fast_pdf_extractor.py /path/to/pdfs/ --workers 8 --output results/

# Vision-based extraction (requires Ollama running)
python src/vision_extractors/vision_extractor.py

# PyMuPDF basic extraction
python src/pdf_processors/pymupdf_processor.py
```

### Q&A Generation
```bash
# Requires a vLLM server running (Qwen2.5-14B-AWQ on localhost:8000)
python src/qa_generators/qa_generator.py --config config/qa_generator.json

# Consolidate Q&A outputs
python scripts/consolidate_qa.py
```

### Embedding Fine-Tuning
```bash
# Train all configured embedding models
python scripts/train_all_models.py --config config/embedding_finetuner.json -y

# Validate setup first
python scripts/validate_embedding_setup.py
```

### LLM Fine-Tuning (QLoRA)
```bash
# Train all configured LLMs
python scripts/train_all_llms.py --config config/llm_finetuner.json -y

# Train a single model
python scripts/train_all_llms.py --model qwen2.5_3b -y

# Smoke test with small data subset
python scripts/train_all_llms.py --subset 50 -y

# Evaluate only (no training)
python scripts/train_all_llms.py --eval-only

# Direct single-model training
python -m src.llm_trainers.llm_finetuner --config config/llm_finetuner.json
```

### GROBID (academic paper parsing)
```bash
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1
```

### vLLM Server
```bash
# Monitor: tail -f logs/vllm_gpu2.log
# Stop: kill $(cat logs/vllm_gpu2.pid)
# Start: ./scripts/start_vllm_background.sh 2 14b 8000 --force
```

## Architecture

### Pipeline Flow
```
PDFs (data/input/) → Text Extraction → data/extracted_text/
    → Q&A Generation → data/qa_outputs/ → scripts/consolidate_qa.py → consolidated_qa.jsonl
    → Embedding Fine-Tuning (sentence-transformers) → models/fine_tuned_embeddings/
    → LLM Fine-Tuning (QLoRA via trl/peft) → models/fine_tuned_llms/
```

### Source Modules (`src/`)

- **`pdf_processors/`** — Multiple PDF-to-text strategies: PyMuPDF direct extraction, LLM-assisted cleaning (single/batch/page-by-page), GROBID for structured academic parsing, multi-format conversion
- **`vision_extractors/`** — `fast_pdf_extractor.py` (recommended, PyMuPDF-based with OCR fallback) and `agentic_vision_extractor.py` (legacy, LangGraph + Ollama vision, slow)
- **`qa_generators/`** — Generates Q&A pairs from extracted text using vLLM-served models (OpenAI-compatible API at localhost:8000)
- **`embedding_trainers/`** — Streaming fine-tuning of sentence-transformer models (e5, bge, nomic-embed). Uses `MultipleNegativesRankingLoss`. Evaluates with Recall@k, MRR, NDCG
- **`llm_trainers/`** — QLoRA fine-tuning using `transformers`, `peft`, `trl` (SFTTrainer) with BitsAndBytes 4-bit quantization. Evaluates with ROUGE, BERTScore, F1, faithfulness, perplexity

### Configuration (`config/`)

All configs are JSON. Key files:
- `config.json` — Ollama/vision extraction settings
- `qa_generator.json` — Q&A generation (vLLM endpoint, parallelism, circuit breaker)
- `embedding_finetuner.json` — Multi-model embedding training config with `models_to_train` list
- `llm_finetuner.json` — Multi-model QLoRA config with `models_to_train` list

### Data Splits
Both embedding and LLM trainers use deterministic hash-based splits (85/10/5 train/val/test) that are compatible with each other.

### Data Directories
- `data/input/` — Source PDFs (named by content hash)
- `data/extracted_text/` — Extracted text files
- `data/qa_outputs/` — Generated Q&A JSONL files
- `data/processed_pdfs/` — Processed PDF outputs

### Model Outputs
- `models/fine_tuned_embeddings/<model_name>/` — Trained embedding models + eval results
- `models/fine_tuned_llms/<model_name>/final_adapter/` — LoRA adapter weights
- `models/fine_tuned_llms/<model_name>/eval_results_test.json` — Per-model metrics
- `paper_results/evaluation/llms_eval_1k/<model_name>/` — Per-model evaluation outputs on the 1K-query subset (deepseek_r1_distill_7b, mistral_7b_v0.3, phi3.5_mini, qwen2.5_7b); `llm_comparison.json` and `summary_n1000.json` aggregate across models

### Paper, Thesis, Proposals, and Release Artifacts (in the parent workspace)

These live one level above this repo, at `/Users/asad/Projects/Thesis/`:

- `archive/paper-v1-original/` — Oldest NeurIPS draft sources (was `paper/` before 2026-05-10).
- `archive/paper-v2-prep-draft/` — Intermediate preparation draft (was `paper_v2/` before 2026-05-10), superseded by `submitted_version/`. `SUBMISSION_GUIDE.md` documents the submission steps that were followed; `openreview_abstract.txt` is the OpenReview abstract.
- `submitted_version/` — **The actual NeurIPS 2026 paper sent to OpenReview** (was `2026-asad-QA/` before 2026-05-10). `main.tex` + `references.bib`, NeurIPS 2026 style; `Paper_V8.pdf` is the submitted PDF. Submission deadline 2026-05-06 has passed — treat as archival.
- `thesis/` — MSc/PhD thesis sources (LaTeX in `thesis/latex/`).
- `proposals/` — Research proposals; PhD application materials. `2026-05-05-innopolis-phd-proposal.{md,tex,pdf}` is the Innopolis PhD proposal (Vladimir Ivanov as proposed supervisor). Compile with: `cd proposals && xelatex -interaction=nonstopmode 2026-05-05-innopolis-phd-proposal.tex` (run twice for cross-refs). XeLaTeX is required for Cyrillic (Russian abstract via polyglossia).
- `github_release/` — Staged GitHub release contents (MSQA-Bench code + LICENSE + README); pushes to `AsadKhalil/MSQA-Bench`.
- `msqa-bench-hf/` — Hugging Face dataset repo; pushes to `asad00027/MSQA-Bench`.
- Public benchmark distribution archive (Zenodo concept DOI `10.5281/zenodo.19831805`) — generated by `scripts/prepare_neurips_release.py`.
- `EXPERIMENT_NOTES.md` — Post-experiment notes / follow-up tracking (in this repo).
- `scripts/submission_diagnostics.py` — Diagnostics for the NeurIPS submission package.
- `LICENSE` — Repository license (MIT).