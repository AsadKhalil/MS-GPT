# MS-GPT

Pipeline that builds **MSQA-Bench** — a large-scale question-answering benchmark for computational mass spectrometry — from peer-reviewed papers, then fine-tunes embedding models and LLMs on the resulting Q&A.

```
PDFs ─► extracted text ─► generated Q&A ─► consolidated JSONL ─►  embedding fine-tuning
                                                              └► QLoRA LLM fine-tuning
                                                              └► retrieval / generation evaluation
                                                              └► two-tier dataset release
```

The companion paper, dataset card, and HF release are linked at the bottom. The public code mirror (NeurIPS Evaluations & Datasets artifact) is `github_release/` in the parent workspace.

## Setup

Python 3.10. CUDA strongly recommended (training and Q&A generation are GPU-bound).

```bash
source .venv/bin/activate
pip install -r requirements.txt              # pinned via config/requirements.txt as well
```

External services used by individual stages:

| Stage | Service | Default endpoint |
| --- | --- | --- |
| Q&A generation | vLLM serving Qwen2.5-14B-Instruct-AWQ | `http://localhost:8000` (OpenAI-compatible) |
| Vision extraction (legacy) | Ollama | per `config/config.json` |
| Structured PDF parsing | GROBID 0.8.1 | `http://localhost:8070` |

Most code paths default to `CUDA_VISIBLE_DEVICES=2`.

## Pipeline commands

### 1. PDF → text

```bash
# Recommended: PyMuPDF + OCR fallback, parallelisable
python src/vision_extractors/fast_pdf_extractor.py /path/to/pdfs/ --workers 8 --output results/

# Vision-based (Ollama, slow; legacy)
python src/vision_extractors/vision_extractor.py

# Plain PyMuPDF
python src/pdf_processors/pymupdf_processor.py

# GROBID (structured academic parsing)
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1
python src/pdf_processors/grobid_batch_processor.py
```

### 2. Text → Q&A

```bash
# vLLM lifecycle
./scripts/start_vllm_background.sh 2 14b 8000 --force
tail -f logs/vllm_gpu2.log
kill $(cat logs/vllm_gpu2.pid)

# Generate Q&A
python src/qa_generators/qa_generator.py --config config/qa_generator.json
python scripts/consolidate_qa.py             # merge per-doc outputs into a single JSONL
```

### 3. Embedding fine-tuning

```bash
python scripts/validate_embedding_setup.py                                       # sanity-check before any training
python scripts/train_all_models.py --config config/embedding_finetuner.json -y   # train every model in the config
```

Streaming sentence-transformers training (e5-base/large, bge-base/large, nomic-embed) with `MultipleNegativesRankingLoss`. Evaluator reports Recall@k, MRR, NDCG.

### 4. LLM fine-tuning (QLoRA)

```bash
python scripts/train_all_llms.py --config config/llm_finetuner.json -y    # train every model in the config
python scripts/train_all_llms.py --model qwen2.5_3b -y                     # single model
python scripts/train_all_llms.py --subset 50 -y                            # smoke test
python scripts/train_all_llms.py --eval-only                               # skip training, evaluate adapters

python -m src.llm_trainers.llm_finetuner --config config/llm_finetuner.json   # direct (no wrapper)
```

QLoRA via `transformers` + `peft` + `trl.SFTTrainer` + `bitsandbytes` 4-bit (NF4). Evaluator reports ROUGE, BERTScore, F1, faithfulness, perplexity.

### 5. Paper experiments and release

```bash
python scripts/run_paper_pipeline.py --config config/paper_pipeline.json   # split + retrieval + audit pipeline
python scripts/evaluate_bm25_baseline.py                                   # controlled 5K-query BM25 baseline
python scripts/eval_llms_1k.py                                             # 1K-query LLM comparison
python scripts/run_rag_ablation.py                                         # RAG ablation grid
python scripts/run_rag_zeroshot_eval.py                                    # zero-shot retrieval-augmented eval
python scripts/llm_judge_gold_set.py                                       # LLM-judge over the audit set
python scripts/eval_zeroshot_openai.py                                     # zero-shot OpenAI baseline
python scripts/compute_iaa.py                                              # Cohen's κ inter-annotator agreement
python scripts/extract_all_results.py                                      # consolidate all metrics → tables
python scripts/visualize_training_results.py                               # training charts
python scripts/visualize_llm_training.py                                   # per-LLM loss curves
python scripts/generate_thesis_figures.py                                  # thesis-grade figure exports
python scripts/verify_references.py ../submitted_version/references.bib   # bib audit on the submitted paper
python scripts/prepare_neurips_release.py \
       --input-dir paper_results/dataset/splits \
       --output-dir paper_results/neurips_release                          # two-tier release (redistributable/restricted)
python scripts/reconstruct_restricted_record.py \
       --restricted-jsonl paper_results/neurips_release/restricted/test.jsonl \
       --pdf-dir /path/to/local/pdfs \
       --output-jsonl reconstructed_test.jsonl                             # local reconstruction of restricted records
python scripts/fix_croissant_schema.py                                     # validate / repair Croissant 1.1
python scripts/submission_diagnostics.py                                   # final NeurIPS package diagnostics
```

## Source layout

```
src/
├── pdf_processors/         PyMuPDF, LLM-clean (single/batch/page-by-page), GROBID, multi-format
├── vision_extractors/      fast_pdf_extractor (recommended) + agentic_vision_extractor (legacy LangGraph + Ollama)
├── qa_generators/          vLLM-driven Q&A generation (OpenAI-compatible API)
├── embedding_trainers/     streaming_finetuner, evaluators, data_utils
└── llm_trainers/           llm_finetuner, evaluators, data_utils, model_comparison
```

```
scripts/                   pipeline entrypoints & release tooling (see "Paper experiments and release" above)
config/
├── config.json                   Ollama / vision settings
├── qa_generator.json             vLLM endpoint, parallelism, circuit breaker
├── embedding_finetuner.json      models_to_train list
├── llm_finetuner.json            QLoRA hyperparameters + models_to_train list
├── paper_pipeline.json           split + retrieval pipeline
├── benchmark_config.json         retrieval baselines
└── fast_extractor.json           fast_pdf_extractor settings
```

## Data and outputs

```
data/
├── input/                  source PDFs (named by content hash)
├── processed_pdfs/         intermediate processed outputs
├── extracted_text/         per-doc text
├── qa_outputs/             per-doc Q&A JSONL
└── consolidated_qa.jsonl   merged Q&A used by trainers and the paper pipeline

models/
├── fine_tuned_embeddings/<model_name>/        sentence-transformers checkpoints + eval results
└── fine_tuned_llms/<model_name>/
    ├── final_adapter/                          LoRA weights
    └── eval_results_test.json                  per-model metrics

paper_results/evaluation/llms_eval_1k/<model_name>/    1K-query eval outputs (deepseek_r1_distill_7b, mistral_7b_v0.3, phi3.5_mini, qwen2.5_7b)
paper_results/evaluation/llms_eval_1k/llm_comparison.json  cross-model comparison
paper_results/evaluation/llms_eval_1k/summary_n1000.json   aggregated summary
```

Both trainers use the same **deterministic, document-level 85/10/5 split** (MD5 hashing of doc hashes) so embedding and LLM evaluation see compatible boundaries and no paper crosses splits.

## Tests

```bash
python -m pytest                                              # full suite
python -m pytest src/vision_extractors/test_vision_extractor.py    # one module
```

Format / lint / type-check (recommended on changed files):

```bash
black src scripts
flake8 src scripts
mypy src
```

## Reproducing the public BM25 row of Table 3

```bash
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("asad00027/MSQA-Bench", "redistributable", split="test")
ds.to_json("/tmp/msqa_redistributable_test.jsonl")
PY

python3 scripts/evaluate_bm25_baseline.py \
        --data /tmp/msqa_redistributable_test.jsonl \
        --output paper_results/evaluation \
        --sample-size 5000
# Expected: recall@10 ≈ 0.742, mrr@10 ≈ 0.668, ndcg@10 ≈ 0.686
```

## How this repo relates to the others

This repository (`MS-GPT/`) is the **active development tree.** It sits inside a parent workspace alongside:

- `github_release/` — public code mirror; `https://github.com/AsadKhalil/MSQA-Bench`. Slimmed-down snapshot of `MS-GPT/` for the NeurIPS code artifact (no `data/`, `models/`, `paper_results/`).
- `msqa-bench-hf/` — Hugging Face dataset repository; `https://huggingface.co/datasets/asad00027/MSQA-Bench`. Holds `redistributable/`, `restricted/`, `sample/`, `croissant.jsonld`, `license_audit.csv`.
- `submitted_version/` — the **actual NeurIPS 2026 paper sent to OpenReview** (was named `2026-asad-QA/` before the 2026-05-10 cleanup).
- `archive/paper-v2-prep-draft/` — intermediate preparation draft (was named `paper_v2/`); superseded by `submitted_version/`.
- `archive/paper-v1-original/` — earliest paper draft + library-style helper modules (since re-implemented as `scripts/` in this repo).
- `paper_results/` — generated experimental outputs cited by all papers/thesis.
- `thesis/`, `proposals/` — thesis and PhD-proposal LaTeX sources.

See `../README.md` and `../CLAUDE.md` for the workspace-level orientation.

## Operational notes

- **Don't commit `.venv/`, `models/`, `paper_results/`, generated logs, or large PDFs.** All are listed in `.gitignore`.
- **Numbers in the paper must trace to files in `paper_results/`.** `scripts/extract_all_results.py` is the consolidator; `scripts/verify_references.py` audits citations.
- **Restricted-tier records are metadata-only.** Never emit `question`/`answer`/`context`/`evidence_spans` for `redistribution_status == "metadata_only"` records. Use `scripts/reconstruct_restricted_record.py` to reconstruct locally with a private PDF collection.
- **Zenodo records are immutable.** To update the archived release, create a *new version* under the existing concept DOI `10.5281/zenodo.19831805` — do not try to overwrite a published file. Always cite the concept DOI in the paper and dataset card, not version DOIs.
- **Generation results are fine-tuned-only.** The paper presents fine-tuned LLM baselines as diagnostic; do not reframe them as base-vs-FT deltas without adding `Base / +FT / Δ` columns.
- **The audit is residual-noise estimation, not gold.** The 191-record human audit (Cohen's κ = 0.31 answer correctness, 0.41 evidence support) estimates label noise; do not describe it as expert-authored ground truth.
- **Half-finished training? Read `EXPERIMENT_NOTES.md` first** — it tracks which checkpoints are converged, what subset training is acceptable, and the reasoning behind 50K-row subset training for late-stage models.

## License

Code is released under the MIT License. Dataset records hosted on Hugging Face follow the two-tier release described in the dataset card (CC-BY-4.0 for the redistributable tier; source-paper terms for the restricted tier).

## Citation

```bibtex
@inproceedings{asad2026msqabench,
  title     = {MSQA-Bench: A Large-Scale Question-Answering Benchmark for Computational Mass Spectrometry with Retrieval and Generation Baselines},
  author    = {Asad, Muhammad and Sulimov, Daniil and Kertesz-Farkas, Attila},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```

## Links

- Paper (submitted version): `../submitted_version/main.tex` (build with `make` from inside that directory)
- Code (this repo, public mirror): https://github.com/AsadKhalil/MSQA-Bench
- Dataset (Hugging Face): https://huggingface.co/datasets/asad00027/MSQA-Bench
- Archival mirror (Zenodo, concept DOI): https://doi.org/10.5281/zenodo.19831805
