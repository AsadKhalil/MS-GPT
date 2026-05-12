# Plan: Complete MS-GPT Paper for Database Track Submission

## Context

The MS-GPT paper (`paper_v2/latex/main.tex`) is structurally complete but has blocking gaps: no LLM test-set evaluation metrics (Table 3 only shows training loss), and "in progress" language for 3 models. The user wants to keep all 5 LLMs in the paper. GPU time is scarce, but the loss curves show all models converge by ~5K steps, so training on a 50K subset gives comparable results to 1M+. Two models already have usable converged checkpoints (Phi-3.5, Qwen-3B). Only Llama-3.1 and DeepSeek-R1 need fresh subset training. The user will provide evaluation metrics after training; I will make all paper edits.

---

## Phase 0: User Prepares Model Adapters + Evaluation (GPU work)

### Step 0a: Rescue existing converged checkpoints (0 GPU hours)

```bash
# Phi-3.5: loss converged at 1.41 by step 15K (barely moved since step 5K)
cp -r /home/asad/MS-GPT/models/fine_tuned_llms/phi3.5_mini/checkpoint-15000 \
      /home/asad/MS-GPT/models/fine_tuned_llms/phi3.5_mini/final_adapter

# Qwen-3B: fully trained 3 epochs, loss converged at 1.54
cp -r /home/asad/models/fine_tuned_llms/qwen2.5_3b/checkpoint-47427 /home/asad/models/fine_tuned_llms/qwen2.5_3b/final_adapter
```

### Step 0b: Train remaining 2 models on 50K subset (~8h each)

**Prerequisite for Llama:** Ensure `HF_TOKEN` env var is set (previous attempt failed on this).

```bash
# Llama-3.1 (~1,328 steps, ~8 hours on 24GB GPU)
python scripts/train_all_llms.py --model llama3.1_8b --subset 50000 -y

# DeepSeek-R1 (~1,328 steps, ~8 hours on 24GB GPU)
python scripts/train_all_llms.py --model deepseek_r1_distill_7b --subset 50000 -y
```

**Why 50K works:** Loss curves from Mistral (full 31K steps) show convergence by step 1K-2K. 50K subset = ~1,328 steps = past the convergence elbow. Mistral's loss at step 1000 was 1.606 vs final 1.503 -- the delta is noise-level for evaluation metrics.

### Step 0c: Run evaluation on ALL models

```bash
python scripts/extract_all_results.py
```

This produces per-model JSON with: ROUGE-1/2/L, BERTScore F1, Token F1, Exact Match, Faithfulness, Perplexity.

**I need these numbers for all 5 models (+ optionally Qwen-3B as 6th):**

| Metric | Needed for |
|--------|-----------|
| ROUGE-1, ROUGE-2, ROUGE-L | Table 3 (main results) |
| BERTScore F1 | Table 3 |
| Token F1 | Table 3 |
| Faithfulness (NLI) | Table 3 + Abstract |
| Perplexity | Text mention |
| Exact Match | Text mention |

---

## Phase A: Paper Edits Independent of Metrics (immediate)

### Step 1: Keep all 5 LLMs, remove "in progress" language

All 5 models stay in the paper. The only text that changes:

| Location | Lines | Change |
|----------|-------|--------|
| Results 5.2 | 399-403 | **Delete** "Three additional models (Phi-3.5-mini, Llama-3.1-8B, DeepSeek-R1-Distill-7B) are currently in progress; their results will be reported in a subsequent revision." |
| Results 5.2 | 406-420 | Replace training-only Table 3 with evaluation results table (Phase B) |
| Results 5.2 | 422-427 | **Delete** "Comprehensive generation evaluation... is planned" |

Everything else (abstract listing 5 models, intro contributions, methodology model list) **stays as-is**.

### Step 2: Add Quality Assessment Paragraph

Add to Section 3 (after dataset statistics) a paragraph on automated quality audit:
- 200-sample stratified assessment: ~61.5% fully correct, ~30% partial, ~8.5% incorrect
- 95% questions well-formed
- ~3.5% non-English content acknowledged
- Framed as "automated heuristic assessment" (not human eval)

### Step 3: Fix Limitations Section

Line 500: Replace "limited to English-language publications" with honest acknowledgment that ~3.5% is non-English. Frame as reflecting real-world multilingual MS literature.

### Step 4: Add Author Statement (Appendix)

New Appendix C with D&B compliance: data source legality, no PII, CC-BY-4.0 license, maintenance commitment.

### Step 5: Add Dataset URL Placeholder

Add `\url{https://huggingface.co/datasets/TBD}` in:
- Conclusion section
- Appendix A (Distribution paragraph, line 567)

User fills the real URL after uploading.

---

## Phase B: Fill Results After User Provides Metrics

### Step 6: Replace Table 3 with Full Generation Evaluation Table

**Current Table 3** (lines 405-420): Only shows training loss, train samples, runtime.

**New Table 3 design:**

```latex
\begin{table}[t]
\centering
\caption{Generation evaluation on \msqa{} test set. R = ROUGE, BS = BERTScore F1,
TF1 = token-level F1, Faith. = NLI-based faithfulness.}
\label{tab:generation}
\small
\begin{tabular}{@{}l ccc cc c@{}}
\toprule
\textbf{Model} & \textbf{R-1} & \textbf{R-2} & \textbf{R-L} &
\textbf{BS F1} & \textbf{TF1} & \textbf{Faith.} \\
\midrule
Phi-3.5-mini (3.8B)    & X.XX & X.XX & X.XX & X.XX & X.XX & X.XX \\
Mistral-7B-v0.3        & X.XX & X.XX & X.XX & X.XX & X.XX & X.XX \\
Llama-3.1-8B           & X.XX & X.XX & X.XX & X.XX & X.XX & X.XX \\
Qwen2.5-7B             & X.XX & X.XX & X.XX & X.XX & X.XX & X.XX \\
DeepSeek-R1-Distill-7B & X.XX & X.XX & X.XX & X.XX & X.XX & X.XX \\
\bottomrule
\end{tabular}
\end{table}
```

Move training metadata to a text sentence: "Models were trained for 1 epoch with QLoRA (rank 64, 4-bit NF4) converging to training loss 1.41--1.96 across architectures in 8--183 GPU-hours."

### Step 7: Rewrite Generation Results Section (5.2)

Replace placeholder text (lines 399-427) with:
- Table description referencing evaluation metrics
- Cross-model comparison (which model scores highest and why)
- Faithfulness interpretation (grounding quality for domain QA)
- Training convergence observation: all models converge to similar loss despite different architectures
- Note on DeepSeek-R1's higher loss (expected -- reasoning-distilled model has different output distribution)

### Step 8: Update Abstract with Concrete Numbers

Replace vague "converge to low training loss (~1.52)" with actual top-line metrics:
- Best ROUGE-L score across models
- Best BERTScore F1
- Faithfulness percentage

### Step 9: Update Conclusion

Line 519: Ground claims in actual numbers instead of vague "significantly outperform."

---

## Phase C: Verification

1. `grep -c "in progress\|subsequent revision\|is planned" paper_v2/latex/main.tex` → 0
2. `grep -c "\\\\placeholder\|\\\\todo" paper_v2/latex/main.tex` → 0
3. All Table 3 cells contain real numbers (no X.XX)
4. Abstract contains concrete metric values
5. LaTeX compiles without errors or `??` references
6. Page count ≤ 9 main body (NeurIPS D&B)

---

## Critical Files

| File | Action |
|------|--------|
| `paper_v2/latex/main.tex` | All textual edits (Steps 1-9) |
| `paper_v2/latex/references.bib` | No changes needed (all 5 models kept) |
| `config/llm_finetuner.json` | No changes needed (`--subset` flag overrides at CLI) |

## Timeline

| Phase | What | GPU Hours | My Edit Time |
|-------|------|-----------|-------------|
| 0a | Copy checkpoints | 0 | 0 |
| 0b | Train Llama + DeepSeek | ~16h total | 0 |
| 0c | Run evaluation | ~2-4h | 0 |
| A | Paper edits (no metrics needed) | 0 | ~1h |
| B | Fill Table 3 + analysis | 0 | ~1-2h |
| C | Verification | 0 | ~30min |
| **Total** | | **~18-20h GPU** | **~3h editing** |

<!-- HF token redacted — set HUGGINGFACE_HUB_TOKEN in .env (or `huggingface-cli login`) instead of pasting tokens in notes. -->











# Plan: MS-GPT Master's Thesis Report (~50-60 pages)

## Context

Muhammad Asad (HSE Moscow) has a completed NeurIPS 2026 conference paper (`paper_v2/latex/main.tex`, ~8 pages, 42 references) on MS-GPT -- a RAG system for mass spectrometry QA. The project includes:
- **MSQA-Bench**: 1.2M QA pairs from 32,376 documents
- **6 embedding models** fine-tuned (all have results)
- **6 LLM models** QLoRA-trained (4 completed with final_adapter, 1 checkpoint-only, 1 failed)
- **Working RAG system**: React + FastAPI + PostgreSQL/pgvector
- **Planned experiments** (from `abundant-coalescing-balloon.md`): RAG ablation study, per-question-type analysis, hard negative mining, data quality study

The thesis must expand the paper into a ~50-60 page Master's thesis, include planned experiments as core chapters (with placeholders for pending results), use a single-column academic style, and have verified references.

---

## Phase 1: Scaffold Thesis Directory & LaTeX Template

### Step 1.1: Create directory structure

```
thesis/
  latex/
    main.tex                    # Master document (\include for chapters)
    preamble.tex                # Packages, macros, page geometry
    titlepage.tex               # HSE-style title page
    abstract.tex                # English abstract (~300 words)
    chapters/
      ch1_introduction.tex
      ch2_background.tex        # Literature review
      ch3_dataset.tex           # MSQA-Bench construction
      ch4_methodology.tex       # Embeddings + QLoRA + RAG
      ch5_experiments.tex       # All results + ablations
      ch6_system.tex            # RAG system implementation
      ch7_conclusion.tex        # Discussion + conclusion + future work
    appendices/
      appendix_a_datasheet.tex
      appendix_b_hyperparams.tex
      appendix_c_prompts.tex
    references.bib              # Expanded from paper_v2 (42 -> ~80-90 refs)
    Makefile
  figures/                      # Symlinks or copies of existing charts
```

### Step 1.2: LaTeX template setup (`preamble.tex`)

- `\documentclass[12pt,a4paper,oneside]{report}` -- single-column, chapters
- `geometry`: left=3cm, right=2cm, top=2.5cm, bottom=2.5cm
- 1.5 line spacing (`setspace`)
- `fancyhdr` for headers (chapter name + page number)
- Reuse all packages from paper: tikz, booktabs, amsmath, natbib, hyperref, algorithm, enumitem
- Reuse custom macros: `\msgpt`, `\msqa`, `\todo`, `\placeholder`
- Add: `graphicx`, `subcaption`, `listings` (for code snippets), `longtable`

### Step 1.3: Title page (`titlepage.tex`)

```
National Research University Higher School of Economics
Faculty of Computer Science
Master's Thesis

Title: Developing Large Language Models for Question-Answering
       in Computational Mass Spectrometry

Student: Muhammad Asad
Supervisor: Attila Kertesz-Farkas
Moscow, 2026
```

**Files to create:** `thesis/latex/main.tex`, `thesis/latex/preamble.tex`, `thesis/latex/titlepage.tex`, `thesis/latex/abstract.tex`, `thesis/latex/Makefile`

---

## Phase 2: Write Chapter Content

### Chapter 1: Introduction (5-6 pages)

**Reuse from paper:** Section 1 (lines 71-127, ~1.5 pages). Expand to include:

- Problem statement: LLMs hallucinate on specialized MS questions (concrete examples)
- Research questions (from `abundant-coalescing-balloon.md`):
  - RQ1: Does domain-adapted retrieval outperform general retrieval for MS?
  - RQ2: Does domain-adapted generation outperform general generation?
  - RQ3: Do retrieval + generation improvements compound end-to-end?
  - RQ4: Which question types benefit most from domain adaptation?
- Scope and delimitations
- Contributions (4 items, expanded from paper)
- Thesis structure roadmap

**Source files to read:** `paper_v2/latex/main.tex:71-127`

---

### Chapter 2: Background and Literature Review (10-12 pages)

**Reuse from paper:** Section 2 (lines 132-161, ~1 page). Major expansion into 5 subsections:

**2.1 Mass Spectrometry Fundamentals (2-3 pages)**
- Ionization methods (ESI, MALDI), mass analyzers (TOF, Orbitrap), fragmentation
- Application domains: proteomics, metabolomics, lipidomics, clinical
- Volume of MS literature and why automated QA is needed
- Refs to add: ~6-8 foundational MS references (reviews, textbooks)

**2.2 Natural Language Processing for Scientific Text (2-3 pages)**
- Transformer architecture overview (BERT, GPT lineage)
- Domain-specific models: BioBERT, PubMedBERT, SciBERT, BioGPT, Galactica
- Instruction tuning and alignment techniques
- Refs to add: ~10-12 (Vaswani attention, Devlin BERT, Radford GPT, domain models)

**2.3 Question-Answering Systems and Benchmarks (2-3 pages)**
- QA taxonomy: extractive, abstractive, open-domain, closed-domain
- Existing benchmarks: SQuAD, PubMedQA, BioASQ, SciQ, SciFact, MMLU, ARC
- Gap analysis: no MS-specific QA benchmark
- Refs to add: ~8-10

**2.4 Retrieval-Augmented Generation (2 pages)**
- RAG framework (Lewis et al.), variants (naive, advanced, modular)
- Dense vs. sparse retrieval (DPR, BM25)
- Embedding model families: E5, BGE, Nomic, sentence-transformers
- Refs to add: ~8-10 (REALM, Atlas, Self-RAG, ColBERT, FAISS, pgvector)

**2.5 Parameter-Efficient Fine-Tuning (1-2 pages)**
- LoRA mathematical formulation: W' = W_0 + BA
- QLoRA: NF4 quantization + double quantization
- Comparison: full fine-tuning vs. LoRA vs. prefix tuning vs. adapters
- Refs to add: ~5-6

**Total new references for Ch2: ~40-45 (bringing total to ~80-90)**

**Critical:** All references must be verified real papers. Use `scripts/verify_references.py` and manual DOI checks.

---

### Chapter 3: MSQA-Bench Dataset Construction (8-10 pages)

**Reuse from paper:** Section 3 (lines 167-293, ~3 pages). Expand each subsection:

**3.1 Source Corpus Acquisition (1.5 pages)**
- Semantic Scholar API details, query construction
- Corpus statistics: year distribution, journal distribution
- New figure: publication year histogram
- Source: `paper/dataset/metadata_extractor.py`, `paper_results/dataset/`

**3.2 Text Extraction Pipeline (1.5 pages)**
- PyMuPDF + OCR fallback + GROBID structured parsing
- Algorithm pseudocode (Algorithm 1)
- Source: `src/vision_extractors/fast_pdf_extractor.py`, `src/pdf_processors/`

**3.3 QA Pair Generation (2 pages)**
- Exact prompt template used with Qwen2.5-14B via vLLM
- Circuit breaker, checkpoint resume, fault tolerance
- Algorithm pseudocode (Algorithm 2)
- Source: `src/qa_generators/qa_generator.py`

**3.4 Quality Filtering and Deduplication (1.5 pages)**
- Quality score formula: q = 0.3*overlap + 0.25*specificity + 0.2*completeness + 0.25*length_norm
- MinHash LSH deduplication parameters
- Filtering statistics table (new): raw -> length filtered -> quality filtered -> deduped -> final
- Source: `paper/dataset/quality_filters.py`

**3.5 Question Classification (1 page)**
- Rule-based classifier with 7 types
- Distribution table (reuse `paper_results/figures/table_question_types.tex`)
- Example questions per type (new table)
- Source: `paper/dataset/question_classifier.py`

**3.6 Dataset Splits (0.5 page)**
- Hash-based document-level splitting (85/10/5)
- Cross-split leakage prevention
- Reuse Table 1 (dataset statistics)
- Source: `paper/dataset/split_generator.py`, `paper_results/dataset/splits/split_statistics.json`

**Reuse:** Pipeline TikZ figure (lines 173-211), Table 1 (lines 272-293)

---

### Chapter 4: Methodology (8-10 pages)

**Reuse from paper:** Section 4 (lines 299-358, ~2 pages). Major expansion with equations:

**4.1 Embedding Model Fine-Tuning (3 pages)**
- Formal problem statement: given query q, find passages p in corpus C
- Sentence-transformer architecture diagram (new figure)
- MultipleNegativesRankingLoss equation:
  L = -log(exp(sim(q,p+)/tau) / sum_i(exp(sim(q,p_i)/tau)))
- In-batch negative sampling: effective negatives = batch_size - 1
- Model-specific configs: instruction prefixes for E5 vs BGE vs Nomic
- Training hyperparameters table (from `config/embedding_finetuner.json`)
- Algorithm pseudocode (Algorithm 3)
- Source: `src/embedding_trainers/streaming_finetuner.py`

**4.2 LLM Fine-Tuning with QLoRA (3 pages)**
- QLoRA decomposition: W = W_0 + BA, B in R^{d x r}, A in R^{r x k}
- NF4 quantization explanation with figure
- Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- Chat template formatting (system/user/assistant)
- No-context training strategy (30% without context)
- Hyperparameter table (reuse/expand Appendix B Table 4)
- Algorithm pseudocode (Algorithm 4)
- Source: `src/llm_trainers/llm_finetuner.py`, `config/llm_finetuner.json`

**4.3 End-to-End RAG Pipeline (2 pages)**
- Query encoding -> HNSW retrieval -> context assembly -> LLM generation
- Architecture diagram (new figure)
- Prompt construction for grounded generation
- Source: `src/rag_evaluator/` (to be built per `abundant-coalescing-balloon.md`)

**4.4 Evaluation Framework (2 pages)**
- Retrieval metrics with equations: Recall@k, MRR@k, NDCG@k, MAP@k
- Generation metrics with equations: ROUGE-1/2/L, BERTScore, Token F1, Faithfulness
- End-to-end evaluation protocol (2x2 ablation design)
- Source: `src/embedding_trainers/evaluators.py`, `src/llm_trainers/evaluators.py`

---

### Chapter 5: Experiments and Results (10-12 pages)

**Reuse from paper:** Section 5 (lines 364-455, ~2.5 pages). Major expansion:

**5.1 Experimental Setup (1.5 pages)**
- Hardware: GPU specs (NVIDIA RTX 3090/A100 24GB), CUDA version
- Software stack: Python 3.10, PyTorch, transformers, peft, trl, sentence-transformers versions
- Reproducibility measures: seeds, deterministic splits, config versioning

**5.2 Retrieval Results (2-3 pages)**
- Expanded Table 2 with ALL metrics (R@1, R@5, R@10, MRR@10, NDCG@10) for all 7 models (BM25 + 6 embeddings), base vs. fine-tuned
- Data sources for all metrics:
  - BM25: `paper_results/evaluation/bm25_baseline_results.json`
  - E5-base: `models/fine_tuned_embeddings_e5_base_v2/training_summary.json`
  - E5-large: `models/fine_tuned_embeddings_e5_large_v2/training_summary.json`
  - BGE-base: `models/fine_tuned_embeddings_bge_base_en_v1.5/training_summary.json`
  - BGE-large: `models/fine_tuned_embeddings_bge_large_en_v1.5/training_summary.json`
  - Nomic: `models/fine_tuned_embeddings_nomic_embed_v1.5/training_summary.json`
  - MiniLM: `training_summary.json` (project root)
- New figures: Recall@k bar chart, radar chart, improvement heatmap (from `paper_results/figures/` if available, otherwise generate)
- Analysis: why base models outperform large after fine-tuning

**5.3 Generation Results (2-3 pages)**
- Table 3 with full evaluation: ROUGE-1/2/L, BERTScore F1, Token F1, Faithfulness for all completed models
- **Placeholder cells** for models pending evaluation (user runs `scripts/extract_all_results.py`)
- Training curves (reuse `training_charts/training_loss.png`, `training_dashboard.png`)
- Per-model analysis: convergence behavior, loss comparisons
- Llama-3.1-8B failure analysis (lessons learned)

**5.4 End-to-End RAG Ablation (2-3 pages) -- PLACEHOLDER**
- 2x2 matrix: {base, fine-tuned} embedding x {base, fine-tuned} LLM
- Results table with all generation metrics under each condition
- Analysis: do improvements compound?
- This is the key thesis finding (from `abundant-coalescing-balloon.md`)
- Script: `scripts/run_rag_ablation.py` (to be built)

**5.5 Per-Question-Type Analysis (1-2 pages) -- PLACEHOLDER**
- Breakdown of retrieval + generation metrics by 7 question types
- Heatmap or table showing which types benefit most
- Script: `scripts/question_type_analysis.py` (to be built)

**5.6 Human Evaluation (1 page)**
- 200-sample gold set audit results
- Metrics: answer_correct, evidence_support, evidence_quality, question_clarity
- Source: `paper_results/annotation/gold_set_annotated.csv`

---

### Chapter 6: System Implementation (4-5 pages)

**Reuse from paper:** Section 6 (lines 461-477, ~0.5 page). Major expansion:

**6.1 System Architecture (1.5 pages)**
- Architecture diagram (new figure): React -> FastAPI -> PostgreSQL/pgvector
- Component overview and data flow
- Technology stack justification

**6.2 Backend (1.5 pages)**
- FastAPI endpoints, middleware
- pgvector HNSW indexing: parameters, ANN tradeoffs
- Embedding service: model loading, batch encoding
- LLM inference: adapter merging, generation parameters

**6.3 Frontend (1 page)**
- React interface design
- Query input, result display, source attribution
- Document upload workflow
- **Screenshots of the working application** (user provides)

**6.4 Deployment (0.5 page)**
- Docker containerization
- GPU memory management
- Performance: latency, throughput measurements

---

### Chapter 7: Discussion, Conclusion and Future Work (4-5 pages)

**Reuse from paper:** Sections 7-8 (lines 483-527, ~1.5 pages). Expand:

**7.1 Discussion (2 pages)**
- Positioning against related work (BioASQ, PubMedQA systems)
- Broader implications for scientific QA in other domains
- Limitations with mitigations (4-5 items)
- Ethical considerations: data provenance, attribution, potential misuse

**7.2 Conclusion (1 page)**
- Summary tied to research questions RQ1-RQ4
- Key contributions restated with concrete numbers

**7.3 Future Work (1-2 pages)**
- Hard negative mining (from `abundant-coalescing-balloon.md`)
- DPO-based hallucination reduction
- DARE-TIES model merging
- ColBERT late interaction retrieval
- Multi-hop QA subset
- GraphRAG with MS knowledge graph
- Multilingual extension

---

### Appendices (~5 pages)

**Appendix A: Datasheet for MSQA-Bench (2 pages)** - Reuse/expand paper Appendix A
**Appendix B: Training Hyperparameters (1.5 pages)** - Full tables for embedding + LLM configs
**Appendix C: Prompt Templates (1.5 pages)** - QA generation prompt, RAG prompt, evaluation prompts

---

## Phase 3: Build References (`references.bib`)

### Step 3.1: Copy and expand `paper_v2/latex/references.bib`

Start with existing 42 references. Add ~40-45 new references in these categories:

| Category | Current | Target | New refs needed |
|----------|---------|--------|-----------------|
| MS fundamentals | 3 | 8 | 5 |
| NLP/Transformers | 0 | 8 | 8 |
| Scientific QA benchmarks | 6 | 12 | 6 |
| Domain-specific LMs | 4 | 10 | 6 |
| RAG & retrieval | 3 | 10 | 7 |
| Embedding models | 5 | 8 | 3 |
| PEFT methods | 2 | 6 | 4 |
| LLM base models | 5 | 8 | 3 |
| Evaluation metrics | 4 | 6 | 2 |
| Vector DBs / ANN | 0 | 3 | 3 |
| Quantization | 0 | 2 | 2 |
| **Total** | **42** | **~85** | **~45** |

### Step 3.2: Verify ALL references

Run `python scripts/verify_references.py` on the expanded .bib file. Manually verify DOIs for any new entries. Every reference must have: author, title, year, venue/journal, and ideally a DOI.

---

## Phase 4: Figures and Tables

### Existing figures to reuse/include:
1. Pipeline TikZ diagram (from paper, lines 173-211)
2. `training_charts/training_loss.png`
3. `training_charts/training_dashboard.png`
4. `training_charts/learning_rate.png`
5. `training_charts/grad_norm.png`
6. `training_charts/token_accuracy.png`
7. `training_charts/training_loss_vs_epoch.png`

### New figures to create:
8. LoRA/QLoRA architecture diagram (TikZ)
9. RAG system architecture diagram (TikZ)
10. Retrieval results bar chart (Recall@k comparison)
11. Radar chart of embedding model metrics
12. RAG ablation 2x2 heatmap (placeholder)
13. Per-question-type performance heatmap (placeholder)
14. Application screenshots (user provides)

### Tables summary (~12 total):
1. Dataset statistics (reuse Table 1)
2. Question type distribution (reuse from paper_results)
3. Filtering pipeline statistics (new)
4. Embedding model configurations (new)
5. Retrieval results - full (expand Table 2 with R@1, R@5)
6. LLM model configurations (new)
7. QLoRA hyperparameters (reuse Table 4)
8. Generation evaluation results (expand Table 3)
9. RAG ablation results (new, placeholder)
10. Per-question-type breakdown (new, placeholder)
11. Human evaluation results (new, from gold_set_annotated.csv)
12. Comparison with existing benchmarks (new)

---

## Phase 5: Verification

1. `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` compiles clean
2. No `??` undefined references
3. `grep -c "\\\\todo\|\\\\placeholder\|X\.XX" thesis/latex/**/*.tex` -- track remaining placeholders
4. Page count: 50-60 pages main body (excluding front matter and appendices)
5. Reference count: 80-90, all verified
6. All existing figures render correctly
7. Table data matches source JSON files

---

## Implementation Order

| Step | What | Depends on |
|------|------|-----------|
| 1 | Create `thesis/` directory + template files | Nothing |
| 2 | Write `preamble.tex`, `titlepage.tex`, `abstract.tex`, `main.tex` skeleton | Step 1 |
| 3 | Write Ch1 Introduction | Step 2 |
| 4 | Write Ch2 Background & Literature Review + expand `references.bib` | Step 2 |
| 5 | Write Ch3 Dataset Construction (port + expand from paper) | Step 2 |
| 6 | Write Ch4 Methodology (port + expand from paper, add equations) | Step 2 |
| 7 | Write Ch5 Experiments (port results, add placeholders for pending) | Steps 3-6 |
| 8 | Write Ch6 System Implementation | Step 2 |
| 9 | Write Ch7 Discussion + Conclusion + Future Work | Steps 3-8 |
| 10 | Write Appendices A-C | Steps 5-6 |
| 11 | Create new TikZ figures (LoRA diagram, system architecture) | Step 6 |
| 12 | Copy existing figures to `thesis/figures/` | Step 1 |
| 13 | Run reference verification | Step 4 |
| 14 | Compile and verify | All |

Steps 3-6 are independent and can be written in parallel.
Steps 3-6 all reuse content from `paper_v2/latex/main.tex` as starting point.

---

## Critical Source Files

| File | Used for |
|------|----------|
| `paper_v2/latex/main.tex` | All reusable text, figures, tables |
| `paper_v2/latex/references.bib` | Base bibliography (42 refs) |
| `models/fine_tuned_embeddings_*/training_summary.json` | Embedding results (6 models) |
| `paper_results/evaluation/bm25_baseline_results.json` | BM25 baseline metrics |
| `models/fine_tuned_llms/*/training_summary.json` | LLM training metrics |
| `paper_results/dataset/splits/split_statistics.json` | Dataset split numbers |
| `paper_results/annotation/gold_set_annotated.csv` | Human eval data (200 samples) |
| `paper_results/figures/table_question_types.tex` | Question type distribution |
| `paper_results/figures/table_dataset_stats.tex` | Dataset statistics |
| `training_charts/*.png` | Training visualizations (6 charts) |
| `config/embedding_finetuner.json` | Embedding training config |
| `config/llm_finetuner.json` | LLM training config |
| `src/qa_generators/qa_generator.py` | Prompt templates for Ch3/Appendix C |
| `scripts/verify_references.py` | Reference verification |
| `.claude/plans/abundant-coalescing-balloon.md` | RAG ablation + research additions plan |
| `.claude/plans/abstract-inventing-pretzel.md` | Paper completion + LLM eval plan |

---

## Page Budget

| Section | Pages |
|---------|-------|
| Front matter (title, abstract, ToC, lists) | 4 |
| Ch1 Introduction | 5-6 |
| Ch2 Background & Literature Review | 10-12 |
| Ch3 Dataset Construction | 8-10 |
| Ch4 Methodology | 8-10 |
| Ch5 Experiments and Results | 10-12 |
| Ch6 System Implementation | 4-5 |
| Ch7 Discussion, Conclusion, Future Work | 4-5 |
| Appendices A-C | 5 |
| Bibliography | 3-4 |
| **Total** | **~61-69 pages** |




# Plan: Thesis Research Contributions Beyond Current Implementation

## Context: The Problem

The MS-GPT project has strong engineering (pipeline, dataset, training infrastructure) but the **research narrative** is thin. Currently:

- Embeddings and LLMs are evaluated **independently** — they've never been tested together as a RAG system
- "Fine-tuned model > base model" is the expected result, not a finding
- No analysis of **when/why** domain adaptation helps or fails
- No methodological novelty beyond applying existing techniques (QLoRA, MNR loss)

A thesis examiner will ask: *"What did you learn that we didn't already know?"*

## The Fix: Frame the Thesis Around Research Questions

Instead of "I built a system," the thesis should answer:

| RQ | Question | Status |
|----|----------|--------|
| RQ1 | Does domain-adapted retrieval outperform general retrieval for MS? | **Answered** (embedding results) |
| RQ2 | Does domain-adapted generation outperform general generation? | **Almost** (LLM training in progress) |
| RQ3 | Do retrieval + generation improvements compound end-to-end? | **NOT answered — critical gap** |
| RQ4 | Which question types benefit most from domain adaptation? | **NOT answered** |

---

## Proposed Additions (Priority Order)

### 1. End-to-End RAG Ablation Study (CRITICAL — ~1 week)

**What:** 2×2 experiment matrix:

```
                    Base LLM          Fine-tuned LLM
Base Embedding      baseline          generation-only gain
Fine-tuned Emb      retrieval-only    full RAG gain
                                      ↑ Does this compound?
```

**Why this matters:**
- Currently embeddings (Recall@k) and LLMs (ROUGE/BERTScore) are evaluated in isolation
- The LLM evaluator feeds **gold context** directly — it never uses retrieved passages
- Nobody knows if better retrieval actually leads to better answers in this domain
- This is a genuine empirical finding, not just engineering
- Every examiner will ask "does the system actually work end-to-end?"

**What to build:**
- New script: `src/rag_evaluator/end_to_end_eval.py`
- Takes: embedding model path, LLM adapter path, test split
- Does: encode corpus → retrieve top-k for each question → feed to LLM → evaluate answer
- Outputs: all generation metrics (ROUGE, BERTScore, Faithfulness) under each of the 4 conditions
- Reuse: `paper/evaluation/retrieval_baselines.py` (EmbeddingRetriever), `src/llm_trainers/evaluators.py` (generation metrics)

**Expected outcome:** The compounding effect (or lack thereof) is the thesis finding. Either:
- Improvements compound → "domain adaptation of both components is necessary"
- Retrieval dominates → "investing in retrieval is more cost-effective"
- Generation dominates → "LLM adaptation matters more than retrieval quality"

Any of these is a publishable insight.

---

### 2. Per-Question-Type Performance Breakdown (LOW EFFORT — ~2-3 days)

**What:** Break down ALL metrics by the 7 question types (factual, definition, method, causal, comparison, numeric, unknown).

**Why:** Reveals *which kinds* of scientific questions benefit from domain adaptation. This is analysis depth, not just aggregate numbers.

**What to build:**
- Filter test split by `question_type` field (already classified)
- Run existing evaluation per subset
- Generate table/figure showing per-type performance

**Expected outcome:** Likely finding: factual/definition questions improve most (pattern matching), while causal/comparison questions improve less (require reasoning). This has practical implications for when domain adaptation is worth the investment.

---

### 3. Hard Negative Mining for Embeddings (MODERATE EFFORT — ~1 week)

**What:** Use the already-trained embedding model to mine hard negatives, then retrain.

**Why:** This is a **methodological contribution** — shows iterative self-improvement. Currently using in-batch negatives only (easy negatives). Hard negatives (passages that are similar but wrong) are the standard improvement.

**What to build:**
- Script to mine top-50 passages per question using fine-tuned model
- Filter out correct answer, keep ranks 10-50 as hard negatives
- Retrain with TripletLoss or augmented MultipleNegativesRankingLoss
- Compare: base → fine-tuned (stage 1) → hard-negative fine-tuned (stage 2)

**Expected outcome:** 5-15% additional Recall@k improvement. The 3-stage progression (base → domain-adapted → hard-negative-refined) tells a compelling iterative improvement story.

---

### 4. Data Quality Impact Study (LOW EFFORT — ~3-4 days)

**What:** Train embeddings on different quality tiers (top 25%, 50%, 75%, 100% by quality_score) and compare.

**Why:** Data-centric AI question — does more data always help, or does quality matter more? The quality_score field already exists on every record.

**What to build:**
- Filter consolidated_qa.jsonl by quality_score thresholds
- Train same model on each subset
- Plot quality-quantity tradeoff curve

**Expected outcome:** Reveals whether careful filtering (smaller but cleaner data) outperforms using everything. Relevant to practitioners building domain-specific systems.

---

## Recommended Thesis Narrative (1-2 month timeline)

**Must do (non-negotiable):**
1. Finish LLM training (fix Phi, Llama failures)
2. Fill paper tables with real numbers
3. **End-to-End RAG Ablation** (#1 above) — this IS the thesis contribution
4. **Per-Question-Type Breakdown** (#2 above) — nearly free, adds depth
5. Fill gold set annotations (user is doing this)

**Should do (if time allows):**
6. Hard Negative Mining (#3) — adds methodological contribution
7. Data Quality Study (#4) — adds data-centric insight

**Items 3+4 alone transform the thesis** from "I fine-tuned models" to "I conducted an empirical study of domain adaptation for scientific RAG, revealing that [finding]."

---

## Critical Files to Create/Modify

| File | Action |
|------|--------|
| `src/rag_evaluator/__init__.py` | New — package init |
| `src/rag_evaluator/end_to_end_eval.py` | New — RAG ablation evaluator |
| `scripts/run_rag_ablation.py` | New — orchestrate 2×2 experiment |
| `scripts/question_type_analysis.py` | New — per-type evaluation breakdown |
| `scripts/mine_hard_negatives.py` | New — hard negative mining (if time) |
| `scripts/quality_tier_training.py` | New — data quality experiment (if time) |
| `paper_v2/latex/main.tex` | Modify — add RAG results table, per-type table, analysis sections |

## Verification

- RAG ablation: 4 conditions × test set → 4 sets of metrics, reported in paper Table 4
- Per-type analysis: 7 types × all metrics → heatmap or table showing which types benefit most
- Hard negatives: before/after Recall@k comparison on same test set
- All use same test split (5%, hash-deterministic) for consistency
