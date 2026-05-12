1. Hard Negative Mining for Embeddings (HIGH IMPACT, MODERATE EFFORT)

  Your current setup uses MultipleNegativesRankingLoss with in-batch negatives only. These are "easy" negatives.
   The state-of-the-art uses mined hard negatives — passages that are semantically similar but not the correct
  answer.

  What to do:
  - Use your trained embedding model to retrieve top-50 passages per question
  - Select passages ranked 10-50 (close but wrong) as hard negatives
  - Retrain with TripletLoss or MultipleNegativesRankingLoss + hard negatives
  - This typically gives 5-15% additional Recall@k improvement

  This is a standard NeurIPS-level ablation and shows your pipeline can iteratively improve.

  ---
  2. ColBERT Late Interaction Retrieval (HIGH IMPACT, MODERATE EFFORT)

  Instead of single-vector dense retrieval, ColBERT uses per-token embeddings with a MaxSim operator. Research
  shows it significantly outperforms single-vector models, especially for out-of-domain generalization — perfect
   for specialized MS vocabulary.

  What to do:
  - Fine-tune ColBERTv2 on your MSQA-Bench training data
  - Compare: BM25 vs single-vector (your current) vs ColBERT
  - ColBERT models 3-5x smaller than dense encoders outperform them in domain-specific tasks
  - There's a dedicated ECIR 2026 workshop on this — very timely

  ---
  3. DPO for Hallucination Reduction (HIGH IMPACT, HIGH EFFORT)

  After your SFT (QLoRA) stage, add a Direct Preference Optimization (DPO) stage. Recent work on F-DPO
  (Factuality-aware DPO) shows consistent hallucination reduction across 1B-14B models.

  What to do:
  - Generate multiple answers per question using your fine-tuned LLM
  - Score them for faithfulness against the context (use your existing faithfulness metrics)
  - Create preference pairs: faithful answer = chosen, hallucinated answer = rejected
  - Run DPO training on top of your QLoRA adapter
  - Compare: Base → QLoRA → QLoRA+DPO (a 3-stage pipeline is a strong contribution)

  This directly addresses the faithfulness angle of your paper and is a novel contribution for domain-specific
  scientific QA.

  ---
  4. GraphRAG with MS Knowledge Graph (HIGH IMPACT, HIGH EFFORT)

  Build a knowledge graph from your 40K papers capturing MS-specific relationships (instruments → techniques →
  analytes → matrices), then use GraphRAG for retrieval.

  What to do:
  - Extract entities/relations from your corpus using an LLM (instruments, methods, compounds, etc.)
  - Build a domain knowledge graph
  - Implement dual-channel retrieval: dense passage retrieval + graph traversal
  - Compare: Standard RAG vs GraphRAG
  - This was accepted at ICLR 2026 and is very hot right now

  ---
  5. Model Merging with DARE-TIES (MODERATE IMPACT, LOW EFFORT)

  You have 5 fine-tuned LLMs on the same domain data. Instead of picking the best one, merge them using
  DARE-TIES to create a single superior model.

  What to do:
  - Use mergekit to merge your 5 QLoRA adapters
  - DARE prunes 90% of redundant delta parameters, TIES resolves sign conflicts
  - The merged model often outperforms any individual model
  - This is a nearly free experiment — just run mergekit on your saved adapters
  - Compare: best single model vs DARE-TIES merge vs simple average merge

  ---
  6. Multi-Hop QA Subset (MODERATE IMPACT, MODERATE EFFORT)

  Add a multi-hop reasoning subset to MSQA-Bench where answering requires combining information from 2-3
  passages (e.g., "What ionization method was used in the study that achieved the lowest detection limit for
  pesticides?").

  What to do:
  - Use your LLM to generate multi-hop questions that chain facts across passages from the same or related
  papers
  - Create a curated set of 500-1000 multi-hop questions
  - Evaluate: single-hop vs multi-hop performance gap
  - This differentiates MSQA-Bench from simpler QA benchmarks and aligns with NeurIPS 2025 trends (GRADE, MINTQA
   benchmarks)

  ---
  7. Adaptive Query Routing (MODERATE IMPACT, LOW EFFORT)

  Implement a query classifier that routes questions to different retrieval/generation strategies based on
  complexity — similar to the R2RAG system that won at NeurIPS 2025 MMU-RAG Competition.

  What to do:
  - Classify queries as simple (factual/definition) vs complex (comparison/causal/multi-hop)
  - Simple → BM25 + small LLM (fast, cheap)
  - Complex → Dense retrieval + large LLM (accurate)
  - Show efficiency-accuracy tradeoffs
  - This is a practical systems contribution reviewers love

  ---
  My Recommendation for Maximum Paper Impact

  If I had to pick 3 experiments to make this a strong NeurIPS D&B paper:

  ┌──────────┬──────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │ Priority │        Technique         │                                Why                                │
  ├──────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 1        │ Hard Negative Mining     │ Easy win, expected by reviewers, big retrieval boost              │
  ├──────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 2        │ DPO for faithfulness     │ Novel contribution for scientific QA, directly addresses          │
  │          │                          │ hallucination                                                     │
  ├──────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 3        │ Model Merging            │ Nearly free experiment, surprising results, good ablation         │
  │          │ (DARE-TIES)              │                                                                   │
  └──────────┴──────────────────────────┴───────────────────────────────────────────────────────────────────┘

  These three together tell a compelling story: better retrieval (hard negatives) → better generation (QLoRA) →
  less hallucination (DPO) → and you can even combine your expert models (merging).

  Want me to implement any of these? I can write the code for hard negative mining, DPO training, or model
  merging.

  ---
  Sources:
  - HNCSE: Contrastive Learning with Hard Negatives for Sentence Embeddings
  - F-DPO: Factuality-Aware Preference Learning
  - LIR: Late Interaction and Multi Vector Retrieval Workshop @ ECIR 2026
  - GraphRAG Survey - ACM
  - Model Merging Survey - ACM Computing Surveys
  - DARE-TIES Model Merging with mergekit
  - R2RAG at NeurIPS 2025 MMU-RAG Competition
  - GRADE: Multi-hop QA Generation
  - Fine-tuning LLMs: Exploration of Training Strategies and Model Merging
  - Domain-Specific LLM Fine-Tuning Guide 2026

✻ Brewed for 1m 28s
 MISSING for a competitive NeurIPS D&B submission:
──────────────────────────────────────────────────────────────────────────────────── msqa-bench-neurips-paper ──
❯    │  #  │        Missing Item         │ Criticality │            Status             │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │     │ Evaluation results tables   │             │                               │
  │ 1   │ (actual numbers for all     │ BLOCKING    │ Need scripts                  │
  │     │ models)                     │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │     │ Base vs fine-tuned          │             │                               │
  │ 2   │ comparison (before/after    │ BLOCKING    │ Need scripts                  │
  │     │ fine-tuning)                │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │ 3   │ Human evaluation /          │ HIGH        │ gold_set_sample.csv exists    │
  │     │ annotation agreement        │             │ but no annotation results     │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │ 4   │ Datasheet for Datasets      │ REQUIRED    │ Missing                       │
  │     │ (NeurIPS D&B requirement)   │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │ 5   │ Hosting, licensing,         │ REQUIRED    │ Missing                       │
  │     │ maintenance plan            │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │     │ Comparison with existing    │             │                               │
  │ 6   │ benchmarks (PubMedQA,       │ HIGH        │ Missing                       │
  │     │ BioASQ scores)              │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │ 7   │ Error analysis / failure    │ MEDIUM      │ Missing                       │
  │     │ modes                       │             │                               │
  ├─────┼─────────────────────────────┼─────────────┼───────────────────────────────┤
  │ 8   │ Broader impact +            │ REQUIRED    │ Missing                       │
  │     │ limitations                 │             │                               │
  └─────┴─────────────────────────────┴─────────────┴───────────────────────────────┘







`````
# Plan: Complete NeurIPS 2026 D&B Submission

## Context

The paper `paper_v2/latex/main.tex` is structurally complete (NeurIPS format, sections, TikZ diagram, Datasheet appendix, 37 verified references) but has ~69 `\placeholder{}` markers in results tables and 3 `\todo{}` blocks for analysis sections. All model weights live on the AIC-lab server (`/home/asad/`), not locally. The gold set (200 samples) is unannotated. No cross-benchmark comparison exists. The goal: fill every gap so the paper is submission-ready.

---

## Phase 1: Server-Side Model Training & Evaluation (BLOCKING)

**Location:** AIC-lab server at `/home/asad/MS-GPT/`

### 1A. Retrain Failed LLMs
Run on server:
```bash
cd /home/asad/MS-GPT && source .venv/bin/activate

# Fix Llama (HF token issue)
huggingface-cli login   # paste token
python scripts/train_all_llms.py --model llama3.1_8b -y

# Phi-3.5 (OOM at 50%) - reduce batch or enable deeper gradient checkpointing
# Edit config/llm_finetuner.json: set phi3.5 batch_size=2, max_seq_length=1536
python scripts/train_all_llms.py --model phi3.5_mini -y

# Verify Qwen2.5 and DeepSeek status
ls -la /home/asad/models/fine_tuned_llms/qwen2.5_7b/final_adapter/
ls -la /home/asad/models/fine_tuned_llms/deepseek_r1_distill_7b/final_adapter/
# Retrain any that didn't complete
```

**Success criteria:** All 5 LLMs have `final_adapter/` directories with adapter weights.

### 1B. Run Full Evaluation Pipeline
```bash
python scripts/extract_all_results.py --all
```
**Script:** `scripts/extract_all_results.py` (already created)
- Evaluates 5 embedding models (base vs fine-tuned): Recall@1/5/10, MRR@10, NDCG@10, MAP@10
- Evaluates 5 LLMs: ROUGE-1/2/L, BERTScore-F1, Token-F1, Faithfulness, Perplexity
- Outputs to `paper_results/model_results/`:
  - `embedding_results.json`, `llm_results.json`
  - `table_embedding_results.tex`, `table_llm_results.tex`

### 1C. Run Retrieval Baselines (BM25)
```bash
python paper/evaluation/retrieval_baselines.py \
  --config config/benchmark_config.json --compare
```
**Script:** `paper/evaluation/retrieval_baselines.py` (already exists, 702 lines)
- BM25 + all-MiniLM-L6-v2 baselines
- Outputs to `paper_results/evaluation/retrieval_results.json`

**Phase 1 deliverables:** All evaluation JSON files + auto-generated LaTeX tables in `paper_results/`.

---

## Phase 2: Fill Paper Placeholders (LOCAL)

**File:** `paper_v2/latex/main.tex`

### 2A. Fill Results Tables (~69 placeholders)
Once Phase 1 outputs are available (scp from server), replace:
- **Table 2** (line ~385-394): Embedding retrieval results — 6 rows × 6 metrics (base/fine-tuned columns)
- **Table 3** (line ~412-419): LLM generation results — 5 rows × 6 metrics
- **Abstract** (lines 61-62): Top-line improvement numbers (Recall@10 gain %, ROUGE-L gain %, faithfulness gain %)

### 2B. Fill TODO Blocks (3 blocks)
- **Line 426**: Base vs fine-tuned comparison analysis — narrative interpreting embedding improvement patterns
- **Line 431**: Per-question-type performance breakdown — add a table or figure showing metric variation across 7 question types (Factual, Definition, Method, Causal, Comparison, Numeric, Unknown)
- **Line 436**: Open-book vs closed-book comparison — compare LLM performance with/without retrieved context to quantify RAG benefit

### 2C. Write Missing Analysis
Create a new script `scripts/generate_analysis.py` to:
1. **Per-question-type eval**: Filter test set by question_type, run LLM eval per type, output breakdown table
2. **Open-book vs closed-book**: Run LLM inference with context=empty string vs with context, compare ROUGE/BERTScore/Faithfulness
3. **Error analysis**: Sample 50 worst-performing examples (lowest Token-F1), categorize failure modes

---

## Phase 3: Human Evaluation / Annotation Agreement

**File:** `paper_results/annotation/gold_set_sample.csv` (200 samples, all annotation columns empty)

### 3A. Annotation Protocol
Create `paper_results/annotation/annotation_guidelines.md`:
- Binary correctness (answer_correct: 0/1)
- Evidence support (evidence_support: 0/1 — is answer grounded in context?)
- Evidence quality (evidence_quality: 1-5 Likert)
- Question clarity (question_clarity: 1-5 Likert)

### 3B. Annotate
- **Minimum**: 2 annotators on all 200 samples (you + supervisor or a colleague)
- **Metric**: Report Cohen's kappa or Fleiss' kappa for inter-annotator agreement
- **Target**: kappa >= 0.6 (substantial agreement)

### 3C. Report in Paper
Add annotation agreement results to Section 4.1 (Dataset Quality) or a new subsection. Report:
- Agreement scores per dimension
- Dataset quality summary (% correct, % evidence-supported)

---

## Phase 4: Cross-Benchmark Comparison

### 4A. Create Comparison Script
Create `scripts/cross_benchmark_comparison.py`:
- Pull published scores for PubMedQA, BioASQ, COVID-QA from papers
- Run MSQA-Bench fine-tuned models on PubMedQA test set (if accessible via HuggingFace datasets)
- Compare domain-adapted vs general models on biomedical QA

### 4B. Add to Paper
Add Table 4 or extend Discussion section:
| Benchmark | Domain | Size | Best Published | Our Model |
|-----------|--------|------|---------------|-----------|
| PubMedQA | Biomedical | 1K | ~82% acc | -- |
| BioASQ | Biomedical | ~5K | varies | -- |
| MSQA-Bench | Mass Spec | 1.2M | N/A (new) | XX |

**Note:** If running on external benchmarks is infeasible before deadline, a literature-based comparison table is acceptable for D&B track (dataset contribution is primary).

---

## Phase 5: Error Analysis & Failure Modes

### 5A. Generate Error Examples
From Phase 2C script output, select 3-5 representative failure categories:
1. **Out-of-context answers** — model hallucinates beyond provided passage
2. **Numerical/formula errors** — model garbles m/z values, retention times
3. **Ambiguous questions** — low-clarity questions yield poor answers
4. **Long-context truncation** — answer information beyond max_seq_length

### 5B. Add to Paper
Add a subsection under Discussion (Section 6) with:
- Table of failure categories + frequency in test set
- 2-3 qualitative examples (question/gold/predicted)

---

## Phase 6: NeurIPS D&B Compliance Checklist

### Already in paper (verify completeness):
- [x] Datasheet for Datasets (Appendix A — check all 12 sections per Gebru et al.)
- [x] Broader Impact (Section 6.2 — verify it covers dual use, bias, environmental cost)

### Must add/complete:
- [ ] **Hosting plan**: State where MSQA-Bench will be hosted (HuggingFace Datasets recommended), under what license (CC BY-SA 4.0 suggested), and long-term maintenance commitment
- [ ] **Author statement**: Add "Author Statement of Responsibility" confirming ethical data collection, consent (N/A for published papers), and compliance with terms of use
- [ ] **URL for dataset**: Create HuggingFace dataset repo (even if private/placeholder initially), include URL in paper
- [ ] **Persistent DOI**: Consider Zenodo backup for archival DOI

### Verify Datasheet completeness (Appendix A):
- Motivation, Composition, Collection process, Preprocessing/cleaning
- Uses, Distribution, Maintenance — all 7 sections must be present

---

## Phase 7: Compile & Final Checks (LOCAL)

### 7A. Install LaTeX
```bash
sudo apt install texlive-full  # or texlive-latex-extra texlive-fonts-recommended
```

### 7B. Compile Paper
```bash
cd paper_v2/latex && make clean && make
```
Fix any LaTeX errors. Verify:
- All tables render correctly (no `\placeholder` or `\todo` remains)
- TikZ diagram compiles
- References resolve (no `??` in PDF)
- Page limit: NeurIPS D&B allows 9 pages + unlimited appendix

### 7C. Final Reference Verification
```bash
python scripts/verify_references.py paper_v2/latex/references.bib
```
Fix any FAIL entries. Ensure all 37+ references pass.

### 7D. Pre-Submission Review
- Grep for remaining placeholders: `grep -c 'placeholder\|todo' main.tex` should be 0
- Check NeurIPS formatting: margins, font size, anonymization (if required for review)
- Proofread abstract and conclusion

---

## Execution Order & Dependencies

```
Phase 1A (retrain LLMs) ──────────────┐
Phase 1C (BM25 baselines) ────────────┤
Phase 3A-3B (human annotation) ───────┤── all on server / parallel
Phase 4A (cross-benchmark script) ────┘
          │
          v
Phase 1B (extract_all_results.py) ── needs 1A complete
          │
          v
Phase 2 (fill placeholders) ── needs 1B + 1C outputs
Phase 5 (error analysis) ── needs 1B outputs
Phase 3C (report annotation) ── needs 3B complete
Phase 4B (comparison table) ── needs 4A outputs
          │
          v
Phase 6 (D&B compliance) ── can start anytime, finish after Phase 2
          │
          v
Phase 7 (compile & verify) ── final step, needs everything above
```

## Critical Path

**Longest chain:** Retrain LLMs (1A, ~24-48h GPU time) -> Extract results (1B, ~4-8h) -> Fill tables (2A, ~1h) -> Compile (7B)

**Parallel work while LLMs train:** Human annotation (Phase 3), cross-benchmark literature review (Phase 4), D&B compliance writing (Phase 6), error analysis script (Phase 2C)

---

## Verification

1. `grep -cP 'placeholder|\\\\todo' paper_v2/latex/main.tex` returns 0
2. `make` in `paper_v2/latex/` produces `main.pdf` with no errors
3. `python scripts/verify_references.py paper_v2/latex/references.bib` shows all PASS
4. `paper_results/model_results/` contains `embedding_results.json` + `llm_results.json`
5. `paper_results/annotation/gold_set_sample.csv` has non-empty annotation columns
6. PDF page count: <= 9 main + appendix
```