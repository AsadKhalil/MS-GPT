# RAG Ablation — Run Guide

Short command-first guide for running `scripts/run_rag_ablation.py` end-to-end on the server.
All commands assume `cd ~/MS-GPT && source .venv/bin/activate`.

The ablation is a 2×2: base vs fine-tuned embedding × base vs fine-tuned LLM.
For a paper/thesis result you need all four cells: `base_base, base_ft, ft_base, ft_ft`.

---

## 1. One-time server setup

```bash
cd ~/MS-GPT
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r config/requirements.txt    # or requirements.txt, whichever your repo has
# vLLM 0.6.6.post1 is pinned to transformers 4.46–4.x. Don't let pip pull 5.x —
# 5.x dropped `all_special_tokens_extended` and tokenizer init crashes.
pip install 'transformers>=4.46,<5' 'tokenizers>=0.20,<0.22' 'sentencepiece>=0.2'
pip install 'vllm==0.6.6.post1'           # only if using MODE=vllm

# Sanity
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(2))"
# expect: True NVIDIA GeForce RTX 4090

huggingface-cli whoami                    # needed for gated Llama-3.1
```

Pre-stage the HF models so the first ablation run doesn't sit on a 14 GB download:

```bash
huggingface-cli download intfloat/e5-base-v2
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
huggingface-cli download roberta-large    # for BERTScore
# Optional: the other 4 LLMs if you'll run the sweep
huggingface-cli download microsoft/Phi-3.5-mini-instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

---

## 2. Get `test.jsonl` onto the server

You have three options. Pick **one**.

### A. Copy from local Mac (simplest, 231 MB, matches the paper exactly)

```bash
# Run on local Mac:
ssh user@server 'mkdir -p ~/MS-GPT/paper_results/dataset/splits'
scp /Users/asad/Projects/Thesis/paper_results/dataset/splits/test.jsonl \
    user@server:~/MS-GPT/paper_results/dataset/splits/test.jsonl
```

### B. Regenerate on the server from `consolidated_qa.jsonl` (if you don't have local access)

```bash
cd ~/MS-GPT
source .venv/bin/activate
python -m src.dataset.split_generator \
    --input      data/consolidated_qa.jsonl \
    --output-dir paper_results/dataset/splits \
    --train-ratio 0.85 --val-ratio 0.10 --test-ratio 0.05
```

Document selection will match the paper (deterministic MD5 hash on `doc_id`). QA-pair count will be higher than the paper's 62,139 because records aren't quality-filtered — fine for running the ablation, slightly different absolute scores.

### C. Regenerate from `enriched.jsonl` (matches the paper exactly)

```bash
cd ~/MS-GPT
source .venv/bin/activate
python -m src.dataset.split_generator \
    --input      paper_results/dataset/enriched.jsonl \
    --output-dir paper_results/dataset/splits
```

---

## 3. Run RAG ablation

`run_rag_ablation.py` defaults already match the paper protocol (`--sample-size 1000`, all four conditions, stratified sampling, `--retrieval-k 10`, `--prompt-k 5`, `--temperature 0.0`, BERTScore + faithfulness ON). You only need to override the LLM backend and where outputs go.

### 3a. Smoke test (≈5 min on 4090) — does it run end-to-end?

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/run_rag_ablation.py \
  --load-4bit \
  --sample-size 20 \
  --conditions base_base \
  --no-bertscore --no-faithfulness \
  --output-dir paper_results/rag_ablation_smoke
```

Once this finishes without errors, you're cleared for the real runs.

### 3b. Paper-grade single-LLM run, local 4-bit (≈5–7 h on 4090)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/run_rag_ablation.py \
  --load-4bit \
  --output-dir paper_results/rag_ablation_mistral_n1000
```

Everything else is the script's default. Runs all four conditions on Mistral-7B with your FT artifacts.

### 3c. Paper-grade single-LLM run, vLLM (≈1–1.5 h on 4090)

Two terminals or tmux panes.

**Terminal 1 — start vLLM:**

```bash
cd ~/MS-GPT && source .venv/bin/activate
tmux new -s vllm

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --enable-lora \
  --lora-modules mistral-ft=./models/fine_tuned_llms/mistral_7b_v0.3/final_adapter \
  --max-lora-rank 64 \
  --port 9000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --hf-overrides '{"head_dim": 128}' \
  --trust-remote-code \
  2>&1 | tee logs/vllm_mistral.log
```

`--hf-overrides head_dim` is the workaround for transformers ≥4.46 exposing
`config.head_dim=None` to vLLM 0.6.6 (Mistral 128, Llama 128, Qwen 128, Phi-3.5 96).
bf16 (default dtype) is used instead of bitsandbytes 4-bit because vLLM 0.6.6's
on-the-fly bnb path is broken for merged-QKV GQA models — Mistral-7B in bf16
still fits in 24 GB at `--gpu-memory-utilization 0.85`.

Wait for `Uvicorn running on http://0.0.0.0:9000`, then verify the model is
serving (in another shell or after `Ctrl-b "`):

```bash
curl -s http://localhost:9000/v1/models | python -m json.tool
# Should list:
#   "id": "mistralai/Mistral-7B-Instruct-v0.3"   (base)
#   "id": "mistral-ft"                            (LoRA adapter)
```

Detach the vLLM tmux pane with `Ctrl-b d`.

**Terminal 2 — run the ablation:**

```bash
cd ~/MS-GPT && source .venv/bin/activate
tmux new -s rag

python scripts/run_rag_ablation.py \
  --llm-backend openai \
  --served-base-llm mistralai/Mistral-7B-Instruct-v0.3 \
  --served-ft-llm   mistral-ft \
  --openai-base-url http://localhost:9000/v1 \
  --openai-concurrency 8 \
  --embedding-device cpu \
  --output-dir paper_results/rag_ablation_mistral_n1000_vllm
# Ctrl-b d to detach
```

When the ablation finishes, stop vLLM:
```bash
tmux attach -t vllm   # Ctrl-c, then exit
```

### 3d. Full sweep — all 5 fine-tuned LLMs (vLLM ≈5–7 h, local ≈20–28 h)

```bash
cd ~/MS-GPT
tmux new -s rag-sweep

# Fast path (vLLM, recommended)
MODE=vllm ./scripts/run_rag_ablation_all_llms.sh

# Slow path (local 4-bit)
./scripts/run_rag_ablation_all_llms.sh
```

Per-LLM logs land in `logs/rag_ablation/`. The script auto-skips any LLM whose `final_adapter/` is missing and prints a cross-model summary table at the end.

Useful env overrides:

```bash
GPU_ID=0                                      # different GPU
SAMPLE_SIZE=200                               # smaller / quicker
VLLM_PORT=9000                                # vLLM port (default 9000)
VLLM_GPU_UTIL=0.75 EMBEDDING_DEVICE=cuda      # put E5 on GPU instead of CPU
VLLM_MAX_MODEL_LEN=2048                       # tighten if Llama-3.1-8B OOMs
```

Sanity-check that vLLM came up between models — while a sweep is running, the
script also exposes the OpenAI-compatible API at `http://localhost:$VLLM_PORT/v1`:

```bash
curl -s http://localhost:9000/v1/models | python -m json.tool
```

---

## 4. Output structure

Each run writes one directory under `--output-dir`:

```
<output-dir>/
  retrieval_base.jsonl                 # base E5 retrieval (15663 ctx, 1000 q)
  retrieval_ft.jsonl                   # FT E5 retrieval
  predictions_base_base.jsonl
  predictions_base_ft.jsonl            # uses your FT LoRA
  predictions_ft_base.jsonl            # uses your FT E5
  predictions_ft_ft.jsonl              # uses BOTH your FT pieces
  metrics_{base_base,base_ft,ft_base,ft_ft}.json
  rag_ablation_summary.json            # cross-condition summary
  rag_ablation_table_rows.tex          # drop directly into the thesis LaTeX
```

After a sweep, the cross-model table lives in `paper_results/rag_ablation_multi_<mode>_n1000/`.

---

## 5. Common errors

| Error | Fix |
|---|---|
| `FileNotFoundError: paper_results/dataset/splits/test.jsonl` | Step 2 didn't land. Verify `ls -lh paper_results/dataset/splits/test.jsonl`. |
| `No module named 'paper'` / `'sentence_transformers'` | Wrong python. Run `source .venv/bin/activate` first. |
| `ImportError: bitsandbytes requires CUDA, but CUDA is not available` | You're on Mac or the CPU build of bitsandbytes. Use the CUDA host. |
| `cannot import name 'DeepseekV3Config' from 'transformers'` | vLLM is newer than your transformers. `pip install 'transformers>=4.46,<5'` (keep the upper bound — 5.x breaks vLLM 0.6.6). |
| `LlamaTokenizer has no attribute all_special_tokens_extended` | You're on transformers 5.x. Same fix as above: pin to `<5`. |
| `TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` in `llama.py` | transformers ≥4.46 exposes `config.head_dim=None` for Mistral/Llama. Add `--hf-overrides '{"head_dim": 128}'` (96 for Phi-3.5-mini). |
| `AssertionError` in `vllm/.../linear.py:978` during weight load | vLLM 0.6.6's `--quantization bitsandbytes --load-format bitsandbytes` path is broken for GQA models. Drop both flags and let vLLM use bf16. |
| vLLM OOM at startup | Drop `--gpu-memory-utilization 0.75` and `--max-model-len 2048`. |
| `*_ft` cells error at load time | Missing FT artifact. Check `models/fine_tuned_embeddings_e5_base_v2/final_model/` and `models/fine_tuned_llms/<llm-run-name>/final_adapter/`. |

---

## 6. Monitoring while it runs

```bash
nvidia-smi -i 2 -l 5                 # live GPU 2 utilization
tail -f logs/rag_ablation/<name>.log
tail -f logs/vllm_mistral.log        # vLLM mode only
tmux attach -t rag-sweep             # rejoin the running sweep
```

To stop a sweep cleanly: `Ctrl-c` inside the tmux session — the script's trap kills vLLM and exits.
