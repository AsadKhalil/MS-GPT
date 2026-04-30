#!/usr/bin/env bash
# Run the missing experiments to close C1 (RAG eval) and C2 (zero-shot baseline).
# Each block is independently runnable; comment out what you don't need.
#
# Hardware: 1x 24GB GPU. Set CUDA_VISIBLE_DEVICES if you have multiple.
# Time budget: ~30 min per (model, base/FT, mode) combo at sample_size=200.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SAMPLE=200
TEST_JSONL=paper_results/dataset/splits/test.jsonl
OUT_DIR=paper_results/model_results/llms

# (base_model, adapter_path, short_name)
MODELS=(
  "microsoft/Phi-3.5-mini-instruct|models/fine_tuned_llms/phi3.5_mini/final_adapter|phi3.5_mini"
  "mistralai/Mistral-7B-Instruct-v0.3|models/fine_tuned_llms/mistral_7b_v0.3/final_adapter|mistral_7b_v0.3"
  "Qwen/Qwen2.5-7B-Instruct|models/fine_tuned_llms/qwen2.5_7b/final_adapter|qwen2.5_7b"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|models/fine_tuned_llms/deepseek_r1_distill_7b/final_adapter|deepseek_r1_distill_7b"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r BASE ADAPTER NAME <<< "$entry"
  MODEL_OUT="$OUT_DIR/$NAME"
  mkdir -p "$MODEL_OUT"

  echo "==> [$NAME] zero-shot, closed-book"
  $PY scripts/run_rag_zeroshot_eval.py \
    --base "$BASE" --mode closed --sample-size $SAMPLE \
    --jsonl "$TEST_JSONL" --load-4bit \
    --output "$MODEL_OUT/eval_zeroshot_closed.json"

  echo "==> [$NAME] zero-shot, open-book (RAG)"
  $PY scripts/run_rag_zeroshot_eval.py \
    --base "$BASE" --mode open --sample-size $SAMPLE \
    --jsonl "$TEST_JSONL" --load-4bit \
    --output "$MODEL_OUT/eval_zeroshot_open.json"

  echo "==> [$NAME] fine-tuned, closed-book"
  $PY scripts/run_rag_zeroshot_eval.py \
    --base "$BASE" --adapter "$ADAPTER" --mode closed --sample-size $SAMPLE \
    --jsonl "$TEST_JSONL" --load-4bit \
    --output "$MODEL_OUT/eval_ft_closed.json"

  echo "==> [$NAME] fine-tuned, open-book (matches paper Table 5)"
  $PY scripts/run_rag_zeroshot_eval.py \
    --base "$BASE" --adapter "$ADAPTER" --mode open --sample-size $SAMPLE \
    --jsonl "$TEST_JSONL" --load-4bit \
    --output "$MODEL_OUT/eval_ft_open.json"
done

echo
echo "Done. New eval JSONs are under $OUT_DIR/<model>/eval_{zeroshot,ft}_{closed,open}.json"
echo "These give you, per model:"
echo "  - Zero-shot closed-book   (C2)"
echo "  - Zero-shot open-book     (C1 + C2 together)"
echo "  - Fine-tuned closed-book  (C1: ablates RAG vs adaptation)"
echo "  - Fine-tuned open-book    (= the paper's current Table 5 numbers)"
