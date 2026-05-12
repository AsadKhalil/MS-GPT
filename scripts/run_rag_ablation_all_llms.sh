#!/usr/bin/env bash
#
# Sweep run_rag_ablation.py across every fine-tuned LLM in config/llm_finetuner.json,
# sequentially on a single GPU.
#
# Each model runs the full 2x2 ablation (base_base, base_ft, ft_base, ft_ft) at
# sample_size=1000. Two backends:
#
#   MODE=local  (default): transformers + bitsandbytes 4-bit. ~20-28 h total on 4090.
#   MODE=vllm           : vLLM serves base (bf16) + LoRA per LLM. ~5-7 h total on 4090.
#                         bf16 (not bnb) — vLLM 0.6.6's on-the-fly bnb path is broken
#                         for merged-QKV GQA models (shape assert in linear.py).
#
# Models whose final_adapter directory is missing are skipped (not failed),
# so a partial run is safe to restart.
#
# Usage:
#   ./scripts/run_rag_ablation_all_llms.sh                         # local 4-bit
#   MODE=vllm ./scripts/run_rag_ablation_all_llms.sh               # vLLM (faster)
#   GPU_ID=0 ./scripts/run_rag_ablation_all_llms.sh                # different GPU
#   SAMPLE_SIZE=200 ./scripts/run_rag_ablation_all_llms.sh         # smaller / quicker
#
# Run under tmux so an SSH disconnect doesn't kill the sweep:
#   tmux new -s rag-sweep
#   MODE=vllm ./scripts/run_rag_ablation_all_llms.sh
#   # detach: Ctrl-b d   reattach: tmux attach -t rag-sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODE="${MODE:-local}"                      # local | vllm
GPU_ID="${GPU_ID:-2}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
MODEL_ROOT="${MODEL_ROOT:-./models}"
OUT_BASE="${OUT_BASE:-paper_results/rag_ablation_multi_${MODE}_n${SAMPLE_SIZE}}"
LOG_DIR="${LOG_DIR:-logs/rag_ablation}"

# vLLM-only knobs
VLLM_PORT="${VLLM_PORT:-9000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-64}"
VLLM_READY_TRIES="${VLLM_READY_TRIES:-120}"   # 120 * 5s = 10 min max wait
OPENAI_CONCURRENCY="${OPENAI_CONCURRENCY:-8}"
# In vLLM mode the GPU is owned by vLLM; default the embedding to CPU to avoid OOM.
# Override with EMBEDDING_DEVICE=cuda if you've lowered VLLM_GPU_UTIL enough to share.
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv python not found at $PYTHON" >&2
    exit 1
fi

if [ "$MODE" = "vllm" ]; then
    if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
        echo "ERROR: vLLM not installed in $PYTHON. Run: $PYTHON -m pip install vllm" >&2
        exit 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: curl is required for vLLM readiness probes" >&2
        exit 1
    fi
fi

mkdir -p "$OUT_BASE" "$LOG_DIR"

# (run_name, HuggingFace base model id, head_dim) — must match config/llm_finetuner.json
# head_dim = hidden_size / num_attention_heads. Required because transformers >=4.46
# exposes config.head_dim = None for Mistral/Llama, which vLLM 0.6.6 then multiplies
# by num_heads and crashes (`int * None`). Injected via --hf-overrides.
LLMS=(
    "mistral_7b_v0.3        mistralai/Mistral-7B-Instruct-v0.3       128"
    "phi3.5_mini            microsoft/Phi-3.5-mini-instruct           96"
    "qwen2.5_7b             Qwen/Qwen2.5-7B-Instruct                 128"
    "deepseek_r1_distill_7b deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  128"
    "llama3.1_8b            meta-llama/Llama-3.1-8B-Instruct         128"
)

echo "============================================================"
echo "RAG ablation sweep"
echo "  MODE             : $MODE"
echo "  GPU              : $GPU_ID (CUDA_DEVICE_ORDER=PCI_BUS_ID)"
echo "  sample_size      : $SAMPLE_SIZE"
echo "  model_root       : $MODEL_ROOT"
echo "  out_base         : $OUT_BASE"
echo "  log_dir          : $LOG_DIR"
echo "  python           : $PYTHON"
if [ "$MODE" = "vllm" ]; then
    echo "  vllm port        : $VLLM_PORT"
    echo "  vllm gpu_util    : $VLLM_GPU_UTIL"
    echo "  vllm max_model_len: $VLLM_MAX_MODEL_LEN"
    echo "  openai concurrency: $OPENAI_CONCURRENCY"
    echo "  embedding device : $EMBEDDING_DEVICE"
fi
echo "============================================================"
nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.free,memory.total --format=csv 2>/dev/null || true
echo

VLLM_PID=""
cleanup_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> stopping vLLM (PID $VLLM_PID)"
        kill -INT "$VLLM_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 5
        done
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # Let GPU memory settle before the next iteration
    sleep 3
}
trap cleanup_vllm EXIT INT TERM

start_vllm() {
    local base_llm="$1"
    local adapter="$2"
    local run_name="$3"
    local head_dim="$4"
    local logfile="$LOG_DIR/vllm_${run_name}.log"

    echo ">>> starting vLLM for $base_llm (LoRA: ${run_name}-ft, head_dim=$head_dim)"
    echo "    log: $logfile"

    # vLLM inherits CUDA_VISIBLE_DEVICES from the script env.
    # bf16 (default dtype) — vLLM 0.6.6's bitsandbytes on-the-fly path is broken
    # for merged-QKV GQA models. 7-8B in bf16 fits in 24 GB at gpu_util=0.85.
    # --hf-overrides head_dim works around transformers >=4.46 exposing
    # config.head_dim=None on Mistral/Llama configs.
    vllm serve "$base_llm" \
        --enable-lora \
        --lora-modules "${run_name}-ft=${adapter}" \
        --max-loras 1 --max-lora-rank "$VLLM_MAX_LORA_RANK" \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --hf-overrides "{\"head_dim\": $head_dim}" \
        --trust-remote-code \
        > "$logfile" 2>&1 &
    VLLM_PID=$!

    local tries="$VLLM_READY_TRIES"
    while [ "$tries" -gt 0 ]; do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo ">>> vLLM ready (PID $VLLM_PID)"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "!!! vLLM died during startup. Tail of $logfile:"
            tail -30 "$logfile" || true
            return 1
        fi
        sleep 5
        tries=$((tries - 1))
    done
    echo "!!! vLLM did not become ready within 10 min. Tail of $logfile:"
    tail -30 "$logfile" || true
    return 1
}

TOTAL_START=$(date +%s)
DONE=0
SKIPPED=0
FAILED=0

for entry in "${LLMS[@]}"; do
    read -r RUN_NAME BASE_LLM HEAD_DIM <<< "$entry"
    ADAPTER="$MODEL_ROOT/fine_tuned_llms/$RUN_NAME/final_adapter"
    OUT_DIR="$OUT_BASE/$RUN_NAME"
    LOG_FILE="$LOG_DIR/${RUN_NAME}_${MODE}.log"

    if [ ! -f "$ADAPTER/adapter_config.json" ]; then
        echo ">>> SKIP $RUN_NAME — adapter missing at $ADAPTER"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo
    echo "============================================================"
    echo ">>> $(date '+%F %T') START $RUN_NAME ($MODE mode)"
    echo "    base_llm   : $BASE_LLM"
    echo "    adapter    : $ADAPTER"
    echo "    out_dir    : $OUT_DIR"
    echo "    log        : $LOG_FILE"
    echo "============================================================"

    START=$(date +%s)
    RC=0

    if [ "$MODE" = "vllm" ]; then
        if ! start_vllm "$BASE_LLM" "$ADAPTER" "$RUN_NAME" "$HEAD_DIM"; then
            FAILED=$((FAILED + 1))
            cleanup_vllm
            continue
        fi
        set +e
        "$PYTHON" -u scripts/run_rag_ablation.py \
            --base-llm "$BASE_LLM" \
            --llm-run-name "$RUN_NAME" \
            --model-root "$MODEL_ROOT" \
            --sample-size "$SAMPLE_SIZE" \
            --llm-backend openai \
            --served-base-llm "$BASE_LLM" \
            --served-ft-llm "${RUN_NAME}-ft" \
            --openai-base-url "http://localhost:${VLLM_PORT}/v1" \
            --openai-concurrency "$OPENAI_CONCURRENCY" \
            --embedding-device "$EMBEDDING_DEVICE" \
            --output-dir "$OUT_DIR" \
            2>&1 | tee "$LOG_FILE"
        RC=${PIPESTATUS[0]}
        set -e
        cleanup_vllm
    else
        # local 4-bit
        set +e
        "$PYTHON" -u scripts/run_rag_ablation.py \
            --base-llm "$BASE_LLM" \
            --llm-run-name "$RUN_NAME" \
            --model-root "$MODEL_ROOT" \
            --sample-size "$SAMPLE_SIZE" \
            --load-4bit \
            --output-dir "$OUT_DIR" \
            2>&1 | tee "$LOG_FILE"
        RC=${PIPESTATUS[0]}
        set -e
    fi

    END=$(date +%s)
    if [ "$RC" -eq 0 ]; then
        echo ">>> $(date '+%F %T') DONE $RUN_NAME in $((END - START)) s"
        DONE=$((DONE + 1))
    else
        echo ">>> $(date '+%F %T') FAIL $RUN_NAME (rc=$RC) after $((END - START)) s — see $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

TOTAL_END=$(date +%s)
echo
echo "============================================================"
echo "Sweep complete: done=$DONE skipped=$SKIPPED failed=$FAILED"
echo "Total wall-clock: $((TOTAL_END - TOTAL_START)) s"
echo "Results under   : $OUT_BASE/"
echo "============================================================"

"$PYTHON" - <<PY
import json, pathlib
base = pathlib.Path("$OUT_BASE")
print()
print(f"{'model':<25} {'condition':<10} {'R@5':>7} {'BERT':>7} {'Faith':>7} {'AQS':>7}")
print("-" * 72)
for sub in sorted(base.iterdir()):
    if not sub.is_dir():
        continue
    summary = sub / "rag_ablation_summary.json"
    if not summary.exists():
        print(f"{sub.name:<25} (no summary.json)")
        continue
    data = json.loads(summary.read_text())
    for r in data["results"]:
        print(f"{sub.name:<25} {r['condition']:<10} "
              f"{r['recall_at_5']:>7.3f} {r['bertscore_f1']:>7.3f} "
              f"{r['faithfulness_score']:>7.3f} {r['answer_quality_score']:>7.3f}")
PY
