#!/bin/bash
# =============================================================================
# Start vLLM Server in Background
# =============================================================================
# Usage:
#   ./scripts/start_vllm_background.sh              # Default: GPU 0, 7B model
#   ./scripts/start_vllm_background.sh 2            # Use GPU 2
#   ./scripts/start_vllm_background.sh 2 32b        # GPU 2 with 32B model
#   ./scripts/start_vllm_background.sh 0 7b 8001    # GPU 0, 7B model, port 8001
# =============================================================================

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Parse arguments
GPU="${1:-0}"
MODEL_SIZE="${2:-7b}"
PORT="${3:-8000}"

# Map model size to full name
case "$MODEL_SIZE" in
    7b|7B)
        MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
        ;;
    14b|14B)
        MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
        ;;
    32b|32B)
        MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
        ;;
    *)
        MODEL="$MODEL_SIZE"  # Allow full model name
        ;;
esac

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Activated .venv"
fi

mkdir -p logs

PID_FILE="logs/vllm_gpu${GPU}.pid"
LOG_FILE="logs/vllm_gpu${GPU}.log"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  vLLM already running on GPU $GPU (PID: $OLD_PID)"
        echo "   Stop it first: kill $OLD_PID"
        exit 1
    fi
fi

echo "=============================================="
echo "🚀 Starting vLLM Server (Background)"
echo "=============================================="
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo "PID: $PID_FILE"
echo "=============================================="

# Start in background
nohup bash -c "
    cd '$PROJECT_DIR'
    source .venv/bin/activate 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
        --model '$MODEL' \
        --host 0.0.0.0 \
        --port $PORT \
        --max-model-len 4096 \
        --dtype auto \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
        --disable-log-requests
" > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo ""
echo "✅ vLLM server starting in background"
echo "   PID: $PID"
echo ""
echo "Wait ~30-60 seconds for model to load, then check:"
echo "   curl http://localhost:$PORT/v1/models"
echo ""
echo "Monitor: tail -f $LOG_FILE"
echo "Stop:    kill \$(cat $PID_FILE)"
