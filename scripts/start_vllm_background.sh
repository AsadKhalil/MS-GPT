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
GPU="${1:-2}"  # Default to GPU 2 (3090 Ti) - supports AWQ
MODEL_SIZE="${2:-7b}"
PORT="${3:-8000}"

# GPU Compute Capabilities:
#   GPU 0 (1080 Ti): 6.1 - NO AWQ support (use FP16 models only)
#   GPU 1 (2080 Ti): 7.5 - AWQ supported
#   GPU 2 (3090 Ti): 8.6 - AWQ supported (RECOMMENDED)

# Map model size to full name
# AWQ models need GPU 1 or 2. FP16 models work on any GPU.
case "$MODEL_SIZE" in
    3b|3B)
        # Small FP16 model - works on any GPU including 1080 Ti
        MODEL="Qwen/Qwen2.5-3B-Instruct"
        ;;
    7b|7B)
        if [ "$GPU" = "0" ]; then
            # GPU 0 doesn't support AWQ, use FP16
            MODEL="Qwen/Qwen2.5-7B-Instruct"
            echo "⚠️  GPU 0 doesn't support AWQ, using FP16 model"
        else
            MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
        fi
        ;;
    14b|14B)
        MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
        if [ "$GPU" = "0" ]; then
            echo "❌ GPU 0 doesn't support AWQ. Use GPU 1 or 2."
            exit 1
        fi
        ;;
    32b|32B)
        MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
        if [ "$GPU" = "0" ]; then
            echo "❌ GPU 0 doesn't support AWQ. Use GPU 1 or 2."
            exit 1
        fi
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
