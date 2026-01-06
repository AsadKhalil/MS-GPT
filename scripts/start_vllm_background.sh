#!/bin/bash
# =============================================================================
# Start vLLM Server in Background
# =============================================================================
# Usage:
#   ./scripts/start_vllm_background.sh              # Default: GPU 2, 7B model
#   ./scripts/start_vllm_background.sh 2            # Use GPU 2
#   ./scripts/start_vllm_background.sh 2 14b        # GPU 2 with 14B model (RECOMMENDED)
#   ./scripts/start_vllm_background.sh 2 32b        # GPU 2 with 32B model
#   ./scripts/start_vllm_background.sh 0 7b 8001    # GPU 0, 7B model, port 8001
#
# Examples:
#   # Run Qwen 14B on GPU 2 (recommended for QA generation)
#   ./scripts/start_vllm_background.sh 2 14b
#
#   # Run Qwen 14B on GPU 2 with custom port
#   ./scripts/start_vllm_background.sh 2 14b 8000
#
# Stop/Kill Running vLLM:
#   # Method 1: Using saved PID file
#   kill $(cat logs/vllm_gpu2.pid)
#
#   # Method 2: Kill all vLLM processes
#   pkill -f "vllm.entrypoints"
#
#   # Method 3: Find and kill manually
#   ps aux | grep vllm
#   kill <PID>
#
# Check if vLLM is running:
#   curl http://localhost:8000/v1/models
#   # or
#   ps aux | grep vllm
#
# Force restart (kills existing vLLM first):
#   ./scripts/start_vllm_background.sh 2 14b 8000 --force
# =============================================================================

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Check for --force flag
FORCE_RESTART=false
for arg in "$@"; do
    if [ "$arg" = "--force" ] || [ "$arg" = "-f" ]; then
        FORCE_RESTART=true
    fi
done

# Parse arguments (filter out --force)
args=()
for arg in "$@"; do
    if [ "$arg" != "--force" ] && [ "$arg" != "-f" ]; then
        args+=("$arg")
    fi
done

GPU="${args[0]:-2}"  # Default to GPU 2 (3090 Ti) - supports AWQ
MODEL_SIZE="${args[1]:-7b}"
PORT="${args[2]:-8000}"

# If --force, kill any existing vLLM processes first
if [ "$FORCE_RESTART" = true ]; then
    echo "🔄 Force restart requested - killing existing vLLM processes..."
    
    # Kill by PID file if exists
    if [ -f "logs/vllm_gpu${GPU}.pid" ]; then
        OLD_PID=$(cat "logs/vllm_gpu${GPU}.pid")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "   Killing vLLM PID $OLD_PID..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
        fi
        rm -f "logs/vllm_gpu${GPU}.pid"
    fi
    
    # Also try to kill any vLLM on the same port
    VLLM_PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$VLLM_PIDS" ]; then
        echo "   Killing processes on port $PORT: $VLLM_PIDS"
        echo "$VLLM_PIDS" | xargs kill 2>/dev/null || true
        sleep 2
    fi
    
    echo "✓ Cleanup complete"
fi

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
        echo ""
        echo "   Options:"
        echo "   1. Stop manually:  kill $OLD_PID"
        echo "   2. Force restart:  $0 $GPU $MODEL_SIZE $PORT --force"
        echo "   3. Kill all vLLM:  pkill -f 'vllm.entrypoints'"
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

# Start in background with proper GPU isolation
# CUDA_DEVICE_ORDER=PCI_BUS_ID ensures GPU indices match nvidia-smi
# CUDA_VISIBLE_DEVICES must be exported BEFORE python starts
nohup env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=$GPU \
    bash -c "
        cd '$PROJECT_DIR'
        source .venv/bin/activate 2>/dev/null || true
        python3 -m vllm.entrypoints.openai.api_server \
            --model '$MODEL' \
            --host 0.0.0.0 \
            --port $PORT \
            --max-model-len 4096 \
            --dtype auto \
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
echo "Commands:"
echo "   Monitor:  tail -f $LOG_FILE"
echo "   Stop:     kill \$(cat $PID_FILE)"
echo "   Kill all: pkill -f 'vllm.entrypoints'"
echo "   Restart:  $0 $GPU $MODEL_SIZE $PORT --force"