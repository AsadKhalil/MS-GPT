#!/bin/bash
# =============================================================================
# Multi-GPU vLLM Server Launcher
# =============================================================================
# Your GPU Setup (from nvidia-smi):
#   GPU 0: GTX 1080 Ti (11GB) - ~11GB free
#   GPU 1: RTX 2080 Ti (11GB) - ~11GB free  
#   GPU 2: RTX 3090 Ti (24GB) - ~17GB free (6.4GB used by other process)
#
# Since GPUs are shared and have different architectures:
# - Tensor parallelism WON'T work (different GPU types)
# - Best: Run single vLLM instance on the GPU with most free VRAM
# - Alternative: Run multiple vLLM instances on different GPUs
# =============================================================================

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Activated .venv"
else
    echo "Warning: .venv not found, using system Python"
fi

# Configuration
HOST="0.0.0.0"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN=4096

# Model options (choose based on available VRAM):
# - Qwen/Qwen2.5-7B-Instruct-AWQ   (~5GB)  - Fast, fits anywhere
# - Qwen/Qwen2.5-14B-Instruct-AWQ  (~9GB)  - Better quality
# - Qwen/Qwen2.5-32B-Instruct-AWQ  (~18GB) - Best quality, needs 3090 Ti

# Default: Use 7B model for shared GPU compatibility
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

# GPU Selection (default to GPU 0 since GPU 2 is already partially used)
GPU="${VLLM_GPU:-0}"

# Create logs directory
mkdir -p logs

# Check for vLLM
python3 -c "import vllm" 2>/dev/null || {
    echo "Error: vLLM not installed. Install with:"
    echo "  pip install vllm"
    exit 1
}

echo "=============================================="
echo "🚀 Starting vLLM Server"
echo "=============================================="
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Host: $HOST:$PORT"
echo "Max context: $MAX_MODEL_LEN tokens"
echo "Project: $PROJECT_DIR"
echo "Log: logs/vllm_gpu${GPU}.log"
echo "=============================================="

# Run vLLM on selected GPU
CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --disable-log-requests \
    2>&1 | tee "logs/vllm_gpu${GPU}.log"
