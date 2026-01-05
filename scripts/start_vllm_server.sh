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

# GPU Compute Capabilities (AWQ requires 7.5+):
#   GPU 0 (1080 Ti): 6.1 - NO AWQ support
#   GPU 1 (2080 Ti): 7.5 - AWQ supported  
#   GPU 2 (3090 Ti): 8.6 - AWQ supported (BEST)

# Model options:
# - Qwen/Qwen2.5-3B-Instruct       (~6GB FP16)  - Works on ANY GPU
# - Qwen/Qwen2.5-7B-Instruct-AWQ   (~5GB AWQ)   - Needs GPU 1 or 2
# - Qwen/Qwen2.5-14B-Instruct-AWQ  (~9GB AWQ)   - Needs GPU 1 or 2
# - Qwen/Qwen2.5-32B-Instruct-AWQ  (~18GB AWQ)  - Needs GPU 2 (3090 Ti)

# Default: Use GPU 2 (3090 Ti) with 7B AWQ model
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
GPU="${VLLM_GPU:-2}"

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
