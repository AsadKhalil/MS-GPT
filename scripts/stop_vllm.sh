#!/bin/bash
# Stop vLLM server(s)

cd "$(dirname "$0")/.."

echo "Stopping vLLM servers..."

for pid_file in logs/vllm_gpu*.pid; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        GPU=$(echo "$pid_file" | grep -o 'gpu[0-9]*')
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  Stopping $GPU (PID: $PID)"
            kill "$PID" 2>/dev/null
            rm -f "$pid_file"
        else
            echo "  $GPU not running (stale PID file)"
            rm -f "$pid_file"
        fi
    fi
done

# Also kill any orphaned vllm processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null && echo "  Killed orphaned vLLM processes" || true

echo "Done."
