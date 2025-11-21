#!/bin/bash
# Background script to run QA generator

cd /home/tk-lpt-0806/Desktop/MS\(GPT\)

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the script in background with nohup
nohup python3 src/qa_generators/qa_generator_jsonl.py \
    --config config/qa_generator.json \
    > logs/qa_generator_$(date +%Y%m%d_%H%M%S).out 2>&1 &

# Get the process ID
PID=$!

# Save PID to file for later reference
echo $PID > logs/qa_generator.pid

echo "QA Generator started in background"
echo "Process ID: $PID"
echo "PID saved to: logs/qa_generator.pid"
echo "Log file: logs/qa_generator_$(date +%Y%m%d_%H%M%S).out"
echo ""
echo "To check status: tail -f logs/qa_generator_*.out"
echo "To stop: kill \$(cat logs/qa_generator.pid)"
