#!/bin/bash
# Quick start script for embedding model fine-tuning

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Embedding Model Fine-Tuning Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default values
CONFIG_FILE="config/embedding_finetuner.json"
INPUT_JSONL="data/qa_outputs/jsonl/questions_answers.jsonl"
OUTPUT_DIR="models/fine_tuned_embeddings"
BASE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EPOCHS=3
BATCH_SIZE=16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --input)
            INPUT_JSONL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE        Configuration file (default: $CONFIG_FILE)"
            echo "  --input FILE         Input JSONL file (default: $INPUT_JSONL)"
            echo "  --output DIR         Output directory (default: $OUTPUT_DIR)"
            echo "  --model NAME         Base model name (default: $BASE_MODEL)"
            echo "  --epochs N           Number of epochs (default: $EPOCHS)"
            echo "  --batch_size N       Batch size (default: $BATCH_SIZE)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [ ! -f "$INPUT_JSONL" ]; then
    echo -e "${RED}ERROR: Input JSONL file not found: $INPUT_JSONL${NC}"
    echo ""
    echo "Please generate Q&A pairs first using:"
    echo "  python src/qa_generators/qa_generator_jsonl.py"
    exit 1
fi

# Count Q&A pairs in input file
QA_COUNT=$(grep -c '"question"' "$INPUT_JSONL" || true)
echo -e "${GREEN}Found $QA_COUNT Q&A pairs in input file${NC}"
echo ""

# Check if dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import torch; import sentence_transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Required dependencies not installed${NC}"
    echo ""
    echo "Please install dependencies:"
    echo "  pip install -r requirements_embedding.txt"
    exit 1
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
GPU_AVAILABLE=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')")
if [ "$GPU_AVAILABLE" = "yes" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}! No GPU detected - training will use CPU (slower)${NC}"
fi
echo ""

# Display configuration
echo -e "${YELLOW}Training Configuration:${NC}"
echo "  Input JSONL:  $INPUT_JSONL"
echo "  Output Dir:   $OUTPUT_DIR"
echo "  Base Model:   $BASE_MODEL"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Q&A Pairs:    $QA_COUNT"
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Run training
python3 src/embedding_trainers/embedding_finetuner.py \
    --config "$CONFIG_FILE" \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Model saved to: $OUTPUT_DIR/final_model"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model:"
    echo "     python examples/use_finetuned_embeddings.py"
    echo ""
    echo "  2. Use the model in your code:"
    echo "     from sentence_transformers import SentenceTransformer"
    echo "     model = SentenceTransformer('$OUTPUT_DIR/final_model')"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the logs for details:"
    echo "  logs/embedding_finetuning.log"
    exit 1
fi

