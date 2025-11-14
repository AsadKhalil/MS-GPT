# Embedding Model Fine-Tuning Guide

## Overview

The Embedding Fine-Tuner is a tool for training custom embedding models on Q&A pairs generated from scientific papers. Fine-tuned embeddings improve semantic search, document retrieval, and question-answering systems by learning domain-specific representations.

## Features

- **Multiple Base Models**: Support for various sentence-transformer models
- **Flexible Training**: Configurable batch sizes, epochs, and learning rates
- **Multiple Loss Functions**: Choose from different training objectives
- **Progress Tracking**: Automatic checkpointing and progress saving
- **Evaluation**: Built-in validation during training
- **GPU Support**: Automatic GPU detection and utilization
- **Resume Capability**: Resume training from checkpoints

## Installation

### Prerequisites

```bash
pip install torch sentence-transformers
```

### Full Requirements

```bash
pip install torch>=2.0.0
pip install sentence-transformers>=2.2.0
pip install transformers>=4.30.0
pip install pydantic>=2.0.0
pip install openai  # If not already installed
```

## Quick Start

### 1. Prepare Your Data

First, generate Q&A pairs using the Q&A generator:

```bash
python src/qa_generators/qa_generator_jsonl.py \
    --config config/qa_generator.json \
    --input_dir grobid_processed_pdf \
    --output_dir data/qa_outputs/jsonl
```

This creates a JSONL file with Q&A pairs in the format:
```json
{
  "id": "uuid",
  "question": "What is the main finding?",
  "answer": "The study found that...",
  "context": "...",
  "file_name": "paper.txt"
}
```

### 2. Configure Training

Edit `config/embedding_finetuner.json` or use command-line arguments:

```json
{
  "input_jsonl": "data/qa_outputs/jsonl/questions_answers.jsonl",
  "output_dir": "models/fine_tuned_embeddings",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "num_epochs": 3,
  "train_batch_size": 16,
  "loss_function": "MultipleNegativesRankingLoss"
}
```

### 3. Start Training

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json
```

### 4. Use the Fine-Tuned Model

```python
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Encode queries and documents
query_embedding = model.encode("What causes climate change?")
doc_embedding = model.encode("Climate change is caused by...")

# Calculate similarity
similarity = model.similarity(query_embedding, doc_embedding)
```

## Configuration Options

### Model Configuration

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `base_model` | Pre-trained model to fine-tune | `all-MiniLM-L6-v2` | See [Available Models](#available-models) |
| `max_seq_length` | Maximum sequence length | `384` | 128-512 |

### Training Configuration

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `num_epochs` | Number of training epochs | `3` | 3-5 |
| `train_batch_size` | Training batch size | `16` | 8-32 (depends on GPU) |
| `eval_batch_size` | Evaluation batch size | `16` | 16-32 |
| `learning_rate` | Learning rate | `2e-5` | 1e-5 to 5e-5 |
| `warmup_steps` | Warmup steps | `500` | 10% of total steps |
| `train_split` | Train/validation split ratio | `0.9` | 0.8-0.95 |

### Loss Functions

| Loss Function | Use Case | Description |
|---------------|----------|-------------|
| `MultipleNegativesRankingLoss` | **Q&A pairs (Recommended)** | Uses in-batch negatives for contrastive learning |
| `ContrastiveLoss` | Binary similarity | Pulls similar pairs closer, pushes dissimilar apart |
| `CosineSimilarityLoss` | Direct similarity | Optimizes cosine similarity directly |
| `OnlineContrastiveLoss` | Hard negatives | Performs online hard negative mining |

### Available Models

#### Lightweight & Fast
- `sentence-transformers/all-MiniLM-L6-v2` ✨ **Recommended for starting**
  - Size: 80MB
  - Speed: Very fast
  - Quality: Good

#### High Quality
- `sentence-transformers/all-mpnet-base-v2`
  - Size: 420MB
  - Speed: Moderate
  - Quality: Excellent

#### Q&A Optimized
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` ✨ **Best for Q&A**
  - Size: 80MB
  - Pre-trained on Q&A datasets
  - Optimized for semantic search

#### Multilingual
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Size: 420MB
  - Supports 50+ languages

## Command-Line Usage

### Basic Training

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json
```

### Override Configuration

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json \
    --input_jsonl data/custom_qa.jsonl \
    --output_dir models/custom_model \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 5 \
    --batch_size 32
```

### Test Existing Model

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json \
    --test_only
```

## Training Process

### 1. Data Loading
- Reads Q&A pairs from JSONL file
- Validates data format
- Reports number of valid pairs

### 2. Data Preparation
- Converts Q&A pairs to `InputExample` format
- Splits data into training and validation sets
- Creates data loaders

### 3. Model Training
- Loads base model
- Applies loss function
- Trains with progress tracking
- Evaluates on validation set
- Saves checkpoints periodically

### 4. Model Saving
- Saves best model based on validation performance
- Saves final model after training
- Stores training metrics and progress

## Monitoring Training

### Progress Tracking

Training progress is saved to `training_progress.json`:

```json
{
  "training_completed": true,
  "completed_at": "2025-11-11T10:30:00Z",
  "duration_seconds": 3600,
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "num_epochs": 3,
  "train_examples": 9000,
  "eval_examples": 1000,
  "final_model_path": "models/fine_tuned_embeddings/final_model"
}
```

### Training Metrics

Metrics are logged to `training_metrics.jsonl`:

```json
{"timestamp": "2025-11-11T10:00:00Z", "epoch": 1, "step": 100, "loss": 0.245}
{"timestamp": "2025-11-11T10:05:00Z", "epoch": 1, "step": 200, "loss": 0.198}
```

### Logs

Detailed logs are written to `logs/embedding_finetuning.log`:

```
2025-11-11 10:00:00 - INFO - Loading Q&A data...
2025-11-11 10:00:05 - INFO - Loaded 10000 Q&A pairs
2025-11-11 10:00:10 - INFO - Training examples: 9000
2025-11-11 10:00:10 - INFO - Validation examples: 1000
```

## GPU vs CPU Training

### GPU Training (Recommended)

- **Requirements**: NVIDIA GPU with CUDA support
- **Speed**: 10-50x faster than CPU
- **Batch Size**: Can use larger batches (32-64)

```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

### CPU Training

- **Use Case**: No GPU available
- **Speed**: Slower but functional
- **Batch Size**: Use smaller batches (8-16)

The script automatically detects and uses available hardware.

## Best Practices

### 1. Data Quality

✅ **Good Q&A Pairs**:
- Clear, specific questions
- Complete, accurate answers
- Relevant to your domain
- Diverse topics

❌ **Poor Q&A Pairs**:
- Vague or ambiguous questions
- Incomplete answers
- Off-topic content
- Duplicate pairs

### 2. Hyperparameter Tuning

**Start with defaults**, then adjust:

- **Small dataset (<1000 pairs)**: 
  - More epochs (5-10)
  - Smaller batch size (8)
  
- **Medium dataset (1000-10000 pairs)**:
  - Default settings work well
  - 3-5 epochs
  
- **Large dataset (>10000 pairs)**:
  - Fewer epochs (2-3)
  - Larger batch size (32-64)

### 3. Model Selection

**Choose based on requirements**:

| Priority | Model | Reason |
|----------|-------|--------|
| Speed | `all-MiniLM-L6-v2` | Fastest inference |
| Quality | `all-mpnet-base-v2` | Best performance |
| Q&A Tasks | `multi-qa-MiniLM-L6-cos-v1` | Pre-trained on Q&A |
| Multilingual | `paraphrase-multilingual-*` | Multiple languages |

### 4. Evaluation

Always evaluate your model:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Your test queries and passages
queries = ["What is X?", "How does Y work?"]
passages = ["X is...", "Y works by..."]
scores = [1.0, 1.0]

evaluator = EmbeddingSimilarityEvaluator(queries, passages, scores)
score = evaluator(model)
print(f"Evaluation score: {score}")
```

## Troubleshooting

### Out of Memory Errors

**Problem**: `CUDA out of memory` or system memory exhausted

**Solutions**:
1. Reduce `train_batch_size` (e.g., 16 → 8)
2. Reduce `max_seq_length` (e.g., 384 → 256)
3. Use a smaller base model
4. Use gradient accumulation (modify script)

### Poor Model Performance

**Problem**: Model doesn't improve search/retrieval

**Solutions**:
1. Verify Q&A data quality
2. Increase training data (more Q&A pairs)
3. Try different loss function
4. Increase number of epochs
5. Use a larger base model

### Training Takes Too Long

**Problem**: Training is very slow

**Solutions**:
1. Use GPU instead of CPU
2. Use smaller base model
3. Increase batch size (if memory allows)
4. Reduce number of epochs
5. Reduce `evaluation_steps`

### Model Not Loading

**Problem**: Can't load fine-tuned model

**Solutions**:
1. Check model path exists: `models/fine_tuned_embeddings/final_model`
2. Verify training completed successfully
3. Check logs for errors
4. Ensure all required files are present

## Advanced Usage

### Custom Evaluation

Create custom evaluation metrics:

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Your corpus
corpus = {
    "doc1": "Document 1 text...",
    "doc2": "Document 2 text..."
}

# Your queries and relevant docs
queries = {
    "q1": "Question 1?"
}

relevant_docs = {
    "q1": {"doc1"}
}

evaluator = InformationRetrievalEvaluator(
    queries, corpus, relevant_docs
)
```

### Batch Processing

Process large collections efficiently:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Batch encode for efficiency
documents = ["Doc 1", "Doc 2", ..., "Doc N"]
embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
```

### Integration with Vector Databases

Use with FAISS, Pinecone, Weaviate, etc.:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Create FAISS index
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)  # Inner product similarity

# Add documents
docs = ["Document 1", "Document 2", ...]
embeddings = model.encode(docs, convert_to_numpy=True)
index.add(embeddings)

# Search
query = "What is machine learning?"
query_embedding = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_embedding, k=5)
```

## Performance Benchmarks

Approximate training times (on RTX 3090):

| Dataset Size | Base Model | Epochs | Batch Size | Time |
|-------------|------------|--------|------------|------|
| 1,000 pairs | MiniLM-L6 | 3 | 16 | ~5 min |
| 5,000 pairs | MiniLM-L6 | 3 | 16 | ~20 min |
| 10,000 pairs | MiniLM-L6 | 3 | 16 | ~40 min |
| 10,000 pairs | MPNet-base | 3 | 16 | ~2 hours |

CPU times are approximately 10-20x longer.

## References

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Training Custom Models](https://www.sbert.net/docs/training/overview.html)
- [Loss Functions](https://www.sbert.net/docs/package_reference/losses.html)
- [Evaluation](https://www.sbert.net/docs/package_reference/evaluation.html)

## Support

For issues or questions:
1. Check the logs: `logs/embedding_finetuning.log`
2. Review training progress: `training_progress.json`
3. Verify data format in input JSONL
4. Consult the troubleshooting section above

