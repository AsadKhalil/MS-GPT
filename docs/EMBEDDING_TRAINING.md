# Embedding Model Training Workflow

This guide provides a complete workflow for training custom embedding models on Q&A pairs extracted from scientific papers.

## Overview

The embedding training pipeline consists of three main steps:

1. **Extract Text** from PDFs using GROBID or vision extractors
2. **Generate Q&A Pairs** from extracted text
3. **Fine-Tune Embedding Model** on Q&A pairs

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements_embedding.txt
```

### Option 1: Using the Shell Script (Easiest)

```bash
# Run the complete training pipeline
./scripts/train_embeddings.sh
```

This script will:
- ✅ Check for required dependencies
- ✅ Verify input data exists
- ✅ Detect GPU availability
- ✅ Show configuration preview
- ✅ Train the model
- ✅ Report results

### Option 2: Manual Python Execution

```bash
# Train with default configuration
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json

# Train with custom parameters
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json \
    --input_jsonl data/qa_outputs/jsonl/questions_answers.jsonl \
    --output_dir models/custom_embeddings \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 5 \
    --batch_size 32
```

## Complete Workflow

### Step 1: Prepare Q&A Data

If you haven't already generated Q&A pairs:

```bash
# Generate Q&A pairs from extracted text
python src/qa_generators/qa_generator_jsonl.py \
    --config config/qa_generator.json \
    --input_dir grobid_processed_pdf \
    --output_dir data/qa_outputs/jsonl
```

This creates: `data/qa_outputs/jsonl/questions_answers.jsonl`

### Step 2: Configure Training

Edit `config/embedding_finetuner.json`:

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

**Key Parameters:**
- `base_model`: Pre-trained model to fine-tune ([see options](#model-selection))
- `num_epochs`: Number of training passes (3-5 recommended)
- `train_batch_size`: Batch size (increase if you have more GPU memory)
- `loss_function`: Training objective (MultipleNegativesRankingLoss recommended for Q&A)

### Step 3: Train the Model

```bash
./scripts/train_embeddings.sh
```

Training will:
1. Load Q&A pairs from JSONL
2. Split data into train/validation sets
3. Fine-tune the embedding model
4. Save checkpoints periodically
5. Save final model to output directory

### Step 4: Test the Model

```bash
# Run example usage scripts
python examples/use_finetuned_embeddings.py
```

Or use in your own code:

```python
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Encode text
embeddings = model.encode([
    "What is machine learning?",
    "Machine learning is a subset of AI..."
])

# Calculate similarity
similarity = model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity}")
```

## Model Selection

### Recommended Models by Use Case

| Use Case | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| **Quick Start** | `all-MiniLM-L6-v2` | 80MB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Q&A Tasks** | `multi-qa-MiniLM-L6-cos-v1` | 80MB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **Best Quality** | `all-mpnet-base-v2` | 420MB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **Multilingual** | `paraphrase-multilingual-MiniLM-L12-v2` | 420MB | ⚡⚡ | ⭐⭐⭐⭐ |

### Model Details

#### all-MiniLM-L6-v2 (Default)
- **Best for**: Getting started, resource-constrained environments
- **Embedding dim**: 384
- **Speed**: Very fast inference
- **Use case**: General-purpose semantic search

#### multi-qa-MiniLM-L6-cos-v1 (Recommended for Q&A)
- **Best for**: Question-answering tasks
- **Embedding dim**: 384
- **Speed**: Very fast inference
- **Pre-trained on**: 215M+ Q&A pairs from various datasets
- **Use case**: Semantic search for questions and answers

#### all-mpnet-base-v2 (Best Quality)
- **Best for**: When quality is more important than speed
- **Embedding dim**: 768
- **Speed**: Moderate inference
- **Use case**: High-quality semantic search and retrieval

## Training Configuration Guide

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 8GB
- Disk: 5GB free space

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better)
- RAM: 16GB+
- Disk: 10GB free space

### Training Time Estimates

Training times on RTX 3090:

| Q&A Pairs | Model | Epochs | Batch Size | Time |
|-----------|-------|--------|------------|------|
| 1,000 | MiniLM-L6 | 3 | 16 | ~5 min |
| 5,000 | MiniLM-L6 | 3 | 16 | ~20 min |
| 10,000 | MiniLM-L6 | 3 | 16 | ~40 min |
| 10,000 | MPNet-base | 3 | 16 | ~2 hours |

**CPU training** is approximately **10-20x slower**.

### Hyperparameter Tuning

#### For Small Datasets (<1,000 pairs)
```json
{
  "num_epochs": 5,
  "train_batch_size": 8,
  "warmup_steps": 100,
  "learning_rate": 2e-5
}
```

#### For Medium Datasets (1,000-10,000 pairs)
```json
{
  "num_epochs": 3,
  "train_batch_size": 16,
  "warmup_steps": 500,
  "learning_rate": 2e-5
}
```

#### For Large Datasets (>10,000 pairs)
```json
{
  "num_epochs": 2,
  "train_batch_size": 32,
  "warmup_steps": 1000,
  "learning_rate": 2e-5
}
```

## Loss Functions

### MultipleNegativesRankingLoss (Default, Recommended)

**Best for**: Q&A pairs, semantic search

**How it works**: Uses in-batch negatives - each question's answer is a positive, all other answers in the batch are negatives.

**Advantages**:
- ✅ No need to mine hard negatives
- ✅ Works well with Q&A format
- ✅ Efficient training
- ✅ Good performance

```json
{
  "loss_function": "MultipleNegativesRankingLoss"
}
```

### ContrastiveLoss

**Best for**: Binary similarity tasks

**How it works**: Pulls similar pairs closer, pushes dissimilar pairs apart.

**Advantages**:
- ✅ Simple and intuitive
- ✅ Good for labeled positive/negative pairs

### CosineSimilarityLoss

**Best for**: Regression tasks with similarity scores

**How it works**: Directly optimizes cosine similarity to match target scores.

### OnlineContrastiveLoss

**Best for**: When you have hard negatives

**How it works**: Mines hard negatives during training.

## Monitoring Training

### Progress Tracking

Training progress is saved to: `models/fine_tuned_embeddings/training_progress.json`

```json
{
  "training_completed": true,
  "completed_at": "2025-11-11T10:30:00Z",
  "duration_seconds": 3600,
  "train_examples": 9000,
  "eval_examples": 1000,
  "final_model_path": "models/fine_tuned_embeddings/final_model"
}
```

### Training Metrics

Metrics are logged to: `models/fine_tuned_embeddings/training_metrics.jsonl`

```json
{"timestamp": "...", "epoch": 1, "step": 100, "loss": 0.245}
{"timestamp": "...", "epoch": 1, "step": 200, "loss": 0.198}
```

### Logs

Detailed logs: `logs/embedding_finetuning.log`

```
2025-11-11 10:00:00 - INFO - Loading Q&A data...
2025-11-11 10:00:05 - INFO - Loaded 10000 Q&A pairs
2025-11-11 10:00:10 - INFO - Training examples: 9000
2025-11-11 10:00:10 - INFO - Validation examples: 1000
2025-11-11 10:00:15 - INFO - Using CUDA device: NVIDIA GeForce RTX 3090
```

## Using the Fine-Tuned Model

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Encode texts
embeddings = model.encode([
    "What is climate change?",
    "Climate change refers to long-term shifts in temperatures..."
])

# Calculate similarity
similarity = model.similarity(embeddings[0], embeddings[1])
```

### Semantic Search

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Index documents
documents = [
    "Machine learning trains models on data.",
    "Deep learning uses neural networks.",
    "NLP processes human language.",
]
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Search
query = "How do computers learn?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Find most similar documents
scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_results = scores.argsort(descending=True)

for idx in top_results[:3]:
    print(f"[{scores[idx]:.4f}] {documents[idx]}")
```

### Integration with Vector Databases

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Create FAISS index
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)

# Add documents
documents = ["Doc 1", "Doc 2", "Doc 3"]
embeddings = model.encode(documents, convert_to_numpy=True)
index.add(embeddings)

# Search
query = "Search query"
query_emb = model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_emb, k=5)
```

## Troubleshooting

### Out of Memory

**Symptoms**: `CUDA out of memory` or system freezes

**Solutions**:
1. Reduce `train_batch_size` (e.g., 16 → 8)
2. Reduce `max_seq_length` (e.g., 384 → 256)
3. Use a smaller model (e.g., MiniLM instead of MPNet)
4. Close other applications

### Poor Performance

**Symptoms**: Low similarity scores, irrelevant search results

**Solutions**:
1. Check Q&A data quality
2. Increase training data (more Q&A pairs)
3. Increase `num_epochs` (e.g., 3 → 5)
4. Try different loss function
5. Use a larger base model

### Slow Training

**Symptoms**: Training takes very long

**Solutions**:
1. Use GPU instead of CPU
2. Increase `train_batch_size` (if memory allows)
3. Use a smaller model
4. Reduce `evaluation_steps`
5. Reduce `save_steps`

### Training Crashes

**Symptoms**: Training stops with errors

**Solutions**:
1. Check logs: `logs/embedding_finetuning.log`
2. Verify JSONL format is correct
3. Ensure sufficient disk space
4. Update dependencies
5. Restart and resume from checkpoint

## Advanced Topics

### Custom Evaluation

Create domain-specific evaluation:

```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Your test data
queries = ["Q1", "Q2", "Q3"]
passages = ["A1", "A2", "A3"]
scores = [1.0, 1.0, 1.0]  # All positive pairs

evaluator = EmbeddingSimilarityEvaluator(
    queries, passages, scores,
    name="custom_eval"
)

# Evaluate model
score = evaluator(model)
print(f"Evaluation score: {score}")
```

### Multi-Task Training

Train on multiple objectives:

```python
# Combine multiple loss functions
train_objectives = [
    (dataloader1, loss1),
    (dataloader2, loss2),
]

model.fit(
    train_objectives=train_objectives,
    epochs=3
)
```

### Transfer Learning

Fine-tune an already fine-tuned model:

```python
# Load your previously fine-tuned model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Continue training on new data
model.fit(
    train_objectives=[(new_dataloader, new_loss)],
    epochs=2
)
```

## Best Practices

### ✅ DO

- Start with a Q&A-optimized model (`multi-qa-MiniLM-L6-cos-v1`)
- Use `MultipleNegativesRankingLoss` for Q&A pairs
- Monitor validation performance during training
- Save checkpoints regularly
- Test on held-out evaluation data
- Version your models

### ❌ DON'T

- Train on poor quality Q&A pairs
- Use too large batch sizes (causes OOM)
- Overtrain (watch for overfitting)
- Skip evaluation
- Ignore GPU/CPU differences in performance

## References

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Training Overview](https://www.sbert.net/docs/training/overview.html)
- [Loss Functions](https://www.sbert.net/docs/package_reference/losses.html)
- [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)

## Support

For issues:
1. Check logs: `logs/embedding_finetuning.log`
2. Review progress: `training_progress.json`
3. Verify JSONL format
4. See [Troubleshooting](#troubleshooting) section
5. Consult [complete guide](embedding_finetuner_guide.md)

