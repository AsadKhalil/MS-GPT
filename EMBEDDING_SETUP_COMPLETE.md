# ✅ Embedding Model Fine-Tuning Setup Complete

This document summarizes the embedding model fine-tuning system that has been set up for you.

## 📁 Files Created

### Core Training Script
- **`src/embedding_trainers/embedding_finetuner.py`** - Main fine-tuning script
  - Loads Q&A pairs from JSONL
  - Fine-tunes embedding models using sentence-transformers
  - Supports multiple loss functions and base models
  - Includes progress tracking and checkpointing
  - GPU/CPU detection and optimization

### Configuration
- **`config/embedding_finetuner.json`** - Training configuration
  - Model selection (default: all-MiniLM-L6-v2)
  - Training hyperparameters (epochs, batch size, learning rate)
  - Loss function selection
  - Input/output paths

### Documentation
- **`docs/embedding_finetuner_guide.md`** - Complete reference guide
  - Detailed explanations of all features
  - Model selection guide
  - Hyperparameter tuning
  - Troubleshooting
  
- **`docs/EMBEDDING_TRAINING.md`** - Quick start workflow guide
  - Step-by-step instructions
  - Common use cases
  - Best practices
  - Example code

### Example Scripts
- **`examples/use_finetuned_embeddings.py`** - Usage examples
  - Semantic search
  - Q&A retrieval
  - Text similarity
  - Batch processing
  - Vector database integration

### Helper Scripts
- **`scripts/train_embeddings.sh`** - Automated training script
  - Dependency checking
  - GPU detection
  - Configuration preview
  - Progress reporting
  
- **`scripts/validate_embedding_setup.py`** - Setup validation
  - Environment verification
  - Dependency checking
  - Data validation
  - Configuration testing

### Requirements
- **`requirements_embedding.txt`** - Python dependencies
  - PyTorch
  - Sentence-Transformers
  - Transformers
  - Pydantic
  - NumPy

## 🚀 Quick Start Guide

### 1. Validate Setup

First, check that everything is configured correctly:

```bash
python scripts/validate_embedding_setup.py
```

This will verify:
- ✅ Python version (3.8+)
- ✅ Required dependencies installed
- ✅ GPU availability (optional but recommended)
- ✅ Input data exists and is valid
- ✅ Configuration is correct
- ✅ Sufficient disk space

### 2. Install Dependencies

If not already installed:

```bash
pip install -r requirements_embedding.txt
```

**Required packages:**
- `torch>=2.0.0`
- `sentence-transformers>=2.2.0`
- `transformers>=4.30.0`
- `pydantic>=2.0.0`
- `numpy>=1.24.0`

### 3. Prepare Q&A Data

If you haven't generated Q&A pairs yet:

```bash
python src/qa_generators/qa_generator_jsonl.py \
    --config config/qa_generator.json \
    --input_dir grobid_processed_pdf \
    --output_dir data/qa_outputs/jsonl
```

This creates: `data/qa_outputs/jsonl/questions_answers.jsonl`

### 4. Configure Training

Edit `config/embedding_finetuner.json` to customize:

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

**Model Options:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight (default)
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A
- `sentence-transformers/all-mpnet-base-v2` - Best quality, slower

### 5. Train the Model

**Option A: Using the automated script (recommended)**

```bash
./scripts/train_embeddings.sh
```

**Option B: Using Python directly**

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json
```

**Option C: With custom parameters**

```bash
python src/embedding_trainers/embedding_finetuner.py \
    --config config/embedding_finetuner.json \
    --input_jsonl data/qa_outputs/jsonl/questions_answers.jsonl \
    --output_dir models/custom_embeddings \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 5 \
    --batch_size 32
```

### 6. Test the Model

```bash
python examples/use_finetuned_embeddings.py
```

This will run 4 examples:
1. **Basic Semantic Search** - Finding similar documents
2. **Q&A Retrieval** - Searching through Q&A pairs
3. **Text Similarity** - Comparing texts
4. **Batch Search** - Processing multiple queries

### 7. Use in Your Code

```python
from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Encode text
query_emb = model.encode("What is machine learning?")
doc_emb = model.encode("Machine learning is a subset of AI...")

# Calculate similarity
similarity = model.similarity(query_emb, doc_emb)
print(f"Similarity: {similarity:.4f}")
```

## 📊 Expected Training Times

On NVIDIA RTX 3090:

| Q&A Pairs | Model | Epochs | Time |
|-----------|-------|--------|------|
| 1,000 | MiniLM-L6 | 3 | ~5 min |
| 5,000 | MiniLM-L6 | 3 | ~20 min |
| 10,000 | MiniLM-L6 | 3 | ~40 min |
| 10,000 | MPNet-base | 3 | ~2 hours |

**Note:** CPU training is 10-20x slower.

## 🎯 Key Features

### 1. Multiple Base Models
Choose from various pre-trained models:
- Lightweight models (80MB) for fast inference
- High-quality models (420MB) for best performance
- Q&A-optimized models for semantic search
- Multilingual models for 50+ languages

### 2. Flexible Loss Functions
- **MultipleNegativesRankingLoss** (recommended for Q&A)
- ContrastiveLoss
- CosineSimilarityLoss
- OnlineContrastiveLoss

### 3. Progress Tracking
- Automatic checkpointing every N steps
- Training metrics logged to JSONL
- Progress saved to JSON
- Detailed logs for debugging

### 4. GPU Acceleration
- Automatic GPU detection
- CUDA optimization
- Batch processing for efficiency

### 5. Resume Capability
- Training can be interrupted and resumed
- Checkpoints saved periodically
- Progress tracking prevents data loss

## 📚 Documentation

### Quick Reference
- **Setup validation**: `python scripts/validate_embedding_setup.py`
- **Training**: `./scripts/train_embeddings.sh`
- **Testing**: `python examples/use_finetuned_embeddings.py`

### Detailed Guides
- **`docs/EMBEDDING_TRAINING.md`** - Complete workflow guide
- **`docs/embedding_finetuner_guide.md`** - Detailed reference
- **`examples/use_finetuned_embeddings.py`** - Usage examples

### Configuration
- **`config/embedding_finetuner.json`** - Training settings

## 🔧 Common Use Cases

### 1. Semantic Search
Build a search engine for scientific papers:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('models/fine_tuned_embeddings/final_model')

# Index your documents
docs = ["Doc 1", "Doc 2", "Doc 3"]
doc_embeddings = model.encode(docs, convert_to_tensor=True)

# Search
query = "Your search query"
query_embedding = model.encode(query, convert_to_tensor=True)
scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# Get top results
top_results = scores.argsort(descending=True)[:5]
```

### 2. Question Answering
Find relevant answers to questions:

```python
# Encode all answers
answers = ["Answer 1", "Answer 2", ...]
answer_embeddings = model.encode(answers)

# Find best answer for a question
question = "What is X?"
question_emb = model.encode(question)
best_match = util.cos_sim(question_emb, answer_embeddings).argmax()
```

### 3. Document Clustering
Group similar documents:

```python
from sklearn.cluster import KMeans

# Encode documents
docs = ["Doc 1", "Doc 2", ...]
embeddings = model.encode(docs)

# Cluster
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)
```

### 4. Duplicate Detection
Find duplicate or near-duplicate content:

```python
# Encode all documents
embeddings = model.encode(documents)

# Find duplicates using high similarity threshold
similarity_matrix = util.cos_sim(embeddings, embeddings)
duplicates = (similarity_matrix > 0.95).nonzero()
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Error:** `CUDA out of memory` or system freeze

**Solution:**
```json
{
  "train_batch_size": 8,  // Reduce from 16
  "max_seq_length": 256   // Reduce from 384
}
```

### Issue: Poor Performance

**Symptoms:** Low similarity scores, irrelevant results

**Solutions:**
1. Check Q&A data quality
2. Increase training data (more Q&A pairs)
3. Increase epochs (3 → 5)
4. Try `multi-qa-MiniLM-L6-cos-v1` model
5. Use `MultipleNegativesRankingLoss`

### Issue: Slow Training

**Solutions:**
1. Use GPU instead of CPU
2. Increase batch size (if memory allows)
3. Use smaller model (MiniLM instead of MPNet)
4. Reduce evaluation frequency

### Issue: Dependencies Not Found

**Solution:**
```bash
pip install -r requirements_embedding.txt
```

Or install manually:
```bash
pip install torch sentence-transformers transformers pydantic numpy
```

## 📈 Monitoring Training

### Progress File
`models/fine_tuned_embeddings/training_progress.json`

```json
{
  "training_completed": true,
  "duration_seconds": 3600,
  "train_examples": 9000,
  "eval_examples": 1000
}
```

### Metrics File
`models/fine_tuned_embeddings/training_metrics.jsonl`

```json
{"epoch": 1, "step": 100, "loss": 0.245}
{"epoch": 1, "step": 200, "loss": 0.198}
```

### Log File
`logs/embedding_finetuning.log`

Contains detailed training logs with timestamps.

## 🎓 Next Steps

1. **Validate setup**: Run `python scripts/validate_embedding_setup.py`
2. **Review configuration**: Check `config/embedding_finetuner.json`
3. **Start training**: Run `./scripts/train_embeddings.sh`
4. **Test model**: Run `python examples/use_finetuned_embeddings.py`
5. **Integrate**: Use the model in your applications

## 📞 Support

For help:
1. Check logs: `logs/embedding_finetuning.log`
2. Review progress: `training_progress.json`
3. Validate setup: `python scripts/validate_embedding_setup.py`
4. Read guides: `docs/EMBEDDING_TRAINING.md`
5. See examples: `examples/use_finetuned_embeddings.py`

## 🔗 Resources

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Training Guide](https://www.sbert.net/docs/training/overview.html)
- [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)
- [Loss Functions](https://www.sbert.net/docs/package_reference/losses.html)

---

**Happy Training! 🚀**

