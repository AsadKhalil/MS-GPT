# Batch Size Configuration & OOM Handling Guide

## 📍 Where Batch Size is Defined

### In Config File (`config/embedding_finetuner.json`)

**Line 14:** `"train_batch_size": 64`
- This is the **actual batch size** sent to GPU in each forward pass
- Controls GPU memory usage directly

**Line 15:** `"eval_batch_size": 32`
- Batch size for evaluation (validation/test)
- Usually smaller than training batch size

**Line 18:** `"gradient_accumulation_steps": 4`
- Number of batches to accumulate before updating weights
- Does NOT affect GPU memory (only training dynamics)

### How They Work Together

```
Effective Batch Size = train_batch_size × gradient_accumulation_steps
                     = 64 × 4
                     = 256
```

**What this means:**
- GPU processes **64 samples** at a time (memory usage)
- After **4 batches**, gradients are averaged and weights updated
- Training behaves as if batch size was **256** (training dynamics)

### In Code (`streaming_finetuner.py`)

**Line 67:** Default `train_batch_size: int = 64`
**Line 71:** Default `gradient_accumulation_steps: int = 4`
**Line 259:** Effective batch calculation: `effective_batch = self.config.train_batch_size * self.config.gradient_accumulation_steps`
**Line 462:** Used in DataLoader: `batch_size=self.config.train_batch_size`

---

## 🚨 Understanding Your OOM Problem

### Current Setup
- **GPU:** RTX 3090 Ti (24 GB)
- **train_batch_size:** 64
- **gradient_accumulation_steps:** 4
- **max_seq_length:** 512 (for large models), 8192 (for nomic)

### Memory Usage Breakdown

| Component | Memory Usage |
|-----------|-------------|
| Model weights (e5-large-v2) | ~1.3 GB |
| Optimizer states | ~2.6 GB (2× model) |
| Activations (batch=64, seq=512) | ~18-20 GB |
| Gradients | ~1.3 GB |
| **Total** | **~23-25 GB** |

**Problem:** With batch_size=64, you're using ~22.5 GB, leaving only ~1 GB free → OOM!

---

## ✅ Best Approaches to Handle OOM

### Strategy 1: Reduce Batch Size + Increase Gradient Accumulation ⭐ **RECOMMENDED**

**Goal:** Keep effective batch size same, reduce memory usage

```json
{
  "train_batch_size": 32,              // Half the memory (64 → 32)
  "gradient_accumulation_steps": 8,    // Double to maintain effective batch (4 → 8)
  "eval_batch_size": 16                // Also reduce eval batch
}
```

**Result:**
- Effective batch: 32 × 8 = **256** (same as before!)
- Memory usage: ~**12-14 GB** (down from 22.5 GB)
- Training quality: **Same** (same effective batch size)
- Speed: ~10-20% slower (more forward passes before backward)

**Pros:**
- ✅ Maintains training dynamics
- ✅ Significant memory reduction
- ✅ Minimal code changes

**Cons:**
- ⚠️ Slightly slower training
- ⚠️ Fewer in-batch negatives per step (but same overall)

---

### Strategy 2: Progressive Batch Size Reduction

**For very tight memory constraints:**

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 16,   // 16 × 16 = 256 effective batch
  "eval_batch_size": 8
}
```

**Result:**
- Memory usage: ~**8-10 GB**
- Effective batch: 256 (maintained)
- Speed: ~30-40% slower

---

### Strategy 3: Model-Specific Batch Sizes

**Different models need different batch sizes:**

```json
{
  "train_batch_size": 64,  // Default for base models
  
  "models_to_train": [
    {
      "name": "e5_large_v2",
      "model_path": "intfloat/e5-large-v2",
      "max_seq_length": 512,
      "_batch_size_override": 32,        // Override for large models
      "_gradient_accumulation_override": 8
    },
    {
      "name": "e5_base_v2",
      "model_path": "intfloat/e5-base-v2",
      "max_seq_length": 512
      // Uses default batch_size=64
    }
  ]
}
```

**Note:** This requires code modification to support per-model batch sizes.

---

### Strategy 4: Reduce Sequence Length (if possible)

**For nomic-embed-v1.5 with max_seq_length=8192:**

```json
{
  "models_to_train": [
    {
      "name": "nomic_embed_v1.5",
      "model_path": "nomic-ai/nomic-embed-text-v1.5",
      "max_seq_length": 4096,  // Reduce from 8192
      // Memory scales quadratically with sequence length!
    }
  ]
}
```

**Memory impact:**
- seq_length=8192: ~22 GB
- seq_length=4096: ~11 GB (50% reduction)
- seq_length=2048: ~6 GB (75% reduction)

---

### Strategy 5: Enable Mixed Precision Training

**Already enabled in code (line 596):**
```python
use_amp=torch.cuda.is_available(),  # Mixed precision if GPU
```

**But you can force it:**
```json
{
  "use_fp16": true,  // Or "use_bf16": true for newer GPUs
}
```

**Memory savings:** ~30-40% reduction

**Note:** Requires code modification to pass this to `model.fit()`

---

### Strategy 6: Gradient Checkpointing

**Trade compute for memory:**

```python
# In streaming_finetuner.py, add to model.fit():
gradient_checkpointing=True
```

**Memory savings:** ~50% reduction
**Speed cost:** ~20-30% slower

---

## 🎯 Recommended Configuration for Your Setup

### For Large Models (e5-large-v2, bge-large-en-v1.5)

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 8,
  "eval_batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 1
}
```

**Expected memory:** ~12-14 GB ✅
**Effective batch:** 256 (same as before)

### For Base Models (e5-base-v2, bge-base-en-v1.5)

```json
{
  "train_batch_size": 64,  // Can keep original
  "gradient_accumulation_steps": 4,
  "eval_batch_size": 32
}
```

**Expected memory:** ~12-16 GB ✅

### For Nomic (8192 seq length)

```json
{
  "models_to_train": [
    {
      "name": "nomic_embed_v1.5",
      "model_path": "nomic-ai/nomic-embed-text-v1.5",
      "max_seq_length": 4096,  // Reduce from 8192
      // Use batch_size=32, gradient_accumulation=8
    }
  ]
}
```

---

## 📊 Memory Estimation Formula

```
Memory ≈ Model_Size + (Batch_Size × Seq_Length × Hidden_Size × 4 bytes × layers)

For e5-large-v2:
- Model: ~1.3 GB
- Batch=64, Seq=512: ~18-20 GB
- Total: ~22-23 GB

For batch=32:
- Model: ~1.3 GB  
- Batch=32, Seq=512: ~9-10 GB
- Total: ~12-14 GB
```

---

## 🔧 Quick Fix: Update Your Config

**Immediate solution for OOM:**

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 8,
  "eval_batch_size": 16
}
```

This will:
- ✅ Fix OOM for large models
- ✅ Maintain training quality
- ✅ Work with your 24 GB GPU

---

## 📝 Summary

1. **Batch size location:** `config/embedding_finetuner.json` line 14 (`train_batch_size`)
2. **Effective batch:** `train_batch_size × gradient_accumulation_steps`
3. **Best OOM fix:** Reduce `train_batch_size` to 32, increase `gradient_accumulation_steps` to 8
4. **Memory scales with:** batch_size, sequence_length, model_size
5. **Keep effective batch same:** Maintains training quality while reducing memory

---

## 🚀 Next Steps

1. Update config with batch_size=32, gradient_accumulation=8
2. Test with one large model first
3. Monitor memory usage during training
4. Adjust further if needed (batch_size=16, gradient_accumulation=16)
