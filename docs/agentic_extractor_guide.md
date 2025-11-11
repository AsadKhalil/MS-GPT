# Agentic PDF Vision Text Extractor Guide

## Overview

The Agentic PDF Vision Extractor uses **LangGraph** to create a self-correcting, robust workflow for extracting text from PDFs using vision models. Unlike the standard extractor, this system:

- **Self-validates** extraction quality
- **Automatically retries** with different strategies
- **Reflects** on failures to improve
- **Adapts** temperature and prompts based on results
- **Detects** repetitive/low-quality outputs

## Architecture

```
┌─────────────┐
│   Extract   │  ← Initial extraction attempt
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Validate   │  ← Quality checks (repetition, length, diversity)
└──────┬──────┘
       │
       ▼
   Decision Point
   ├─── Quality OK? ──→ [Complete]
   │
   ├─── Needs Retry? ──→ [Reflect] ──→ [Retry Strategy] ──→ [Extract]
   │
   └─── Max Retries? ──→ [Complete]
```

## Key Features

### 1. **Quality Validation**

The validator checks:
- **Text length** (minimum 50 characters by default)
- **Repetition ratio** (detects "urokinase-urokinase..." loops)
- **Word diversity** (unique words / total words)
- **Special character ratio** (flags OCR noise)
- **Error patterns** (detects model fallback phrases)

Each issue reduces the quality score. If score < threshold (0.6), retry is triggered.

### 2. **Adaptive Retry Strategies**

Based on failure type:
- **High repetition** → Lower temperature (0.0), use 'detailed' strategy
- **Too short** → Increase temperature slightly, use 'detailed' strategy
- **Special characters/OCR noise** → Temperature 0.0, 'ocr-focused' strategy
- **Other** → Cycle through strategies: standard → detailed → ocr-focused

### 3. **Reflection Node**

Analyzes failures and provides feedback:
- Aggregates validation issues
- Suggests strategy improvements
- Logs reasoning for debugging

### 4. **Multi-Strategy Prompts**

**Standard**: Balanced extraction
```
Extract main text content clearly and accurately.
```

**Detailed**: Comprehensive capture
```
Focus on capturing ALL text content from the image, even small details.
Read carefully and extract every word from paragraphs.
```

**OCR-focused**: Precision transcription
```
Focus on accurate character recognition.
Carefully transcribe exactly what you see.
```

## Installation

```bash
cd /home/tk-lpt-0806/Desktop/MS(GPT)
pip install -r config/requirements.txt
```

This installs:
- `langgraph>=0.2.0`
- `langchain>=0.3.0`
- `langchain-community>=0.3.0`

## Configuration

Add to `config/config.json`:

```json
{
  "agentic": {
    "max_retries": 3,
    "quality_threshold": 0.6,
    "min_text_length": 50,
    "max_repetition_ratio": 0.3,
    "enable_reflection": true,
    "strategies": ["standard", "detailed", "ocr-focused"]
  }
}
```

### Parameters:

- `max_retries` (int): Maximum retry attempts per page (default: 3)
- `quality_threshold` (float): Minimum quality score to accept (0.0-1.0, default: 0.6)
- `min_text_length` (int): Minimum extracted text length (default: 50)
- `max_repetition_ratio` (float): Max ratio of most common word (default: 0.3)
- `enable_reflection` (bool): Enable reflection node (default: true)
- `strategies` (list): Available extraction strategies

## Usage

### Basic Usage

```bash
# Single file
python src/vision_extractors/agentic_vision_extractor.py data/input/paper.pdf

# Batch processing (uses config input_dir)
python src/vision_extractors/agentic_vision_extractor.py

# With custom config
python src/vision_extractors/agentic_vision_extractor.py -c config/config.json
```

### Background Execution

```bash
nohup /home/asad/MS-GPT/.venv/bin/python3 -u src/vision_extractors/agentic_vision_extractor.py \
  > logs/agentic_extraction.out 2>&1 &
```

Monitor progress:
```bash
tail -f logs/agentic_extraction.out
tail -f logs/pdf_extraction.log
```

### Advanced Options

```bash
python src/vision_extractors/agentic_vision_extractor.py \
  data/input/ \
  --batch \
  -m llama3.2-vision:latest \
  -o extracted_text/vision \
  --start-page 1 \
  --end-page 10 \
  -v
```

Options:
- `-c, --config`: Config file path
- `-m, --model`: Ollama vision model
- `-o, --output`: Output directory
- `-u, --url`: Ollama server URL
- `--start-page`: Start page (1-indexed)
- `--end-page`: End page
- `--batch`: Force batch mode
- `--no-resume`: Disable resume
- `--progress-file`: Custom progress file
- `--log-file`: Custom log file
- `-v, --verbose`: Debug logging

## How It Solves the Repetition Problem

Given your issue with `urokinase-urokinase-...`:

1. **Extract** node gets the repeated text
2. **Validate** node calculates:
   - Repetition ratio: `count("urokinase") / total_words` = ~0.9
   - This is > threshold (0.3) → quality_score -= 0.5
3. **Decision**: quality_score < 0.6 → needs retry
4. **Reflect** node logs: "High repetition ratio (0.90)"
5. **Retry Strategy** node:
   - Detects "repetition" in feedback
   - Lowers temperature to 0.0 (deterministic)
   - Switches to 'detailed' strategy
6. **Extract** (retry): New attempt with better params
7. **Validate** (retry): Checks new output
8. If still bad → max retries (3) → marks page as failed

## Monitoring Quality

Progress file (`logs/extraction_progress.json`) tracks:
```json
{
  "abc123...": {
    "file_path": "/path/to/file.pdf",
    "processed_pages": [1, 2, 3],
    "quality_scores": {
      "1": 0.85,
      "2": 0.92,
      "3": 0.65
    },
    "average_quality": 0.81,
    "completed": true
  }
}
```

Low quality pages need review.

## Comparison: Standard vs Agentic

| Feature | Standard Extractor | Agentic Extractor |
|---------|-------------------|-------------------|
| Retry logic | ❌ None | ✅ Automatic with strategy |
| Quality checks | ❌ Manual | ✅ Automated validation |
| Repetition detection | ❌ None | ✅ Built-in |
| Self-correction | ❌ None | ✅ Reflection + retry |
| Progress tracking | ✅ Yes | ✅ Yes + quality scores |
| Temperature adaptation | ❌ Fixed | ✅ Dynamic |
| Multiple strategies | ❌ One prompt | ✅ 3 strategies |

## Troubleshooting

### Still getting repetitions after max retries?

1. **Increase max_retries** in config: `"max_retries": 5`
2. **Lower quality_threshold**: `"quality_threshold": 0.5`
3. **Try different model**: `ollama pull llama3.2-vision:11b`
4. **Increase DPI**: `"image_dpi": 300` (better image quality)

### All pages failing validation?

- Check `logs/pdf_extraction.log` for validation feedback
- Lower `quality_threshold` temporarily
- Disable reflection: `"enable_reflection": false` (faster)

### Slow processing?

- Reduce `max_retries`: `"max_retries": 2`
- Use smaller/faster model: `qwen3-vl:2b`
- Disable reflection: `"enable_reflection": false`

## Example Output

```
============================================================
Processing page 5/120
============================================================
2025-11-09 16:30:15 - INFO - Extracting page 5 (attempt 1/3)
2025-11-09 16:30:42 - INFO - Extracted 1245 characters
2025-11-09 16:30:42 - INFO - Quality score: 0.45 - High repetition ratio (0.75)
2025-11-09 16:30:42 - INFO - Reflecting on extraction quality...
2025-11-09 16:30:42 - INFO - Reflection: Quality score 0.45 is below threshold 0.60 | Issues: High repetition ratio (0.75) | Consider reducing temperature
2025-11-09 16:30:42 - INFO - Planning retry 1/3
2025-11-09 16:30:42 - INFO - Lowering temperature to 0.0
2025-11-09 16:30:42 - INFO - Retry strategy: detailed
2025-11-09 16:30:42 - INFO - Extracting page 5 (attempt 2/3)
2025-11-09 16:31:18 - INFO - Extracted 856 characters
2025-11-09 16:31:18 - INFO - Quality score: 0.85 - OK
2025-11-09 16:31:18 - INFO - ✓ Page 5 extracted successfully (quality: 0.85, attempts: 2)
```

## API (Programmatic Usage)

```python
from src.vision_extractors.agentic_vision_extractor import (
    AgenticPDFVisionExtractor
)

# Initialize
extractor = AgenticPDFVisionExtractor(
    config_path="config/config.json",
    model="llama3.2-vision:latest"
)

# Single file
success = extractor.extract_from_pdf(
    "data/input/paper.pdf",
    output_filename="paper_extracted.txt"
)

# Batch
results = extractor.batch_extract("data/input/", resume=True)
print(f"Success rate: {sum(results.values())}/{len(results)}")
```

## Next Steps

1. Install dependencies: `pip install -r config/requirements.txt`
2. Update config: Add `agentic` section to `config/config.json`
3. Test on single file first
4. Run batch processing
5. Review quality scores in progress file
6. Tune parameters based on results

## Support

For issues or questions:
- Check logs: `logs/pdf_extraction.log`
- Review progress: `logs/extraction_progress.json`
- Adjust config parameters
- Try different vision models

