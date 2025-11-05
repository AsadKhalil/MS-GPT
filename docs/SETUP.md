# Setup Guide

## Quick Start

1. **Environment Setup**
   ```bash
   # Make setup script executable
   chmod +x scripts/setup.sh
   
   # Run setup (creates virtual environment and installs dependencies)
   ./scripts/setup.sh
   ```

2. **Activate Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r config/requirements.txt
   ```

## Configuration

### API Keys
If using LLM-based processing, update the API keys in the processor files:
- OpenAI API key in `src/pdf_processors/llm_pdf_processor.py`
- Ollama configuration in `src/qa_generators/qa_generator.py`

### GROBID Setup (Optional)
For academic paper processing:
```bash
# Run GROBID Docker container
docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1
```

## Directory Structure

- **Input PDFs**: Place in `data/input/`
- **Processed Text**: Output goes to `data/extracted_text/`
- **Q&A Pairs**: Generated in `data/qa_outputs/`
- **Logs**: Check `logs/` for processing logs
- **Temp Files**: Intermediate files in `temp/`

## Usage Examples

### Basic Text Extraction
```bash
python src/pdf_processors/pymupdf_processor.py
```

### Batch Processing with LLM
```bash
python src/pdf_processors/batch_llm_processor.py
```

### Generate Q&A Pairs
```bash
python src/qa_generators/qa_generator.py
```

## Troubleshooting

1. **Import Errors**: Ensure you're in the project root and virtual environment is activated
2. **API Errors**: Check API keys and network connectivity
3. **GROBID Errors**: Ensure GROBID Docker container is running on port 8070
4. **Memory Issues**: For large PDFs, use page-by-page processing

## Development

- Add new processors to `src/pdf_processors/`
- Add new Q&A generators to `src/qa_generators/`
- Add new vision extractors to `src/vision_extractors/`
- Update tests in respective directories
