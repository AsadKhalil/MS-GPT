# PDF Processing and Analysis Suite

A comprehensive toolkit for processing PDF documents, extracting text, and generating question-answer pairs using various methods including LLM-based processing, vision extraction, and traditional text extraction.

## Project Structure

```
pdf_to_process/
├── src/                          # Source code modules
│   ├── pdf_processors/           # PDF text extraction modules
│   ├── qa_generators/            # Question-Answer generation modules
│   └── vision_extractors/        # Vision-based text extraction modules
├── data/                         # Data directories
│   ├── input/                    # Input PDF files
│   ├── processed_pdfs/           # Processed PDF outputs
│   ├── extracted_text/           # Extracted text files
│   └── qa_outputs/               # Generated Q&A pairs
├── scripts/                      # Utility and setup scripts
├── config/                       # Configuration files
├── logs/                         # Log files
├── temp/                         # Temporary files
└── docs/                         # Documentation
```

## Modules Overview

### PDF Processors (`src/pdf_processors/`)

- **`llm_pdf_processor.py`** - Single PDF processing using LLM for text cleaning
- **`batch_llm_processor.py`** - Batch processing of PDFs using LLM
- **`page_by_page_processor.py`** - Page-by-page PDF processing with LLM
- **`pymupdf_processor.py`** - Basic PDF text extraction using PyMuPDF
- **`multi_format_converter.py`** - Convert PDFs to multiple formats (TXT, HTML, MD)
- **`grobid_processor.py`** - GROBID-based academic paper processing
- **`grobid_batch_processor.py`** - Batch processing using GROBID

### QA Generators (`src/qa_generators/`)

- **`qa_generator.py`** - Generate question-answer pairs from text
- **`qa_generator_v2.py`** - Enhanced Q&A generation (version 2)
- **`qa_generator_v1.py`** - Basic Q&A generation (version 1)

### Vision Extractors (`src/vision_extractors/`)

- **`vision_extractor.py`** - Vision-based text extraction from PDFs
- **`vision_extractor_clean.py`** - Clean version of vision extractor
- **`test_vision_extractor.py`** - Tests for vision extraction functionality

### Scripts (`scripts/`)

- **`pdf_preprocessing.py`** - PDF preprocessing utilities
- **`batch_processing.py`** - Batch processing workflows
- **`example_usage.py`** - Usage examples and demonstrations
- **`setup.sh`** - Environment setup script
- **`download_files.sh`** - File download utilities

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pdf_to_process
   ```

2. **Set up the environment:**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Install dependencies:**
   ```bash
   pip install -r config/requirements.txt
   ```

## Configuration

Configuration files are located in the `config/` directory:

- **`config.json`** - Main configuration file
- **`requirements.txt`** - Python dependencies

## Usage Examples

### Basic PDF Text Extraction

```python
from src.pdf_processors.pymupdf_processor import extract_text_from_pdf

# Extract text from a single PDF
text = extract_text_from_pdf('data/input/document.pdf')
```

### LLM-based Text Cleaning

```python
from src.pdf_processors.llm_pdf_processor import process_pdf_with_llm

# Process PDF with LLM cleaning
cleaned_text = process_pdf_with_llm('data/input/document.pdf')
```

### Generate Q&A Pairs

```python
from src.qa_generators.qa_generator import generate_qa_pairs

# Generate questions and answers from text
qa_pairs = generate_qa_pairs(text_content)
```

### Vision-based Extraction

```python
from src.vision_extractors.vision_extractor import extract_with_vision

# Extract text using vision models
extracted_text = extract_with_vision('data/input/document.pdf')
```

## Data Flow

1. **Input**: Place PDF files in `data/input/`
2. **Processing**: Use appropriate processors from `src/pdf_processors/`
3. **Text Storage**: Extracted text saved to `data/extracted_text/`
4. **Q&A Generation**: Use generators from `src/qa_generators/`
5. **Output**: Q&A pairs saved to `data/qa_outputs/`

## Supported Processing Methods

### Traditional Text Extraction
- PyMuPDF-based extraction
- Multi-format conversion (TXT, HTML, Markdown)

### LLM-Enhanced Processing
- OpenAI API integration
- Ollama local model support
- Text cleaning and formatting

### Academic Paper Processing
- GROBID integration for structured extraction
- Citation and reference handling

### Vision-based Extraction
- Image-to-text conversion
- OCR capabilities for scanned documents

## Logging

All processing activities are logged to the `logs/` directory:
- Processing logs
- Error logs
- Q&A generation logs

## Temporary Files

Intermediate processing files are stored in `temp/` directory and can be safely cleaned periodically.

## Contributing

1. Follow the modular structure when adding new features
2. Place new processors in appropriate `src/` subdirectories
3. Update documentation when adding new functionality
4. Add tests for new modules

## Requirements

- Python 3.8+
- PyMuPDF
- OpenAI API (optional, for LLM features)
- Ollama (optional, for local LLM processing)
- GROBID server (optional, for academic paper processing)

## License

[Add your license information here]

## Support

For issues and questions, please refer to the documentation in the `docs/` directory or create an issue in the repository.



see