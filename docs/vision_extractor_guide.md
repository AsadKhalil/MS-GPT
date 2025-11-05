# PDF Vision Text Extractor

A powerful Python script that uses Ollama's vision models to extract and clean text from PDF files. Unlike traditional OCR tools, this uses advanced vision language models for more accurate text extraction and intelligent content cleaning.

## Features

- **Vision-based extraction**: Uses Ollama vision models (LLaVA, BakLLaVA, etc.) for accurate text extraction
- **Intelligent text cleaning**: Removes citations, figure references, headers/footers automatically
- **Batch processing**: Process multiple PDFs in a directory
- **Configurable**: Customizable models, output formats, and processing options
- **Robust error handling**: Comprehensive logging and fallback mechanisms
- **Page range selection**: Process specific page ranges
- **High-quality output**: Maintains text structure and fixes OCR errors

## Prerequisites

1. **Ollama**: Make sure Ollama is installed and running
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   
   # Pull a vision model (choose one)
   ollama pull llava:13b      # Recommended for quality
   ollama pull llava:7b       # Faster alternative
   ollama pull bakllava       # Another good option
   ```

2. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Usage

```bash
# Extract text from a single PDF
python pdf_vision_extractor.py input/document.pdf

# Batch process all PDFs in a directory
python pdf_vision_extractor.py input/ --batch

# Use specific model and output directory
python pdf_vision_extractor.py input/document.pdf -m llava:7b -o my_output/

# Process specific page range
python pdf_vision_extractor.py input/document.pdf --start-page 5 --end-page 15

# Verbose logging
python pdf_vision_extractor.py input/document.pdf -v
```

### Python API Usage

```python
from pdf_vision_extractor import PDFVisionExtractor

# Initialize extractor
extractor = PDFVisionExtractor(
    model="llava:13b",
    output_dir="extracted_text"
)

# Extract from single file
success = extractor.extract_from_pdf("input/document.pdf")

# Batch process directory
results = extractor.batch_extract("input/")
```

## Command Line Options

```
positional arguments:
  pdf_path              Path to PDF file or directory

optional arguments:
  -h, --help            Show help message
  -m, --model MODEL     Ollama vision model to use (default: llava:13b)
  -o, --output OUTPUT   Output filename or directory
  -u, --url URL         Ollama server URL (default: http://localhost:11434)
  --start-page N        Start page (1-indexed, default: 1)
  --end-page N          End page (optional)
  --batch               Process all PDFs in directory
  -v, --verbose         Enable verbose logging
```

## Supported Models

### Recommended Models
- **llava:13b** - Best quality, slower processing
- **llava:7b** - Good balance of speed and quality
- **bakllava** - Alternative high-quality model
- **llava-phi3** - Efficient model for faster processing
- **moondream** - Lightweight option

### Model Selection Tips
- Use `llava:13b` for best quality on important documents
- Use `llava:7b` for batch processing or when speed matters
- Try `bakllava` if LLaVA models aren't working well
- Use smaller models for testing or low-resource environments

## Configuration

The script includes intelligent text cleaning that:

✅ **Preserves**:
- Main body text paragraphs
- Natural paragraph structure
- Proper sentence flow
- Corrected OCR errors (when certain)

❌ **Removes**:
- References and citations `[1]`, `(Smith 2020)`
- Figure/table mentions `"Figure 1:"`, `"Table 2 shows..."`
- Headers, footers, page numbers
- Footnotes and marginalia
- Mathematical equations (unless part of main text)
- OCR artifacts and stray characters

## Output

- **Text files**: Clean, readable text with proper paragraph structure
- **Logging**: Detailed logs saved to `pdf_extraction.log`
- **Progress tracking**: Real-time progress updates during processing

## Examples

### Single File
```bash
python pdf_vision_extractor.py research_paper.pdf
# Output: extracted_text/research_paper_extracted.txt
```

### Batch Processing
```bash
python pdf_vision_extractor.py documents/ --batch -m llava:7b
# Processes all PDFs in documents/ directory
```

### Custom Output
```bash
python pdf_vision_extractor.py paper.pdf -o clean_text.txt --start-page 3 --end-page 20
# Extract pages 3-20 to clean_text.txt
```

## Troubleshooting

### Common Issues

1. **"Failed to connect to Ollama"**
   ```bash
   # Make sure Ollama is running
   ollama serve
   ```

2. **"Model not found"**
   ```bash
   # Pull the required model
   ollama pull llava:13b
   ```

3. **"Empty response from vision model"**
   - Try a different model (`llava:7b` instead of `llava:13b`)
   - Check if the PDF pages contain actual text (not just images)
   - Increase timeout in the script if processing large/complex pages

4. **Slow processing**
   - Use `llava:7b` instead of `llava:13b`
   - Reduce image DPI in the script (change `dpi=200` to `dpi=150`)
   - Process fewer pages at once

### Performance Tips

- **For speed**: Use `llava:7b` model
- **For quality**: Use `llava:13b` model  
- **For batch processing**: Use smaller models and process during off-peak hours
- **Memory usage**: Process large PDFs in smaller page ranges

## Advanced Usage

See `example_usage.py` for detailed examples including:
- Custom model selection
- Error handling
- Batch processing with results analysis
- Page range processing

## Dependencies

- PyMuPDF (fitz) - PDF processing
- Pillow (PIL) - Image handling
- requests - Ollama API communication
- unidecode - Text normalization

## License

This script is provided as-is for educational and research purposes.

