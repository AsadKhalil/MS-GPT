#!/bin/bash
# Setup script for PDF Vision Extractor

echo "=== PDF Vision Extractor Setup ==="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "   Please install Ollama from: https://ollama.ai/download"
    echo "   Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "✓ Ollama found: $(ollama --version)"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama server is not running."
    echo "   Starting Ollama server..."
    ollama serve &
    sleep 5
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "❌ Failed to start Ollama server"
        echo "   Please run manually: ollama serve"
        exit 1
    fi
fi

echo "✓ Ollama server is running"

# Check for vision models
echo "🔍 Checking for vision models..."

models=$(ollama list | grep -E "(llava|bakllava|moondream)")
if [ -z "$models" ]; then
    echo "⚠️  No vision models found. Installing llava:7b (recommended for testing)..."
    ollama pull llava:7b
    
    if [ $? -eq 0 ]; then
        echo "✓ llava:7b model installed successfully"
    else
        echo "❌ Failed to install llava:7b model"
        echo "   Please run manually: ollama pull llava:7b"
        exit 1
    fi
else
    echo "✓ Vision models found:"
    echo "$models"
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p extracted_text test_output batch_extracted

echo "✓ Output directories created"

# Make scripts executable
chmod +x pdf_vision_extractor_clean.py
chmod +x test_vision_extractor.py
chmod +x example_usage.py

echo "✓ Scripts made executable"

echo
echo "🎉 Setup completed successfully!"
echo
echo "Next steps:"
echo "1. Test the installation:"
echo "   python3 test_vision_extractor.py"
echo
echo "2. Extract text from a PDF:"
echo "   python3 pdf_vision_extractor_clean.py input/your_file.pdf"
echo
echo "3. Batch process PDFs:"
echo "   python3 pdf_vision_extractor_clean.py input/ --batch"
echo
echo "4. See more examples:"
echo "   python3 example_usage.py"
echo

