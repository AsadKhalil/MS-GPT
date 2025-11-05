#!/usr/bin/env python3
"""
Test script for PDF Vision Extractor
"""

import sys
from pathlib import Path
from pdf_vision_extractor_clean import PDFVisionExtractor


def test_single_pdf():
    """Test extraction from a single PDF file."""
    print("=== Testing Single PDF Extraction ===")
    
    # Use one of the existing PDFs
    pdf_path = "input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Test PDF not found: {pdf_path}")
        return False
    
    try:
        # Initialize extractor
        extractor = PDFVisionExtractor(
            model="llava:7b",  # Use smaller model for testing
            output_dir="test_output",
            log_level="INFO"
        )
        
        # Extract first 2 pages only for testing
        success = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            output_filename="test_extraction.txt",
            start_page=1,
            end_page=2
        )
        
        if success:
            print("✓ Single PDF test passed!")
            
            # Check if output file was created
            output_file = Path("test_output/test_extraction.txt")
            if output_file.exists():
                print(f"✓ Output file created: {output_file}")
                print(f"  File size: {output_file.stat().st_size} bytes")
                
                # Show first few lines
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')[:5]
                    print("  First few lines:")
                    for i, line in enumerate(lines, 1):
                        if line.strip():
                            print(f"    {i}: {line[:80]}...")
            return True
        else:
            print("✗ Single PDF test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


def test_connection():
    """Test Ollama connection."""
    print("=== Testing Ollama Connection ===")
    
    try:
        extractor = PDFVisionExtractor(
            model="llava:7b",
            output_dir="test_output",
            log_level="INFO"
        )
        print("✓ Ollama connection test passed!")
        return True
        
    except SystemExit:
        print("✗ Ollama connection test failed!")
        print("  Make sure Ollama is running: ollama serve")
        print("  And the model is available: ollama pull llava:7b")
        return False
    except Exception as e:
        print(f"✗ Connection test failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("PDF Vision Extractor Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Connection
    if test_connection():
        tests_passed += 1
    
    print()
    
    # Test 2: Single PDF extraction
    if test_single_pdf():
        tests_passed += 1
    
    print()
    print("=" * 40)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

