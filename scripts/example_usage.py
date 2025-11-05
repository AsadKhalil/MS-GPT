#!/usr/bin/env python3
"""
Example usage of the PDF Vision Extractor
"""

from pdf_vision_extractor import PDFVisionExtractor
import logging

def example_single_file():
    """Example: Extract text from a single PDF file."""
    print("=== Single File Extraction ===")
    
    # Initialize extractor
    extractor = PDFVisionExtractor(
        ollama_url="http://localhost:11434",
        model="llava:13b",  # or "llava:7b" for faster processing
        output_dir="extracted_text",
        log_level="INFO"
    )
    
    # Extract text from a single PDF
    pdf_path = "input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"
    success = extractor.extract_from_pdf(
        pdf_path=pdf_path,
        output_filename="example_output.txt",
        start_page=1,
        end_page=5  # Only process first 5 pages
    )
    
    if success:
        print("✓ Text extraction successful!")
    else:
        print("✗ Text extraction failed!")

def example_batch_processing():
    """Example: Batch process all PDFs in a directory."""
    print("\n=== Batch Processing ===")
    
    # Initialize extractor with different model
    extractor = PDFVisionExtractor(
        model="llava:7b",  # Faster model for batch processing
        output_dir="batch_extracted",
        log_level="INFO"
    )
    
    # Process all PDFs in input directory
    results = extractor.batch_extract("input/")
    
    # Print results
    successful = sum(results.values())
    total = len(results)
    print(f"Processed {successful}/{total} files successfully")

def example_with_custom_settings():
    """Example: Using custom settings and error handling."""
    print("\n=== Custom Settings Example ===")
    
    try:
        # Initialize with custom settings
        extractor = PDFVisionExtractor(
            ollama_url="http://localhost:11434",
            model="bakllava",  # Alternative vision model
            output_dir="custom_output",
            log_level="DEBUG"  # More verbose logging
        )
        
        # Process with specific page range
        pdf_path = "input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"
        success = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            start_page=3,
            end_page=10
        )
        
        print(f"Extraction {'successful' if success else 'failed'}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run examples
    example_single_file()
    example_batch_processing()
    example_with_custom_settings()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("Check the output directories for extracted text files.")

