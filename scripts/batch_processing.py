import os
import pdfplumber

def process_pdfs(input_dir, output_dir):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            print(filename)
            file_path = os.path.join(input_dir, filename)
            
            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            
            # Create a .txt file to save the extracted text
            txt_filename = os.path.splitext(filename)[0] + '.txt'  # Change extension to .txt
            output_path = os.path.join(output_dir, txt_filename)
            
            # Write the extracted text to the text file
            with open(output_path, 'w') as output_file:
                output_file.write(text)
            
            print(f"Processed {filename} and saved as {txt_filename}")

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            # Get starting points of bounding box
            bbox = page.bbox
            x_start = bbox[0]
            y_start = bbox[1]

            left_page = page.crop((x_start, y_start, page.width / 2, page.height))  # left part of the page
            right_page = page.crop((page.width / 2, y_start, page.width, page.height))  # right part of the page
            
            left_text = left_page.extract_text()
            right_text = right_page.extract_text()

            # Split the extracted text into words and then join with spaces 
            # This is done to avoid concatenation issues
            if left_text:
                left_text = ' '.join(left_text.split())
            if right_text:
                right_text = ' '.join(right_text.split())
            
            text += f"{left_text} {right_text}"
        
        text = text.replace('\n', ' ')  # Replaces '\n' with a space

        # Finding start index
        search_terms = ['abstract', 'introduction']
        start_indices = [text.lower().find(term) for term in search_terms]
        start_indices = [i for i in start_indices if i != -1]  # Filter out -1 values
        
        # Only call min() if we found any valid start indices
        if start_indices:
            start_index = min(start_indices)
        else:
            start_index = -1  # No valid start index found

        # Find end index
        end_index = text.lower().find('references')
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract all the text after the start keyword and before the end keyword
            text = text[start_index:end_index]
            
        return text

# Example usage
input_directory = '/home/tk-lpt-0806/Desktop/pdf_to_process'  # Directory containing PDFs
output_directory = '/home/tk-lpt-0806/Desktop/pdf_to_process/output'  # Directory to save .txt files

process_pdfs(input_directory, output_directory)
