import pdfplumber


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
        start_index = min([i for i in [text.lower().find(x) for x in ['abstract', 'introduction']] if i != -1])


        # Find end index
        end_index = min([i for i in [text.lower().find('references')] if i != -1])
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract all the text after the start keyword and before the end keyword
            text = text[start_index:end_index]
            
        return text



text_two_columned = extract_text_from_pdf('/home/dasulimov/pdf_papers/9676ef4b1bb13762786d3f9783ec199e6b82fc41.pdf')
text_one_columned = extract_text_from_pdf('/home/dasulimov/pdf_papers/631f01cad4c43a6eecaa8c5af64b459fe097ea56.pdf')

with open('one_columned_pdf.txt', 'w') as output_one_columned:
	output_one_columned.write(text_one_columned)

with open('two_columned_pdf.txt', 'w') as output_two_columned:
	output_two_columned.write(text_two_columned)
