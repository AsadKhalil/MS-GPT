import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode

# Path to the PDF file
pdf_path = '/home/tk-lpt-0806/Desktop/pdf_to_process/input/fbf30d7e8741a78d9eb6d228e02ed03a80499a36.pdf'

# Open the PDF
doc = fitz.open(pdf_path)

# Initialize an empty list to hold all text blocks
all_blocks = []

# Loop through each page in the document
for page_num, page in enumerate(doc, start=1):
    # Get the text blocks with sort=True to handle multi-column layouts
    blocks = page.get_text("blocks", sort=True)
    for block in blocks:
        if block[6] == 0:  # Only include text blocks
            # Append page number and block details
            all_blocks.append((page_num,) + block)

# Define the columns for the DataFrame
columns = ['page', 'x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'block_type']

# Create the DataFrame
df = pd.DataFrame(all_blocks, columns=columns)

# Apply unidecode to the 'text' column to handle Unicode characters
df['text'] = df['text'].apply(unidecode)

# Reset the index
df = df.reset_index(drop=True)


with open('output.txt', 'w', encoding='utf-8') as file:
    for index, row in df.iterrows():
        file.write(f"Page: {row['page']}, Block: {row['block_no']}, Text: {row['text']}\n")

# Option 2: Save as a CSV file (recommended for structured data)
df.to_csv('output.csv', index=False, encoding='utf-8')

# Close the document
doc.close()

print("DataFrame has been saved to 'output.txt' and 'output.csv'.")