import fitz  # PyMuPDF
import pandas as pd
from unidecode import unidecode
import json
from openai import OpenAI, APIError
import re

# --- Configuration ---
pdf_path = "/home/tk-lpt-0806/Desktop/pdf_to_process/input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"
# pdf_path = (
#     "/tear/yoda/pdf_to_txt_using_llm/input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"
# )
output_cleaned_file = "cleaned_output.txt"
intermediate_file = "intermediate_output.txt"
# ollama_api_url = "http://localhost:11434/v1"
ollama_model = "o3-mini"
key="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"
# base_url=ollama_api_url,
# --- Initialize OpenAI Client for Ollama ---
try:
    client = OpenAI(
        api_key=key,  # Required by the client library, but Ollama doesn't use it
    )

    print(f"OpenAI client initialized to target Ollama at")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

# --- PDF Parsing (Your existing code) ---
try:
    doc = fitz.open(pdf_path)
except Exception as e:
    print(f"Error opening PDF: {e}")
    exit()

all_blocks = []
for page_num, page in enumerate(doc, start=1):
    # Get text blocks, sorted for better reading order on the page
    blocks = page.get_text("blocks", sort=True)
    for block in blocks:
        # block format: (x0, y0, x1, y1, "text", block_no, block_type)
        if block[6] == 0:  # block_type 0 is text
            # Clean up text slightly - remove leading/trailing whitespace and replace multiple newlines
            text = block[4].strip().replace("\r", "\n").replace("\n\n", "\n")
            if text:  # Only add if there's actual text after stripping
                # Apply unidecode *here* if desired, or later
                # text = unidecode(text)
                all_blocks.append(
                    (page_num, block[5], text)
                )  # page_num, block_no, text

# Define the columns for the DataFrame
columns = ["page", "block_no", "text"]

# Create the DataFrame
df = pd.DataFrame(all_blocks, columns=columns)

# Ensure blocks are sorted correctly by page and then block number
df = df.sort_values(by=["page", "block_no"]).reset_index(drop=True)

# Optional: Apply unidecode to the entire column if preferred
df["text"] = df["text"].apply(unidecode)

# --- Chunking and Processing with Ollama ---
df.to_csv("output.csv", index=False, encoding="utf-8")

# Combine all text from all pages
full_text = "\n".join(df["text"])

with open(intermediate_file, "w", encoding="utf-8") as f:
    f.write(full_text)  # Fallback
print(f"Processing complete document (total {len(df)} blocks)...")

# --- Prepare the Prompt for Ollama ---
# Be very specific about the cleaning task.
prompt = f"""
    Extract and clean the raw text content from this PDF document following these strict rules:
    1. **Remove ALL**:
    - References/citations (e.g., [1], (Smith 2020))
    - Figure/table mentions (e.g., "Figure 1:", "Table 2 shows...")
    - Captions, footnotes, or marginalia
    - Page numbers/headers/footers
    - Stray characters, symbols, or OCR artifacts

    2. Preserve ONLY:
    - Main body text paragraphs
    - Natural paragraph breaks (single blank line between paragraphs)
    - Corrected OCR errors ONLY when absolutely certain (e.g., "teh" → "the")

    3. Formatting:
    - Single spaces between words
    - Single line breaks between paragraphs
    - No leading/trailing whitespace

    4. Absolutely DO NOT:
    - Add any commentary, explanations, or metadata
    - Summarize or rephrase content
    - Include any non-text elements
    - Create placeholder text like "[...]"

    Return ONLY the cleaned text content with no additional text from you.
    Text to clean:
    --- START TEXT ---
    {full_text}
    --- END TEXT ---

    Cleaned text:"""
messages = [
    {
        "role": "system",
        "content": "You are an assistant that cleans text according to specific rules and outputs the result *only* .You do not provide explanations or thoughts.",
    },
    {"role": "user", "content": prompt},
]

# --- Call Ollama API ---
try:
    print("Sending full document for processing...")
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        # temperature=0.1, # Optional: Lower temp for more deterministic cleaning
    )

    if (
        response.choices
        and response.choices[0].message
        and response.choices[0].message.content
    ):
        full_response_content = response.choices[0].message.content

        # Clean up the response text
        cleaned_text = full_response_content.strip()

        # --- Save Final Output ---
        with open(output_cleaned_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print("Document processed successfully.")

    else:
        # The API response structure itself was invalid or empty
        print("Warning: Received empty or unexpected response structure from API.")
        print("Using original text due to empty/invalid response.")
        # with open(output_cleaned_file, "w", encoding="utf-8") as f:
        #     f.write(full_text)  # Fallback

except Exception as e:  # General catch-all for other unexpected errors
    print(f"An unexpected error occurred during API response processing: {e}")
    print("Using original text due to unexpected error.")
    # with open(output_cleaned_file, "w", encoding="utf-8") as f:
    #     f.write(full_text)  # Fallback

# --- Close PDF ---
doc.close()

print(f"\nProcessing complete. Cleaned text saved to '{output_cleaned_file}'.")
