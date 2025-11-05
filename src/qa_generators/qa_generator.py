import os

import time
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json
import time
import logging
import math
from pydantic import BaseModel, ValidationError
import openai

# ---------------------- Configuration ----------------------
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

# Directory containing input text files
INPUT_DIR = "sample"
# Maximum number of retries for API calls
MAX_RETRIES = 5
# Output file to store Q&A pairs
OUTPUT_FILE = "questions_answers.txt"

# OpenAI model to use
MODEL = "deepseek-r1:1.5b"
# Time to wait between API calls to respect rate limits (in seconds)
API_CALL_DELAY = 1


# ---------------------- Functions ---------------------------
class QA(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: List[QA]


def is_valid_line(line: str) -> bool:
    """
    Determines if a line is valid for Q&A generation.
    """
    exclude_prefixes = [
        "Acknowledgments:",
        "Funding:",
        "Data Availability Statement:",
        "Author Contributions:",
        "Conclusions:",
        "References:",
        "Declarations:",
    ]
    if not line:
        return False
    if any(line.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if len(line.split()) < 5:  # Adjust the minimum word count as needed
        return False
    return True


def generate_question_answer(line: str) -> List[QA]:
    """
    Generates multiple questions and answers based on the input line using OpenAI API.

    Args:
        line (str): A single line of text from the input file.

    Returns:
        List[QA]: A list of generated QA objects.
    """
    prompt = (
        f"Analyze the following statement and generate multiple relevant question and answer pairs based solely on the information provided.\n\n"
        f'Statement: "{line}"\n\n'
        f"Instructions:\n"
        f"1. Generate as many as necessary clear and concise questions that can be directly answered using the information from the statement.\n"
        f"2. Provide accurate answers to each generated question.\n\n"
        f"Ensure that:\n"
        f"- Each question is directly related to the content of the statement.\n"
        f"- Each answer is extracted or inferred strictly from the statement without adding external information.\n\n"
        f'Format your response as a JSON object with a key "qa_pairs" containing a list of objects, each with "question" and "answer" fields.\n'
        f"Example:\n"
        f"{{\n"
        f'  "qa_pairs": [\n'
        f"    {{\n"
        f'      "question": "Your first generated question",\n'
        f'      "answer": "Answer to the first question"\n'
        f"    }},\n"
        f"    {{\n"
        f'      "question": "Your second generated question",\n'
        f'      "answer": "Answer to the second question"\n'
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent assistant that generates relevant questions and accurate answers based solely on provided statements."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=500,  # Increased to accommodate multiple Q&A pairs
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.1,
            )

            if not response.choices:
                return []

            reply_content = response.choices[0].message.content.strip()
            # Remove the surrounding markdown code block if present
            if reply_content.startswith("```") and reply_content.endswith("```"):
                # reply_content = reply_content[3:-3].strip()
                reply_content = reply_content[len("```json") : -len("```")].strip()

            # Attempt to parse the JSON response
            try:

                qa_response = QAResponse.model_validate_json(reply_content)

                return qa_response.qa_pairs
            except ValidationError as ve:
                return []
            except json.JSONDecodeError as jde:
                return []
        except Exception as e:
            logging.error(f"Error generating Q&A for line: {line}\nError: {e}")
            return []

    logging.error(f"Max retries exceeded for line: {line}")


# ---------------------- Main Process ------------------------


def main():
    # Check if input directory exists
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory '{INPUT_DIR}' does not exist.")
        print(f"Input directory '{INPUT_DIR}' does not exist.")
        return

    qa_results = []

    # Iterate through each file in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_DIR, filename)
            logging.info(f"Processing file: {filepath}")
            print(f"Processing file: {filepath}")

            # Open and read each line in the file
            with open(filepath, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not is_valid_line(line):
                        continue  # Skip invalid lines

                    logging.info(f"Generating Q&A for line {line_number} in {filename}")
                    print(f"Generating Q&A for line {line_number} in {filename}")

                    # Generate question and answer
                    qa_pairs = generate_question_answer(line)

                    # Append each QA pair to the results
                    for qa in qa_pairs:
                        qa_results.append(
                            {
                                "question": qa.question,
                                "answer": qa.answer,
                            }
                        )

                    # Wait to respect API rate limits
                    time.sleep(API_CALL_DELAY)

    # Write all Q&A pairs to the output JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        json.dump(qa_results, outfile, ensure_ascii=False, indent=4)

    logging.info(f"All Q&A pairs have been written to '{OUTPUT_FILE}'.")
    print(f"All Q&A pairs have been written to '{OUTPUT_FILE}'.")


# ---------------------- Entry Point -------------------------

if __name__ == "__main__":
    main()
