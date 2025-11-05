import os
import time
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
import json
import logging
import argparse

# ---------------------- Configuration ----------------------
# List of models to use for Q&A generation
MODELS = ["llama3.3:latest", "deepseek-r1:70b"]

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

# Directory containing input text files
INPUT_DIR = "grobid_proccessed_pdf"
MAX_RETRIES = 5
API_CALL_DELAY = 1
FINAL_OUTPUT_FILE = "questions_answers_temp.json"
PROGRESS_FILE = "progress.json"
MATRIX_FILE = "matrix.json"


# ---------------------- Functions ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Q&A Generation Script")
    # Optionally, a command line model name can override MODELS
    parser.add_argument(
        "--model-name",
        type=str,
        help="(Optional) The model name for Q&A generation. Overrides MODELS list if provided.",
    )
    return parser.parse_args()


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
        "References:",
        "Declarations:",
    ]
    if not line:
        return False
    if any(line.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if len(line.split()) < 10:  # Adjust the minimum word count as needed
        return False
    return True


def generate_question_answer(line: str, model: str):
    """
    Generates multiple questions and answers based on the input line using OpenAI API.

    Returns:
        A tuple: (qa_pairs, total_tokens, completion_tokens, prompt_tokens)
        or ([], 0, 0, 0) on error.
    """
    prompt = (
        f"Analyze the following statement and generate as many relevant question and answer pairs as possible based solely on the information provided.\n\n"
        f'Statement: "{line}"\n\n'
        f"Instructions:\n"
        f"1. Generate clear and concise questions that can be directly answered using the information from the statement.\n"
        f"2. Provide accurate answers to each generated question.\n\n"
        f"Ensure that:\n"
        f"- Each question is directly related to the content of the statement.\n"
        f"- Each answer is extracted or inferred strictly from the statement without adding external information.\n\n"
        f'Format your response as a JSON object with a key "qa_pairs" containing a list of objects, each with "question" and "answer" fields.\n'
        f"Example:\n"
        f"{{\n"
        f'  "qa_pairs": [\n'
        f"    {{\n"
        f'      "question": "What is the main topic of the statement?",\n'
        f'      "answer": "The main topic is ..."\n'
        f"    }},\n"
        f"    {{\n"
        f'      "question": "Why is ... important?",\n'
        f'      "answer": "Because ..."\n'
        f"    }},\n"
        f"    {{\n"
        f'      "question": "How does ... affect ...?",\n'
        f'      "answer": "It affects ... by ..."\n'
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent assistant that generates relevant questions and accurate answers based solely on provided statements."
                            " Give maximum possible questions and answers according to response format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qa_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "qa_pairs": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question": {
                                                "type": "string",
                                                "description": "A relevant question generated from the statement.",
                                            },
                                            "answer": {
                                                "type": "string",
                                                "description": "The corresponding answer to the generated question.",
                                            },
                                        },
                                        "required": ["question", "answer"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["qa_pairs"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.1,
            )

            if not response.choices:
                logging.warning(f"No choices returned for line: {line}")
                return [], 0, 0, 0

            reply = response.choices[0]
            reply_content = reply.message.content.strip()

            try:
                qa_response = QAResponse.model_validate_json(reply_content)
                total_tokens = response.usage.total_tokens
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
                return (
                    qa_response.qa_pairs,
                    total_tokens,
                    completion_tokens,
                    prompt_tokens,
                )
            except (ValidationError, json.JSONDecodeError) as e:
                logging.warning(
                    f"JSON parsing failed on attempt {attempt} for line: {line}\nError: {e}"
                )
                logging.debug(
                    f"Attempt {attempt} - Raw reply content:\n{reply_content}"
                )

                if attempt < MAX_RETRIES:
                    logging.info(
                        f"Retrying... (Attempt {attempt + 1} of {MAX_RETRIES})"
                    )
                    time.sleep(API_CALL_DELAY)
                    continue
                else:
                    logging.error(f"Max retries exceeded for line: {line}")
                    return [], 0, 0, 0

        except Exception as e:
            logging.error(f"Error generating Q&A for line: {line}\nError: {e}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying... (Attempt {attempt + 1} of {MAX_RETRIES})")
                time.sleep(API_CALL_DELAY)
                continue
            else:
                logging.error(f"Max retries exceeded for line: {line}")
                return [], 0, 0, 0

    return [], 0, 0, 0


def load_progress(PROGRESS_FILE):
    """Load progress from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress, PROGRESS_FILE):
    """Save progress to the progress file."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)


def load_matrix(MATRIX_FILE):
    """Load matrix information from file."""
    if os.path.exists(MATRIX_FILE):
        with open(MATRIX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_usage": 0, "total_prompt": 0, "total_completion": 0}


def save_matrix(matrix, MATRIX_FILE):
    """Save matrix information."""
    with open(MATRIX_FILE, "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=4)


# ---------------------- Main Process ------------------------
def main():
    args = parse_args()
    models_to_use = [args.model_name] if args.model_name else MODELS

    output_dir = "qa_generation_deepseek_vs_llama"
    os.makedirs(output_dir, exist_ok=True)

    FINAL_OUTPUT_FILE = os.path.join(output_dir, "questions_answers.json")
    PROGRESS_FILE = os.path.join(output_dir, "progress.json")
    MATRIX_FILE = os.path.join(output_dir, "matrix.json")

    log_file = os.path.join(output_dir, "qa_generation.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )
    start_time = time.time()

    progress = load_progress(PROGRESS_FILE)
    matrix = load_matrix(MATRIX_FILE)

    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Input directory '{INPUT_DIR}' does not exist.")
        print(f"Input directory '{INPUT_DIR}' does not exist.")
        return

    qa_results = []

    # Process each file in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_DIR, filename)
            file_progress = progress.get(filename, {"line_number": 0})
            start_line = file_progress["line_number"]

            logging.info(f"Processing file: {filepath}")
            print(f"Processing file: {filepath}")

            with open(filepath, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    if line_number < start_line:
                        continue

                    line = line.strip()
                    if not is_valid_line(line):
                        logging.debug(
                            f"Skipping invalid line {line_number} in {filename}"
                        )
                        continue

                    # Create an object for the current context that will hold results for all models
                    context_obj = {"context": line, "model_results": []}

                    # Loop through each model
                    for model in models_to_use:
                        logging.info(
                            f"Generating Q&A for line {line_number} in {filename} using model {model}"
                        )
                        print(
                            f"Generating Q&A for line {line_number} in {filename} using model {model}"
                        )

                        qa_pairs, total_tokens, completion_tokens, prompt_tokens = (
                            generate_question_answer(line, model)
                        )

                        # Append results only if we have valid Q&A pairs
                        if qa_pairs:
                            context_obj["model_results"].append(
                                {
                                    "model_name": model,
                                    "qa_pairs": [
                                        {"question": qa.question, "answer": qa.answer}
                                        for qa in qa_pairs
                                    ],
                                }
                            )

                        matrix["total_usage"] += total_tokens
                        matrix["total_prompt"] += prompt_tokens
                        matrix["total_completion"] += completion_tokens
                        save_matrix(matrix, MATRIX_FILE)

                        time.sleep(API_CALL_DELAY)

                    # Append the context object if any model generated Q&A pairs
                    if context_obj["model_results"]:
                        qa_results.append(context_obj)

                    # Update progress for the file
                    progress[filename] = {"line_number": line_number + 1}
                    save_progress(progress, PROGRESS_FILE)

    try:
        with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
            json.dump(qa_results, outfile, ensure_ascii=False, indent=4)
        logging.info(f"All Q&A pairs have been written to '{FINAL_OUTPUT_FILE}'.")
        print(f"All Q&A pairs have been written to '{FINAL_OUTPUT_FILE}'.")
    except Exception as e:
        logging.error(f"Failed to write Q&A pairs to '{FINAL_OUTPUT_FILE}'. Error: {e}")
        print(
            f"Failed to write Q&A pairs to '{FINAL_OUTPUT_FILE}'. Check logs for details."
        )

    total_time = time.time() - start_time
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
    print(f"Total time taken: {total_time:.2f} seconds.")
    matrix["total_time"] = total_time
    save_matrix(matrix, MATRIX_FILE)


if __name__ == "__main__":
    main()
