import os
import sys
import json
import time
import uuid
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, ValidationError
from openai import OpenAI


class QA(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: List[QA]


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_valid_line(line: str) -> bool:
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
    if len(line.split()) < 10:
        return False
    return True


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def call_model(client: OpenAI, model: str, line: str, max_retries: int, api_delay: float) -> Tuple[List[QA], Dict]:
    prompt = (
        f"Analyze the following statement and generate as many relevant question and answer pairs as possible based solely on the information provided.\n\n"
        f'Statement: "{line}"\n\n'
        f"Instructions:\n"
        f"1. Generate clear and concise questions that can be directly answered using the information from the statement.\n"
        f"2. Provide accurate answers to each generated question.\n\n"
        f"Ensure that:\n"
        f"- Each question is directly related to the content of the statement.\n"
        f"- Each answer is extracted or inferred strictly from the statement without adding external information.\n\n"
        f'Format your response as a JSON object with a key "qa_pairs" containing a list of objects, each with "question" and "answer" fields.'
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that returns only JSON per the provided schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
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
                                            "question": {"type": "string"},
                                            "answer": {"type": "string"},
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
            )

            if not response.choices:
                return [], {}

            content = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            usage_dict = {}
            if usage is not None:
                try:
                    usage_dict = {
                        "total_tokens": usage.total_tokens,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                except Exception:
                    usage_dict = {}

            try:
                qa_resp = QAResponse.model_validate_json(content)
                return qa_resp.qa_pairs, usage_dict
            except (ValidationError, json.JSONDecodeError):
                if attempt < max_retries:
                    time.sleep(api_delay)
                    continue
                return [], usage_dict
        except Exception as e:
            logging.error(f"Model call failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(api_delay)
                continue
            return [], {}


def main():
    parser = argparse.ArgumentParser(description="Q&A JSONL Generator with resume")
    parser.add_argument("--config", type=str, default=str(Path("config/qa_generator.json")), help="Path to config JSON")
    parser.add_argument("--input_dir", type=str, help="Override input dir of .txt files")
    parser.add_argument("--output_dir", type=str, help="Override output dir for artifacts")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--base_url", type=str, help="Override OpenAI-compatible base URL")
    parser.add_argument("--api_key", type=str, help="Override API key")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    cfg = load_json(cfg_path, {})

    input_dir = Path(args.input_dir or cfg.get("input_dir", "grobid_proccessed_pdf"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "data/qa_outputs/jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or cfg.get("model", "deepseek-r1:1.5b")
    base_url = args.base_url or cfg.get("base_url", "http://localhost:11434/v1")
    api_key = args.api_key or cfg.get("api_key", "ollama")

    max_retries = int(cfg.get("max_retries", 5))
    api_delay = float(cfg.get("api_delay", 1))

    jsonl_file = output_dir / cfg.get("jsonl_output", "questions_answers.jsonl")
    progress_file = output_dir / cfg.get("progress_file", "progress.json")
    log_file = output_dir / cfg.get("log_file", "qa_generation.log")

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    if not input_dir.exists():
        logging.error(f"Input directory '{input_dir}' does not exist.")
        print(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    progress: Dict[str, Dict] = load_json(progress_file, {})
    client = build_client(base_url=base_url, api_key=api_key)

    # Open JSONL for append. Ensure file exists.
    if not jsonl_file.exists():
        jsonl_file.touch()

    files = sorted([p for p in input_dir.glob("*.txt")])
    if not files:
        print(f"No .txt files found in '{input_dir}'.")
        return

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    with jsonl_file.open("a", encoding="utf-8") as out_f:
        for file_path in files:
            fname = file_path.name
            file_prog = progress.get(fname, {"line_number": 0})
            start_line = int(file_prog.get("line_number", 0))

            try:
                with file_path.open("r", encoding="utf-8") as infile:
                    for idx, raw_line in enumerate(infile, start=1):
                        if idx <= start_line:
                            continue
                        line = raw_line.strip()
                        if not is_valid_line(line):
                            continue

                        qa_pairs, usage = call_model(
                            client=client,
                            model=model_name,
                            line=line,
                            max_retries=max_retries,
                            api_delay=api_delay,
                        )
                        if not qa_pairs:
                            # Still advance progress to avoid re-looping a problematic line forever
                            progress[fname] = {"line_number": idx}
                            save_json(progress_file, progress)
                            continue

                        ts = datetime.utcnow().isoformat() + "Z"
                        for qa in qa_pairs:
                            record = {
                                "id": str(uuid.uuid4()),
                                "run_id": run_id,
                                "created_at": ts,
                                "model": model_name,
                                "file_name": fname,
                                "file_path": str(file_path.resolve()),
                                "line_number": idx,
                                "context": line,
                                "question": qa.question,
                                "answer": qa.answer,
                            }
                            if usage:
                                record["usage"] = usage
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()

                        progress[fname] = {"line_number": idx}
                        save_json(progress_file, progress)
                        time.sleep(api_delay)
            except KeyboardInterrupt:
                print("\nInterrupted. Progress saved.")
                save_json(progress_file, progress)
                sys.exit(0)
            except Exception as e:
                logging.error(f"Error processing file {fname}: {e}")
                save_json(progress_file, progress)
                continue

    print(f"Done. JSONL written to: {jsonl_file}")


if __name__ == "__main__":
    main()


