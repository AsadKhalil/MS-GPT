import os
import sys
import json
import time
import uuid
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pydantic import BaseModel, ValidationError
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError


# --- Configuration ---
class QA(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: List[QA]


# --- Utilities ---
def setup_logging(log_file: Path) -> None:
    """Configure logging with both file and console output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_json(path: Path, default):
    """Safely load JSON with error handling."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}. Using defaults.")
            return default
    return default


def save_json_atomic(path: Path, data) -> None:
    """Atomic JSON save to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.tmp')
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.move(str(temp_path), str(path))
    except Exception as e:
        logging.error(f"Failed to save {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()


def is_valid_paragraph(paragraph: str) -> bool:
    """Validate paragraph content with improved filtering."""
    if not paragraph or len(paragraph) < 50:  # Minimum length
        return False
    
    exclude_prefixes = [
        "acknowledgments:", "funding:", "data availability statement:",
        "author contributions:", "references:", "declarations:",
        "acknowledgements:",  # British spelling
        "acknowledgment:",    # Singular
    ]
    
    first_line = paragraph.split('\n')[0].strip().lower()
    if any(first_line.startswith(prefix) for prefix in exclude_prefixes):
        return False
        
    # Additional check for reference lists
    if 'doi:' in paragraph.lower() or 'http' in paragraph.lower():
        return False
        
    return True


def validate_qa_pairs(qa_pairs: List[QA]) -> List[QA]:
    """Filter out invalid QA pairs."""
    valid = []
    for qa in qa_pairs:
        q_clean = qa.question.strip()
        a_clean = qa.answer.strip()
        if (q_clean and a_clean and 
            len(q_clean) > 15 and len(a_clean) > 20 and
            not q_clean.startswith(('?', 'What?', 'How?'))):  # Avoid generic questions
            valid.append(QA(question=q_clean, answer=a_clean))
    return valid


def iter_paragraphs(file_path: Path) -> Iterable[Tuple[str, int, int]]:
    """Yield (paragraph, start_line, paragraph_index) with better sentence reconstruction."""
    buffer: List[str] = []
    start_line = 0
    
    with file_path.open("r", encoding="utf-8", errors="replace") as infile:
        for idx, raw_line in enumerate(infile, start=1):
            stripped = raw_line.strip()
            if stripped:
                if not buffer:
                    start_line = idx
                buffer.append(stripped)
            elif buffer:
                paragraph = " ".join(buffer)
                # Ensure proper ending
                if not paragraph.rstrip().endswith(('.', '?', '!')):
                    paragraph += "."
                yield paragraph, start_line, idx
                buffer = []
                start_line = 0
        
        if buffer:
            paragraph = " ".join(buffer)
            if not paragraph.rstrip().endswith(('.', '?', '!')):
                paragraph += "."
            yield paragraph, start_line, idx


# --- Core Processing ---
def process_file_worker(
    file_path: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    api_delay: float,
    run_id: str,
    per_file_dir: Path,
    progress_file: Path,
    failed_records_file: Path,
) -> Dict:
    """Worker function for parallel processing."""
    # Each worker gets its own client
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)
    
    fname = file_path.name
    progress = load_json(progress_file, {})
    file_prog = progress.get(fname, {"paragraph_index": 0, "line_number": 0})
    start_paragraph = int(file_prog.get("paragraph_index", 0))
    start_line = int(file_prog.get("line_number", 0))
    
    file_output_dir = per_file_dir / file_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    file_jsonl = file_output_dir / "qa.jsonl"
    failed_jsonl = file_output_dir / "failed.jsonl"
    
    stats = {
        "file": fname,
        "attempted": 0,
        "filtered": 0,
        "success": 0,
        "failed": 0,
        "qa_records": 0,
    }
    
    try:
        with file_jsonl.open("a", encoding="utf-8") as out_f, \
             failed_jsonl.open("a", encoding="utf-8") as fail_f:
            
            for p_idx, (paragraph, para_line, _) in enumerate(
                iter_paragraphs(file_path), start=1
            ):
                # Resume logic
                if p_idx < start_paragraph or (p_idx == start_paragraph and para_line <= start_line):
                    continue
                
                # Validation
                if not is_valid_paragraph(paragraph):
                    stats["filtered"] += 1
                    # Batched progress save (every 50 paragraphs)
                    if p_idx % 50 == 0:
                        progress[fname] = {"paragraph_index": p_idx, "line_number": para_line}
                        save_json_atomic(progress_file, progress)
                    continue
                
                stats["attempted"] += 1
                qa_pairs, usage = call_model_with_backoff(
                    client, model_name, paragraph, max_retries, api_delay
                )
                
                # Validation and filtering
                qa_pairs = validate_qa_pairs(qa_pairs)
                
                if not qa_pairs:
                    stats["failed"] += 1
                    # Save failure record
                    failure_record = {
                        "file_name": fname,
                        "file_path": str(file_path.resolve()),
                        "paragraph_index": p_idx,
                        "line_number": para_line,
                        "context": paragraph[:500],  # Truncate long contexts
                        "model_attempted": model_name,
                        "run_id": run_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    fail_f.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                else:
                    stats["success"] += 1
                    stats["qa_records"] += len(qa_pairs)
                    
                    # Write QA records
                    ts = datetime.utcnow().isoformat() + "Z"
                    for qa in qa_pairs:
                        record = {
                            "id": str(uuid.uuid4()),
                            "run_id": run_id,
                            "created_at": ts,
                            "model": model_name,
                            "file_name": fname,
                            "file_path": str(file_path.resolve()),
                            "line_number": para_line,
                            "paragraph_index": p_idx,
                            "context": paragraph,
                            "question": qa.question,
                            "answer": qa.answer,
                        }
                        if usage:
                            record["usage"] = usage
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Batched progress and flush
                if p_idx % 20 == 0:
                    out_f.flush()
                if p_idx % 50 == 0:
                    progress[fname] = {"paragraph_index": p_idx, "line_number": para_line}
                    save_json_atomic(progress_file, progress)
                
                # Circuit breaker: skip file if failure rate > 30%
                if stats["attempted"] > 10 and stats["failed"] / stats["attempted"] > 0.3:
                    logging.warning(f"Circuit breaker triggered for {fname}: >30% failure rate")
                    break
                
    except Exception as e:
        logging.error(f"Fatal error processing {fname}: {e}")
    
    # Final save
    progress[fname] = {"paragraph_index": p_idx, "line_number": para_line}
    save_json_atomic(progress_file, progress)
    
    return stats


def call_model_with_backoff(
    client: OpenAI,
    model: str,
    context: str,
    max_retries: int,
    base_delay: float,
) -> Tuple[List[QA], Dict]:
    """Call model with exponential backoff and specific error handling."""
    prompt = f"""Analyze the following text and generate diverse, specific question-answer pairs.

Text: \"{context[:2000]}\"  # Truncate if too long

Requirements:
1. Each question must be answerable from the text
2. Questions should vary in complexity and type (what, why, how, compare)
3. Answers must be concise and accurate
4. Return ONLY JSON with "qa_pairs" array

JSON Schema: {{\"qa_pairs\": [{{\"question\": \"...\", \"answer\": \"...\"}}]}}"""

    for attempt in range(1, max_retries + 1):
        try:
            # Adjust timeout based on context length
            timeout = min(180.0, 30.0 + len(context) / 100)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise QA generation assistant. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},  # Simpler format
                temperature=0.4,  # Lower for consistency
                top_p=0.95,
                timeout=timeout,
            )
            
            if not response.choices:
                return [], {}
            
            content = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            usage_dict = {
                "total_tokens": getattr(usage, "total_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            }
            
            # Parse JSON and validate
            data = json.loads(content)
            qa_resp = QAResponse.model_validate(data)
            return qa_resp.qa_pairs, usage_dict
            
        except (ValidationError, json.JSONDecodeError) as e:
            logging.warning(f"Parse error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(base_delay * attempt)  # Linear backoff
                continue
            return [], {}
            
        except RateLimitError:
            # Exponential backoff for rate limits
            wait = base_delay * (2 ** attempt)
            logging.warning(f"Rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
            
        except (APIConnectionError, APITimeoutError) as e:
            logging.warning(f"Connection error (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(base_delay * attempt)
                continue
            return [], {}
            
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt < max_retries:
                time.sleep(base_delay)
                continue
            return [], {}
    
    return [], {}


# --- Main Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Parallel Q&A Generator")
    parser.add_argument("--config", type=str, default="config/qa_generator.json")
    parser.add_argument("--input_dir", type=str, help="Override input dir")
    parser.add_argument("--output_dir", type=str, help="Override output dir")
    parser.add_argument("--model", type=str, help="Override model")
    parser.add_argument("--base_url", type=str, help="Override base URL")
    parser.add_argument("--api_key", type=str, help="Override API key")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    args = parser.parse_args()

    # Load config
    cfg = load_json(Path(args.config), {})
    
    input_dir = Path(args.input_dir or cfg.get("input_dir", "grobid_proccessed_pdf"))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "data/qa_outputs/jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model or cfg.get("model", "qwen2.5:32b-q5_k_m")
    base_url = args.base_url or cfg.get("base_url", "http://localhost:11434/v1")
    api_key = args.api_key or cfg.get("api_key", "ollama")
    
    max_retries = int(cfg.get("max_retries", 3))
    api_delay = float(cfg.get("api_delay", 0.5))
    num_workers = args.workers or int(cfg.get("workers", 4))
    
    per_file_dir = output_dir / cfg.get("per_file_output_dir", "qa_by_file")
    progress_file = output_dir / cfg.get("progress_file", "progress.json")
    log_file = output_dir / cfg.get("log_file", "qa_generation.log")
    failed_records_file = output_dir / cfg.get("failed_records_file", "failed_paragraphs.jsonl")
    
    setup_logging(log_file)
    logging.info("Starting parallel QA generation with %d workers", num_workers)
    
    if not input_dir.exists():
        logging.error(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    # Get all files
    all_files = list(input_dir.glob("*.txt"))
    if not all_files:
        logging.error(f"No .txt files found in '{input_dir}'.")
        sys.exit(1)
    
    logging.info(f"Found {len(all_files)} files to process")
    
    # Prepare arguments for workers
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    worker_args = [
        (
            file_path,
            model_name,
            base_url,
            api_key,
            max_retries,
            api_delay,
            run_id,
            per_file_dir,
            progress_file,
            failed_records_file,
        )
        for file_path in all_files
    ]
    
    # Process files in parallel
    overall_stats = {
        "total_files": len(all_files),
        "completed_files": 0,
        "total_paragraphs_attempted": 0,
        "total_qa_records": 0,
        "total_failures": 0,
    }
    
    with tqdm(total=len(all_files), desc="Files", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_worker, *args): args[0]
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    stats = future.result()
                    overall_stats["completed_files"] += 1
                    overall_stats["total_paragraphs_attempted"] += stats["attempted"]
                    overall_stats["total_qa_records"] += stats["qa_records"]
                    overall_stats["total_failures"] += stats["failed"]
                    
                    pbar.set_postfix({
                        "QA": stats["qa_records"],
                        "Fails": stats["failed"]
                    })
                    pbar.update(1)
                    
                    logging.info(
                        f"Completed {stats['file']}: "
                        f"{stats['qa_records']} QA pairs, "
                        f"{stats['failed']} failures"
                    )
                    
                except Exception as e:
                    logging.error(f"Worker failed for {file_path}: {e}")
                    pbar.update(1)
    
    # Save final summary
    summary_file = output_dir / "run_summary.json"
    save_json_atomic(summary_file, {
        "run_id": run_id,
        "config": cfg,
        "overall_stats": overall_stats,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    })
    
    logging.info("Run complete. Summary: %s", overall_stats)
    print(f"\n✅ Done! Processed {overall_stats['completed_files']} files.")
    print(f"📊 Generated {overall_stats['total_qa_records']} QA pairs.")
    print(f"📁 Output: {output_dir}")
    print(f"📋 Summary: {summary_file}")


if __name__ == "__main__":
    main()