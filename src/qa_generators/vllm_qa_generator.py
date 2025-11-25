#!/usr/bin/env python3
"""
vLLM Batch QA Generator - GPU 2 Isolation
Optimized for 40k files with persistent progress and crash recovery.
Reads your existing config file.
"""
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
import psutil

# --- Configuration ---
class QA(BaseModel):
    question: str
    answer: str

class QAResponse(BaseModel):
    qa_pairs: List[QA]

# --- GPU Utilities ---
def get_gpu_2_status():
    """Get RTX 3090 Ti (GPU 2) status."""
    try:
        import nvidia-ml-py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(2)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        return {
            "memory_used_gb": mem.used / 1024**3,
            "memory_total_gb": mem.total / 1024**3,
            "utilization_pct": util.gpu,
            "temperature_c": temp
        }
    except:
        return None

def log_gpu_status():
    """Log GPU 2 stats every 100 requests."""
    stats = get_gpu_2_status()
    if stats:
        logging.info(
            f"GPU 2 | VRAM: {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB "
            f"({stats['utilization_pct']}%), Temp: {stats['temperature_c']}°C"
        )

# --- Utilities ---
def setup_logging(log_file: Path):
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
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}. Using defaults.")
            return default
    return default

def save_json_atomic(path: Path, data) -> None:
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
    if not paragraph or len(paragraph) < 50:
        return False
    exclude_prefixes = [
        "acknowledgments:", "funding:", "data availability statement:",
        "author contributions:", "references:", "declarations:",
        "acknowledgements:", "acknowledgment:",
    ]
    first_line = paragraph.split('\n')[0].strip().lower()
    if any(first_line.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if 'doi:' in paragraph.lower() or 'http' in paragraph.lower():
        return False
    return True

def validate_qa_pairs(qa_pairs: List[QA]) -> List[QA]:
    valid = []
    for qa in qa_pairs:
        q_clean = qa.question.strip()
        a_clean = qa.answer.strip()
        if (q_clean and a_clean and 
            len(q_clean) > 15 and len(a_clean) > 20 and
            not q_clean.startswith(('?', 'What?', 'How?'))):
            valid.append(QA(question=q_clean, answer=a_clean))
    return valid

def iter_paragraphs(file_path: Path) -> Iterable[Tuple[str, int, int]]:
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
class QAProcessor:
    def __init__(self, config: Dict):
        # Persistent client with connection pooling
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=300.0,
            max_retries=0,
        )
        self.model_name = config["model"]
        self.max_retries = config["max_retries"]
        self.api_delay = config["api_delay"]
        self.run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.request_count = 0
        self.config = config
        
    def call_model(self, paragraph: str) -> Tuple[List[QA], Dict]:
        prompt = f"""Generate diverse, specific QA pairs from this text:

Text: \"{paragraph[:2000]}\"

Requirements:
1. Questions must be answerable from text
2. Vary complexity (what, why, how, compare)
3. Answers concise and accurate
4. Return ONLY JSON: {{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}"""

        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    sleep_time = self.api_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_time)
                
                # GPU monitoring
                self.request_count += 1
                if self.request_count % 100 == 0:
                    log_gpu_status()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise QA generation assistant. Return only valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    top_p=0.95,
                    max_tokens=1024,
                    extra_body={
                        "repetition_penalty": 1.05,
                        "skip_special_tokens": True,
                    }
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
                
                data = json.loads(content)
                qa_resp = QAResponse.model_validate(data)
                return qa_resp.qa_pairs, usage_dict
                
            except (ValidationError, json.JSONDecodeError) as e:
                logging.warning(f"Parse error (attempt {attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    return [], {}
                    
            except RateLimitError:
                wait = self.api_delay * (2 ** attempt)
                logging.warning(f"Rate limited, waiting {wait}s")
                time.sleep(wait)
                
            except (APIConnectionError, APITimeoutError) as e:
                logging.warning(f"Connection error (attempt {attempt}): {e}")
                if attempt == self.max_retries:
                    return [], {}
                    
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                if attempt == self.max_retries:
                    return [], {}
        
        return [], {}

def process_file(
    file_path: Path,
    processor: QAProcessor,
    config: Dict,
) -> Dict:
    fname = file_path.name
    progress_file = Path(config["output_dir"]) / config["progress_file"]
    per_file_dir = Path(config["output_dir"]) / config["per_file_output_dir"]
    failed_records_file = Path(config["output_dir"]) / config["failed_records_file"]
    
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
                if p_idx < start_paragraph or (p_idx == start_paragraph and para_line <= start_line):
                    continue
                
                if not is_valid_paragraph(paragraph):
                    stats["filtered"] += 1
                    if p_idx % 50 == 0:
                        progress[fname] = {"paragraph_index": p_idx, "line_number": para_line, "completed": False}
                        save_json_atomic(progress_file, progress)
                    continue
                
                stats["attempted"] += 1
                qa_pairs, usage = processor.call_model(paragraph)
                qa_pairs = validate_qa_pairs(qa_pairs)
                
                if not qa_pairs:
                    stats["failed"] += 1
                    failure_record = {
                        "file_name": fname,
                        "file_path": str(file_path.resolve()),
                        "paragraph_index": p_idx,
                        "line_number": para_line,
                        "context": paragraph[:500],
                        "model_attempted": processor.model_name,
                        "run_id": processor.run_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    fail_f.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                else:
                    stats["success"] += 1
                    stats["qa_records"] += len(qa_pairs)
                    ts = datetime.utcnow().isoformat() + "Z"
                    for qa in qa_pairs:
                        record = {
                            "id": str(uuid.uuid4()),
                            "run_id": processor.run_id,
                            "created_at": ts,
                            "model": processor.model_name,
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
                
                # Periodic flush and progress save
                if p_idx % 20 == 0:
                    out_f.flush()
                if p_idx % 50 == 0:
                    progress[fname] = {"paragraph_index": p_idx, "line_number": para_line, "completed": False}
                    save_json_atomic(progress_file, progress)
                
                # Memory check
                if p_idx % 100 == 0:
                    mem = psutil.virtual_memory()
                    if mem.percent > 85:
                        logging.warning(f"Memory usage high: {mem.percent}%")
                
                # Circuit breaker
                if stats["attempted"] > 20 and stats["failed"] / stats["attempted"] > 0.3:
                    logging.warning(f"Circuit breaker triggered for {fname}: >30% failure rate")
                    break
                
    except Exception as e:
        logging.error(f"Fatal error processing {fname}: {e}", exc_info=True)
    
    # Mark file as completed
    progress[fname] = {"paragraph_index": p_idx, "line_number": para_line, "completed": True}
    save_json_atomic(progress_file, progress)
    
    return stats

# --- Main Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="vLLM Batch QA Generator")
    parser.add_argument("--config", type=str, default="config/qa_generator.json")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_json(config_path, {})
    
    # Override with absolute paths
    config["input_dir"] = str(Path(config["input_dir"]).resolve())
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    
    # Setup paths
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    log_file = Path(config["output_dir"]) / config["log_file"]
    setup_logging(log_file)
    
    # Model mapping: Ollama -> vLLM
    model_mapping = {
        "qwen2.5:32b-q5_k_m": "qwen/Qwen2.5-32B-Instruct-AWQ",
        "qwen2.5:32b": "qwen/Qwen2.5-32B-Instruct",
    }
    if config["model"] in model_mapping:
        config["model"] = model_mapping[config["model"]]
        logging.info(f"Mapped Ollama model to vLLM: {config['model']}")
    
    logging.info("="*60)
    logging.info(f"🚀 Starting vLLM batch QA generation")
    logging.info(f"📁 Input: {config['input_dir']}")
    logging.info(f"💾 Output: {config['output_dir']}")
    logging.info(f"🎯 Target GPU: 2 (RTX 3090 Ti)")
    logging.info(f"🧠 Model: {config['model']}")
    logging.info(f"👷 Workers: {config['workers']}")
    logging.info("="*60)
    
    input_dir = Path(config["input_dir"])
    if not input_dir.exists():
        logging.error(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    all_files = sorted(list(input_dir.glob("*.txt")))
    if not all_files:
        logging.error(f"No .txt files found in '{input_dir}'.")
        sys.exit(1)
    
    # Resume logic
    if args.resume:
        progress_file = Path(config["output_dir"]) / config["progress_file"]
        progress = load_json(progress_file, {})
        completed_files = {k for k, v in progress.items() if v.get("completed", False)}
        all_files = [f for f in all_files if f.name not in completed_files]
        logging.info(f"📌 Resuming: {len(completed_files)} files already completed")
    
    logging.info(f"📄 Processing {len(all_files)} files...")
    
    # Initialize processor
    processor = QAProcessor(config)
    
    overall_stats = {
        "run_id": processor.run_id,
        "config": config,
        "total_files": len(all_files),
        "completed_files": 0,
        "total_paragraphs_attempted": 0,
        "total_qa_records": 0,
        "total_failures": 0,
        "start_time": time.time(),
    }
    
    # Process with ThreadPoolExecutor
    with tqdm(total=len(all_files), desc="Files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            future_to_file = {}
            
            # Submit files in batches
            batch_size = config["workers"] * 2
            for i in range(0, len(all_files), batch_size):
                batch = all_files[i:i + batch_size]
                
                for file_path in batch:
                    future = executor.submit(
                        process_file,
                        file_path=file_path,
                        processor=processor,
                        config=config,
                    )
                    future_to_file[future] = file_path
                
                # Wait for batch completion
                for future in as_completed(future_to_file):
                    file_path = future_to_file.pop(future)
                    try:
                        stats = future.result()
                        overall_stats["completed_files"] += 1
                        overall_stats["total_paragraphs_attempted"] += stats["attempted"]
                        overall_stats["total_qa_records"] += stats["qa_records"]
                        overall_stats["total_failures"] += stats["failed"]
                        
                        pbar.set_postfix({
                            "QA": overall_stats["total_qa_records"],
                            "Fails": overall_stats["total_failures"],
                            "Mem": f"{psutil.virtual_memory().percent:.0f}%"
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        logging.error(f"Worker failed for {file_path}: {e}", exc_info=True)
                        pbar.update(1)
                
                # Brief pause between batches
                time.sleep(0.5)
    
    # Final summary
    overall_stats["duration_seconds"] = time.time() - overall_stats["start_time"]
    overall_stats["completed_at"] = datetime.utcnow().isoformat() + "Z"
    
    summary_file = Path(config["output_dir"]) / "run_summary.json"
    save_json_atomic(summary_file, overall_stats)
    
    logging.info("="*60)
    logging.info("🏁 RUN COMPLETE")
    logging.info(f"⏱️  Duration: {overall_stats['duration_seconds']:.1f}s ({overall_stats['duration_seconds']/3600:.1f}h)")
    logging.info(f"📊 QA Pairs: {overall_stats['total_qa_records']:,}")
    logging.info(f"❌ Failures: {overall_stats['total_failures']:,}")
    logging.info("="*60)
    
    print(f"\n✅ Done! Processed {overall_stats['completed_files']} files.")
    print(f"📊 Generated {overall_stats['total_qa_records']:,} QA pairs.")
    print(f"⏱️  Duration: {overall_stats['duration_seconds']/60:.1f} minutes")
    print(f"📁 Output: {config['output_dir']}")
    print(f"📋 Summary: {summary_file}")

if __name__ == "__main__":
    main()