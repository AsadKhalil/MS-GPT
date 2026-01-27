#!/usr/bin/env python3
"""
MSQA-Bench Full Pipeline Script.

Runs the complete benchmark generation and evaluation pipeline:
1. Extract metadata from documents
2. Enrich QA schema
3. Apply quality filters
4. Classify question types
5. Generate document-level splits
6. Evaluate retrieval baselines
7. Run RAG evaluation
8. Compute faithfulness metrics
9. Generate paper tables

Usage:
    python scripts/run_benchmark_pipeline.py --config config/benchmark_config.json
    python scripts/run_benchmark_pipeline.py --step enrich  # Run specific step
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from paper.dataset.metadata_extractor import batch_extract_metadata
from paper.dataset.schema_enricher import enrich_qa_file
from paper.dataset.quality_filters import run_quality_pipeline, FilterConfig
from paper.dataset.question_classifier import classify_qa_file
from paper.dataset.split_generator import split_qa_file, verify_no_leakage
from paper.annotation.gold_set_sampler import sample_for_annotation, generate_annotation_guidelines
from paper.figures.generate_tables import generate_all_tables, compute_dataset_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with config_path.open('r') as f:
        return json.load(f)


def step_extract_metadata(config: Dict[str, Any]) -> None:
    """Step 1: Extract metadata from documents."""
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting metadata from documents")
    logger.info("=" * 60)
    
    text_dir = Path(config['dataset']['text_dir'])
    output_file = Path(config['dataset']['output_dir']) / "raw" / "metadata.jsonl"
    
    if not text_dir.exists():
        logger.warning(f"Text directory not found: {text_dir}")
        return
    
    batch_extract_metadata(
        text_dir,
        output_file,
        use_api=config.get('metadata', {}).get('use_api', False),
        workers=4,
    )
    
    logger.info(f"Metadata saved to: {output_file}")


def step_enrich_schema(config: Dict[str, Any]) -> None:
    """Step 2: Enrich QA schema with benchmark fields."""
    logger.info("=" * 60)
    logger.info("STEP 2: Enriching QA schema")
    logger.info("=" * 60)
    
    input_file = Path(config['dataset']['input_qa_file'])
    output_dir = Path(config['dataset']['output_dir'])
    text_dir = Path(config['dataset']['text_dir'])
    
    metadata_file = output_dir / "raw" / "metadata.jsonl"
    output_file = output_dir / "processed" / "enriched_qa.jsonl"
    public_file = output_dir / "release" / "msqa_bench_public.jsonl"
    
    stats = enrich_qa_file(
        input_file,
        output_file,
        text_dir=text_dir if text_dir.exists() else None,
        metadata_file=metadata_file if metadata_file.exists() else None,
        public_output=public_file,
    )
    
    logger.info(f"Enrichment stats: {stats}")
    logger.info(f"Enriched file: {output_file}")
    logger.info(f"Public release file: {public_file}")


def step_quality_filter(config: Dict[str, Any]) -> None:
    """Step 3: Apply quality filters."""
    logger.info("=" * 60)
    logger.info("STEP 3: Applying quality filters")
    logger.info("=" * 60)
    
    output_dir = Path(config['dataset']['output_dir'])
    input_file = output_dir / "processed" / "enriched_qa.jsonl"
    output_file = output_dir / "processed" / "filtered_qa.jsonl"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    filter_config = FilterConfig(
        min_question_length=config['dataset']['quality_filters']['min_question_length'],
        max_question_length=config['dataset']['quality_filters']['max_question_length'],
        min_answer_length=config['dataset']['quality_filters']['min_answer_length'],
        max_answer_length=config['dataset']['quality_filters']['max_answer_length'],
        min_quality_score=config['dataset']['quality_filters']['min_quality_score'],
    )
    
    stats = run_quality_pipeline(
        input_file,
        output_file,
        config=filter_config,
        deduplicate=config['dataset']['quality_filters']['deduplicate'],
    )
    
    logger.info(f"Filter stats: {stats}")
    logger.info(f"Filtered file: {output_file}")


def step_classify_questions(config: Dict[str, Any]) -> None:
    """Step 4: Classify question types."""
    logger.info("=" * 60)
    logger.info("STEP 4: Classifying question types")
    logger.info("=" * 60)
    
    output_dir = Path(config['dataset']['output_dir'])
    input_file = output_dir / "processed" / "filtered_qa.jsonl"
    output_file = output_dir / "processed" / "classified_qa.jsonl"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    type_counts = classify_qa_file(input_file, output_file)
    
    logger.info(f"Question type distribution: {type_counts}")
    logger.info(f"Classified file: {output_file}")


def step_generate_splits(config: Dict[str, Any]) -> None:
    """Step 5: Generate document-level splits."""
    logger.info("=" * 60)
    logger.info("STEP 5: Generating document-level splits")
    logger.info("=" * 60)
    
    output_dir = Path(config['dataset']['output_dir'])
    input_file = output_dir / "processed" / "classified_qa.jsonl"
    splits_dir = output_dir / "splits"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    split_config = config['dataset']['splits']
    
    stats = split_qa_file(
        input_file,
        splits_dir,
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio'],
    )
    
    # Verify no leakage
    no_leakage = verify_no_leakage(
        splits_dir / "train.jsonl",
        splits_dir / "val.jsonl",
        splits_dir / "test.jsonl",
    )
    
    if not no_leakage:
        logger.error("DATA LEAKAGE DETECTED! Check split generation.")
    
    logger.info(f"Split stats: {stats}")
    logger.info(f"Split files in: {splits_dir}")


def step_sample_annotation(config: Dict[str, Any]) -> None:
    """Step 6: Sample QA pairs for human annotation."""
    logger.info("=" * 60)
    logger.info("STEP 6: Sampling for human annotation")
    logger.info("=" * 60)
    
    output_dir = Path(config['dataset']['output_dir'])
    
    # Sample from test set for annotation
    input_file = output_dir / "splits" / "test.jsonl"
    if not input_file.exists():
        input_file = output_dir / "processed" / "classified_qa.jsonl"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    annotation_config = config.get('annotation', {})
    
    output_file = output_dir / "gold" / f"annotation_sample.{annotation_config.get('output_format', 'csv')}"
    
    sampled = sample_for_annotation(
        input_file,
        output_file,
        target_size=annotation_config.get('target_size', 500),
        format=annotation_config.get('output_format', 'csv'),
    )
    
    # Generate guidelines
    guidelines_file = Path(annotation_config.get('guidelines_file', output_dir / "gold" / "annotation_guidelines.md"))
    generate_annotation_guidelines(guidelines_file)
    
    logger.info(f"Sampled {len(sampled)} records for annotation")
    logger.info(f"Annotation file: {output_file}")
    logger.info(f"Guidelines: {guidelines_file}")


def step_generate_tables(config: Dict[str, Any]) -> None:
    """Step 7: Generate paper tables."""
    logger.info("=" * 60)
    logger.info("STEP 7: Generating paper tables")
    logger.info("=" * 60)
    
    output_dir = Path(config['dataset']['output_dir'])
    paper_config = config.get('paper', {})
    
    # Use the most processed file available
    qa_file = None
    for candidate in [
        output_dir / "splits" / "train.jsonl",
        output_dir / "processed" / "classified_qa.jsonl",
        output_dir / "processed" / "filtered_qa.jsonl",
        Path(config['dataset']['input_qa_file']),
    ]:
        if candidate.exists():
            qa_file = candidate
            break
    
    if qa_file is None:
        logger.error("No QA file found for table generation")
        return
    
    tables_dir = Path(paper_config.get('tables_dir', 'paper/figures/tables'))
    
    tables = generate_all_tables(
        qa_file,
        output_dir=tables_dir,
    )
    
    logger.info(f"Generated {len(tables)} tables in: {tables_dir}")


def run_full_pipeline(config: Dict[str, Any]) -> None:
    """Run the complete pipeline."""
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("MSQA-BENCH PIPELINE STARTING")
    logger.info(f"Time: {start_time}")
    logger.info("=" * 70)
    
    steps = [
        ("extract_metadata", step_extract_metadata),
        ("enrich", step_enrich_schema),
        ("filter", step_quality_filter),
        ("classify", step_classify_questions),
        ("split", step_generate_splits),
        ("annotate", step_sample_annotation),
        ("tables", step_generate_tables),
    ]
    
    for step_name, step_func in steps:
        try:
            step_func(config)
        except Exception as e:
            logger.error(f"Step '{step_name}' failed: {e}")
            logger.exception(e)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 70)
    logger.info("MSQA-BENCH PIPELINE COMPLETE")
    logger.info(f"Duration: {duration}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MSQA-Bench Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/benchmark_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--step", "-s",
        type=str,
        choices=[
            "metadata", "enrich", "filter", "classify", 
            "split", "annotate", "tables", "all"
        ],
        default="all",
        help="Run specific step or 'all'"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Run specified step or full pipeline
    step_mapping = {
        "metadata": step_extract_metadata,
        "enrich": step_enrich_schema,
        "filter": step_quality_filter,
        "classify": step_classify_questions,
        "split": step_generate_splits,
        "annotate": step_sample_annotation,
        "tables": step_generate_tables,
    }
    
    if args.step == "all":
        run_full_pipeline(config)
    else:
        step_func = step_mapping.get(args.step)
        if step_func:
            step_func(config)
        else:
            logger.error(f"Unknown step: {args.step}")
            sys.exit(1)


if __name__ == "__main__":
    main()
