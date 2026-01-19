#!/usr/bin/env python3
"""
Compare base model vs fine-tuned model performance.

This script evaluates both models on a held-out test set and produces
a detailed comparison report.

Usage:
    python scripts/compare_models.py \
        --base sentence-transformers/all-MiniLM-L6-v2 \
        --finetuned models/fine_tuned_embeddings/final_model \
        --jsonl consolidated_qa.jsonl \
        --output eval_results/

    # Quick test with smaller sample
    python scripts/compare_models.py \
        --base sentence-transformers/all-MiniLM-L6-v2 \
        --finetuned models/fine_tuned_embeddings/final_model \
        --jsonl consolidated_qa.jsonl \
        --sample_size 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_trainers.data_utils import DataConfig, load_split_samples
from src.embedding_trainers.evaluators import (
    compare_models,
    evaluate_model,
    ManualInspectionEvaluator,
)
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_manual_inspection(
    base_model_path: str,
    finetuned_model_path: str,
    jsonl_path: str,
    sample_queries: list = None
):
    """
    Run manual inspection to compare search results qualitatively.
    """
    # Load models
    logger.info("Loading models for manual inspection...")
    base_model = SentenceTransformer(base_model_path)
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    # Load corpus from test set
    config = DataConfig()
    data = load_split_samples(jsonl_path, "test", sample_size=1000, config=config)
    corpus = list(data['corpus'].values())
    
    # Default sample queries
    if sample_queries is None:
        sample_queries = [
            "What is the mechanism of action?",
            "What are the main findings of this study?",
            "What methods were used in the experiment?",
            "What are the limitations of this approach?",
            "How does temperature affect the results?",
        ]
    
    print("\n" + "=" * 80)
    print("MANUAL INSPECTION: Side-by-side comparison")
    print("=" * 80)
    
    for query in sample_queries:
        print(f"\n{'#' * 80}")
        print(f"QUERY: {query}")
        print(f"{'#' * 80}")
        
        # Base model results
        print("\n--- BASE MODEL Results ---")
        base_inspector = ManualInspectionEvaluator(base_model, corpus)
        base_results = base_inspector.search(query, top_k=3)
        for r in base_results:
            text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
            print(f"  {r['rank']}. [{r['score']:.4f}] {text}")
        
        # Fine-tuned model results
        print("\n--- FINE-TUNED MODEL Results ---")
        ft_inspector = ManualInspectionEvaluator(finetuned_model, corpus)
        ft_results = ft_inspector.search(query, top_k=3)
        for r in ft_results:
            text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
            print(f"  {r['rank']}. [{r['score']:.4f}] {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model vs fine-tuned model"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name or path"
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default="models/fine_tuned_embeddings/final_model",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="consolidated_qa.jsonl",
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Number of test samples to evaluate on"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Run manual inspection (qualitative comparison)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/embedding_finetuner.json",
        help="Config file for data filtering settings"
    )
    
    args = parser.parse_args()
    
    # Check if finetuned model exists
    finetuned_path = Path(args.finetuned)
    if not finetuned_path.exists():
        logger.error(f"Fine-tuned model not found: {finetuned_path}")
        logger.info("Please train a model first using:")
        logger.info("  python -m src.embedding_trainers.streaming_finetuner --config config/embedding_finetuner.json")
        sys.exit(1)
    
    # Check if JSONL exists
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        sys.exit(1)
    
    # Load data config
    config = DataConfig()
    if Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                cfg_data = json.load(f)
            config = DataConfig(
                min_question_length=cfg_data.get('min_question_length', 10),
                max_question_length=cfg_data.get('max_question_length', 256),
                min_answer_length=cfg_data.get('min_answer_length', 20),
                max_answer_length=cfg_data.get('max_answer_length', 512),
                clean_answers=cfg_data.get('clean_answers', True),
                train_ratio=cfg_data.get('train_ratio', 0.85),
                val_ratio=cfg_data.get('val_ratio', 0.10),
            )
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run quantitative comparison
    logger.info("=" * 70)
    logger.info("QUANTITATIVE EVALUATION")
    logger.info("=" * 70)
    
    results = compare_models(
        base_model_path=args.base,
        finetuned_model_path=args.finetuned,
        jsonl_path=args.jsonl,
        sample_size=args.sample_size,
        config=config,
        output_dir=args.output,
    )
    
    # Run manual inspection if requested
    if args.manual:
        run_manual_inspection(
            args.base,
            args.finetuned,
            args.jsonl,
        )
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir / 'model_comparison.json'}")
    print("\nKey takeaways:")
    
    base_recall = results['base'].recall_at_10
    ft_recall = results['finetuned'].recall_at_10
    improvement = ((ft_recall - base_recall) / base_recall * 100) if base_recall > 0 else 0
    
    if improvement > 5:
        print(f"  ✓ Fine-tuning improved Recall@10 by {improvement:.1f}%")
        print(f"  ✓ Model is ready for use: {args.finetuned}")
    elif improvement > 0:
        print(f"  ~ Modest improvement of {improvement:.1f}%")
        print(f"  ~ Consider training longer or with more data")
    else:
        print(f"  ✗ No improvement detected ({improvement:.1f}%)")
        print(f"  ✗ Check data quality or try different hyperparameters")
    
    # Suggest next steps
    print("\nNext steps:")
    print("  1. Review model_comparison.json for detailed metrics")
    if not args.manual:
        print("  2. Run with --manual flag for qualitative inspection")
    print("  3. Test with domain-specific queries from your use case")


if __name__ == "__main__":
    main()
