"""
Paper Table Generator for MSQA-Bench.

Generates LaTeX tables for:
1. Dataset statistics
2. Retrieval comparison results
3. RAG evaluation results
4. Faithfulness metrics
5. Human annotation results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for the dataset."""
    total_documents: int = 0
    total_qa_pairs: int = 0
    
    # Splits
    train_docs: int = 0
    train_pairs: int = 0
    val_docs: int = 0
    val_pairs: int = 0
    test_docs: int = 0
    test_pairs: int = 0
    
    # Question types
    question_types: Dict[str, int] = None
    
    # Answer styles
    extractive_count: int = 0
    abstractive_count: int = 0
    
    # Metadata coverage
    with_doi: int = 0
    open_access: int = 0
    
    # Quality
    avg_quality_score: float = 0.0
    avg_question_length: float = 0.0
    avg_answer_length: float = 0.0
    avg_context_length: float = 0.0
    
    def __post_init__(self):
        if self.question_types is None:
            self.question_types = {}


def compute_dataset_stats(qa_file: Path) -> DatasetStats:
    """
    Compute dataset statistics from a QA file.
    
    Args:
        qa_file: JSONL file with QA records
        
    Returns:
        DatasetStats object
    """
    stats = DatasetStats()
    
    doc_ids = set()
    train_docs = set()
    val_docs = set()
    test_docs = set()
    
    question_types = Counter()
    quality_scores = []
    question_lengths = []
    answer_lengths = []
    context_lengths = []
    
    with qa_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            stats.total_qa_pairs += 1
            
            doc_id = record.get('doc_id', record.get('file_name', ''))
            doc_ids.add(doc_id)
            
            # Split tracking
            split = record.get('split', 'train')
            if split == 'train':
                stats.train_pairs += 1
                train_docs.add(doc_id)
            elif split == 'val':
                stats.val_pairs += 1
                val_docs.add(doc_id)
            elif split == 'test':
                stats.test_pairs += 1
                test_docs.add(doc_id)
            
            # Question types
            qtype = record.get('question_type', 'unknown')
            question_types[qtype] += 1
            
            # Answer style
            style = record.get('answer_style', 'unknown')
            if style == 'extractive':
                stats.extractive_count += 1
            elif style == 'abstractive':
                stats.abstractive_count += 1
            
            # Metadata coverage
            if record.get('doi'):
                stats.with_doi += 1
            if record.get('license') == 'open_access':
                stats.open_access += 1
            
            # Quality and lengths
            if 'quality_score' in record:
                quality_scores.append(record['quality_score'])
            
            question = record.get('question', '')
            answer = record.get('answer', '')
            context = record.get('context', '')
            
            question_lengths.append(len(question))
            answer_lengths.append(len(answer))
            context_lengths.append(len(context))
    
    stats.total_documents = len(doc_ids)
    stats.train_docs = len(train_docs)
    stats.val_docs = len(val_docs)
    stats.test_docs = len(test_docs)
    stats.question_types = dict(question_types)
    
    if quality_scores:
        stats.avg_quality_score = sum(quality_scores) / len(quality_scores)
    if question_lengths:
        stats.avg_question_length = sum(question_lengths) / len(question_lengths)
    if answer_lengths:
        stats.avg_answer_length = sum(answer_lengths) / len(answer_lengths)
    if context_lengths:
        stats.avg_context_length = sum(context_lengths) / len(context_lengths)
    
    return stats


def generate_dataset_stats_table(
    stats: DatasetStats,
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate LaTeX table for dataset statistics.
    
    Args:
        stats: DatasetStats object
        output_file: Optional file to save table
        
    Returns:
        LaTeX table string
    """
    latex = r"""
\begin{table}[t]
\centering
\caption{MSQA-Bench Dataset Statistics}
\label{tab:dataset_stats}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Overall}} \\
Total Documents & """ + f"{stats.total_documents:,}" + r""" \\
Total QA Pairs & """ + f"{stats.total_qa_pairs:,}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Splits}} \\
Train (docs / pairs) & """ + f"{stats.train_docs:,} / {stats.train_pairs:,}" + r""" \\
Validation (docs / pairs) & """ + f"{stats.val_docs:,} / {stats.val_pairs:,}" + r""" \\
Test (docs / pairs) & """ + f"{stats.test_docs:,} / {stats.test_pairs:,}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Metadata Coverage}} \\
With DOI & """ + f"{stats.with_doi:,} ({stats.with_doi/stats.total_qa_pairs*100:.1f}\\%)" + r""" \\
Open Access & """ + f"{stats.open_access:,} ({stats.open_access/stats.total_qa_pairs*100:.1f}\\%)" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Answer Style}} \\
Extractive & """ + f"{stats.extractive_count:,} ({stats.extractive_count/stats.total_qa_pairs*100:.1f}\\%)" + r""" \\
Abstractive & """ + f"{stats.abstractive_count:,} ({stats.abstractive_count/stats.total_qa_pairs*100:.1f}\\%)" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Average Lengths (chars)}} \\
Question & """ + f"{stats.avg_question_length:.0f}" + r""" \\
Answer & """ + f"{stats.avg_answer_length:.0f}" + r""" \\
Context & """ + f"{stats.avg_context_length:.0f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
        logger.info(f"Saved dataset stats table to {output_file}")
    
    return latex


def generate_question_type_table(
    stats: DatasetStats,
    output_file: Optional[Path] = None,
) -> str:
    """Generate LaTeX table for question type distribution."""
    
    total = sum(stats.question_types.values())
    
    rows = []
    for qtype, count in sorted(stats.question_types.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        rows.append(f"{qtype.capitalize()} & {count:,} & {pct:.1f}\\%")
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Question Type Distribution}
\label{tab:question_types}
\begin{tabular}{lrr}
\toprule
\textbf{Question Type} & \textbf{Count} & \textbf{Percentage} \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\midrule
Total & """ + f"{total:,}" + r""" & 100.0\% \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
    
    return latex


def generate_retrieval_results_table(
    results: Dict[str, Dict[str, float]],
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate LaTeX table for retrieval comparison.
    
    Args:
        results: Dict mapping model name to metrics dict
        output_file: Optional file to save table
        
    Returns:
        LaTeX table string
    """
    rows = []
    for model_name, metrics in results.items():
        row = (
            f"{model_name} & "
            f"{metrics.get('recall@1', 0):.3f} & "
            f"{metrics.get('recall@5', 0):.3f} & "
            f"{metrics.get('recall@10', 0):.3f} & "
            f"{metrics.get('mrr@10', 0):.3f} & "
            f"{metrics.get('ndcg@10', 0):.3f}"
        )
        rows.append(row)
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Retrieval Performance Comparison}
\label{tab:retrieval_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR@10} & \textbf{NDCG@10} \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
        logger.info(f"Saved retrieval results table to {output_file}")
    
    return latex


def generate_faithfulness_table(
    results: Dict[str, Dict[str, float]],
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate LaTeX table for faithfulness metrics.
    
    Args:
        results: Dict mapping method name to faithfulness metrics
        output_file: Optional file to save table
        
    Returns:
        LaTeX table string
    """
    rows = []
    for method_name, metrics in results.items():
        row = (
            f"{method_name} & "
            f"{metrics.get('avg_faithfulness_score', 0):.3f} & "
            f"{metrics.get('avg_citation_precision', 0):.3f} & "
            f"{metrics.get('avg_citation_recall', 0):.3f} & "
            f"{metrics.get('hallucination_rate', 0):.3f} & "
            f"{metrics.get('abstention_rate', 0):.3f}"
        )
        rows.append(row)
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Faithfulness Evaluation Results}
\label{tab:faithfulness}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Faith.} & \textbf{Cite-P} & \textbf{Cite-R} & \textbf{Halluc.} & \textbf{Abstain} \\
\midrule
""" + " \\\\\n".join(rows) + r""" \\
\bottomrule
\end{tabular}
\vspace{1mm}
\small
Faith.=Faithfulness Score, Cite-P=Citation Precision, Cite-R=Citation Recall, 
Halluc.=Hallucination Rate, Abstain=Abstention Rate
\end{table}
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
        logger.info(f"Saved faithfulness table to {output_file}")
    
    return latex


def generate_human_annotation_table(
    summary: Dict[str, Any],
    output_file: Optional[Path] = None,
) -> str:
    """
    Generate LaTeX table for human annotation results.
    
    Args:
        summary: Annotation summary dictionary
        output_file: Optional file to save table
        
    Returns:
        LaTeX table string
    """
    correctness = summary.get('answer_correctness', {})
    evidence = summary.get('evidence_support', {})
    quality = summary.get('evidence_quality', {})
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Human Evaluation Results (n=""" + str(summary.get('total_annotated', 0)) + r""")}
\label{tab:human_eval}
\begin{tabular}{lrrr}
\toprule
\textbf{Criterion} & \textbf{Yes/Good} & \textbf{Partial/Weak} & \textbf{No/Missing} \\
\midrule
Answer Correct & """ + f"{correctness.get('yes', 0)}" + r""" & """ + f"{correctness.get('partial', 0)}" + r""" & """ + f"{correctness.get('no', 0)}" + r""" \\
Evidence Support & """ + f"{evidence.get('yes', 0)}" + r""" & """ + f"{evidence.get('partial', 0)}" + r""" & """ + f"{evidence.get('no', 0)}" + r""" \\
Evidence Quality & """ + f"{quality.get('good', 0)}" + r""" & """ + f"{quality.get('weak', 0)}" + r""" & """ + f"{quality.get('missing', 0)}" + r""" \\
\midrule
\multicolumn{4}{l}{Answer Accuracy: """ + f"{correctness.get('accuracy', 0)*100:.1f}" + r"""\%} \\
\multicolumn{4}{l}{Inter-annotator Agreement (Kappa): """ + f"{summary.get('cohens_kappa', 'N/A')}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(latex)
        logger.info(f"Saved human annotation table to {output_file}")
    
    return latex


def generate_all_tables(
    qa_file: Path,
    retrieval_results: Optional[Dict[str, Dict]] = None,
    faithfulness_results: Optional[Dict[str, Dict]] = None,
    annotation_summary: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Generate all paper tables.
    
    Args:
        qa_file: JSONL file with QA data
        retrieval_results: Retrieval comparison results
        faithfulness_results: Faithfulness evaluation results
        annotation_summary: Human annotation summary
        output_dir: Directory to save tables
        
    Returns:
        Dict mapping table name to LaTeX string
    """
    tables = {}
    
    # Dataset statistics
    stats = compute_dataset_stats(qa_file)
    
    tables['dataset_stats'] = generate_dataset_stats_table(
        stats,
        output_dir / "table_dataset_stats.tex" if output_dir else None,
    )
    
    tables['question_types'] = generate_question_type_table(
        stats,
        output_dir / "table_question_types.tex" if output_dir else None,
    )
    
    # Retrieval results
    if retrieval_results:
        tables['retrieval'] = generate_retrieval_results_table(
            retrieval_results,
            output_dir / "table_retrieval.tex" if output_dir else None,
        )
    
    # Faithfulness results
    if faithfulness_results:
        tables['faithfulness'] = generate_faithfulness_table(
            faithfulness_results,
            output_dir / "table_faithfulness.tex" if output_dir else None,
        )
    
    # Human annotation
    if annotation_summary:
        tables['human_eval'] = generate_human_annotation_table(
            annotation_summary,
            output_dir / "table_human_eval.tex" if output_dir else None,
        )
    
    logger.info(f"Generated {len(tables)} tables")
    
    return tables


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--qa-file", "-q", required=True, help="QA JSONL file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--retrieval", "-r", help="Retrieval results JSON")
    parser.add_argument("--faithfulness", "-f", help="Faithfulness results JSON")
    parser.add_argument("--annotation", "-a", help="Annotation summary JSON")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load optional results
    retrieval_results = None
    if args.retrieval and Path(args.retrieval).exists():
        with open(args.retrieval) as f:
            retrieval_results = json.load(f)
    
    faithfulness_results = None
    if args.faithfulness and Path(args.faithfulness).exists():
        with open(args.faithfulness) as f:
            faithfulness_results = json.load(f)
    
    annotation_summary = None
    if args.annotation and Path(args.annotation).exists():
        with open(args.annotation) as f:
            annotation_summary = json.load(f)
    
    tables = generate_all_tables(
        Path(args.qa_file),
        retrieval_results=retrieval_results,
        faithfulness_results=faithfulness_results,
        annotation_summary=annotation_summary,
        output_dir=Path(args.output_dir),
    )
    
    print(f"\nGenerated {len(tables)} tables:")
    for name in tables:
        print(f"  - {name}")
