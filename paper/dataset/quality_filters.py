"""
Quality Filters for MSQA-Bench.

Implements automatic quality control filters:
1. Answer-context entailment checking
2. Near-duplicate detection (MinHash + embeddings)
3. Trivial question removal
4. Quality score computation
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    logger.warning("datasketch not installed. MinHash deduplication disabled.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class QualityMetrics:
    """Quality metrics for a QA pair."""
    # Answer-context relationship
    answer_context_overlap: float = 0.0  # Word overlap ratio
    answer_in_context: bool = False      # Is answer substring of context
    
    # Question quality
    question_length: int = 0
    question_specificity: float = 0.0    # How specific is the question
    is_trivial_question: bool = False
    
    # Answer quality
    answer_length: int = 0
    answer_completeness: float = 0.0     # Not truncated
    
    # Duplication
    is_near_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    # Overall score
    quality_score: float = 0.0
    
    # Flags
    filter_reasons: List[str] = field(default_factory=list)
    should_keep: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer_context_overlap': self.answer_context_overlap,
            'answer_in_context': self.answer_in_context,
            'question_length': self.question_length,
            'question_specificity': self.question_specificity,
            'is_trivial_question': self.is_trivial_question,
            'answer_length': self.answer_length,
            'answer_completeness': self.answer_completeness,
            'is_near_duplicate': self.is_near_duplicate,
            'duplicate_of': self.duplicate_of,
            'quality_score': self.quality_score,
            'filter_reasons': self.filter_reasons,
            'should_keep': self.should_keep,
        }


@dataclass
class FilterConfig:
    """Configuration for quality filters."""
    # Length thresholds
    min_question_length: int = 15
    max_question_length: int = 500
    min_answer_length: int = 20
    max_answer_length: int = 2000
    min_context_length: int = 50
    
    # Quality thresholds
    min_answer_context_overlap: float = 0.1  # At least 10% word overlap
    max_answer_context_overlap: float = 0.95  # Not just copy-paste
    min_question_specificity: float = 0.3
    
    # Deduplication
    minhash_threshold: float = 0.8  # Similarity threshold for duplicates
    minhash_num_perm: int = 128
    
    # Trivial question patterns
    trivial_patterns: List[str] = field(default_factory=lambda: [
        r'^what\?$',
        r'^why\?$',
        r'^how\?$',
        r'^what is (this|that|it)\??$',
        r'^what is the main (topic|idea|point)\??$',
        r'^what does this (mean|say)\??$',
        r'^can you explain\??$',
        r'^tell me about',
    ])
    
    # Minimum quality score to keep
    min_quality_score: float = 0.4


class QualityFilter:
    """
    Apply quality filters to QA pairs.
    
    Filters:
    1. Length checks (question, answer, context)
    2. Answer-context relationship (overlap, entailment proxy)
    3. Question specificity (avoid generic questions)
    4. Near-duplicate detection
    5. Trivial question removal
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize quality filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        
        # Compile trivial patterns
        self.trivial_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.trivial_patterns
        ]
        
        # MinHash LSH for deduplication
        self.lsh: Optional[Any] = None
        self.minhash_cache: Dict[str, Any] = {}
        
        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(
                threshold=self.config.minhash_threshold,
                num_perm=self.config.minhash_num_perm,
            )
    
    def compute_metrics(self, record: Dict[str, Any]) -> QualityMetrics:
        """
        Compute quality metrics for a QA record.
        
        Args:
            record: QA record with question, answer, context
            
        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics()
        
        question = record.get('question', '').strip()
        answer = record.get('answer', '').strip()
        context = record.get('context', '').strip()
        
        # Length metrics
        metrics.question_length = len(question)
        metrics.answer_length = len(answer)
        
        # Answer-context overlap
        metrics.answer_context_overlap = self._compute_word_overlap(answer, context)
        metrics.answer_in_context = self._is_substring(answer, context)
        
        # Question specificity
        metrics.question_specificity = self._compute_specificity(question)
        
        # Trivial question check
        metrics.is_trivial_question = self._is_trivial_question(question)
        
        # Answer completeness (check for truncation indicators)
        metrics.answer_completeness = self._check_completeness(answer)
        
        # Apply filters and collect reasons
        self._apply_filters(metrics, question, answer, context)
        
        # Compute overall quality score
        metrics.quality_score = self._compute_quality_score(metrics)
        
        # Final decision
        if metrics.quality_score < self.config.min_quality_score:
            metrics.should_keep = False
            if 'low_quality_score' not in metrics.filter_reasons:
                metrics.filter_reasons.append('low_quality_score')
        
        return metrics
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap ratio between two texts."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / len(words1)
    
    def _is_substring(self, needle: str, haystack: str) -> bool:
        """Check if needle is a substring of haystack (normalized)."""
        if not needle or not haystack:
            return False
        
        needle_norm = ' '.join(needle.lower().split())
        haystack_norm = ' '.join(haystack.lower().split())
        
        return needle_norm in haystack_norm
    
    def _compute_specificity(self, question: str) -> float:
        """
        Compute question specificity score.
        
        Higher score = more specific question.
        """
        if not question:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus (longer questions tend to be more specific)
        if len(question) > 50:
            score += 0.1
        if len(question) > 100:
            score += 0.1
        
        # Contains specific terms (numbers, technical terms, etc.)
        if re.search(r'\d+', question):
            score += 0.1
        
        # Contains quoted terms
        if re.search(r'["\'].*?["\']', question):
            score += 0.1
        
        # Starts with specific question words
        specific_starters = ['how does', 'what is the', 'why does', 'what are the',
                           'how can', 'what causes', 'how is', 'what determines']
        question_lower = question.lower()
        for starter in specific_starters:
            if question_lower.startswith(starter):
                score += 0.1
                break
        
        # Penalty for very short questions
        if len(question) < 20:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _is_trivial_question(self, question: str) -> bool:
        """Check if question matches trivial patterns."""
        question_clean = question.strip().lower()
        
        for pattern in self.trivial_patterns:
            if pattern.match(question_clean):
                return True
        
        # Also check for very short questions
        if len(question_clean) < 10:
            return True
        
        return False
    
    def _check_completeness(self, answer: str) -> float:
        """
        Check if answer appears complete (not truncated).
        
        Returns score 0-1 where 1 = complete.
        """
        if not answer:
            return 0.0
        
        score = 1.0
        
        # Check for truncation indicators
        truncation_indicators = [
            '...',
            '…',
            ' and so on',
            ' etc.',
            ' et cetera',
        ]
        
        answer_lower = answer.lower()
        for indicator in truncation_indicators:
            if answer_lower.endswith(indicator):
                score -= 0.3
        
        # Check if ends mid-sentence (no punctuation)
        if answer and answer[-1] not in '.!?)"\'':
            score -= 0.2
        
        return max(0.0, score)
    
    def _apply_filters(
        self,
        metrics: QualityMetrics,
        question: str,
        answer: str,
        context: str,
    ) -> None:
        """Apply all filters and record reasons."""
        # Length filters
        if metrics.question_length < self.config.min_question_length:
            metrics.should_keep = False
            metrics.filter_reasons.append('question_too_short')
        
        if metrics.question_length > self.config.max_question_length:
            metrics.should_keep = False
            metrics.filter_reasons.append('question_too_long')
        
        if metrics.answer_length < self.config.min_answer_length:
            metrics.should_keep = False
            metrics.filter_reasons.append('answer_too_short')
        
        if metrics.answer_length > self.config.max_answer_length:
            metrics.should_keep = False
            metrics.filter_reasons.append('answer_too_long')
        
        if len(context) < self.config.min_context_length:
            metrics.should_keep = False
            metrics.filter_reasons.append('context_too_short')
        
        # Answer-context relationship
        if metrics.answer_context_overlap < self.config.min_answer_context_overlap:
            # Answer might not be supported by context
            metrics.filter_reasons.append('low_answer_context_overlap')
        
        if metrics.answer_context_overlap > self.config.max_answer_context_overlap:
            # Might be trivial copy-paste
            metrics.filter_reasons.append('excessive_answer_context_overlap')
        
        # Question specificity
        if metrics.question_specificity < self.config.min_question_specificity:
            metrics.filter_reasons.append('low_question_specificity')
        
        # Trivial question
        if metrics.is_trivial_question:
            metrics.should_keep = False
            metrics.filter_reasons.append('trivial_question')
    
    def _compute_quality_score(self, metrics: QualityMetrics) -> float:
        """Compute overall quality score from metrics."""
        score = 0.0
        
        # Answer-context relationship (most important)
        if 0.2 <= metrics.answer_context_overlap <= 0.8:
            score += 0.3
        elif metrics.answer_context_overlap > 0.1:
            score += 0.15
        
        # Question specificity
        score += metrics.question_specificity * 0.25
        
        # Answer completeness
        score += metrics.answer_completeness * 0.2
        
        # Length appropriateness
        if 20 <= metrics.question_length <= 200:
            score += 0.1
        if 30 <= metrics.answer_length <= 500:
            score += 0.1
        
        # Penalties
        if metrics.is_trivial_question:
            score -= 0.3
        if metrics.is_near_duplicate:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def add_to_dedup_index(self, record_id: str, text: str) -> None:
        """Add a record to the deduplication index."""
        if not HAS_DATASKETCH or not self.lsh:
            return
        
        minhash = self._compute_minhash(text)
        self.minhash_cache[record_id] = minhash
        
        try:
            self.lsh.insert(record_id, minhash)
        except ValueError:
            # Already exists
            pass
    
    def check_duplicate(self, record_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a record is a near-duplicate of existing records.
        
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        if not HAS_DATASKETCH or not self.lsh:
            return False, None
        
        minhash = self._compute_minhash(text)
        
        # Query LSH
        candidates = self.lsh.query(minhash)
        
        # Remove self if present
        candidates = [c for c in candidates if c != record_id]
        
        if candidates:
            return True, candidates[0]
        
        return False, None
    
    def _compute_minhash(self, text: str) -> Any:
        """Compute MinHash for text."""
        if not HAS_DATASKETCH:
            return None
        
        minhash = MinHash(num_perm=self.config.minhash_num_perm)
        
        # Use word shingles
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            minhash.update(shingle.encode('utf-8'))
        
        return minhash


def run_quality_pipeline(
    input_file: Path,
    output_file: Path,
    config: Optional[FilterConfig] = None,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Run the full quality control pipeline on a QA file.
    
    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file (filtered records)
        config: Filter configuration
        deduplicate: Whether to perform deduplication
        
    Returns:
        Statistics dictionary
    """
    qf = QualityFilter(config)
    
    stats = {
        'total': 0,
        'kept': 0,
        'filtered': 0,
        'filter_reasons': defaultdict(int),
        'duplicates_removed': 0,
        'quality_score_distribution': defaultdict(int),
    }
    
    # First pass: build dedup index and compute metrics
    records_with_metrics: List[Tuple[Dict, QualityMetrics]] = []
    
    logger.info(f"Processing {input_file}...")
    
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            stats['total'] += 1
            record = json.loads(line)
            
            # Compute quality metrics
            metrics = qf.compute_metrics(record)
            
            # Check for duplicates
            if deduplicate:
                question_text = record.get('question', '')
                record_id = record.get('id', str(stats['total']))
                
                is_dup, dup_of = qf.check_duplicate(record_id, question_text)
                if is_dup:
                    metrics.is_near_duplicate = True
                    metrics.duplicate_of = dup_of
                    metrics.should_keep = False
                    metrics.filter_reasons.append('near_duplicate')
                    stats['duplicates_removed'] += 1
                else:
                    # Add to index for future checks
                    qf.add_to_dedup_index(record_id, question_text)
            
            records_with_metrics.append((record, metrics))
            
            # Track quality score distribution
            score_bucket = int(metrics.quality_score * 10) / 10
            stats['quality_score_distribution'][f"{score_bucket:.1f}"] += 1
    
    # Write filtered records
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open('w', encoding='utf-8') as f:
        for record, metrics in records_with_metrics:
            # Add quality metrics to record
            record['quality_score'] = metrics.quality_score
            record['quality_metrics'] = metrics.to_dict()
            
            if metrics.should_keep:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['kept'] += 1
            else:
                stats['filtered'] += 1
                for reason in metrics.filter_reasons:
                    stats['filter_reasons'][reason] += 1
    
    # Convert defaultdicts for JSON serialization
    stats['filter_reasons'] = dict(stats['filter_reasons'])
    stats['quality_score_distribution'] = dict(stats['quality_score_distribution'])
    
    logger.info(f"Quality filtering complete: {stats['kept']}/{stats['total']} kept")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quality filters on QA data")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--min-quality", type=float, default=0.4, help="Min quality score")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = FilterConfig(min_quality_score=args.min_quality)
    
    stats = run_quality_pipeline(
        Path(args.input),
        Path(args.output),
        config=config,
        deduplicate=not args.no_dedup,
    )
    
    print("\nQuality Filter Statistics:")
    print(f"  Total records: {stats['total']}")
    print(f"  Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"  Filtered: {stats['filtered']}")
    
    if stats['filter_reasons']:
        print("\nFilter Reasons:")
        for reason, count in sorted(stats['filter_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
