"""
Faithfulness Metrics for MSQA-Bench.

Measures hallucination and faithfulness in RAG outputs:
1. Unsupported claim rate - claims without evidence
2. Citation precision - do citations support claims
3. Citation recall - are all claims cited
4. Answer-context faithfulness - is answer grounded in context
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional NLI model for entailment checking
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class Claim:
    """A claim extracted from an answer."""
    text: str
    start_pos: int
    end_pos: int
    citations: List[int] = field(default_factory=list)  # Citation indices
    
    # Evaluation
    is_supported: Optional[bool] = None
    support_score: float = 0.0
    supporting_passage_idx: Optional[int] = None


@dataclass
class FaithfulnessResult:
    """Faithfulness evaluation for a single QA pair."""
    query_id: str
    answer: str
    
    # Claims
    claims: List[Claim] = field(default_factory=list)
    num_claims: int = 0
    
    # Citation metrics
    num_citations: int = 0
    citation_precision: float = 0.0  # Cited passages that support claims
    citation_recall: float = 0.0     # Claims that are cited
    
    # Faithfulness metrics
    supported_claims: int = 0
    unsupported_claims: int = 0
    unsupported_claim_rate: float = 0.0
    
    # Overall
    faithfulness_score: float = 0.0
    has_hallucination: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'num_claims': self.num_claims,
            'num_citations': self.num_citations,
            'citation_precision': round(self.citation_precision, 4),
            'citation_recall': round(self.citation_recall, 4),
            'supported_claims': self.supported_claims,
            'unsupported_claims': self.unsupported_claims,
            'unsupported_claim_rate': round(self.unsupported_claim_rate, 4),
            'faithfulness_score': round(self.faithfulness_score, 4),
            'has_hallucination': self.has_hallucination,
        }


@dataclass
class AggregatedFaithfulness:
    """Aggregated faithfulness metrics across multiple samples."""
    num_samples: int
    
    # Averages
    avg_claims_per_answer: float = 0.0
    avg_citations_per_answer: float = 0.0
    avg_citation_precision: float = 0.0
    avg_citation_recall: float = 0.0
    avg_unsupported_claim_rate: float = 0.0
    avg_faithfulness_score: float = 0.0
    
    # Rates
    hallucination_rate: float = 0.0  # % of answers with any hallucination
    abstention_rate: float = 0.0     # % of answers that abstained
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_samples': self.num_samples,
            'avg_claims_per_answer': round(self.avg_claims_per_answer, 2),
            'avg_citations_per_answer': round(self.avg_citations_per_answer, 2),
            'avg_citation_precision': round(self.avg_citation_precision, 4),
            'avg_citation_recall': round(self.avg_citation_recall, 4),
            'avg_unsupported_claim_rate': round(self.avg_unsupported_claim_rate, 4),
            'avg_faithfulness_score': round(self.avg_faithfulness_score, 4),
            'hallucination_rate': round(self.hallucination_rate, 4),
            'abstention_rate': round(self.abstention_rate, 4),
        }
    
    def __str__(self) -> str:
        return (
            f"Faithfulness: {self.avg_faithfulness_score:.3f}, "
            f"Hallucination Rate: {self.hallucination_rate:.3f}, "
            f"Citation Precision: {self.avg_citation_precision:.3f}, "
            f"Citation Recall: {self.avg_citation_recall:.3f}"
        )


class FaithfulnessEvaluator:
    """
    Evaluate faithfulness of RAG outputs.
    
    Metrics:
    - Unsupported claim rate: proportion of claims not supported by context
    - Citation precision: proportion of citations that actually support claims
    - Citation recall: proportion of claims that have citations
    - Faithfulness score: overall groundedness of answer
    """
    
    def __init__(self, use_nli: bool = False, nli_model: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            use_nli: Whether to use NLI model for entailment checking
            nli_model: Model name for NLI (default: roberta-large-mnli)
        """
        self.use_nli = use_nli and HAS_TRANSFORMERS
        self.nli_pipeline = None
        
        if self.use_nli:
            model_name = nli_model or "roberta-large-mnli"
            logger.info(f"Loading NLI model: {model_name}")
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0,  # Use GPU if available
            )
    
    def evaluate(
        self,
        answer: str,
        passages: List[str],
        query_id: Optional[str] = None,
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness of an answer against retrieved passages.
        
        Args:
            answer: Generated answer text
            passages: List of retrieved passage texts
            query_id: Optional identifier
            
        Returns:
            FaithfulnessResult with all metrics
        """
        result = FaithfulnessResult(
            query_id=query_id or "",
            answer=answer,
        )
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        result.claims = claims
        result.num_claims = len(claims)
        
        if not claims:
            result.faithfulness_score = 1.0  # No claims = nothing to verify
            return result
        
        # Extract citations
        citations = self._extract_citations(answer)
        result.num_citations = len(set(citations))
        
        # Check support for each claim
        for claim in claims:
            support_score, supporting_idx = self._check_claim_support(
                claim.text, passages
            )
            claim.support_score = support_score
            claim.is_supported = support_score > 0.5
            claim.supporting_passage_idx = supporting_idx
            
            if claim.is_supported:
                result.supported_claims += 1
            else:
                result.unsupported_claims += 1
        
        # Compute metrics
        result.unsupported_claim_rate = (
            result.unsupported_claims / result.num_claims
            if result.num_claims > 0 else 0.0
        )
        
        # Citation precision: Are cited passages supporting claims?
        if citations:
            cited_supports = sum(
                1 for c in claims 
                if c.citations and c.is_supported
            )
            result.citation_precision = cited_supports / len(set(citations))
        
        # Citation recall: Are claims with evidence cited?
        claims_with_support = sum(1 for c in claims if c.is_supported)
        claims_with_citations = sum(1 for c in claims if c.citations)
        result.citation_recall = (
            claims_with_citations / claims_with_support
            if claims_with_support > 0 else 0.0
        )
        
        # Overall faithfulness score
        result.faithfulness_score = self._compute_faithfulness_score(result)
        result.has_hallucination = result.unsupported_claim_rate > 0.2
        
        return result
    
    def _extract_claims(self, answer: str) -> List[Claim]:
        """
        Extract individual claims from an answer.
        
        Uses sentence splitting with some heuristics for scientific text.
        """
        claims = []
        
        # Split by sentence-ending punctuation
        # Handle abbreviations and numbers
        sentence_pattern = r'(?<!\b[A-Z])(?<!\b[a-z])(?<!\d)\.(?=\s+[A-Z]|\s*$)|[!?]'
        
        # Simple sentence split
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Find position in original
            start = answer.find(sentence, pos)
            if start == -1:
                start = pos
            end = start + len(sentence)
            pos = end
            
            # Extract any citations in this sentence
            citation_pattern = r'\[(\d+)\]'
            citations = [int(m.group(1)) for m in re.finditer(citation_pattern, sentence)]
            
            # Remove citations from claim text for evaluation
            claim_text = re.sub(citation_pattern, '', sentence).strip()
            
            if claim_text:
                claims.append(Claim(
                    text=claim_text,
                    start_pos=start,
                    end_pos=end,
                    citations=citations,
                ))
        
        return claims
    
    def _extract_citations(self, answer: str) -> List[int]:
        """Extract all citation indices from answer."""
        pattern = r'\[(\d+)\]'
        return [int(m.group(1)) for m in re.finditer(pattern, answer)]
    
    def _check_claim_support(
        self,
        claim: str,
        passages: List[str],
    ) -> Tuple[float, Optional[int]]:
        """
        Check if a claim is supported by any passage.
        
        Args:
            claim: The claim text
            passages: List of passage texts
            
        Returns:
            Tuple of (support_score, supporting_passage_index)
        """
        if not passages:
            return 0.0, None
        
        best_score = 0.0
        best_idx = None
        
        for idx, passage in enumerate(passages):
            score = self._compute_support_score(claim, passage)
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_score, best_idx
    
    def _compute_support_score(self, claim: str, passage: str) -> float:
        """
        Compute support score between claim and passage.
        
        Uses NLI if available, otherwise falls back to lexical overlap.
        """
        if self.use_nli and self.nli_pipeline:
            return self._nli_support_score(claim, passage)
        else:
            return self._lexical_support_score(claim, passage)
    
    def _nli_support_score(self, claim: str, passage: str) -> float:
        """Use NLI model to check entailment."""
        try:
            # Truncate if too long
            passage_truncated = passage[:500]
            
            result = self.nli_pipeline(
                f"{passage_truncated} [SEP] {claim}",
                truncation=True,
            )
            
            # Get entailment probability
            for r in result:
                if r['label'].lower() == 'entailment':
                    return r['score']
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"NLI error: {e}")
            return self._lexical_support_score(claim, passage)
    
    def _lexical_support_score(self, claim: str, passage: str) -> float:
        """
        Compute lexical support score (fallback when NLI unavailable).
        
        Based on word overlap and n-gram matching.
        """
        claim_words = set(claim.lower().split())
        passage_words = set(passage.lower().split())
        
        if not claim_words:
            return 0.0
        
        # Word overlap
        overlap = len(claim_words & passage_words)
        word_score = overlap / len(claim_words)
        
        # Check for key phrase matches (bigrams, trigrams)
        claim_lower = claim.lower()
        passage_lower = passage.lower()
        
        # Extract n-grams from claim
        words = claim_lower.split()
        ngram_matches = 0
        ngram_total = 0
        
        for n in [2, 3]:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngram_total += 1
                if ngram in passage_lower:
                    ngram_matches += 1
        
        ngram_score = ngram_matches / ngram_total if ngram_total > 0 else 0
        
        # Combine scores
        score = 0.6 * word_score + 0.4 * ngram_score
        
        return min(1.0, score)
    
    def _compute_faithfulness_score(self, result: FaithfulnessResult) -> float:
        """Compute overall faithfulness score."""
        if result.num_claims == 0:
            return 1.0
        
        # Weight different factors
        support_rate = result.supported_claims / result.num_claims
        
        # Penalize unsupported claims more heavily
        penalty = result.unsupported_claim_rate * 0.5
        
        # Bonus for proper citations
        citation_bonus = 0.0
        if result.num_citations > 0:
            citation_bonus = 0.1 * result.citation_precision
        
        score = support_rate - penalty + citation_bonus
        
        return max(0.0, min(1.0, score))


def compute_faithfulness_metrics(
    rag_results_file: Path,
    output_file: Optional[Path] = None,
    use_nli: bool = False,
) -> AggregatedFaithfulness:
    """
    Compute faithfulness metrics for RAG results.
    
    Args:
        rag_results_file: JSONL file with RAG results
        output_file: Optional file to save detailed results
        use_nli: Whether to use NLI model
        
    Returns:
        AggregatedFaithfulness metrics
    """
    evaluator = FaithfulnessEvaluator(use_nli=use_nli)
    
    results: List[FaithfulnessResult] = []
    abstentions = 0
    
    with rag_results_file.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Check for abstention
            if data.get('has_abstained', False):
                abstentions += 1
                continue
            
            answer = data.get('generated_answer', '')
            passages = [p.get('text', '') for p in data.get('retrieved_passages', [])]
            query_id = data.get('query_id', '')
            
            result = evaluator.evaluate(answer, passages, query_id)
            results.append(result)
    
    # Aggregate metrics
    n = len(results)
    if n == 0:
        return AggregatedFaithfulness(num_samples=0)
    
    aggregated = AggregatedFaithfulness(
        num_samples=n,
        avg_claims_per_answer=sum(r.num_claims for r in results) / n,
        avg_citations_per_answer=sum(r.num_citations for r in results) / n,
        avg_citation_precision=sum(r.citation_precision for r in results) / n,
        avg_citation_recall=sum(r.citation_recall for r in results) / n,
        avg_unsupported_claim_rate=sum(r.unsupported_claim_rate for r in results) / n,
        avg_faithfulness_score=sum(r.faithfulness_score for r in results) / n,
        hallucination_rate=sum(1 for r in results if r.has_hallucination) / n,
        abstention_rate=abstentions / (n + abstentions) if (n + abstentions) > 0 else 0,
    )
    
    # Save detailed results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            # Write summary first
            f.write(json.dumps({'summary': aggregated.to_dict()}, ensure_ascii=False) + '\n')
            
            # Write individual results
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved faithfulness results to {output_file}")
    
    logger.info(f"Faithfulness evaluation: {aggregated}")
    
    return aggregated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute faithfulness metrics")
    parser.add_argument("--input", "-i", required=True, help="RAG results JSONL")
    parser.add_argument("--output", "-o", help="Output file for detailed results")
    parser.add_argument("--use-nli", action="store_true", help="Use NLI model")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    metrics = compute_faithfulness_metrics(
        Path(args.input),
        output_file=Path(args.output) if args.output else None,
        use_nli=args.use_nli,
    )
    
    print(f"\nFaithfulness Metrics:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v}")
