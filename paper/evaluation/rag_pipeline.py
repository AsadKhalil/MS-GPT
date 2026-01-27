"""
RAG Pipeline for MSQA-Bench.

Implements Retrieval-Augmented Generation with:
1. Standard RAG: Retrieve + Generate
2. Cite-then-Answer RAG: Force citations in answers
3. RAG with Abstention: Allow "I don't know" responses

Supports citation tracking for faithfulness evaluation.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    retriever_type: str = "embedding"  # "bm25" or "embedding"
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    
    # Generation settings
    generator_model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    generator_base_url: str = "http://localhost:8000/v1"
    generator_api_key: str = "not-needed"
    temperature: float = 0.3
    max_tokens: int = 512
    
    # RAG mode
    mode: str = "standard"  # "standard", "cite", "abstain"
    
    # Citation settings
    require_citations: bool = False
    citation_format: str = "[{idx}]"  # e.g., [1], [2]
    
    # Abstention settings
    allow_abstention: bool = False
    abstention_phrase: str = "I cannot find sufficient evidence"


@dataclass
class Citation:
    """A citation in the generated answer."""
    index: int           # Citation number [1], [2], etc.
    passage_index: int   # Which retrieved passage (0-indexed)
    passage_text: str    # The cited passage text
    start_pos: int       # Position in answer
    end_pos: int         # Position in answer


@dataclass
class RAGResult:
    """Result of RAG generation."""
    query: str
    query_id: Optional[str] = None
    
    # Retrieval
    retrieved_passages: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    
    # Generation
    generated_answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    
    # Evaluation
    has_abstained: bool = False
    abstention_reason: Optional[str] = None
    
    # Reference (if available)
    reference_answer: Optional[str] = None
    
    # Metadata
    generation_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'query_id': self.query_id,
            'retrieved_passages': self.retrieved_passages,
            'retrieval_scores': self.retrieval_scores,
            'generated_answer': self.generated_answer,
            'citations': [
                {
                    'index': c.index,
                    'passage_index': c.passage_index,
                    'start_pos': c.start_pos,
                    'end_pos': c.end_pos,
                }
                for c in self.citations
            ],
            'has_abstained': self.has_abstained,
            'abstention_reason': self.abstention_reason,
            'reference_answer': self.reference_answer,
            'generation_time': self.generation_time,
            'total_time': self.total_time,
        }


class RAGPipeline:
    """
    RAG pipeline with retrieval, generation, and citation tracking.
    
    Modes:
    - standard: Basic RAG (retrieve + generate)
    - cite: Force citations in answer format
    - abstain: Allow model to say "I don't know"
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.retriever = None
        self.generator_client = None
        
        # Initialize retriever
        self._init_retriever()
        
        # Initialize generator
        if HAS_OPENAI:
            self.generator_client = OpenAI(
                base_url=config.generator_base_url,
                api_key=config.generator_api_key,
                timeout=120,
            )
    
    def _init_retriever(self) -> None:
        """Initialize the retriever."""
        from .retrieval_baselines import BM25Retriever, EmbeddingRetriever
        
        if self.config.retriever_type == "bm25":
            self.retriever = BM25Retriever()
        else:
            self.retriever = EmbeddingRetriever(
                self.config.retriever_model,
                device="cuda",
            )
    
    def index_corpus(self, corpus: Dict[str, str]) -> None:
        """
        Index the corpus for retrieval.
        
        Args:
            corpus: Dict mapping doc_id to document text
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        self.retriever.index(corpus)
        self.corpus = corpus
        logger.info(f"Indexed {len(corpus)} documents")
    
    def query(
        self,
        question: str,
        query_id: Optional[str] = None,
        reference_answer: Optional[str] = None,
    ) -> RAGResult:
        """
        Run RAG pipeline for a single query.
        
        Args:
            question: The query/question
            query_id: Optional identifier
            reference_answer: Optional ground truth answer
            
        Returns:
            RAGResult with answer and metadata
        """
        import time
        start_time = time.time()
        
        result = RAGResult(
            query=question,
            query_id=query_id,
            reference_answer=reference_answer,
        )
        
        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question, self.config.top_k)
        
        for doc_id, score in retrieved:
            passage_text = self.corpus.get(doc_id, "")
            result.retrieved_passages.append({
                'doc_id': doc_id,
                'text': passage_text,
            })
            result.retrieval_scores.append(score)
        
        # Step 2: Generate
        gen_start = time.time()
        
        if self.generator_client is None:
            logger.warning("Generator not available, skipping generation")
            result.generated_answer = "[Generator not available]"
        else:
            answer, citations, abstained = self._generate(
                question, 
                result.retrieved_passages
            )
            result.generated_answer = answer
            result.citations = citations
            result.has_abstained = abstained
            
            if abstained:
                result.abstention_reason = "insufficient_evidence"
        
        result.generation_time = time.time() - gen_start
        result.total_time = time.time() - start_time
        
        return result
    
    def _generate(
        self,
        question: str,
        passages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Citation], bool]:
        """
        Generate answer using LLM.
        
        Args:
            question: The question
            passages: Retrieved passages
            
        Returns:
            Tuple of (answer, citations, has_abstained)
        """
        # Build prompt based on mode
        if self.config.mode == "cite":
            prompt = self._build_citation_prompt(question, passages)
        elif self.config.mode == "abstain":
            prompt = self._build_abstention_prompt(question, passages)
        else:
            prompt = self._build_standard_prompt(question, passages)
        
        try:
            response = self.generator_client.chat.completions.create(
                model=self.config.generator_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract citations if present
            citations = self._extract_citations(answer, passages)
            
            # Check for abstention
            has_abstained = self._check_abstention(answer)
            
            return answer, citations, has_abstained
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error: {e}]", [], False
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on mode."""
        base = "You are a helpful scientific assistant that answers questions based on provided evidence."
        
        if self.config.mode == "cite":
            return (
                f"{base} You MUST cite your sources using [1], [2], etc. "
                "Every claim must have a citation. Format: 'Claim [1]. Another claim [2].'"
            )
        elif self.config.mode == "abstain":
            return (
                f"{base} If the provided passages do not contain sufficient evidence "
                f"to answer the question, respond with: '{self.config.abstention_phrase}'"
            )
        else:
            return base
    
    def _build_standard_prompt(
        self, 
        question: str, 
        passages: List[Dict[str, Any]]
    ) -> str:
        """Build standard RAG prompt."""
        context = "\n\n".join([
            f"Passage {i+1}:\n{p['text'][:1000]}"
            for i, p in enumerate(passages)
        ])
        
        return f"""Based on the following passages, answer the question.

{context}

Question: {question}

Answer:"""
    
    def _build_citation_prompt(
        self, 
        question: str, 
        passages: List[Dict[str, Any]]
    ) -> str:
        """Build prompt that requires citations."""
        context = "\n\n".join([
            f"[{i+1}] {p['text'][:1000]}"
            for i, p in enumerate(passages)
        ])
        
        return f"""Based on the following numbered passages, answer the question.
You MUST cite passages using [1], [2], etc. for every claim you make.

{context}

Question: {question}

Answer (with citations):"""
    
    def _build_abstention_prompt(
        self, 
        question: str, 
        passages: List[Dict[str, Any]]
    ) -> str:
        """Build prompt that allows abstention."""
        context = "\n\n".join([
            f"Passage {i+1}:\n{p['text'][:1000]}"
            for i, p in enumerate(passages)
        ])
        
        return f"""Based on the following passages, answer the question.
If the passages do not contain enough information to answer the question confidently,
respond with: "{self.config.abstention_phrase}"

{context}

Question: {question}

Answer:"""
    
    def _extract_citations(
        self, 
        answer: str, 
        passages: List[Dict[str, Any]]
    ) -> List[Citation]:
        """Extract citations from answer text."""
        citations = []
        
        # Find all citation patterns [1], [2], etc.
        pattern = r'\[(\d+)\]'
        
        for match in re.finditer(pattern, answer):
            try:
                idx = int(match.group(1))
                if 1 <= idx <= len(passages):
                    citation = Citation(
                        index=idx,
                        passage_index=idx - 1,
                        passage_text=passages[idx - 1]['text'][:200],
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                    citations.append(citation)
            except ValueError:
                continue
        
        return citations
    
    def _check_abstention(self, answer: str) -> bool:
        """Check if the answer indicates abstention."""
        abstention_indicators = [
            self.config.abstention_phrase.lower(),
            "cannot find",
            "no evidence",
            "not mentioned",
            "insufficient information",
            "unable to determine",
            "i don't know",
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in abstention_indicators)
    
    def run_batch(
        self,
        queries: List[Dict[str, Any]],
        corpus: Dict[str, str],
        output_file: Optional[Path] = None,
    ) -> List[RAGResult]:
        """
        Run RAG on a batch of queries.
        
        Args:
            queries: List of dicts with 'question' and optionally 'answer', 'id'
            corpus: Document corpus
            output_file: Optional file to save results
            
        Returns:
            List of RAGResult objects
        """
        # Index corpus if not already done
        if not hasattr(self, 'corpus') or self.corpus != corpus:
            self.index_corpus(corpus)
        
        results = []
        
        for i, q in enumerate(queries):
            question = q.get('question', '')
            query_id = q.get('id', str(i))
            reference = q.get('answer')
            
            result = self.query(question, query_id, reference)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        # Save results
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open('w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
            logger.info(f"Saved results to {output_file}")
        
        return results


def run_rag_evaluation(
    qa_file: Path,
    corpus_file: Path,
    output_file: Path,
    config: Optional[RAGConfig] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run RAG evaluation on a QA file.
    
    Args:
        qa_file: JSONL with questions and answers
        corpus_file: JSONL with corpus documents
        output_file: Output file for results
        config: RAG configuration
        sample_size: Optional sample size
        
    Returns:
        Evaluation statistics
    """
    config = config or RAGConfig()
    
    # Load queries
    queries = []
    with qa_file.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    if sample_size and sample_size < len(queries):
        import random
        queries = random.sample(queries, sample_size)
    
    # Load or build corpus
    corpus = {}
    if corpus_file.exists():
        with corpus_file.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    corpus[doc.get('id', doc.get('doc_id'))] = doc.get('text', doc.get('context', ''))
    else:
        # Build corpus from QA contexts
        for q in queries:
            ctx_id = q.get('context_id', q.get('id'))
            corpus[ctx_id] = q.get('context', '')
    
    # Run RAG
    pipeline = RAGPipeline(config)
    results = pipeline.run_batch(queries, corpus, output_file)
    
    # Compute statistics
    stats = {
        'total_queries': len(results),
        'abstentions': sum(1 for r in results if r.has_abstained),
        'with_citations': sum(1 for r in results if r.citations),
        'avg_citations': (
            sum(len(r.citations) for r in results) / len(results) 
            if results else 0
        ),
        'avg_generation_time': (
            sum(r.generation_time for r in results) / len(results)
            if results else 0
        ),
    }
    
    logger.info(f"RAG evaluation complete: {stats}")
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("--qa-file", "-q", required=True, help="QA JSONL file")
    parser.add_argument("--corpus", "-c", help="Corpus JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--mode", choices=["standard", "cite", "abstain"], 
                       default="standard")
    parser.add_argument("--sample", "-s", type=int, help="Sample size")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = RAGConfig(
        mode=args.mode,
        generator_model=args.model,
        generator_base_url=args.base_url,
        require_citations=(args.mode == "cite"),
        allow_abstention=(args.mode == "abstain"),
    )
    
    corpus_file = Path(args.corpus) if args.corpus else Path(args.qa_file)
    
    stats = run_rag_evaluation(
        Path(args.qa_file),
        corpus_file,
        Path(args.output),
        config=config,
        sample_size=args.sample,
    )
    
    print(f"\nRAG Statistics: {json.dumps(stats, indent=2)}")
