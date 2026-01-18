"""
Cross-Encoder Reranker for RAG Pipeline.
Reranks retrieved documents using cross-encoder models for better relevance.
AWS Lambda compatible - numpy is optional.
"""
from __future__ import annotations
from typing import List, Tuple

# numpy is optional for Lambda
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving retrieval quality.
    
    Uses a cross-encoder model to score query-document pairs directly,
    providing more accurate relevance scores than bi-encoder retrieval.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=512)
        except Exception as e:
            print(f"Warning: Could not load cross-encoder model: {e}")
            self._model = "fallback"
    
    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank passages based on relevance to query.
        
        Args:
            query: The search query
            passages: List of passage texts
            top_k: Number of top results to return (None = all)
        
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not passages:
            return []
        
        self._load_model()
        
        if self._model == "fallback":
            # Fallback: return original order with dummy scores
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k or len(passages), len(passages)))]
        
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Score pairs
        try:
            scores = self._model.predict(pairs)
        except Exception:
            # Fallback on error
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k or len(passages), len(passages)))]
        
        # Sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return [(idx, float(score)) for idx, score in indexed_scores]


class LLMReranker:
    """
    LLM-based reranker using the main LLM for relevance scoring.
    
    Useful when cross-encoder is not available or for more nuanced reranking.
    """
    
    def __init__(self):
        from app.services.llm import get_llm
        self.llm = get_llm()
    
    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rerank using LLM scoring."""
        if not passages:
            return []
        
        # For efficiency, only rerank top 20 passages
        passages_to_rank = passages[:20]
        
        # Build prompt
        passages_text = ""
        for i, p in enumerate(passages_to_rank):
            preview = p[:200] + "..." if len(p) > 200 else p
            passages_text += f"\n[{i}] {preview}"
        
        prompt = f"""Rank these passages by relevance to the query.
Return a JSON array of passage indices in order of relevance (most relevant first).
Return ONLY the indices, nothing else.

Query: {query}

Passages:{passages_text}

Example output: [2, 5, 0, 3, 1, 4]"""
        
        try:
            response = self.llm.generate(prompt, temperature=0)
            
            # Parse response
            import json
            import re
            match = re.search(r'\[[\d,\s]+\]', response.text)
            if match:
                indices = json.loads(match.group())
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < len(passages_to_rank)]
                
                # Create scored results
                results = [(idx, 1.0 - rank * 0.05) for rank, idx in enumerate(valid_indices)]
                return results[:top_k]
        except Exception:
            pass
        
        # Fallback
        return [(i, 1.0 - i * 0.05) for i in range(min(top_k, len(passages_to_rank)))]
