"""
Local Hybrid Index for RAG.
Combines FAISS (semantic) + BM25 (keyword) search with RRF fusion.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import numpy as np
import re

from app.config import settings
from app.services.rag.embeddings import get_embedding_service


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    doc_id: str
    chunk_id: str
    source_path: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalHybridIndex:
    """
    Hybrid index combining semantic and keyword search.
    
    Uses:
    - FAISS for semantic (vector) search
    - BM25 for keyword search
    - Reciprocal Rank Fusion (RRF) for combining results
    """
    
    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir or settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedder = get_embedding_service()
        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self.bm25_index = None
        
        # Load existing index if available
        self._load()
    
    def _load(self):
        """Load existing index from disk."""
        chunks_path = self.index_dir / "chunks.pkl"
        faiss_path = self.index_dir / "faiss.index"
        bm25_path = self.index_dir / "bm25.pkl"
        
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        
        if faiss_path.exists():
            try:
                import faiss
                self.faiss_index = faiss.read_index(str(faiss_path))
            except Exception:
                pass
        
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)
    
    def _save(self):
        """Save index to disk."""
        chunks_path = self.index_dir / "chunks.pkl"
        faiss_path = self.index_dir / "faiss.index"
        bm25_path = self.index_dir / "bm25.pkl"
        
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        if self.faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_index, str(faiss_path))
            except Exception:
                pass
        
        if self.bm25_index is not None:
            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25_index, f)
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add document chunks to the index."""
        if not chunks:
            return
        
        # Add to chunk list
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        
        # Update FAISS index
        self._update_faiss(embeddings, start_idx)
        
        # Update BM25 index
        self._update_bm25()
        
        # Save to disk
        self._save()
    
    def _update_faiss(self, embeddings: np.ndarray, start_idx: int):
        """Update FAISS index with new embeddings."""
        try:
            import faiss
        except ImportError:
            return
        
        if embeddings.shape[0] == 0:
            return
        
        dim = embeddings.shape[1]
        
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
    
    def _update_bm25(self):
        """Rebuild BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        
        # Tokenize all chunks
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
        
        Returns:
            List of (chunk, score) tuples
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        if not self.chunks:
            return []
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k * 2)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine with RRF
        combined = self._rrf_fusion(
            semantic_results, 
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Return top_k
        return combined[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform semantic search using FAISS."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        try:
            import faiss
        except ImportError:
            return []
        
        # Get query embedding
        query_emb = self.embedder.embed_query(query).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_emb, k)
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform keyword search using BM25."""
        if self.bm25_index is None:
            return []
        
        tokens = self._tokenize(query)
        if not tokens:
            return []
        
        scores = self.bm25_index.get_scores(tokens)
        
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _rrf_fusion(
        self,
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        semantic_weight: float,
        keyword_weight: float,
        k: int = 60
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(weight / (k + rank))
        """
        scores = {}
        
        # Add semantic scores
        for rank, (idx, _) in enumerate(semantic_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += semantic_weight / (k + rank + 1)
        
        # Add keyword scores
        for rank, (idx, _) in enumerate(keyword_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += keyword_weight / (k + rank + 1)
        
        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return with chunks
        return [(self.chunks[idx], score) for idx, score in sorted_results if idx < len(self.chunks)]
    
    def clear(self):
        """Clear the entire index."""
        self.chunks = []
        self.faiss_index = None
        self.bm25_index = None
        
        # Remove persisted files
        for file in ["chunks.pkl", "faiss.index", "bm25.pkl"]:
            path = self.index_dir / file
            if path.exists():
                path.unlink()
