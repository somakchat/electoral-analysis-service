"""
Embedding Service for RAG Pipeline.
Supports multiple embedding providers.
AWS Lambda compatible - numpy is optional.
"""
from __future__ import annotations
from typing import List, Optional, Union
from functools import lru_cache

# Make numpy optional for Lambda
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from app.config import settings


class EmbeddingService:
    """Unified embedding service supporting multiple providers."""
    
    def __init__(self):
        self._model = None
        self._provider = None
        self._dimension = 768  # Default dimension
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        # Ensure provider is initialized so dimension reflects the active model
        self._init_provider()
        return self._dimension
    
    def _init_provider(self):
        """Initialize the embedding provider lazily."""
        if self._model is not None:
            return
        
        # Try OpenAI first (preferred for AWS Lambda)
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._model = OpenAI(api_key=settings.openai_api_key)
                self._provider = "openai"
                # text-embedding-3-large has 3072 dimensions
                if settings.openai_embed_model == "text-embedding-3-large":
                    self._dimension = 3072
                else:
                    self._dimension = 1536
                return
            except Exception:
                pass
        
        # Try Gemini
        if settings.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self._model = genai
                self._provider = "gemini"
                self._dimension = 768
                return
            except Exception:
                pass
        
        # Fallback to local sentence transformers (not available on Lambda)
        if HAS_NUMPY:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._provider = "local"
                self._dimension = 384
                return
            except Exception:
                pass
        
        # Ultimate fallback: mock embeddings for testing
        self._provider = "mock"
        self._dimension = 384
    
    def embed(self, texts: List[str]) -> Union[List[List[float]], 'np.ndarray']:
        """Generate embeddings for texts. Returns list of lists for Lambda compatibility."""
        self._init_provider()
        
        if not texts:
            return []
        
        if self._provider == "openai":
            return self._embed_openai(texts)
        elif self._provider == "gemini":
            return self._embed_gemini(texts)
        elif self._provider == "local":
            return self._embed_local(texts)
        else:
            return self._embed_mock(texts)
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI. Returns list of lists."""
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._model.embeddings.create(
                model=settings.openai_embed_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Embed using Google Gemini. Returns list of lists."""
        embeddings = []
        for text in texts:
            result = self._model.embed_content(
                model=f"models/{settings.gemini_embed_model}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings
    
    def _embed_local(self, texts: List[str]) -> Union[List[List[float]], 'np.ndarray']:
        """Embed using local sentence transformers."""
        result = self._model.encode(texts, convert_to_numpy=True)
        if HAS_NUMPY:
            return result
        return [list(e) for e in result]
    
    def _embed_mock(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for testing."""
        import random
        return [[random.gauss(0, 1) for _ in range(self._dimension)] for _ in range(len(texts))]
    
    def embed_query(self, query: str) -> Union[List[float], 'np.ndarray']:
        """Embed a single query."""
        embeddings = self.embed([query])
        if len(embeddings) > 0:
            return embeddings[0]
        return [0.0] * self._dimension


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service instance."""
    return EmbeddingService()
