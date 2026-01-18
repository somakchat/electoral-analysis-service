"""
Unified Vector Store - Auto-selects backend based on configuration.

Priority:
1. OpenSearch (if OPENSEARCH_ENDPOINT is configured)
2. Local FAISS (fallback for development without OpenSearch)

This provides a single interface for the application regardless of backend.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os

from app.config import settings


@dataclass
class SearchResult:
    """Unified search result."""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    constituency: str = ""
    district: str = ""


class VectorStoreBase(ABC):
    """Abstract base for vector stores."""
    
    @abstractmethod
    def index_documents(self, documents: List[Dict]) -> int:
        """Index documents with text and metadata."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict[str, str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """Search for documents."""
        pass
    
    @abstractmethod
    def delete_by_source(self, source_file: str) -> int:
        """Delete documents by source file."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total document count."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        pass


class OpenSearchVectorStore(VectorStoreBase):
    """OpenSearch-backed vector store using PoliticalOpenSearchClient."""
    
    def __init__(self):
        from app.services.rag.political_opensearch import (
            PoliticalOpenSearchClient,
            DEFAULT_INDEX_NAME
        )
        self._client = PoliticalOpenSearchClient(
            index_name=os.getenv("OPENSEARCH_INDEX", DEFAULT_INDEX_NAME),
            endpoint=settings.opensearch_endpoint,
            region=settings.aws_region
        )
        print(f"[VectorStore] Using OpenSearch: {settings.opensearch_endpoint}")
    
    def index_documents(self, documents: List[Dict]) -> int:
        """Index documents using PoliticalOpenSearchClient."""
        # Ensure index exists
        self._client.ensure_index()
        
        # Index documents (client handles embedding generation)
        success, failed = self._client.index_documents(documents)
        return success
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict[str, str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """Search using PoliticalOpenSearchClient."""
        search_type = (search_type or "hybrid").lower()

        if search_type == "keyword":
            results = self._client.bm25_search_sync(query=query, top_k=top_k, filters=filters)
        elif search_type == "vector":
            results = self._client.knn_search_sync(query=query, top_k=top_k, filters=filters)
        else:
            # Hybrid (LocalHybridIndex-style RRF fusion)
            results = self._client.hybrid_search_sync(query, top_k, filters)
        
        return [
            SearchResult(
                doc_id=r.doc_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                source_file=r.source_file,
                constituency=r.constituency,
                district=r.district
            )
            for r in results
        ]
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete documents by source file."""
        return self._client.delete_by_source(source_file)
    
    def get_document_count(self) -> int:
        """Get total document count."""
        return self._client.get_document_count()
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch health."""
        return self._client.health_check()


class LocalFAISSVectorStore(VectorStoreBase):
    """Local FAISS-backed vector store (fallback)."""
    
    def __init__(self):
        from app.services.rag.local_store import LocalHybridIndex, DocumentChunk
        self._index = LocalHybridIndex(settings.index_dir)
        self._DocumentChunk = DocumentChunk
        print(f"[VectorStore] Using Local FAISS: {settings.index_dir}")
    
    def index_documents(self, documents: List[Dict]) -> int:
        chunks = []
        for doc in documents:
            chunks.append(self._DocumentChunk(
                doc_id=doc.get("doc_id", ""),
                chunk_id=doc.get("chunk_id", doc.get("doc_id", "")),
                source_path=doc.get("source_file", doc.get("metadata", {}).get("source_file", "")),
                text=doc.get("text", doc.get("content", "")),
                metadata=doc.get("metadata", {})
            ))
        
        self._index.add_chunks(chunks)
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict[str, str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        # FAISS local store doesn't support filters well, but we can post-filter
        if search_type == "keyword":
            semantic_weight, keyword_weight = 0.0, 1.0
        elif search_type == "vector":
            semantic_weight, keyword_weight = 1.0, 0.0
        else:
            semantic_weight, keyword_weight = 0.7, 0.3
        
        results = self._index.search(
            query, 
            top_k=top_k * 2 if filters else top_k,  # Get more if filtering
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        search_results = []
        for chunk, score in results:
            # Apply post-filtering
            if filters:
                match = True
                for field, value in filters.items():
                    chunk_value = chunk.metadata.get(field, "")
                    if isinstance(value, list):
                        if chunk_value not in value:
                            match = False
                            break
                    elif chunk_value.upper() != value.upper():
                        match = False
                        break
                if not match:
                    continue
            
            search_results.append(SearchResult(
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
                source_file=chunk.source_path,
                constituency=chunk.metadata.get("constituency", ""),
                district=chunk.metadata.get("district", "")
            ))
            
            if len(search_results) >= top_k:
                break
        
        return search_results
    
    def delete_by_source(self, source_file: str) -> int:
        # FAISS doesn't support easy deletion, would need to rebuild
        # For now, just clear and let re-indexing handle it
        count_before = len(self._index.chunks)
        self._index.chunks = [c for c in self._index.chunks if c.source_path != source_file]
        count_after = len(self._index.chunks)
        return count_before - count_after
    
    def get_document_count(self) -> int:
        return len(self._index.chunks)
    
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "backend": "local_faiss",
            "index_dir": str(self._index.index_dir),
            "document_count": self.get_document_count(),
            "faiss_loaded": self._index.faiss_index is not None,
            "bm25_loaded": self._index.bm25_index is not None
        }


class UnifiedVectorStore:
    """
    Unified vector store that auto-selects the best backend.
    
    Usage:
        store = UnifiedVectorStore()
        store.index_documents([...])
        results = store.search("query")
    """
    
    def __init__(self, force_backend: str = None):
        """
        Initialize vector store.
        
        Args:
            force_backend: Force a specific backend ("opensearch" or "local")
        """
        self._store: VectorStoreBase = self._select_backend(force_backend)
    
    def _select_backend(self, force_backend: str = None) -> VectorStoreBase:
        """Select the appropriate backend."""
        if force_backend == "opensearch":
            return OpenSearchVectorStore()
        elif force_backend == "local":
            return LocalFAISSVectorStore()
        
        # Auto-select based on configuration
        if settings.opensearch_endpoint:
            try:
                store = OpenSearchVectorStore()
                # Test connection
                health = store.health_check()
                if health.get("status") == "healthy":
                    return store
                else:
                    print(f"[VectorStore] OpenSearch unhealthy, falling back to local")
            except Exception as e:
                print(f"[VectorStore] OpenSearch failed: {e}, falling back to local")
        
        # Fallback to local FAISS
        return LocalFAISSVectorStore()
    
    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        if isinstance(self._store, OpenSearchVectorStore):
            return "opensearch"
        return "local_faiss"
    
    def index_documents(self, documents: List[Dict]) -> int:
        """Index documents."""
        return self._store.index_documents(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Dict[str, str] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters (e.g., {"district": "BANKURA"})
            search_type: "hybrid", "vector", or "keyword"
        
        Returns:
            List of SearchResult objects
        """
        return self._store.search(query, top_k, filters, search_type)
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete documents by source file."""
        return self._store.delete_by_source(source_file)
    
    def get_document_count(self) -> int:
        """Get total document count."""
        return self._store.get_document_count()
    
    def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        return self._store.health_check()


# Singleton instance
_store: Optional[UnifiedVectorStore] = None


def get_vector_store(force_backend: str = None) -> UnifiedVectorStore:
    """
    Get or create vector store instance.
    
    Args:
        force_backend: Force a specific backend ("opensearch" or "local")
    
    Returns:
        UnifiedVectorStore instance
    """
    global _store
    if _store is None or force_backend:
        _store = UnifiedVectorStore(force_backend)
    return _store

