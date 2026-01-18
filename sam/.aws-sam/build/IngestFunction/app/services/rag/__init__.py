"""
RAG (Retrieval-Augmented Generation) Pipeline Module.

Implements Advanced RAG with:
- Query decomposition
- Hybrid search (semantic + keyword) via Unified Vector Store
- Cross-encoder reranking
- Contextual compression
- Knowledge Graph for structured political data
- Zero-hallucination retrieval with citations
- Query routing for optimal retrieval strategy

Vector Store: Automatically selects OpenSearch (if configured) or FAISS (fallback)
"""

# Conditional imports for AWS Lambda compatibility
# LocalHybridIndex requires FAISS/numpy - only for local development
try:
    from app.services.rag.local_store import LocalHybridIndex, DocumentChunk
    LOCAL_STORE_AVAILABLE = True
except ImportError:
    LOCAL_STORE_AVAILABLE = False
    LocalHybridIndex = None
    # Define a minimal DocumentChunk for compatibility
    from dataclasses import dataclass, field
    from typing import Any, Dict
    
    @dataclass
    class DocumentChunk:
        """Represents a chunk of a document."""
        doc_id: str
        chunk_id: str
        source_path: str
        text: str
        metadata: Dict[str, Any] = field(default_factory=dict)

from app.services.rag.vector_store import UnifiedVectorStore, get_vector_store, SearchResult
from app.services.rag.advanced_rag import AdvancedRAG
from app.services.rag.rerank import CrossEncoderReranker
from app.services.rag.political_rag import PoliticalRAGSystem, create_political_rag, RAGResponse
from app.services.rag.knowledge_graph import PoliticalKnowledgeGraph
from app.services.rag.data_schema import (
    ConstituencyProfile, FactWithCitation, VerifiedAnswer,
    ElectionResult, SurveyResponse
)

# structured_ingest requires pandas/numpy - optional
try:
    from app.services.rag.structured_ingest import StructuredIngestionPipeline
    STRUCTURED_INGEST_AVAILABLE = True
except ImportError:
    STRUCTURED_INGEST_AVAILABLE = False
    StructuredIngestionPipeline = None

from app.services.rag.verified_retrieval import VerifiedRetriever, HallucinationGuard
from app.services.rag.query_router import QueryRouter, QueryExecutor

__all__ = [
    # Unified Vector Store (primary)
    "UnifiedVectorStore",
    "get_vector_store",
    "SearchResult",
    # Legacy components (backward compatibility)
    "LocalHybridIndex",
    "DocumentChunk",
    "AdvancedRAG",
    "CrossEncoderReranker",
    # Political RAG system
    "PoliticalRAGSystem",
    "create_political_rag",
    "RAGResponse",
    # Knowledge Graph
    "PoliticalKnowledgeGraph",
    # Data schemas
    "ConstituencyProfile",
    "FactWithCitation",
    "VerifiedAnswer",
    "ElectionResult",
    "SurveyResponse",
    # Ingestion
    "StructuredIngestionPipeline",
    # Retrieval
    "VerifiedRetriever",
    "HallucinationGuard",
    "QueryRouter",
    "QueryExecutor",
]
