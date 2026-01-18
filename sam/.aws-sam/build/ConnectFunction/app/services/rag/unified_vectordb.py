"""
Unified Vector Database using OpenSearch.

Works for both local development and AWS production:
- Local: Connects to OpenSearch via direct endpoint (LocalStack or AWS)
- AWS: Uses IAM authentication with AWS4Auth

Supports:
- Hybrid search (kNN + BM25)
- Vector-only search
- Keyword-only search (BM25)
- Filtering by metadata
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import json
import hashlib

from app.config import settings


@dataclass
class Document:
    """Document with text, embedding, and metadata."""
    doc_id: str
    text: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class VectorDBConfig:
    """Configuration for vector database."""
    
    # OpenSearch settings
    OPENSEARCH_ENDPOINT: str = os.getenv("OPENSEARCH_ENDPOINT", settings.opensearch_endpoint or "")
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", settings.opensearch_index or "political-data")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "443"))
    AWS_REGION: str = os.getenv("AWS_REGION", settings.aws_region or "us-east-1")
    
    # Embedding settings
    EMBEDDING_DIM: int = 3072  # text-embedding-3-large
    EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    
    # Search settings
    DEFAULT_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    KNN_EF_SEARCH: int = 100
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if OpenSearch is properly configured."""
        return bool(cls.OPENSEARCH_ENDPOINT)


class OpenSearchVectorDB:
    """
    Unified OpenSearch Vector Database.
    
    Works with both AWS OpenSearch Service and OpenSearch Serverless.
    Uses hybrid search combining kNN vectors and BM25 keyword matching.
    """
    
    # Index mapping for political data
    INDEX_MAPPING = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 3072
                },
                "text": {"type": "text", "analyzer": "standard"},
                "doc_id": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                # Political-specific fields
                "constituency": {"type": "keyword"},
                "district": {"type": "keyword"},
                "party": {"type": "keyword"},
                "year": {"type": "keyword"},
                "data_type": {"type": "keyword"},
                # Metadata object
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "keyword"},
                        "constituency": {"type": "keyword"},
                        "district": {"type": "keyword"},
                        "party": {"type": "keyword"},
                        "year": {"type": "keyword"},
                        "data_type": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                    }
                }
            }
        }
    }
    
    def __init__(
        self,
        endpoint: str = None,
        index_name: str = None,
        region: str = None,
        use_ssl: bool = True
    ):
        self.endpoint = endpoint or VectorDBConfig.OPENSEARCH_ENDPOINT
        self.index_name = index_name or VectorDBConfig.OPENSEARCH_INDEX
        self.region = region or VectorDBConfig.AWS_REGION
        self.use_ssl = use_ssl
        
        self._client = None
        self._embedder = None
        self._is_serverless = "aoss" in self.endpoint if self.endpoint else False
    
    @property
    def client(self):
        """Lazy-load OpenSearch client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def embedder(self):
        """Lazy-load embedding service."""
        if self._embedder is None:
            from app.services.rag.embeddings import get_embedding_service
            self._embedder = get_embedding_service()
        return self._embedder
    
    def _create_client(self):
        """Create OpenSearch client with appropriate authentication."""
        from opensearchpy import OpenSearch, RequestsHttpConnection
        
        if not self.endpoint:
            raise ValueError("OpenSearch endpoint not configured. Set OPENSEARCH_ENDPOINT.")
        
        # Parse endpoint
        host = self.endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        
        # Try AWS authentication first
        auth = self._get_aws_auth()
        
        if auth:
            print(f"[OpenSearch] Connecting with AWS auth to {host}")
            return OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=auth,
                connection_class=RequestsHttpConnection,
                use_ssl=True,
                verify_certs=True,
                timeout=60,
                max_retries=3,
                retry_on_timeout=True,
            )
        else:
            # Fallback to no auth (for local OpenSearch)
            print(f"[OpenSearch] Connecting without auth to {host}")
            port = 9200 if "localhost" in host or "127.0.0.1" in host else 443
            return OpenSearch(
                hosts=[{"host": host, "port": port}],
                use_ssl=self.use_ssl and port == 443,
                verify_certs=False if "localhost" in host else True,
                timeout=60,
            )
    
    def _get_aws_auth(self):
        """Get AWS4Auth for OpenSearch authentication."""
        try:
            import boto3
            from requests_aws4auth import AWS4Auth
            
            session = boto3.Session()
            creds = session.get_credentials()
            
            if creds is None:
                return None
            
            frozen = creds.get_frozen_credentials()
            
            # Use "aoss" for serverless, "es" for managed
            service = "aoss" if self._is_serverless else "es"
            
            return AWS4Auth(
                frozen.access_key,
                frozen.secret_key,
                self.region,
                service,
                session_token=frozen.token,
            )
        except Exception as e:
            print(f"[OpenSearch] AWS auth not available: {e}")
            return None
    
    def ensure_index(self, embedding_dim: int = 3072) -> bool:
        """Create index if it doesn't exist."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                # Update dimension in mapping
                mapping = self.INDEX_MAPPING.copy()
                mapping["mappings"]["properties"]["vector"]["dimension"] = embedding_dim
                
                self.client.indices.create(index=self.index_name, body=mapping)
                print(f"[OpenSearch] Created index '{self.index_name}'")
                return True
            else:
                print(f"[OpenSearch] Index '{self.index_name}' already exists")
                return True
        except Exception as e:
            print(f"[OpenSearch] Error creating index: {e}")
            return False
    
    def index_documents(self, documents: List[Document], batch_size: int = 100) -> int:
        """
        Index documents with embeddings.
        
        Args:
            documents: List of Document objects
            batch_size: Number of documents to index per batch
        
        Returns:
            Number of documents indexed
        """
        from opensearchpy import helpers
        
        if not documents:
            return 0
        
        # Ensure index exists
        self.ensure_index()
        
        # Generate embeddings for documents without them
        texts_to_embed = []
        indices_to_embed = []
        
        for i, doc in enumerate(documents):
            if not doc.embedding:
                texts_to_embed.append(doc.text)
                indices_to_embed.append(i)
        
        if texts_to_embed:
            embeddings = self.embedder.embed(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                documents[idx].embedding = emb
        
        # Prepare bulk actions
        actions = []
        for doc in documents:
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_source": {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "vector": doc.embedding,
                    "source_file": doc.metadata.get("source_file", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "constituency": doc.metadata.get("constituency", ""),
                    "district": doc.metadata.get("district", ""),
                    "party": doc.metadata.get("party", ""),
                    "year": doc.metadata.get("year", ""),
                    "data_type": doc.metadata.get("data_type", ""),
                    "metadata": doc.metadata,
                }
            }
            # OpenSearch Serverless doesn't support explicit _id
            if not self._is_serverless:
                action["_id"] = doc.doc_id
            actions.append(action)
        
        # Bulk index
        success, failed = helpers.bulk(
            self.client, 
            actions, 
            chunk_size=batch_size,
            raise_on_error=False
        )
        
        failed_count = len(failed) if failed else 0
        print(f"[OpenSearch] Indexed {success} documents, {failed_count} failed")
        if failed_count:
            try:
                # Print a compact view of a few failures to aid debugging (esp. AOSS restrictions)
                sample = failed[:3]
                print("[OpenSearch] Sample failures (first 3):")
                print(json.dumps(sample, indent=2)[:4000])
            except Exception as e:
                print(f"[OpenSearch] Could not serialize failures: {e}")
        return success
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, str] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Document]:
        """
        Hybrid search combining kNN vectors and BM25.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"district": "BANKURA"})
            vector_weight: Weight for vector similarity
            bm25_weight: Weight for BM25 keyword matching
        
        Returns:
            List of matching documents with scores
        """
        top_k = top_k or VectorDBConfig.DEFAULT_TOP_K
        
        # Get query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Build query
        must_clauses = []
        
        # kNN vector search
        must_clauses.append({
            "knn": {
                "vector": {
                    "vector": query_embedding,
                    "k": top_k * 2  # Get more for fusion
                }
            }
        })
        
        # BM25 text search
        must_clauses.append({
            "multi_match": {
                "query": query,
                "fields": ["text^3", "constituency^2", "district^2", "party"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        })
        
        # Build filter clauses
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value:
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {field: value}})
                    else:
                        filter_clauses.append({"term": {field: value}})
        
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": must_clauses,
                    "filter": filter_clauses,
                    "minimum_should_match": 1
                }
            },
            "_source": ["doc_id", "text", "metadata", "source_file", "constituency", "district", "party", "year"]
        }
        
        return self._execute_search(query_body)
    
    def vector_search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, str] = None
    ) -> List[Document]:
        """Pure vector similarity search."""
        top_k = top_k or VectorDBConfig.DEFAULT_TOP_K
        
        query_embedding = self.embedder.embed_query(query)
        
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value:
                    filter_clauses.append({"term": {field: value}})
        
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{
                        "knn": {
                            "vector": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    }],
                    "filter": filter_clauses
                }
            },
            "_source": ["doc_id", "text", "metadata", "source_file", "constituency", "district", "party", "year"]
        }
        
        return self._execute_search(query_body)
    
    def keyword_search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict[str, str] = None
    ) -> List[Document]:
        """BM25 keyword search."""
        top_k = top_k or VectorDBConfig.DEFAULT_TOP_K
        
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                if value:
                    filter_clauses.append({"term": {field: value}})
        
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query": query,
                            "fields": ["text^3", "constituency^2", "district^2", "party"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    }],
                    "filter": filter_clauses
                }
            },
            "_source": ["doc_id", "text", "metadata", "source_file", "constituency", "district", "party", "year"]
        }
        
        return self._execute_search(query_body)
    
    def _execute_search(self, query_body: Dict) -> List[Document]:
        """Execute search and convert results to Documents."""
        try:
            response = self.client.search(index=self.index_name, body=query_body)
            hits = response.get("hits", {}).get("hits", [])
            
            documents = []
            for hit in hits:
                source = hit.get("_source", {})
                doc = Document(
                    doc_id=source.get("doc_id", hit.get("_id", "")),
                    text=source.get("text", ""),
                    metadata=source.get("metadata", {}),
                    score=hit.get("_score", 0.0)
                )
                # Add top-level fields to metadata
                for field in ["source_file", "constituency", "district", "party", "year"]:
                    if source.get(field):
                        doc.metadata[field] = source[field]
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"[OpenSearch] Search error: {e}")
            return []
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete all documents from a source file."""
        try:
            response = self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"source_file": source_file}
                    }
                }
            )
            deleted = response.get("deleted", 0)
            print(f"[OpenSearch] Deleted {deleted} documents from {source_file}")
            return deleted
        except Exception as e:
            print(f"[OpenSearch] Delete error: {e}")
            return 0
    
    def get_document_count(self) -> int:
        """Get total document count in index."""
        try:
            response = self.client.count(index=self.index_name)
            return response.get("count", 0)
        except Exception:
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch connection health."""
        try:
            info = self.client.info()
            count = self.get_document_count()
            return {
                "status": "healthy",
                "cluster_name": info.get("cluster_name", "unknown"),
                "version": info.get("version", {}).get("number", "unknown"),
                "index": self.index_name,
                "document_count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "index": self.index_name
            }


# Singleton instance
_vectordb: Optional[OpenSearchVectorDB] = None


def get_vectordb() -> OpenSearchVectorDB:
    """Get or create vector database instance."""
    global _vectordb
    if _vectordb is None:
        _vectordb = OpenSearchVectorDB()
    return _vectordb


def create_vectordb(
    endpoint: str = None,
    index_name: str = None,
    region: str = None
) -> OpenSearchVectorDB:
    """Create a new vector database instance."""
    return OpenSearchVectorDB(
        endpoint=endpoint,
        index_name=index_name,
        region=region
    )

