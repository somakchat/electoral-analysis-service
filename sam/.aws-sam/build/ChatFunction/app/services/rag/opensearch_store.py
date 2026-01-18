"""
OpenSearch Serverless Store for AWS Production.
Implements hybrid search using OpenSearch kNN + BM25 capabilities.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import hashlib

from app.config import settings
from app.services.rag.embeddings import get_embedding_service


@dataclass
class OpenSearchChunk:
    """Document chunk stored in OpenSearch."""
    doc_id: str
    chunk_id: str
    source_path: str
    text: str
    metadata: Dict[str, Any]


class OpenSearchHybridStore:
    """
    OpenSearch Serverless implementation for production.
    
    Features:
    - Hybrid search (kNN + BM25) in single query
    - Entity filtering
    - Metadata-based filtering
    """
    
    def __init__(self):
        self.embedder = get_embedding_service()
        self._client = None
        self.index_name = settings.opensearch_index
    
    def _get_client(self):
        """Get OpenSearch client with AWS authentication."""
        if self._client is not None:
            return self._client
        
        if not settings.opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT not configured")
        
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        import boto3
        
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            settings.aws_region,
            'aoss',  # OpenSearch Serverless
            session_token=credentials.token
        )
        
        self._client = OpenSearch(
            hosts=[{'host': settings.opensearch_endpoint, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        
        return self._client
    
    def create_index(self):
        """Create the OpenSearch index with hybrid search configuration."""
        client = self._get_client()
        
        dimension = self.embedder.dimension
        
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "source_path": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "metadata": {"type": "object", "enabled": True},
                    "entity_type": {"type": "keyword"},
                    "entity_name": {"type": "keyword"},
                    "constituency": {"type": "keyword"},
                    "party": {"type": "keyword"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        
        if not client.indices.exists(index=self.index_name):
            client.indices.create(index=self.index_name, body=index_body)
    
    def index_chunk(self, chunk: OpenSearchChunk, embedding: List[float] = None):
        """Index a single chunk."""
        client = self._get_client()
        
        if embedding is None:
            embedding = self.embedder.embed([chunk.text])[0].tolist()
        
        doc = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "text": chunk.text,
            "embedding": embedding,
            "metadata": chunk.metadata,
            "entity_type": chunk.metadata.get("entity_type"),
            "entity_name": chunk.metadata.get("entity_name"),
            "constituency": chunk.metadata.get("constituency"),
            "party": chunk.metadata.get("party"),
        }
        
        client.index(
            index=self.index_name,
            id=chunk.chunk_id,
            body=doc
        )
    
    def index_chunks_bulk(self, chunks: List[OpenSearchChunk]):
        """Bulk index chunks for efficiency."""
        if not chunks:
            return
        
        client = self._get_client()
        
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        
        # Build bulk request
        actions = []
        for chunk, embedding in zip(chunks, embeddings):
            actions.append({"index": {"_index": self.index_name, "_id": chunk.chunk_id}})
            actions.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "text": chunk.text,
                "embedding": embedding.tolist(),
                "metadata": chunk.metadata,
                "entity_type": chunk.metadata.get("entity_type"),
                "entity_name": chunk.metadata.get("entity_name"),
                "constituency": chunk.metadata.get("constituency"),
                "party": chunk.metadata.get("party"),
            })
        
        # Execute bulk
        response = client.bulk(body=actions)
        return response
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Tuple[OpenSearchChunk, float]]:
        """
        Execute hybrid search combining kNN and BM25.
        
        Uses OpenSearch's hybrid query capability for efficient single-request search.
        """
        client = self._get_client()
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query).tolist()
        
        # Build hybrid query
        body = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        },
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "boost": keyword_weight / semantic_weight
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            
            body["query"] = {
                "bool": {
                    "must": body["query"],
                    "filter": filter_clauses
                }
            }
        
        # Execute search
        response = client.search(index=self.index_name, body=body)
        
        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            chunk = OpenSearchChunk(
                doc_id=source["doc_id"],
                chunk_id=source["chunk_id"],
                source_path=source["source_path"],
                text=source["text"],
                metadata=source.get("metadata", {})
            )
            results.append((chunk, hit["_score"]))
        
        return results
    
    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks for a document."""
        client = self._get_client()
        
        client.delete_by_query(
            index=self.index_name,
            body={
                "query": {
                    "term": {"doc_id": doc_id}
                }
            }
        )
