"""
Political Strategy Maker - OpenSearch Client.

Unified OpenSearch client for both local and AWS production:
- Follows the same patterns as eib_search_app
- Optimized for political electoral data
- Supports hybrid search (kNN + BM25)
- Political-specific field mappings and filters

Index Name: political-strategy-maker
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth

from app.config import settings

# Optional langchain import - use our own embeddings if not available
try:
    from langchain_openai import OpenAIEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    OpenAIEmbeddings = None

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Default index name for Political Strategy Maker
DEFAULT_INDEX_NAME = "political-strategy-maker"


@dataclass
class PoliticalSearchResult:
    """Search result with political-specific fields."""
    doc_id: str
    text: str
    score: float
    constituency: str = ""
    district: str = ""
    party: str = ""
    year: str = ""
    data_type: str = ""
    source_file: str = ""
    winner_2021: str = ""
    predicted_winner_2026: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class _EmbeddingWrapper:
    """Wrapper to make our embedding service compatible with langchain API."""
    def __init__(self, service):
        self._service = service
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Langchain-compatible method for embedding multiple documents."""
        result = self._service.embed(texts)
        # Ensure we return list of lists
        if hasattr(result, 'tolist'):
            return result.tolist()
        return [list(e) if hasattr(e, '__iter__') else e for e in result]
    
    def embed_query(self, text: str) -> List[float]:
        """Langchain-compatible method for embedding a query."""
        result = self._service.embed_query(text)
        if hasattr(result, 'tolist'):
            return result.tolist()
        return list(result) if hasattr(result, '__iter__') else result


class PoliticalOpenSearchClient:
    """
    OpenSearch client optimized for Political Strategy Maker.
    
    Features:
    - AWS IAM authentication (AOSS/Serverless compatible)
    - Hybrid search (kNN + BM25)
    - Political field filtering (constituency, district, party, year)
    - Batch indexing with proper mappings
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
                # Vector field for semantic search
                "vector": {
                    "type": "knn_vector",
                    "dimension": 3072  # text-embedding-3-large
                },
                # Text fields for BM25 search
                "text": {"type": "text", "analyzer": "standard"},
                "summary": {"type": "text"},
                
                # Political-specific keyword fields (for filtering)
                "doc_id": {"type": "keyword"},
                "constituency": {"type": "keyword"},
                "district": {"type": "keyword"},
                "party": {"type": "keyword"},
                "year": {"type": "keyword"},
                "data_type": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "winner_2021": {"type": "keyword"},
                "predicted_winner_2026": {"type": "keyword"},
                "race_rating": {"type": "keyword"},
                
                # Numeric fields
                "margin_2021": {"type": "float"},
                "predicted_margin_2026": {"type": "float"},
                "tmc_vote_share": {"type": "float"},
                "bjp_vote_share": {"type": "float"},
                "swing": {"type": "float"},
                
                # Metadata object
                "metadata": {
                    "type": "object",
                    "properties": {
                        "constituency": {"type": "keyword"},
                        "district": {"type": "keyword"},
                        "party": {"type": "keyword"},
                        "year": {"type": "keyword"},
                        "data_type": {"type": "keyword"},
                        "source_file": {"type": "keyword"},
                        "chunk_index": {"type": "integer"}
                    }
                }
            }
        }
    }
    
    def __init__(
        self,
        index_name: str = None,
        endpoint: str = None,
        region: str = None
    ):
        """
        Initialize the Political OpenSearch client.
        
        Args:
            index_name: OpenSearch index name (default: political-strategy-maker)
            endpoint: OpenSearch endpoint URL
            region: AWS region
        """
        self.index_name = index_name or os.getenv("OPENSEARCH_INDEX", DEFAULT_INDEX_NAME)
        self.endpoint = endpoint or settings.opensearch_endpoint
        self.region = region or os.getenv("AWS_REGION", settings.aws_region or "us-east-1")
        
        self._client = None
        self._embedder = None
        self._is_serverless = "aoss" in (self.endpoint or "")
        # Small in-memory cache to avoid repeated embedding calls for identical queries
        self._query_embedding_cache: Dict[str, List[float]] = {}
        self._query_embedding_cache_max: int = 128
        
        logger.info(f"[PoliticalOpenSearch] Index: {self.index_name}")
        logger.info(f"[PoliticalOpenSearch] Endpoint: {self.endpoint}")
        logger.info(f"[PoliticalOpenSearch] Serverless: {self._is_serverless}")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get (cached) query embedding to reduce OpenAI embedding rate-limit pressure."""
        q = (query or "").strip()
        if not q:
            return []
        cached = self._query_embedding_cache.get(q)
        if cached is not None:
            return cached
        emb = self.embedder.embed_query(q)
        # naive FIFO eviction
        if len(self._query_embedding_cache) >= self._query_embedding_cache_max:
            try:
                oldest_key = next(iter(self._query_embedding_cache.keys()))
                self._query_embedding_cache.pop(oldest_key, None)
            except Exception:
                self._query_embedding_cache.clear()
        self._query_embedding_cache[q] = emb
        return emb
    
    @property
    def client(self) -> OpenSearch:
        """Lazy-load OpenSearch client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    @property
    def embedder(self):
        """Lazy-load embedding model. Uses langchain if available, otherwise our service."""
        if self._embedder is None:
            if HAS_LANGCHAIN:
                self._embedder = OpenAIEmbeddings(
                    model=settings.openai_embed_model or "text-embedding-3-large",
                    openai_api_key=settings.openai_api_key
                )
            else:
                # Use our own embedding service wrapped for compatibility
                from app.services.rag.embeddings import get_embedding_service
                self._embedder = _EmbeddingWrapper(get_embedding_service())
        return self._embedder
    
    def _create_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS authentication."""
        if not self.endpoint:
            raise ValueError("OpenSearch endpoint not configured. Set OPENSEARCH_ENDPOINT.")
        
        # Get AWS credentials
        session = boto3.Session()
        creds = session.get_credentials().get_frozen_credentials()
        
        # Use "aoss" for serverless, "es" for managed
        service = "aoss" if self._is_serverless else "es"
        
        awsauth = AWS4Auth(
            creds.access_key,
            creds.secret_key,
            self.region,
            service,
            session_token=creds.token,
        )
        
        # Parse host from endpoint
        host = self.endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        
        logger.info(f"[PoliticalOpenSearch] Connecting to {host} with {service} auth")
        
        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            connection_class=RequestsHttpConnection,
            use_ssl=True,
            verify_certs=True,
            timeout=60,
            max_retries=5,
            retry_on_timeout=True,
        )
    
    def ensure_index(self) -> bool:
        """Create index with proper mappings if it doesn't exist."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"[PoliticalOpenSearch] Creating index: {self.index_name}")
                self.client.indices.create(index=self.index_name, body=self.INDEX_MAPPING)
                logger.info(f"[PoliticalOpenSearch] Index created successfully")
                return True
            else:
                logger.info(f"[PoliticalOpenSearch] Index already exists: {self.index_name}")
                return True
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Error creating index: {e}")
            return False
    
    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> Tuple[int, int]:
        """
        Index documents with embeddings.
        
        Args:
            documents: List of documents with text and metadata
            batch_size: Number of documents per batch
            
        Returns:
            Tuple of (success_count, failed_count)
        """
        if not documents:
            return 0, 0
        
        # Ensure index exists
        self.ensure_index()
        
        # Generate embeddings for all documents
        logger.info(f"[PoliticalOpenSearch] Generating embeddings for {len(documents)} documents...")
        texts = [doc.get("text", "") for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        
        # Prepare bulk actions
        # Note: AOSS (OpenSearch Serverless) doesn't support explicit document IDs
        actions = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc_{i}")
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                # Don't include _id for AOSS - store it in the source instead
                "_source": {
                    "vector": embeddings[i],
                    "text": doc.get("text", ""),
                    "doc_id": doc_id,  # Store doc_id in source for reference
                    "constituency": doc.get("constituency", doc.get("metadata", {}).get("constituency", "")),
                    "district": doc.get("district", doc.get("metadata", {}).get("district", "")),
                    "party": doc.get("party", doc.get("metadata", {}).get("party", "")),
                    "year": doc.get("year", doc.get("metadata", {}).get("year", "")),
                    "data_type": doc.get("data_type", doc.get("metadata", {}).get("data_type", "")),
                    "source_file": doc.get("source_file", doc.get("metadata", {}).get("source_file", "")),
                    "winner_2021": doc.get("winner_2021", ""),
                    "predicted_winner_2026": doc.get("predicted_winner_2026", ""),
                    "race_rating": doc.get("race_rating", ""),
                    "margin_2021": doc.get("margin_2021", 0.0),
                    "predicted_margin_2026": doc.get("predicted_margin_2026", 0.0),
                    "tmc_vote_share": doc.get("tmc_vote_share", 0.0),
                    "bjp_vote_share": doc.get("bjp_vote_share", 0.0),
                    "swing": doc.get("swing", 0.0),
                    "metadata": doc.get("metadata", {})
                }
            }
            # Only add _id for non-serverless OpenSearch
            if not self._is_serverless:
                action["_id"] = doc_id
            actions.append(action)
        
        # Bulk index
        logger.info(f"[PoliticalOpenSearch] Bulk indexing {len(actions)} documents...")
        try:
            success, errors = helpers.bulk(
                self.client,
                actions,
                chunk_size=batch_size,
                raise_on_error=False,
                stats_only=False
            )
            
            # helpers.bulk returns (success_count, list_of_errors)
            failed_count = len(errors) if errors else 0
            logger.info(f"[PoliticalOpenSearch] Indexed: {success} success, {failed_count} failed")
            
            if errors:
                for err in errors[:5]:  # Log first 5 errors
                    logger.warning(f"[PoliticalOpenSearch] Error: {err}")
            
            return success, failed_count
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Bulk index error: {e}")
            # Try individual inserts as fallback
            success_count = 0
            for action in actions:
                try:
                    # For AOSS, don't specify id
                    if self._is_serverless:
                        self.client.index(
                            index=self.index_name,
                            body=action["_source"]
                        )
                    else:
                        self.client.index(
                            index=self.index_name,
                            id=action.get("_id"),
                            body=action["_source"]
                        )
                    success_count += 1
                except Exception as idx_err:
                    logger.warning(f"[PoliticalOpenSearch] Single index error: {idx_err}")
            logger.info(f"[PoliticalOpenSearch] Fallback indexed: {success_count} documents")
            return success_count, len(actions) - success_count
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """
        Hybrid search combining kNN vectors and BM25 text matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (constituency, district, party, year)
            
        Returns:
            List of PoliticalSearchResult objects
        """
        try:
            logger.info(f"[PoliticalOpenSearch] Hybrid search: {query[:50]}...")
            
            # Generate query embedding (cached)
            embedding = self._get_query_embedding(query)
            
            # Build query body (SAM-aligned):
            # Use SHOULD for kNN and BM25 so either can match (not overly strict).
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "vector": {
                                        "vector": embedding,
                                        "k": top_k * 2
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^3", "constituency^2", "district^2", "party"],
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": []
                    }
                },
                "_source": {
                    "includes": [
                        "doc_id","text","constituency","district","party","year","data_type","source_file","metadata"
                    ]
                }
            }
            
            # Add filters (support term + terms)
            if filters:
                for key, val in filters.items():
                    if not val:
                        continue
                    if isinstance(val, list):
                        query_body["query"]["bool"]["filter"].append({"terms": {key: val}})
                    else:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})
            
            # Execute search
            res = self.client.search(index=self.index_name, body=query_body)
            hits = res["hits"]["hits"]
            
            logger.info(f"[PoliticalOpenSearch] Found {len(hits)} results")
            return self._format_results(hits)
            
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Hybrid search error: {e}")
            return []

    def _extract_key_entities(self, query: str) -> List[str]:
        """Extract key named entities from query for boosted search."""
        import re
        entities = []
        
        # Common political figures in West Bengal
        known_names = [
            "Sukanta Majumdar", "Suvendu Adhikari", "Dilip Ghosh", "Mamata Banerjee",
            "Abhishek Banerjee", "Narendra Modi", "Amit Shah",
            "সুকান্ত মজুমদার", "শুভেন্দু অধিকারী", "দিলীপ ঘোষ", "মমতা বন্দ্যোপাধ্যায়"
        ]
        
        for name in known_names:
            if name.lower() in query.lower():
                entities.append(name)
        
        # Extract capitalized multi-word phrases (likely names)
        caps_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        matches = re.findall(caps_pattern, query)
        entities.extend(matches)
        
        # Key political terms to boost
        key_terms = ["Chief Minister", "CM", "মুখ্যমন্ত্রী", "BJP", "TMC", "বিজেপি", "তৃণমূল"]
        for term in key_terms:
            if term.lower() in query.lower():
                entities.append(term)
        
        return list(set(entities))

    async def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """BM25-only search with entity-aware boosting for better relevance."""
        try:
            # Extract key entities for boosting
            entities = self._extract_key_entities(query)
            
            # Build should clauses for entity boosting
            should_clauses = []
            for entity in entities:
                should_clauses.append({
                    "match_phrase": {
                        "text": {
                            "query": entity,
                            "boost": 5.0  # High boost for exact entity matches
                        }
                    }
                })
            
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^3", "source_file^2", "constituency", "district"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "minimum_should_match": "25%"
                                }
                            }
                        ],
                        "should": should_clauses,  # Boost docs with key entities
                        "filter": []
                    }
                },
                "_source": {"includes": ["doc_id","text","constituency","district","party","year","data_type","source_file","metadata"]}
            }
            if filters:
                for key, val in filters.items():
                    if not val:
                        continue
                    if isinstance(val, list):
                        query_body["query"]["bool"]["filter"].append({"terms": {key: val}})
                    else:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})

            res = self.client.search(index=self.index_name, body=query_body)
            return self._format_results(res["hits"]["hits"])
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] BM25 search error: {e}")
            return []

    async def knn_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """kNN-only search (SAM pattern)."""
        try:
            embedding = self._get_query_embedding(query)
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "filter": [],
                        "should": [
                            {"knn": {"vector": {"vector": embedding, "k": top_k * 2}}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "_source": {"includes": ["doc_id","text","constituency","district","party","year","data_type","source_file","metadata"]}
            }
            if filters:
                for key, val in filters.items():
                    if not val:
                        continue
                    if isinstance(val, list):
                        query_body["query"]["bool"]["filter"].append({"terms": {key: val}})
                    else:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})

            res = self.client.search(index=self.index_name, body=query_body)
            return self._format_results(res["hits"]["hits"])
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] kNN search error: {e}")
            return []
    
    def knn_search_sync(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """Synchronous kNN-only search."""
        try:
            embedding = self._get_query_embedding(query)
            if not embedding:
                return []

            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "filter": [],
                        "should": [
                            {"knn": {"vector": {"vector": embedding, "k": top_k}}}
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "_source": {"includes": ["doc_id","text","constituency","district","party","year","data_type","source_file","metadata"]},
            }
            if filters:
                for key, val in filters.items():
                    if not val:
                        continue
                    if isinstance(val, list):
                        query_body["query"]["bool"]["filter"].append({"terms": {key: val}})
                    else:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})

            res = self.client.search(index=self.index_name, body=query_body)
            return self._format_results(res["hits"]["hits"])
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] kNN search error: {e}")
            return []

    def bm25_search_sync(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """Synchronous BM25-only search with entity-aware boosting."""
        try:
            # Extract key entities for boosting
            entities = self._extract_key_entities(query)
            
            # Build should clauses for entity boosting
            should_clauses = []
            for entity in entities:
                should_clauses.append({
                    "match_phrase": {
                        "text": {
                            "query": entity,
                            "boost": 5.0  # High boost for exact entity matches
                        }
                    }
                })
            
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^3", "source_file^2", "constituency", "district"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "minimum_should_match": "25%"
                                }
                            }
                        ],
                        "should": should_clauses,  # Boost docs with key entities
                        "filter": [],
                    }
                },
                "_source": {"includes": ["doc_id","text","constituency","district","party","year","data_type","source_file","metadata"]},
            }
            if filters:
                for key, val in filters.items():
                    if not val:
                        continue
                    if isinstance(val, list):
                        query_body["query"]["bool"]["filter"].append({"terms": {key: val}})
                    else:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})

            res = self.client.search(index=self.index_name, body=query_body)
            return self._format_results(res["hits"]["hits"])
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] BM25 search error: {e}")
            return []

    def _rrf_fuse_hits(
        self,
        semantic_hits: List[Dict[str, Any]],
        keyword_hits: List[Dict[str, Any]],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) to mimic LocalHybridIndex behavior.
        Uses rank-based fusion: score += weight / (k + rank + 1)
        """
        scores: Dict[str, float] = {}
        best_hit: Dict[str, Dict[str, Any]] = {}

        def _doc_id(hit: Dict[str, Any]) -> str:
            src = hit.get("_source") or {}
            return src.get("doc_id") or hit.get("_id") or ""

        def _add(hits: List[Dict[str, Any]], weight: float) -> None:
            if weight <= 0:
                return
            for rank, hit in enumerate(hits):
                did = _doc_id(hit)
                if not did:
                    continue
                scores[did] = scores.get(did, 0.0) + (weight / (k + rank + 1))
                # Keep the first hit we saw for this doc_id (contains _source)
                if did not in best_hit:
                    best_hit[did] = hit

        _add(semantic_hits, semantic_weight)
        _add(keyword_hits, keyword_weight)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_hits: List[Dict[str, Any]] = []
        for did, fused_score in ranked:
            hit = best_hit.get(did, {})
            fused_hits.append(
                {
                    "_id": hit.get("_id"),
                    "_source": hit.get("_source", {}),
                    "_score": fused_score,
                }
            )
        return fused_hits

    def hybrid_search_sync(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """
        Hybrid search that matches LocalHybridIndex behavior:
        - run kNN + BM25 separately
        - fuse with Reciprocal Rank Fusion (RRF)
        """
        try:
            candidate_k = max(top_k * 2, top_k)

            # Separate retrieval
            knn_results = self.knn_search_sync(query=query, top_k=candidate_k, filters=filters)
            bm25_results = self.bm25_search_sync(query=query, top_k=candidate_k, filters=filters)

            # Convert back to "hits" shape so we can reuse formatter with RRF scores
            knn_hits = [{"_source": r.__dict__ if hasattr(r, "__dict__") else {}, "_score": r.score} for r in knn_results]
            bm25_hits = [{"_source": r.__dict__ if hasattr(r, "__dict__") else {}, "_score": r.score} for r in bm25_results]

            fused_hits = self._rrf_fuse_hits(
                semantic_hits=knn_hits,
                keyword_hits=bm25_hits,
                semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
                keyword_weight=float(os.getenv("KEYWORD_WEIGHT", "0.3")),
                k=int(os.getenv("RRF_K", "60")),
            )

            # Take top_k
            return self._format_results(fused_hits[:top_k])
            
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Search error: {e}")
            return []
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[PoliticalSearchResult]:
        """Pure vector similarity search."""
        try:
            embedding = self.embedder.embed_query(query)
            
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "vector": {
                                        "vector": embedding,
                                        "k": top_k
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                }
            }
            
            if filters:
                for key, val in filters.items():
                    if val:
                        query_body["query"]["bool"]["filter"].append({"term": {key: val}})
            
            res = self.client.search(index=self.index_name, body=query_body)
            return self._format_results(res["hits"]["hits"])
            
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Vector search error: {e}")
            return []
    
    
    def _format_results(self, hits: List[Dict]) -> List[PoliticalSearchResult]:
        """Format OpenSearch hits to PoliticalSearchResult objects."""
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            results.append(PoliticalSearchResult(
                doc_id=src.get("doc_id", hit.get("_id", "")),
                text=src.get("text", ""),
                score=hit.get("_score", 0.0),
                constituency=src.get("constituency", ""),
                district=src.get("district", ""),
                party=src.get("party", ""),
                year=src.get("year", ""),
                data_type=src.get("data_type", ""),
                source_file=src.get("source_file", ""),
                winner_2021=src.get("winner_2021", ""),
                predicted_winner_2026=src.get("predicted_winner_2026", ""),
                metadata=src.get("metadata", {})
            ))
        return results
    
    def get_document_count(self) -> int:
        """Get total document count in index."""
        try:
            res = self.client.count(index=self.index_name)
            return res.get("count", 0)
        except Exception:
            return 0
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete all documents from a specific source file."""
        try:
            res = self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"source_file": source_file}
                    }
                }
            )
            deleted = res.get("deleted", 0)
            logger.info(f"[PoliticalOpenSearch] Deleted {deleted} docs from {source_file}")
            return deleted
        except Exception as e:
            logger.exception(f"[PoliticalOpenSearch] Delete error: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch connection health."""
        try:
            # For AOSS serverless, info() may not work, try indices.exists instead
            if self._is_serverless:
                # Try a simple operation to verify connectivity
                try:
                    exists = self.client.indices.exists(index=self.index_name)
                    count = self.get_document_count() if exists else 0
                    return {
                        "status": "healthy",
                        "cluster_name": "aoss-serverless",
                        "version": "serverless",
                        "index": self.index_name,
                        "index_exists": exists,
                        "document_count": count
                    }
                except Exception as idx_error:
                    # If we get a 404, connection is working but index doesn't exist
                    if "404" in str(idx_error) or "NotFoundError" in str(idx_error):
                        return {
                            "status": "healthy",
                            "cluster_name": "aoss-serverless",
                            "version": "serverless",
                            "index": self.index_name,
                            "index_exists": False,
                            "document_count": 0
                        }
                    raise
            else:
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
_opensearch_client: Optional[PoliticalOpenSearchClient] = None


def get_political_opensearch(
    index_name: str = None,
    endpoint: str = None
) -> PoliticalOpenSearchClient:
    """Get or create Political OpenSearch client."""
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = PoliticalOpenSearchClient(
            index_name=index_name,
            endpoint=endpoint
        )
    return _opensearch_client


def create_political_opensearch(
    index_name: str = None,
    endpoint: str = None
) -> PoliticalOpenSearchClient:
    """Create a new Political OpenSearch client instance."""
    return PoliticalOpenSearchClient(
        index_name=index_name,
        endpoint=endpoint
    )

