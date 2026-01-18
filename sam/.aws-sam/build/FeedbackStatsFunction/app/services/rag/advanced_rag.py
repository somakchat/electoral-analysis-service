"""
Advanced RAG Pipeline with Query Decomposition, Hybrid Search, and Reranking.

This implements the full Advanced RAG pipeline as per architecture:
1. Query Decomposer - Break complex queries into sub-queries
2. Hybrid Search - kNN semantic + BM25 keyword search
3. RRF Fusion - Combine search results
4. Cross-Encoder Reranker - Improve relevance ranking
5. Contextual Compressor - Extract relevant parts of documents
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import re

from app.models import Evidence
from app.services.llm import get_llm, BaseLLM
from app.services.rag.rerank import CrossEncoderReranker
from app.config import settings

# Conditional imports for AWS Lambda compatibility
try:
    from app.services.rag.local_store import LocalHybridIndex, DocumentChunk
except ImportError:
    LocalHybridIndex = None
    from dataclasses import dataclass, field
    from typing import Any as AnyType
    
    @dataclass
    class DocumentChunk:
        doc_id: str
        chunk_id: str
        source_path: str
        text: str
        metadata: dict = field(default_factory=dict)


class AdvancedRAG:
    """
    Multi-hop RAG with query decomposition, hybrid search, and contextual compression.
    
    Pipeline:
    User Query → Query Decomposer → [Sub-Query 1, Sub-Query 2, Sub-Query 3]
                                          ↓
    For each sub-query:
        Hybrid Search (kNN + BM25) → RRF Fusion → Results
                                          ↓
    All Results → Cross-Encoder Reranker → Top Results
                                          ↓
    Top Results → Contextual Compressor → Final Context
    """
    
    def __init__(self, index) -> None:
        """Initialize with any compatible index (LocalHybridIndex or OpenSearchHybridStore)."""
        self.index = index
        self.llm: BaseLLM = get_llm()
        self.reranker = CrossEncoderReranker()
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex political strategy query into concrete sub-queries.
        
        Example:
        Input: "Design a winning strategy for BJP in Nandigram for 2026"
        Output: ["BJP historical performance in Nandigram",
                 "Nandigram voter demographics and segments",
                 "Key issues affecting voters in Nandigram",
                 "Opposition strength in Nandigram"]
        """
        system = """You decompose political strategy questions into specific, searchable sub-queries.
Each sub-query should focus on ONE specific aspect that can be searched independently.
Return ONLY a JSON array of strings, nothing else."""
        
        prompt = f"""Break this political strategy question into 2-4 specific sub-queries
that can be searched independently to gather comprehensive information.

Question: {query}

Consider these aspects:
- Historical voting patterns and trends
- Demographic composition and voter segments
- Current issues and grievances
- Opposition analysis
- Local power structures

Return as JSON array: ["sub-query 1", "sub-query 2", ...]"""
        
        try:
            resp = self.llm.generate(prompt=prompt, system=system, temperature=0.1)
            
            # Extract JSON array
            match = re.search(r'\[[\s\S]*\]', resp.text)
            if match:
                arr = json.loads(match.group())
                arr = [str(x).strip() for x in arr if str(x).strip()]
                return arr[:4] if arr else [query]
        except Exception:
            pass
        
        # Fallback: split by common conjunctions
        parts = re.split(r'\b(and|or|then|also|with)\b', query, flags=re.I)
        cleaned = [p.strip(" ,.;:-") for p in parts if len(p.strip()) > 10]
        return cleaned[:4] if cleaned else [query]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        """
        Perform hybrid search combining semantic and keyword search.
        """
        results = self.index.search(query, top_k=top_k)
        
        evidences = []
        for chunk, score in results:
            # Apply filters if provided
            if filters:
                match = all(
                    chunk.metadata.get(k) == v 
                    for k, v in filters.items()
                )
                if not match:
                    continue
            
            evidences.append(Evidence(
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                chunk_id=chunk.chunk_id,
                score=float(score),
                text=chunk.text,
                metadata=chunk.metadata,
            ))
        
        return evidences
    
    def rerank(self, query: str, evidences: List[Evidence], top_k: int = 10) -> List[Evidence]:
        """
        Rerank evidences using cross-encoder for better relevance.
        """
        if not evidences:
            return []
        
        passages = [e.text for e in evidences]
        ranked = self.reranker.rerank(query, passages, top_k=top_k)
        
        reranked = []
        for idx, score in ranked:
            if idx < len(evidences):
                e = evidences[idx]
                # Update score with reranker score
                reranked.append(Evidence(
                    doc_id=e.doc_id,
                    source_path=e.source_path,
                    chunk_id=e.chunk_id,
                    score=float(score),
                    text=e.text,
                    metadata=e.metadata,
                ))
        
        return reranked
    
    def compress_context(self, query: str, evidences: List[Evidence]) -> List[Evidence]:
        """
        Contextually compress evidences to keep only relevant parts.
        
        Uses LLM to extract only the sentences that directly support
        answering the query, reducing noise and improving answer quality.
        """
        if not evidences:
            return evidences
        
        # Build evidence text with markers
        evidence_text = ""
        for i, e in enumerate(evidences[:8]):
            evidence_text += f"\n[E{i+1}] {e.text[:500]}"
        
        system = """You are a strict evidence compressor. 
Extract ONLY the exact sentences that directly answer or support the query.
Do not add new information. Output JSON only."""
        
        prompt = f"""Given the user question and evidence snippets, keep ONLY the exact sentences
that directly support answering the question. Drop irrelevant content.

Question: {query}

Evidence:{evidence_text}

Return JSON array: [{{"keep_from": "E1", "text": "relevant sentence..."}}, ...]"""
        
        try:
            out = self.llm.generate(prompt=prompt, system=system, temperature=0.0)
            
            match = re.search(r'\[[\s\S]*\]', out.text)
            if match:
                arr = json.loads(match.group())
                
                # Map kept content back to evidences
                keep_map = {}
                for obj in arr:
                    key = str(obj.get("keep_from", "")).strip()
                    text = str(obj.get("text", "")).strip()
                    if key and text:
                        keep_map.setdefault(key, []).append(text)
                
                compressed = []
                for i, e in enumerate(evidences[:8]):
                    key = f"E{i+1}"
                    if key in keep_map:
                        compressed.append(Evidence(
                            doc_id=e.doc_id,
                            source_path=e.source_path,
                            chunk_id=e.chunk_id,
                            score=e.score,
                            text="\n".join(keep_map[key]),
                            metadata=e.metadata
                        ))
                
                return compressed if compressed else evidences[:5]
        except Exception:
            pass
        
        # Fallback: return top evidences uncompressed
        return evidences[:5]
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        decompose: bool = True,
        rerank: bool = True,
        compress: bool = True
    ) -> List[Evidence]:
        """
        Full Advanced RAG search pipeline.
        
        Steps:
        1. Decompose query into sub-queries
        2. Run hybrid search for each sub-query
        3. Deduplicate and aggregate results
        4. Rerank with cross-encoder
        5. Compress context
        """
        # Step 1: Decompose query
        if decompose:
            sub_queries = self.decompose_query(query)
        else:
            sub_queries = [query]
        
        # Step 2: Hybrid search for each sub-query
        gathered: List[Evidence] = []
        for sq in sub_queries:
            hits = self.hybrid_search(sq, top_k=settings.top_k_retrieval, filters=filters)
            gathered.extend(hits)
        
        # Step 3: Deduplicate by chunk_id, keeping highest score
        best = {}
        for e in gathered:
            if e.chunk_id not in best or e.score > best[e.chunk_id].score:
                best[e.chunk_id] = e
        
        dedup = list(best.values())
        dedup.sort(key=lambda e: e.score, reverse=True)
        
        # Step 4: Rerank top results
        if rerank:
            top_for_rerank = dedup[:20]
            reranked = self.rerank(query, top_for_rerank, top_k=settings.top_k_rerank)
        else:
            reranked = dedup[:settings.top_k_rerank]
        
        # Step 5: Compress context
        if compress:
            return self.compress_context(query, reranked)
        
        return reranked
