"""
Political RAG System - Unified zero-hallucination retrieval-augmented generation.

This is the main entry point for the Political Strategy Maker's RAG system.
It combines:
1. Knowledge Graph for structured data
2. Unified Vector Store (OpenSearch or FAISS) for semantic search
3. Query Router for optimal retrieval
4. Fact Verification for hallucination prevention
5. Citation-based responses

Vector Store Selection:
- If OPENSEARCH_ENDPOINT is configured → Uses OpenSearch (production-ready)
- Otherwise → Uses local FAISS (development fallback)
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from .data_schema import VerifiedAnswer, FactWithCitation
from .knowledge_graph import PoliticalKnowledgeGraph
from .vector_store import UnifiedVectorStore, get_vector_store, SearchResult
from .verified_retrieval import VerifiedRetriever, HallucinationGuard
from .query_router import QueryRouter, QueryExecutor, RouteDecision
from ..llm import get_llm

# Optional import - structured_ingest requires pandas/numpy
try:
    from .structured_ingest import StructuredIngestionPipeline, TextChunkGenerator
    STRUCTURED_INGEST_AVAILABLE = True
except ImportError:
    STRUCTURED_INGEST_AVAILABLE = False
    StructuredIngestionPipeline = None
    TextChunkGenerator = None


@dataclass
class RAGResponse:
    """Complete response from the RAG system."""
    answer: str
    confidence: float
    sources: List[str]
    facts_used: List[Dict[str, Any]]
    route_used: str
    verification_status: str
    caveats: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "facts_used": self.facts_used,
            "route_used": self.route_used,
            "verification_status": self.verification_status,
            "caveats": self.caveats
        }


class PoliticalRAGSystem:
    """
    Main Political RAG System.
    
    Usage:
        rag = PoliticalRAGSystem(data_dir, index_dir)
        rag.initialize()  # Load/build knowledge
        response = rag.query("Who won Nandigram in 2021?")
    """
    
    def __init__(self, 
                 data_dir: Path,
                 index_dir: Path,
                 auto_initialize: bool = True):
        """
        Initialize the Political RAG system.
        
        Args:
            data_dir: Directory containing political data files
            index_dir: Directory for storing indexes
            auto_initialize: Whether to auto-initialize on first query
        """
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.auto_initialize = auto_initialize
        
        # Core components (lazy initialized)
        self._kg: Optional[PoliticalKnowledgeGraph] = None
        self._vector_store: Optional[UnifiedVectorStore] = None
        self._retriever: Optional[VerifiedRetriever] = None
        self._executor: Optional[QueryExecutor] = None
        self._guard: Optional[HallucinationGuard] = None
        self._llm = None
        
        self._initialized = False
        
        # Paths
        self.kg_path = self.index_dir / "knowledge_graph.json"
        
    @property
    def kg(self) -> PoliticalKnowledgeGraph:
        """Get knowledge graph (initialize if needed)."""
        if self._kg is None:
            if self.auto_initialize:
                self.initialize()
            else:
                raise RuntimeError("RAG system not initialized. Call initialize() first.")
        return self._kg
    
    @property
    def vector_store(self) -> UnifiedVectorStore:
        """Get vector store (initialize if needed)."""
        if self._vector_store is None:
            if self.auto_initialize:
                self.initialize()
            else:
                raise RuntimeError("RAG system not initialized. Call initialize() first.")
        return self._vector_store
    
    @property
    def index(self) -> UnifiedVectorStore:
        """Alias for vector_store (backward compatibility)."""
        return self.vector_store
    
    @property
    def llm(self):
        """Get LLM instance."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def initialize(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize the RAG system.
        
        Args:
            force_rebuild: Force rebuild of indexes even if they exist
            
        Returns:
            Statistics about the initialization
        """
        if self._initialized and not force_rebuild:
            return {"status": "already_initialized"}
        
        stats = {
            "status": "initializing",
            "kg_loaded": False,
            "index_loaded": False,
            "data_ingested": False
        }
        
        # Create directories
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge graph
        self._kg = PoliticalKnowledgeGraph(storage_path=self.kg_path)
        
        # Initialize unified vector store (auto-selects OpenSearch or FAISS)
        self._vector_store = get_vector_store()
        print(f"[RAG] Vector store backend: {self._vector_store.backend_name}")
        
        # Check if we need to ingest data
        need_ingest = force_rebuild or (
            len(self._kg.constituency_profiles) == 0 and 
            self.data_dir.exists()
        )
        
        if need_ingest:
            stats["data_ingested"] = True
            ingest_stats = self._ingest_all_data()
            stats["ingest_stats"] = ingest_stats
        else:
            stats["kg_loaded"] = True
            stats["index_loaded"] = True
        
        # Initialize retriever and executor
        self._retriever = VerifiedRetriever(self._kg, self._vector_store)
        self._executor = QueryExecutor(self._kg, self._retriever)
        self._guard = HallucinationGuard(self._kg)
        
        self._initialized = True
        stats["status"] = "initialized"
        stats["kg_stats"] = self._kg.get_statistics()
        
        return stats
    
    def _ingest_all_data(self) -> Dict[str, Any]:
        """Ingest all data from the data directory."""
        print(f"Ingesting data from {self.data_dir}...")
        
        # Use structured ingestion pipeline
        pipeline = StructuredIngestionPipeline(
            data_dir=self.data_dir,
            kg_storage_path=self.kg_path
        )
        
        # Run full ingestion
        stats = pipeline.run_full_ingestion()
        
        # Get knowledge graph from pipeline
        self._kg = pipeline.get_knowledge_graph()
        
        # Generate searchable chunks and add to vector store
        chunks = pipeline.generate_searchable_chunks()
        
        print(f"Adding {len(chunks)} chunks to vector store ({self._vector_store.backend_name})...")
        
        # Convert to unified document format
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "doc_id": f"kg_{chunk['metadata'].get('type', 'unknown')}_{i}",
                "chunk_id": f"chunk_{i}",
                "text": chunk["text"],
                "source_file": "knowledge_graph",
                "metadata": chunk["metadata"]
            })
        
        indexed = self._vector_store.index_documents(documents)
        
        stats["chunks_indexed"] = indexed
        
        return stats
    
    def query(self, 
              question: str, 
              use_llm: bool = True,
              include_reasoning: bool = False) -> RAGResponse:
        """
        Query the Political RAG system.
        
        Args:
            question: User's question
            use_llm: Whether to use LLM for final answer generation
            include_reasoning: Include detailed reasoning steps
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        if not self._initialized:
            self.initialize()
        
        # Route and execute query
        router = QueryRouter(self._kg)
        decision = router.route(question)
        
        # Get verified answer
        verified = self._executor.execute(question)
        
        # Generate final answer
        if use_llm and verified.confidence > 0:
            final_answer = self._generate_llm_answer(question, verified, decision)
        else:
            final_answer = verified.answer_text
        
        # Validate response
        validated_answer, corrections = self._guard.validate_response(
            final_answer, question
        )
        
        if corrections:
            verification_status = "corrected"
        elif verified.confidence >= 0.9:
            verification_status = "verified"
        elif verified.confidence >= 0.7:
            verification_status = "high_confidence"
        else:
            verification_status = "moderate_confidence"
        
        return RAGResponse(
            answer=validated_answer,
            confidence=verified.confidence,
            sources=verified.sources,
            facts_used=[
                {
                    "text": f.fact_text,
                    "source": f.source_file,
                    "confidence": f.confidence
                }
                for f in verified.facts[:10]
            ],
            route_used=decision.route.value,
            verification_status=verification_status,
            caveats=verified.caveats + (corrections if corrections else [])
        )
    
    def _generate_llm_answer(self, 
                            question: str, 
                            verified: VerifiedAnswer,
                            decision: RouteDecision) -> str:
        """Generate LLM answer grounded in verified facts."""
        
        # Build grounded prompt
        prompt = self._guard.get_grounded_response_prompt(question, verified.facts)
        
        # Add context from verified answer
        context = f"""
ADDITIONAL VERIFIED CONTEXT:
{verified.answer_text[:3000]}

QUESTION TYPE: {decision.route.value}
CONFIDENCE LEVEL: {verified.confidence:.0%}
"""
        
        full_prompt = f"""{prompt}

{context}

Remember: Only use the facts provided. Include citations. Acknowledge limitations.

FINAL ANSWER:"""
        
        try:
            response = self.llm.generate(
                full_prompt,
                system="You are a political analyst for West Bengal elections. Answer ONLY based on verified data provided. Never invent statistics.",
                temperature=0.3
            )
            return response.text
        except Exception as e:
            # Fallback to verified answer if LLM fails
            return verified.answer_text
    
    def get_constituency_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed constituency profile."""
        if not self._initialized:
            self.initialize()
        
        profile = self._kg.get_constituency(name)
        if not profile:
            return None
        
        return {
            "ac_no": profile.ac_no,
            "ac_name": profile.ac_name,
            "district": profile.district,
            "type": profile.constituency_type.value if hasattr(profile.constituency_type, 'value') else profile.constituency_type,
            "parent_pc": profile.parent_pc,
            "2021": {
                "winner": profile.winner_2021,
                "tmc_vs": profile.tmc_vote_share_2021,
                "bjp_vs": profile.bjp_vote_share_2021,
                "margin": profile.margin_2021
            },
            "lok_sabha": {
                "tmc_2019": profile.pc_tmc_vs_2019,
                "bjp_2019": profile.pc_bjp_vs_2019,
                "tmc_2024": profile.pc_tmc_vs_2024,
                "bjp_2024": profile.pc_bjp_vs_2024,
                "swing": profile.pc_swing_2019_2024
            },
            "2026_prediction": {
                "winner": profile.predicted_winner_2026,
                "margin": profile.predicted_margin_2026,
                "rating": profile.race_rating
            },
            "vulnerability": profile.vulnerability_tag,
            "sources": profile.source_files
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Direct search without LLM processing."""
        if not self._initialized:
            self.initialize()
        
        results = self._retriever.retrieve(query, top_k=top_k)
        
        return [
            {
                "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                "source": r.source,
                "score": r.relevance_score,
                "verified": r.verification_status
            }
            for r in results
        ]
    
    def get_predictions_summary(self) -> Dict[str, Any]:
        """Get overall 2026 predictions summary."""
        if not self._initialized:
            self.initialize()
        
        seats_2021 = self._kg.count_seats_by_party(2021)
        seats_2026 = self._kg.count_predicted_seats()
        ratings = self._kg.count_by_race_rating()
        
        # Calculate changes
        changes = {}
        for party in set(list(seats_2021.keys()) + list(seats_2026.keys())):
            s21 = seats_2021.get(party, 0)
            s26 = seats_2026.get(party, 0)
            changes[party] = s26 - s21
        
        return {
            "seats_2021": seats_2021,
            "seats_2026_predicted": seats_2026,
            "changes": changes,
            "race_ratings": ratings,
            "total_constituencies": len(self._kg.constituency_profiles),
            "swing_seats": len(self._kg.get_swing_seats(5.0)),
            "bjp_vulnerable": len(self._kg.get_vulnerable_seats("BJP")),
            "tmc_vulnerable": len(self._kg.get_vulnerable_seats("TMC"))
        }
    
    def get_district_summary(self, district: str) -> str:
        """Get district-level summary."""
        if not self._initialized:
            self.initialize()
        
        return self._kg.generate_district_summary(district)
    
    def list_constituencies(self, 
                           district: Optional[str] = None,
                           pc: Optional[str] = None,
                           winner_2021: Optional[str] = None,
                           predicted_2026: Optional[str] = None,
                           race_rating: Optional[str] = None) -> List[Dict[str, Any]]:
        """List constituencies with optional filters."""
        if not self._initialized:
            self.initialize()
        
        results = []
        
        for name, profile in self._kg.constituency_profiles.items():
            # Apply filters
            if district and profile.district.upper() != district.upper():
                continue
            if pc and profile.parent_pc.upper() != pc.upper():
                continue
            if winner_2021 and profile.winner_2021.upper() != winner_2021.upper():
                continue
            if predicted_2026 and profile.predicted_winner_2026.upper() != predicted_2026.upper():
                continue
            if race_rating and profile.race_rating.lower() != race_rating.lower():
                continue
            
            results.append({
                "ac_no": profile.ac_no,
                "name": profile.ac_name,
                "district": profile.district,
                "pc": profile.parent_pc,
                "winner_2021": profile.winner_2021,
                "predicted_2026": profile.predicted_winner_2026,
                "margin_2026": profile.predicted_margin_2026,
                "race_rating": profile.race_rating
            })
        
        return sorted(results, key=lambda x: x["ac_no"])
    
    def get_swing_analysis(self) -> Dict[str, Any]:
        """Get comprehensive swing analysis."""
        if not self._initialized:
            self.initialize()
        
        # Group by PC
        pcs = {}
        for profile in self._kg.constituency_profiles.values():
            if profile.parent_pc not in pcs:
                pcs[profile.parent_pc] = {
                    "swing": profile.pc_swing_2019_2024,
                    "seats": 0,
                    "tmc_2021": 0,
                    "bjp_2021": 0,
                    "tmc_2026": 0,
                    "bjp_2026": 0
                }
            
            pcs[profile.parent_pc]["seats"] += 1
            
            if profile.winner_2021.upper() in ["TMC", "AITC"]:
                pcs[profile.parent_pc]["tmc_2021"] += 1
            elif profile.winner_2021.upper() == "BJP":
                pcs[profile.parent_pc]["bjp_2021"] += 1
            
            if profile.predicted_winner_2026.upper() == "TMC":
                pcs[profile.parent_pc]["tmc_2026"] += 1
            elif profile.predicted_winner_2026.upper() == "BJP":
                pcs[profile.parent_pc]["bjp_2026"] += 1
        
        # Sort by swing
        tmc_gaining = {k: v for k, v in pcs.items() if v["swing"] > 0}
        bjp_gaining = {k: v for k, v in pcs.items() if v["swing"] < 0}
        
        return {
            "pcs": pcs,
            "tmc_gaining": dict(sorted(tmc_gaining.items(), key=lambda x: -x[1]["swing"])),
            "bjp_gaining": dict(sorted(bjp_gaining.items(), key=lambda x: x[1]["swing"])),
            "average_swing": sum(p["swing"] for p in pcs.values()) / len(pcs) if pcs else 0
        }


# Convenience function to create RAG system
def create_political_rag(data_dir: str = None, index_dir: str = None) -> PoliticalRAGSystem:
    """Create a Political RAG system with default paths."""
    from ...config import settings
    
    if data_dir is None:
        # Look for political-data folder
        possible_paths = [
            Path(settings.data_dir).parent / "political-data",
            Path(__file__).parent.parent.parent.parent.parent / "political-data",
            Path.cwd() / "political-data"
        ]
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break
        if data_dir is None:
            data_dir = Path(settings.data_dir)
    
    if index_dir is None:
        index_dir = Path(settings.index_dir)
    
    return PoliticalRAGSystem(
        data_dir=Path(data_dir),
        index_dir=Path(index_dir)
    )

