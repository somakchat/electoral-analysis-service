"""
Orchestrator - Advanced Hierarchical Strategy Management.

This orchestrator:
1. Uses Evidence-Only RAG for grounded, citation-based answers
2. Coordinates evidence-based specialist agents (custom + CrewAI)
3. Ensures zero-hallucination through verification
4. Provides detailed reasoning chains with mandatory citations
5. Supports autonomous multi-agent collaboration via CrewAI

Evidence-Only Strategy:
- Query understanding with constituency/intent extraction
- Two-pass retrieval (constituency-specific + global context)
- Cross-encoder reranking
- Constrained generation with citation enforcement
- Confidence gating and contradiction checks
"""
from __future__ import annotations
from typing import Any, Dict, List, Callable, Awaitable, Optional
from pathlib import Path
import asyncio
import time
from datetime import datetime
import re

from app.models import Evidence, AgentUpdate, AgentStatus, FinalResponse, StrategyResult
from app.config import settings

# Conditional imports for AWS Lambda compatibility
# LocalHybridIndex uses FAISS/numpy - only available locally
try:
    from app.services.rag.local_store import LocalHybridIndex
    LOCAL_STORE_AVAILABLE = True
except ImportError:
    LOCAL_STORE_AVAILABLE = False
    LocalHybridIndex = None

# OpenSearch store for AWS (preferred)
try:
    from app.services.rag.opensearch_store import OpenSearchHybridStore
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    OpenSearchHybridStore = None

from app.services.rag.advanced_rag import AdvancedRAG
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.agents.strategic_orchestrator import StrategicOrchestrator, OrchestratedResponse
from app.services.agents import (
    IntelligenceAgent, VoterAnalystAgent, OppositionResearchAgent, GroundStrategyAgent,
    ResourceOptimizerAgent, SentimentDecoderAgent, DataScientistAgent, StrategicReporterAgent
)
from app.services.agents.base import AgentResult, AgentContext, ManagerAgent

# Evidence-Only RAG (production-grade)
try:
    from app.services.rag.evidence_rag import EvidenceOnlyRAG, create_evidence_rag
    EVIDENCE_RAG_AVAILABLE = True
except ImportError:
    EVIDENCE_RAG_AVAILABLE = False
    EvidenceOnlyRAG = None
    create_evidence_rag = None

# Query Classifier and Statistics Retriever for aggregation queries
try:
    from app.services.rag.query_classifier import QueryClassifier, QueryType, ClassifiedQuery
    from app.services.rag.statistics_retriever import StatisticsRetriever
    STATISTICS_AVAILABLE = True
    print("[Orchestrator] STATISTICS modules imported successfully")
except ImportError as e:
    STATISTICS_AVAILABLE = False
    QueryClassifier = None
    StatisticsRetriever = None
    print(f"[Orchestrator] STATISTICS import failed: {e}")

# StructuredDataIngester requires pandas (optional, for ingestion only)
try:
    from app.services.rag.structured_data_ingester import StructuredDataIngester
except ImportError:
    StructuredDataIngester = None

# CrewAI agents (optional - only if crewai is available)
try:
    from app.services.agents.crew_agents import PoliticalStrategyCrew, QuickAnalysisCrew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    PoliticalStrategyCrew = None
    QuickAnalysisCrew = None


class Orchestrator:
    """
    Main orchestrator implementing hierarchical crew architecture.
    
    Now with:
    - Political RAG integration for verified data
    - Evidence-based agent coordination
    - Multi-step reasoning with citations
    - Zero-hallucination guarantees
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                 Strategic Orchestrator                   │
    │           (Evidence-Based Coordination)                  │
    └─────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │Constituency │    │  Electoral  │    │  Campaign   │
    │  Analyst    │    │ Strategist  │    │ Strategist  │
    └─────────────┘    └─────────────┘    └─────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Political RAG  │
                    │  (Knowledge     │
                    │   Graph)        │
                    └─────────────────┘
    """
    
    def __init__(self, index_dir: str = None, use_crewai: bool = False) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.use_crewai = use_crewai and CREWAI_AVAILABLE
        
        # Initialize RAG systems - prefer OpenSearch for AWS, fallback to local
        if OPENSEARCH_AVAILABLE and settings.opensearch_endpoint:
            print("[Orchestrator] Using OpenSearch for vector search")
            self.index = OpenSearchHybridStore()
        elif LOCAL_STORE_AVAILABLE:
            print("[Orchestrator] Using LocalHybridIndex (FAISS)")
            self.index = LocalHybridIndex(index_dir=str(self.index_dir))
        else:
            print("[Orchestrator] WARNING: No vector store available!")
            self.index = None
        
        self.rag = AdvancedRAG(self.index) if self.index else None
        
        # Initialize Political RAG (lazy loaded)
        self._political_rag: Optional[PoliticalRAGSystem] = None
        self._strategic_orchestrator: Optional[StrategicOrchestrator] = None
        
        # Evidence-Only RAG (lazy loaded) - PRIMARY FOR DOCUMENT-BASED QUERIES
        self._evidence_rag: Optional[EvidenceOnlyRAG] = None
        
        # Human-in-the-Loop orchestrator (lazy loaded)
        self._hitl_orchestrator = None
        
        # Query Classifier and Statistics Retriever (for aggregation queries)
        self._query_classifier = None
        self._statistics_retriever = None
        self._structured_ingester = None
        
        # CrewAI agents (lazy loaded)
        self._crewai_full: Optional[PoliticalStrategyCrew] = None
        self._crewai_quick: Optional[QuickAnalysisCrew] = None
        
        # Legacy agents (for backward compatibility)
        self.intel = IntelligenceAgent(self.rag)
        self.voter = VoterAnalystAgent(self.rag)
        self.opp = OppositionResearchAgent(self.rag)
        self.ground = GroundStrategyAgent(self.rag)
        self.resource = ResourceOptimizerAgent(self.rag)
        self.sentiment = SentimentDecoderAgent(self.rag)
        self.data = DataScientistAgent(self.rag)
        self.reporter = StrategicReporterAgent(self.rag)
        
        # Teams
        self.research_team = [self.intel, self.opp, self.sentiment]
        self.analysis_team = [self.data, self.voter]
        self.strategy_team = [self.ground, self.resource]

        # All agents (used by legacy flows / reporting)
        self.all_agents = [
            self.intel, self.voter, self.opp, self.ground,
            self.resource, self.sentiment, self.data, self.reporter
        ]

        # Memory store (lazy, to avoid import-time side effects)
        self._memory_store = None

    def _resolve_kg_source_file(self, source: str, evidence_like: Dict[str, Any]) -> str:
        """
        If a citation comes from the Knowledge Graph, prefer showing the *real source file name*
        (CSV/XLSX/DOCX) rather than generic labels like 'knowledge_graph' or KG chunk ids.
        """
        try:
            if not source:
                return source

            src_lower = str(source).lower()
            is_kg = src_lower in {"knowledge_graph", "kg", "knowledge graph"} or src_lower.startswith("kg_")
            if not is_kg:
                return source

            # 1) If evidence dict already has a source_file, use it
            sf = None
            if isinstance(evidence_like, dict):
                sf = evidence_like.get("source_file") or evidence_like.get("source")
                if sf and str(sf).lower() not in {"knowledge_graph", "kg", "knowledge graph"}:
                    return str(sf)

            # 2) If constituency is present, resolve to constituency profile source_files
            constituency = None
            if isinstance(evidence_like, dict):
                constituency = evidence_like.get("constituency")
                if not constituency:
                    consts = evidence_like.get("constituencies") or evidence_like.get("constituencies_cited") or []
                    if isinstance(consts, list) and consts:
                        constituency = consts[0]

            if constituency:
                rag = self._get_political_rag()
                kg = getattr(rag, "kg", None)
                prof = None
                if kg and hasattr(kg, "constituency_profiles"):
                    prof = kg.constituency_profiles.get(str(constituency).upper())
                if prof and getattr(prof, "source_files", None):
                    return prof.source_files[0]

            # 3) Fallback: keep 'Knowledge Graph' label
            return "Knowledge Graph"
        except Exception:
            return source or "Knowledge Graph"

    def _get_memory_store(self):
        if self._memory_store is None:
            from app.services.memory import get_memory_store
            self._memory_store = get_memory_store()
        return self._memory_store

    def _is_pronoun_followup(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        # Very common follow-up pronouns / vague references
        return bool(re.search(r"\b(him|his|he|her|she|them|their|they|this person|that person|the person)\b", q))

    def _extract_last_person(self, turns: List[Dict[str, Any]]) -> Optional[str]:
        """
        Heuristic extraction of the most recently discussed person name from conversation turns.
        Prioritizes explicit "about <NAME>" patterns, then falls back to capitalized name patterns.
        """
        if not turns:
            return None

        # Common non-person phrases to exclude
        bad = {
            "WEST BENGAL", "STRATEGY AI", "BJP", "TMC", "CONGRESS", "CPM",
            "ASSEMBLY CONSTITUENCY", "PARLIAMENTARY CONSTITUENCY",
            "LOK SABHA", "RAJYA SABHA", "INDIA"
        }

        # Scan newest -> oldest
        for t in reversed(turns):
            text = (t.get("content") or "").strip()
            if not text:
                continue

            # 1) Pattern: "opinion about X", "about X"
            m = re.search(r"\babout\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\b", text)
            if m:
                cand = m.group(1).strip()
                if cand.upper() not in bad:
                    return cand

            # 2) Fallback: pick a plausible 2-4 token capitalized name
            # Avoid matching ALL CAPS, and avoid "West Bengal" etc.
            for m2 in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
                cand = m2.group(1).strip()
                if cand.upper() in bad:
                    continue
                # Filter common geo phrases
                if cand.upper().endswith("DISTRICT") or cand.upper().endswith("CONSTITUENCY"):
                    continue
                return cand

        return None

    def _rewrite_followup_query(self, query: str, session_id: str) -> str:
        """
        If query is a pronoun-based follow-up, rewrite it using the last discussed person
        from session history so the RAG pipeline can retrieve correct evidence.
        """
        if not self._is_pronoun_followup(query):
            return query

        mem = self._get_memory_store()
        turns = mem.get_recent_context(session_id, n_turns=12) if mem else []

        # Remove current user query turn if it was already appended by API layer
        if turns and turns[-1].get("role") == "user" and (turns[-1].get("content") == query):
            turns = turns[:-1]

        last_person = self._extract_last_person(turns)
        if not last_person:
            return query

        # Replace pronoun phrases with the resolved person name.
        # Keep it simple: rewrite to an explicit query.
        q = query.strip()
        if re.search(r"\bconstituency\b", q, re.IGNORECASE) and not re.search(r"\b(assembly|parliamentary|lok sabha|mp|mla)\b", q, re.IGNORECASE):
            # If user asked "constituency" without specifying AC vs PC, ask for both.
            return f"{q.rstrip(' ?') } of {last_person} (specify if you mean Assembly constituency (AC/MLA) or Parliamentary constituency (PC/MP); if evidence mentions both, list both)."

        # Generic pronoun replacement
        return re.sub(r"\b(him|his|he|her|she|them|their|they|this person|that person|the person)\b", last_person, q, flags=re.IGNORECASE)
    
    def _get_political_rag(self) -> PoliticalRAGSystem:
        """Get or initialize Political RAG system."""
        if self._political_rag is None:
            # Find political data folder
            data_paths = [
                self.index_dir.parent / "political-data",
                Path(__file__).parent.parent.parent.parent / "political-data",
                Path.cwd() / "political-data",
            ]
            
            data_dir = None
            for path in data_paths:
                if path.exists():
                    data_dir = path
                    break
            
            if data_dir is None:
                data_dir = data_paths[0]  # Use first path even if doesn't exist
            
            self._political_rag = PoliticalRAGSystem(
                data_dir=data_dir,
                index_dir=self.index_dir,
                auto_initialize=True
            )
        
        return self._political_rag
    
    def _get_strategic_orchestrator(self) -> StrategicOrchestrator:
        """Get or initialize Strategic Orchestrator."""
        if self._strategic_orchestrator is None:
            rag = self._get_political_rag()
            self._strategic_orchestrator = StrategicOrchestrator(rag)
        
        return self._strategic_orchestrator
    
    def _get_evidence_rag(self) -> Optional[EvidenceOnlyRAG]:
        """
        Get or initialize Evidence-Only RAG system.
        
        This is the primary system for document-based queries with:
        - Constituency-aware retrieval
        - Two-pass search (local + global)
        - Cross-encoder reranking
        - Citation enforcement
        """
        if not EVIDENCE_RAG_AVAILABLE:
            print("[Orchestrator] Evidence RAG not available")
            return None
        
        if self._evidence_rag is None:
            try:
                # Pass KG so we can resolve AC numbers (e.g., "AC-261") to official constituency names.
                rag = self._get_political_rag()
                self._evidence_rag = create_evidence_rag(kg=rag.kg)
                print("[Orchestrator] Evidence-Only RAG initialized")
            except Exception as e:
                print(f"[Orchestrator] Evidence RAG init failed: {e}")
                return None
        
        return self._evidence_rag
    
    def _get_query_classifier(self):
        """Get or initialize Query Classifier for routing queries."""
        print(f"[Orchestrator] _get_query_classifier called, STATISTICS_AVAILABLE={STATISTICS_AVAILABLE}")
        if not STATISTICS_AVAILABLE:
            print("[Orchestrator] Query Classifier not available (STATISTICS_AVAILABLE=False)")
            return None
        if self._query_classifier is None:
            try:
                self._query_classifier = QueryClassifier()
                print("[Orchestrator] Query Classifier initialized successfully")
            except Exception as e:
                print(f"[Orchestrator] Query Classifier init error: {e}")
                return None
        return self._query_classifier
    
    def _get_statistics_retriever(self):
        """Get or initialize Statistics Retriever for aggregation queries."""
        if not STATISTICS_AVAILABLE:
            return None
        if self._statistics_retriever is None:
            try:
                rag = self._get_political_rag()
                ingester = self._get_structured_ingester()
                self._statistics_retriever = StatisticsRetriever(
                    kg=rag.kg if rag else None,
                    ingester=ingester
                )
                print("[Orchestrator] Statistics Retriever initialized")
            except Exception as e:
                print(f"[Orchestrator] Statistics Retriever init failed: {e}")
                return None
        return self._statistics_retriever
    
    def _get_structured_ingester(self):
        """Get or initialize Structured Data Ingester."""
        if not STATISTICS_AVAILABLE:
            return None
        if self._structured_ingester is None:
            try:
                rag = self._get_political_rag()
                self._structured_ingester = StructuredDataIngester(kg=rag.kg if rag else None)
                print("[Orchestrator] Structured Data Ingester initialized")
            except Exception as e:
                print(f"[Orchestrator] Structured Ingester init failed: {e}")
                return None
        return self._structured_ingester
    
    def _get_crewai_full(self) -> Optional[PoliticalStrategyCrew]:
        """Get or initialize full CrewAI crew."""
        if not CREWAI_AVAILABLE:
            return None
        if self._crewai_full is None:
            rag = self._get_political_rag()
            self._crewai_full = PoliticalStrategyCrew(rag)
        return self._crewai_full
    
    def _get_crewai_quick(self) -> Optional[QuickAnalysisCrew]:
        """Get or initialize quick CrewAI crew."""
        if not CREWAI_AVAILABLE:
            return None
        if self._crewai_quick is None:
            rag = self._get_political_rag()
            self._crewai_quick = QuickAnalysisCrew(rag)
        return self._crewai_quick
    
    def _get_hitl_orchestrator(self):
        """Get or initialize Human-in-the-Loop orchestrator."""
        if self._hitl_orchestrator is None:
            from app.services.human_in_loop import create_hitl_orchestrator
            from app.services.llm import get_llm
            rag = self._get_political_rag()
            self._hitl_orchestrator = create_hitl_orchestrator(rag.kg, get_llm())
        return self._hitl_orchestrator
    
    async def _enhance_statistics_with_analysis(
        self,
        query: str,
        stats_result,
        classified_query,
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> str:
        """
        Generate human-like comprehensive analysis combining:
        1. Accurate statistics (pre-computed)
        2. Qualitative insights (from survey data)
        3. Strategic recommendations (LLM synthesis)
        """
        from app.services.llm import OpenAILLM, get_llm
        
        # Filter out long comments - only keep actual candidate names
        filtered_breakdown = {
            k: v for k, v in stats_result.breakdown.items() 
            if len(str(k)) <= 50
        }
        
        sorted_candidates = sorted(
            filtered_breakdown.items(), 
            key=lambda x: x[1].get('count', 0), 
            reverse=True
        )[:10]
        
        total = stats_result.total_count
        
        # === PHASE 1: Build the quantitative section (fast, no LLM) ===
        top3 = sorted_candidates[:3]
        top1_name, top1_data = top3[0] if top3 else ("Unknown", {"count": 0, "percentage": 0})
        top2_name, top2_data = top3[1] if len(top3) > 1 else ("Unknown", {"count": 0, "percentage": 0})
        top3_name, top3_data = top3[2] if len(top3) > 2 else ("Unknown", {"count": 0, "percentage": 0})
        
        # Build statistics table
        stats_table = "| Candidate | Votes | Percentage |\n|-----------|-------|------------|\n"
        for name, data in sorted_candidates[:7]:
            stats_table += f"| {name} | {data.get('count', 0)} | {data.get('percentage', 0):.1f}% |\n"
        
        # === PHASE 2: Generate qualitative analysis with LLM ===
        try:
            fast_llm = OpenAILLM(force_model="gpt-4o-mini")
        except:
            fast_llm = get_llm()
        
        # Focused prompt - only ask for qualitative insights (faster)
        analysis_prompt = f"""You are an expert political analyst for West Bengal BJP. Based on survey data showing CM preferences, provide ONLY the qualitative analysis sections.

SURVEY DATA (417 responses total):
- {top1_name}: {top1_data.get('percentage', 0):.1f}% ({top1_data.get('count', 0)} votes) - LEADER
- {top2_name}: {top2_data.get('percentage', 0):.1f}% ({top2_data.get('count', 0)} votes) - Runner-up  
- {top3_name}: {top3_data.get('percentage', 0):.1f}% ({top3_data.get('count', 0)} votes) - Third

Generate ONLY these 2 sections (be specific and insightful):

## Qualitative Insights: Why These Candidates?

**A. {top1_name}**
- Key strengths and appeal (based on typical political positioning)
- Themes: [list 3-4 themes like "Academic," "Clean image," etc.]
- Voter sentiment summary

**B. {top2_name}**  
- Key strengths and appeal
- Themes: [list 3-4 themes]
- Voter sentiment summary

**C. {top3_name}**
- Key strengths and appeal
- Themes: [list 3-4 themes]
- Voter sentiment summary

## Strategic Projection

Provide 2-3 strategic recommendations for the party based on this preference data. Consider:
- Who should be the projected face?
- What partnership/balance might work best?
- Key message for different voter segments

Be concise but insightful. Write professionally."""

        if send_update:
            await send_update(AgentUpdate(
                agent="qualitative_analyzer",
                status=AgentStatus.WORKING,
                task="Generating comprehensive political analysis"
            ))
        
        try:
            loop = asyncio.get_event_loop()
            llm_response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: fast_llm.generate(analysis_prompt, max_tokens=1200)),
                timeout=18.0
            )
            
            qualitative_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
        except asyncio.TimeoutError:
            print("[Orchestrator] LLM timed out, using template")
            qualitative_text = self._generate_template_analysis(top1_name, top2_name, top3_name)
        except Exception as e:
            print(f"[Orchestrator] LLM error: {e}")
            qualitative_text = self._generate_template_analysis(top1_name, top2_name, top3_name)
        
        if send_update:
            await send_update(AgentUpdate(
                agent="qualitative_analyzer",
                status=AgentStatus.DONE,
                task="Analysis complete"
            ))
        
        # === PHASE 3: Combine into final human-like output ===
        final_output = f"""## Executive Summary

The survey data shows a clear preference for **{top1_name}** as the primary choice for Chief Minister, securing **{top1_data.get('percentage', 0):.1f}%** of the vote ({top1_data.get('count', 0)} out of {total} responses). **{top2_name}** remains a strong second at **{top2_data.get('percentage', 0):.1f}%**, while **{top3_name}** maintains a solid base at **{top3_data.get('percentage', 0):.1f}%**.

---

## Quantitative Analysis: The Frontrunners

Based on {total} survey responses, the distribution of preferences is:

{stats_table}

### Key Observations:
- **{top1_name}** commands the strongest support with nearly a third of all preferences
- The top 3 candidates account for over 60% of total preferences
- There is a clear two-tier structure: the frontrunner, followed by a competitive second tier

---

{qualitative_text}

---

*Analysis based on verified survey data ({total} responses)*"""
        
        return final_output
    
    def _generate_template_analysis(self, top1: str, top2: str, top3: str) -> str:
        """Generate template analysis when LLM times out."""
        return f"""## Qualitative Insights

**{top1}** - The Leading Choice
- Positioned as a credible, party-aligned leader
- Appeals to voters seeking stability and organizational strength

**{top2}** - The Strong Contender  
- Known for aggressive political stance
- Appeals to voters seeking active opposition to current government

**{top3}** - The Grassroots Option
- Strong party organization background
- Appeals to core party workers and booth-level supporters

## Strategic Projection

Based on the data, the party's strongest positioning would be to project the leading candidate while leveraging the strengths of the runner-up for active campaigning."""
    
    def _extract_constituency(self, query: str) -> Optional[str]:
        """Extract constituency name from query."""
        import re
        
        # Common West Bengal constituency names
        constituencies = [
            'Karimpur', 'Tehatta', 'Ranaghat', 'Krishnanagar', 'Nabadwip',
            'Chakdaha', 'Haringhata', 'Bagda', 'Bangaon', 'Bongaon',
            'Basirhat', 'Hingalganj', 'Gosaba', 'Diamond Harbour',
            'Kolkata', 'Howrah', 'Hooghly', 'Murshidabad', 'Malda',
            'Darjeeling', 'Siliguri', 'Jalpaiguri', 'Cooch Behar',
            'Purulia', 'Bankura', 'Birbhum', 'Burdwan', 'Asansol'
        ]
        
        query_lower = query.lower()
        for const in constituencies:
            if const.lower() in query_lower:
                return const.upper()
        
        return None
    
    async def _generate_constituency_ground_report(
        self,
        query: str,
        constituency: str,
        party: Optional[str],
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive ground report for a constituency with actual survey feedback."""
        from app.services.llm import OpenAILLM, get_llm
        
        party = party or 'BJP'
        
        if send_update:
            await send_update(AgentUpdate(
                agent="ground_report",
                status=AgentStatus.WORKING,
                task=f"Gathering ground-level feedback from {constituency}"
            ))
        
        # Direct OpenSearch search for survey data (bypasses constituency metadata filter)
        from app.services.rag.political_opensearch import PoliticalOpenSearchClient
        
        all_citations = []
        all_feedback = []
        
        try:
            os_client = PoliticalOpenSearchClient()
            
            # Search for survey responses containing constituency name in text
            search_body = {
                'query': {
                    'bool': {
                        'should': [
                            {'match_phrase': {'text': f'{constituency} {party}'}},
                            {'match': {'text': f'Assembly Constituency {constituency} feeling {party}'}},
                            {'match': {'text': f'{constituency} local problems issues hospital'}},
                            {'bool': {
                                'must': [
                                    {'match': {'text': constituency}},
                                    {'match': {'source_file': 'Nagarik'}}
                                ]
                            }},
                            {'bool': {
                                'must': [
                                    {'match': {'text': constituency}},
                                    {'match': {'text': f'{party} opinion feeling'}}
                                ]
                            }}
                        ],
                        'minimum_should_match': 1
                    }
                },
                'size': 15,
                '_source': ['text', 'source_file', 'constituency', 'district']
            }
            
            result = os_client.client.search(index=os_client.index_name, body=search_body)
            hits = result.get('hits', {}).get('hits', [])
            
            print(f"[Orchestrator] Direct OpenSearch for {constituency}: {len(hits)} hits")
            
            for hit in hits:
                source = hit.get('_source', {})
                text = source.get('text', '')
                source_file = source.get('source_file', 'Survey')
                
                # Filter for data containing constituency name
                if text and constituency.lower() in text.lower() and len(text) > 100:
                    # Check if it's survey/feedback data
                    if any(kw in text.lower() for kw in ['feeling', 'opinion', 'local problems', 'timestamp']):
                        if text[:100] not in [f.get('text_preview', '')[:100] for f in all_citations]:
                            all_feedback.append(f"[Source: {source_file}]\n{text[:900]}")
                            all_citations.append({
                                'text_preview': text[:600],
                                'source_file': source_file,
                                'constituency': constituency,
                                'content': text[:600]
                            })
            
        except Exception as e:
            print(f"[Orchestrator] OpenSearch search error: {e}")
        
        if not all_feedback:
            print(f"[Orchestrator] No ground survey data found for {constituency}")
            return None
        
        feedback_text = "\n\n---\n\n".join(all_feedback[:6])  # Limit to 6 passages
        
        # Use LLM to synthesize a balanced ground report
        try:
            fast_llm = OpenAILLM(force_model="gpt-4o-mini")
        except:
            fast_llm = get_llm()
        
        # Determine incumbent party (usually TMC in West Bengal)
        incumbent = "TMC"  # Default for WB constituencies
        
        prompt = f"""You are a political analyst creating a ground report based on ACTUAL survey responses and feedback from {constituency} residents.

CONSTITUENCY: {constituency}
PARTY BEING ANALYZED: {party}
INCUMBENT GOVERNMENT: {incumbent} (the current ruling party)
QUERY: {query}

ACTUAL SURVEY RESPONSES AND FEEDBACK FROM {constituency} RESIDENTS:
{feedback_text}

Based ONLY on the above actual feedback, create a ground report. DO NOT invent or generalize - use only what's in the data.

## Ground Report: Public Opinion on {party} in {constituency}

### Overview
(2-3 sentences summarizing the actual feedback received from {constituency} residents about {party})

### What People Like About {party} (Positive Feedback)
- Extract ANY positive comments/opinions about {party} from the survey data
- Look for phrases like "support on rise", "good leadership", "positive change", etc.
- Quote the exact feedback with [Source N] citations
- Example: If someone says "{party} has weak organisation but support of people on rise", the positive part is "support of people on rise"

### What People Dislike About {party} (Criticisms)  
- Extract criticisms and concerns about {party} from the survey data
- Look for phrases like "weak organisation", "no face", "chaos", "not good", etc.
- Quote the exact feedback with [Source N] citations
- Include organizational weaknesses mentioned by respondents

### Local Issues Mentioned by {constituency} Residents
- List the ACTUAL local problems mentioned by respondents (these are issues under {incumbent} governance)
- These represent opportunities for {party} to address if they win
- Examples: roads, hospitals, employment, corruption, etc.

### Voter Sentiment Summary
(2-3 sentences on the overall sentiment towards {party} based on the actual feedback)

### Strategic Opportunities for {party}
- Based on the local issues mentioned (which are under {incumbent} governance), what can {party} promise/address?
- How can {party} convert criticisms into campaign points?

CRITICAL RULES:
1. Only use information that is ACTUALLY in the survey data above
2. Use [Source 1], [Source 2], etc. to cite specific feedback
3. If data is limited, say so - don't invent opinions
4. Local issues are problems under the CURRENT {incumbent} government - they are opportunities for {party}
5. Be specific to {constituency} - no generic political commentary"""

        if send_update:
            await send_update(AgentUpdate(
                agent="ground_report",
                status=AgentStatus.WORKING,
                task="Synthesizing balanced ground report from actual feedback"
            ))
        
        try:
            loop = asyncio.get_event_loop()
            llm_response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: fast_llm.generate(prompt, max_tokens=1800)),
                timeout=20.0
            )
            
            response_text = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
            
            if send_update:
                await send_update(AgentUpdate(
                    agent="ground_report",
                    status=AgentStatus.DONE,
                    task=f"Ground report complete with {len(all_citations)} sources"
                ))
            
            # Return structured response with citations
            return {
                "answer": response_text,
                "citations": all_citations,
                "sources_count": len(all_citations)
            }
            
        except Exception as e:
            print(f"[Orchestrator] Ground report generation error: {e}")
            return None
    
    async def run(
        self,
        query: str,
        session_id: str = None,
        constituency: str = None,
        party: str = None,
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None,
        use_crewai: bool = False,
        enable_hitl: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the full hierarchical workflow with evidence-based agents.
        
        Args:
            query: User's strategy question
            session_id: Session ID for context
            constituency: Target constituency (optional)
            party: Target party (optional)
            send_update: Callback for real-time updates
            use_crewai: Use CrewAI autonomous agents (default: False)
            enable_hitl: Enable Human-in-the-Loop interactions (default: True)
        
        Returns:
            Dict with answer, citations, confidence, interactions, etc.
        """
        start_time = time.time()
        session_id = session_id or "default"

        # Resolve pronoun-based follow-ups using session memory (interactive chatbot behavior)
        try:
            rewritten = self._rewrite_followup_query(query, session_id)
            if rewritten and rewritten != query:
                if send_update:
                    await send_update(AgentUpdate(
                        agent="context",
                        status=AgentStatus.WORKING,
                        task=f"Resolved follow-up reference using conversation context"
                    ))
                query = rewritten
        except Exception as _coref_err:
            # Never block main flow on coref issues
            pass
        
        # Build context
        context = {}
        if constituency:
            context['constituency'] = constituency.upper()
        if party:
            context['party'] = party.upper()
        
        # ===== HUMAN-IN-THE-LOOP: Pre-processing =====
        if enable_hitl:
            try:
                hitl = self._get_hitl_orchestrator()

                # Check if clarification is needed OR resume original query after user selects an option
                if hasattr(hitl, "pre_process_and_rewrite"):
                    clarification, effective_query = hitl.pre_process_and_rewrite(query, session_id)
                    query = effective_query
                else:
                    clarification = hitl.pre_process_query(query, session_id)

                if clarification:
                    # Return clarification request instead of processing
                    if send_update:
                        await send_update(AgentUpdate(
                            agent="hitl",
                            status=AgentStatus.WORKING,
                            task=f"Clarification needed: {clarification.message}"
                        ))
                    
                    return {
                        "answer": clarification.message,
                        "needs_clarification": True,
                        "interaction": clarification.to_dict(),
                        "citations": [],
                        "agents_used": ["hitl"],
                        "confidence": 0.5
                    }
                
                # Update context with entities from conversation
                hitl_context = hitl.get_context(session_id)
                context.update(hitl_context.entities)
                
            except Exception as hitl_error:
                # HITL errors shouldn't block main processing
                print(f"HITL pre-processing warning: {hitl_error}")
        
        # ===== QUERY CLASSIFICATION: Route aggregation queries to statistics =====
        print(f"[Orchestrator] Starting query classification for: {query[:50]}...")
        classifier = self._get_query_classifier()
        classified_query = None
        
        print(f"[Orchestrator] Classifier returned: {classifier is not None}")
        if classifier and STATISTICS_AVAILABLE:
            try:
                classified_query = classifier.classify(query)
                print(f"[Orchestrator] Query classified as: {classified_query.query_type.value} ({classified_query.intent.value})")
                
                # For aggregation/survey queries, try statistics retriever first
                if classified_query.use_statistics and classified_query.query_type.value in ['aggregation', 'survey']:
                    stats_retriever = self._get_statistics_retriever()
                    
                    if stats_retriever:
                        if send_update:
                            await send_update(AgentUpdate(
                                agent="statistics",
                                status=AgentStatus.WORKING,
                                task="Retrieving pre-computed statistics for accurate counts"
                            ))
                        
                        stats_result = stats_retriever.retrieve(classified_query)
                        
                        if stats_result.found:
                            if send_update:
                                await send_update(AgentUpdate(
                                    agent="statistics",
                                    status=AgentStatus.DONE,
                                    task=f"Found accurate statistics from {len(stats_result.sources)} sources"
                                ))
                            
                            # For survey queries, enhance statistics with qualitative analysis
                            enhanced_answer = await self._enhance_statistics_with_analysis(
                                query=query,
                                stats_result=stats_result,
                                classified_query=classified_query,
                                send_update=send_update
                            )
                            
                            result = {
                                "answer": enhanced_answer,
                                "citations": [
                                    Evidence(
                                        source=src,
                                        content=f"Pre-computed statistics from survey data",
                                        relevance_score=stats_result.confidence,
                                        doc_id=f"stats_{src[:20]}",
                                        source_type="statistics"
                                    )
                                    for src in stats_result.sources
                                ],
                                "agents_used": ["statistics_retriever", "qualitative_analyzer"],
                                "confidence": stats_result.confidence,
                                "verification_status": "statistics_verified",
                                "query_type": classified_query.query_type.value,
                                "statistics_data": {
                                    "total_count": stats_result.total_count,
                                    "value_count": stats_result.value_count,
                                    "percentage": stats_result.percentage,
                                    "breakdown": stats_result.breakdown
                                }
                            }
                            
                            # Post-process and return
                            if enable_hitl:
                                try:
                                    hitl = self._get_hitl_orchestrator()
                                    result = hitl.post_process_response(query, result, session_id)
                                except Exception as hitl_error:
                                    print(f"HITL post-processing warning: {hitl_error}")
                                    result['interactions'] = []
                            
                            return result
                        else:
                            print(f"[Orchestrator] Statistics not found, falling back to RAG")
                
                # For constituency-specific opinion queries, use ground report handler
                if classified_query.query_type.value == 'qualitative' and classified_query.intent.value == 'opinion':
                    # Check if this is about a specific constituency
                    constituency = self._extract_constituency(query)
                    if constituency:
                        print(f"[Orchestrator] Constituency opinion query detected: {constituency}")
                        ground_report = await self._generate_constituency_ground_report(
                            query=query,
                            constituency=constituency,
                            party=classified_query.entities[0] if classified_query.entities else None,
                            send_update=send_update
                        )
                        if ground_report and isinstance(ground_report, dict):
                            # Build proper citations from the retrieved evidence
                            citations = []
                            for i, cite in enumerate(ground_report.get('citations', [])[:10]):
                                citations.append(Evidence(
                                    source=cite.get('source_file', f'Survey Response {i+1}'),
                                    content=cite.get('text_preview', cite.get('content', ''))[:400],
                                    relevance_score=cite.get('score', 0.85),
                                    doc_id=cite.get('chunk_id', f'ground_{i}'),
                                    source_type='survey_response',
                                    constituencies=[constituency] if constituency else [],
                                    districts=[cite.get('district')] if cite.get('district') else []
                                ))
                            
                            result = {
                                "answer": ground_report.get('answer', ''),
                                "citations": citations,
                                "agents_used": ["ground_report_analyzer"],
                                "confidence": 0.9,
                                "verification_status": "ground_verified",
                                "query_type": "ground_report",
                                "sources_used": ground_report.get('sources_count', len(citations))
                            }
                            return result
                
            except Exception as classifier_error:
                print(f"[Orchestrator] Query classification warning: {classifier_error}")
        
        # ===== PRIMARY: Evidence-Only RAG (for grounded, citation-based answers) =====
        evidence_rag = self._get_evidence_rag()
        if evidence_rag and EVIDENCE_RAG_AVAILABLE:
            try:
                if send_update:
                    await send_update(AgentUpdate(
                        agent="evidence_rag",
                        status=AgentStatus.WORKING,
                        task="Searching documents with constituency-aware retrieval"
                    ))
                
                # Get conversation history for context-aware generation
                conversation_history = []
                try:
                    mem = self._get_memory_store()
                    if mem:
                        recent_turns = mem.get_recent_context(session_id, n_turns=10)
                        # Convert to simple format for the prompt
                        conversation_history = [
                            {"role": t.get("role", "user"), "content": t.get("content", "")}
                            for t in recent_turns
                        ]
                except Exception as mem_err:
                    print(f"[Orchestrator] Warning: Could not get conversation history: {mem_err}")
                
                # Run evidence-only RAG with conversation context
                evidence_result = await evidence_rag.answer(query, conversation_history=conversation_history)
                
                # Check if we got sufficient evidence
                if evidence_result.evidence_quality in ["high", "medium"]:
                    if send_update:
                        await send_update(AgentUpdate(
                            agent="evidence_rag",
                            status=AgentStatus.DONE,
                            task=f"Found {len(evidence_result.citations)} relevant sources"
                        ))
                    
                    # Convert to standard response format
                    result = {
                        "answer": evidence_result.answer,
                        "citations": [
                            Evidence(
                                source=self._resolve_kg_source_file(c.get("source", "Document"), c),
                                content=c.get("text_preview", ""),
                                relevance_score=c.get("score", 0.8),
                                doc_id=c.get("chunk_id", ""),
                                source_type=c.get("source_type", "document"),
                                constituencies=[c.get("constituency")] if c.get("constituency") else [],
                                districts=[c.get("district")] if c.get("district") else []
                            )
                            for c in evidence_result.citations
                        ],
                        "agents_used": ["evidence_rag"],
                        "confidence": evidence_result.confidence,
                        "verification_status": f"evidence_quality:{evidence_result.evidence_quality}",
                        "warnings": evidence_result.warnings
                    }
                    
                    # Post-process and return
                    if enable_hitl:
                        try:
                            hitl = self._get_hitl_orchestrator()
                            result = hitl.post_process_response(query, result, session_id)
                        except Exception as hitl_error:
                            print(f"HITL post-processing warning: {hitl_error}")
                            result['interactions'] = []
                    
                    return result
                else:
                    # Evidence insufficient - for strategy queries, return honest response
                    # Don't fall back to KG-only data which could mislead users
                    if send_update:
                        await send_update(AgentUpdate(
                            agent="evidence_rag",
                            status=AgentStatus.DONE,
                            task=f"Evidence quality: {evidence_result.evidence_quality}. Returning available information."
                        ))
                    
                    print(f"[Orchestrator] Evidence quality: {evidence_result.evidence_quality}")
                    
                    # Return the honest "insufficient evidence" response
                    # This prevents hallucination by being transparent about data gaps
                    result = {
                        "answer": evidence_result.answer,
                        "citations": [
                            Evidence(
                                source=self._resolve_kg_source_file(c.get("source", "Document"), c),
                                content=c.get("text_preview", ""),
                                relevance_score=c.get("score", 0.8),
                                doc_id=c.get("chunk_id", ""),
                                source_type=c.get("source_type", "document"),
                                constituencies=[c.get("constituency")] if c.get("constituency") else [],
                                districts=[c.get("district")] if c.get("district") else []
                            )
                            for c in evidence_result.citations
                        ],
                        "agents_used": ["evidence_rag"],
                        "confidence": evidence_result.confidence,
                        "verification_status": f"evidence_quality:{evidence_result.evidence_quality}",
                        "warnings": evidence_result.warnings,
                        "evidence_insufficient": True
                    }
                    
                    # Post-process and return
                    if enable_hitl:
                        try:
                            hitl = self._get_hitl_orchestrator()
                            result = hitl.post_process_response(query, result, session_id)
                        except Exception as hitl_error:
                            print(f"HITL post-processing warning: {hitl_error}")
                            result['interactions'] = []
                    
                    return result
                    
            except Exception as evidence_error:
                print(f"[Orchestrator] Evidence RAG error: {evidence_error}")
                import traceback
                traceback.print_exc()
        
        # ===== FALLBACK: CrewAI or Standard Orchestrator =====
        # Use CrewAI if requested and available
        if (use_crewai or self.use_crewai) and CREWAI_AVAILABLE:
            result = await self._run_crewai(query, context, send_update)
        else:
            # Use standard orchestrator
            try:
                result = await self._run_standard(query, context, send_update)
            except Exception as e:
                # Fallback to legacy workflow
                import traceback
                traceback.print_exc()
                
                if send_update:
                    await send_update(AgentUpdate(
                        agent="fallback",
                        status=AgentStatus.WORKING,
                        task=f"Using legacy pipeline: {str(e)[:100]}"
                    ))
                
                result = await self._legacy_run(query, session_id, constituency, party, send_update)
        
        # ===== HUMAN-IN-THE-LOOP: Post-processing =====
        if enable_hitl:
            try:
                hitl = self._get_hitl_orchestrator()
                result = hitl.post_process_response(query, result, session_id)
            except Exception as hitl_error:
                # HITL errors shouldn't affect the response
                print(f"HITL post-processing warning: {hitl_error}")
                result['interactions'] = []
        
        return result
    
    async def _run_with_fallback(
        self,
        query: str,
        context: Dict[str, Any],
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None,
        session_id: str = None,
        constituency: str = None,
        party: str = None
    ) -> Dict[str, Any]:
        """Run query with fallback handling - extracted for cleaner code."""
        try:
            return await self._run_standard(query, context, send_update)
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            if send_update:
                await send_update(AgentUpdate(
                    agent="fallback",
                    status=AgentStatus.WORKING,
                    task=f"Using legacy pipeline: {str(e)[:100]}"
                ))
            
            return await self._legacy_run(query, session_id, constituency, party, send_update)
    
    async def _run_crewai(
        self,
        query: str,
        context: Dict[str, Any],
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Run query using CrewAI autonomous agents.
        
        This provides:
        - Multi-agent collaboration
        - Autonomous decision making
        - Tool-based data access
        - Quality control and verification
        """
        start_time = time.time()
        
        if send_update:
            await send_update(AgentUpdate(
                agent="crewai",
                status=AgentStatus.WORKING,
                task="Initializing autonomous agent crew"
            ))
        
        try:
            # Determine if we need full crew or quick analysis
            is_complex = any(w in query.lower() for w in [
                'strategy', 'plan', 'campaign', 'win', 'compare', 'why', 'how'
            ])
            
            if is_complex:
                if send_update:
                    await send_update(AgentUpdate(
                        agent="crewai",
                        status=AgentStatus.WORKING,
                        task="Deploying full agent crew for strategic analysis"
                    ))
                
                crew = self._get_crewai_full()
                result = await asyncio.to_thread(
                    crew.analyze_query, query, context
                )
            else:
                if send_update:
                    await send_update(AgentUpdate(
                        agent="crewai",
                        status=AgentStatus.WORKING,
                        task="Quick analysis with research agent"
                    ))
                
                crew = self._get_crewai_quick()
                result = await asyncio.to_thread(
                    crew.analyze, query
                )
            
            if send_update:
                await send_update(AgentUpdate(
                    agent="crewai",
                    status=AgentStatus.DONE,
                    task="Analysis complete"
                ))
            
            execution_time = time.time() - start_time
            
            # Format response
            return {
                "answer": result.get("answer", "No response generated"),
                "citations": [],
                "agents_used": result.get("agents_used", ["crewai"]),
                "confidence": result.get("confidence", 0.7),
                "verification_status": "crewai_verified",
                "execution_time_ms": int(execution_time * 1000),
                "framework": "crewai"
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            if send_update:
                await send_update(AgentUpdate(
                    agent="crewai",
                    status=AgentStatus.ERROR,
                    task=f"CrewAI error: {str(e)[:100]}"
                ))
            
            # Fallback to standard orchestrator
            return await self._run_standard(query, context, send_update)
    
    async def _run_standard(
        self,
        query: str,
        context: Dict[str, Any],
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Standard orchestrator run (non-CrewAI)."""
        start_time = time.time()
        
        try:
            orchestrator = self._get_strategic_orchestrator()
            
            async def forward_update(update: dict):
                if send_update:
                    if update.get('type') == 'agent_activity':
                        await send_update(AgentUpdate(
                            agent=update.get('agent', 'system'),
                            status=AgentStatus.WORKING if update.get('status') == 'working' else AgentStatus.DONE,
                            task=update.get('task', update.get('message', ''))
                        ))
                    else:
                        await send_update(AgentUpdate(
                            agent='orchestrator',
                            status=AgentStatus.WORKING,
                            task=update.get('message', str(update))
                        ))
            
            result = await orchestrator.process_query(
                query=query,
                context=context,
                send_update=forward_update
            )
            
            # Build detailed citations from evidence
            citations = []
            for i, ev in enumerate(result.evidence[:10]):
                # Extract all available fields for rich citations
                source = ev.get('source') or ev.get('source_path') or 'Electoral Data'
                content = ev.get('content') or ev.get('text') or ''
                score = ev.get('score') or ev.get('relevance_score') or 0.85
                source_type = ev.get('source_type') or 'Data Source'

                # If this is KG evidence, resolve to underlying source file when possible
                source = self._resolve_kg_source_file(source, ev)
                
                # Build document ID from source info
                doc_id = f"{source_type}_{i+1}"
                if ev.get('constituencies_cited'):
                    doc_id = f"Constituency: {ev['constituencies_cited'][0]}" if len(ev['constituencies_cited']) == 1 else f"Constituencies: {len(ev['constituencies_cited'])}"
                elif ev.get('districts_cited'):
                    doc_id = f"District: {ev['districts_cited'][0]}" if len(ev['districts_cited']) == 1 else f"Districts: {len(ev['districts_cited'])}"
                
                citations.append(Evidence(
                    source=source,
                    content=content[:500] if content else f"Analysis from {source}",
                    relevance_score=float(score) if score else 0.85,
                    doc_id=doc_id,
                    source_type=source_type,
                    constituencies=ev.get('constituencies_cited', []),
                    districts=ev.get('districts_cited', [])
                ))
            
            # Add data sources if available and not already cited
            existing_sources = {c.source for c in citations}
            for source in result.sources[:5]:
                if source not in existing_sources:
                    citations.append(Evidence(
                        source=source,
                        content=f"Data retrieved from {source}",
                        relevance_score=0.80,
                        doc_id=f"Source: {source}",
                        source_type="Data Source"
                    ))
            
            execution_time = time.time() - start_time
            
            return {
                "answer": result.answer,
                "citations": citations,
                "agents_used": result.agents_used,
                "confidence": result.confidence,
                "verification_status": result.verification_status,
                "claims": result.claims,
                "reasoning_trace": result.reasoning_trace,
                "execution_time_ms": int(execution_time * 1000)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Analysis error: {str(e)}",
                "citations": [],
                "agents_used": [],
                "confidence": 0.0
            }
    
    async def _legacy_run(
        self,
        query: str,
        session_id: str = None,
        constituency: str = None,
        party: str = None,
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Legacy workflow using original agents."""
        
        context = AgentContext(
            query=query,
            session_id=session_id,
            constituency=constituency or "",
            party=party or "",
            metadata={}
        )
        
        results = []
        citations = []
        
        # Phase 1: Research
        if send_update:
            await send_update(AgentUpdate(
                agent="intelligence",
                status=AgentStatus.WORKING,
                task="Gathering constituency intelligence"
            ))
        
        try:
            intel_result = await asyncio.to_thread(self.intel.execute, context)
            results.append(intel_result)
            citations.extend(intel_result.citations)
        except Exception as e:
            print(f"Intel agent error: {e}")
        
        if send_update:
            await send_update(AgentUpdate(
                agent="intelligence",
                status=AgentStatus.DONE
            ))
        
        # Phase 2: Analysis
        if send_update:
            await send_update(AgentUpdate(
                agent="data_scientist",
                status=AgentStatus.WORKING,
                task="Analyzing patterns"
            ))
        
        try:
            data_context = AgentContext(
                query=query,
                session_id=session_id,
                constituency=constituency or "",
                party=party or "",
                metadata={"previous_results": [r.output for r in results]}
            )
            data_result = await asyncio.to_thread(self.data.execute, data_context)
            results.append(data_result)
            citations.extend(data_result.citations)
        except Exception as e:
            print(f"Data agent error: {e}")
        
        if send_update:
            await send_update(AgentUpdate(
                agent="data_scientist",
                status=AgentStatus.DONE
            ))
        
        # Phase 3: Synthesis
        if send_update:
            await send_update(AgentUpdate(
                agent="reporter",
                status=AgentStatus.WORKING,
                task="Synthesizing strategy"
            ))
        
        try:
            reporter_context = AgentContext(
                query=query,
                session_id=session_id,
                constituency=constituency or "",
                party=party or "",
                metadata={"all_results": [r.output for r in results]}
            )
            final_result = await asyncio.to_thread(self.reporter.execute, reporter_context)
            citations.extend(final_result.citations)
            answer = final_result.output.get("answer", "Analysis completed.")
        except Exception as e:
            answer = f"Analysis completed with partial results. Error in synthesis: {str(e)}"
        
        if send_update:
            await send_update(AgentUpdate(
                agent="reporter",
                status=AgentStatus.DONE
            ))
        
        return {
            "answer": answer,
            "citations": citations[:10],
            "agents_used": ["intelligence", "data_scientist", "reporter"],
            "confidence": 0.6,
            "workflow": "legacy"
        }
    
    async def quick_analysis(self, query: str) -> Dict[str, Any]:
        """Quick analysis using Political RAG directly."""
        try:
            rag = self._get_political_rag()
            response = rag.query(query, use_llm=True)
            
            citations = [
                Evidence(source=s, content=f"From {s}", relevance_score=0.9)
                for s in response.sources[:5]
            ]
            
            return {
                "answer": response.answer,
                "citations": citations,
                "agents_used": ["political_rag"],
                "confidence": response.confidence,
                "verification_status": response.verification_status
            }
        except Exception as e:
            return {
                "answer": f"Analysis error: {str(e)}",
                "citations": [],
                "agents_used": [],
                "confidence": 0.0
            }
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get 2026 predictions summary."""
        rag = self._get_political_rag()
        return rag.get_predictions_summary()
    
    def get_constituency(self, name: str) -> Optional[Dict]:
        """Get constituency profile."""
        rag = self._get_political_rag()
        return rag.get_constituency_profile(name)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search political data."""
        rag = self._get_political_rag()
        return rag.search(query, top_k=top_k)
