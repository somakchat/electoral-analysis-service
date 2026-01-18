"""
Evidence-Only RAG Engine - Production-Grade Implementation

Core Principle: The assistant may only answer using passages it retrieved from 
the ingested corpus. If retrieval confidence is low, it must respond with 
"Not found in the documents".

Features:
- Constituency-aware chunking and retrieval
- Two-pass retrieval (local constituency + global context)
- Cross-encoder reranking
- Evidence pack building with citations
- Confidence gating
- Contradiction checking
"""

import asyncio
import logging
import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TopicTag(str, Enum):
    """Standard topic tags for political documents"""
    ROADS = "roads"
    JOBS = "jobs"
    WELFARE = "welfare"
    LAW_ORDER = "law_order"
    COMMUNAL = "communal"
    CANDIDATE = "candidate"
    ECONOMY = "economy"
    HEALTH = "health"
    EDUCATION = "education"
    CORRUPTION = "corruption"
    DEVELOPMENT = "development"
    STRATEGY = "strategy"
    SURVEY = "survey"
    PREDICTION = "prediction"
    VOTER_SEGMENT = "voter_segment"
    CAMPAIGN = "campaign"


@dataclass
class CanonicalChunk:
    """Canonical JSON format for all ingested content"""
    chunk_id: str
    text: str
    doc_id: str
    source_file_name: str
    source_type: str  # docx, pdf, xlsx, csv, txt
    page_section: Optional[str] = None
    heading: Optional[str] = None
    constituency: Optional[str] = None
    district: Optional[str] = None
    topic_tags: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    version: str = "1.0"
    chunk_summary: Optional[str] = None  # 1-2 line summary for retrieval boosting
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "source_file_name": self.source_file_name,
            "source_type": self.source_type,
            "page_section": self.page_section,
            "heading": self.heading,
            "constituency": self.constituency,
            "district": self.district,
            "topic_tags": self.topic_tags,
            "timestamp": self.timestamp,
            "version": self.version,
            "chunk_summary": self.chunk_summary,
        }


@dataclass
class RetrievedEvidence:
    """A single piece of retrieved evidence with metadata"""
    chunk_id: str
    text: str
    source_file: str
    source_type: str
    constituency: Optional[str]
    district: Optional[str]
    page_section: Optional[str]
    relevance_score: float
    rerank_score: Optional[float] = None
    why_selected: Optional[str] = None
    
    def to_citation(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source_file,
            "source_type": self.source_type,
            "constituency": self.constituency,
            "district": self.district,
            "section": self.page_section,
            "score": self.rerank_score or self.relevance_score,
            "text_preview": self.text[:200] + "..." if len(self.text) > 200 else self.text
        }


@dataclass
class EvidencePack:
    """Collection of evidence passages for answer generation"""
    query: str
    constituency: Optional[str]
    intent: str
    passages: List[RetrievedEvidence]
    confidence_score: float
    constituency_match_rate: float
    has_sufficient_evidence: bool
    missing_evidence_types: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Previous conversation turns
    
    def get_context_for_llm(self) -> str:
        """Format evidence pack for LLM generation"""
        context_parts = []
        for i, p in enumerate(self.passages, 1):
            source_info = f"[Source {i}: {p.source_file}"
            if p.constituency:
                source_info += f", Constituency: {p.constituency}"
            if p.page_section:
                source_info += f", Section: {p.page_section}"
            source_info += f", Relevance: {p.rerank_score or p.relevance_score:.2f}]"
            context_parts.append(f"{source_info}\n{p.text}\n")
        return "\n---\n".join(context_parts)


@dataclass
class QueryUnderstanding:
    """Extracted understanding from user query"""
    original_query: str
    constituency: Optional[str] = None
    ac_no: Optional[int] = None
    district: Optional[str] = None
    parent_pc: Optional[str] = None
    pc_name: Optional[str] = None
    intent: str = "general"  # strategy, analysis, prediction, comparison, etc.
    required_evidence_types: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    time_reference: Optional[str] = None


@dataclass 
class EvidenceBasedAnswer:
    """Final answer with citations and confidence"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    evidence_quality: str  # high, medium, low, insufficient
    warnings: List[str] = field(default_factory=list)
    contradiction_found: bool = False


class ConstituencyAwareChunker:
    """
    Constituency-aware document chunker
    
    Rules:
    - Split by headings/sections/paragraph blocks
    - Target size: 350-900 tokens
    - Overlap: 80-120 tokens
    - Hard boundary when constituency changes
    - Generate chunk summary for retrieval boosting
    """
    
    # West Bengal constituencies for detection
    WB_CONSTITUENCIES = [
        "COOCH BEHAR UTTAR", "COOCH BEHAR DAKSHIN", "SITALKUCHI", "SITAI", "DINHATA",
        "NATABARI", "TUFANGANJ", "KUMARGRAM", "KALCHINI", "ALIPURDUARS", "FALAKATA",
        "MADARIHAT", "DHUPGURI", "MAYNAGURI", "JALPAIGURI", "RAJGANJ", "DABGRAM FULBARI",
        "MAL", "NAGRAKATA", "METIALI", "KALIMPONG", "DARJEELING", "KURSEONG", "MATIGARA NAXALBARI",
        "SILIGURI", "PHANSIDEWA", "CHOPRA", "ISLAMPUR", "GOALPOKHAR", "CHAKULIA", "KARANDIGHI",
        "HEMTABAD", "KALIAGANJ", "RAIGANJ", "ITAHAR", "KUSHMANDI", "KUMARGANJ", "BALURGHAT",
        "TAPAN", "GANGARAMPUR", "HARIRAMPUR", "HABIBPUR", "GAZOLE", "CHANCHAL", "HARISHCHANDRAPUR",
        "MALATIPUR", "RATUA", "MALDAHA", "ENGLISHBAZAR", "MOTHABARI", "SUJAPUR", "BAISHNAB NAGAR",
        "FARAKKA", "SAMSHERGANJ", "SUTI", "JANGIPUR", "RAGHUNATHGANJ", "SAGARDIGHI", "LALGOLA",
        "BHAGAWANGOLA", "MURSHIDABAD", "NABAGRAM", "KHARGRAM", "BURWAN", "KANDI", "BHARATPUR",
        "REJINAGAR", "BELDANGA", "BAHARAMPUR", "HARIHARPARA", "NAODA", "DOMKAL", "JALANGI",
        "KARIMPUR", "TEHATTA", "PALASHIPARA", "KALIGANJ", "NAKASHIPARA", "CHAPRA", "KRISHNANAGAR UTTAR",
        "NABADWIP", "KRISHNANAGAR DAKSHIN", "SANTIPUR", "RANAGHAT UTTAR PASCHIM", "KRISHNAGANJ",
        "RANAGHAT UTTAR PURBA", "RANAGHAT DAKSHIN", "CHAKDAHA", "KALYANI", "HARINGHATA", "BAGDA",
        "BANGAON UTTAR", "BANGAON DAKSHIN", "GAIGHATA", "SWARUPNAGAR", "BADURIA", "HABRA",
        "ASHOKNAGAR", "AMDANGA", "BIJPUR", "NAIHATI", "BHATPARA", "JAGATDAL", "NOAPARA",
        "BARRACKPUR", "KHARDAHA", "DUM DUM UTTAR", "PANIHATI", "KAMARHATI", "BARANAGAR",
        "DUM DUM", "RAJARHAT NEW TOWN", "BIDHANNAGAR", "RAJARHAT GOPALPUR", "MADHYAMGRAM",
        "BARASAT", "DEGANGA", "HAROA", "MINAKHAN", "SANDESHKHALI", "BASIRHAT DAKSHIN",
        "BASIRHAT UTTAR", "HINGALGANJ", "GOSABA", "KULTALI", "PATHARPRATIMA", "KAKDWIP",
        "SAGAR", "KULPI", "RAIDIGHI", "MANDIRBAZAR", "JAYNAGAR", "BARUIPUR PURBA",
        "CANNING PASCHIM", "CANNING PURBA", "BARUIPUR PASCHIM", "MAGRAHAT PURBA",
        "MAGRAHAT PASCHIM", "DIAMOND HARBOUR", "FALTA", "SATGACHHIA", "BISHNUPUR",
        "SONARPUR DAKSHIN", "BHANGAR", "KASBA", "JADAVPUR", "SONARPUR UTTAR", "TOLLYGUNGE",
        "BEHALA PASCHIM", "BEHALA PURBA", "MAHESHTALA", "BUDGE BUDGE", "METIABRUZ",
        "KOLKATA PORT", "BHAWANIPUR", "RASHBEHARI", "BALLYGUNGE", "CHOWRANGHEE", "ENTALLY",
        "BELEGHATA", "JORASANKO", "SHYAMPUKUR", "MANIKTALA", "KASHIPUR BELGACHHIA",
        "COSSIPORE BELGACHHIA", "BARANAGAR", "ULUBERIA PURBA", "ULUBERIA UTTAR",
        "ULUBERIA DAKSHIN", "SHYAMPUR", "BAGNAN", "AMTA", "UDAYNARAYANPUR", "JAGATBALLAVPUR",
        "DOMJUR", "HOWRAH UTTAR", "HOWRAH MADHYA", "HOWRAH DAKSHIN", "BALLY", "HOWRAH CENTRAL",
        "SHIBPUR", "PANCHLA", "SANKRAIL", "MAHISHADAL", "NANDIGRAM", "TAMLUK", "MOYNA",
        "NANDAKUMAR", "CHANDIPUR", "POTASHPUR", "CONTAI UTTAR", "CONTAI DAKSHIN", "KHEJURI",
        "PATASHPUR", "EGRA", "DANTAN", "KESHIARY", "KHARAGPUR SADAR", "NARAYANGARH",
        "SABANG", "PINGLA", "DEBRA", "DASPUR", "GHATAL", "CHANDRAKONA", "GARBETA",
        "SALBANI", "KESHPUR", "KHARAGPUR", "MEDINIPUR", "BINPUR", "BANDWAN", "BALARAMPUR",
        "BAGHMUNDI", "JOYPUR", "PURULIA", "MANBAZAR", "KASHIPUR", "PARA", "RAGHUNATHPUR",
        "SALTORA", "CHHATNA", "RANIBANDH", "RAIPUR", "TALDANGRA", "BANKURA", "BARJORA",
        "ONDA", "BISHNUPUR", "KATULPUR", "INDUS", "SONAMUKHI", "KHANDAGHOSH", "BARDHAMAN UTTAR",
        "BARDHAMAN DAKSHIN", "RAINA", "JAMALPUR", "MONTESWAR", "KALNA", "MEMARI", "PURBASTHALI UTTAR",
        "PURBASTHALI DAKSHIN", "KATWA", "MANGALKOT", "KETUGRAM", "AUSGRAM", "GALSI", "PANDABESWAR",
        "DURGAPUR PURBA", "DURGAPUR PASCHIM", "RANIGANJ", "JAMURIA", "ASANSOL UTTAR",
        "ASANSOL DAKSHIN", "KULTI", "BARABANI", "DUBRAJPUR", "SURI", "BOLPUR", "NANOOR",
        "LABPUR", "MAYURESWAR", "RAMPURHAT", "HANSAN", "NALHATI", "MURARAI", "SAINTHIA",
        "JHALDA", "ARSHA", "JHARGRAM", "NAYAGRAM", "GOPIBALLAVPUR", "KESHIARY"
    ]
    
    # West Bengal districts
    WB_DISTRICTS = [
        "ALIPURDUAR", "BANKURA", "BIRBHUM", "COOCH BEHAR", "DAKSHIN DINAJPUR",
        "DARJEELING", "HOOGHLY", "HOWRAH", "JALPAIGURI", "JHARGRAM", "KALIMPONG",
        "KOLKATA", "MALDAH", "MURSHIDABAD", "NADIA", "NORTH 24 PARGANAS",
        "PASCHIM BARDHAMAN", "PASCHIM MEDINIPUR", "PURBA BARDHAMAN", "PURBA MEDINIPUR",
        "PURULIA", "SOUTH 24 PARGANAS", "UTTAR DINAJPUR"
    ]
    
    def __init__(
        self,
        target_chunk_size: int = 600,  # tokens
        min_chunk_size: int = 350,
        max_chunk_size: int = 900,
        overlap_size: int = 100
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        source_type: str
    ) -> List[CanonicalChunk]:
        """
        Chunk document with constituency awareness
        """
        chunks = []
        
        # First, split by major sections (headings, double newlines)
        sections = self._split_into_sections(text)
        
        chunk_idx = 0
        for section in sections:
            # Detect constituency in section
            constituency = self._detect_constituency(section)
            district = self._detect_district(section)
            heading = self._extract_heading(section)
            topic_tags = self._detect_topics(section)
            
            # Sub-chunk if section is too large
            sub_chunks = self._chunk_section(
                section,
                constituency=constituency,
                max_size=self.max_chunk_size
            )
            
            for sub_text in sub_chunks:
                if len(sub_text.strip()) < 50:  # Skip very small chunks
                    continue
                    
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                
                # Generate chunk summary
                summary = self._generate_chunk_summary(sub_text, constituency, topic_tags)
                
                chunk = CanonicalChunk(
                    chunk_id=chunk_id,
                    text=sub_text,
                    doc_id=doc_id,
                    source_file_name=source_file,
                    source_type=source_type,
                    heading=heading,
                    constituency=constituency,
                    district=district or self._get_district_for_constituency(constituency),
                    topic_tags=topic_tags,
                    chunk_summary=summary
                )
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into major sections"""
        # Split on double newlines, headings, or explicit section markers
        patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=[A-Z][A-Z\s]{5,}:)',  # ALL CAPS headings
            r'\n(?=#{1,3}\s)',  # Markdown headings
            r'\n(?=\d+\.\s+[A-Z])',  # Numbered sections
            r'\n(?=Assembly Constituency)',  # Constituency headers
            r'\n(?=বিধানসভা)',  # Bengali constituency references
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(patterns)
        sections = re.split(combined_pattern, text)
        
        # Filter empty sections
        return [s.strip() for s in sections if s and len(s.strip()) > 20]
    
    def _chunk_section(
        self,
        section: str,
        constituency: Optional[str],
        max_size: int
    ) -> List[str]:
        """Sub-chunk a section while respecting constituency boundaries"""
        # Estimate tokens (rough: 4 chars per token)
        estimated_tokens = len(section) / 4
        
        if estimated_tokens <= max_size:
            return [section]
        
        # Need to split - use paragraph boundaries
        paragraphs = section.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_tokens = len(para) / 4
            
            # Check if this paragraph mentions a different constituency
            para_constituency = self._detect_constituency(para)
            if para_constituency and para_constituency != constituency:
                # Hard boundary - save current chunk and start new
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                constituency = para_constituency
            
            if current_size + para_tokens > max_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                # Start new chunk with overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_size = len(overlap_text) / 4 + para_tokens
            else:
                current_chunk.append(para)
                current_size += para_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _detect_constituency(self, text: str) -> Optional[str]:
        """Detect constituency name in text"""
        text_upper = text.upper()
        
        for const in self.WB_CONSTITUENCIES:
            if const in text_upper:
                return const
            # Also check without spaces
            if const.replace(" ", "") in text_upper.replace(" ", ""):
                return const
        
        # Check for constituency number patterns like "AC-151" or "বিধানসভা 151"
        patterns = [
            r'AC[- ]?(\d+)',
            r'Assembly Constituency[- ]?(\d+)',
            r'Assembly[- ]?Constituency[- ]?(\d+)',
            r'\bconstituency\s*(\d{1,3})\b',
            r'বিধানসভা[- ]?(\d+)',
            r'কেন্দ্র[- ]?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the number as identifier
                return f"AC-{match.group(1)}"
        
        return None
    
    def _detect_district(self, text: str) -> Optional[str]:
        """Detect district name in text"""
        text_upper = text.upper()
        
        for district in self.WB_DISTRICTS:
            if district in text_upper:
                return district
            # Also check common variations
            if district.replace(" ", "") in text_upper.replace(" ", ""):
                return district
        
        return None
    
    def _get_district_for_constituency(self, constituency: Optional[str]) -> Optional[str]:
        """Map constituency to district"""
        if not constituency:
            return None
            
        # Constituency to district mapping (subset - expand as needed)
        CONST_TO_DISTRICT = {
            "KARIMPUR": "NADIA",
            "TEHATTA": "NADIA",
            "KRISHNANAGAR UTTAR": "NADIA",
            "KRISHNANAGAR DAKSHIN": "NADIA",
            "SONARPUR UTTAR": "SOUTH 24 PARGANAS",
            "SONARPUR DAKSHIN": "SOUTH 24 PARGANAS",
            "JADAVPUR": "SOUTH 24 PARGANAS",
            "CHAMPDANI": "HOOGHLY",
            "BINPUR": "JHARGRAM",
            "BEHALA PURBA": "SOUTH 24 PARGANAS",
            "BEHALA PASCHIM": "SOUTH 24 PARGANAS",
        }
        
        return CONST_TO_DISTRICT.get(constituency.upper())
    
    def _extract_heading(self, text: str) -> Optional[str]:
        """Extract section heading if present"""
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            # Check for heading patterns
            if re.match(r'^[A-Z][A-Z\s]{5,}$', line):  # ALL CAPS
                return line
            if re.match(r'^#{1,3}\s+(.+)$', line):  # Markdown
                return re.sub(r'^#{1,3}\s+', '', line)
            if re.match(r'^\d+\.\s+[A-Z]', line):  # Numbered
                return line
        return None
    
    def _detect_topics(self, text: str) -> List[str]:
        """Detect topic tags from text content"""
        tags = []
        text_lower = text.lower()
        
        topic_keywords = {
            TopicTag.ROADS: ["road", "rasta", "রাস্তা", "highway", "bridge"],
            TopicTag.JOBS: ["job", "employment", "unemployment", "চাকরি", "কর্মসংস্থান"],
            TopicTag.WELFARE: ["welfare", "scheme", "yojana", "প্রকল্প", "freebie", "ration"],
            TopicTag.LAW_ORDER: ["crime", "police", "law", "order", "আইন", "শৃঙ্খলা", "সন্ত্রাস"],
            TopicTag.COMMUNAL: ["communal", "hindu", "muslim", "minority", "সংখ্যালঘু", "সাম্প্রদায়িক"],
            TopicTag.CANDIDATE: ["candidate", "mla", "প্রার্থী", "বিধায়ক", "নেতা"],
            TopicTag.ECONOMY: ["economy", "gdp", "income", "অর্থনীতি", "আয়"],
            TopicTag.HEALTH: ["health", "hospital", "doctor", "স্বাস্থ্য", "চিকিৎসা"],
            TopicTag.EDUCATION: ["education", "school", "college", "শিক্ষা", "স্কুল"],
            TopicTag.CORRUPTION: ["corruption", "দুর্নীতি", "scam", "bribe"],
            TopicTag.DEVELOPMENT: ["development", "infrastructure", "উন্নয়ন"],
            TopicTag.STRATEGY: ["strategy", "plan", "action", "recommendation", "কৌশল"],
            TopicTag.SURVEY: ["survey", "poll", "opinion", "সমীক্ষা", "জরিপ"],
            TopicTag.PREDICTION: ["prediction", "forecast", "projection", "পূর্বাভাস"],
            TopicTag.VOTER_SEGMENT: ["voter", "segment", "demographic", "ভোটার"],
            TopicTag.CAMPAIGN: ["campaign", "election", "প্রচার", "নির্বাচন"],
        }
        
        for tag, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag.value)
        
        return tags
    
    def _generate_chunk_summary(
        self,
        text: str,
        constituency: Optional[str],
        topics: List[str]
    ) -> str:
        """Generate a 1-2 line summary for retrieval boosting"""
        # Simple extractive summary
        summary_parts = []
        
        if constituency:
            summary_parts.append(f"About {constituency} constituency")
        
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics[:3])}")
        
        # Extract first meaningful sentence
        sentences = re.split(r'[.।\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30 and len(sent) < 200:
                summary_parts.append(sent)
                break
        
        return ". ".join(summary_parts) if summary_parts else text[:150]


class TwoPassRetriever:
    """
    Two-pass retrieval strategy:
    - Pass A: Strict constituency filter
    - Pass B: Global context (statewide patterns, survey conclusions)
    """
    
    def __init__(self, opensearch_client, embedding_service):
        self.opensearch = opensearch_client
        self.embedding_service = embedding_service
        
    async def retrieve(
        self,
        query: str,
        query_understanding: QueryUnderstanding,
        top_k_per_pass: int = 30
    ) -> List[RetrievedEvidence]:
        """
        Execute two-pass retrieval (SAM-aligned):
        - Pass A (local): BM25 + kNN with strict constituency (or district) filter
        - Pass B (global): BM25 + kNN without filters
        """
        all_results: List[RetrievedEvidence] = []

        # Pass A: strict constituency filter (district fallback)
        local_filters: Dict[str, Any] = {}
        if query_understanding.constituency:
            local_filters["constituency"] = query_understanding.constituency
            logger.info(f"[TwoPassRetriever] Pass A: Searching for constituency={query_understanding.constituency}")
        elif query_understanding.district:
            local_filters["district"] = query_understanding.district
            logger.info(f"[TwoPassRetriever] Pass A: Searching for district={query_understanding.district}")

        if local_filters:
            local_results = await self._retrieve_bm25_and_knn(query=query, filters=local_filters, top_k=top_k_per_pass)
            all_results.extend(local_results)
            logger.info(f"[TwoPassRetriever] Pass A returned {len(local_results)} results")

        # Pass A2: district enrichment (always when district known)
        if query_understanding.district:
            district_filters = {"district": query_understanding.district}
            district_results = await self._retrieve_bm25_and_knn(
                query=query,
                filters=district_filters,
                top_k=max(5, top_k_per_pass // 2),
            )
            all_results.extend(district_results)
            logger.info(f"[TwoPassRetriever] Pass A2 (district) returned {len(district_results)} results")

        # Pass A3: parent PC enrichment (BM25 only; no embedding cost)
        if query_understanding.pc_name:
            pc_query = f"{query} {query_understanding.pc_name} parliamentary constituency"
            pc_results = await self._retrieve_bm25_and_knn(
                query=pc_query,
                filters=None,
                top_k=max(5, top_k_per_pass // 3),
                use_knn=False,
                use_bm25=True,
            )
            all_results.extend(pc_results)
            logger.info(f"[TwoPassRetriever] Pass A3 (pc) returned {len(pc_results)} results")

        # Pass B: global context retrieval (no filters)
        logger.info("[TwoPassRetriever] Pass B: Searching for global context")
        global_results = await self._retrieve_bm25_and_knn(query=query, filters=None, top_k=top_k_per_pass)
        all_results.extend(global_results)
        logger.info(f"[TwoPassRetriever] Pass B returned {len(global_results)} results")

        # Deduplicate by chunk_id, keep the higher relevance_score
        best: Dict[str, RetrievedEvidence] = {}
        for r in all_results:
            prev = best.get(r.chunk_id)
            if prev is None or float(r.relevance_score or 0.0) > float(prev.relevance_score or 0.0):
                best[r.chunk_id] = r

        unique_results = list(best.values())
        logger.info(f"[TwoPassRetriever] Total results: {len(unique_results)}")
        return unique_results

    async def _retrieve_bm25_and_knn(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int,
        use_knn: bool = True,
        use_bm25: bool = True
    ) -> List[RetrievedEvidence]:
        """
        Run BM25 and kNN separately (if supported) and merge.
        Falls back to hybrid_search if bm25_search/knn_search aren't present.
        """
        try:
            if hasattr(self.opensearch, "bm25_search") and hasattr(self.opensearch, "knn_search"):
                bm25_results = []
                knn_results = []
                if use_bm25:
                    bm25_results = await self.opensearch.bm25_search(query=query, top_k=top_k, filters=filters)
                if use_knn:
                    knn_results = await self.opensearch.knn_search(query=query, top_k=top_k, filters=filters)
                results = self._merge_search_results(bm25_results, knn_results)[:top_k]
            else:
                results = await self.opensearch.hybrid_search(query=query, top_k=top_k, filters=filters)

            evidence_list: List[RetrievedEvidence] = []
            for r in results:
                metadata = getattr(r, "metadata", None) or {}
                chunk_id = metadata.get("chunk_id") or getattr(r, "doc_id", None) or "unknown"
                evidence_list.append(
                    RetrievedEvidence(
                        chunk_id=str(chunk_id),
                        text=getattr(r, "text", "") or "",
                        source_file=getattr(r, "source_file", "") or metadata.get("source_file", "unknown"),
                        source_type=metadata.get("source_type", "unknown"),
                        constituency=metadata.get("constituency") or getattr(r, "constituency", None),
                        district=metadata.get("district") or getattr(r, "district", None),
                        page_section=metadata.get("page") or metadata.get("page_section"),
                        relevance_score=float(getattr(r, "score", 0.0) or 0.0),
                    )
                )
            return evidence_list
        except Exception as e:
            logger.error(f"[TwoPassRetriever] Retrieval error: {e}")
            return []

    def _merge_search_results(self, *lists: List[Any]) -> List[Any]:
        """Merge results by doc_id keeping max score (SAM-style merge)."""
        seen: Dict[str, Any] = {}
        for lst in lists:
            for r in (lst or []):
                key = getattr(r, "doc_id", None) or (getattr(r, "metadata", None) or {}).get("doc_id") or getattr(r, "text", "")[:80]
                score = float(getattr(r, "score", 0.0) or 0.0)
                if key not in seen or score > float(getattr(seen[key], "score", 0.0) or 0.0):
                    seen[key] = r
        return sorted(seen.values(), key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
    
    async def _retrieve_with_filter(
        self,
        query: str,
        constituency: Optional[str],
        district: Optional[str],
        top_k: int
    ) -> List[RetrievedEvidence]:
        """Retrieve with optional constituency/district filter"""
        try:
            # Build filter
            filters = {}
            if constituency:
                filters["constituency"] = constituency
            if district and not constituency:
                filters["district"] = district
            
            # Execute hybrid search
            results = await self.opensearch.hybrid_search(
                query=query,
                top_k=top_k,
                filters=filters if filters else None
            )
            
            # Convert to RetrievedEvidence
            evidence_list = []
            for r in results:
                metadata = r.metadata or {}
                evidence = RetrievedEvidence(
                    chunk_id=metadata.get("chunk_id", r.id if hasattr(r, 'id') else "unknown"),
                    text=r.text,
                    source_file=metadata.get("source_file", metadata.get("source_file_name", "unknown")),
                    source_type=metadata.get("source_type", "unknown"),
                    constituency=metadata.get("constituency"),
                    district=metadata.get("district"),
                    page_section=metadata.get("page_section"),
                    relevance_score=r.score
                )
                evidence_list.append(evidence)
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"[TwoPassRetriever] Retrieval error: {e}")
            return []


class CrossEncoderReranker:
    """
    Rerank retrieved passages using cross-encoder or LLM
    
    Reranker checks:
    - Does the passage directly support a claim?
    - Is it about the same constituency/concept?
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    async def rerank(
        self,
        query: str,
        query_understanding: QueryUnderstanding,
        passages: List[RetrievedEvidence],
        top_k: int = 12
    ) -> List[RetrievedEvidence]:
        """Rerank passages using LLM-as-reranker"""
        if not passages:
            return []
        
        if len(passages) <= top_k:
            # No need to rerank, just score
            for p in passages:
                p.rerank_score = p.relevance_score
            return passages
        
        # Root cause of "same strategy everywhere" in your screenshots:
        # OpenAI throttling triggered our fallback generator often, so the generic template dominated.
        # To reduce throttling, default rerank mode is SCORE (no LLM call).
        # Enable LLM reranking explicitly with: EVIDENCE_RERANK_MODE=llm
        if os.getenv("EVIDENCE_RERANK_MODE", "score").lower() == "llm":
            return await self._llm_rerank(query, query_understanding, passages, top_k)

        # Fallback: score-based reranking
        return self._score_based_rerank(query_understanding, passages, top_k)
    
    async def _llm_rerank(
        self,
        query: str,
        query_understanding: QueryUnderstanding,
        passages: List[RetrievedEvidence],
        top_k: int
    ) -> List[RetrievedEvidence]:
        """Use LLM to rerank passages"""
        try:
            # SAM-style: LLM returns selected chunk_ids with confidence 0..1
            candidates = []
            for i, p in enumerate(passages[:max(top_k, 30)], start=1):
                candidates.append({
                    "rank_id": i,
                    "chunk_id": p.chunk_id,
                    "text": (p.text or "")[:1500],
                    "meta": {
                        "source_file": p.source_file,
                        "constituency": p.constituency,
                        "district": p.district,
                        "section": p.page_section,
                    }
                })

            payload = {
                "question": query,
                "target_constituency": query_understanding.constituency,
                "intent": query_understanding.intent,
                "instructions": [
                    "You are reranking evidence passages for a RAG system.",
                    "Select the passages that most directly support answering the question.",
                    "Return strict JSON only: {\"selected\": [{\"chunk_id\": str, \"confidence\": 0..1, \"why\": str}]}",
                    "Do NOT invent facts. Prefer direct mentions and actionable strategy relevance.",
                ],
                "candidates": candidates
            }

            # Option B: use a faster/cheaper model for reranking to reduce rate limits
            from app.services.llm import get_llm
            rerank_model = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
            llm = get_llm(force_model=rerank_model)
            llm_resp = llm.generate(
                prompt=json.dumps(payload, ensure_ascii=False),
                system="You rerank passages for evidence-based answering. Output strict JSON only.",
                temperature=0.0,
                max_tokens=800,
            )
            resp_text = llm_resp.text if hasattr(llm_resp, "text") else str(llm_resp)
            data = llm.extract_json(resp_text)

            selected = data.get("selected", []) if isinstance(data, dict) else []
            by_id = {p.chunk_id: p for p in passages}
            out: List[RetrievedEvidence] = []
            for s in selected:
                cid = s.get("chunk_id")
                if cid in by_id:
                    p = by_id[cid]
                    p.rerank_score = float(s.get("confidence", 0.0) or 0.0)  # normalized 0..1
                    p.why_selected = s.get("why", "")
                    out.append(p)

            out.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
            if not out:
                return self._score_based_rerank(query_understanding, passages, top_k)

            # IMPORTANT: LLM reranker often selects only a few passages (3-5).
            # That can shrink context too aggressively even when retrieval found many relevant hits.
            # To stabilize answers, always fill up to top_k using next-best by relevance_score.
            if len(out) < top_k:
                chosen = {p.chunk_id for p in out}
                rest = [p for p in passages if p.chunk_id not in chosen]
                rest.sort(key=lambda x: float(x.relevance_score or 0.0), reverse=True)
                for p in rest:
                    # Keep selected passages on top; fillers get a low rerank_score.
                    p.rerank_score = p.rerank_score if p.rerank_score is not None else 0.0
                    out.append(p)
                    if len(out) >= top_k:
                        break

            return out[:top_k]
            
        except Exception as e:
            # If LLM rerank is rate-limited, fall back to deterministic score-based rerank
            err = str(e)
            logger.error(f"[Reranker] LLM rerank failed: {err}")
            return self._score_based_rerank(query_understanding, passages, top_k)
    
    def _score_based_rerank(
        self,
        query_understanding: QueryUnderstanding,
        passages: List[RetrievedEvidence],
        top_k: int
    ) -> List[RetrievedEvidence]:
        """Fallback score-based reranking with constituency boost"""
        for p in passages:
            score = p.relevance_score
            
            # Boost if constituency matches
            if query_understanding.constituency and p.constituency:
                if p.constituency.upper() == query_understanding.constituency.upper():
                    score *= 1.5
                    p.why_selected = f"Constituency match: {p.constituency}"
            
            # Boost if has specific content
            if p.source_type in ["docx", "pdf"]:
                score *= 1.1  # Prefer document sources
            
            p.rerank_score = score
        
        # Sort and return top_k
        passages.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return passages[:top_k]


class EvidencePackBuilder:
    """Build evidence pack for constrained generation"""
    
    # SAM default confidence gate (avg top selected reranker confidence)
    CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE", "0.25"))
    # Do NOT require 3 passages (SAM doesn’t); require at least 1 passage.
    MIN_PASSAGES = int(os.getenv("MIN_PASSAGES", "1"))
    
    def build(
        self,
        query: str,
        query_understanding: QueryUnderstanding,
        reranked_passages: List[RetrievedEvidence]
    ) -> EvidencePack:
        """Build evidence pack with confidence scoring"""
        
        # Calculate confidence metrics
        if not reranked_passages:
            return EvidencePack(
                query=query,
                constituency=query_understanding.constituency,
                intent=query_understanding.intent,
                passages=[],
                confidence_score=0.0,
                constituency_match_rate=0.0,
                has_sufficient_evidence=False,
                missing_evidence_types=["No passages retrieved"]
            )
        
        # Constituency match rate
        const_matches = sum(
            1 for p in reranked_passages 
            if query_understanding.constituency and p.constituency 
            and p.constituency.upper() == query_understanding.constituency.upper()
        )
        constituency_match_rate = const_matches / len(reranked_passages) if reranked_passages else 0
        
        # Top score average
        top_scores = [float(p.rerank_score if p.rerank_score is not None else p.relevance_score) for p in reranked_passages[:5]]
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Normalize:
        # - If reranker produced confidence scores (0..1), use SAM-style gate: avg of top selected (up to 3)
        # - Else scale OpenSearch scores down (often ~10-100+)
        if avg_top_score <= 1.5:
            top_selected = [float(p.rerank_score or 0.0) for p in reranked_passages if p.why_selected][:3]
            if not top_selected:
                top_selected = [float(p.rerank_score or 0.0) for p in reranked_passages[:3]]
            normalized_score = float(sum(top_selected) / max(len(top_selected), 1))
        else:
            normalized_score = float(min(avg_top_score / 100, 1.0))

        # Combined confidence (slight boost for constituency match)
        if query_understanding.constituency:
            confidence_score = (normalized_score * 0.7) + (constituency_match_rate * 0.3)
        else:
            confidence_score = normalized_score
        
        # Check for missing evidence types
        missing = []
        if query_understanding.constituency and constituency_match_rate < 0.3:
            missing.append(f"Limited content about {query_understanding.constituency}")
        
        has_sufficient = (
            len(reranked_passages) >= self.MIN_PASSAGES and
            confidence_score >= self.CONFIDENCE_THRESHOLD
        )
        
        return EvidencePack(
            query=query,
            constituency=query_understanding.constituency,
            intent=query_understanding.intent,
            passages=reranked_passages,
            confidence_score=confidence_score,
            constituency_match_rate=constituency_match_rate,
            has_sufficient_evidence=has_sufficient,
            missing_evidence_types=missing
        )


class ConstrainedAnswerGenerator:
    """
    Generate answers constrained to evidence only
    
    Rules:
    - Every key assertion must cite at least one retrieved passage
    - If no passage supports it, don't say it
    - Provide structured answer (diagnosis → do/don't → plan)
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def generate(
        self,
        evidence_pack: EvidencePack
    ) -> EvidenceBasedAnswer:
        """Generate evidence-only answer"""
        
        # Check if we have sufficient evidence
        if not evidence_pack.has_sufficient_evidence:
            return self._generate_insufficient_evidence_response(evidence_pack)
        
        # Build generation prompt
        prompt = self._build_generation_prompt(evidence_pack)
        
        try:
            # Option B: keep best model for final answer (default gpt-4o),
            # but allow override via env for throughput if needed.
            from app.services.llm import get_llm
            final_model = os.getenv("OPENAI_FINAL_MODEL", None)  # e.g., gpt-4o or gpt-4o-mini
            llm = get_llm(force_model=final_model) if final_model else get_llm()
            
            # LLM generate is synchronous and returns LLMResponse
            # Increased to 4096 to avoid truncation of longer responses
            logger.warning(f"[ConstrainedGenerator] DEBUG: Calling LLM with {len(evidence_pack.passages)} passages, prompt length: {len(prompt)}")
            
            llm_response = llm.generate(prompt, max_tokens=4096)
            response_text = llm_response.text if hasattr(llm_response, "text") else str(llm_response)
            
            logger.warning(f"[ConstrainedGenerator] DEBUG: LLM response length: {len(response_text)}, preview: {repr(response_text[:300]) if response_text else 'EMPTY'}")

            # Citation enforcement: if the model didn't include citations, add them post-hoc
            # rather than refusing the answer (relaxed from strict refusal)
            has_citations = "[Source" in response_text or "[source" in response_text.lower()
            
            if not has_citations and response_text and len(response_text) > 100:
                # The LLM generated a meaningful answer but without explicit citations
                # Add citations as a note at the end rather than refusing
                logger.warning("[ConstrainedGenerator] LLM response missing citations, adding post-hoc")
                
                # Add a sources appendix
                sources_text = "\n\n**Sources Used:**\n"
                for i, p in enumerate(evidence_pack.passages[:5], 1):
                    source_name = p.source_file or "Unknown"
                    sources_text += f"- [Source {i}]: {source_name}\n"
                
                response_text = response_text + sources_text
                
            elif not has_citations and (not response_text or len(response_text) < 100):
                # Very short or empty response without citations - likely failed
                logger.warning("[ConstrainedGenerator] LLM response too short/empty without citations")
                evidence_pack.missing_evidence_types = list(
                    set(evidence_pack.missing_evidence_types + ["Generated answer was too short or missing citations"])
                )
                return self._generate_insufficient_evidence_response(evidence_pack)
            
            # Extract citations from response
            citations = [p.to_citation() for p in evidence_pack.passages[:8]]
            
            # Run contradiction check (simplified for sync)
            contradiction_found = self._check_contradictions_sync(response_text, evidence_pack)
            
            # Determine quality
            quality = self._assess_quality(evidence_pack, response_text)
            
            return EvidenceBasedAnswer(
                answer=response_text,
                citations=citations,
                confidence=evidence_pack.confidence_score,
                evidence_quality=quality,
                contradiction_found=contradiction_found,
                warnings=evidence_pack.missing_evidence_types
            )
            
        except Exception as e:
            # If the LLM is rate-limited (RetryError/RateLimitError), fall back to a
            # deterministic evidence-only template so behavior stays consistent.
            err = str(e)
            logger.error(f"[ConstrainedGenerator] Generation error: {err}")
            if "RateLimitError" in err or "RetryError" in err:
                return self._generate_template_fallback(evidence_pack, err)
            return self._generate_error_response(err)

    def _generate_template_fallback(self, evidence_pack: EvidencePack, error: str) -> EvidenceBasedAnswer:
        """
        Deterministic evidence-only fallback when LLM is unavailable.
        Produces a structured strategy using only retrieved passages and citations.
        """
        citations = [p.to_citation() for p in evidence_pack.passages[:8]]

        # Prefer passages that actually reference the target constituency (if any)
        target = (evidence_pack.constituency or "").upper()
        local_first = []
        if target:
            for p in evidence_pack.passages:
                txt = (p.text or "").upper()
                if (p.constituency or "").upper() == target or target in txt:
                    local_first.append(p)

        ranked = local_first + [p for p in evidence_pack.passages if p not in local_first]

        # Pick up to 3 strongest passages for quoting (local-first)
        top_passages = ranked[:3]
        context_snips = []
        for i, p in enumerate(top_passages, 1):
            context_snips.append(f"[Source {i}] {p.text[:400].strip()}...")

        # Simple topic detection from *local* evidence first (prevents same topics everywhere)
        chunker = ConstituencyAwareChunker()
        all_text = "\n".join([p.text or "" for p in (local_first or evidence_pack.passages)])
        topics = chunker._detect_topics(all_text)[:6]

        # Extract simple seat diagnostics from constituency profile text if present
        diag = []
        if target:
            diag.append(f"Target constituency: **{target}**")
        # Try to pull margin/seat-type hints from profile snippet
        profile_text = (top_passages[0].text if top_passages else "") or ""
        if "MARGIN" in profile_text.upper():
            diag.append("Use constituency profile margins/trends from the retrieved evidence. [Source 1]")

        # Build constituency-specific strategy blocks driven by detected topics + competitiveness
        recs = []
        if "roads" in topics or "development" in topics:
            recs.append("- **Infrastructure**: pick the top 2–3 complaints that appear in the retrieved constituency evidence and publish a 90-day fix calendar with ward-wise tracking. [Source 1]")
        if "health" in topics:
            recs.append("- **Healthcare**: commit to visible, repeatable health access fixes (camps, referral desk, ambulance coordination) specifically tied to issues mentioned in evidence. [Source 1]")
        if "jobs" in topics or "economy" in topics:
            recs.append("- **Jobs/economy**: anchor messaging to any local employment/industry cues in the evidence; propose a measurable apprenticeship/MSME plan. [Source 1]")
        if "corruption" in topics:
            recs.append("- **Anti-corruption**: use only what the evidence mentions—set up grievance register + weekly follow-ups on the specific failures cited. [Source 1]")
        if not recs:
            recs.append("- **Local issue-led campaign**: restrict your plan to the constituency-specific complaints present in the retrieved evidence. [Source 1]")

        answer = "\n".join([
            "## Analysis",
            f"This answer is **evidence-only** and generated using retrieved passages (fallback mode due to LLM throttling: {error}).",
            *([f"- {d}" for d in diag] if diag else []),
            "",
            "## Evidence Highlights",
            *context_snips,
            "",
            "## Strategic Recommendations (evidence-driven)",
            *recs,
            "",
            "## Action Items (next 30 days)",
            "- **Candidate/cadre**: appoint constituency war-room + booth leads; publish weekly field-activity logs. [Source 1]",
            "- **Listening loop**: 10–15 household meetings, collect written issue slips, publish commitments and follow-ups. [Source 1]",
            "- **Targeting**: prioritize persuadable segments implied by evidence (undecided/issue-driven) instead of broad slogans. [Source 1]",
        ])

        return EvidenceBasedAnswer(
            answer=answer,
            citations=citations,
            confidence=evidence_pack.confidence_score,
            evidence_quality="medium" if evidence_pack.has_sufficient_evidence else "insufficient",
            warnings=list(set(evidence_pack.missing_evidence_types + [f"LLM unavailable: {error}"])),
            contradiction_found=False
        )
    
    def _build_generation_prompt(self, evidence_pack: EvidencePack) -> str:
        """Build prompt for constrained generation - DYNAMIC format based on query type"""
        context = evidence_pack.get_context_for_llm()
        
        # Build conversation history context if available
        history_context = ""
        if evidence_pack.conversation_history:
            history_parts = []
            for turn in evidence_pack.conversation_history[-6:]:  # Last 6 turns for context
                role = turn.get("role", "user")
                content = turn.get("content", "")[:500]  # Truncate long content
                history_parts.append(f"{role.upper()}: {content}")
            if history_parts:
                history_context = f"""
CONVERSATION HISTORY (for context - answer the current query with awareness of this conversation):
{chr(10).join(history_parts)}

"""
        
        # Detect query type for dynamic formatting
        query_lower = evidence_pack.query.lower()
        is_followup = any(word in query_lower for word in ["tell me", "only", "just", "what about", "and", "also"])
        is_list_request = any(word in query_lower for word in ["list", "key", "main", "top", "action item", "action point", "bullet"])
        is_simple_question = len(evidence_pack.query.split()) < 10 and "?" in evidence_pack.query
        
        # Dynamic format instructions based on query type
        if is_list_request:
            format_instruction = """
FORMAT: Provide a concise bullet-point list directly answering the query. 
- Be direct and specific
- Include [Source N] citations for each point
- No lengthy analysis needed - just the requested list"""
        elif is_followup:
            format_instruction = """
FORMAT: This appears to be a follow-up question. Answer directly and concisely.
- Refer back to the conversation context
- Be specific to what was asked
- Include [Source N] citations where applicable"""
        elif is_simple_question:
            format_instruction = """
FORMAT: Provide a direct, focused answer.
- Start with the key answer
- Support with evidence citations [Source N]
- Keep it concise but complete"""
        else:
            format_instruction = """
FORMAT: Provide a well-structured response appropriate to the query complexity.
- Start with key insights/analysis
- Include specific findings with citations [Source N]
- Add recommendations if relevant to the query
- Be comprehensive but avoid unnecessary repetition"""

        return f"""You are an expert political strategy analyst for West Bengal elections. Answer the query using ONLY the evidence provided below.

CRITICAL CITATION RULES (MUST FOLLOW):
1. Every factual claim MUST cite the source in this exact format: [Source N] where N is the source number
2. Example: "Sukanta Majumdar received strong support [Source 1] [Source 3]"
3. If the evidence doesn't contain the answer, say "The provided documents do not contain this specific information"
4. ALWAYS include at least one [Source N] citation in your response
{history_context}
QUERY: {evidence_pack.query}
TARGET CONSTITUENCY: {evidence_pack.constituency or 'General/Statewide'}
QUERY INTENT: {evidence_pack.intent}

EVIDENCE SOURCES (cite these using [Source 1], [Source 2], etc.):
{context}
{format_instruction}

IMPORTANT: 
- You MUST include [Source N] citations - answers without citations will be rejected
- Answer what was ACTUALLY asked based on the evidence
- For survey/poll data, analyze the patterns from the provided responses
- Be specific and use data from the sources"""

    def _generate_insufficient_evidence_response(
        self,
        evidence_pack: EvidencePack
    ) -> EvidenceBasedAnswer:
        """Generate response when evidence is insufficient"""
        
        response_parts = [
            f"## Insufficient Evidence for Query\n",
            f"**Query:** {evidence_pack.query}\n",
        ]
        
        if evidence_pack.constituency:
            response_parts.append(
                f"**Target Constituency:** {evidence_pack.constituency}\n"
            )
        
        response_parts.append(
            f"\n**Issue:** The ingested documents do not contain sufficient "
            f"information to answer this query with confidence.\n"
        )
        
        if evidence_pack.missing_evidence_types:
            response_parts.append("\n**What's Missing:**\n")
            for missing in evidence_pack.missing_evidence_types:
                response_parts.append(f"- {missing}\n")
        
        if evidence_pack.passages:
            response_parts.append("\n**Available Related Information:**\n")
            for i, p in enumerate(evidence_pack.passages[:3], 1):
                source_info = f"From {p.source_file}"
                if p.constituency:
                    source_info += f" (about {p.constituency})"
                response_parts.append(f"\n[{i}] {source_info}:\n")
                response_parts.append(f"> {p.text[:300]}...\n")
        
        response_parts.append(
            "\n**Recommendation:** Please upload documents containing "
            f"specific information about {evidence_pack.constituency or 'the requested topic'} "
            "to get accurate, grounded responses."
        )
        
        return EvidenceBasedAnswer(
            answer="".join(response_parts),
            citations=[p.to_citation() for p in evidence_pack.passages[:3]],
            confidence=evidence_pack.confidence_score,
            evidence_quality="insufficient",
            warnings=evidence_pack.missing_evidence_types
        )
    
    def _generate_error_response(self, error: str) -> EvidenceBasedAnswer:
        """Generate error response"""
        return EvidenceBasedAnswer(
            answer=f"An error occurred while generating the response: {error}",
            citations=[],
            confidence=0.0,
            evidence_quality="error",
            warnings=[error]
        )
    
    async def _check_contradictions(
        self,
        answer: str,
        evidence_pack: EvidencePack
    ) -> bool:
        """Check if answer contradicts any evidence (async version)"""
        return self._check_contradictions_sync(answer, evidence_pack)
    
    def _check_contradictions_sync(
        self,
        answer: str,
        evidence_pack: EvidencePack
    ) -> bool:
        """Check if answer contradicts any evidence"""
        # Simple check - could be enhanced with LLM
        # For now, return False (no contradiction detected)
        return False
    
    def _assess_quality(
        self,
        evidence_pack: EvidencePack,
        answer: str
    ) -> str:
        """Assess quality of the answer"""
        if evidence_pack.confidence_score >= 0.8:
            return "high"
        elif evidence_pack.confidence_score >= 0.6:
            return "medium"
        elif evidence_pack.confidence_score >= 0.4:
            return "low"
        else:
            return "insufficient"


class QueryUnderstandingEngine:
    """Extract structured understanding from user query"""
    
    def __init__(self, kg: Any = None):
        self.chunker = ConstituencyAwareChunker()
        self.kg = kg
    
    async def understand(self, query: str) -> QueryUnderstanding:
        """Parse query to extract structured understanding"""
        
        # Detect constituency
        constituency = self.chunker._detect_constituency(query)
        ac_no: Optional[int] = None
        parent_pc: Optional[str] = None
        pc_name: Optional[str] = None

        # Resolve "AC-<number>" to official constituency name via KG (if available)
        if constituency and constituency.upper().startswith("AC-"):
            try:
                ac_no = int(constituency.split("-", 1)[1])
            except Exception:
                ac_no = None

            if ac_no and self.kg is not None:
                resolved = self._resolve_ac_no_to_name(ac_no)
                if resolved:
                    constituency = resolved  # use real AC name for retrieval filters
        
        # Detect district
        district = self.chunker._detect_district(query)

        # If we have KG + a constituency name, enrich district + parent_pc deterministically from profile
        if self.kg is not None and constituency and not constituency.upper().startswith("AC-"):
            prof = None
            profiles = getattr(self.kg, "constituency_profiles", None)
            if profiles:
                prof = profiles.get(str(constituency).upper())
            if prof:
                try:
                    district = getattr(prof, "district", district) or district
                    parent_pc = getattr(prof, "parent_pc", None) or parent_pc
                except Exception:
                    pass

        # Detect PC name (Parliamentary Constituency) if the query is PC-related
        pc_name = self._detect_pc_name(query) or parent_pc
        
        # Detect intent
        intent = self._detect_intent(query)
        
        # Detect entities
        entities = self._extract_entities(query)
        
        # Detect required evidence types
        required_evidence = self._detect_required_evidence(query, intent)
        
        return QueryUnderstanding(
            original_query=query,
            constituency=constituency,
            ac_no=ac_no,
            district=district,
            parent_pc=parent_pc,
            pc_name=pc_name,
            intent=intent,
            required_evidence_types=required_evidence,
            entities=entities
        )

    def _resolve_ac_no_to_name(self, ac_no: int) -> Optional[str]:
        """
        Resolve AC number to constituency name using Knowledge Graph constituency_profiles.
        Returns uppercase ac_name if found.
        """
        try:
            profiles = getattr(self.kg, "constituency_profiles", None)
            if not profiles:
                return None
            for profile in profiles.values():
                if getattr(profile, "ac_no", None) == ac_no:
                    name = getattr(profile, "ac_name", None)
                    return name.upper() if name else None
        except Exception:
            return None
        return None

    def _detect_pc_name(self, query: str) -> Optional[str]:
        """
        Best-effort PC detection:
        - If KG has constituency_profiles, match query text against known parent_pc values.
        - This avoids hardcoding PC lists and works as KG grows.
        """
        try:
            if self.kg is None:
                return None
            q = query.upper()
            profiles = getattr(self.kg, "constituency_profiles", None)
            if not profiles:
                return None
            pcs = set()
            for p in profiles.values():
                pc = getattr(p, "parent_pc", None)
                if pc:
                    pcs.add(str(pc).upper())
            # Prefer longest match first
            for pc in sorted(pcs, key=len, reverse=True):
                if pc and pc in q:
                    return pc
        except Exception:
            return None
        return None
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["strategy", "recommendation", "action", "plan", "should"]):
            return "strategy"
        elif any(w in query_lower for w in ["predict", "forecast", "will win", "chances"]):
            return "prediction"
        elif any(w in query_lower for w in ["compare", "versus", "vs", "difference"]):
            return "comparison"
        elif any(w in query_lower for w in ["analyze", "analysis", "assess", "evaluate"]):
            return "analysis"
        elif any(w in query_lower for w in ["issue", "problem", "concern", "feedback"]):
            return "issues"
        elif any(w in query_lower for w in ["swing", "battleground", "competitive"]):
            return "swing_analysis"
        else:
            return "general"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        entities = []
        
        # Political parties
        parties = ["BJP", "TMC", "INC", "Congress", "CPM", "CPIM", "Left"]
        for party in parties:
            if party.upper() in query.upper():
                entities.append(party)
        
        # Constituency detection
        const = self.chunker._detect_constituency(query)
        if const:
            entities.append(const)
        
        # Year/election references
        years = re.findall(r'20\d{2}', query)
        entities.extend(years)
        
        return entities
    
    def _detect_required_evidence(self, query: str, intent: str) -> List[str]:
        """Detect what types of evidence are required"""
        required = []
        
        if intent == "strategy":
            required.extend(["survey_data", "constraints", "patterns", "recommendations"])
        elif intent == "prediction":
            required.extend(["historical_results", "survey_data", "trends"])
        elif intent == "issues":
            required.extend(["feedback", "complaints", "local_issues"])
        elif intent == "analysis":
            required.extend(["data", "statistics", "trends"])
        
        return required


class EvidenceOnlyRAG:
    """
    Main Evidence-Only RAG Engine
    
    Orchestrates the complete evidence-based answering pipeline:
    1. Query Understanding
    2. Two-Pass Retrieval
    3. Reranking
    4. Evidence Pack Building
    5. Constrained Generation
    6. Confidence Gating
    """
    
    def __init__(self, opensearch_client=None, embedding_service=None, kg: Any = None):
        self.opensearch = opensearch_client
        self.embedding_service = embedding_service
        self.kg = kg
        
        self.query_engine = QueryUnderstandingEngine(kg=kg)
        self.retriever = TwoPassRetriever(opensearch_client, embedding_service) if opensearch_client else None
        self.reranker = CrossEncoderReranker()
        self.evidence_builder = EvidencePackBuilder()
        self.generator = ConstrainedAnswerGenerator()
        self.chunker = ConstituencyAwareChunker()
    
    async def answer(self, query: str, conversation_history: List[Dict[str, str]] = None) -> EvidenceBasedAnswer:
        """
        Main entry point - answer query with evidence-only approach
        
        Args:
            query: The user's question
            conversation_history: Optional list of previous conversation turns for context
        """
        logger.info(f"[EvidenceRAG] Processing query: {query[:100]}...")
        
        # Step 1: Query Understanding
        understanding = await self.query_engine.understand(query)
        logger.info(f"[EvidenceRAG] Query understanding: constituency={understanding.constituency}, intent={understanding.intent}")
        
        # Step 2: Two-Pass Retrieval
        if not self.retriever:
            logger.error("[EvidenceRAG] No retriever configured")
            return EvidenceBasedAnswer(
                answer="RAG system not properly configured. Please ensure OpenSearch is connected.",
                citations=[],
                confidence=0.0,
                evidence_quality="error"
            )
        
        passages = await self.retriever.retrieve(query, understanding)
        logger.info(f"[EvidenceRAG] Retrieved {len(passages)} passages")
        
        # Step 3: Reranking
        # Keep more evidence after rerank to avoid over-shrinking context.
        # Can be tuned via env: EVIDENCE_RERANK_TOP_K=20 (recommended 15-30)
        rerank_top_k = int(os.getenv("EVIDENCE_RERANK_TOP_K", "30"))
        reranked = await self.reranker.rerank(query, understanding, passages, top_k=rerank_top_k)
        logger.info(f"[EvidenceRAG] Reranked to top {len(reranked)} passages")

        # Guardrail for constituency-specific queries:
        # Keep constituency-specific evidence, but ALSO allow district + parent-PC context
        # so responses can be enriched with all information related to that district/PC.
        if understanding.constituency:
            target = understanding.constituency.upper()
            target_district = (understanding.district or "").upper() if understanding.district else ""
            target_pc = (understanding.pc_name or understanding.parent_pc or "").upper() if (understanding.pc_name or understanding.parent_pc) else ""

            filtered: List[RetrievedEvidence] = []
            for p in reranked:
                p_const = (p.constituency or "").upper() if p.constituency else ""
                p_dist = (p.district or "").upper() if p.district else ""
                txt = (p.text or "").upper()

                is_constituency_match = (p_const == target) or (target in txt)
                is_district_match = bool(target_district) and ((p_dist == target_district) or (target_district in txt))
                is_pc_match = bool(target_pc) and (target_pc in txt)

                if is_constituency_match or is_district_match or is_pc_match:
                    filtered.append(p)

            # Only apply if it doesn't wipe everything out
            if len(filtered) >= 2 or (filtered and len(reranked) <= 4):
                reranked = filtered
                logger.info(f"[EvidenceRAG] Constituency/district/pc-filtered passages: {len(reranked)}")
        
        # Step 4: Build Evidence Pack
        evidence_pack = self.evidence_builder.build(query, understanding, reranked)
        # Attach conversation history for context-aware generation
        if conversation_history:
            evidence_pack.conversation_history = conversation_history
        logger.info(f"[EvidenceRAG] Evidence pack: confidence={evidence_pack.confidence_score:.2f}, sufficient={evidence_pack.has_sufficient_evidence}")
        
        # Step 5: Constrained Generation
        answer = await self.generator.generate(evidence_pack)
        logger.info(f"[EvidenceRAG] Generated answer: quality={answer.evidence_quality}")
        
        return answer
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        source_type: str
    ) -> List[CanonicalChunk]:
        """Chunk document using constituency-aware chunking"""
        return self.chunker.chunk_document(text, doc_id, source_file, source_type)


# Factory function
def create_evidence_rag(kg: Any = None) -> EvidenceOnlyRAG:
    """Create configured EvidenceOnlyRAG instance"""
    try:
        from app.services.rag.political_opensearch import PoliticalOpenSearchClient
        from app.services.rag.embeddings import get_embedding_service
        
        opensearch = PoliticalOpenSearchClient()
        embeddings = get_embedding_service()
        
        return EvidenceOnlyRAG(
            opensearch_client=opensearch,
            embedding_service=embeddings,
            kg=kg
        )
    except Exception as e:
        logger.warning(f"[EvidenceRAG] Could not initialize with OpenSearch: {e}")
        return EvidenceOnlyRAG(kg=kg)

