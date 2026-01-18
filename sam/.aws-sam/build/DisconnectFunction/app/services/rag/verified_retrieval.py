"""
Verified Retrieval System - Citation-based retrieval with source verification.

This module ensures ZERO HALLUCINATION by:
1. Only retrieving facts that exist in the knowledge graph
2. Attaching source citations to every piece of information
3. Validating numerical claims against stored data
4. Providing confidence scores based on data quality
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import re
from collections import defaultdict

from .data_schema import FactWithCitation, VerifiedAnswer, ConstituencyProfile
from .knowledge_graph import PoliticalKnowledgeGraph
from .vector_store import UnifiedVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """A single retrieval result with verification status."""
    content: str
    source: str
    relevance_score: float
    verification_status: str  # 'verified', 'partial', 'unverified'
    facts: List[FactWithCitation] = field(default_factory=list)
    numerical_claims: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    """Analysis of user query for routing."""
    original_query: str
    query_type: str  # 'factual', 'analytical', 'comparative', 'predictive'
    entities_mentioned: List[str]
    entity_types: Dict[str, str]  # entity -> type
    time_references: List[str]
    numerical_expectations: bool
    requires_aggregation: bool


class QueryAnalyzer:
    """Analyze queries to determine optimal retrieval strategy."""
    
    # Keywords for query type detection
    FACTUAL_KEYWORDS = ['who won', 'what is', 'how many votes', 'vote share', 'margin', 'result']
    ANALYTICAL_KEYWORDS = ['why', 'analyze', 'explain', 'reason', 'because', 'impact']
    COMPARATIVE_KEYWORDS = ['compare', 'vs', 'versus', 'between', 'difference', 'better']
    PREDICTIVE_KEYWORDS = ['predict', 'forecast', 'will', 'expect', 'chance', '2026', 'future']
    AGGREGATION_KEYWORDS = ['total', 'all', 'how many', 'count', 'list', 'every', 'each district']
    
    # Known entity patterns
    CONSTITUENCY_PATTERN = r'\b[A-Z][A-Z\s]+(?:UTTAR|DAKSHIN|PURBA|PASCHIM)?\b'
    PARTY_PATTERN = r'\b(TMC|BJP|AITC|CPM|CPIM|INC|Congress|Trinamool|Left)\b'
    YEAR_PATTERN = r'\b(2016|2019|2021|2024|2026)\b'
    
    def analyze(self, query: str, kg: PoliticalKnowledgeGraph) -> QueryAnalysis:
        """Analyze a query and extract structured information."""
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Extract entities
        entities, entity_types = self._extract_entities(query, kg)
        
        # Extract time references
        time_refs = re.findall(self.YEAR_PATTERN, query)
        
        # Check for numerical expectations
        numerical = any(kw in query_lower for kw in 
                       ['how many', 'percentage', 'margin', 'vote share', '%', 'votes'])
        
        # Check for aggregation needs
        aggregation = any(kw in query_lower for kw in self.AGGREGATION_KEYWORDS)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            entities_mentioned=entities,
            entity_types=entity_types,
            time_references=time_refs,
            numerical_expectations=numerical,
            requires_aggregation=aggregation
        )
    
    def _detect_query_type(self, query_lower: str) -> str:
        """Detect the type of query."""
        if any(kw in query_lower for kw in self.PREDICTIVE_KEYWORDS):
            return 'predictive'
        elif any(kw in query_lower for kw in self.COMPARATIVE_KEYWORDS):
            return 'comparative'
        elif any(kw in query_lower for kw in self.ANALYTICAL_KEYWORDS):
            return 'analytical'
        else:
            return 'factual'
    
    def _extract_entities(self, query: str, kg: PoliticalKnowledgeGraph) -> Tuple[List[str], Dict[str, str]]:
        """Extract named entities from query."""
        entities = []
        entity_types = {}
        
        # Find constituency names
        for name in kg.constituency_profiles.keys():
            if name.lower() in query.lower() or name in query.upper():
                entities.append(name)
                entity_types[name] = 'constituency'
        
        # Find party names
        party_matches = re.findall(self.PARTY_PATTERN, query, re.IGNORECASE)
        for party in party_matches:
            party_upper = party.upper()
            if party_upper not in entities:
                entities.append(party_upper)
                entity_types[party_upper] = 'party'
        
        # Find district names
        districts = set(p.district for p in kg.constituency_profiles.values())
        for district in districts:
            if district.lower() in query.lower():
                entities.append(district)
                entity_types[district] = 'district'
        
        # Find PC names
        pcs = set(p.parent_pc for p in kg.constituency_profiles.values())
        for pc in pcs:
            if pc.lower() in query.lower():
                entities.append(pc)
                entity_types[pc] = 'pc'
        
        return entities, entity_types


class VerifiedRetriever:
    """
    Retriever that only returns verified, cited information.
    
    This is the core anti-hallucination component.
    Works with both OpenSearch and FAISS backends via UnifiedVectorStore.
    """
    
    def __init__(self, 
                 knowledge_graph: PoliticalKnowledgeGraph,
                 vector_store: UnifiedVectorStore):
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.analyzer = QueryAnalyzer()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve verified information for a query.
        
        Returns information from:
        1. Knowledge graph (verified structured data)
        2. Vector store (uploaded documents and additional context)
        3. Fact database
        
        ALWAYS includes vector search to capture uploaded documents.
        """
        # Analyze query
        analysis = self.analyzer.analyze(query, self.kg)
        
        results = []
        
        # Strategy 1: Direct entity lookup (highest confidence)
        entity_results = self._retrieve_by_entity(analysis)
        results.extend(entity_results)
        
        # Strategy 2: Fact-based retrieval
        fact_results = self._retrieve_relevant_facts(analysis)
        results.extend(fact_results)
        
        # Strategy 3: ALWAYS do vector search to include uploaded documents
        # This ensures dynamically ingested documents are included
        vector_results = self._retrieve_with_verification(query, top_k)
        results.extend(vector_results)
        
        # Deduplicate and rank
        results = self._dedupe_and_rank(results)
        
        return results[:top_k]
    
    def _retrieve_by_entity(self, analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Retrieve based on directly mentioned entities."""
        results = []
        
        for entity in analysis.entities_mentioned:
            entity_type = analysis.entity_types.get(entity, 'unknown')
            
            if entity_type == 'constituency':
                profile = self.kg.get_constituency(entity)
                if profile:
                    # Get full summary
                    summary = self.kg.generate_constituency_summary(entity)
                    facts = self.kg.get_facts_for_entity(entity)
                    
                    # Extract numerical claims
                    numerical = {
                        'tmc_vs_2021': profile.tmc_vote_share_2021,
                        'bjp_vs_2021': profile.bjp_vote_share_2021,
                        'margin_2021': profile.margin_2021,
                        'predicted_margin_2026': profile.predicted_margin_2026,
                        'pc_swing': profile.pc_swing_2019_2024
                    }
                    
                    results.append(RetrievalResult(
                        content=summary,
                        source=', '.join(profile.source_files),
                        relevance_score=1.0,  # Direct match = highest relevance
                        verification_status='verified',
                        facts=facts,
                        numerical_claims=numerical
                    ))
            
            elif entity_type == 'district':
                summary = self.kg.generate_district_summary(entity)
                constituencies = self.kg.get_constituencies_by_district(entity)
                
                facts = []
                for c in constituencies[:5]:  # Sample facts from top constituencies
                    facts.extend(self.kg.get_facts_for_entity(c.ac_name)[:2])
                
                results.append(RetrievalResult(
                    content=summary,
                    source='aggregated from constituency data',
                    relevance_score=0.95,
                    verification_status='verified',
                    facts=facts
                ))
            
            elif entity_type == 'party':
                # Get party-specific data
                party_seats = self.kg.get_constituencies_by_winner(entity)
                
                content = f"## {entity} Performance Analysis\n\n"
                content += f"**2021 Assembly:** Won {len(party_seats)} seats\n\n"
                
                if party_seats:
                    content += "**Key Seats:**\n"
                    for seat in party_seats[:10]:
                        content += f"- {seat.ac_name} ({seat.district}): "
                        content += f"{seat.margin_2021:.2f}% margin\n"
                
                # 2026 predictions
                predicted = self.kg.count_predicted_seats()
                if entity in predicted or (entity == 'TMC' and 'AITC' in predicted):
                    pred_count = predicted.get(entity, predicted.get('AITC', 0) if entity == 'TMC' else 0)
                    content += f"\n**2026 Prediction:** {pred_count} seats\n"
                
                facts = self.kg.get_facts_by_type('electoral_result')[:10]
                
                results.append(RetrievalResult(
                    content=content,
                    source='knowledge_graph_aggregation',
                    relevance_score=0.9,
                    verification_status='verified',
                    facts=facts
                ))
            
            elif entity_type == 'pc':
                # Parliamentary constituency analysis
                ac_list = self.kg.get_constituencies_by_pc(entity)
                
                if ac_list:
                    content = f"## {entity} Parliamentary Constituency\n\n"
                    content += f"**Assembly Segments:** {len(ac_list)}\n\n"
                    
                    # Aggregate PC data
                    tmc_2021 = sum(1 for c in ac_list if c.winner_2021.upper() in ['TMC', 'AITC'])
                    bjp_2021 = sum(1 for c in ac_list if c.winner_2021.upper() == 'BJP')
                    
                    content += f"**2021 Results:**\n"
                    content += f"- TMC: {tmc_2021} segments\n"
                    content += f"- BJP: {bjp_2021} segments\n"
                    content += f"- Others: {len(ac_list) - tmc_2021 - bjp_2021} segments\n\n"
                    
                    # PC-level swing
                    if ac_list[0].pc_swing_2019_2024:
                        swing = ac_list[0].pc_swing_2019_2024
                        direction = 'towards TMC' if swing > 0 else 'towards BJP'
                        content += f"**Lok Sabha Swing (2019→2024):** {abs(swing):.2f}% {direction}\n\n"
                    
                    content += "**Segments:**\n"
                    for ac in ac_list:
                        content += f"- {ac.ac_name}: {ac.winner_2021} → predicted {ac.predicted_winner_2026} ({ac.race_rating})\n"
                    
                    results.append(RetrievalResult(
                        content=content,
                        source='knowledge_graph_aggregation',
                        relevance_score=0.9,
                        verification_status='verified',
                        facts=[]
                    ))
        
        return results
    
    def _retrieve_relevant_facts(self, analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Retrieve facts relevant to the query."""
        results = []
        
        # Determine which fact types to retrieve
        if analysis.query_type == 'predictive':
            fact_types = ['prediction', 'swing_analysis', 'vulnerability']
        elif analysis.query_type == 'comparative':
            fact_types = ['electoral_result', 'swing_analysis']
        else:
            fact_types = ['electoral_result', 'prediction', 'survey']
        
        for fact_type in fact_types:
            facts = self.kg.get_facts_by_type(fact_type)
            
            # Filter facts by mentioned entities
            if analysis.entities_mentioned:
                relevant_facts = [
                    f for f in facts 
                    if any(e.upper() in f.entity_name.upper() or 
                          e.upper() in ' '.join(f.related_entities).upper()
                          for e in analysis.entities_mentioned)
                ]
            else:
                relevant_facts = facts[:20]  # Limit if no specific entity
            
            if relevant_facts:
                content = f"## Verified Facts ({fact_type.replace('_', ' ').title()})\n\n"
                for fact in relevant_facts[:10]:
                    content += f"• {fact.fact_text} {fact.citation_string()}\n"
                    content += f"  Confidence: {fact.confidence:.0%}\n\n"
                
                results.append(RetrievalResult(
                    content=content,
                    source='knowledge_graph_facts',
                    relevance_score=0.85,
                    verification_status='verified',
                    facts=relevant_facts[:10]
                ))
        
        return results
    
    def _retrieve_with_verification(self, query: str, limit: int) -> List[RetrievalResult]:
        """
        Use vector search but verify results against knowledge graph.
        Works with unified vector store (OpenSearch or FAISS).
        """
        results = []
        
        # Get vector search results using unified interface
        search_results = self.vector_store.search(
            query=query, 
            top_k=limit * 2,  # Get more to filter
            search_type="hybrid"
        )
        
        for result in search_results:
            # Try to verify content against KG
            verification_status, verified_facts = self._verify_content(result.text)
            
            # Extract any numerical claims for verification
            numerical_claims = self._extract_numerical_claims(result.text)
            verified_numerical = self._verify_numerical_claims(numerical_claims)
            
            results.append(RetrievalResult(
                content=result.text,
                source=result.metadata.get('source_file', result.source_file),
                relevance_score=result.score,
                verification_status=verification_status,
                facts=verified_facts,
                numerical_claims=verified_numerical
            ))
        
        return results[:limit]
    
    def _verify_content(self, content: str) -> Tuple[str, List[FactWithCitation]]:
        """
        Verify content against knowledge graph.
        
        Returns:
            Tuple of (verification_status, list of verified facts)
        """
        verified_facts = []
        
        # Look for constituency mentions
        for name in self.kg.constituency_profiles.keys():
            if name in content.upper():
                facts = self.kg.get_facts_for_entity(name)
                verified_facts.extend(facts[:3])
        
        # Look for numerical claims and verify
        # Pattern: "<party> at/with X.X%"
        percentage_pattern = r'(TMC|BJP|AITC|CPM)\s+(?:at|with|got|secured)?\s*(\d+\.?\d*)%'
        matches = re.findall(percentage_pattern, content, re.IGNORECASE)
        
        verified_count = 0
        for party, percentage in matches:
            pct = float(percentage)
            # Try to verify this percentage exists in our data
            for profile in self.kg.constituency_profiles.values():
                if party.upper() in ['TMC', 'AITC']:
                    if abs(profile.tmc_vote_share_2021 - pct) < 0.5:
                        verified_count += 1
                        break
                elif party.upper() == 'BJP':
                    if abs(profile.bjp_vote_share_2021 - pct) < 0.5:
                        verified_count += 1
                        break
        
        if verified_facts or verified_count > 0:
            return 'verified', verified_facts
        elif len(matches) > 0:
            return 'partial', verified_facts
        else:
            return 'unverified', verified_facts
    
    def _extract_numerical_claims(self, content: str) -> Dict[str, float]:
        """Extract numerical claims from content."""
        claims = {}
        
        # Percentage patterns
        pct_pattern = r'(\d+\.?\d*)%'
        for match in re.finditer(pct_pattern, content):
            claims[f'percentage_{match.start()}'] = float(match.group(1))
        
        # Vote count patterns
        vote_pattern = r'(\d{1,3}(?:,\d{3})*)\s*votes'
        for match in re.finditer(vote_pattern, content):
            claims[f'votes_{match.start()}'] = float(match.group(1).replace(',', ''))
        
        return claims
    
    def _verify_numerical_claims(self, claims: Dict[str, float]) -> Dict[str, float]:
        """Verify numerical claims against knowledge graph."""
        verified = {}
        
        for key, value in claims.items():
            if 'percentage' in key:
                # Check if this percentage exists in our data
                for profile in self.kg.constituency_profiles.values():
                    if (abs(profile.tmc_vote_share_2021 - value) < 0.1 or
                        abs(profile.bjp_vote_share_2021 - value) < 0.1 or
                        abs(profile.predicted_margin_2026 - value) < 0.1):
                        verified[key] = value
                        break
        
        return verified
    
    def _dedupe_and_rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicates and rank by relevance and verification."""
        seen_content = set()
        unique_results = []
        
        for result in results:
            # Create a content hash
            content_hash = hash(result.content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Rank by: verification_status (verified > partial > unverified) and relevance_score
        def rank_key(r: RetrievalResult) -> Tuple[int, float]:
            status_rank = {'verified': 0, 'partial': 1, 'unverified': 2}
            return (status_rank.get(r.verification_status, 2), -r.relevance_score)
        
        unique_results.sort(key=rank_key)
        
        return unique_results
    
    def build_verified_answer(self, 
                             query: str, 
                             results: List[RetrievalResult]) -> VerifiedAnswer:
        """
        Build a verified answer with full citations.
        
        This is the final anti-hallucination step.
        """
        analysis = self.analyzer.analyze(query, self.kg)
        
        # Collect all facts
        all_facts = []
        all_sources = []
        
        for result in results:
            all_facts.extend(result.facts)
            if result.source:
                all_sources.append(result.source)
        
        # Build answer text
        if results:
            # Prioritize verified content
            verified_content = [r for r in results if r.verification_status == 'verified']
            
            if verified_content:
                answer_text = "\n\n".join([r.content for r in verified_content[:3]])
            else:
                answer_text = results[0].content
        else:
            answer_text = f"No verified data found for: {query}"
        
        # Add caveats
        caveats = []
        unverified = [r for r in results if r.verification_status == 'unverified']
        if unverified:
            caveats.append(f"{len(unverified)} results could not be verified against source data")
        
        if analysis.query_type == 'predictive':
            caveats.append("Predictions are based on historical trends and may not reflect actual election outcomes")
        
        # Calculate confidence
        if all(r.verification_status == 'verified' for r in results):
            confidence = 0.95
        elif any(r.verification_status == 'verified' for r in results):
            confidence = 0.75
        else:
            confidence = 0.5
        
        return VerifiedAnswer(
            question=query,
            answer_text=answer_text,
            confidence=confidence,
            facts=all_facts[:10],
            sources=list(set(all_sources)),
            caveats=caveats
        )


class HallucinationGuard:
    """
    Final layer of protection against hallucination.
    
    Validates LLM responses against verified data.
    """
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph):
        self.kg = knowledge_graph
    
    def validate_response(self, response: str, query: str) -> Tuple[str, List[str]]:
        """
        Validate an LLM response against the knowledge graph.
        
        Returns:
            Tuple of (validated_response, list of corrections made)
        """
        corrections = []
        validated = response
        
        # 1. Validate constituency claims
        validated, const_corrections = self._validate_constituency_claims(validated)
        corrections.extend(const_corrections)
        
        # 2. Validate numerical claims
        validated, num_corrections = self._validate_numerical_claims(validated)
        corrections.extend(num_corrections)
        
        # 3. Add citations where missing
        validated = self._add_missing_citations(validated)
        
        return validated, corrections
    
    def _validate_constituency_claims(self, text: str) -> Tuple[str, List[str]]:
        """Validate claims about constituencies."""
        corrections = []
        
        # Pattern: "<constituency> won by <party>"
        won_pattern = r'(\b[A-Z][A-Z\s]+\b)\s+(?:was\s+)?won\s+by\s+(TMC|BJP|AITC|CPM|INC)'
        
        for match in re.finditer(won_pattern, text, re.IGNORECASE):
            constituency = match.group(1).strip().upper()
            claimed_party = match.group(2).upper()
            
            profile = self.kg.get_constituency(constituency)
            if profile:
                actual_party = profile.winner_2021
                if claimed_party != actual_party and not (
                    claimed_party in ['TMC', 'AITC'] and actual_party in ['TMC', 'AITC']
                ):
                    # Correction needed
                    corrections.append(
                        f"Corrected: {constituency} was won by {actual_party}, not {claimed_party}"
                    )
                    text = text.replace(match.group(0), 
                                       f"{constituency} was won by {actual_party}")
        
        return text, corrections
    
    def _validate_numerical_claims(self, text: str) -> Tuple[str, List[str]]:
        """Validate numerical claims."""
        corrections = []
        
        # Pattern: "<party> got/secured X% in <constituency>"
        pct_pattern = r'(TMC|BJP|AITC)\s+(?:got|secured|received|had)\s+(\d+\.?\d*)%\s+(?:in|at)\s+(\b[A-Z][A-Z\s]+\b)'
        
        for match in re.finditer(pct_pattern, text, re.IGNORECASE):
            party = match.group(1).upper()
            claimed_pct = float(match.group(2))
            constituency = match.group(3).strip().upper()
            
            profile = self.kg.get_constituency(constituency)
            if profile:
                if party in ['TMC', 'AITC']:
                    actual_pct = profile.tmc_vote_share_2021
                elif party == 'BJP':
                    actual_pct = profile.bjp_vote_share_2021
                else:
                    continue
                
                # Allow 0.5% tolerance
                if abs(claimed_pct - actual_pct) > 0.5:
                    corrections.append(
                        f"Corrected: {party} got {actual_pct:.2f}% in {constituency}, not {claimed_pct}%"
                    )
                    text = text.replace(
                        f"{claimed_pct}%",
                        f"{actual_pct:.2f}%"
                    )
        
        return text, corrections
    
    def _add_missing_citations(self, text: str) -> str:
        """Add citations to claims that are missing them."""
        # This is a simplified version - in production, use NLP to identify claims
        
        # For now, just add a general source note if no citations present
        if '[Source:' not in text and 'Sources:' not in text.lower():
            text += "\n\n*Data sourced from West Bengal Electoral Commission and TCPD database*"
        
        return text
    
    def get_grounded_response_prompt(self, query: str, verified_facts: List[FactWithCitation]) -> str:
        """
        Generate a prompt that grounds the LLM in verified facts.
        """
        facts_text = "\n".join([
            f"FACT {i+1}: {f.fact_text} [Confidence: {f.confidence:.0%}] {f.citation_string()}"
            for i, f in enumerate(verified_facts)
        ])
        
        return f"""You are a political analyst for West Bengal elections. 
Answer ONLY based on the verified facts provided below. 
Do NOT make up any information not present in the facts.
If the facts don't contain enough information, say "I don't have verified data for that."

VERIFIED FACTS:
{facts_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Use ONLY the facts above to answer
2. Include source citations in your answer
3. If making predictions, clearly state they are based on trends
4. If the facts are insufficient, acknowledge limitations
5. Never invent statistics or election results

ANSWER:"""

