"""
Electoral Strategist Agent - Party-focused strategic planning.

This agent specializes in:
- Party strength/weakness analysis
- Victory path planning
- Resource prioritization
- Opposition strategy
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .evidence_framework import (
    PoliticalAgentBase, Evidence, Claim, EvidenceType, ConfidenceLevel,
    AgentCapability, calculate_seat_metrics, format_constituency_brief
)
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.rag.data_schema import ConstituencyProfile


class ElectoralStrategistAgent(PoliticalAgentBase):
    """
    Party-level electoral strategy with evidence-based recommendations.
    
    Capabilities:
    - Party performance analysis
    - Victory path calculation
    - Seat prioritization
    - Resource allocation recommendations
    """
    
    name = "Electoral Strategist"
    role = "Strategic campaign planning and party positioning expert"
    expertise = [
        "Party performance analysis",
        "Victory path modeling",
        "Seat prioritization frameworks",
        "Resource allocation optimization",
        "Opposition strategy assessment"
    ]
    
    capabilities = [
        AgentCapability(
            name="party_analysis",
            description="Comprehensive party performance analysis",
            input_types=["party_name"],
            output_types=["analysis", "claims", "evidence"],
            required_data=["electoral_results", "predictions"]
        ),
        AgentCapability(
            name="victory_path",
            description="Calculate path to victory for a party",
            input_types=["party_name", "target_seats"],
            output_types=["strategy", "priority_seats"],
            required_data=["predictions", "swing_analysis"]
        ),
        AgentCapability(
            name="resource_allocation",
            description="Recommend resource allocation across seats",
            input_types=["party_name", "budget"],
            output_types=["allocation_plan"],
            required_data=["predictions", "vulnerability"]
        )
    ]
    
    QUERY_KEYWORDS = [
        'party', 'bjp', 'tmc', 'trinamool', 'congress', 'cpm', 'left',
        'strategy', 'win', 'victory', 'seats', 'campaign', 'resources',
        'allocation', 'priority', 'focus', 'target', 'stronghold', 'weakness'
    ]
    
    def can_handle(self, query: str) -> Tuple[bool, float]:
        """Check if this agent can handle the query."""
        query_lower = query.lower()
        
        # Party-specific queries
        if any(p in query_lower for p in ['bjp', 'tmc', 'trinamool', 'congress', 'cpm']):
            if any(s in query_lower for s in ['strategy', 'win', 'seats', 'analysis', 'performance']):
                return True, 0.95
            return True, 0.7
        
        # Strategy keywords
        keyword_matches = sum(1 for kw in self.QUERY_KEYWORDS if kw in query_lower)
        if keyword_matches >= 3:
            return True, 0.85
        elif keyword_matches >= 2:
            return True, 0.7
        
        return False, 0.0
    
    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main analysis entry point - routes based on query intent."""
        self.reset()
        context = context or {}
        query_lower = query.lower()
        
        # First, try to answer the specific query using RAG
        rag_response = self._try_rag_query(query)
        if rag_response and rag_response.get('confidence', 0) >= 0.7:
            return rag_response
        
        # Identify party from context or query
        party = context.get('party') or self._extract_party(query)
        
        # Check for district-level queries FIRST (before constituency)
        # This is important because some district names match constituency names
        if 'district' in query_lower:
            district = self._extract_district(query)
            if district:
                # If party also mentioned, do party-specific district analysis
                if party:
                    return self._party_district_analysis(party, district, query, context)
                return self._district_analysis(district, query, context)
        
        # Handle constituency-specific queries
        constituency = context.get('constituency') or self._extract_constituency(query)
        if constituency and 'district' not in query_lower:
            return self._constituency_analysis(constituency, query, context)
        
        # Handle specific analysis types with party context
        if party:
            # HIGHEST PRIORITY: Strategic recommendations and action points queries
            # This should come BEFORE prediction check since strategy queries often mention future years
            if any(w in query_lower for w in ['strategic decision', 'action point', 'implement', 
                                              'recommend', 'should do', 'improve', 'steps', 'plan',
                                              'what should', 'how to improve', 'how can', 'win more',
                                              'strategy for', 'strategies', 'action', 'steps to']):
                return self._strategic_recommendations(party, query, context)
        
        # Handle prediction/forecast queries (only if not already a strategy query)
        if any(w in query_lower for w in ['predict', 'forecast', 'will win', 'expected']):
            if party:
                return self._comprehensive_party_analysis(party, query, context)
            else:
                return self._overall_electoral_analysis(query)
        
        # Handle other specific analysis types with party context
        if party:
            # Victory path analysis
            if any(w in query_lower for w in ['victory path', 'path to victory']):
                return self._victory_path_analysis(party, query, context)
            elif any(w in query_lower for w in ['resource', 'allocate', 'prioritize', 'budget']):
                return self._resource_allocation(party, query, context)
            elif any(w in query_lower for w in ['vulnerable', 'risk', 'lose', 'threat', 'at risk']):
                return self._vulnerability_analysis(party, query, context)
            elif any(w in query_lower for w in ['safe', 'bastion', 'fortress', 'stronghold']):
                return self._strength_analysis(party, query, context)
            else:
                return self._comprehensive_party_analysis(party, query, context)
        
        # Default: Use RAG to find relevant information for the query
        return self._query_based_analysis(query, context)
    
    def _extract_party(self, query: str) -> Optional[str]:
        """Extract party from query."""
        query_lower = query.lower()
        
        if 'bjp' in query_lower or 'bharatiya janata' in query_lower:
            return 'BJP'
        elif 'tmc' in query_lower or 'trinamool' in query_lower or 'aitc' in query_lower:
            return 'TMC'
        elif 'congress' in query_lower or 'inc' in query_lower:
            return 'INC'
        elif 'cpm' in query_lower or 'cpim' in query_lower or 'communist' in query_lower or 'left' in query_lower:
            return 'CPM'
        
        return None
    
    def _extract_constituency(self, query: str) -> Optional[str]:
        """Extract constituency name from query."""
        query_upper = query.upper()
        
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                return name
        return None
    
    def _try_rag_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Try to answer using RAG search for relevant data."""
        try:
            # Use search_data to find relevant information
            results = self.search_data(query, top_k=5)
            
            if results and len(results) > 0:
                # Check if results have enough context
                total_score = sum(r.get('score', 0) for r in results)
                if total_score > 3.0:  # Good relevance
                    # Build answer from search results
                    answer_parts = []
                    sources = []
                    for r in results:
                        if r.get('text'):
                            answer_parts.append(r['text'])
                        if r.get('source'):
                            sources.append(r['source'])
                    
                    if answer_parts:
                        return {
                            "answer": "\n\n".join(answer_parts[:3]),
                            "confidence": min(0.85, total_score / 5),
                            "sources": list(set(sources)),
                            "verification_status": "high_confidence",
                            "claims": [],
                            "evidence": [{"content": a[:200], "source": s} 
                                        for a, s in zip(answer_parts, sources)]
                        }
        except Exception as e:
            pass
        return None
    
    def _constituency_analysis(self, constituency: str, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze a specific constituency with strategy recommendations."""
        profile = self.kg.constituency_profiles.get(constituency.upper())
        
        if not profile:
            return {
                "answer": f"No data found for constituency: {constituency}",
                "confidence": 0.3,
                "claims": [],
                "evidence": []
            }
        
        # Determine party focus from query
        party = context.get('party') or self._extract_party(query)
        
        answer = f"## {profile.ac_name} Constituency Analysis\n\n"
        answer += f"**Location:** {profile.district} district, part of {profile.parent_pc} PC\n"
        answer += f"**Category:** {profile.constituency_type.value if hasattr(profile.constituency_type, 'value') else profile.constituency_type} seat\n\n"
        
        answer += "### 2021 Assembly Election\n"
        answer += f"- **Winner:** {profile.winner_2021}\n"
        answer += f"- **TMC Vote Share:** {profile.tmc_vote_share_2021:.2f}%\n"
        answer += f"- **BJP Vote Share:** {profile.bjp_vote_share_2021:.2f}%\n"
        answer += f"- **Margin:** {profile.margin_2021:.2f}%\n\n"
        
        if profile.pc_swing_2019_2024:
            answer += f"### Lok Sabha Trends ({profile.parent_pc} PC)\n"
            tmc_2019 = profile.tmc_vote_share_2019_pc or 0
            bjp_2019 = profile.bjp_vote_share_2019_pc or 0
            tmc_2024 = profile.tmc_vote_share_2024_pc or 0
            bjp_2024 = profile.bjp_vote_share_2024_pc or 0
            answer += f"- **2019:** TMC {tmc_2019:.2f}% vs BJP {bjp_2019:.2f}%\n"
            answer += f"- **2024:** TMC {tmc_2024:.2f}% vs BJP {bjp_2024:.2f}%\n"
            swing_dir = "towards TMC" if profile.pc_swing_2019_2024 > 0 else "towards BJP"
            answer += f"- **Swing:** {abs(profile.pc_swing_2019_2024):.2f}% {swing_dir}\n\n"
        
        answer += "### 2026 Prediction\n"
        answer += f"- **Predicted Winner:** {profile.predicted_winner_2026}\n"
        answer += f"- **Predicted Margin:** {abs(profile.predicted_margin_2026):.2f}%\n"
        answer += f"- **Race Rating:** {profile.race_rating}\n\n"
        
        # ===== ADD STRATEGY RECOMMENDATIONS =====
        # Search for additional strategic context from uploaded documents
        # Use DIRECT vector search to bypass KG-prioritized retrieval
        evidence_list = []
        strategy_insights = []
        district_name = profile.district if profile else ""
        
        try:
            # First, try direct vector store search to get uploaded documents
            from app.services.rag.vector_store import get_vector_store
            vector_store = get_vector_store()
            
            # Search for strategy content with multiple query variations
            # Include constituency-specific, district-level, and general party strategy searches
            search_queries = [
                f"{party or 'BJP'} strategy {constituency} recommendations",
                f"campaign strategy {constituency} election",
                f"what should {party or 'BJP'} do in {constituency}",
                # Add district-level search
                f"{party or 'BJP'} strategy {district_name} district" if district_name else None,
                f"ground report {district_name}" if district_name else None,
                # Add general party strategy search (important for cross-constituency insights)
                f"{party or 'BJP'} strategy West Bengal election 2026",
                f"{party or 'BJP'} campaign problems issues Bengal"
            ]
            search_queries = [q for q in search_queries if q]  # Remove None
            
            all_vector_results = []
            for sq in search_queries:
                try:
                    results = vector_store.search(sq, top_k=5, search_type="hybrid")
                    all_vector_results.extend(results)
                except:
                    pass
            
            # Deduplicate and filter for strategy content
            seen_texts = set()
            for result in all_vector_results:
                text = result.text if hasattr(result, 'text') else str(result)
                source = result.source_file if hasattr(result, 'source_file') else 'Unknown'
                
                # Skip KG entries for strategy queries - we want uploaded docs
                if source == 'knowledge_graph' or 'knowledge_graph' in str(source).lower():
                    continue
                
                text_hash = hash(text[:100])
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
                
                # Filter for strategy-related content (broader keywords)
                if any(w in text.lower() for w in ['strategy', 'recommend', 'should', 'focus', 'campaign', 
                                                    'action', 'plan', 'priority', 'target', 'strengthen',
                                                    'mobilize', 'outreach', 'consolidate', 'voter',
                                                    'problem', 'issue', 'feedback', 'ground report',
                                                    'grievance', 'corruption', 'local', 'mla']):
                    strategy_insights.append({
                        'text': text[:600],
                        'source': source,
                        'is_constituency_specific': constituency.lower() in text.lower()
                    })
                    evidence_list.append({
                        "content": text[:400],
                        "source": source,
                        "source_type": "uploaded_document",
                        "score": result.score if hasattr(result, 'score') else 0
                    })
        except Exception as e:
            # Fallback to regular search
            pass
        
        # Also try regular search_data as fallback
        if not strategy_insights:
            search_query = f"strategy {constituency} {party or 'election'}"
            try:
                search_results = self.search_data(search_query, top_k=5)
                
                for result in search_results:
                    text = result.get('text', result.get('content', ''))
                    source = result.get('source_file', result.get('source', 'Unknown'))
                    
                    # Skip KG sources
                    if 'knowledge_graph' in str(source).lower():
                        continue
                    
                    # Filter for strategy-related content
                    if any(w in text.lower() for w in ['strategy', 'recommend', 'should', 'focus', 'campaign', 'action', 'plan', 'problem', 'issue']):
                        strategy_insights.append({
                            'text': text[:500],
                            'source': source,
                            'is_constituency_specific': constituency.lower() in text.lower()
                        })
                        evidence_list.append({
                            "content": text[:300],
                            "source": source,
                            "source_type": "uploaded_document"
                        })
            except:
                pass
            
        if strategy_insights:
            # Separate constituency-specific and general insights
            constituency_specific = [s for s in strategy_insights if s.get('is_constituency_specific', False)]
            general_insights = [s for s in strategy_insights if not s.get('is_constituency_specific', False)]
            
            if constituency_specific:
                answer += f"\n### Strategic Insights for {constituency} (from uploaded documents)\n"
                for i, insight in enumerate(constituency_specific[:3], 1):
                    text = insight['text'].replace('\n', ' ').strip()
                    if len(text) > 250:
                        text = text[:250] + "..."
                    answer += f"{i}. {text}\n"
                    answer += f"   *Source: {insight['source']}*\n\n"
            
            if general_insights and len(constituency_specific) < 2:
                answer += f"\n### Relevant Insights from Other Areas (may apply to {constituency})\n"
                for i, insight in enumerate(general_insights[:3 - len(constituency_specific)], 1):
                    text = insight['text'].replace('\n', ' ').strip()
                    if len(text) > 200:
                        text = text[:200] + "..."
                    answer += f"{i}. {text}\n"
                    answer += f"   *Source: {insight['source']}*\n\n"
        
        # Generate party-specific strategy
        if party:
            answer += f"\n### Strategy for {party} in {constituency}\n"
            
            if party.upper() == 'BJP':
                if profile.winner_2021 == 'BJP':
                    answer += f"- **Seat Status:** Currently held by BJP\n"
                    answer += f"- **Priority:** Defend with margin of {profile.margin_2021:.1f}%\n"
                    answer += f"- **Focus:** Strengthen booth management, address local grievances\n"
                else:
                    gap = profile.tmc_vote_share_2021 - profile.bjp_vote_share_2021
                    answer += f"- **Seat Status:** Currently held by {profile.winner_2021}\n"
                    answer += f"- **Gap to Close:** {gap:.1f}%\n"
                    
                    if gap <= 5:
                        answer += f"- **Assessment:** Winnable with focused campaign\n"
                        answer += f"- **Strategy:** Intensify ground presence, target swing voters\n"
                    elif gap <= 10:
                        answer += f"- **Assessment:** Competitive but challenging\n"
                        answer += f"- **Strategy:** Build local organization, anti-incumbency campaign\n"
                    else:
                        answer += f"- **Assessment:** Difficult, requires significant swing\n"
                        answer += f"- **Strategy:** Long-term organization building, issue-based campaign\n"
            
            elif party.upper() in ['TMC', 'AITC']:
                if profile.winner_2021 in ['TMC', 'AITC']:
                    answer += f"- **Seat Status:** Currently held by TMC\n"
                    answer += f"- **Priority:** Defend with strong local presence\n"
                    answer += f"- **Focus:** Development delivery, voter connect\n"
                else:
                    answer += f"- **Seat Status:** Currently held by {profile.winner_2021}\n"
                    answer += f"- **Strategy:** Target opposition weaknesses\n"
        
        # Key insights
        answer += "\n### Key Insights\n"
        if profile.race_rating in ['Toss-up', 'Lean']:
            answer += f"- ⚠️ {constituency} is a **swing seat** - highly competitive\n"
        if profile.margin_2021 < 5:
            answer += f"- 📊 Very narrow margin in 2021 ({profile.margin_2021:.1f}%) indicates volatility\n"
        answer += f"- {constituency} was won by {profile.winner_2021} in 2021 with a margin of {profile.margin_2021:.2f}%\n"
        
        ev = self.create_evidence_from_profile(profile, "2026_prediction")
        claims = [self.make_claim(
            statement=f"{constituency} is predicted to be won by {profile.predicted_winner_2026} with {abs(profile.predicted_margin_2026):.1f}% margin",
            evidence=[ev],
            reasoning="Based on swing analysis and historical voting patterns"
        )]
        
        # Combine KG evidence with search evidence
        all_evidence = [{"content": ev.content, "source": ev.source}] + evidence_list
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": all_evidence,
            "confidence": 0.9,
            "constituency": constituency
        }
    
    def _query_based_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle generic queries by understanding what the user is asking."""
        query_lower = query.lower()
        
        # FIRST: Check for district/location questions - these are very specific
        district = self._extract_district(query)
        if district:
            # Check if asking about strength/dominance/which party
            if any(w in query_lower for w in ['stronger', 'dominant', 'which party', 'who is leading', 'who controls', 'stronghold']):
                return self._district_strength_analysis(district, query, context)
            return self._district_analysis(district, query, context)
        
        # Voter segment / persuadable groups questions
        if any(w in query_lower for w in ['voter', 'segment', 'persuadable', 'demographic', 'groups', 'target']):
            return self._voter_segment_analysis(query, context)
        
        # Who won questions
        if 'who won' in query_lower or 'winner' in query_lower:
            return self._winner_analysis(query, context)
        
        # Seat count questions
        if 'how many seats' in query_lower or 'seat count' in query_lower or 'total seats' in query_lower:
            return self._overall_electoral_analysis(query)
        
        # Swing/trend questions
        if 'swing' in query_lower or 'trend' in query_lower or 'change' in query_lower:
            return self._swing_analysis(query, context)
        
        # Party strength questions (without district)
        if any(w in query_lower for w in ['stronger', 'dominant', 'which party']):
            return self._overall_strength_analysis(query, context)
        
        # Default to overall electoral analysis
        return self._overall_electoral_analysis(query)
    
    # Common district aliases/mappings
    DISTRICT_ALIASES = {
        'kolkata': ['KOLKATA', 'SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS'],  # Kolkata area spans multiple districts
        'calcutta': ['KOLKATA', 'SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS'],
        '24 parganas': ['SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS'],
        'parganas': ['SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS'],
        'midnapore': ['PASCHIM MEDINIPUR', 'PURBA MEDINIPUR'],
        'medinipur': ['PASCHIM MEDINIPUR', 'PURBA MEDINIPUR'],
        'dinajpur': ['DAKSHIN DINAJPUR', 'UTTAR DINAJPUR'],
        'bardhaman': ['PURBA BARDHAMAN', 'PASCHIM BARDHAMAN'],
        'burdwan': ['PURBA BARDHAMAN', 'PASCHIM BARDHAMAN'],
    }
    
    def _extract_district(self, query: str) -> Optional[str]:
        """Extract district name from query."""
        districts = set(p.district for p in self.kg.constituency_profiles.values())
        query_lower = query.lower()
        
        # First check exact district matches
        for dist in districts:
            if dist.lower() in query_lower:
                return dist
        
        # Then check aliases
        for alias, actual_districts in self.DISTRICT_ALIASES.items():
            if alias in query_lower:
                # Return the first matching district that exists in our data
                for actual in actual_districts:
                    if actual in districts:
                        return actual
                # If asking about "KOLKATA" which spans multiple districts, 
                # return a special marker for multi-district analysis
                if alias in ['kolkata', 'calcutta']:
                    return 'KOLKATA_AREA'
        
        return None
    
    def _get_available_districts(self) -> List[str]:
        """Get list of available districts."""
        return sorted(set(p.district for p in self.kg.constituency_profiles.values()))
    
    def _winner_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze winner questions."""
        # Check for year context
        year = "2021"
        if "2026" in query or "predict" in query.lower():
            year = "2026"
        elif "2019" in query:
            year = "2019"
        
        # Check for constituency
        constituency = self._extract_constituency(query)
        if constituency:
            profile = self.kg.constituency_profiles.get(constituency)
            if profile:
                if year == "2026":
                    winner = profile.predicted_winner_2026
                    answer = f"**{constituency}** is predicted to be won by **{winner}** in 2026 with a margin of {abs(profile.predicted_margin_2026):.1f}%."
                else:
                    winner = profile.winner_2021
                    answer = f"**{constituency}** was won by **{winner}** in 2021 with a margin of {abs(profile.margin_2021):.1f}%."
                
                return {
                    "answer": answer,
                    "confidence": 0.95,
                    "claims": [],
                    "evidence": [{"content": answer, "source": "Electoral data"}]
                }
        
        # General winner info
        return self._overall_electoral_analysis(query)
    
    def _voter_segment_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze voter segments and identify persuadable groups."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Categorize constituencies by competitiveness (persuadability proxy)
        tossup_seats = [c for c in all_seats if c.race_rating.lower() == 'toss-up']
        lean_seats = [c for c in all_seats if c.race_rating.lower() == 'lean']
        likely_seats = [c for c in all_seats if c.race_rating.lower() == 'likely']
        safe_seats = [c for c in all_seats if c.race_rating.lower() == 'safe']
        
        # Identify swing voters (constituencies with high swing)
        high_swing = sorted([c for c in all_seats if abs(c.pc_swing_2019_2024) > 10],
                           key=lambda x: abs(x.pc_swing_2019_2024), reverse=True)
        
        # Close margin seats (most persuadable)
        close_margin = sorted([c for c in all_seats if abs(c.predicted_margin_2026) < 5],
                             key=lambda x: abs(x.predicted_margin_2026))
        
        # Flipping seats (changed allegiance)
        flipping_to_tmc = [c for c in all_seats 
                          if c.winner_2021 == 'BJP' and c.predicted_winner_2026 in ['TMC', 'AITC']]
        flipping_to_bjp = [c for c in all_seats 
                          if c.winner_2021 in ['TMC', 'AITC'] and c.predicted_winner_2026 == 'BJP']
        
        # Constituency type distribution
        gen_seats = [c for c in all_seats if 'GEN' in str(c.constituency_type).upper()]
        sc_seats = [c for c in all_seats if 'SC' in str(c.constituency_type).upper()]
        st_seats = [c for c in all_seats if 'ST' in str(c.constituency_type).upper()]
        
        answer = "## Voter Segment Analysis & Persuadable Groups\n\n"
        
        # Segment 1: Swing Voters
        answer += "### 1. Swing Voters (High Volatility Areas)\n"
        answer += f"**Total High-Swing Constituencies (>10% swing):** {len(high_swing)}\n\n"
        answer += "These areas showed significant voter movement between 2019-2024:\n"
        for c in high_swing[:8]:
            direction = "→ TMC" if c.pc_swing_2019_2024 > 0 else "→ BJP"
            answer += f"- **{c.ac_name}** ({c.district}): {abs(c.pc_swing_2019_2024):.1f}% {direction}\n"
        answer += "\n**Strategy:** These voters are highly persuadable. Focus on recent governance issues.\n\n"
        
        # Segment 2: Fence-Sitters (Toss-up)
        answer += "### 2. Fence-Sitters (Undecided/Toss-up Areas)\n"
        answer += f"**Total Toss-up Constituencies:** {len(tossup_seats)}\n\n"
        answer += "Top undecided battlegrounds:\n"
        for c in tossup_seats[:8]:
            answer += f"- **{c.ac_name}** ({c.district}): {c.predicted_winner_2026} +{abs(c.predicted_margin_2026):.1f}%\n"
        answer += "\n**Strategy:** Intensive ground campaign, door-to-door contact, local issue focus.\n\n"
        
        # Segment 3: Soft Supporters (Lean seats)
        answer += "### 3. Soft Supporters (Lean Seats - Persuadable)\n"
        answer += f"**Total Lean Constituencies:** {len(lean_seats)}\n\n"
        
        lean_tmc = [c for c in lean_seats if c.predicted_winner_2026 in ['TMC', 'AITC']]
        lean_bjp = [c for c in lean_seats if c.predicted_winner_2026 == 'BJP']
        
        answer += f"- Leaning TMC: {len(lean_tmc)} seats\n"
        answer += f"- Leaning BJP: {len(lean_bjp)} seats\n\n"
        answer += "**Strategy:** Reinforce messaging, address doubts, mobilize base.\n\n"
        
        # Segment 4: Reserved Constituency Voters
        answer += "### 4. Reserved Constituency Demographics\n"
        answer += f"- **General (GEN):** {len(gen_seats)} seats\n"
        answer += f"- **Scheduled Caste (SC):** {len(sc_seats)} seats\n"
        answer += f"- **Scheduled Tribe (ST):** {len(st_seats)} seats\n\n"
        
        # SC Analysis
        sc_tmc = sum(1 for c in sc_seats if c.predicted_winner_2026 in ['TMC', 'AITC'])
        sc_bjp = sum(1 for c in sc_seats if c.predicted_winner_2026 == 'BJP')
        answer += f"SC Seat Prediction: TMC {sc_tmc}, BJP {sc_bjp}\n"
        
        # ST Analysis  
        st_tmc = sum(1 for c in st_seats if c.predicted_winner_2026 in ['TMC', 'AITC'])
        st_bjp = sum(1 for c in st_seats if c.predicted_winner_2026 == 'BJP')
        answer += f"ST Seat Prediction: TMC {st_tmc}, BJP {st_bjp}\n\n"
        
        # Segment 5: Defectors (Flipping seats)
        answer += "### 5. Defecting Voters (Constituency Flips)\n"
        answer += f"- **Flipping BJP → TMC:** {len(flipping_to_tmc)} seats\n"
        answer += f"- **Flipping TMC → BJP:** {len(flipping_to_bjp)} seats\n\n"
        
        if flipping_to_tmc:
            answer += "BJP seats at risk:\n"
            for c in flipping_to_tmc[:5]:
                answer += f"- {c.ac_name} ({c.district}): now TMC +{abs(c.predicted_margin_2026):.1f}%\n"
        
        if flipping_to_bjp:
            answer += "\nTMC seats at risk:\n"
            for c in flipping_to_bjp[:5]:
                answer += f"- {c.ac_name} ({c.district}): now BJP +{abs(c.predicted_margin_2026):.1f}%\n"
        
        # Persuadability Matrix
        answer += "\n### 📊 Persuadability Matrix\n\n"
        answer += "| Segment | Count | Persuadability | Priority |\n"
        answer += "|---------|-------|----------------|----------|\n"
        answer += f"| Toss-up Seats | {len(tossup_seats)} | Very High | ⭐⭐⭐⭐⭐ |\n"
        answer += f"| Lean Seats | {len(lean_seats)} | High | ⭐⭐⭐⭐ |\n"
        answer += f"| High-Swing Areas | {len(high_swing)} | High | ⭐⭐⭐⭐ |\n"
        answer += f"| Flipping Seats | {len(flipping_to_tmc) + len(flipping_to_bjp)} | Medium-High | ⭐⭐⭐ |\n"
        answer += f"| Likely Seats | {len(likely_seats)} | Medium | ⭐⭐ |\n"
        answer += f"| Safe Seats | {len(safe_seats)} | Low | ⭐ |\n"
        
        # Strategic Recommendations
        answer += "\n### 🎯 Strategic Recommendations\n\n"
        answer += "1. **Primary Target:** Focus 50% resources on toss-up + lean seats\n"
        answer += "2. **High-Swing Areas:** Deploy strong local candidates with clean image\n"
        answer += "3. **SC/ST Seats:** Tailor messaging to welfare schemes and representation\n"
        answer += "4. **Flipping Seats:** Understand why voters are changing, address concerns\n"
        answer += "5. **Close Margins (<2%):** Every vote counts - maximize turnout\n"
        
        claims = [self.make_claim(
            statement=f"Identified {len(tossup_seats)} toss-up and {len(lean_seats)} lean constituencies as most persuadable",
            evidence=[self.create_aggregated_evidence("Voter segment analysis", tossup_seats + lean_seats)],
            reasoning="Based on race ratings and margin analysis"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": "Voter segment analysis", "source": "Electoral data"}],
            "confidence": 0.85,
            "segments": {
                "tossup": len(tossup_seats),
                "lean": len(lean_seats),
                "high_swing": len(high_swing),
                "flipping": len(flipping_to_tmc) + len(flipping_to_bjp)
            }
        }
    
    def _swing_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze swing/trends - covers 'swing seats' queries."""
        all_seats = list(self.kg.constituency_profiles.values())
        query_lower = query.lower()
        
        # Check if asking specifically about "swing seats" (competitive seats)
        if 'swing seat' in query_lower or 'swing constituency' in query_lower or 'battleground' in query_lower:
            return self._find_swing_seats(query, context)
        
        # Calculate swings using correct attribute name
        positive_swing = sorted([c for c in all_seats if c.pc_swing_2019_2024 and c.pc_swing_2019_2024 > 0], 
                                key=lambda x: x.pc_swing_2019_2024, reverse=True)
        negative_swing = sorted([c for c in all_seats if c.pc_swing_2019_2024 and c.pc_swing_2019_2024 < 0], 
                                key=lambda x: x.pc_swing_2019_2024)
        
        answer = "## Swing Analysis (2019-2024 Lok Sabha Comparison)\n\n"
        
        answer += "### Top Gains (Pro-TMC Swing)\n"
        for c in positive_swing[:10]:
            answer += f"- **{c.ac_name}** ({c.district}): +{c.pc_swing_2019_2024:.1f}%\n"
        
        answer += "\n### Top Losses (Pro-BJP Swing)\n"
        for c in negative_swing[:10]:
            answer += f"- **{c.ac_name}** ({c.district}): {c.pc_swing_2019_2024:.1f}%\n"
        
        # Average swing
        swings = [c.pc_swing_2019_2024 for c in all_seats if c.pc_swing_2019_2024]
        if swings:
            avg_swing = sum(swings) / len(swings)
            answer += f"\n**Average State Swing:** {avg_swing:+.2f}%\n"
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "claims": [],
            "evidence": [{"content": "Swing analysis from Lok Sabha 2019 vs 2024", "source": "Electoral data"}]
        }
    
    def _find_swing_seats(self, query: str, context: Dict) -> Dict[str, Any]:
        """Find swing seats (competitive/close race constituencies) for 2026."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Swing seats: competitive races with close margins
        swing_seats = []
        for c in all_seats:
            margin = abs(c.predicted_margin_2026) if c.predicted_margin_2026 else 99
            rating = (c.race_rating or '').lower()
            
            # Include seats with <5% margin or rated as toss-up/lean
            if margin < 5 or rating in ['toss-up', 'lean', 'lean tmc', 'lean bjp', 'competitive']:
                swing_seats.append(c)
        
        # Sort by closest margin
        swing_seats = sorted(swing_seats, key=lambda x: abs(x.predicted_margin_2026) if x.predicted_margin_2026 else 99)
        
        answer = f"## Swing Seats in West Bengal (2026 Predictions)\n\n"
        answer += f"**Definition:** Constituencies with predicted margin <5% or rated as competitive\n\n"
        answer += f"**Total Swing Seats:** {len(swing_seats)}\n\n"
        
        # Table format
        answer += "| # | Constituency | District | 2021 Winner | 2026 Predicted | Margin | Rating |\n"
        answer += "|---|--------------|----------|-------------|----------------|--------|--------|\n"
        
        for i, c in enumerate(swing_seats[:20], 1):
            answer += f"| {i} | {c.ac_name} | {c.district} | {c.winner_2021} | {c.predicted_winner_2026} | {abs(c.predicted_margin_2026):.1f}% | {c.race_rating} |\n"
        
        if len(swing_seats) > 20:
            answer += f"\n*...and {len(swing_seats) - 20} more swing seats*\n"
        
        # Analysis
        tmc_defend = [c for c in swing_seats if c.winner_2021 in ['TMC', 'AITC']]
        bjp_defend = [c for c in swing_seats if c.winner_2021 == 'BJP']
        
        answer += f"\n### Strategic Implications\n"
        answer += f"- **TMC defending {len(tmc_defend)} swing seats** (vulnerable)\n"
        answer += f"- **BJP defending {len(bjp_defend)} swing seats** (at risk)\n"
        answer += f"- These {len(swing_seats)} seats will likely decide the 2026 election outcome\n"
        
        # Campaign recommendations
        answer += f"\n### Campaign Focus Areas\n"
        answer += f"1. Prioritize resource allocation to these swing constituencies\n"
        answer += f"2. Deploy strongest candidates in toss-up races\n"
        answer += f"3. Intensive ground campaign in close districts\n"
        
        evidence = [{
            "content": f"Identified {len(swing_seats)} swing seats based on 2026 predictions",
            "source": "Electoral Prediction Model"
        }]
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "claims": [{"statement": f"West Bengal has {len(swing_seats)} swing seats for 2026", "confidence": "high"}],
            "evidence": evidence,
            "swing_seats_count": len(swing_seats),
            "tmc_vulnerable": len(tmc_defend),
            "bjp_vulnerable": len(bjp_defend)
        }
    
    def _party_district_analysis(self, party: str, district: str, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze party's position in a specific district."""
        all_seats = list(self.kg.constituency_profiles.values())
        district_seats = [c for c in all_seats if c.district.upper() == district.upper()]
        
        if not district_seats:
            return {
                "answer": f"No data found for district: {district}",
                "confidence": 0.3,
                "claims": [],
                "evidence": []
            }
        
        # Party-specific analysis
        party_2021 = [c for c in district_seats if c.winner_2021.upper() == party.upper()]
        party_2026 = [c for c in district_seats if c.predicted_winner_2026.upper() == party.upper()]
        
        # Vulnerable seats (won in 2021, losing in 2026)
        vulnerable = [c for c in district_seats 
                     if c.winner_2021.upper() == party.upper() 
                     and c.predicted_winner_2026.upper() != party.upper()]
        
        # Potential gains (not won in 2021, winning in 2026)
        gains = [c for c in district_seats 
                if c.winner_2021.upper() != party.upper() 
                and c.predicted_winner_2026.upper() == party.upper()]
        
        answer = f"## {party} Position in {district} District\n\n"
        answer += f"**Total Seats in {district}:** {len(district_seats)}\n\n"
        
        answer += f"### Seat Count\n"
        answer += f"- **2021 Wins:** {len(party_2021)} seats\n"
        answer += f"- **2026 Predicted:** {len(party_2026)} seats\n"
        answer += f"- **Net Change:** {len(party_2026) - len(party_2021):+d}\n\n"
        
        if vulnerable:
            answer += f"### Vulnerable Seats ({len(vulnerable)})\n"
            answer += "*Seats won in 2021 but predicted to lose in 2026:*\n"
            for c in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026), reverse=True):
                answer += f"- **{c.ac_name}**: Trailing by {abs(c.predicted_margin_2026):.1f}%\n"
            answer += "\n"
        
        if gains:
            answer += f"### Potential Gains ({len(gains)})\n"
            answer += "*Seats not won in 2021 but predicted to win in 2026:*\n"
            for c in sorted(gains, key=lambda x: abs(x.predicted_margin_2026), reverse=True):
                answer += f"- **{c.ac_name}**: Leading by {abs(c.predicted_margin_2026):.1f}%\n"
            answer += "\n"
        
        # Safe seats
        safe = [c for c in party_2026 if c.race_rating.lower() == 'safe']
        if safe:
            answer += f"### Safe Seats ({len(safe)})\n"
            for c in safe[:5]:
                answer += f"- {c.ac_name}: +{abs(c.predicted_margin_2026):.1f}%\n"
            if len(safe) > 5:
                answer += f"- ... and {len(safe)-5} more\n"
        
        # Strategic recommendation
        answer += f"\n### Strategic Summary\n"
        if len(party_2026) > len(party_2021):
            answer += f"- {party} is set to **gain** {len(party_2026) - len(party_2021)} seats in {district}\n"
        elif len(party_2026) < len(party_2021):
            answer += f"- {party} is at risk of **losing** {len(party_2021) - len(party_2026)} seats in {district}\n"
        else:
            answer += f"- {party} position is **stable** in {district}\n"
        
        claims = [self.make_claim(
            statement=f"{party} holds {len(party_2021)} seats in {district} (2021), predicted to have {len(party_2026)} in 2026",
            evidence=[self.create_aggregated_evidence(f"{party} {district} analysis", district_seats)],
            reasoning="District-level aggregation of constituency data"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": f"{party} analysis for {district}", "source": "aggregated_analysis"}],
            "confidence": 0.9,
            "party": party,
            "district": district,
            "metrics": {
                "seats_2021": len(party_2021),
                "seats_2026": len(party_2026),
                "vulnerable": len(vulnerable),
                "gains": len(gains)
            }
        }
    
    def _district_strength_analysis(self, district: str, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze which party is stronger in a specific district."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Handle multi-district areas like Kolkata
        if district == 'KOLKATA_AREA':
            kolkata_districts = ['KOLKATA', 'SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS']
            district_seats = [c for c in all_seats if c.district.upper() in kolkata_districts]
            district = "Kolkata Area (South & North 24 Parganas)"
        else:
            district_seats = [c for c in all_seats if c.district.upper() == district.upper()]
        
        if not district_seats:
            return {
                "answer": f"No data found for district: {district}",
                "confidence": 0.3,
                "claims": [],
                "evidence": []
            }
        
        # Count by party (2021)
        party_2021 = defaultdict(int)
        party_2021_votes = defaultdict(float)
        for c in district_seats:
            party_2021[c.winner_2021] += 1
            party_2021_votes['TMC'] += c.tmc_vote_share_2021
            party_2021_votes['BJP'] += c.bjp_vote_share_2021
        
        # Count by party (2026 predicted)
        party_2026 = defaultdict(int)
        for c in district_seats:
            party_2026[c.predicted_winner_2026] += 1
        
        # Determine strongest party
        strongest_2021 = max(party_2021.items(), key=lambda x: x[1])
        strongest_2026 = max(party_2026.items(), key=lambda x: x[1])
        
        # Average vote share
        avg_tmc = party_2021_votes['TMC'] / len(district_seats) if district_seats else 0
        avg_bjp = party_2021_votes['BJP'] / len(district_seats) if district_seats else 0
        
        # Determine overall stronger party
        if strongest_2021[1] > len(district_seats) * 0.6:
            dominance = "strongly dominates"
        elif strongest_2021[1] > len(district_seats) * 0.5:
            dominance = "leads"
        else:
            dominance = "has a slight edge in"
        
        answer = f"## Party Strength in {district} District\n\n"
        
        # Direct answer to the question
        answer += f"### Answer: **{strongest_2021[0]}** {dominance} {district}\n\n"
        
        answer += f"**Total Seats:** {len(district_seats)}\n\n"
        
        # Seat distribution
        answer += "### Seat Distribution\n\n"
        answer += "| Party | 2021 Seats | 2026 Predicted | Change |\n"
        answer += "|-------|------------|----------------|--------|\n"
        
        all_parties = set(party_2021.keys()) | set(party_2026.keys())
        for party in sorted(all_parties, key=lambda x: -party_2021.get(x, 0)):
            s2021 = party_2021.get(party, 0)
            s2026 = party_2026.get(party, 0)
            change = s2026 - s2021
            change_str = f"+{change}" if change > 0 else str(change)
            answer += f"| {party} | {s2021} | {s2026} | {change_str} |\n"
        
        # Vote share analysis
        answer += f"\n### Average Vote Share (2021)\n"
        answer += f"- **TMC:** {avg_tmc:.1f}%\n"
        answer += f"- **BJP:** {avg_bjp:.1f}%\n"
        answer += f"- **Lead:** {'TMC' if avg_tmc > avg_bjp else 'BJP'} by {abs(avg_tmc - avg_bjp):.1f}%\n"
        
        # Competitive seats
        competitive = [c for c in district_seats if abs(c.predicted_margin_2026) < 5]
        if competitive:
            answer += f"\n### Competitive Seats in {district} ({len(competitive)})\n"
            for c in sorted(competitive, key=lambda x: abs(x.predicted_margin_2026)):
                leader = c.predicted_winner_2026
                answer += f"- **{c.ac_name}**: {leader} +{abs(c.predicted_margin_2026):.1f}%\n"
        
        # Strongholds
        tmc_safe = [c for c in district_seats if c.predicted_winner_2026 in ['TMC', 'AITC'] and c.race_rating.lower() == 'safe']
        bjp_safe = [c for c in district_seats if c.predicted_winner_2026 == 'BJP' and c.race_rating.lower() == 'safe']
        
        answer += f"\n### Strongholds\n"
        answer += f"- **TMC Safe Seats:** {len(tmc_safe)}\n"
        answer += f"- **BJP Safe Seats:** {len(bjp_safe)}\n"
        
        # Strategic insight
        answer += f"\n### Strategic Insight\n"
        if strongest_2021[0] == strongest_2026[0]:
            answer += f"- {strongest_2021[0]} maintains dominance in {district}\n"
        else:
            answer += f"- Power shift: {strongest_2021[0]} (2021) -> {strongest_2026[0]} (2026)\n"
        
        claims = [self.make_claim(
            statement=f"{strongest_2021[0]} is the stronger party in {district} with {strongest_2021[1]} of {len(district_seats)} seats in 2021",
            evidence=[self.create_aggregated_evidence(f"{district} district analysis", district_seats)],
            reasoning=f"Based on seat count and vote share analysis in {district} district"
        )]
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "district": district,
            "stronger_party": strongest_2021[0],
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": f"District strength analysis for {district}", "source": "Electoral data"}]
        }
    
    def _overall_strength_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze overall party strength in West Bengal."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Count by party
        tmc_2021 = sum(1 for c in all_seats if c.winner_2021 in ['TMC', 'AITC'])
        bjp_2021 = sum(1 for c in all_seats if c.winner_2021 == 'BJP')
        tmc_2026 = sum(1 for c in all_seats if c.predicted_winner_2026 in ['TMC', 'AITC'])
        bjp_2026 = sum(1 for c in all_seats if c.predicted_winner_2026 == 'BJP')
        
        stronger = "TMC" if tmc_2021 > bjp_2021 else "BJP"
        
        answer = f"## Party Strength in West Bengal\n\n"
        answer += f"### Answer: **{stronger}** is the stronger party in West Bengal\n\n"
        
        answer += "### Seat Comparison\n\n"
        answer += "| Party | 2021 Seats | 2026 Predicted | Change |\n"
        answer += "|-------|------------|----------------|--------|\n"
        answer += f"| TMC | {tmc_2021} | {tmc_2026} | {tmc_2026 - tmc_2021:+d} |\n"
        answer += f"| BJP | {bjp_2021} | {bjp_2026} | {bjp_2026 - bjp_2021:+d} |\n"
        
        answer += f"\n### Dominance Ratio\n"
        answer += f"- TMC controls **{tmc_2021/len(all_seats)*100:.1f}%** of seats (2021)\n"
        answer += f"- BJP controls **{bjp_2021/len(all_seats)*100:.1f}%** of seats (2021)\n"
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "stronger_party": stronger,
            "claims": [],
            "evidence": [{"content": "Overall strength analysis", "source": "Electoral data"}]
        }
    
    def _district_analysis(self, district: str, query: str, context: Dict) -> Dict[str, Any]:
        """Analyze a specific district."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Handle multi-district areas like Kolkata
        if district == 'KOLKATA_AREA':
            kolkata_districts = ['KOLKATA', 'SOUTH 24 PARGANAS', 'NORTH 24 PARGANAS']
            district_seats = [c for c in all_seats if c.district.upper() in kolkata_districts]
            district = "Kolkata Area"
        else:
            district_seats = [c for c in all_seats if c.district.upper() == district.upper()]
        
        if not district_seats:
            available = self._get_available_districts()
            return {
                "answer": f"No data found for district: {district}\n\n**Available districts:** {', '.join(available[:10])}{'...' if len(available) > 10 else ''}",
                "confidence": 0.3,
                "claims": [],
                "evidence": []
            }
        
        # Count by party (2021)
        party_2021 = defaultdict(int)
        for c in district_seats:
            party_2021[c.winner_2021] += 1
        
        # Count by party (2026 predicted)
        party_2026 = defaultdict(int)
        for c in district_seats:
            party_2026[c.predicted_winner_2026] += 1
        
        answer = f"## {district} District Analysis\n\n"
        answer += f"**Total Seats:** {len(district_seats)}\n\n"
        
        answer += "### 2021 Results\n"
        for party, count in sorted(party_2021.items(), key=lambda x: -x[1]):
            answer += f"- **{party}:** {count} seats\n"
        
        answer += "\n### 2026 Predictions\n"
        for party, count in sorted(party_2026.items(), key=lambda x: -x[1]):
            answer += f"- **{party}:** {count} seats\n"
        
        answer += "\n### Constituencies\n"
        for c in sorted(district_seats, key=lambda x: x.ac_name):
            answer += f"- {c.ac_name}: {c.winner_2021} (2021) -> {c.predicted_winner_2026} (2026)\n"
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "district": district,
            "claims": [],
            "evidence": [{"content": f"District analysis for {district}", "source": "Electoral data"}]
        }
    
    def _comprehensive_party_analysis(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive party analysis."""
        
        self.add_reasoning_step(
            action=f"Analyze {party} Electoral Position",
            input_data=query,
            output=f"Gathering data for {party}"
        )
        
        # Get seats
        seats_2021 = self.get_party_seats(party)
        vulnerable = self.get_vulnerable_seats(party)
        
        # Calculate predictions
        all_seats = list(self.kg.constituency_profiles.values())
        predicted_2026 = [c for c in all_seats if c.predicted_winner_2026.upper() == party.upper()]
        
        # For TMC, also check AITC
        if party == 'TMC':
            predicted_2026.extend([c for c in all_seats if c.predicted_winner_2026.upper() == 'AITC'])
            seats_2021.extend([c for c in all_seats if c.winner_2021.upper() == 'AITC'])
        
        # Remove duplicates
        seats_2021 = list({c.ac_name: c for c in seats_2021}.values())
        predicted_2026 = list({c.ac_name: c for c in predicted_2026}.values())
        
        claims = []
        evidence = []
        
        # Current position claim
        ev_current = self.create_aggregated_evidence(
            f"{party} won {len(seats_2021)} seats in 2021",
            seats_2021
        )
        evidence.append(ev_current)
        
        claims.append(self.make_claim(
            statement=f"{party} won {len(seats_2021)} seats in the 2021 Assembly election",
            evidence=[ev_current],
            reasoning="Direct count from 2021 electoral data"
        ))
        
        # Prediction claim
        ev_pred = self.create_aggregated_evidence(
            f"{party} predicted {len(predicted_2026)} seats in 2026",
            predicted_2026
        )
        evidence.append(ev_pred)
        
        change = len(predicted_2026) - len(seats_2021)
        direction = "gain" if change > 0 else "lose"
        
        claims.append(self.make_claim(
            statement=f"{party} is predicted to win {len(predicted_2026)} seats in 2026, a {direction} of {abs(change)} seats",
            evidence=[ev_pred],
            reasoning="Based on swing analysis and prediction model"
        ))
        
        # Vulnerability claim
        if vulnerable:
            ev_vuln = self.create_aggregated_evidence(
                f"{len(vulnerable)} {party} seats at risk",
                vulnerable
            )
            evidence.append(ev_vuln)
            
            claims.append(self.make_claim(
                statement=f"{party} has {len(vulnerable)} seats at risk of flipping to the opposition",
                evidence=[ev_vuln],
                reasoning="Seats won in 2021 but predicted to lose in 2026"
            ))
        
        self.add_reasoning_step(
            action="Calculate Electoral Metrics",
            input_data="Electoral data",
            output=f"2021: {len(seats_2021)} seats, 2026 predicted: {len(predicted_2026)} seats",
            claims=claims
        )
        
        # District breakdown
        by_district = defaultdict(lambda: {"2021": 0, "2026": 0})
        for c in seats_2021:
            by_district[c.district]["2021"] += 1
        for c in predicted_2026:
            by_district[c.district]["2026"] += 1
        
        # Build answer
        answer = f"## {party} Electoral Analysis\n\n"
        
        # Summary
        answer += "### Summary\n"
        answer += f"- **2021 Seats:** {len(seats_2021)}\n"
        answer += f"- **2026 Predicted:** {len(predicted_2026)}\n"
        answer += f"- **Net Change:** {'+' if change > 0 else ''}{change} seats\n"
        answer += f"- **Vulnerable Seats:** {len(vulnerable)}\n\n"
        
        # District Performance
        answer += "### District-wise Breakdown\n\n"
        answer += "| District | 2021 | 2026 Pred | Change |\n"
        answer += "|----------|------|-----------|--------|\n"
        
        for dist in sorted(by_district.keys()):
            data = by_district[dist]
            chg = data["2026"] - data["2021"]
            chg_str = f"+{chg}" if chg > 0 else str(chg)
            answer += f"| {dist} | {data['2021']} | {data['2026']} | {chg_str} |\n"
        
        # Vulnerable seats
        if vulnerable:
            answer += "\n### Vulnerable Seats (At Risk)\n"
            for seat in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026), reverse=True)[:10]:
                answer += f"- **{seat.ac_name}** ({seat.district}): "
                answer += f"Lost by {abs(seat.predicted_margin_2026):.1f}% predicted margin\n"
        
        # Gains
        gains = [c for c in predicted_2026 if c.ac_name not in [s.ac_name for s in seats_2021]]
        if gains:
            answer += "\n### Potential Gains\n"
            for seat in sorted(gains, key=lambda x: abs(x.predicted_margin_2026), reverse=True)[:10]:
                answer += f"- **{seat.ac_name}** ({seat.district}): "
                answer += f"Winning by {abs(seat.predicted_margin_2026):.1f}% predicted\n"
        
        # Strategic recommendations
        answer += "\n### Strategic Recommendations\n"
        if change > 0:
            answer += f"- {party} is positioned for gains; focus on consolidating leads in close races\n"
        else:
            answer += f"- {party} faces headwinds; defensive strategy in vulnerable seats is critical\n"
        
        if len(vulnerable) > 10:
            answer += f"- High vulnerability ({len(vulnerable)} seats at risk) requires intensive ground game\n"
        
        swing_seats = [c for c in predicted_2026 if c.race_rating.lower() in ['toss-up', 'lean']]
        if swing_seats:
            answer += f"- Focus resources on {len(swing_seats)} swing seats for maximum impact\n"
        
        reasoning_chain = self.get_reasoning_chain()
        reasoning_chain.question = query
        reasoning_chain.final_answer = answer
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": e.content, "source": e.source} for e in evidence],
            "confidence": reasoning_chain.average_confidence,
            "reasoning": reasoning_chain,
            "party": party,
            "metrics": {
                "seats_2021": len(seats_2021),
                "seats_2026_predicted": len(predicted_2026),
                "change": change,
                "vulnerable": len(vulnerable)
            }
        }
    
    def _victory_path_analysis(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate path to victory."""
        
        target_seats = context.get('target_seats', 148)  # Majority in 294-seat assembly
        
        self.add_reasoning_step(
            action=f"Calculate Victory Path for {party}",
            input_data=f"Target: {target_seats} seats",
            output="Analyzing seat categories"
        )
        
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Categorize seats
        safe_seats = [c for c in all_seats 
                     if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
                     and c.race_rating.lower() == 'safe']
        
        likely_seats = [c for c in all_seats 
                       if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
                       and c.race_rating.lower() == 'likely']
        
        lean_seats = [c for c in all_seats 
                     if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
                     and c.race_rating.lower() == 'lean']
        
        tossup_seats = [c for c in all_seats 
                       if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
                       and c.race_rating.lower() == 'toss-up']
        
        # Opponent's vulnerable seats (potential flips)
        opp_party = 'TMC' if party == 'BJP' else 'BJP'
        potential_flips = [c for c in all_seats
                         if c.winner_2021.upper() in [opp_party.upper(), 'AITC' if opp_party == 'TMC' else '']
                         and c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        claims = []
        
        # Current prediction
        current_predicted = len(safe_seats) + len(likely_seats) + len(lean_seats) + len(tossup_seats)
        
        claims.append(self.make_claim(
            statement=f"Based on current projections, {party} is predicted to win {current_predicted} seats",
            evidence=[self.create_aggregated_evidence(f"{party} seat prediction", 
                                                      safe_seats + likely_seats + lean_seats + tossup_seats)],
            reasoning="Sum of seats across all rating categories"
        ))
        
        # Victory path calculation
        shortfall = target_seats - current_predicted
        
        if shortfall <= 0:
            path_status = "ON TRACK"
            path_message = f"{party} is projected to exceed the target of {target_seats} seats"
        else:
            path_status = "NEEDS GAINS"
            path_message = f"{party} needs {shortfall} additional seats beyond current projection to reach {target_seats}"
        
        claims.append(self.make_claim(
            statement=f"Victory Path Status: {path_status}. {path_message}",
            evidence=[],
            reasoning=f"Target: {target_seats}, Projected: {current_predicted}"
        ))
        
        self.add_reasoning_step(
            action="Calculate Victory Path",
            input_data=f"Target: {target_seats}",
            output=f"Status: {path_status}",
            claims=claims
        )
        
        # Build answer
        answer = f"## {party} Victory Path Analysis\n\n"
        answer += f"**Target:** {target_seats} seats (majority)\n"
        answer += f"**Currently Projected:** {current_predicted} seats\n"
        answer += f"**Status:** {path_status}\n\n"
        
        answer += "### Seat Distribution by Confidence\n\n"
        answer += f"| Category | Seats | Examples |\n"
        answer += f"|----------|-------|----------|\n"
        answer += f"| Safe | {len(safe_seats)} | {', '.join(c.ac_name for c in safe_seats[:3])}... |\n"
        answer += f"| Likely | {len(likely_seats)} | {', '.join(c.ac_name for c in likely_seats[:3])}... |\n"
        answer += f"| Lean | {len(lean_seats)} | {', '.join(c.ac_name for c in lean_seats[:3])}... |\n"
        answer += f"| Toss-up | {len(tossup_seats)} | {', '.join(c.ac_name for c in tossup_seats[:3])}... |\n"
        
        answer += f"\n### Pathway Strategy\n\n"
        
        if shortfall > 0:
            answer += f"To reach {target_seats} seats, {party} must:\n\n"
            answer += f"1. **Secure all predicted seats** ({current_predicted})\n"
            answer += f"2. **Convert toss-ups** (currently have {len(tossup_seats)})\n"
            answer += f"3. **Flip opponent seats** (potential: {len(potential_flips)})\n\n"
            
            answer += "**Priority Flip Targets:**\n"
            for c in sorted(potential_flips, key=lambda x: abs(x.predicted_margin_2026))[:10]:
                answer += f"- {c.ac_name} ({c.district}): +{abs(c.predicted_margin_2026):.1f}%\n"
        else:
            answer += f"{party} is on track to exceed target. Focus on:\n\n"
            answer += f"1. **Defend vulnerable seats** to prevent erosion\n"
            answer += f"2. **Maximize margin** in safe seats for buffer\n"
            answer += f"3. **Push in lean seats** to secure majority\n"
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "confidence": 0.85,
            "party": party,
            "victory_path": {
                "target": target_seats,
                "projected": current_predicted,
                "shortfall": max(0, shortfall),
                "status": path_status,
                "safe": len(safe_seats),
                "likely": len(likely_seats),
                "lean": len(lean_seats),
                "tossup": len(tossup_seats),
                "potential_flips": len(potential_flips)
            }
        }
    
    def _strategic_recommendations(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive strategic recommendations and action points for a party.
        This method provides detailed, actionable strategies based on electoral data.
        """
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Gather data for analysis
        seats_2021 = [c for c in all_seats if c.winner_2021.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        predicted_2026 = [c for c in all_seats if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        vulnerable = [c for c in seats_2021 if c.predicted_winner_2026.upper() not in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        # Opposition analysis
        opp_party = 'TMC' if party == 'BJP' else 'BJP'
        opp_vulnerable = [c for c in all_seats 
                        if c.winner_2021.upper() in [opp_party.upper(), 'AITC' if opp_party == 'TMC' else '']
                        and c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        # Swing seats
        swing_seats = [c for c in all_seats if (c.race_rating or '').lower() in ['toss-up', 'lean']]
        party_swing = [c for c in swing_seats if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        # Districts with gains/losses
        district_change = {}
        for c in all_seats:
            d = c.district
            if d not in district_change:
                district_change[d] = {'2021': 0, '2026': 0, 'vulnerable': 0, 'gains': 0}
            if c.winner_2021.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']:
                district_change[d]['2021'] += 1
            if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']:
                district_change[d]['2026'] += 1
            if c in vulnerable:
                district_change[d]['vulnerable'] += 1
            if c in opp_vulnerable:
                district_change[d]['gains'] += 1
        
        # Identify priority districts
        declining_districts = [(d, v) for d, v in district_change.items() if v['2026'] < v['2021']]
        growing_districts = [(d, v) for d, v in district_change.items() if v['2026'] > v['2021']]
        
        # Build comprehensive strategic recommendations
        change = len(predicted_2026) - len(seats_2021)
        
        answer = f"## {party} Strategic Decision Framework & Action Points\n\n"
        
        answer += "### Executive Summary\n"
        answer += f"- **Current Position:** {len(seats_2021)} seats (2021)\n"
        answer += f"- **Projected Position:** {len(predicted_2026)} seats (2026)\n"
        answer += f"- **Net Trajectory:** {'+' if change >= 0 else ''}{change} seats\n"
        answer += f"- **Vulnerable Seats:** {len(vulnerable)} (at risk)\n"
        answer += f"- **Potential Gains:** {len(opp_vulnerable)} (from {opp_party})\n\n"
        
        # IMMEDIATE ACTION POINTS (0-30 Days)
        answer += "---\n\n## 🚨 IMMEDIATE ACTION POINTS (0-30 Days)\n\n"
        
        answer += "### 1. Seat Defense Emergency Measures\n"
        if vulnerable:
            answer += f"**{len(vulnerable)} seats require immediate intervention:**\n\n"
            high_priority = sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]
            for c in high_priority:
                answer += f"- **{c.ac_name}** ({c.district}): Deploy senior leader visits, intensify booth management\n"
            answer += "\n**Actions:**\n"
            answer += "- Conduct emergency survey in each vulnerable constituency\n"
            answer += "- Identify and address local grievances within 2 weeks\n"
            answer += "- Deploy dedicated campaign manager for each vulnerable seat\n"
        else:
            answer += "No immediate seat defense emergencies identified.\n"
        
        answer += "\n### 2. Candidate Selection & Finalization\n"
        answer += "- Finalize candidates for all contested seats within 30 days\n"
        answer += "- Prioritize candidates with strong local networks in swing seats\n"
        answer += "- Ensure candidate announcements are staggered for media impact\n"
        
        answer += "\n### 3. Opposition Research\n"
        answer += f"- Complete dossiers on all {opp_party} sitting MLAs in target seats\n"
        answer += "- Identify anti-incumbency narratives in vulnerable opposition seats\n"
        answer += "- Document development failures in opposition-held districts\n"
        
        # SHORT-TERM STRATEGY (1-3 Months)
        answer += "\n---\n\n## 📋 SHORT-TERM STRATEGY (1-3 Months)\n\n"
        
        answer += "### 1. District-Level Focus Areas\n\n"
        
        if growing_districts:
            answer += "**Growth Districts (Double Down):**\n"
            for d, v in sorted(growing_districts, key=lambda x: x[1]['2026'] - x[1]['2021'], reverse=True)[:5]:
                gain = v['2026'] - v['2021']
                answer += f"- **{d}**: +{gain} seats projected - Increase resource allocation by 50%\n"
            answer += "\n"
        
        if declining_districts:
            answer += "**Declining Districts (Defensive Focus):**\n"
            for d, v in sorted(declining_districts, key=lambda x: x[1]['2026'] - x[1]['2021'])[:5]:
                loss = v['2021'] - v['2026']
                answer += f"- **{d}**: -{loss} seats at risk - Investigate causes, deploy damage control\n"
            answer += "\n"
        
        answer += "### 2. Swing Seat Capture Strategy\n"
        answer += f"**Total Swing Seats in Play:** {len(swing_seats)}\n"
        answer += f"**Currently Projected for {party}:** {len(party_swing)}\n\n"
        answer += "**Actions:**\n"
        answer += "- Deploy strongest candidates in toss-up races\n"
        answer += "- Allocate 3x normal campaign budget to swing seats\n"
        answer += "- Implement micro-targeting based on voter segmentation\n"
        
        answer += "\n### 3. Ground Organization\n"
        answer += "- Activate booth-level committees in all priority constituencies\n"
        answer += "- Train 50+ volunteers per constituency for swing seats\n"
        answer += "- Establish war rooms in each district headquarters\n"
        
        # MEDIUM-TERM STRATEGY (3-6 Months)
        answer += "\n---\n\n## 📊 MEDIUM-TERM STRATEGY (3-6 Months)\n\n"
        
        answer += "### 1. Voter Outreach Programs\n"
        answer += "- Launch door-to-door campaigns in all 294 constituencies\n"
        answer += "- Organize 5+ public rallies in each swing district\n"
        answer += "- Implement targeted social media campaigns by constituency\n"
        
        answer += "\n### 2. Coalition & Alliance Strategy\n"
        if party == 'BJP':
            answer += "- Explore strategic understanding with smaller parties\n"
            answer += "- Coordinate with NDA allies for vote transfer\n"
            answer += "- Avoid triangular contests in swing seats\n"
        else:
            answer += "- Consolidate alliance partnerships\n"
            answer += "- Ensure seat-sharing agreements are finalized early\n"
            answer += "- Build coalition ground coordination\n"
        
        answer += "\n### 3. Issue-Based Campaigning\n"
        if party == 'BJP':
            answer += "**Recommended Themes:**\n"
            answer += "- Law and order / Security\n"
            answer += "- Central government scheme benefits\n"
            answer += "- Anti-corruption narrative\n"
            answer += "- Development and infrastructure\n"
        else:
            answer += "**Recommended Themes:**\n"
            answer += "- State welfare scheme successes\n"
            answer += "- Regional pride and identity\n"
            answer += "- Anti-central interference\n"
            answer += "- Women empowerment programs\n"
        
        # SPECIFIC CONSTITUENCY ACTIONS
        answer += "\n---\n\n## 🎯 SPECIFIC CONSTITUENCY ACTION PLAN\n\n"
        
        answer += "### Must-Win Constituencies (Top 10 Priority)\n\n"
        answer += "| # | Constituency | District | Action Required | Priority |\n"
        answer += "|---|--------------|----------|-----------------|----------|\n"
        
        # Mix of defense and offense
        priority_list = []
        for c in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            priority_list.append((c, "DEFEND", "Critical"))
        for c in sorted(opp_vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            priority_list.append((c, "ATTACK", "High"))
        
        for i, (c, action, priority) in enumerate(priority_list[:10], 1):
            answer += f"| {i} | {c.ac_name} | {c.district} | {action} | {priority} |\n"
        
        # RESOURCE ALLOCATION
        answer += "\n---\n\n## 💰 RECOMMENDED RESOURCE ALLOCATION\n\n"
        answer += "| Category | Allocation % | Rationale |\n"
        answer += "|----------|-------------|------------|\n"
        answer += f"| Vulnerable Seat Defense | 35% | Protect {len(vulnerable)} at-risk seats |\n"
        answer += f"| Swing Seat Capture | 30% | Contest {len(party_swing)} winnable seats |\n"
        answer += f"| Opposition Flips | 20% | Target {len(opp_vulnerable)} weak {opp_party} seats |\n"
        answer += "| Safe Seat Maintenance | 10% | Ensure turnout in strongholds |\n"
        answer += "| Statewide Campaign | 5% | Brand building and media |\n"
        
        # KEY PERFORMANCE INDICATORS
        answer += "\n---\n\n## 📈 KEY PERFORMANCE INDICATORS\n\n"
        answer += "Track these metrics weekly:\n\n"
        answer += "1. **Booth Coverage:** % of booths with active committees\n"
        answer += "2. **Candidate Strength:** Ground surveys in swing seats\n"
        answer += "3. **Media Sentiment:** Coverage ratio vs opposition\n"
        answer += "4. **Rally Attendance:** Average footfall trends\n"
        answer += "5. **Social Media Engagement:** Constituency-wise reach\n"
        
        # FINAL VERDICT
        answer += "\n---\n\n## 🎯 BOTTOM LINE\n\n"
        if change > 0:
            answer += f"**{party} is positioned for gains.** Focus on:\n"
            answer += f"1. Converting projected gains into actual wins\n"
            answer += f"2. Preventing complacency in growing districts\n"
            answer += f"3. Aggressive campaigning in {len(party_swing)} swing seats\n"
        elif change < 0:
            answer += f"**{party} faces significant challenges.** Immediate priorities:\n"
            answer += f"1. Emergency intervention in {len(vulnerable)} vulnerable seats\n"
            answer += f"2. Candidate refresh in declining districts\n"
            answer += f"3. Targeted messaging to address vote erosion\n"
            answer += f"4. Focus resources on {len(party_swing)} defensible swing seats\n"
        else:
            answer += f"**{party} is holding steady.** Strategy:\n"
            answer += f"1. Maintain current position while seeking targeted gains\n"
            answer += f"2. Focus on swing seats for marginal improvement\n"
        
        # Generate detailed, query-specific citations
        evidence_list = []
        
        # Citation 1: 2021 Assembly Election Results
        evidence_list.append({
            "content": f"{party} won {len(seats_2021)} seats in West Bengal Assembly Election 2021. "
                      f"Data includes vote share, margin, and winning candidate for each constituency.",
            "source": "West Bengal Assembly Election Results 2021",
            "source_type": "Electoral Results",
            "constituencies_cited": [c.ac_name for c in seats_2021[:5]],
            "score": 0.95
        })
        
        # Citation 2: 2026 Predictions based on swing analysis
        evidence_list.append({
            "content": f"2026 prediction model projects {len(predicted_2026)} seats for {party}. "
                      f"Based on Lok Sabha 2019-2024 swing analysis applied to assembly segments. "
                      f"Net change: {'+' if change >= 0 else ''}{change} seats.",
            "source": "2026 Prediction Model (Lok Sabha Swing Analysis)",
            "source_type": "Prediction Model",
            "methodology": "Vote swing from LS 2019 to LS 2024 applied to AC segments",
            "score": 0.88
        })
        
        # Citation 3: Vulnerable Seats Analysis
        if vulnerable:
            vuln_names = [c.ac_name for c in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]]
            evidence_list.append({
                "content": f"{len(vulnerable)} {party} seats from 2021 are projected to flip in 2026. "
                          f"Most vulnerable: {', '.join(vuln_names)}. "
                          f"Analysis based on local swing trends and predicted margins.",
                "source": "Vulnerability Analysis - Seat Flip Projections",
                "source_type": "Analytical",
                "constituencies_cited": vuln_names,
                "score": 0.85
            })
        
        # Citation 4: Potential Gains from Opposition
        if opp_vulnerable:
            gain_names = [c.ac_name for c in sorted(opp_vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]]
            evidence_list.append({
                "content": f"{len(opp_vulnerable)} {opp_party} seats from 2021 may flip to {party} in 2026. "
                          f"Top targets: {', '.join(gain_names)}. "
                          f"Based on anti-incumbency and swing momentum.",
                "source": f"Opposition Vulnerability Analysis - {opp_party} Weak Seats",
                "source_type": "Analytical",
                "constituencies_cited": gain_names,
                "score": 0.82
            })
        
        # Citation 5: District-Level Analysis
        if declining_districts:
            decline_names = [d for d, v in sorted(declining_districts, key=lambda x: x[1]['2026'] - x[1]['2021'])[:3]]
            evidence_list.append({
                "content": f"Districts showing {party} decline: {', '.join(decline_names)}. "
                          f"Combined loss: {sum(v['2021'] - v['2026'] for d, v in declining_districts)} seats. "
                          f"Analysis based on 2021 results vs 2026 projections.",
                "source": "District-wise Seat Change Analysis",
                "source_type": "District Analysis",
                "districts_cited": decline_names,
                "score": 0.87
            })
        
        # Citation 6: Swing Seat Analysis
        if party_swing:
            swing_names = [c.ac_name for c in sorted(party_swing, key=lambda x: abs(x.predicted_margin_2026))[:5]]
            evidence_list.append({
                "content": f"{len(party_swing)} swing seats projected for {party}. "
                          f"Closest races: {', '.join(swing_names)}. "
                          f"Defined as seats with margin <5% or rated 'Toss-up'/'Lean'.",
                "source": "Swing Seat Classification Model",
                "source_type": "Competitive Analysis",
                "constituencies_cited": swing_names,
                "score": 0.83
            })
        
        # Citation 7: Lok Sabha Trend Data
        evidence_list.append({
            "content": f"Lok Sabha vote share trends 2019-2024 used for swing calculation. "
                      f"State-level swing and PC-wise constituency mapping applied. "
                      f"Total 294 assembly constituencies analyzed across 42 Lok Sabha segments.",
            "source": "Lok Sabha Election Data 2019 & 2024",
            "source_type": "Electoral Results",
            "data_points": "Vote share by party, constituency-wise results",
            "score": 0.92
        })
        
        # Generate detailed claims with evidence references
        claims = [
            {
                "statement": f"{party} won {len(seats_2021)} seats in 2021 Assembly election",
                "confidence": "high",
                "evidence_ref": 1,
                "verification": "Direct from electoral records"
            },
            {
                "statement": f"{party} is projected to win {len(predicted_2026)} seats in 2026 ({'+' if change >= 0 else ''}{change} from 2021)",
                "confidence": "high",
                "evidence_ref": 2,
                "verification": "Swing analysis model"
            },
            {
                "statement": f"{len(vulnerable)} current {party} seats are at high risk of flipping",
                "confidence": "high",
                "evidence_ref": 3,
                "verification": "Margin and swing analysis"
            },
            {
                "statement": f"{party} can potentially gain {len(opp_vulnerable)} seats from {opp_party}",
                "confidence": "medium",
                "evidence_ref": 4,
                "verification": "Opposition vulnerability analysis"
            },
            {
                "statement": f"{len(party_swing)} seats are in 'swing' category requiring focused attention",
                "confidence": "high",
                "evidence_ref": 6,
                "verification": "Competitive seat classification"
            }
        ]
        
        return {
            "answer": answer,
            "confidence": 0.92,
            "claims": claims,
            "evidence": evidence_list,
            "party": party,
            "strategic_summary": {
                "current_seats": len(seats_2021),
                "projected_seats": len(predicted_2026),
                "change": change,
                "vulnerable": len(vulnerable),
                "potential_gains": len(opp_vulnerable),
                "swing_seats": len(party_swing)
            },
            "data_sources": [
                "West Bengal Assembly Election 2021",
                "Lok Sabha Election 2019",
                "Lok Sabha Election 2024",
                "2026 Prediction Model",
                "Constituency Profile Database"
            ]
        }
    
    def _resource_allocation(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resource allocation recommendations."""
        
        all_seats = list(self.kg.constituency_profiles.values())
        party_seats = [c for c in all_seats 
                      if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        # Categorize by priority
        tier1_defensive = []  # Vulnerable seats we must defend
        tier1_offensive = []  # Opponent's seats we can flip
        tier2_consolidate = []  # Close races we're winning
        tier3_maintain = []  # Safe seats
        
        for c in all_seats:
            is_our_2021 = c.winner_2021.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
            is_our_2026 = c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
            
            if is_our_2021 and not is_our_2026:
                tier1_defensive.append(c)
            elif not is_our_2021 and is_our_2026:
                tier1_offensive.append(c)
            elif is_our_2026 and c.race_rating.lower() in ['toss-up', 'lean']:
                tier2_consolidate.append(c)
            elif is_our_2026 and c.race_rating.lower() in ['safe', 'likely']:
                tier3_maintain.append(c)
        
        answer = f"## {party} Resource Allocation Strategy\n\n"
        
        answer += "### Tier 1: Critical (Highest Priority)\n"
        answer += f"*Allocate 50% of resources*\n\n"
        
        answer += f"**Defensive ({len(tier1_defensive)} seats)** - Prevent losses\n"
        for c in sorted(tier1_defensive, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            answer += f"- {c.ac_name}: 2021 won, now trailing by {abs(c.predicted_margin_2026):.1f}%\n"
        
        answer += f"\n**Offensive ({len(tier1_offensive)} seats)** - Capture from opponent\n"
        for c in sorted(tier1_offensive, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            answer += f"- {c.ac_name}: Flipping with {abs(c.predicted_margin_2026):.1f}% lead\n"
        
        answer += f"\n### Tier 2: Important ({len(tier2_consolidate)} seats)\n"
        answer += f"*Allocate 35% of resources*\n\n"
        answer += "Consolidate close leads:\n"
        for c in sorted(tier2_consolidate, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            answer += f"- {c.ac_name}: Leading by only {abs(c.predicted_margin_2026):.1f}%\n"
        
        answer += f"\n### Tier 3: Maintenance ({len(tier3_maintain)} seats)\n"
        answer += f"*Allocate 15% of resources*\n\n"
        answer += f"Safe seats requiring minimal investment: {len(tier3_maintain)} seats\n"
        
        # ROI calculation
        answer += "\n### Return on Investment Analysis\n\n"
        answer += "| Priority | Seats | Impact if Won | Impact if Lost |\n"
        answer += "|----------|-------|---------------|----------------|\n"
        answer += f"| Tier 1 Defensive | {len(tier1_defensive)} | Prevent -{len(tier1_defensive)} seats | -{len(tier1_defensive)} |\n"
        answer += f"| Tier 1 Offensive | {len(tier1_offensive)} | +{len(tier1_offensive)} seats | +0 |\n"
        answer += f"| Tier 2 | {len(tier2_consolidate)} | +{len(tier2_consolidate)} seats | Potential loss |\n"
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "party": party,
            "allocation": {
                "tier1_defensive": len(tier1_defensive),
                "tier1_offensive": len(tier1_offensive),
                "tier2": len(tier2_consolidate),
                "tier3": len(tier3_maintain)
            }
        }
    
    def _vulnerability_analysis(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze party vulnerabilities."""
        
        vulnerable = self.get_vulnerable_seats(party)
        
        answer = f"## {party} Vulnerability Analysis\n\n"
        answer += f"**Total Vulnerable Seats:** {len(vulnerable)}\n\n"
        
        if not vulnerable:
            answer += f"No vulnerable seats identified for {party}.\n"
            return {"answer": answer, "confidence": 0.9, "party": party}
        
        # Group by district
        by_district = defaultdict(list)
        for c in vulnerable:
            by_district[c.district].append(c)
        
        answer += "### By District\n"
        for dist, seats in sorted(by_district.items(), key=lambda x: -len(x[1])):
            answer += f"\n**{dist}** ({len(seats)} seats):\n"
            for c in seats:
                answer += f"- {c.ac_name}: {abs(c.predicted_margin_2026):.1f}% behind\n"
        
        # Most vulnerable
        answer += "\n### Most Vulnerable (Largest Deficits)\n"
        for c in sorted(vulnerable, key=lambda x: -abs(x.predicted_margin_2026))[:10]:
            answer += f"- **{c.ac_name}**: {abs(c.predicted_margin_2026):.1f}% behind "
            answer += f"(2021 margin: {abs(c.margin_2021):.1f}%)\n"
        
        claims = [self.make_claim(
            statement=f"{party} has {len(vulnerable)} vulnerable seats, primarily concentrated in {list(by_district.keys())[:3]}",
            evidence=[self.create_aggregated_evidence(f"{party} vulnerability", vulnerable)],
            reasoning="Seats won in 2021 but predicted to lose in 2026"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "confidence": 0.9,
            "party": party,
            "vulnerable_count": len(vulnerable)
        }
    
    def _strength_analysis(self, party: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze party strengths."""
        
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Safe seats
        safe = [c for c in all_seats 
               if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
               and c.race_rating.lower() == 'safe']
        
        # High margin seats
        high_margin = [c for c in all_seats 
                      if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
                      and abs(c.predicted_margin_2026) > 15]
        
        answer = f"## {party} Strength Analysis\n\n"
        answer += f"**Safe Seats:** {len(safe)}\n"
        answer += f"**High Margin (>15%):** {len(high_margin)}\n\n"
        
        # Group by district
        by_district = defaultdict(list)
        for c in safe:
            by_district[c.district].append(c)
        
        answer += "### Stronghold Districts\n"
        for dist, seats in sorted(by_district.items(), key=lambda x: -len(x[1]))[:10]:
            answer += f"- **{dist}**: {len(seats)} safe seats\n"
        
        answer += "\n### Fortress Seats (Highest Margins)\n"
        for c in sorted(high_margin, key=lambda x: abs(x.predicted_margin_2026), reverse=True)[:10]:
            answer += f"- **{c.ac_name}** ({c.district}): +{abs(c.predicted_margin_2026):.1f}%\n"
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "party": party,
            "safe_count": len(safe),
            "high_margin_count": len(high_margin)
        }
    
    def _overall_electoral_analysis(self, query: str) -> Dict[str, Any]:
        """General electoral landscape analysis."""
        
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Count by party (2021)
        party_2021 = defaultdict(int)
        for c in all_seats:
            party_2021[c.winner_2021] += 1
        
        # Count by party (2026 predicted)
        party_2026 = defaultdict(int)
        for c in all_seats:
            party_2026[c.predicted_winner_2026] += 1
        
        # Normalize TMC/AITC
        if 'AITC' in party_2021:
            party_2021['TMC'] = party_2021.get('TMC', 0) + party_2021.pop('AITC')
        if 'AITC' in party_2026:
            party_2026['TMC'] = party_2026.get('TMC', 0) + party_2026.pop('AITC')
        
        answer = "## West Bengal 2026 Electoral Landscape\n\n"
        
        answer += "### Seat Distribution\n\n"
        answer += "| Party | 2021 | 2026 Pred | Change |\n"
        answer += "|-------|------|-----------|--------|\n"
        
        for party in ['TMC', 'BJP', 'INC', 'CPM']:
            s21 = party_2021.get(party, 0)
            s26 = party_2026.get(party, 0)
            chg = s26 - s21
            chg_str = f"+{chg}" if chg > 0 else str(chg)
            answer += f"| {party} | {s21} | {s26} | {chg_str} |\n"
        
        # Race ratings
        ratings = defaultdict(int)
        for c in all_seats:
            ratings[c.race_rating] += 1
        
        answer += "\n### Race Ratings\n"
        for rating in ['Safe', 'Likely', 'Lean', 'Toss-up']:
            answer += f"- **{rating}:** {ratings.get(rating, 0)} seats\n"
        
        # Key battlegrounds
        swing = self.get_swing_seats(5.0)
        answer += f"\n### Key Battlegrounds\n"
        answer += f"**Total Swing Seats (margin <5%):** {len(swing)}\n\n"
        
        for c in sorted(swing, key=lambda x: abs(x.predicted_margin_2026))[:10]:
            answer += f"- {c.ac_name}: {c.predicted_winner_2026} +{abs(c.predicted_margin_2026):.1f}%\n"
        
        return {
            "answer": answer,
            "confidence": 0.9,
            "party_2021": dict(party_2021),
            "party_2026": dict(party_2026),
            "swing_seats": len(swing)
        }

