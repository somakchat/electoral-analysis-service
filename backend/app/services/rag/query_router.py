"""
Query Router - Intelligent routing for structured vs unstructured queries.

This module determines the optimal retrieval strategy for each query:
1. Structured queries → Direct knowledge graph lookup
2. Aggregation queries → KG aggregation functions
3. Comparison queries → Multi-entity retrieval
4. Free-form queries → Hybrid search with verification
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import re

from .data_schema import VerifiedAnswer, FactWithCitation
from .knowledge_graph import PoliticalKnowledgeGraph
from .verified_retrieval import VerifiedRetriever, QueryAnalyzer, RetrievalResult


class QueryRoute(str, Enum):
    """Types of query routes."""
    CONSTITUENCY_LOOKUP = "constituency_lookup"
    DISTRICT_AGGREGATION = "district_aggregation"
    PARTY_ANALYSIS = "party_analysis"
    COMPARISON = "comparison"
    PREDICTION = "prediction"
    VULNERABILITY = "vulnerability"
    SWING_ANALYSIS = "swing_analysis"
    SURVEY_DATA = "survey_data"
    HYBRID_SEARCH = "hybrid_search"
    DIRECT_ANSWER = "direct_answer"


@dataclass
class RouteDecision:
    """Decision about how to route a query."""
    route: QueryRoute
    confidence: float
    parameters: Dict[str, Any]
    explanation: str


class QueryRouter:
    """
    Routes queries to the optimal retrieval strategy.
    
    Key routing logic:
    - If query mentions specific constituency → CONSTITUENCY_LOOKUP
    - If query asks "how many" or "total" → aggregation route
    - If query compares entities → COMPARISON
    - If query mentions 2026/prediction → PREDICTION
    - If query mentions swing/trend → SWING_ANALYSIS
    - Default → HYBRID_SEARCH
    """
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph):
        self.kg = knowledge_graph
        self.analyzer = QueryAnalyzer()
        
        # Build route patterns
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize routing patterns."""
        self.route_patterns = {
            QueryRoute.CONSTITUENCY_LOOKUP: [
                r"(?:tell me about|what about|analyze|details of|profile of)\s+([A-Z][A-Z\s]+)(?:\s+constituency)?",
                r"([A-Z][A-Z\s]+)\s+(?:election|result|winner|data)",
                r"(?:who won|winner in|result of)\s+([A-Z][A-Z\s]+)",
            ],
            QueryRoute.DISTRICT_AGGREGATION: [
                r"(?:how many|total|all)\s+(?:seats?|constituencies)\s+in\s+(\w+)\s+district",
                r"(\w+)\s+district\s+(?:analysis|overview|summary)",
                r"(?:seats?|constituencies)\s+in\s+(\w+)",
            ],
            QueryRoute.PARTY_ANALYSIS: [
                r"(?:BJP|TMC|AITC|CPM)\s+(?:performance|seats?|won|lost|position)",
                r"(?:how many|total)\s+(?:seats?)\s+(?:did|will)\s+(BJP|TMC|AITC|CPM)",
                r"(BJP|TMC|AITC|CPM)\s+(?:analysis|strategy|strength|weakness)",
            ],
            QueryRoute.PREDICTION: [
                r"(?:2026|predict|forecast|will|expect|chances?)",
                r"(?:who will win|predicted winner|projection)",
                r"(?:likely|probability|odds)\s+(?:of|for)",
            ],
            QueryRoute.VULNERABILITY: [
                r"(?:vulnerable|at risk|swing|battleground|marginal)",
                r"(?:seats? at risk|could lose|may flip)",
                r"(?:close race|tight contest|toss-?up)",
            ],
            QueryRoute.SWING_ANALYSIS: [
                r"(?:swing|trend|shift|change)\s+(?:in|of|from|between)",
                r"(?:2019|2021|2024)\s+(?:to|vs|versus|compared to)",
                r"(?:momentum|direction|movement)",
            ],
            QueryRoute.COMPARISON: [
                r"(?:compare|versus|vs|between|difference)",
                r"(\w+)\s+(?:vs?|versus|or|and)\s+(\w+)",
                r"(?:better|worse|more|less)\s+(?:than|compared to)",
            ],
            QueryRoute.SURVEY_DATA: [
                r"(?:survey|poll|opinion|sentiment|public view)",
                r"(?:what do people|voters? think|popular opinion)",
                r"(?:preference|favorite|support for)",
            ],
        }
    
    def route(self, query: str) -> RouteDecision:
        """
        Determine the best route for a query.
        
        Returns:
            RouteDecision with route type, confidence, and parameters
        """
        query_lower = query.lower()
        analysis = self.analyzer.analyze(query, self.kg)
        
        # Check for specific constituency mention first (highest priority)
        if analysis.entity_types:
            for entity, etype in analysis.entity_types.items():
                if etype == 'constituency':
                    profile = self.kg.get_constituency(entity)
                    if profile:
                        return RouteDecision(
                            route=QueryRoute.CONSTITUENCY_LOOKUP,
                            confidence=0.95,
                            parameters={"constituency": entity, "profile": profile},
                            explanation=f"Direct lookup for constituency: {entity}"
                        )
        
        # Check prediction queries
        if any(kw in query_lower for kw in ['2026', 'predict', 'will win', 'forecast', 'next election']):
            # Check if specific entity
            if analysis.entities_mentioned:
                return RouteDecision(
                    route=QueryRoute.PREDICTION,
                    confidence=0.9,
                    parameters={"entities": analysis.entities_mentioned},
                    explanation="Prediction query with specific entities"
                )
            else:
                return RouteDecision(
                    route=QueryRoute.PREDICTION,
                    confidence=0.85,
                    parameters={"type": "general"},
                    explanation="General prediction query"
                )
        
        # Check vulnerability queries
        if any(kw in query_lower for kw in ['vulnerable', 'at risk', 'may lose', 'could flip', 'swing seat']):
            party = None
            if 'bjp' in query_lower:
                party = 'BJP'
            elif 'tmc' in query_lower or 'trinamool' in query_lower:
                party = 'TMC'
            
            return RouteDecision(
                route=QueryRoute.VULNERABILITY,
                confidence=0.9,
                parameters={"party": party},
                explanation=f"Vulnerability analysis for {party or 'all parties'}"
            )
        
        # Check aggregation queries
        if any(kw in query_lower for kw in ['how many', 'total', 'count', 'all seats', 'number of']):
            # District aggregation
            for entity, etype in analysis.entity_types.items():
                if etype == 'district':
                    return RouteDecision(
                        route=QueryRoute.DISTRICT_AGGREGATION,
                        confidence=0.9,
                        parameters={"district": entity},
                        explanation=f"District aggregation for {entity}"
                    )
            
            # Party aggregation
            if any(p in query_lower for p in ['bjp', 'tmc', 'trinamool', 'cpm', 'left']):
                party = 'BJP' if 'bjp' in query_lower else 'TMC' if 'tmc' in query_lower or 'trinamool' in query_lower else 'CPM'
                return RouteDecision(
                    route=QueryRoute.PARTY_ANALYSIS,
                    confidence=0.9,
                    parameters={"party": party, "aggregation": True},
                    explanation=f"Party seat aggregation for {party}"
                )
        
        # Check comparison queries
        if any(kw in query_lower for kw in ['compare', 'versus', ' vs ', 'between', 'difference']):
            return RouteDecision(
                route=QueryRoute.COMPARISON,
                confidence=0.85,
                parameters={"entities": analysis.entities_mentioned},
                explanation="Comparison query"
            )
        
        # Check swing/trend analysis
        if any(kw in query_lower for kw in ['swing', 'trend', 'shift', 'movement', 'momentum']):
            return RouteDecision(
                route=QueryRoute.SWING_ANALYSIS,
                confidence=0.85,
                parameters={"entities": analysis.entities_mentioned},
                explanation="Swing/trend analysis"
            )
        
        # Check survey queries
        if any(kw in query_lower for kw in ['survey', 'poll', 'opinion', 'sentiment', 'what do people think']):
            return RouteDecision(
                route=QueryRoute.SURVEY_DATA,
                confidence=0.85,
                parameters={},
                explanation="Survey data query"
            )
        
        # Check party analysis (without aggregation)
        for party in ['bjp', 'tmc', 'trinamool', 'cpm', 'congress', 'left']:
            if party in query_lower:
                party_code = 'BJP' if party == 'bjp' else 'TMC' if party in ['tmc', 'trinamool'] else 'CPM' if party in ['cpm', 'left'] else 'INC'
                return RouteDecision(
                    route=QueryRoute.PARTY_ANALYSIS,
                    confidence=0.8,
                    parameters={"party": party_code},
                    explanation=f"Party analysis for {party_code}"
                )
        
        # Default to hybrid search
        return RouteDecision(
            route=QueryRoute.HYBRID_SEARCH,
            confidence=0.7,
            parameters={"query": query, "analysis": analysis},
            explanation="Using hybrid search for free-form query"
        )


class QueryExecutor:
    """
    Executes queries based on route decisions.
    """
    
    def __init__(self, 
                 knowledge_graph: PoliticalKnowledgeGraph,
                 retriever: VerifiedRetriever):
        self.kg = knowledge_graph
        self.retriever = retriever
        self.router = QueryRouter(knowledge_graph)
    
    def execute(self, query: str) -> VerifiedAnswer:
        """
        Execute a query through the routing system.
        """
        # Route the query
        decision = self.router.route(query)
        
        # Execute based on route
        if decision.route == QueryRoute.CONSTITUENCY_LOOKUP:
            return self._execute_constituency_lookup(query, decision)
        elif decision.route == QueryRoute.DISTRICT_AGGREGATION:
            return self._execute_district_aggregation(query, decision)
        elif decision.route == QueryRoute.PARTY_ANALYSIS:
            return self._execute_party_analysis(query, decision)
        elif decision.route == QueryRoute.PREDICTION:
            return self._execute_prediction(query, decision)
        elif decision.route == QueryRoute.VULNERABILITY:
            return self._execute_vulnerability(query, decision)
        elif decision.route == QueryRoute.SWING_ANALYSIS:
            return self._execute_swing_analysis(query, decision)
        elif decision.route == QueryRoute.COMPARISON:
            return self._execute_comparison(query, decision)
        elif decision.route == QueryRoute.SURVEY_DATA:
            return self._execute_survey(query, decision)
        else:
            return self._execute_hybrid(query, decision)
    
    def _execute_constituency_lookup(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute constituency lookup."""
        constituency = decision.parameters.get("constituency", "")
        profile = decision.parameters.get("profile")
        
        if not profile:
            profile = self.kg.get_constituency(constituency)
        
        if not profile:
            return VerifiedAnswer(
                question=query,
                answer_text=f"No data found for constituency: {constituency}",
                confidence=0.0,
                caveats=["Constituency not found in database"]
            )
        
        # Generate comprehensive summary
        summary = self.kg.generate_constituency_summary(constituency)
        facts = self.kg.get_facts_for_entity(constituency)
        
        return VerifiedAnswer(
            question=query,
            answer_text=summary,
            confidence=0.95,
            facts=facts,
            sources=profile.source_files,
            caveats=[]
        )
    
    def _execute_district_aggregation(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute district aggregation."""
        district = decision.parameters.get("district", "")
        
        summary = self.kg.generate_district_summary(district)
        constituencies = self.kg.get_constituencies_by_district(district)
        
        facts = []
        sources = set()
        for c in constituencies[:10]:
            facts.extend(self.kg.get_facts_for_entity(c.ac_name)[:2])
            sources.update(c.source_files)
        
        return VerifiedAnswer(
            question=query,
            answer_text=summary,
            confidence=0.9,
            facts=facts,
            sources=list(sources),
            caveats=[]
        )
    
    def _execute_party_analysis(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute party analysis."""
        party = decision.parameters.get("party", "")
        
        # Get seats
        seats_2021 = self.kg.get_constituencies_by_winner(party)
        seats_2026 = self.kg.count_predicted_seats()
        
        text = f"## {party} Analysis\n\n"
        text += f"### 2021 Assembly Election\n"
        text += f"**Seats Won:** {len(seats_2021)}\n\n"
        
        if seats_2021:
            # Group by district
            by_district = {}
            for seat in seats_2021:
                if seat.district not in by_district:
                    by_district[seat.district] = []
                by_district[seat.district].append(seat)
            
            text += "**By District:**\n"
            for dist, dist_seats in sorted(by_district.items(), key=lambda x: -len(x[1])):
                text += f"- {dist}: {len(dist_seats)} seats\n"
            
            text += "\n**Strongest Seats (by margin):**\n"
            sorted_seats = sorted(seats_2021, 
                                 key=lambda x: abs(x.margin_2021) if x.winner_2021.upper() in [party.upper(), 'AITC'] else -abs(x.margin_2021),
                                 reverse=True)
            for seat in sorted_seats[:10]:
                text += f"- {seat.ac_name}: {abs(seat.margin_2021):.2f}% margin\n"
        
        text += f"\n### 2026 Predictions\n"
        pred_count = seats_2026.get(party, seats_2026.get('TMC', 0) if party == 'AITC' else 0)
        text += f"**Predicted Seats:** {pred_count}\n"
        
        change = pred_count - len(seats_2021)
        if change > 0:
            text += f"**Change:** +{change} seats (gain)\n"
        else:
            text += f"**Change:** {change} seats ({'loss' if change < 0 else 'no change'})\n"
        
        # Get vulnerable seats
        vulnerable = self.kg.get_vulnerable_seats(party)
        if vulnerable:
            text += f"\n**Seats at Risk:** {len(vulnerable)}\n"
            for seat in vulnerable[:5]:
                other_party = 'BJP' if party in ['TMC', 'AITC'] else 'TMC'
                text += f"- {seat.ac_name}: {other_party} predicted by {abs(seat.predicted_margin_2026):.2f}%\n"
        
        facts = self.kg.get_facts_by_type('electoral_result')[:10]
        
        return VerifiedAnswer(
            question=query,
            answer_text=text,
            confidence=0.9,
            facts=facts,
            sources=["knowledge_graph_aggregation"],
            caveats=["Predictions based on historical trends and current polling"]
        )
    
    def _execute_prediction(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute prediction query."""
        entities = decision.parameters.get("entities", [])
        
        if entities:
            # Specific entity predictions
            text = "## 2026 Election Predictions\n\n"
            facts = []
            
            for entity in entities:
                profile = self.kg.get_constituency(entity)
                if profile:
                    text += f"### {entity}\n"
                    text += f"- **2021 Winner:** {profile.winner_2021}\n"
                    text += f"- **Predicted 2026 Winner:** {profile.predicted_winner_2026}\n"
                    text += f"- **Predicted Margin:** {abs(profile.predicted_margin_2026):.2f}%\n"
                    text += f"- **Race Rating:** {profile.race_rating}\n"
                    
                    if profile.vulnerability_tag:
                        text += f"- **Status:** {profile.vulnerability_tag}\n"
                    
                    text += f"- **PC Swing (2019→2024):** {abs(profile.pc_swing_2019_2024):.2f}% "
                    text += f"{'towards TMC' if profile.pc_swing_2019_2024 > 0 else 'towards BJP'}\n\n"
                    
                    facts.extend(self.kg.get_facts_for_entity(entity))
            
            return VerifiedAnswer(
                question=query,
                answer_text=text,
                confidence=0.85,
                facts=facts,
                sources=["WB_Assembly_2026_predictions.csv"],
                caveats=["Predictions based on swing analysis and historical trends"]
            )
        else:
            # General predictions
            seats_2026 = self.kg.count_predicted_seats()
            ratings = self.kg.count_by_race_rating()
            
            text = "## 2026 West Bengal Assembly Predictions\n\n"
            text += "### Projected Seat Distribution\n\n"
            
            for party, count in sorted(seats_2026.items(), key=lambda x: -x[1]):
                text += f"- **{party}:** {count} seats\n"
            
            text += "\n### Race Ratings Breakdown\n\n"
            for rating in ['Safe', 'Likely', 'Lean', 'Toss-up']:
                if rating in ratings:
                    text += f"**{rating}:**\n"
                    for party, count in sorted(ratings[rating].items(), key=lambda x: -x[1]):
                        text += f"  - {party}: {count}\n"
            
            # Swing seats
            swing = self.kg.get_swing_seats(5.0)
            text += f"\n### Key Swing Seats ({len(swing)} total)\n\n"
            for seat in sorted(swing, key=lambda x: abs(x.predicted_margin_2026))[:15]:
                text += f"- **{seat.ac_name}** ({seat.district}): "
                text += f"{seat.predicted_winner_2026} by {abs(seat.predicted_margin_2026):.2f}% "
                text += f"[{seat.race_rating}]\n"
            
            return VerifiedAnswer(
                question=query,
                answer_text=text,
                confidence=0.8,
                facts=[],
                sources=["WB_Assembly_2026_predictions.csv"],
                caveats=["Predictions are estimates based on historical trends and may change"]
            )
    
    def _execute_vulnerability(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute vulnerability analysis."""
        party = decision.parameters.get("party")
        
        text = "## Vulnerability Analysis: 2026\n\n"
        
        if party:
            vulnerable = self.kg.get_vulnerable_seats(party)
            text += f"### {party} Seats at Risk ({len(vulnerable)} seats)\n\n"
            
            for seat in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026), reverse=True):
                other_party = 'TMC' if party == 'BJP' else 'BJP'
                text += f"**{seat.ac_name}** ({seat.district})\n"
                text += f"- 2021: {party} won with {abs(seat.margin_2021):.2f}% margin\n"
                text += f"- 2026: {other_party} predicted by {abs(seat.predicted_margin_2026):.2f}%\n"
                text += f"- PC Swing: {abs(seat.pc_swing_2019_2024):.2f}% towards {other_party if seat.pc_swing_2019_2024 < 0 else party}\n"
                if seat.vulnerability_tag:
                    text += f"- Status: {seat.vulnerability_tag}\n"
                text += "\n"
        else:
            # Both parties
            bjp_vulnerable = self.kg.get_vulnerable_seats("BJP")
            tmc_vulnerable = self.kg.get_vulnerable_seats("TMC")
            
            text += f"### BJP Seats at Risk: {len(bjp_vulnerable)}\n"
            for seat in bjp_vulnerable[:10]:
                text += f"- {seat.ac_name}: TMC predicted by {abs(seat.predicted_margin_2026):.2f}%\n"
            
            text += f"\n### TMC Seats at Risk: {len(tmc_vulnerable)}\n"
            for seat in tmc_vulnerable[:10]:
                text += f"- {seat.ac_name}: BJP predicted by {abs(seat.predicted_margin_2026):.2f}%\n"
        
        facts = self.kg.get_facts_by_type('vulnerability')
        
        return VerifiedAnswer(
            question=query,
            answer_text=text,
            confidence=0.85,
            facts=facts,
            sources=["WB_2026_BJP_vulnerable_to_TMC_estimated.csv", 
                    "WB_2026_TMC_vulnerable_to_BJP_estimated.csv"],
            caveats=[]
        )
    
    def _execute_swing_analysis(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute swing analysis."""
        text = "## Swing Analysis: 2019-2024\n\n"
        
        # Get all PCs with swing data
        pcs = {}
        for profile in self.kg.constituency_profiles.values():
            if profile.parent_pc not in pcs:
                pcs[profile.parent_pc] = {
                    'swing': profile.pc_swing_2019_2024,
                    'seats': []
                }
            pcs[profile.parent_pc]['seats'].append(profile)
        
        # Sort by swing magnitude
        sorted_pcs = sorted(pcs.items(), key=lambda x: abs(x[1]['swing']), reverse=True)
        
        text += "### Parliamentary Constituencies by Swing\n\n"
        text += "| PC Name | Swing | Direction | Impact |\n"
        text += "|---------|-------|-----------|--------|\n"
        
        for pc_name, data in sorted_pcs:
            swing = data['swing']
            direction = "→ TMC" if swing > 0 else "→ BJP"
            n_seats = len(data['seats'])
            text += f"| {pc_name} | {abs(swing):.2f}% | {direction} | {n_seats} ACs |\n"
        
        # Biggest gainers
        tmc_gains = [p for p in sorted_pcs if p[1]['swing'] > 0]
        bjp_gains = [p for p in sorted_pcs if p[1]['swing'] < 0]
        
        text += "\n### Top TMC Gaining PCs\n"
        for pc, data in tmc_gains[:5]:
            text += f"- **{pc}**: +{data['swing']:.2f}% swing\n"
        
        text += "\n### Top BJP Gaining PCs\n"
        for pc, data in bjp_gains[:5]:
            text += f"- **{pc}**: +{abs(data['swing']):.2f}% swing\n"
        
        facts = self.kg.get_facts_by_type('swing_analysis')
        
        return VerifiedAnswer(
            question=query,
            answer_text=text,
            confidence=0.9,
            facts=facts,
            sources=["WB_Assembly_2026_predictions.csv"],
            caveats=["Swing calculated from Lok Sabha 2019 to 2024 results"]
        )
    
    def _execute_comparison(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute comparison query."""
        entities = decision.parameters.get("entities", [])
        
        if len(entities) < 2:
            return self._execute_hybrid(query, decision)
        
        text = f"## Comparison: {' vs '.join(entities[:3])}\n\n"
        
        # Check if constituencies
        profiles = [self.kg.get_constituency(e) for e in entities]
        profiles = [p for p in profiles if p]
        
        if profiles:
            text += "| Metric | " + " | ".join([p.ac_name for p in profiles]) + " |\n"
            text += "|--------|" + "|".join(["--------"] * len(profiles)) + "|\n"
            text += "| District | " + " | ".join([p.district for p in profiles]) + " |\n"
            text += "| 2021 Winner | " + " | ".join([p.winner_2021 for p in profiles]) + " |\n"
            text += "| TMC 2021 | " + " | ".join([f"{p.tmc_vote_share_2021:.2f}%" for p in profiles]) + " |\n"
            text += "| BJP 2021 | " + " | ".join([f"{p.bjp_vote_share_2021:.2f}%" for p in profiles]) + " |\n"
            text += "| 2026 Predicted | " + " | ".join([p.predicted_winner_2026 for p in profiles]) + " |\n"
            text += "| Pred Margin | " + " | ".join([f"{abs(p.predicted_margin_2026):.2f}%" for p in profiles]) + " |\n"
            text += "| Race Rating | " + " | ".join([p.race_rating for p in profiles]) + " |\n"
        
        return VerifiedAnswer(
            question=query,
            answer_text=text,
            confidence=0.9,
            facts=[],
            sources=["WB_Assembly_2026_predictions.csv"],
            caveats=[]
        )
    
    def _execute_survey(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute survey data query."""
        text = "## Survey Data Summary\n\n"
        
        for survey_id, survey in self.kg.surveys.items():
            text += f"### {survey.survey_name}\n"
            text += f"**Question:** {survey.question}\n"
            text += f"**Responses:** {survey.total_responses}\n\n"
            
            text += "| Response | Count | Percentage |\n"
            text += "|----------|-------|------------|\n"
            
            for response, count in sorted(survey.response_distribution.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]:
                pct = survey.response_percentages.get(response, 0)
                text += f"| {response[:50]} | {count} | {pct:.1f}% |\n"
            
            text += "\n"
        
        facts = self.kg.get_facts_by_type('survey')
        
        return VerifiedAnswer(
            question=query,
            answer_text=text,
            confidence=0.85,
            facts=facts,
            sources=[s.source_file for s in self.kg.surveys.values()],
            caveats=["Survey results reflect respondent opinions and may have selection bias"]
        )
    
    def _execute_hybrid(self, query: str, decision: RouteDecision) -> VerifiedAnswer:
        """Execute hybrid search as fallback."""
        results = self.retriever.retrieve(query, top_k=5)
        return self.retriever.build_verified_answer(query, results)

