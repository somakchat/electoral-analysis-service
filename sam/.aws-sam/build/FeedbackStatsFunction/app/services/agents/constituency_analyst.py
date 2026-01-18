"""
Constituency Intelligence Analyst - Deep analysis of electoral constituencies.

This agent specializes in:
- Individual constituency profiling
- Multi-constituency comparisons
- Constituency clustering and patterns
- Historical performance analysis
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .evidence_framework import (
    PoliticalAgentBase, Evidence, Claim, EvidenceType, ConfidenceLevel,
    AgentCapability, calculate_seat_metrics, format_constituency_brief
)
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.rag.data_schema import ConstituencyProfile


class ConstituencyIntelligenceAgent(PoliticalAgentBase):
    """
    Deep constituency analysis with evidence-based insights.
    
    Capabilities:
    - Profile any constituency with full data
    - Compare constituencies
    - Identify patterns across constituencies
    - Analyze constituency clusters
    """
    
    name = "Constituency Intelligence Analyst"
    role = "Electoral geography and constituency dynamics expert"
    expertise = [
        "Constituency profiling",
        "Electoral history analysis",
        "Vote share decomposition",
        "Demographic-electoral correlations",
        "Swing pattern analysis"
    ]
    
    capabilities = [
        AgentCapability(
            name="constituency_profile",
            description="Generate comprehensive constituency profile",
            input_types=["constituency_name"],
            output_types=["profile", "claims", "evidence"],
            required_data=["electoral_results", "predictions"]
        ),
        AgentCapability(
            name="compare_constituencies",
            description="Compare multiple constituencies",
            input_types=["constituency_names"],
            output_types=["comparison", "claims"],
            required_data=["electoral_results"]
        ),
        AgentCapability(
            name="find_similar",
            description="Find constituencies with similar characteristics",
            input_types=["constituency_name", "criteria"],
            output_types=["similar_constituencies"],
            required_data=["electoral_results"]
        )
    ]
    
    # Keywords for query matching
    QUERY_KEYWORDS = [
        'constituency', 'seat', 'ac', 'assembly', 'vidhan sabha',
        'profile', 'analysis', 'compare', 'similar', 'like'
    ]
    
    def can_handle(self, query: str) -> Tuple[bool, float]:
        """Check if this agent can handle the query."""
        query_lower = query.lower()
        
        # Check for constituency names
        for name in self.kg.constituency_profiles.keys():
            if name.lower() in query_lower:
                return True, 0.95
        
        # Check for keywords
        keyword_matches = sum(1 for kw in self.QUERY_KEYWORDS if kw in query_lower)
        if keyword_matches >= 2:
            return True, 0.8
        elif keyword_matches == 1:
            return True, 0.6
        
        return False, 0.0
    
    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main analysis entry point."""
        self.reset()
        
        context = context or {}
        query_lower = query.lower()
        
        # Check for constituency name first
        constituency_name = context.get('constituency') or self._extract_constituency_name(query)
        
        # If no constituency mentioned, try RAG-based general search
        if not constituency_name:
            # For general queries, use RAG search to find relevant data
            return self._general_constituency_query(query, context)
        
        # Determine analysis type
        if self._is_comparison_query(query):
            return self._compare_constituencies(query, context)
        elif self._is_cluster_query(query):
            return self._analyze_cluster(query, context)
        else:
            return self._profile_constituency(query, context)
    
    def _general_constituency_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries without specific constituency using RAG search."""
        query_lower = query.lower()
        
        # Try to understand what the user is asking about
        # Swing seats / competitive seats
        if any(w in query_lower for w in ['swing', 'competitive', 'close', 'toss-up', 'battleground']):
            return self._find_swing_seats(query, context)
        
        # Safe seats / strongholds
        if any(w in query_lower for w in ['safe', 'strong', 'bastion', 'fortress']):
            return self._find_safe_seats(query, context)
        
        # All constituencies / list
        if any(w in query_lower for w in ['all', 'list', 'total', 'how many']):
            return self._list_constituencies(query, context)
        
        # Use OpenSearch for semantic search
        try:
            results = self.search_data(query, top_k=10)
            
            if results and len(results) > 0:
                answer_parts = []
                evidence_list = []
                
                for r in results[:5]:
                    if r.get('text'):
                        answer_parts.append(r['text'])
                    evidence_list.append({
                        "content": r.get('text', '')[:200],
                        "source": r.get('source', 'Electoral Data'),
                        "score": r.get('score', 0)
                    })
                
                if answer_parts:
                    return {
                        "answer": "\n\n---\n\n".join(answer_parts[:3]),
                        "claims": [],
                        "evidence": evidence_list[:5],
                        "confidence": min(0.85, sum(r.get('score', 0) for r in results[:3]) / 15),
                        "sources": list(set(r.get('source', '') for r in results if r.get('source'))),
                        "reasoning": self.get_reasoning_chain()
                    }
        except Exception as e:
            pass
        
        # Last resort: provide a helpful message
        return {
            "answer": "I can help you with constituency-specific analysis. Please specify a constituency name, or try asking about swing seats, safe seats, or competitive constituencies in West Bengal.",
            "claims": [],
            "evidence": [],
            "confidence": 0.5,
            "reasoning": self.get_reasoning_chain()
        }
    
    def _find_swing_seats(self, query: str, context: Dict) -> Dict[str, Any]:
        """Find and analyze swing/competitive seats."""
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Define swing seats: close margin and/or high swing
        swing_seats = []
        for c in all_seats:
            margin = abs(c.predicted_margin_2026) if c.predicted_margin_2026 else 0
            rating = c.race_rating.lower() if c.race_rating else ''
            
            if margin < 5 or rating in ['toss-up', 'lean', 'lean tmc', 'lean bjp']:
                swing_seats.append(c)
        
        # Sort by predicted margin (closest races first)
        swing_seats = sorted(swing_seats, key=lambda x: abs(x.predicted_margin_2026) if x.predicted_margin_2026 else 99)
        
        answer = f"## Swing Seats in West Bengal (2026 Predictions)\n\n"
        answer += f"**Total Swing/Competitive Seats:** {len(swing_seats)}\n\n"
        answer += "These are constituencies where the race is expected to be close (<5% margin):\n\n"
        
        answer += "| Constituency | District | 2021 Winner | Predicted 2026 | Margin | Rating |\n"
        answer += "|--------------|----------|-------------|----------------|--------|--------|\n"
        
        for c in swing_seats[:15]:
            answer += f"| {c.ac_name} | {c.district} | {c.winner_2021} | {c.predicted_winner_2026} | {abs(c.predicted_margin_2026):.1f}% | {c.race_rating} |\n"
        
        if len(swing_seats) > 15:
            answer += f"\n*...and {len(swing_seats) - 15} more swing seats*\n"
        
        # Group by party
        tmc_defend = [c for c in swing_seats if c.winner_2021 == 'TMC']
        bjp_defend = [c for c in swing_seats if c.winner_2021 == 'BJP']
        
        answer += f"\n### Summary\n"
        answer += f"- **TMC defending:** {len(tmc_defend)} swing seats\n"
        answer += f"- **BJP defending:** {len(bjp_defend)} swing seats\n"
        
        claims = [self.make_claim(
            statement=f"There are {len(swing_seats)} swing seats in West Bengal for 2026",
            evidence=[Evidence(
                content=f"Based on predicted margins <5%",
                evidence_type=EvidenceType.STATISTICAL,
                source="Prediction Model"
            )],
            reasoning="Swing seats defined as constituencies with predicted margin under 5%"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": "Swing seat analysis based on 2026 predictions", "source": "Electoral Data"}],
            "confidence": 0.9,
            "swing_seats": len(swing_seats)
        }
    
    def _find_safe_seats(self, query: str, context: Dict) -> Dict[str, Any]:
        """Find and analyze safe seats."""
        all_seats = list(self.kg.constituency_profiles.values())
        query_lower = query.lower()
        
        # Determine party filter
        party = None
        if 'bjp' in query_lower:
            party = 'BJP'
        elif 'tmc' in query_lower or 'trinamool' in query_lower:
            party = 'TMC'
        
        # Safe seats: high margin (>15%)
        safe_seats = [c for c in all_seats if abs(c.predicted_margin_2026 or 0) > 15]
        
        if party:
            safe_seats = [c for c in safe_seats if c.predicted_winner_2026 == party]
        
        safe_seats = sorted(safe_seats, key=lambda x: abs(x.predicted_margin_2026 or 0), reverse=True)
        
        title = f"Safe Seats" + (f" for {party}" if party else " in West Bengal")
        answer = f"## {title}\n\n"
        answer += f"**Total Safe Seats (margin >15%):** {len(safe_seats)}\n\n"
        
        for c in safe_seats[:10]:
            answer += f"- **{c.ac_name}** ({c.district}): {c.predicted_winner_2026} by {abs(c.predicted_margin_2026):.1f}%\n"
        
        return {
            "answer": answer,
            "claims": [],
            "evidence": [{"content": f"Safe seat analysis", "source": "Electoral Data"}],
            "confidence": 0.85,
            "safe_seats": len(safe_seats)
        }
    
    def _list_constituencies(self, query: str, context: Dict) -> Dict[str, Any]:
        """List constituencies based on query criteria."""
        all_seats = list(self.kg.constituency_profiles.values())
        query_lower = query.lower()
        
        # Check for district filter
        district = None
        for d in set(c.district for c in all_seats):
            if d.lower() in query_lower:
                district = d
                break
        
        if district:
            filtered = [c for c in all_seats if c.district == district]
            answer = f"## Constituencies in {district} District\n\n"
            answer += f"**Total:** {len(filtered)}\n\n"
            for c in filtered:
                answer += f"- **{c.ac_name}**: {c.winner_2021} (2021) -> {c.predicted_winner_2026} (2026 pred)\n"
        else:
            answer = f"## All Constituencies in West Bengal\n\n"
            answer += f"**Total:** {len(all_seats)} Assembly Constituencies\n\n"
            
            # Group by district
            by_district = {}
            for c in all_seats:
                if c.district not in by_district:
                    by_district[c.district] = []
                by_district[c.district].append(c)
            
            for d, seats in sorted(by_district.items()):
                answer += f"### {d} ({len(seats)} seats)\n"
        
        return {
            "answer": answer,
            "claims": [],
            "evidence": [{"content": "Constituency listing", "source": "Electoral Data"}],
            "confidence": 0.95
        }
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query asks for comparison."""
        comparison_words = ['compare', 'versus', ' vs ', 'difference', 'between']
        return any(w in query.lower() for w in comparison_words)
    
    def _is_cluster_query(self, query: str) -> bool:
        """Check if query asks about groups of constituencies."""
        cluster_words = ['all', 'district', 'region', 'similar', 'like', 'pattern']
        return any(w in query.lower() for w in cluster_words)
    
    def _profile_constituency(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive constituency profile."""
        
        # Step 1: Identify constituency
        constituency_name = context.get('constituency') or self._extract_constituency_name(query)
        
        self.add_reasoning_step(
            action="Identify Target Constituency",
            input_data=query,
            output=f"Identified constituency: {constituency_name or 'Not found'}"
        )
        
        if not constituency_name:
            return {
                "answer": "I couldn't identify a specific constituency in your query. Please specify a constituency name.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0,
                "reasoning": self.get_reasoning_chain()
            }
        
        # Step 2: Retrieve constituency data
        profile = self.get_constituency_data(constituency_name)
        
        if not profile:
            # Try fuzzy match
            matches = self.kg.search_constituency(constituency_name)
            if matches:
                profile = matches[0]
                constituency_name = profile.ac_name
        
        if not profile:
            return {
                "answer": f"No data found for constituency: {constituency_name}. Please check the spelling or try a different constituency.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0,
                "reasoning": self.get_reasoning_chain()
            }
        
        # Step 3: Build evidence
        evidence_list = []
        claims = []
        
        # 2021 Result Evidence
        ev_2021 = self.create_evidence_from_profile(profile, "2021_winner")
        evidence_list.append(ev_2021)
        
        claims.append(self.make_claim(
            statement=f"{profile.ac_name} was won by {profile.winner_2021} in 2021 with a margin of {abs(profile.margin_2021):.2f}%",
            evidence=[ev_2021],
            reasoning="Direct electoral result from 2021 assembly election data"
        ))
        
        # 2026 Prediction Evidence
        ev_2026 = self.create_evidence_from_profile(profile, "2026_prediction")
        evidence_list.append(ev_2026)
        
        claims.append(self.make_claim(
            statement=f"In 2026, {profile.ac_name} is predicted to be won by {profile.predicted_winner_2026} with margin of {abs(profile.predicted_margin_2026):.2f}% (rated '{profile.race_rating}')",
            evidence=[ev_2026],
            reasoning="Based on swing analysis and prediction model"
        ))
        
        # Swing Evidence
        ev_swing = self.create_evidence_from_profile(profile, "swing")
        evidence_list.append(ev_swing)
        
        swing_direction = "towards TMC" if profile.pc_swing_2019_2024 > 0 else "towards BJP"
        claims.append(self.make_claim(
            statement=f"The {profile.parent_pc} PC (which includes {profile.ac_name}) saw a swing of {abs(profile.pc_swing_2019_2024):.2f}% {swing_direction} between 2019 and 2024 Lok Sabha",
            evidence=[ev_swing],
            reasoning="Calculated from Lok Sabha 2019 vs 2024 vote shares"
        ))
        
        self.add_reasoning_step(
            action="Gather Electoral Evidence",
            input_data=f"Profile for {constituency_name}",
            output=f"Collected {len(evidence_list)} evidence points",
            claims=claims
        )
        
        # Step 4: Generate strategic analysis
        strategic_claims = self._generate_strategic_insights(profile)
        claims.extend(strategic_claims)
        
        self.add_reasoning_step(
            action="Generate Strategic Insights",
            input_data="Electoral patterns",
            output=f"Generated {len(strategic_claims)} strategic insights",
            claims=strategic_claims
        )
        
        # Step 5: Build comprehensive answer
        answer = self._build_constituency_answer(profile, claims)
        
        reasoning_chain = self.get_reasoning_chain()
        reasoning_chain.question = query
        reasoning_chain.final_answer = answer
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": e.content, "source": e.source, "type": e.evidence_type.value} for e in evidence_list],
            "confidence": reasoning_chain.average_confidence,
            "reasoning": reasoning_chain,
            "constituency": constituency_name,
            "profile_data": {
                "ac_no": profile.ac_no,
                "ac_name": profile.ac_name,
                "district": profile.district,
                "type": profile.constituency_type.value if hasattr(profile.constituency_type, 'value') else str(profile.constituency_type),
                "parent_pc": profile.parent_pc,
                "winner_2021": profile.winner_2021,
                "predicted_2026": profile.predicted_winner_2026,
                "race_rating": profile.race_rating
            }
        }
    
    def _compare_constituencies(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple constituencies."""
        
        # Extract constituency names
        names = self._extract_multiple_constituencies(query)
        
        if len(names) < 2:
            return {
                "answer": "Please specify at least two constituencies to compare.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0
            }
        
        profiles = [self.get_constituency_data(n) for n in names]
        profiles = [p for p in profiles if p is not None]
        
        if len(profiles) < 2:
            return {
                "answer": f"Could not find sufficient constituency data. Found: {[p.ac_name for p in profiles]}",
                "claims": [],
                "evidence": [],
                "confidence": 0.3
            }
        
        # Build comparison
        claims = []
        evidence = []
        
        # Compare 2021 results
        comparison_text = "**2021 Results Comparison:**\n"
        for p in profiles:
            ev = self.create_evidence_from_profile(p, "2021_winner")
            evidence.append(ev)
            comparison_text += f"- {p.ac_name}: {p.winner_2021} (TMC {p.tmc_vote_share_2021:.1f}% vs BJP {p.bjp_vote_share_2021:.1f}%)\n"
        
        # Compare predictions
        comparison_text += "\n**2026 Predictions:**\n"
        for p in profiles:
            comparison_text += f"- {p.ac_name}: {p.predicted_winner_2026} predicted ({p.race_rating})\n"
        
        # Identify key differences
        differences = []
        if len(set(p.winner_2021 for p in profiles)) > 1:
            differences.append("Different 2021 winners")
        if len(set(p.predicted_winner_2026 for p in profiles)) > 1:
            differences.append("Different 2026 predictions")
        
        margin_range = max(p.predicted_margin_2026 for p in profiles) - min(p.predicted_margin_2026 for p in profiles)
        if margin_range > 10:
            differences.append(f"Significant margin variance ({margin_range:.1f}%)")
        
        claims.append(self.make_claim(
            statement=f"Key differences: {', '.join(differences) if differences else 'Constituencies show similar patterns'}",
            evidence=evidence,
            reasoning="Comparison of electoral metrics"
        ))
        
        return {
            "answer": comparison_text,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [{"content": e.content, "source": e.source} for e in evidence],
            "confidence": 0.9,
            "constituencies_compared": [p.ac_name for p in profiles]
        }
    
    def _analyze_cluster(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a cluster of constituencies (district, similar characteristics)."""
        
        query_lower = query.lower()
        
        # Check for district
        for district in set(p.district for p in self.kg.constituency_profiles.values()):
            if district.lower() in query_lower:
                return self._analyze_district(district)
        
        # Check for rating-based cluster
        for rating in ['safe', 'likely', 'lean', 'toss-up']:
            if rating in query_lower:
                return self._analyze_by_rating(rating)
        
        # Default: swing seats
        if 'swing' in query_lower or 'battleground' in query_lower:
            return self._analyze_swing_seats()
        
        return {
            "answer": "Please specify a district, rating category, or constituency pattern to analyze.",
            "claims": [],
            "evidence": [],
            "confidence": 0.0
        }
    
    def _analyze_district(self, district: str) -> Dict[str, Any]:
        """Analyze all constituencies in a district."""
        
        constituencies = self.get_constituencies_by_filter(district=district)
        
        if not constituencies:
            return {
                "answer": f"No constituencies found in {district} district.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0
            }
        
        metrics = calculate_seat_metrics(constituencies)
        
        claims = []
        
        # District overview claim
        claims.append(self.make_claim(
            statement=f"{district} has {metrics['total']} constituencies. In 2021: TMC won {metrics['tmc_2021']}, BJP won {metrics['bjp_2021']}. In 2026: TMC predicted {metrics['tmc_2026']}, BJP predicted {metrics['bjp_2026']}.",
            evidence=[self.create_aggregated_evidence(
                f"Analysis of {metrics['total']} constituencies in {district}",
                constituencies
            )],
            reasoning="Aggregated from constituency-level data"
        ))
        
        # Swing analysis
        if metrics['avg_swing'] != 0:
            claims.append(self.make_claim(
                statement=f"Average swing in {district}: {abs(metrics['avg_swing']):.2f}% towards {metrics['swing_direction']}",
                evidence=[self.create_aggregated_evidence(
                    f"Swing data from {len(constituencies)} constituencies",
                    constituencies
                )],
                reasoning="Calculated from PC-level Lok Sabha swing data"
            ))
        
        # Build answer
        answer = f"## {district} District Analysis\n\n"
        answer += f"**Total Seats:** {metrics['total']}\n\n"
        answer += f"**2021 Results:**\n"
        answer += f"- TMC: {metrics['tmc_2021']} seats\n"
        answer += f"- BJP: {metrics['bjp_2021']} seats\n"
        answer += f"- Others: {metrics['total'] - metrics['tmc_2021'] - metrics['bjp_2021']} seats\n\n"
        answer += f"**2026 Predictions:**\n"
        answer += f"- TMC: {metrics['tmc_2026']} seats\n"
        answer += f"- BJP: {metrics['bjp_2026']} seats\n"
        answer += f"- Swing Seats: {metrics['swing_seats']}\n\n"
        
        # Key battlegrounds
        swing_in_district = [c for c in constituencies if c.race_rating.lower() in ['toss-up', 'lean']]
        if swing_in_district:
            answer += "**Key Battlegrounds:**\n"
            for c in sorted(swing_in_district, key=lambda x: abs(x.predicted_margin_2026)):
                answer += f"- {c.ac_name}: {c.predicted_winner_2026} by {abs(c.predicted_margin_2026):.1f}% [{c.race_rating}]\n"
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [],
            "confidence": 0.95,
            "district": district,
            "metrics": metrics
        }
    
    def _analyze_by_rating(self, rating: str) -> Dict[str, Any]:
        """Analyze constituencies by race rating."""
        
        constituencies = self.kg.get_seats_by_rating(rating)
        
        if not constituencies:
            return {
                "answer": f"No constituencies found with '{rating}' rating.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0
            }
        
        # Group by predicted winner
        by_winner = {}
        for c in constituencies:
            winner = c.predicted_winner_2026
            if winner not in by_winner:
                by_winner[winner] = []
            by_winner[winner].append(c)
        
        answer = f"## '{rating.title()}' Rated Constituencies\n\n"
        answer += f"**Total:** {len(constituencies)} seats\n\n"
        
        for winner, seats in sorted(by_winner.items(), key=lambda x: -len(x[1])):
            answer += f"**{winner} Predicted ({len(seats)} seats):**\n"
            for c in sorted(seats, key=lambda x: abs(x.predicted_margin_2026)):
                answer += f"- {c.ac_name} ({c.district}): margin {abs(c.predicted_margin_2026):.1f}%\n"
            answer += "\n"
        
        claims = [self.make_claim(
            statement=f"There are {len(constituencies)} constituencies rated '{rating}', distributed as: {', '.join(f'{w}: {len(s)}' for w, s in by_winner.items())}",
            evidence=[self.create_aggregated_evidence(f"'{rating}' seats analysis", constituencies)],
            reasoning="Aggregated from prediction model ratings"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [],
            "confidence": 0.9,
            "rating": rating,
            "total": len(constituencies)
        }
    
    def _analyze_swing_seats(self) -> Dict[str, Any]:
        """Analyze swing/battleground seats."""
        
        swing = self.get_swing_seats(5.0)
        
        if not swing:
            return {
                "answer": "No swing seats found with margin less than 5%.",
                "claims": [],
                "evidence": [],
                "confidence": 0.0
            }
        
        # Sort by closeness
        swing_sorted = sorted(swing, key=lambda x: abs(x.predicted_margin_2026))
        
        # Count by predicted winner
        tmc_swing = sum(1 for c in swing if c.predicted_winner_2026.upper() == 'TMC')
        bjp_swing = sum(1 for c in swing if c.predicted_winner_2026.upper() == 'BJP')
        
        answer = f"## Swing Seats Analysis (Margin < 5%)\n\n"
        answer += f"**Total Swing Seats:** {len(swing)}\n"
        answer += f"- TMC Leading: {tmc_swing}\n"
        answer += f"- BJP Leading: {bjp_swing}\n\n"
        answer += "**Closest Races:**\n"
        
        for c in swing_sorted[:15]:
            answer += f"- **{c.ac_name}** ({c.district}): {c.predicted_winner_2026} by {abs(c.predicted_margin_2026):.2f}%\n"
            answer += f"  2021: {c.winner_2021} | Swing: {abs(c.pc_swing_2019_2024):.1f}% {'→TMC' if c.pc_swing_2019_2024 > 0 else '→BJP'}\n"
        
        claims = [self.make_claim(
            statement=f"{len(swing)} swing seats identified: TMC leads in {tmc_swing}, BJP leads in {bjp_swing}. These seats will determine the election outcome.",
            evidence=[self.create_aggregated_evidence("Swing seat analysis", swing)],
            reasoning="Constituencies with predicted margin under 5%"
        )]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "evidence": [],
            "confidence": 0.85,
            "swing_count": len(swing),
            "tmc_leading": tmc_swing,
            "bjp_leading": bjp_swing
        }
    
    def _generate_strategic_insights(self, profile: ConstituencyProfile) -> List[Claim]:
        """Generate strategic insights from constituency data."""
        claims = []
        
        # Flip detection
        if profile.winner_2021 != profile.predicted_winner_2026:
            claims.append(self.make_claim(
                statement=f"{profile.ac_name} is predicted to FLIP from {profile.winner_2021} to {profile.predicted_winner_2026} in 2026",
                evidence=[
                    self.create_evidence_from_profile(profile, "2021_winner"),
                    self.create_evidence_from_profile(profile, "2026_prediction")
                ],
                reasoning=f"2021 winner ({profile.winner_2021}) differs from 2026 prediction ({profile.predicted_winner_2026})"
            ))
        
        # Vulnerability assessment
        if profile.vulnerability_tag:
            claims.append(self.make_claim(
                statement=f"Vulnerability Status: {profile.vulnerability_tag}",
                evidence=[Evidence(
                    evidence_type=EvidenceType.HISTORICAL_TREND,
                    content=profile.vulnerability_tag,
                    source="vulnerability_analysis",
                    confidence=0.85
                )],
                reasoning="Based on swing patterns and historical trends"
            ))
        
        # Swing impact
        if abs(profile.pc_swing_2019_2024) > 10:
            direction = "TMC" if profile.pc_swing_2019_2024 > 0 else "BJP"
            claims.append(self.make_claim(
                statement=f"SIGNIFICANT SWING: The parent PC saw {abs(profile.pc_swing_2019_2024):.1f}% shift towards {direction}, indicating strong momentum",
                evidence=[self.create_evidence_from_profile(profile, "swing")],
                reasoning="Large swings (>10%) typically carry over to assembly elections"
            ))
        
        return claims
    
    def _build_constituency_answer(self, profile: ConstituencyProfile, claims: List[Claim]) -> str:
        """Build comprehensive constituency answer."""
        
        answer = f"## {profile.ac_name} Constituency Analysis\n\n"
        
        # Basic info
        answer += f"**Location:** {profile.district} district, part of {profile.parent_pc} PC\n"
        answer += f"**Category:** {profile.constituency_type.value if hasattr(profile.constituency_type, 'value') else profile.constituency_type} seat\n\n"
        
        # 2021 Results
        answer += "### 2021 Assembly Election\n"
        answer += f"- **Winner:** {profile.winner_2021}\n"
        answer += f"- **TMC Vote Share:** {profile.tmc_vote_share_2021:.2f}%\n"
        answer += f"- **BJP Vote Share:** {profile.bjp_vote_share_2021:.2f}%\n"
        answer += f"- **Margin:** {abs(profile.margin_2021):.2f}%\n\n"
        
        # Lok Sabha Trends
        answer += f"### Lok Sabha Trends ({profile.parent_pc} PC)\n"
        answer += f"- **2019:** TMC {profile.pc_tmc_vs_2019:.2f}% vs BJP {profile.pc_bjp_vs_2019:.2f}%\n"
        answer += f"- **2024:** TMC {profile.pc_tmc_vs_2024:.2f}% vs BJP {profile.pc_bjp_vs_2024:.2f}%\n"
        swing_dir = "TMC" if profile.pc_swing_2019_2024 > 0 else "BJP"
        answer += f"- **Swing:** {abs(profile.pc_swing_2019_2024):.2f}% towards {swing_dir}\n\n"
        
        # 2026 Prediction
        answer += "### 2026 Prediction\n"
        answer += f"- **Predicted Winner:** {profile.predicted_winner_2026}\n"
        answer += f"- **Predicted Margin:** {abs(profile.predicted_margin_2026):.2f}%\n"
        answer += f"- **Race Rating:** {profile.race_rating}\n"
        if profile.vulnerability_tag:
            answer += f"- **Status:** {profile.vulnerability_tag}\n"
        answer += "\n"
        
        # Strategic Claims
        answer += "### Key Insights\n"
        for claim in claims:
            if claim.confidence in [ConfidenceLevel.CERTAIN, ConfidenceLevel.HIGH]:
                answer += f"- {claim.statement}\n"
        
        return answer
    
    def _extract_constituency_name(self, query: str) -> Optional[str]:
        """Extract constituency name from query."""
        query_upper = query.upper()
        
        # Direct match
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                return name
        
        # Partial match
        for name in self.kg.constituency_profiles.keys():
            if any(word in query_upper for word in name.split()):
                return name
        
        return None
    
    def _extract_multiple_constituencies(self, query: str) -> List[str]:
        """Extract multiple constituency names from query."""
        query_upper = query.upper()
        found = []
        
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                found.append(name)
        
        return found[:5]  # Limit to 5

