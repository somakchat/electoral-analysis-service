"""
Campaign Strategist Agent - Ground-level campaign planning.

This agent specializes in:
- Campaign messaging and narrative
- Ground game strategy
- Issue-based targeting
- Voter outreach planning
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .evidence_framework import (
    PoliticalAgentBase, Evidence, Claim, EvidenceType, ConfidenceLevel,
    AgentCapability, calculate_seat_metrics
)
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.rag.data_schema import ConstituencyProfile


class CampaignStrategistAgent(PoliticalAgentBase):
    """
    Ground-level campaign strategy with tactical recommendations.
    
    Capabilities:
    - Campaign narrative design
    - Ground game planning
    - Issue identification
    - Voter outreach strategy
    """
    
    name = "Campaign Strategist"
    role = "Ground campaign operations and voter engagement expert"
    expertise = [
        "Campaign messaging",
        "Ground operations",
        "Voter outreach",
        "Issue identification",
        "Local campaign tactics",
        "Rally and booth management"
    ]
    
    QUERY_KEYWORDS = [
        'campaign', 'ground', 'booth', 'rally', 'message', 'narrative',
        'voter', 'outreach', 'canvass', 'door-to-door', 'issue', 'local',
        'strategy', 'plan', 'organize', 'mobilize'
    ]
    
    def can_handle(self, query: str) -> Tuple[bool, float]:
        """Check if this agent can handle the query."""
        query_lower = query.lower()
        
        keyword_matches = sum(1 for kw in self.QUERY_KEYWORDS if kw in query_lower)
        if keyword_matches >= 2:
            return True, 0.9
        elif keyword_matches == 1:
            return True, 0.6
        
        if 'how to win' in query_lower or 'winning strategy' in query_lower:
            return True, 0.8
        
        return False, 0.0
    
    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main analysis entry point."""
        self.reset()
        context = context or {}
        
        party = context.get('party') or self._extract_party(query)
        constituency = context.get('constituency') or self._extract_constituency(query)
        
        query_lower = query.lower()
        
        if constituency:
            return self._constituency_campaign_plan(constituency, party, query)
        elif 'message' in query_lower or 'narrative' in query_lower:
            return self._messaging_strategy(party, query)
        elif 'ground' in query_lower or 'booth' in query_lower:
            return self._ground_game_strategy(party, query)
        elif 'issue' in query_lower:
            return self._issue_analysis(party, query)
        else:
            return self._comprehensive_campaign_plan(party, query)
    
    def _extract_party(self, query: str) -> Optional[str]:
        """Extract party from query."""
        query_lower = query.lower()
        if 'bjp' in query_lower:
            return 'BJP'
        elif 'tmc' in query_lower or 'trinamool' in query_lower:
            return 'TMC'
        return None
    
    def _extract_constituency(self, query: str) -> Optional[str]:
        """Extract constituency from query."""
        query_upper = query.upper()
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                return name
        return None
    
    def _constituency_campaign_plan(self, constituency: str, party: str, query: str) -> Dict[str, Any]:
        """Create constituency-specific campaign plan."""
        
        profile = self.get_constituency_data(constituency)
        
        if not profile:
            return {
                "answer": f"No data found for constituency: {constituency}",
                "confidence": 0.0
            }
        
        party = party or ('BJP' if profile.predicted_winner_2026 == 'BJP' else 'TMC')
        is_incumbent = profile.winner_2021.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
        is_predicted_win = profile.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']
        
        answer = f"## Campaign Strategy: {constituency} for {party}\n\n"
        
        # Situation Assessment
        answer += "### Situation Assessment\n\n"
        answer += f"- **2021 Status:** {'Incumbent (won)' if is_incumbent else 'Opposition (lost)'}\n"
        answer += f"- **2026 Projection:** {'Retaining' if is_predicted_win and is_incumbent else 'Capturing' if is_predicted_win else 'Defending' if is_incumbent else 'Challenging'}\n"
        answer += f"- **Race Rating:** {profile.race_rating}\n"
        answer += f"- **Predicted Margin:** {abs(profile.predicted_margin_2026):.1f}%\n"
        answer += f"- **Lok Sabha Trend:** {abs(profile.pc_swing_2019_2024):.1f}% {'towards TMC' if profile.pc_swing_2019_2024 > 0 else 'towards BJP'}\n\n"
        
        # Campaign Strategy based on situation
        answer += "### Recommended Strategy\n\n"
        
        if is_predicted_win and abs(profile.predicted_margin_2026) > 10:
            # Comfortable lead
            answer += "**Strategy Type:** Defensive Consolidation\n\n"
            answer += "1. **Focus on turnout** - Maximize supporter turnout\n"
            answer += "2. **Avoid controversies** - Maintain positive narrative\n"
            answer += "3. **Community outreach** - Strengthen local connections\n"
            answer += "4. **Booth management** - Ensure strong presence at all booths\n"
        
        elif is_predicted_win and abs(profile.predicted_margin_2026) <= 10:
            # Close race, leading
            answer += "**Strategy Type:** Aggressive Consolidation\n\n"
            answer += "1. **Intensive ground game** - Door-to-door in every ward\n"
            answer += "2. **Target swing voters** - Focus on undecided demographics\n"
            answer += "3. **Counter opposition narrative** - Active fact-checking\n"
            answer += "4. **High-visibility rallies** - Star campaigner visits\n"
            answer += "5. **Booth-level monitoring** - Prevent last-minute swings\n"
        
        elif not is_predicted_win and abs(profile.predicted_margin_2026) <= 5:
            # Close race, trailing
            answer += "**Strategy Type:** Aggressive Offense\n\n"
            answer += "1. **Change the narrative** - Highlight opposition failures\n"
            answer += "2. **Maximum ground contact** - 3x normal canvassing\n"
            answer += "3. **Local issues focus** - Constituency-specific problems\n"
            answer += "4. **Star power** - Multiple high-profile visits\n"
            answer += "5. **Social media blitz** - Viral local content\n"
            answer += "6. **Vote banking** - Secure every sympathizer\n"
        
        else:
            # Significant deficit
            answer += "**Strategy Type:** Disruption Campaign\n\n"
            answer += "1. **Contrast messaging** - Clear differentiation\n"
            answer += "2. **Issue insurgency** - Local problems as campaign focus\n"
            answer += "3. **Coalition building** - Unite anti-incumbent vote\n"
            answer += "4. **Guerrilla tactics** - Unconventional outreach\n"
            answer += "5. **Youth mobilization** - First-time voter focus\n"
        
        # Ground Operations
        answer += "\n### Ground Operations Plan\n\n"
        answer += "**Booth-Level Activities:**\n"
        answer += "- Booth committees with local influencers\n"
        answer += "- Daily morning/evening voter contact\n"
        answer += "- WhatsApp groups for each polling booth\n"
        answer += "- Women's wing engagement programs\n\n"
        
        answer += "**Voter Contact Targets:**\n"
        margin_gap = abs(profile.predicted_margin_2026)
        if margin_gap < 5:
            answer += "- Contact every registered voter minimum 3 times\n"
            answer += "- Identify and secure every sympathizer\n"
        elif margin_gap < 10:
            answer += "- Contact 80% of voters at least twice\n"
            answer += "- Focus on fence-sitters and new voters\n"
        else:
            answer += "- Contact 60% of voters at least once\n"
            answer += "- Prioritize strong supporter mobilization\n"
        
        # Messaging
        answer += "\n### Messaging Framework\n\n"
        if party == 'BJP':
            answer += "**Core Themes:**\n"
            answer += "- Development and infrastructure\n"
            answer += "- Governance and law & order\n"
            answer += "- National integration\n"
            answer += "- Anti-corruption\n"
        else:
            answer += "**Core Themes:**\n"
            answer += "- Bengal identity and culture\n"
            answer += "- Welfare schemes and benefits\n"
            answer += "- Local development\n"
            answer += "- Inclusive governance\n"
        
        claims = [
            self.make_claim(
                statement=f"Campaign strategy for {constituency} should be {'defensive' if is_predicted_win else 'offensive'} based on {abs(profile.predicted_margin_2026):.1f}% {'lead' if is_predicted_win else 'deficit'}",
                evidence=[self.create_evidence_from_profile(profile, "2026_prediction")],
                reasoning="Strategy calibrated to electoral position"
            )
        ]
        
        return {
            "answer": answer,
            "claims": [{"statement": c.statement, "confidence": c.confidence.value} for c in claims],
            "confidence": 0.85,
            "constituency": constituency,
            "party": party
        }
    
    def _messaging_strategy(self, party: str, query: str) -> Dict[str, Any]:
        """Develop party messaging strategy."""
        
        party = party or 'BJP'
        all_seats = list(self.kg.constituency_profiles.values())
        
        # Analyze party position
        party_seats = [c for c in all_seats 
                      if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']]
        
        answer = f"## {party} Messaging Strategy for 2026\n\n"
        
        answer += "### Core Narrative Framework\n\n"
        
        if party == 'BJP':
            answer += "**Primary Narrative:** 'Bengal Deserves Better'\n\n"
            answer += "**Key Messages:**\n"
            answer += "1. **Development Gap:** Contrast Gujarat/UP development with Bengal\n"
            answer += "2. **Governance:** Law & order, corruption-free administration\n"
            answer += "3. **Welfare Delivery:** Central schemes blocked by state\n"
            answer += "4. **National Pride:** Bengal's role in New India\n"
            answer += "5. **Youth Focus:** Jobs, entrepreneurship, skills\n\n"
            
            answer += "**Counter-Narratives:**\n"
            answer += "- vs 'Outsider' tag: 'Development has no borders'\n"
            answer += "- vs 'Culture attack': 'BJP celebrates Bengali culture'\n"
            answer += "- vs 'Divisive': 'Unity for development'\n"
        
        else:  # TMC
            answer += "**Primary Narrative:** 'Bengal's Own Government'\n\n"
            answer += "**Key Messages:**\n"
            answer += "1. **Bengal Identity:** Protect Bengali language, culture\n"
            answer += "2. **Welfare Champion:** Lakshmir Bhandar, Swasthya Sathi\n"
            answer += "3. **Women Empowerment:** Kanyashree, women's safety\n"
            answer += "4. **Against Outsiders:** Keep Bengal in Bengali hands\n"
            answer += "5. **Inclusive Growth:** All communities, all regions\n\n"
            
            answer += "**Counter-Narratives:**\n"
            answer += "- vs 'Corruption': 'Opposition's false propaganda'\n"
            answer += "- vs 'Appeasement': 'Equal treatment for all'\n"
            answer += "- vs 'Misgovernance': 'Compare with BJP states'\n"
        
        # Segment-specific messaging
        answer += "\n### Voter Segment Messaging\n\n"
        answer += "| Segment | Key Message | Evidence to Use |\n"
        answer += "|---------|-------------|------------------|\n"
        answer += "| Youth | Jobs & opportunities | Unemployment data |\n"
        answer += "| Women | Safety & welfare | Scheme enrollments |\n"
        answer += "| Farmers | MSP & support | Agricultural investment |\n"
        answer += "| Urban | Development & services | Infrastructure projects |\n"
        answer += "| Rural | Roads & electricity | Connectivity improvements |\n"
        
        # Regional adaptation
        answer += "\n### Regional Message Adaptation\n\n"
        
        # Analyze by region
        regions = defaultdict(list)
        for c in all_seats:
            # Simplified regional grouping
            if c.district in ['DARJEELING', 'JALPAIGURI', 'COOCH BEHAR', 'ALIPURDUAR']:
                regions['North Bengal'].append(c)
            elif c.district in ['KOLKATA', 'HOWRAH', 'NORTH 24 PARGANAS', 'SOUTH 24 PARGANAS']:
                regions['Kolkata Metro'].append(c)
            elif c.district in ['BANKURA', 'PURULIA', 'BARDHAMAN', 'BIRBHUM']:
                regions['Rarh Bengal'].append(c)
            else:
                regions['Other'].append(c)
        
        for region, seats in regions.items():
            metrics = calculate_seat_metrics(seats)
            answer += f"**{region}** ({metrics['total']} seats):\n"
            if metrics['swing_direction'] == 'TMC':
                answer += f"  - Trend: +{abs(metrics['avg_swing']):.1f}% towards TMC\n"
            else:
                answer += f"  - Trend: +{abs(metrics['avg_swing']):.1f}% towards BJP\n"
            answer += f"  - Focus: {'Consolidate' if party == metrics['swing_direction'] else 'Counter-narrative'}\n\n"
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "party": party
        }
    
    def _ground_game_strategy(self, party: str, query: str) -> Dict[str, Any]:
        """Ground operations strategy."""
        
        party = party or 'BJP'
        all_seats = list(self.kg.constituency_profiles.values())
        
        swing_seats = self.get_swing_seats(5.0)
        vulnerable = self.get_vulnerable_seats(party)
        
        answer = f"## {party} Ground Game Strategy\n\n"
        
        answer += "### Booth-Level Organization\n\n"
        answer += "**Booth Committee Structure:**\n"
        answer += "- 1 Booth President + 1 Vice President\n"
        answer += "- 5-7 Active workers per booth\n"
        answer += "- 1 Women's coordinator\n"
        answer += "- 1 Youth coordinator\n"
        answer += "- WhatsApp group for coordination\n\n"
        
        answer += "**Booth Activities:**\n"
        answer += "- Daily morning meeting (6:30 AM)\n"
        answer += "- Voter list verification\n"
        answer += "- House-to-house contact (evenings)\n"
        answer += "- Issue collection and reporting\n"
        answer += "- D-Day preparation\n\n"
        
        answer += "### Resource Prioritization\n\n"
        
        # High priority constituencies
        high_priority = [c for c in swing_seats if abs(c.predicted_margin_2026) < 3]
        
        answer += f"**Tier 1 - Maximum Resources ({len(high_priority)} constituencies)**\n"
        for c in sorted(high_priority, key=lambda x: abs(x.predicted_margin_2026))[:10]:
            answer += f"- {c.ac_name}: {abs(c.predicted_margin_2026):.1f}% margin\n"
        
        answer += f"\n**Tier 2 - High Resources ({len(vulnerable)} vulnerable seats)**\n"
        for c in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026))[:5]:
            answer += f"- {c.ac_name}: Defend from {abs(c.predicted_margin_2026):.1f}% deficit\n"
        
        answer += "\n### Voter Contact Metrics\n\n"
        answer += "| Constituency Type | Contact Target | Frequency |\n"
        answer += "|-------------------|----------------|----------|\n"
        answer += "| Toss-up | 100% voters | 3+ times |\n"
        answer += "| Lean | 90% voters | 2+ times |\n"
        answer += "| Likely | 75% voters | 2 times |\n"
        answer += "| Safe | 50% voters | 1 time |\n"
        
        answer += "\n### Election Day Operations\n\n"
        answer += "**Booth Monitoring:**\n"
        answer += "- Polling agent at every booth\n"
        answer += "- Hourly voter turnout tracking\n"
        answer += "- Vehicle arrangement for voters\n"
        answer += "- Refreshment for workers\n"
        answer += "- Issue resolution cell active\n"
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "party": party,
            "high_priority_count": len(high_priority)
        }
    
    def _issue_analysis(self, party: str, query: str) -> Dict[str, Any]:
        """Analyze key issues for campaign."""
        
        answer = "## Issue Analysis for West Bengal 2026\n\n"
        
        answer += "### State-wide Issues\n\n"
        answer += "**Economic Issues:**\n"
        answer += "- Unemployment (especially youth)\n"
        answer += "- Industrial stagnation\n"
        answer += "- Agricultural distress\n"
        answer += "- Price rise and inflation\n\n"
        
        answer += "**Governance Issues:**\n"
        answer += "- Law and order concerns\n"
        answer += "- Corruption allegations\n"
        answer += "- Administrative efficiency\n"
        answer += "- Central-state coordination\n\n"
        
        answer += "**Identity Issues:**\n"
        answer += "- Bengali identity preservation\n"
        answer += "- Religious harmony\n"
        answer += "- Cultural protection\n"
        answer += "- Language concerns\n\n"
        
        answer += "### Regional Variations\n\n"
        answer += "| Region | Primary Issues | Secondary Issues |\n"
        answer += "|--------|----------------|------------------|\n"
        answer += "| North Bengal | Tea garden wages, tourism | Gorkhaland, connectivity |\n"
        answer += "| Kolkata Metro | Jobs, traffic, services | Housing, pollution |\n"
        answer += "| Rarh Bengal | Agriculture, industry | Mining, tribal rights |\n"
        answer += "| Coastal Bengal | Fishing, cyclone relief | Sundarban conservation |\n"
        answer += "| Murshidabad/Malda | Border issues, trade | Employment, education |\n"
        
        return {
            "answer": answer,
            "confidence": 0.8
        }
    
    def _comprehensive_campaign_plan(self, party: str, query: str) -> Dict[str, Any]:
        """Comprehensive campaign plan."""
        
        party = party or 'BJP'
        
        answer = f"## Comprehensive Campaign Plan: {party} for WB 2026\n\n"
        
        # Overview
        seats_2021 = len(self.get_party_seats(party))
        predicted_2026 = len([c for c in self.kg.constituency_profiles.values()
                            if c.predicted_winner_2026.upper() in [party.upper(), 'AITC' if party == 'TMC' else '']])
        vulnerable = len(self.get_vulnerable_seats(party))
        swing = len(self.get_swing_seats(5.0))
        
        answer += f"### Current Position\n"
        answer += f"- 2021 Seats: {seats_2021}\n"
        answer += f"- 2026 Projected: {predicted_2026}\n"
        answer += f"- Vulnerable Seats: {vulnerable}\n"
        answer += f"- Swing Seats: {swing}\n\n"
        
        # Campaign phases
        answer += "### Campaign Phases\n\n"
        answer += "**Phase 1: Foundation (T-6 months)**\n"
        answer += "- Booth committee activation\n"
        answer += "- Voter list preparation\n"
        answer += "- Issue identification survey\n"
        answer += "- Candidate selection initiation\n\n"
        
        answer += "**Phase 2: Outreach (T-3 months)**\n"
        answer += "- Mass contact programs\n"
        answer += "- Local leader engagement\n"
        answer += "- Women's wing programs\n"
        answer += "- Youth mobilization\n\n"
        
        answer += "**Phase 3: Intensive (T-1 month)**\n"
        answer += "- Daily rallies and meetings\n"
        answer += "- Star campaigner visits\n"
        answer += "- Media blitz\n"
        answer += "- Door-to-door peak\n\n"
        
        answer += "**Phase 4: Closing (Last 10 days)**\n"
        answer += "- Vote consolidation\n"
        answer += "- Booth-level tracking\n"
        answer += "- Last-mile outreach\n"
        answer += "- D-Day preparation\n"
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "party": party
        }

