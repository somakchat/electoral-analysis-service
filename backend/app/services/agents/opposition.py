"""
Opposition Research Agent - Opposition Research Director.

Specialization: Competitor Analysis
Micro-Level Capabilities: Candidate strengths/weaknesses, anti-incumbency mapping, defection tracking
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class OppositionResearchAgent(SpecialistAgent):
    """
    Opposition Research Director.
    
    Analyzes competitor strategies, candidate vulnerabilities, and counter-narratives.
    Tracks every move of competing parties, identifies weaknesses, and develops counter-strategies.
    """
    
    name = "Opposition Research Agent"
    role = "Opposition Research Director"
    goal = "Analyze competitor strategies, candidate vulnerabilities, and counter-narratives"
    backstory = """You are an opposition researcher who tracks every move of competing parties. 
You identify weaknesses in opponent candidates, track defections, measure anti-incumbency 
sentiment, and develop counter-strategies. Your expertise includes:
- Candidate profile analysis
- Vulnerability assessment
- Anti-incumbency measurement
- Vote transfer scenario analysis
- Counter-narrative development"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Perform opposition research analysis."""
        
        if not evidences:
            return {
                "competing_parties": [],
                "opponent_candidates": [],
                "vulnerabilities": [],
                "anti_incumbency_factors": [],
                "potential_defections": [],
                "vote_transfer_scenarios": [],
                "counter_strategies": [],
                "data_gaps": ["No evidence available for opposition analysis"]
            }
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        system = self._build_system_prompt()
        prompt = f"""Analyze opposition and competitors from the evidence.

QUERY: {query}

EVIDENCE:
{evidence_text}

Create a comprehensive opposition analysis:

1. COMPETING PARTIES: All parties contesting in this constituency
2. OPPONENT CANDIDATES: Profile each opponent candidate
3. VULNERABILITIES: Weaknesses that can be exploited
4. ANTI-INCUMBENCY FACTORS: Any anti-incumbency sentiment
5. POTENTIAL DEFECTIONS: Leaders who might switch sides
6. VOTE TRANSFER SCENARIOS: How votes might transfer between parties
7. COUNTER STRATEGIES: Recommended counter-narratives and strategies

For each opponent candidate, provide:
- name
- party
- strengths
- weaknesses
- past_performance
- local_base
- vulnerability_score (1-10)

Return as JSON:
{{
    "competing_parties": [
        {{"party": "TMC", "local_strength": "Strong", "key_leaders": ["..."]}}
    ],
    "opponent_candidates": [
        {{
            "name": "Candidate Name",
            "party": "Party",
            "strengths": ["strong local base", ...],
            "weaknesses": ["corruption allegations", ...],
            "past_performance": "Won 2019 by 15000 votes",
            "local_base": "Urban areas",
            "vulnerability_score": 7
        }}
    ],
    "vulnerabilities": [
        {{"target": "TMC", "vulnerability": "Anti-incumbency in urban areas", "exploitation_strategy": "..."}}
    ],
    "anti_incumbency_factors": ["factor1", "factor2"],
    "potential_defections": [
        {{"leader": "Name", "current_party": "...", "likelihood": "high/medium/low"}}
    ],
    "vote_transfer_scenarios": [
        {{"scenario": "If X defects", "vote_transfer": "5-7%", "impact": "..."}}
    ],
    "counter_strategies": [
        {{"opponent_narrative": "...", "counter_narrative": "...", "execution": "..."}}
    ],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.15)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["competing_parties", "opponent_candidates", "vulnerabilities",
                    "anti_incumbency_factors", "potential_defections", 
                    "vote_transfer_scenarios", "counter_strategies", "data_gaps"]:
            if key not in content:
                content[key] = []
        
        return content


class CandidateProfileTool:
    """Tool for profiling candidates."""
    
    name = "candidate_profile_tool"
    description = "Build detailed profile of a candidate"
    
    def __init__(self, rag):
        self.rag = rag
    
    def run(self, candidate_name: str, party: str = None) -> Dict[str, Any]:
        query = f"Profile of {candidate_name}"
        if party:
            query += f" from {party}"
        
        evidences = self.rag.search(query)
        
        return {
            "candidate": candidate_name,
            "party": party,
            "data_found": len(evidences) > 0,
            "raw_data": [{"text": e.text, "source": e.source_path} for e in evidences[:5]]
        }


class VulnerabilityScorerTool:
    """Tool for scoring candidate vulnerabilities."""
    
    name = "vulnerability_scorer_tool"
    description = "Score vulnerabilities of opponent candidates"
    
    def run(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score vulnerabilities based on available data."""
        score = 5  # Base score
        
        weaknesses = candidate_data.get("weaknesses", [])
        strengths = candidate_data.get("strengths", [])
        
        # Adjust based on weaknesses
        if "corruption" in str(weaknesses).lower():
            score += 2
        if "anti-incumbency" in str(weaknesses).lower():
            score += 1.5
        if "unpopular" in str(weaknesses).lower():
            score += 1
        
        # Adjust based on strengths
        if "popular" in str(strengths).lower():
            score -= 1
        if "strong base" in str(strengths).lower():
            score -= 1
        
        return {
            "candidate": candidate_data.get("name", "Unknown"),
            "vulnerability_score": min(10, max(1, score)),
            "assessment": "High vulnerability" if score > 7 else "Medium vulnerability" if score > 4 else "Low vulnerability"
        }
