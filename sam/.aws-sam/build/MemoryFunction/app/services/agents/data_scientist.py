"""
Data Scientist Agent - Electoral Data Scientist.

Specialization: Statistical Analysis
Micro-Level Capabilities: Swing calculations, turnout modeling, vote transfer matrices
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class DataScientistAgent(SpecialistAgent):
    """
    Electoral Data Scientist.
    
    Performs statistical analysis, predictive modeling, and scenario simulations.
    Calculates swing percentages, models turnout, and analyzes vote transfer patterns.
    """
    
    name = "Data Scientist Agent"
    role = "Electoral Data Scientist"
    goal = "Perform statistical analysis, predictive modeling, and scenario simulations"
    backstory = """You are an electoral data scientist who builds predictive models. 
You calculate swing percentages, model turnout scenarios, analyze vote transfer matrices, 
and quantify the impact of various campaign interventions. Your expertise includes:
- Swing calculation and analysis
- Turnout modeling by segment
- Vote transfer matrix construction
- Statistical projection of outcomes
- Scenario simulation and sensitivity analysis"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Perform statistical and predictive analysis."""
        
        if not evidences:
            return {
                "historical_data": {},
                "swing_analysis": {},
                "turnout_model": {},
                "vote_transfer_matrix": {},
                "projections": [],
                "scenario_simulations": [],
                "statistical_insights": [],
                "data_gaps": ["No evidence available for statistical analysis"]
            }
        
        # Get previous analysis if available
        previous_data = {}
        if context and context.previous_results:
            for result in context.previous_results:
                if "voter_segments" in result.content:
                    previous_data["segments"] = result.content["voter_segments"]
                if "historical_trends" in result.content:
                    previous_data["trends"] = result.content["historical_trends"]
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        context_text = ""
        if previous_data:
            context_text = f"\n\nPREVIOUS DATA:\n{previous_data}"
        
        system = self._build_system_prompt()
        prompt = f"""Perform comprehensive statistical analysis from the evidence.

QUERY: {query}
{context_text}

EVIDENCE:
{evidence_text}

Create a data-driven electoral analysis:

1. HISTORICAL DATA: Key numbers from past elections
2. SWING ANALYSIS: Projected swing in vote shares
3. TURNOUT MODEL: Expected turnout by segment
4. VOTE TRANSFER MATRIX: How votes might shift between parties
5. PROJECTIONS: Statistical projections with confidence intervals
6. SCENARIO SIMULATIONS: Multiple outcome scenarios
7. STATISTICAL INSIGHTS: Key data-driven insights

Return as JSON:
{{
    "historical_data": {{
        "last_election": {{
            "year": 2021,
            "winner": "TMC",
            "winning_margin": 15000,
            "vote_shares": {{"TMC": "48%", "BJP": "38%", "Others": "14%"}},
            "turnout": "82%"
        }},
        "trend_over_elections": [
            {{"year": 2016, "winner": "TMC", "margin": 25000}},
            {{"year": 2021, "winner": "TMC", "margin": 15000}}
        ]
    }},
    "swing_analysis": {{
        "projected_swing": {{
            "from_tmc_to_bjp": "3-5%",
            "from_others_to_bjp": "2-3%",
            "from_tmc_to_others": "1-2%"
        }},
        "swing_drivers": ["anti-incumbency", "candidate factor", "national mood"],
        "net_impact": "BJP gains 4-6% overall"
    }},
    "turnout_model": {{
        "base_turnout": "82%",
        "projected_turnout": "80-84%",
        "by_segment": [
            {{"segment": "Urban", "projected": "75%", "vs_last": "-2%"}},
            {{"segment": "Rural", "projected": "85%", "vs_last": "+1%"}},
            {{"segment": "Youth", "projected": "70%", "vs_last": "-3%"}}
        ],
        "turnout_sensitivity": "Each 1% turnout change = ~2500 votes"
    }},
    "vote_transfer_matrix": {{
        "description": "Estimated vote flow between parties",
        "matrix": [
            {{"from": "TMC", "to": "BJP", "percentage": "8%", "reason": "Anti-incumbency"}},
            {{"from": "TMC", "to": "Stayed", "percentage": "85%", "reason": "Core vote"}},
            {{"from": "Left", "to": "BJP", "percentage": "40%", "reason": "Tactical voting"}}
        ]
    }},
    "projections": [
        {{
            "scenario": "Base case",
            "vote_share": {{"BJP": "42%", "TMC": "44%", "Others": "14%"}},
            "margin": "-3000 to -5000",
            "outcome": "Close loss",
            "probability": "45%"
        }},
        {{
            "scenario": "High mobilization",
            "vote_share": {{"BJP": "46%", "TMC": "42%", "Others": "12%"}},
            "margin": "+5000 to +8000",
            "outcome": "Narrow win",
            "probability": "35%"
        }},
        {{
            "scenario": "Status quo",
            "vote_share": {{"BJP": "40%", "TMC": "46%", "Others": "14%"}},
            "margin": "-10000 to -15000",
            "outcome": "Clear loss",
            "probability": "20%"
        }}
    ],
    "scenario_simulations": [
        {{
            "scenario_name": "High youth turnout",
            "assumptions": ["Youth turnout increases by 10%"],
            "impact": "BJP vote share +2%",
            "required_action": "Aggressive youth campaign"
        }},
        {{
            "scenario_name": "Booth-level focus",
            "assumptions": ["Win 60 out of 100 marginal booths"],
            "impact": "Net gain of 8000 votes",
            "required_action": "Targeted booth management"
        }}
    ],
    "statistical_insights": [
        "Margin of victory likely to be less than 10000 votes (tight contest)",
        "Youth segment is the swing decider - every 1% youth swing = 1500 votes",
        "12 booths with <100 vote difference in last election - priority targets",
        "Muslim consolidation key variable - 38% population, currently 12% support"
    ],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.1)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["historical_data", "swing_analysis", "turnout_model",
                    "vote_transfer_matrix", "projections", "scenario_simulations",
                    "statistical_insights", "data_gaps"]:
            if key not in content:
                content[key] = {} if key in ["historical_data", "swing_analysis", 
                                              "turnout_model", "vote_transfer_matrix"] else []
        
        return content


class StatisticalCalculatorTool:
    """Tool for electoral statistical calculations."""
    
    name = "statistical_calculator_tool"
    description = "Perform electoral statistical calculations"
    
    def calculate_swing(
        self,
        previous_results: Dict[str, float],
        current_polls: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate swing between elections."""
        
        swings = {}
        for party in previous_results:
            prev = previous_results.get(party, 0)
            curr = current_polls.get(party, 0)
            swings[party] = round(curr - prev, 2)
        
        return {
            "swings": swings,
            "biggest_gainer": max(swings.items(), key=lambda x: x[1])[0] if swings else None,
            "biggest_loser": min(swings.items(), key=lambda x: x[1])[0] if swings else None
        }
    
    def project_votes(
        self,
        total_voters: int,
        turnout_percentage: float,
        vote_shares: Dict[str, float]
    ) -> Dict[str, Any]:
        """Project absolute votes from shares."""
        
        total_votes = int(total_voters * turnout_percentage / 100)
        projections = {}
        
        for party, share in vote_shares.items():
            projections[party] = int(total_votes * share / 100)
        
        # Sort by votes
        sorted_projections = dict(sorted(projections.items(), key=lambda x: x[1], reverse=True))
        parties = list(sorted_projections.keys())
        
        return {
            "total_voters": total_voters,
            "expected_turnout": total_votes,
            "projections": sorted_projections,
            "winner": parties[0] if parties else None,
            "margin": sorted_projections[parties[0]] - sorted_projections[parties[1]] if len(parties) >= 2 else 0
        }


class ScenarioSimulatorTool:
    """Tool for simulating election scenarios."""
    
    name = "scenario_simulator_tool"
    description = "Simulate election outcomes under different scenarios"
    
    def run(
        self,
        base_projection: Dict[str, float],
        scenarios: List[Dict[str, Any]],
        total_voters: int,
        base_turnout: float = 80
    ) -> List[Dict[str, Any]]:
        """Run scenario simulations."""
        
        results = []
        
        for scenario in scenarios:
            name = scenario.get("name", "Unnamed")
            adjustments = scenario.get("adjustments", {})
            turnout_adj = scenario.get("turnout_adjustment", 0)
            
            # Apply adjustments
            adjusted_shares = {}
            for party, base_share in base_projection.items():
                adj = adjustments.get(party, 0)
                adjusted_shares[party] = max(0, min(100, base_share + adj))
            
            # Normalize to 100%
            total_share = sum(adjusted_shares.values())
            if total_share > 0:
                adjusted_shares = {k: v * 100 / total_share for k, v in adjusted_shares.items()}
            
            # Calculate votes
            turnout = base_turnout + turnout_adj
            total_votes = int(total_voters * turnout / 100)
            
            votes = {party: int(total_votes * share / 100) for party, share in adjusted_shares.items()}
            sorted_votes = dict(sorted(votes.items(), key=lambda x: x[1], reverse=True))
            parties = list(sorted_votes.keys())
            
            results.append({
                "scenario_name": name,
                "adjusted_vote_shares": {k: f"{v:.1f}%" for k, v in adjusted_shares.items()},
                "projected_votes": sorted_votes,
                "winner": parties[0] if parties else None,
                "margin": sorted_votes[parties[0]] - sorted_votes[parties[1]] if len(parties) >= 2 else 0,
                "turnout": f"{turnout:.1f}%"
            })
        
        return results
