"""
Advanced Decision-Making Tools for Political Strategy.

Implements:
1. SWOT Analysis Tool
2. Scenario Simulator Tool
3. Resource Allocation Optimizer
4. Micro-Targeting Tool
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import re
from functools import lru_cache

from app.services.llm import get_llm, BaseLLM


# ============= SWOT Analysis Tool =============

class SWOTAnalysisTool:
    """
    Generate SWOT analysis for a party/candidate in a constituency.
    """
    
    name = "swot_analysis"
    description = "Generate SWOT analysis for a party/candidate in a constituency"
    
    def __init__(self, rag=None):
        self.rag = rag
        self.llm = get_llm()
    
    def run(self, party: str, constituency: str, evidence: str = None) -> Dict[str, Any]:
        """
        Generate SWOT analysis grounded in evidence.
        
        Args:
            party: Political party name
            constituency: Constituency name
            evidence: Evidence text to base analysis on
        
        Returns:
            SWOT analysis with strengths, weaknesses, opportunities, threats, and priority actions
        """
        # Get evidence from RAG if not provided
        if not evidence and self.rag:
            query = f"SWOT analysis for {party} in {constituency}"
            evidences = self.rag.search(query)
            evidence = "\n\n".join([e.text for e in evidences[:5]])
        
        if not evidence:
            return {
                "party": party,
                "constituency": constituency,
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "priority_actions": [],
                "data_quality": "insufficient"
            }
        
        system = "You produce SWOT analysis strictly grounded in evidence. Output JSON only."
        prompt = f"""Create a comprehensive SWOT analysis for {party} in {constituency} based ONLY on the evidence below.
If evidence is insufficient for a quadrant, return an empty list for that quadrant.

Evidence:
{evidence[:3000]}

Return JSON with keys:
- strengths: List of strengths
- weaknesses: List of weaknesses  
- opportunities: List of opportunities
- threats: List of threats
- priority_actions: Top 5 recommended actions based on SWOT"""
        
        try:
            response = self.llm.generate(prompt, system=system, temperature=0.1)
            match = re.search(r'\{[\s\S]*\}', response.text)
            if match:
                result = json.loads(match.group())
                result["party"] = party
                result["constituency"] = constituency
                result["data_quality"] = "good" if len(evidence) > 500 else "limited"
                return result
        except Exception:
            pass
        
        return {
            "party": party,
            "constituency": constituency,
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "priority_actions": [],
            "data_quality": "error"
        }


# ============= Scenario Simulator Tool =============

class ScenarioSimulatorTool:
    """
    Simulate election outcomes under different scenarios.
    """
    
    name = "scenario_simulator"
    description = "Simulate election outcomes under different scenarios"
    
    def run(
        self,
        constituency: str,
        base_projection: Dict[str, float],
        scenarios: List[Dict[str, Any]],
        total_voters: int = 200000
    ) -> Dict[str, Any]:
        """
        Simulate multiple election scenarios.
        
        Args:
            constituency: Constituency name
            base_projection: Base vote share projection by party
            scenarios: List of scenario definitions with adjustments
            total_voters: Total voters in constituency
        
        Returns:
            Simulation results for all scenarios
        """
        results = []
        
        for scenario in scenarios:
            name = scenario.get("name", "Unnamed Scenario")
            adjustments = scenario.get("adjustments", {})
            turnout_adj = scenario.get("turnout_adjustment", 0)
            base_turnout = scenario.get("base_turnout", 80)
            
            # Apply adjustments to vote shares
            adjusted_shares = {}
            for party, base_share in base_projection.items():
                adj = adjustments.get(party, 0)
                adjusted_shares[party] = max(0, min(100, base_share + adj))
            
            # Normalize to 100%
            total = sum(adjusted_shares.values())
            if total > 0:
                adjusted_shares = {k: v * 100 / total for k, v in adjusted_shares.items()}
            
            # Calculate projected votes
            turnout = base_turnout + turnout_adj
            total_votes = int(total_voters * turnout / 100)
            
            votes = {party: int(total_votes * share / 100) 
                    for party, share in adjusted_shares.items()}
            
            # Determine winner and margin
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            winner = sorted_votes[0][0] if sorted_votes else None
            margin = sorted_votes[0][1] - sorted_votes[1][1] if len(sorted_votes) >= 2 else 0
            
            # Determine outcome description
            if margin > 20000:
                outcome = "Comfortable win"
            elif margin > 10000:
                outcome = "Clear win"
            elif margin > 5000:
                outcome = "Narrow win"
            elif margin > 0:
                outcome = "Very close win"
            else:
                outcome = "Tie"
            
            results.append({
                "scenario_name": name,
                "vote_shares": {k: f"{v:.1f}%" for k, v in adjusted_shares.items()},
                "projected_votes": votes,
                "winner": winner,
                "margin": margin,
                "outcome": outcome,
                "turnout": f"{turnout:.1f}%",
                "assumptions": scenario.get("assumptions", [])
            })
        
        return {
            "constituency": constituency,
            "total_voters": total_voters,
            "base_projection": {k: f"{v:.1f}%" for k, v in base_projection.items()},
            "scenarios": results,
            "sensitivity_analysis": self._sensitivity_analysis(base_projection, total_voters)
        }
    
    def _sensitivity_analysis(
        self,
        base_projection: Dict[str, float],
        total_voters: int
    ) -> Dict[str, Any]:
        """Analyze sensitivity to key variables."""
        analysis = {
            "turnout_sensitivity": f"Each 1% turnout change = ~{int(total_voters/100)} votes",
            "swing_sensitivity": {}
        }
        
        # Calculate votes per 1% swing for each party
        base_turnout = 80
        total_votes = int(total_voters * base_turnout / 100)
        
        for party in base_projection:
            votes_per_percent = int(total_votes / 100)
            analysis["swing_sensitivity"][party] = f"1% swing = ~{votes_per_percent} votes"
        
        return analysis


# ============= Resource Allocation Optimizer =============

class ResourceOptimizerTool:
    """
    Optimize campaign resource allocation across constituencies.
    """
    
    name = "resource_optimizer"
    description = "Optimize campaign resource allocation across constituencies"
    
    def run(
        self,
        total_budget: float,
        constituencies: List[Dict[str, Any]],
        objective: str = "maximize_seats"
    ) -> Dict[str, Any]:
        """
        Optimize resource allocation using constrained optimization.
        
        Args:
            total_budget: Total budget available
            constituencies: List of constituency data with win probabilities
            objective: Optimization objective (maximize_seats, maximize_margin)
        
        Returns:
            Optimal allocation and expected outcomes
        """
        if not constituencies:
            return {
                "total_budget": total_budget,
                "allocation": [],
                "expected_seats": 0
            }
        
        # Calculate strategic value for each constituency
        for const in constituencies:
            win_prob = const.get("win_probability", 0.5)
            marginal_impact = const.get("marginal_impact", 1.0)
            
            # Prioritize swing constituencies (0.3 to 0.7 win probability)
            if 0.3 <= win_prob <= 0.7:
                strategic_value = marginal_impact * 2  # Double priority for swing seats
            elif win_prob < 0.3:
                strategic_value = marginal_impact * 0.5  # Lower priority for long shots
            else:
                strategic_value = marginal_impact  # Normal priority for safe seats
            
            const["strategic_value"] = strategic_value
        
        # Sort by strategic value
        sorted_const = sorted(constituencies, key=lambda x: x["strategic_value"], reverse=True)
        
        # Allocate budget proportionally to strategic value
        total_value = sum(c["strategic_value"] for c in sorted_const)
        
        allocations = []
        for const in sorted_const:
            if total_value > 0:
                share = const["strategic_value"] / total_value
                allocated = total_budget * share
            else:
                allocated = total_budget / len(sorted_const)
            
            allocations.append({
                "constituency": const.get("name", "Unknown"),
                "allocated_budget": round(allocated, 2),
                "win_probability": const.get("win_probability", 0.5),
                "strategic_value": round(const["strategic_value"], 2),
                "priority": "high" if const["strategic_value"] > 1.5 else 
                          "medium" if const["strategic_value"] > 0.8 else "low"
            })
        
        # Calculate expected seats
        expected_seats = sum(
            1 if a["win_probability"] > 0.5 else 0.5 * a["win_probability"]
            for a in allocations
        )
        
        return {
            "total_budget": total_budget,
            "objective": objective,
            "allocation": allocations,
            "expected_seats": round(expected_seats, 1),
            "efficiency_score": round(expected_seats / max(1, len(allocations)) * 100, 1)
        }


# ============= Micro-Targeting Tool =============

class MicroTargetingTool:
    """
    Identify micro-segments and targeting strategies.
    """
    
    name = "micro_targeting"
    description = "Identify micro-segments and targeting strategies"
    
    def __init__(self, rag=None):
        self.rag = rag
        self.llm = get_llm()
    
    def run(
        self,
        constituency: str,
        party: str,
        evidence: str = None
    ) -> Dict[str, Any]:
        """
        Identify voter micro-segments and recommend targeting strategies.
        
        Args:
            constituency: Constituency name
            party: Party to develop strategy for
            evidence: Evidence text to base analysis on
        
        Returns:
            Segment analysis with targeting recommendations
        """
        # Get evidence from RAG if not provided
        if not evidence and self.rag:
            query = f"Voter demographics and segments in {constituency} for {party}"
            evidences = self.rag.search(query)
            evidence = "\n\n".join([e.text for e in evidences[:5]])
        
        if not evidence:
            return {
                "constituency": constituency,
                "party": party,
                "segments": [],
                "priority_segments": [],
                "data_quality": "insufficient"
            }
        
        system = "You extract voter micro-segments strictly from evidence. Output JSON only."
        prompt = f"""From the evidence below, identify voter segments relevant to {party} in {constituency}.

For each segment provide:
- segment_name
- population_share (percentage if known, null if unknown)
- historical_support (% support in past elections if available)
- current_sentiment (positive/negative/neutral/mixed)
- persuadability_score (1-10, where 10 is most persuadable)
- key_issues (list of top issues)
- recommended_messaging (specific message approach)
- outreach_channels (best ways to reach them)
- expected_impact (if targeted successfully)

Evidence:
{evidence[:3000]}

Return JSON:
{{
    "constituency": "{constituency}",
    "party": "{party}",
    "total_voters": null or number,
    "segments": [...],
    "priority_segments": ["segment1", "segment2"],  // Top 3 to focus on
    "resource_recommendation": "Where to invest most effort"
}}"""
        
        try:
            response = self.llm.generate(prompt, system=system, temperature=0.2)
            match = re.search(r'\{[\s\S]*\}', response.text)
            if match:
                result = json.loads(match.group())
                result["data_quality"] = "good" if len(evidence) > 500 else "limited"
                return result
        except Exception:
            pass
        
        return {
            "constituency": constituency,
            "party": party,
            "segments": [],
            "priority_segments": [],
            "data_quality": "error"
        }


# ============= Risk Assessment Tool =============

class RiskAssessmentTool:
    """
    Assess campaign risks and provide mitigation strategies.
    """
    
    name = "risk_assessment"
    description = "Assess campaign risks and provide mitigation strategies"
    
    def run(
        self,
        swot: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile risk assessment from various analyses.
        """
        risks = []
        
        # Extract risks from SWOT weaknesses and threats
        for weakness in swot.get("weaknesses", []):
            risks.append({
                "risk": weakness,
                "source": "SWOT - Weakness",
                "severity": "medium",
                "mitigation": "Address through targeted campaign"
            })
        
        for threat in swot.get("threats", []):
            risks.append({
                "risk": threat,
                "source": "SWOT - Threat",
                "severity": "high",
                "mitigation": "Monitor and prepare counter-strategy"
            })
        
        # Extract risks from negative scenarios
        for scenario in scenarios:
            if "loss" in str(scenario.get("outcome", "")).lower():
                risks.append({
                    "risk": f"Scenario '{scenario.get('scenario_name')}' projects loss",
                    "source": "Scenario Analysis",
                    "severity": "high" if "clear" in str(scenario.get("outcome", "")).lower() else "medium",
                    "mitigation": "Focus on changing assumptions"
                })
        
        # Extract risks from sentiment
        if sentiment.get("overall_mood") == "negative":
            risks.append({
                "risk": "Overall negative public mood",
                "source": "Sentiment Analysis",
                "severity": "high",
                "mitigation": "Address key grievances immediately"
            })
        
        for trigger in sentiment.get("negative_triggers", []):
            risks.append({
                "risk": trigger.get("trigger", "Unknown trigger"),
                "source": "Sentiment - Negative Trigger",
                "severity": "medium",
                "mitigation": trigger.get("mitigation", "Develop counter-narrative")
            })
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        risks.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return {
            "total_risks": len(risks),
            "high_severity_count": len([r for r in risks if r.get("severity") == "high"]),
            "risks": risks[:15],  # Top 15 risks
            "overall_risk_level": "high" if any(r.get("severity") == "high" for r in risks) else "medium"
        }
