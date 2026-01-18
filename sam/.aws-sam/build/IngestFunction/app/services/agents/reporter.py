"""
Strategic Reporter Agent - Strategic Reporter.

Specialization: Synthesis
Capabilities: Actionable briefs, risk alerts, daily strategy recommendations
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence, StrategyResult


class StrategicReporterAgent(SpecialistAgent):
    """
    Strategic Reporter.
    
    Synthesizes analysis from all specialist agents into comprehensive,
    actionable strategy documents with risk alerts and recommendations.
    """
    
    name = "Strategic Reporter Agent"
    role = "Strategic Reporter"
    goal = "Synthesize all analysis into actionable briefs, risk alerts, and strategy recommendations"
    backstory = """You are a strategic communications expert who synthesizes complex 
political intelligence into clear, actionable strategy documents. You compile insights 
from all specialist agents and create comprehensive briefs that campaign managers can 
act on immediately. Your expertise includes:
- Executive summary writing
- Risk identification and alerting
- Priority action sequencing
- Success metrics definition
- Contingency planning"""

    async def run(
        self,
        query: str,
        context: Optional[AgentContext] = None
    ) -> AgentResult:
        """Compile final strategy report from all agent analyses."""
        
        # Gather all previous results
        all_analyses = {}
        all_evidences = []
        
        if context and context.previous_results:
            for result in context.previous_results:
                all_analyses[result.agent] = result.content
                all_evidences.extend(result.evidences)
        
        # Compile the final report
        content = await self._compile_report(query, all_analyses, context)
        
        # Deduplicate evidences
        seen = set()
        unique_evidences = []
        for e in all_evidences:
            if e.chunk_id not in seen:
                seen.add(e.chunk_id)
                unique_evidences.append(e)
        
        return AgentResult(
            agent=self.name,
            content=content,
            evidences=unique_evidences[:20],
            confidence=self._calculate_confidence(content, unique_evidences)
        )

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Not used directly - run() overridden."""
        return {}

    async def _compile_report(
        self,
        query: str,
        all_analyses: Dict[str, Dict[str, Any]],
        context: Optional[AgentContext]
    ) -> Dict[str, Any]:
        """Compile comprehensive strategy report."""
        
        # Format all analyses for the prompt
        analyses_text = ""
        for agent, analysis in all_analyses.items():
            analyses_text += f"\n\n=== {agent} ===\n"
            analyses_text += str(analysis)[:2000]  # Limit each analysis
        
        constituency = context.constituency if context else "the constituency"
        party = context.party if context else "the party"
        
        system = self._build_system_prompt()
        prompt = f"""Compile a comprehensive micro-level political strategy from all specialist analyses.

ORIGINAL QUERY: {query}

CONSTITUENCY: {constituency}
PARTY: {party}

SPECIALIST ANALYSES:
{analyses_text}

Create a comprehensive strategy document with:

1. EXECUTIVE SUMMARY: 3-4 paragraph overview of the situation and key recommendations
2. CONSTITUENCY PROFILE: Key facts about the constituency
3. SWOT ANALYSIS: Synthesized from all analyses
4. VOTER SEGMENTS: Priority segments with targeting strategies
5. GROUND PLAN: Field operations blueprint
6. RESOURCE ALLOCATION: Budget and manpower recommendations
7. SCENARIOS: Key outcome scenarios
8. PRIORITY ACTIONS: Immediate, short-term, and medium-term
9. RISK FACTORS: Key risks and mitigation strategies
10. SUCCESS METRICS: How to measure campaign progress

Return as JSON:
{{
    "answer": "Full narrative answer synthesizing all insights...",
    "executive_summary": "3-4 paragraph strategic overview...",
    "constituency_profile": {{
        "name": "{constituency}",
        "total_voters": 234567,
        "booths": 287,
        "historical_trend": "TMC stronghold, BJP improved 12% in 2021",
        "key_demographics": {{}}
    }},
    "swot_analysis": {{
        "strengths": ["..."],
        "weaknesses": ["..."],
        "opportunities": ["..."],
        "threats": ["..."],
        "priority_actions": ["..."]
    }},
    "voter_segments": [
        {{
            "segment_name": "...",
            "population_share": 38,
            "current_support": "12%",
            "persuadability": "low",
            "key_issues": ["..."],
            "strategy": "..."
        }}
    ],
    "ground_plan": {{
        "priority_booths": ["..."],
        "rally_locations": ["..."],
        "influencer_targets": ["..."],
        "door_to_door_routes": ["..."]
    }},
    "resource_allocation": {{
        "total_recommended": "₹2.5 Cr",
        "breakdown": {{}}
    }},
    "scenarios": [
        {{
            "name": "Base Case",
            "projected_vote_share": "42%",
            "outcome": "Close loss by 3,000 votes"
        }},
        {{
            "name": "High Mobilization",
            "projected_vote_share": "46%",
            "outcome": "Narrow win by 5,000 votes"
        }}
    ],
    "priority_actions": [
        "Immediate: Announce local OBC candidate",
        "Week 1-2: Intensive booth-level worker deployment",
        "Week 3-4: Targeted social media campaign for youth"
    ],
    "risk_factors": [
        "Muslim consolidation against BJP",
        "Low youth turnout",
        "Opponent's strong ground game"
    ],
    "success_metrics": [
        "Achieve 50% booth-level coverage by week 2",
        "10000+ WhatsApp group members",
        "5 local influencer endorsements"
    ]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.2)
        content = self._extract_json(response.text)
        
        # Ensure answer field exists
        if "answer" not in content:
            content["answer"] = content.get("executive_summary", "Analysis complete. See detailed sections for strategy.")
        
        # Ensure all required keys exist
        required_keys = ["executive_summary", "constituency_profile", "swot_analysis",
                        "voter_segments", "ground_plan", "resource_allocation",
                        "scenarios", "priority_actions", "risk_factors", "success_metrics"]
        
        for key in required_keys:
            if key not in content:
                content[key] = {} if key in ["constituency_profile", "swot_analysis", 
                                              "ground_plan", "resource_allocation"] else []
        
        return content


class RiskAlertTool:
    """Tool for generating risk alerts."""
    
    name = "risk_alert_tool"
    description = "Generate and prioritize risk alerts"
    
    def run(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prioritize risks from analyses."""
        
        risks = []
        
        # Extract risks from different analyses
        if "opposition" in analyses:
            opp = analyses["opposition"]
            if "anti_incumbency_factors" in opp:
                for factor in opp["anti_incumbency_factors"]:
                    risks.append({"risk": factor, "source": "opposition", "severity": "medium"})
        
        if "sentiment" in analyses:
            sent = analyses["sentiment"]
            if sent.get("overall_mood") == "negative":
                risks.append({"risk": "Negative public mood", "source": "sentiment", "severity": "high"})
        
        if "data_scientist" in analyses:
            ds = analyses["data_scientist"]
            for projection in ds.get("projections", []):
                if "loss" in str(projection.get("outcome", "")).lower():
                    risks.append({"risk": f"Projected: {projection.get('outcome')}", 
                                 "source": "projections", "severity": "high"})
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        risks.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return {
            "total_risks": len(risks),
            "high_severity": len([r for r in risks if r.get("severity") == "high"]),
            "risks": risks[:10]
        }


class ActionSequencerTool:
    """Tool for sequencing priority actions."""
    
    name = "action_sequencer_tool"
    description = "Sequence and prioritize campaign actions"
    
    def run(
        self,
        actions: List[Dict[str, Any]],
        days_to_election: int = 30
    ) -> Dict[str, Any]:
        """Sequence actions into a timeline."""
        
        immediate = []  # Do now
        short_term = []  # Week 1-2
        medium_term = []  # Week 3-4
        
        for action in actions:
            urgency = action.get("urgency", "medium")
            if urgency == "high":
                immediate.append(action)
            elif urgency == "medium":
                short_term.append(action)
            else:
                medium_term.append(action)
        
        return {
            "days_to_election": days_to_election,
            "immediate": immediate[:5],
            "short_term": short_term[:5],
            "medium_term": medium_term[:5],
            "timeline": {
                "today": [a.get("action") for a in immediate[:3]],
                f"week_1_2": [a.get("action") for a in short_term[:3]],
                f"week_3_4": [a.get("action") for a in medium_term[:3]]
            }
        }
