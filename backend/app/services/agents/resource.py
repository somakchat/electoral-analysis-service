"""
Resource Optimizer Agent - Campaign Resource Optimizer.

Specialization: Budget & Manpower
Micro-Level Capabilities: Fund allocation by constituency, volunteer deployment, media spend ROI
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class ResourceOptimizerAgent(SpecialistAgent):
    """
    Campaign Resource Optimizer.
    
    Optimizes allocation of funds, manpower, and media across constituencies.
    Maximizes campaign ROI through data-driven resource distribution.
    """
    
    name = "Resource Optimizer Agent"
    role = "Campaign Resource Optimizer"
    goal = "Optimize allocation of funds, manpower, and media across constituencies"
    backstory = """You are a resource allocation expert who maximizes campaign ROI. 
You determine optimal fund distribution, volunteer deployment strategies, and media 
spending based on constituency-level win probability. Your expertise includes:
- Budget allocation optimization
- ROI calculation for campaign activities
- Manpower deployment planning
- Media spend optimization
- Resource-to-impact modeling"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Optimize resource allocation."""
        
        if not evidences:
            return {
                "budget_allocation": {},
                "manpower_plan": {},
                "media_strategy": {},
                "roi_projections": {},
                "efficiency_recommendations": [],
                "data_gaps": ["No evidence available for resource optimization"]
            }
        
        # Get previous analysis if available
        previous_analysis = {}
        if context and context.previous_results:
            for result in context.previous_results:
                if "priority_areas" in result.content:
                    previous_analysis["ground_priorities"] = result.content["priority_areas"]
                if "voter_segments" in result.content:
                    previous_analysis["segments"] = result.content["voter_segments"]
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        context_text = ""
        if previous_analysis:
            context_text = f"\n\nPREVIOUS ANALYSIS:\n{previous_analysis}"
        
        system = self._build_system_prompt()
        prompt = f"""Create an optimized resource allocation plan from the evidence.

QUERY: {query}
{context_text}

EVIDENCE:
{evidence_text}

Design a comprehensive resource optimization plan:

1. BUDGET ALLOCATION: How to distribute campaign funds
2. MANPOWER PLAN: Volunteer and worker deployment
3. MEDIA STRATEGY: Media spend across channels
4. ROI PROJECTIONS: Expected return on each investment
5. EFFICIENCY RECOMMENDATIONS: How to maximize impact

Return as JSON:
{{
    "budget_allocation": {{
        "total_recommended": "₹2.5 Cr",
        "breakdown": [
            {{"activity": "Ground operations", "amount": "₹80 Lakhs", "percentage": 32, "priority": "high"}},
            {{"activity": "Media & advertising", "amount": "₹50 Lakhs", "percentage": 20, "priority": "medium"}},
            {{"activity": "Rally & events", "amount": "₹40 Lakhs", "percentage": 16, "priority": "high"}},
            {{"activity": "Social media", "amount": "₹30 Lakhs", "percentage": 12, "priority": "high"}},
            {{"activity": "Logistics", "amount": "₹30 Lakhs", "percentage": 12, "priority": "medium"}},
            {{"activity": "Contingency", "amount": "₹20 Lakhs", "percentage": 8, "priority": "low"}}
        ],
        "phase_wise": {{
            "phase_1_launch": "30%",
            "phase_2_mobilization": "45%",
            "phase_3_final_push": "25%"
        }}
    }},
    "manpower_plan": {{
        "total_volunteers_needed": 500,
        "deployment": [
            {{"role": "Booth workers", "count": 300, "per_booth": 3}},
            {{"role": "Canvassers", "count": 100, "coverage": "door-to-door"}},
            {{"role": "Rally coordinators", "count": 50, "per_rally": 10}},
            {{"role": "Social media team", "count": 20, "platform": "all"}},
            {{"role": "Logistics support", "count": 30, "function": "transport, materials"}}
        ],
        "training_schedule": "Week 1-2: Training, Week 3+: Deployment",
        "coordination_structure": "Zone-wise with block coordinators"
    }},
    "media_strategy": {{
        "total_media_spend": "₹50 Lakhs",
        "channel_allocation": [
            {{"channel": "TV ads", "spend": "₹15 Lakhs", "rationale": "Mass reach"}},
            {{"channel": "Social media ads", "spend": "₹15 Lakhs", "rationale": "Youth targeting"}},
            {{"channel": "Print ads", "spend": "₹10 Lakhs", "rationale": "Local newspapers"}},
            {{"channel": "Radio", "spend": "₹5 Lakhs", "rationale": "Rural reach"}},
            {{"channel": "Outdoor/banners", "spend": "₹5 Lakhs", "rationale": "Visibility"}}
        ],
        "timing": "Peak spend in last 2 weeks"
    }},
    "roi_projections": {{
        "expected_vote_gain": "4-6%",
        "cost_per_vote_estimate": "₹150-200",
        "high_roi_activities": ["Social media in urban", "Ground game in rural"],
        "low_roi_activities": ["Generic TV ads"]
    }},
    "efficiency_recommendations": [
        {{"recommendation": "Focus 40% budget on top 50 swing booths", "expected_impact": "2% vote share gain"}},
        {{"recommendation": "Redirect print to WhatsApp campaign", "expected_impact": "30% cost saving"}},
        {{"recommendation": "Deploy experienced workers in weak areas", "expected_impact": "Better conversion"}}
    ],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.2)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["budget_allocation", "manpower_plan", "media_strategy",
                    "roi_projections", "efficiency_recommendations", "data_gaps"]:
            if key not in content:
                content[key] = {} if key != "efficiency_recommendations" and key != "data_gaps" else []
        
        return content


class BudgetAllocatorTool:
    """Tool for optimal budget allocation using linear programming."""
    
    name = "budget_allocator_tool"
    description = "Optimize budget allocation across constituencies"
    
    def run(
        self,
        total_budget: float,
        constituencies: List[Dict[str, Any]],
        objective: str = "maximize_seats"
    ) -> Dict[str, Any]:
        """Allocate budget optimally."""
        
        if not constituencies:
            return {"total_budget": total_budget, "allocation": []}
        
        # Simple allocation based on win probability and margin
        allocations = []
        total_weight = sum(c.get("marginal_impact", 1) for c in constituencies)
        
        for const in constituencies:
            weight = const.get("marginal_impact", 1)
            allocated = total_budget * (weight / total_weight)
            
            allocations.append({
                "constituency": const.get("name", "Unknown"),
                "allocated_budget": round(allocated, 2),
                "win_probability": const.get("win_probability", 0.5),
                "priority": "high" if weight > 1.5 else "medium" if weight > 0.8 else "low"
            })
        
        # Sort by priority
        allocations.sort(key=lambda x: x["allocated_budget"], reverse=True)
        
        return {
            "total_budget": total_budget,
            "objective": objective,
            "allocation": allocations,
            "expected_seats": sum(1 for a in allocations if a["win_probability"] > 0.5)
        }


class ROICalculatorTool:
    """Tool for calculating campaign ROI."""
    
    name = "roi_calculator_tool"
    description = "Calculate ROI for campaign activities"
    
    def run(
        self,
        investment: float,
        expected_vote_gain_percentage: float,
        total_voters: int,
        cost_per_vote_target: float = 200
    ) -> Dict[str, Any]:
        """Calculate ROI metrics."""
        
        votes_gained = total_voters * (expected_vote_gain_percentage / 100)
        actual_cost_per_vote = investment / max(1, votes_gained)
        roi_ratio = cost_per_vote_target / actual_cost_per_vote
        
        return {
            "investment": investment,
            "expected_votes_gained": int(votes_gained),
            "cost_per_vote": round(actual_cost_per_vote, 2),
            "target_cost_per_vote": cost_per_vote_target,
            "roi_ratio": round(roi_ratio, 2),
            "assessment": "Good ROI" if roi_ratio >= 1 else "Needs optimization"
        }
