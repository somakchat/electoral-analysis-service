"""
Ground Strategy Agent - Ground Operations Strategist.

Specialization: Field Operations
Micro-Level Capabilities: Rally locations, door-to-door coverage, local influencer mapping
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class GroundStrategyAgent(SpecialistAgent):
    """
    Ground Operations Strategist.
    
    Designs micro-level field campaign strategies for maximum voter contact.
    Expert in rally planning, door-to-door canvassing, and influencer engagement.
    """
    
    name = "Ground Strategy Agent"
    role = "Ground Operations Strategist"
    goal = "Design micro-level field campaign strategies for maximum voter contact"
    backstory = """You are a ground game expert who has organized grassroots campaigns. 
You know how to optimize rally locations, plan door-to-door canvassing routes, identify 
local influencers, and maximize ground presence efficiently. Your expertise includes:
- Rally and public meeting location optimization
- Door-to-door canvassing route planning
- Local influencer identification and engagement
- Booth-level worker deployment
- Last-mile voter mobilization strategies"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Design ground campaign strategy."""
        
        if not evidences:
            return {
                "rally_strategy": {},
                "canvassing_plan": {},
                "influencer_map": [],
                "booth_deployment": {},
                "mobilization_plan": {},
                "priority_areas": [],
                "data_gaps": ["No evidence available for ground strategy"]
            }
        
        # Get previous analysis if available
        previous_analysis = {}
        if context and context.previous_results:
            for result in context.previous_results:
                if "voter_segments" in result.content:
                    previous_analysis["voter_segments"] = result.content["voter_segments"]
                if "opponent_candidates" in result.content:
                    previous_analysis["opposition"] = result.content["opponent_candidates"]
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        context_text = ""
        if previous_analysis:
            context_text = f"\n\nPREVIOUS ANALYSIS:\n{previous_analysis}"
        
        system = self._build_system_prompt()
        prompt = f"""Design a comprehensive ground campaign strategy from the evidence.

QUERY: {query}
{context_text}

EVIDENCE:
{evidence_text}

Create a detailed ground operations plan:

1. RALLY STRATEGY: Optimal locations, timing, expected footfall
2. CANVASSING PLAN: Door-to-door routes, target areas, coverage schedule
3. INFLUENCER MAP: Local leaders, opinion makers to engage
4. BOOTH DEPLOYMENT: Worker allocation per booth
5. MOBILIZATION PLAN: Last-mile voter turnout strategy
6. PRIORITY AREAS: Areas needing most attention

Return as JSON:
{{
    "rally_strategy": {{
        "recommended_locations": [
            {{"location": "...", "rationale": "...", "expected_footfall": 5000, "optimal_timing": "..."}}
        ],
        "rally_sequence": ["location1", "location2"],
        "logistics_notes": "..."
    }},
    "canvassing_plan": {{
        "priority_wards": ["ward1", "ward2"],
        "routes": [
            {{"route_id": "R1", "areas_covered": ["..."], "households": 500, "schedule": "Day 1-3"}}
        ],
        "target_coverage": "80%",
        "team_size_needed": 50
    }},
    "influencer_map": [
        {{"name": "Local Leader", "influence_area": "Ward 5", "contact_priority": "high", "approach": "..."}}
    ],
    "booth_deployment": {{
        "total_booths": 287,
        "priority_booths": ["B1", "B2"],
        "workers_per_booth": 3,
        "training_needs": ["polling agent training", ...]
    }},
    "mobilization_plan": {{
        "d_minus_7": ["action1", "action2"],
        "d_minus_3": ["action3", "action4"],
        "d_day": ["action5", "action6"],
        "transport_arrangements": "...",
        "special_attention_areas": ["area1", "area2"]
    }},
    "priority_areas": [
        {{"area": "Urban slums", "priority": "high", "reason": "Low penetration", "recommended_action": "..."}}
    ],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.2)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["rally_strategy", "canvassing_plan", "influencer_map",
                    "booth_deployment", "mobilization_plan", "priority_areas", "data_gaps"]:
            if key not in content:
                content[key] = {} if key in ["rally_strategy", "canvassing_plan", 
                                              "booth_deployment", "mobilization_plan"] else []
        
        return content


class LocationOptimizerTool:
    """Tool for optimizing rally and meeting locations."""
    
    name = "location_optimizer_tool"
    description = "Optimize rally and public meeting locations"
    
    def run(
        self,
        constituency: str,
        voter_density_data: Dict[str, Any],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize locations based on voter density and constraints."""
        
        # Simple heuristic optimization
        locations = []
        
        # Prioritize high-density areas
        if "high_density_areas" in voter_density_data:
            for area in voter_density_data["high_density_areas"]:
                locations.append({
                    "location": area,
                    "priority": "high",
                    "rationale": "High voter density"
                })
        
        return {
            "constituency": constituency,
            "recommended_locations": locations[:5],
            "optimization_criteria": ["voter_density", "accessibility", "visibility"]
        }


class CoverageCalculatorTool:
    """Tool for calculating canvassing coverage."""
    
    name = "coverage_calculator_tool"
    description = "Calculate door-to-door canvassing coverage"
    
    def run(
        self,
        total_households: int,
        team_size: int,
        days_available: int,
        households_per_person_per_day: int = 30
    ) -> Dict[str, Any]:
        """Calculate coverage feasibility."""
        
        total_capacity = team_size * days_available * households_per_person_per_day
        coverage_percentage = min(100, (total_capacity / total_households) * 100)
        
        return {
            "total_households": total_households,
            "team_size": team_size,
            "days_available": days_available,
            "total_capacity": total_capacity,
            "coverage_percentage": round(coverage_percentage, 1),
            "feasibility": "Achievable" if coverage_percentage >= 80 else "Needs more resources",
            "additional_team_needed": max(0, int((total_households * 0.8 - total_capacity) / (days_available * households_per_person_per_day)))
        }
