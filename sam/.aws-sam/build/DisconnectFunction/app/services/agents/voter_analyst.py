"""
Voter Analyst Agent - Voter Segmentation Expert.

Specialization: Demographic Analysis
Micro-Level Capabilities: Caste/community segments, age cohorts, occupation-based voting, first-time voters
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence, VoterSegment


class VoterAnalystAgent(SpecialistAgent):
    """
    Voter Segmentation Expert.
    
    Analyzes voter demographics and identifies micro-segments for targeted outreach.
    Understands caste dynamics, religious demographics, occupation-based voting patterns,
    and generational differences.
    """
    
    name = "Voter Analyst Agent"
    role = "Voter Segmentation Expert"
    goal = "Analyze voter demographics and identify micro-segments for targeted outreach"
    backstory = """You are a demographic analyst who can dissect voter populations into 
actionable segments. You understand caste dynamics, religious demographics, occupation-based 
voting patterns, and generational differences in West Bengal. Your expertise includes:
- Caste/community segment analysis
- Age cohort voting behavior
- Occupation-based political preferences
- First-time voter identification and targeting
- Rural vs urban voting patterns"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Perform voter segmentation analysis."""
        
        if not evidences:
            return {
                "total_voters_estimate": None,
                "demographic_breakdown": {},
                "voter_segments": [],
                "persuadable_segments": [],
                "turnout_patterns": [],
                "targeting_priorities": [],
                "data_gaps": ["No evidence available for voter analysis"]
            }
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        system = self._build_system_prompt()
        prompt = f"""Analyze voter demographics and segments from the evidence.

QUERY: {query}

EVIDENCE:
{evidence_text}

Create a comprehensive voter segmentation analysis:

1. TOTAL VOTERS ESTIMATE: If available in evidence
2. DEMOGRAPHIC BREAKDOWN: Caste, religion, age, occupation distributions
3. VOTER SEGMENTS: Detailed analysis of each segment
4. PERSUADABLE SEGMENTS: Which segments can be won over
5. TURNOUT PATTERNS: Expected turnout by segment
6. TARGETING PRIORITIES: Prioritized list for campaign focus

For each voter segment, provide:
- segment_name
- population_share (percentage if known, null if unknown)
- current_support (estimated support level)
- persuadability (low/medium/high)
- key_issues (what matters to them)
- strategy (how to approach)
- recommended_messaging
- outreach_channels

Return as JSON:
{{
    "total_voters_estimate": 234567 or null,
    "demographic_breakdown": {{
        "caste": {{"SC": "18%", "OBC": "32%", ...}},
        "religion": {{"Hindu": "65%", "Muslim": "32%", ...}},
        "age": {{"18-25": "22%", ...}},
        "occupation": {{"farmers": "35%", ...}}
    }},
    "voter_segments": [
        {{
            "segment_name": "Muslim voters",
            "population_share": 38,
            "current_support": "12%",
            "persuadability": "low",
            "key_issues": ["development", "security"],
            "strategy": "Focus on development, avoid polarization",
            "recommended_messaging": "...",
            "outreach_channels": ["local leaders", "mosques"]
        }}
    ],
    "persuadable_segments": ["segment1", "segment2"],
    "turnout_patterns": ["pattern1", "pattern2"],
    "targeting_priorities": ["priority1", "priority2"],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.15)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["voter_segments", "persuadable_segments", "turnout_patterns", 
                    "targeting_priorities", "data_gaps"]:
            if key not in content:
                content[key] = []
        
        if "demographic_breakdown" not in content:
            content["demographic_breakdown"] = {}
        
        return content


class DemographicQueryTool:
    """Tool for querying demographic data."""
    
    name = "demographic_query_tool"
    description = "Query demographic data for a constituency"
    
    def __init__(self, rag):
        self.rag = rag
    
    def run(self, constituency: str, demographic_type: str = "all") -> Dict[str, Any]:
        query = f"Demographic data {demographic_type} for {constituency}"
        evidences = self.rag.search(query)
        
        return {
            "constituency": constituency,
            "demographic_type": demographic_type,
            "data_found": len(evidences) > 0,
            "raw_data": [{"text": e.text, "source": e.source_path} for e in evidences[:5]]
        }


class SegmentCalculatorTool:
    """Tool for calculating segment sizes and priorities."""
    
    name = "segment_calculator_tool"
    description = "Calculate segment sizes and targeting priorities"
    
    def run(
        self, 
        total_voters: int, 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate segment metrics."""
        results = []
        for seg in segments:
            share = seg.get("population_share", 0) or 0
            absolute = int(total_voters * share / 100)
            persuadability = seg.get("persuadability", "medium")
            
            # Calculate priority score
            persuade_multiplier = {"low": 0.3, "medium": 0.6, "high": 1.0}
            priority_score = absolute * persuade_multiplier.get(persuadability, 0.5)
            
            results.append({
                "segment_name": seg.get("segment_name"),
                "absolute_voters": absolute,
                "priority_score": priority_score
            })
        
        # Sort by priority
        results.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return {
            "total_voters": total_voters,
            "segments": results
        }
