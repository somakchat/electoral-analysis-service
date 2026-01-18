"""
Intelligence Agent - Constituency Intelligence Specialist.

Specialization: Data Retrieval
Micro-Level Capabilities: Booth-level data, ward-wise patterns, historical trends by polling station
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import re

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class IntelligenceAgent(SpecialistAgent):
    """
    Constituency Intelligence Specialist.
    
    Gathers granular, booth-level political intelligence for any constituency.
    Understands historical voting patterns, local power structures, and ground realities.
    """
    
    name = "Intelligence Agent"
    role = "Constituency Intelligence Specialist"
    goal = "Gather granular, booth-level political intelligence for any constituency"
    backstory = """You are a field intelligence expert who understands every nuance of 
West Bengal politics down to the booth level. You know the historical voting patterns, 
local power structures, and ground realities of each assembly segment. You have access to:
- Booth-level electoral data
- Ward-wise voting patterns
- Historical trend analysis by polling station
- Local power broker networks
- Ground-level political dynamics"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Gather comprehensive constituency intelligence from evidence."""
        
        if not evidences:
            return {
                "key_facts": [],
                "historical_trends": [],
                "booth_level_data": [],
                "power_structures": [],
                "notable_local_issues": [],
                "data_gaps": ["No evidence available for analysis"]
            }
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        system = self._build_system_prompt()
        prompt = f"""Analyze the following evidence to create a comprehensive constituency intelligence brief.

QUERY: {query}

EVIDENCE:
{evidence_text}

Extract and organize the following (use ONLY information from evidence):

1. KEY FACTS: Important political facts about the constituency
2. HISTORICAL TRENDS: Voting patterns from past elections
3. BOOTH LEVEL DATA: Any booth-specific or polling station data
4. POWER STRUCTURES: Local leaders, influencers, power brokers
5. NOTABLE LOCAL ISSUES: Key issues affecting voters
6. DATA GAPS: What important information is missing

Return as JSON:
{{
    "key_facts": ["fact1", "fact2", ...],
    "historical_trends": ["trend1", "trend2", ...],
    "booth_level_data": [{{"booth_id": "...", "info": "..."}}],
    "power_structures": [{{"name": "...", "role": "...", "influence": "..."}}],
    "notable_local_issues": ["issue1", "issue2", ...],
    "data_gaps": ["gap1", "gap2", ...]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.1)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["key_facts", "historical_trends", "booth_level_data", 
                    "power_structures", "notable_local_issues", "data_gaps"]:
            if key not in content:
                content[key] = []
        
        return content


class HistoricalTrendTool:
    """Tool for analyzing historical voting trends."""
    
    name = "historical_trend_tool"
    description = "Analyze historical voting trends for a constituency across elections"
    
    def __init__(self, rag):
        self.rag = rag
    
    def run(self, constituency: str, elections: List[str] = None) -> Dict[str, Any]:
        """Query RAG for historical trend data."""
        query = f"Historical voting patterns and trends in {constituency}"
        if elections:
            query += f" for elections: {', '.join(elections)}"
        
        evidences = self.rag.search(query)
        
        return {
            "constituency": constituency,
            "data_found": len(evidences) > 0,
            "evidence_count": len(evidences),
            "raw_data": [{"text": e.text, "source": e.source_path} for e in evidences[:5]]
        }


class ConstituencyDataTool:
    """Tool for retrieving constituency-specific data."""
    
    name = "constituency_data_tool"
    description = "Retrieve detailed data about a specific constituency"
    
    def __init__(self, rag):
        self.rag = rag
    
    def run(self, constituency: str, data_type: str = "all") -> Dict[str, Any]:
        """Query RAG for constituency data."""
        query = f"Constituency data for {constituency}"
        if data_type != "all":
            query += f" focusing on {data_type}"
        
        evidences = self.rag.search(query)
        
        return {
            "constituency": constituency,
            "data_type": data_type,
            "data_found": len(evidences) > 0,
            "raw_data": [{"text": e.text, "source": e.source_path} for e in evidences[:5]]
        }
