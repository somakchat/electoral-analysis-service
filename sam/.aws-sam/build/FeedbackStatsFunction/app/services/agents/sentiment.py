"""
Sentiment Decoder Agent - Public Sentiment Decoder.

Specialization: Opinion Analysis
Micro-Level Capabilities: Issue-wise sentiment, leader perception, local grievances extraction
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.agents.base import SpecialistAgent, AgentResult, AgentContext
from app.models import Evidence


class SentimentDecoderAgent(SpecialistAgent):
    """
    Public Sentiment Decoder.
    
    Extracts and analyzes public opinion at granular issue and locality level.
    Decodes sentiment from survey data, social signals, and ground reports.
    """
    
    name = "Sentiment Decoder Agent"
    role = "Public Sentiment Decoder"
    goal = "Extract and analyze public opinion at granular issue and locality level"
    backstory = """You are a sentiment analysis expert who decodes public opinion from 
survey data, social signals, and ground reports. You identify key issues driving voter 
behavior, leader perceptions, and local grievances that can be addressed. Your expertise includes:
- Issue-wise sentiment analysis
- Leader perception measurement
- Local grievance extraction
- Mood tracking and trend analysis
- Social media sentiment decoding"""

    async def _analyze(
        self,
        query: str,
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Decode public sentiment from evidence."""
        
        if not evidences:
            return {
                "overall_mood": "Unknown",
                "issue_sentiments": [],
                "leader_perceptions": [],
                "local_grievances": [],
                "positive_triggers": [],
                "negative_triggers": [],
                "sentiment_trends": [],
                "data_gaps": ["No evidence available for sentiment analysis"]
            }
        
        # Compile evidence text
        evidence_text = "\n\n".join([
            f"[Source: {e.source_path}]\n{e.text}" 
            for e in evidences[:10]
        ])
        
        system = self._build_system_prompt()
        prompt = f"""Analyze public sentiment and opinion from the evidence.

QUERY: {query}

EVIDENCE:
{evidence_text}

Decode sentiment across multiple dimensions:

1. OVERALL MOOD: General public sentiment (positive/negative/mixed/neutral)
2. ISSUE SENTIMENTS: Sentiment on specific issues
3. LEADER PERCEPTIONS: How different leaders are perceived
4. LOCAL GRIEVANCES: Unaddressed problems causing dissatisfaction
5. POSITIVE TRIGGERS: What generates positive response
6. NEGATIVE TRIGGERS: What causes negative reaction
7. SENTIMENT TRENDS: How sentiment is changing over time

Return as JSON:
{{
    "overall_mood": "mixed",
    "mood_score": 0.45,  // 0-1 scale, 0.5 is neutral
    "issue_sentiments": [
        {{
            "issue": "Employment",
            "sentiment": "negative",
            "intensity": "high",
            "key_concerns": ["lack of jobs", "low wages"],
            "opportunity": "Promise skill development programs"
        }},
        {{
            "issue": "Development",
            "sentiment": "mixed",
            "intensity": "medium",
            "key_concerns": ["road conditions", "electricity"],
            "opportunity": "Highlight completed projects"
        }}
    ],
    "leader_perceptions": [
        {{
            "leader": "Current MLA",
            "party": "TMC",
            "perception": "negative",
            "reasons": ["inaccessible", "unfulfilled promises"],
            "approval_estimate": "35%"
        }},
        {{
            "leader": "BJP Candidate",
            "party": "BJP",
            "perception": "positive",
            "reasons": ["new face", "good speaker"],
            "approval_estimate": "42%"
        }}
    ],
    "local_grievances": [
        {{
            "grievance": "Poor road conditions in Block A",
            "affected_population": "~15000",
            "duration": "3 years",
            "political_opportunity": "Campaign promise + immediate action if possible"
        }},
        {{
            "grievance": "Water supply issues in Ward 5",
            "affected_population": "~8000",
            "duration": "1 year",
            "political_opportunity": "Highlight if resolved, address if ongoing"
        }}
    ],
    "positive_triggers": [
        {{"trigger": "Development projects", "response": "Moderate positive", "audience": "Urban voters"}},
        {{"trigger": "Welfare schemes", "response": "Strong positive", "audience": "Rural/poor"}}
    ],
    "negative_triggers": [
        {{"trigger": "Price rise", "response": "Strong negative", "audience": "All"}},
        {{"trigger": "Unemployment", "response": "Strong negative", "audience": "Youth"}}
    ],
    "sentiment_trends": [
        {{"timeframe": "Last 6 months", "trend": "Slight negative shift", "reason": "Price rise"}},
        {{"timeframe": "Last 1 month", "trend": "Stabilizing", "reason": "Election announcements"}}
    ],
    "data_gaps": ["gap1", "gap2"]
}}"""
        
        response = self.llm.generate(prompt, system=system, temperature=0.2)
        content = self._extract_json(response.text)
        
        # Ensure required keys exist
        for key in ["issue_sentiments", "leader_perceptions", "local_grievances",
                    "positive_triggers", "negative_triggers", "sentiment_trends", "data_gaps"]:
            if key not in content:
                content[key] = []
        
        if "overall_mood" not in content:
            content["overall_mood"] = "Unknown"
        
        return content


class SentimentAnalyzerTool:
    """Tool for analyzing sentiment in text."""
    
    name = "sentiment_analyzer_tool"
    description = "Analyze sentiment in text data"
    
    def run(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of given texts."""
        
        # Simple keyword-based sentiment (in production, use ML model)
        positive_words = {"good", "great", "excellent", "support", "happy", "satisfied", "progress", "development"}
        negative_words = {"bad", "poor", "corrupt", "angry", "disappointed", "failed", "problem", "issue"}
        
        results = []
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            results.append({
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment,
                "positive_signals": pos_count,
                "negative_signals": neg_count
            })
        
        # Overall summary
        sentiments = [r["sentiment"] for r in results]
        
        return {
            "analyzed_count": len(texts),
            "overall_sentiment": max(set(sentiments), key=sentiments.count) if sentiments else "neutral",
            "sentiment_distribution": {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            },
            "details": results[:5]  # Return first 5 details
        }


class GrievanceMapperTool:
    """Tool for mapping local grievances."""
    
    name = "grievance_mapper_tool"
    description = "Map and categorize local grievances"
    
    def run(self, grievances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize and prioritize grievances."""
        
        categories = {}
        for g in grievances:
            cat = g.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(g)
        
        # Prioritize by affected population
        prioritized = sorted(grievances, 
                            key=lambda x: int(x.get("affected_population", "0").replace("~", "").replace(",", "")),
                            reverse=True)
        
        return {
            "total_grievances": len(grievances),
            "by_category": {k: len(v) for k, v in categories.items()},
            "top_priority": prioritized[:5] if prioritized else [],
            "categories": list(categories.keys())
        }
