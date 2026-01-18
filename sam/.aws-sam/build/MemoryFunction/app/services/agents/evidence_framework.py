"""
Evidence-Based Agent Framework - Foundation for human-like political reasoning.

This framework ensures:
1. Every claim is backed by data evidence
2. Logical reasoning chains are explicit
3. Confidence is calibrated based on evidence quality
4. Uncertainty is acknowledged transparently
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import json
import re

from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.rag.knowledge_graph import PoliticalKnowledgeGraph
from app.services.rag.data_schema import FactWithCitation, VerifiedAnswer, ConstituencyProfile
from app.services.llm import get_llm, LLMResponse


class EvidenceType(str, Enum):
    """Types of evidence for claims."""
    ELECTORAL_RESULT = "electoral_result"      # Hard data from elections
    PREDICTION_MODEL = "prediction_model"       # Model-based predictions
    SURVEY_DATA = "survey_data"                 # Survey responses
    HISTORICAL_TREND = "historical_trend"       # Historical patterns
    DEMOGRAPHIC = "demographic"                  # Population data
    SWING_ANALYSIS = "swing_analysis"           # Vote swing data
    EXPERT_INFERENCE = "expert_inference"       # Logical deduction
    AGGREGATED = "aggregated"                   # Computed from multiple sources


class ConfidenceLevel(str, Enum):
    """Confidence levels for claims."""
    CERTAIN = "certain"           # 95%+ - Direct data match
    HIGH = "high"                 # 80-95% - Strong evidence
    MODERATE = "moderate"         # 60-80% - Multiple indicators
    LOW = "low"                   # 40-60% - Limited evidence
    SPECULATIVE = "speculative"   # <40% - Inference with uncertainty


@dataclass
class Evidence:
    """A piece of evidence supporting a claim."""
    evidence_type: EvidenceType
    content: str
    source: str
    numerical_value: Optional[float] = None
    confidence: float = 1.0
    timestamp: str = ""
    
    def to_citation(self) -> str:
        """Format as inline citation."""
        conf_str = f"{self.confidence:.0%}" if self.confidence < 1.0 else ""
        return f"[{self.source}{', ' + conf_str if conf_str else ''}]"


@dataclass
class Claim:
    """A claim made by an agent with supporting evidence."""
    statement: str
    evidence: List[Evidence]
    confidence: ConfidenceLevel
    reasoning: str
    
    def get_confidence_score(self) -> float:
        """Calculate numerical confidence score."""
        base_scores = {
            ConfidenceLevel.CERTAIN: 0.95,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MODERATE: 0.70,
            ConfidenceLevel.LOW: 0.50,
            ConfidenceLevel.SPECULATIVE: 0.30
        }
        
        # Adjust based on evidence quality
        if not self.evidence:
            return 0.2
        
        avg_evidence_conf = sum(e.confidence for e in self.evidence) / len(self.evidence)
        return base_scores[self.confidence] * avg_evidence_conf
    
    def to_text(self) -> str:
        """Format claim with evidence."""
        citations = " ".join(e.to_citation() for e in self.evidence)
        return f"{self.statement} {citations}"


@dataclass
class ReasoningStep:
    """A single step in multi-step reasoning."""
    step_number: int
    action: str
    input_data: str
    output: str
    claims: List[Claim]
    duration_ms: int = 0


@dataclass
class ReasoningChain:
    """Complete chain of reasoning from question to answer."""
    question: str
    steps: List[ReasoningStep]
    final_answer: str
    total_claims: int
    average_confidence: float
    sources_used: List[str]
    
    def to_explanation(self) -> str:
        """Generate human-readable explanation."""
        lines = [f"**Question:** {self.question}\n"]
        
        for step in self.steps:
            lines.append(f"\n**Step {step.step_number}: {step.action}**")
            if step.claims:
                for claim in step.claims:
                    lines.append(f"- {claim.to_text()}")
        
        lines.append(f"\n**Conclusion:**\n{self.final_answer}")
        lines.append(f"\n*Confidence: {self.average_confidence:.0%} | Sources: {len(self.sources_used)}*")
        
        return "\n".join(lines)


@dataclass
class AgentCapability:
    """Definition of what an agent can do."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    required_data: List[str]


class PoliticalAgentBase:
    """
    Base class for evidence-based political strategy agents.
    
    All specialized agents inherit from this to ensure:
    - RAG integration for data access
    - Evidence-based claim generation
    - Explicit reasoning chains
    - Calibrated confidence
    """
    
    # Agent identity
    name: str = "Base Agent"
    role: str = "Generic political analyst"
    expertise: List[str] = []
    
    # Capabilities
    capabilities: List[AgentCapability] = []
    
    def __init__(self, rag: PoliticalRAGSystem):
        self.rag = rag
        self.kg = rag.kg
        self.llm = get_llm()
        self.reasoning_chain: List[ReasoningStep] = []
        self.claims: List[Claim] = []
        self.step_counter = 0
    
    def reset(self):
        """Reset reasoning state for new query."""
        self.reasoning_chain = []
        self.claims = []
        self.step_counter = 0
    
    # ================================================================
    # DATA ACCESS METHODS - All data access goes through these
    # ================================================================
    
    def get_constituency_data(self, name: str) -> Optional[ConstituencyProfile]:
        """Get verified constituency data."""
        return self.kg.get_constituency(name)
    
    def get_constituencies_by_filter(self, 
                                     district: str = None,
                                     pc: str = None,
                                     winner_2021: str = None,
                                     race_rating: str = None) -> List[ConstituencyProfile]:
        """Get filtered constituency list."""
        results = []
        for profile in self.kg.constituency_profiles.values():
            if district and profile.district.upper() != district.upper():
                continue
            if pc and profile.parent_pc.upper() != pc.upper():
                continue
            if winner_2021 and profile.winner_2021.upper() != winner_2021.upper():
                continue
            if race_rating and profile.race_rating.lower() != race_rating.lower():
                continue
            results.append(profile)
        return results
    
    def get_party_seats(self, party: str, year: int = 2021) -> List[ConstituencyProfile]:
        """Get seats won by a party."""
        return self.kg.get_constituencies_by_winner(party)
    
    def get_vulnerable_seats(self, party: str) -> List[ConstituencyProfile]:
        """Get vulnerable seats for a party."""
        return self.kg.get_vulnerable_seats(party)
    
    def get_swing_seats(self, margin_threshold: float = 5.0) -> List[ConstituencyProfile]:
        """Get swing/battleground seats."""
        return self.kg.get_swing_seats(margin_threshold)
    
    def get_facts(self, entity: str) -> List[FactWithCitation]:
        """Get verified facts about an entity."""
        return self.kg.get_facts_for_entity(entity)
    
    def search_data(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search with verification."""
        return self.rag.search(query, top_k=top_k)
    
    # ================================================================
    # EVIDENCE CREATION METHODS
    # ================================================================
    
    def create_evidence_from_profile(self, 
                                     profile: ConstituencyProfile,
                                     aspect: str) -> Evidence:
        """Create evidence from constituency profile."""
        if aspect == "2021_winner":
            return Evidence(
                evidence_type=EvidenceType.ELECTORAL_RESULT,
                content=f"{profile.ac_name} won by {profile.winner_2021} with TMC {profile.tmc_vote_share_2021:.2f}% vs BJP {profile.bjp_vote_share_2021:.2f}%",
                source=profile.source_files[0] if profile.source_files else "electoral_data",
                numerical_value=profile.margin_2021,
                confidence=1.0
            )
        elif aspect == "2026_prediction":
            return Evidence(
                evidence_type=EvidenceType.PREDICTION_MODEL,
                content=f"{profile.ac_name} predicted {profile.predicted_winner_2026} by {abs(profile.predicted_margin_2026):.2f}% ({profile.race_rating})",
                source=profile.source_files[0] if profile.source_files else "prediction_model",
                numerical_value=profile.predicted_margin_2026,
                confidence=0.8  # Predictions inherently less certain
            )
        elif aspect == "swing":
            direction = "towards TMC" if profile.pc_swing_2019_2024 > 0 else "towards BJP"
            return Evidence(
                evidence_type=EvidenceType.SWING_ANALYSIS,
                content=f"{profile.parent_pc} PC swung {abs(profile.pc_swing_2019_2024):.2f}% {direction} (2019-2024)",
                source="lok_sabha_comparison",
                numerical_value=profile.pc_swing_2019_2024,
                confidence=1.0
            )
        else:
            return Evidence(
                evidence_type=EvidenceType.ELECTORAL_RESULT,
                content=f"Data for {profile.ac_name}",
                source="electoral_data",
                confidence=0.9
            )
    
    def create_evidence_from_fact(self, fact: FactWithCitation) -> Evidence:
        """Create evidence from a verified fact."""
        type_map = {
            "electoral_result": EvidenceType.ELECTORAL_RESULT,
            "prediction": EvidenceType.PREDICTION_MODEL,
            "survey": EvidenceType.SURVEY_DATA,
            "swing_analysis": EvidenceType.SWING_ANALYSIS,
            "vulnerability": EvidenceType.HISTORICAL_TREND
        }
        
        return Evidence(
            evidence_type=type_map.get(fact.fact_type, EvidenceType.ELECTORAL_RESULT),
            content=fact.fact_text,
            source=fact.source_file,
            numerical_value=fact.numerical_value,
            confidence=fact.confidence
        )
    
    def create_aggregated_evidence(self,
                                  description: str,
                                  data_points: List[Any],
                                  source: str = "aggregated_analysis") -> Evidence:
        """Create evidence from aggregated data."""
        return Evidence(
            evidence_type=EvidenceType.AGGREGATED,
            content=description,
            source=source,
            numerical_value=len(data_points),
            confidence=0.9
        )
    
    # ================================================================
    # CLAIM CREATION METHODS
    # ================================================================
    
    def make_claim(self,
                   statement: str,
                   evidence: List[Evidence],
                   reasoning: str = "") -> Claim:
        """Create a claim with evidence and determine confidence."""
        # Determine confidence based on evidence quality
        if not evidence:
            confidence = ConfidenceLevel.SPECULATIVE
        else:
            avg_conf = sum(e.confidence for e in evidence) / len(evidence)
            has_electoral = any(e.evidence_type == EvidenceType.ELECTORAL_RESULT for e in evidence)
            has_multiple = len(evidence) >= 2
            
            if avg_conf >= 0.95 and has_electoral:
                confidence = ConfidenceLevel.CERTAIN
            elif avg_conf >= 0.8 and (has_electoral or has_multiple):
                confidence = ConfidenceLevel.HIGH
            elif avg_conf >= 0.6:
                confidence = ConfidenceLevel.MODERATE
            elif avg_conf >= 0.4:
                confidence = ConfidenceLevel.LOW
            else:
                confidence = ConfidenceLevel.SPECULATIVE
        
        claim = Claim(
            statement=statement,
            evidence=evidence,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.claims.append(claim)
        return claim
    
    # ================================================================
    # REASONING METHODS
    # ================================================================
    
    def add_reasoning_step(self,
                          action: str,
                          input_data: str,
                          output: str,
                          claims: List[Claim] = None) -> ReasoningStep:
        """Add a step to the reasoning chain."""
        self.step_counter += 1
        
        step = ReasoningStep(
            step_number=self.step_counter,
            action=action,
            input_data=input_data,
            output=output,
            claims=claims or []
        )
        
        self.reasoning_chain.append(step)
        return step
    
    def get_reasoning_chain(self) -> ReasoningChain:
        """Get complete reasoning chain."""
        all_claims = [c for step in self.reasoning_chain for c in step.claims]
        avg_conf = sum(c.get_confidence_score() for c in all_claims) / len(all_claims) if all_claims else 0.5
        
        sources = set()
        for step in self.reasoning_chain:
            for claim in step.claims:
                for ev in claim.evidence:
                    sources.add(ev.source)
        
        return ReasoningChain(
            question="",  # Set by caller
            steps=self.reasoning_chain,
            final_answer="",  # Set by caller
            total_claims=len(all_claims),
            average_confidence=avg_conf,
            sources_used=list(sources)
        )
    
    # ================================================================
    # LLM INTERACTION WITH GROUNDING
    # ================================================================
    
    def query_llm_grounded(self,
                          prompt: str,
                          context_data: Dict[str, Any],
                          task: str = "analysis") -> str:
        """
        Query LLM with grounded context - prevents hallucination.
        """
        # Build context from verified data
        context_parts = []
        
        for key, value in context_data.items():
            if isinstance(value, list):
                context_parts.append(f"\n{key.upper()}:")
                for item in value[:20]:  # Limit items
                    if isinstance(item, ConstituencyProfile):
                        context_parts.append(f"  - {item.ac_name}: {item.winner_2021} (2021), predicted {item.predicted_winner_2026} (2026)")
                    elif isinstance(item, Evidence):
                        context_parts.append(f"  - {item.content}")
                    elif isinstance(item, Claim):
                        context_parts.append(f"  - {item.statement}")
                    else:
                        context_parts.append(f"  - {str(item)[:200]}")
            else:
                context_parts.append(f"{key.upper()}: {value}")
        
        context_str = "\n".join(context_parts)
        
        system_prompt = f"""You are {self.name}, a {self.role}.

CRITICAL RULES:
1. ONLY use information from the VERIFIED DATA below
2. NEVER invent statistics, names, or results
3. If data is insufficient, say "Based on available data..." 
4. Include specific numbers and sources when making claims
5. Acknowledge uncertainty with phrases like "likely", "appears to", "data suggests"

YOUR EXPERTISE: {', '.join(self.expertise)}"""

        full_prompt = f"""VERIFIED DATA:
{context_str}

TASK: {task}

{prompt}

Provide your analysis based ONLY on the verified data above. Include specific evidence for each claim."""

        try:
            response = self.llm.generate(
                full_prompt,
                system=system_prompt,
                temperature=0.3  # Low temperature for factual accuracy
            )
            return response.text
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    # ================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ================================================================
    
    def analyze(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis method - must be implemented by subclasses.
        
        Returns:
            Dict with 'answer', 'claims', 'evidence', 'confidence', 'reasoning'
        """
        raise NotImplementedError("Subclasses must implement analyze()")
    
    def can_handle(self, query: str) -> Tuple[bool, float]:
        """
        Check if this agent can handle the query.
        
        Returns:
            Tuple of (can_handle, confidence)
        """
        raise NotImplementedError("Subclasses must implement can_handle()")


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def calculate_seat_metrics(constituencies: List[ConstituencyProfile]) -> Dict[str, Any]:
    """Calculate aggregate metrics for a list of constituencies."""
    if not constituencies:
        return {}
    
    tmc_2021 = sum(1 for c in constituencies if c.winner_2021.upper() in ['TMC', 'AITC'])
    bjp_2021 = sum(1 for c in constituencies if c.winner_2021.upper() == 'BJP')
    
    tmc_2026 = sum(1 for c in constituencies if c.predicted_winner_2026.upper() == 'TMC')
    bjp_2026 = sum(1 for c in constituencies if c.predicted_winner_2026.upper() == 'BJP')
    
    swing_seats = [c for c in constituencies if c.race_rating.lower() in ['toss-up', 'lean']]
    safe_seats = [c for c in constituencies if c.race_rating.lower() == 'safe']
    
    avg_swing = sum(c.pc_swing_2019_2024 for c in constituencies) / len(constituencies)
    
    return {
        "total": len(constituencies),
        "tmc_2021": tmc_2021,
        "bjp_2021": bjp_2021,
        "tmc_2026": tmc_2026,
        "bjp_2026": bjp_2026,
        "swing_seats": len(swing_seats),
        "safe_seats": len(safe_seats),
        "avg_swing": avg_swing,
        "swing_direction": "TMC" if avg_swing > 0 else "BJP"
    }


def format_constituency_brief(profile: ConstituencyProfile) -> str:
    """Format brief constituency description."""
    return f"{profile.ac_name} ({profile.district}): {profile.winner_2021}→{profile.predicted_winner_2026} [{profile.race_rating}]"

