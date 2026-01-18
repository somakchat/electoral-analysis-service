"""
Political Strategy Maker - Advanced Agent Module.

This module implements evidence-based political strategy agents:

CORE FRAMEWORK:
- Evidence Framework - Base classes for evidence-based reasoning
- Strategic Orchestrator - Intelligent agent coordination

SPECIALIST AGENTS:
- Constituency Intelligence Analyst - Deep constituency analysis
- Electoral Strategist - Party-level strategy and victory paths
- Campaign Strategist - Ground operations and messaging

LEGACY AGENTS (for backward compatibility):
- Intelligence Agent, Voter Analyst, Opposition Research, etc.
"""

# Advanced evidence-based agents
from app.services.agents.evidence_framework import (
    PoliticalAgentBase, Evidence, Claim, ReasoningChain,
    EvidenceType, ConfidenceLevel, AgentCapability
)
from app.services.agents.constituency_analyst import ConstituencyIntelligenceAgent
from app.services.agents.electoral_strategist import ElectoralStrategistAgent
from app.services.agents.campaign_strategist import CampaignStrategistAgent
from app.services.agents.strategic_orchestrator import (
    StrategicOrchestrator, QueryAnalyzer, OrchestratedResponse
)
from app.services.agents.autonomous_orchestrator import (
    AutonomousPoliticalOrchestrator, AutonomousResponse, 
    create_autonomous_orchestrator, QueryComplexity, ResponseQuality
)

# Legacy agents (for backward compatibility)
from app.services.agents.intelligence import IntelligenceAgent
from app.services.agents.voter_analyst import VoterAnalystAgent
from app.services.agents.opposition import OppositionResearchAgent
from app.services.agents.ground import GroundStrategyAgent
from app.services.agents.resource import ResourceOptimizerAgent
from app.services.agents.sentiment import SentimentDecoderAgent
from app.services.agents.data_scientist import DataScientistAgent
from app.services.agents.reporter import StrategicReporterAgent

__all__ = [
    # Core framework
    "PoliticalAgentBase",
    "Evidence",
    "Claim",
    "ReasoningChain",
    "EvidenceType",
    "ConfidenceLevel",
    "AgentCapability",
    # Orchestrators
    "StrategicOrchestrator",
    "QueryAnalyzer",
    "OrchestratedResponse",
    # Autonomous Orchestrator (Enhanced)
    "AutonomousPoliticalOrchestrator",
    "AutonomousResponse",
    "create_autonomous_orchestrator",
    "QueryComplexity",
    "ResponseQuality",
    # Advanced agents
    "ConstituencyIntelligenceAgent",
    "ElectoralStrategistAgent",
    "CampaignStrategistAgent",
    # Legacy agents
    "IntelligenceAgent",
    "VoterAnalystAgent", 
    "OppositionResearchAgent",
    "GroundStrategyAgent",
    "ResourceOptimizerAgent",
    "SentimentDecoderAgent",
    "DataScientistAgent",
    "StrategicReporterAgent",
]
