"""
Pydantic models for Political Strategy Maker API.
Defines request/response schemas and data structures.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


# ============= Enums =============

class WorkflowType(str, Enum):
    COMPREHENSIVE_STRATEGY = "comprehensive_strategy"
    QUICK_ANALYSIS = "quick_analysis"
    CONSTITUENCY_INTEL = "constituency_intel"
    VOTER_TARGETING = "voter_targeting"
    OPPOSITION_ANALYSIS = "opposition_analysis"


class DepthLevel(str, Enum):
    MICRO = "micro"  # Booth-level analysis
    MESO = "meso"    # Constituency-level
    MACRO = "macro"  # Regional/State-level


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    DELEGATING = "delegating"
    DONE = "done"
    ERROR = "error"


# ============= Core Data Models =============

class Evidence(BaseModel):
    """Evidence chunk from RAG retrieval with detailed citation info."""
    # Required fields
    source: str = ""  # Source file or identifier
    content: str = ""  # The evidence content/text
    
    # Optional fields for backward compatibility
    doc_id: Optional[str] = None
    source_path: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None  # Alias for content
    relevance_score: float = 0.0  # Score between 0-1
    metadata: Dict[str, Any] = {}
    
    # New fields for detailed citations
    source_type: Optional[str] = None  # Electoral Results, Prediction Model, Analysis, etc.
    constituencies: List[str] = []  # Constituencies referenced in this evidence
    districts: List[str] = []  # Districts referenced in this evidence
    methodology: Optional[str] = None  # How the data was derived
    data_points: Optional[str] = None  # What data points are included
    year: Optional[str] = None  # Year of the data
    
    def __init__(self, **data):
        # Handle backward compatibility
        if 'text' in data and 'content' not in data:
            data['content'] = data['text']
        if 'source_path' in data and 'source' not in data:
            data['source'] = data['source_path']
        if 'score' in data and 'relevance_score' not in data:
            data['relevance_score'] = data['score']
        super().__init__(**data)


class EntityReference(BaseModel):
    """Entity extracted from documents or queries."""
    entity_type: str  # constituency, candidate, party, issue, leader
    entity_name: str
    attributes: Dict[str, Any] = {}
    source_doc_ids: List[str] = []
    confidence: float = 0.0


class VoterSegment(BaseModel):
    """Voter segment analysis result."""
    segment_name: str
    population_share: Optional[float] = None
    current_support: Optional[str] = None
    persuadability: str = "medium"  # low, medium, high
    key_issues: List[str] = []
    strategy: Optional[str] = None
    recommended_messaging: Optional[str] = None
    outreach_channels: List[str] = []


class ScenarioResult(BaseModel):
    """Election scenario simulation result."""
    name: str
    projected_vote_share: str
    outcome: str
    confidence: float = 0.0
    key_assumptions: List[str] = []


class ResourceAllocation(BaseModel):
    """Resource allocation for a constituency."""
    constituency: str
    budget: float
    manpower: Optional[int] = None
    media_spend: Optional[float] = None
    priority_level: str = "medium"


class SWOTAnalysis(BaseModel):
    """SWOT analysis result."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    opportunities: List[str] = []
    threats: List[str] = []
    priority_actions: List[str] = []


# ============= API Request Models =============

class IngestRequest(BaseModel):
    """Request for document ingestion."""
    document_id: Optional[str] = None
    extract_entities: bool = True


class IngestResponse(BaseModel):
    """Response after document ingestion."""
    document_id: str
    chunks_indexed: int
    entities_extracted: int = 0
    file_name: Optional[str] = None
    # Per-index status (for UI visibility)
    index_status: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    """WebSocket chat request."""
    session_id: str = Field(..., description="Client session id")
    query: str
    workflow: WorkflowType = WorkflowType.COMPREHENSIVE_STRATEGY
    depth: DepthLevel = DepthLevel.MICRO
    include_scenarios: bool = True
    constituency: Optional[str] = None
    party: Optional[str] = None


# ============= Real-time Updates =============

class AgentUpdate(BaseModel):
    """Real-time agent activity update."""
    type: str = "agent_activity"
    agent: str
    status: AgentStatus
    task: str
    timestamp: Optional[datetime] = None
    details: Dict[str, Any] = {}


class DelegationUpdate(BaseModel):
    """Manager delegation update."""
    type: str = "delegation"
    from_agent: str
    to_agent: str
    task_description: str


# ============= Final Response Models =============

class ConstituencyProfile(BaseModel):
    """Constituency intelligence profile."""
    name: str
    total_voters: Optional[int] = None
    booths: Optional[int] = None
    historical_trend: Optional[str] = None
    key_demographics: Dict[str, Any] = {}


class GroundPlan(BaseModel):
    """Ground operations plan."""
    priority_booths: List[str] = []
    rally_locations: List[str] = []
    influencer_targets: List[str] = []
    door_to_door_routes: List[str] = []


class StrategyResult(BaseModel):
    """Complete micro-strategy output."""
    executive_summary: str = ""
    constituency_profile: Optional[ConstituencyProfile] = None
    swot_analysis: Optional[SWOTAnalysis] = None
    voter_segments: List[VoterSegment] = []
    ground_plan: Optional[GroundPlan] = None
    resource_allocation: Optional[Dict[str, Any]] = None
    scenarios: List[ScenarioResult] = []
    priority_actions: List[str] = []
    risk_factors: List[str] = []
    success_metrics: List[str] = []


class FinalResponse(BaseModel):
    """Final chat response with complete strategy."""
    type: str = "final_response"
    answer: str
    strategy: Optional[StrategyResult] = None
    citations: List[Evidence] = []
    agents_used: List[str] = []
    confidence: float = 0.0
    memory_stored: bool = False
    # Interactive chatbot fields (HITL)
    needs_clarification: bool = False
    interaction: Optional[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = []
    conversation_context: Optional[Dict[str, Any]] = None


# ============= Memory Models =============

class MemoryItem(BaseModel):
    """Single memory item."""
    memory_id: str
    session_id: str
    memory_type: str  # short_term, long_term, entity
    content: str
    embedding_text: Optional[str] = None
    metadata: Dict[str, Any] = {}
    relevance_score: float = 0.0
    timestamp: datetime
    ttl: Optional[int] = None  # Time to live in seconds


class SessionHistory(BaseModel):
    """Session conversation history."""
    session_id: str
    turns: List[Dict[str, Any]] = []
    entities_mentioned: List[EntityReference] = []
    created_at: datetime
    updated_at: datetime
