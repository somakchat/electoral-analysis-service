"""
Political Data Schema - Structured data models for zero-hallucination RAG.

This module defines strongly-typed data structures that ensure:
1. All facts are traceable to source documents
2. Numerical data is stored with precision
3. Relationships between entities are explicit
4. Temporal context is preserved
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum


class ElectionType(str, Enum):
    """Types of elections tracked."""
    ASSEMBLY = "assembly"
    LOK_SABHA = "lok_sabha"
    PANCHAYAT = "panchayat"
    MUNICIPAL = "municipal"


class PartyCode(str, Enum):
    """Major party codes in West Bengal."""
    TMC = "TMC"      # Trinamool Congress (AITC)
    BJP = "BJP"      # Bharatiya Janata Party
    INC = "INC"      # Indian National Congress
    CPM = "CPM"      # Communist Party of India (Marxist)
    CPIM = "CPIM"    # Alias for CPM
    AIFB = "AIFB"    # All India Forward Bloc
    RSP = "RSP"      # Revolutionary Socialist Party
    CPI = "CPI"      # Communist Party of India
    SUCI = "SUCI"    # Socialist Unity Centre of India
    BSP = "BSP"      # Bahujan Samaj Party
    IND = "IND"      # Independent
    NOTA = "NOTA"    # None of the Above
    OTHER = "OTHER"  # Other parties


class ConstituencyType(str, Enum):
    """Reservation status of constituency."""
    GENERAL = "GEN"
    SC = "SC"        # Scheduled Caste
    ST = "ST"        # Scheduled Tribe


class VulnerabilityTag(str, Enum):
    """Seat vulnerability classification."""
    SAFE = "safe"
    LIKELY = "likely"
    LEAN = "lean"
    TOSSUP = "toss-up"
    AT_RISK = "at_risk"
    VULNERABLE = "vulnerable"


@dataclass
class ElectionResult:
    """A single election result with full provenance."""
    constituency_name: str
    year: int
    election_type: ElectionType
    winner_party: str
    winner_candidate: Optional[str]
    winner_votes: int
    total_valid_votes: int
    vote_share_percent: float
    margin_votes: int
    margin_percent: float
    runner_up_party: Optional[str] = None
    runner_up_candidate: Optional[str] = None
    runner_up_votes: Optional[int] = None
    turnout_percent: Optional[float] = None
    total_electors: Optional[int] = None
    source_file: str = ""
    source_row: int = 0
    
    @property
    def citation(self) -> str:
        """Generate citation for this data point."""
        return f"[Source: {self.source_file}, Row: {self.source_row}]"


@dataclass  
class ConstituencyProfile:
    """Complete constituency profile with all electoral history."""
    ac_no: int
    ac_name: str
    district: str
    constituency_type: ConstituencyType
    parent_pc: str  # Parent Parliamentary Constituency
    
    # 2021 Assembly results
    winner_2021: str
    tmc_vote_share_2021: float
    bjp_vote_share_2021: float
    margin_2021: float
    
    # 2019 Lok Sabha segment data
    pc_tmc_vs_2019: float
    pc_bjp_vs_2019: float
    
    # 2024 Lok Sabha segment data
    pc_tmc_vs_2024: float
    pc_bjp_vs_2024: float
    
    # Swing analysis
    pc_swing_2019_2024: float
    
    # 2026 Prediction
    predicted_margin_2026: float
    predicted_winner_2026: str
    race_rating: str
    
    # Vulnerability assessment
    vulnerability_tag: Optional[str] = None
    swing_history: Optional[str] = None
    
    # Source tracking
    source_files: List[str] = field(default_factory=list)
    
    def to_natural_text(self) -> str:
        """Convert to searchable natural language text."""
        lines = [
            f"=== {self.ac_name} ({self.ac_no}) CONSTITUENCY PROFILE ===",
            f"District: {self.district} | Type: {self.constituency_type.value} | Parent PC: {self.parent_pc}",
            "",
            "--- 2021 ASSEMBLY ELECTION ---",
            f"Winner: {self.winner_2021}",
            f"TMC Vote Share: {self.tmc_vote_share_2021:.2f}%",
            f"BJP Vote Share: {self.bjp_vote_share_2021:.2f}%",
            f"Margin (TMC-BJP): {self.margin_2021:.2f}%",
            "",
            "--- LOK SABHA TRENDS ({self.parent_pc}) ---",
            f"2019: TMC {self.pc_tmc_vs_2019:.2f}% vs BJP {self.pc_bjp_vs_2019:.2f}%",
            f"2024: TMC {self.pc_tmc_vs_2024:.2f}% vs BJP {self.pc_bjp_vs_2024:.2f}%",
            f"Swing (2019→2024): {self.pc_swing_2019_2024:.2f}% towards {'TMC' if self.pc_swing_2019_2024 > 0 else 'BJP'}",
            "",
            "--- 2026 PREDICTION ---",
            f"Predicted Winner: {self.predicted_winner_2026}",
            f"Predicted Margin: {abs(self.predicted_margin_2026):.2f}% ({self.race_rating})",
        ]
        
        if self.vulnerability_tag:
            lines.append(f"Vulnerability: {self.vulnerability_tag}")
        if self.swing_history:
            lines.append(f"Swing History: {self.swing_history}")
            
        return "\n".join(lines)
    
    @property
    def is_swing_seat(self) -> bool:
        """Check if this is a swing/toss-up seat."""
        return self.race_rating.lower() in ["toss-up", "lean"]
    
    @property
    def is_bjp_at_risk(self) -> bool:
        """Check if BJP seat is at risk."""
        return self.winner_2021 == "BJP" and self.predicted_winner_2026 == "TMC"
    
    @property
    def is_tmc_at_risk(self) -> bool:
        """Check if TMC seat is at risk."""
        return self.winner_2021 in ["TMC", "AITC"] and self.predicted_winner_2026 == "BJP"


@dataclass
class ParliamentaryConstituency:
    """Parliamentary constituency with segment-wise data."""
    pc_name: str
    state: str
    segments: List[str]  # List of AC names
    
    # 2019 Results
    winner_2019: str
    tmc_votes_2019: int
    bjp_votes_2019: int
    tmc_vote_share_2019: float
    bjp_vote_share_2019: float
    margin_2019: int
    
    # 2024 Results
    winner_2024: str
    tmc_votes_2024: int
    bjp_votes_2024: int
    tmc_vote_share_2024: float
    bjp_vote_share_2024: float
    margin_2024: int
    
    # Swing
    swing_percentage: float  # Positive = towards TMC
    
    source_files: List[str] = field(default_factory=list)
    
    def to_natural_text(self) -> str:
        """Convert to searchable text."""
        return f"""
=== {self.pc_name} PARLIAMENTARY CONSTITUENCY ===
Segments: {', '.join(self.segments)}

2019 LOK SABHA:
Winner: {self.winner_2019}
TMC: {self.tmc_vote_share_2019:.2f}% ({self.tmc_votes_2019:,} votes)
BJP: {self.bjp_vote_share_2019:.2f}% ({self.bjp_votes_2019:,} votes)
Margin: {self.margin_2019:,} votes

2024 LOK SABHA:
Winner: {self.winner_2024}
TMC: {self.tmc_vote_share_2024:.2f}% ({self.tmc_votes_2024:,} votes)
BJP: {self.bjp_vote_share_2024:.2f}% ({self.bjp_votes_2024:,} votes)
Margin: {self.margin_2024:,} votes

SWING: {abs(self.swing_percentage):.2f}% towards {'TMC' if self.swing_percentage > 0 else 'BJP'}
"""


@dataclass
class CandidateProfile:
    """Individual candidate with electoral history."""
    name: str
    party: str
    constituency: str
    year: int
    election_type: ElectionType
    
    # Results
    position: int
    votes: int
    vote_share: float
    won: bool
    
    # Demographics
    age: Optional[int] = None
    sex: Optional[str] = None
    education: Optional[str] = None
    profession: Optional[str] = None
    
    # Political history
    incumbent: bool = False
    terms_served: int = 0
    previous_party: Optional[str] = None
    turncoat: bool = False
    
    source_file: str = ""


@dataclass
class SurveyResponse:
    """Aggregated survey response data."""
    survey_name: str
    question: str
    total_responses: int
    response_date: Optional[datetime] = None
    
    # Aggregated results
    response_distribution: Dict[str, int] = field(default_factory=dict)
    response_percentages: Dict[str, float] = field(default_factory=dict)
    
    # Geographic breakdown (if available)
    constituency_responses: Dict[str, Dict[str, int]] = field(default_factory=dict)
    district_responses: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    source_file: str = ""
    
    def to_natural_text(self) -> str:
        """Convert to searchable text."""
        lines = [
            f"=== SURVEY: {self.survey_name} ===",
            f"Question: {self.question}",
            f"Total Responses: {self.total_responses}",
            "",
            "RESPONSE BREAKDOWN:"
        ]
        
        for response, count in sorted(self.response_distribution.items(), 
                                      key=lambda x: x[1], reverse=True):
            pct = self.response_percentages.get(response, 0)
            lines.append(f"  {response}: {count} ({pct:.1f}%)")
        
        return "\n".join(lines)


@dataclass
class FactWithCitation:
    """A verified fact with full source citation."""
    fact_type: str  # 'electoral_result', 'prediction', 'survey', 'demographic'
    fact_text: str
    numerical_value: Optional[float] = None
    entity_name: str = ""
    entity_type: str = ""  # 'constituency', 'candidate', 'party', 'pc'
    time_period: str = ""  # '2021', '2024', '2026_predicted'
    
    # Citation info
    source_file: str = ""
    source_row: Optional[int] = None
    confidence: float = 1.0  # 1.0 for direct data, lower for derived
    
    # Related entities
    related_entities: List[str] = field(default_factory=list)
    
    def citation_string(self) -> str:
        """Get formatted citation."""
        if self.source_row:
            return f"[{self.source_file}:row_{self.source_row}]"
        return f"[{self.source_file}]"


@dataclass
class DataQualityReport:
    """Track data quality for transparency."""
    total_records: int
    valid_records: int
    missing_fields: Dict[str, int]
    data_source: str
    extraction_date: datetime
    notes: List[str] = field(default_factory=list)


# ============================================================
# STRUCTURED QUERY TYPES
# ============================================================

@dataclass
class ConstituencyQuery:
    """Query for constituency-specific data."""
    constituency_name: str
    query_type: Literal["profile", "history", "prediction", "comparison"]
    years: Optional[List[int]] = None
    compare_with: Optional[List[str]] = None  # Other constituencies


@dataclass
class AggregateQuery:
    """Query for aggregated statistics."""
    aggregation_type: Literal["count", "sum", "average", "distribution"]
    filter_party: Optional[str] = None
    filter_district: Optional[str] = None
    filter_race_rating: Optional[str] = None
    filter_type: Optional[str] = None  # SC/ST/GEN
    group_by: Optional[str] = None  # 'district', 'pc', 'race_rating'


@dataclass
class ComparisonQuery:
    """Query for comparative analysis."""
    entity_type: Literal["constituency", "district", "pc"]
    entities: List[str]
    metrics: List[str]  # ['vote_share', 'swing', 'margin']
    years: List[int]


@dataclass
class TrendQuery:
    """Query for trend analysis."""
    entity_name: str
    entity_type: str
    metric: str
    years: List[int]


# ============================================================
# ANSWER GENERATION STRUCTURES
# ============================================================

@dataclass
class VerifiedAnswer:
    """An answer with full verification trail."""
    question: str
    answer_text: str
    confidence: float
    
    # Supporting facts
    facts: List[FactWithCitation] = field(default_factory=list)
    
    # Data sources used
    sources: List[str] = field(default_factory=list)
    
    # Caveats and limitations
    caveats: List[str] = field(default_factory=list)
    
    # Timestamp
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_response(self) -> str:
        """Format as user-facing response."""
        response_parts = [self.answer_text, ""]
        
        if self.facts:
            response_parts.append("**Supporting Data:**")
            for fact in self.facts[:5]:  # Limit to top 5
                response_parts.append(f"- {fact.fact_text} {fact.citation_string()}")
        
        if self.caveats:
            response_parts.append("\n**Note:**")
            for caveat in self.caveats:
                response_parts.append(f"- {caveat}")
        
        if self.sources:
            response_parts.append(f"\n*Sources: {', '.join(set(self.sources))}*")
        
        return "\n".join(response_parts)

