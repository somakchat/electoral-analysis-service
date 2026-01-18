"""
Political Knowledge Graph - Constituency-centric entity relationships.

This module builds a knowledge graph that:
1. Centers on constituencies as the primary unit of analysis
2. Tracks all relationships (constituency ↔ PC, candidate ↔ party, etc.)
3. Enables precise fact retrieval with full provenance
4. Supports aggregation queries across hierarchies
"""
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import re

from .data_schema import (
    ConstituencyProfile, ParliamentaryConstituency, CandidateProfile,
    ElectionResult, SurveyResponse, FactWithCitation, ConstituencyType
)


@dataclass
class EntityNode:
    """A node in the knowledge graph."""
    entity_id: str
    entity_type: str  # 'constituency', 'pc', 'district', 'party', 'candidate'
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relationship_type: str  # 'belongs_to', 'won_in', 'contested_from', 'located_in'
    properties: Dict[str, Any] = field(default_factory=dict)
    year: Optional[int] = None


class PoliticalKnowledgeGraph:
    """
    Knowledge graph for political entities in West Bengal.
    
    Hierarchy:
    - State → District → Constituency (AC)
    - State → Parliamentary Constituency (PC) → Constituency (AC)
    - Party → Candidate → Constituency
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        
        # Entity storage
        self.entities: Dict[str, EntityNode] = {}
        
        # Relationship storage
        self.relationships: List[Relationship] = []
        self.relationships_by_source: Dict[str, List[Relationship]] = defaultdict(list)
        self.relationships_by_target: Dict[str, List[Relationship]] = defaultdict(list)
        self.relationships_by_type: Dict[str, List[Relationship]] = defaultdict(list)
        
        # Indexed lookups for fast retrieval
        self.constituencies_by_name: Dict[str, str] = {}  # name → entity_id
        self.constituencies_by_district: Dict[str, List[str]] = defaultdict(list)
        self.constituencies_by_pc: Dict[str, List[str]] = defaultdict(list)
        self.candidates_by_party: Dict[str, List[str]] = defaultdict(list)
        self.candidates_by_constituency: Dict[str, List[str]] = defaultdict(list)
        
        # Fact store with citations
        self.facts: List[FactWithCitation] = []
        self.facts_by_entity: Dict[str, List[FactWithCitation]] = defaultdict(list)
        self.facts_by_type: Dict[str, List[FactWithCitation]] = defaultdict(list)
        
        # Constituency profiles (the core data structure)
        self.constituency_profiles: Dict[str, ConstituencyProfile] = {}
        
        # Survey data
        self.surveys: Dict[str, SurveyResponse] = {}
        
        if storage_path and storage_path.exists():
            self.load()
    
    # ============================================================
    # ENTITY MANAGEMENT
    # ============================================================
    
    def add_entity(self, entity: EntityNode) -> str:
        """Add an entity to the graph."""
        self.entities[entity.entity_id] = entity
        
        # Update indexes
        if entity.entity_type == "constituency":
            self.constituencies_by_name[entity.name.upper()] = entity.entity_id
            if "district" in entity.properties:
                self.constituencies_by_district[entity.properties["district"]].append(entity.entity_id)
            if "parent_pc" in entity.properties:
                self.constituencies_by_pc[entity.properties["parent_pc"]].append(entity.entity_id)
        
        return entity.entity_id
    
    def add_relationship(self, rel: Relationship):
        """Add a relationship to the graph."""
        self.relationships.append(rel)
        self.relationships_by_source[rel.source_id].append(rel)
        self.relationships_by_target[rel.target_id].append(rel)
        self.relationships_by_type[rel.relationship_type].append(rel)
    
    def add_fact(self, fact: FactWithCitation):
        """Add a verified fact with citation."""
        self.facts.append(fact)
        if fact.entity_name:
            self.facts_by_entity[fact.entity_name.upper()].append(fact)
        self.facts_by_type[fact.fact_type].append(fact)
    
    # ============================================================
    # CONSTITUENCY PROFILE MANAGEMENT
    # ============================================================
    
    def add_constituency_profile(self, profile: ConstituencyProfile):
        """Add a constituency profile."""
        entity_id = f"ac_{profile.ac_no}"
        
        # Create entity node
        entity = EntityNode(
            entity_id=entity_id,
            entity_type="constituency",
            name=profile.ac_name,
            properties={
                "ac_no": profile.ac_no,
                "district": profile.district,
                "type": profile.constituency_type.value if isinstance(profile.constituency_type, ConstituencyType) else profile.constituency_type,
                "parent_pc": profile.parent_pc,
                "winner_2021": profile.winner_2021,
                "tmc_vs_2021": profile.tmc_vote_share_2021,
                "bjp_vs_2021": profile.bjp_vote_share_2021,
                "predicted_winner_2026": profile.predicted_winner_2026,
                "predicted_margin_2026": profile.predicted_margin_2026,
                "race_rating": profile.race_rating,
            },
            source_files=profile.source_files
        )
        
        self.add_entity(entity)
        self.constituency_profiles[profile.ac_name.upper()] = profile
        
        # Add facts with citations
        source_file = profile.source_files[0] if profile.source_files else "unknown"
        
        # 2021 result fact
        self.add_fact(FactWithCitation(
            fact_type="electoral_result",
            fact_text=f"{profile.ac_name} was won by {profile.winner_2021} in 2021 with TMC at {profile.tmc_vote_share_2021:.2f}% and BJP at {profile.bjp_vote_share_2021:.2f}%",
            entity_name=profile.ac_name,
            entity_type="constituency",
            time_period="2021",
            source_file=source_file,
            confidence=1.0
        ))
        
        # Prediction fact
        self.add_fact(FactWithCitation(
            fact_type="prediction",
            fact_text=f"{profile.ac_name} is predicted to be won by {profile.predicted_winner_2026} in 2026 with margin of {abs(profile.predicted_margin_2026):.2f}% (Rating: {profile.race_rating})",
            numerical_value=profile.predicted_margin_2026,
            entity_name=profile.ac_name,
            entity_type="constituency",
            time_period="2026_predicted",
            source_file=source_file,
            confidence=0.8  # Predictions have lower confidence
        ))
        
        # Swing fact
        if profile.pc_swing_2019_2024:
            direction = "towards TMC" if profile.pc_swing_2019_2024 > 0 else "towards BJP"
            self.add_fact(FactWithCitation(
                fact_type="swing_analysis",
                fact_text=f"{profile.parent_pc} PC saw a swing of {abs(profile.pc_swing_2019_2024):.2f}% {direction} from 2019 to 2024",
                numerical_value=profile.pc_swing_2019_2024,
                entity_name=profile.parent_pc,
                entity_type="pc",
                time_period="2019-2024",
                source_file=source_file,
                confidence=1.0,
                related_entities=[profile.ac_name]
            ))
        
        # Add relationships
        # Constituency → District
        self.add_relationship(Relationship(
            source_id=entity_id,
            target_id=f"district_{profile.district.lower().replace(' ', '_')}",
            relationship_type="located_in"
        ))
        
        # Constituency → PC
        self.add_relationship(Relationship(
            source_id=entity_id,
            target_id=f"pc_{profile.parent_pc.lower().replace(' ', '_')}",
            relationship_type="belongs_to_pc"
        ))
    
    # ============================================================
    # QUERY METHODS
    # ============================================================
    
    def get_constituency(self, name: str) -> Optional[ConstituencyProfile]:
        """Get constituency profile by name."""
        return self.constituency_profiles.get(name.upper())
    
    def search_constituency(self, query: str) -> List[ConstituencyProfile]:
        """Fuzzy search for constituencies."""
        query_upper = query.upper()
        matches = []
        
        for name, profile in self.constituency_profiles.items():
            if query_upper in name or name in query_upper:
                matches.append(profile)
        
        return matches
    
    def get_constituencies_by_district(self, district: str) -> List[ConstituencyProfile]:
        """Get all constituencies in a district."""
        return [
            self.constituency_profiles[self.entities[eid].name.upper()]
            for eid in self.constituencies_by_district.get(district, [])
            if self.entities[eid].name.upper() in self.constituency_profiles
        ]
    
    def get_constituencies_by_pc(self, pc_name: str) -> List[ConstituencyProfile]:
        """Get all constituencies in a parliamentary constituency."""
        # Normalize PC name
        pc_upper = pc_name.upper().strip()
        
        matches = []
        for name, profile in self.constituency_profiles.items():
            if profile.parent_pc.upper().strip() == pc_upper:
                matches.append(profile)
        
        return matches
    
    def get_constituencies_by_winner(self, party: str, year: int = 2021) -> List[ConstituencyProfile]:
        """Get constituencies won by a party."""
        party_upper = party.upper().strip()
        # Handle aliases
        if party_upper in ["TMC", "TRINAMOOL", "AITC"]:
            party_variants = ["TMC", "AITC", "TRINAMOOL"]
        elif party_upper == "BJP":
            party_variants = ["BJP"]
        else:
            party_variants = [party_upper]
        
        return [
            profile for profile in self.constituency_profiles.values()
            if profile.winner_2021.upper() in party_variants
        ]
    
    def get_swing_seats(self, threshold: float = 5.0) -> List[ConstituencyProfile]:
        """Get seats with predicted close margins (swing seats)."""
        return [
            profile for profile in self.constituency_profiles.values()
            if abs(profile.predicted_margin_2026) <= threshold
        ]
    
    def get_vulnerable_seats(self, party: str) -> List[ConstituencyProfile]:
        """Get seats vulnerable for a party."""
        party_upper = party.upper()
        vulnerable = []
        
        for profile in self.constituency_profiles.values():
            if party_upper in ["TMC", "AITC"]:
                # TMC vulnerable = TMC won 2021 but BJP predicted 2026
                if profile.winner_2021.upper() in ["TMC", "AITC"] and profile.predicted_winner_2026.upper() == "BJP":
                    vulnerable.append(profile)
            elif party_upper == "BJP":
                # BJP vulnerable = BJP won 2021 but TMC predicted 2026
                if profile.winner_2021.upper() == "BJP" and profile.predicted_winner_2026.upper() == "TMC":
                    vulnerable.append(profile)
        
        return vulnerable
    
    def get_seats_by_rating(self, rating: str) -> List[ConstituencyProfile]:
        """Get seats by race rating."""
        rating_lower = rating.lower()
        return [
            profile for profile in self.constituency_profiles.values()
            if profile.race_rating.lower() == rating_lower
        ]
    
    def get_facts_for_entity(self, entity_name: str) -> List[FactWithCitation]:
        """Get all facts about an entity."""
        return self.facts_by_entity.get(entity_name.upper(), [])
    
    def get_facts_by_type(self, fact_type: str) -> List[FactWithCitation]:
        """Get all facts of a specific type."""
        return self.facts_by_type.get(fact_type, [])
    
    # ============================================================
    # AGGREGATION QUERIES
    # ============================================================
    
    def count_seats_by_party(self, year: int = 2021) -> Dict[str, int]:
        """Count seats won by each party."""
        counts = defaultdict(int)
        for profile in self.constituency_profiles.values():
            counts[profile.winner_2021] += 1
        return dict(counts)
    
    def count_predicted_seats(self) -> Dict[str, int]:
        """Count predicted seats for 2026."""
        counts = defaultdict(int)
        for profile in self.constituency_profiles.values():
            counts[profile.predicted_winner_2026] += 1
        return dict(counts)
    
    def count_by_race_rating(self) -> Dict[str, Dict[str, int]]:
        """Count seats by race rating and predicted winner."""
        ratings = defaultdict(lambda: defaultdict(int))
        for profile in self.constituency_profiles.values():
            ratings[profile.race_rating][profile.predicted_winner_2026] += 1
        return {k: dict(v) for k, v in ratings.items()}
    
    def count_by_district(self) -> Dict[str, Dict[str, int]]:
        """Count seats by district and 2021 winner."""
        districts = defaultdict(lambda: defaultdict(int))
        for profile in self.constituency_profiles.values():
            districts[profile.district][profile.winner_2021] += 1
        return {k: dict(v) for k, v in districts.items()}
    
    def get_district_swing(self, district: str) -> Tuple[float, int]:
        """Get average swing and seat count for a district."""
        constituencies = self.get_constituencies_by_district(district)
        if not constituencies:
            return 0.0, 0
        
        swings = [c.pc_swing_2019_2024 for c in constituencies if c.pc_swing_2019_2024]
        if not swings:
            return 0.0, len(constituencies)
        
        return sum(swings) / len(swings), len(constituencies)
    
    # ============================================================
    # NATURAL LANGUAGE GENERATION
    # ============================================================
    
    def generate_constituency_summary(self, name: str) -> str:
        """Generate natural language summary for a constituency."""
        profile = self.get_constituency(name)
        if not profile:
            return f"No data found for constituency: {name}"
        
        # Get related facts
        facts = self.get_facts_for_entity(name)
        
        summary_parts = [
            f"## {profile.ac_name} Constituency Analysis",
            f"",
            f"**Location:** {profile.district} district, part of {profile.parent_pc} PC",
            f"**Category:** {profile.constituency_type.value if isinstance(profile.constituency_type, ConstituencyType) else profile.constituency_type} seat",
            f"",
            f"### 2021 Assembly Election",
            f"- Winner: **{profile.winner_2021}**",
            f"- TMC Vote Share: {profile.tmc_vote_share_2021:.2f}%",
            f"- BJP Vote Share: {profile.bjp_vote_share_2021:.2f}%",
            f"- Margin: {abs(profile.tmc_vote_share_2021 - profile.bjp_vote_share_2021):.2f}%",
            f"",
            f"### Lok Sabha Trends ({profile.parent_pc} PC)",
            f"- 2019: TMC {profile.pc_tmc_vs_2019:.2f}% vs BJP {profile.pc_bjp_vs_2019:.2f}%",
            f"- 2024: TMC {profile.pc_tmc_vs_2024:.2f}% vs BJP {profile.pc_bjp_vs_2024:.2f}%",
            f"- Swing: {abs(profile.pc_swing_2019_2024):.2f}% {'towards TMC' if profile.pc_swing_2019_2024 > 0 else 'towards BJP'}",
            f"",
            f"### 2026 Prediction",
            f"- Predicted Winner: **{profile.predicted_winner_2026}**",
            f"- Predicted Margin: {abs(profile.predicted_margin_2026):.2f}%",
            f"- Race Rating: **{profile.race_rating}**",
        ]
        
        if profile.vulnerability_tag:
            summary_parts.append(f"- Status: {profile.vulnerability_tag}")
        
        return "\n".join(summary_parts)
    
    def generate_district_summary(self, district: str) -> str:
        """Generate summary for a district."""
        constituencies = self.get_constituencies_by_district(district)
        if not constituencies:
            return f"No data found for district: {district}"
        
        # Calculate stats
        tmc_2021 = sum(1 for c in constituencies if c.winner_2021.upper() in ["TMC", "AITC"])
        bjp_2021 = sum(1 for c in constituencies if c.winner_2021.upper() == "BJP")
        other_2021 = len(constituencies) - tmc_2021 - bjp_2021
        
        tmc_2026 = sum(1 for c in constituencies if c.predicted_winner_2026.upper() == "TMC")
        bjp_2026 = sum(1 for c in constituencies if c.predicted_winner_2026.upper() == "BJP")
        
        swing_seats = [c for c in constituencies if c.race_rating.lower() in ["toss-up", "lean"]]
        
        avg_swing, _ = self.get_district_swing(district)
        
        return f"""## {district} District Analysis

**Total Seats:** {len(constituencies)}

### 2021 Results
- TMC: {tmc_2021} seats
- BJP: {bjp_2021} seats
- Others: {other_2021} seats

### 2026 Predictions
- TMC: {tmc_2026} seats
- BJP: {bjp_2026} seats
- Change: TMC {'gains' if tmc_2026 > tmc_2021 else 'loses'} {abs(tmc_2026 - tmc_2021)} seats

### Swing Analysis
- Average PC Swing (2019→2024): {abs(avg_swing):.2f}% {'towards TMC' if avg_swing > 0 else 'towards BJP'}
- Close Races (Toss-up/Lean): {len(swing_seats)} seats

### Key Battlegrounds
{chr(10).join([f"- {c.ac_name}: {c.race_rating} ({c.predicted_winner_2026} by {abs(c.predicted_margin_2026):.1f}%)" for c in swing_seats[:5]])}
"""
    
    # ============================================================
    # PERSISTENCE
    # ============================================================
    
    def save(self):
        """Save knowledge graph to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": {k: {"entity_id": v.entity_id, "entity_type": v.entity_type, 
                           "name": v.name, "properties": v.properties, 
                           "source_files": v.source_files} for k, v in self.entities.items()},
            "constituency_profiles": {
                k: {
                    "ac_no": v.ac_no, "ac_name": v.ac_name, "district": v.district,
                    "constituency_type": v.constituency_type.value if isinstance(v.constituency_type, ConstituencyType) else v.constituency_type,
                    "parent_pc": v.parent_pc, "winner_2021": v.winner_2021,
                    "tmc_vote_share_2021": v.tmc_vote_share_2021,
                    "bjp_vote_share_2021": v.bjp_vote_share_2021,
                    "margin_2021": v.margin_2021,
                    "pc_tmc_vs_2019": v.pc_tmc_vs_2019,
                    "pc_bjp_vs_2019": v.pc_bjp_vs_2019,
                    "pc_tmc_vs_2024": v.pc_tmc_vs_2024,
                    "pc_bjp_vs_2024": v.pc_bjp_vs_2024,
                    "pc_swing_2019_2024": v.pc_swing_2019_2024,
                    "predicted_margin_2026": v.predicted_margin_2026,
                    "predicted_winner_2026": v.predicted_winner_2026,
                    "race_rating": v.race_rating,
                    "vulnerability_tag": v.vulnerability_tag,
                    "source_files": v.source_files
                }
                for k, v in self.constituency_profiles.items()
            },
            "facts": [
                {
                    "fact_type": f.fact_type, "fact_text": f.fact_text,
                    "numerical_value": f.numerical_value, "entity_name": f.entity_name,
                    "entity_type": f.entity_type, "time_period": f.time_period,
                    "source_file": f.source_file, "source_row": f.source_row,
                    "confidence": f.confidence, "related_entities": f.related_entities
                }
                for f in self.facts
            ]
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self):
        """Load knowledge graph from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore entities
        for eid, edata in data.get("entities", {}).items():
            entity = EntityNode(**edata)
            self.entities[eid] = entity
        
        # Restore constituency profiles
        for name, pdata in data.get("constituency_profiles", {}).items():
            # Convert string to enum
            ctype = pdata.get("constituency_type", "GEN")
            try:
                ctype = ConstituencyType(ctype)
            except ValueError:
                ctype = ConstituencyType.GENERAL
            
            profile = ConstituencyProfile(
                ac_no=pdata["ac_no"],
                ac_name=pdata["ac_name"],
                district=pdata["district"],
                constituency_type=ctype,
                parent_pc=pdata["parent_pc"],
                winner_2021=pdata["winner_2021"],
                tmc_vote_share_2021=pdata["tmc_vote_share_2021"],
                bjp_vote_share_2021=pdata["bjp_vote_share_2021"],
                margin_2021=pdata.get("margin_2021", 0),
                pc_tmc_vs_2019=pdata["pc_tmc_vs_2019"],
                pc_bjp_vs_2019=pdata["pc_bjp_vs_2019"],
                pc_tmc_vs_2024=pdata["pc_tmc_vs_2024"],
                pc_bjp_vs_2024=pdata["pc_bjp_vs_2024"],
                pc_swing_2019_2024=pdata["pc_swing_2019_2024"],
                predicted_margin_2026=pdata["predicted_margin_2026"],
                predicted_winner_2026=pdata["predicted_winner_2026"],
                race_rating=pdata["race_rating"],
                vulnerability_tag=pdata.get("vulnerability_tag"),
                source_files=pdata.get("source_files", [])
            )
            self.constituency_profiles[name] = profile
        
        # Restore facts
        for fdata in data.get("facts", []):
            fact = FactWithCitation(**fdata)
            self.facts.append(fact)
            if fact.entity_name:
                self.facts_by_entity[fact.entity_name.upper()].append(fact)
            self.facts_by_type[fact.fact_type].append(fact)
        
        # Rebuild indexes
        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes after loading."""
        self.constituencies_by_name.clear()
        self.constituencies_by_district.clear()
        self.constituencies_by_pc.clear()
        
        for name, profile in self.constituency_profiles.items():
            self.constituencies_by_name[name] = f"ac_{profile.ac_no}"
            self.constituencies_by_district[profile.district].append(f"ac_{profile.ac_no}")
            self.constituencies_by_pc[profile.parent_pc].append(f"ac_{profile.ac_no}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_constituencies": len(self.constituency_profiles),
            "total_entities": len(self.entities),
            "total_facts": len(self.facts),
            "total_relationships": len(self.relationships),
            "districts": len(self.constituencies_by_district),
            "pcs": len(self.constituencies_by_pc),
            "seats_2021": self.count_seats_by_party(2021),
            "seats_2026_predicted": self.count_predicted_seats(),
        }

