"""
Structured Data Ingester - Properly ingest survey/tabular data with statistics.

This module solves the aggregation problem by:
1. Detecting structured data (surveys, election results)
2. Pre-computing statistics during ingestion
3. Storing aggregated results in Knowledge Graph
4. Creating statistics chunks for OpenSearch
"""
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging

import pandas as pd

from .data_schema import SurveyResponse, FactWithCitation
from .knowledge_graph import PoliticalKnowledgeGraph, EntityNode, Relationship

logger = logging.getLogger(__name__)


@dataclass
class SurveyStatistics:
    """Pre-computed statistics for a survey question."""
    survey_id: str
    survey_name: str
    question_column: str
    question_text: str
    total_responses: int
    response_date: Optional[str] = None
    
    # Results - sorted by count
    results: Dict[str, int] = field(default_factory=dict)
    percentages: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    source_file: str = ""
    
    def to_natural_text(self) -> str:
        """Generate comprehensive natural language summary."""
        lines = [
            f"## SURVEY STATISTICS: {self.survey_name}",
            f"## Question: {self.question_text}",
            f"## Total Responses: {self.total_responses}",
            f"## Source: {self.source_file}",
            "",
            "### COMPLETE RESULTS (All responses, sorted by votes):",
            ""
        ]
        
        # Sort by count descending
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (option, count) in enumerate(sorted_results, 1):
            pct = self.percentages.get(option, 0)
            lines.append(f"{rank}. **{option}**: {count} responses ({pct:.1f}%)")
        
        # Top choice summary
        if sorted_results:
            top_option, top_count = sorted_results[0]
            top_pct = self.percentages.get(top_option, 0)
            lines.extend([
                "",
                "### SUMMARY:",
                f"- **Most Popular Choice**: {top_option}",
                f"- **Votes**: {top_count} out of {self.total_responses}",
                f"- **Percentage**: {top_pct:.1f}%",
                "",
                "**Note**: These are the COMPLETE and ACCURATE counts from the full dataset.",
                f"This data is from {self.source_file} with {self.total_responses} total responses."
            ])
        
        return "\n".join(lines)
    
    def to_kg_facts(self) -> List[FactWithCitation]:
        """Convert to Knowledge Graph facts."""
        facts = []
        
        # Overall survey fact
        sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_results:
            top_option, top_count = sorted_results[0]
            top_pct = self.percentages.get(top_option, 0)
            
            facts.append(FactWithCitation(
                fact_type="survey_result",
                fact_text=f"In the survey '{self.question_text}', {top_option} received the most support with {top_count} votes ({top_pct:.1f}%) out of {self.total_responses} total responses.",
                numerical_value=top_pct,
                entity_name=top_option,
                entity_type="survey_response",
                time_period="2025",
                source_file=self.source_file,
                confidence=1.0,
                related_entities=[opt for opt, _ in sorted_results[:5]]
            ))
        
        # Individual option facts
        for option, count in sorted_results[:10]:  # Top 10
            pct = self.percentages.get(option, 0)
            facts.append(FactWithCitation(
                fact_type="survey_option",
                fact_text=f"'{option}' received {count} votes ({pct:.1f}%) in the survey: {self.question_text}",
                numerical_value=pct,
                entity_name=option,
                entity_type="survey_option",
                time_period="2025",
                source_file=self.source_file,
                confidence=1.0
            ))
        
        return facts


@dataclass
class ElectionStatistics:
    """Pre-computed statistics for election data."""
    dataset_name: str
    election_type: str  # 'assembly', 'lok_sabha'
    year: int
    
    # Party-wise results
    seats_by_party: Dict[str, int] = field(default_factory=dict)
    vote_share_by_party: Dict[str, float] = field(default_factory=dict)
    
    # Geographic breakdown
    by_district: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Metadata
    total_seats: int = 0
    source_file: str = ""
    
    def to_natural_text(self) -> str:
        """Generate natural language summary."""
        lines = [
            f"## ELECTION STATISTICS: {self.dataset_name}",
            f"## Election: {self.election_type.upper()} {self.year}",
            f"## Total Seats: {self.total_seats}",
            "",
            "### PARTY-WISE RESULTS:",
            ""
        ]
        
        sorted_results = sorted(self.seats_by_party.items(), key=lambda x: x[1], reverse=True)
        
        for party, seats in sorted_results:
            vote_share = self.vote_share_by_party.get(party, 0)
            pct = (seats / self.total_seats * 100) if self.total_seats else 0
            lines.append(f"- **{party}**: {seats} seats ({pct:.1f}%), Vote Share: {vote_share:.1f}%")
        
        return "\n".join(lines)


class StructuredDataIngester:
    """
    Ingester for structured data (surveys, election results).
    
    Responsibilities:
    1. Detect data type (survey, election, demographic)
    2. Compute aggregations and statistics
    3. Store in Knowledge Graph as entities and facts
    4. Create statistics chunks for OpenSearch
    """
    
    def __init__(self, kg: PoliticalKnowledgeGraph):
        self.kg = kg
        self.survey_stats: Dict[str, SurveyStatistics] = {}
        self.election_stats: Dict[str, ElectionStatistics] = {}
    
    def detect_data_type(self, df: pd.DataFrame, filename: str) -> str:
        """Detect the type of structured data."""
        filename_lower = filename.lower()
        columns_lower = [str(c).lower() for c in df.columns]
        
        # Survey detection
        survey_indicators = ['response', 'survey', 'opinion', 'মতামত', 'সার্ভে']
        if any(ind in filename_lower for ind in survey_indicators):
            return 'survey'
        
        if 'timestamp' in columns_lower or any('response' in c for c in columns_lower):
            return 'survey'
        
        # Election data detection
        election_indicators = ['election', 'result', 'vote', 'assembly', 'lok_sabha']
        if any(ind in filename_lower for ind in election_indicators):
            return 'election'
        
        if any(c in columns_lower for c in ['votes', 'vote_share', 'winner', 'margin']):
            return 'election'
        
        # Prediction data
        if 'prediction' in filename_lower or 'forecast' in filename_lower:
            return 'prediction'
        
        return 'generic'
    
    def find_survey_question_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that contain survey questions (limited unique values)."""
        question_columns = []
        
        # Key survey columns that should always be included (higher unique value limit)
        key_column_keywords = ['alternative', 'বিকল্প', 'preference', 'পছন্দ', 'choice', 'candidate', 'প্রার্থী']
        
        for col in df.columns:
            col_str = str(col)
            col_lower = col_str.lower()
            
            # Skip metadata columns
            skip_keywords = ['timestamp', 'email', 'phone', 'contact', 'date', 'time']
            if any(kw in col_lower for kw in skip_keywords):
                continue
            
            # Check unique value count
            unique_count = df[col].nunique()
            total_count = len(df)
            
            # Key survey columns (like CM preferences) can have more unique values
            is_key_column = any(kw in col_lower for kw in key_column_keywords)
            max_unique = 100 if is_key_column else 50
            
            # Good question column: 2-max_unique unique values, not too sparse
            if 2 <= unique_count <= max_unique:
                non_null_ratio = df[col].notna().sum() / total_count
                if non_null_ratio > 0.5:  # At least 50% responses
                    question_columns.append(col)
                    if is_key_column:
                        logger.info(f"Found key survey column: {col_str[:50]} ({unique_count} unique values)")
        
        return question_columns
    
    def compute_survey_statistics(self, df: pd.DataFrame, filename: str) -> List[SurveyStatistics]:
        """Compute statistics for all survey questions."""
        stats_list = []
        
        question_columns = self.find_survey_question_columns(df)
        logger.info(f"Found {len(question_columns)} survey question columns in {filename}")
        
        for col in question_columns:
            # Compute value counts
            counts = df[col].value_counts().to_dict()
            total = len(df)
            
            # Compute percentages
            percentages = {k: (v / total * 100) for k, v in counts.items()}
            
            # Create survey ID
            survey_id = hashlib.md5(f"{filename}_{col}".encode()).hexdigest()[:12]
            
            stats = SurveyStatistics(
                survey_id=survey_id,
                survey_name=filename,
                question_column=str(col),
                question_text=str(col),
                total_responses=total,
                results=counts,
                percentages=percentages,
                source_file=filename
            )
            
            stats_list.append(stats)
            self.survey_stats[survey_id] = stats
        
        return stats_list
    
    def compute_election_statistics(self, df: pd.DataFrame, filename: str) -> Optional[ElectionStatistics]:
        """Compute statistics for election data."""
        # Find winner column
        winner_cols = [c for c in df.columns if 'winner' in str(c).lower()]
        party_cols = [c for c in df.columns if 'party' in str(c).lower()]
        
        if not winner_cols and not party_cols:
            return None
        
        # Determine election year from filename
        year_match = re.search(r'(20\d{2})', filename)
        year = int(year_match.group(1)) if year_match else 2021
        
        # Determine election type
        election_type = 'assembly'
        if 'lok' in filename.lower() or 'ls' in filename.lower():
            election_type = 'lok_sabha'
        
        # Count seats by party
        seats_by_party = defaultdict(int)
        
        if winner_cols:
            winner_col = winner_cols[0]
            for party in df[winner_col].dropna():
                party_clean = str(party).strip().upper()
                # Normalize party names
                if party_clean in ['AITC', 'TMC', 'TRINAMOOL']:
                    party_clean = 'TMC'
                seats_by_party[party_clean] += 1
        
        stats = ElectionStatistics(
            dataset_name=filename,
            election_type=election_type,
            year=year,
            seats_by_party=dict(seats_by_party),
            total_seats=len(df),
            source_file=filename
        )
        
        return stats
    
    def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a structured data file.
        
        Returns:
            Dict with ingestion results:
            - data_type: detected type
            - statistics: list of computed statistics
            - facts_added: count of facts added to KG
            - chunks: list of statistics chunks for OpenSearch
        """
        result = {
            "file": file_path.name,
            "data_type": "unknown",
            "statistics": [],
            "facts_added": 0,
            "chunks": []
        }
        
        try:
            # Read file
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                else:
                    return result
            else:
                return result
            
            logger.info(f"Ingesting {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
            
            # Detect data type
            data_type = self.detect_data_type(df, file_path.name)
            result["data_type"] = data_type
            
            # Process based on type
            if data_type == 'survey':
                stats_list = self.compute_survey_statistics(df, file_path.name)
                result["statistics"] = stats_list
                
                # Add to Knowledge Graph and create chunks
                for stats in stats_list:
                    # Add facts to KG
                    for fact in stats.to_kg_facts():
                        self.kg.add_fact(fact)
                        result["facts_added"] += 1
                    
                    # Create statistics chunk for OpenSearch
                    result["chunks"].append({
                        "text": stats.to_natural_text(),
                        "source_file": file_path.name,
                        "data_type": "SURVEY_STATISTICS",
                        "survey_id": stats.survey_id,
                        "question": stats.question_text,
                        "total_responses": stats.total_responses
                    })
                    
                    # Also add survey entity to KG
                    self._add_survey_to_kg(stats)
            
            elif data_type == 'election':
                stats = self.compute_election_statistics(df, file_path.name)
                if stats:
                    result["statistics"] = [stats]
                    result["chunks"].append({
                        "text": stats.to_natural_text(),
                        "source_file": file_path.name,
                        "data_type": "ELECTION_STATISTICS",
                        "year": stats.year,
                        "election_type": stats.election_type
                    })
            
            logger.info(f"Ingested {file_path.name}: {len(result['chunks'])} statistics chunks, {result['facts_added']} facts")
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path.name}: {e}")
            result["error"] = str(e)
        
        return result
    
    def _add_survey_to_kg(self, stats: SurveyStatistics):
        """Add survey results as entities in the Knowledge Graph."""
        # Create survey entity
        survey_entity = EntityNode(
            entity_id=f"survey_{stats.survey_id}",
            entity_type="survey",
            name=stats.survey_name,
            properties={
                "question": stats.question_text,
                "total_responses": stats.total_responses,
                "source_file": stats.source_file,
                "results": stats.results,
                "percentages": stats.percentages
            },
            source_files=[stats.source_file]
        )
        self.kg.add_entity(survey_entity)
        
        # Create entities for top choices
        sorted_results = sorted(stats.results.items(), key=lambda x: x[1], reverse=True)
        
        for option, count in sorted_results[:10]:
            option_id = hashlib.md5(f"{stats.survey_id}_{option}".encode()).hexdigest()[:12]
            
            option_entity = EntityNode(
                entity_id=f"survey_option_{option_id}",
                entity_type="survey_option",
                name=option,
                properties={
                    "votes": count,
                    "percentage": stats.percentages.get(option, 0),
                    "survey_id": stats.survey_id,
                    "question": stats.question_text
                },
                source_files=[stats.source_file]
            )
            self.kg.add_entity(option_entity)
            
            # Add relationship
            self.kg.add_relationship(Relationship(
                source_id=f"survey_{stats.survey_id}",
                target_id=f"survey_option_{option_id}",
                relationship_type="has_result",
                properties={"votes": count, "percentage": stats.percentages.get(option, 0)}
            ))
    
    def get_survey_statistics(self, survey_id: str) -> Optional[SurveyStatistics]:
        """Get pre-computed survey statistics by ID."""
        return self.survey_stats.get(survey_id)
    
    def search_survey_by_keyword(self, keyword: str) -> List[SurveyStatistics]:
        """Search surveys by keyword in question or results."""
        keyword_lower = keyword.lower()
        matches = []
        
        for stats in self.survey_stats.values():
            # Check question
            if keyword_lower in stats.question_text.lower():
                matches.append(stats)
                continue
            
            # Check results
            for option in stats.results.keys():
                if keyword_lower in str(option).lower():
                    matches.append(stats)
                    break
        
        return matches
    
    def get_candidate_survey_stats(self, candidate_name: str) -> Dict[str, Any]:
        """Get survey statistics for a specific candidate."""
        result = {
            "candidate": candidate_name,
            "surveys": [],
            "total_mentions": 0,
            "average_percentage": 0.0
        }
        
        candidate_lower = candidate_name.lower()
        
        for stats in self.survey_stats.values():
            for option, count in stats.results.items():
                if candidate_lower in str(option).lower():
                    result["surveys"].append({
                        "question": stats.question_text,
                        "votes": count,
                        "percentage": stats.percentages.get(option, 0),
                        "total_responses": stats.total_responses,
                        "source": stats.source_file
                    })
                    result["total_mentions"] += count
        
        if result["surveys"]:
            result["average_percentage"] = sum(s["percentage"] for s in result["surveys"]) / len(result["surveys"])
        
        return result

