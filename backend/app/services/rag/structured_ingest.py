"""
Structured Data Ingestion - Type-aware data loading for zero hallucination.

This module provides specialized loaders for each data type:
1. Electoral results (CSV with vote counts, margins)
2. Constituency predictions (with swing analysis)  
3. Vulnerability assessments
4. Survey responses (Bengali and English)
5. Narrative documents (DOCX analysis)

Each loader:
- Validates data before ingestion
- Creates structured FactWithCitation objects
- Builds the knowledge graph
- Generates searchable text chunks with embedded citations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
import json
import hashlib

from .data_schema import (
    ConstituencyProfile, ConstituencyType, ElectionResult,
    CandidateProfile, SurveyResponse, FactWithCitation, ElectionType
)
from .knowledge_graph import PoliticalKnowledgeGraph


class DataValidator:
    """Validate data quality before ingestion."""
    
    @staticmethod
    def validate_constituency_name(name: str) -> str:
        """Normalize constituency name."""
        if not name or pd.isna(name):
            return ""
        return str(name).strip().upper()
    
    @staticmethod
    def validate_percentage(value: Any, allow_negative: bool = True) -> Optional[float]:
        """Validate and normalize percentage value."""
        if pd.isna(value):
            return None
        try:
            val = float(value)
            if not allow_negative and val < 0:
                return None
            if val > 100 or val < -100:
                return None
            return round(val, 2)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def validate_party_code(code: str) -> str:
        """Normalize party code."""
        if not code or pd.isna(code):
            return "UNKNOWN"
        
        code = str(code).strip().upper()
        
        # Normalize common variants
        party_map = {
            "AITC": "TMC",
            "TRINAMOOL": "TMC",
            "ALL INDIA TRINAMOOL CONGRESS": "TMC",
            "BHARATIYA JANATA PARTY": "BJP",
            "INDIAN NATIONAL CONGRESS": "INC",
            "CONGRESS": "INC",
            "COMMUNIST PARTY OF INDIA (MARXIST)": "CPM",
            "CPI(M)": "CPM",
            "CPIM": "CPM",
        }
        
        return party_map.get(code, code)


class ConstituencyDataLoader:
    """Load constituency prediction and results data."""
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph):
        self.kg = knowledge_graph
        self.validator = DataValidator()
    
    def load_predictions_csv(self, file_path: Path) -> Tuple[int, List[str]]:
        """
        Load WB_Assembly_2026_predictions_by_AC_sorted.csv
        
        Expected columns:
        AC_No, AC_Name, District, Type, Parent_PC, Winner_2021,
        TMC_VS_2021, BJP_VS_2021, TwoPartySum_2021,
        PC2019_TMC_VS, PC2019_BJP_VS, PC2024_TMC_VS, PC2024_BJP_VS,
        PC_Swing_2019_2024, Swing_Blend, Predicted_TMCminusBJP_Margin_2026,
        Predicted_Winner_2026, Race_Rating
        """
        loaded = 0
        errors = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            return 0, [f"Failed to read file: {e}"]
        
        source_file = file_path.name
        
        for idx, row in df.iterrows():
            try:
                # Validate required fields
                ac_name = self.validator.validate_constituency_name(row.get('AC_Name', ''))
                if not ac_name:
                    errors.append(f"Row {idx}: Missing AC_Name")
                    continue
                
                # Parse constituency type
                type_str = str(row.get('Type', 'GEN')).strip().upper()
                try:
                    ctype = ConstituencyType(type_str)
                except ValueError:
                    ctype = ConstituencyType.GENERAL
                
                # Build profile
                profile = ConstituencyProfile(
                    ac_no=int(row.get('AC_No', idx)),
                    ac_name=ac_name,
                    district=str(row.get('District', '')).strip(),
                    constituency_type=ctype,
                    parent_pc=str(row.get('Parent_PC', '')).strip(),
                    winner_2021=self.validator.validate_party_code(str(row.get('Winner_2021', ''))),
                    tmc_vote_share_2021=self.validator.validate_percentage(row.get('TMC_VS_2021')) or 0.0,
                    bjp_vote_share_2021=self.validator.validate_percentage(row.get('BJP_VS_2021')) or 0.0,
                    margin_2021=0.0,  # Will calculate
                    pc_tmc_vs_2019=self.validator.validate_percentage(row.get('PC2019_TMC_VS')) or 0.0,
                    pc_bjp_vs_2019=self.validator.validate_percentage(row.get('PC2019_BJP_VS')) or 0.0,
                    pc_tmc_vs_2024=self.validator.validate_percentage(row.get('PC2024_TMC_VS')) or 0.0,
                    pc_bjp_vs_2024=self.validator.validate_percentage(row.get('PC2024_BJP_VS')) or 0.0,
                    pc_swing_2019_2024=self.validator.validate_percentage(row.get('PC_Swing_2019_2024')) or 0.0,
                    predicted_margin_2026=self.validator.validate_percentage(row.get('Predicted_TMCminusBJP_Margin_2026')) or 0.0,
                    predicted_winner_2026=self.validator.validate_party_code(str(row.get('Predicted_Winner_2026', ''))),
                    race_rating=str(row.get('Race_Rating', 'Unknown')).strip(),
                    source_files=[source_file]
                )
                
                # Calculate margin
                profile.margin_2021 = profile.tmc_vote_share_2021 - profile.bjp_vote_share_2021
                
                # Add to knowledge graph
                self.kg.add_constituency_profile(profile)
                loaded += 1
                
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        return loaded, errors
    
    def load_vulnerability_csv(self, file_path: Path, vulnerability_type: str) -> Tuple[int, List[str]]:
        """
        Load vulnerability assessment CSV.
        
        vulnerability_type: 'BJP_vulnerable_to_TMC' or 'TMC_vulnerable_to_BJP'
        """
        updated = 0
        errors = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            return 0, [f"Failed to read file: {e}"]
        
        source_file = file_path.name
        
        for idx, row in df.iterrows():
            try:
                ac_name = self.validator.validate_constituency_name(row.get('AC', ''))
                if not ac_name:
                    continue
                
                # Check if constituency exists in KG
                profile = self.kg.get_constituency(ac_name)
                if profile:
                    # Update vulnerability info
                    profile.vulnerability_tag = str(row.get('Vulnerability_Tag', ''))
                    profile.swing_history = str(row.get('Swing/History (2016→2024)', ''))
                    profile.source_files.append(source_file)
                    
                    # Add vulnerability fact
                    self.kg.add_fact(FactWithCitation(
                        fact_type="vulnerability",
                        fact_text=f"{ac_name}: {profile.vulnerability_tag}",
                        entity_name=ac_name,
                        entity_type="constituency",
                        time_period="2026_analysis",
                        source_file=source_file,
                        source_row=idx + 1,
                        confidence=0.85
                    ))
                    
                    updated += 1
                    
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        return updated, errors


class ElectoralResultsLoader:
    """Load detailed electoral results with candidate data."""
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph):
        self.kg = knowledge_graph
        self.validator = DataValidator()
    
    def load_assembly_results(self, file_path: Path) -> Tuple[int, List[str]]:
        """
        Load wb_assembly_2016_2021.csv - detailed TCPD format data.
        """
        loaded = 0
        errors = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except Exception as e:
            return 0, [f"Failed to read file: {e}"]
        
        source_file = file_path.name
        
        # Group by constituency and year
        for (constituency, year), group in df.groupby(['Constituency_Name', 'Year']):
            try:
                constituency = self.validator.validate_constituency_name(constituency)
                if not constituency:
                    continue
                
                # Get winner (position 1)
                winner_row = group[group['Position'] == 1].iloc[0] if len(group[group['Position'] == 1]) > 0 else None
                
                if winner_row is None:
                    continue
                
                # Create election result
                result = ElectionResult(
                    constituency_name=constituency,
                    year=int(year),
                    election_type=ElectionType.ASSEMBLY,
                    winner_party=self.validator.validate_party_code(str(winner_row.get('Party', ''))),
                    winner_candidate=str(winner_row.get('Candidate', '')),
                    winner_votes=int(winner_row.get('Votes', 0)),
                    total_valid_votes=int(winner_row.get('Valid_Votes', 0)),
                    vote_share_percent=float(winner_row.get('Vote_Share_Percentage', 0)),
                    margin_votes=int(winner_row.get('Margin', 0)),
                    margin_percent=float(winner_row.get('Margin_Percentage', 0)),
                    turnout_percent=float(winner_row.get('Turnout_Percentage', 0)) if 'Turnout_Percentage' in winner_row else None,
                    total_electors=int(winner_row.get('Electors', 0)) if 'Electors' in winner_row else None,
                    source_file=source_file,
                    source_row=int(winner_row.name) + 1
                )
                
                # Get runner-up
                runner_row = group[group['Position'] == 2].iloc[0] if len(group[group['Position'] == 2]) > 0 else None
                if runner_row is not None:
                    result.runner_up_party = self.validator.validate_party_code(str(runner_row.get('Party', '')))
                    result.runner_up_candidate = str(runner_row.get('Candidate', ''))
                    result.runner_up_votes = int(runner_row.get('Votes', 0))
                
                # Add fact to knowledge graph
                self.kg.add_fact(FactWithCitation(
                    fact_type="electoral_result",
                    fact_text=f"In {year}, {constituency} was won by {result.winner_candidate} ({result.winner_party}) with {result.vote_share_percent:.2f}% votes, defeating {result.runner_up_party or 'runner-up'} by {result.margin_votes:,} votes ({result.margin_percent:.2f}%)",
                    numerical_value=result.vote_share_percent,
                    entity_name=constituency,
                    entity_type="constituency",
                    time_period=str(year),
                    source_file=source_file,
                    source_row=int(winner_row.name) + 1,
                    confidence=1.0,
                    related_entities=[result.winner_candidate, result.winner_party]
                ))
                
                # Add candidate facts
                for _, cand_row in group.iterrows():
                    profile = CandidateProfile(
                        name=str(cand_row.get('Candidate', '')),
                        party=self.validator.validate_party_code(str(cand_row.get('Party', ''))),
                        constituency=constituency,
                        year=int(year),
                        election_type=ElectionType.ASSEMBLY,
                        position=int(cand_row.get('Position', 0)),
                        votes=int(cand_row.get('Votes', 0)),
                        vote_share=float(cand_row.get('Vote_Share_Percentage', 0)),
                        won=(int(cand_row.get('Position', 0)) == 1),
                        age=int(cand_row.get('Age', 0)) if pd.notna(cand_row.get('Age')) else None,
                        sex=str(cand_row.get('Sex', '')) if pd.notna(cand_row.get('Sex')) else None,
                        education=str(cand_row.get('MyNeta_education', '')) if pd.notna(cand_row.get('MyNeta_education')) else None,
                        profession=str(cand_row.get('TCPD_Prof_Main_Desc', '')) if pd.notna(cand_row.get('TCPD_Prof_Main_Desc')) else None,
                        incumbent=bool(cand_row.get('Incumbent')),
                        terms_served=int(cand_row.get('No_Terms', 0)),
                        turncoat=bool(cand_row.get('Turncoat')),
                        source_file=source_file
                    )
                    
                    # Add candidate fact
                    if profile.won:
                        self.kg.add_fact(FactWithCitation(
                            fact_type="candidate",
                            fact_text=f"{profile.name} won from {constituency} in {year} as {profile.party} candidate with {profile.vote_share:.2f}% votes. {'Incumbent' if profile.incumbent else 'First-time winner'}.",
                            entity_name=profile.name,
                            entity_type="candidate",
                            time_period=str(year),
                            source_file=source_file,
                            confidence=1.0,
                            related_entities=[constituency, profile.party]
                        ))
                
                loaded += 1
                
            except Exception as e:
                errors.append(f"Constituency {constituency}: {str(e)}")
        
        return loaded, errors
    
    def load_lok_sabha_results(self, file_path: Path, year: int) -> Tuple[int, List[str]]:
        """Load Lok Sabha election results. Handles multiple column naming formats."""
        loaded = 0
        errors = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            return 0, [f"Failed to read file: {e}"]
        
        source_file = file_path.name
        
        # Detect column name format and normalize
        # PC Name column variations
        pc_col = None
        for col in ['PC Name', 'Constituency_Name', 'pc_name', 'constituency_name', 'PC_Name']:
            if col in df.columns:
                pc_col = col
                break
        
        if not pc_col:
            return 0, [f"Could not find PC/Constituency name column. Available: {list(df.columns)}"]
        
        # Votes column variations
        votes_col = None
        for col in ['Total Votes', 'Votes', 'votes', 'total_votes', 'Total_Votes']:
            if col in df.columns:
                votes_col = col
                break
        
        # Vote share column variations
        vote_share_col = None
        for col in ['Vote Share', 'Vote_Share_Percentage', 'vote_share', 'Vote_Share', 'vote_share_percentage']:
            if col in df.columns:
                vote_share_col = col
                break
        
        # Filter West Bengal if state column exists
        state_col = None
        for col in ['State', 'State_Name', 'state', 'state_name']:
            if col in df.columns:
                state_col = col
                break
        
        if state_col:
            df = df[df[state_col].str.contains('West', case=False, na=False)]
        
        if len(df) == 0:
            return 0, ["No West Bengal data found in file"]
        
        # Group by PC
        for pc_name, group in df.groupby(pc_col):
            try:
                pc_name_clean = str(pc_name).strip().upper()
                
                # Get winner (highest votes)
                if votes_col and votes_col in group.columns:
                    winner_row = group.loc[group[votes_col].idxmax()]
                else:
                    # Try to find winner by position
                    if 'Position' in group.columns:
                        winner_row = group[group['Position'] == 1].iloc[0] if len(group[group['Position'] == 1]) > 0 else group.iloc[0]
                    else:
                        winner_row = group.iloc[0]
                
                # Get vote share
                vote_share = 0.0
                if vote_share_col and vote_share_col in winner_row.index:
                    vote_share = float(winner_row[vote_share_col]) if pd.notna(winner_row[vote_share_col]) else 0.0
                
                # Get candidate and party
                candidate = winner_row.get('Candidate', 'Unknown')
                party = winner_row.get('Party', 'Unknown')
                
                # Add fact
                self.kg.add_fact(FactWithCitation(
                    fact_type="lok_sabha_result",
                    fact_text=f"In Lok Sabha {year}, {pc_name_clean} was won by {candidate} ({party}) with {vote_share:.2f}% votes",
                    numerical_value=float(vote_share),
                    entity_name=pc_name_clean,
                    entity_type="pc",
                    time_period=str(year),
                    source_file=source_file,
                    confidence=1.0
                ))
                
                loaded += 1
                
            except Exception as e:
                errors.append(f"PC {pc_name}: {str(e)}")
        
        return loaded, errors


class SurveyDataLoader:
    """Load and process survey response data (including Bengali)."""
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph):
        self.kg = knowledge_graph
    
    def load_survey_xlsx(self, file_path: Path) -> Tuple[int, List[str]]:
        """
        Load survey responses from Excel file.
        Handles Bengali text in filenames and content.
        """
        loaded = 0
        errors = []
        
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            return 0, [f"Failed to read file: {e}"]
        
        source_file = file_path.name
        
        # Detect survey type from filename
        survey_type = self._detect_survey_type(source_file)
        
        # Get the main question column (usually first non-timestamp column)
        question_col = None
        for col in df.columns:
            if 'Timestamp' not in col and 'timestamp' not in col.lower():
                question_col = col
                break
        
        if not question_col:
            return 0, ["Could not identify question column"]
        
        # Aggregate responses
        response_counts = df[question_col].value_counts().to_dict()
        total = sum(response_counts.values())
        
        # Create survey response object
        survey = SurveyResponse(
            survey_name=survey_type,
            question=question_col,
            total_responses=total,
            response_distribution=response_counts,
            response_percentages={k: (v / total * 100) for k, v in response_counts.items()},
            source_file=source_file
        )
        
        # Store in KG
        survey_id = hashlib.md5(source_file.encode()).hexdigest()[:8]
        self.kg.surveys[survey_id] = survey
        
        # Create facts from survey
        for response, count in sorted(response_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / total) * 100
            self.kg.add_fact(FactWithCitation(
                fact_type="survey",
                fact_text=f"Survey '{survey_type}': {pct:.1f}% ({count} respondents) answered '{response}'",
                numerical_value=pct,
                entity_name=str(response),
                entity_type="survey_response",
                time_period="2024-2025",
                source_file=source_file,
                confidence=0.9  # Surveys have inherent uncertainty
            ))
        
        # Add summary fact
        top_response = max(response_counts.items(), key=lambda x: x[1])
        self.kg.add_fact(FactWithCitation(
            fact_type="survey_summary",
            fact_text=f"In survey '{survey_type}' (n={total}), the top response was '{top_response[0]}' with {(top_response[1]/total)*100:.1f}%",
            numerical_value=total,
            entity_name=survey_type,
            entity_type="survey",
            time_period="2024-2025",
            source_file=source_file,
            confidence=0.9
        ))
        
        loaded = len(response_counts)
        
        return loaded, errors
    
    def _detect_survey_type(self, filename: str) -> str:
        """Detect survey type from filename."""
        # Bengali to English mapping
        if 'তৃণমূল' in filename and 'প্রার্থী' in filename:
            return "TMC Candidate Preference Survey"
        elif 'বিজেপি' in filename and 'মুখ্যমন্ত্রী' in filename:
            return "BJP CM Preference Survey"
        elif 'তৃণমূল কংগ্রেস সম্পর্কে' in filename:
            return "TMC Opinion Survey"
        elif 'বিজেপি সম্পর্কে' in filename:
            return "BJP Opinion Survey"
        elif 'সিপিআইএম' in filename:
            return "CPIM Opinion Survey"
        elif 'নাগরিক সমাজ' in filename or 'Nagarik Samaj' in filename:
            return "Civil Society Opinion Survey"
        else:
            return f"Survey: {filename[:50]}"


class TextChunkGenerator:
    """Generate searchable text chunks with embedded citations."""
    
    def __init__(self, knowledge_graph: PoliticalKnowledgeGraph, chunk_size: int = 1000):
        self.kg = knowledge_graph
        self.chunk_size = chunk_size
    
    def generate_constituency_chunks(self) -> List[Dict[str, Any]]:
        """Generate chunks for all constituencies."""
        chunks = []
        
        for name, profile in self.kg.constituency_profiles.items():
            # Main constituency chunk
            chunk_text = profile.to_natural_text()
            
            # Add related facts
            facts = self.kg.get_facts_for_entity(name)
            if facts:
                chunk_text += "\n\n--- ADDITIONAL DATA ---\n"
                for fact in facts[:5]:
                    chunk_text += f"\n• {fact.fact_text} {fact.citation_string()}"
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "type": "constituency_profile",
                    "constituency": name,
                    "district": profile.district,
                    "parent_pc": profile.parent_pc,
                    "winner_2021": profile.winner_2021,
                    "predicted_winner_2026": profile.predicted_winner_2026,
                    "race_rating": profile.race_rating,
                    "sources": profile.source_files
                }
            })
        
        return chunks
    
    def generate_district_chunks(self) -> List[Dict[str, Any]]:
        """Generate summary chunks for each district."""
        chunks = []
        districts = set(p.district for p in self.kg.constituency_profiles.values())
        
        for district in districts:
            summary = self.kg.generate_district_summary(district)
            
            chunks.append({
                "text": summary,
                "metadata": {
                    "type": "district_summary",
                    "district": district,
                    "sources": ["aggregated"]
                }
            })
        
        return chunks
    
    def generate_comparison_chunks(self) -> List[Dict[str, Any]]:
        """Generate comparison and analysis chunks."""
        chunks = []
        
        # Swing seats analysis
        swing_seats = self.kg.get_swing_seats(5.0)
        if swing_seats:
            text = "## KEY SWING SEATS FOR 2026\n\n"
            text += "These constituencies have predicted margins of 5% or less:\n\n"
            
            for seat in sorted(swing_seats, key=lambda x: abs(x.predicted_margin_2026)):
                text += f"• **{seat.ac_name}** ({seat.district}): {seat.predicted_winner_2026} by {abs(seat.predicted_margin_2026):.2f}% [{seat.race_rating}]\n"
                text += f"  2021: {seat.winner_2021} | 2024 PC Swing: {abs(seat.pc_swing_2019_2024):.1f}% {'→TMC' if seat.pc_swing_2019_2024 > 0 else '→BJP'}\n\n"
            
            chunks.append({
                "text": text,
                "metadata": {
                    "type": "swing_analysis",
                    "count": len(swing_seats),
                    "sources": ["aggregated"]
                }
            })
        
        # BJP vulnerable seats
        bjp_vulnerable = self.kg.get_vulnerable_seats("BJP")
        if bjp_vulnerable:
            text = "## BJP SEATS AT RISK (2026)\n\n"
            text += f"BJP won these {len(bjp_vulnerable)} seats in 2021 but may lose in 2026:\n\n"
            
            for seat in sorted(bjp_vulnerable, key=lambda x: x.predicted_margin_2026):
                text += f"• **{seat.ac_name}** ({seat.district}): TMC lead {abs(seat.predicted_margin_2026):.2f}%\n"
                text += f"  2021 margin: BJP +{seat.margin_2021:.2f}% | Vulnerability: {seat.vulnerability_tag or 'N/A'}\n\n"
            
            chunks.append({
                "text": text,
                "metadata": {
                    "type": "vulnerability_analysis",
                    "party": "BJP",
                    "count": len(bjp_vulnerable),
                    "sources": ["aggregated"]
                }
            })
        
        # TMC vulnerable seats
        tmc_vulnerable = self.kg.get_vulnerable_seats("TMC")
        if tmc_vulnerable:
            text = "## TMC SEATS AT RISK (2026)\n\n"
            text += f"TMC won these {len(tmc_vulnerable)} seats in 2021 but may lose in 2026:\n\n"
            
            for seat in sorted(tmc_vulnerable, key=lambda x: -x.predicted_margin_2026):
                text += f"• **{seat.ac_name}** ({seat.district}): BJP lead {abs(seat.predicted_margin_2026):.2f}%\n"
                text += f"  2021 margin: TMC +{abs(seat.margin_2021):.2f}% | Vulnerability: {seat.vulnerability_tag or 'N/A'}\n\n"
            
            chunks.append({
                "text": text,
                "metadata": {
                    "type": "vulnerability_analysis",
                    "party": "TMC",
                    "count": len(tmc_vulnerable),
                    "sources": ["aggregated"]
                }
            })
        
        # Seat projection summary
        seats_2021 = self.kg.count_seats_by_party(2021)
        seats_2026 = self.kg.count_predicted_seats()
        
        text = "## 2026 SEAT PROJECTION SUMMARY\n\n"
        text += "### 2021 Results vs 2026 Predictions\n\n"
        text += "| Party | 2021 | 2026 (Predicted) | Change |\n"
        text += "|-------|------|------------------|--------|\n"
        
        for party in ['TMC', 'BJP', 'AITC']:
            if party in seats_2021 or party in seats_2026:
                s21 = seats_2021.get(party, 0)
                s26 = seats_2026.get(party, 0)
                change = s26 - s21
                text += f"| {party} | {s21} | {s26} | {'+' if change > 0 else ''}{change} |\n"
        
        # Race rating breakdown
        ratings = self.kg.count_by_race_rating()
        text += "\n### Seats by Race Rating\n\n"
        for rating, parties in ratings.items():
            text += f"**{rating}:** {sum(parties.values())} seats\n"
            for party, count in sorted(parties.items(), key=lambda x: -x[1]):
                text += f"  - {party}: {count}\n"
        
        chunks.append({
            "text": text,
            "metadata": {
                "type": "projection_summary",
                "sources": ["aggregated"]
            }
        })
        
        return chunks
    
    def generate_survey_chunks(self) -> List[Dict[str, Any]]:
        """Generate chunks from survey data."""
        chunks = []
        
        for survey_id, survey in self.kg.surveys.items():
            text = survey.to_natural_text()
            
            chunks.append({
                "text": text,
                "metadata": {
                    "type": "survey",
                    "survey_name": survey.survey_name,
                    "total_responses": survey.total_responses,
                    "source": survey.source_file
                }
            })
        
        return chunks
    
    def generate_all_chunks(self) -> List[Dict[str, Any]]:
        """Generate all chunk types."""
        all_chunks = []
        
        all_chunks.extend(self.generate_constituency_chunks())
        all_chunks.extend(self.generate_district_chunks())
        all_chunks.extend(self.generate_comparison_chunks())
        all_chunks.extend(self.generate_survey_chunks())
        
        return all_chunks


class StructuredIngestionPipeline:
    """Main pipeline for structured data ingestion."""
    
    def __init__(self, data_dir: Path, kg_storage_path: Path):
        self.data_dir = data_dir
        self.kg = PoliticalKnowledgeGraph(storage_path=kg_storage_path)
        
        self.constituency_loader = ConstituencyDataLoader(self.kg)
        self.results_loader = ElectoralResultsLoader(self.kg)
        self.survey_loader = SurveyDataLoader(self.kg)
        self.chunk_generator = TextChunkGenerator(self.kg)
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """Run complete ingestion pipeline."""
        stats = {
            "predictions_loaded": 0,
            "vulnerabilities_updated": 0,
            "election_results_loaded": 0,
            "surveys_loaded": 0,
            "errors": []
        }
        
        # 1. Load predictions (primary data source)
        for csv_file in self.data_dir.glob("WB_Assembly_2026_predictions*.csv"):
            loaded, errors = self.constituency_loader.load_predictions_csv(csv_file)
            stats["predictions_loaded"] += loaded
            stats["errors"].extend(errors)
            print(f"Loaded {loaded} constituencies from {csv_file.name}")
        
        # 2. Load vulnerability assessments
        for csv_file in self.data_dir.glob("WB_2026_*vulnerable*.csv"):
            if "BJP" in csv_file.name:
                loaded, errors = self.constituency_loader.load_vulnerability_csv(
                    csv_file, "BJP_vulnerable"
                )
            else:
                loaded, errors = self.constituency_loader.load_vulnerability_csv(
                    csv_file, "TMC_vulnerable"
                )
            stats["vulnerabilities_updated"] += loaded
            stats["errors"].extend(errors)
            print(f"Updated {loaded} vulnerability records from {csv_file.name}")
        
        # 3. Load detailed election results
        for csv_file in self.data_dir.glob("wb_assembly_2016_2021.csv"):
            loaded, errors = self.results_loader.load_assembly_results(csv_file)
            stats["election_results_loaded"] += loaded
            stats["errors"].extend(errors)
            print(f"Loaded {loaded} election results from {csv_file.name}")
        
        # 4. Load Lok Sabha results
        for csv_file in self.data_dir.glob("lok_sabha_2024_results.csv"):
            loaded, errors = self.results_loader.load_lok_sabha_results(csv_file, 2024)
            stats["election_results_loaded"] += loaded
            stats["errors"].extend(errors)
            print(f"Loaded {loaded} Lok Sabha results from {csv_file.name}")
        
        for csv_file in self.data_dir.glob("west_bengal_lok_sabha_2019*.csv"):
            loaded, errors = self.results_loader.load_lok_sabha_results(csv_file, 2019)
            stats["election_results_loaded"] += loaded
            stats["errors"].extend(errors)
            print(f"Loaded {loaded} Lok Sabha results from {csv_file.name}")
        
        # 5. Load surveys
        for xlsx_file in self.data_dir.glob("*.xlsx"):
            loaded, errors = self.survey_loader.load_survey_xlsx(xlsx_file)
            stats["surveys_loaded"] += loaded
            stats["errors"].extend(errors)
            try:
                print(f"Loaded {loaded} survey responses from {xlsx_file.name}")
            except UnicodeEncodeError:
                print(f"Loaded {loaded} survey responses from [Bengali filename]")
        
        # Save knowledge graph
        self.kg.save()
        
        # Generate statistics
        stats["kg_stats"] = self.kg.get_statistics()
        
        return stats
    
    def generate_searchable_chunks(self) -> List[Dict[str, Any]]:
        """Generate all searchable chunks for RAG."""
        return self.chunk_generator.generate_all_chunks()
    
    def get_knowledge_graph(self) -> PoliticalKnowledgeGraph:
        """Get the knowledge graph instance."""
        return self.kg

