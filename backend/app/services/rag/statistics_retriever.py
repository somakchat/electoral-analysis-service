"""
Statistics Retriever - Retrieve pre-computed statistics for aggregation queries.

This module provides:
1. Retrieval of pre-computed survey statistics
2. Retrieval of election statistics from KG
3. Integration with OpenSearch for statistics chunks
4. Accurate answers for count/percentage queries
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging

from .query_classifier import ClassifiedQuery, QueryType, QueryIntent
from .knowledge_graph import PoliticalKnowledgeGraph

# Optional import - StructuredDataIngester requires pandas (not available in Lambda)
try:
    from .structured_data_ingester import StructuredDataIngester, SurveyStatistics
except ImportError:
    StructuredDataIngester = None
    SurveyStatistics = None

logger = logging.getLogger(__name__)


@dataclass
class StatisticsResult:
    """Result from statistics retrieval."""
    query: str
    found: bool
    answer_text: str
    
    # Statistics data
    total_count: int = 0
    value_count: int = 0
    percentage: float = 0.0
    
    # Full breakdown (for surveys)
    breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Sources
    sources: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0
    
    def to_formatted_answer(self) -> str:
        """Generate formatted answer with citations and executive summary."""
        if not self.found:
            return self.answer_text
        
        lines = []
        
        # Filter out long comments - only keep actual candidate names (under 50 chars)
        def is_valid_candidate(name: str) -> bool:
            if len(name) > 50:
                return False  # Too long to be a name
            if name.lower() in ['nobody', 'no one', 'none', 'other']:
                return True  # Valid "none" option
            return True
        
        # Get filtered breakdown
        filtered_breakdown = {
            k: v for k, v in self.breakdown.items() 
            if is_valid_candidate(k)
        } if self.breakdown else {}
        
        # Add executive summary for survey data
        if filtered_breakdown and self.total_count > 0:
            sorted_breakdown = sorted(filtered_breakdown.items(), 
                                     key=lambda x: x[1].get('count', 0), 
                                     reverse=True)
            if sorted_breakdown:
                top1_name, top1_data = sorted_breakdown[0]
                top1_pct = top1_data.get('percentage', 0)
                top1_count = top1_data.get('count', 0)
                
                lines.append("## Executive Summary")
                lines.append(f"Based on survey data with **{self.total_count} responses**, "
                           f"**{top1_name}** leads with **{top1_pct:.1f}%** ({top1_count} votes).")
                
                if len(sorted_breakdown) >= 2:
                    top2_name, top2_data = sorted_breakdown[1]
                    top2_pct = top2_data.get('percentage', 0)
                    lines.append(f"Runner-up is **{top2_name}** at **{top2_pct:.1f}%**.")
                
                lines.append("")
        
        # Add complete breakdown (filtered, no long comments)
        if filtered_breakdown:
            lines.append("### Survey Results:")
            for option, data in sorted(filtered_breakdown.items(), 
                                       key=lambda x: x[1].get('count', 0), 
                                       reverse=True)[:10]:
                count = data.get('count', 0)
                pct = data.get('percentage', 0)
                lines.append(f"- **{option}**: {count} votes ({pct:.1f}%)")
        
        # Add source as footnote (cleaner format)
        if self.sources:
            source_name = self.sources[0] if self.sources else ""
            # Clean up source name - just show "Survey Data" instead of full filename
            lines.append(f"\n*Data from verified survey responses*")
        
        return "\n".join(lines)


class StatisticsRetriever:
    """
    Retrieves pre-computed statistics for aggregation queries.
    
    This class:
    1. Searches for matching statistics in memory (from StructuredDataIngester)
    2. Searches OpenSearch for STATISTICS chunks
    3. Queries Knowledge Graph for entity-based statistics
    4. Returns accurate, pre-computed answers
    """
    
    def __init__(
        self, 
        kg: Optional[PoliticalKnowledgeGraph] = None,
        ingester: Optional[StructuredDataIngester] = None,
        opensearch_client: Optional[Any] = None
    ):
        self.kg = kg
        self.ingester = ingester
        self.opensearch_client = opensearch_client
        
        # Cache for frequently accessed statistics
        self._cache: Dict[str, StatisticsResult] = {}
    
    def retrieve(self, classified_query: ClassifiedQuery) -> StatisticsResult:
        """
        Retrieve statistics for a classified query.
        
        Args:
            classified_query: Query classified by QueryClassifier
            
        Returns:
            StatisticsResult with answer and breakdown
        """
        query = classified_query.original_query
        
        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self._cache:
            logger.info(f"Cache hit for: {query[:50]}...")
            return self._cache[cache_key]
        
        result = StatisticsResult(
            query=query,
            found=False,
            answer_text="Statistics not found for this query."
        )
        
        try:
            # Route based on query type and intent
            if classified_query.query_type == QueryType.SURVEY:
                result = self._retrieve_survey_stats(classified_query)
            
            elif classified_query.query_type == QueryType.AGGREGATION:
                if classified_query.intent == QueryIntent.PERCENTAGE:
                    result = self._retrieve_percentage(classified_query)
                elif classified_query.intent == QueryIntent.COUNT:
                    result = self._retrieve_count(classified_query)
                elif classified_query.intent == QueryIntent.RANKING:
                    result = self._retrieve_ranking(classified_query)
                else:
                    result = self._retrieve_generic_stats(classified_query)
            
            # Cache the result
            if result.found:
                self._cache[cache_key] = result
        
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            result.answer_text = f"Error retrieving statistics: {str(e)}"
        
        return result
    
    def _retrieve_survey_stats(self, classified: ClassifiedQuery) -> StatisticsResult:
        """Retrieve survey statistics."""
        result = StatisticsResult(
            query=classified.original_query,
            found=False,
            answer_text=""
        )
        
        # Extract candidate/entity from query
        target_entity = None
        for entity in classified.entities:
            if entity.lower() not in ['bjp', 'tmc', 'congress', 'cpm']:
                target_entity = entity
                break
        
        if not target_entity:
            # Extract from aggregation target
            target_entity = classified.aggregation_target
        
        # Search in ingester's survey stats
        if self.ingester:
            for survey_id, stats in self.ingester.survey_stats.items():
                # Check if this survey is relevant
                if self._is_relevant_survey(stats, classified):
                    # Find the target entity in results
                    for option, count in stats.results.items():
                        if target_entity and target_entity.lower() in str(option).lower():
                            pct = stats.percentages.get(option, 0)
                            
                            result.found = True
                            result.total_count = stats.total_responses
                            result.value_count = count
                            result.percentage = pct
                            result.breakdown = {
                                opt: {"count": cnt, "percentage": stats.percentages.get(opt, 0)}
                                for opt, cnt in stats.results.items()
                            }
                            result.sources = [stats.source_file]
                            result.confidence = 1.0
                            
                            result.answer_text = self._format_survey_answer(
                                target_entity, count, pct, 
                                stats.total_responses, stats.question_text,
                                stats.source_file, stats.results
                            )
                            
                            return result
                    
                    # If no specific target, return full breakdown
                    if not target_entity:
                        result.found = True
                        result.total_count = stats.total_responses
                        result.breakdown = {
                            opt: {"count": cnt, "percentage": stats.percentages.get(opt, 0)}
                            for opt, cnt in stats.results.items()
                        }
                        result.sources = [stats.source_file]
                        result.confidence = 1.0
                        
                        result.answer_text = self._format_full_survey_answer(stats)
                        return result
        
        # Fallback 1: Search OpenSearch for STATISTICS chunks (preferred for Lambda)
        opensearch_result = self._search_opensearch_statistics(classified, target_entity)
        if opensearch_result and opensearch_result.found:
            return opensearch_result
        
        # Fallback 2: search in KG facts
        if self.kg and target_entity:
            facts = self.kg.get_facts_for_entity(target_entity)
            survey_facts = [f for f in facts if f.fact_type in ['survey_result', 'survey_option']]
            
            if survey_facts:
                fact = survey_facts[0]
                result.found = True
                result.answer_text = fact.fact_text
                result.sources = [fact.source_file]
                result.confidence = fact.confidence
                return result
        
        result.answer_text = f"No survey statistics found for '{target_entity or 'this query'}'."
        return result
    
    def _search_opensearch_statistics(self, classified: ClassifiedQuery, target_entity: Optional[str]) -> Optional[StatisticsResult]:
        """Search OpenSearch for pre-computed statistics chunks."""
        try:
            # Get OpenSearch client
            os_client = self._get_opensearch_client()
            if not os_client:
                return None
            
            # Build query to find SURVEY STATISTICS chunks specifically
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match_phrase": {"text": "SURVEY STATISTICS"}}
                        ],
                        "should": []
                    }
                },
                "size": 10
            }
            
            # Add entity-specific boost if we have a target
            if target_entity:
                query["query"]["bool"]["should"].append(
                    {"match": {"text": {"query": target_entity, "boost": 3.0}}}
                )
            
            # Add survey-related keywords based on query
            original_query_lower = classified.original_query.lower()
            keywords = ["CM", "Chief Minister", "percentage", "votes", "preference"]
            for kw in keywords:
                if kw.lower() in original_query_lower:
                    query["query"]["bool"]["should"].append(
                        {"match": {"text": {"query": kw, "boost": 1.5}}}
                    )
            
            # Special handling for "face of party" or "leader" queries - find CM preference survey
            if 'face' in original_query_lower or 'মুখ' in original_query_lower or 'leader' in original_query_lower or 'নেতা' in original_query_lower:
                query["query"]["bool"]["should"].append(
                    {"match": {"text": {"query": "মুখ্যমন্ত্রী Chief Minister CM", "boost": 3.0}}}
                )
                query["query"]["bool"]["should"].append(
                    {"match": {"text": {"query": "সম্ভাব্য বিকল্প possible alternatives", "boost": 3.0}}}
                )
            
            # If asking about BJP specifically, boost BJP-related surveys
            if 'bjp' in original_query_lower or 'বিজেপি' in original_query_lower:
                query["query"]["bool"]["should"].append(
                    {"match": {"text": {"query": "বিজেপির মুখ্যমন্ত্রী BJP CM", "boost": 2.0}}}
                )
            
            response = os_client.search(index="political-strategy-maker-v2", body=query)
            hits = response.get("hits", {}).get("hits", [])
            
            if not hits:
                logger.info("No STATISTICS chunks found in OpenSearch")
                return None
            
            # Parse the best matching statistics chunk
            for hit in hits:
                text = hit["_source"].get("text", "")
                source = hit["_source"].get("source_file", "Unknown")
                
                # Check if this is a real statistics chunk (not the summary doc)
                if "## SURVEY STATISTICS" in text and "Total Responses:" in text:
                    result = self._parse_statistics_chunk(text, source, target_entity)
                    if result and result.found:
                        logger.info(f"Found statistics from OpenSearch: {source}")
                        return result
            
            return None
            
        except Exception as e:
            logger.error(f"OpenSearch statistics search error: {e}")
            return None
    
    def _get_opensearch_client(self):
        """Get OpenSearch client for statistics queries."""
        if self.opensearch_client:
            return self.opensearch_client
        
        try:
            import os
            from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
            import boto3
            
            endpoint = os.environ.get("OPENSEARCH_ENDPOINT")
            if not endpoint:
                return None
            
            # Clean endpoint
            endpoint = endpoint.replace("https://", "").rstrip("/")
            
            session = boto3.Session()
            creds = session.get_credentials()
            auth = AWSV4SignerAuth(creds, os.environ.get("AWS_REGION", "us-east-1"), "aoss")
            
            self.opensearch_client = OpenSearch(
                hosts=[{"host": endpoint, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
            return self.opensearch_client
        except Exception as e:
            logger.error(f"Failed to create OpenSearch client: {e}")
            return None
    
    def _parse_statistics_chunk(self, text: str, source: str, target_entity: Optional[str]) -> Optional[StatisticsResult]:
        """Parse a statistics chunk to extract counts and percentages."""
        import re
        
        result = StatisticsResult(
            query="",
            found=False,
            answer_text=""
        )
        
        try:
            # Extract total responses
            total_match = re.search(r"Total Responses:\s*(\d+)", text)
            if not total_match:
                return None
            total_responses = int(total_match.group(1))
            
            # Extract all results with counts and percentages
            # Format can be:
            # 1. **Option Name**: COUNT responses (PERCENT%)
            # or - Option Name: COUNT (PERCENT%)
            pattern = r"(?:\d+\.\s*)?[*-]*\s*\*?\*?(.+?)\*?\*?:\s*(\d+)\s*(?:responses\s*)?\((\d+\.?\d*)%\)"
            matches = re.findall(pattern, text)
            
            # Clean up option names (remove markdown artifacts)
            def clean_option(opt):
                opt = opt.strip()
                # Remove leading numbers, asterisks, hyphens
                opt = re.sub(r'^[\d\.\s*\-]+', '', opt)
                opt = re.sub(r'\*+', '', opt)
                return opt.strip()
            
            if not matches:
                return None
            
            breakdown = {}
            for option, count, pct in matches:
                option = clean_option(option)
                if option:  # Skip empty options
                    breakdown[option] = {
                        "count": int(count),
                        "percentage": float(pct)
                    }
            
            # If we have a target entity, find it
            if target_entity:
                target_lower = target_entity.lower()
                found_option = None
                found_count = 0
                found_pct = 0.0
                
                for option, data in breakdown.items():
                    if target_lower in option.lower():
                        found_option = option
                        found_count = data["count"]
                        found_pct = data["percentage"]
                        break
                
                if found_option:
                    result.found = True
                    result.total_count = total_responses
                    result.value_count = found_count
                    result.percentage = found_pct
                    result.breakdown = breakdown
                    result.sources = [source]
                    result.confidence = 1.0
                    
                    result.answer_text = (
                        f"## Survey Results: {target_entity}\n\n"
                        f"Based on survey data with **{total_responses} responses**:\n\n"
                        f"- **{found_option}**: **{found_count} votes ({found_pct:.1f}%)**\n\n"
                        f"### Top 5 Results:\n"
                    )
                    
                    # Add top 5
                    sorted_options = sorted(breakdown.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
                    for opt, data in sorted_options:
                        result.answer_text += f"- **{opt}**: {data['count']} ({data['percentage']:.1f}%)\n"
                    
                    result.answer_text += f"\n*Source: {source}*"
                    return result
            else:
                # Return full breakdown
                result.found = True
                result.total_count = total_responses
                result.breakdown = breakdown
                result.sources = [source]
                result.confidence = 1.0
                
                result.answer_text = f"## Survey Results (Total: {total_responses})\n\n"
                sorted_options = sorted(breakdown.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
                for opt, data in sorted_options:
                    result.answer_text += f"- **{opt}**: {data['count']} ({data['percentage']:.1f}%)\n"
                result.answer_text += f"\n*Source: {source}*"
                return result
                
        except Exception as e:
            logger.error(f"Error parsing statistics chunk: {e}")
            return None
        
        return None
    
    def _retrieve_percentage(self, classified: ClassifiedQuery) -> StatisticsResult:
        """Retrieve percentage statistics."""
        # This is similar to survey stats but focused on percentage
        return self._retrieve_survey_stats(classified)
    
    def _retrieve_count(self, classified: ClassifiedQuery) -> StatisticsResult:
        """Retrieve count statistics."""
        result = StatisticsResult(
            query=classified.original_query,
            found=False,
            answer_text=""
        )
        
        # Check if asking about election seats
        query_lower = classified.original_query.lower()
        
        if self.kg and ('seat' in query_lower or 'constituency' in query_lower or 'won' in query_lower):
            # Check for party
            party = None
            for entity in classified.entities:
                if entity.upper() in ['BJP', 'TMC', 'AITC', 'CONGRESS', 'CPM']:
                    party = entity.upper()
                    break
            
            if party:
                # Count from KG
                if '2021' in query_lower or 'won' in query_lower:
                    seats = self.kg.get_constituencies_by_winner(party, 2021)
                    count = len(seats)
                    
                    result.found = True
                    result.value_count = count
                    result.total_count = len(self.kg.constituency_profiles)
                    result.percentage = (count / result.total_count * 100) if result.total_count else 0
                    
                    result.answer_text = f"**{party}** won **{count} seats** out of {result.total_count} in the 2021 Assembly election ({result.percentage:.1f}%)."
                    result.sources = ["Election Results Data"]
                    result.confidence = 1.0
                    
                    return result
                
                elif '2026' in query_lower or 'predict' in query_lower:
                    counts = self.kg.count_predicted_seats()
                    count = counts.get(party, 0)
                    total = sum(counts.values())
                    
                    result.found = True
                    result.value_count = count
                    result.total_count = total
                    result.percentage = (count / total * 100) if total else 0
                    
                    result.answer_text = f"**{party}** is predicted to win **{count} seats** out of {total} in 2026 ({result.percentage:.1f}%)."
                    result.sources = ["2026 Predictions Data"]
                    result.confidence = 0.8
                    
                    return result
        
        # Fallback to survey stats
        return self._retrieve_survey_stats(classified)
    
    def _retrieve_ranking(self, classified: ClassifiedQuery) -> StatisticsResult:
        """Retrieve ranking statistics."""
        result = StatisticsResult(
            query=classified.original_query,
            found=False,
            answer_text=""
        )
        
        # Search for relevant survey with rankings
        if self.ingester:
            for survey_id, stats in self.ingester.survey_stats.items():
                if self._is_relevant_survey(stats, classified):
                    # Sort by votes
                    sorted_results = sorted(stats.results.items(), key=lambda x: x[1], reverse=True)
                    
                    result.found = True
                    result.total_count = stats.total_responses
                    result.breakdown = {
                        opt: {"count": cnt, "percentage": stats.percentages.get(opt, 0), "rank": i+1}
                        for i, (opt, cnt) in enumerate(sorted_results)
                    }
                    result.sources = [stats.source_file]
                    result.confidence = 1.0
                    
                    # Format ranking answer
                    lines = [
                        f"## Ranking for: {stats.question_text}",
                        f"Total responses: {stats.total_responses}",
                        ""
                    ]
                    
                    for rank, (option, count) in enumerate(sorted_results[:10], 1):
                        pct = stats.percentages.get(option, 0)
                        lines.append(f"{rank}. **{option}**: {count} votes ({pct:.1f}%)")
                    
                    result.answer_text = "\n".join(lines)
                    return result
        
        result.answer_text = "No ranking statistics found for this query."
        return result
    
    def _retrieve_generic_stats(self, classified: ClassifiedQuery) -> StatisticsResult:
        """Retrieve generic statistics."""
        # Try survey stats first
        result = self._retrieve_survey_stats(classified)
        if result.found:
            return result
        
        # Try count stats
        result = self._retrieve_count(classified)
        return result
    
    def _is_relevant_survey(self, stats: SurveyStatistics, classified: ClassifiedQuery) -> bool:
        """Check if a survey is relevant to the query."""
        query_lower = classified.original_query.lower()
        question_lower = stats.question_text.lower()
        
        # Check for keyword matches
        keywords = classified.keywords
        
        # Specific checks for common queries
        if 'cm' in query_lower or 'chief minister' in query_lower or 'মুখ্যমন্ত্রী' in query_lower:
            if 'cm' in question_lower or 'chief' in question_lower or 'মুখ্যমন্ত্রী' in question_lower:
                return True
        
        if 'bjp' in query_lower:
            if 'bjp' in question_lower or 'বিজেপি' in question_lower:
                return True
        
        # Check entity matches
        for entity in classified.entities:
            if entity.lower() in question_lower:
                return True
            
            # Check in results
            for option in stats.results.keys():
                if entity.lower() in str(option).lower():
                    return True
        
        return False
    
    def _format_survey_answer(
        self, 
        entity: str, 
        count: int, 
        percentage: float,
        total: int,
        question: str,
        source: str,
        all_results: Dict[str, int]
    ) -> str:
        """Format a survey statistics answer."""
        # Filter out long comments (keep only actual candidate names)
        filtered_results = {k: v for k, v in all_results.items() if len(str(k)) <= 50}
        
        # Find entity's rank in filtered results
        sorted_results = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)
        rank = 1
        for i, (opt, _) in enumerate(sorted_results, 1):
            if entity.lower() in str(opt).lower():
                rank = i
                break
        
        lines = [
            f"## Survey Results for {entity}",
            "",
            f"**Total Responses:** {total}",
            "",
            f"### {entity}:",
            f"- **Votes:** {count}",
            f"- **Percentage:** {percentage:.1f}%",
            f"- **Rank:** #{rank}",
            "",
            "### Top 5 Candidates:",
        ]
        
        for i, (opt, cnt) in enumerate(sorted_results[:5], 1):
            pct = (cnt / total * 100) if total else 0
            marker = "👉 " if entity.lower() in str(opt).lower() else ""
            lines.append(f"{i}. {marker}**{opt}**: {cnt} votes ({pct:.1f}%)")
        
        lines.append("\n*Data from verified survey responses*")
        
        return "\n".join(lines)
    
    def _format_full_survey_answer(self, stats: SurveyStatistics) -> str:
        """Format a full survey breakdown answer."""
        # Filter out long comments (keep only actual candidate names)
        filtered_results = {k: v for k, v in stats.results.items() if len(str(k)) <= 50}
        sorted_results = sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)
        
        lines = [
            "## Survey Results",
            "",
            f"**Total Responses:** {stats.total_responses}",
            "",
            "### Results:",
            ""
        ]
        
        for rank, (option, count) in enumerate(sorted_results[:10], 1):
            pct = stats.percentages.get(option, 0)
            lines.append(f"{rank}. **{option}**: {count} votes ({pct:.1f}%)")
        
        lines.append("\n*Data from verified survey responses*")
        
        return "\n".join(lines)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        # Normalize query
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return normalized[:100]  # Limit key length
    
    def clear_cache(self):
        """Clear the statistics cache."""
        self._cache.clear()
    
    def search_opensearch_statistics(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for statistics chunks in OpenSearch."""
        if not self.opensearch_client:
            return []
        
        try:
            # Search with high boost for SURVEY_STATISTICS data type
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 1.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "data_type": {
                                        "query": "SURVEY_STATISTICS",
                                        "boost": 5.0
                                    }
                                }
                            },
                            {
                                "match": {
                                    "data_type": {
                                        "query": "ELECTION_STATISTICS",
                                        "boost": 3.0
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            response = self.opensearch_client.search(
                index="political-strategy-maker-v2",
                body=search_body
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
        
        except Exception as e:
            logger.error(f"OpenSearch statistics search error: {e}")
            return []

