"""
Query Classifier - Detect query type and route to appropriate handler.

This module solves the query routing problem by:
1. Classifying queries into types (aggregation, qualitative, entity, comparison)
2. Detecting intent (count, percentage, opinion, strategy, etc.)
3. Extracting entities and keywords for targeted retrieval
4. Routing to the appropriate handler (KG, Statistics, RAG)
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    AGGREGATION = "aggregation"      # Count, percentage, sum, average
    SURVEY = "survey"                # Survey-specific questions
    ENTITY = "entity"                # Specific entity lookup (constituency, candidate)
    COMPARISON = "comparison"        # Compare entities
    TEMPORAL = "temporal"            # Time-based queries (trend, swing)
    QUALITATIVE = "qualitative"      # Opinion, strategy, recommendation
    FACTUAL = "factual"              # Simple fact lookup


class QueryIntent(str, Enum):
    """Specific intent within a query."""
    COUNT = "count"                  # How many, count of
    PERCENTAGE = "percentage"        # What %, percent, proportion
    SUM = "sum"                      # Total, sum
    AVERAGE = "average"              # Average, mean
    TOP_N = "top_n"                  # Top 5, first 3, best
    RANKING = "ranking"              # Rank, position, standing
    OPINION = "opinion"              # Opinion, view, think
    STRATEGY = "strategy"            # Strategy, approach, plan
    RECOMMENDATION = "recommendation" # Recommend, suggest
    COMPARISON = "comparison"        # Compare, vs, versus
    TREND = "trend"                  # Trend, change, swing
    PREDICTION = "prediction"        # Predict, forecast, expect
    LOOKUP = "lookup"                # What is, who is, tell me about


@dataclass
class ClassifiedQuery:
    """Result of query classification."""
    original_query: str
    query_type: QueryType
    intent: QueryIntent
    confidence: float
    
    # Extracted information
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    
    # Routing recommendation
    use_statistics: bool = False
    use_kg: bool = False
    use_rag: bool = True
    
    # For aggregation queries
    aggregation_target: Optional[str] = None  # What to count/aggregate
    filter_conditions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set routing based on query type."""
        if self.query_type in [QueryType.AGGREGATION, QueryType.SURVEY]:
            self.use_statistics = True
            self.use_kg = True
            self.use_rag = True  # Also use RAG for context
        elif self.query_type == QueryType.ENTITY:
            self.use_kg = True
            self.use_rag = True
        elif self.query_type == QueryType.COMPARISON:
            self.use_kg = True
            self.use_rag = True
        else:
            self.use_rag = True


class QueryClassifier:
    """
    Classifies queries to route them to the appropriate handler.
    
    Key capabilities:
    1. Detect aggregation queries (count, %, sum)
    2. Detect survey-related queries
    3. Extract entities (candidates, parties, constituencies)
    4. Determine query intent
    """
    
    # Aggregation keywords (English and Bengali)
    AGGREGATION_PATTERNS = {
        'count': [
            r'how many', r'count', r'number of', r'total',
            r'কত', r'কতজন', r'সংখ্যা', r'মোট'
        ],
        'percentage': [
            r'what percent', r'what %', r'percentage', r'proportion',
            r'শতাংশ', r'শতকরা', r'%'
        ],
        'sum': [
            r'sum of', r'total', r'aggregate',
            r'মোট', r'সর্বমোট'
        ],
        'average': [
            r'average', r'mean', r'avg',
            r'গড়', r'গড়পড়তা'
        ],
        'top_n': [
            r'top \d+', r'first \d+', r'best \d+', r'leading',
            r'শীর্ষ', r'প্রথম'
        ],
        'ranking': [
            r'rank', r'position', r'standing', r'where does .* stand',
            r'স্থান', r'অবস্থান'
        ]
    }
    
    # Survey-related patterns (Bengali patterns using Unicode for reliability)
    SURVEY_PATTERNS = [
        r'survey', r'poll', r'voted for', r'support', r'preference',
        r'who do .* want', r'who .* prefer', r'favorite',
        # Bengali: জনমত, সার্ভে, মতামত, পছন্দ, সমর্থন, কাকে চান, কাকে চাইছে
        '\u099c\u09a8\u09ae\u09a4',  # জনমত
        '\u09b8\u09be\u09b0\u09cd\u09ad\u09c7',  # সার্ভে  
        '\u09ae\u09a4\u09be\u09ae\u09a4',  # মতামত
        '\u09aa\u099b\u09a8\u09cd\u09a6',  # পছন্দ
        '\u09b8\u09ae\u09b0\u09cd\u09a5\u09a8',  # সমর্থন
        '\u0995\u09be\u0995\u09c7 \u099a\u09be\u09a8',  # কাকে চান
        '\u0995\u09be\u0995\u09c7 \u099a\u09be\u0987\u099b\u09c7',  # কাকে চাইছে
        r'who .* want as', r'face of', r'cm candidate',
        # CM face preference patterns
        r'cm face', r'chief minister.*face', r'acceptable.*cm', r'acceptable.*chief',
        r'cm.*candidate', r'chief minister.*candidate', r'who.*cm.*face',
        r'মুখ্যমন্ত্রী.*মুখ', r'সিএম.*ফেস'
    ]
    
    # Opinion/Strategy patterns
    OPINION_PATTERNS = [
        r'opinion', r'think', r'view', r'perception',
        r'মতামত', r'মনে করে', r'দৃষ্টিভঙ্গি'
    ]
    
    STRATEGY_PATTERNS = [
        r'strategy', r'approach', r'plan', r'should', r'how can',
        r'what reform', r'action item', r'recommendation',
        r'কৌশল', r'পরিকল্পনা', r'উচিত'
    ]
    
    # Entity patterns (candidates, parties)
    CANDIDATE_PATTERNS = [
        r'sukanta majumdar', r'suvendu adhikari', r'dilip ghosh',
        r'mamata banerjee', r'samik bhattacharya', r'ashok lahiri',
        r'সুকান্ত মজুমদার', r'শুভেন্দু অধিকারী', r'দিলীপ ঘোষ',
        r'মমতা বন্দ্যোপাধ্যায়', r'সমিক ভট্টাচার্য'
    ]
    
    PARTY_PATTERNS = [
        r'\bbjp\b', r'\btmc\b', r'\bcongress\b', r'\bcpm\b', r'\bcpim\b',
        r'\baitc\b', r'trinamool', r'bharatiya janata',
        r'বিজেপি', r'তৃণমূল', r'কংগ্রেস'
    ]
    
    CONSTITUENCY_PATTERNS = [
        r'constituency', r'seat', r'assembly', r'karimpur', r'nandigram',
        r'বিধানসভা', r'আসন'
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.agg_patterns_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.AGGREGATION_PATTERNS.items()
        }
        
        self.survey_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.SURVEY_PATTERNS
        ]
        
        self.opinion_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.OPINION_PATTERNS
        ]
        
        self.strategy_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.STRATEGY_PATTERNS
        ]
        
        self.candidate_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.CANDIDATE_PATTERNS
        ]
        
        self.party_patterns_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.PARTY_PATTERNS
        ]
    
    def classify(self, query: str) -> ClassifiedQuery:
        """
        Classify a query and determine routing.
        
        Args:
            query: The user's query string
            
        Returns:
            ClassifiedQuery with type, intent, entities, and routing
        """
        query_lower = query.lower()
        
        # Initialize result
        result = ClassifiedQuery(
            original_query=query,
            query_type=QueryType.QUALITATIVE,
            intent=QueryIntent.LOOKUP,
            confidence=0.5
        )
        
        # Extract entities first
        result.entities = self._extract_entities(query)
        result.keywords = self._extract_keywords(query)
        result.numbers = self._extract_numbers(query)
        
        # Check for aggregation queries
        agg_type, agg_score = self._detect_aggregation(query_lower)
        if agg_score > 0.7:
            result.query_type = QueryType.AGGREGATION
            result.intent = QueryIntent[agg_type.upper()]
            result.confidence = agg_score
            result.use_statistics = True
            result.use_kg = True
            
            # Extract what to aggregate
            result.aggregation_target = self._extract_aggregation_target(query)
            
            logger.info(f"Classified as AGGREGATION ({agg_type}): {query[:50]}...")
            return result
        
        # Check for survey-related queries
        survey_score = self._detect_survey(query_lower)
        if survey_score > 0.6:
            result.query_type = QueryType.SURVEY
            result.intent = QueryIntent.COUNT if 'how many' in query_lower or '%' in query else QueryIntent.RANKING
            result.confidence = survey_score
            result.use_statistics = True
            result.use_kg = True
            
            logger.info(f"Classified as SURVEY: {query[:50]}...")
            return result
        
        # Check for strategy/recommendation queries
        if self._detect_strategy(query_lower):
            result.query_type = QueryType.QUALITATIVE
            result.intent = QueryIntent.STRATEGY
            result.confidence = 0.8
            result.use_rag = True
            
            logger.info(f"Classified as STRATEGY: {query[:50]}...")
            return result
        
        # Check for opinion queries
        if self._detect_opinion(query_lower):
            result.query_type = QueryType.QUALITATIVE
            result.intent = QueryIntent.OPINION
            result.confidence = 0.8
            result.use_rag = True
            
            logger.info(f"Classified as OPINION: {query[:50]}...")
            return result
        
        # Check for entity lookup
        if result.entities and len(result.entities) <= 2:
            result.query_type = QueryType.ENTITY
            result.intent = QueryIntent.LOOKUP
            result.confidence = 0.7
            result.use_kg = True
            
            logger.info(f"Classified as ENTITY: {query[:50]}...")
            return result
        
        # Default to qualitative
        result.use_rag = True
        logger.info(f"Classified as QUALITATIVE (default): {query[:50]}...")
        return result
    
    def _detect_aggregation(self, query: str) -> Tuple[str, float]:
        """Detect if query is an aggregation query."""
        max_score = 0.0
        detected_type = 'count'
        
        for agg_type, patterns in self.agg_patterns_compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    # Weight based on pattern specificity
                    score = 0.8 if agg_type in ['percentage', 'ranking'] else 0.7
                    
                    # Boost if combined with survey keywords
                    if any(p.search(query) for p in self.survey_patterns_compiled):
                        score += 0.15
                    
                    # Boost if combined with entity
                    if any(p.search(query) for p in self.candidate_patterns_compiled):
                        score += 0.1
                    
                    if score > max_score:
                        max_score = score
                        detected_type = agg_type
        
        return detected_type, min(max_score, 1.0)
    
    def _detect_survey(self, query: str) -> float:
        """Detect if query is about survey data."""
        score = 0.0
        
        for pattern in self.survey_patterns_compiled:
            if pattern.search(query):
                score += 0.4
        
        # Boost if asking about voting/preference
        if 'vote' in query or 'voted' in query or 'prefer' in query or 'want' in query:
            score += 0.3
        
        # Boost for "acceptable" in context of CM/face (preference query)
        if 'acceptable' in query and ('cm' in query or 'face' in query or 'chief' in query):
            score += 0.4
        
        # Boost if mentions CM/Chief Minister (Bengali: মুখ্যমন্ত্রী)
        if 'cm' in query or 'chief minister' in query or '\u09ae\u09c1\u0996\u09cd\u09af\u09ae\u09a8\u09cd\u09a4\u09cd\u09b0\u09c0' in query:
            score += 0.3
        
        # Boost if asking about party face/leader (Bengali: মুখ, নেতা)
        if 'face' in query or '\u09ae\u09c1\u0996' in query or 'leader' in query or '\u09a8\u09c7\u09a4\u09be' in query:
            score += 0.3
        
        # Strong boost for "CM face" combination (direct survey topic)
        if ('cm' in query or 'chief' in query) and 'face' in query:
            score += 0.4
        
        # Boost if asking "who do people want" pattern (Bengali: কে চাইছে, কাকে চাইছে, কাকে চান)
        bengali_want_pattern = '\u0995\u09c7 \u099a\u09be\u0987\u099b\u09c7|\u0995\u09be\u0995\u09c7 \u099a\u09be\u0987\u099b\u09c7|\u0995\u09be\u0995\u09c7 \u099a\u09be\u09a8'
        if re.search(r'who do .* want|who .* prefer|who would be|' + bengali_want_pattern, query, re.IGNORECASE):
            score += 0.4
        
        # Boost if mentions "people/public preference" (Bengali: জনগণ, জনতা)
        if 'people' in query or 'public' in query or '\u099c\u09a8\u0997\u09a3' in query or '\u099c\u09a8\u09a4\u09be' in query:
            score += 0.2
        
        # Boost if asking about "most popular/preferred"
        if 'most popular' in query or 'most preferred' in query or 'top choice' in query:
            score += 0.4
        
        # Boost if asking about "candidate" for CM/leader position (Bengali: প্রার্থী)
        if ('candidate' in query or '\u09aa\u09cd\u09b0\u09be\u09b0\u09cd\u09a5\u09c0' in query) and ('cm' in query or 'chief' in query or 'leader' in query or '\u09ae\u09c1\u0996\u09cd\u09af\u09ae\u09a8\u09cd\u09a4\u09cd\u09b0\u09c0' in query):
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_strategy(self, query: str) -> bool:
        """Detect if query is asking for strategy/recommendations."""
        return any(p.search(query) for p in self.strategy_patterns_compiled)
    
    def _detect_opinion(self, query: str) -> bool:
        """Detect if query is asking for opinions."""
        return any(p.search(query) for p in self.opinion_patterns_compiled)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        # Candidates
        for pattern in self.candidate_patterns_compiled:
            match = pattern.search(query)
            if match:
                entities.append(match.group(0).title())
        
        # Parties
        for pattern in self.party_patterns_compiled:
            match = pattern.search(query)
            if match:
                entities.append(match.group(0).upper())
        
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'of', 'for', 'in', 'to', 'from', 
                      'how', 'many', 'much', 'who', 'which', 'where', 'when', 'why',
                      'about', 'tell', 'me', 'please', 'can', 'you', 'give'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit to top 10
    
    def _extract_numbers(self, query: str) -> List[int]:
        """Extract numbers from query (for top-N queries)."""
        numbers = re.findall(r'\b(\d+)\b', query)
        return [int(n) for n in numbers]
    
    def _extract_aggregation_target(self, query: str) -> Optional[str]:
        """Extract what the query wants to aggregate."""
        # Common patterns
        patterns = [
            r'(?:how many|count|number of) (.+?)(?:\s+in|\s+for|\s+from|\?|$)',
            r'(?:percentage|%|percent) (?:of )?(.+?)(?:\s+who|\s+that|\?|$)',
            r'voted for (.+?)(?:\s+as|\s+in|\?|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def get_routing_recommendation(self, classified: ClassifiedQuery) -> Dict[str, Any]:
        """Get detailed routing recommendation for a classified query."""
        recommendation = {
            "primary_source": "rag",
            "secondary_sources": [],
            "search_strategy": "hybrid",
            "boost_statistics": False,
            "boost_kg": False,
            "required_chunks": 10
        }
        
        if classified.query_type == QueryType.AGGREGATION:
            recommendation["primary_source"] = "statistics"
            recommendation["secondary_sources"] = ["kg", "rag"]
            recommendation["boost_statistics"] = True
            recommendation["search_strategy"] = "keyword_first"
            recommendation["required_chunks"] = 5
        
        elif classified.query_type == QueryType.SURVEY:
            recommendation["primary_source"] = "statistics"
            recommendation["secondary_sources"] = ["kg", "rag"]
            recommendation["boost_statistics"] = True
            recommendation["boost_kg"] = True
            recommendation["required_chunks"] = 8
        
        elif classified.query_type == QueryType.ENTITY:
            recommendation["primary_source"] = "kg"
            recommendation["secondary_sources"] = ["rag"]
            recommendation["boost_kg"] = True
            recommendation["required_chunks"] = 5
        
        elif classified.intent in [QueryIntent.STRATEGY, QueryIntent.RECOMMENDATION]:
            recommendation["primary_source"] = "rag"
            recommendation["search_strategy"] = "semantic"
            recommendation["required_chunks"] = 15
        
        return recommendation


# Convenience function
def classify_query(query: str) -> ClassifiedQuery:
    """Classify a query using the default classifier."""
    classifier = QueryClassifier()
    return classifier.classify(query)

