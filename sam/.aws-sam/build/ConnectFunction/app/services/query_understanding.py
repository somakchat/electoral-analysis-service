"""
Advanced Query Understanding System.

This module provides intelligent query analysis with:
1. Intent Classification - What does the user want?
2. Entity Extraction - What entities are mentioned?
3. Query Decomposition - Break complex queries into sub-queries
4. Semantic Routing - Route to the best handler
5. Context Enrichment - Add relevant context
6. Confidence Scoring - How confident are we?
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import re
import json
from datetime import datetime

from app.services.llm import get_llm


class QueryIntent(str, Enum):
    """Primary intent of the query."""
    # Factual - Asking for specific facts
    FACTUAL_WHO = "factual_who"           # Who won, who is candidate
    FACTUAL_WHAT = "factual_what"         # What happened, what is the result
    FACTUAL_WHEN = "factual_when"         # When did something happen
    FACTUAL_WHERE = "factual_where"       # Where is constituency
    FACTUAL_COUNT = "factual_count"       # How many seats, votes
    FACTUAL_RESULT = "factual_result"     # Election results
    FACTUAL_PROFILE = "factual_profile"   # Constituency profile
    
    # Analytical - Asking for analysis
    ANALYTICAL_WHY = "analytical_why"     # Why did party lose
    ANALYTICAL_HOW = "analytical_how"     # How did they win
    ANALYTICAL_TREND = "analytical_trend" # What is the trend
    ANALYTICAL_COMPARE = "analytical_compare"  # Compare X vs Y
    ANALYTICAL_COMPARISON = "analytical_comparison"  # Alias for compare
    ANALYTICAL_PATTERN = "analytical_pattern"  # Patterns in data
    
    # Strategic - Asking for strategy
    STRATEGIC_PLAN = "strategic_plan"     # Design a strategy
    STRATEGIC_RECOMMEND = "strategic_recommend"  # What should party do
    STRATEGIC_PRIORITIZE = "strategic_prioritize"  # Where to focus
    STRATEGIC_RESOURCE = "strategic_resource"  # Resource allocation
    STRATEGIC_VOTER = "strategic_voter"   # Voter strategy/segments
    STRATEGIC_CAMPAIGN = "strategic_campaign"  # Campaign strategy
    STRATEGIC_OPPOSITION = "strategic_opposition"  # Opposition analysis
    
    # Predictive - Asking about future
    PREDICTIVE_WILL = "predictive_will"   # Will party win
    PREDICTIVE_FORECAST = "predictive_forecast"  # Forecast results
    PREDICTIVE_OUTCOME = "predictive_outcome"  # Predicted outcome
    PREDICTIVE_PROBABILITY = "predictive_probability"  # Win probability
    PREDICTIVE_SCENARIO = "predictive_scenario"  # What-if scenarios
    
    # Segment Analysis
    SEGMENT_VOTER = "segment_voter"       # Voter segments
    SEGMENT_DEMOGRAPHIC = "segment_demographic"  # Demographics
    
    # Exploratory
    EXPLORATORY_OVERVIEW = "exploratory_overview"  # Overview/summary
    EXPLORATORY_LIST = "exploratory_list"  # List constituencies/data
    EXPLORATORY_SEARCH = "exploratory_search"  # Search for info
    
    # Unknown
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities in political context."""
    CONSTITUENCY = "constituency"
    DISTRICT = "district"
    PARTY = "party"
    CANDIDATE = "candidate"
    YEAR = "year"
    PC = "parliamentary_constituency"
    REGION = "region"


@dataclass
class ExtractedEntity:
    """An extracted entity from the query."""
    text: str
    entity_type: EntityType
    normalized: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class QueryAnalysis:
    """Complete analysis of a user query."""
    original_query: str
    cleaned_query: str
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    entities: List[ExtractedEntity]
    time_context: List[str]
    topic_keywords: List[str]
    sub_queries: List[str]
    complexity_score: int  # 1-5
    confidence: float
    suggested_handler: str
    context_hints: Dict[str, Any]
    requires_llm: bool
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "primary_intent": self.primary_intent.value,
            "entities": [
                {"text": e.text, "type": e.entity_type.value, "normalized": e.normalized}
                for e in self.entities
            ],
            "time_context": self.time_context,
            "complexity": self.complexity_score,
            "confidence": self.confidence,
            "handler": self.suggested_handler,
            "requires_llm": self.requires_llm
        }


class IntentClassifier:
    """
    Classifies query intent using pattern matching and LLM.
    """
    
    # Intent patterns - ordered by priority
    INTENT_PATTERNS = {
        # Factual patterns
        QueryIntent.FACTUAL_WHO: [
            r'\bwho\s+(won|is|was|will|are|were)\b',
            r'\bwinner\s+of\b',
            r'\bcandidate\b',
            r'\bwho\s+is\s+the\b'
        ],
        QueryIntent.FACTUAL_COUNT: [
            r'\bhow\s+many\s+(seats?|votes?|constituencies?)\b',
            r'\btotal\s+(seats?|votes?)\b',
            r'\bnumber\s+of\b',
            r'\bcount\b',
            r'\bseat\s+count\b'
        ],
        QueryIntent.FACTUAL_WHAT: [
            r'\bwhat\s+(is|are|was|were)\b',
            r'\bresult\s+of\b',
            r'\bmargin\s+in\b'
        ],
        QueryIntent.FACTUAL_RESULT: [
            r'\belection\s+result\b',
            r'\bvoting\s+result\b',
            r'\bwon\s+by\b',
            r'\blost\s+by\b'
        ],
        QueryIntent.FACTUAL_PROFILE: [
            r'\bprofile\s+of\b',
            r'\btell\s+me\s+about\b',
            r'\binformation\s+on\b',
            r'\bdetails?\s+(of|about)\b'
        ],
        
        # Analytical patterns
        QueryIntent.ANALYTICAL_WHY: [
            r'\bwhy\s+(did|does|is|was|will)\b',
            r'\breason\s+for\b',
            r'\bcause\s+of\b',
            r'\bexplain\s+why\b'
        ],
        QueryIntent.ANALYTICAL_HOW: [
            r'\bhow\s+(did|does|can|will)\b',
            r'\bin\s+what\s+way\b'
        ],
        QueryIntent.ANALYTICAL_TREND: [
            r'\btrend\b',
            r'\bswing\b',
            r'\bshift\b',
            r'\bchange\s+(in|over)\b',
            r'\bmomentum\b'
        ],
        QueryIntent.ANALYTICAL_COMPARISON: [
            r'\bcompare\b',
            r'\bversus\b',
            r'\bvs\.?\b',
            r'\bdifference\s+between\b',
            r'\bbetter\s+than\b'
        ],
        QueryIntent.ANALYTICAL_PATTERN: [
            r'\bpattern\b',
            r'\bcorrelation\b',
            r'\brelationship\s+between\b'
        ],
        
        # Strategic patterns
        QueryIntent.STRATEGIC_PLAN: [
            r'\bstrategy\b',
            r'\bplan\b',
            r'\bdesign\b',
            r'\bcreate\s+a\b',
            r'\bdevelop\s+a\b'
        ],
        QueryIntent.STRATEGIC_RECOMMEND: [
            r'\bshould\b',
            r'\brecommend\b',
            r'\badvise\b',
            r'\bsuggest\b',
            r'\bwhat\s+to\s+do\b'
        ],
        QueryIntent.STRATEGIC_PRIORITIZE: [
            r'\bprioritize\b',
            r'\bfocus\s+on\b',
            r'\bkey\s+areas?\b',
            r'\bwhere\s+to\s+(invest|focus|concentrate)\b',
            r'\bresource\s+allocation\b'
        ],
        QueryIntent.STRATEGIC_RESOURCE: [
            r'\bresource\b',
            r'\bbudget\b',
            r'\ballocat\b',
            r'\bspending\b',
            r'\binvest\b'
        ],
        QueryIntent.STRATEGIC_VOTER: [
            r'\bvoter\s+segment\b',
            r'\bpersuadable\b',
            r'\btarget\s+(voters?|groups?|audience)\b',
            r'\bvoter\s+(groups?|categories?|turnout)\b',
            r'\belectorate\b',
            r'\bidentify\s+.*\s+(voter|group)\b'
        ],
        QueryIntent.STRATEGIC_CAMPAIGN: [
            r'\bcampaign\b',
            r'\bground\s+game\b',
            r'\brally\b',
            r'\boutreach\b',
            r'\bmobilization\b'
        ],
        QueryIntent.STRATEGIC_OPPOSITION: [
            r'\bopposition\b',
            r'\bopponent\b',
            r'\brival\b',
            r'\bagainst\s+bjp\b',
            r'\bagainst\s+tmc\b',
            r'\bcounter\s+strategy\b'
        ],
        
        # Predictive patterns
        QueryIntent.PREDICTIVE_WILL: [
            r'\bwill\s+(win|lose|get)\b',
            r'\bgoing\s+to\s+(win|lose)\b',
            r'\bexpected\s+to\b'
        ],
        QueryIntent.PREDICTIVE_OUTCOME: [
            r'\bpredict(ed)?\s+outcome\b',
            r'\blikely\s+winner\b',
            r'\bexpected\s+result\b'
        ],
        QueryIntent.PREDICTIVE_FORECAST: [
            r'\bpredict(ion)?\b',
            r'\bforecast\b',
            r'\b2026\b',
            r'\bnext\s+election\b',
            r'\bprojection\b'
        ],
        QueryIntent.PREDICTIVE_PROBABILITY: [
            r'\bprobability\b',
            r'\bchances?\s+of\b',
            r'\blikelihood\b',
            r'\bodds\b'
        ],
        QueryIntent.PREDICTIVE_SCENARIO: [
            r'\bwhat\s+if\b',
            r'\bscenario\b',
            r'\bif\s+.*\s+then\b',
            r'\bhypothetical\b'
        ],
        
        # Segment patterns
        QueryIntent.SEGMENT_VOTER: [
            r'\bvoter\s+segment\b',
            r'\bpersuadable\b',
            r'\btarget\s+(voters?|groups?|audience)\b',
            r'\bvoter\s+(groups?|categories?)\b',
            r'\belectorate\b'
        ],
        QueryIntent.SEGMENT_DEMOGRAPHIC: [
            r'\bdemographic\b',
            r'\bpopulation\b',
            r'\bsc\s+st\b',
            r'\breserved\s+seats?\b',
            r'\bcaste\b',
            r'\breligion\b',
            r'\brural\s+urban\b'
        ],
        
        # Exploratory patterns
        QueryIntent.EXPLORATORY_OVERVIEW: [
            r'\boverview\b',
            r'\bsummar(y|ize)\b',
            r'\boverall\b',
            r'\bgeneral\s+picture\b'
        ],
        QueryIntent.EXPLORATORY_LIST: [
            r'\blist\s+(all|the)\b',
            r'\bshow\s+(me\s+)?(all|the)\b',
            r'\benumerate\b'
        ],
        QueryIntent.EXPLORATORY_SEARCH: [
            r'\bfind\b',
            r'\bsearch\s+for\b',
            r'\blook\s+up\b',
            r'\bwhere\s+can\s+i\b'
        ],
    }
    
    def classify(self, query: str) -> Tuple[QueryIntent, List[QueryIntent], float]:
        """
        Classify query intent.
        
        Returns:
            Tuple of (primary_intent, secondary_intents, confidence)
        """
        query_lower = query.lower()
        intent_scores: Dict[QueryIntent, float] = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.UNKNOWN, [], 0.3
        
        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: -x[1])
        
        primary = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]
        
        # Secondary intents (score > 0.5 of primary)
        secondary = [
            intent for intent, score in sorted_intents[1:]
            if score >= primary_score * 0.5
        ]
        
        # Confidence based on match strength
        confidence = min(0.95, 0.5 + (primary_score * 0.15))
        
        return primary, secondary, confidence


class EntityExtractor:
    """
    Extracts political entities from queries.
    """
    
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self._constituency_names: Set[str] = set()
        self._district_names: Set[str] = set()
        self._pc_names: Set[str] = set()
        
        if self.kg:
            self._load_entity_lists()
    
    def _load_entity_lists(self):
        """Load entity lists from knowledge graph."""
        if hasattr(self.kg, 'constituency_profiles'):
            for name, profile in self.kg.constituency_profiles.items():
                self._constituency_names.add(name.upper())
                self._district_names.add(profile.district.upper())
                self._pc_names.add(profile.parent_pc.upper())
    
    def set_knowledge_graph(self, kg):
        """Set knowledge graph and reload entities."""
        self.kg = kg
        self._load_entity_lists()
    
    # Party patterns
    PARTY_PATTERNS = {
        'BJP': [r'\bbjp\b', r'\bbharatiya\s+janata\b', r'\bbjp\'s\b'],
        'TMC': [r'\btmc\b', r'\btrinamool\b', r'\baitc\b', r'\btmc\'s\b', r'\bmamata\b'],
        'INC': [r'\bcongress\b', r'\binc\b', r'\bcongress\'s\b'],
        'CPM': [r'\bcpm\b', r'\bcpim\b', r'\bcommunist\b', r'\bleft\s+front\b', r'\bleft\b'],
        'AIMIM': [r'\baimim\b', r'\bowaisi\b'],
        'ISF': [r'\bisf\b', r'\bindian\s+secular\s+front\b']
    }
    
    # Year patterns
    YEAR_PATTERN = r'\b(20\d{2}|19\d{2})\b'
    
    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract all entities from query."""
        entities = []
        query_upper = query.upper()
        query_lower = query.lower()
        
        # Extract constituencies
        for name in self._constituency_names:
            if name in query_upper:
                entities.append(ExtractedEntity(
                    text=name,
                    entity_type=EntityType.CONSTITUENCY,
                    normalized=name,
                    confidence=0.95
                ))
        
        # Extract districts
        for name in self._district_names:
            # Avoid matching if already matched as constituency
            if name in query_upper and name not in [e.normalized for e in entities]:
                # Check if "district" is mentioned nearby
                if 'district' in query_lower:
                    entities.append(ExtractedEntity(
                        text=name,
                        entity_type=EntityType.DISTRICT,
                        normalized=name,
                        confidence=0.9
                    ))
                else:
                    # Could be constituency or district - lower confidence
                    entities.append(ExtractedEntity(
                        text=name,
                        entity_type=EntityType.DISTRICT,
                        normalized=name,
                        confidence=0.7
                    ))
        
        # Extract PCs
        for name in self._pc_names:
            if name in query_upper and name not in [e.normalized for e in entities]:
                if any(w in query_lower for w in ['pc', 'lok sabha', 'parliamentary']):
                    entities.append(ExtractedEntity(
                        text=name,
                        entity_type=EntityType.PC,
                        normalized=name,
                        confidence=0.9
                    ))
        
        # Extract parties
        for party, patterns in self.PARTY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    entities.append(ExtractedEntity(
                        text=party,
                        entity_type=EntityType.PARTY,
                        normalized=party,
                        confidence=0.95
                    ))
                    break
        
        # Extract years
        year_matches = re.findall(self.YEAR_PATTERN, query)
        for year in year_matches:
            entities.append(ExtractedEntity(
                text=year,
                entity_type=EntityType.YEAR,
                normalized=year,
                confidence=0.99
            ))
        
        return entities


class QueryDecomposer:
    """
    Decomposes complex queries into simpler sub-queries.
    """
    
    def __init__(self):
        self.llm = None
    
    def _get_llm(self):
        if self.llm is None:
            self.llm = get_llm()
        return self.llm
    
    def decompose(self, query: str, complexity: int) -> List[str]:
        """
        Decompose query into sub-queries.
        
        For simple queries (complexity <= 2), returns [query].
        For complex queries, breaks into logical sub-parts.
        """
        if complexity <= 2:
            return [query]
        
        # Pattern-based decomposition
        sub_queries = self._pattern_decompose(query)
        
        if len(sub_queries) > 1:
            return sub_queries
        
        # LLM-based decomposition for very complex queries
        if complexity >= 4:
            return self._llm_decompose(query)
        
        return [query]
    
    def _pattern_decompose(self, query: str) -> List[str]:
        """Decompose using patterns."""
        sub_queries = []
        
        # Split on "and also", "additionally", etc.
        connectors = [
            r'\s+and\s+also\s+',
            r'\s+additionally\s+',
            r'\s+moreover\s+',
            r'\s+plus\s+',
            r'\s+as\s+well\s+as\s+'
        ]
        
        parts = [query]
        for connector in connectors:
            new_parts = []
            for part in parts:
                split = re.split(connector, part, flags=re.IGNORECASE)
                new_parts.extend(split)
            parts = new_parts
        
        # Clean and filter
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) > 10:  # Minimum length
                sub_queries.append(cleaned)
        
        return sub_queries if len(sub_queries) > 1 else [query]
    
    def _llm_decompose(self, query: str) -> List[str]:
        """Use LLM to decompose complex query."""
        try:
            llm = self._get_llm()
            
            prompt = f"""Break down this complex political query into 2-4 simpler, focused sub-questions.

QUERY: {query}

Return ONLY a JSON array of sub-questions. Example:
["What is X?", "How does Y compare to Z?"]

SUB-QUESTIONS:"""
            
            response = llm.generate(prompt, temperature=0.2)
            
            # Extract JSON array
            match = re.search(r'\[[\s\S]*?\]', response.text)
            if match:
                sub_qs = json.loads(match.group())
                if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                    return sub_qs[:4]
        except Exception:
            pass
        
        return [query]


class HandlerRouter:
    """
    Routes queries to appropriate handlers based on analysis.
    """
    
    # Handler mapping
    INTENT_HANDLER_MAP = {
        # Factual -> constituency_analyst or electoral_strategist
        QueryIntent.FACTUAL_WHO: "constituency_analyst",
        QueryIntent.FACTUAL_WHAT: "constituency_analyst",
        QueryIntent.FACTUAL_COUNT: "electoral_strategist",
        QueryIntent.FACTUAL_WHEN: "constituency_analyst",
        QueryIntent.FACTUAL_WHERE: "constituency_analyst",
        QueryIntent.FACTUAL_RESULT: "electoral_strategist",
        QueryIntent.FACTUAL_PROFILE: "constituency_analyst",
        
        # Analytical -> electoral_strategist
        QueryIntent.ANALYTICAL_WHY: "electoral_strategist",
        QueryIntent.ANALYTICAL_HOW: "electoral_strategist",
        QueryIntent.ANALYTICAL_TREND: "electoral_strategist",
        QueryIntent.ANALYTICAL_COMPARE: "electoral_strategist",
        QueryIntent.ANALYTICAL_COMPARISON: "electoral_strategist",
        QueryIntent.ANALYTICAL_PATTERN: "electoral_strategist",
        
        # Strategic -> varies
        QueryIntent.STRATEGIC_PLAN: "campaign_strategist",
        QueryIntent.STRATEGIC_RECOMMEND: "campaign_strategist",
        QueryIntent.STRATEGIC_PRIORITIZE: "campaign_strategist",
        QueryIntent.STRATEGIC_RESOURCE: "campaign_strategist",
        QueryIntent.STRATEGIC_VOTER: "electoral_strategist",
        QueryIntent.STRATEGIC_CAMPAIGN: "campaign_strategist",
        QueryIntent.STRATEGIC_OPPOSITION: "electoral_strategist",
        
        # Predictive -> electoral_strategist
        QueryIntent.PREDICTIVE_WILL: "electoral_strategist",
        QueryIntent.PREDICTIVE_FORECAST: "electoral_strategist",
        QueryIntent.PREDICTIVE_OUTCOME: "electoral_strategist",
        QueryIntent.PREDICTIVE_PROBABILITY: "electoral_strategist",
        QueryIntent.PREDICTIVE_SCENARIO: "electoral_strategist",
        
        # Segments -> electoral_strategist
        QueryIntent.SEGMENT_VOTER: "electoral_strategist",
        QueryIntent.SEGMENT_DEMOGRAPHIC: "electoral_strategist",
        
        # Exploratory
        QueryIntent.EXPLORATORY_OVERVIEW: "electoral_strategist",
        QueryIntent.EXPLORATORY_LIST: "constituency_analyst",
        QueryIntent.EXPLORATORY_SEARCH: "constituency_analyst",
        
        # Unknown -> electoral_strategist (default)
        QueryIntent.UNKNOWN: "electoral_strategist"
    }
    
    def route(self, analysis: QueryAnalysis) -> str:
        """Determine best handler for the query."""
        
        # Check if constituency-specific with high confidence
        constituency_entity = next(
            (e for e in analysis.entities if e.entity_type == EntityType.CONSTITUENCY),
            None
        )
        
        if constituency_entity and constituency_entity.confidence > 0.8:
            # Constituency-specific queries
            if analysis.primary_intent in [QueryIntent.FACTUAL_WHO, QueryIntent.FACTUAL_WHAT]:
                return "constituency_analyst"
        
        # Check for party-specific strategic queries
        party_entity = next(
            (e for e in analysis.entities if e.entity_type == EntityType.PARTY),
            None
        )
        
        if party_entity and analysis.primary_intent in [
            QueryIntent.STRATEGIC_PLAN, 
            QueryIntent.STRATEGIC_RECOMMEND,
            QueryIntent.STRATEGIC_PRIORITIZE
        ]:
            return "campaign_strategist"
        
        # Default mapping
        return self.INTENT_HANDLER_MAP.get(
            analysis.primary_intent, 
            "electoral_strategist"
        )


class QueryUnderstandingEngine:
    """
    Main engine for query understanding.
    
    Combines all components for comprehensive query analysis.
    """
    
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor(knowledge_graph)
        self.decomposer = QueryDecomposer()
        self.router = HandlerRouter()
    
    def set_knowledge_graph(self, kg):
        """Update knowledge graph reference."""
        self.kg = kg
        self.entity_extractor.set_knowledge_graph(kg)
    
    def analyze(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: User's query
            context: Additional context (session, previous queries, etc.)
        
        Returns:
            QueryAnalysis with all extracted information
        """
        context = context or {}
        
        # Clean query
        cleaned = self._clean_query(query)
        
        # Classify intent
        primary_intent, secondary_intents, intent_confidence = \
            self.intent_classifier.classify(cleaned)
        
        # Extract entities
        entities = self.entity_extractor.extract(cleaned)
        
        # Extract time context
        time_context = self._extract_time_context(cleaned)
        
        # Extract topic keywords
        topic_keywords = self._extract_keywords(cleaned)
        
        # Calculate complexity
        complexity = self._calculate_complexity(
            cleaned, entities, primary_intent, secondary_intents
        )
        
        # Decompose if complex
        sub_queries = self.decomposer.decompose(cleaned, complexity)
        
        # Build context hints
        context_hints = self._build_context_hints(entities, context)
        
        # Determine if LLM is required
        requires_llm = self._requires_llm(primary_intent, complexity)
        
        # Create analysis
        analysis = QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            time_context=time_context,
            topic_keywords=topic_keywords,
            sub_queries=sub_queries,
            complexity_score=complexity,
            confidence=intent_confidence,
            suggested_handler="",  # Will be set by router
            context_hints=context_hints,
            requires_llm=requires_llm
        )
        
        # Route to handler
        analysis.suggested_handler = self.router.route(analysis)
        
        return analysis
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        # Remove some common filler words at start
        cleaned = re.sub(r'^(please|can you|could you|i want to know|tell me)\s+', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def _extract_time_context(self, query: str) -> List[str]:
        """Extract time references."""
        time_refs = []
        
        # Years
        years = re.findall(r'\b(20\d{2})\b', query)
        time_refs.extend(years)
        
        # Relative time
        if any(w in query.lower() for w in ['last election', 'previous']):
            time_refs.append('2021')
        if any(w in query.lower() for w in ['next election', 'upcoming', 'future']):
            time_refs.append('2026')
        if 'lok sabha' in query.lower():
            if '2024' not in time_refs:
                time_refs.append('2024')
            if '2019' not in time_refs:
                time_refs.append('2019')
        
        return list(set(time_refs))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract topic keywords."""
        keywords = []
        
        # Political keywords
        political_terms = [
            'election', 'vote', 'seat', 'margin', 'swing', 'campaign',
            'strategy', 'voter', 'constituency', 'district', 'candidate',
            'win', 'lose', 'prediction', 'poll', 'survey'
        ]
        
        query_lower = query.lower()
        for term in political_terms:
            if term in query_lower:
                keywords.append(term)
        
        return keywords
    
    def _calculate_complexity(
        self, 
        query: str, 
        entities: List[ExtractedEntity],
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent]
    ) -> int:
        """Calculate query complexity (1-5)."""
        complexity = 1
        
        # Length factor
        if len(query) > 100:
            complexity += 1
        if len(query) > 200:
            complexity += 1
        
        # Multiple entities
        if len(entities) >= 2:
            complexity += 1
        if len(entities) >= 4:
            complexity += 1
        
        # Multiple intents
        if len(secondary_intents) >= 2:
            complexity += 1
        
        # Strategic intents are more complex
        if primary_intent in [QueryIntent.STRATEGIC_PLAN, QueryIntent.STRATEGIC_RECOMMEND]:
            complexity += 1
        
        return min(complexity, 5)
    
    def _build_context_hints(
        self, 
        entities: List[ExtractedEntity],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context hints for handlers."""
        hints = {}
        
        # Add entity hints
        for entity in entities:
            if entity.entity_type == EntityType.CONSTITUENCY:
                hints['constituency'] = entity.normalized
            elif entity.entity_type == EntityType.DISTRICT:
                hints['district'] = entity.normalized
            elif entity.entity_type == EntityType.PARTY:
                hints['party'] = entity.normalized
            elif entity.entity_type == EntityType.YEAR:
                hints.setdefault('years', []).append(entity.normalized)
        
        # Add context hints
        if context.get('previous_constituency'):
            hints.setdefault('constituency', context['previous_constituency'])
        if context.get('previous_party'):
            hints.setdefault('party', context['previous_party'])
        
        return hints
    
    def _requires_llm(self, intent: QueryIntent, complexity: int) -> bool:
        """Determine if LLM is required for this query."""
        # Strategic queries need LLM
        if intent in [QueryIntent.STRATEGIC_PLAN, QueryIntent.STRATEGIC_RECOMMEND]:
            return True
        
        # Complex queries need LLM
        if complexity >= 4:
            return True
        
        # Why/How questions need LLM for explanation
        if intent in [QueryIntent.ANALYTICAL_WHY, QueryIntent.ANALYTICAL_HOW]:
            return True
        
        return False


@dataclass
class AdvancedQueryAnalysis:
    """
    Advanced query analysis result with rich understanding.
    This is the format expected by the orchestrator.
    """
    original_query: str
    cleaned_query: str
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    entities: List[ExtractedEntity]
    time_context: List[str]
    topic_keywords: List[str]
    sub_queries: List[str]
    complexity_score: int
    confidence: float
    suggested_agents: List[str]
    reasoning: str
    is_complex: bool
    requires_llm: bool
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "cleaned_query": self.cleaned_query,
            "primary_intent": self.primary_intent.value,
            "secondary_intents": [i.value for i in self.secondary_intents],
            "entities": [
                {"text": e.text, "type": e.entity_type.value, "normalized": e.normalized}
                for e in self.entities
            ],
            "time_context": self.time_context,
            "sub_queries": self.sub_queries,
            "complexity": self.complexity_score,
            "confidence": self.confidence,
            "suggested_agents": self.suggested_agents,
            "reasoning": self.reasoning,
            "is_complex": self.is_complex,
            "requires_llm": self.requires_llm
        }


class EnhancedQueryEngine(QueryUnderstandingEngine):
    """
    Enhanced query understanding engine with advanced NLU.
    
    This is the main entry point used by the orchestrator.
    Provides the `understand` method for query analysis.
    """
    
    def __init__(self, knowledge_graph=None, use_llm: bool = True):
        super().__init__(knowledge_graph)
        self.use_llm = use_llm
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None and self.use_llm:
            self._llm = get_llm()
        return self._llm
    
    def understand(self, query: str, context: Dict[str, Any] = None) -> AdvancedQueryAnalysis:
        """
        Perform advanced query understanding.
        
        This is the main entry point for query analysis.
        Returns AdvancedQueryAnalysis compatible with the orchestrator.
        """
        context = context or {}
        
        # Clean query
        cleaned = self._clean_query(query)
        
        # Classify intent
        primary_intent, secondary_intents, intent_confidence = \
            self.intent_classifier.classify(cleaned)
        
        # Extract entities
        entities = self.entity_extractor.extract(cleaned)
        
        # Extract time context
        time_context = self._extract_time_context(cleaned)
        
        # Extract topic keywords
        topic_keywords = self._extract_keywords(cleaned)
        
        # Calculate complexity
        complexity = self._calculate_complexity(
            cleaned, entities, primary_intent, secondary_intents
        )
        
        # Determine if complex
        is_complex = complexity >= 3
        
        # Decompose if complex
        sub_queries = self.decomposer.decompose(cleaned, complexity)
        
        # Determine if LLM is required
        requires_llm = self._requires_llm(primary_intent, complexity)
        
        # Get suggested agents using router
        basic_analysis = QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            time_context=time_context,
            topic_keywords=topic_keywords,
            sub_queries=sub_queries,
            complexity_score=complexity,
            confidence=intent_confidence,
            suggested_handler=self.router.route(QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned,
                primary_intent=primary_intent,
                secondary_intents=secondary_intents,
                entities=entities,
                time_context=time_context,
                topic_keywords=topic_keywords,
                sub_queries=sub_queries,
                complexity_score=complexity,
                confidence=intent_confidence,
                suggested_handler="",
                context_hints={},
                requires_llm=requires_llm
            )),
            context_hints=self._build_context_hints(entities, context),
            requires_llm=requires_llm
        )
        
        # Get agents for intent
        suggested_agents = self._get_suggested_agents(primary_intent, secondary_intents, entities, is_complex)
        
        # Build reasoning explanation
        reasoning = self._build_reasoning(primary_intent, entities, complexity)
        
        return AdvancedQueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            time_context=time_context,
            topic_keywords=topic_keywords,
            sub_queries=sub_queries,
            complexity_score=complexity,
            confidence=intent_confidence,
            suggested_agents=suggested_agents,
            reasoning=reasoning,
            is_complex=is_complex,
            requires_llm=requires_llm
        )
    
    def _get_suggested_agents(
        self, 
        primary_intent: QueryIntent, 
        secondary_intents: List[QueryIntent],
        entities: List[ExtractedEntity],
        is_complex: bool
    ) -> List[str]:
        """Get suggested agents based on intent and entities."""
        agents = []
        
        # Map primary intent to agent
        primary_handler = self.router.INTENT_HANDLER_MAP.get(primary_intent, "electoral_strategist")
        agents.append(primary_handler)
        
        # Add agents for secondary intents if complex
        if is_complex:
            for sec_intent in secondary_intents[:2]:
                sec_handler = self.router.INTENT_HANDLER_MAP.get(sec_intent)
                if sec_handler and sec_handler not in agents:
                    agents.append(sec_handler)
        
        # Boost based on entities
        entity_types = set(e.entity_type for e in entities)
        
        if EntityType.CONSTITUENCY in entity_types and 'constituency_analyst' not in agents:
            agents.insert(0, 'constituency_analyst')
        
        if EntityType.PARTY in entity_types and 'electoral_strategist' not in agents:
            agents.append('electoral_strategist')
        
        return agents
    
    def _build_reasoning(
        self, 
        primary_intent: QueryIntent, 
        entities: List[ExtractedEntity],
        complexity: int
    ) -> str:
        """Build human-readable reasoning for query understanding."""
        parts = []
        
        # Intent reasoning
        intent_name = primary_intent.value.replace('_', ' ').title()
        parts.append(f"Detected intent: {intent_name}")
        
        # Entity reasoning
        if entities:
            entity_strs = [f"{e.normalized} ({e.entity_type.value})" for e in entities[:3]]
            parts.append(f"Found entities: {', '.join(entity_strs)}")
        else:
            parts.append("No specific entities detected")
        
        # Complexity reasoning
        complexity_levels = {1: "simple", 2: "moderate", 3: "complex", 4: "very complex", 5: "highly complex"}
        parts.append(f"Query complexity: {complexity_levels.get(complexity, 'unknown')}")
        
        return ". ".join(parts) + "."


# Singleton instance
_query_engine: Optional[EnhancedQueryEngine] = None


def get_query_engine(kg=None, use_llm: bool = True) -> EnhancedQueryEngine:
    """Get or create enhanced query understanding engine."""
    global _query_engine
    if _query_engine is None:
        _query_engine = EnhancedQueryEngine(kg, use_llm)
    elif kg is not None:
        _query_engine.set_knowledge_graph(kg)
    return _query_engine


def create_query_engine(kg=None, use_llm: bool = True) -> EnhancedQueryEngine:
    """Create a new enhanced query understanding engine."""
    return EnhancedQueryEngine(kg, use_llm)
