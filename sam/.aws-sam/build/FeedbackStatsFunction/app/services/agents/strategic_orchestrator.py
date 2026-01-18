"""
Strategic Orchestrator - Intelligent coordination of specialized agents.

This orchestrator:
1. Uses advanced query understanding for intent detection
2. Decomposes complex questions into sub-tasks
3. Routes to appropriate specialist agents based on intent
4. Synthesizes responses with full evidence
5. Ensures no hallucination through verification
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import asyncio
import time
from datetime import datetime

from .evidence_framework import (
    PoliticalAgentBase, Evidence, Claim, ReasoningChain, EvidenceType, ConfidenceLevel
)
from .constituency_analyst import ConstituencyIntelligenceAgent
from .electoral_strategist import ElectoralStrategistAgent
from .campaign_strategist import CampaignStrategistAgent
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.llm import get_llm
from app.services.query_understanding import (
    QueryUnderstandingEngine, 
    QueryAnalysis,
    AdvancedQueryAnalysis,
    QueryIntent, 
    create_query_engine
)


class LegacyQueryIntent(str, Enum):
    """Legacy intent types for backward compatibility."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    STRATEGIC = "strategic"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    OPERATIONAL = "operational"


@dataclass
class LegacyQueryAnalysis:
    """Legacy analysis for backward compatibility."""
    original_query: str
    intent: LegacyQueryIntent
    entities: List[str]
    entity_types: Dict[str, str]
    time_context: List[str]
    complexity: int  # 1-5 scale
    sub_questions: List[str]
    required_agents: List[str]


@dataclass
class SubTask:
    """A sub-task for an agent."""
    task_id: str
    description: str
    agent_name: str
    query: str
    dependencies: List[str]
    context: Dict[str, Any]
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class OrchestratedResponse:
    """Final orchestrated response."""
    answer: str
    confidence: float
    claims: List[Dict]
    evidence: List[Dict]
    sources: List[str]
    agents_used: List[str]
    reasoning_trace: List[Dict]
    sub_tasks: List[SubTask]
    execution_time_ms: int
    verification_status: str


class QueryAnalyzer:
    """Legacy query analyzer for backward compatibility."""
    
    INTENT_KEYWORDS = {
        LegacyQueryIntent.FACTUAL: ['who', 'what', 'when', 'where', 'how many', 'which'],
        LegacyQueryIntent.ANALYTICAL: ['why', 'how', 'explain', 'reason', 'cause', 'because'],
        LegacyQueryIntent.STRATEGIC: ['strategy', 'plan', 'should', 'recommend', 'best way', 'how to win'],
        LegacyQueryIntent.COMPARATIVE: ['compare', 'versus', 'vs', 'difference', 'better', 'between'],
        LegacyQueryIntent.PREDICTIVE: ['will', 'predict', 'forecast', 'expect', '2026', 'future', 'likely'],
        LegacyQueryIntent.OPERATIONAL: ['campaign', 'organize', 'execute', 'implement', 'action', 'ground']
    }
    
    def __init__(self, kg):
        self.kg = kg
        self.llm = get_llm()
    
    def analyze(self, query: str) -> LegacyQueryAnalysis:
        """Analyze a query for routing."""
        query_lower = query.lower()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract entities
        entities, entity_types = self._extract_entities(query)
        
        # Detect time context
        time_context = self._extract_time_context(query)
        
        # Calculate complexity
        complexity = self._calculate_complexity(query, entities)
        
        # Generate sub-questions for complex queries
        sub_questions = self._decompose_query(query, complexity)
        
        # Determine required agents
        required_agents = self._determine_agents(intent, entity_types, query_lower)
        
        return LegacyQueryAnalysis(
            original_query=query,
            intent=intent,
            entities=entities,
            entity_types=entity_types,
            time_context=time_context,
            complexity=complexity,
            sub_questions=sub_questions,
            required_agents=required_agents
        )
    
    def _detect_intent(self, query_lower: str) -> LegacyQueryIntent:
        """Detect the primary intent of the query."""
        intent_scores = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            intent_scores[intent] = score
        
        # Get highest scoring intent
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return LegacyQueryIntent.FACTUAL  # Default
    
    def _extract_entities(self, query: str) -> Tuple[List[str], Dict[str, str]]:
        """Extract named entities from query."""
        entities = []
        entity_types = {}
        query_upper = query.upper()
        
        # Constituencies
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                entities.append(name)
                entity_types[name] = 'constituency'
        
        # Districts
        districts = set(p.district for p in self.kg.constituency_profiles.values())
        for district in districts:
            if district.lower() in query.lower():
                entities.append(district)
                entity_types[district] = 'district'
        
        # Parties
        party_patterns = [
            ('BJP', 'party'), ('TMC', 'party'), ('AITC', 'party'),
            ('Congress', 'party'), ('CPM', 'party'), ('Left', 'party')
        ]
        for pattern, etype in party_patterns:
            if pattern.lower() in query.lower():
                entities.append(pattern.upper())
                entity_types[pattern.upper()] = etype
        
        return entities, entity_types
    
    def _extract_time_context(self, query: str) -> List[str]:
        """Extract time references."""
        years = []
        for year in ['2016', '2019', '2021', '2024', '2026']:
            if year in query:
                years.append(year)
        return years
    
    def _calculate_complexity(self, query: str, entities: List[str]) -> int:
        """Calculate query complexity (1-5)."""
        complexity = 1
        
        # Length factor
        if len(query) > 100:
            complexity += 1
        
        # Multiple entities
        if len(entities) >= 2:
            complexity += 1
        if len(entities) >= 4:
            complexity += 1
        
        # Multiple aspects
        aspects = ['strategy', 'campaign', 'predict', 'compare', 'analyze', 'explain']
        if sum(1 for a in aspects if a in query.lower()) >= 2:
            complexity += 1
        
        return min(complexity, 5)
    
    def _decompose_query(self, query: str, complexity: int) -> List[str]:
        """Decompose complex query into sub-questions."""
        if complexity <= 2:
            return [query]
        
        # For complex queries, use LLM to decompose
        try:
            prompt = f"""Decompose this complex political query into 2-4 simpler sub-questions that together answer the original.

Query: {query}

Return ONLY a JSON array of strings, each a sub-question. Example:
["What is the current seat distribution?", "What are the key swing seats?"]

Sub-questions:"""
            
            response = self.llm.generate(prompt, temperature=0.1)
            
            import json
            import re
            match = re.search(r'\[[\s\S]*\]', response.text)
            if match:
                sub_qs = json.loads(match.group())
                return sub_qs[:4]  # Max 4 sub-questions
        except:
            pass
        
        return [query]
    
    def _determine_agents(self, intent: LegacyQueryIntent, entity_types: Dict, query_lower: str) -> List[str]:
        """Determine which agents to involve."""
        agents = []
        
        # Based on intent
        if intent == LegacyQueryIntent.FACTUAL:
            if 'constituency' in entity_types.values():
                agents.append('constituency_analyst')
            else:
                agents.append('electoral_strategist')
        
        elif intent == LegacyQueryIntent.ANALYTICAL:
            agents.append('constituency_analyst')
            agents.append('electoral_strategist')
        
        elif intent == LegacyQueryIntent.STRATEGIC:
            agents.append('electoral_strategist')
            agents.append('campaign_strategist')
        
        elif intent == LegacyQueryIntent.COMPARATIVE:
            agents.append('constituency_analyst')
        
        elif intent == LegacyQueryIntent.PREDICTIVE:
            agents.append('electoral_strategist')
        
        elif intent == LegacyQueryIntent.OPERATIONAL:
            agents.append('campaign_strategist')
        
        # Add based on keywords
        if 'campaign' in query_lower or 'ground' in query_lower:
            if 'campaign_strategist' not in agents:
                agents.append('campaign_strategist')
        
        if 'strategy' in query_lower or 'win' in query_lower:
            if 'electoral_strategist' not in agents:
                agents.append('electoral_strategist')
        
        return agents if agents else ['electoral_strategist']


class StrategicOrchestrator:
    """
    Main orchestrator for political strategy queries.
    
    Coordinates multiple specialist agents using advanced
    query understanding for intelligent routing and response.
    """
    
    def __init__(self, rag: PoliticalRAGSystem, feedback_manager=None):
        self.rag = rag
        self.kg = rag.kg
        self.llm = get_llm()
        self.feedback_manager = feedback_manager
        
        # Advanced query understanding engine
        self.query_engine = create_query_engine(self.kg, use_llm=True)
        
        # Legacy analyzer for backward compatibility
        self.analyzer = QueryAnalyzer(self.kg)
        
        # Initialize agents
        self.agents = {
            'constituency_analyst': ConstituencyIntelligenceAgent(rag),
            'electoral_strategist': ElectoralStrategistAgent(rag),
            'campaign_strategist': CampaignStrategistAgent(rag)
        }
        
        # Intent to agent mapping - comprehensive for all intents
        self._intent_agent_map = {
            # Factual intents
            QueryIntent.FACTUAL_WHO: ['constituency_analyst'],
            QueryIntent.FACTUAL_WHAT: ['constituency_analyst'],
            QueryIntent.FACTUAL_WHEN: ['constituency_analyst'],
            QueryIntent.FACTUAL_WHERE: ['constituency_analyst'],
            QueryIntent.FACTUAL_RESULT: ['electoral_strategist'],
            QueryIntent.FACTUAL_COUNT: ['electoral_strategist'],
            QueryIntent.FACTUAL_PROFILE: ['constituency_analyst'],
            
            # Analytical intents
            QueryIntent.ANALYTICAL_WHY: ['electoral_strategist', 'constituency_analyst'],
            QueryIntent.ANALYTICAL_HOW: ['electoral_strategist'],
            QueryIntent.ANALYTICAL_TREND: ['electoral_strategist'],
            QueryIntent.ANALYTICAL_COMPARE: ['constituency_analyst', 'electoral_strategist'],
            QueryIntent.ANALYTICAL_COMPARISON: ['constituency_analyst', 'electoral_strategist'],
            QueryIntent.ANALYTICAL_PATTERN: ['electoral_strategist'],
            
            # Strategic intents
            QueryIntent.STRATEGIC_PLAN: ['electoral_strategist', 'campaign_strategist'],
            QueryIntent.STRATEGIC_RECOMMEND: ['electoral_strategist', 'campaign_strategist'],
            QueryIntent.STRATEGIC_PRIORITIZE: ['campaign_strategist'],
            QueryIntent.STRATEGIC_RESOURCE: ['campaign_strategist'],
            QueryIntent.STRATEGIC_VOTER: ['electoral_strategist'],
            QueryIntent.STRATEGIC_CAMPAIGN: ['campaign_strategist'],
            QueryIntent.STRATEGIC_OPPOSITION: ['electoral_strategist'],
            
            # Predictive intents
            QueryIntent.PREDICTIVE_WILL: ['electoral_strategist'],
            QueryIntent.PREDICTIVE_FORECAST: ['electoral_strategist'],
            QueryIntent.PREDICTIVE_OUTCOME: ['electoral_strategist'],
            QueryIntent.PREDICTIVE_PROBABILITY: ['electoral_strategist'],
            QueryIntent.PREDICTIVE_SCENARIO: ['electoral_strategist', 'campaign_strategist'],
            
            # Segment intents
            QueryIntent.SEGMENT_VOTER: ['electoral_strategist'],
            QueryIntent.SEGMENT_DEMOGRAPHIC: ['electoral_strategist'],
            
            # Exploratory intents
            QueryIntent.EXPLORATORY_OVERVIEW: ['electoral_strategist'],
            QueryIntent.EXPLORATORY_LIST: ['constituency_analyst'],
            QueryIntent.EXPLORATORY_SEARCH: ['constituency_analyst'],
            
            # Unknown
            QueryIntent.UNKNOWN: ['electoral_strategist'],
        }
    
    async def process_query(self, 
                           query: str,
                           session_id: str = None,
                           context: Dict[str, Any] = None,
                           send_update: Callable = None) -> OrchestratedResponse:
        """
        Process a query through the orchestration pipeline.
        
        Args:
            query: User's question
            session_id: Session identifier
            context: Additional context (constituency, party, etc.)
            send_update: Callback for streaming updates
        
        Returns:
            OrchestratedResponse with full answer and evidence
        """
        start_time = time.time()
        context = context or {}
        
        # Step 1: Advanced query understanding
        query_analysis = self.query_engine.understand(query, context)
        
        if send_update:
            await send_update({
                "type": "query_understanding",
                "message": f"Understanding query: {query_analysis.primary_intent.value}",
                "intent": query_analysis.primary_intent.value,
                "entities": [e.normalized for e in query_analysis.entities],
                "confidence": query_analysis.confidence,
                "reasoning": query_analysis.reasoning,
                "is_complex": query_analysis.is_complex,
                "sub_queries": query_analysis.sub_queries if query_analysis.is_complex else []
            })
        
        # Map to agents using advanced understanding
        required_agents = self._get_agents_for_intent(query_analysis)
        
        # Also run legacy analysis for compatibility
        analysis = self.analyzer.analyze(query)
        # Update analysis with advanced understanding
        analysis.required_agents = required_agents
        analysis.sub_questions = query_analysis.sub_queries
        
        if send_update:
            await send_update({
                "type": "analysis",
                "message": f"Query classified: {query_analysis.primary_intent.value}, complexity {'high' if query_analysis.is_complex else 'standard'}",
                "entities": [e.normalized for e in query_analysis.entities],
                "suggested_agents": required_agents
            })
        
        # Step 2: Create sub-tasks
        sub_tasks = self._create_sub_tasks(analysis, context)
        
        if send_update:
            await send_update({
                "type": "planning",
                "message": f"Created {len(sub_tasks)} sub-tasks",
                "agents": analysis.required_agents
            })
        
        # Step 3: Execute sub-tasks
        results = []
        all_claims = []
        all_evidence = []
        all_sources = set()
        agents_used = set()
        reasoning_trace = []
        
        for task in sub_tasks:
            if send_update:
                await send_update({
                    "type": "agent_activity",
                    "agent": task.agent_name,
                    "status": "working",
                    "task": task.description
                })
            
            try:
                agent = self.agents.get(task.agent_name)
                if agent:
                    result = agent.analyze(task.query, task.context)
                    task.result = result
                    task.status = "completed"
                    
                    results.append(result)
                    agents_used.add(task.agent_name)
                    
                    # Collect claims and evidence
                    if 'claims' in result:
                        all_claims.extend(result['claims'])
                    if 'evidence' in result:
                        all_evidence.extend(result['evidence'])
                    if 'reasoning' in result and hasattr(result['reasoning'], 'sources_used'):
                        all_sources.update(result['reasoning'].sources_used)
                    
                    reasoning_trace.append({
                        "agent": task.agent_name,
                        "task": task.description,
                        "confidence": result.get('confidence', 0.5)
                    })
                    
                    if send_update:
                        await send_update({
                            "type": "agent_activity",
                            "agent": task.agent_name,
                            "status": "completed",
                            "confidence": result.get('confidence', 0.5)
                        })
                else:
                    task.status = "skipped"
                    task.error = f"Agent {task.agent_name} not found"
                    
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                
                if send_update:
                    await send_update({
                        "type": "agent_activity",
                        "agent": task.agent_name,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Step 4: Synthesize response
        if send_update:
            await send_update({
                "type": "synthesizing",
                "message": "Combining agent insights"
            })
        
        final_answer = self._synthesize_response(query, analysis, results, all_claims)
        
        # Calculate confidence
        if results:
            avg_confidence = sum(r.get('confidence', 0.5) for r in results) / len(results)
        else:
            avg_confidence = 0.3
        
        # Verify response
        verification_status = "verified" if avg_confidence >= 0.8 else "high_confidence" if avg_confidence >= 0.6 else "moderate_confidence"
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return OrchestratedResponse(
            answer=final_answer,
            confidence=avg_confidence,
            claims=all_claims,
            evidence=all_evidence,
            sources=list(all_sources),
            agents_used=list(agents_used),
            reasoning_trace=reasoning_trace,
            sub_tasks=sub_tasks,
            execution_time_ms=execution_time,
            verification_status=verification_status
        )
    
    def _create_sub_tasks(self, analysis: LegacyQueryAnalysis, context: Dict) -> List[SubTask]:
        """Create sub-tasks based on query analysis."""
        tasks = []
        
        # Create task for each sub-question and required agent
        for i, sub_q in enumerate(analysis.sub_questions):
            for agent_name in analysis.required_agents:
                agent = self.agents.get(agent_name)
                if agent:
                    can_handle, confidence = agent.can_handle(sub_q)
                    if can_handle and confidence >= 0.5:
                        tasks.append(SubTask(
                            task_id=f"task_{i}_{agent_name}",
                            description=f"Analyze: {sub_q[:50]}...",
                            agent_name=agent_name,
                            query=sub_q,
                            dependencies=[],
                            context={
                                **context,
                                'entities': analysis.entities,
                                'entity_types': analysis.entity_types
                            }
                        ))
                        break  # One agent per sub-question
        
        # If no tasks created, create default task
        if not tasks:
            tasks.append(SubTask(
                task_id="task_default",
                description="General analysis",
                agent_name="electoral_strategist",
                query=analysis.original_query,
                dependencies=[],
                context=context
            ))
        
        return tasks
    
    def _synthesize_response(self, 
                            original_query: str,
                            analysis: LegacyQueryAnalysis,
                            results: List[Dict],
                            claims: List[Dict]) -> str:
        """Synthesize final response from agent results."""
        
        if not results:
            return "I don't have sufficient data to answer this query. Please try a more specific question about West Bengal electoral data."
        
        # If single result, return directly
        if len(results) == 1:
            return results[0].get('answer', 'No answer generated.')
        
        # For multiple results, synthesize
        combined_answers = "\n\n".join(r.get('answer', '') for r in results if r.get('answer'))
        
        # Use LLM to create cohesive response
        try:
            synthesis_prompt = f"""You are synthesizing multiple expert analyses into a single cohesive response.

ORIGINAL QUESTION: {original_query}

EXPERT ANALYSES:
{combined_answers[:6000]}

INSTRUCTIONS:
1. Create a unified, well-structured response
2. Remove redundancy while keeping all unique insights
3. Maintain evidence citations where present
4. Use markdown formatting for clarity
5. Keep the factual accuracy - don't add new claims
6. Preserve numerical data exactly as provided

SYNTHESIZED RESPONSE:"""

            response = self.llm.generate(
                synthesis_prompt,
                system="You are a senior political analyst synthesizing expert insights. Be accurate, clear, and evidence-based.",
                temperature=0.3
            )
            
            return response.text
            
        except Exception as e:
            # Fallback: combine answers with headers
            return combined_answers
    
    def query_sync(self, query: str, context: Dict[str, Any] = None) -> OrchestratedResponse:
        """Synchronous wrapper for process_query."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_query(query, context=context))
    
    def _get_agents_for_intent(self, query_analysis) -> List[str]:
        """Get appropriate agents based on advanced query analysis."""
        primary_intent = query_analysis.primary_intent
        query_lower = query_analysis.original_query.lower()
        
        # Special routing for general analytical queries (no specific constituency)
        # These should go to electoral_strategist first
        general_analytical_keywords = [
            'swing seat', 'competitive', 'battleground', 'safe seat', 'stronghold',
            'all constituencies', 'total seats', 'how many', 'list', 'overview',
            'west bengal', 'state', 'overall', 'prediction', 'forecast',
            'party strength', 'who will win', 'election result'
        ]
        
        is_general_query = any(kw in query_lower for kw in general_analytical_keywords)
        
        # Check if specific constituency is mentioned
        entity_types = [e.entity_type.value for e in query_analysis.entities]
        has_constituency = 'constituency' in entity_types
        
        # Route general queries without specific constituency to electoral_strategist
        if is_general_query and not has_constituency:
            return ['electoral_strategist']
        
        # Get agents for primary intent
        agents = list(self._intent_agent_map.get(primary_intent, ['electoral_strategist']))
        
        # Add agents for secondary intents if complex
        if query_analysis.is_complex:
            for sec_intent in query_analysis.secondary_intents[:2]:
                sec_agents = self._intent_agent_map.get(sec_intent, [])
                for agent in sec_agents:
                    if agent not in agents:
                        agents.append(agent)
        
        # Boost certain agents based on entities
        if has_constituency and 'constituency_analyst' not in agents:
            agents.insert(0, 'constituency_analyst')
        
        if 'party' in entity_types and 'electoral_strategist' not in agents:
            agents.append('electoral_strategist')
        
        # Default to electoral_strategist for any unknown routing
        if not agents:
            agents = ['electoral_strategist']
        
        return agents
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents."""
        return {
            name: agent.expertise
            for name, agent in self.agents.items()
        }
    
    def get_query_understanding_summary(self, query: str) -> Dict:
        """Get a summary of how the system understands a query."""
        analysis = self.query_engine.understand(query)
        return {
            "original_query": analysis.original_query,
            "cleaned_query": analysis.cleaned_query,
            "primary_intent": analysis.primary_intent.value,
            "secondary_intents": [i.value for i in analysis.secondary_intents],
            "entities": [
                {"text": e.text, "type": e.entity_type.value, "normalized": e.normalized}
                for e in analysis.entities
            ],
            "time_context": analysis.time_context,
            "is_complex": analysis.is_complex,
            "sub_queries": analysis.sub_queries,
            "suggested_agents": analysis.suggested_agents,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning
        }

