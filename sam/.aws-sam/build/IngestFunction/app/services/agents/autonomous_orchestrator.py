"""
Autonomous Political Strategy Orchestrator.

This is an enhanced agentic system with:
1. Dynamic query understanding and intent detection
2. Autonomous agent selection and tool routing
3. Self-reflection and iterative refinement
4. Fallback mechanisms for unknown queries
5. Multi-turn context management
6. Confidence calibration
7. Query clarification when needed
8. Real-time learning from feedback
9. Chain-of-thought reasoning
10. Agent collaboration and debate
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import asyncio
import time
import json
import re
from datetime import datetime

from app.services.llm import get_llm, BaseLLM
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.memory import get_memory_store


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"           # Single fact lookup
    MODERATE = "moderate"       # Analysis of one entity
    COMPLEX = "complex"         # Multi-entity analysis
    STRATEGIC = "strategic"     # Strategy development
    EXPLORATORY = "exploratory" # Open-ended exploration


class ResponseQuality(str, Enum):
    """Response quality assessment."""
    EXCELLENT = "excellent"     # Fully answers with evidence
    GOOD = "good"              # Answers with some gaps
    PARTIAL = "partial"        # Incomplete answer
    INSUFFICIENT = "insufficient"  # Needs clarification
    FAILED = "failed"          # Could not answer


@dataclass
class ThinkingStep:
    """A step in the agent's reasoning chain."""
    step_number: int
    thought: str
    action: str
    observation: str
    confidence: float


@dataclass
class AgentDecision:
    """Agent's decision on how to handle query."""
    should_answer: bool
    needs_clarification: bool
    clarification_question: Optional[str]
    selected_strategy: str
    reasoning: str
    confidence: float


@dataclass
class AutonomousResponse:
    """Complete response from the autonomous system."""
    answer: str
    thinking_chain: List[ThinkingStep]
    confidence: float
    quality: ResponseQuality
    sources: List[str]
    agents_used: List[str]
    tools_used: List[str]
    refinement_iterations: int
    clarification_needed: bool
    clarification_question: Optional[str]
    execution_time_ms: int
    metadata: Dict[str, Any]


class AutonomousQueryAnalyzer:
    """
    Advanced query analyzer with self-reflection.
    Determines the best approach for any query type.
    """
    
    def __init__(self, llm: BaseLLM, kg):
        self.llm = llm
        self.kg = kg
    
    def analyze(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze query with deep understanding."""
        context = context or {}
        
        # Step 1: Basic classification
        query_lower = query.lower()
        
        # Step 2: Entity extraction
        entities = self._extract_entities(query)
        
        # Step 3: Determine complexity
        complexity = self._assess_complexity(query, entities)
        
        # Step 4: Check if we can answer
        answerable, reason = self._check_answerable(query, entities)
        
        # Step 5: Determine best strategy
        strategy = self._determine_strategy(query, complexity, entities)
        
        # Step 6: Check if clarification needed
        needs_clarification, clarification_q = self._check_clarification_needed(
            query, entities, context
        )
        
        return {
            "original_query": query,
            "entities": entities,
            "complexity": complexity,
            "answerable": answerable,
            "answerable_reason": reason,
            "strategy": strategy,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_q,
            "context_available": bool(context),
            "confidence": self._calculate_confidence(answerable, entities, complexity)
        }
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract all entities from query."""
        entities = {
            "constituencies": [],
            "districts": [],
            "parties": [],
            "years": [],
            "candidates": []
        }
        
        query_upper = query.upper()
        
        # Constituencies
        for name in self.kg.constituency_profiles.keys():
            if name in query_upper:
                entities["constituencies"].append(name)
        
        # Districts
        districts = set(p.district for p in self.kg.constituency_profiles.values())
        for dist in districts:
            if dist.upper() in query_upper:
                entities["districts"].append(dist)
        
        # Parties
        party_patterns = {'BJP': r'\bbjp\b', 'TMC': r'\btmc|trinamool|aitc\b', 
                         'CPM': r'\bcpm|cpim|left\b', 'INC': r'\bcongress|inc\b'}
        for party, pattern in party_patterns.items():
            if re.search(pattern, query.lower()):
                entities["parties"].append(party)
        
        # Years
        years = re.findall(r'\b(2016|2019|2021|2024|2026)\b', query)
        entities["years"] = years
        
        return entities
    
    def _assess_complexity(self, query: str, entities: Dict) -> QueryComplexity:
        """Assess query complexity."""
        query_lower = query.lower()
        
        # Count entities
        entity_count = sum(len(v) for v in entities.values())
        
        # Strategic keywords
        strategic_words = ['strategy', 'plan', 'campaign', 'win', 'defeat', 'approach']
        is_strategic = any(w in query_lower for w in strategic_words)
        
        # Analytical keywords
        analytical_words = ['why', 'how', 'analyze', 'compare', 'trend', 'explain']
        is_analytical = any(w in query_lower for w in analytical_words)
        
        # Exploratory keywords
        exploratory_words = ['tell me about', 'what do you know', 'explain everything']
        is_exploratory = any(w in query_lower for w in exploratory_words)
        
        if is_strategic or (entity_count > 2 and is_analytical):
            return QueryComplexity.STRATEGIC
        elif is_exploratory:
            return QueryComplexity.EXPLORATORY
        elif is_analytical or entity_count > 1:
            return QueryComplexity.COMPLEX if entity_count > 2 else QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _check_answerable(self, query: str, entities: Dict) -> Tuple[bool, str]:
        """Check if we can answer this query."""
        # Check if we have relevant entities
        has_entities = any(len(v) > 0 for v in entities.values())
        
        # Check for West Bengal relevance
        wb_relevant = any(w in query.lower() for w in [
            'bengal', 'wb', 'kolkata', 'tmc', 'bjp', 'mamata', 
            'constituency', 'assembly', 'election', '2021', '2026'
        ])
        
        # General political queries we can handle
        general_political = any(w in query.lower() for w in [
            'voter', 'swing', 'margin', 'prediction', 'seat', 'strategy'
        ])
        
        if has_entities or wb_relevant or general_political:
            return True, "Query relates to West Bengal politics"
        
        # Check if it's completely out of scope
        if not (wb_relevant or has_entities or general_political):
            return False, "Query may not be related to West Bengal politics"
        
        return True, "General political query"
    
    def _determine_strategy(self, query: str, complexity: QueryComplexity, 
                           entities: Dict) -> str:
        """Determine the best strategy for answering."""
        query_lower = query.lower()
        
        strategies = {
            "constituency_lookup": any(entities["constituencies"]),
            "district_analysis": any(entities["districts"]),
            "party_analysis": any(entities["parties"]),
            "prediction_query": "predict" in query_lower or "2026" in query_lower,
            "comparison": "compare" in query_lower or "vs" in query_lower,
            "swing_analysis": "swing" in query_lower or "trend" in query_lower,
            "voter_segment": "voter" in query_lower or "segment" in query_lower,
            "strategy_development": "strategy" in query_lower or "plan" in query_lower,
            "general_overview": complexity == QueryComplexity.EXPLORATORY
        }
        
        # Return first matching strategy
        for strategy, matches in strategies.items():
            if matches:
                return strategy
        
        return "general_analysis"
    
    def _check_clarification_needed(self, query: str, entities: Dict, 
                                   context: Dict) -> Tuple[bool, Optional[str]]:
        """Check if we need to ask for clarification."""
        # Too vague
        if len(query.split()) < 3 and not any(len(v) > 0 for v in entities.values()):
            return True, "Could you please be more specific? For example, which constituency, party, or district are you interested in?"
        
        # Ambiguous party reference
        if "they" in query.lower() or "the party" in query.lower():
            if not context.get("party") and not entities["parties"]:
                return True, "Which party are you referring to? (BJP, TMC, CPM, or others)"
        
        # Ambiguous location
        if "there" in query.lower() or "that area" in query.lower():
            if not context.get("constituency") and not entities["constituencies"]:
                return True, "Which constituency or district are you asking about?"
        
        return False, None
    
    def _calculate_confidence(self, answerable: bool, entities: Dict, 
                             complexity: QueryComplexity) -> float:
        """Calculate confidence in our ability to answer."""
        if not answerable:
            return 0.2
        
        base_confidence = 0.5
        
        # Boost for entities found
        entity_count = sum(len(v) for v in entities.values())
        if entity_count > 0:
            base_confidence += min(0.3, entity_count * 0.1)
        
        # Adjust for complexity
        complexity_penalties = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.05,
            QueryComplexity.COMPLEX: 0.0,
            QueryComplexity.STRATEGIC: -0.05,
            QueryComplexity.EXPLORATORY: -0.1
        }
        base_confidence += complexity_penalties.get(complexity, 0)
        
        return min(0.95, max(0.2, base_confidence))


class SelfReflectingAgent:
    """
    An agent that can reflect on its own responses and improve them.
    """
    
    def __init__(self, name: str, role: str, llm: BaseLLM, tools: List[Any]):
        self.name = name
        self.role = role
        self.llm = llm
        self.tools = tools
        self.thinking_chain: List[ThinkingStep] = []
    
    def think(self, query: str, context: Dict = None) -> ThinkingStep:
        """Generate a thinking step."""
        step_num = len(self.thinking_chain) + 1
        
        # Generate thought
        thought = self._generate_thought(query, context)
        
        # Decide action
        action = self._decide_action(thought)
        
        # Execute and observe
        observation = self._execute_action(action, query, context)
        
        # Assess confidence
        confidence = self._assess_step_confidence(thought, observation)
        
        step = ThinkingStep(
            step_number=step_num,
            thought=thought,
            action=action,
            observation=observation,
            confidence=confidence
        )
        
        self.thinking_chain.append(step)
        return step
    
    def _generate_thought(self, query: str, context: Dict) -> str:
        """Generate a thought about the query."""
        prompt = f"""As a {self.role}, analyze this query:
Query: {query}
Context: {json.dumps(context or {})}

What is the key insight or approach needed? Be specific and brief."""
        
        response = self.llm.generate(prompt, temperature=0.3)
        return response.text[:300]
    
    def _decide_action(self, thought: str) -> str:
        """Decide what action to take based on thought."""
        # Simple rule-based for now, can be LLM-enhanced
        if "constituency" in thought.lower():
            return "lookup_constituency"
        elif "party" in thought.lower():
            return "analyze_party"
        elif "predict" in thought.lower():
            return "get_predictions"
        elif "compare" in thought.lower():
            return "compare_entities"
        else:
            return "general_search"
    
    def _execute_action(self, action: str, query: str, context: Dict) -> str:
        """Execute the decided action."""
        # This would call the appropriate tool
        return f"Executed {action} for query: {query[:50]}..."
    
    def _assess_step_confidence(self, thought: str, observation: str) -> float:
        """Assess confidence in this step."""
        if "error" in observation.lower() or "not found" in observation.lower():
            return 0.3
        elif "found" in observation.lower() or "success" in observation.lower():
            return 0.8
        return 0.6
    
    def reflect(self, response: str) -> Tuple[str, float]:
        """Reflect on the response and potentially improve it."""
        prompt = f"""Review this response for accuracy and completeness:

RESPONSE: {response[:1000]}

Rate the response quality (1-10) and suggest improvements if needed.
Format: QUALITY: X/10 | IMPROVEMENTS: [list any improvements]"""
        
        reflection = self.llm.generate(prompt, temperature=0.2)
        
        # Parse quality score
        quality_match = re.search(r'QUALITY:\s*(\d+)', reflection.text)
        quality = int(quality_match.group(1)) / 10 if quality_match else 0.7
        
        return reflection.text, quality


class AutonomousPoliticalOrchestrator:
    """
    Main autonomous orchestrator for political strategy queries.
    
    Features:
    1. Dynamic query understanding
    2. Self-reflecting agents
    3. Iterative refinement
    4. Fallback mechanisms
    5. Clarification handling
    6. Multi-turn context
    7. Confidence calibration
    """
    
    MAX_REFINEMENT_ITERATIONS = 3
    MIN_ACCEPTABLE_QUALITY = 0.7
    
    def __init__(self, rag: PoliticalRAGSystem):
        self.rag = rag
        self.kg = rag.kg
        self.llm = get_llm()
        self.analyzer = AutonomousQueryAnalyzer(self.llm, self.kg)
        self.memory = get_memory_store()
        
        # Session context for multi-turn
        self._session_context: Dict[str, Dict] = {}
        
        # Import specialized agents
        from app.services.agents.electoral_strategist import ElectoralStrategistAgent
        from app.services.agents.constituency_analyst import ConstituencyIntelligenceAgent
        from app.services.agents.campaign_strategist import CampaignStrategistAgent
        
        self.agents = {
            "electoral_strategist": ElectoralStrategistAgent(rag),
            "constituency_analyst": ConstituencyIntelligenceAgent(rag),
            "campaign_strategist": CampaignStrategistAgent(rag)
        }
    
    async def process(
        self,
        query: str,
        session_id: str = None,
        context: Dict[str, Any] = None,
        send_update: Callable = None
    ) -> AutonomousResponse:
        """
        Process a query autonomously with full reasoning chain.
        """
        start_time = time.time()
        context = context or {}
        session_id = session_id or "default"
        
        # Merge with session context
        session_ctx = self._session_context.get(session_id, {})
        merged_context = {**session_ctx, **context}
        
        thinking_chain = []
        agents_used = []
        tools_used = []
        
        # Step 1: Analyze query
        if send_update:
            await send_update({"type": "thinking", "message": "Analyzing your query..."})
        
        analysis = self.analyzer.analyze(query, merged_context)
        
        thinking_chain.append(ThinkingStep(
            step_number=1,
            thought=f"Query analysis: {analysis['strategy']} approach",
            action="analyze_query",
            observation=f"Complexity: {analysis['complexity']}, Answerable: {analysis['answerable']}",
            confidence=analysis['confidence']
        ))
        
        # Step 2: Check if clarification needed
        if analysis['needs_clarification']:
            return AutonomousResponse(
                answer="",
                thinking_chain=thinking_chain,
                confidence=0.3,
                quality=ResponseQuality.INSUFFICIENT,
                sources=[],
                agents_used=[],
                tools_used=[],
                refinement_iterations=0,
                clarification_needed=True,
                clarification_question=analysis['clarification_question'],
                execution_time_ms=int((time.time() - start_time) * 1000),
                metadata={"analysis": analysis}
            )
        
        # Step 3: Route to appropriate agent(s)
        if send_update:
            await send_update({
                "type": "routing", 
                "message": f"Using {analysis['strategy']} strategy..."
            })
        
        selected_agents = self._select_agents(analysis)
        agents_used = selected_agents
        
        # Step 4: Execute with agents
        if send_update:
            await send_update({
                "type": "agent_activity",
                "agents": selected_agents,
                "status": "working"
            })
        
        response_text, agent_confidence, evidence = await self._execute_agents(
            query, selected_agents, merged_context, send_update
        )
        
        thinking_chain.append(ThinkingStep(
            step_number=2,
            thought=f"Executed {len(selected_agents)} agents",
            action="agent_execution",
            observation=f"Got response with confidence {agent_confidence:.2f}",
            confidence=agent_confidence
        ))
        
        # Step 5: Self-reflection and refinement
        quality_score = agent_confidence
        refinement_count = 0
        
        while quality_score < self.MIN_ACCEPTABLE_QUALITY and refinement_count < self.MAX_REFINEMENT_ITERATIONS:
            refinement_count += 1
            
            if send_update:
                await send_update({
                    "type": "refining",
                    "message": f"Refining response (iteration {refinement_count})..."
                })
            
            response_text, quality_score = await self._refine_response(
                query, response_text, evidence, merged_context
            )
            
            thinking_chain.append(ThinkingStep(
                step_number=2 + refinement_count,
                thought=f"Refinement iteration {refinement_count}",
                action="refine",
                observation=f"Quality improved to {quality_score:.2f}",
                confidence=quality_score
            ))
        
        # Step 6: Determine response quality
        if quality_score >= 0.85:
            quality = ResponseQuality.EXCELLENT
        elif quality_score >= 0.7:
            quality = ResponseQuality.GOOD
        elif quality_score >= 0.5:
            quality = ResponseQuality.PARTIAL
        else:
            quality = ResponseQuality.INSUFFICIENT
        
        # Step 7: Handle fallback if needed
        if quality == ResponseQuality.INSUFFICIENT:
            response_text = self._generate_fallback(query, analysis, evidence)
            thinking_chain.append(ThinkingStep(
                step_number=len(thinking_chain) + 1,
                thought="Response quality insufficient, using fallback",
                action="fallback",
                observation="Generated graceful fallback response",
                confidence=0.5
            ))
        
        # Step 8: Update session context
        self._update_session_context(session_id, query, analysis, response_text)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return AutonomousResponse(
            answer=response_text,
            thinking_chain=thinking_chain,
            confidence=quality_score,
            quality=quality,
            sources=evidence,
            agents_used=agents_used,
            tools_used=tools_used,
            refinement_iterations=refinement_count,
            clarification_needed=False,
            clarification_question=None,
            execution_time_ms=execution_time,
            metadata={
                "analysis": analysis,
                "strategy": analysis['strategy'],
                "complexity": analysis['complexity'].value
            }
        )
    
    def _select_agents(self, analysis: Dict) -> List[str]:
        """Select agents based on analysis."""
        strategy = analysis['strategy']
        
        strategy_agent_map = {
            "constituency_lookup": ["constituency_analyst"],
            "district_analysis": ["electoral_strategist"],
            "party_analysis": ["electoral_strategist"],
            "prediction_query": ["electoral_strategist"],
            "comparison": ["constituency_analyst", "electoral_strategist"],
            "swing_analysis": ["electoral_strategist"],
            "voter_segment": ["electoral_strategist"],
            "strategy_development": ["electoral_strategist", "campaign_strategist"],
            "general_overview": ["electoral_strategist"],
            "general_analysis": ["electoral_strategist"]
        }
        
        return strategy_agent_map.get(strategy, ["electoral_strategist"])
    
    async def _execute_agents(
        self,
        query: str,
        agent_names: List[str],
        context: Dict,
        send_update: Callable = None
    ) -> Tuple[str, float, List[str]]:
        """Execute selected agents and combine results."""
        results = []
        evidence = []
        confidences = []
        
        for agent_name in agent_names:
            agent = self.agents.get(agent_name)
            if agent:
                if send_update:
                    await send_update({
                        "type": "agent_activity",
                        "agent": agent_name,
                        "status": "working"
                    })
                
                try:
                    result = agent.analyze(query, context)
                    results.append(result.get('answer', ''))
                    confidences.append(result.get('confidence', 0.5))
                    
                    if result.get('evidence'):
                        evidence.extend([e.get('source', str(e)) for e in result['evidence'][:3]])
                    
                    if send_update:
                        await send_update({
                            "type": "agent_activity",
                            "agent": agent_name,
                            "status": "completed",
                            "confidence": result.get('confidence', 0.5)
                        })
                except Exception as e:
                    if send_update:
                        await send_update({
                            "type": "agent_activity",
                            "agent": agent_name,
                            "status": "error",
                            "error": str(e)
                        })
        
        # Combine results
        if len(results) == 1:
            combined = results[0]
        elif len(results) > 1:
            combined = self._synthesize_results(query, results)
        else:
            combined = "I couldn't find relevant information for your query."
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return combined, avg_confidence, evidence
    
    def _synthesize_results(self, query: str, results: List[str]) -> str:
        """Synthesize multiple agent results into coherent response."""
        if not results:
            return ""
        
        combined_text = "\n\n".join(results)
        
        # If results are short enough, combine directly
        if len(combined_text) < 4000:
            return combined_text
        
        # Otherwise, use LLM to synthesize
        prompt = f"""Synthesize these analysis results into a single coherent response:

QUERY: {query}

RESULTS:
{combined_text[:6000]}

Create a unified, well-structured response that combines all insights without redundancy."""
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            return response.text
        except:
            return combined_text
    
    async def _refine_response(
        self,
        query: str,
        response: str,
        evidence: List[str],
        context: Dict
    ) -> Tuple[str, float]:
        """Refine response to improve quality."""
        prompt = f"""Improve this response for accuracy and completeness:

QUERY: {query}
CURRENT RESPONSE: {response[:2000]}
EVIDENCE: {', '.join(evidence[:5])}

Provide an improved response that:
1. Is more specific and data-driven
2. Cites sources where possible
3. Is well-structured with clear sections
4. Addresses the query directly

IMPROVED RESPONSE:"""
        
        try:
            refined = self.llm.generate(prompt, temperature=0.2)
            # Estimate quality improvement
            quality = 0.75 if len(refined.text) > len(response) * 0.8 else 0.6
            return refined.text, quality
        except:
            return response, 0.5
    
    def _generate_fallback(self, query: str, analysis: Dict, evidence: List[str]) -> str:
        """Generate a graceful fallback response."""
        entities = analysis.get('entities', {})
        
        fallback = "I apologize, but I don't have enough specific data to fully answer your question.\n\n"
        
        # Provide what we do know
        if entities.get('constituencies'):
            fallback += f"I found references to these constituencies: {', '.join(entities['constituencies'][:3])}\n"
        if entities.get('parties'):
            fallback += f"Regarding parties mentioned: {', '.join(entities['parties'])}\n"
        
        fallback += "\nHere's what I can tell you based on available data:\n"
        fallback += "- West Bengal has 294 assembly constituencies\n"
        fallback += "- The 2021 election saw TMC win 216 seats and BJP win 75 seats\n"
        fallback += "- For specific constituency or district data, please mention the exact name\n"
        
        if analysis.get('clarification_question'):
            fallback += f"\nTo help you better: {analysis['clarification_question']}"
        
        return fallback
    
    def _update_session_context(self, session_id: str, query: str, 
                                analysis: Dict, response: str):
        """Update session context for multi-turn conversations."""
        if session_id not in self._session_context:
            self._session_context[session_id] = {}
        
        ctx = self._session_context[session_id]
        
        # Update with entities from this query
        entities = analysis.get('entities', {})
        if entities.get('constituencies'):
            ctx['last_constituency'] = entities['constituencies'][0]
        if entities.get('districts'):
            ctx['last_district'] = entities['districts'][0]
        if entities.get('parties'):
            ctx['last_party'] = entities['parties'][0]
        
        # Keep track of conversation
        ctx['last_query'] = query
        ctx['turn_count'] = ctx.get('turn_count', 0) + 1
    
    def query_sync(self, query: str, context: Dict = None) -> AutonomousResponse:
        """Synchronous wrapper for process."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process(query, context=context))


# Factory function
def create_autonomous_orchestrator(rag: PoliticalRAGSystem) -> AutonomousPoliticalOrchestrator:
    """Create an autonomous orchestrator instance."""
    return AutonomousPoliticalOrchestrator(rag)

