"""
Base Agent Classes for Political Strategy Maker.
Implements hierarchical agent architecture with delegation support.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import time
import json
import re
from datetime import datetime

from app.models import Evidence, AgentUpdate, AgentStatus
from app.services.llm import get_llm, BaseLLM


@dataclass
class AgentResult:
    """Result from agent execution."""
    agent: str
    content: Dict[str, Any]
    evidences: List[Evidence] = field(default_factory=list)
    execution_time: float = 0.0
    sub_tasks_completed: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass 
class AgentContext:
    """Context passed between agents in hierarchical execution."""
    query: str
    session_id: str
    constituency: Optional[str] = None
    party: Optional[str] = None
    depth: str = "micro"
    previous_results: List[AgentResult] = field(default_factory=list)
    accumulated_evidence: List[Evidence] = field(default_factory=list)
    entity_context: Dict[str, Any] = field(default_factory=dict)


class SpecialistAgent(ABC):
    """
    Base class for specialist agents.
    
    Each agent has:
    - A specific role/specialization
    - Goal and backstory for context
    - Access to RAG and LLM
    - Ability to use tools
    """
    
    # Agent identity
    name: str = "Specialist Agent"
    role: str = "Specialist"
    goal: str = "Assist in political strategy analysis"
    backstory: str = "You are a political strategy specialist."
    
    # Agent configuration
    verbose: bool = True
    max_iterations: int = 15
    
    def __init__(self, rag=None) -> None:
        self.rag = rag
        self.llm: BaseLLM = get_llm()
        self._tools: List[Any] = []
    
    @property
    def tools(self) -> List[Any]:
        """Override in subclass to provide agent-specific tools."""
        return self._tools
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt including role, goal and backstory."""
        return f"""You are {self.role}.

GOAL: {self.goal}

BACKSTORY: {self.backstory}

IMPORTANT RULES:
1. Use ONLY information from the evidence provided. Do not make up facts.
2. If evidence is insufficient, clearly state what information is missing.
3. Structure your analysis clearly with actionable insights.
4. Be specific to the political context (constituencies, booths, local dynamics).
5. Return your response in valid JSON format.
"""
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON object
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        # Return as wrapped text if JSON extraction fails
        return {"raw_text": text}
    
    async def run(self, query: str, context: Optional[AgentContext] = None) -> AgentResult:
        """Execute the agent's task."""
        start_time = time.time()
        
        # Get evidence from RAG if available
        evidences = []
        if self.rag:
            evidences = self.rag.search(query)
        
        # Run the specialized analysis
        content = await self._analyze(query, evidences, context)
        
        execution_time = time.time() - start_time
        
        return AgentResult(
            agent=self.name,
            content=content,
            evidences=evidences,
            execution_time=execution_time,
            confidence=self._calculate_confidence(content, evidences)
        )
    
    @abstractmethod
    async def _analyze(
        self, 
        query: str, 
        evidences: List[Evidence],
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Perform the agent's specialized analysis."""
        pass
    
    def _calculate_confidence(self, content: Dict[str, Any], evidences: List[Evidence]) -> float:
        """Calculate confidence score based on evidence quality."""
        if not evidences:
            return 0.3
        
        # Average evidence scores
        avg_score = sum(e.score for e in evidences) / len(evidences)
        
        # Adjust based on content completeness
        content_completeness = 0.5
        if content:
            non_empty = sum(1 for v in content.values() if v)
            content_completeness = min(1.0, non_empty / max(1, len(content)))
        
        return min(1.0, (avg_score * 0.6 + content_completeness * 0.4))


class ManagerAgent:
    """
    Strategy Manager Agent - The Chief Political Strategist.
    
    Orchestrates the hierarchical crew by:
    - Breaking down complex queries into tasks
    - Delegating to specialist agents
    - Synthesizing results into comprehensive strategy
    """
    
    name: str = "Strategy Manager"
    role: str = "Chief Political Strategist"
    goal: str = "Orchestrate comprehensive political strategy by delegating to specialist agents"
    backstory: str = """You are a master political strategist who has managed 50+ winning campaigns.
You excel at breaking down complex political challenges into actionable tasks and delegating
to the right specialists. You synthesize micro-level insights into winning strategies."""
    
    def __init__(self, specialists: List[SpecialistAgent]) -> None:
        self.specialists = {s.name: s for s in specialists}
        self.llm = get_llm()
    
    async def run(
        self,
        query: str,
        context: AgentContext,
        send_update: Optional[Callable[[AgentUpdate], Awaitable[None]]] = None
    ) -> AgentResult:
        """
        Execute hierarchical strategy workflow.
        
        Flow:
        1. Analyze query and plan task delegation
        2. Execute Research Team (Intelligence, Opposition, Sentiment)
        3. Execute Analysis Team (Data Scientist, Voter Analyst)
        4. Execute Strategy Team (Ground Strategist, Resource Optimizer)
        5. Compile final report (Strategic Reporter)
        """
        start_time = time.time()
        
        async def notify(agent: str, status: AgentStatus, task: str, details: Dict = None):
            if send_update:
                await send_update(AgentUpdate(
                    agent=agent,
                    status=status,
                    task=task,
                    timestamp=datetime.now(),
                    details=details or {}
                ))
        
        # Notify start
        await notify(self.name, AgentStatus.WORKING, "Analyzing query and planning delegation")
        
        # Phase 1: Research Team (parallel)
        await notify(self.name, AgentStatus.DELEGATING, "Delegating to Research Team")
        
        research_agents = ["Intelligence Agent", "Opposition Research Agent", "Sentiment Decoder Agent"]
        research_tasks = []
        for agent_name in research_agents:
            if agent_name in self.specialists:
                agent = self.specialists[agent_name]
                await notify(agent_name, AgentStatus.WORKING, f"Researching: {query[:50]}...")
                research_tasks.append(self._run_agent_with_notify(agent, query, context, notify))
        
        research_results = await asyncio.gather(*research_tasks)
        
        # Update context with research findings
        for result in research_results:
            context.previous_results.append(result)
            context.accumulated_evidence.extend(result.evidences)
        
        # Phase 2: Analysis Team (parallel)
        await notify(self.name, AgentStatus.DELEGATING, "Delegating to Analysis Team")
        
        analysis_agents = ["Data Scientist Agent", "Voter Analyst Agent"]
        analysis_tasks = []
        for agent_name in analysis_agents:
            if agent_name in self.specialists:
                agent = self.specialists[agent_name]
                await notify(agent_name, AgentStatus.WORKING, f"Analyzing: {query[:50]}...")
                analysis_tasks.append(self._run_agent_with_notify(agent, query, context, notify))
        
        analysis_results = await asyncio.gather(*analysis_tasks)
        
        for result in analysis_results:
            context.previous_results.append(result)
            context.accumulated_evidence.extend(result.evidences)
        
        # Phase 3: Strategy Team (parallel)
        await notify(self.name, AgentStatus.DELEGATING, "Delegating to Strategy Team")
        
        strategy_agents = ["Ground Strategy Agent", "Resource Optimizer Agent"]
        strategy_tasks = []
        for agent_name in strategy_agents:
            if agent_name in self.specialists:
                agent = self.specialists[agent_name]
                await notify(agent_name, AgentStatus.WORKING, f"Strategizing: {query[:50]}...")
                strategy_tasks.append(self._run_agent_with_notify(agent, query, context, notify))
        
        strategy_results = await asyncio.gather(*strategy_tasks)
        
        for result in strategy_results:
            context.previous_results.append(result)
            context.accumulated_evidence.extend(result.evidences)
        
        # Phase 4: Final Report
        if "Strategic Reporter Agent" in self.specialists:
            reporter = self.specialists["Strategic Reporter Agent"]
            await notify(reporter.name, AgentStatus.WORKING, "Compiling final strategy report")
            final_result = await reporter.run(query, context)
            await notify(reporter.name, AgentStatus.DONE, "Report ready")
        else:
            # Fallback: compile results manually
            final_result = self._compile_results(context)
        
        await notify(self.name, AgentStatus.DONE, f"Strategy complete in {time.time()-start_time:.1f}s")
        
        return final_result
    
    async def _run_agent_with_notify(
        self,
        agent: SpecialistAgent,
        query: str,
        context: AgentContext,
        notify: Callable
    ) -> AgentResult:
        """Run an agent and send completion notification."""
        result = await agent.run(query, context)
        await notify(agent.name, AgentStatus.DONE, f"Completed in {result.execution_time:.1f}s")
        return result
    
    def _compile_results(self, context: AgentContext) -> AgentResult:
        """Fallback result compilation if no reporter agent."""
        combined_content = {}
        for result in context.previous_results:
            combined_content[result.agent] = result.content
        
        # Deduplicate evidences
        seen_chunks = set()
        unique_evidences = []
        for e in context.accumulated_evidence:
            if e.chunk_id not in seen_chunks:
                seen_chunks.add(e.chunk_id)
                unique_evidences.append(e)
        
        return AgentResult(
            agent=self.name,
            content={"compiled_analysis": combined_content},
            evidences=unique_evidences[:20],  # Limit to top 20
            confidence=sum(r.confidence for r in context.previous_results) / max(1, len(context.previous_results))
        )
