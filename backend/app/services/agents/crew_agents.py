"""
CrewAI-based Autonomous Political Strategy Agents.

This module implements a robust multi-agent system using CrewAI framework with:
1. Autonomous decision-making and reasoning
2. Tool use for data access and analysis
3. Hierarchical task delegation
4. Memory systems (short-term and long-term)
5. Self-reflection and quality control
6. Collaborative problem solving
"""
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime
import json

from app.config import settings
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.rag.data_schema import ConstituencyProfile


# ============================================================================
# CUSTOM TOOLS FOR AGENTS
# ============================================================================

class ConstituencySearchInput(BaseModel):
    """Input for constituency search tool."""
    query: str = Field(description="Search query for constituency data")
    top_k: int = Field(default=5, description="Number of results to return")


class ConstituencySearchTool(BaseTool):
    """Tool for searching constituency information."""
    name: str = "constituency_search"
    description: str = """
    Search for constituency information in West Bengal.
    Use this when you need to find data about specific constituencies, 
    their electoral history, predictions, or vote shares.
    Returns detailed constituency profiles with 2021 results and 2026 predictions.
    """
    args_schema: Type[BaseModel] = ConstituencySearchInput
    rag: PoliticalRAGSystem = None
    
    def __init__(self, rag: PoliticalRAGSystem, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag', rag)
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Execute constituency search."""
        try:
            results = self.rag.search(query, top_k=top_k)
            if not results:
                return "No constituencies found matching your query."
            
            output = []
            for r in results[:top_k]:
                output.append(f"- {r.get('text', r.get('content', str(r)))[:300]}")
            return "\n".join(output)
        except Exception as e:
            return f"Search error: {str(e)}"


class PartyAnalysisInput(BaseModel):
    """Input for party analysis tool."""
    party: str = Field(description="Party name (BJP, TMC, CPM, INC)")
    analysis_type: str = Field(default="overview", description="Type: overview, seats, vulnerable, strong")


class PartyAnalysisTool(BaseTool):
    """Tool for analyzing party performance."""
    name: str = "party_analysis"
    description: str = """
    Analyze a political party's electoral position in West Bengal.
    Returns seat counts, vulnerable seats, strongholds, and predictions.
    Use for BJP, TMC, CPM, or INC analysis.
    """
    args_schema: Type[BaseModel] = PartyAnalysisInput
    rag: PoliticalRAGSystem = None
    
    def __init__(self, rag: PoliticalRAGSystem, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag', rag)
    
    def _run(self, party: str, analysis_type: str = "overview") -> str:
        """Execute party analysis."""
        try:
            party = party.upper()
            all_seats = list(self.rag.kg.constituency_profiles.values())
            
            seats_2021 = [c for c in all_seats if c.winner_2021.upper() == party]
            seats_2026 = [c for c in all_seats if c.predicted_winner_2026.upper() == party]
            vulnerable = [c for c in all_seats 
                         if c.winner_2021.upper() == party 
                         and c.predicted_winner_2026.upper() != party]
            
            if analysis_type == "overview":
                return f"""
{party} Electoral Analysis:
- 2021 Seats Won: {len(seats_2021)}
- 2026 Predicted: {len(seats_2026)}
- Net Change: {len(seats_2026) - len(seats_2021):+d}
- Vulnerable Seats: {len(vulnerable)}
- Safe Seats: {len([c for c in seats_2026 if c.race_rating.lower() == 'safe'])}
"""
            elif analysis_type == "vulnerable":
                vuln_list = [f"  - {c.ac_name} ({c.district}): trailing by {abs(c.predicted_margin_2026):.1f}%"
                            for c in sorted(vulnerable, key=lambda x: abs(x.predicted_margin_2026), reverse=True)[:10]]
                return f"{party} Vulnerable Seats:\n" + "\n".join(vuln_list)
            else:
                return f"{party} has {len(seats_2021)} seats (2021) and predicted {len(seats_2026)} in 2026"
                
        except Exception as e:
            return f"Analysis error: {str(e)}"


class DistrictAnalysisInput(BaseModel):
    """Input for district analysis tool."""
    district: str = Field(description="District name in West Bengal")


class DistrictAnalysisTool(BaseTool):
    """Tool for analyzing district-level electoral data."""
    name: str = "district_analysis"
    description: str = """
    Analyze electoral data for a specific district in West Bengal.
    Returns constituency-wise breakdown, party performance, and predictions.
    """
    args_schema: Type[BaseModel] = DistrictAnalysisInput
    rag: PoliticalRAGSystem = None
    
    def __init__(self, rag: PoliticalRAGSystem, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag', rag)
    
    def _run(self, district: str) -> str:
        """Execute district analysis."""
        try:
            district = district.upper()
            all_seats = list(self.rag.kg.constituency_profiles.values())
            district_seats = [c for c in all_seats if c.district.upper() == district]
            
            if not district_seats:
                return f"No data found for district: {district}"
            
            # Count by party
            tmc_2021 = sum(1 for c in district_seats if c.winner_2021 in ['TMC', 'AITC'])
            bjp_2021 = sum(1 for c in district_seats if c.winner_2021 == 'BJP')
            tmc_2026 = sum(1 for c in district_seats if c.predicted_winner_2026 in ['TMC', 'AITC'])
            bjp_2026 = sum(1 for c in district_seats if c.predicted_winner_2026 == 'BJP')
            
            return f"""
{district} District Analysis:
- Total Seats: {len(district_seats)}
- 2021: TMC {tmc_2021}, BJP {bjp_2021}, Others {len(district_seats) - tmc_2021 - bjp_2021}
- 2026: TMC {tmc_2026}, BJP {bjp_2026}, Others {len(district_seats) - tmc_2026 - bjp_2026}
- TMC Change: {tmc_2026 - tmc_2021:+d}
- BJP Change: {bjp_2026 - bjp_2021:+d}
"""
        except Exception as e:
            return f"Analysis error: {str(e)}"


class SwingAnalysisTool(BaseTool):
    """Tool for analyzing electoral swing patterns."""
    name: str = "swing_analysis"
    description: str = """
    Analyze electoral swing patterns between 2019 and 2024 Lok Sabha elections.
    Identifies trends and momentum shifts that will impact 2026 Assembly elections.
    """
    rag: PoliticalRAGSystem = None
    
    def __init__(self, rag: PoliticalRAGSystem, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag', rag)
    
    def _run(self, query: str = "") -> str:
        """Execute swing analysis."""
        try:
            all_seats = list(self.rag.kg.constituency_profiles.values())
            
            pro_tmc = sorted([c for c in all_seats if c.pc_swing_2019_2024 > 5],
                           key=lambda x: x.pc_swing_2019_2024, reverse=True)[:5]
            pro_bjp = sorted([c for c in all_seats if c.pc_swing_2019_2024 < -5],
                           key=lambda x: x.pc_swing_2019_2024)[:5]
            
            avg_swing = sum(c.pc_swing_2019_2024 for c in all_seats) / len(all_seats)
            
            output = f"Swing Analysis (2019-2024 Lok Sabha):\n"
            output += f"- Average Swing: {avg_swing:+.2f}% ({'TMC' if avg_swing > 0 else 'BJP'})\n\n"
            output += "Highest Pro-TMC Swing:\n"
            for c in pro_tmc:
                output += f"  - {c.ac_name}: +{c.pc_swing_2019_2024:.1f}%\n"
            output += "\nHighest Pro-BJP Swing:\n"
            for c in pro_bjp:
                output += f"  - {c.ac_name}: {c.pc_swing_2019_2024:.1f}%\n"
            
            return output
        except Exception as e:
            return f"Swing analysis error: {str(e)}"


class PredictionQueryTool(BaseTool):
    """Tool for querying 2026 election predictions."""
    name: str = "prediction_query"
    description: str = """
    Query 2026 West Bengal Assembly election predictions.
    Get seat counts, winning probabilities, and race ratings.
    """
    rag: PoliticalRAGSystem = None
    
    def __init__(self, rag: PoliticalRAGSystem, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag', rag)
    
    def _run(self, query: str = "") -> str:
        """Execute prediction query."""
        try:
            summary = self.rag.get_predictions_summary()
            return f"""
2026 West Bengal Assembly Predictions:
- TMC Predicted: {summary.get('tmc_predicted', 0)} seats
- BJP Predicted: {summary.get('bjp_predicted', 0)} seats
- Others: {summary.get('others_predicted', 0)} seats
- Safe Seats: {summary.get('safe', 0)}
- Toss-up Seats: {summary.get('tossup', 0)}
- Total: 294 seats (148 needed for majority)
"""
        except Exception as e:
            return f"Prediction error: {str(e)}"


# ============================================================================
# CREWAI AGENTS
# ============================================================================

class PoliticalCrewFactory:
    """Factory for creating CrewAI-based political analysis crews."""
    
    def __init__(self, rag: PoliticalRAGSystem):
        self.rag = rag
        self.llm = self._get_llm()
        self.tools = self._create_tools()
    
    def _get_llm(self):
        """Get LLM for agents."""
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.3
        )
    
    def _create_tools(self) -> List[BaseTool]:
        """Create all tools for agents."""
        return [
            ConstituencySearchTool(rag=self.rag),
            PartyAnalysisTool(rag=self.rag),
            DistrictAnalysisTool(rag=self.rag),
            SwingAnalysisTool(rag=self.rag),
            PredictionQueryTool(rag=self.rag)
        ]
    
    def create_research_agent(self) -> Agent:
        """Create the Research Analyst agent."""
        return Agent(
            role='Political Research Analyst',
            goal='Gather comprehensive electoral data and facts from the knowledge base',
            backstory="""You are a meticulous political researcher with deep expertise in 
            West Bengal electoral politics. You specialize in finding accurate data about 
            constituencies, parties, and voting patterns. You NEVER make up statistics - 
            you only report what you find in the data. You always cite your sources.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            memory=True
        )
    
    def create_strategist_agent(self) -> Agent:
        """Create the Electoral Strategist agent."""
        return Agent(
            role='Electoral Strategist',
            goal='Develop winning electoral strategies based on data-driven insights',
            backstory="""You are a seasoned electoral strategist who has worked on 
            multiple successful campaigns in India. You understand the nuances of 
            West Bengal politics, caste dynamics, and regional issues. You make 
            strategic recommendations backed by data and evidence.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            memory=True
        )
    
    def create_analyst_agent(self) -> Agent:
        """Create the Data Analyst agent."""
        return Agent(
            role='Electoral Data Analyst',
            goal='Analyze electoral trends, swings, and voting patterns to identify opportunities',
            backstory="""You are an expert psephologist and data analyst. You excel at 
            identifying electoral trends, calculating swings, and predicting outcomes 
            based on historical data. You present data in clear, actionable formats.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            memory=True
        )
    
    def create_campaign_agent(self) -> Agent:
        """Create the Campaign Manager agent."""
        return Agent(
            role='Campaign Manager',
            goal='Design ground-level campaign strategies and resource allocation plans',
            backstory="""You are an experienced campaign manager who understands 
            booth-level management, voter mobilization, and ground operations. You 
            focus on practical, implementable strategies that translate data into 
            electoral success.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            memory=True
        )
    
    def create_quality_agent(self) -> Agent:
        """Create the Quality Controller agent for fact-checking."""
        return Agent(
            role='Quality Controller & Fact Checker',
            goal='Verify all claims, ensure accuracy, and prevent hallucinations',
            backstory="""You are a rigorous fact-checker and quality controller. Your job 
            is to verify that all statistics and claims made by other agents are accurate 
            and grounded in actual data. You flag any unsupported claims and ensure the 
            final output is reliable and trustworthy.""",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            memory=True
        )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

class PoliticalStrategyCrew:
    """
    Main CrewAI-based orchestration for political strategy queries.
    
    Features:
    - Autonomous multi-agent collaboration
    - Hierarchical task delegation
    - Tool-based data access
    - Quality control and fact-checking
    - Memory for context retention
    """
    
    def __init__(self, rag: PoliticalRAGSystem):
        self.rag = rag
        self.factory = PoliticalCrewFactory(rag)
        
        # Create agents
        self.research_agent = self.factory.create_research_agent()
        self.strategist_agent = self.factory.create_strategist_agent()
        self.analyst_agent = self.factory.create_analyst_agent()
        self.campaign_agent = self.factory.create_campaign_agent()
        self.quality_agent = self.factory.create_quality_agent()
    
    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a political strategy query using the full agent crew.
        
        Args:
            query: User's question or strategy request
            context: Additional context (party, constituency, etc.)
            
        Returns:
            Comprehensive analysis with citations and confidence
        """
        context = context or {}
        
        # Determine query complexity and required agents
        query_type = self._classify_query(query)
        
        # Create appropriate tasks based on query type
        tasks = self._create_tasks(query, query_type, context)
        
        # Create and run crew
        crew = Crew(
            agents=self._select_agents(query_type),
            tasks=tasks,
            process=Process.sequential,  # Can be hierarchical for complex queries
            verbose=True,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        )
        
        try:
            result = crew.kickoff()
            
            return {
                "answer": str(result),
                "query_type": query_type,
                "agents_used": [a.role for a in self._select_agents(query_type)],
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "answer": f"Analysis encountered an error: {str(e)}",
                "error": str(e),
                "confidence": 0.0
            }
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for agent selection."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['strategy', 'win', 'campaign', 'plan']):
            return "strategic"
        elif any(w in query_lower for w in ['predict', 'forecast', '2026', 'will']):
            return "predictive"
        elif any(w in query_lower for w in ['compare', 'versus', 'difference']):
            return "comparative"
        elif any(w in query_lower for w in ['why', 'how', 'reason', 'explain']):
            return "analytical"
        else:
            return "factual"
    
    def _select_agents(self, query_type: str) -> List[Agent]:
        """Select appropriate agents based on query type."""
        agent_map = {
            "factual": [self.research_agent, self.quality_agent],
            "analytical": [self.research_agent, self.analyst_agent, self.quality_agent],
            "predictive": [self.research_agent, self.analyst_agent, self.quality_agent],
            "strategic": [self.research_agent, self.analyst_agent, self.strategist_agent, self.quality_agent],
            "comparative": [self.research_agent, self.analyst_agent, self.quality_agent]
        }
        return agent_map.get(query_type, [self.research_agent, self.quality_agent])
    
    def _create_tasks(self, query: str, query_type: str, context: Dict) -> List[Task]:
        """Create tasks based on query type."""
        tasks = []
        
        # Task 1: Research and data gathering
        research_task = Task(
            description=f"""
            Research and gather all relevant electoral data for the following query:
            
            QUERY: {query}
            CONTEXT: {json.dumps(context)}
            
            Use the available tools to find:
            1. Relevant constituency data
            2. Party performance statistics
            3. Historical election results
            4. 2026 predictions
            
            Be thorough and cite all sources. Do NOT make up any statistics.
            """,
            agent=self.research_agent,
            expected_output="Comprehensive data report with citations"
        )
        tasks.append(research_task)
        
        # Task 2: Analysis (for analytical/predictive/strategic queries)
        if query_type in ["analytical", "predictive", "strategic", "comparative"]:
            analysis_task = Task(
                description=f"""
                Analyze the research findings to answer: {query}
                
                Perform:
                1. Trend analysis
                2. Pattern identification
                3. Risk assessment
                4. Opportunity identification
                
                Base ALL conclusions on the data provided by the Research Analyst.
                """,
                agent=self.analyst_agent,
                expected_output="Detailed analysis with insights"
            )
            tasks.append(analysis_task)
        
        # Task 3: Strategy (for strategic queries)
        if query_type == "strategic":
            strategy_task = Task(
                description=f"""
                Based on the research and analysis, develop a strategic recommendation for: {query}
                
                Include:
                1. Priority seats/areas
                2. Resource allocation suggestions
                3. Key messaging themes
                4. Risk mitigation strategies
                
                Ensure all recommendations are data-driven and practical.
                """,
                agent=self.strategist_agent,
                expected_output="Strategic recommendations with action items"
            )
            tasks.append(strategy_task)
        
        # Task 4: Quality control (always last)
        quality_task = Task(
            description=f"""
            Review the analysis for the query: {query}
            
            Verify:
            1. All statistics are accurate and from verified sources
            2. No hallucinated or made-up data
            3. Conclusions are logically supported
            4. Answer is complete and addresses the query
            
            Flag any issues and provide the final verified response.
            """,
            agent=self.quality_agent,
            expected_output="Verified final response with quality assessment"
        )
        tasks.append(quality_task)
        
        return tasks


# ============================================================================
# QUICK ANALYSIS (For simpler queries)
# ============================================================================

class QuickAnalysisCrew:
    """
    Lightweight crew for simple factual queries.
    Uses a single agent with all tools for fast response.
    """
    
    def __init__(self, rag: PoliticalRAGSystem):
        self.rag = rag
        self.factory = PoliticalCrewFactory(rag)
        self.agent = self.factory.create_research_agent()
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Quick analysis for simple queries."""
        task = Task(
            description=f"""
            Answer this political query accurately and concisely:
            
            {query}
            
            Use the available tools to find relevant data.
            Provide a clear, factual answer with key statistics.
            """,
            agent=self.agent,
            expected_output="Concise factual answer"
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        try:
            result = crew.kickoff()
            return {
                "answer": str(result),
                "agents_used": ["research_agent"],
                "confidence": 0.8
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "confidence": 0.0
            }

