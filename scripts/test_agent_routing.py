"""
Test agent routing and query handling for Political Strategy Maker.
This script tests various query scenarios to ensure proper routing and response.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.agents.strategic_orchestrator import StrategicOrchestrator
from app.config import settings

print("=" * 70)
print("POLITICAL STRATEGY MAKER - AGENT ROUTING TEST")
print("=" * 70)

# Initialize RAG and Orchestrator
print("\n[1] Initializing system...")
data_dir = Path(settings.data_dir)
index_dir = Path(settings.index_dir)

rag = PoliticalRAGSystem(data_dir=data_dir, index_dir=index_dir)
# Initialize loads the knowledge graph
rag.initialize()
print(f"    Knowledge Graph: {len(rag.kg.constituency_profiles)} constituencies")

orchestrator = StrategicOrchestrator(rag)
print("    Orchestrator initialized")

# Test queries
test_queries = [
    # General analytical queries (should go to electoral_strategist)
    "What are the swing seats in 2026?",
    "What are the swing seats in West Bengal?",
    "Which constituencies are competitive?",
    "How many seats does TMC hold?",
    "What is the overall prediction for 2026?",
    
    # Specific constituency queries (should go to constituency_analyst)
    "Tell me about JADAVPUR constituency",
    "Who won SALTORA in 2021?",
    
    # Party queries (electoral_strategist)
    "How is BJP performing in West Bengal?",
    "What is TMC's stronghold?",
    
    # District queries (electoral_strategist)
    "Which party is stronger in BANKURA district?",
]

print(f"\n[2] Testing {len(test_queries)} queries...")
print("-" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}]: {query}")
    
    try:
        # Get query understanding
        understanding = orchestrator.get_query_understanding_summary(query)
        intent = understanding.get('primary_intent', 'unknown')
        agents = understanding.get('suggested_agents', [])
        entities = [f"{e['normalized']}({e['type']})" for e in understanding.get('entities', [])]
        
        print(f"  Intent: {intent}")
        print(f"  Entities: {entities if entities else 'None detected'}")
        print(f"  Routed to: {agents}")
        
        # Execute query
        response = orchestrator.query_sync(query)
        
        # Check response
        answer = response.answer[:200] if response.answer else "No answer"
        confidence = response.confidence
        agents_used = response.agents_used
        
        print(f"  Agents used: {agents_used}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Answer preview: {answer}...")
        
        # Check for error responses
        if "couldn't identify" in answer.lower() or "no data" in answer.lower():
            print(f"  [WARNING] Possible routing issue!")
        else:
            print(f"  [OK] Response generated")
            
    except Exception as e:
        print(f"  [ERROR]: {str(e)}")

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)

