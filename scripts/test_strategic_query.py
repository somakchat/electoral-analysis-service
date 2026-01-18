"""Test strategic recommendations query."""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.agents.strategic_orchestrator import StrategicOrchestrator
from app.config import settings

print("=" * 70)
print("TESTING STRATEGIC RECOMMENDATIONS QUERY")
print("=" * 70)

# Initialize
data_dir = Path(settings.data_dir)
index_dir = Path(settings.index_dir)
rag = PoliticalRAGSystem(data_dir=data_dir, index_dir=index_dir)
rag.initialize()

orchestrator = StrategicOrchestrator(rag)

# Test the strategic query
query = "What strategic decision or action point BJP should implement immediately for upcoming election to improve seat share?"

print(f"\n[QUERY]: {query}")
print("-" * 70)

response = orchestrator.query_sync(query)

print(f"\n[AGENTS USED]: {response.agents_used}")
print(f"[CONFIDENCE]: {response.confidence:.2f}")
print("-" * 70)
print("\n[ANSWER]:\n")
print(response.answer)
print("\n" + "=" * 70)

