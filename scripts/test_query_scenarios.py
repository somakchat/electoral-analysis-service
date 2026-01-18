"""
Comprehensive test for Political Strategy Maker queries.
Tests multiple scenarios to ensure the system handles all query types.
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.rag.political_opensearch import create_political_opensearch

print("=" * 70)
print("POLITICAL STRATEGY MAKER - QUERY SCENARIO TESTING")
print("=" * 70)

client = create_political_opensearch()

# Test queries that should work
test_queries = [
    # General analytical queries
    ("What are the swing seats in 2026?", None),
    ("What are the swing seats in West Bengal?", None),
    ("Which constituencies are competitive?", None),
    ("Where can BJP win in 2026?", None),
    ("Which party is stronger in BANKURA district?", None),
    
    # Specific constituency queries
    ("Who won JADAVPUR in 2021?", None),
    ("Tell me about SALTORA constituency", None),
    
    # Party-specific queries
    ("Where did TMC win in 2021?", None),
    ("What is BJP's performance in West Bengal?", None),
    
    # District queries
    ("Analyze PURBA MEDINIPUR district", None),
    
    # Prediction queries
    ("What are the 2026 election predictions?", None),
    ("Which seats will change hands in 2026?", None),
]

print(f"\nTesting {len(test_queries)} query scenarios...\n")

for i, (query, filters) in enumerate(test_queries, 1):
    print(f"[{i:2d}] Query: \"{query}\"")
    results = client.hybrid_search_sync(query, top_k=5, filters=filters)
    
    if results:
        print(f"     Found {len(results)} results")
        for j, r in enumerate(results[:3]):
            name = r.constituency or r.district or "Summary"
            data_type = r.data_type or "unknown"
            print(f"       - {name} ({data_type}, score: {r.score:.2f})")
    else:
        print("     [WARNING] No results found!")
    print()

print("=" * 70)
print("CHECKING DATA TYPES IN INDEX")
print("=" * 70)

# Check what data types exist
results = client.hybrid_search_sync("election results", top_k=50)
data_types = {}
districts = set()
parties = set()

for r in results:
    dt = r.data_type or "unknown"
    data_types[dt] = data_types.get(dt, 0) + 1
    if r.district:
        districts.add(r.district)
    if r.party:
        parties.add(r.party)

print(f"\nData Types: {data_types}")
print(f"\nDistricts Found: {len(districts)}")
print(f"  Examples: {list(districts)[:5]}")
print(f"\nParties Found: {parties}")

print("\n" + "=" * 70)
print("TESTING COMPLETED")
print("=" * 70)

