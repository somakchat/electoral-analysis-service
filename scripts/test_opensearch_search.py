"""Test OpenSearch search functionality."""
import sys
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.rag.political_opensearch import create_political_opensearch

print("=" * 50)
print("Testing OpenSearch Search")
print("=" * 50)

client = create_political_opensearch()

# Wait for index refresh
print("\nWaiting for index refresh...")
time.sleep(3)

print("\n[1] Checking document count...")
count = client.get_document_count()
print(f"    Documents in index: {count}")

print("\n[2] Test search: 'Who won JADAVPUR in 2021?'")
results = client.hybrid_search_sync("Who won JADAVPUR in 2021?", top_k=3)
print(f"    Found {len(results)} results")
for i, r in enumerate(results[:3]):
    name = r.constituency or "Summary"
    print(f"    {i+1}. {name} (score: {r.score:.3f})")
    print(f"       {r.text[:80]}...")

print("\n[3] Test search: 'Which party is stronger in BANKURA district?'")
results = client.hybrid_search_sync("Which party is stronger in BANKURA district?", top_k=3)
print(f"    Found {len(results)} results")
for i, r in enumerate(results[:3]):
    name = r.constituency or r.district or "Summary"
    print(f"    {i+1}. {name} (score: {r.score:.3f})")

print("\n[4] Test search with filter: 'predictions' in KOLKATA")
results = client.hybrid_search_sync(
    "election predictions 2026",
    top_k=3,
    filters={"district": "KOLKATA"}
)
print(f"    Found {len(results)} results")
for i, r in enumerate(results[:3]):
    print(f"    {i+1}. {r.constituency} - {r.district}")

print("\n" + "=" * 50)
print("Search tests completed!")
print("=" * 50)

