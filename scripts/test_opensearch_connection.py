"""Test OpenSearch connection and create index."""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.rag.political_opensearch import create_political_opensearch

print("=" * 50)
print("Step 2: OpenSearch Connection & Index Creation")
print("=" * 50)

try:
    client = create_political_opensearch()
    print()
    print("[1] Testing connection...")
    health = client.health_check()
    status = health.get("status", "unknown")
    print(f"    Status: {status}")
    
    if status == "healthy":
        print()
        print("[2] Creating index with political mappings...")
        if client.ensure_index():
            print("    Index created/verified successfully!")
            print()
            print("[3] Checking document count...")
            count = client.get_document_count()
            print(f"    Current documents in index: {count}")
            print()
            print("SUCCESS! Ready for data ingestion.")
        else:
            print("    Index creation failed!")
            sys.exit(1)
    else:
        print(f"    Connection failed: {health}")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

