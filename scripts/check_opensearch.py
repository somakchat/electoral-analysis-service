"""Check OpenSearch document breakdown."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.rag.unified_vectordb import OpenSearchVectorDB

c = OpenSearchVectorDB()

# Total count
total = c.client.count(index='political-strategy-maker')
print(f"Total documents: {total['count']}")

# Sample documents to see structure
print("\n=== Sample Documents ===")
r = c.client.search(index='political-strategy-maker', body={'size': 5, 'query': {'match_all': {}}})
for i, hit in enumerate(r['hits']['hits'], 1):
    src = hit['_source']
    keys = list(src.keys())
    text = src.get('text', '')[:100] if src.get('text') else 'N/A'
    source_file = src.get('source_file', src.get('metadata', {}).get('source_file', 'N/A'))
    print(f"\n{i}. Keys: {keys}")
    print(f"   Source: {source_file}")
    print(f"   Text: {text}...")

# Count by different query patterns
print("\n=== Document Source Analysis ===")

# Knowledge graph data
kg_query = {'query': {'match': {'text': 'constituency analysis'}}}
kg_count = c.client.count(index='political-strategy-maker', body=kg_query)
print(f"Knowledge Graph related: ~{kg_count['count']}")

# Uploaded docs
doc_query = {'query': {'match': {'text': 'BJP strategy'}}}
doc_count = c.client.count(index='political-strategy-maker', body=doc_query)
print(f"BJP strategy related: ~{doc_count['count']}")

