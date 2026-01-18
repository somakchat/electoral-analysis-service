"""Check what documents are in OpenSearch and if the uploaded file is there."""
import sys
sys.path.insert(0, 'D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend')

from app.services.rag.political_opensearch import PoliticalOpenSearchClient

client = PoliticalOpenSearchClient()

print("=== DOCUMENTS IN OPENSEARCH ===")
print(f"Index: {client.index_name}")

# Get total count
count_result = client.client.count(index=client.index_name)
print(f"Total documents: {count_result['count']}")

# Get unique sources
try:
    result = client.client.search(
        index=client.index_name,
        body={
            'size': 0,
            'aggs': {
                'sources': {
                    'terms': {'field': 'source_file.keyword', 'size': 100}
                }
            }
        }
    )
    
    if 'aggregations' in result:
        print("\n=== SOURCES ===")
        for bucket in result['aggregations']['sources']['buckets']:
            print(f"  {bucket['key']}: {bucket['doc_count']} chunks")
except Exception as e:
    print(f"Aggregation failed: {e}")
    # Try getting a sample
    result = client.client.search(
        index=client.index_name,
        body={'size': 50, '_source': ['source_file', 'text']}
    )
    sources = {}
    for hit in result.get('hits', {}).get('hits', []):
        src = hit.get('_source', {}).get('source_file', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    print("\n=== SOURCES (from sample) ===")
    for s, count in sorted(sources.items()):
        print(f"  {s}: ~{count}+ chunks")

# Search specifically for "AI agent" or the uploaded file
print("\n=== SEARCHING FOR UPLOADED 'AI agent' FILE ===")
search_result = client.client.search(
    index=client.index_name,
    body={
        'size': 10,
        'query': {
            'bool': {
                'should': [
                    {'match': {'source_file': 'AI agent'}},
                    {'match': {'text': 'Karimpur strategy'}},
                ]
            }
        },
        '_source': ['source_file', 'text']
    }
)

hits = search_result.get('hits', {}).get('hits', [])
print(f"Found {len(hits)} matching documents")
for hit in hits[:5]:
    src = hit.get('_source', {}).get('source_file', 'unknown')
    text = hit.get('_source', {}).get('text', '')[:200]
    print(f"\nSource: {src}")
    print(f"Text: {text}...")

