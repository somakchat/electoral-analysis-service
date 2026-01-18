#!/usr/bin/env python
"""Debug script to check OpenSearch retrieval"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import os

region = 'us-east-1'
service = 'aoss'
creds = boto3.Session().get_credentials()
awsauth = AWS4Auth(creds.access_key, creds.secret_key, region, service, session_token=creds.token)

host = os.environ['OPENSEARCH_ENDPOINT'].replace('https://', '')
client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30,
)

index_name = 'political-strategy-maker-v2'

# Count documents
count = client.count(index=index_name)
print(f'Total documents in OpenSearch index [{index_name}]: {count["count"]}')

# Search for Sukanta Majumdar using BM25
print()
print('--- BM25 Search for "Sukanta Majumdar chief minister" ---')
bm25_query = {
    'query': {
        'match': {
            'text': 'Sukanta Majumdar chief minister percentage'
        }
    },
    'size': 5
}
results = client.search(index=index_name, body=bm25_query)
total_hits = results["hits"]["total"]["value"]
print(f'BM25 hits: {total_hits}')
for hit in results['hits']['hits'][:3]:
    score = hit["_score"]
    source = hit["_source"].get("source", "N/A")[:60]
    text = hit["_source"].get("text", "")[:200]
    print(f'  Score: {score:.2f} | Source: {source}')
    print(f'  Text: {text}...')
    print()

# Check for Bengali file
print()
print('--- Search for CM preference survey file ---')
bm25_query2 = {
    'query': {
        'bool': {
            'should': [
                {'match': {'text': 'মুখ্যমন্ত্রী'}},
                {'match': {'text': 'Chief Minister preference'}},
                {'match': {'source': 'Responses'}}
            ]
        }
    },
    'size': 5
}
results2 = client.search(index=index_name, body=bm25_query2)
total_hits2 = results2["hits"]["total"]["value"]
print(f'Hits for CM preference: {total_hits2}')
for hit in results2['hits']['hits'][:3]:
    score = hit["_score"]
    source = hit["_source"].get("source", "N/A")[:80]
    text = hit["_source"].get("text", "")[:200]
    print(f'  Score: {score:.2f} | Source: {source}')
    print(f'  Text: {text}...')
    print()

# Check index mapping
print()
print('--- Index Mapping ---')
mapping = client.indices.get_mapping(index=index_name)
fields = list(mapping[index_name]['mappings']['properties'].keys())
print(f'Fields: {fields}')

