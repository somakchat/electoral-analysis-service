#!/usr/bin/env python
"""Debug script to check OpenSearch retrieval for Sukanta Majumdar CM survey"""

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

# Search for Sukanta Majumdar Chief Minister survey
print()
print('--- Search 1: "Sukanta Majumdar Chief Minister percentage" ---')
query1 = {
    'query': {
        'multi_match': {
            'query': 'Sukanta Majumdar Chief Minister percentage vote',
            'fields': ['text^3'],
            'operator': 'or'
        }
    },
    'size': 5
}
results1 = client.search(index=index_name, body=query1)
total_hits1 = results1["hits"]["total"]["value"]
print(f'Hits: {total_hits1}')
for hit in results1['hits']['hits'][:3]:
    score = hit["_score"]
    source = hit["_source"].get("source_file", "N/A")[:80]
    text = hit["_source"].get("text", "")[:300]
    print(f'  Score: {score:.2f} | Source: {source}')
    print(f'  Text: {text}...')
    print()

# Search for the specific Bengali file name
print()
print('--- Search 2: Bengali CM survey file ---')
query2 = {
    'query': {
        'bool': {
            'should': [
                {'match': {'text': 'সুকান্ত মজুমদার'}},
                {'match': {'text': 'Sukanta Majumdar'}},
                {'match': {'text': 'মুখ্যমন্ত্রী'}},
                {'match': {'text': 'Chief Minister BJP Bengal'}}
            ]
        }
    },
    'size': 10
}
results2 = client.search(index=index_name, body=query2)
total_hits2 = results2["hits"]["total"]["value"]
print(f'Hits: {total_hits2}')
for hit in results2['hits']['hits'][:5]:
    score = hit["_score"]
    source = hit["_source"].get("source_file", "N/A")[:80]
    metadata = hit["_source"].get("metadata", {})
    chunk_id = hit["_source"].get("doc_id", "")[:50]
    text = hit["_source"].get("text", "")[:200]
    print(f'  Score: {score:.2f}')
    print(f'  Source File: {source}')
    print(f'  Chunk ID: {chunk_id}')
    print(f'  Metadata: {metadata}')
    print(f'  Text: {text}...')
    print()

# Count candidates in CM survey
print()
print('--- Aggregation: Candidate mentions ---')
agg_query = {
    'query': {
        'bool': {
            'should': [
                {'match': {'text': 'সুকান্ত'}},
                {'match': {'text': 'Sukanta'}},
                {'match': {'text': 'শুভেন্দু'}},
                {'match': {'text': 'Suvendu'}},
                {'match': {'text': 'দিলীপ'}},
                {'match': {'text': 'Dilip'}}
            ],
            'minimum_should_match': 1
        }
    },
    'size': 0,
    'aggs': {
        'total_hits': {
            'value_count': {
                'field': 'doc_id'
            }
        }
    }
}
try:
    agg_results = client.search(index=index_name, body=agg_query)
    print(f'Documents mentioning candidates: {agg_results["hits"]["total"]["value"]}')
except Exception as e:
    print(f'Aggregation error: {e}')

# Count Sukanta specifically
print()
print('--- Count: Sukanta Majumdar mentions ---')
sukanta_count_query = {
    'query': {
        'bool': {
            'should': [
                {'match': {'text': 'সুকান্ত'}},
                {'match': {'text': 'Sukanta'}},
                {'match': {'text': 'সুকান্ত মজুমদার'}},
                {'match': {'text': 'Sukanta Majumdar'}}
            ],
            'minimum_should_match': 1
        }
    },
    'size': 0
}
sukanta_results = client.search(index=index_name, body=sukanta_count_query)
print(f'Documents mentioning Sukanta Majumdar: {sukanta_results["hits"]["total"]["value"]}')

