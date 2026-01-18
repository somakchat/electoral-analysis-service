#!/usr/bin/env python
"""Debug the specific query to understand retrieval issues"""

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

# Test with the FIXED BM25 query (OR operator, 30% match)
print('=' * 70)
print('Testing Fixed BM25 Query (OR operator, 30% minimum_should_match)')
print('=' * 70)

query = 'what percentage of people voted for Sukanta Majumdar as the Chief Minister of bengal from bjp'
print(f'Query: {query}')
print()

query_body = {
    'size': 10,
    'query': {
        'bool': {
            'must': [
                {
                    'multi_match': {
                        'query': query,
                        'fields': ['text^3', 'source_file^2', 'constituency', 'district'],
                        'type': 'best_fields',
                        'operator': 'or',
                        'minimum_should_match': '30%'
                    }
                }
            ]
        }
    },
    '_source': {'includes': ['doc_id','text','source_file']}
}
results = client.search(index=index_name, body=query_body)
total = results['hits']['total']['value']
print(f'Total Hits: {total}')
print()

for i, hit in enumerate(results['hits']['hits'][:7], 1):
    score = hit['_score']
    source = hit['_source'].get('source_file', 'N/A')[:80]
    text = hit['_source'].get('text', '')[:200]
    print(f'{i}. Score: {score:.2f}')
    print(f'   Source: {source}')
    print(f'   Text: {text}...')
    print()

# Now test direct search for Sukanta
print('=' * 70)
print('Direct Search for "Sukanta Majumdar Chief Minister"')
print('=' * 70)

query2 = 'Sukanta Majumdar Chief Minister BJP'
query_body2 = {
    'size': 10,
    'query': {
        'bool': {
            'must': [
                {
                    'multi_match': {
                        'query': query2,
                        'fields': ['text^3', 'source_file^2'],
                        'type': 'best_fields',
                        'operator': 'or',
                        'minimum_should_match': '50%'
                    }
                }
            ]
        }
    },
    '_source': {'includes': ['doc_id','text','source_file']}
}
results2 = client.search(index=index_name, body=query_body2)
total2 = results2['hits']['total']['value']
print(f'Total Hits: {total2}')
print()

for i, hit in enumerate(results2['hits']['hits'][:5], 1):
    score = hit['_score']
    source = hit['_source'].get('source_file', 'N/A')[:80]
    text = hit['_source'].get('text', '')[:200]
    print(f'{i}. Score: {score:.2f}')
    print(f'   Source: {source}')
    print(f'   Text: {text}...')
    print()

# Check if CM survey file exists
print('=' * 70)
print('Checking for CM Survey File')
print('=' * 70)

query3 = {
    'size': 5,
    'query': {
        'bool': {
            'should': [
                {'match': {'source_file': 'মুখ্যমন্ত্রী'}},
                {'match': {'text': 'মুখ্যমন্ত্রী'}},
                {'match': {'text': 'সুকান্ত মজুমদার'}}
            ],
            'minimum_should_match': 1
        }
    },
    '_source': {'includes': ['doc_id','text','source_file']}
}
results3 = client.search(index=index_name, body=query3)
total3 = results3['hits']['total']['value']
print(f'Total Hits for Bengali CM terms: {total3}')
print()

for i, hit in enumerate(results3['hits']['hits'][:5], 1):
    score = hit['_score']
    source = hit['_source'].get('source_file', 'N/A')[:80]
    text = hit['_source'].get('text', '')[:200]
    print(f'{i}. Score: {score:.2f}')
    print(f'   Source: {source}')
    print(f'   Text: {text}...')
    print()

