#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Structured Data Ingestion Script.

This script:
1. Processes all survey and election data files
2. Pre-computes statistics (counts, percentages)
3. Creates STATISTICS chunks for OpenSearch
4. Adds survey results to Knowledge Graph
5. Enables accurate aggregation queries

Usage:
    python ingest_structured_data.py
    python ingest_structured_data.py --verify   # Verify without ingesting
"""
import sys
import os
import io
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "backend"))

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import pandas as pd

# Import from backend
from app.services.rag.structured_data_ingester import StructuredDataIngester, SurveyStatistics
from app.services.rag.knowledge_graph import PoliticalKnowledgeGraph

# Configuration
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', 'https://uzh6nlog7cqvgizfij49.us-east-1.aoss.amazonaws.com')
INDEX_NAME = 'political-strategy-maker-v2'
REGION = 'us-east-1'
DATA_DIR = script_dir / "political-data"
INDEX_DIR = script_dir / "index"


def get_opensearch_client():
    """Create OpenSearch client with AWS auth."""
    host = OPENSEARCH_ENDPOINT.replace('https://', '').replace('http://', '')
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, 'aoss')
    
    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )


def get_embeddings(texts: list, batch_size: int = 20) -> list:
    """Generate embeddings using OpenAI."""
    from openai import OpenAI
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        sm = boto3.client('secretsmanager', region_name=REGION)
        secret = sm.get_secret_value(SecretId='political-strategy/openai-api-key')
        api_key = secret['SecretString'].strip()
    
    client = OpenAI(api_key=api_key)
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t[:8000] for t in batch]
        
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
            dimensions=1024
        )
        
        for item in response.data:
            all_embeddings.append(item.embedding)
        
        print(f"    Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
    
    return all_embeddings


def ingest_statistics_to_opensearch(client: OpenSearch, chunks: list) -> int:
    """Ingest statistics chunks to OpenSearch."""
    if not chunks:
        return 0
    
    texts = [c["text"] for c in chunks]
    print(f"  Generating embeddings for {len(texts)} statistics chunks...")
    
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        print(f"  Embedding error: {e}")
        return 0
    
    indexed = 0
    
    for chunk, embedding in zip(chunks, embeddings):
        doc = {
            "doc_id": hashlib.md5(chunk["text"][:100].encode()).hexdigest(),
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "data_type": chunk["data_type"],
            "embedding": embedding
        }
        
        # Add optional metadata
        if "survey_id" in chunk:
            doc["survey_id"] = chunk["survey_id"]
        if "question" in chunk:
            doc["question"] = chunk["question"]
        if "total_responses" in chunk:
            doc["total_responses"] = chunk["total_responses"]
        
        try:
            result = client.index(index=INDEX_NAME, body=doc)
            if result.get("result") in ["created", "updated"]:
                indexed += 1
        except Exception as e:
            print(f"  Index error: {e}")
    
    return indexed


def save_statistics_locally(stats_list: list, output_path: Path):
    """Save statistics to a local JSON file for faster loading."""
    serializable = []
    
    for stats in stats_list:
        serializable.append({
            "survey_id": stats.survey_id,
            "survey_name": stats.survey_name,
            "question_column": stats.question_column,
            "question_text": stats.question_text,
            "total_responses": stats.total_responses,
            "results": {str(k): v for k, v in stats.results.items()},
            "percentages": {str(k): v for k, v in stats.percentages.items()},
            "source_file": stats.source_file
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(serializable)} statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest structured data with statistics")
    parser.add_argument("--verify", action="store_true", help="Verify only, don't ingest")
    parser.add_argument("--skip-opensearch", action="store_true", help="Skip OpenSearch ingestion")
    args = parser.parse_args()
    
    print("=" * 70)
    print("STRUCTURED DATA INGESTION WITH STATISTICS")
    print("=" * 70)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"OpenSearch Endpoint: {OPENSEARCH_ENDPOINT}")
    print(f"Index: {INDEX_NAME}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize Knowledge Graph
    print("\n[1/5] Initializing Knowledge Graph...")
    kg_path = INDEX_DIR / "knowledge_graph.json"
    kg = PoliticalKnowledgeGraph(storage_path=kg_path)
    print(f"  KG has {len(kg.facts)} existing facts")
    
    # Initialize Structured Data Ingester
    print("\n[2/5] Initializing Structured Data Ingester...")
    ingester = StructuredDataIngester(kg=kg)
    
    # Find all survey/structured data files
    print("\n[3/5] Processing structured data files...")
    
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    # Filter out temp files
    xlsx_files = [f for f in xlsx_files if not f.name.startswith("~$")]
    
    print(f"  Found {len(xlsx_files)} xlsx files and {len(csv_files)} csv files")
    
    all_chunks = []
    all_stats = []
    total_facts = 0
    
    # Process xlsx files (surveys)
    for file_path in xlsx_files:
        print(f"\n  📊 {file_path.name}")
        
        result = ingester.ingest_file(file_path)
        
        if result.get("error"):
            print(f"    ❌ Error: {result['error']}")
            continue
        
        print(f"    Type: {result['data_type']}")
        print(f"    Statistics chunks: {len(result['chunks'])}")
        print(f"    Facts added to KG: {result['facts_added']}")
        
        all_chunks.extend(result["chunks"])
        all_stats.extend(result.get("statistics", []))
        total_facts += result["facts_added"]
        
        # Show sample statistics
        for stats in result.get("statistics", [])[:2]:
            if isinstance(stats, SurveyStatistics):
                sorted_results = sorted(stats.results.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"    Top 3 for '{stats.question_text[:40]}...':")
                for opt, cnt in sorted_results:
                    pct = stats.percentages.get(opt, 0)
                    print(f"      - {opt}: {cnt} ({pct:.1f}%)")
    
    # Process csv files (election data)
    for file_path in csv_files:
        # Skip very large files or non-survey CSVs
        if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
            print(f"\n  ⏭️ Skipping large file: {file_path.name}")
            continue
        
        if not any(kw in file_path.name.lower() for kw in ['prediction', 'result', 'vote', 'vulnerable']):
            continue
        
        print(f"\n  📈 {file_path.name}")
        
        result = ingester.ingest_file(file_path)
        
        if result.get("error"):
            print(f"    ❌ Error: {result['error']}")
            continue
        
        print(f"    Type: {result['data_type']}")
        print(f"    Statistics chunks: {len(result['chunks'])}")
        
        all_chunks.extend(result["chunks"])
    
    # Summary before ingestion
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE - READY FOR INGESTION")
    print("=" * 70)
    print(f"\n  Total statistics chunks created: {len(all_chunks)}")
    print(f"  Total facts added to KG: {total_facts}")
    print(f"  Total survey statistics: {len(all_stats)}")
    print(f"  Total surveys tracked: {len(ingester.survey_stats)}")
    
    if args.verify:
        print("\n  ⚠️ VERIFY MODE - Skipping actual ingestion")
        
        # Still save statistics locally
        stats_path = INDEX_DIR / "survey_statistics.json"
        save_statistics_locally(all_stats, stats_path)
        
        # Save KG
        kg.save()
        print(f"  Knowledge Graph saved to {kg_path}")
        
        return
    
    # Save statistics locally (for faster loading)
    print("\n[4/5] Saving statistics locally...")
    stats_path = INDEX_DIR / "survey_statistics.json"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_statistics_locally(all_stats, stats_path)
    
    # Save Knowledge Graph
    kg.save()
    print(f"  Knowledge Graph saved with {len(kg.facts)} total facts")
    
    # Ingest to OpenSearch
    if not args.skip_opensearch:
        print("\n[5/5] Ingesting statistics to OpenSearch...")
        
        try:
            client = get_opensearch_client()
            indexed = ingest_statistics_to_opensearch(client, all_chunks)
            print(f"  ✅ Indexed {indexed} statistics chunks to OpenSearch")
        except Exception as e:
            print(f"  ❌ OpenSearch ingestion error: {e}")
    else:
        print("\n[5/5] Skipping OpenSearch ingestion (--skip-opensearch)")
    
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNow aggregation queries like 'What % voted for X?' will use")
    print("pre-computed statistics instead of trying to count from raw data.")
    
    # Show example queries that will now work correctly
    print("\n📌 Example queries that will now work correctly:")
    print("  - What percentage of people voted for Sukanta Majumdar as CM?")
    print("  - Who is the top choice for BJP CM face?")
    print("  - How many people prefer Suvendu Adhikari?")
    print("  - বিজেপির মুখ হিসাবে জনগণ কাকে চাইছে?")


if __name__ == "__main__":
    main()

