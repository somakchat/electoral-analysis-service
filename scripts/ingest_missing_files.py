#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest Missing Files to OpenSearch.

This script ingests the 5 files that were missed in the initial ingestion:
1. wb_assembly_2016_2021.csv
2. WB_Assembly_2026_predictions_by_AC_sorted.csv
3. West_Bengal_AE.csv
4. west_bengal_lok_sabha_2019_all_candidates_votes.csv
5. WhatsApp Chat with West Bengal Election 2026 AI prediction.txt
"""
import sys
import os
import io
from pathlib import Path
import hashlib
import time

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
import numpy as np


# Configuration
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', 'https://uzh6nlog7cqvgizfij49.us-east-1.aoss.amazonaws.com')
INDEX_NAME = 'political-strategy-maker-v2'
REGION = 'us-east-1'
DATA_DIR = script_dir / "political-data"

# Files to ingest
MISSING_FILES = [
    "wb_assembly_2016_2021.csv",
    "WB_Assembly_2026_predictions_by_AC_sorted.csv",
    "West_Bengal_AE.csv",
    "west_bengal_lok_sabha_2019_all_candidates_votes.csv",
    "WhatsApp Chat with West Bengal Election 2026 AI prediction.txt"
]

# Limit chunks per file to avoid excessive API calls
MAX_CHUNKS_PER_FILE = 200  # Set to None for no limit


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
    
    # Get API key from environment or AWS Secrets Manager
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        try:
            import boto3
            sm = boto3.client('secretsmanager', region_name=REGION)
            secret = sm.get_secret_value(SecretId='political-strategy/openai-api-key')
            secret_str = secret['SecretString']
            # Check if it's JSON or plain string
            try:
                import json
                secret_data = json.loads(secret_str)
                api_key = secret_data.get('OPENAI_API_KEY') or secret_data.get('api_key')
            except json.JSONDecodeError:
                # It's a plain string
                api_key = secret_str.strip()
        except Exception as e:
            print(f"Could not get API key from Secrets Manager: {e}")
            raise
    
    client = OpenAI(api_key=api_key)
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Truncate texts to avoid token limits
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


def process_csv(file_path: Path) -> list:
    """Process CSV file into chunks."""
    chunks = []
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                continue
        else:
            print(f"    Could not read CSV with any encoding")
            return []
        
        print(f"    CSV has {len(df)} rows, {len(df.columns)} columns")
        
        # Create summary chunk
        summary = f"Data from {file_path.name}:\n"
        summary += f"Columns: {', '.join(df.columns)}\n"
        summary += f"Total rows: {len(df)}\n"
        
        # Add sample values for key columns
        for col in df.columns[:10]:
            if df[col].dtype == 'object':
                unique = df[col].dropna().unique()[:5]
                if len(unique) > 0:
                    summary += f"{col}: {', '.join(str(v) for v in unique)}\n"
        
        chunks.append({
            "text": summary[:2000],
            "source_file": file_path.name,
            "data_type": "csv_summary"
        })
        
        # Create chunks for groups of rows (5 rows per chunk for better context)
        for i in range(0, len(df), 5):
            batch = df.iloc[i:i+5]
            text = f"Data from {file_path.name} (rows {i+1}-{min(i+5, len(df))}):\n"
            text += batch.to_string(index=False)
            
            chunks.append({
                "text": text[:1500],
                "source_file": file_path.name,
                "data_type": "csv_data",
                "row_range": f"{i+1}-{min(i+5, len(df))}"
            })
        
        return chunks
        
    except Exception as e:
        print(f"    CSV error: {e}")
        return []


def process_txt(file_path: Path) -> list:
    """Process TXT file (WhatsApp chat) into chunks."""
    chunks = []
    
    try:
        # Read with UTF-8
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        print(f"    TXT has {len(content)} characters")
        
        # Split by lines and group messages
        lines = content.split('\n')
        
        # Create chunks of ~1000 characters
        current_chunk = ""
        chunk_count = 0
        
        for line in lines:
            current_chunk += line + "\n"
            
            if len(current_chunk) > 1000:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source_file": file_path.name,
                    "data_type": "whatsapp_chat",
                    "chunk_index": chunk_count
                })
                current_chunk = ""
                chunk_count += 1
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "source_file": file_path.name,
                "data_type": "whatsapp_chat",
                "chunk_index": chunk_count
            })
        
        return chunks
        
    except Exception as e:
        print(f"    TXT error: {e}")
        return []


def ingest_to_opensearch(client: OpenSearch, chunks: list, source_file: str):
    """Ingest chunks to OpenSearch with embeddings."""
    if not chunks:
        return 0
    
    # Generate embeddings
    texts = [c["text"] for c in chunks]
    print(f"    Generating embeddings for {len(texts)} chunks...")
    
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        print(f"    Embedding error: {e}")
        return 0
    
    # Prepare bulk request
    indexed = 0
    batch_size = 50
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        bulk_body = []
        for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
            # Generate a doc_id for reference (but don't use it for indexing in OpenSearch Serverless)
            doc_id = hashlib.md5(f"{source_file}_{i+j}_{chunk['text'][:50]}".encode()).hexdigest()
            
            # Index action - NO _id for OpenSearch Serverless
            bulk_body.append({"index": {"_index": INDEX_NAME}})
            
            # Document
            doc = {
                "doc_id": doc_id,  # Store as a field for reference
                "text": chunk["text"],
                "source_file": source_file,
                "data_type": chunk.get("data_type", "unknown"),
                "embedding": embedding
            }
            
            # Add optional fields
            if "constituency" in chunk:
                doc["constituency"] = chunk["constituency"]
            if "district" in chunk:
                doc["district"] = chunk["district"]
            if "party" in chunk:
                doc["party"] = chunk["party"]
                
            bulk_body.append(doc)
        
        try:
            response = client.bulk(body=bulk_body)
            if not response.get("errors"):
                indexed += len(batch_chunks)
            else:
                # Count successful items and show errors
                success_count = 0
                for item in response.get("items", []):
                    if "index" in item:
                        if item["index"].get("status") in [200, 201]:
                            success_count += 1
                        else:
                            # Show first error
                            if success_count == 0:
                                print(f"    Index error: {item['index'].get('error', {}).get('reason', 'unknown')}")
                indexed += success_count
        except Exception as e:
            print(f"    Bulk index error: {e}")
            import traceback
            traceback.print_exc()
    
    return indexed


def verify_all_files(client: OpenSearch, expected_files: list):
    """Verify all expected files are in OpenSearch."""
    query = {
        "size": 0,
        "aggs": {
            "source_files": {
                "terms": {
                    "field": "source_file",
                    "size": 100
                }
            }
        }
    }
    
    result = client.search(index=INDEX_NAME, body=query)
    buckets = result['aggregations']['source_files']['buckets']
    
    indexed_files = {b['key']: b['doc_count'] for b in buckets}
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"\nTotal unique source files in OpenSearch: {len(indexed_files)}")
    
    all_present = True
    missing = []
    
    for f in expected_files:
        # Check for exact match or partial match
        found = False
        for indexed_file, count in indexed_files.items():
            if f in indexed_file or indexed_file in f:
                print(f"  ✅ {f}: {count} chunks")
                found = True
                break
        
        if not found:
            print(f"  ❌ {f}: NOT FOUND")
            missing.append(f)
            all_present = False
    
    if missing:
        print(f"\n⚠️  Missing files: {missing}")
    else:
        print(f"\n✅ All {len(expected_files)} files verified!")
    
    return all_present, indexed_files


def main():
    print("=" * 70)
    print("INGEST MISSING FILES TO OPENSEARCH")
    print("=" * 70)
    print(f"\nOpenSearch Endpoint: {OPENSEARCH_ENDPOINT}")
    print(f"Index: {INDEX_NAME}")
    print(f"Data Directory: {DATA_DIR}")
    
    # Initialize OpenSearch client
    print("\n[1/4] Connecting to OpenSearch...")
    client = get_opensearch_client()
    
    # Test connection by checking index
    try:
        # For OpenSearch Serverless, we check if we can access the index
        result = client.indices.exists(index=INDEX_NAME)
        if result:
            print(f"  ✅ Connected - Index '{INDEX_NAME}' exists")
        else:
            print(f"  ⚠️ Connected - Index '{INDEX_NAME}' does not exist, will be created")
    except Exception as e:
        print(f"  Connection test: {e}")
        # Continue anyway - might work for indexing
    
    # Process missing files
    print("\n[2/4] Processing missing files...")
    
    total_indexed = 0
    
    for filename in MISSING_FILES:
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            print(f"\n  ⚠️ File not found: {filename}")
            continue
        
        print(f"\n  📄 {filename}")
        
        # Process based on file type
        if filename.endswith('.csv'):
            chunks = process_csv(file_path)
        elif filename.endswith('.txt'):
            chunks = process_txt(file_path)
        else:
            print(f"    Unsupported file type")
            continue
        
        # Apply chunk limit if set
        original_count = len(chunks)
        if MAX_CHUNKS_PER_FILE and len(chunks) > MAX_CHUNKS_PER_FILE:
            chunks = chunks[:MAX_CHUNKS_PER_FILE]
            print(f"    Created {original_count} chunks (limited to {len(chunks)})")
        else:
            print(f"    Created {len(chunks)} chunks")
        
        if chunks:
            indexed = ingest_to_opensearch(client, chunks, filename)
            total_indexed += indexed
            print(f"    ✅ Indexed {indexed} chunks")
    
    print(f"\n[3/4] Total chunks indexed: {total_indexed}")
    
    # Verify all files
    print("\n[4/4] Verifying all files in OpenSearch...")
    
    # All expected files (19 total)
    all_expected_files = [
        "2019-assembly-segment-wise-information-electors.csv",
        "AI agent for WB 2026 Assembly Election1.docx",
        "chat_compilation.docx",
        "lok_sabha_2024_results.csv",
        "Nagarik Samaj Opinion - Nov 18 2025.xlsx",
        "results_2024.csv",
        "WB_2026_BJP_vulnerable_to_TMC_estimated.csv",
        "WB_2026_TMC_vulnerable_to_BJP_estimated.csv",
        "wb_assembly_2016_2021.csv",
        "WB_Assembly_2026_predictions_by_AC_sorted.csv",
        "West Bengal Politics (2019-2025).docx",
        "West_Bengal_AE.csv",
        "west_bengal_lok_sabha_2019_all_candidates_votes.csv",
        "WhatsApp Chat with West Bengal Election 2026 AI prediction.txt",
    ]
    
    verify_all_files(client, all_expected_files)
    
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

