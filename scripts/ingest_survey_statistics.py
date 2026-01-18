#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest Survey Statistics to OpenSearch.

This script creates pre-computed statistics chunks from survey data,
so that aggregation queries (count, percentage, etc.) can be answered accurately.

The problem with standard RAG:
- Survey has 417 rows → 415 chunks
- RAG retrieves only 8-10 chunks
- LLM sees only ~2% of data → CANNOT compute accurate statistics

Solution:
- Pre-compute statistics during ingestion
- Create special "STATISTICS" chunks with aggregated data
- These chunks are retrieved for count/percentage queries
"""
import sys
import os
import io
from pathlib import Path
import hashlib

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

# Configuration
OPENSEARCH_ENDPOINT = os.environ.get('OPENSEARCH_ENDPOINT', 'https://uzh6nlog7cqvgizfij49.us-east-1.aoss.amazonaws.com')
INDEX_NAME = 'political-strategy-maker-v2'
REGION = 'us-east-1'
DATA_DIR = script_dir / "political-data"


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
    
    return all_embeddings


def compute_survey_statistics(file_path: Path) -> list:
    """Compute statistics from a survey/xlsx file."""
    stats_chunks = []
    
    try:
        df = pd.read_excel(file_path)
        total_responses = len(df)
        
        if total_responses == 0:
            return []
        
        print(f"  Processing {file_path.name}: {total_responses} responses")
        
        # Find columns that look like survey questions (have limited unique values)
        for col in df.columns:
            col_str = str(col)
            unique_count = df[col].nunique()
            
            # Skip columns with too many unique values (like timestamps, emails)
            if unique_count > 50 or unique_count < 2:
                continue
            
            # Skip timestamp/email columns
            if any(skip in col_str.lower() for skip in ['timestamp', 'email', 'phone', 'contact']):
                continue
            
            # Compute value counts
            counts = df[col].value_counts()
            
            # Create statistics text
            stats_text = f"""## SURVEY STATISTICS: {file_path.name}
## Question/Column: {col}
## Total Responses: {total_responses}

### Complete Results (All {total_responses} responses):
"""
            
            for value, count in counts.items():
                percentage = (count / total_responses) * 100
                stats_text += f"- {value}: {count} responses ({percentage:.1f}%)\n"
            
            stats_text += f"""
### Summary:
- Total survey responses: {total_responses}
- Most popular choice: {counts.index[0]} with {counts.iloc[0]} votes ({counts.iloc[0]/total_responses*100:.1f}%)
- This is the COMPLETE and ACCURATE count from the full dataset.
"""
            
            stats_chunks.append({
                "text": stats_text,
                "source_file": file_path.name,
                "data_type": "SURVEY_STATISTICS",
                "question": col_str,
                "total_responses": total_responses
            })
            
            print(f"    Created statistics for: {col_str[:50]}...")
        
        # Also create a file-level summary
        summary_text = f"""## SURVEY SUMMARY: {file_path.name}

### Dataset Overview:
- Total Responses: {total_responses}
- Number of Questions: {len(df.columns)}
- Columns: {', '.join(str(c) for c in df.columns)}

This survey contains {total_responses} total responses. 
For accurate vote counts and percentages, refer to the SURVEY_STATISTICS chunks.
"""
        
        stats_chunks.append({
            "text": summary_text,
            "source_file": file_path.name,
            "data_type": "SURVEY_SUMMARY",
            "total_responses": total_responses
        })
        
        return stats_chunks
        
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")
        return []


def ingest_to_opensearch(client: OpenSearch, chunks: list):
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
        
        try:
            result = client.index(index=INDEX_NAME, body=doc)
            if result.get("result") in ["created", "updated"]:
                indexed += 1
        except Exception as e:
            print(f"  Index error: {e}")
    
    return indexed


def main():
    print("=" * 70)
    print("INGEST SURVEY STATISTICS TO OPENSEARCH")
    print("=" * 70)
    print("\nThis creates pre-computed statistics chunks for accurate aggregation queries.")
    
    # Initialize OpenSearch client
    print("\n[1/3] Connecting to OpenSearch...")
    client = get_opensearch_client()
    
    # Find survey files
    print("\n[2/3] Processing survey files...")
    
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))
    print(f"  Found {len(xlsx_files)} xlsx files")
    
    total_indexed = 0
    
    for file_path in xlsx_files:
        if file_path.name.startswith("~$"):
            continue
        
        # Process survey files (those with "Responses" in name or many rows)
        if "Responses" in file_path.name or "Opinion" in file_path.name:
            stats_chunks = compute_survey_statistics(file_path)
            
            if stats_chunks:
                indexed = ingest_to_opensearch(client, stats_chunks)
                total_indexed += indexed
                print(f"  ✅ Indexed {indexed} statistics chunks from {file_path.name}")
    
    print(f"\n[3/3] Total statistics chunks indexed: {total_indexed}")
    
    print("\n" + "=" * 70)
    print("STATISTICS INGESTION COMPLETE")
    print("=" * 70)
    print("\nNow aggregation queries like 'what percentage voted for X' will use")
    print("the pre-computed statistics instead of trying to count from raw data chunks.")


if __name__ == "__main__":
    main()

