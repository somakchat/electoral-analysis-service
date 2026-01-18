#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Political Data Ingestion Script - Advanced RAG with Knowledge Graph.

This script uses the new structured ingestion pipeline that:
1. Builds a Knowledge Graph from electoral data
2. Creates verified facts with citations
3. Generates searchable chunks with metadata
4. Supports zero-hallucination retrieval
"""
import sys
import os
import io
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for Unicode (Bengali filenames)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Now import after path is set
from app.config import settings
from app.services.rag.political_rag import PoliticalRAGSystem


# Data folder path - check multiple locations
def find_data_folder():
    """Find the political-data folder."""
    candidates = [
        Path(__file__).parent.parent / "political-data",
        Path(__file__).parent.parent.parent / "political-data",
        Path(settings.data_dir).parent / "political-data",
        Path.cwd() / "political-data",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return candidates[0]  # Return first option even if doesn't exist


def main():
    """Main ingestion function using Advanced RAG."""
    print("=" * 70)
    print("Political Strategy Maker - Advanced RAG Ingestion")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_folder = find_data_folder()
    index_dir = Path(settings.index_dir)
    
    print(f"Data folder: {data_folder}")
    print(f"Index folder: {index_dir}")
    print()
    
    if not data_folder.exists():
        print(f"ERROR: Data folder not found at {data_folder}")
        print("Please ensure political-data folder exists with your CSV/XLSX files.")
        return
    
    # List files found
    files = list(data_folder.glob("*.csv")) + list(data_folder.glob("*.xlsx")) + list(data_folder.glob("*.csv.gz"))
    print(f"Found {len(files)} data files:")
    for f in files:
        size = f.stat().st_size / 1024  # KB
        print(f"  - {f.name} ({size:.1f} KB)")
    print()
    
    # Initialize Political RAG System
    print("Initializing Political RAG System...")
    print("-" * 50)
    
    rag = PoliticalRAGSystem(
        data_dir=data_folder,
        index_dir=index_dir,
        auto_initialize=False
    )
    
    # Force rebuild to re-ingest all data
    stats = rag.initialize(force_rebuild=True)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    
    if "ingest_stats" in stats:
        ingest = stats["ingest_stats"]
        print(f"Predictions loaded: {ingest.get('predictions_loaded', 0)}")
        print(f"Vulnerabilities updated: {ingest.get('vulnerabilities_updated', 0)}")
        print(f"Election results loaded: {ingest.get('election_results_loaded', 0)}")
        print(f"Surveys loaded: {ingest.get('surveys_loaded', 0)}")
        print(f"Chunks indexed: {ingest.get('chunks_indexed', 0)}")
        
        if ingest.get("errors"):
            print(f"\nWarnings/Errors ({len(ingest['errors'])}):")
            for err in ingest["errors"][:10]:
                print(f"  - {err}")
    
    if "kg_stats" in stats:
        kg = stats["kg_stats"]
        print(f"\nKnowledge Graph Statistics:")
        print(f"  - Total constituencies: {kg.get('total_constituencies', 0)}")
        print(f"  - Total entities: {kg.get('total_entities', 0)}")
        print(f"  - Total facts: {kg.get('total_facts', 0)}")
        print(f"  - Districts: {kg.get('districts', 0)}")
        print(f"  - Parliamentary constituencies: {kg.get('pcs', 0)}")
        
        if "seats_2021" in kg:
            print(f"\n  2021 Seat Distribution:")
            for party, count in sorted(kg["seats_2021"].items(), key=lambda x: -x[1]):
                print(f"    - {party}: {count}")
        
        if "seats_2026_predicted" in kg:
            print(f"\n  2026 Predicted Distribution:")
            for party, count in sorted(kg["seats_2026_predicted"].items(), key=lambda x: -x[1]):
                print(f"    - {party}: {count}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test queries
    print("\n" + "-" * 70)
    print("Testing RAG System...")
    print("-" * 70)
    
    test_queries = [
        "Who won Nandigram in 2021?",
        "What are the BJP vulnerable seats in 2026?",
        "How many seats will TMC win in 2026?",
        "Tell me about Bankura district",
        "What is the swing analysis for Asansol PC?",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            response = rag.query(query, use_llm=False)  # Skip LLM for testing
            print(f"   Route: {response.route_used}")
            print(f"   Confidence: {response.confidence:.0%}")
            print(f"   Verification: {response.verification_status}")
            
            # Show first 200 chars of answer
            answer_preview = response.answer[:200].replace('\n', ' ')
            print(f"   Answer: {answer_preview}...")
            
            if response.sources:
                print(f"   Sources: {', '.join(response.sources[:3])}")
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 70)
    print("RAG System Ready!")
    print("=" * 70)
    print(f"\nYou can now start the backend server with:")
    print(f"  cd backend && python -m uvicorn app.main:app --reload")
    print(f"\nOr use the RAG system directly in Python:")
    print(f"  from app.services.rag import create_political_rag")
    print(f"  rag = create_political_rag()")
    print(f"  response = rag.query('Your question here')")


if __name__ == "__main__":
    main()

