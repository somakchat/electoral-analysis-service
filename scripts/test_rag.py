#!/usr/bin/env python
"""
Test script for Political RAG System.

Run this script to verify the RAG system is working correctly.
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag.political_rag import PoliticalRAGSystem
from app.config import settings


def find_data_folder():
    """Find the political-data folder."""
    candidates = [
        Path(__file__).parent.parent / "political-data",
        Path(settings.data_dir).parent / "political-data",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    """Test the Political RAG System."""
    print("=" * 70)
    print("Political RAG System - Test Suite")
    print("=" * 70)
    
    data_folder = find_data_folder()
    if not data_folder:
        print("ERROR: Could not find political-data folder")
        return
    
    print(f"Data folder: {data_folder}")
    print(f"Index folder: {settings.index_dir}")
    print()
    
    # Initialize RAG
    print("Initializing RAG system...")
    rag = PoliticalRAGSystem(
        data_dir=data_folder,
        index_dir=Path(settings.index_dir)
    )
    
    # Initialize (will build KG and index if needed)
    stats = rag.initialize()
    print(f"Initialization status: {stats['status']}")
    
    if 'kg_stats' in stats:
        print(f"Constituencies: {stats['kg_stats'].get('total_constituencies', 0)}")
        print(f"Facts: {stats['kg_stats'].get('total_facts', 0)}")
    
    print()
    print("-" * 70)
    print("RUNNING TESTS")
    print("-" * 70)
    
    tests = [
        # Constituency lookups
        ("Who won Nandigram in 2021?", "constituency_lookup"),
        ("Tell me about Bankura constituency", "constituency_lookup"),
        
        # Predictions
        ("What are the 2026 predictions for West Bengal?", "prediction"),
        ("Will BJP win more seats in 2026?", "prediction"),
        
        # Vulnerability
        ("Which BJP seats are vulnerable in 2026?", "vulnerability"),
        ("What are the TMC vulnerable constituencies?", "vulnerability"),
        
        # Aggregations
        ("How many seats did TMC win in 2021?", "party_analysis"),
        ("What is the seat distribution in Bankura district?", "district_aggregation"),
        
        # Swing analysis
        ("What is the swing trend in Asansol?", "swing_analysis"),
        
        # Comparisons
        ("Compare Nandigram and Singur", "comparison"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_route in tests:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Expected route: {expected_route}")
        
        try:
            response = rag.query(query, use_llm=False)
            
            print(f"   ✓ Route used: {response.route_used}")
            print(f"   ✓ Confidence: {response.confidence:.0%}")
            print(f"   ✓ Verification: {response.verification_status}")
            print(f"   ✓ Sources: {len(response.sources)}")
            
            # Check if answer is non-empty
            if response.answer and len(response.answer) > 50:
                print(f"   ✓ Answer length: {len(response.answer)} chars")
                passed += 1
            else:
                print(f"   ✗ Answer too short or empty")
                failed += 1
            
            # Preview answer
            preview = response.answer[:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    # Test API endpoints
    print()
    print("API ENDPOINTS AVAILABLE:")
    print("-" * 70)
    print("GET  /rag/constituency/{name}  - Get constituency profile")
    print("GET  /rag/constituencies       - List constituencies (with filters)")
    print("GET  /rag/predictions          - Get 2026 predictions summary")
    print("GET  /rag/swing-analysis       - Get swing analysis")
    print("GET  /rag/district/{name}      - Get district summary")
    print("GET  /rag/search?q=...         - Direct search")
    print("POST /rag/query                - Query with verification")
    print("POST /rag/initialize           - Initialize/rebuild RAG")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = main()
    sys.exit(0 if failed == 0 else 1)

