#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script to test agent functionality."""
import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.config import settings
from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.agents.electoral_strategist import ElectoralStrategistAgent
from pathlib import Path

def main():
    # Load RAG
    data_dir = Path(__file__).parent.parent / "political-data"
    print(f"Data dir: {data_dir}")
    print(f"Exists: {data_dir.exists()}")
    
    rag = PoliticalRAGSystem(
        data_dir=data_dir, 
        index_dir=Path(settings.index_dir), 
        auto_initialize=True
    )
    print(f"RAG loaded with {len(rag.kg.constituency_profiles)} constituencies")
    
    # Create agent
    agent = ElectoralStrategistAgent(rag)
    print("Agent created successfully")
    
    # Test query
    query = "Analyze BJP's position in Bankura district"
    print(f"\nQuery: {query}")
    print("Expected: Party-district analysis for BJP in Bankura")
    
    try:
        result = agent.analyze(query, {"party": "BJP"})
        print(f"\nResult keys: {list(result.keys())}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"\nAnswer preview:\n{result.get('answer', '')[:500]}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

