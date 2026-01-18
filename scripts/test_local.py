#!/usr/bin/env python
"""
Local Test Script for Political Strategy Maker.

Tests the core functionality without external API calls.
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag.local_store import LocalHybridIndex, DocumentChunk
from app.services.rag.advanced_rag import AdvancedRAG
from app.services.orchestrator import Orchestrator
from app.models import AgentUpdate

async def test_basic_functionality():
    """Test basic RAG and orchestrator functionality."""
    print("=" * 60)
    print("Political Strategy Maker - Local Test")
    print("=" * 60)
    
    # Test 1: Local Hybrid Index
    print("\n[Test 1] Local Hybrid Index...")
    index = LocalHybridIndex(index_dir="./test_index")
    
    # Add sample chunks
    sample_chunks = [
        DocumentChunk(
            doc_id="test_doc_1",
            chunk_id="test_doc_1_chunk_0",
            source_path="test_data.xlsx",
            text="Nandigram constituency has 234,567 voters across 287 booths. BJP improved by 12% in 2021 election.",
            metadata={"source_type": "excel", "constituency": "Nandigram"}
        ),
        DocumentChunk(
            doc_id="test_doc_1",
            chunk_id="test_doc_1_chunk_1",
            source_path="test_data.xlsx",
            text="Key voter segments in Nandigram: Muslim voters (38%), Hindu OBC (28%), SC (18%), General (16%). Youth voters account for 22%.",
            metadata={"source_type": "excel", "constituency": "Nandigram"}
        ),
        DocumentChunk(
            doc_id="test_doc_2",
            chunk_id="test_doc_2_chunk_0",
            source_path="test_report.docx",
            text="Anti-incumbency is a major factor against TMC. Local grievances include poor road conditions and water supply issues.",
            metadata={"source_type": "docx", "constituency": "Nandigram"}
        ),
    ]
    
    index.add_chunks(sample_chunks)
    print(f"  ✓ Added {len(sample_chunks)} test chunks")
    
    # Test 2: Search
    print("\n[Test 2] Hybrid Search...")
    results = index.search("voter demographics Nandigram", top_k=3)
    print(f"  ✓ Found {len(results)} results for 'voter demographics Nandigram'")
    for chunk, score in results[:2]:
        print(f"    - {chunk.chunk_id}: score={score:.3f}")
        print(f"      {chunk.text[:80]}...")
    
    # Test 3: Advanced RAG
    print("\n[Test 3] Advanced RAG Pipeline...")
    rag = AdvancedRAG(index)
    
    # Test query decomposition
    sub_queries = rag.decompose_query("Design a winning strategy for BJP in Nandigram")
    print(f"  ✓ Query decomposed into {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        print(f"    - {sq}")
    
    # Full RAG search
    evidences = rag.search("What are the voter segments in Nandigram?")
    print(f"  ✓ RAG search returned {len(evidences)} evidences")
    
    # Test 4: Orchestrator (lightweight test)
    print("\n[Test 4] Orchestrator (lightweight test)...")
    orchestrator = Orchestrator(index_dir="./test_index")
    
    # Test quick analysis
    async def log_update(update: AgentUpdate):
        print(f"    [{update.agent}] {update.status}: {update.task[:50]}...")
    
    print("  Running quick analysis...")
    result = await orchestrator.quick_analysis(
        "What are the key voter segments in Nandigram?",
        send_update=log_update
    )
    
    print(f"\n  ✓ Quick analysis complete!")
    print(f"    - Agents used: {result.get('agents_used', [])}")
    print(f"    - Confidence: {result.get('confidence', 0):.2f}")
    print(f"    - Citations: {len(result.get('citations', []))}")
    
    if result.get('answer'):
        print(f"\n  Answer preview:")
        print(f"    {result['answer'][:200]}...")
    
    # Cleanup
    print("\n[Cleanup] Removing test index...")
    index.clear()
    import shutil
    if os.path.exists("./test_index"):
        shutil.rmtree("./test_index")
    print("  ✓ Test index removed")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

def main():
    """Main entry point."""
    try:
        asyncio.run(test_basic_functionality())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

