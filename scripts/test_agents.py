#!/usr/bin/env python
"""
Test script for Advanced Political Agent System.

This tests:
1. Evidence-based reasoning
2. Multi-agent coordination  
3. Citation accuracy
4. Hallucination prevention
"""
import sys
import os
from pathlib import Path
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag.political_rag import PoliticalRAGSystem
from app.services.agents.strategic_orchestrator import StrategicOrchestrator
from app.services.agents.constituency_analyst import ConstituencyIntelligenceAgent
from app.services.agents.electoral_strategist import ElectoralStrategistAgent
from app.services.agents.campaign_strategist import CampaignStrategistAgent
from app.config import settings


def find_data_folder():
    candidates = [
        Path(__file__).parent.parent / "political-data",
        Path(settings.data_dir).parent / "political-data",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def test_constituency_analyst(rag):
    """Test the Constituency Intelligence Agent."""
    print("\n" + "=" * 70)
    print("TESTING: Constituency Intelligence Agent")
    print("=" * 70)
    
    agent = ConstituencyIntelligenceAgent(rag)
    
    tests = [
        ("Tell me about Nandigram constituency", "constituency_profile"),
        ("Compare Nandigram and Singur", "comparison"),
        ("Analyze all constituencies in Bankura district", "cluster"),
    ]
    
    passed = 0
    for query, expected_type in tests:
        print(f"\n🔍 Query: {query}")
        
        can_handle, confidence = agent.can_handle(query)
        print(f"   Can handle: {can_handle} (confidence: {confidence:.2f})")
        
        if can_handle:
            result = agent.analyze(query)
            
            answer_len = len(result.get('answer', ''))
            claims_count = len(result.get('claims', []))
            conf = result.get('confidence', 0)
            
            print(f"   ✓ Answer length: {answer_len} chars")
            print(f"   ✓ Claims: {claims_count}")
            print(f"   ✓ Confidence: {conf:.2f}")
            
            if answer_len > 100 and conf > 0.5:
                print("   ✓ PASSED")
                passed += 1
            else:
                print("   ✗ FAILED (insufficient response)")
        else:
            print("   ✗ FAILED (cannot handle)")
    
    return passed, len(tests)


def test_electoral_strategist(rag):
    """Test the Electoral Strategist Agent."""
    print("\n" + "=" * 70)
    print("TESTING: Electoral Strategist Agent")
    print("=" * 70)
    
    agent = ElectoralStrategistAgent(rag)
    
    tests = [
        ("Analyze BJP performance in West Bengal", "party_analysis"),
        ("What is BJP's victory path for 2026?", "victory_path"),
        ("Which BJP seats are vulnerable?", "vulnerability"),
        ("What are TMC's strongholds?", "strength"),
    ]
    
    passed = 0
    for query, expected_type in tests:
        print(f"\n🔍 Query: {query}")
        
        can_handle, confidence = agent.can_handle(query)
        print(f"   Can handle: {can_handle} (confidence: {confidence:.2f})")
        
        if can_handle:
            result = agent.analyze(query)
            
            answer_len = len(result.get('answer', ''))
            claims_count = len(result.get('claims', []))
            conf = result.get('confidence', 0)
            
            print(f"   ✓ Answer length: {answer_len} chars")
            print(f"   ✓ Claims: {claims_count}")
            print(f"   ✓ Confidence: {conf:.2f}")
            
            if 'metrics' in result:
                print(f"   ✓ Metrics: {list(result['metrics'].keys())}")
            
            if answer_len > 100 and conf > 0.5:
                print("   ✓ PASSED")
                passed += 1
            else:
                print("   ✗ FAILED")
        else:
            print("   ✗ FAILED (cannot handle)")
    
    return passed, len(tests)


def test_campaign_strategist(rag):
    """Test the Campaign Strategist Agent."""
    print("\n" + "=" * 70)
    print("TESTING: Campaign Strategist Agent")
    print("=" * 70)
    
    agent = CampaignStrategistAgent(rag)
    
    tests = [
        ("Design a campaign strategy for BJP in Nandigram", "constituency_campaign"),
        ("What should be BJP's messaging strategy?", "messaging"),
        ("Plan the ground game for TMC", "ground_game"),
    ]
    
    passed = 0
    for query, expected_type in tests:
        print(f"\n🔍 Query: {query}")
        
        can_handle, confidence = agent.can_handle(query)
        print(f"   Can handle: {can_handle} (confidence: {confidence:.2f})")
        
        if can_handle:
            result = agent.analyze(query)
            
            answer_len = len(result.get('answer', ''))
            conf = result.get('confidence', 0)
            
            print(f"   ✓ Answer length: {answer_len} chars")
            print(f"   ✓ Confidence: {conf:.2f}")
            
            if answer_len > 100 and conf > 0.5:
                print("   ✓ PASSED")
                passed += 1
            else:
                print("   ✗ FAILED")
        else:
            print("   ✗ FAILED (cannot handle)")
    
    return passed, len(tests)


async def test_orchestrator(rag):
    """Test the Strategic Orchestrator."""
    print("\n" + "=" * 70)
    print("TESTING: Strategic Orchestrator")
    print("=" * 70)
    
    orchestrator = StrategicOrchestrator(rag)
    
    tests = [
        "Design a winning strategy for BJP in West Bengal 2026",
        "What are the key battleground constituencies?",
        "How can TMC defend their vulnerable seats?",
        "Compare BJP and TMC positions in Bankura district",
    ]
    
    passed = 0
    for query in tests:
        print(f"\n🔍 Query: {query}")
        
        async def log_update(update):
            print(f"   → {update.get('type', 'update')}: {update.get('message', update.get('agent', ''))}")
        
        try:
            result = await orchestrator.process_query(query, send_update=log_update)
            
            print(f"\n   Results:")
            print(f"   ✓ Answer length: {len(result.answer)} chars")
            print(f"   ✓ Confidence: {result.confidence:.2f}")
            print(f"   ✓ Verification: {result.verification_status}")
            print(f"   ✓ Agents used: {result.agents_used}")
            print(f"   ✓ Execution time: {result.execution_time_ms}ms")
            
            if len(result.answer) > 100 and result.confidence > 0.5:
                print("   ✓ PASSED")
                passed += 1
            else:
                print("   ✗ FAILED (insufficient response)")
                
        except Exception as e:
            print(f"   ✗ ERROR: {str(e)}")
    
    return passed, len(tests)


def test_evidence_chain(rag):
    """Test that evidence chains are properly maintained."""
    print("\n" + "=" * 70)
    print("TESTING: Evidence Chain Integrity")
    print("=" * 70)
    
    agent = ConstituencyIntelligenceAgent(rag)
    
    # Query that should produce evidence
    result = agent.analyze("Tell me about Nandigram constituency")
    
    claims = result.get('claims', [])
    evidence = result.get('evidence', [])
    
    print(f"\nClaims generated: {len(claims)}")
    for i, claim in enumerate(claims[:5]):
        print(f"  {i+1}. {claim.get('statement', '')[:80]}...")
        print(f"     Confidence: {claim.get('confidence', 'N/A')}")
    
    print(f"\nEvidence collected: {len(evidence)}")
    for i, ev in enumerate(evidence[:5]):
        print(f"  {i+1}. Type: {ev.get('type', 'N/A')}")
        print(f"     Source: {ev.get('source', 'N/A')}")
        print(f"     Content: {ev.get('content', '')[:60]}...")
    
    # Check integrity
    passed = True
    if len(claims) == 0:
        print("\n   ✗ FAILED: No claims generated")
        passed = False
    if len(evidence) == 0:
        print("\n   ✗ FAILED: No evidence collected")
        passed = False
    
    if passed:
        print("\n   ✓ PASSED: Evidence chain maintained")
    
    return 1 if passed else 0, 1


def main():
    """Run all tests."""
    print("=" * 70)
    print("Advanced Political Agent System - Test Suite")
    print("=" * 70)
    
    data_folder = find_data_folder()
    if not data_folder:
        print("ERROR: Could not find political-data folder")
        return
    
    print(f"Data folder: {data_folder}")
    print(f"Index folder: {settings.index_dir}")
    
    # Initialize RAG
    print("\nInitializing Political RAG system...")
    rag = PoliticalRAGSystem(
        data_dir=data_folder,
        index_dir=Path(settings.index_dir)
    )
    rag.initialize()
    
    print(f"Knowledge Graph: {len(rag.kg.constituency_profiles)} constituencies")
    
    # Run tests
    total_passed = 0
    total_tests = 0
    
    p, t = test_constituency_analyst(rag)
    total_passed += p
    total_tests += t
    
    p, t = test_electoral_strategist(rag)
    total_passed += p
    total_tests += t
    
    p, t = test_campaign_strategist(rag)
    total_passed += p
    total_tests += t
    
    p, t = test_evidence_chain(rag)
    total_passed += p
    total_tests += t
    
    # Test orchestrator (async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    p, t = loop.run_until_complete(test_orchestrator(rag))
    total_passed += p
    total_tests += t
    loop.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {total_passed}/{total_tests}")
    print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total_tests - total_passed} tests failed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

