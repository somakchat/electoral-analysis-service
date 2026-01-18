"""Debug the strategy search in constituency analysis."""
import sys
sys.path.insert(0, 'D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend')
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag.vector_store import get_vector_store

constituency = "KARIMPUR"
party = "BJP"
district_name = "NADIA"

vs = get_vector_store()

# Simulate the search queries from electoral_strategist.py
search_queries = [
    f"{party} strategy {constituency} recommendations",
    f"campaign strategy {constituency} election",
    f"what should {party} do in {constituency}",
    f"{party} strategy {district_name} district",
    f"ground report {district_name}",
    f"{party} strategy West Bengal election 2026",
    f"{party} campaign problems issues Bengal"
]

print("=== SEARCHING FOR STRATEGY CONTENT ===")
strategy_keywords = ['strategy', 'recommend', 'should', 'focus', 'campaign', 
                     'action', 'plan', 'priority', 'target', 'strengthen',
                     'mobilize', 'outreach', 'consolidate', 'voter',
                     'problem', 'issue', 'feedback', 'ground report',
                     'grievance', 'corruption', 'local', 'mla']

all_results = []
for sq in search_queries:
    print(f"\nQuery: {sq}")
    try:
        results = vs.search(sq, top_k=5, search_type="hybrid")
        for r in results:
            source = r.source_file if hasattr(r, 'source_file') else 'Unknown'
            text = r.text if hasattr(r, 'text') else str(r)
            
            # Skip KG entries
            if 'knowledge_graph' in str(source).lower():
                continue
            
            # Check for strategy keywords
            has_keywords = any(w in text.lower() for w in strategy_keywords)
            
            if has_keywords:
                print(f"  ✅ Source: {source}")
                print(f"     Has keywords: {has_keywords}")
                print(f"     Text: {text[:200]}...")
                all_results.append({'source': source, 'text': text})
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n\n=== TOTAL STRATEGY INSIGHTS FOUND: {len(all_results)} ===")
for i, r in enumerate(all_results[:5]):
    print(f"\n{i+1}. {r['source']}")
    print(f"   {r['text'][:300]}...")

