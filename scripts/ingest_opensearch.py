"""
OpenSearch Ingestion Script for Political Strategy Maker.

This script:
1. Creates the 'political-strategy-maker' index in OpenSearch
2. Loads the knowledge graph from local storage
3. Converts constituency profiles to searchable documents
4. Indexes all documents with embeddings

Usage:
    python scripts/ingest_opensearch.py
"""
import os
import sys
import json
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Set encoding for Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

from app.config import settings
from app.services.rag.political_opensearch import (
    PoliticalOpenSearchClient, 
    create_political_opensearch,
    DEFAULT_INDEX_NAME
)


def load_knowledge_graph(kg_path: Path) -> dict:
    """Load knowledge graph from JSON file."""
    if not kg_path.exists():
        print(f"[ERROR] Knowledge graph not found at: {kg_path}")
        return {}
    
    with open(kg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_constituency_documents(kg: dict) -> list:
    """Convert constituency profiles to searchable documents."""
    documents = []
    
    profiles = kg.get("constituency_profiles", {})
    print(f"[INFO] Processing {len(profiles)} constituency profiles...")
    
    for name, profile in profiles.items():
        # Create detailed text for semantic search
        text = f"""
{name} Assembly Constituency Analysis

District: {profile.get('district', 'Unknown')}
Parliamentary Constituency: {profile.get('parent_pc', 'Unknown')}
Category: {profile.get('constituency_type', 'GEN')}

2021 Assembly Election Results:
- Winner: {profile.get('winner_2021', 'Unknown')}
- TMC Vote Share: {profile.get('tmc_vote_share_2021', 0):.2f}%
- BJP Vote Share: {profile.get('bjp_vote_share_2021', 0):.2f}%
- Margin: {profile.get('margin_2021', 0):.2f}%

2026 Predictions:
- Predicted Winner: {profile.get('predicted_winner_2026', 'Unknown')}
- Predicted Margin: {profile.get('predicted_margin_2026', 0):.2f}%
- Race Rating: {profile.get('race_rating', 'Unknown')}

Lok Sabha Trends:
- 2019 TMC: {profile.get('tmc_ls_2019', 0):.2f}%
- 2019 BJP: {profile.get('bjp_ls_2019', 0):.2f}%
- 2024 TMC: {profile.get('tmc_ls_2024', 0):.2f}%
- 2024 BJP: {profile.get('bjp_ls_2024', 0):.2f}%
- Swing 2019-2024: {profile.get('pc_swing_2019_2024', 0):.2f}%
""".strip()
        
        doc = {
            "doc_id": f"constituency_{name}",
            "text": text,
            "constituency": name,
            "district": profile.get("district", ""),
            "party": profile.get("winner_2021", ""),
            "year": "2021",
            "data_type": "constituency_profile",
            "source_file": "knowledge_graph",
            "winner_2021": profile.get("winner_2021", ""),
            "predicted_winner_2026": profile.get("predicted_winner_2026", ""),
            "race_rating": profile.get("race_rating", ""),
            "margin_2021": profile.get("margin_2021", 0.0),
            "predicted_margin_2026": profile.get("predicted_margin_2026", 0.0),
            "tmc_vote_share": profile.get("tmc_vote_share_2021", 0.0),
            "bjp_vote_share": profile.get("bjp_vote_share_2021", 0.0),
            "swing": profile.get("pc_swing_2019_2024", 0.0),
            "metadata": {
                "constituency": name,
                "district": profile.get("district", ""),
                "parent_pc": profile.get("parent_pc", ""),
                "constituency_type": str(profile.get("constituency_type", "")),
                "data_type": "constituency_profile"
            }
        }
        documents.append(doc)
    
    return documents


def prepare_summary_documents(kg: dict) -> list:
    """Create summary documents for overall analysis."""
    documents = []
    profiles = kg.get("constituency_profiles", {})
    
    if not profiles:
        return documents
    
    # Overall summary
    total_seats = len(profiles)
    tmc_2021 = sum(1 for p in profiles.values() if p.get("winner_2021") in ["TMC", "AITC"])
    bjp_2021 = sum(1 for p in profiles.values() if p.get("winner_2021") == "BJP")
    tmc_2026 = sum(1 for p in profiles.values() if p.get("predicted_winner_2026") in ["TMC", "AITC"])
    bjp_2026 = sum(1 for p in profiles.values() if p.get("predicted_winner_2026") == "BJP")
    
    summary_text = f"""
West Bengal Assembly Elections - Overall Summary

Total Seats: {total_seats}

2021 Results:
- TMC (Trinamool Congress): {tmc_2021} seats ({tmc_2021/total_seats*100:.1f}%)
- BJP (Bharatiya Janata Party): {bjp_2021} seats ({bjp_2021/total_seats*100:.1f}%)
- Others: {total_seats - tmc_2021 - bjp_2021} seats

2026 Predictions:
- TMC Predicted: {tmc_2026} seats ({tmc_2026/total_seats*100:.1f}%)
- BJP Predicted: {bjp_2026} seats ({bjp_2026/total_seats*100:.1f}%)
- Change: TMC {tmc_2026 - tmc_2021:+d}, BJP {bjp_2026 - bjp_2021:+d}

Key Statistics:
- Majority Required: 148 seats
- TMC Current Lead: {tmc_2021 - bjp_2021} seats over BJP
""".strip()
    
    documents.append({
        "doc_id": "summary_overall",
        "text": summary_text,
        "constituency": "",
        "district": "",
        "party": "",
        "year": "2026",
        "data_type": "summary",
        "source_file": "knowledge_graph"
    })
    
    # District summaries
    districts = {}
    for name, profile in profiles.items():
        dist = profile.get("district", "Unknown")
        if dist not in districts:
            districts[dist] = {"seats": [], "tmc_2021": 0, "bjp_2021": 0, "tmc_2026": 0, "bjp_2026": 0}
        districts[dist]["seats"].append(name)
        if profile.get("winner_2021") in ["TMC", "AITC"]:
            districts[dist]["tmc_2021"] += 1
        elif profile.get("winner_2021") == "BJP":
            districts[dist]["bjp_2021"] += 1
        if profile.get("predicted_winner_2026") in ["TMC", "AITC"]:
            districts[dist]["tmc_2026"] += 1
        elif profile.get("predicted_winner_2026") == "BJP":
            districts[dist]["bjp_2026"] += 1
    
    for dist, data in districts.items():
        district_text = f"""
{dist} District Electoral Summary

Total Seats: {len(data['seats'])}
Constituencies: {', '.join(data['seats'][:10])}{'...' if len(data['seats']) > 10 else ''}

2021 Results:
- TMC: {data['tmc_2021']} seats
- BJP: {data['bjp_2021']} seats
- Others: {len(data['seats']) - data['tmc_2021'] - data['bjp_2021']} seats

2026 Predictions:
- TMC: {data['tmc_2026']} seats
- BJP: {data['bjp_2026']} seats

Dominant Party: {'TMC' if data['tmc_2021'] > data['bjp_2021'] else 'BJP' if data['bjp_2021'] > data['tmc_2021'] else 'Contested'}
""".strip()
        
        documents.append({
            "doc_id": f"summary_district_{dist.replace(' ', '_').lower()}",
            "text": district_text,
            "constituency": "",
            "district": dist,
            "party": "",
            "year": "2026",
            "data_type": "district_summary",
            "source_file": "knowledge_graph"
        })
    
    return documents


def main():
    print("=" * 60)
    print("Political Strategy Maker - OpenSearch Ingestion")
    print("=" * 60)
    
    # Check configuration
    print(f"\n[CONFIG] OpenSearch Endpoint: {settings.opensearch_endpoint}")
    print(f"[CONFIG] Index Name: {DEFAULT_INDEX_NAME}")
    print(f"[CONFIG] Region: {settings.aws_region}")
    
    if not settings.opensearch_endpoint:
        print("\n[ERROR] OPENSEARCH_ENDPOINT not set in .env file")
        print("Please add: OPENSEARCH_ENDPOINT=https://your-domain.us-east-1.aoss.amazonaws.com")
        return 1
    
    # Initialize OpenSearch client
    print("\n[STEP 1] Initializing OpenSearch client...")
    try:
        client = create_political_opensearch(index_name=DEFAULT_INDEX_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to create client: {e}")
        return 1
    
    # Health check
    print("\n[STEP 2] Checking OpenSearch health...")
    health = client.health_check()
    if health.get("status") != "healthy":
        print(f"[ERROR] OpenSearch unhealthy: {health}")
        return 1
    print(f"[OK] OpenSearch healthy: {health}")
    
    # Create index
    print("\n[STEP 3] Creating/verifying index...")
    if not client.ensure_index():
        print("[ERROR] Failed to create index")
        return 1
    print(f"[OK] Index '{DEFAULT_INDEX_NAME}' ready")
    
    # Load knowledge graph
    kg_path = backend_path / "index" / "knowledge_graph.json"
    print(f"\n[STEP 4] Loading knowledge graph from {kg_path}...")
    kg = load_knowledge_graph(kg_path)
    
    if not kg:
        print("[ERROR] No knowledge graph data found")
        return 1
    print(f"[OK] Loaded {len(kg.get('constituency_profiles', {}))} constituencies")
    
    # Prepare documents
    print("\n[STEP 5] Preparing documents for indexing...")
    documents = []
    
    # Add constituency documents
    constituency_docs = prepare_constituency_documents(kg)
    documents.extend(constituency_docs)
    print(f"  - Constituency documents: {len(constituency_docs)}")
    
    # Add summary documents
    summary_docs = prepare_summary_documents(kg)
    documents.extend(summary_docs)
    print(f"  - Summary documents: {len(summary_docs)}")
    
    print(f"  - Total documents: {len(documents)}")
    
    # Index documents
    print("\n[STEP 6] Indexing documents to OpenSearch...")
    print("  (This may take a few minutes for embedding generation)")
    
    try:
        success, failed = client.index_documents(documents, batch_size=50)
        print(f"\n[RESULT] Indexed {success} documents, {failed} failed")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")
        return 1
    
    # Verify
    print("\n[STEP 7] Verifying index...")
    count = client.get_document_count()
    print(f"[OK] Total documents in index: {count}")
    
    # Test search
    print("\n[STEP 8] Testing search...")
    test_results = client.hybrid_search_sync("Who won JADAVPUR in 2021?", top_k=3)
    print(f"[OK] Test search returned {len(test_results)} results")
    if test_results:
        print(f"  - Top result: {test_results[0].constituency or test_results[0].text[:50]}...")
    
    print("\n" + "=" * 60)
    print("Ingestion completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

