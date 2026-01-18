#!/usr/bin/env python
"""
Sample Data Ingestion Script.

Creates sample political data for testing the system.
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag.local_store import LocalHybridIndex, DocumentChunk
from app.config import settings


def create_sample_data():
    """Create sample political data chunks for testing."""
    
    sample_data = [
        # Nandigram Constituency Data
        {
            "doc_id": "nandigram_electoral_2021",
            "source": "West Bengal Electoral Data 2021",
            "chunks": [
                """Nandigram Assembly Constituency (AC No. 210)
                Total Voters: 234,567
                Total Booths: 287
                District: Purba Medinipur
                
                2021 Election Results:
                - BJP: Suvendu Adhikari - 110,764 votes (47.3%)
                - TMC: Mamata Banerjee - 107,637 votes (45.9%)
                - Others: 15,963 votes (6.8%)
                Winning Margin: 3,127 votes
                Turnout: 87.5%""",
                
                """Demographic Profile - Nandigram:
                Religion:
                - Hindu: 62%
                - Muslim: 38%
                
                Caste Composition:
                - OBC (Hindu): 28%
                - Scheduled Caste: 18%
                - General: 16%
                - Muslim: 38%
                
                Age Distribution:
                - 18-25 years: 22%
                - 26-40 years: 35%
                - 41-60 years: 30%
                - 60+ years: 13%""",
                
                """Historical Voting Pattern - Nandigram:
                2016: TMC won by 25,000 votes (62% vote share)
                2011: TMC won by 45,000 votes (68% vote share)
                2006: Left Front won (before Nandigram land acquisition movement)
                
                Key Observation: Sharp swing towards BJP in 2021. TMC lost traditional
                stronghold after Suvendu Adhikari's defection. Anti-incumbency 
                sentiment visible in rural areas.""",
                
                """Booth-Level Analysis - Nandigram:
                Strong BJP Booths (>60% vote share): 78 booths
                Strong TMC Booths (>60% vote share): 65 booths
                Swing Booths (<5% margin): 92 booths
                Muslim-majority Booths: 108 booths
                
                Key Insight: BJP performed well in Hindu-majority rural areas.
                TMC retained Muslim-majority booths but with reduced margins.""",
            ]
        },
        
        # Diamond Harbour Constituency
        {
            "doc_id": "diamond_harbour_electoral_2021",
            "source": "West Bengal Electoral Data 2021",
            "chunks": [
                """Diamond Harbour Assembly Constituency (AC No. 133)
                Total Voters: 298,456
                Total Booths: 312
                District: South 24 Parganas
                
                2021 Election Results:
                - TMC: Dipak Halder - 152,345 votes (51.0%)
                - BJP: Dipak Kumar Roy - 135,678 votes (45.5%)
                - Others: 10,433 votes (3.5%)
                Winning Margin: 16,667 votes
                Turnout: 82.3%""",
                
                """Demographic Profile - Diamond Harbour:
                Religion:
                - Hindu: 55%
                - Muslim: 45%
                
                Occupation:
                - Farmers/Agricultural Workers: 40%
                - Fishermen: 15%
                - Small Business: 20%
                - Service/Others: 25%
                
                Rural-Urban Split:
                - Rural: 70%
                - Semi-Urban: 30%""",
            ]
        },
        
        # Regional Analysis
        {
            "doc_id": "west_bengal_political_analysis_2024",
            "source": "Political Intelligence Report 2024",
            "chunks": [
                """Current Political Sentiment - West Bengal (2024):
                
                Key Issues Affecting Voters:
                1. Employment/Unemployment - Top concern for youth
                2. Price Rise - Affecting middle class and poor
                3. Law and Order - Perception of deterioration
                4. Development Work - Mixed response to infrastructure projects
                5. Corruption Allegations - Both parties face allegations
                
                Party Perception:
                - TMC: Strong organization but anti-incumbency building
                - BJP: Growing support but lacks local cadre depth
                - Left: Struggling for relevance, some tactical voting potential""",
                
                """Ground Intelligence - Rural Bengal:
                
                Observations from field surveys:
                1. Farm distress continues, demands for better MSP
                2. Scheme implementation patchy - Lakshmir Bhandar popular but delays
                3. Local grievances often ignored by distant MLAs
                4. Youth migration to cities for jobs
                5. Social media influence increasing in semi-urban areas
                
                Recommendation: Focus on hyperlocal issues, deploy young candidates,
                address livelihood concerns directly.""",
                
                """Opposition Strategy Analysis:
                
                BJP Strengths:
                - National brand, PM Modi factor
                - Strong financial resources
                - Growing social media presence
                - Effective booth management in select areas
                
                BJP Weaknesses:
                - Lack of strong local leadership post-Adhikari
                - Perceived as outsider party
                - Limited understanding of Bengal culture
                - Over-reliance on polarization narrative
                
                TMC Counter-Strategy:
                - Emphasizing Bengali identity/pride
                - Women-centric schemes
                - Coalition building with minority leaders
                - Aggressive ground presence""",
            ]
        },
    ]
    
    return sample_data


def ingest_sample_data():
    """Ingest sample data into the local index."""
    print("=" * 60)
    print("Political Strategy Maker - Sample Data Ingestion")
    print("=" * 60)
    
    # Initialize index
    index = LocalHybridIndex(index_dir=settings.index_dir)
    print(f"\nUsing index directory: {settings.index_dir}")
    
    # Get sample data
    sample_data = create_sample_data()
    
    total_chunks = 0
    for doc in sample_data:
        doc_id = doc["doc_id"]
        source = doc["source"]
        chunks = doc["chunks"]
        
        print(f"\nProcessing: {doc_id}")
        print(f"  Source: {source}")
        
        doc_chunks = []
        for i, text in enumerate(chunks):
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{i}",
                source_path=source,
                text=text.strip(),
                metadata={
                    "source_type": "sample_data",
                    "source_document": source,
                    "chunk_index": i
                }
            )
            doc_chunks.append(chunk)
        
        index.add_chunks(doc_chunks)
        print(f"  ✓ Added {len(doc_chunks)} chunks")
        total_chunks += len(doc_chunks)
    
    print(f"\n" + "=" * 60)
    print(f"Ingestion complete! Total chunks indexed: {total_chunks}")
    print("=" * 60)
    
    # Test search
    print("\nTesting search...")
    results = index.search("voter demographics Nandigram", top_k=3)
    print(f"Search 'voter demographics Nandigram' returned {len(results)} results")
    
    for chunk, score in results[:2]:
        print(f"\n  [{chunk.chunk_id}] Score: {score:.3f}")
        print(f"  {chunk.text[:150]}...")


def main():
    """Main entry point."""
    try:
        ingest_sample_data()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

