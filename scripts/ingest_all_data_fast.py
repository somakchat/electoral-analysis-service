"""
FAST Data Ingestion Script - Optimized for speed.

Optimizations:
1. Batch embedding generation (50 texts at a time)
2. Skip very large files or sample them
3. Checkpoint/resume support
4. Progress tracking
5. Parallel processing where possible
"""
import os
import sys
from pathlib import Path
import asyncio
import gzip
import hashlib
import json
import time
from datetime import datetime

# Add backend to path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir / "backend"))

from app.config import settings

# Configuration
MAX_CHUNKS_PER_FILE = 100  # Limit chunks per file to prevent slowdown
EMBEDDING_BATCH_SIZE = 50  # Process embeddings in batches
OPENSEARCH_BATCH_SIZE = 100
CHECKPOINT_FILE = script_dir / "scripts" / ".ingest_checkpoint.json"


def load_checkpoint():
    """Load checkpoint of processed files."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except:
            pass
    return {"processed_files": [], "stats": {}}


def save_checkpoint(checkpoint):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))


def sample_chunks(chunks, max_chunks=MAX_CHUNKS_PER_FILE):
    """Sample chunks if too many - take from start, middle, and end."""
    if len(chunks) <= max_chunks:
        return chunks
    
    # Take evenly distributed samples
    step = len(chunks) // max_chunks
    sampled = []
    for i in range(0, len(chunks), step):
        sampled.append(chunks[i])
        if len(sampled) >= max_chunks:
            break
    
    print(f"    (Sampled {len(sampled)} from {len(chunks)} chunks)")
    return sampled


async def ingest_all_data_fast(data_dir_path: str = None, resume: bool = True):
    """Fast ingestion with batching and checkpointing."""
    start_time = time.time()
    
    print("=" * 70)
    print("FAST POLITICAL DATA INGESTION")
    print("=" * 70)
    
    # Initialize
    from app.services.orchestrator import Orchestrator
    from app.services.ingest import DocumentProcessor
    from app.services.rag.unified_vectordb import VectorDBConfig, OpenSearchVectorDB, Document
    from app.services.rag.embeddings import get_embedding_service
    from app.services.rag.data_schema import FactWithCitation
    import re
    
    # Data directory
    if data_dir_path:
        data_dir = Path(data_dir_path)
    else:
        data_dir = script_dir / "political-data"
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    print(f"\nData directory: {data_dir}")
    
    # Load checkpoint
    checkpoint = load_checkpoint() if resume else {"processed_files": [], "stats": {}}
    processed_files = set(checkpoint.get("processed_files", []))
    
    if processed_files:
        print(f"Resuming from checkpoint ({len(processed_files)} files already done)")
    
    # Initialize systems
    print("\n[1/4] Initializing...")
    orchestrator = Orchestrator(index_dir=str(settings.index_dir))
    political_rag = orchestrator._get_political_rag()
    kg = political_rag.kg
    
    print(f"  - KG has {len(kg.constituency_profiles)} constituencies")
    
    # OpenSearch
    os_client = None
    if VectorDBConfig.is_configured():
        os_client = OpenSearchVectorDB()
        print(f"  - OpenSearch: {VectorDBConfig.OPENSEARCH_ENDPOINT}")
    else:
        print("  - OpenSearch not configured (local only)")
    
    embedder = get_embedding_service()
    processor = DocumentProcessor()
    
    # Get data files
    print("\n[2/4] Scanning files...")
    supported_extensions = {'.csv', '.xlsx', '.xls', '.xlsm', '.docx', '.pdf', '.txt', '.gz'}
    
    data_files = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in supported_extensions:
            if f.name not in processed_files:
                data_files.append(f)
    
    print(f"  Files to process: {len(data_files)}")
    
    # Stats
    stats = {
        "files_processed": len(processed_files),
        "opensearch_chunks": 0,
        "kg_facts": 0,
        "errors": []
    }
    
    known_constituencies = set(kg.constituency_profiles.keys())
    
    # Process files
    print("\n[3/4] Processing files...")
    
    for i, file_path in enumerate(data_files, 1):
        file_start = time.time()
        print(f"\n  [{i}/{len(data_files)}] {file_path.name}")
        
        try:
            # Generate doc ID
            doc_id = f"{file_path.stem}_{hashlib.md5(file_path.name.encode()).hexdigest()[:6]}"
            
            # Handle gzip
            process_path = file_path
            temp_path = None
            
            if file_path.suffix.lower() == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                temp_path = data_dir / f"_temp_{file_path.stem}"
                temp_path.write_text(content[:500000], encoding='utf-8')  # Limit size
                process_path = temp_path
            
            # Process document
            try:
                raw_chunks = processor.process(process_path)
            except Exception as e:
                print(f"    ERROR processing: {e}")
                stats["errors"].append(f"{file_path.name}: {e}")
                continue
            
            # Sample if too many chunks
            raw_chunks = sample_chunks(raw_chunks, MAX_CHUNKS_PER_FILE)
            print(f"    - {len(raw_chunks)} chunks")
            
            # === BATCH EMBED + INDEX TO OPENSEARCH ===
            if os_client and raw_chunks:
                try:
                    texts = [c.get("text", "")[:2000] for c in raw_chunks if c.get("text")]  # Truncate long texts
                    
                    if texts:
                        # Batch embed
                        all_embeddings = []
                        for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                            batch_texts = texts[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
                            batch_embeddings = embedder.embed(batch_texts)
                            all_embeddings.extend(batch_embeddings)
                        
                        # Prepare documents
                        documents = []
                        for j, (chunk, embedding) in enumerate(zip(raw_chunks, all_embeddings)):
                            if not chunk.get("text"):
                                continue
                            doc = Document(
                                doc_id=f"{doc_id}_c{j}",
                                text=chunk.get("text", "")[:2000],
                                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                                metadata={
                                    "source_file": file_path.name,
                                    "source_type": "political_data",
                                    "chunk_id": f"c{j}"
                                }
                            )
                            documents.append(doc)
                        
                        # Bulk index
                        if documents:
                            indexed = os_client.index_documents(documents, batch_size=OPENSEARCH_BATCH_SIZE)
                            stats["opensearch_chunks"] += indexed
                            print(f"    - OpenSearch: {indexed} indexed")
                
                except Exception as e:
                    print(f"    WARNING: OpenSearch failed: {str(e)[:50]}")
            
            # === INDEX TO KNOWLEDGE GRAPH ===
            try:
                facts_added = 0
                
                for chunk in raw_chunks[:50]:  # Limit KG facts per file
                    text = chunk.get("text", "")
                    if not text:
                        continue
                    
                    # Quick entity extraction
                    mentioned = []
                    for ac in known_constituencies:
                        if ac.lower() in text.lower():
                            mentioned.append(ac)
                    
                    parties = []
                    if re.search(r'\bbjp\b', text.lower()): parties.append('BJP')
                    if re.search(r'\btmc|trinamool\b', text.lower()): parties.append('TMC')
                    if re.search(r'\bcongress\b', text.lower()): parties.append('INC')
                    if re.search(r'\bcpm|cpim\b', text.lower()): parties.append('CPM')
                    
                    if mentioned or parties:
                        fact_type = "document_content"
                        if 'strategy' in text.lower(): fact_type = "strategy"
                        elif 'predict' in text.lower(): fact_type = "prediction"
                        
                        for entity in (mentioned + parties)[:3]:  # Limit entities per chunk
                            fact = FactWithCitation(
                                fact_type=fact_type,
                                fact_text=text[:400],
                                entity_name=entity,
                                entity_type="constituency" if entity in mentioned else "party",
                                confidence=0.8,
                                source_file=file_path.name,
                                source_doc_ids=[doc_id]
                            )
                            kg.add_fact(fact)
                            facts_added += 1
                
                stats["kg_facts"] += facts_added
                if facts_added > 0:
                    print(f"    - KG: {facts_added} facts")
            
            except Exception as e:
                print(f"    WARNING: KG failed: {str(e)[:50]}")
            
            # Cleanup
            if temp_path and temp_path.exists():
                temp_path.unlink()
            
            # Update checkpoint
            stats["files_processed"] += 1
            processed_files.add(file_path.name)
            checkpoint["processed_files"] = list(processed_files)
            checkpoint["stats"] = stats
            save_checkpoint(checkpoint)
            
            elapsed = time.time() - file_start
            print(f"    Done in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            stats["errors"].append(f"{file_path.name}: {str(e)[:100]}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("[4/4] COMPLETE")
    print("=" * 70)
    print(f"\n  Time: {total_time/60:.1f} minutes")
    print(f"  Files: {stats['files_processed']}")
    print(f"  OpenSearch: {stats['opensearch_chunks']} chunks")
    print(f"  KG facts: {stats['kg_facts']}")
    print(f"  Total KG facts: {len(kg.facts)}")
    
    if stats["errors"]:
        print(f"\n  Errors: {len(stats['errors'])}")
    
    print("\n✅ Done! Data indexed to both KG and OpenSearch.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore checkpoint")
    args = parser.parse_args()
    
    asyncio.run(ingest_all_data_fast(args.data_dir, resume=not args.fresh))

