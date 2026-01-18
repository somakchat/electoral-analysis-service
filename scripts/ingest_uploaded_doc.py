"""Manually ingest the uploaded AI agent document to OpenSearch."""
import sys
sys.path.insert(0, 'D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend')

from pathlib import Path
from app.services.ingest import ingest_file
from app.services.rag.unified_vectordb import VectorDBConfig, OpenSearchVectorDB, Document
from app.services.rag.embeddings import get_embedding_service
import os

# File path
file_path = Path("D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend/data/uploads/AI agent for WB 2026 Assembly Election1 (1).docx")

print(f"File exists: {file_path.exists()}")
print(f"File size: {file_path.stat().st_size if file_path.exists() else 'N/A'} bytes")

# Step 1: Process the file to get chunks
print("\n=== STEP 1: Process file to get chunks ===")
try:
    chunks, entities = ingest_file(str(file_path))
    print(f"Extracted {len(chunks)} chunks")
    print(f"Extracted {len(entities)} entities")
    
    # Show sample chunk
    if chunks:
        print(f"\nSample chunk text (first 300 chars):")
        print(chunks[0].text[:300] if hasattr(chunks[0], 'text') else str(chunks[0])[:300])
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error processing file: {e}")
    chunks = []

# Step 2: Index to OpenSearch
print("\n=== STEP 2: Index to OpenSearch ===")
if chunks:
    try:
        if not VectorDBConfig.is_configured():
            print("OpenSearch NOT configured!")
        else:
            print("OpenSearch is configured, proceeding...")
            
            os_client = OpenSearchVectorDB()
            embedder = get_embedding_service()
            
            # Convert chunks to proper format
            raw_chunks = []
            for chunk in chunks:
                if hasattr(chunk, 'text'):
                    raw_chunks.append({
                        "text": chunk.text,
                        "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
                    })
                elif isinstance(chunk, dict):
                    raw_chunks.append(chunk)
                else:
                    raw_chunks.append({"text": str(chunk)})
            
            print(f"Processing {len(raw_chunks)} chunks for OpenSearch...")
            
            # Generate embeddings
            texts = [c.get("text", "") for c in raw_chunks if c.get("text")]
            print(f"Generating embeddings for {len(texts)} texts...")
            
            # Batch in smaller groups
            batch_size = 20
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
                embs = embedder.embed(batch)
                all_embeddings.extend(embs)
            
            # Create documents
            documents = []
            doc_id = "AI_agent_WB_2026"
            file_name = "AI agent for WB 2026 Assembly Election1 (1).docx"
            
            for i, (chunk, embedding) in enumerate(zip(raw_chunks, all_embeddings)):
                text = chunk.get("text", "")
                if not text:
                    continue
                    
                doc = Document(
                    doc_id=f"{doc_id}_chunk_{i}",
                    text=text,
                    embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                    metadata={
                        "source_doc_id": doc_id,
                        "source_type": "uploaded_document",
                        "source_file": file_name,
                        "chunk_id": f"chunk_{i}",
                        "data_type": "strategy_document"
                    }
                )
                documents.append(doc)
            
            print(f"Indexing {len(documents)} documents to OpenSearch...")
            indexed = os_client.index_documents(documents)
            print(f"Successfully indexed {indexed} documents!")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error indexing: {e}")
else:
    print("No chunks to index")

# Step 3: Verify the document is searchable
print("\n=== STEP 3: Verify document is searchable ===")
try:
    from app.services.rag.vector_store import get_vector_store
    vs = get_vector_store()
    
    results = vs.search("BJP strategy Karimpur recommendations AI agent", top_k=10, search_type="hybrid")
    print(f"Found {len(results)} results")
    
    for i, r in enumerate(results[:5]):
        text = r.text if hasattr(r, 'text') else str(r)[:200]
        source = r.source_file if hasattr(r, 'source_file') else getattr(r, 'metadata', {}).get('source_file', 'unknown')
        print(f"\n{i+1}. Source: {source}")
        print(f"   Text: {text[:150]}...")
        
except Exception as e:
    import traceback
    traceback.print_exc()

