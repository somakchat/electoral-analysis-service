#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest local `index/chunks.pkl` (LocalHybridIndex chunks) into OpenSearch.

Why:
- Local RAG uses `index/chunks.pkl` (FAISS+BM25) which contains ~31k chunks.
- The deployed OpenSearch index in the new AWS account was only populated with ~314
  knowledge-graph summary docs, so retrieval quality is degraded.

This script indexes the same chunk corpus into OpenSearch so AWS retrieval matches local.

Notes:
- OpenSearch Serverless (AOSS) does NOT support explicit document IDs, so this script is
  designed to run once into a fresh index name (recommended), or you can recreate the index.
- Embeddings are generated using the configured embedding provider (OpenAI by default).

Usage (recommended):
  # create a NEW index and ingest (avoids duplicates)
  cd backend
  python ..\\scripts\\ingest_chunks_pkl_to_opensearch.py --index political-strategy-maker-v2 --recreate

  # after ingestion finishes, update Lambda OPENSEARCH_INDEX to the new index.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path


def _get_default_chunks_path() -> Path:
    # repo_root/scripts/ -> repo_root/index/chunks.pkl
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "index" / "chunks.pkl"


def _load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        try:
            return json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_checkpoint(checkpoint_path: Path, data: dict) -> None:
    checkpoint_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest index/chunks.pkl into OpenSearch")
    parser.add_argument("--chunks-path", default=str(_get_default_chunks_path()), help="Path to chunks.pkl")
    parser.add_argument(
        "--faiss-path",
        default=str((Path(__file__).resolve().parent.parent / "index" / "faiss.index")),
        help="Optional path to faiss.index to reuse embeddings (saves OpenAI embedding cost/time)",
    )
    parser.add_argument("--no-faiss", action="store_true", help="Disable reuse of FAISS embeddings even if available")
    parser.add_argument("--index", default=os.getenv("OPENSEARCH_INDEX", "political-strategy-maker-v2"), help="OpenSearch index name")
    parser.add_argument("--endpoint", default=os.getenv("OPENSEARCH_ENDPOINT", ""), help="OpenSearch endpoint (host or https://host)")
    parser.add_argument("--region", default=os.getenv("DEPLOYMENT_REGION", os.getenv("AWS_REGION", "us-east-1")), help="AWS region")
    parser.add_argument("--batch", type=int, default=100, help="Docs per indexing batch")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the index before ingestion")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint offset")
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / ".chunks_ingest_checkpoint.json"))
    parser.add_argument("--start", type=int, default=0, help="Start offset in chunks list")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to ingest (0 = all)")
    args = parser.parse_args()

    chunks_path = Path(args.chunks_path).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if not chunks_path.exists():
        print(f"[ERROR] chunks.pkl not found: {chunks_path}")
        return 1

    # Import backend after args parsing (keeps startup fast)
    repo_root = Path(__file__).resolve().parent.parent
    backend_dir = repo_root / "backend"
    import sys

    sys.path.insert(0, str(backend_dir))

    from app.services.rag.unified_vectordb import OpenSearchVectorDB, Document

    # Configure endpoint/index via env (VectorDBConfig reads env)
    if args.endpoint:
        os.environ["OPENSEARCH_ENDPOINT"] = args.endpoint
    os.environ["OPENSEARCH_INDEX"] = args.index
    os.environ["AWS_REGION"] = args.region

    print("=" * 70)
    print("Ingest chunks.pkl -> OpenSearch")
    print("=" * 70)
    print(f"[CONFIG] chunks.pkl: {chunks_path}")
    print(f"[CONFIG] endpoint:  {os.environ.get('OPENSEARCH_ENDPOINT')}")
    print(f"[CONFIG] index:     {args.index}")
    print(f"[CONFIG] region:    {args.region}")
    print(f"[CONFIG] batch:     {args.batch}")
    print(f"[CONFIG] recreate:  {args.recreate}")
    print(f"[CONFIG] resume:    {args.resume}")
    print(f"[CONFIG] faiss:     {'disabled' if args.no_faiss else args.faiss_path}")

    # Load chunks
    t0 = time.time()
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"[OK] Loaded {len(chunks)} chunks in {time.time() - t0:.1f}s")

    # Determine range
    start = int(args.start)
    end = len(chunks) if args.limit <= 0 else min(len(chunks), start + int(args.limit))

    # Resume from checkpoint if requested
    if args.resume:
        ck = _load_checkpoint(checkpoint_path) or {}
        start = max(start, int(ck.get("next_offset", start)))
        print(f"[RESUME] checkpoint={checkpoint_path} next_offset={start}")

    if start >= end:
        print(f"[DONE] Nothing to ingest (start={start}, end={end})")
        return 0

    # Init OpenSearch client
    osdb = OpenSearchVectorDB(index_name=args.index, endpoint=os.environ.get("OPENSEARCH_ENDPOINT"), region=args.region)

    # Optional: reuse embeddings from FAISS (local index) to match LocalHybridIndex behavior
    faiss_index = None
    faiss_ntotal = 0
    faiss_dim = 0
    if not args.no_faiss:
        try:
            faiss_path = Path(args.faiss_path).resolve()
            if faiss_path.exists():
                import faiss  # type: ignore

                faiss_index = faiss.read_index(str(faiss_path))
                faiss_ntotal = int(getattr(faiss_index, "ntotal", 0))
                faiss_dim = int(getattr(faiss_index, "d", 0))
                print(f"[FAISS] Loaded {faiss_ntotal} vectors (dim={faiss_dim}) from {faiss_path}")
            else:
                print(f"[FAISS] Not found: {faiss_path} (will embed via OpenAI)")
        except Exception as e:
            print(f"[FAISS] Failed to load faiss.index (will embed via OpenAI): {e}")
            faiss_index = None
            faiss_ntotal = 0
            faiss_dim = 0

    # Recreate index (recommended for serverless to avoid duplicates)
    if args.recreate:
        try:
            if osdb.client.indices.exists(index=args.index):
                print(f"[INDEX] Deleting existing index: {args.index}")
                osdb.client.indices.delete(index=args.index)
        except Exception as e:
            print(f"[WARN] Could not delete index '{args.index}': {e}")

    # Ensure index exists (dimension inferred from embedder)
    embedding_dim = getattr(osdb.embedder, "dimension", 3072)
    if faiss_index is not None and faiss_dim and faiss_dim != int(embedding_dim):
        print(f"[FAISS] Dimension mismatch vs embedder (faiss={faiss_dim}, embedder={embedding_dim}). Disabling FAISS reuse.")
        faiss_index = None
        faiss_ntotal = 0
    if not osdb.ensure_index(embedding_dim=embedding_dim):
        print("[ERROR] Failed to create/verify index")
        return 1

    # Ingest
    total = end - start
    print(f"[INGEST] Range: {start}..{end} ({total} chunks)")

    t_ingest = time.time()
    indexed_total = 0

    for offset in range(start, end, args.batch):
        batch_chunks = chunks[offset : min(end, offset + args.batch)]

        # If we have FAISS vectors for this contiguous range, reconstruct them in one shot
        faiss_vecs = None
        if faiss_index is not None and offset < faiss_ntotal:
            n = min(len(batch_chunks), faiss_ntotal - offset)
            try:
                faiss_vecs = faiss_index.reconstruct_n(offset, n)
            except Exception as e:
                print(f"[FAISS] reconstruct_n failed at offset={offset}: {e}")
                faiss_vecs = None

        documents = []
        for c in batch_chunks:
            # DocumentChunk fields: doc_id, chunk_id, source_path, text, metadata
            source_path = getattr(c, "source_path", "") or ""
            source_file = Path(source_path).name if source_path else ""
            chunk_id = getattr(c, "chunk_id", "") or ""
            doc_id = chunk_id or f"{getattr(c, 'doc_id', 'doc')}_{offset}"

            metadata = dict(getattr(c, "metadata", {}) or {})
            metadata.setdefault("source_path", source_path)
            metadata.setdefault("source_file", source_file)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault("source_type", "local_chunks_pkl")

            # Reuse FAISS embedding when available; otherwise let OpenSearchVectorDB embed via OpenAI
            emb = []
            if faiss_vecs is not None:
                rel = len(documents)
                if rel < len(faiss_vecs):
                    try:
                        emb = faiss_vecs[rel].tolist()
                    except Exception:
                        emb = []

            documents.append(
                Document(
                    doc_id=doc_id,
                    text=(getattr(c, "text", "") or "").strip(),
                    embedding=emb,
                    metadata=metadata,
                )
            )

        # Skip empty texts
        documents = [d for d in documents if d.text]
        if not documents:
            continue

        try:
            indexed = osdb.index_documents(documents, batch_size=args.batch)
            indexed_total += int(indexed)
        except Exception as e:
            print(f"[ERROR] Batch failed at offset={offset}: {e}")
            # Save checkpoint so user can resume
            _save_checkpoint(checkpoint_path, {"next_offset": offset, "index": args.index, "time": time.time()})
            return 1

        # Save checkpoint for resume
        _save_checkpoint(checkpoint_path, {"next_offset": min(end, offset + args.batch), "index": args.index, "time": time.time()})

        # Progress
        done = min(end, offset + args.batch) - start
        pct = (done / total) * 100.0
        elapsed = time.time() - t_ingest
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"[PROGRESS] {done}/{total} ({pct:.1f}%) indexed={indexed_total} rate={rate:.1f} chunks/s eta={eta/60:.1f}m")

    print("=" * 70)
    print(f"[DONE] Indexed {indexed_total} documents into '{args.index}' in {(time.time() - t_ingest)/60:.1f} minutes")
    print(f"[NEXT] Update Lambda OPENSEARCH_INDEX to '{args.index}' and redeploy backend.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


