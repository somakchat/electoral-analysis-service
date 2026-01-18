"""
Political Strategy Maker - FastAPI Main Application.

Provides REST and WebSocket APIs for:
- Document ingestion
- Real-time strategy chat with agent streaming
- Memory retrieval
- Zero-hallucination political data queries
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import json
import uuid
from datetime import datetime
import structlog

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.models import (
    IngestResponse, ChatRequest, FinalResponse, AgentUpdate,
    Evidence, WorkflowType, DepthLevel
)
from app.services.orchestrator import Orchestrator
from app.services.ingest import ingest_file
from app.services.memory import get_memory_store
from app.services.rag.political_rag import PoliticalRAGSystem

# Configure logging - simple config compatible with PrintLogger
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),  # 0 = DEBUG level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Initialize FastAPI
app = FastAPI(
    title="Political Strategy Maker",
    description="Advanced Multi-Agent Political Strategy System",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Log configuration on startup."""
    logger.info("=" * 60)
    logger.info("Political Strategy Maker - Starting Up")
    logger.info("=" * 60)
    logger.info("configuration", 
                app_env=settings.app_env,
                llm_provider=settings.llm_provider,
                openai_model=settings.openai_model,
                openai_api_key="SET" if settings.openai_api_key else "NOT SET",
                gemini_api_key="SET" if settings.gemini_api_key else "NOT SET",
                data_dir=settings.data_dir,
                index_dir=settings.index_dir)
    logger.info("=" * 60)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data directory
data_dir = Path(settings.data_dir)
data_dir.mkdir(parents=True, exist_ok=True)
(data_dir / "uploads").mkdir(exist_ok=True)

# Find political data folder
def find_political_data_folder():
    candidates = [
        Path(settings.data_dir).parent / "political-data",
        Path(__file__).parent.parent.parent / "political-data",
        Path.cwd() / "political-data",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

# Initialize services
orchestrator = Orchestrator(index_dir=settings.index_dir)
memory = get_memory_store()

# Initialize Political RAG System (lazy loaded)
_political_rag: Optional[PoliticalRAGSystem] = None

def get_political_rag() -> PoliticalRAGSystem:
    """Get or initialize the Political RAG system."""
    global _political_rag
    if _political_rag is None:
        political_data = find_political_data_folder()
        _political_rag = PoliticalRAGSystem(
            data_dir=political_data,
            index_dir=Path(settings.index_dir),
            auto_initialize=True
        )
    return _political_rag


# ============= Health Check =============

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "version": "1.0.0"
    }


# ============= Document Ingestion =============

async def _index_to_opensearch(chunks: List[dict], doc_id: str, file_name: str = ""):
    """
    Index chunks to OpenSearch using constituency-aware chunking.
    
    Uses the Evidence-Only RAG chunker for:
    - Constituency detection and tagging
    - District mapping
    - Topic tag extraction
    - Proper metadata for filtered retrieval
    """
    try:
        from app.services.rag.unified_vectordb import VectorDBConfig, OpenSearchVectorDB, Document
        from app.services.rag.embeddings import get_embedding_service
        from app.services.rag.evidence_rag import ConstituencyAwareChunker
        
        if not VectorDBConfig.is_configured():
            logger.info("opensearch_skip", reason="OpenSearch not configured")
            return 0
        
        # Initialize components
        os_client = OpenSearchVectorDB()
        embedder = get_embedding_service()
        chunker = ConstituencyAwareChunker()
        
        # Determine source type from filename
        file_ext = Path(file_name).suffix.lower() if file_name else ".txt"
        source_type_map = {
            '.docx': 'docx',
            '.pdf': 'pdf',
            '.xlsx': 'xlsx',
            '.xls': 'xlsx',
            '.csv': 'csv',
            '.txt': 'txt'
        }
        source_type = source_type_map.get(file_ext, 'document')
        
        # Combine all chunk texts for constituency-aware re-chunking
        full_text = ""
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                full_text += text + "\n\n"
        
        if not full_text.strip():
            return 0
        
        # Use constituency-aware chunker for better retrieval
        canonical_chunks = chunker.chunk_document(
            text=full_text,
            doc_id=doc_id,
            source_file=file_name,
            source_type=source_type
        )
        
        if not canonical_chunks:
            # Fallback to original chunks if re-chunking fails
            logger.warning("constituency_chunker_fallback", reason="No chunks produced")
            canonical_chunks = []
            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                if text:
                    from app.services.rag.evidence_rag import CanonicalChunk
                    canonical_chunks.append(CanonicalChunk(
                        chunk_id=f"{doc_id}_chunk_{i}",
                        text=text,
                        doc_id=doc_id,
                        source_file_name=file_name,
                        source_type=source_type,
                        constituency=chunker._detect_constituency(text),
                        district=chunker._detect_district(text),
                        topic_tags=chunker._detect_topics(text)
                    ))
        
        # Generate embeddings in batch
        texts_to_embed = [c.text for c in canonical_chunks]
        if not texts_to_embed:
            return 0
            
        embeddings = embedder.embed(texts_to_embed)
        
        # Prepare Document objects with rich metadata
        documents = []
        for chunk, embedding in zip(canonical_chunks, embeddings):
            doc = Document(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                metadata={
                    "source_doc_id": doc_id,
                    "source_type": chunk.source_type,
                    "source_file": chunk.source_file_name,
                    "chunk_id": chunk.chunk_id,
                    "data_type": "document",
                    # Constituency-aware metadata for filtered retrieval
                    "constituency": chunk.constituency,
                    "district": chunk.district,
                    "topic_tags": chunk.topic_tags,
                    "heading": chunk.heading,
                    "chunk_summary": chunk.chunk_summary
                }
            )
            documents.append(doc)
        
        # Bulk index to OpenSearch
        if documents:
            indexed = os_client.index_documents(documents)
            
            # Log details
            constituencies_found = set(d.metadata.get("constituency") for d in documents if d.metadata.get("constituency"))
            districts_found = set(d.metadata.get("district") for d in documents if d.metadata.get("district"))
            
            logger.info("opensearch_indexed", 
                       doc_id=doc_id, 
                       chunks=indexed,
                       constituencies=list(constituencies_found),
                       districts=list(districts_found))
            return indexed
        
        return 0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.warning("opensearch_index_failed", error=str(e))
        return 0


async def _index_to_knowledge_graph(chunks: List[dict], doc_id: str, file_name: str = ""):
    """
    Index document chunks to the Knowledge Graph for entity-based retrieval.
    
    Extracts facts from chunks and adds them to the KG so that when
    users query about specific constituencies, the KG includes this data.
    """
    try:
        import re
        
        # Get the Political RAG system's knowledge graph
        political_rag = orchestrator._get_political_rag()
        if not political_rag or not political_rag.kg:
            logger.info("kg_skip", reason="Knowledge graph not available")
            return 0
        
        kg = political_rag.kg
        facts_added = 0
        
        # Known constituencies for entity extraction
        known_constituencies = set(kg.constituency_profiles.keys()) if hasattr(kg, 'constituency_profiles') else set()
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if not text:
                continue
            
            # Extract constituency mentions from the text
            mentioned_constituencies = []
            for ac_name in known_constituencies:
                if ac_name.lower() in text.lower():
                    mentioned_constituencies.append(ac_name)
            
            # Extract party mentions
            parties_mentioned = []
            party_patterns = {
                'BJP': r'\b(bjp|bharatiya janata)\b',
                'TMC': r'\b(tmc|trinamool|aitc)\b',
                'INC': r'\b(congress|inc)\b',
                'CPM': r'\b(cpm|cpim|left front|communist)\b'
            }
            for party, pattern in party_patterns.items():
                if re.search(pattern, text.lower()):
                    parties_mentioned.append(party)
            
            # Create facts for the knowledge graph
            if mentioned_constituencies or parties_mentioned:
                from app.services.rag.data_schema import FactWithCitation
                
                # Determine fact type based on content
                fact_type = "document_content"
                if any(w in text.lower() for w in ['strategy', 'recommend', 'action', 'plan']):
                    fact_type = "strategy"
                elif any(w in text.lower() for w in ['predict', 'forecast', 'expect']):
                    fact_type = "prediction"
                elif any(w in text.lower() for w in ['result', 'won', 'lost', 'vote share']):
                    fact_type = "electoral_result"
                
                # Create fact for each mentioned constituency
                for ac_name in mentioned_constituencies:
                    fact = FactWithCitation(
                        fact_type=fact_type,
                        entity_name=ac_name,
                        fact_text=text[:500],  # Truncate for storage
                        confidence=0.85,
                        source_file=file_name,
                        source_row=i,
                        related_entities=list(set(mentioned_constituencies + parties_mentioned))
                    )
                    
                    # Add to knowledge graph via add_fact method
                    kg.add_fact(fact)
                    facts_added += 1
                
                # Also create party-level facts
                for party in parties_mentioned:
                    fact = FactWithCitation(
                        fact_type=fact_type,
                        entity_name=party,
                        fact_text=text[:500],
                        confidence=0.85,
                        source_file=file_name,
                        source_row=i,
                        related_entities=list(set(mentioned_constituencies + parties_mentioned))
                    )
                    
                    kg.add_fact(fact)
                    facts_added += 1
        
        logger.info("kg_indexed", doc_id=doc_id, facts=facts_added)
        return facts_added
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.warning("kg_index_failed", error=str(e))
        return 0

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(default=None),
    extract_entities: bool = Form(default=True)
):
    """
    Ingest a document into the RAG system.
    
    Supported formats: XLSX, XLS, DOCX, PDF, TXT
    
    Documents are indexed to:
    1. Local FAISS index (for fast local search)
    2. OpenSearch (for production hybrid search)
    """
    # Validate file type
    allowed_extensions = {'.xlsx', '.xls', '.xlsm', '.docx', '.pdf', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    dest = data_dir / "uploads" / file.filename
    content = await file.read()
    dest.write_bytes(content)
    
    try:
        # Ingest into local index
        doc_id, n_chunks, entities = ingest_file(
            orchestrator.index,
            dest,
            document_id=document_id,
            extract_entities=extract_entities
        )
        
        # Re-process to get raw chunks for indexing
        from app.services.ingest import DocumentProcessor
        processor = DocumentProcessor()
        raw_chunks = processor.process(dest)
        
        # Index to OpenSearch for vector/hybrid search
        opensearch_indexed = await _index_to_opensearch(raw_chunks, doc_id, file.filename)
        
        # Index to Knowledge Graph for entity-based retrieval
        kg_facts = await _index_to_knowledge_graph(raw_chunks, doc_id, file.filename)
        
        # Store extracted entities
        for entity in entities:
            memory.store_entity(entity)
        
        logger.info("document_ingested", 
                   doc_id=doc_id, 
                   local_chunks=n_chunks, 
                   opensearch_chunks=opensearch_indexed,
                   kg_facts=kg_facts,
                   entities=len(entities))
        
        return IngestResponse(
            document_id=doc_id,
            chunks_indexed=n_chunks + opensearch_indexed,
            entities_extracted=len(entities),
            file_name=file.filename,
            index_status={
                "local_index": {"status": "ok", "chunks_indexed": n_chunks},
                "opensearch": {"status": "ok" if opensearch_indexed > 0 else "skipped_or_failed", "chunks_indexed": opensearch_indexed},
                "knowledge_graph": {"status": "ok" if kg_facts > 0 else "skipped_or_failed", "facts_added": kg_facts},
            }
        )
    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ============= Memory Endpoints =============

@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Get session memory and conversation history."""
    return memory.get_session(session_id)


@app.get("/entities/{entity_type}")
async def get_entities(entity_type: str):
    """Get all entities of a specific type."""
    entities = memory.get_entities_by_type(entity_type)
    return {"entity_type": entity_type, "entities": [e.model_dump() for e in entities]}


@app.get("/entities/search/{query}")
async def search_entities(query: str, limit: int = 10):
    """Search entities by name or attributes."""
    entities = memory.search_entities(query, limit=limit)
    return {"query": query, "results": [e.model_dump() for e in entities]}


# ============= WebSocket Chat =============

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


manager = ConnectionManager()


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time strategy chat with streaming.
    
    Message format:
    {
        "type": "query",  // or "feedback"
        "session_id": "uuid",
        "query": "Design a winning strategy for BJP in Nandigram",
        "constituency": "Nandigram",  // optional
        "party": "BJP",  // optional
        "stream": true  // Enable token-by-token streaming
    }
    
    Feedback message:
    {
        "type": "feedback",
        "response_id": "...",
        "feedback_type": "correction",
        "original_text": "...",
        "corrected_text": "...",
        "rating": 5
    }
    
    Streaming responses:
    - Quick ack: {"type": "ack", "message": "Processing..."}
    - Agent activity: {"type": "agent_activity", "agent": "...", "status": "...", "task": "..."}
    - Stream chunk: {"type": "stream", "chunk": "...", "done": false}
    - Final response: {"type": "final_response", "answer": "...", ...}
    - Feedback ack: {"type": "feedback_received", "applied": true}
    """
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message
            payload = await websocket.receive_json()
            msg_type = payload.get("type", "query")
            
            # Handle feedback messages
            if msg_type == "feedback":
                await _handle_ws_feedback(websocket, payload)
                continue
            
            # Handle query messages
            session_id = payload.get("session_id", session_id)
            query = payload.get("query", "")
            constituency = payload.get("constituency")
            party = payload.get("party")
            stream = payload.get("stream", True)
            
            if not query:
                await websocket.send_json({"type": "error", "message": "Query is required"})
                continue
            
            # Send immediate acknowledgment
            await websocket.send_json({
                "type": "ack",
                "message": f"Processing: {query[:50]}...",
                "session_id": session_id,
                "response_id": f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            })
            
            # Store user message in memory
            memory.append_turn(session_id, "user", query)
            
            # Define update callback for streaming
            async def send_update(update: AgentUpdate):
                await websocket.send_json({
                    "type": "agent_activity",
                    **update.model_dump()
                })
            
            try:
                # Run orchestrator
                result = await orchestrator.run(
                    query=query,
                    session_id=session_id,
                    constituency=constituency,
                    party=party,
                    send_update=send_update
                )
                
                # Store assistant response
                answer = result.get("answer", "")
                memory.append_turn(session_id, "assistant", answer)
                
                # Stream response in chunks if enabled
                if stream and len(answer) > 100:
                    chunks = answer.split('\n')
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            await websocket.send_json({
                                "type": "stream",
                                "chunk": chunk + '\n',
                                "done": i == len(chunks) - 1
                            })
                            await asyncio.sleep(0.05)  # Small delay for visual effect
                
                # Build final response
                citations = result.get("citations", [])
                response_id = f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                interactions = result.get("interactions", []) or []
                interaction = result.get("interaction")
                if interaction and not interactions:
                    interactions = [interaction]
                
                final = {
                    "type": "final_response",
                    "response_id": response_id,
                    "answer": answer,
                    "citations": [c.model_dump() if hasattr(c, 'model_dump') else c for c in citations],
                    "agents_used": result.get("agents_used", []),
                    "confidence": result.get("confidence", 0.5),
                    "memory_stored": True,
                    "can_provide_feedback": True,
                    # HITL / interactive fields (for multi-turn clarification + followups)
                    "needs_clarification": result.get("needs_clarification", False),
                    "interaction": interaction,
                    "interactions": interactions,
                    "conversation_context": result.get("conversation_context"),
                }
                
                await websocket.send_json(final)
                
            except Exception as e:
                logger.error("chat_error", error=str(e), session=session_id)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing query: {str(e)}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info("websocket_disconnected", session=session_id)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        manager.disconnect(session_id)


async def _handle_ws_feedback(websocket: WebSocket, payload: dict):
    """Handle feedback submitted via WebSocket."""
    try:
        from app.services.feedback_learning import FeedbackRequest
        
        request = FeedbackRequest(
            session_id=payload.get("session_id", "ws_session"),
            response_id=payload.get("response_id", ""),
            feedback_type=payload.get("feedback_type", "comment"),
            rating=payload.get("rating"),
            original_text=payload.get("original_text"),
            corrected_text=payload.get("corrected_text"),
            comment=payload.get("comment"),
            entity_type=payload.get("entity_type"),
            entity_name=payload.get("entity_name")
        )
        
        response = learning_engine.process_feedback(request)
        
        await websocket.send_json({
            "type": "feedback_received",
            "feedback_id": response.feedback_id,
            "status": response.status,
            "message": response.message,
            "correction_applied": response.correction_applied,
            "learning_updated": response.learning_updated
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Feedback error: {str(e)}"
        })


# ============= REST Chat (Alternative) =============

class ChatRequestREST(BaseModel):
    """REST chat request model."""
    session_id: str
    query: str
    constituency: Optional[str] = None
    party: Optional[str] = None
    use_crewai: bool = False  # Set to True for autonomous CrewAI agents


@app.post("/chat")
async def chat_rest(request: ChatRequestREST):
    """
    REST endpoint for strategy chat (non-streaming).
    Use WebSocket for real-time agent updates.
    
    Set use_crewai=true for autonomous multi-agent collaboration.
    """
    # Store user message
    memory.append_turn(request.session_id, "user", request.query)
    
    try:
        # Run orchestrator (with optional CrewAI mode)
        result = await orchestrator.run(
            query=request.query,
            session_id=request.session_id,
            constituency=request.constituency,
            party=request.party,
            use_crewai=request.use_crewai
        )
        
        # Store response
        answer = result.get("answer", "")
        memory.append_turn(request.session_id, "assistant", answer)

        # Normalize HITL fields: include blocking interaction inside interactions list for UI
        interactions = result.get("interactions", []) or []
        interaction = result.get("interaction")
        if interaction and not interactions:
            interactions = [interaction]
        
        return FinalResponse(
            answer=answer,
            strategy=result.get("strategy"),
            citations=result.get("citations", []),
            agents_used=result.get("agents_used", []),
            confidence=result.get("confidence", 0.5),
            memory_stored=True,
            needs_clarification=bool(result.get("needs_clarification", False)),
            interaction=interaction,
            interactions=interactions,
            conversation_context=result.get("conversation_context"),
        )
    
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============= Quick Analysis =============

@app.post("/quick-analysis")
async def quick_analysis(request: ChatRequestREST):
    """
    Quick analysis endpoint using only Intelligence + Reporter.
    Faster but less comprehensive.
    """
    try:
        result = await orchestrator.quick_analysis(request.query)
        
        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "agents_used": result.get("agents_used", []),
            "confidence": result.get("confidence", 0.5)
        }
    
    except Exception as e:
        logger.error("quick_analysis_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============= Political RAG Endpoints =============

class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    use_llm: bool = True


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """
    Query the Political RAG system with zero-hallucination guarantees.
    
    Returns verified, cited information about West Bengal politics.
    """
    try:
        rag = get_political_rag()
        response = rag.query(request.query, use_llm=request.use_llm)
        return response.to_dict()
    except Exception as e:
        logger.error("rag_query_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/constituency/{name}")
async def get_constituency(name: str):
    """Get detailed constituency profile."""
    try:
        rag = get_political_rag()
        profile = rag.get_constituency_profile(name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Constituency not found: {name}")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error("constituency_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/constituencies")
async def list_constituencies(
    district: Optional[str] = Query(None),
    pc: Optional[str] = Query(None),
    winner_2021: Optional[str] = Query(None),
    predicted_2026: Optional[str] = Query(None),
    race_rating: Optional[str] = Query(None)
):
    """List constituencies with optional filters."""
    try:
        rag = get_political_rag()
        return rag.list_constituencies(
            district=district,
            pc=pc,
            winner_2021=winner_2021,
            predicted_2026=predicted_2026,
            race_rating=race_rating
        )
    except Exception as e:
        logger.error("constituencies_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/predictions")
async def get_predictions():
    """Get 2026 election predictions summary."""
    try:
        rag = get_political_rag()
        return rag.get_predictions_summary()
    except Exception as e:
        logger.error("predictions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/swing-analysis")
async def get_swing_analysis():
    """Get comprehensive swing analysis."""
    try:
        rag = get_political_rag()
        return rag.get_swing_analysis()
    except Exception as e:
        logger.error("swing_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/district/{name}")
async def get_district_summary(name: str):
    """Get district-level summary."""
    try:
        rag = get_political_rag()
        summary = rag.get_district_summary(name)
        return {"district": name, "summary": summary}
    except Exception as e:
        logger.error("district_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search")
async def rag_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results")
):
    """Direct search without LLM processing."""
    try:
        rag = get_political_rag()
        return rag.search(q, top_k=top_k)
    except Exception as e:
        logger.error("search_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/initialize")
async def initialize_rag(force_rebuild: bool = False):
    """Initialize or rebuild the RAG system."""
    try:
        rag = get_political_rag()
        stats = rag.initialize(force_rebuild=force_rebuild)
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error("init_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============= Query Understanding System =============

from app.services.query_understanding import create_query_engine, QueryIntent

@app.post("/query/understand")
async def understand_query(query: str = Form(...)):
    """
    Analyze and understand a query before processing.
    
    Returns detailed breakdown of:
    - Primary and secondary intents
    - Extracted entities (constituencies, parties, etc.)
    - Time context
    - Complexity assessment
    - Suggested agents to handle the query
    - Confidence in understanding
    """
    try:
        rag = get_political_rag()
        engine = create_query_engine(rag.kg, use_llm=True)
        analysis = engine.understand(query)
        
        return {
            "original_query": analysis.original_query,
            "cleaned_query": analysis.cleaned_query,
            "primary_intent": {
                "value": analysis.primary_intent.value,
                "description": _get_intent_description(analysis.primary_intent)
            },
            "secondary_intents": [
                {"value": i.value, "description": _get_intent_description(i)}
                for i in analysis.secondary_intents
            ],
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type.value,
                    "normalized": e.normalized,
                    "confidence": e.confidence
                }
                for e in analysis.entities
            ],
            "time_context": analysis.time_context,
            "comparison_items": analysis.comparison_items,
            "is_complex": analysis.is_complex,
            "sub_queries": analysis.sub_queries,
            "required_data": analysis.required_data,
            "suggested_agents": analysis.suggested_agents,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning
        }
    except Exception as e:
        logger.error("query_understand_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/intents")
async def list_intents():
    """List all recognized query intents with descriptions."""
    return {
        "intents": [
            {"value": intent.value, "description": _get_intent_description(intent)}
            for intent in QueryIntent
        ]
    }


def _get_intent_description(intent: QueryIntent) -> str:
    """Get human-readable description for an intent."""
    descriptions = {
        QueryIntent.FACTUAL_RESULT: "Asking about election results (who won, margins)",
        QueryIntent.FACTUAL_COUNT: "Asking about counts or numbers (seats, votes)",
        QueryIntent.FACTUAL_PROFILE: "Requesting profile of a specific entity",
        QueryIntent.ANALYTICAL_COMPARISON: "Comparing two or more entities",
        QueryIntent.ANALYTICAL_TREND: "Analyzing trends or changes over time",
        QueryIntent.ANALYTICAL_WHY: "Seeking causal explanation (why questions)",
        QueryIntent.ANALYTICAL_PATTERN: "Looking for patterns or correlations",
        QueryIntent.PREDICTIVE_OUTCOME: "Predicting future election outcomes",
        QueryIntent.PREDICTIVE_PROBABILITY: "Asking about probabilities/chances",
        QueryIntent.PREDICTIVE_SCENARIO: "What-if scenario analysis",
        QueryIntent.STRATEGIC_PLAN: "Requesting strategic advice/planning",
        QueryIntent.STRATEGIC_RESOURCE: "Resource allocation decisions",
        QueryIntent.STRATEGIC_VOTER: "Voter segmentation and targeting",
        QueryIntent.STRATEGIC_CAMPAIGN: "Campaign design and operations",
        QueryIntent.STRATEGIC_OPPOSITION: "Opposition analysis/vulnerabilities",
        QueryIntent.EXPLORATORY_OVERVIEW: "General overview or landscape",
        QueryIntent.EXPLORATORY_LIST: "Listing entities",
        QueryIntent.EXPLORATORY_SEARCH: "Searching for information",
        QueryIntent.META_CAPABILITY: "Asking about system capabilities",
        QueryIntent.META_DATA_SOURCE: "Asking about data sources",
        QueryIntent.UNKNOWN: "Intent not recognized"
    }
    return descriptions.get(intent, "Unknown intent")


# ============= Feedback & Learning System =============

from app.services.feedback_learning import (
    FeedbackRequest, FeedbackResponse, 
    get_feedback_store, get_learning_engine, get_quick_response_manager
)

feedback_store = get_feedback_store()
learning_engine = get_learning_engine()
quick_manager = get_quick_response_manager()


class InteractiveChatRequest(BaseModel):
    """Interactive chat request with streaming support."""
    session_id: str
    query: str
    constituency: Optional[str] = None
    party: Optional[str] = None
    stream: bool = False  # Enable streaming response
    apply_learnings: bool = True  # Apply previous corrections


@app.post("/chat/interactive")
async def interactive_chat(request: InteractiveChatRequest):
    """
    Interactive chat with quick acknowledgment and learning integration.
    
    Features:
    - Immediate acknowledgment
    - Applies learned corrections
    - Faster response time
    - Progressive detail loading
    """
    # Store user message
    memory.append_turn(request.session_id, "user", request.query)
    
    # Get quick summary for immediate response
    quick_summary = await quick_manager.get_quick_summary(request.query)
    
    try:
        # Check cache first
        cached = quick_manager.get_cached_response(
            request.query, 
            {"constituency": request.constituency, "party": request.party}
        )
        
        if cached:
            return {
                "type": "cached_response",
                "quick_summary": "Retrieved from recent analysis",
                "answer": cached,
                "cached": True,
                "confidence": 0.9
            }
        
        # Run orchestrator
        result = await orchestrator.run(
            query=request.query,
            session_id=request.session_id,
            constituency=request.constituency,
            party=request.party
        )
        
        answer = result.get("answer", "")
        
        # Apply learned corrections if enabled
        corrections_applied = []
        if request.apply_learnings:
            entities = []
            if request.constituency:
                entities.append(request.constituency)
            
            answer, corrections_applied = learning_engine.apply_learnings_to_response(
                answer, entities
            )
        
        # Cache the response
        quick_manager.cache_response(
            request.query, answer,
            {"constituency": request.constituency, "party": request.party}
        )
        
        # Store in memory
        memory.append_turn(request.session_id, "assistant", answer)
        
        # Generate response ID for feedback
        response_id = f"{request.session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        interactions = result.get("interactions", []) or []
        interaction = result.get("interaction")
        if interaction and not interactions:
            interactions = [interaction]

        return {
            "type": "full_response",
            "response_id": response_id,
            "quick_summary": quick_summary,
            "answer": answer,
            "citations": result.get("citations", []),
            "agents_used": result.get("agents_used", []),
            "confidence": result.get("confidence", 0.7),
            "corrections_applied": corrections_applied,
            "cached": False,
            "can_provide_feedback": True,
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "interaction": interaction,
            "interactions": interactions,
            "conversation_context": result.get("conversation_context"),
        }
    
    except Exception as e:
        logger.error("interactive_chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a response.
    
    Feedback types:
    - rating: 1-5 star rating
    - correction: Factual correction (provide original_text and corrected_text)
    - addition: New information (provide corrected_text)
    - clarification: Request more detail
    - disagreement: User disagrees (provide comment)
    - confirmation: User confirms correctness
    - comment: General comment
    
    Corrections are immediately applied to improve future responses.
    """
    try:
        # Get original query from session if available
        original_query = ""
        session_data = memory.get_session(request.session_id)
        if session_data and session_data.get("turns"):
            for turn in reversed(session_data["turns"]):
                if turn.get("role") == "user":
                    original_query = turn.get("content", "")
                    break
        
        response = learning_engine.process_feedback(request, original_query)
        
        logger.info(
            "feedback_received",
            feedback_id=response.feedback_id,
            feedback_type=request.feedback_type,
            correction_applied=response.correction_applied
        )
        
        return response
    
    except Exception as e:
        logger.error("feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback and learning statistics."""
    return feedback_store.get_feedback_stats()


@app.get("/feedback/learnings")
async def get_all_learnings():
    """Get all learned facts from user feedback."""
    return feedback_store.get_all_learnings()


@app.get("/feedback/learnings/{entity_name}")
async def get_entity_learnings(entity_name: str):
    """Get learned facts for a specific entity."""
    learnings = feedback_store.get_learnings(entity_name)
    if not learnings:
        return {"entity": entity_name, "learnings": {}, "message": "No learnings found"}
    return {"entity": entity_name, "learnings": learnings}


class KnowledgeUpdateRequest(BaseModel):
    """Request to update knowledge base."""
    entity_type: str  # constituency, party, prediction
    entity_name: str
    field: str
    old_value: Optional[str] = None
    new_value: str
    reason: str


@app.post("/knowledge/update")
async def update_knowledge(request: KnowledgeUpdateRequest):
    """
    Directly update knowledge base with new information.
    
    Use this for administrative corrections or verified data updates.
    """
    try:
        feedback_store.add_learning(
            entity_name=request.entity_name,
            key=request.field,
            value={
                "value": request.new_value,
                "old_value": request.old_value,
                "reason": request.reason,
                "updated_at": datetime.now().isoformat()
            },
            source="admin_update"
        )
        
        logger.info(
            "knowledge_updated",
            entity=request.entity_name,
            field=request.field
        )
        
        return {
            "status": "success",
            "message": f"Knowledge updated for {request.entity_name}.{request.field}",
            "entity": request.entity_name,
            "field": request.field
        }
    
    except Exception as e:
        logger.error("knowledge_update_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/correct")
async def correct_knowledge(
    entity_name: str,
    original_text: str,
    corrected_text: str,
    reason: Optional[str] = None
):
    """
    Quick correction endpoint for knowledge base.
    
    Example: Correct a wrong statistic or name.
    """
    feedback_request = FeedbackRequest(
        session_id="admin",
        response_id="direct_correction",
        feedback_type="correction",
        entity_name=entity_name,
        original_text=original_text,
        corrected_text=corrected_text,
        comment=reason
    )
    
    response = learning_engine.process_feedback(feedback_request)
    return response


# ============= Local Run =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
