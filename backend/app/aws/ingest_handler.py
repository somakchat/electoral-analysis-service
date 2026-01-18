"""
Lambda handler for document ingestion (REST API).
"""
from __future__ import annotations
import json
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict

from app.aws.ws_common import success_response, error_response
from app.services.orchestrator import Orchestrator
from app.services.ingest import ingest_file
from app.services.memory import get_memory_store


# Initialize services
orchestrator = None
memory = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator(index_dir="/tmp/index")
    return orchestrator


def get_memory():
    global memory
    if memory is None:
        memory = get_memory_store()
    return memory


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle document ingestion via REST API.
    
    Expects multipart/form-data with:
    - file: The document file (base64 encoded in Lambda)
    - document_id: Optional custom document ID
    - extract_entities: Whether to extract entities (default: true)
    """
    try:
        # Parse request
        if event.get("isBase64Encoded"):
            body = base64.b64decode(event["body"])
        else:
            body = event.get("body", "").encode()
        
        # For API Gateway, we need to parse multipart form data
        # This is simplified - in production use a proper multipart parser
        content_type = event.get("headers", {}).get("content-type", "")
        
        if "multipart/form-data" in content_type:
            # Parse multipart form data
            file_content, file_name, document_id, extract_entities = parse_multipart(
                body, content_type
            )
        else:
            # Assume JSON body with base64 file
            json_body = json.loads(body.decode())
            file_content = base64.b64decode(json_body.get("file_content", ""))
            file_name = json_body.get("file_name", "document.txt")
            document_id = json_body.get("document_id")
            extract_entities = json_body.get("extract_entities", True)
        
        if not file_content:
            return error_response(400, "No file provided")
        
        # Save to temp file
        suffix = Path(file_name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)
        
        try:
            # Ingest file
            orch = get_orchestrator()
            doc_id, n_chunks, entities = ingest_file(
                orch.index,
                tmp_path,
                document_id=document_id,
                extract_entities=extract_entities
            )
            
            # Store entities
            mem = get_memory()
            for entity in entities:
                mem.store_entity(entity)
            
            return success_response({
                "document_id": doc_id,
                "chunks_indexed": n_chunks,
                "entities_extracted": len(entities),
                "file_name": file_name
            })
            
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)
    
    except Exception as e:
        print(f"Ingest error: {e}")
        return error_response(500, f"Ingestion failed: {str(e)}")


def parse_multipart(body: bytes, content_type: str):
    """Parse multipart form data (simplified)."""
    # Extract boundary
    import re
    boundary_match = re.search(r'boundary=(.+)', content_type)
    if not boundary_match:
        raise ValueError("No boundary in multipart content-type")
    
    boundary = boundary_match.group(1).encode()
    parts = body.split(b'--' + boundary)
    
    file_content = None
    file_name = "document.txt"
    document_id = None
    extract_entities = True
    
    for part in parts:
        if b'name="file"' in part:
            # Extract filename
            name_match = re.search(b'filename="([^"]+)"', part)
            if name_match:
                file_name = name_match.group(1).decode()
            
            # Extract content (after double CRLF)
            content_start = part.find(b'\r\n\r\n')
            if content_start != -1:
                file_content = part[content_start + 4:].rstrip(b'\r\n--')
        
        elif b'name="document_id"' in part:
            content_start = part.find(b'\r\n\r\n')
            if content_start != -1:
                document_id = part[content_start + 4:].strip().decode()
        
        elif b'name="extract_entities"' in part:
            content_start = part.find(b'\r\n\r\n')
            if content_start != -1:
                value = part[content_start + 4:].strip().decode().lower()
                extract_entities = value in ('true', '1', 'yes')
    
    return file_content, file_name, document_id, extract_entities
