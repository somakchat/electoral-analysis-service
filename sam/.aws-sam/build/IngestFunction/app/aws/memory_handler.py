"""
Lambda handler for memory retrieval (REST API).
"""
from __future__ import annotations
import json
from typing import Any, Dict

from app.aws.ws_common import success_response, error_response
from app.services.memory import get_memory_store


memory = None


def get_memory():
    global memory
    if memory is None:
        memory = get_memory_store()
    return memory


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle memory retrieval requests.
    
    Routes:
    - GET /memory/{session_id}: Get session history
    - GET /entities/{entity_type}: Get entities by type
    - GET /entities/search/{query}: Search entities
    """
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "")
    path_params = event.get("pathParameters", {}) or {}
    query_params = event.get("queryStringParameters", {}) or {}
    
    mem = get_memory()
    
    try:
        if "/memory/" in path:
            session_id = path_params.get("session_id")
            if not session_id:
                return error_response(400, "session_id required")
            
            session = mem.get_session(session_id)
            return success_response(session)
        
        elif "/entities/search/" in path:
            query = path_params.get("query", "")
            limit = int(query_params.get("limit", 10))
            
            entities = mem.search_entities(query, limit=limit)
            return success_response({
                "query": query,
                "results": [e.model_dump() for e in entities]
            })
        
        elif "/entities/" in path:
            entity_type = path_params.get("entity_type")
            if not entity_type:
                return error_response(400, "entity_type required")
            
            entities = mem.get_entities_by_type(entity_type)
            return success_response({
                "entity_type": entity_type,
                "entities": [e.model_dump() for e in entities]
            })
        
        else:
            return error_response(404, "Not found")
    
    except Exception as e:
        print(f"Memory handler error: {e}")
        return error_response(500, str(e))
