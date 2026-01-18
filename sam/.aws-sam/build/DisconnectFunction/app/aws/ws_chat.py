"""
WebSocket chat handler for Lambda.
Handles the 'chat' route action.
"""
from __future__ import annotations
import json
import asyncio
from typing import Any, Dict

from app.aws.ws_common import (
    get_apigw_client, build_endpoint_url, send_to_connection,
    success_response, error_response
)
from app.models import AgentUpdate, FinalResponse
from app.services.orchestrator import Orchestrator
from app.services.memory import get_memory_store
from app.config import settings


# Initialize services (cold start)
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
    Handle chat message via WebSocket.
    
    Message format:
    {
        "action": "chat",
        "session_id": "uuid",
        "query": "Design a winning strategy...",
        "constituency": "Nandigram",
        "party": "BJP"
    }
    """
    # Debug trace to confirm invocation and payload
    print("[ws_chat] event received:", json.dumps({
        "connectionId": event.get("requestContext", {}).get("connectionId"),
        "routeKey": event.get("requestContext", {}).get("routeKey"),
        "domain": event.get("requestContext", {}).get("domainName"),
        "stage": event.get("requestContext", {}).get("stage"),
        "body": event.get("body")
    }))
    connection_id = event["requestContext"]["connectionId"]
    
    # Parse message body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return error_response(400, "Invalid JSON body")
    
    session_id = body.get("session_id", connection_id)
    query = body.get("query", "")
    constituency = body.get("constituency")
    party = body.get("party")
    
    if not query:
        return error_response(400, "Query is required")
    
    # Get API Gateway client
    endpoint_url = build_endpoint_url(event)
    apigw_client = get_apigw_client(endpoint_url)
    
    # Run async handler
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            process_chat(
                apigw_client=apigw_client,
                connection_id=connection_id,
                session_id=session_id,
                query=query,
                constituency=constituency,
                party=party
            )
        )
        return success_response(result)
    except Exception as e:
        print(f"Chat error: {e}")
        # Send error to client
        loop.run_until_complete(
            send_to_connection(apigw_client, connection_id, {
                "type": "error",
                "message": str(e)
            })
        )
        return error_response(500, str(e))
    finally:
        loop.close()


async def process_chat(
    apigw_client,
    connection_id: str,
    session_id: str,
    query: str,
    constituency: str = None,
    party: str = None
) -> Dict[str, Any]:
    """Process chat request and stream updates."""
    
    orch = get_orchestrator()
    mem = get_memory()
    
    # Store user message
    mem.append_turn(session_id, "user", query)
    
    # Define callback to send agent updates
    async def send_update(update: AgentUpdate):
        await send_to_connection(apigw_client, connection_id, update.model_dump())
    
    # Run orchestrator
    result = await orch.run(
        query=query,
        session_id=session_id,
        constituency=constituency,
        party=party,
        send_update=send_update
    )
    
    # Store assistant response
    answer = result.get("answer", "")
    mem.append_turn(session_id, "assistant", answer)
    
    # Build final response
    final = FinalResponse(
        answer=answer,
        strategy=result.get("strategy"),
        citations=result.get("citations", []),
        agents_used=result.get("agents_used", []),
        confidence=result.get("confidence", 0.5),
        memory_stored=True
    )
    
    # Send final response
    await send_to_connection(apigw_client, connection_id, final.model_dump())
    
    return {"status": "completed", "session_id": session_id}
