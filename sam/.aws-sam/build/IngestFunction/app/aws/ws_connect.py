"""
WebSocket $connect handler.
"""
from __future__ import annotations
import json
from typing import Any, Dict

from app.aws.ws_common import success_response, error_response


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle WebSocket connection.
    
    This is called when a client establishes a WebSocket connection.
    We can store the connection ID for later use.
    """
    connection_id = event["requestContext"]["connectionId"]
    
    # Optional: Store connection in DynamoDB for tracking
    # For now, just accept the connection
    print(f"WebSocket connected: {connection_id}")
    
    return success_response({"message": "Connected", "connectionId": connection_id})
