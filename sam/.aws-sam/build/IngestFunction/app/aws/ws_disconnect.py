"""
WebSocket $disconnect handler.
"""
from __future__ import annotations
from typing import Any, Dict

from app.aws.ws_common import success_response


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Handle WebSocket disconnection.
    
    Clean up any resources associated with the connection.
    """
    connection_id = event["requestContext"]["connectionId"]
    
    # Optional: Remove connection from DynamoDB tracking
    print(f"WebSocket disconnected: {connection_id}")
    
    return success_response({"message": "Disconnected"})
