"""
Common utilities for WebSocket Lambda handlers.
"""
from __future__ import annotations
import json
import boto3
from typing import Any, Dict

from app.config import settings


def get_apigw_client(endpoint_url: str):
    """Get API Gateway Management API client."""
    return boto3.client(
        'apigatewaymanagementapi',
        endpoint_url=endpoint_url,
        region_name=settings.aws_region
    )


def build_endpoint_url(event: Dict[str, Any]) -> str:
    """Build API Gateway endpoint URL from event."""
    domain = event["requestContext"]["domainName"]
    stage = event["requestContext"]["stage"]
    return f"https://{domain}/{stage}"


async def send_to_connection(
    apigw_client,
    connection_id: str,
    data: Dict[str, Any]
) -> bool:
    """Send data to a WebSocket connection."""
    try:
        apigw_client.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(data).encode('utf-8')
        )
        return True
    except apigw_client.exceptions.GoneException:
        # Connection is gone
        return False
    except Exception as e:
        print(f"Error sending to connection {connection_id}: {e}")
        return False


def success_response(body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build successful Lambda response."""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body or {"message": "Success"})
    }


def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Build error Lambda response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({"error": message})
    }
