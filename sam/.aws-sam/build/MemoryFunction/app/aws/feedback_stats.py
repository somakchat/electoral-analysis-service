"""
Simple feedback stats endpoint (stub) to satisfy frontend and CORS.
"""
from __future__ import annotations
from typing import Dict, Any
import json


def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
    }


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Return minimal feedback stats.
    """
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": ""
        }

    body = {
        "total_feedback": 0,
        "total_corrections": 0,
        "average_rating": 0.0,
    }

    return {
        "statusCode": 200,
        "headers": {**_cors_headers(), "Content-Type": "application/json"},
        "body": json.dumps(body)
    }

