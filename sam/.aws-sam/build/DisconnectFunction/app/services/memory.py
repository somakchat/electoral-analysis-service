"""
Memory System for Political Strategy Maker.

Implements three types of memory:
1. Short-Term Memory - Session context and recent interactions
2. Long-Term Memory - Persistent learnings across sessions
3. Entity Memory - Tracks political entities (constituencies, candidates, parties)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import time

from app.config import settings
from app.models import EntityReference, MemoryItem


class LocalMemoryStore:
    """
    Local file-based memory store for development.
    """
    
    def __init__(self, data_dir: str = None) -> None:
        self.root = Path(data_dir or settings.data_dir) / "memory"
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories for different memory types
        (self.root / "sessions").mkdir(exist_ok=True)
        (self.root / "long_term").mkdir(exist_ok=True)
        (self.root / "entities").mkdir(exist_ok=True)
    
    # ===== Session / Short-Term Memory =====
    
    def _session_path(self, session_id: str) -> Path:
        return self.root / "sessions" / f"{session_id}.json"
    
    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a conversation turn to session history."""
        path = self._session_path(session_id)
        
        data = {
            "session_id": session_id,
            "turns": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        
        data["turns"].append({
            "timestamp": int(time.time()),
            "role": role,
            "content": content
        })
        data["updated_at"] = datetime.now().isoformat()
        
        # Keep only last N turns
        if len(data["turns"]) > settings.short_term_max_items:
            data["turns"] = data["turns"][-settings.short_term_max_items:]
        
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get full session history."""
        path = self._session_path(session_id)
        
        if not path.exists():
            return {"session_id": session_id, "turns": [], "entities_mentioned": []}
        
        return json.loads(path.read_text(encoding="utf-8"))
    
    def get_recent_context(self, session_id: str, n_turns: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation turns for context."""
        session = self.get_session(session_id)
        return session.get("turns", [])[-n_turns:]
    
    # ===== Long-Term Memory =====
    
    def _memory_path(self, memory_id: str) -> Path:
        return self.root / "long_term" / f"{memory_id}.json"
    
    def store_learning(
        self,
        content: str,
        source_session: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a learning/insight for long-term memory."""
        memory_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]
        
        data = {
            "memory_id": memory_id,
            "content": content,
            "source_session": source_session,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        
        path = self._memory_path(memory_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        return memory_id
    
    def retrieve_relevant_memories(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant long-term memories."""
        memories = []
        
        for path in (self.root / "long_term").glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                
                # Simple relevance scoring based on keyword overlap
                query_words = set(query.lower().split())
                content_words = set(data["content"].lower().split())
                overlap = len(query_words & content_words)
                
                if overlap > 0:
                    data["relevance_score"] = overlap
                    memories.append(data)
            except Exception:
                continue
        
        # Sort by relevance and return top
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return memories[:limit]
    
    # ===== Entity Memory =====
    
    def _entity_path(self, entity_type: str, entity_name: str) -> Path:
        safe_name = entity_name.replace(" ", "_").replace("/", "_")[:50]
        return self.root / "entities" / f"{entity_type}_{safe_name}.json"
    
    def store_entity(self, entity: EntityReference) -> None:
        """Store or update an entity."""
        path = self._entity_path(entity.entity_type, entity.entity_name)
        
        data = {
            "entity_type": entity.entity_type,
            "entity_name": entity.entity_name,
            "attributes": entity.attributes,
            "source_doc_ids": entity.source_doc_ids,
            "confidence": entity.confidence,
            "updated_at": datetime.now().isoformat()
        }
        
        # Merge with existing if present
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            # Merge attributes
            existing["attributes"].update(data["attributes"])
            # Merge source docs
            existing["source_doc_ids"] = list(set(
                existing.get("source_doc_ids", []) + data["source_doc_ids"]
            ))
            data = existing
            data["updated_at"] = datetime.now().isoformat()
        
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def get_entity(self, entity_type: str, entity_name: str) -> Optional[EntityReference]:
        """Retrieve an entity."""
        path = self._entity_path(entity_type, entity_name)
        
        if not path.exists():
            return None
        
        data = json.loads(path.read_text(encoding="utf-8"))
        return EntityReference(**data)
    
    def get_entities_by_type(self, entity_type: str) -> List[EntityReference]:
        """Get all entities of a given type."""
        entities = []
        
        for path in (self.root / "entities").glob(f"{entity_type}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entities.append(EntityReference(**data))
            except Exception:
                continue
        
        return entities
    
    def search_entities(self, query: str, limit: int = 10) -> List[EntityReference]:
        """Search entities by name or attributes."""
        results = []
        query_lower = query.lower()
        
        for path in (self.root / "entities").glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                
                # Check name match
                if query_lower in data["entity_name"].lower():
                    results.append(EntityReference(**data))
                # Check attribute match
                elif query_lower in str(data["attributes"]).lower():
                    results.append(EntityReference(**data))
            except Exception:
                continue
        
        return results[:limit]


class DynamoDBMemoryStore:
    """
    DynamoDB-based memory store for AWS production.
    """
    
    def __init__(self):
        self._client = None
        self._sessions_table = settings.ddb_table_sessions
        self._memory_table = settings.ddb_table_memory
    
    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.resource('dynamodb', region_name=settings.aws_region)
        return self._client
    
    # ===== Session / Short-Term Memory =====
    
    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a conversation turn."""
        table = self._get_client().Table(self._sessions_table)
        
        turn = {
            "timestamp": int(time.time()),
            "role": role,
            "content": content
        }
        
        try:
            # Try to update existing session
            table.update_item(
                Key={"session_id": session_id},
                UpdateExpression="SET turns = list_append(if_not_exists(turns, :empty), :turn), updated_at = :now",
                ExpressionAttributeValues={
                    ":turn": [turn],
                    ":empty": [],
                    ":now": datetime.now().isoformat()
                }
            )
        except Exception:
            # Create new session
            table.put_item(Item={
                "session_id": session_id,
                "turns": [turn],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session history."""
        table = self._get_client().Table(self._sessions_table)
        
        response = table.get_item(Key={"session_id": session_id})
        return response.get("Item", {"session_id": session_id, "turns": []})
    
    # ===== Long-Term Memory =====
    
    def store_learning(
        self,
        content: str,
        source_session: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a learning."""
        table = self._get_client().Table(self._memory_table)
        
        memory_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]
        
        table.put_item(Item={
            "pk": "MEMORY",
            "sk": f"LEARNING#{memory_id}",
            "content": content,
            "source_session": source_session,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "ttl": int((datetime.now() + timedelta(days=90)).timestamp())  # 90 days TTL
        })
        
        return memory_id
    
    # ===== Entity Memory =====
    
    def store_entity(self, entity: EntityReference) -> None:
        """Store an entity."""
        table = self._get_client().Table(self._memory_table)
        
        table.put_item(Item={
            "pk": f"ENTITY#{entity.entity_type}",
            "sk": entity.entity_name,
            "attributes": entity.attributes,
            "source_doc_ids": entity.source_doc_ids,
            "confidence": str(entity.confidence),
            "updated_at": datetime.now().isoformat()
        })
    
    def get_entity(self, entity_type: str, entity_name: str) -> Optional[EntityReference]:
        """Get an entity."""
        table = self._get_client().Table(self._memory_table)
        
        response = table.get_item(Key={
            "pk": f"ENTITY#{entity_type}",
            "sk": entity_name
        })
        
        item = response.get("Item")
        if not item:
            return None
        
        return EntityReference(
            entity_type=entity_type,
            entity_name=item["sk"],
            attributes=item.get("attributes", {}),
            source_doc_ids=item.get("source_doc_ids", []),
            confidence=float(item.get("confidence", 0))
        )
    
    def get_entities_by_type(self, entity_type: str) -> List[EntityReference]:
        """Get all entities of a type."""
        table = self._get_client().Table(self._memory_table)
        
        response = table.query(
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": f"ENTITY#{entity_type}"}
        )
        
        entities = []
        for item in response.get("Items", []):
            entities.append(EntityReference(
                entity_type=entity_type,
                entity_name=item["sk"],
                attributes=item.get("attributes", {}),
                source_doc_ids=item.get("source_doc_ids", []),
                confidence=float(item.get("confidence", 0))
            ))
        
        return entities


def get_memory_store():
    """Factory function to get appropriate memory store."""
    if settings.is_aws and settings.ddb_table_sessions:
        return DynamoDBMemoryStore()
    return LocalMemoryStore()
