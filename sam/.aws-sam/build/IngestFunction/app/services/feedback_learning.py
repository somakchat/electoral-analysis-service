"""
Feedback & Learning System - Interactive Knowledge Updates.

This module provides:
1. User feedback collection (ratings, corrections, comments)
2. Knowledge base corrections and updates
3. Learning from user interactions
4. Real-time knowledge refinement
5. Correction propagation to agents
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of user feedback."""
    RATING = "rating"                    # 1-5 star rating
    CORRECTION = "correction"            # Factual correction
    ADDITION = "addition"                # New information
    CLARIFICATION = "clarification"      # Request for more detail
    DISAGREEMENT = "disagreement"        # User disagrees with analysis
    CONFIRMATION = "confirmation"        # User confirms correctness
    COMMENT = "comment"                  # General comment


class CorrectionStatus(str, Enum):
    """Status of a correction."""
    PENDING = "pending"
    VERIFIED = "verified"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass
class UserFeedback:
    """User feedback on a response."""
    feedback_id: str
    session_id: str
    query: str
    response_id: str
    feedback_type: FeedbackType
    rating: Optional[int] = None  # 1-5
    original_text: Optional[str] = None
    corrected_text: Optional[str] = None
    comment: Optional[str] = None
    entity_type: Optional[str] = None  # constituency, party, prediction
    entity_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False


@dataclass
class KnowledgeCorrection:
    """A correction to the knowledge base."""
    correction_id: str
    feedback_id: str
    entity_type: str
    entity_name: str
    field_name: str
    original_value: Any
    corrected_value: Any
    reason: str
    source: str  # "user_feedback"
    confidence: float
    status: CorrectionStatus
    verified_by: Optional[str] = None
    applied_at: Optional[datetime] = None


class FeedbackRequest(BaseModel):
    """API model for feedback submission."""
    session_id: str
    response_id: str
    feedback_type: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    original_text: Optional[str] = None
    corrected_text: Optional[str] = None
    comment: Optional[str] = None
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None


class FeedbackResponse(BaseModel):
    """API response for feedback submission."""
    feedback_id: str
    status: str
    message: str
    correction_applied: bool = False
    learning_updated: bool = False


class FeedbackStore:
    """
    Persistent storage for feedback and corrections.
    Enables learning from user interactions.
    """
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/feedback")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.storage_dir / "feedback.json"
        self.corrections_file = self.storage_dir / "corrections.json"
        self.learnings_file = self.storage_dir / "learnings.json"
        
        # In-memory caches
        self._feedback: Dict[str, UserFeedback] = {}
        self._corrections: Dict[str, KnowledgeCorrection] = {}
        self._learnings: Dict[str, Dict] = {}  # entity -> learned facts
        
        self._load()
    
    def _load(self):
        """Load from persistent storage."""
        try:
            if self.feedback_file.exists():
                data = json.loads(self.feedback_file.read_text(encoding='utf-8'))
                for item in data:
                    fb = UserFeedback(**item)
                    self._feedback[fb.feedback_id] = fb
        except Exception:
            pass
        
        try:
            if self.corrections_file.exists():
                data = json.loads(self.corrections_file.read_text(encoding='utf-8'))
                for item in data:
                    corr = KnowledgeCorrection(**item)
                    self._corrections[corr.correction_id] = corr
        except Exception:
            pass
        
        try:
            if self.learnings_file.exists():
                self._learnings = json.loads(self.learnings_file.read_text(encoding='utf-8'))
        except Exception:
            self._learnings = {}
    
    def _save(self):
        """Persist to storage."""
        try:
            feedback_data = [
                {**fb.__dict__, 'timestamp': fb.timestamp.isoformat()}
                for fb in self._feedback.values()
            ]
            self.feedback_file.write_text(
                json.dumps(feedback_data, indent=2, default=str),
                encoding='utf-8'
            )
            
            corrections_data = [
                {**c.__dict__, 'applied_at': c.applied_at.isoformat() if c.applied_at else None}
                for c in self._corrections.values()
            ]
            self.corrections_file.write_text(
                json.dumps(corrections_data, indent=2, default=str),
                encoding='utf-8'
            )
            
            self.learnings_file.write_text(
                json.dumps(self._learnings, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def add_feedback(self, feedback: UserFeedback) -> str:
        """Add new feedback."""
        self._feedback[feedback.feedback_id] = feedback
        self._save()
        return feedback.feedback_id
    
    def add_correction(self, correction: KnowledgeCorrection) -> str:
        """Add new correction."""
        self._corrections[correction.correction_id] = correction
        self._save()
        return correction.correction_id
    
    def get_feedback(self, feedback_id: str) -> Optional[UserFeedback]:
        """Get feedback by ID."""
        return self._feedback.get(feedback_id)
    
    def get_corrections_for_entity(self, entity_type: str, entity_name: str) -> List[KnowledgeCorrection]:
        """Get all corrections for an entity."""
        return [
            c for c in self._corrections.values()
            if c.entity_type == entity_type and c.entity_name.upper() == entity_name.upper()
            and c.status == CorrectionStatus.APPLIED
        ]
    
    def get_learnings(self, entity_name: str) -> Dict:
        """Get learned facts for an entity."""
        return self._learnings.get(entity_name.upper(), {})
    
    def add_learning(self, entity_name: str, key: str, value: Any, source: str = "user"):
        """Add learned fact for an entity."""
        entity_key = entity_name.upper()
        if entity_key not in self._learnings:
            self._learnings[entity_key] = {}
        
        self._learnings[entity_key][key] = {
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        self._save()
    
    def get_all_learnings(self) -> Dict:
        """Get all learned facts."""
        return self._learnings
    
    def get_session_feedback(self, session_id: str) -> List[UserFeedback]:
        """Get all feedback for a session."""
        return [fb for fb in self._feedback.values() if fb.session_id == session_id]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        ratings = [fb.rating for fb in self._feedback.values() if fb.rating]
        corrections = [c for c in self._corrections.values() if c.status == CorrectionStatus.APPLIED]
        
        return {
            "total_feedback": len(self._feedback),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "total_corrections": len(corrections),
            "total_learnings": sum(len(v) for v in self._learnings.values())
        }


class InteractiveLearningEngine:
    """
    Engine for interactive learning from user feedback.
    
    Features:
    - Process user corrections
    - Update knowledge base
    - Learn from patterns
    - Improve response accuracy
    """
    
    def __init__(self, feedback_store: FeedbackStore, knowledge_graph=None):
        self.store = feedback_store
        self.kg = knowledge_graph
    
    def process_feedback(self, request: FeedbackRequest, original_query: str = "") -> FeedbackResponse:
        """
        Process user feedback and apply learnings.
        
        Returns response with status of learning application.
        """
        # Generate feedback ID
        feedback_id = hashlib.md5(
            f"{request.session_id}:{request.response_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create feedback record
        feedback = UserFeedback(
            feedback_id=feedback_id,
            session_id=request.session_id,
            query=original_query,
            response_id=request.response_id,
            feedback_type=FeedbackType(request.feedback_type),
            rating=request.rating,
            original_text=request.original_text,
            corrected_text=request.corrected_text,
            comment=request.comment,
            entity_type=request.entity_type,
            entity_name=request.entity_name
        )
        
        self.store.add_feedback(feedback)
        
        correction_applied = False
        learning_updated = False
        message = "Feedback recorded. Thank you!"
        
        # Process based on feedback type
        if feedback.feedback_type == FeedbackType.CORRECTION:
            correction_applied = self._process_correction(feedback)
            if correction_applied:
                message = "Thank you! Your correction has been applied to improve future responses."
                learning_updated = True
        
        elif feedback.feedback_type == FeedbackType.ADDITION:
            learning_updated = self._process_addition(feedback)
            if learning_updated:
                message = "Thank you! New information has been added to the knowledge base."
        
        elif feedback.feedback_type == FeedbackType.CONFIRMATION:
            self._boost_confidence(feedback)
            message = "Thank you for confirming! This helps improve accuracy."
        
        elif feedback.feedback_type == FeedbackType.DISAGREEMENT:
            self._log_disagreement(feedback)
            message = "Your disagreement has been noted for review."
        
        elif feedback.feedback_type == FeedbackType.RATING:
            if feedback.rating and feedback.rating <= 2:
                message = "We're sorry the response wasn't helpful. Your feedback helps us improve."
            elif feedback.rating and feedback.rating >= 4:
                message = "Thank you for the positive feedback!"
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message=message,
            correction_applied=correction_applied,
            learning_updated=learning_updated
        )
    
    def _process_correction(self, feedback: UserFeedback) -> bool:
        """Process a factual correction."""
        if not feedback.entity_name or not feedback.corrected_text:
            return False
        
        # Create correction record
        correction_id = hashlib.md5(
            f"{feedback.feedback_id}:{feedback.entity_name}".encode()
        ).hexdigest()[:12]
        
        correction = KnowledgeCorrection(
            correction_id=correction_id,
            feedback_id=feedback.feedback_id,
            entity_type=feedback.entity_type or "general",
            entity_name=feedback.entity_name,
            field_name="user_correction",
            original_value=feedback.original_text,
            corrected_value=feedback.corrected_text,
            reason=feedback.comment or "User correction",
            source="user_feedback",
            confidence=0.9,  # User corrections have high confidence
            status=CorrectionStatus.APPLIED
        )
        
        self.store.add_correction(correction)
        
        # Add to learnings
        self.store.add_learning(
            entity_name=feedback.entity_name,
            key=f"correction_{correction_id}",
            value={
                "original": feedback.original_text,
                "corrected": feedback.corrected_text,
                "reason": feedback.comment
            },
            source="user_correction"
        )
        
        # If knowledge graph is available, try to update it
        if self.kg and hasattr(self.kg, 'add_user_correction'):
            try:
                self.kg.add_user_correction(
                    entity_name=feedback.entity_name,
                    original=feedback.original_text,
                    corrected=feedback.corrected_text
                )
            except Exception:
                pass
        
        return True
    
    def _process_addition(self, feedback: UserFeedback) -> bool:
        """Process new information addition."""
        if not feedback.corrected_text:
            return False
        
        entity_name = feedback.entity_name or "general"
        
        # Add to learnings
        self.store.add_learning(
            entity_name=entity_name,
            key=f"addition_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            value={
                "new_info": feedback.corrected_text,
                "context": feedback.comment
            },
            source="user_addition"
        )
        
        return True
    
    def _boost_confidence(self, feedback: UserFeedback):
        """Boost confidence for confirmed information."""
        if feedback.entity_name:
            self.store.add_learning(
                entity_name=feedback.entity_name,
                key="confidence_boost",
                value={
                    "confirmed_at": datetime.now().isoformat(),
                    "context": feedback.original_text
                },
                source="user_confirmation"
            )
    
    def _log_disagreement(self, feedback: UserFeedback):
        """Log user disagreement for review."""
        entity_name = feedback.entity_name or "general"
        self.store.add_learning(
            entity_name=entity_name,
            key=f"disagreement_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            value={
                "disputed_text": feedback.original_text,
                "user_view": feedback.corrected_text or feedback.comment,
                "requires_review": True
            },
            source="user_disagreement"
        )
    
    def get_learnings_for_query(self, query: str, entities: List[str] = None) -> Dict:
        """Get relevant learnings for a query."""
        learnings = {}
        
        # Get learnings for mentioned entities
        if entities:
            for entity in entities:
                entity_learnings = self.store.get_learnings(entity)
                if entity_learnings:
                    learnings[entity] = entity_learnings
        
        # Get general learnings
        general = self.store.get_learnings("general")
        if general:
            learnings["general"] = general
        
        return learnings
    
    def apply_learnings_to_response(self, response: str, entities: List[str] = None) -> Tuple[str, List[str]]:
        """
        Apply learned corrections to a response.
        
        Returns:
            Tuple of (corrected_response, list of applied corrections)
        """
        applied = []
        corrected_response = response
        
        if not entities:
            return response, []
        
        for entity in entities:
            corrections = self.store.get_corrections_for_entity("constituency", entity)
            for corr in corrections:
                if corr.original_value and corr.original_value in corrected_response:
                    corrected_response = corrected_response.replace(
                        corr.original_value,
                        corr.corrected_value
                    )
                    applied.append(f"Applied correction for {entity}: {corr.corrected_value}")
        
        return corrected_response, applied


class QuickResponseManager:
    """
    Manager for quick, interactive responses.
    
    Features:
    - Streaming responses for perceived speed
    - Cached common queries
    - Instant feedback acknowledgment
    - Progressive detail loading
    """
    
    def __init__(self, rag=None):
        self.rag = rag
        self._cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
    
    def _get_cache_key(self, query: str, context: Dict = None) -> str:
        """Generate cache key for a query."""
        context_str = json.dumps(context or {}, sort_keys=True)
        return hashlib.md5(f"{query}:{context_str}".encode()).hexdigest()
    
    def get_cached_response(self, query: str, context: Dict = None) -> Optional[str]:
        """Get cached response if available and fresh."""
        key = self._get_cache_key(query, context)
        if key in self._cache:
            response, timestamp = self._cache[key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return response
            else:
                del self._cache[key]
        return None
    
    def cache_response(self, query: str, response: str, context: Dict = None):
        """Cache a response."""
        key = self._get_cache_key(query, context)
        self._cache[key] = (response, datetime.now())
    
    async def get_quick_summary(self, query: str) -> str:
        """Get a quick summary for immediate response."""
        query_lower = query.lower()
        
        # Common query patterns with quick responses
        quick_responses = {
            "how many seats": "West Bengal has 294 Assembly constituencies. Let me get the detailed breakdown...",
            "who won": "Checking the electoral data...",
            "bjp seats": "Looking up BJP's seat count...",
            "tmc seats": "Looking up TMC's seat count...",
            "predict": "Analyzing 2026 predictions...",
            "strategy": "Developing strategic recommendations...",
            "vulnerable": "Identifying vulnerable seats...",
        }
        
        for pattern, response in quick_responses.items():
            if pattern in query_lower:
                return response
        
        return "Analyzing your query..."
    
    def get_progressive_response(self, full_response: str, chunk_size: int = 100) -> List[str]:
        """
        Split response into chunks for progressive loading.
        
        Useful for streaming responses to frontend.
        """
        chunks = []
        lines = full_response.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks


# Singleton instances
_feedback_store: Optional[FeedbackStore] = None
_learning_engine: Optional[InteractiveLearningEngine] = None
_quick_response_manager: Optional[QuickResponseManager] = None


def get_feedback_store() -> FeedbackStore:
    """Get or create feedback store singleton."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store


def get_learning_engine(kg=None) -> InteractiveLearningEngine:
    """Get or create learning engine singleton."""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = InteractiveLearningEngine(get_feedback_store(), kg)
    return _learning_engine


def get_quick_response_manager(rag=None) -> QuickResponseManager:
    """Get or create quick response manager singleton."""
    global _quick_response_manager
    if _quick_response_manager is None:
        _quick_response_manager = QuickResponseManager(rag)
    return _quick_response_manager

