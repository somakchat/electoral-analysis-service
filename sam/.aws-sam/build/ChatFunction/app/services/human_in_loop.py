"""
Human-in-the-Loop (HITL) System for Political Strategy Maker.

This module enables interactive conversations between users and agents:
- Clarification requests for ambiguous queries
- Suggested follow-up actions
- Confirmation workflows for critical decisions
- Interactive refinement of outputs
- Multi-turn conversation context
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime
import json
import re


class InteractionType(str, Enum):
    """Types of human-in-the-loop interactions."""
    CLARIFICATION = "clarification"  # Need more info from user
    CONFIRMATION = "confirmation"    # Confirm before proceeding
    CHOICE = "choice"                # Choose from options
    REFINEMENT = "refinement"        # Refine/modify output
    FOLLOW_UP = "follow_up"          # Suggested follow-ups
    FEEDBACK = "feedback"            # Request feedback
    INFO = "info"                    # Informational (no action needed)


class InteractionPriority(str, Enum):
    """Priority levels for interactions."""
    BLOCKING = "blocking"      # Must respond before continuing
    SUGGESTED = "suggested"    # Optional but recommended
    OPTIONAL = "optional"      # Nice to have


@dataclass
class InteractionOption:
    """An option in a HITL interaction."""
    id: str
    label: str
    description: str = ""
    action: str = ""  # Action to trigger if selected
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HITLInteraction:
    """A human-in-the-loop interaction request."""
    interaction_id: str
    interaction_type: InteractionType
    priority: InteractionPriority
    message: str
    options: List[InteractionOption] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    default_option: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "type": self.interaction_type.value,
            "priority": self.priority.value,
            "message": self.message,
            "options": [
                {
                    "id": opt.id,
                    "label": opt.label,
                    "description": opt.description,
                    "action": opt.action,
                    "metadata": opt.metadata
                }
                for opt in self.options
            ],
            "context": self.context,
            "default_option": self.default_option,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    interaction: Optional[HITLInteraction] = None


@dataclass
class ConversationContext:
    """Multi-turn conversation context."""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)  # Extracted entities
    pending_interaction: Optional[HITLInteraction] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, role: str, content: str, **kwargs) -> None:
        self.turns.append(ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=kwargs
        ))
    
    def get_recent_context(self, n_turns: int = 5) -> str:
        """Get recent conversation as context string."""
        recent = self.turns[-n_turns:] if len(self.turns) > n_turns else self.turns
        return "\n".join([f"{t.role}: {t.content}" for t in recent])
    
    def update_entities(self, entities: Dict[str, str]) -> None:
        """Update extracted entities from conversation."""
        self.entities.update(entities)


class ClarificationDetector:
    """Detects when clarification is needed from the user."""
    
    # Ambiguous query patterns
    AMBIGUOUS_PATTERNS = [
        (r'\b(it|this|that|these|those|they|them)\b(?!\s+is|\s+are|\s+was|\s+were)', 'pronoun_reference'),
        (r'\?.*\?', 'multiple_questions'),
        (r'\b(somewhere|somehow|something|anyone|anything)\b', 'vague_reference'),
        (r'^(what|which|who|where|how)\s+(about|is|are)\s*$', 'incomplete_question'),
    ]
    
    # Topics that may need clarification
    CLARIFICATION_TOPICS = {
        'party': ['bjp', 'tmc', 'congress', 'cpm', 'left', 'trinamool'],
        'region': ['district', 'constituency', 'area', 'region', 'zone'],
        'timeframe': ['when', 'timeline', 'deadline', 'schedule'],
        'scope': ['all', 'some', 'few', 'many', 'specific']
    }
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
    
    def analyze_query(self, query: str, context: ConversationContext) -> Optional[HITLInteraction]:
        """Analyze query and determine if clarification is needed."""
        query_lower = query.lower()
        
        # Check for ambiguous patterns
        for pattern, pattern_type in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, query_lower):
                return self._create_clarification(query, pattern_type, context)
        
        # Check if query lacks specificity
        needs_clarification = self._check_specificity(query_lower, context)
        if needs_clarification:
            return needs_clarification
        
        # Check for multiple possible interpretations
        interpretations = self._detect_ambiguity(query_lower)
        if len(interpretations) > 1:
            return self._create_interpretation_choice(query, interpretations)
        
        return None
    
    def _check_specificity(self, query: str, context: ConversationContext) -> Optional[HITLInteraction]:
        """Check if query needs more specific information."""
        
        # Generic strategy query without party
        if any(w in query for w in ['strategy', 'win', 'improve', 'campaign']) and \
           not any(p in query for p in ['bjp', 'tmc', 'congress', 'trinamool']):
            if 'party' not in context.entities:
                return HITLInteraction(
                    interaction_id=f"clarify_{datetime.now().timestamp()}",
                    interaction_type=InteractionType.CLARIFICATION,
                    priority=InteractionPriority.SUGGESTED,
                    message="Which party's strategy would you like me to analyze?",
                    options=[
                        InteractionOption("bjp", "BJP", "Bharatiya Janata Party strategy"),
                        InteractionOption("tmc", "TMC", "Trinamool Congress strategy"),
                        InteractionOption("both", "Compare Both", "Compare BJP and TMC strategies"),
                    ],
                    context={"original_query": query, "missing": "party"}
                )
        
        # District-level query without specific district
        if 'district' in query and not any(d.lower() in query for d in self._get_districts()):
            return HITLInteraction(
                interaction_id=f"clarify_{datetime.now().timestamp()}",
                interaction_type=InteractionType.CLARIFICATION,
                priority=InteractionPriority.SUGGESTED,
                message="Which district would you like me to focus on?",
                options=self._get_district_options()[:6],
                context={"original_query": query, "missing": "district"}
            )

        # Constituency question without specifying AC vs PC
        # Example: "what is the constituency of X" could mean:
        # - Assembly constituency (AC/MLA seat)
        # - Parliamentary constituency (PC/MP seat)
        if "constituency" in query and any(k in query for k in ["which", "what", "name", "constituency of"]):
            if not any(k in query for k in ["assembly", "ac", "mla", "parliamentary", "pc", "mp", "lok sabha"]):
                return HITLInteraction(
                    interaction_id=f"clarify_{datetime.now().timestamp()}",
                    interaction_type=InteractionType.CLARIFICATION,
                    priority=InteractionPriority.BLOCKING,
                    message="Do you mean **Assembly constituency (AC/MLA)** or **Parliamentary constituency (PC/MP)**?",
                    options=[
                        InteractionOption("ac", "Assembly (AC/MLA)", "Return the assembly constituency / MLA seat"),
                        InteractionOption("pc", "Parliamentary (PC/MP)", "Return the parliamentary constituency / MP seat"),
                        InteractionOption("both", "Both", "Return both AC and PC if evidence mentions them"),
                    ],
                    context={"original_query": query, "missing": "constituency_type"}
                )
        
        return None
    
    def _detect_ambiguity(self, query: str) -> List[Dict]:
        """Detect multiple possible interpretations of a query."""
        interpretations = []
        
        # "seats" could mean many things
        if 'seats' in query and 'swing' not in query and 'safe' not in query:
            interpretations = [
                {"id": "all_seats", "label": "All seats overview"},
                {"id": "swing_seats", "label": "Swing/competitive seats"},
                {"id": "safe_seats", "label": "Safe seats"},
                {"id": "vulnerable_seats", "label": "Vulnerable seats"}
            ]
        
        return interpretations
    
    def _create_clarification(self, query: str, pattern_type: str, context: ConversationContext) -> HITLInteraction:
        """Create a clarification request based on pattern type."""
        messages = {
            'pronoun_reference': "Could you please clarify what you're referring to?",
            'multiple_questions': "I noticed multiple questions. Which would you like me to answer first?",
            'vague_reference': "Could you be more specific about what you're looking for?",
            'incomplete_question': "Your question seems incomplete. Could you provide more details?"
        }
        
        return HITLInteraction(
            interaction_id=f"clarify_{datetime.now().timestamp()}",
            interaction_type=InteractionType.CLARIFICATION,
            priority=InteractionPriority.BLOCKING,
            message=messages.get(pattern_type, "Could you please clarify your question?"),
            context={"original_query": query, "pattern": pattern_type}
        )
    
    def _create_interpretation_choice(self, query: str, interpretations: List[Dict]) -> HITLInteraction:
        """Create a choice interaction for ambiguous queries."""
        return HITLInteraction(
            interaction_id=f"choice_{datetime.now().timestamp()}",
            interaction_type=InteractionType.CHOICE,
            priority=InteractionPriority.SUGGESTED,
            message="What type of seat analysis are you looking for?",
            options=[
                InteractionOption(i["id"], i["label"], action=i["id"])
                for i in interpretations
            ],
            context={"original_query": query}
        )
    
    def _get_districts(self) -> List[str]:
        """Get list of districts from knowledge graph."""
        if self.kg and hasattr(self.kg, 'constituency_profiles'):
            return list(set(c.district for c in self.kg.constituency_profiles.values()))
        return []
    
    def _get_district_options(self) -> List[InteractionOption]:
        """Get district options for selection."""
        districts = self._get_districts()
        return [
            InteractionOption(d.lower().replace(' ', '_'), d, f"Analyze {d} district")
            for d in sorted(districts)[:10]
        ]


class FollowUpGenerator:
    """Generates intelligent follow-up suggestions based on responses."""
    
    # Follow-up templates by response type
    FOLLOW_UP_TEMPLATES = {
        'party_analysis': [
            ("victory_path", "How can {party} reach 148 seats?", "Calculate victory path"),
            ("vulnerable", "Which {party} seats are most vulnerable?", "Show vulnerabilities"),
            ("resource", "How should {party} allocate resources?", "Resource strategy"),
            ("compare", "How does {party} compare to {opponent}?", "Compare parties"),
        ],
        'constituency_analysis': [
            ("similar", "Which constituencies are similar to {constituency}?", "Find similar"),
            ("history", "What is the electoral history of {constituency}?", "Show history"),
            ("strategy", "What's the winning strategy for {constituency}?", "Campaign strategy"),
        ],
        'district_analysis': [
            ("constituencies", "List all constituencies in {district}", "Show constituencies"),
            ("swing", "Which seats are swing in {district}?", "Swing analysis"),
            ("party_strength", "Compare party strength in {district}", "Party comparison"),
        ],
        'strategic_recommendations': [
            ("deep_dive", "Explain more about {topic}", "Deep dive"),
            ("alternatives", "What are alternative strategies?", "Show alternatives"),
            ("risks", "What are the risks of this strategy?", "Risk analysis"),
            ("timeline", "What's the implementation timeline?", "Show timeline"),
            ("budget", "What budget is needed?", "Budget estimation"),
        ],
        'prediction': [
            ("assumptions", "What assumptions is this based on?", "Show assumptions"),
            ("scenarios", "What if scenarios change?", "Scenario analysis"),
            ("confidence", "How confident is this prediction?", "Confidence details"),
        ]
    }
    
    def generate_follow_ups(
        self, 
        response_type: str,
        context: Dict[str, Any],
        max_suggestions: int = 4
    ) -> List[InteractionOption]:
        """Generate follow-up suggestions based on response type and context."""
        templates = self.FOLLOW_UP_TEMPLATES.get(response_type, [])
        
        follow_ups = []
        for template_id, template_text, description in templates[:max_suggestions]:
            # Fill in template with context
            try:
                filled_text = template_text.format(**context)
                follow_ups.append(InteractionOption(
                    id=f"followup_{template_id}",
                    label=filled_text,
                    description=description,
                    action=template_id,
                    metadata=context
                ))
            except KeyError:
                # Skip if required context is missing
                continue
        
        return follow_ups
    
    def detect_response_type(self, answer: str, query: str) -> str:
        """Detect the type of response for generating appropriate follow-ups."""
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        if 'strategic decision' in answer_lower or 'action point' in answer_lower:
            return 'strategic_recommendations'
        elif 'constituency analysis' in answer_lower or 'constituency profile' in answer_lower:
            return 'constituency_analysis'
        elif 'district analysis' in answer_lower or 'district' in query_lower:
            return 'district_analysis'
        elif 'prediction' in answer_lower or 'projected' in answer_lower:
            return 'prediction'
        elif any(p in query_lower for p in ['bjp', 'tmc', 'congress']):
            return 'party_analysis'
        else:
            return 'general'


class ConfirmationManager:
    """Manages confirmation workflows for critical decisions."""
    
    # Actions that require confirmation
    CONFIRMATION_REQUIRED = [
        'resource_allocation',
        'candidate_recommendation',
        'strategy_change',
        'priority_shift'
    ]
    
    def check_needs_confirmation(
        self, 
        action_type: str, 
        impact_level: str = "medium"
    ) -> bool:
        """Check if an action needs user confirmation."""
        if action_type in self.CONFIRMATION_REQUIRED:
            return True
        if impact_level == "high":
            return True
        return False
    
    def create_confirmation_request(
        self,
        action: str,
        details: Dict[str, Any],
        consequences: List[str]
    ) -> HITLInteraction:
        """Create a confirmation request for a critical action."""
        return HITLInteraction(
            interaction_id=f"confirm_{datetime.now().timestamp()}",
            interaction_type=InteractionType.CONFIRMATION,
            priority=InteractionPriority.BLOCKING,
            message=f"Please confirm this action: {action}",
            options=[
                InteractionOption("confirm", "Yes, proceed", "Confirm and execute", action="confirm"),
                InteractionOption("modify", "Modify", "Modify before proceeding", action="modify"),
                InteractionOption("cancel", "Cancel", "Cancel this action", action="cancel"),
            ],
            context={
                "action": action,
                "details": details,
                "consequences": consequences
            }
        )


class RefinementHandler:
    """Handles user refinement requests for outputs."""
    
    REFINEMENT_TYPES = {
        'more_detail': "Provide more detailed analysis",
        'less_detail': "Summarize more concisely",
        'different_focus': "Focus on a different aspect",
        'specific_area': "Zoom in on specific area",
        'add_data': "Include more data/numbers",
        'simplify': "Simplify the explanation"
    }
    
    def create_refinement_options(self, response: str) -> HITLInteraction:
        """Create refinement options for a response."""
        options = [
            InteractionOption(
                id=ref_id,
                label=ref_label,
                description=f"Modify response to {ref_label.lower()}",
                action=ref_id
            )
            for ref_id, ref_label in self.REFINEMENT_TYPES.items()
        ]
        
        return HITLInteraction(
            interaction_id=f"refine_{datetime.now().timestamp()}",
            interaction_type=InteractionType.REFINEMENT,
            priority=InteractionPriority.OPTIONAL,
            message="Would you like me to refine this response?",
            options=options,
            context={"original_response": response[:500]}
        )
    
    def apply_refinement(self, original: str, refinement_type: str, llm) -> str:
        """Apply refinement to the original response."""
        refinement_prompts = {
            'more_detail': "Expand this analysis with more specific details and examples:\n\n",
            'less_detail': "Summarize this analysis more concisely, keeping only key points:\n\n",
            'different_focus': "Reframe this analysis from a different perspective:\n\n",
            'add_data': "Add more numerical data and statistics to this analysis:\n\n",
            'simplify': "Simplify this explanation for a non-expert audience:\n\n"
        }
        
        prompt = refinement_prompts.get(refinement_type, "Refine this response:\n\n")
        
        try:
            response = llm.generate(
                prompt + original,
                system="You are refining a political analysis. Keep the same factual content but adjust the presentation as requested."
            )
            return response.text
        except Exception:
            return original


class HITLOrchestrator:
    """
    Main orchestrator for Human-in-the-Loop interactions.
    
    Integrates all HITL components:
    - Clarification detection
    - Follow-up generation
    - Confirmation management
    - Refinement handling
    - Conversation context
    """
    
    def __init__(self, knowledge_graph, llm=None):
        self.kg = knowledge_graph
        self.llm = llm
        
        # Initialize components
        self.clarification_detector = ClarificationDetector(knowledge_graph)
        self.follow_up_generator = FollowUpGenerator()
        self.confirmation_manager = ConfirmationManager()
        self.refinement_handler = RefinementHandler()
        
        # Conversation contexts by session
        self.contexts: Dict[str, ConversationContext] = {}
    
    def get_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for a session."""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(session_id=session_id)
        return self.contexts[session_id]
    
    def pre_process_query(
        self, 
        query: str, 
        session_id: str
    ) -> Optional[HITLInteraction]:
        """
        Pre-process a query to check if clarification is needed.
        
        Returns an interaction if clarification needed, None otherwise.
        """
        context = self.get_context(session_id)
        context.add_turn("user", query)
        
        # Check if this is a response to a pending interaction
        if context.pending_interaction:
            return self._handle_pending_response(query, context)
        
        # Check if clarification is needed
        clarification = self.clarification_detector.analyze_query(query, context)
        if clarification:
            context.pending_interaction = clarification
            return clarification
        
        return None

    def pre_process_and_rewrite(
        self,
        query: str,
        session_id: str
    ):
        """
        Pre-process a query and return (clarification_interaction, effective_query).
        This enables true multi-turn interactions:
        - If user answers a clarification (by clicking an option), we resume the ORIGINAL query
          with that choice applied.
        """
        context = self.get_context(session_id)
        context.add_turn("user", query)

        # If this is a response to a pending interaction, resolve and rewrite
        if context.pending_interaction:
            effective = self._handle_pending_response_with_rewrite(query, context)
            if effective:
                return None, effective
            # If we couldn't resolve, ask again
            return context.pending_interaction, context.pending_interaction.context.get("original_query", query)

        clarification = self.clarification_detector.analyze_query(query, context)
        if clarification:
            context.pending_interaction = clarification
            return clarification, query

        return None, query
    
    def _handle_pending_response(
        self, 
        response: str, 
        context: ConversationContext
    ) -> Optional[HITLInteraction]:
        """Handle a response to a pending interaction."""
        pending = context.pending_interaction
        context.pending_interaction = None
        
        if pending.interaction_type == InteractionType.CLARIFICATION:
            # Extract entity from response
            original_query = pending.context.get('original_query', '')
            missing = pending.context.get('missing', '')
            
            # Try to match response to options
            for opt in pending.options:
                if opt.id in response.lower() or opt.label.lower() in response.lower():
                    context.update_entities({missing: opt.id})
                    return None  # Proceed with enhanced query
            
            # Use raw response as entity
            context.update_entities({missing: response})
        
        return None

    def _handle_pending_response_with_rewrite(
        self,
        response: str,
        context: ConversationContext
    ) -> Optional[str]:
        """
        Handle a pending clarification response and return the enhanced query to execute.
        """
        pending = context.pending_interaction
        if not pending:
            return None

        # Clear pending; re-set only if we fail to resolve.
        context.pending_interaction = None

        if pending.interaction_type != InteractionType.CLARIFICATION:
            return None

        original_query = (pending.context or {}).get("original_query", "")
        missing = (pending.context or {}).get("missing", "")
        if not original_query or not missing:
            return None

        resp_lower = (response or "").strip().lower()
        if not resp_lower:
            context.pending_interaction = pending
            return None

        selected_opt: Optional[InteractionOption] = None
        for opt in pending.options:
            if opt.id and opt.id.lower() == resp_lower:
                selected_opt = opt
                break
            if opt.id and opt.id.lower() in resp_lower:
                selected_opt = opt
                break
            if opt.label and opt.label.lower() in resp_lower:
                selected_opt = opt
                break

        chosen_value = (selected_opt.id if selected_opt else resp_lower).strip()
        if not chosen_value:
            context.pending_interaction = pending
            return None

        context.update_entities({missing: chosen_value})

        # Apply missing value to original query
        if missing == "constituency_type":
            if chosen_value in {"ac", "assembly"}:
                return f"{original_query} (Assembly constituency / AC / MLA)"
            if chosen_value in {"pc", "parliamentary"}:
                return f"{original_query} (Parliamentary constituency / PC / MP)"
            if chosen_value in {"both", "all"}:
                return f"{original_query} (Provide both Assembly (AC/MLA) and Parliamentary (PC/MP) constituencies if available)"
            return f"{original_query} ({chosen_value})"

        if chosen_value not in original_query.lower():
            return f"{original_query} ({missing}: {chosen_value})"
        return original_query
    
    def post_process_response(
        self, 
        query: str,
        response: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Post-process a response to add interactive elements.
        
        Adds:
        - Follow-up suggestions
        - Refinement options
        - Confirmation requests if needed
        """
        context = self.get_context(session_id)
        answer = response.get('answer', '')
        
        context.add_turn("assistant", answer[:500])
        
        # Detect response type
        response_type = self.follow_up_generator.detect_response_type(answer, query)
        
        # Build context for follow-ups
        follow_up_context = self._extract_context_for_followups(query, answer, context)
        
        # Generate follow-up suggestions
        follow_ups = self.follow_up_generator.generate_follow_ups(
            response_type, 
            follow_up_context,
            max_suggestions=4
        )
        
        # Create interactions
        interactions = []
        
        # Add follow-up interaction
        if follow_ups:
            interactions.append(HITLInteraction(
                interaction_id=f"followup_{datetime.now().timestamp()}",
                interaction_type=InteractionType.FOLLOW_UP,
                priority=InteractionPriority.SUGGESTED,
                message="Would you like to explore further?",
                options=follow_ups
            ).to_dict())
        
        # Check if refinement should be offered
        if len(answer) > 500:  # Long responses can be refined
            refinement = self.refinement_handler.create_refinement_options(answer)
            interactions.append(refinement.to_dict())
        
        # Add interactions to response
        response['interactions'] = interactions
        response['conversation_context'] = {
            'session_id': session_id,
            'turn_count': len(context.turns),
            'entities': context.entities,
            'response_type': response_type
        }
        
        return response
    
    def _extract_context_for_followups(
        self, 
        query: str, 
        answer: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Extract context variables for follow-up generation."""
        ctx = dict(context.entities)
        
        # Extract party
        for party in ['BJP', 'TMC', 'Congress', 'CPM']:
            if party.lower() in query.lower() or party.lower() in answer.lower():
                ctx['party'] = party
                ctx['opponent'] = 'TMC' if party == 'BJP' else 'BJP'
                break
        
        # Extract constituency
        if self.kg and hasattr(self.kg, 'constituency_profiles'):
            for name in self.kg.constituency_profiles.keys():
                if name.lower() in query.lower():
                    ctx['constituency'] = name
                    break
        
        # Extract district
        if self.kg and hasattr(self.kg, 'constituency_profiles'):
            districts = set(c.district for c in self.kg.constituency_profiles.values())
            for district in districts:
                if district.lower() in query.lower():
                    ctx['district'] = district
                    break
        
        # Extract topic from query
        topics = ['swing seats', 'vulnerable seats', 'resource allocation', 'campaign strategy']
        for topic in topics:
            if topic in query.lower():
                ctx['topic'] = topic
                break
        
        return ctx
    
    def handle_interaction_response(
        self, 
        interaction_id: str,
        selected_option: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle user's response to an interaction."""
        context = self.get_context(session_id)
        
        # Find the option details
        option_details = {"id": selected_option}
        
        # Build new query based on interaction type
        if selected_option.startswith("followup_"):
            # Generate follow-up query
            action = selected_option.replace("followup_", "")
            return {
                "action": "new_query",
                "query": self._generate_followup_query(action, context)
            }
        elif selected_option in ["more_detail", "less_detail", "simplify"]:
            # Refinement action
            return {
                "action": "refine",
                "refinement_type": selected_option
            }
        elif selected_option in ["confirm", "cancel", "modify"]:
            # Confirmation response
            return {
                "action": selected_option,
                "proceed": selected_option == "confirm"
            }
        else:
            # Option selection (party, district, etc.)
            return {
                "action": "continue",
                "selected": selected_option,
                "entities": {selected_option.split("_")[0]: selected_option}
            }
    
    def _generate_followup_query(self, action: str, context: ConversationContext) -> str:
        """Generate a follow-up query based on action and context."""
        templates = {
            "victory_path": "How can {party} reach 148 seats to form government?",
            "vulnerable": "Which {party} seats are most vulnerable to losing?",
            "resource": "How should {party} allocate resources across constituencies?",
            "compare": "Compare {party} and {opponent} strategies",
            "similar": "Which constituencies are similar to {constituency}?",
            "history": "What is the electoral history of {constituency}?",
            "deep_dive": "Provide more details about {topic}",
            "risks": "What are the risks and challenges?",
            "timeline": "What is the recommended timeline for implementation?"
        }
        
        template = templates.get(action, "Tell me more about this")
        
        try:
            return template.format(**context.entities)
        except KeyError:
            return template.replace("{", "").replace("}", "")


# Factory function
def create_hitl_orchestrator(knowledge_graph, llm=None) -> HITLOrchestrator:
    """Create an HITL orchestrator instance."""
    return HITLOrchestrator(knowledge_graph, llm)


