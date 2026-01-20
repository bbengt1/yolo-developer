"""Type definitions for cross-agent correlation (Story 11.5).

This module provides the data types used for correlating decisions across agents:

- CorrelationType: Literal type for correlation mechanisms
- DecisionChain: Groups related decisions for navigation
- CausalRelation: Explicit cause-effect relationship between decisions
- AgentTransition: Records workflow handoffs between agents
- TimelineEntry: Single entry in a timeline view (Task 5)

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.correlation_types import (
    ...     DecisionChain,
    ...     CausalRelation,
    ...     AgentTransition,
    ... )
    >>>
    >>> chain = DecisionChain(
    ...     id="chain-001",
    ...     decisions=("dec-001", "dec-002"),
    ...     chain_type="session",
    ...     created_at="2026-01-18T12:00:00Z",
    ... )
    >>> chain.to_dict()
    {'id': 'chain-001', ...}

References:
    - FR85: System can correlate decisions across agent boundaries
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from yolo_developer.audit.types import Decision

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

CorrelationType = Literal["causal", "temporal", "session", "artifact"]
"""Type of correlation mechanism.

Values:
    causal: Direct cause-effect relationships (A caused B)
    temporal: Time-based correlation (same time window)
    session: Same session correlation (same session_id)
    artifact: Related to same artifact (via trace_links)
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

VALID_CORRELATION_TYPES: frozenset[str] = frozenset(
    {
        "causal",
        "temporal",
        "session",
        "artifact",
    }
)
"""Set of valid correlation type values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class DecisionChain:
    """Groups related decisions for navigation.

    Decision chains represent a collection of correlated decisions that
    can be navigated together. A decision can belong to multiple chains.

    Attributes:
        id: Unique identifier for the chain
        decisions: Tuple of decision IDs in this chain
        chain_type: Type of correlation (causal, temporal, session, artifact)
        created_at: ISO 8601 timestamp when chain was created
        metadata: Additional key-value data (optional)

    Example:
        >>> chain = DecisionChain(
        ...     id="chain-001",
        ...     decisions=("dec-001", "dec-002", "dec-003"),
        ...     chain_type="session",
        ...     created_at="2026-01-18T12:00:00Z",
        ... )
        >>> chain.chain_type
        'session'
    """

    id: str
    decisions: tuple[str, ...]
    chain_type: CorrelationType
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chain data and log warnings for issues."""
        if not self.id:
            _logger.warning("DecisionChain id is empty")
        if not self.decisions:
            _logger.warning("DecisionChain decisions is empty for id=%s", self.id)
        if self.chain_type not in VALID_CORRELATION_TYPES:
            _logger.warning(
                "DecisionChain chain_type='%s' is not a valid correlation type for id=%s",
                self.chain_type,
                self.id,
            )
        if not self.created_at:
            _logger.warning("DecisionChain created_at is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the decision chain.
        """
        return {
            "id": self.id,
            "decisions": list(self.decisions),
            "chain_type": self.chain_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CausalRelation:
    """Explicit cause-effect relationship between two decisions.

    Tracks causal relationships where one decision directly caused or
    influenced another. Uses parent_decision_id from DecisionContext
    for auto-detection.

    Attributes:
        id: Unique identifier for the relation
        cause_decision_id: ID of the decision that caused the effect
        effect_decision_id: ID of the decision that was caused
        relation_type: Description of the relationship (e.g., "derives_from", "triggers")
        evidence: Explanation of why this relationship exists (optional)
        created_at: ISO 8601 timestamp when relation was created

    Example:
        >>> relation = CausalRelation(
        ...     id="rel-001",
        ...     cause_decision_id="dec-001",
        ...     effect_decision_id="dec-002",
        ...     relation_type="derives_from",
        ...     evidence="Parent decision ID reference",
        ...     created_at="2026-01-18T12:00:00Z",
        ... )
        >>> relation.relation_type
        'derives_from'
    """

    id: str
    cause_decision_id: str
    effect_decision_id: str
    relation_type: str
    created_at: str
    evidence: str = ""

    def __post_init__(self) -> None:
        """Validate relation data and log warnings for issues."""
        if not self.id:
            _logger.warning("CausalRelation id is empty")
        if not self.cause_decision_id:
            _logger.warning("CausalRelation cause_decision_id is empty for id=%s", self.id)
        if not self.effect_decision_id:
            _logger.warning("CausalRelation effect_decision_id is empty for id=%s", self.id)
        if not self.relation_type:
            _logger.warning("CausalRelation relation_type is empty for id=%s", self.id)
        if not self.created_at:
            _logger.warning("CausalRelation created_at is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the causal relation.
        """
        return {
            "id": self.id,
            "cause_decision_id": self.cause_decision_id,
            "effect_decision_id": self.effect_decision_id,
            "relation_type": self.relation_type,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AgentTransition:
    """Records workflow handoff between agents.

    Captures when work transitions from one agent to another, essential
    for timeline visualization and workflow reconstruction.

    Attributes:
        id: Unique identifier for the transition
        from_agent: Name of the agent handing off work
        to_agent: Name of the agent receiving work
        decision_id: ID of the decision that triggered the transition
        timestamp: ISO 8601 timestamp when transition occurred
        context: Additional context data (optional)

    Example:
        >>> transition = AgentTransition(
        ...     id="trans-001",
        ...     from_agent="analyst",
        ...     to_agent="pm",
        ...     decision_id="dec-001",
        ...     timestamp="2026-01-18T12:00:00Z",
        ... )
        >>> transition.from_agent
        'analyst'
    """

    id: str
    from_agent: str
    to_agent: str
    decision_id: str
    timestamp: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transition data and log warnings for issues."""
        if not self.id:
            _logger.warning("AgentTransition id is empty")
        if not self.from_agent:
            _logger.warning("AgentTransition from_agent is empty for id=%s", self.id)
        if not self.to_agent:
            _logger.warning("AgentTransition to_agent is empty for id=%s", self.id)
        if not self.decision_id:
            _logger.warning("AgentTransition decision_id is empty for id=%s", self.id)
        if not self.timestamp:
            _logger.warning("AgentTransition timestamp is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the agent transition.
        """
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "context": self.context,
        }


@dataclass(frozen=True)
class TimelineEntry:
    """Single entry in a timeline view (Task 5).

    Represents a decision with its position in a chronological timeline.
    Used by get_timeline_view to provide ordered, sequenced decision data.

    Attributes:
        decision: The Decision object
        sequence_number: Position in timeline (1-indexed)
        timestamp: ISO 8601 timestamp (copied from decision for convenience)
        agent_transition: AgentTransition if this decision triggered a handoff (optional)
        previous_agent: Name of the previous agent if a transition occurred (optional)

    Example:
        >>> entry = TimelineEntry(
        ...     decision=decision,
        ...     sequence_number=1,
        ...     timestamp="2026-01-18T12:00:00Z",
        ...     agent_transition=None,
        ...     previous_agent=None,
        ... )
        >>> entry.sequence_number
        1
    """

    decision: Decision
    sequence_number: int
    timestamp: str
    agent_transition: AgentTransition | None = None
    previous_agent: str | None = None

    def __post_init__(self) -> None:
        """Validate timeline entry data and log warnings for issues."""
        if self.sequence_number < 1:
            _logger.warning("TimelineEntry sequence_number=%d is less than 1", self.sequence_number)
        if not self.timestamp:
            _logger.warning(
                "TimelineEntry timestamp is empty for sequence_number=%d",
                self.sequence_number,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the timeline entry.
        """
        return {
            "decision": self.decision.to_dict(),
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "agent_transition": self.agent_transition.to_dict() if self.agent_transition else None,
            "previous_agent": self.previous_agent,
        }
