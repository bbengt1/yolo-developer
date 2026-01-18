"""Type definitions for audit trail and decision logging (Story 11.1).

This module provides the data types used by the audit module:

- DecisionType: Literal type for decision categories
- DecisionSeverity: Literal type for decision importance levels
- AgentIdentity: Identifies which agent made a decision
- DecisionContext: Context about when/where a decision was made
- Decision: Complete decision record with content, rationale, and metadata

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.types import (
    ...     Decision,
    ...     AgentIdentity,
    ...     DecisionContext,
    ... )
    >>>
    >>> agent = AgentIdentity(
    ...     agent_name="analyst",
    ...     agent_type="analyst",
    ...     session_id="session-123",
    ... )
    >>> decision = Decision(
    ...     id="dec-001",
    ...     decision_type="requirement_analysis",
    ...     content="Crystallized requirement",
    ...     rationale="Clear and testable",
    ...     agent=agent,
    ...     context=DecisionContext(),
    ...     timestamp="2026-01-17T12:00:00Z",
    ... )
    >>> decision.to_dict()
    {'id': 'dec-001', ...}

References:
    - FR81: System can log all agent decisions with rationale
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

DecisionType = Literal[
    "requirement_analysis",
    "story_creation",
    "architecture_choice",
    "implementation_choice",
    "test_strategy",
    "orchestration",
    "quality_gate",
    "escalation",
]
"""Type of decision made by an agent.

Values:
    requirement_analysis: Analyst crystallizing requirements
    story_creation: PM creating or modifying stories
    architecture_choice: Architect making design decisions
    implementation_choice: Dev choosing implementation approach
    test_strategy: TEA deciding testing approach
    orchestration: SM making workflow decisions
    quality_gate: Gate evaluation pass/fail decisions
    escalation: Decision to escalate to another agent or human
"""

DecisionSeverity = Literal["info", "warning", "critical"]
"""Severity level of a decision.

Values:
    info: Normal decision, informational
    warning: Decision that may need attention
    critical: Decision that significantly impacts system behavior
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

VALID_DECISION_TYPES: frozenset[str] = frozenset(
    {
        "requirement_analysis",
        "story_creation",
        "architecture_choice",
        "implementation_choice",
        "test_strategy",
        "orchestration",
        "quality_gate",
        "escalation",
    }
)
"""Set of valid decision type values."""

VALID_DECISION_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
"""Set of valid decision severity values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class AgentIdentity:
    """Identifies which agent made a decision.

    Captures the agent's name, type, and session for traceability.

    Attributes:
        agent_name: Human-readable agent name (e.g., "analyst")
        agent_type: Agent type for categorization (e.g., "analyst", "pm", "dev")
        session_id: Session identifier for grouping related decisions

    Example:
        >>> agent = AgentIdentity(
        ...     agent_name="analyst",
        ...     agent_type="analyst",
        ...     session_id="session-123",
        ... )
        >>> agent.agent_name
        'analyst'
    """

    agent_name: str
    agent_type: str
    session_id: str

    def __post_init__(self) -> None:
        """Validate agent identity data and log warnings for issues."""
        if not self.agent_name:
            _logger.warning("AgentIdentity agent_name is empty")
        if not self.agent_type:
            _logger.warning("AgentIdentity agent_type is empty")
        if not self.session_id:
            _logger.warning("AgentIdentity session_id is empty")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the agent identity.
        """
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class DecisionContext:
    """Context about when/where a decision was made.

    Provides optional contextual information for traceability,
    linking decisions to sprints, stories, artifacts, parent decisions,
    and trace links from the traceability system (Story 11.2).

    Attributes:
        sprint_id: Sprint identifier (optional)
        story_id: Story identifier (optional)
        artifact_id: Related artifact identifier (optional)
        parent_decision_id: ID of parent decision in a chain (optional)
        trace_links: List of TraceLink IDs from traceability system (optional)

    Example:
        >>> context = DecisionContext(
        ...     sprint_id="sprint-1",
        ...     story_id="1-2-user-auth",
        ...     trace_links=["link-001", "link-002"],
        ... )
        >>> context.sprint_id
        'sprint-1'
    """

    sprint_id: str | None = None
    story_id: str | None = None
    artifact_id: str | None = None
    parent_decision_id: str | None = None
    trace_links: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the decision context.
        """
        return {
            "sprint_id": self.sprint_id,
            "story_id": self.story_id,
            "artifact_id": self.artifact_id,
            "parent_decision_id": self.parent_decision_id,
            "trace_links": list(self.trace_links),
        }


@dataclass(frozen=True)
class Decision:
    """Complete decision record with content, rationale, and metadata.

    Represents a single logged decision made by an agent, including
    the decision content, rationale, agent identity, context, and metadata.

    Attributes:
        id: Unique decision identifier (UUID)
        decision_type: Type of decision (from DecisionType)
        content: What was decided
        rationale: Why this decision was made
        agent: Identity of the agent making the decision
        context: Contextual information (sprint, story, etc.)
        timestamp: ISO 8601 timestamp when decision was made
        metadata: Additional key-value data (optional)
        severity: Decision importance level (default: "info")

    Example:
        >>> decision = Decision(
        ...     id="dec-001",
        ...     decision_type="requirement_analysis",
        ...     content="OAuth2 authentication required",
        ...     rationale="Industry standard security",
        ...     agent=agent,
        ...     context=context,
        ...     timestamp="2026-01-17T12:00:00Z",
        ... )
        >>> decision.content
        'OAuth2 authentication required'
    """

    id: str
    decision_type: DecisionType
    content: str
    rationale: str
    agent: AgentIdentity
    context: DecisionContext
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    severity: DecisionSeverity = "info"

    def __post_init__(self) -> None:
        """Validate decision data and log warnings for issues."""
        if not self.id:
            _logger.warning("Decision id is empty")
        if self.decision_type not in VALID_DECISION_TYPES:
            _logger.warning(
                "Decision decision_type='%s' is not a valid type for id=%s",
                self.decision_type,
                self.id,
            )
        if not self.content:
            _logger.warning("Decision content is empty for id=%s", self.id)
        if not self.rationale:
            _logger.warning("Decision rationale is empty for id=%s", self.id)
        if not self.timestamp:
            _logger.warning("Decision timestamp is empty for id=%s", self.id)
        if self.severity not in VALID_DECISION_SEVERITIES:
            _logger.warning(
                "Decision severity='%s' is not a valid severity for id=%s",
                self.severity,
                self.id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation with nested agent and context.
        """
        return {
            "id": self.id,
            "decision_type": self.decision_type,
            "content": self.content,
            "rationale": self.rationale,
            "agent": self.agent.to_dict(),
            "context": self.context.to_dict(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "severity": self.severity,
        }
