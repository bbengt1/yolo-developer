"""Type definitions for SM (Scrum Master) agent (Story 10.2).

This module provides the data types used by the SM agent:

- RoutingDecision: Literal type for routing target agents
- EscalationReason: Literal type for escalation reasons
- SMOutput: Complete output from SM processing
- AgentExchange: Record of a message exchange between agents (for circular logic detection)

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.types import (
    ...     SMOutput,
    ...     AgentExchange,
    ... )
    >>>
    >>> output = SMOutput(
    ...     routing_decision="pm",
    ...     routing_rationale="Analyst completed requirements crystallization",
    ...     circular_logic_detected=False,
    ...     escalation_triggered=False,
    ... )
    >>> output.to_dict()
    {'routing_decision': 'pm', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

RoutingDecision = Literal[
    "analyst",
    "pm",
    "architect",
    "dev",
    "tea",
    "sm",
    "escalate",
]
"""Target agent for routing decision.

Values:
    analyst: Route to Analyst agent for requirement analysis
    pm: Route to PM agent for story transformation
    architect: Route to Architect agent for design decisions
    dev: Route to Dev agent for implementation
    tea: Route to TEA agent for validation
    sm: Route back to SM for re-evaluation
    escalate: Escalate to human intervention
"""

EscalationReason = Literal[
    "human_requested",
    "circular_logic",
    "gate_blocked_unresolvable",
    "conflict_unresolved",
    "agent_failure",
    "unknown",
]
"""Reason for escalating to human intervention.

Values:
    human_requested: Human escalation was explicitly requested via state flag
    circular_logic: Detected circular logic pattern (>3 exchanges)
    gate_blocked_unresolvable: Gate is blocked and cannot be resolved automatically
    conflict_unresolved: Agent conflict that could not be mediated
    agent_failure: Agent encountered unrecoverable error
    unknown: Unknown reason for escalation
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class AgentExchange:
    """Record of a message exchange between agents.

    Used for tracking agent-to-agent communication and detecting
    circular logic patterns (per FR12: >3 exchanges on same issue).

    Attributes:
        source_agent: Agent that sent the message
        target_agent: Agent that received the message
        exchange_type: Type of exchange (handoff, response, escalation)
        topic: Brief topic/reason for exchange
        timestamp: When the exchange occurred

    Example:
        >>> exchange = AgentExchange(
        ...     source_agent="analyst",
        ...     target_agent="pm",
        ...     exchange_type="handoff",
        ...     topic="requirements_crystallized",
        ... )
    """

    source_agent: str
    target_agent: str
    exchange_type: Literal["handoff", "response", "escalation", "query"]
    topic: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the exchange.
        """
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "exchange_type": self.exchange_type,
            "topic": self.topic,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SMOutput:
    """Complete output from SM agent processing.

    Contains the routing decision, rationale, and detection flags
    from SM control plane evaluation.

    Attributes:
        routing_decision: The target agent to route to
        routing_rationale: Explanation of why this routing was chosen
        circular_logic_detected: Whether circular logic pattern was detected
        escalation_triggered: Whether escalation was triggered
        escalation_reason: Reason for escalation if triggered
        exchange_count: Number of recent exchanges tracked
        recent_exchanges: Tuple of recent exchanges for tracking
        gate_blocked: Whether a gate is currently blocking progress
        recovery_agent: Agent to route to for recovery (if gate_blocked)
        processing_notes: Additional notes about the decision process
        sprint_plan: Optional sprint plan when SM is in planning mode (Story 10.3)
        delegation_result: Optional delegation result when task delegated (Story 10.4)
        health_status: Optional health status when monitoring is enabled (Story 10.5)
        created_at: ISO timestamp when output was created

    Example:
        >>> output = SMOutput(
        ...     routing_decision="pm",
        ...     routing_rationale="Analyst completed requirements crystallization",
        ...     circular_logic_detected=False,
        ...     escalation_triggered=False,
        ... )
        >>> output.to_dict()
        {'routing_decision': 'pm', ...}
    """

    routing_decision: RoutingDecision
    routing_rationale: str
    circular_logic_detected: bool = False
    escalation_triggered: bool = False
    escalation_reason: EscalationReason | None = None
    exchange_count: int = 0
    recent_exchanges: tuple[AgentExchange, ...] = field(default_factory=tuple)
    gate_blocked: bool = False
    recovery_agent: str | None = None
    processing_notes: str = ""
    sprint_plan: dict[str, Any] | None = None
    delegation_result: dict[str, Any] | None = None
    health_status: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested exchanges.
        """
        return {
            "routing_decision": self.routing_decision,
            "routing_rationale": self.routing_rationale,
            "circular_logic_detected": self.circular_logic_detected,
            "escalation_triggered": self.escalation_triggered,
            "escalation_reason": self.escalation_reason,
            "exchange_count": self.exchange_count,
            "recent_exchanges": [e.to_dict() for e in self.recent_exchanges],
            "gate_blocked": self.gate_blocked,
            "recovery_agent": self.recovery_agent,
            "processing_notes": self.processing_notes,
            "sprint_plan": self.sprint_plan,
            "delegation_result": self.delegation_result,
            "health_status": self.health_status,
            "created_at": self.created_at,
        }


# Threshold for circular logic detection (per FR12)
CIRCULAR_LOGIC_THRESHOLD = 3
"""Number of exchanges between same agents before circular logic is detected.

Per FR12: Detects and breaks out of circular/looping behavior by detecting
when agents ping-pong on the same issue more than 3 times.
"""

# Valid target agents for routing
VALID_AGENTS: frozenset[str] = frozenset(
    {
        "analyst",
        "pm",
        "architect",
        "dev",
        "tea",
        "sm",
        "escalate",
    }
)
"""Set of valid agent names for routing decisions."""

# Natural successor mapping for standard workflow flow
# Values are RoutingDecision literals to ensure type safety
NATURAL_SUCCESSOR: dict[str, RoutingDecision] = {
    "analyst": "pm",
    "pm": "architect",
    "architect": "dev",
    "dev": "tea",
    "tea": "dev",  # TEA routes back to dev if issues, or ends
    "sm": "analyst",  # SM defaults to analyst for new work
}
"""Mapping of agent to natural successor in standard workflow."""
