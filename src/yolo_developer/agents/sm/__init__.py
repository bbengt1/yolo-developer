"""SM (Scrum Master) agent module for orchestration control plane (Story 10.2).

The SM agent serves as the control plane for orchestration decisions,
providing centralized routing logic for the multi-agent workflow.

Key Responsibilities:
- Routing decisions: Determines next agent based on state analysis
- Circular logic detection: Detects agent ping-pong patterns (>3 exchanges per FR12)
- Escalation handling: Triggers human intervention when needed
- Gate-blocked recovery: Routes to appropriate agent for recovery

Example:
    >>> from yolo_developer.agents.sm import (
    ...     sm_node,
    ...     SMOutput,
    ...     AgentExchange,
    ... )
    >>>
    >>> # Run the SM node
    >>> result = await sm_node(state)
    >>> result["sm_output"]["routing_decision"]
    'pm'

Architecture:
    The sm_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with sm_routing gate (non-blocking)

References:
    - ADR-005: Inter-Agent Communication
    - ADR-007: Error Handling Strategy
    - FR10: Task delegation
    - FR11: Health monitoring
    - FR12: Circular logic detection (>3 exchanges)
    - FR13: Conflict mediation
"""

from __future__ import annotations

from yolo_developer.agents.sm.node import sm_node
from yolo_developer.agents.sm.types import (
    CIRCULAR_LOGIC_THRESHOLD,
    NATURAL_SUCCESSOR,
    VALID_AGENTS,
    AgentExchange,
    EscalationReason,
    RoutingDecision,
    SMOutput,
)

__all__ = [
    "CIRCULAR_LOGIC_THRESHOLD",
    "NATURAL_SUCCESSOR",
    "VALID_AGENTS",
    "AgentExchange",
    "EscalationReason",
    "RoutingDecision",
    "SMOutput",
    "sm_node",
]
