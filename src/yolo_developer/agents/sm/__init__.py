"""SM (Scrum Master) agent module for orchestration control plane.

The SM agent serves as the control plane for orchestration decisions,
providing centralized routing logic for the multi-agent workflow.

Key Responsibilities:
- Routing decisions: Determines next agent based on state analysis (Story 10.2)
- Circular logic detection: Detects agent ping-pong patterns (>3 exchanges per FR12)
- Escalation handling: Triggers human intervention when needed
- Gate-blocked recovery: Routes to appropriate agent for recovery
- Sprint planning: Plan sprints by prioritizing and sequencing stories (Story 10.3)

Example:
    >>> from yolo_developer.agents.sm import (
    ...     sm_node,
    ...     SMOutput,
    ...     AgentExchange,
    ...     plan_sprint,
    ...     SprintPlan,
    ... )
    >>>
    >>> # Run the SM node
    >>> result = await sm_node(state)
    >>> result["sm_output"]["routing_decision"]
    'pm'
    >>>
    >>> # Plan a sprint (Story 10.3)
    >>> stories = [{"story_id": "1-1", "title": "Setup"}]
    >>> plan = await plan_sprint(stories)
    >>> plan.sprint_id
    'sprint-20260112'

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
    - FR9: SM Agent can plan sprints by prioritizing and sequencing stories
    - FR10: Task delegation
    - FR11: Health monitoring
    - FR12: Circular logic detection (>3 exchanges)
    - FR13: Conflict mediation
    - FR65: SM Agent can calculate weighted priority scores for story selection
"""

from __future__ import annotations

from yolo_developer.agents.sm.node import sm_node
from yolo_developer.agents.sm.planning import (
    CircularDependencyError,
    plan_sprint,
)
from yolo_developer.agents.sm.planning_types import (
    DEFAULT_DEPENDENCY_WEIGHT,
    DEFAULT_MAX_POINTS,
    DEFAULT_MAX_STORIES,
    DEFAULT_TECH_DEBT_WEIGHT,
    DEFAULT_VALUE_WEIGHT,
    DEFAULT_VELOCITY_WEIGHT,
    PlanningConfig,
    SprintPlan,
    SprintStory,
)
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
    "DEFAULT_DEPENDENCY_WEIGHT",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MAX_STORIES",
    "DEFAULT_TECH_DEBT_WEIGHT",
    "DEFAULT_VALUE_WEIGHT",
    "DEFAULT_VELOCITY_WEIGHT",
    "NATURAL_SUCCESSOR",
    "VALID_AGENTS",
    "AgentExchange",
    "CircularDependencyError",
    "EscalationReason",
    "PlanningConfig",
    "RoutingDecision",
    "SMOutput",
    "SprintPlan",
    "SprintStory",
    "plan_sprint",
    "sm_node",
]
