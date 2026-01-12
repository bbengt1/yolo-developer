"""Type definitions for task delegation (Story 10.4).

This module provides the data types used by the delegation module:

- TaskType: Literal type for task categories
- DelegationRequest: Request to delegate a task to an agent
- DelegationResult: Result of a delegation attempt
- DelegationConfig: Configuration for delegation behavior

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.delegation_types import (
    ...     DelegationRequest,
    ...     DelegationResult,
    ...     DelegationConfig,
    ...     TASK_TO_AGENT,
    ... )
    >>>
    >>> # Get agent for task type
    >>> TASK_TO_AGENT["implementation"]
    'dev'
    >>>
    >>> # Create delegation request
    >>> request = DelegationRequest(
    ...     task_type="implementation",
    ...     task_description="Implement user auth",
    ...     source_agent="sm",
    ...     target_agent="dev",
    ...     context={"story_id": "1-2"},
    ... )
    >>> request.to_dict()
    {'task_type': 'implementation', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR10: SM Agent can delegate tasks to appropriate specialized agents
    - FR15: System can handle agent handoffs with context preservation
    - FR68: SM Agent can trigger inter-agent sync protocols
    - FR69: SM Agent can inject context when agents lack information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

TaskType = Literal[
    "requirement_analysis",
    "story_creation",
    "architecture_design",
    "implementation",
    "validation",
    "orchestration",
]
"""Type of task that can be delegated to an agent.

Values:
    requirement_analysis: For Analyst agent - analyzing and crystallizing requirements
    story_creation: For PM agent - transforming requirements into stories
    architecture_design: For Architect agent - designing system architecture
    implementation: For Dev agent - implementing code and tests
    validation: For TEA agent - validating tests and quality
    orchestration: For SM agent - coordinating agents and workflow
"""

Priority = Literal["low", "normal", "high", "critical"]
"""Priority level for delegated tasks.

Values:
    low: Non-urgent task, can be deferred
    normal: Standard priority (default)
    high: Should be addressed promptly
    critical: Requires immediate attention
"""

# =============================================================================
# Constants
# =============================================================================

DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS: float = 30.0
"""Default timeout in seconds for acknowledgment verification."""

DEFAULT_MAX_RETRY_ATTEMPTS: int = 3
"""Default maximum retry attempts for delegation."""

VALID_TASK_TYPES: frozenset[str] = frozenset(
    {
        "requirement_analysis",
        "story_creation",
        "architecture_design",
        "implementation",
        "validation",
        "orchestration",
    }
)
"""Set of all valid task type values."""

AGENT_EXPERTISE: dict[str, tuple[TaskType, ...]] = {
    "analyst": ("requirement_analysis",),
    "pm": ("story_creation",),
    "architect": ("architecture_design",),
    "dev": ("implementation",),
    "tea": ("validation",),
    "sm": ("orchestration",),
}
"""Mapping of agent names to their areas of expertise (per FR10).

Each agent has specific expertise and should only receive tasks
that match their capabilities.
"""

TASK_TO_AGENT: dict[TaskType, str] = {
    "requirement_analysis": "analyst",
    "story_creation": "pm",
    "architecture_design": "architect",
    "implementation": "dev",
    "validation": "tea",
    "orchestration": "sm",
}
"""Mapping of task types to the agent responsible for that task type.

This is the inverse of AGENT_EXPERTISE, providing O(1) lookup for
determining which agent should handle a given task type.
"""

# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class DelegationConfig:
    """Configuration for delegation behavior.

    Controls timeouts, retries, and other delegation parameters.

    Attributes:
        acknowledgment_timeout_seconds: Max time to wait for agent acknowledgment
        max_retry_attempts: Maximum attempts to delegate before giving up
        allow_self_delegation: Whether SM can delegate to itself (default False)

    Example:
        >>> config = DelegationConfig(
        ...     acknowledgment_timeout_seconds=60.0,
        ...     max_retry_attempts=5,
        ... )
        >>> config.max_retry_attempts
        5
    """

    acknowledgment_timeout_seconds: float = DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS
    max_retry_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS
    allow_self_delegation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "acknowledgment_timeout_seconds": self.acknowledgment_timeout_seconds,
            "max_retry_attempts": self.max_retry_attempts,
            "allow_self_delegation": self.allow_self_delegation,
        }


@dataclass(frozen=True)
class DelegationRequest:
    """Request to delegate a task to an agent.

    Contains all the information needed to delegate work from the SM
    to a specialized agent, including task details and context.

    Attributes:
        task_type: Type of task being delegated (determines target agent)
        task_description: Human-readable description of what needs to be done
        source_agent: Agent requesting the delegation (usually "sm")
        target_agent: Agent that will receive the delegated task
        context: Dictionary of context data relevant to the task
        priority: Priority level for the task (default "normal")
        created_at: ISO timestamp when request was created (auto-generated)

    Example:
        >>> request = DelegationRequest(
        ...     task_type="implementation",
        ...     task_description="Implement user authentication",
        ...     source_agent="sm",
        ...     target_agent="dev",
        ...     context={"story_id": "1-2-user-auth"},
        ...     priority="high",
        ... )
        >>> request.target_agent
        'dev'
    """

    task_type: TaskType
    task_description: str
    source_agent: str
    target_agent: str
    context: dict[str, Any]
    priority: Priority = "normal"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the request.
        """
        return {
            "task_type": self.task_type,
            "task_description": self.task_description,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "context": self.context,
            "priority": self.priority,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DelegationResult:
    """Result of a delegation attempt.

    Contains the outcome of attempting to delegate a task, including
    whether it succeeded, was acknowledged, and any error details.

    Attributes:
        request: The original delegation request
        success: Whether the delegation was successful
        acknowledged: Whether the target agent acknowledged the delegation
        acknowledgment_timestamp: When acknowledgment was received (if any)
        error_message: Error details if delegation failed
        handoff_context: Context dict for state updates (if successful)

    Example:
        >>> result = DelegationResult(
        ...     request=request,
        ...     success=True,
        ...     acknowledged=True,
        ...     acknowledgment_timestamp="2026-01-12T12:00:00+00:00",
        ...     handoff_context={"source_agent": "sm", "target_agent": "dev"},
        ... )
        >>> result.success
        True
    """

    request: DelegationRequest
    success: bool
    acknowledged: bool
    acknowledgment_timestamp: str | None = None
    error_message: str | None = None
    handoff_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested request.
        """
        return {
            "request": self.request.to_dict(),
            "success": self.success,
            "acknowledged": self.acknowledged,
            "acknowledgment_timestamp": self.acknowledgment_timestamp,
            "error_message": self.error_message,
            "handoff_context": self.handoff_context,
        }
