"""Type definitions for rollback coordination (Story 10.15).

This module provides the data types used by the rollback coordination module:

- RollbackReason: Literal type for reasons triggering rollback
- RollbackStatus: Literal type for rollback lifecycle stages
- RollbackStep: A single step in a rollback plan
- RollbackPlan: Complete plan for executing a rollback
- RollbackResult: Result of a rollback operation
- RollbackConfig: Configuration for rollback behavior

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.rollback_types import (
    ...     RollbackConfig,
    ...     RollbackPlan,
    ...     RollbackStep,
    ...     RollbackResult,
    ... )
    >>>
    >>> # Create a rollback step
    >>> step = RollbackStep(
    ...     step_id="step-001",
    ...     action="restore_field",
    ...     target_field="current_agent",
    ...     previous_value="analyst",
    ...     current_value="dev",
    ...     executed=False,
    ...     success=None,
    ... )
    >>> step.target_field
    'current_agent'
    >>>
    >>> # Create rollback plan with steps
    >>> plan = RollbackPlan(
    ...     plan_id="plan-001",
    ...     reason="checkpoint_recovery",
    ...     checkpoint_id="chk-12345678",
    ...     steps=(step,),
    ...     estimated_impact="moderate",
    ... )
    >>> plan.reason
    'checkpoint_recovery'

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR71: SM Agent can coordinate rollback operations as emergency sprints
    - ADR-007: Checkpoint-based recovery patterns
    - Story 10.10: Emergency Protocols (Checkpoint type)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

RollbackReason = Literal[
    "checkpoint_recovery",
    "emergency_recovery",
    "manual_request",
    "conflict_resolution",
    "gate_failure",
]
"""Reason that triggered the rollback operation.

Values:
    checkpoint_recovery: Rollback to a saved checkpoint after failure
    emergency_recovery: Rollback triggered by emergency protocol
    manual_request: User explicitly requested rollback
    conflict_resolution: Rollback to resolve agent conflict
    gate_failure: Rollback due to quality gate failure
"""

RollbackStatus = Literal[
    "pending",
    "planning",
    "executing",
    "completed",
    "failed",
    "escalated",
]
"""Current status of the rollback operation.

Values:
    pending: Rollback initiated but not yet started
    planning: Creating rollback plan with steps
    executing: Rollback steps being executed
    completed: Rollback completed successfully
    failed: Rollback failed (partial or complete failure)
    escalated: Rollback escalated to human intervention
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

DEFAULT_MAX_ROLLBACK_STEPS: int = 100
"""Maximum number of rollback steps allowed in a single plan."""

DEFAULT_ALLOW_PARTIAL_ROLLBACK: bool = False
"""Default setting for allowing partial rollback on failure."""

DEFAULT_LOG_ROLLBACKS: bool = True
"""Default setting for logging rollback events."""

DEFAULT_AUTO_ESCALATE_ON_FAILURE: bool = True
"""Default setting for auto-escalating on rollback failure."""

MIN_DURATION_MS: float = 0.0
"""Minimum duration value for rollback operations."""

MAX_DURATION_MS: float = 86_400_000.0
"""Maximum duration value (24 hours in milliseconds)."""

VALID_ROLLBACK_REASONS: frozenset[str] = frozenset(
    {
        "checkpoint_recovery",
        "emergency_recovery",
        "manual_request",
        "conflict_resolution",
        "gate_failure",
    }
)
"""Set of valid rollback reason values."""

VALID_ROLLBACK_STATUSES: frozenset[str] = frozenset(
    {"pending", "planning", "executing", "completed", "failed", "escalated"}
)
"""Set of valid rollback status values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class RollbackStep:
    """A single step in a rollback plan.

    Represents one atomic operation to revert state to a previous value.

    Attributes:
        step_id: Unique identifier for this step (e.g., "step-001")
        action: Type of action to perform (e.g., "restore_field", "clear_field")
        target_field: State field to modify
        previous_value: Value to restore (from checkpoint)
        current_value: Current value being replaced
        executed: Whether this step has been executed
        success: Result of execution (None if not executed, True/False after)

    Example:
        >>> step = RollbackStep(
        ...     step_id="step-001",
        ...     action="restore_field",
        ...     target_field="current_agent",
        ...     previous_value="analyst",
        ...     current_value="dev",
        ...     executed=False,
        ...     success=None,
        ... )
        >>> step.action
        'restore_field'
    """

    step_id: str
    action: str
    target_field: str
    previous_value: Any
    current_value: Any
    executed: bool
    success: bool | None

    def __post_init__(self) -> None:
        """Validate step data and log warnings for issues."""
        if not self.step_id:
            _logger.warning(
                "RollbackStep step_id is empty for target_field=%s",
                self.target_field,
            )
        if not self.action:
            _logger.warning(
                "RollbackStep action is empty for step_id=%s",
                self.step_id,
            )
        if not self.target_field:
            _logger.warning(
                "RollbackStep target_field is empty for step_id=%s",
                self.step_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the step.
        """
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target_field": self.target_field,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "executed": self.executed,
            "success": self.success,
        }


@dataclass(frozen=True)
class RollbackPlan:
    """Complete plan for executing a rollback operation.

    Contains all steps needed to restore state to a checkpoint.

    Attributes:
        plan_id: Unique identifier for this plan (e.g., "plan-001")
        reason: Why the rollback was triggered
        checkpoint_id: ID of the checkpoint to restore to
        steps: Ordered tuple of rollback steps to execute
        created_at: ISO timestamp when plan was created (auto-generated)
        estimated_impact: Impact level ("minimal", "moderate", "significant")

    Example:
        >>> plan = RollbackPlan(
        ...     plan_id="plan-001",
        ...     reason="checkpoint_recovery",
        ...     checkpoint_id="chk-12345678",
        ...     steps=(step1, step2),
        ...     estimated_impact="moderate",
        ... )
        >>> plan.checkpoint_id
        'chk-12345678'
    """

    plan_id: str
    reason: RollbackReason
    checkpoint_id: str
    steps: tuple[RollbackStep, ...]
    estimated_impact: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate plan data and log warnings for issues."""
        if not self.plan_id:
            _logger.warning(
                "RollbackPlan plan_id is empty for checkpoint_id=%s",
                self.checkpoint_id,
            )
        if self.reason not in VALID_ROLLBACK_REASONS:
            _logger.warning(
                "RollbackPlan reason='%s' is not a valid rollback reason for plan_id=%s",
                self.reason,
                self.plan_id,
            )
        if not self.checkpoint_id:
            _logger.warning(
                "RollbackPlan checkpoint_id is empty for plan_id=%s",
                self.plan_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation with nested steps.
        """
        return {
            "plan_id": self.plan_id,
            "reason": self.reason,
            "checkpoint_id": self.checkpoint_id,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "estimated_impact": self.estimated_impact,
        }


@dataclass(frozen=True)
class RollbackResult:
    """Result of a rollback operation.

    Captures the outcome of executing a rollback plan.

    Attributes:
        plan: The rollback plan that was executed
        status: Final status of the rollback operation
        steps_executed: Number of steps successfully executed
        steps_failed: Number of steps that failed
        rollback_complete: Whether rollback completed successfully
        duration_ms: Time taken to execute rollback in milliseconds
        error_message: Error message if rollback failed (None on success)

    Example:
        >>> result = RollbackResult(
        ...     plan=plan,
        ...     status="completed",
        ...     steps_executed=2,
        ...     steps_failed=0,
        ...     rollback_complete=True,
        ...     duration_ms=150.5,
        ...     error_message=None,
        ... )
        >>> result.rollback_complete
        True
    """

    plan: RollbackPlan
    status: RollbackStatus
    steps_executed: int
    steps_failed: int
    rollback_complete: bool
    duration_ms: float
    error_message: str | None

    def __post_init__(self) -> None:
        """Validate result data and log warnings for issues."""
        if self.status not in VALID_ROLLBACK_STATUSES:
            _logger.warning(
                "RollbackResult status='%s' is not a valid rollback status for plan_id=%s",
                self.status,
                self.plan.plan_id,
            )
        if self.duration_ms < MIN_DURATION_MS:
            _logger.warning(
                "RollbackResult duration_ms=%.2f is negative for plan_id=%s",
                self.duration_ms,
                self.plan.plan_id,
            )
        if self.rollback_complete and self.steps_failed > 0:
            _logger.warning(
                "RollbackResult rollback_complete=True but steps_failed=%d for plan_id=%s",
                self.steps_failed,
                self.plan.plan_id,
            )
        if self.status == "completed" and not self.rollback_complete:
            _logger.warning(
                "RollbackResult status='completed' but rollback_complete=False for plan_id=%s",
                self.plan.plan_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation with nested plan.
        """
        return {
            "plan": self.plan.to_dict(),
            "status": self.status,
            "steps_executed": self.steps_executed,
            "steps_failed": self.steps_failed,
            "rollback_complete": self.rollback_complete,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class RollbackConfig:
    """Configuration for rollback behavior.

    Controls limits, behavior, and logging for rollback operations.

    Attributes:
        max_steps: Maximum rollback steps allowed (default 100)
        allow_partial_rollback: Whether to allow partial rollback on failure (default False)
        log_rollbacks: Whether to log rollback events (default True)
        auto_escalate_on_failure: Whether to auto-escalate on failure (default True)

    Example:
        >>> config = RollbackConfig(max_steps=50)
        >>> config.max_steps
        50
    """

    max_steps: int = DEFAULT_MAX_ROLLBACK_STEPS
    allow_partial_rollback: bool = DEFAULT_ALLOW_PARTIAL_ROLLBACK
    log_rollbacks: bool = DEFAULT_LOG_ROLLBACKS
    auto_escalate_on_failure: bool = DEFAULT_AUTO_ESCALATE_ON_FAILURE

    def __post_init__(self) -> None:
        """Validate config values and log warnings for issues."""
        if self.max_steps <= 0:
            _logger.warning(
                "RollbackConfig max_steps=%d should be positive (negative or zero)",
                self.max_steps,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "max_steps": self.max_steps,
            "allow_partial_rollback": self.allow_partial_rollback,
            "log_rollbacks": self.log_rollbacks,
            "auto_escalate_on_failure": self.auto_escalate_on_failure,
        }
