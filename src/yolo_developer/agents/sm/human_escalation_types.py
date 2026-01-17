"""Type definitions for human escalation (Story 10.14).

This module provides the data types used by the human escalation module:

- EscalationTrigger: Literal type for escalation triggers
- EscalationStatus: Literal type for escalation lifecycle stages
- EscalationOption: A selectable option for human decision
- EscalationRequest: Request for human intervention
- EscalationResponse: User's decision response
- EscalationResult: Complete result of escalation operation
- EscalationConfig: Configuration for escalation behavior

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.human_escalation_types import (
    ...     EscalationConfig,
    ...     EscalationRequest,
    ...     EscalationOption,
    ...     EscalationResponse,
    ...     EscalationResult,
    ... )
    >>>
    >>> # Create an escalation option
    >>> option = EscalationOption(
    ...     option_id="opt-1",
    ...     label="Retry",
    ...     description="Retry the failed operation",
    ...     action="retry",
    ...     is_recommended=True,
    ... )
    >>>
    >>> # Create escalation request with options
    >>> request = EscalationRequest(
    ...     request_id="esc-12345",
    ...     trigger="circular_logic",
    ...     agent="architect",
    ...     summary="Circular logic detected",
    ...     context={"exchanges": 5},
    ...     options=(option,),
    ...     recommended_option="opt-1",
    ... )
    >>> request.trigger
    'circular_logic'

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR70: SM Agent can escalate to human when circular logic persists
    - Story 10.6: Circular Logic Detection (escalation_triggered flag)
    - Story 10.7: Conflict Mediation (escalations_triggered tuple)
    - Story 10.10: Emergency Protocols (escalate_emergency function)
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

EscalationTrigger = Literal[
    "circular_logic",
    "conflict_unresolved",
    "gate_blocked",
    "system_error",
    "agent_stuck",
    "user_requested",
]
"""Reason that triggered escalation to human.

Values:
    circular_logic: Circular logic persists after intervention (FR70)
    conflict_unresolved: Conflict mediation failed to resolve
    gate_blocked: Quality gate blocked with no recovery path
    system_error: Unrecoverable system error occurred
    agent_stuck: Agent stuck beyond idle threshold
    user_requested: User explicitly requested escalation
"""

EscalationStatus = Literal[
    "pending",
    "presented",
    "resolved",
    "timed_out",
    "cancelled",
]
"""Current status of the escalation.

Values:
    pending: Escalation created, not yet presented to user
    presented: Escalation shown to user, awaiting response
    resolved: User provided decision and it was integrated
    timed_out: User didn't respond within timeout period
    cancelled: Escalation was cancelled (system resolved issue)
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

DEFAULT_ESCALATION_TIMEOUT_SECONDS: int = 300
"""Default timeout for user response (5 minutes)."""

DEFAULT_LOG_ESCALATIONS: bool = True
"""Default setting for logging escalation events."""

DEFAULT_MAX_PENDING: int = 5
"""Default maximum pending escalations before queueing."""

MIN_DURATION_MS: float = 0.0
"""Minimum duration value for escalation operations."""

MAX_DURATION_MS: float = 86_400_000.0
"""Maximum duration value (24 hours in milliseconds)."""

VALID_ESCALATION_TRIGGERS: frozenset[str] = frozenset(
    {
        "circular_logic",
        "conflict_unresolved",
        "gate_blocked",
        "system_error",
        "agent_stuck",
        "user_requested",
    }
)
"""Set of valid escalation trigger values."""

VALID_ESCALATION_STATUSES: frozenset[str] = frozenset(
    {"pending", "presented", "resolved", "timed_out", "cancelled"}
)
"""Set of valid escalation status values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class EscalationOption:
    """A selectable option for human decision.

    Represents one choice the user can make when responding to an escalation.

    Attributes:
        option_id: Unique identifier for this option (e.g., "opt-1")
        label: Short label for UI display (e.g., "Retry")
        description: Detailed explanation of what this option does
        action: The action to take if selected (e.g., "retry", "skip", "abort")
        is_recommended: Whether this is the recommended option

    Example:
        >>> option = EscalationOption(
        ...     option_id="opt-1",
        ...     label="Retry",
        ...     description="Retry the failed operation",
        ...     action="retry",
        ...     is_recommended=True,
        ... )
        >>> option.is_recommended
        True
    """

    option_id: str
    label: str
    description: str
    action: str
    is_recommended: bool

    def __post_init__(self) -> None:
        """Validate option data and log warnings for issues."""
        if not self.option_id:
            _logger.warning(
                "EscalationOption option_id is empty for label=%s",
                self.label,
            )
        if not self.label:
            _logger.warning(
                "EscalationOption label is empty for option_id=%s",
                self.option_id,
            )
        if not self.description:
            _logger.warning(
                "EscalationOption description is empty for option_id=%s",
                self.option_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the option.
        """
        return {
            "option_id": self.option_id,
            "label": self.label,
            "description": self.description,
            "action": self.action,
            "is_recommended": self.is_recommended,
        }


@dataclass(frozen=True)
class EscalationRequest:
    """Request for human intervention.

    Created when the system detects an unresolvable issue that requires
    human decision-making.

    Attributes:
        request_id: Unique identifier for this request (e.g., "esc-12345")
        trigger: What triggered the escalation
        agent: Agent that was active when escalation was triggered
        summary: Human-readable summary of the issue
        context: Detailed context (exchanges, decisions, errors)
        options: Available options for the user to choose from
        recommended_option: option_id of recommended choice (None if no recommendation)
        created_at: ISO timestamp when request was created (auto-generated)

    Example:
        >>> request = EscalationRequest(
        ...     request_id="esc-12345",
        ...     trigger="circular_logic",
        ...     agent="architect",
        ...     summary="Circular logic detected",
        ...     context={"exchanges": 5},
        ...     options=(option,),
        ...     recommended_option="opt-1",
        ... )
        >>> request.trigger
        'circular_logic'
    """

    request_id: str
    trigger: EscalationTrigger
    agent: str
    summary: str
    context: dict[str, Any]
    options: tuple[EscalationOption, ...]
    recommended_option: str | None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate request data and log warnings for issues."""
        if not self.request_id:
            _logger.warning(
                "EscalationRequest request_id is empty for agent=%s",
                self.agent,
            )
        if not self.agent:
            _logger.warning(
                "EscalationRequest agent is empty for request_id=%s",
                self.request_id,
            )
        if self.trigger not in VALID_ESCALATION_TRIGGERS:
            _logger.warning(
                "EscalationRequest trigger='%s' is not a valid trigger for request_id=%s",
                self.trigger,
                self.request_id,
            )
        if not self.summary:
            _logger.warning(
                "EscalationRequest summary is empty for request_id=%s",
                self.request_id,
            )
        if self.recommended_option is not None:
            option_ids = {opt.option_id for opt in self.options}
            if self.recommended_option not in option_ids:
                _logger.warning(
                    "EscalationRequest recommended_option='%s' not in options for request_id=%s",
                    self.recommended_option,
                    self.request_id,
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation with nested options.
        """
        return {
            "request_id": self.request_id,
            "trigger": self.trigger,
            "agent": self.agent,
            "summary": self.summary,
            "context": dict(self.context),
            "options": [opt.to_dict() for opt in self.options],
            "recommended_option": self.recommended_option,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EscalationResponse:
    """User's decision response to an escalation.

    Captures the user's choice and optional rationale.

    Attributes:
        request_id: ID of the request this responds to
        selected_option: option_id of the user's chosen option
        user_rationale: Optional explanation from user for their choice
        responded_at: ISO timestamp when response was received (auto-generated)

    Example:
        >>> response = EscalationResponse(
        ...     request_id="esc-12345",
        ...     selected_option="opt-1",
        ...     user_rationale="Seems like the safest choice",
        ... )
        >>> response.selected_option
        'opt-1'
    """

    request_id: str
    selected_option: str
    user_rationale: str | None
    responded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate response data and log warnings for issues."""
        if not self.request_id:
            _logger.warning("EscalationResponse request_id is empty")
        if not self.selected_option:
            _logger.warning(
                "EscalationResponse selected_option is empty for request_id=%s",
                self.request_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the response.
        """
        return {
            "request_id": self.request_id,
            "selected_option": self.selected_option,
            "user_rationale": self.user_rationale,
            "responded_at": self.responded_at,
        }


@dataclass(frozen=True)
class EscalationResult:
    """Complete result of an escalation operation.

    Captures the full lifecycle from request through resolution.

    Attributes:
        request: The original escalation request
        response: User's response (None if timed out or cancelled)
        status: Final status of the escalation
        resolution_action: Action taken to resolve (from selected option)
        integration_success: Whether decision was successfully integrated
        duration_ms: Time from request creation to resolution in milliseconds

    Example:
        >>> result = EscalationResult(
        ...     request=request,
        ...     response=response,
        ...     status="resolved",
        ...     resolution_action="retry",
        ...     integration_success=True,
        ...     duration_ms=5000.0,
        ... )
        >>> result.integration_success
        True
    """

    request: EscalationRequest
    response: EscalationResponse | None
    status: EscalationStatus
    resolution_action: str | None
    integration_success: bool
    duration_ms: float

    def __post_init__(self) -> None:
        """Validate result data and log warnings for issues."""
        if self.status not in VALID_ESCALATION_STATUSES:
            _logger.warning(
                "EscalationResult status='%s' is not a valid status for request_id=%s",
                self.status,
                self.request.request_id,
            )
        if self.duration_ms < MIN_DURATION_MS:
            _logger.warning(
                "EscalationResult duration_ms=%.2f is negative for request_id=%s",
                self.duration_ms,
                self.request.request_id,
            )
        if self.integration_success and self.response is None:
            _logger.warning(
                "EscalationResult integration_success=True but response is None for request_id=%s",
                self.request.request_id,
            )
        if self.status == "resolved" and not self.integration_success:
            _logger.warning(
                "EscalationResult status='resolved' but integration_success=False for request_id=%s",
                self.request.request_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation with nested request and response.
        """
        return {
            "request": self.request.to_dict(),
            "response": self.response.to_dict() if self.response else None,
            "status": self.status,
            "resolution_action": self.resolution_action,
            "integration_success": self.integration_success,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class EscalationConfig:
    """Configuration for escalation behavior.

    Controls timeouts, defaults, and logging for escalations.

    Attributes:
        timeout_seconds: Seconds to wait for user response (default 300)
        default_action: Action to take on timeout (default "skip")
        log_escalations: Whether to log escalation events (default True)
        max_pending: Maximum pending escalations before queueing (default 5)

    Example:
        >>> config = EscalationConfig(timeout_seconds=60)
        >>> config.timeout_seconds
        60
    """

    timeout_seconds: int = DEFAULT_ESCALATION_TIMEOUT_SECONDS
    default_action: str = "skip"
    log_escalations: bool = DEFAULT_LOG_ESCALATIONS
    max_pending: int = DEFAULT_MAX_PENDING

    def __post_init__(self) -> None:
        """Validate config values and log warnings for issues."""
        if self.timeout_seconds < 0:
            _logger.warning(
                "EscalationConfig timeout_seconds=%d should be non-negative",
                self.timeout_seconds,
            )
        if self.max_pending < 1:
            _logger.warning(
                "EscalationConfig max_pending=%d should be at least 1",
                self.max_pending,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "timeout_seconds": self.timeout_seconds,
            "default_action": self.default_action,
            "log_escalations": self.log_escalations,
            "max_pending": self.max_pending,
        }
