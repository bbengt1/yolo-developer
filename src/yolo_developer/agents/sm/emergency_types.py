"""Type definitions for emergency protocols (Story 10.10).

This module provides the data types used by the emergency protocol module:

- EmergencyType: Literal type for types of emergencies detected
- ProtocolStatus: Literal type for emergency protocol lifecycle stages
- RecoveryAction: Literal type for possible recovery actions
- EmergencyTrigger: Information about what triggered the emergency
- Checkpoint: Captured state checkpoint for recovery
- RecoveryOption: A potential recovery action with evaluation
- EmergencyProtocol: Complete emergency protocol execution record
- EmergencyConfig: Configuration for emergency protocol behavior

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.emergency_types import (
    ...     EmergencyConfig,
    ...     EmergencyProtocol,
    ...     EmergencyTrigger,
    ...     Checkpoint,
    ...     RecoveryOption,
    ... )
    >>>
    >>> # Create emergency config with custom thresholds
    >>> config = EmergencyConfig(max_recovery_attempts=5)
    >>> config.max_recovery_attempts
    5
    >>>
    >>> # Create emergency trigger
    >>> trigger = EmergencyTrigger(
    ...     emergency_type="health_degraded",
    ...     severity="critical",
    ...     source_agent="dev",
    ...     trigger_reason="System health critical",
    ...     health_status={"status": "critical"},
    ... )
    >>> trigger.emergency_type
    'health_degraded'

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR17: SM Agent can trigger emergency protocols when system health degrades
    - FR70: SM Agent can escalate to human when circular logic persists
    - FR71: SM Agent can coordinate rollback operations as emergency sprints
    - ADR-007: Retry with exponential backoff + SM-coordinated recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

EmergencyType = Literal[
    "health_degraded", "circular_logic", "gate_blocked", "agent_stuck", "system_error"
]
"""Type of emergency detected.

Values:
    health_degraded: System health fell below critical thresholds
    circular_logic: Agents in ping-pong loop (>3 exchanges per FR12)
    gate_blocked: Repeated gate failures blocking progress
    agent_stuck: Agent idle beyond threshold (stuck processing)
    system_error: Unrecoverable error in orchestration
"""

ProtocolStatus = Literal[
    "pending", "active", "checkpointed", "recovering", "resolved", "escalated"
]
"""Current status of the emergency protocol.

Values:
    pending: Emergency detected, protocol not yet started
    active: Protocol actively executing
    checkpointed: State has been checkpointed
    recovering: Recovery action in progress
    resolved: Emergency resolved successfully
    escalated: Escalated to human intervention
"""

RecoveryAction = Literal["retry", "rollback", "skip", "escalate", "terminate"]
"""Recovery action to take.

Values:
    retry: Retry the failed operation
    rollback: Rollback to checkpoint and try different path
    skip: Skip the problematic step and continue
    escalate: Escalate to human for manual intervention
    terminate: Gracefully terminate the current sprint
"""

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_RECOVERY_ATTEMPTS: int = 3
"""Maximum automatic recovery attempts before escalation."""

DEFAULT_ESCALATION_THRESHOLD: float = 0.5
"""Confidence threshold below which to escalate (0.0-1.0)."""

VALID_EMERGENCY_TYPES: frozenset[str] = frozenset(
    {"health_degraded", "circular_logic", "gate_blocked", "agent_stuck", "system_error"}
)
"""Set of all valid emergency type values."""

VALID_PROTOCOL_STATUSES: frozenset[str] = frozenset(
    {"pending", "active", "checkpointed", "recovering", "resolved", "escalated"}
)
"""Set of all valid protocol status values."""

VALID_RECOVERY_ACTIONS: frozenset[str] = frozenset(
    {"retry", "rollback", "skip", "escalate", "terminate"}
)
"""Set of all valid recovery action values."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class EmergencyTrigger:
    """Information about what triggered the emergency.

    Captures the conditions that led to emergency protocol activation.

    Attributes:
        emergency_type: Type of emergency detected
        severity: How severe the emergency is ("warning", "critical")
        source_agent: Agent that was active when emergency triggered (None for system-wide)
        trigger_reason: Human-readable description of why emergency was triggered
        health_status: Snapshot of HealthStatus at time of trigger (None if not health-related)
        detected_at: ISO timestamp when emergency was detected (auto-generated)

    Example:
        >>> trigger = EmergencyTrigger(
        ...     emergency_type="health_degraded",
        ...     severity="critical",
        ...     source_agent="dev",
        ...     trigger_reason="System health fell below critical thresholds",
        ...     health_status={"status": "critical"},
        ... )
        >>> trigger.emergency_type
        'health_degraded'
    """

    emergency_type: EmergencyType
    severity: str
    source_agent: str | None
    trigger_reason: str
    health_status: dict[str, Any] | None
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the trigger.
        """
        return {
            "emergency_type": self.emergency_type,
            "severity": self.severity,
            "source_agent": self.source_agent,
            "trigger_reason": self.trigger_reason,
            "health_status": self.health_status,
            "detected_at": self.detected_at,
        }


@dataclass(frozen=True)
class Checkpoint:
    """Captured state checkpoint for recovery.

    Stores a snapshot of state that can be used for rollback.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        state_snapshot: Serialized state at time of checkpoint
        created_at: ISO timestamp when checkpoint was created
        trigger_type: Type of emergency that triggered the checkpoint
        metadata: Additional metadata about the checkpoint (default empty dict)

    Example:
        >>> checkpoint = Checkpoint(
        ...     checkpoint_id="chk-12345678",
        ...     state_snapshot={"current_agent": "dev"},
        ...     created_at="2026-01-16T10:00:00+00:00",
        ...     trigger_type="health_degraded",
        ... )
        >>> checkpoint.checkpoint_id
        'chk-12345678'
    """

    checkpoint_id: str
    state_snapshot: dict[str, Any]
    created_at: str
    trigger_type: EmergencyType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the checkpoint.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "state_snapshot": self.state_snapshot,
            "created_at": self.created_at,
            "trigger_type": self.trigger_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecoveryOption:
    """A potential recovery action with evaluation.

    Represents one possible way to recover from the emergency.

    Attributes:
        action: The recovery action type
        description: Human-readable description of what this action does
        confidence: Confidence level that this action will succeed (0.0-1.0)
        risks: Tuple of potential risks associated with this action
        estimated_impact: Impact level ("minimal", "moderate", "significant")

    Example:
        >>> option = RecoveryOption(
        ...     action="retry",
        ...     description="Retry the failed operation",
        ...     confidence=0.8,
        ...     risks=("May fail again",),
        ...     estimated_impact="minimal",
        ... )
        >>> option.confidence
        0.8
    """

    action: RecoveryAction
    description: str
    confidence: float
    risks: tuple[str, ...]
    estimated_impact: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the recovery option.
        """
        return {
            "action": self.action,
            "description": self.description,
            "confidence": self.confidence,
            "risks": list(self.risks),
            "estimated_impact": self.estimated_impact,
        }


@dataclass(frozen=True)
class EmergencyProtocol:
    """Complete emergency protocol execution record.

    Tracks the full lifecycle of an emergency from detection to resolution.

    Attributes:
        protocol_id: Unique identifier for this protocol execution
        trigger: Information about what triggered the emergency
        status: Current status in the protocol lifecycle
        checkpoint: State checkpoint if created (None if not checkpointed)
        recovery_options: Tuple of evaluated recovery options
        selected_action: The action selected for recovery (None if escalated)
        escalation_reason: Why escalation occurred (None if not escalated)
        created_at: ISO timestamp when protocol started (auto-generated)
        resolved_at: ISO timestamp when protocol resolved (None if not resolved)

    Example:
        >>> from yolo_developer.agents.sm.emergency_types import (
        ...     EmergencyProtocol,
        ...     EmergencyTrigger,
        ... )
        >>> trigger = EmergencyTrigger(
        ...     emergency_type="health_degraded",
        ...     severity="critical",
        ...     source_agent=None,
        ...     trigger_reason="Test",
        ...     health_status=None,
        ... )
        >>> protocol = EmergencyProtocol(
        ...     protocol_id="emergency-12345678",
        ...     trigger=trigger,
        ...     status="pending",
        ...     checkpoint=None,
        ...     recovery_options=(),
        ...     selected_action=None,
        ...     escalation_reason=None,
        ... )
        >>> protocol.status
        'pending'
    """

    protocol_id: str
    trigger: EmergencyTrigger
    status: ProtocolStatus
    checkpoint: Checkpoint | None
    recovery_options: tuple[RecoveryOption, ...]
    selected_action: RecoveryAction | None
    escalation_reason: str | None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested trigger, checkpoint, and options.
        """
        return {
            "protocol_id": self.protocol_id,
            "trigger": self.trigger.to_dict(),
            "status": self.status,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "recovery_options": [o.to_dict() for o in self.recovery_options],
            "selected_action": self.selected_action,
            "escalation_reason": self.escalation_reason,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass(frozen=True)
class EmergencyConfig:
    """Configuration for emergency protocol behavior.

    Controls automatic recovery and escalation thresholds.

    Attributes:
        auto_checkpoint: Whether to automatically checkpoint state (default True)
        max_recovery_attempts: Maximum automatic recovery attempts (default 3)
        escalation_threshold: Confidence below which to escalate (default 0.5)
        enable_auto_recovery: Whether to enable automatic recovery (default True)

    Example:
        >>> config = EmergencyConfig(max_recovery_attempts=5)
        >>> config.max_recovery_attempts
        5
    """

    auto_checkpoint: bool = True
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD
    enable_auto_recovery: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "auto_checkpoint": self.auto_checkpoint,
            "max_recovery_attempts": self.max_recovery_attempts,
            "escalation_threshold": self.escalation_threshold,
            "enable_auto_recovery": self.enable_auto_recovery,
        }
