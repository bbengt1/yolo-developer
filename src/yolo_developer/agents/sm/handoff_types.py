"""Type definitions for agent handoff management (Story 10.8).

This module defines the data types used by the handoff management system
for preserving context during agent transitions in the orchestration workflow.

Key Types:
- HandoffStatus: Lifecycle states for a handoff operation
- HandoffMetrics: Performance metrics captured during handoff
- HandoffRecord: Complete record of a handoff event for audit trail
- HandoffResult: Result of a manage_handoff() operation
- HandoffConfig: Configuration for handoff behavior

Architecture Notes:
- All dataclasses are frozen for immutability (per ADR-001)
- to_dict() methods provided for serialization to state
- Constants define sensible defaults per NFR-PERF-1

References:
- FR14: System can execute agents in defined sequence
- FR15: System can handle agent handoffs with context preservation
- NFR-PERF-1: Agent handoff latency <5 seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types (Task 1.2)
# =============================================================================

HandoffStatus = Literal["pending", "in_progress", "completed", "failed"]
"""Status of a handoff operation through its lifecycle."""

# =============================================================================
# Constants (Task 1.8)
# =============================================================================

DEFAULT_TIMEOUT_SECONDS: int = 5
"""Default timeout for handoff operations per NFR-PERF-1 (<5s latency)."""

DEFAULT_MAX_CONTEXT_SIZE: int = 1_000_000
"""Default maximum context size in bytes (1MB)."""

VALID_HANDOFF_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed", "failed"}
)
"""Valid handoff status values for validation."""


# =============================================================================
# Dataclasses (Tasks 1.3-1.7)
# =============================================================================


@dataclass(frozen=True)
class HandoffMetrics:
    """Metrics captured during a handoff operation.

    Measures timing and size characteristics for performance monitoring
    and NFR-PERF-1 compliance verification.

    Attributes:
        duration_ms: Time taken for the handoff operation in milliseconds.
        context_size_bytes: Size of the transferred context in bytes.
        messages_transferred: Number of messages included in handoff.
        decisions_transferred: Number of decisions included in handoff.
        memory_refs_transferred: Number of memory references included.

    Example:
        >>> metrics = HandoffMetrics(
        ...     duration_ms=150.5,
        ...     context_size_bytes=1024,
        ...     messages_transferred=10,
        ...     decisions_transferred=3,
        ... )
        >>> metrics.duration_ms
        150.5
    """

    duration_ms: float
    context_size_bytes: int
    messages_transferred: int
    decisions_transferred: int
    memory_refs_transferred: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the metrics.
        """
        return {
            "duration_ms": self.duration_ms,
            "context_size_bytes": self.context_size_bytes,
            "messages_transferred": self.messages_transferred,
            "decisions_transferred": self.decisions_transferred,
            "memory_refs_transferred": self.memory_refs_transferred,
        }


@dataclass(frozen=True)
class HandoffRecord:
    """Record of a single handoff event.

    Captures the full lifecycle of a handoff for audit trail and
    observability. Includes timing, participants, and outcome.

    Attributes:
        handoff_id: Unique identifier for this handoff event.
        source_agent: Agent handing off work (e.g., "analyst").
        target_agent: Agent receiving work (e.g., "pm").
        status: Current lifecycle status of the handoff.
        started_at: ISO timestamp when handoff started.
        completed_at: ISO timestamp when handoff completed (if done).
        metrics: Performance metrics captured during handoff.
        context_checksum: SHA-256 checksum of transferred context.
        error_message: Error details if handoff failed.

    Example:
        >>> record = HandoffRecord(
        ...     handoff_id="handoff_analyst_pm_123",
        ...     source_agent="analyst",
        ...     target_agent="pm",
        ...     status="completed",
        ... )
        >>> record.status
        'completed'
    """

    handoff_id: str
    source_agent: str
    target_agent: str
    status: HandoffStatus
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    metrics: HandoffMetrics | None = None
    context_checksum: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the record.
        """
        return {
            "handoff_id": self.handoff_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "context_checksum": self.context_checksum,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class HandoffResult:
    """Result of a managed handoff operation.

    Returned by manage_handoff() with complete outcome details including
    the handoff record, success status, and state updates to apply.

    Attributes:
        record: The HandoffRecord capturing the handoff event.
        success: Whether the handoff completed successfully.
        context_validated: Whether context integrity was validated.
        state_updates: Dictionary of state updates to apply (if successful).
        warnings: Any warnings encountered during handoff.

    Example:
        >>> result = HandoffResult(
        ...     record=record,
        ...     success=True,
        ...     context_validated=True,
        ...     state_updates={"current_agent": "pm"},
        ... )
        >>> result.success
        True
    """

    record: HandoffRecord
    success: bool
    context_validated: bool
    state_updates: dict[str, Any] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "record": self.record.to_dict(),
            "success": self.success,
            "context_validated": self.context_validated,
            "state_updates": self.state_updates,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HandoffConfig:
    """Configuration for handoff management.

    Allows customization of validation, timing, and context transfer
    behavior for managed handoffs.

    Attributes:
        validate_context_integrity: Whether to validate state checksums.
        log_timing: Whether to log timing metrics.
        timeout_seconds: Maximum time for handoff operation.
        max_context_size_bytes: Maximum allowed context size.
        include_all_messages: Include all messages vs. recent only.
        max_messages_to_transfer: Maximum messages to include.

    Example:
        >>> config = HandoffConfig(timeout_seconds=10.0)
        >>> config.timeout_seconds
        10.0
    """

    validate_context_integrity: bool = True
    log_timing: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_context_size_bytes: int = DEFAULT_MAX_CONTEXT_SIZE
    include_all_messages: bool = False
    max_messages_to_transfer: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "validate_context_integrity": self.validate_context_integrity,
            "log_timing": self.log_timing,
            "timeout_seconds": self.timeout_seconds,
            "max_context_size_bytes": self.max_context_size_bytes,
            "include_all_messages": self.include_all_messages,
            "max_messages_to_transfer": self.max_messages_to_transfer,
        }
