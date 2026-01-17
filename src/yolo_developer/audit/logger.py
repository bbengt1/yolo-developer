"""Decision logger implementation (Story 11.1).

This module provides the DecisionLogger class for logging agent decisions
with automatic ID generation, timestamping, and structured logging.

Example:
    >>> from yolo_developer.audit.logger import DecisionLogger, get_logger
    >>> from yolo_developer.audit.memory_store import InMemoryDecisionStore
    >>>
    >>> store = InMemoryDecisionStore()
    >>> logger = get_logger(store)
    >>>
    >>> decision_id = await logger.log(
    ...     agent_name="analyst",
    ...     agent_type="analyst",
    ...     decision_type="requirement_analysis",
    ...     content="OAuth2 authentication required",
    ...     rationale="Industry standard security",
    ... )

References:
    - FR81: System can log all agent decisions with rationale
    - ADR-007: Log errors, don't block callers
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.audit.store import DecisionStore
    from yolo_developer.audit.types import (
        DecisionContext,
        DecisionSeverity,
        DecisionType,
    )

_logger = structlog.get_logger(__name__)


class DecisionLogger:
    """Logger for agent decisions.

    Provides a high-level interface for logging decisions with automatic
    generation of IDs, timestamps, and session IDs.

    Attributes:
        _store: The DecisionStore to persist decisions.
        _session_id: Session ID for grouping related decisions.

    Example:
        >>> logger = DecisionLogger(store)
        >>> decision_id = await logger.log(
        ...     agent_name="analyst",
        ...     agent_type="analyst",
        ...     decision_type="requirement_analysis",
        ...     content="Decision content",
        ...     rationale="Decision rationale",
        ... )
    """

    def __init__(
        self,
        store: DecisionStore,
        session_id: str | None = None,
    ) -> None:
        """Initialize the decision logger.

        Args:
            store: The DecisionStore to persist decisions.
            session_id: Optional session ID. If not provided, generates a new UUID.
        """
        self._store = store
        self._session_id = session_id or str(uuid.uuid4())

    async def log(
        self,
        agent_name: str,
        agent_type: str,
        decision_type: DecisionType,
        content: str,
        rationale: str,
        context: DecisionContext | None = None,
        metadata: dict[str, Any] | None = None,
        severity: DecisionSeverity = "info",
    ) -> str:
        """Log a decision and return its ID.

        Creates a Decision object with auto-generated ID and timestamp,
        stores it in the decision store, and emits structured log output.

        Args:
            agent_name: Human-readable agent name (e.g., "analyst").
            agent_type: Agent type for categorization (e.g., "analyst", "pm").
            decision_type: Type of decision being made.
            content: What was decided.
            rationale: Why this decision was made.
            context: Optional context (sprint, story, etc.).
            metadata: Optional additional key-value data.
            severity: Decision importance level (default: "info").

        Returns:
            The decision ID (UUID).
        """
        from yolo_developer.audit.types import (
            AgentIdentity,
            Decision,
            DecisionContext,
        )

        # Auto-generate ID and timestamp
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Create agent identity
        agent = AgentIdentity(
            agent_name=agent_name,
            agent_type=agent_type,
            session_id=self._session_id,
        )

        # Use provided context or create empty one
        decision_context = context or DecisionContext()

        # Create decision
        decision = Decision(
            id=decision_id,
            decision_type=decision_type,
            content=content,
            rationale=rationale,
            agent=agent,
            context=decision_context,
            timestamp=timestamp,
            metadata=metadata or {},
            severity=severity,
        )

        # Store the decision (per ADR-007: log errors, don't block callers)
        try:
            await self._store.log_decision(decision)
        except Exception:
            _logger.exception(
                "decision_store_failed",
                decision_id=decision_id,
                agent_name=agent_name,
                decision_type=decision_type,
            )
            # Still return the decision_id - decision was created but may not be persisted

        # Emit structured log
        _logger.info(
            "decision_logged",
            decision_id=decision_id,
            agent_name=agent_name,
            agent_type=agent_type,
            decision_type=decision_type,
            severity=severity,
            content_preview=content[:100] if len(content) > 100 else content,
        )

        return decision_id


def get_logger(
    store: DecisionStore | None = None,
    session_id: str | None = None,
) -> DecisionLogger:
    """Factory function to create a DecisionLogger.

    Creates a DecisionLogger with the provided store, or creates a default
    InMemoryDecisionStore if none is provided.

    Args:
        store: Optional DecisionStore. If not provided, creates InMemoryDecisionStore.
        session_id: Optional session ID for grouping decisions.

    Returns:
        A configured DecisionLogger instance.

    Example:
        >>> logger = get_logger()  # Uses default InMemoryDecisionStore
        >>> logger = get_logger(my_store, session_id="my-session")
    """
    if store is None:
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()

    return DecisionLogger(store, session_id=session_id)
