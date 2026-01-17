"""In-memory decision store implementation (Story 11.1).

This module provides an in-memory implementation of the DecisionStore protocol.
It's suitable for testing and single-session use cases.

The implementation uses thread-safe storage with threading.Lock for concurrent access.

Example:
    >>> from yolo_developer.audit.memory_store import InMemoryDecisionStore
    >>> from yolo_developer.audit.types import Decision, AgentIdentity, DecisionContext
    >>>
    >>> store = InMemoryDecisionStore()
    >>> decision = Decision(
    ...     id="dec-001",
    ...     decision_type="requirement_analysis",
    ...     content="OAuth2 authentication required",
    ...     rationale="Industry standard security",
    ...     agent=AgentIdentity("analyst", "analyst", "session-1"),
    ...     context=DecisionContext(),
    ...     timestamp="2026-01-17T12:00:00Z",
    ... )
    >>> await store.log_decision(decision)
    'dec-001'

References:
    - FR81: System can log all agent decisions with rationale
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yolo_developer.audit.store import DecisionFilters
    from yolo_developer.audit.types import Decision


class InMemoryDecisionStore:
    """In-memory implementation of the DecisionStore protocol.

    Stores decisions in memory with thread-safe access. Decisions are
    stored in a dictionary keyed by ID and returned in chronological
    order when queried.

    This implementation is suitable for:
    - Unit and integration testing
    - Single-session use cases
    - Development and debugging

    For persistent storage, use a file-based or database implementation.

    Attributes:
        _decisions: Internal storage mapping decision IDs to Decision objects.
        _lock: Threading lock for concurrent access safety.

    Example:
        >>> store = InMemoryDecisionStore()
        >>> await store.log_decision(decision)
        >>> count = await store.get_decision_count()
    """

    def __init__(self) -> None:
        """Initialize the in-memory decision store."""
        self._decisions: dict[str, Decision] = {}
        self._lock = threading.Lock()

    async def log_decision(self, decision: Decision) -> str:
        """Store a decision and return its ID.

        Thread-safe operation that stores the decision in memory.

        Args:
            decision: The Decision to store.

        Returns:
            The decision ID.
        """
        with self._lock:
            self._decisions[decision.id] = decision
        return decision.id

    async def get_decision(self, decision_id: str) -> Decision | None:
        """Retrieve a decision by its ID.

        Thread-safe operation that looks up a decision by ID.

        Args:
            decision_id: The ID of the decision to retrieve.

        Returns:
            The Decision if found, None otherwise.
        """
        with self._lock:
            return self._decisions.get(decision_id)

    async def get_decisions(
        self,
        filters: DecisionFilters | None = None,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Returns decisions in chronological order (oldest first).
        Applies filters if provided.

        Args:
            filters: Optional filters to apply. None returns all decisions.

        Returns:
            List of matching Decision objects, ordered by timestamp.
        """
        with self._lock:
            decisions = list(self._decisions.values())

        # Apply filters if provided
        if filters is not None:
            decisions = self._apply_filters(decisions, filters)

        # Sort by timestamp (chronological order)
        decisions.sort(key=lambda d: d.timestamp)

        return decisions

    async def get_decision_count(self) -> int:
        """Get the total number of stored decisions.

        Returns:
            Count of all stored decisions.
        """
        with self._lock:
            return len(self._decisions)

    def _apply_filters(
        self,
        decisions: list[Decision],
        filters: DecisionFilters,
    ) -> list[Decision]:
        """Apply filters to a list of decisions.

        Args:
            decisions: List of decisions to filter.
            filters: Filters to apply.

        Returns:
            Filtered list of decisions.
        """
        result = decisions

        # Filter by agent_name
        if filters.agent_name is not None:
            result = [d for d in result if d.agent.agent_name == filters.agent_name]

        # Filter by decision_type
        if filters.decision_type is not None:
            result = [d for d in result if d.decision_type == filters.decision_type]

        # Filter by start_time (inclusive)
        if filters.start_time is not None:
            result = [d for d in result if d.timestamp >= filters.start_time]

        # Filter by end_time (inclusive)
        if filters.end_time is not None:
            result = [d for d in result if d.timestamp <= filters.end_time]

        # Filter by sprint_id
        if filters.sprint_id is not None:
            result = [d for d in result if d.context.sprint_id == filters.sprint_id]

        # Filter by story_id
        if filters.story_id is not None:
            result = [d for d in result if d.context.story_id == filters.story_id]

        return result
