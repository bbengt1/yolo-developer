"""Decision store protocol and filters (Story 11.1).

This module defines the DecisionStore protocol and DecisionFilters dataclass
for storing and retrieving decision records.

The Protocol pattern enables:
- Easy mocking in tests
- Future implementations (JSON file, SQLite, etc.)
- Dependency injection

Example:
    >>> from yolo_developer.audit.store import DecisionStore, DecisionFilters
    >>>
    >>> # DecisionStore is a Protocol - implementations must provide these methods
    >>> class MyStore(DecisionStore):
    ...     async def log_decision(self, decision: Decision) -> str: ...
    ...     async def get_decision(self, decision_id: str) -> Decision | None: ...
    ...     async def get_decisions(self, filters: DecisionFilters | None = None) -> list[Decision]: ...
    ...     async def get_decision_count(self) -> int: ...

References:
    - FR81: System can log all agent decisions with rationale
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.types import Decision


# =============================================================================
# Filter Dataclass (Subtask 2.3)
# =============================================================================


@dataclass(frozen=True)
class DecisionFilters:
    """Filters for querying decisions from the store.

    All fields are optional; None means no filtering on that field.

    Attributes:
        agent_name: Filter by agent name
        decision_type: Filter by decision type
        start_time: Filter decisions after this timestamp (inclusive)
        end_time: Filter decisions before this timestamp (inclusive)
        sprint_id: Filter by sprint ID
        story_id: Filter by story ID

    Example:
        >>> filters = DecisionFilters(
        ...     agent_name="analyst",
        ...     sprint_id="sprint-1",
        ... )
        >>> decisions = await store.get_decisions(filters)
    """

    agent_name: str | None = None
    decision_type: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    sprint_id: str | None = None
    story_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the filters.
        """
        return {
            "agent_name": self.agent_name,
            "decision_type": self.decision_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sprint_id": self.sprint_id,
            "story_id": self.story_id,
        }


# =============================================================================
# Store Protocol (Subtasks 2.1, 2.2)
# =============================================================================


@runtime_checkable
class DecisionStore(Protocol):
    """Protocol for decision storage implementations.

    Defines the interface for storing and retrieving decision records.
    Implementations can use various backends (memory, file, database).

    Methods:
        log_decision: Store a decision and return its ID
        get_decision: Retrieve a decision by ID
        get_decisions: Query decisions with optional filters
        get_decision_count: Get total number of stored decisions

    Example:
        >>> class InMemoryDecisionStore:
        ...     async def log_decision(self, decision: Decision) -> str:
        ...         # Store decision
        ...         return decision.id
    """

    async def log_decision(self, decision: Decision) -> str:
        """Store a decision and return its ID.

        Args:
            decision: The Decision to store.

        Returns:
            The decision ID.
        """
        ...

    async def get_decision(self, decision_id: str) -> Decision | None:
        """Retrieve a decision by its ID.

        Args:
            decision_id: The ID of the decision to retrieve.

        Returns:
            The Decision if found, None otherwise.
        """
        ...

    async def get_decisions(
        self,
        filters: DecisionFilters | None = None,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Returns decisions in chronological order (oldest first).

        Args:
            filters: Optional filters to apply. None returns all decisions.

        Returns:
            List of matching Decision objects, ordered by timestamp.
        """
        ...

    async def get_decision_count(self) -> int:
        """Get the total number of stored decisions.

        Returns:
            Count of all stored decisions.
        """
        ...
