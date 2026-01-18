"""Cost store protocol and filters (Story 11.6).

This module defines the CostStore protocol and CostFilters dataclass
for storing and retrieving cost records.

The Protocol pattern enables:
- Easy mocking in tests
- Future implementations (JSON file, SQLite, etc.)
- Dependency injection

Example:
    >>> from yolo_developer.audit.cost_store import CostStore, CostFilters
    >>>
    >>> # CostStore is a Protocol - implementations must provide these methods
    >>> class MyCostStore(CostStore):
    ...     async def store_cost(self, record: CostRecord) -> None: ...
    ...     async def get_cost(self, cost_id: str) -> CostRecord | None: ...
    ...     # ... other methods

References:
    - FR86: System can track token usage and cost per operation
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.cost_types import CostAggregation, CostRecord


# =============================================================================
# Filter Dataclass (Subtask 2.1)
# =============================================================================


@dataclass(frozen=True)
class CostFilters:
    """Filters for querying cost records from the store.

    All fields are optional; None means no filtering on that field.

    Attributes:
        agent_name: Filter by agent name
        story_id: Filter by story ID
        sprint_id: Filter by sprint ID
        session_id: Filter by session ID
        model: Filter by model name
        tier: Filter by model tier (routine, complex, critical)
        start_time: Filter costs after this timestamp (inclusive)
        end_time: Filter costs before this timestamp (inclusive)

    Example:
        >>> filters = CostFilters(
        ...     agent_name="analyst",
        ...     session_id="session-123",
        ... )
        >>> costs = await store.get_costs(filters)
    """

    agent_name: str | None = None
    story_id: str | None = None
    sprint_id: str | None = None
    session_id: str | None = None
    model: str | None = None
    tier: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the filters.
        """
        return {
            "agent_name": self.agent_name,
            "story_id": self.story_id,
            "sprint_id": self.sprint_id,
            "session_id": self.session_id,
            "model": self.model,
            "tier": self.tier,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# =============================================================================
# Store Protocol (Subtasks 2.2, 2.3)
# =============================================================================


@runtime_checkable
class CostStore(Protocol):
    """Protocol for cost record storage implementations.

    Defines the interface for storing and retrieving cost data.
    Implementations can use various backends (memory, file, database).

    Methods:
        store_cost: Store a cost record
        get_cost: Retrieve a cost record by ID
        get_costs: Retrieve cost records with optional filtering
        get_aggregation: Get aggregated cost statistics
        get_grouped_aggregation: Get aggregated costs grouped by a dimension

    Example:
        >>> class InMemoryCostStore:
        ...     async def store_cost(self, record: CostRecord) -> None:
        ...         # Store record
        ...         pass
    """

    async def store_cost(self, record: CostRecord) -> None:
        """Store a cost record.

        Args:
            record: The CostRecord to store.
        """
        ...

    async def get_cost(self, cost_id: str) -> CostRecord | None:
        """Retrieve a cost record by its ID.

        Args:
            cost_id: The ID of the cost record to retrieve.

        Returns:
            The CostRecord if found, None otherwise.
        """
        ...

    async def get_costs(
        self,
        filters: CostFilters | None = None,
    ) -> list[CostRecord]:
        """Retrieve cost records with optional filtering.

        Args:
            filters: Optional filters to apply. None returns all records.

        Returns:
            List of matching CostRecords.
        """
        ...

    async def get_aggregation(
        self,
        filters: CostFilters | None = None,
    ) -> CostAggregation:
        """Get aggregated cost statistics.

        Computes totals across all matching cost records.

        Args:
            filters: Optional filters to apply before aggregation.

        Returns:
            CostAggregation with totals for matching records.
        """
        ...

    async def get_grouped_aggregation(
        self,
        group_by: str,
        filters: CostFilters | None = None,
    ) -> dict[str, CostAggregation]:
        """Get aggregated costs grouped by a dimension.

        Computes separate aggregations for each unique value
        of the specified dimension (agent, story, sprint, model, tier).

        Args:
            group_by: Dimension to group by (from CostGroupBy).
            filters: Optional filters to apply before grouping.

        Returns:
            Dictionary mapping group values to their aggregations.
            Example: {"analyst": CostAggregation(...), "pm": CostAggregation(...)}
        """
        ...
