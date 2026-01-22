"""In-memory cost store implementation (Story 11.6).

This module provides an in-memory implementation of the CostStore protocol.
It's suitable for testing and single-session use cases.

The implementation uses thread-safe storage with threading.Lock for concurrent access.

Example:
    >>> from yolo_developer.audit.cost_memory_store import InMemoryCostStore
    >>> from yolo_developer.audit.cost_types import CostRecord, TokenUsage
    >>>
    >>> store = InMemoryCostStore()
    >>> usage = TokenUsage(100, 50, 150)
    >>> record = CostRecord(
    ...     id="cost-001",
    ...     timestamp="2026-01-18T12:00:00Z",
    ...     model="gpt-5.2-instant",
    ...     tier="routine",
    ...     token_usage=usage,
    ...     cost_usd=0.0015,
    ...     agent_name="analyst",
    ...     session_id="session-123",
    ... )
    >>> await store.store_cost(record)

References:
    - FR86: System can track token usage and cost per operation
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from yolo_developer.audit.cost_types import VALID_GROUPBY_VALUES, CostAggregation

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from yolo_developer.audit.cost_store import CostFilters
    from yolo_developer.audit.cost_types import CostRecord


class InMemoryCostStore:
    """In-memory implementation of the CostStore protocol.

    Stores cost records in memory with thread-safe access.

    This implementation is suitable for:
    - Unit and integration testing
    - Single-session use cases
    - Development and debugging

    For persistent storage, use a file-based or database implementation.

    Attributes:
        _costs: Internal storage mapping cost IDs to CostRecord objects.
        _lock: Threading lock for concurrent access safety.

    Example:
        >>> store = InMemoryCostStore()
        >>> await store.store_cost(record)
        >>> costs = await store.get_costs()
    """

    def __init__(self) -> None:
        """Initialize the in-memory cost store."""
        self._costs: dict[str, CostRecord] = {}
        self._lock = threading.Lock()

    async def store_cost(self, record: CostRecord) -> None:
        """Store a cost record.

        Thread-safe operation that stores the record in memory.

        Args:
            record: The CostRecord to store.
        """
        with self._lock:
            self._costs[record.id] = record

    async def get_cost(self, cost_id: str) -> CostRecord | None:
        """Retrieve a cost record by its ID.

        Thread-safe operation that looks up a record by ID.

        Args:
            cost_id: The ID of the cost record to retrieve.

        Returns:
            The CostRecord if found, None otherwise.
        """
        with self._lock:
            return self._costs.get(cost_id)

    async def get_costs(
        self,
        filters: CostFilters | None = None,
    ) -> list[CostRecord]:
        """Retrieve cost records with optional filtering.

        Applies filters to find matching records.

        Args:
            filters: Optional filters to apply. None returns all records.

        Returns:
            List of matching CostRecords.
        """
        with self._lock:
            costs = list(self._costs.values())

        # Apply filters if provided
        if filters is not None:
            costs = [c for c in costs if self._matches_filters(c, filters)]

        return costs

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
        costs = await self.get_costs(filters)
        return self._aggregate(costs)

    async def get_grouped_aggregation(
        self,
        group_by: str,
        filters: CostFilters | None = None,
    ) -> dict[str, CostAggregation]:
        """Get aggregated costs grouped by a dimension.

        Computes separate aggregations for each unique value
        of the specified dimension (agent, story, sprint, model, tier).

        Args:
            group_by: Dimension to group by (agent, story, sprint, model, tier).
            filters: Optional filters to apply before grouping.

        Returns:
            Dictionary mapping group values to their aggregations.
        """
        costs = await self.get_costs(filters)

        # Group costs by the specified dimension
        groups: dict[str, list[CostRecord]] = {}
        for cost in costs:
            key = self._get_group_key(cost, group_by)
            if key is not None:  # Skip None values (e.g., no story_id)
                if key not in groups:
                    groups[key] = []
                groups[key].append(cost)

        # Aggregate each group
        return {key: self._aggregate(records) for key, records in groups.items()}

    def _matches_filters(self, record: CostRecord, filters: CostFilters) -> bool:
        """Check if a cost record matches the given filters.

        Args:
            record: The CostRecord to check.
            filters: The filters to apply.

        Returns:
            True if the record matches all filters, False otherwise.
        """
        # Check agent_name filter
        if filters.agent_name is not None and record.agent_name != filters.agent_name:
            return False

        # Check story_id filter
        if filters.story_id is not None and record.story_id != filters.story_id:
            return False

        # Check sprint_id filter
        if filters.sprint_id is not None and record.sprint_id != filters.sprint_id:
            return False

        # Check session_id filter
        if filters.session_id is not None and record.session_id != filters.session_id:
            return False

        # Check model filter
        if filters.model is not None and record.model != filters.model:
            return False

        # Check tier filter
        if filters.tier is not None and record.tier != filters.tier:
            return False

        # Check start_time filter (inclusive)
        if filters.start_time is not None and record.timestamp < filters.start_time:
            return False

        # Check end_time filter (inclusive)
        if filters.end_time is not None and record.timestamp > filters.end_time:
            return False

        return True

    def _aggregate(self, costs: list[CostRecord]) -> CostAggregation:
        """Compute aggregated statistics for a list of cost records.

        Args:
            costs: List of CostRecords to aggregate.

        Returns:
            CostAggregation with computed totals.
        """
        if not costs:
            return CostAggregation(
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                call_count=0,
                models=(),
            )

        total_prompt_tokens = sum(c.token_usage.prompt_tokens for c in costs)
        total_completion_tokens = sum(c.token_usage.completion_tokens for c in costs)
        total_tokens = sum(c.token_usage.total_tokens for c in costs)
        total_cost_usd = sum(c.cost_usd for c in costs)
        call_count = len(costs)

        # Collect unique models, sorted for consistent ordering
        models = tuple(sorted({c.model for c in costs}))

        # Determine period boundaries if costs exist
        timestamps = [c.timestamp for c in costs]
        period_start = min(timestamps)
        period_end = max(timestamps)

        return CostAggregation(
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            call_count=call_count,
            models=models,
            period_start=period_start,
            period_end=period_end,
        )

    def _get_group_key(self, record: CostRecord, group_by: str) -> str | None:
        """Get the grouping key for a record based on the group_by dimension.

        Args:
            record: The CostRecord to get the key for.
            group_by: The dimension to group by.

        Returns:
            The grouping key value, or None if not applicable.
        """
        if group_by not in VALID_GROUPBY_VALUES:
            _logger.warning(
                "Invalid group_by value '%s'. Valid values: %s",
                group_by,
                VALID_GROUPBY_VALUES,
            )
            return None

        if group_by == "agent":
            return record.agent_name
        elif group_by == "story":
            return record.story_id
        elif group_by == "sprint":
            return record.sprint_id
        elif group_by == "model":
            return record.model
        elif group_by == "tier":
            return record.tier
        else:
            return None
