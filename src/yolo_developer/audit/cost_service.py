"""Cost tracking service for LLM token and cost monitoring (Story 11.6).

This module provides the CostTrackingService for recording and querying
LLM token usage and costs:

- CostTrackingService: Main service for cost tracking operations
- get_cost_tracking_service: Factory function for creating service instances

Example:
    >>> from yolo_developer.audit.cost_service import get_cost_tracking_service
    >>> from yolo_developer.audit.cost_memory_store import InMemoryCostStore
    >>>
    >>> store = InMemoryCostStore()
    >>> service = get_cost_tracking_service(store)
    >>> record = await service.record_llm_call(
    ...     model="gpt-4o-mini",
    ...     tier="routine",
    ...     prompt_tokens=100,
    ...     completion_tokens=50,
    ...     cost_usd=0.0015,
    ...     agent_name="analyst",
    ...     session_id="session-123",
    ... )

References:
    - FR86: System can track token usage and cost per operation
    - AC #1: Tokens per call are recorded
    - AC #2: Costs are calculated
    - AC #3: Per-agent breakdown is available
    - AC #4: Per-story breakdown is available
    - AC #5: Totals are aggregated
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.audit.cost_memory_store import InMemoryCostStore
from yolo_developer.audit.cost_store import CostFilters
from yolo_developer.audit.cost_types import CostAggregation, CostRecord, TokenUsage

if TYPE_CHECKING:
    from yolo_developer.audit.cost_store import CostStore

logger = structlog.get_logger(__name__)


class CostTrackingService:
    """Service for tracking LLM token usage and costs.

    Provides methods for recording LLM calls and querying cost data
    with breakdowns by agent, story, session, and sprint.

    Attributes:
        _cost_store: Store for cost records
        _enabled: Whether tracking is enabled

    Example:
        >>> service = CostTrackingService(cost_store)
        >>> record = await service.record_llm_call(
        ...     model="gpt-4o-mini",
        ...     tier="routine",
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     cost_usd=0.0015,
        ...     agent_name="analyst",
        ...     session_id="session-123",
        ... )
    """

    def __init__(
        self,
        cost_store: CostStore,
        enabled: bool = True,
    ) -> None:
        """Initialize the cost tracking service.

        Args:
            cost_store: Store for cost records
            enabled: Whether to enable tracking (default: True)
        """
        self._cost_store = cost_store
        self._enabled = enabled

    async def record_llm_call(
        self,
        model: str,
        tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        agent_name: str,
        session_id: str,
        story_id: str | None = None,
        sprint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record token usage and cost for an LLM call.

        Creates a CostRecord with the provided data and stores it
        if tracking is enabled.

        Args:
            model: LLM model identifier (e.g., "gpt-4o-mini")
            tier: Model tier (routine, complex, critical)
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost_usd: Cost in USD
            agent_name: Name of the agent making the call
            session_id: Session identifier
            story_id: Story identifier (optional)
            sprint_id: Sprint identifier (optional)
            metadata: Additional metadata (optional)

        Returns:
            The created CostRecord
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        record = CostRecord(
            id=record_id,
            timestamp=timestamp,
            model=model,
            tier=tier,
            token_usage=token_usage,
            cost_usd=cost_usd,
            agent_name=agent_name,
            session_id=session_id,
            story_id=story_id,
            sprint_id=sprint_id,
            metadata=metadata or {},
        )

        if self._enabled:
            await self._cost_store.store_cost(record)

            logger.info(
                "recorded_llm_call",
                record_id=record_id,
                model=model,
                tier=tier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                agent_name=agent_name,
                session_id=session_id,
                story_id=story_id,
            )

        return record

    async def get_agent_costs(
        self,
        agent_name: str | None = None,
    ) -> dict[str, CostAggregation]:
        """Get cost breakdown by agent.

        Returns aggregated costs grouped by agent name.
        Optionally filter to a specific agent.

        Args:
            agent_name: Filter to specific agent (optional, returns all if None)

        Returns:
            Dictionary mapping agent names to their cost aggregations
        """
        filters = CostFilters(agent_name=agent_name) if agent_name else None
        return await self._cost_store.get_grouped_aggregation("agent", filters)

    async def get_story_costs(
        self,
        story_id: str | None = None,
    ) -> dict[str, CostAggregation]:
        """Get cost breakdown by story.

        Returns aggregated costs grouped by story ID.
        Optionally filter to a specific story.

        Args:
            story_id: Filter to specific story (optional, returns all if None)

        Returns:
            Dictionary mapping story IDs to their cost aggregations
        """
        filters = CostFilters(story_id=story_id) if story_id else None
        return await self._cost_store.get_grouped_aggregation("story", filters)

    async def get_session_total(
        self,
        session_id: str,
    ) -> CostAggregation:
        """Get total costs for a session.

        Computes aggregated totals for all LLM calls in the specified session.

        Args:
            session_id: Session ID to get totals for

        Returns:
            CostAggregation with session totals
        """
        filters = CostFilters(session_id=session_id)
        return await self._cost_store.get_aggregation(filters)

    async def get_sprint_total(
        self,
        sprint_id: str,
    ) -> CostAggregation:
        """Get total costs for a sprint.

        Computes aggregated totals for all LLM calls in the specified sprint.

        Args:
            sprint_id: Sprint ID to get totals for

        Returns:
            CostAggregation with sprint totals
        """
        filters = CostFilters(sprint_id=sprint_id)
        return await self._cost_store.get_aggregation(filters)


def get_cost_tracking_service(
    cost_store: CostStore | None = None,
    enabled: bool = True,
) -> CostTrackingService:
    """Factory function to create a CostTrackingService instance.

    Args:
        cost_store: Store for cost records (default: InMemoryCostStore)
        enabled: Whether to enable tracking (default: True)

    Returns:
        Configured CostTrackingService instance
    """
    if cost_store is None:
        cost_store = InMemoryCostStore()

    return CostTrackingService(cost_store, enabled)
