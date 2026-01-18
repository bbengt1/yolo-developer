"""Audit filter service for unified filtering across all stores (Story 11.7).

This module provides the AuditFilterService class that coordinates
filtering across all audit stores (decisions, traceability, costs).

The service enables unified querying of audit data with consistent
filter semantics across different store implementations.

Example:
    >>> from yolo_developer.audit.filter_service import (
    ...     AuditFilterService,
    ...     get_audit_filter_service,
    ... )
    >>> from yolo_developer.audit.filter_types import AuditFilters
    >>>
    >>> service = get_audit_filter_service(
    ...     decision_store=decision_store,
    ...     traceability_store=traceability_store,
    ...     cost_store=cost_store,
    ... )
    >>>
    >>> filters = AuditFilters(agent_name="analyst")
    >>> results = await service.filter_all(filters)
    >>> results["decisions"]  # Decision records for analyst
    >>> results["artifacts"]  # Traceable artifacts
    >>> results["costs"]  # Cost records for analyst

References:
    - FR87: Users can filter audit trail by agent, time range, or artifact
    - Story 11.1: DecisionStore pattern
    - Story 11.2: TraceabilityStore pattern
    - Story 11.6: CostStore pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.audit.cost_store import CostStore
    from yolo_developer.audit.cost_types import CostRecord
    from yolo_developer.audit.filter_types import AuditFilters
    from yolo_developer.audit.store import DecisionStore
    from yolo_developer.audit.traceability_store import TraceabilityStore
    from yolo_developer.audit.traceability_types import TraceableArtifact
    from yolo_developer.audit.types import Decision

_logger = structlog.get_logger(__name__)


class AuditFilterService:
    """Service for filtering audit data across all stores.

    Provides unified filtering interface that queries decisions,
    traceability, and cost data with combined filters.

    The service coordinates filtering across multiple audit stores,
    converting unified AuditFilters to store-specific filter types.

    Attributes:
        _decision_store: Store for decision records
        _traceability_store: Store for traceable artifacts
        _cost_store: Optional store for cost records

    Example:
        >>> service = AuditFilterService(
        ...     decision_store=decision_store,
        ...     traceability_store=traceability_store,
        ...     cost_store=cost_store,
        ... )
        >>> filters = AuditFilters(agent_name="analyst")
        >>> decisions = await service.filter_decisions(filters)
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        traceability_store: TraceabilityStore,
        cost_store: CostStore | None = None,
    ) -> None:
        """Initialize the audit filter service.

        Args:
            decision_store: Store for decision records.
            traceability_store: Store for traceable artifacts.
            cost_store: Optional store for cost records.
        """
        self._decision_store = decision_store
        self._traceability_store = traceability_store
        self._cost_store = cost_store
        _logger.debug(
            "audit_filter_service_initialized",
            has_cost_store=cost_store is not None,
        )

    async def filter_decisions(
        self,
        filters: AuditFilters,
    ) -> list[Decision]:
        """Filter decisions using unified filters.

        Converts AuditFilters to DecisionFilters and queries
        the decision store.

        Args:
            filters: Unified filters to apply.

        Returns:
            List of Decision records matching the filters.
        """
        decision_filters = filters.to_decision_filters()
        _logger.debug(
            "filtering_decisions",
            agent_name=filters.agent_name,
            decision_type=filters.decision_type,
            start_time=filters.start_time,
            end_time=filters.end_time,
        )
        return await self._decision_store.get_decisions(decision_filters)

    async def filter_artifacts(
        self,
        filters: AuditFilters,
    ) -> list[TraceableArtifact]:
        """Filter traceability artifacts using unified filters.

        Uses artifact_type and time range filters from AuditFilters
        to query the traceability store.

        Args:
            filters: Unified filters to apply.

        Returns:
            List of TraceableArtifact records matching the filters.
        """
        _logger.debug(
            "filtering_artifacts",
            artifact_type=filters.artifact_type,
            start_time=filters.start_time,
            end_time=filters.end_time,
        )
        traceability_filters = filters.to_traceability_filters()
        return await self._traceability_store.get_artifacts_by_filters(
            **traceability_filters,
        )

    async def filter_costs(
        self,
        filters: AuditFilters,
    ) -> list[CostRecord]:
        """Filter cost records using unified filters.

        Converts AuditFilters to CostFilters and queries
        the cost store. Returns empty list if no cost store
        is configured.

        Args:
            filters: Unified filters to apply.

        Returns:
            List of CostRecord records matching the filters,
            or empty list if no cost store is configured.
        """
        if self._cost_store is None:
            _logger.debug("cost_store_not_configured")
            return []

        cost_filters = filters.to_cost_filters()
        _logger.debug(
            "filtering_costs",
            agent_name=filters.agent_name,
            session_id=filters.session_id,
            start_time=filters.start_time,
            end_time=filters.end_time,
        )
        return await self._cost_store.get_costs(cost_filters)

    async def filter_all(
        self,
        filters: AuditFilters,
    ) -> dict[str, Any]:
        """Filter all audit data and return combined results.

        Queries all configured stores with the provided filters
        and returns a unified result dictionary.

        Args:
            filters: Unified filters to apply across all stores.

        Returns:
            Dictionary with keys:
                - 'decisions': List of matching Decision records
                - 'artifacts': List of matching TraceableArtifact records
                - 'costs': List of matching CostRecord records
                - 'filters_applied': Dictionary of filter values used
        """
        _logger.info(
            "filtering_all_audit_data",
            agent_name=filters.agent_name,
            artifact_type=filters.artifact_type,
            start_time=filters.start_time,
            end_time=filters.end_time,
        )

        decisions = await self.filter_decisions(filters)
        artifacts = await self.filter_artifacts(filters)
        costs = await self.filter_costs(filters)

        _logger.info(
            "filter_results",
            decision_count=len(decisions),
            artifact_count=len(artifacts),
            cost_count=len(costs),
        )

        return {
            "decisions": decisions,
            "artifacts": artifacts,
            "costs": costs,
            "filters_applied": filters.to_dict(),
        }


def get_audit_filter_service(
    decision_store: DecisionStore,
    traceability_store: TraceabilityStore,
    cost_store: CostStore | None = None,
) -> AuditFilterService:
    """Factory function to create AuditFilterService.

    Creates an AuditFilterService with the provided stores.

    Args:
        decision_store: Store for decision records.
        traceability_store: Store for traceable artifacts.
        cost_store: Optional store for cost records.

    Returns:
        Configured AuditFilterService instance.

    Example:
        >>> service = get_audit_filter_service(
        ...     decision_store=decision_store,
        ...     traceability_store=traceability_store,
        ... )
    """
    return AuditFilterService(
        decision_store=decision_store,
        traceability_store=traceability_store,
        cost_store=cost_store,
    )
