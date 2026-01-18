"""Audit view service for human-readable output (Story 11.3).

This module provides the AuditViewService class that orchestrates
viewing decisions, trace chains, coverage reports, and summaries.

Example:
    >>> from yolo_developer.audit import (
    ...     AuditViewService,
    ...     InMemoryDecisionStore,
    ...     InMemoryTraceabilityStore,
    ...     get_audit_view_service,
    ... )
    >>>
    >>> decision_store = InMemoryDecisionStore()
    >>> traceability_store = InMemoryTraceabilityStore()
    >>> service = get_audit_view_service(decision_store, traceability_store)
    >>>
    >>> output = await service.view_decisions()
    >>> print(output)

References:
    - FR83: Users can view audit trail in human-readable format
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

from yolo_developer.audit.formatter_types import FormatOptions
from yolo_developer.audit.plain_formatter import PlainAuditFormatter

if TYPE_CHECKING:
    from yolo_developer.audit.formatter_protocol import AuditFormatter
    from yolo_developer.audit.store import DecisionFilters, DecisionStore
    from yolo_developer.audit.traceability_store import TraceabilityStore

_logger = structlog.get_logger(__name__)


class AuditViewService:
    """Service for viewing audit data in human-readable formats.

    Orchestrates retrieval from decision and traceability stores,
    formatting output using the configured formatter.

    Attributes:
        _decision_store: Store for decision data
        _traceability_store: Store for traceability data
        _formatter: Formatter for output rendering

    Example:
        >>> service = AuditViewService(
        ...     decision_store=decision_store,
        ...     traceability_store=traceability_store,
        ... )
        >>> output = await service.view_decisions()
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        traceability_store: TraceabilityStore,
        formatter: AuditFormatter | None = None,
    ) -> None:
        """Initialize the audit view service.

        Args:
            decision_store: Store for retrieving decisions.
            traceability_store: Store for retrieving traceability data.
            formatter: Optional formatter (uses PlainAuditFormatter if None).
        """
        self._decision_store = decision_store
        self._traceability_store = traceability_store
        self._formatter: AuditFormatter = formatter or PlainAuditFormatter()

    async def view_decisions(
        self,
        filters: DecisionFilters | None = None,
        options: FormatOptions | None = None,
    ) -> str:
        """View all decisions with optional filtering.

        Args:
            filters: Optional filters to apply.
            options: Optional format options.

        Returns:
            Formatted string representation of decisions.
        """
        _logger.debug("viewing_decisions", filters=filters, options=options)

        decisions = await self._decision_store.get_decisions(filters)
        return self._formatter.format_decisions(decisions, options)

    async def view_decision(
        self,
        decision_id: str,
        options: FormatOptions | None = None,
    ) -> str:
        """View a single decision by ID.

        Args:
            decision_id: The decision ID to retrieve.
            options: Optional format options.

        Returns:
            Formatted string representation of the decision.
        """
        _logger.debug("viewing_decision", decision_id=decision_id)

        decision = await self._decision_store.get_decision(decision_id)
        if decision is None:
            return f"Decision not found: {decision_id}"

        return self._formatter.format_decision(decision, options)

    async def view_trace_chain(
        self,
        artifact_id: str,
        direction: Literal["upstream", "downstream"],
        options: FormatOptions | None = None,
    ) -> str:
        """View trace chain for an artifact.

        Args:
            artifact_id: The starting artifact ID.
            direction: Direction to traverse (upstream/downstream).
            options: Optional format options.

        Returns:
            Formatted string representation of the trace chain.
        """
        _logger.debug("viewing_trace_chain", artifact_id=artifact_id, direction=direction)

        # Get the starting artifact
        start_artifact = await self._traceability_store.get_artifact(artifact_id)

        # Get the trace chain
        chain_artifacts = await self._traceability_store.get_trace_chain(artifact_id, direction)

        # Include start artifact if found
        artifacts = []
        if start_artifact:
            artifacts.append(start_artifact)
        artifacts.extend(chain_artifacts)

        # Get all links involving these artifacts
        links = []
        for artifact in artifacts:
            from_links = await self._traceability_store.get_links_from(artifact.id)
            to_links = await self._traceability_store.get_links_to(artifact.id)
            links.extend(from_links)
            links.extend(to_links)

        # Deduplicate links
        seen_link_ids: set[str] = set()
        unique_links = []
        for link in links:
            if link.id not in seen_link_ids:
                seen_link_ids.add(link.id)
                unique_links.append(link)

        return self._formatter.format_trace_chain(artifacts, unique_links, options)

    async def view_coverage(
        self,
        options: FormatOptions | None = None,
    ) -> str:
        """View coverage report showing requirement coverage.

        Args:
            options: Optional format options.

        Returns:
            Formatted string representation of coverage statistics.
        """
        _logger.debug("viewing_coverage")

        # Get requirements with no outgoing links (in our model, requirements don't
        # have outgoing links - stories link TO requirements, so this returns ALL requirements)
        all_requirements = await self._traceability_store.get_unlinked_artifacts(
            "requirement"
        )
        total_requirements = len(all_requirements)

        # Find covered vs uncovered requirements
        # A requirement is "covered" if something links TO it (i.e., has incoming links)
        covered_count = 0
        unlinked_requirements = []

        for req in all_requirements:
            links_to = await self._traceability_store.get_links_to(req.id)
            if links_to:
                covered_count += 1
            else:
                unlinked_requirements.append(req)

        unlinked_count = len(unlinked_requirements)

        # Calculate coverage percentage
        coverage_percentage = (
            (covered_count / total_requirements * 100) if total_requirements > 0 else 0.0
        )

        # Build coverage report
        report = {
            "total_requirements": total_requirements,
            "covered_requirements": covered_count,
            "coverage_percentage": coverage_percentage,
            "unlinked_requirements": [r.id for r in unlinked_requirements],
            "unlinked_count": unlinked_count,
        }

        return self._formatter.format_coverage_report(report, options)

    async def view_summary(
        self,
        filters: DecisionFilters | None = None,
        options: FormatOptions | None = None,
    ) -> str:
        """View summary statistics for decisions.

        Args:
            filters: Optional filters to apply.
            options: Optional format options.

        Returns:
            Formatted string with summary statistics.
        """
        _logger.debug("viewing_summary", filters=filters)

        decisions = await self._decision_store.get_decisions(filters)
        return self._formatter.format_summary(decisions, options)


def get_audit_view_service(
    decision_store: DecisionStore,
    traceability_store: TraceabilityStore,
    formatter: AuditFormatter | None = None,
) -> AuditViewService:
    """Factory function to create AuditViewService.

    Args:
        decision_store: Store for retrieving decisions.
        traceability_store: Store for retrieving traceability data.
        formatter: Optional formatter (uses PlainAuditFormatter if None).

    Returns:
        Configured AuditViewService instance.
    """
    return AuditViewService(
        decision_store=decision_store,
        traceability_store=traceability_store,
        formatter=formatter,
    )
