"""Unified audit filter types (Story 11.7).

This module provides unified filter types for querying audit data across
all audit stores (decisions, traceability, costs).

The AuditFilters dataclass combines all filtering capabilities into a single
unified interface, with conversion methods to store-specific filter types.

Example:
    >>> from yolo_developer.audit.filter_types import AuditFilters
    >>>
    >>> # Create unified filters
    >>> filters = AuditFilters(
    ...     agent_name="analyst",
    ...     start_time="2026-01-01T00:00:00Z",
    ...     artifact_type="requirement",
    ... )
    >>>
    >>> # Convert to store-specific filters
    >>> decision_filters = filters.to_decision_filters()
    >>> cost_filters = filters.to_cost_filters()

References:
    - FR87: Users can filter audit trail by agent, time range, or artifact
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yolo_developer.audit.traceability_types import VALID_ARTIFACT_TYPES

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from yolo_developer.audit.cost_store import CostFilters
    from yolo_developer.audit.store import DecisionFilters


@dataclass(frozen=True)
class AuditFilters:
    """Unified filters for querying audit data across all stores.

    All fields are optional; None means no filtering on that field.
    Multiple filters are combined with AND logic.

    Attributes:
        agent_name: Filter by agent name
        decision_type: Filter by decision type
        artifact_type: Filter by artifact type (requirement, story, design_decision, code, test)
        start_time: Filter items after this timestamp (inclusive, ISO 8601)
        end_time: Filter items before this timestamp (inclusive, ISO 8601)
        sprint_id: Filter by sprint ID
        story_id: Filter by story ID
        session_id: Filter by session ID
        severity: Filter by decision severity

    Example:
        >>> filters = AuditFilters(
        ...     agent_name="analyst",
        ...     artifact_type="requirement",
        ...     start_time="2026-01-01T00:00:00Z",
        ... )
        >>> filters.to_dict()
        {'agent_name': 'analyst', 'artifact_type': 'requirement', ...}
    """

    agent_name: str | None = None
    decision_type: str | None = None
    artifact_type: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    sprint_id: str | None = None
    story_id: str | None = None
    session_id: str | None = None
    severity: str | None = None

    def __post_init__(self) -> None:
        """Validate filter values and log warnings for issues."""
        if self.artifact_type is not None and self.artifact_type not in VALID_ARTIFACT_TYPES:
            _logger.warning(
                "AuditFilters artifact_type='%s' is not a valid artifact type. Valid types: %s",
                self.artifact_type,
                VALID_ARTIFACT_TYPES,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the filters.
        """
        return {
            "agent_name": self.agent_name,
            "decision_type": self.decision_type,
            "artifact_type": self.artifact_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sprint_id": self.sprint_id,
            "story_id": self.story_id,
            "session_id": self.session_id,
            "severity": self.severity,
        }

    def to_decision_filters(self) -> DecisionFilters:
        """Convert to DecisionFilters for decision store queries.

        Returns:
            DecisionFilters instance with applicable fields mapped.
        """
        from yolo_developer.audit.store import DecisionFilters

        return DecisionFilters(
            agent_name=self.agent_name,
            decision_type=self.decision_type,
            start_time=self.start_time,
            end_time=self.end_time,
            sprint_id=self.sprint_id,
            story_id=self.story_id,
        )

    def to_cost_filters(self) -> CostFilters:
        """Convert to CostFilters for cost store queries.

        Returns:
            CostFilters instance with applicable fields mapped.
        """
        from yolo_developer.audit.cost_store import CostFilters

        return CostFilters(
            agent_name=self.agent_name,
            story_id=self.story_id,
            sprint_id=self.sprint_id,
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def to_traceability_filters(self) -> dict[str, str | None]:
        """Convert to traceability filter parameters.

        Returns a dictionary of filter parameters for TraceabilityStore's
        get_artifacts_by_filters() method.

        Returns:
            Dictionary with artifact_type, created_after, created_before keys.
        """
        return {
            "artifact_type": self.artifact_type,
            "created_after": self.start_time,
            "created_before": self.end_time,
        }
