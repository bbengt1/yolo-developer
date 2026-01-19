"""ADR types for auto-generated Architecture Decision Records (Story 11.8).

This module provides types for ADRs auto-generated from audit trail decisions.

The AutoADR dataclass captures ADRs created from Decision records with
decision_type="architecture_choice", linking them back to their source
decisions and related stories.

Example:
    >>> from yolo_developer.audit.adr_types import AutoADR
    >>>
    >>> adr = AutoADR(
    ...     id="ADR-001",
    ...     title="Use PostgreSQL for Data Storage",
    ...     status="proposed",
    ...     context="Database selection needed for persistence layer.",
    ...     decision="Selected PostgreSQL for ACID compliance.",
    ...     consequences="Positive: Strong consistency. Trade-off: Operational complexity.",
    ...     source_decision_id="dec-123",
    ...     story_ids=("1-2-database-setup",),
    ... )
    >>> adr.to_dict()
    {'id': 'ADR-001', ...}

References:
    - FR88: System can generate Architecture Decision Records automatically
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
    - Story 7.3: ADR generation pattern (ArchitectAgent)
    - Story 11.1: Decision types and logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types
# =============================================================================

ADRStatus = Literal["proposed", "accepted", "deprecated", "superseded"]
"""Status of an Architecture Decision Record.

Values:
    proposed: ADR is newly created and awaiting review
    accepted: ADR has been reviewed and accepted
    deprecated: ADR is no longer relevant but kept for history
    superseded: ADR has been replaced by a newer decision
"""

VALID_ADR_STATUSES: frozenset[str] = frozenset(
    {"proposed", "accepted", "deprecated", "superseded"}
)
"""Set of valid ADR status values."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class AutoADR:
    """Architecture Decision Record auto-generated from audit decisions.

    Captures ADRs created automatically from Decision records in the audit
    trail, specifically those with decision_type="architecture_choice".

    Attributes:
        id: Unique ADR identifier (format: ADR-{number:03d})
        title: Descriptive ADR title summarizing the decision
        status: ADR status (proposed by default for auto-generated)
        context: Why this decision was needed (problem statement)
        decision: What was decided and the chosen approach
        consequences: Positive/negative effects and trade-offs
        source_decision_id: ID of the Decision that triggered this ADR
        story_ids: Stories this ADR relates to (from decision context)
        created_at: ISO 8601 timestamp when ADR was generated

    Example:
        >>> adr = AutoADR(
        ...     id="ADR-001",
        ...     title="Use PostgreSQL",
        ...     status="proposed",
        ...     context="Need persistent storage.",
        ...     decision="Selected PostgreSQL.",
        ...     consequences="Strong ACID compliance.",
        ...     source_decision_id="dec-123",
        ... )
        >>> adr.id
        'ADR-001'
    """

    id: str
    title: str
    status: ADRStatus
    context: str
    decision: str
    consequences: str
    source_decision_id: str
    story_ids: tuple[str, ...] = ()
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate ADR data and log warnings for issues."""
        if not self.id:
            _logger.warning("AutoADR id is empty")
        if not self.id.startswith("ADR-"):
            _logger.warning(
                "AutoADR id='%s' does not follow ADR-XXX format",
                self.id,
            )
        if not self.title:
            _logger.warning("AutoADR title is empty for id=%s", self.id)
        if self.status not in VALID_ADR_STATUSES:
            _logger.warning(
                "AutoADR status='%s' is not a valid status for id=%s",
                self.status,
                self.id,
            )
        if not self.context:
            _logger.warning("AutoADR context is empty for id=%s", self.id)
        if not self.decision:
            _logger.warning("AutoADR decision is empty for id=%s", self.id)
        if not self.consequences:
            _logger.warning("AutoADR consequences is empty for id=%s", self.id)
        if not self.source_decision_id:
            _logger.warning(
                "AutoADR source_decision_id is empty for id=%s", self.id
            )
        if not self.created_at:
            _logger.warning("AutoADR created_at is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the ADR with all fields.
        """
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "source_decision_id": self.source_decision_id,
            "story_ids": list(self.story_ids),
            "created_at": self.created_at,
        }
