"""Type definitions for Architect agent (Story 7.1).

This module provides the data types used by the Architect agent:

- DesignDecisionType: Literal type for design decision categories
- DesignDecision: A design decision made for a story
- ADRStatus: Literal type for ADR lifecycle status
- ADR: An Architecture Decision Record
- ArchitectOutput: Complete output from architect processing

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.architect.types import (
    ...     DesignDecision,
    ...     ADR,
    ...     ArchitectOutput,
    ... )
    >>>
    >>> decision = DesignDecision(
    ...     id="design-001",
    ...     story_id="story-001",
    ...     decision_type="pattern",
    ...     description="Use Repository pattern",
    ...     rationale="Decouples data access from business logic",
    ...     alternatives_considered=("Active Record", "DAO"),
    ... )
    >>> decision.to_dict()
    {'id': 'design-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

DesignDecisionType = Literal[
    "pattern",
    "technology",
    "integration",
    "data",
    "security",
    "infrastructure",
]
"""Type of architectural design decision.

Values:
    pattern: Architectural pattern selection (e.g., Repository, Factory)
    technology: Technology/framework choice (e.g., PostgreSQL, Redis)
    integration: Integration approach (e.g., REST API, message queue)
    data: Data model/storage decisions (e.g., event sourcing, CQRS)
    security: Security architecture (e.g., OAuth2, JWT)
    infrastructure: Infrastructure choice (e.g., Docker, Kubernetes)
"""

ADRStatus = Literal[
    "proposed",
    "accepted",
    "deprecated",
    "superseded",
]
"""Lifecycle status of an Architecture Decision Record.

Values:
    proposed: Decision is being considered
    accepted: Decision has been accepted and is in effect
    deprecated: Decision is no longer recommended
    superseded: Decision has been replaced by another ADR
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class DesignDecision:
    """A design decision made for a story.

    Represents an architectural decision with full context including
    the rationale and alternatives that were considered.

    Attributes:
        id: Unique identifier for the decision (format: design-{timestamp}-{counter})
        story_id: ID of the story this decision applies to
        decision_type: Category of the decision
        description: Clear description of what was decided
        rationale: Why this decision was made
        alternatives_considered: Other options that were evaluated
        created_at: ISO timestamp when decision was created

    Example:
        >>> decision = DesignDecision(
        ...     id="design-1704412345-001",
        ...     story_id="story-001",
        ...     decision_type="pattern",
        ...     description="Use Repository pattern for data access",
        ...     rationale="Provides clean separation of concerns",
        ...     alternatives_considered=("Active Record", "DAO"),
        ... )
    """

    id: str
    story_id: str
    decision_type: DesignDecisionType
    description: str
    rationale: str
    alternatives_considered: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the design decision.
        """
        return {
            "id": self.id,
            "story_id": self.story_id,
            "decision_type": self.decision_type,
            "description": self.description,
            "rationale": self.rationale,
            "alternatives_considered": list(self.alternatives_considered),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ADR:
    """An Architecture Decision Record.

    Follows the standard ADR format with Title, Status, Context,
    Decision, and Consequences sections.

    Attributes:
        id: Unique identifier (format: ADR-{number:03d})
        title: Descriptive title of the decision
        status: Current lifecycle status
        context: Why this decision was needed
        decision: What was decided
        consequences: Positive and negative effects
        story_ids: Stories this ADR relates to
        created_at: ISO timestamp when ADR was created

    Example:
        >>> adr = ADR(
        ...     id="ADR-001",
        ...     title="Use PostgreSQL for persistence",
        ...     status="accepted",
        ...     context="Need reliable ACID-compliant database",
        ...     decision="PostgreSQL for all persistent data",
        ...     consequences="Good: Reliability, Bad: Operational complexity",
        ...     story_ids=("story-001", "story-002"),
        ... )
    """

    id: str
    title: str
    status: ADRStatus
    context: str
    decision: str
    consequences: str
    story_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the ADR.
        """
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "story_ids": list(self.story_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ArchitectOutput:
    """Complete output from Architect agent processing.

    Contains all design decisions and ADRs generated during
    architect_node execution, plus processing notes.

    Attributes:
        design_decisions: Tuple of design decisions made
        adrs: Tuple of ADRs generated
        processing_notes: Notes about the processing (stats, issues, etc.)

    Example:
        >>> output = ArchitectOutput(
        ...     design_decisions=(decision1, decision2),
        ...     adrs=(adr1,),
        ...     processing_notes="Processed 2 stories, generated 2 decisions",
        ... )
        >>> output.to_dict()
        {'design_decisions': [...], 'adrs': [...], ...}
    """

    design_decisions: tuple[DesignDecision, ...] = field(default_factory=tuple)
    adrs: tuple[ADR, ...] = field(default_factory=tuple)
    processing_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested design decisions and ADRs.
        """
        return {
            "design_decisions": [d.to_dict() for d in self.design_decisions],
            "adrs": [a.to_dict() for a in self.adrs],
            "processing_notes": self.processing_notes,
        }
