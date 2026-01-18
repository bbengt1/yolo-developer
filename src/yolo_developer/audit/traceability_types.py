"""Type definitions for requirement traceability (Story 11.2).

This module provides the data types used for traceability:

- ArtifactType: Literal type for traceable artifact categories
- LinkType: Literal type for trace link relationships
- TraceableArtifact: Represents a traceable item (requirement, story, code, etc.)
- TraceLink: Represents a link between two artifacts

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.traceability_types import (
    ...     TraceableArtifact,
    ...     TraceLink,
    ... )
    >>>
    >>> artifact = TraceableArtifact(
    ...     id="FR82",
    ...     artifact_type="requirement",
    ...     name="Requirement Traceability",
    ...     description="System can trace code to requirements",
    ...     created_at="2026-01-18T12:00:00Z",
    ... )
    >>> artifact.to_dict()
    {'id': 'FR82', ...}

References:
    - FR82: System can generate decision traceability from requirement to code
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

ArtifactType = Literal["requirement", "story", "design_decision", "code", "test"]
"""Type of traceable artifact.

Values:
    requirement: A functional or non-functional requirement
    story: A user story derived from requirements
    design_decision: An architectural or design decision
    code: Source code implementing a design
    test: Test code validating an implementation
"""

LinkType = Literal["derives_from", "implements", "tests", "documents"]
"""Type of trace link relationship.

Values:
    derives_from: Target artifact is derived from source (e.g., story from requirement)
    implements: Target artifact implements source (e.g., code implements design)
    tests: Target artifact tests source (e.g., test validates code)
    documents: Target artifact documents source (e.g., ADR documents decision)
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

VALID_ARTIFACT_TYPES: frozenset[str] = frozenset(
    {
        "requirement",
        "story",
        "design_decision",
        "code",
        "test",
    }
)
"""Set of valid artifact type values."""

VALID_LINK_TYPES: frozenset[str] = frozenset(
    {
        "derives_from",
        "implements",
        "tests",
        "documents",
    }
)
"""Set of valid link type values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class TraceableArtifact:
    """Represents a traceable artifact in the requirement-to-code chain.

    Traceable artifacts form a directed acyclic graph (DAG):
    Requirement → Story → Design Decision → Code → Test

    Attributes:
        id: Unique identifier for the artifact
        artifact_type: Type of artifact (requirement, story, design_decision, code, test)
        name: Human-readable name of the artifact
        description: Detailed description of the artifact
        created_at: ISO 8601 timestamp when artifact was created
        metadata: Additional key-value data (optional)

    Example:
        >>> artifact = TraceableArtifact(
        ...     id="FR82",
        ...     artifact_type="requirement",
        ...     name="Requirement Traceability",
        ...     description="System can trace code to requirements",
        ...     created_at="2026-01-18T12:00:00Z",
        ... )
        >>> artifact.name
        'Requirement Traceability'
    """

    id: str
    artifact_type: ArtifactType
    name: str
    description: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate artifact data and log warnings for issues."""
        if not self.id:
            _logger.warning("TraceableArtifact id is empty")
        if self.artifact_type not in VALID_ARTIFACT_TYPES:
            _logger.warning(
                "TraceableArtifact artifact_type='%s' is not a valid artifact type for id=%s",
                self.artifact_type,
                self.id,
            )
        if not self.name:
            _logger.warning("TraceableArtifact name is empty for id=%s", self.id)
        if not self.created_at:
            _logger.warning("TraceableArtifact created_at is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the traceable artifact.
        """
        return {
            "id": self.id,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TraceLink:
    """Represents a link between two traceable artifacts.

    Links form the edges of the traceability DAG, connecting artifacts
    in the requirement-to-code chain.

    Attributes:
        id: Unique identifier for the link
        source_id: ID of the source artifact (the artifact being traced from)
        source_type: Type of the source artifact
        target_id: ID of the target artifact (the artifact being traced to)
        target_type: Type of the target artifact
        link_type: Type of relationship between artifacts
        created_at: ISO 8601 timestamp when link was created
        metadata: Additional key-value data (optional)

    Example:
        >>> link = TraceLink(
        ...     id="link-001",
        ...     source_id="story-001",
        ...     source_type="story",
        ...     target_id="req-001",
        ...     target_type="requirement",
        ...     link_type="derives_from",
        ...     created_at="2026-01-18T12:00:00Z",
        ... )
        >>> link.link_type
        'derives_from'
    """

    id: str
    source_id: str
    source_type: ArtifactType
    target_id: str
    target_type: ArtifactType
    link_type: LinkType
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate link data and log warnings for issues."""
        if not self.id:
            _logger.warning("TraceLink id is empty")
        if not self.source_id:
            _logger.warning("TraceLink source_id is empty for id=%s", self.id)
        if not self.target_id:
            _logger.warning("TraceLink target_id is empty for id=%s", self.id)
        if self.source_type not in VALID_ARTIFACT_TYPES:
            _logger.warning(
                "TraceLink source_type='%s' is not a valid artifact type for id=%s",
                self.source_type,
                self.id,
            )
        if self.target_type not in VALID_ARTIFACT_TYPES:
            _logger.warning(
                "TraceLink target_type='%s' is not a valid artifact type for id=%s",
                self.target_type,
                self.id,
            )
        if self.link_type not in VALID_LINK_TYPES:
            _logger.warning(
                "TraceLink link_type='%s' is not a valid link type for id=%s",
                self.link_type,
                self.id,
            )
        if not self.created_at:
            _logger.warning("TraceLink created_at is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the trace link.
        """
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "link_type": self.link_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
