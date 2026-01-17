"""Type definitions for conflict mediation (Story 10.7).

This module provides the data types used by the conflict mediation system:

- ConflictType: Literal type for conflict categories
- ConflictSeverity: Literal type for conflict severity levels
- ResolutionStrategy: Literal type for resolution strategies
- ConflictParty: An agent's position in a conflict
- Conflict: Detected conflict between agents
- ConflictResolution: Resolution of a conflict
- MediationResult: Complete result of conflict mediation
- ConflictMediationConfig: Configuration for conflict mediation

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.conflict_types import (
    ...     ConflictParty,
    ...     Conflict,
    ...     ConflictResolution,
    ...     MediationResult,
    ... )
    >>>
    >>> party = ConflictParty(
    ...     agent="architect",
    ...     position="Use microservices architecture",
    ...     rationale="Better scalability and maintainability",
    ...     artifacts=("adr-001",),
    ... )
    >>> party.to_dict()
    {'agent': 'architect', ...}

References:
    - FR13: SM Agent can mediate conflicts between agents with different recommendations
    - ADR-001: Internal state uses frozen dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

ConflictType = Literal[
    "design_conflict", "priority_conflict", "approach_conflict", "scope_conflict"
]
"""Type of conflict between agents.

Values:
    design_conflict: Conflicting architectural or design recommendations
    priority_conflict: Conflicting assessments of task/story priority
    approach_conflict: Conflicting implementation approaches for the same goal
    scope_conflict: Conflicting assessments of scope or boundaries
"""

ConflictSeverity = Literal["minor", "moderate", "major", "blocking"]
"""Severity level for detected conflicts.

Values:
    minor: Low impact, can be deferred or auto-resolved
    moderate: Noticeable impact, should be addressed soon
    major: Significant impact, requires prompt resolution
    blocking: Blocks progress, must be resolved immediately
"""

ResolutionStrategy = Literal[
    "accept_first", "accept_second", "compromise", "defer", "escalate_human"
]
"""Strategy for resolving conflicts.

Values:
    accept_first: Accept the first party's position
    accept_second: Accept the second party's position
    compromise: Merge or partially accept both positions
    defer: Defer resolution until more context is available
    escalate_human: Escalate to human intervention
"""

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_MEDIATION_ROUNDS: int = 3
"""Default maximum mediation attempts before escalation."""

VALID_CONFLICT_TYPES: frozenset[str] = frozenset(
    {"design_conflict", "priority_conflict", "approach_conflict", "scope_conflict"}
)
"""Set of valid conflict type values."""

VALID_CONFLICT_SEVERITIES: frozenset[str] = frozenset({"minor", "moderate", "major", "blocking"})
"""Set of valid conflict severity values."""

VALID_RESOLUTION_STRATEGIES: frozenset[str] = frozenset(
    {"accept_first", "accept_second", "compromise", "defer", "escalate_human"}
)
"""Set of valid resolution strategy values."""

# Resolution principles hierarchy (higher weight = higher priority)
RESOLUTION_PRINCIPLES: dict[str, dict[str, Any]] = {
    "safety": {
        "description": "Security and safety concerns take precedence",
        "weight": 1.0,
        "keywords": ("security", "vulnerability", "risk", "safety", "exposure", "attack"),
    },
    "correctness": {
        "description": "Functional correctness beats convenience",
        "weight": 0.9,
        "keywords": ("correct", "accurate", "valid", "spec", "requirement", "bug", "fix"),
    },
    "simplicity": {
        "description": "Simpler solutions preferred",
        "weight": 0.7,
        "keywords": ("simple", "straightforward", "maintainable", "clear", "readable"),
    },
    "performance": {
        "description": "Better performance when correctness equal",
        "weight": 0.5,
        "keywords": ("fast", "efficient", "scalable", "optimized", "performance"),
    },
    "speed": {
        "description": "Faster delivery when all else equal",
        "weight": 0.3,
        "keywords": ("quick", "rapid", "soon", "deadline", "velocity"),
    },
}
"""Principles hierarchy for conflict resolution with weights and keywords."""

DEFAULT_PRINCIPLES_HIERARCHY: tuple[str, ...] = (
    "safety",
    "correctness",
    "simplicity",
    "performance",
    "speed",
)
"""Default order of principles for conflict resolution."""

# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class ConflictParty:
    """An agent's position in a conflict.

    Represents one side of a disagreement with supporting evidence.

    Attributes:
        agent: Name of the agent holding this position
        position: Brief statement of the position
        rationale: Why the agent holds this position
        artifacts: Related artifact IDs (decisions, requirements, etc.)

    Example:
        >>> party = ConflictParty(
        ...     agent="architect",
        ...     position="Use microservices architecture",
        ...     rationale="Better scalability for expected growth",
        ...     artifacts=("adr-001", "req-123"),
        ... )
    """

    agent: str
    position: str
    rationale: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the party.
        """
        return {
            "agent": self.agent,
            "position": self.position,
            "rationale": self.rationale,
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class Conflict:
    """Detected conflict between agents.

    Captures the nature of the disagreement and the parties involved.

    Attributes:
        conflict_id: Unique identifier for this conflict
        conflict_type: Type of conflict (design, priority, approach, scope)
        severity: Severity level of the conflict
        parties: Tuple of parties involved (usually 2, but could be more)
        topic: What the conflict is about
        detected_at: ISO timestamp when the conflict was detected
        blocking_progress: Whether this conflict blocks workflow progress

    Example:
        >>> conflict = Conflict(
        ...     conflict_id="conflict-arch-001",
        ...     conflict_type="design_conflict",
        ...     severity="major",
        ...     parties=(party1, party2),
        ...     topic="service_architecture",
        ... )
    """

    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    parties: tuple[ConflictParty, ...]
    topic: str
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    blocking_progress: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested parties.
        """
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "parties": [p.to_dict() for p in self.parties],
            "topic": self.topic,
            "detected_at": self.detected_at,
            "blocking_progress": self.blocking_progress,
        }


@dataclass(frozen=True)
class ConflictResolution:
    """Resolution of a conflict.

    Documents the resolution decision and rationale.

    Attributes:
        conflict_id: ID of the conflict being resolved
        strategy: Resolution strategy used
        resolution_rationale: Why this resolution was chosen
        winning_position: Position accepted (if accept_first/second)
        compromises: Compromises made (if compromise strategy)
        principles_applied: Which principles drove the decision
        documented_at: ISO timestamp when resolution was documented

    Example:
        >>> resolution = ConflictResolution(
        ...     conflict_id="conflict-arch-001",
        ...     strategy="accept_first",
        ...     resolution_rationale="Security concerns take precedence",
        ...     winning_position="Use microservices with API gateway",
        ...     compromises=(),
        ...     principles_applied=("safety", "correctness"),
        ... )
    """

    conflict_id: str
    strategy: ResolutionStrategy
    resolution_rationale: str
    winning_position: str | None = None
    compromises: tuple[str, ...] = field(default_factory=tuple)
    principles_applied: tuple[str, ...] = field(default_factory=tuple)
    documented_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the resolution.
        """
        return {
            "conflict_id": self.conflict_id,
            "strategy": self.strategy,
            "resolution_rationale": self.resolution_rationale,
            "winning_position": self.winning_position,
            "compromises": list(self.compromises),
            "principles_applied": list(self.principles_applied),
            "documented_at": self.documented_at,
        }


@dataclass(frozen=True)
class MediationResult:
    """Complete result of conflict mediation.

    Returned by mediate_conflicts() with full mediation outcome.

    Attributes:
        conflicts_detected: Tuple of detected conflicts
        resolutions: Tuple of resolutions for each conflict
        notifications_sent: Agent names that were notified
        escalations_triggered: Conflict IDs requiring escalation
        success: Whether all conflicts were resolved
        mediation_notes: Additional notes about the mediation
        mediated_at: ISO timestamp when mediation was performed

    Example:
        >>> result = MediationResult(
        ...     conflicts_detected=(conflict,),
        ...     resolutions=(resolution,),
        ...     notifications_sent=("architect", "dev"),
        ...     escalations_triggered=(),
        ...     success=True,
        ...     mediation_notes="Resolved via safety principle",
        ... )
        >>> result.to_dict()
        {'conflicts_detected': [...], ...}
    """

    conflicts_detected: tuple[Conflict, ...]
    resolutions: tuple[ConflictResolution, ...]
    notifications_sent: tuple[str, ...]
    escalations_triggered: tuple[str, ...]
    success: bool
    mediation_notes: str = ""
    mediated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested conflicts and resolutions.
        """
        return {
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "notifications_sent": list(self.notifications_sent),
            "escalations_triggered": list(self.escalations_triggered),
            "success": self.success,
            "mediation_notes": self.mediation_notes,
            "mediated_at": self.mediated_at,
        }


@dataclass(frozen=True)
class ConflictMediationConfig:
    """Configuration for conflict mediation.

    Configurable thresholds and behavior settings for detecting
    and resolving conflicts between agents.

    Attributes:
        auto_resolve_minor: Whether to automatically resolve minor conflicts
        escalate_blocking: Whether to escalate blocking conflicts to human
        max_mediation_rounds: Maximum attempts to resolve before escalation
        principles_hierarchy: Order of principles for resolution decisions
        score_threshold: Minimum score difference to accept a winner

    Example:
        >>> config = ConflictMediationConfig(
        ...     auto_resolve_minor=True,
        ...     escalate_blocking=True,
        ...     max_mediation_rounds=3,
        ... )
    """

    auto_resolve_minor: bool = True
    escalate_blocking: bool = True
    max_mediation_rounds: int = DEFAULT_MAX_MEDIATION_ROUNDS
    principles_hierarchy: tuple[str, ...] = DEFAULT_PRINCIPLES_HIERARCHY
    score_threshold: float = 0.1  # Minimum score difference for clear winner
