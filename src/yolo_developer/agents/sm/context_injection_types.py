"""Type definitions for context injection (Story 10.13).

This module provides the data types used by the context injection module:

- ContextSource: Literal type for context sources
- GapReason: Literal type for gap detection reasons
- ContextGap: Detected context gap requiring injection
- RetrievedContext: Context retrieved from a source
- InjectionResult: Result of a context injection operation
- InjectionConfig: Configuration for context injection

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.context_injection_types import (
    ...     ContextGap,
    ...     RetrievedContext,
    ...     InjectionResult,
    ...     InjectionConfig,
    ... )
    >>>
    >>> gap = ContextGap(
    ...     gap_id="gap-12345",
    ...     agent="architect",
    ...     reason="clarification_requested",
    ...     context_query="authentication requirements",
    ...     confidence=0.9,
    ...     indicators=("clarification_message_detected",),
    ... )
    >>> gap.to_dict()
    {'gap_id': 'gap-12345', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.

References:
    - FR69: SM Agent can inject context when agents lack information
    - Story 10.5: Health Monitoring (cycle time metrics for gap detection)
    - Story 10.6: Circular Logic Detection (cycle analysis input)
    - Story 10.8: Handoff Management (context validation complements injection)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Task 1.1, 1.2)
# =============================================================================

ContextSource = Literal["memory", "state", "sprint", "architecture"]
"""Source from which context can be retrieved.

Values:
    memory: Context retrieved from memory store (vector/graph)
    state: Context extracted from current orchestration state
    sprint: Context from sprint plan and progress
    architecture: Context from architecture decisions and ADRs
"""

GapReason = Literal[
    "clarification_requested",
    "circular_logic",
    "long_cycle_time",
    "gate_failure",
    "explicit_flag",
]
"""Reason why a context gap was detected.

Values:
    clarification_requested: Agent explicitly requested clarification
    circular_logic: Circular pattern detected indicating missing info
    long_cycle_time: Agent cycle time exceeds baseline significantly
    gate_failure: Quality gate failed due to missing context
    explicit_flag: Agent explicitly flagged missing_context
"""


# =============================================================================
# Constants (Task 1.7)
# =============================================================================

DEFAULT_MAX_CONTEXT_ITEMS: int = 5
"""Default maximum number of context items to retrieve."""

DEFAULT_MIN_RELEVANCE_SCORE: float = 0.7
"""Default minimum relevance score for context inclusion (0.0-1.0)."""

DEFAULT_MAX_CONTEXT_SIZE_BYTES: int = 100_000
"""Default maximum context size in bytes (100KB)."""

DEFAULT_LOG_INJECTIONS: bool = True
"""Default setting for logging injection events."""

MIN_CONFIDENCE: float = 0.0
"""Minimum confidence value for gap detection."""

MAX_CONFIDENCE: float = 1.0
"""Maximum confidence value for gap detection."""

MIN_RELEVANCE: float = 0.0
"""Minimum relevance score for retrieved context."""

MAX_RELEVANCE: float = 1.0
"""Maximum relevance score for retrieved context."""

LONG_CYCLE_TIME_MULTIPLIER: float = 2.0
"""Multiplier for detecting long cycle times (2x baseline = gap)."""

VALID_CONTEXT_SOURCES: frozenset[str] = frozenset({"memory", "state", "sprint", "architecture"})
"""Set of valid context source values."""

VALID_GAP_REASONS: frozenset[str] = frozenset(
    {
        "clarification_requested",
        "circular_logic",
        "long_cycle_time",
        "gate_failure",
        "explicit_flag",
    }
)
"""Set of valid gap reason values."""


# =============================================================================
# Data Classes (Tasks 1.3-1.6)
# =============================================================================


@dataclass(frozen=True)
class ContextGap:
    """Detected context gap requiring injection.

    Represents a gap in context detected by the SM agent that
    may prevent an agent from completing its work effectively.

    Attributes:
        gap_id: Unique identifier for this gap (e.g., "gap-12345")
        agent: Agent needing additional context (e.g., "architect")
        reason: Reason the gap was detected
        context_query: Query string describing what context is needed
        confidence: Confidence level in gap detection (0.0-1.0)
        indicators: Evidence that led to gap detection
        detected_at: ISO timestamp when gap was detected (auto-generated)

    Example:
        >>> gap = ContextGap(
        ...     gap_id="gap-12345",
        ...     agent="architect",
        ...     reason="clarification_requested",
        ...     context_query="authentication requirements",
        ...     confidence=0.9,
        ...     indicators=("clarification_message_detected",),
        ... )
        >>> gap.agent
        'architect'
    """

    gap_id: str
    agent: str
    reason: GapReason
    context_query: str
    confidence: float
    indicators: tuple[str, ...]
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate context gap data and log warnings for issues."""
        if not self.gap_id:
            _logger.warning(
                "ContextGap gap_id is empty for agent=%s",
                self.agent,
            )
        if not self.agent:
            _logger.warning(
                "ContextGap agent is empty for gap_id=%s",
                self.gap_id,
            )
        if self.reason not in VALID_GAP_REASONS:
            _logger.warning(
                "ContextGap reason='%s' is not a valid reason value for gap_id=%s",
                self.reason,
                self.gap_id,
            )
        if self.confidence < MIN_CONFIDENCE or self.confidence > MAX_CONFIDENCE:
            _logger.warning(
                "ContextGap confidence=%.3f is outside valid range [%.1f, %.1f] for gap_id=%s",
                self.confidence,
                MIN_CONFIDENCE,
                MAX_CONFIDENCE,
                self.gap_id,
            )
        if not self.context_query:
            _logger.warning(
                "ContextGap context_query is empty for gap_id=%s",
                self.gap_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the context gap.
        """
        return {
            "gap_id": self.gap_id,
            "agent": self.agent,
            "reason": self.reason,
            "context_query": self.context_query,
            "confidence": self.confidence,
            "indicators": list(self.indicators),
            "detected_at": self.detected_at,
        }


@dataclass(frozen=True)
class RetrievedContext:
    """Context retrieved from a source for injection.

    Represents a piece of context retrieved from memory, state,
    or other sources that may help fill a detected gap.

    Attributes:
        source: Source from which context was retrieved
        content: The actual context content (text)
        relevance_score: Relevance to the context query (0.0-1.0)
        metadata: Additional metadata about the retrieved context
        retrieved_at: ISO timestamp when context was retrieved (auto-generated)

    Example:
        >>> context = RetrievedContext(
        ...     source="memory",
        ...     content="OAuth2 is required for authentication",
        ...     relevance_score=0.95,
        ...     metadata={"type": "decision", "agent": "architect"},
        ... )
        >>> context.relevance_score
        0.95
    """

    source: ContextSource
    content: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate retrieved context data and log warnings for issues."""
        if self.source not in VALID_CONTEXT_SOURCES:
            _logger.warning(
                "RetrievedContext source='%s' is not a valid source value",
                self.source,
            )
        if not self.content:
            _logger.warning(
                "RetrievedContext content is empty for source=%s",
                self.source,
            )
        if self.relevance_score < MIN_RELEVANCE or self.relevance_score > MAX_RELEVANCE:
            _logger.warning(
                "RetrievedContext relevance_score=%.3f is outside valid range [%.1f, %.1f]",
                self.relevance_score,
                MIN_RELEVANCE,
                MAX_RELEVANCE,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the retrieved context.
        """
        return {
            "source": self.source,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class InjectionResult:
    """Result of a context injection operation.

    Captures the outcome of detecting a gap, retrieving context,
    and injecting it into the orchestration state.

    Attributes:
        gap: The detected context gap that triggered injection
        contexts_retrieved: Contexts that were retrieved for injection
        injected: Whether context was successfully injected
        injection_target: State key where context was injected
        total_context_size: Total size of injected context in bytes
        duration_ms: Time taken for the injection operation in milliseconds

    Example:
        >>> result = InjectionResult(
        ...     gap=gap,
        ...     contexts_retrieved=(context1, context2),
        ...     injected=True,
        ...     injection_target="injected_context",
        ...     total_context_size=1024,
        ...     duration_ms=50.0,
        ... )
        >>> result.injected
        True
    """

    gap: ContextGap
    contexts_retrieved: tuple[RetrievedContext, ...]
    injected: bool
    injection_target: str
    total_context_size: int
    duration_ms: float

    def __post_init__(self) -> None:
        """Validate injection result data and log warnings for issues."""
        if self.total_context_size < 0:
            _logger.warning(
                "InjectionResult total_context_size=%d is negative for gap_id=%s",
                self.total_context_size,
                self.gap.gap_id,
            )
        if self.duration_ms < 0:
            _logger.warning(
                "InjectionResult duration_ms=%.2f is negative for gap_id=%s",
                self.duration_ms,
                self.gap.gap_id,
            )
        if self.injected and not self.injection_target:
            _logger.warning(
                "InjectionResult injected=True but injection_target is empty for gap_id=%s",
                self.gap.gap_id,
            )
        if self.injected and not self.contexts_retrieved:
            _logger.warning(
                "InjectionResult injected=True but contexts_retrieved is empty for gap_id=%s",
                self.gap.gap_id,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the injection result.
        """
        return {
            "gap": self.gap.to_dict(),
            "contexts_retrieved": [c.to_dict() for c in self.contexts_retrieved],
            "injected": self.injected,
            "injection_target": self.injection_target,
            "total_context_size": self.total_context_size,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class InjectionConfig:
    """Configuration for context injection behavior.

    Allows customization of which sources to search, relevance
    thresholds, and size limits for context injection.

    Attributes:
        max_context_items: Maximum number of context items to retrieve
        min_relevance_score: Minimum relevance score for inclusion (0.0-1.0)
        max_context_size_bytes: Maximum total context size in bytes
        enabled_sources: Tuple of context sources to search
        log_injections: Whether to log injection events

    Example:
        >>> config = InjectionConfig(max_context_items=10)
        >>> config.max_context_items
        10
    """

    max_context_items: int = DEFAULT_MAX_CONTEXT_ITEMS
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE
    max_context_size_bytes: int = DEFAULT_MAX_CONTEXT_SIZE_BYTES
    enabled_sources: tuple[ContextSource, ...] = ("memory", "state")
    log_injections: bool = DEFAULT_LOG_INJECTIONS

    def __post_init__(self) -> None:
        """Validate config values and log warnings for issues."""
        if self.max_context_items < 1:
            _logger.warning(
                "InjectionConfig max_context_items=%d should be at least 1",
                self.max_context_items,
            )
        if self.min_relevance_score < MIN_RELEVANCE or self.min_relevance_score > MAX_RELEVANCE:
            _logger.warning(
                "InjectionConfig min_relevance_score=%.3f should be between %.1f and %.1f",
                self.min_relevance_score,
                MIN_RELEVANCE,
                MAX_RELEVANCE,
            )
        if self.max_context_size_bytes < 1:
            _logger.warning(
                "InjectionConfig max_context_size_bytes=%d should be at least 1",
                self.max_context_size_bytes,
            )
        for source in self.enabled_sources:
            if source not in VALID_CONTEXT_SOURCES:
                _logger.warning(
                    "InjectionConfig enabled_sources contains invalid source='%s'",
                    source,
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for state storage.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "max_context_items": self.max_context_items,
            "min_relevance_score": self.min_relevance_score,
            "max_context_size_bytes": self.max_context_size_bytes,
            "enabled_sources": list(self.enabled_sources),
            "log_injections": self.log_injections,
        }
