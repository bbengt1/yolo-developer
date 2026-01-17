"""Type definitions for circular logic detection (Story 10.6).

This module provides the data types used by the circular logic detection system:

- CycleSeverity: Literal type for cycle severity levels
- InterventionStrategy: Literal type for intervention strategies
- CircularPattern: Detected circular pattern in agent exchanges
- CycleLog: Audit log entry for a detected cycle
- CircularLogicConfig: Configuration for circular logic detection
- CycleAnalysis: Complete analysis result from circular logic detection

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.sm.circular_detection_types import (
    ...     CircularPattern,
    ...     CycleAnalysis,
    ...     CircularLogicConfig,
    ... )
    >>>
    >>> pattern = CircularPattern(
    ...     pattern_type="agent_pair",
    ...     agents_involved=("analyst", "pm"),
    ...     topic="requirements",
    ...     exchange_count=5,
    ...     first_exchange_at="2026-01-16T10:00:00Z",
    ...     last_exchange_at="2026-01-16T10:30:00Z",
    ...     duration_seconds=1800.0,
    ...     severity="medium",
    ... )
    >>> pattern.to_dict()
    {'pattern_type': 'agent_pair', ...}

References:
    - FR12: SM Agent can detect circular logic between agents (>3 exchanges)
    - FR70: SM Agent can escalate to human when circular logic persists
    - ADR-001: Internal state uses frozen dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

CycleSeverity = Literal["low", "medium", "high", "critical"]
"""Severity level for detected circular logic patterns.

Values:
    low: Just at threshold (3-4 exchanges) - monitoring only
    medium: Clearly cycling (5-7 exchanges) - intervention recommended
    high: Persistent cycle (8-11 exchanges) - intervention required
    critical: Severe cycle (12+ exchanges) - immediate escalation
"""

InterventionStrategy = Literal["break_cycle", "inject_context", "escalate_human", "none"]
"""Strategy for intervening when circular logic is detected.

Values:
    break_cycle: Force routing to a different agent to break the cycle
    inject_context: Add clarifying context to help agents resolve the impasse
    escalate_human: Escalate to human intervention immediately
    none: No intervention (monitoring only)
"""

PatternType = Literal["agent_pair", "multi_agent", "topic_cycle"]
"""Type of circular pattern detected.

Values:
    agent_pair: Two agents exchanging back and forth (A->B->A)
    multi_agent: Three or more agents in a cycle (A->B->C->A)
    topic_cycle: Repeated exchanges on the same topic across agents
"""

# =============================================================================
# Constants
# =============================================================================

DEFAULT_EXCHANGE_THRESHOLD: int = 3
"""Default number of exchanges before circular logic is detected (per FR12)."""

DEFAULT_TIME_WINDOW_SECONDS: float = 600.0
"""Default time window in seconds for tracking exchanges (10 minutes)."""

VALID_CYCLE_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})
"""Set of valid cycle severity values."""

VALID_INTERVENTION_STRATEGIES: frozenset[str] = frozenset(
    {"break_cycle", "inject_context", "escalate_human", "none"}
)
"""Set of valid intervention strategy values."""

VALID_PATTERN_TYPES: frozenset[str] = frozenset({"agent_pair", "multi_agent", "topic_cycle"})
"""Set of valid pattern type values."""

# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CircularPattern:
    """Detected circular pattern in agent exchanges.

    Represents a detected cycle with context about the agents, topic, and severity.

    Attributes:
        pattern_type: Type of pattern (agent_pair, multi_agent, topic_cycle)
        agents_involved: Ordered sequence of agents in the cycle
        topic: Semantic topic/issue being cycled on
        exchange_count: Number of exchanges in this cycle
        first_exchange_at: ISO timestamp of first exchange
        last_exchange_at: ISO timestamp of last exchange
        duration_seconds: Time span of the cycle in seconds
        severity: Severity level of the cycle

    Example:
        >>> pattern = CircularPattern(
        ...     pattern_type="agent_pair",
        ...     agents_involved=("analyst", "pm"),
        ...     topic="requirements_clarification",
        ...     exchange_count=5,
        ...     first_exchange_at="2026-01-16T10:00:00Z",
        ...     last_exchange_at="2026-01-16T10:30:00Z",
        ...     duration_seconds=1800.0,
        ...     severity="medium",
        ... )
    """

    pattern_type: PatternType
    agents_involved: tuple[str, ...]
    topic: str
    exchange_count: int
    first_exchange_at: str
    last_exchange_at: str
    duration_seconds: float
    severity: CycleSeverity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the pattern.
        """
        return {
            "pattern_type": self.pattern_type,
            "agents_involved": list(self.agents_involved),
            "topic": self.topic,
            "exchange_count": self.exchange_count,
            "first_exchange_at": self.first_exchange_at,
            "last_exchange_at": self.last_exchange_at,
            "duration_seconds": self.duration_seconds,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class CycleLog:
    """Audit log entry for a detected cycle.

    Complete record for post-mortem analysis of circular logic.

    Attributes:
        cycle_id: Unique identifier for this cycle detection
        detected_at: ISO timestamp when the cycle was detected
        patterns: Tuple of detected patterns
        intervention_taken: Strategy used to intervene
        escalation_triggered: Whether escalation was triggered
        resolution: Description of how the cycle was resolved

    Example:
        >>> log = CycleLog(
        ...     cycle_id="cycle-123",
        ...     detected_at="2026-01-16T10:30:00Z",
        ...     patterns=(pattern,),
        ...     intervention_taken="break_cycle",
        ...     escalation_triggered=False,
        ...     resolution="Routed to architect to break cycle",
        ... )
    """

    cycle_id: str
    detected_at: str
    patterns: tuple[CircularPattern, ...]
    intervention_taken: InterventionStrategy
    escalation_triggered: bool
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the log entry.
        """
        return {
            "cycle_id": self.cycle_id,
            "detected_at": self.detected_at,
            "patterns": [p.to_dict() for p in self.patterns],
            "intervention_taken": self.intervention_taken,
            "escalation_triggered": self.escalation_triggered,
            "resolution": self.resolution,
        }


def _default_severity_thresholds() -> dict[str, int]:
    """Create default severity thresholds dictionary.

    Returns:
        Dictionary mapping severity levels to exchange counts.
    """
    return {
        "low": 3,  # Just at threshold
        "medium": 5,  # Clearly cycling
        "high": 8,  # Persistent cycle
        "critical": 12,  # Severe, immediate escalation
    }


@dataclass(frozen=True)
class CircularLogicConfig:
    """Configuration for circular logic detection.

    Configurable thresholds and behavior settings for detecting
    and responding to circular logic patterns.

    Attributes:
        exchange_threshold: Number of exchanges before detection (default: 3 per FR12)
        time_window_seconds: Time window for tracking exchanges (default: 600s)
        severity_thresholds: Mapping of severity to exchange counts
        auto_escalate_severity: Severity level that triggers automatic escalation
        enable_topic_detection: Whether to enable topic-based cycle detection
        enable_multi_agent_detection: Whether to enable multi-agent cycle detection

    Example:
        >>> config = CircularLogicConfig(
        ...     exchange_threshold=5,
        ...     auto_escalate_severity="high",
        ... )
    """

    exchange_threshold: int = DEFAULT_EXCHANGE_THRESHOLD
    time_window_seconds: float = DEFAULT_TIME_WINDOW_SECONDS
    severity_thresholds: dict[str, int] = field(default_factory=_default_severity_thresholds)
    auto_escalate_severity: CycleSeverity = "critical"
    enable_topic_detection: bool = True
    enable_multi_agent_detection: bool = True


@dataclass(frozen=True)
class CycleAnalysis:
    """Complete analysis result from circular logic detection.

    Returned by detect_circular_logic() with all detection results,
    intervention decisions, and logging information.

    Attributes:
        circular_detected: Whether circular logic was detected
        patterns_found: Tuple of detected circular patterns
        intervention_strategy: Strategy for intervention
        intervention_message: Human-readable intervention message
        escalation_triggered: Whether escalation was triggered
        escalation_reason: Reason for escalation if triggered
        topic_exchanges: Mapping of topics to exchange IDs
        total_exchange_count: Total number of exchanges analyzed
        cycle_log: Audit log entry for the detection
        analyzed_at: ISO timestamp when analysis was performed

    Example:
        >>> analysis = CycleAnalysis(
        ...     circular_detected=True,
        ...     patterns_found=(pattern,),
        ...     intervention_strategy="break_cycle",
        ...     intervention_message="Breaking cycle by routing to architect",
        ...     escalation_triggered=False,
        ...     escalation_reason=None,
        ...     topic_exchanges={"requirements": ["ex-1", "ex-2"]},
        ...     total_exchange_count=5,
        ...     cycle_log=log,
        ... )
        >>> analysis.to_dict()
        {'circular_detected': True, ...}
    """

    circular_detected: bool
    patterns_found: tuple[CircularPattern, ...]
    intervention_strategy: InterventionStrategy
    intervention_message: str
    escalation_triggered: bool
    escalation_reason: str | None
    topic_exchanges: dict[str, list[str]]
    total_exchange_count: int
    cycle_log: CycleLog | None
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested patterns and log.
        """
        return {
            "circular_detected": self.circular_detected,
            "patterns_found": [p.to_dict() for p in self.patterns_found],
            "intervention_strategy": self.intervention_strategy,
            "intervention_message": self.intervention_message,
            "escalation_triggered": self.escalation_triggered,
            "escalation_reason": self.escalation_reason,
            "topic_exchanges": dict(self.topic_exchanges),
            "total_exchange_count": self.total_exchange_count,
            "cycle_log": self.cycle_log.to_dict() if self.cycle_log else None,
            "analyzed_at": self.analyzed_at,
        }
