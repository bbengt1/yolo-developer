"""Gate types and data structures for quality gate framework.

This module defines the core data structures used by the quality gate
decorator and evaluators. All types are immutable (frozen dataclasses)
for consistency and thread safety.

Example:
    >>> from yolo_developer.gates.types import GateResult, GateMode, GateContext
    >>>
    >>> # Create a gate result
    >>> result = GateResult(passed=True, gate_name="testability")
    >>> result.to_dict()
    {'passed': True, 'gate_name': 'testability', 'reason': None, 'timestamp': '...'}
    >>>
    >>> # Create gate context for evaluation
    >>> context = GateContext(state={"messages": []}, gate_name="testability")

Security Note:
    Gate results and contexts should not contain sensitive information.
    State dictionaries passed to gates are project-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class GateMode(Enum):
    """Mode of operation for a quality gate.

    Determines how the gate responds to evaluation failures.

    Attributes:
        BLOCKING: Gate failure prevents node execution and blocks handoff.
        ADVISORY: Gate failure logs a warning but allows execution to continue.

    Example:
        >>> mode = GateMode.BLOCKING
        >>> mode.value
        'blocking'
    """

    BLOCKING = "blocking"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class GateResult:
    """Result of a quality gate evaluation.

    Captures whether the gate passed, the gate name, failure reason,
    score, threshold, and timestamp. Results are immutable for audit trail integrity.

    Attributes:
        passed: Whether the gate evaluation passed.
        gate_name: Name of the gate that was evaluated.
        reason: Reason for failure (None if passed).
        score: Evaluation score (0.0-1.0), None if not applicable.
        threshold: Threshold used for evaluation, None if not applicable.
        timestamp: When the evaluation occurred.

    Example:
        >>> result = GateResult(
        ...     passed=False,
        ...     gate_name="testability",
        ...     reason="Missing unit tests for core functionality",
        ...     score=0.65,
        ...     threshold=0.80,
        ... )
        >>> result.passed
        False
        >>> result.score
        0.65
    """

    passed: bool
    gate_name: str
    reason: str | None = None
    score: float | None = None
    threshold: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for logging and audit.

        Returns:
            Dictionary representation of the gate result with ISO timestamp.

        Example:
            >>> result = GateResult(passed=True, gate_name="test")
            >>> d = result.to_dict()
            >>> d["passed"]
            True
        """
        return {
            "passed": self.passed,
            "gate_name": self.gate_name,
            "reason": self.reason,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class GateContext:
    """Context passed to gate evaluators.

    Provides the state dictionary and metadata needed for gate evaluation.
    Context objects are immutable to prevent accidental state mutation
    during evaluation.

    Attributes:
        state: The current workflow state dictionary.
        gate_name: Name of the gate being evaluated.
        artifact_id: Optional ID of the artifact being validated.
        metadata: Optional additional metadata for evaluation.

    Example:
        >>> context = GateContext(
        ...     state={"messages": [], "current_agent": "analyst"},
        ...     gate_name="testability",
        ...     artifact_id="story-001",
        ... )
        >>> context.gate_name
        'testability'
    """

    state: dict[str, Any]
    gate_name: str
    artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
