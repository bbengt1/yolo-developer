"""Gate failure report types for quality gate framework (Story 3.8 - Task 1).

This module defines the core data structures for structured failure reports:
- Severity: Enum for issue severity levels (blocking vs warning)
- GateIssue: Represents a single issue found during gate evaluation
- GateFailureReport: Structured report containing all issues and metadata

All types are immutable (frozen dataclasses) for audit trail integrity
and thread safety.

Example:
    >>> from yolo_developer.gates.report_types import (
    ...     GateFailureReport, GateIssue, Severity
    ... )
    >>>
    >>> # Create an issue
    >>> issue = GateIssue(
    ...     location="req-001",
    ...     issue_type="vague_term",
    ...     description="Contains vague term 'fast'",
    ...     severity=Severity.BLOCKING,
    ... )
    >>>
    >>> # Create a report
    >>> report = GateFailureReport(
    ...     gate_name="testability",
    ...     issues=(issue,),
    ...     score=0.75,
    ...     threshold=0.80,
    ...     summary="Score below threshold",
    ... )
    >>> report.to_dict()
    {'gate_name': 'testability', 'issues': [...], 'score': 0.75, ...}

Security Note:
    Gate reports may contain file paths and requirement content.
    Reports should not be exposed to untrusted parties without review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    """Severity level for gate issues.

    Determines how issues affect gate evaluation and workflow continuation.

    Attributes:
        BLOCKING: Issue prevents gate from passing and blocks workflow.
        WARNING: Issue is noted but doesn't prevent gate from passing.

    Example:
        >>> severity = Severity.BLOCKING
        >>> severity.value
        'blocking'
    """

    BLOCKING = "blocking"
    WARNING = "warning"


@dataclass(frozen=True)
class GateIssue:
    """Represents a single issue found during gate evaluation.

    Captures the location, type, description, and severity of an issue
    discovered during quality gate validation. Issues are immutable
    for audit trail integrity.

    Attributes:
        location: Where the issue was found (e.g., requirement ID, file path).
        issue_type: Category of issue (e.g., "vague_term", "coverage_gap").
        description: Human-readable description of the issue.
        severity: Whether this issue is blocking or advisory.
        context: Optional additional context about the issue.

    Example:
        >>> issue = GateIssue(
        ...     location="req-001",
        ...     issue_type="vague_term",
        ...     description="Contains vague term 'fast'",
        ...     severity=Severity.BLOCKING,
        ...     context={"term": "fast", "position": 42},
        ... )
        >>> issue.to_dict()
        {'location': 'req-001', 'issue_type': 'vague_term', ...}
    """

    location: str
    issue_type: str
    description: str
    severity: Severity
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary for logging and serialization.

        Converts the severity enum to its string value for JSON compatibility.

        Returns:
            Dictionary representation of the issue.

        Example:
            >>> issue = GateIssue(
            ...     location="test",
            ...     issue_type="test",
            ...     description="test",
            ...     severity=Severity.WARNING,
            ... )
            >>> issue.to_dict()["severity"]
            'warning'
        """
        return {
            "location": self.location,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity.value,
            "context": self.context,
        }


@dataclass(frozen=True)
class GateFailureReport:
    """Structured report of gate evaluation results.

    Aggregates all issues found during gate evaluation along with
    score and threshold information. Reports are immutable for
    audit trail integrity.

    Attributes:
        gate_name: Name of the gate that generated the report.
        issues: Tuple of issues found during evaluation (immutable).
        score: Numeric score (0.0-1.0) achieved.
        threshold: Required threshold for passing.
        summary: Brief summary of the evaluation result.

    Example:
        >>> issue = GateIssue(
        ...     location="req-001",
        ...     issue_type="vague_term",
        ...     description="Contains 'fast'",
        ...     severity=Severity.BLOCKING,
        ... )
        >>> report = GateFailureReport(
        ...     gate_name="testability",
        ...     issues=(issue,),
        ...     score=0.75,
        ...     threshold=0.80,
        ...     summary="Score 75% below threshold 80%",
        ... )
        >>> report.to_dict()
        {'gate_name': 'testability', 'issues': [...], ...}
    """

    gate_name: str
    issues: tuple[GateIssue, ...]
    score: float
    threshold: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for logging and serialization.

        Serializes all issues using their to_dict() method.

        Returns:
            Dictionary representation of the report.

        Example:
            >>> report = GateFailureReport(
            ...     gate_name="test",
            ...     issues=(),
            ...     score=1.0,
            ...     threshold=0.80,
            ...     summary="Passed",
            ... )
            >>> result = report.to_dict()
            >>> result["issues"]
            []
        """
        return {
            "gate_name": self.gate_name,
            "issues": [issue.to_dict() for issue in self.issues],
            "score": self.score,
            "threshold": self.threshold,
            "summary": self.summary,
        }
