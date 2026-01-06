"""Report generator utility for quality gate failures (Story 3.8 - Task 2).

This module provides utilities for generating structured failure reports
and formatting them for human-readable output.

Functions:
    generate_failure_report: Create a structured GateFailureReport
    format_report_text: Format a report as human-readable text

Example:
    >>> from yolo_developer.gates.report_generator import (
    ...     generate_failure_report, format_report_text
    ... )
    >>> from yolo_developer.gates.report_types import GateIssue, Severity
    >>>
    >>> issues = [
    ...     GateIssue("req-001", "vague_term", "Contains 'fast'", Severity.BLOCKING)
    ... ]
    >>> report = generate_failure_report("testability", issues, 0.75, 0.80)
    >>> print(format_report_text(report))
    Testability Gate Report
    ...

Security Note:
    Reports may contain file paths and code references.
    Review output before exposing to external parties.
"""

from __future__ import annotations

import structlog

from yolo_developer.gates.remediation import get_remediation_suggestion
from yolo_developer.gates.report_types import GateFailureReport, GateIssue, Severity

logger = structlog.get_logger(__name__)


def generate_failure_report(
    gate_name: str,
    issues: list[GateIssue],
    score: float,
    threshold: float,
) -> GateFailureReport:
    """Generate a structured failure report for a gate.

    Creates a GateFailureReport with summary statistics about the
    issues found during gate evaluation.

    Args:
        gate_name: Name of the gate that was evaluated.
        issues: List of issues found during evaluation.
        score: Achieved score (0.0-1.0).
        threshold: Required threshold for passing.

    Returns:
        Structured GateFailureReport with summary.

    Example:
        >>> issues = [GateIssue("r1", "vague_term", "Issue", Severity.BLOCKING)]
        >>> report = generate_failure_report("testability", issues, 0.75, 0.80)
        >>> report.gate_name
        'testability'
        >>> "75%" in report.summary
        True
    """
    blocking_count = sum(1 for i in issues if i.severity == Severity.BLOCKING)
    warning_count = sum(1 for i in issues if i.severity == Severity.WARNING)

    score_pct = int(score * 100)
    threshold_pct = int(threshold * 100)

    summary = (
        f"{gate_name} score {score_pct}% below threshold {threshold_pct}%. "
        f"Found {blocking_count} blocking issue(s) and {warning_count} warning(s)."
    )

    logger.debug(
        "gate_failure_report_generated",
        gate_name=gate_name,
        score=score,
        threshold=threshold,
        blocking_count=blocking_count,
        warning_count=warning_count,
        total_issues=len(issues),
    )

    return GateFailureReport(
        gate_name=gate_name,
        issues=tuple(issues),
        score=score,
        threshold=threshold,
        summary=summary,
    )


def format_report_text(report: GateFailureReport) -> str:
    """Format a failure report as human-readable text.

    Creates a formatted text report suitable for CLI or log output.
    Groups issues by location and includes remediation suggestions.

    Args:
        report: The failure report to format.

    Returns:
        Formatted text string suitable for display.

    Example:
        >>> from yolo_developer.gates.report_types import GateFailureReport
        >>> report = GateFailureReport(
        ...     gate_name="testability",
        ...     issues=(),
        ...     score=1.0,
        ...     threshold=0.80,
        ...     summary="All passed",
        ... )
        >>> text = format_report_text(report)
        >>> "Testability Gate Report" in text
        True
    """
    # Format gate name for display (replace underscores, title case)
    display_name = report.gate_name.replace("_", " ").title()

    lines: list[str] = [
        f"{display_name} Gate Report",
        "=" * 50,
        "",
        report.summary,
        "",
    ]

    if not report.issues:
        return "\n".join(lines)

    # Group issues by location
    issues_by_location: dict[str, list[GateIssue]] = {}
    for issue in report.issues:
        if issue.location not in issues_by_location:
            issues_by_location[issue.location] = []
        issues_by_location[issue.location].append(issue)

    # Sort locations alphabetically
    for location in sorted(issues_by_location.keys()):
        location_issues = issues_by_location[location]
        lines.append(f"Location: {location}")
        lines.append("-" * 40)

        for issue in location_issues:
            severity_marker = "[BLOCKING]" if issue.severity == Severity.BLOCKING else "[WARNING]"
            lines.append(f"  {severity_marker} {issue.description}")

            # Add remediation suggestion if available
            suggestion = get_remediation_suggestion(issue.issue_type, report.gate_name)
            if suggestion:
                lines.append(f"    Suggestion: {suggestion}")

        lines.append("")

    return "\n".join(lines)
