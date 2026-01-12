"""Gap analysis types and functions (Story 9.8).

This module provides data types and functions for test gap analysis:

- GapType: Literal type for gap classification
- GapSeverity: Literal type for gap severity levels
- SuggestionTestType: Literal type for suggested test types
- TestGap: A test coverage gap with location and description
- GapPriority: Priority scoring for a gap
- TestSuggestion: A suggested test to address a gap
- GapAnalysisSummary: Summary statistics for gap analysis
- GapAnalysisReport: Complete gap analysis report

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.gap_analysis import (
    ...     TestGap,
    ...     GapPriority,
    ...     TestSuggestion,
    ... )
    >>>
    >>> gap = TestGap(
    ...     gap_id="GAP-a1b2c3-001",
    ...     file_path="src/module.py",
    ...     function_names=("calculate_total",),
    ...     uncovered_lines=((10, 20),),
    ...     gap_type="no_tests",
    ...     description="Module has no tests",
    ... )
    >>> gap.to_dict()
    {'gap_id': 'GAP-a1b2c3-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from yolo_developer.agents.tea.coverage import CoverageReport

logger = structlog.get_logger(__name__)

# Default coverage threshold for gap analysis
DEFAULT_GAP_THRESHOLD = 80.0

# Time threshold constants for effort estimation (in minutes)
MINUTES_PER_HOUR = 60
MINUTES_4_HOURS = 240
MINUTES_8_HOURS = 480
MINUTES_PER_SUGGESTION = 20

# =============================================================================
# Literal Types
# =============================================================================

GapType = Literal[
    "no_tests",
    "partial_coverage",
    "untested_branch",
]
"""Classification of a test coverage gap.

Values:
    no_tests: Code file/function has no associated tests
    partial_coverage: Code has tests but coverage is below threshold
    untested_branch: Conditional branches without test coverage
"""

GapSeverity = Literal[
    "critical",
    "high",
    "medium",
    "low",
]
"""Severity level for test coverage gaps.

Values:
    critical: Gap in critical path, must be addressed immediately
    high: Significant gap that should be addressed soon
    medium: Moderate gap that should be addressed
    low: Minor gap that can be addressed when convenient
"""

SuggestionTestType = Literal[
    "unit",
    "integration",
    "e2e",
]
"""Type of test suggested to address a gap.

Values:
    unit: Unit test for isolated function/method testing
    integration: Integration test for component interaction testing
    e2e: End-to-end test for user flow testing
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TestGap:
    """A test coverage gap with location and description.

    Represents a specific area of code lacking adequate test coverage.

    Attributes:
        gap_id: Unique identifier for the gap (e.g., "GAP-a1b2c3-001")
        file_path: Path to the file containing the gap
        function_names: Tuple of untested function/method names
        uncovered_lines: Tuple of (start, end) line ranges without coverage
        gap_type: Classification of the gap type
        description: Human-readable description of the gap

    Example:
        >>> gap = TestGap(
        ...     gap_id="GAP-a1b2c3-001",
        ...     file_path="src/module.py",
        ...     function_names=("calculate_total",),
        ...     uncovered_lines=((10, 20),),
        ...     gap_type="no_tests",
        ...     description="Module has no tests",
        ... )
        >>> gap.gap_type
        'no_tests'
    """

    gap_id: str
    file_path: str
    function_names: tuple[str, ...] = field(default_factory=tuple)
    uncovered_lines: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    gap_type: GapType = "no_tests"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the gap.
        """
        return {
            "gap_id": self.gap_id,
            "file_path": self.file_path,
            "function_names": list(self.function_names),
            "uncovered_lines": [list(r) for r in self.uncovered_lines],
            "gap_type": self.gap_type,
            "description": self.description,
        }


@dataclass(frozen=True)
class GapPriority:
    """Priority scoring for a test coverage gap.

    Contains severity and risk scoring used to prioritize gap remediation.

    Attributes:
        gap_id: Reference to the associated TestGap
        severity: Severity level of the gap
        risk_score: Risk score from 0-100 (higher = more urgent)
        priority_rank: Ranking among all gaps (1 = highest priority)

    Example:
        >>> priority = GapPriority(
        ...     gap_id="GAP-a1b2c3-001",
        ...     severity="critical",
        ...     risk_score=95,
        ...     priority_rank=1,
        ... )
        >>> priority.severity
        'critical'
    """

    gap_id: str
    severity: GapSeverity
    risk_score: int
    priority_rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the priority.
        """
        return {
            "gap_id": self.gap_id,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "priority_rank": self.priority_rank,
        }


@dataclass(frozen=True)
class TestSuggestion:
    """A suggested test to address a coverage gap.

    Provides actionable guidance for addressing a specific gap.

    Attributes:
        suggestion_id: Unique identifier for the suggestion (e.g., "SUG-001")
        target_gap_id: Reference to the gap this addresses
        test_type: Type of test suggested
        description: What the test should verify
        estimated_impact: Estimated coverage improvement percentage
        example_signature: Suggested test function name

    Example:
        >>> suggestion = TestSuggestion(
        ...     suggestion_id="SUG-001",
        ...     target_gap_id="GAP-a1b2c3-001",
        ...     test_type="unit",
        ...     description="Test calculate_total returns correct sum",
        ...     estimated_impact=5.0,
        ...     example_signature="test_calculate_total_returns_correct_sum",
        ... )
        >>> suggestion.test_type
        'unit'
    """

    suggestion_id: str
    target_gap_id: str
    test_type: SuggestionTestType
    description: str
    estimated_impact: float
    example_signature: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the suggestion.
        """
        return {
            "suggestion_id": self.suggestion_id,
            "target_gap_id": self.target_gap_id,
            "test_type": self.test_type,
            "description": self.description,
            "estimated_impact": self.estimated_impact,
            "example_signature": self.example_signature,
        }


@dataclass(frozen=True)
class GapAnalysisSummary:
    """Summary statistics for gap analysis.

    Provides aggregate counts and effort estimates.

    Attributes:
        total_gaps: Total number of gaps identified
        critical_gaps: Count of critical severity gaps
        high_gaps: Count of high severity gaps
        medium_gaps: Count of medium severity gaps
        low_gaps: Count of low severity gaps
        total_suggestions: Total number of test suggestions
        estimated_effort: Human-readable effort estimate

    Example:
        >>> summary = GapAnalysisSummary(
        ...     total_gaps=10,
        ...     critical_gaps=2,
        ...     high_gaps=3,
        ...     medium_gaps=3,
        ...     low_gaps=2,
        ...     total_suggestions=15,
        ...     estimated_effort="2-4 hours",
        ... )
        >>> summary.total_gaps
        10
    """

    total_gaps: int
    critical_gaps: int
    high_gaps: int
    medium_gaps: int
    low_gaps: int
    total_suggestions: int
    estimated_effort: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "total_gaps": self.total_gaps,
            "critical_gaps": self.critical_gaps,
            "high_gaps": self.high_gaps,
            "medium_gaps": self.medium_gaps,
            "low_gaps": self.low_gaps,
            "total_suggestions": self.total_suggestions,
            "estimated_effort": self.estimated_effort,
        }


@dataclass(frozen=True)
class GapAnalysisReport:
    """Complete gap analysis report.

    Contains all gaps, priorities, suggestions, and summary statistics.

    Attributes:
        gaps: Tuple of TestGap objects identified
        priorities: Tuple of GapPriority objects for each gap
        suggestions: Tuple of TestSuggestion objects
        summary: GapAnalysisSummary with aggregate statistics
        coverage_baseline: Current coverage percentage
        projected_coverage: Projected coverage if all suggestions implemented
        created_at: ISO timestamp when report was created

    Example:
        >>> report = GapAnalysisReport(
        ...     gaps=(gap1, gap2),
        ...     priorities=(priority1, priority2),
        ...     suggestions=(suggestion1,),
        ...     summary=summary,
        ...     coverage_baseline=75.0,
        ...     projected_coverage=90.0,
        ... )
        >>> report.coverage_baseline
        75.0
    """

    gaps: tuple[TestGap, ...] = field(default_factory=tuple)
    priorities: tuple[GapPriority, ...] = field(default_factory=tuple)
    suggestions: tuple[TestSuggestion, ...] = field(default_factory=tuple)
    summary: GapAnalysisSummary = field(
        default_factory=lambda: GapAnalysisSummary(
            total_gaps=0,
            critical_gaps=0,
            high_gaps=0,
            medium_gaps=0,
            low_gaps=0,
            total_suggestions=0,
            estimated_effort="None",
        )
    )
    coverage_baseline: float = 100.0
    projected_coverage: float = 100.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested objects.
        """
        return {
            "gaps": [g.to_dict() for g in self.gaps],
            "priorities": [p.to_dict() for p in self.priorities],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "summary": self.summary.to_dict(),
            "coverage_baseline": self.coverage_baseline,
            "projected_coverage": self.projected_coverage,
            "created_at": self.created_at,
        }


# =============================================================================
# Gap Identification Functions (AC: 1)
# =============================================================================


def _generate_gap_id(file_path: str, sequence: int) -> str:
    """Generate unique gap ID from file path hash.

    Args:
        file_path: Path to the file with the gap.
        sequence: Sequence number for the gap within the file.

    Returns:
        Gap ID in format "GAP-{hash[:6]}-{seq:03d}".

    Example:
        >>> _generate_gap_id("src/module.py", 1)
        'GAP-a1b2c3-001'

    """
    # Use SHA256 for secure hashing (avoid MD5 per security scanners)
    path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:6]
    return f"GAP-{path_hash}-{sequence:03d}"


def identify_untested_functions(coverage_report: CoverageReport) -> list[TestGap]:
    """Identify files with no test coverage (0%).

    Args:
        coverage_report: CoverageReport with coverage analysis results.

    Returns:
        List of TestGap objects for files with 0% coverage.

    Example:
        >>> report = CoverageReport(results=(result,), ...)
        >>> gaps = identify_untested_functions(report)
    """
    gaps: list[TestGap] = []
    sequence = 1

    for result in coverage_report.results:
        if result.coverage_percentage == 0.0 and result.lines_total > 0:
            gap = TestGap(
                gap_id=_generate_gap_id(result.file_path, sequence),
                file_path=result.file_path,
                function_names=(),  # Could be extracted from code analysis
                uncovered_lines=result.uncovered_lines,
                gap_type="no_tests",
                description=f"File has no test coverage ({result.lines_total} lines uncovered)",
            )
            gaps.append(gap)
            sequence += 1

    logger.debug(
        "identified_untested_functions",
        gap_count=len(gaps),
    )

    return gaps


def identify_partial_coverage_gaps(
    coverage_report: CoverageReport,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> list[TestGap]:
    """Identify files with partial coverage below threshold.

    Args:
        coverage_report: CoverageReport with coverage analysis results.
        threshold: Coverage threshold percentage (default 80.0).

    Returns:
        List of TestGap objects for files with partial coverage.

    Example:
        >>> report = CoverageReport(results=(result,), ...)
        >>> gaps = identify_partial_coverage_gaps(report, threshold=80.0)
    """
    gaps: list[TestGap] = []
    sequence = 1

    for result in coverage_report.results:
        # Skip files with 0% (handled by identify_untested_functions)
        # and files meeting threshold
        if result.coverage_percentage > 0.0 and result.coverage_percentage < threshold:
            gap_percentage = threshold - result.coverage_percentage
            gap = TestGap(
                gap_id=_generate_gap_id(result.file_path, sequence),
                file_path=result.file_path,
                function_names=(),
                uncovered_lines=result.uncovered_lines,
                gap_type="partial_coverage",
                description=(
                    f"File has {result.coverage_percentage:.1f}% coverage "
                    f"({gap_percentage:.1f}% below {threshold:.1f}% threshold)"
                ),
            )
            gaps.append(gap)
            sequence += 1

    logger.debug(
        "identified_partial_coverage_gaps",
        gap_count=len(gaps),
        threshold=threshold,
    )

    return gaps


def identify_untested_branches(coverage_report: CoverageReport) -> list[TestGap]:
    """Identify files with uncovered branch/line ranges.

    Finds files that have test coverage but still have specific
    uncovered line ranges (potential branches without tests).

    Args:
        coverage_report: CoverageReport with coverage analysis results.

    Returns:
        List of TestGap objects for files with untested branches.

    Example:
        >>> report = CoverageReport(results=(result,), ...)
        >>> gaps = identify_untested_branches(report)
    """
    gaps: list[TestGap] = []
    sequence = 1

    for result in coverage_report.results:
        # Files with some coverage but still have uncovered ranges
        if (
            result.coverage_percentage > 0.0
            and result.coverage_percentage < 100.0
            and result.uncovered_lines
        ):
            # Only create branch gap if not already zero coverage
            gap = TestGap(
                gap_id=_generate_gap_id(result.file_path, sequence),
                file_path=result.file_path,
                function_names=(),
                uncovered_lines=result.uncovered_lines,
                gap_type="untested_branch",
                description=(
                    f"File has {len(result.uncovered_lines)} uncovered line range(s) "
                    f"({result.lines_total - result.lines_covered} lines)"
                ),
            )
            gaps.append(gap)
            sequence += 1

    logger.debug(
        "identified_untested_branches",
        gap_count=len(gaps),
    )

    return gaps


def identify_gaps(
    coverage_report: CoverageReport,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> tuple[TestGap, ...]:
    """Identify all test coverage gaps.

    Combines gap identification from:
    - identify_untested_functions (0% coverage)
    - identify_partial_coverage_gaps (below threshold)
    - identify_untested_branches (uncovered ranges)

    Args:
        coverage_report: CoverageReport with coverage analysis results.
        threshold: Coverage threshold percentage (default 80.0).

    Returns:
        Tuple of deduplicated TestGap objects.

    Example:
        >>> report = CoverageReport(results=(result,), ...)
        >>> gaps = identify_gaps(report)
        >>> len(gaps)
        5
    """
    all_gaps: list[TestGap] = []

    # Identify untested files (0% coverage)
    untested = identify_untested_functions(coverage_report)
    all_gaps.extend(untested)

    # Track files already identified
    untested_files = {g.file_path for g in untested}

    # Identify partial coverage gaps (not already in untested)
    partial = identify_partial_coverage_gaps(coverage_report, threshold)
    for gap in partial:
        if gap.file_path not in untested_files:
            all_gaps.append(gap)

    # Identify branch gaps (files with coverage but uncovered ranges)
    # Skip files already in untested or partial
    partial_files = {g.file_path for g in partial}
    branches = identify_untested_branches(coverage_report)
    for gap in branches:
        if gap.file_path not in untested_files and gap.file_path not in partial_files:
            all_gaps.append(gap)

    # Re-number gaps sequentially
    renumbered_gaps: list[TestGap] = []
    for i, gap in enumerate(all_gaps, start=1):
        renumbered = TestGap(
            gap_id=_generate_gap_id(gap.file_path, i),
            file_path=gap.file_path,
            function_names=gap.function_names,
            uncovered_lines=gap.uncovered_lines,
            gap_type=gap.gap_type,
            description=gap.description,
        )
        renumbered_gaps.append(renumbered)

    logger.info(
        "identified_all_gaps",
        total_gaps=len(renumbered_gaps),
        untested_count=len(untested),
        partial_count=len(partial),
        branch_count=len(branches),
    )

    return tuple(renumbered_gaps)


# =============================================================================
# Gap Prioritization Functions (AC: 2)
# =============================================================================


def _calculate_risk_score(gap: TestGap, critical_paths: tuple[str, ...]) -> int:
    """Calculate risk score for a gap (0-100).

    Risk is based on:
    - Critical path location (+50 points)
    - Gap type (no_tests=+30, partial=+20, branch=+10)
    - Number of uncovered lines (+1 per 10 lines, max +20)

    Args:
        gap: TestGap to score.
        critical_paths: Tuple of critical path patterns.

    Returns:
        Risk score from 0-100.
    """
    score = 0

    # Critical path bonus
    for pattern in critical_paths:
        if pattern in gap.file_path:
            score += 50
            break

    # Gap type scoring
    if gap.gap_type == "no_tests":
        score += 30
    elif gap.gap_type == "partial_coverage":
        score += 20
    else:  # untested_branch
        score += 10

    # Uncovered lines scoring
    total_uncovered = sum(end - start + 1 for start, end in gap.uncovered_lines)
    lines_score = min(total_uncovered // 10, 20)
    score += lines_score

    return min(score, 100)


def _determine_severity(
    gap: TestGap,
    risk_score: int,
    critical_paths: tuple[str, ...],
) -> GapSeverity:
    """Determine severity level for a gap.

    Args:
        gap: TestGap to evaluate.
        risk_score: Calculated risk score.
        critical_paths: Tuple of critical path patterns.

    Returns:
        Severity level.
    """
    # Critical paths always get critical severity
    for pattern in critical_paths:
        if pattern in gap.file_path:
            return "critical"

    # Map risk score to severity
    if risk_score >= 70:
        return "high"
    elif risk_score >= 40:
        return "medium"
    else:
        return "low"


def prioritize_gaps(
    gaps: tuple[TestGap, ...],
    critical_paths: tuple[str, ...] | None = None,
) -> tuple[GapPriority, ...]:
    """Prioritize gaps by severity and risk score.

    Args:
        gaps: Tuple of TestGap objects.
        critical_paths: Optional tuple of critical path patterns.

    Returns:
        Tuple of GapPriority objects sorted by priority (1 = highest).
    """
    if critical_paths is None:
        critical_paths = ()

    # Calculate priorities
    priorities: list[tuple[GapPriority, int]] = []
    for gap in gaps:
        risk_score = _calculate_risk_score(gap, critical_paths)
        severity = _determine_severity(gap, risk_score, critical_paths)
        priority = GapPriority(
            gap_id=gap.gap_id,
            severity=severity,
            risk_score=risk_score,
            priority_rank=0,  # Temporary, will be assigned after sorting
        )
        priorities.append((priority, risk_score))

    # Sort by severity (critical > high > medium > low) then by risk score
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    priorities.sort(key=lambda x: (severity_order[x[0].severity], -x[1]))

    # Assign ranks
    ranked: list[GapPriority] = []
    for rank, (priority, _) in enumerate(priorities, start=1):
        ranked.append(
            GapPriority(
                gap_id=priority.gap_id,
                severity=priority.severity,
                risk_score=priority.risk_score,
                priority_rank=rank,
            )
        )

    logger.debug(
        "prioritized_gaps",
        gap_count=len(ranked),
        critical_count=sum(1 for p in ranked if p.severity == "critical"),
    )

    return tuple(ranked)


# =============================================================================
# Test Suggestion Generation Functions (AC: 3)
# =============================================================================


def _generate_suggestion_id(sequence: int) -> str:
    """Generate suggestion ID.

    Args:
        sequence: Sequence number.

    Returns:
        Suggestion ID in format "SUG-{seq:03d}".
    """
    return f"SUG-{sequence:03d}"


def _determine_test_type(gap: TestGap) -> SuggestionTestType:
    """Determine appropriate test type for a gap.

    Args:
        gap: TestGap to analyze.

    Returns:
        Suggested test type.
    """
    # Check for integration indicators
    integration_patterns = ("api", "service", "client", "handler", "endpoint")
    file_lower = gap.file_path.lower()
    for pattern in integration_patterns:
        if pattern in file_lower:
            return "integration"

    # Check for e2e indicators
    e2e_patterns = ("cli", "command", "workflow", "orchestrat")
    for pattern in e2e_patterns:
        if pattern in file_lower:
            return "e2e"

    # Default to unit
    return "unit"


def _generate_test_signature(gap: TestGap) -> str:
    """Generate suggested test function name.

    Args:
        gap: TestGap to generate signature for.

    Returns:
        Suggested test function name.
    """
    # Extract module name from file path
    file_name = gap.file_path.split("/")[-1].replace(".py", "")

    # If we have function names, use the first one
    if gap.function_names:
        func_name = gap.function_names[0]
        return f"test_{func_name}"

    # Otherwise use module name
    return f"test_{file_name}_basic_functionality"


def _estimate_impact(gap: TestGap, coverage_baseline: float) -> float:
    """Estimate coverage improvement from addressing a gap.

    Args:
        gap: TestGap to estimate.
        coverage_baseline: Current coverage percentage.

    Returns:
        Estimated coverage improvement percentage.
    """
    # Estimate based on uncovered lines
    total_uncovered = sum(end - start + 1 for start, end in gap.uncovered_lines)

    # Simple heuristic: each test might cover ~10-20 lines
    # Assume covering 50% of the uncovered lines
    potential_lines = total_uncovered * 0.5

    # Rough estimate of impact (varies by project size)
    # Assume project has ~1000 executable lines for estimation
    estimated_project_lines = max(1000, total_uncovered * 10)
    impact = (potential_lines / estimated_project_lines) * 100

    return max(impact, 0.5)  # Minimum 0.5% impact


def generate_test_suggestions(
    gaps: tuple[TestGap, ...],
    priorities: tuple[GapPriority, ...],
    coverage_baseline: float = 0.0,
    max_suggestions: int = 20,
) -> tuple[TestSuggestion, ...]:
    """Generate test suggestions for gaps.

    Args:
        gaps: Tuple of TestGap objects.
        priorities: Tuple of GapPriority objects.
        coverage_baseline: Current coverage percentage.
        max_suggestions: Maximum number of suggestions to generate.

    Returns:
        Tuple of TestSuggestion objects.
    """
    # Create gap lookup
    gap_map = {g.gap_id: g for g in gaps}

    # Sort by priority rank
    sorted_priorities = sorted(priorities, key=lambda p: p.priority_rank)

    suggestions: list[TestSuggestion] = []
    for i, priority in enumerate(sorted_priorities[:max_suggestions], start=1):
        gap = gap_map.get(priority.gap_id)
        if not gap:
            continue

        test_type = _determine_test_type(gap)
        signature = _generate_test_signature(gap)
        impact = _estimate_impact(gap, coverage_baseline)

        suggestion = TestSuggestion(
            suggestion_id=_generate_suggestion_id(i),
            target_gap_id=gap.gap_id,
            test_type=test_type,
            description=f"Add {test_type} test for {gap.file_path}",
            estimated_impact=impact,
            example_signature=signature,
        )
        suggestions.append(suggestion)

    logger.debug(
        "generated_test_suggestions",
        suggestion_count=len(suggestions),
        max_suggestions=max_suggestions,
    )

    return tuple(suggestions)


# =============================================================================
# Summary Generation Functions (AC: 5)
# =============================================================================


def _count_gaps_by_severity(priorities: tuple[GapPriority, ...]) -> dict[GapSeverity, int]:
    """Count gaps by severity level.

    Args:
        priorities: Tuple of GapPriority objects.

    Returns:
        Dict mapping severity to count.
    """
    counts: dict[GapSeverity, int] = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for priority in priorities:
        counts[priority.severity] += 1

    return counts


def _estimate_total_effort(
    suggestions: tuple[TestSuggestion, ...],
) -> str:
    """Estimate total effort to address all gaps.

    Args:
        suggestions: Tuple of TestSuggestion objects.

    Returns:
        Human-readable effort estimate.

    """
    if not suggestions:
        return "No effort required"

    # Estimate ~15-30 minutes per test suggestion
    total_minutes = len(suggestions) * MINUTES_PER_SUGGESTION

    if total_minutes < MINUTES_PER_HOUR:
        return f"{total_minutes} minutes"
    if total_minutes < MINUTES_4_HOURS:
        hours = total_minutes / MINUTES_PER_HOUR
        return f"{hours:.1f}-{hours * 1.5:.1f} hours"
    if total_minutes < MINUTES_8_HOURS:
        return "0.5-1 day"
    days = total_minutes / MINUTES_8_HOURS
    return f"{days:.0f}-{days * 1.5:.0f} days"


def generate_summary(
    gaps: tuple[TestGap, ...],
    priorities: tuple[GapPriority, ...],
    suggestions: tuple[TestSuggestion, ...],
) -> GapAnalysisSummary:
    """Generate gap analysis summary.

    Args:
        gaps: Tuple of TestGap objects.
        priorities: Tuple of GapPriority objects.
        suggestions: Tuple of TestSuggestion objects.

    Returns:
        GapAnalysisSummary with aggregate statistics.

    """
    _ = gaps  # Reserved for future use (e.g., per-gap effort estimation)
    counts = _count_gaps_by_severity(priorities)
    effort = _estimate_total_effort(suggestions)

    return GapAnalysisSummary(
        total_gaps=len(gaps),
        critical_gaps=counts["critical"],
        high_gaps=counts["high"],
        medium_gaps=counts["medium"],
        low_gaps=counts["low"],
        total_suggestions=len(suggestions),
        estimated_effort=effort,
    )


# =============================================================================
# Report Generation Functions (AC: 4)
# =============================================================================


def generate_gap_analysis_report(
    coverage_report: CoverageReport,
    critical_paths: tuple[str, ...] | None = None,
    max_suggestions: int = 20,
) -> GapAnalysisReport:
    """Generate complete gap analysis report.

    Runs the full gap analysis pipeline:
    1. Identify gaps from coverage data
    2. Prioritize gaps
    3. Generate test suggestions
    4. Create summary statistics
    5. Calculate projected coverage

    Args:
        coverage_report: CoverageReport with coverage data.
        critical_paths: Optional tuple of critical path patterns.
        max_suggestions: Maximum test suggestions to generate.

    Returns:
        Complete GapAnalysisReport.
    """
    # Import here to avoid circular imports at module load
    from yolo_developer.agents.tea.coverage import get_critical_paths_from_config

    # Use config paths if not provided
    if critical_paths is None:
        critical_paths = get_critical_paths_from_config()

    # Step 1: Identify gaps
    gaps = identify_gaps(coverage_report)

    # Step 2: Prioritize gaps
    priorities = prioritize_gaps(gaps, critical_paths)

    # Step 3: Generate suggestions
    coverage_baseline = coverage_report.overall_coverage
    suggestions = generate_test_suggestions(
        gaps,
        priorities,
        coverage_baseline=coverage_baseline,
        max_suggestions=max_suggestions,
    )

    # Step 4: Generate summary
    summary = generate_summary(gaps, priorities, suggestions)

    # Step 5: Calculate projected coverage
    total_impact = sum(s.estimated_impact for s in suggestions)
    projected_coverage = min(coverage_baseline + total_impact, 100.0)

    logger.info(
        "generated_gap_analysis_report",
        gap_count=len(gaps),
        suggestion_count=len(suggestions),
        coverage_baseline=coverage_baseline,
        projected_coverage=projected_coverage,
    )

    return GapAnalysisReport(
        gaps=gaps,
        priorities=priorities,
        suggestions=suggestions,
        summary=summary,
        coverage_baseline=coverage_baseline,
        projected_coverage=projected_coverage,
    )


# =============================================================================
# Export Functions (AC: 6)
# =============================================================================


def export_to_json(report: GapAnalysisReport) -> str:
    """Export report to JSON format.

    Args:
        report: GapAnalysisReport to export.

    Returns:
        JSON string representation.
    """
    return json.dumps(report.to_dict(), indent=2)


def export_to_markdown(report: GapAnalysisReport) -> str:
    """Export report to Markdown format.

    Args:
        report: GapAnalysisReport to export.

    Returns:
        Markdown string representation.
    """
    lines: list[str] = []

    lines.append("# Gap Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {report.created_at}")
    lines.append(f"**Coverage Baseline:** {report.coverage_baseline:.1f}%")
    lines.append(f"**Projected Coverage:** {report.projected_coverage:.1f}%")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Gaps:** {report.summary.total_gaps}")
    lines.append(f"- **Critical:** {report.summary.critical_gaps}")
    lines.append(f"- **High:** {report.summary.high_gaps}")
    lines.append(f"- **Medium:** {report.summary.medium_gaps}")
    lines.append(f"- **Low:** {report.summary.low_gaps}")
    lines.append(f"- **Total Suggestions:** {report.summary.total_suggestions}")
    lines.append(f"- **Estimated Effort:** {report.summary.estimated_effort}")
    lines.append("")

    # Gaps section
    lines.append("## Gaps")
    lines.append("")
    if report.gaps:
        # Create priority lookup
        priority_map = {p.gap_id: p for p in report.priorities}

        for gap in report.gaps:
            priority = priority_map.get(gap.gap_id)
            severity = priority.severity if priority else "unknown"
            rank = priority.priority_rank if priority else 0

            lines.append(f"### {gap.gap_id} ({severity.upper()}, Rank #{rank})")
            lines.append("")
            lines.append(f"- **File:** `{gap.file_path}`")
            lines.append(f"- **Type:** {gap.gap_type}")
            lines.append(f"- **Description:** {gap.description}")
            if gap.uncovered_lines:
                ranges = ", ".join(f"{s}-{e}" for s, e in gap.uncovered_lines)
                lines.append(f"- **Uncovered Lines:** {ranges}")
            lines.append("")
    else:
        lines.append("No gaps identified.")
        lines.append("")

    # Suggestions section
    lines.append("## Suggestions")
    lines.append("")
    if report.suggestions:
        for suggestion in report.suggestions:
            lines.append(f"### {suggestion.suggestion_id}")
            lines.append("")
            lines.append(f"- **Target Gap:** {suggestion.target_gap_id}")
            lines.append(f"- **Test Type:** {suggestion.test_type}")
            lines.append(f"- **Description:** {suggestion.description}")
            lines.append(f"- **Estimated Impact:** {suggestion.estimated_impact:.1f}%")
            lines.append(f"- **Example:** `{suggestion.example_signature}`")
            lines.append("")
    else:
        lines.append("No suggestions generated.")
        lines.append("")

    return "\n".join(lines)


def export_to_csv(report: GapAnalysisReport) -> str:
    """Export gaps and priorities to CSV format.

    Args:
        report: GapAnalysisReport to export.

    Returns:
        CSV string with gaps and priorities.
    """
    lines: list[str] = []

    # Header
    lines.append("gap_id,file_path,gap_type,severity,risk_score,priority_rank,description")

    # Create priority lookup
    priority_map = {p.gap_id: p for p in report.priorities}

    # Data rows
    for gap in report.gaps:
        priority = priority_map.get(gap.gap_id)
        severity = priority.severity if priority else ""
        risk_score = priority.risk_score if priority else 0
        rank = priority.priority_rank if priority else 0

        # Escape description for CSV
        description = gap.description.replace('"', '""')

        lines.append(
            f'{gap.gap_id},{gap.file_path},{gap.gap_type},{severity},{risk_score},{rank},"{description}"'
        )

    return "\n".join(lines)
