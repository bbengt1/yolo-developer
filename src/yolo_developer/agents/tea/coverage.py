"""Coverage validation types and functions (Story 9.2).

This module provides the data types and functions used for test coverage validation:

- CoverageResult: Individual file coverage result with uncovered line ranges
- CoverageReport: Aggregate coverage report with threshold checking

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.coverage import (
    ...     CoverageResult,
    ...     CoverageReport,
    ... )
    >>>
    >>> result = CoverageResult(
    ...     file_path="src/module.py",
    ...     lines_total=100,
    ...     lines_covered=80,
    ...     coverage_percentage=80.0,
    ...     uncovered_lines=((10, 15), (30, 35)),
    ... )
    >>> result.to_dict()
    {'file_path': 'src/module.py', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from yolo_developer.agents.tea.types import Finding

logger = structlog.get_logger(__name__)

# Default critical paths that require 100% coverage
DEFAULT_CRITICAL_PATH_PATTERNS = (
    "orchestrator/",
    "gates/",
    "agents/",
)

# Default coverage threshold percentage (80%)
DEFAULT_COVERAGE_THRESHOLD = 80.0


def get_coverage_threshold_from_config() -> float:
    """Get coverage threshold from YoloConfig if available.

    Loads the threshold from the quality.test_coverage_threshold config
    and converts from ratio (0.0-1.0) to percentage (0.0-100.0).

    Returns:
        Coverage threshold as percentage (e.g., 80.0 for 80%).
        Defaults to 80.0 if config is unavailable or on expected errors.
    """
    try:
        from yolo_developer.config import load_config
        from yolo_developer.config.loader import ConfigurationError

        config = load_config()
        # Config stores as ratio (0.0-1.0), convert to percentage
        return config.quality.test_coverage_threshold * 100.0
    except FileNotFoundError:
        # No config file - expected in some contexts
        logger.debug("coverage_config_file_not_found_using_default")
        return DEFAULT_COVERAGE_THRESHOLD
    except ConfigurationError as e:
        # Invalid config - log warning but continue with default
        logger.warning("coverage_config_error_using_default", error=str(e))
        return DEFAULT_COVERAGE_THRESHOLD
    except ImportError:
        # Config module not available (e.g., during testing)
        logger.debug("coverage_config_module_not_available_using_default")
        return DEFAULT_COVERAGE_THRESHOLD
    except Exception as e:
        # Unexpected error - log warning for debugging
        logger.warning(
            "coverage_config_unexpected_error_using_default",
            error=str(e),
            error_type=type(e).__name__,
        )
        return DEFAULT_COVERAGE_THRESHOLD


def get_critical_paths_from_config() -> tuple[str, ...]:
    """Get critical path patterns from configuration.

    Loads the critical path patterns from YoloConfig.quality.critical_paths
    if available. These paths require 100% test coverage.

    Returns:
        Tuple of critical path patterns. Defaults to orchestrator/, gates/, agents/.
    """
    try:
        from yolo_developer.config import load_config
        from yolo_developer.config.loader import ConfigurationError

        config = load_config()
        # Return configured critical paths as tuple
        return tuple(config.quality.critical_paths)
    except FileNotFoundError:
        logger.debug("critical_paths_config_file_not_found_using_default")
        return DEFAULT_CRITICAL_PATH_PATTERNS
    except ConfigurationError as e:
        logger.warning("critical_paths_config_error_using_default", error=str(e))
        return DEFAULT_CRITICAL_PATH_PATTERNS
    except ImportError:
        logger.debug("critical_paths_config_module_not_available_using_default")
        return DEFAULT_CRITICAL_PATH_PATTERNS
    except Exception as e:
        logger.warning(
            "critical_paths_config_unexpected_error_using_default",
            error=str(e),
            error_type=type(e).__name__,
        )
        return DEFAULT_CRITICAL_PATH_PATTERNS


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CoverageResult:
    """Individual file coverage result with uncovered line ranges.

    Represents coverage analysis for a single code file, including
    line counts and specific uncovered line ranges.

    Attributes:
        file_path: Path to the code file being analyzed
        lines_total: Total number of executable lines in the file
        lines_covered: Number of lines covered by tests
        coverage_percentage: Coverage as a percentage (0.0 to 100.0)
        uncovered_lines: Tuple of (start, end) line ranges that are uncovered
        created_at: ISO timestamp when result was created

    Example:
        >>> result = CoverageResult(
        ...     file_path="src/module.py",
        ...     lines_total=100,
        ...     lines_covered=80,
        ...     coverage_percentage=80.0,
        ...     uncovered_lines=((10, 15), (30, 35)),
        ... )
        >>> result.coverage_percentage
        80.0
    """

    file_path: str
    lines_total: int
    lines_covered: int
    coverage_percentage: float
    uncovered_lines: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the coverage result.
        """
        return {
            "file_path": self.file_path,
            "lines_total": self.lines_total,
            "lines_covered": self.lines_covered,
            "coverage_percentage": self.coverage_percentage,
            "uncovered_lines": [list(r) for r in self.uncovered_lines],
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Aggregate coverage report with threshold checking.

    Contains coverage results for all analyzed files, plus aggregate
    metrics and threshold validation status.

    Attributes:
        results: Tuple of CoverageResult for each analyzed file
        overall_coverage: Weighted average coverage across all files (0.0 to 100.0)
        threshold: Coverage threshold required to pass (e.g., 80.0)
        passed: Whether overall coverage meets or exceeds threshold
        critical_files_coverage: Dict mapping critical file paths to their coverage
        created_at: ISO timestamp when report was created

    Example:
        >>> report = CoverageReport(
        ...     results=(result1, result2),
        ...     overall_coverage=85.0,
        ...     threshold=80.0,
        ...     passed=True,
        ...     critical_files_coverage={"src/core.py": 95.0},
        ... )
        >>> report.passed
        True
    """

    results: tuple[CoverageResult, ...] = field(default_factory=tuple)
    overall_coverage: float = 100.0
    threshold: float = 80.0
    passed: bool = True
    critical_files_coverage: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested coverage results.
        """
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_coverage": self.overall_coverage,
            "threshold": self.threshold,
            "passed": self.passed,
            "critical_files_coverage": dict(self.critical_files_coverage),
            "created_at": self.created_at,
        }


# =============================================================================
# Coverage Analysis Functions
# =============================================================================


def _is_critical_path(file_path: str) -> bool:
    """Check if a file path is in a critical path.

    Args:
        file_path: Path to check.

    Returns:
        True if file is in a critical path.
    """
    critical_patterns = get_critical_paths_from_config()
    return any(pattern in file_path for pattern in critical_patterns)


def _count_executable_lines(content: str) -> int:
    """Count executable lines in Python code.

    Counts lines that contain actual code (not just comments, docstrings,
    or blank lines).

    Args:
        content: Python source code content.

    Returns:
        Number of executable lines.
    """
    if not content.strip():
        return 0

    lines = content.splitlines()
    executable_count = 0
    in_multiline_string = False
    multiline_quote: str | None = None  # Track which quote style we're in

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip comment-only lines (but only if not in multiline string)
        if not in_multiline_string and stripped.startswith("#"):
            continue

        # Handle multiline string state
        if in_multiline_string:
            # Check if this line closes the multiline string
            if multiline_quote and multiline_quote in stripped:
                in_multiline_string = False
                multiline_quote = None
            # Don't count lines inside multiline strings
            continue

        # Check for triple quotes on this line
        has_triple_double = '"""' in stripped
        has_triple_single = "'''" in stripped

        if has_triple_double or has_triple_single:
            # Determine which quote type
            quote = '"""' if has_triple_double else "'''"

            # Count occurrences of the quote
            count = stripped.count(quote)

            if count == 1:
                # Single occurrence - entering or exiting multiline
                # If line starts with quote, it's a docstring start (skip)
                if stripped.startswith(quote):
                    in_multiline_string = True
                    multiline_quote = quote
                    continue
                # Quote appears mid-line - might be string in code
                # Count as executable
                executable_count += 1
            elif count >= 2:
                # Two or more quotes - could be complete docstring on one line
                # e.g., """docstring""" or def f(): """doc"""
                if stripped.startswith(quote):
                    # Standalone docstring line - skip
                    continue
                # Quote appears in code (e.g., x = """text""")
                executable_count += 1
        else:
            # No triple quotes - regular code line
            executable_count += 1

    return executable_count


def _extract_function_names(content: str) -> set[str]:
    """Extract function names from Python code.

    Args:
        content: Python source code content.

    Returns:
        Set of function names defined in the code.
    """
    # Match def function_name( or async def function_name(
    pattern = r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    matches = re.findall(pattern, content)
    return set(matches)


def _estimate_coverage(
    code_content: str,
    test_content: str,
    code_functions: set[str],
) -> float:
    """Estimate coverage based on test presence heuristics.

    This is a stub implementation for MVP. It estimates coverage based on:
    - Presence of test functions that reference code functions
    - Assertion density in test code

    Args:
        code_content: Source code content.
        test_content: Combined test file content.
        code_functions: Set of function names in the code.

    Returns:
        Estimated coverage percentage (0.0 to 100.0).
    """
    if not code_functions:
        # No functions to test = 100% covered
        return 100.0

    if not test_content.strip():
        # No tests = low coverage
        return 10.0

    # Count how many code functions appear to have tests
    tested_functions = 0
    for func_name in code_functions:
        # Look for test_<func_name> or <func_name> in test content
        if f"test_{func_name}" in test_content or func_name in test_content:
            tested_functions += 1

    base_coverage = (tested_functions / len(code_functions)) * 100.0

    # Boost coverage based on assertion density
    assertion_count = test_content.count("assert")
    assertion_boost = min(assertion_count * 2, 20)  # Max 20% boost

    return min(base_coverage + assertion_boost, 100.0)


def _get_indent_level(line: str) -> int:
    """Get the indentation level of a line, handling both spaces and tabs.

    Args:
        line: A line of code.

    Returns:
        Number of leading whitespace characters.
    """
    return len(line) - len(line.lstrip())


def _detect_uncovered_functions(
    code_content: str,
    code_functions: set[str],
    test_content: str,
) -> list[tuple[int, int]]:
    """Detect functions without test coverage.

    Args:
        code_content: Source code content.
        code_functions: Set of function names in the code.
        test_content: Combined test file content.

    Returns:
        List of (start_line, end_line) tuples for uncovered functions.
    """
    uncovered_ranges: list[tuple[int, int]] = []
    lines = code_content.splitlines()

    for func_name in code_functions:
        # Check if function appears to be tested
        if f"test_{func_name}" in test_content or func_name in test_content:
            continue

        # Find the function definition line
        for i, line in enumerate(lines):
            if re.match(rf"(?:async\s+)?def\s+{func_name}\s*\(", line.strip()):
                start_line = i + 1  # 1-indexed

                # Find end of function (next def at same or lower indent, or end of file)
                end_line = start_line
                func_indent = _get_indent_level(line)

                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    # Skip empty lines
                    if not next_line.strip():
                        end_line = j + 1
                        continue

                    next_indent = _get_indent_level(next_line)

                    # If we find a line at same or lower indent level that's a def,
                    # the function has ended
                    if next_indent <= func_indent:
                        if next_line.strip().startswith(("def ", "async def ", "class ")):
                            break
                        # Decorator or other top-level code also ends the function
                        if next_line.strip().startswith("@"):
                            break
                    end_line = j + 1

                uncovered_ranges.append((start_line, end_line))
                break

    return uncovered_ranges


def analyze_coverage(
    code_files: list[dict[str, Any]],
    test_files: list[dict[str, Any]],
    threshold: float = 80.0,
) -> CoverageReport:
    """Analyze coverage for a set of code files against test files.

    This is a heuristic-based implementation for MVP. It estimates coverage
    based on test file presence and function-level test matching.

    Args:
        code_files: List of code file dicts with 'artifact_id' and 'content' keys.
        test_files: List of test file dicts with 'artifact_id' and 'content' keys.
        threshold: Coverage threshold percentage (default 80.0).

    Returns:
        CoverageReport with coverage results and threshold status.

    Example:
        >>> code_files = [{"artifact_id": "src/module.py", "content": "..."}]
        >>> test_files = [{"artifact_id": "tests/test_module.py", "content": "..."}]
        >>> report = analyze_coverage(code_files, test_files)
        >>> report.passed
        True
    """
    if not code_files:
        logger.debug("no_code_files_to_analyze")
        return CoverageReport(
            results=(),
            overall_coverage=100.0,
            threshold=threshold,
            passed=True,
            critical_files_coverage={},
        )

    # Combine all test content for heuristic matching
    combined_test_content = "\n".join(tf.get("content", "") for tf in test_files)

    results: list[CoverageResult] = []
    critical_files_coverage: dict[str, float] = {}
    total_lines = 0
    total_covered = 0

    for code_file in code_files:
        file_path = code_file.get("artifact_id", "unknown")
        content = code_file.get("content", "")

        # Count executable lines
        lines_total = _count_executable_lines(content)

        if lines_total == 0:
            # Empty file - consider 100% covered
            result = CoverageResult(
                file_path=file_path,
                lines_total=0,
                lines_covered=0,
                coverage_percentage=100.0,
                uncovered_lines=(),
            )
            results.append(result)
            continue

        # Extract functions from code
        code_functions = _extract_function_names(content)

        # Estimate coverage
        coverage_percentage = _estimate_coverage(content, combined_test_content, code_functions)

        # Calculate lines covered based on percentage
        lines_covered = int(lines_total * coverage_percentage / 100.0)

        # Detect uncovered functions
        uncovered_ranges = _detect_uncovered_functions(
            content, code_functions, combined_test_content
        )

        result = CoverageResult(
            file_path=file_path,
            lines_total=lines_total,
            lines_covered=lines_covered,
            coverage_percentage=coverage_percentage,
            uncovered_lines=tuple(uncovered_ranges),
        )
        results.append(result)

        # Track critical files
        if _is_critical_path(file_path):
            critical_files_coverage[file_path] = coverage_percentage

        total_lines += lines_total
        total_covered += lines_covered

    # Calculate overall coverage
    if total_lines > 0:
        overall_coverage = (total_covered / total_lines) * 100.0
    else:
        overall_coverage = 100.0

    passed = overall_coverage >= threshold

    logger.info(
        "coverage_analysis_complete",
        file_count=len(results),
        overall_coverage=overall_coverage,
        threshold=threshold,
        passed=passed,
        critical_files=len(critical_files_coverage),
    )

    return CoverageReport(
        results=tuple(results),
        overall_coverage=overall_coverage,
        threshold=threshold,
        passed=passed,
        critical_files_coverage=critical_files_coverage,
    )


def validate_critical_paths(report: CoverageReport) -> list[Finding]:
    """Validate that critical paths have 100% coverage.

    Critical paths (orchestrator/, gates/, agents/) are required to have
    complete test coverage. Any gap generates a critical severity finding.

    Args:
        report: CoverageReport with coverage analysis results.

    Returns:
        List of Finding objects for critical paths below 100% coverage.

    Example:
        >>> report = analyze_coverage(code_files, test_files)
        >>> findings = validate_critical_paths(report)
        >>> len(findings)
        0  # All critical paths have 100% coverage
    """
    # Import here to avoid circular imports
    from yolo_developer.agents.tea.types import Finding

    findings: list[Finding] = []
    finding_counter = 1

    for file_path, coverage in report.critical_files_coverage.items():
        if coverage < 100.0:
            finding = Finding(
                finding_id=f"CRIT-{finding_counter:03d}",
                category="test_coverage",
                severity="critical",
                description=(
                    f"Critical path '{file_path}' has {coverage:.1f}% coverage. "
                    f"Critical paths require 100% test coverage."
                ),
                location=file_path,
                remediation=(
                    f"Add tests to achieve 100% coverage for {file_path}. "
                    f"Currently at {coverage:.1f}%, need to cover remaining "
                    f"{100.0 - coverage:.1f}% of code."
                ),
            )
            findings.append(finding)
            finding_counter += 1

            logger.warning(
                "critical_path_coverage_gap",
                file_path=file_path,
                coverage=coverage,
                required=100.0,
            )

    if not findings:
        logger.debug("all_critical_paths_fully_covered")

    return findings


def check_coverage_threshold(
    report: CoverageReport,
    threshold: float = 80.0,
) -> tuple[bool, list[Finding]]:
    """Check if coverage meets the required threshold.

    Compares overall coverage against the threshold and generates
    findings for any gaps, with severity based on the coverage level.

    Severity mapping:
    - critical: coverage < 50%
    - high: 50% <= coverage < threshold
    - medium: coverage is just below threshold (within 10%)

    Args:
        report: CoverageReport with coverage analysis results.
        threshold: Required coverage percentage (default 80.0).

    Returns:
        Tuple of (passed, findings):
        - passed: True if overall coverage >= threshold
        - findings: List of Finding objects for coverage gaps

    Example:
        >>> report = analyze_coverage(code_files, test_files)
        >>> passed, findings = check_coverage_threshold(report, threshold=80.0)
        >>> passed
        True
    """
    # Import here to avoid circular imports
    from yolo_developer.agents.tea.types import Finding, FindingSeverity

    findings: list[Finding] = []
    passed = report.overall_coverage >= threshold

    if passed:
        logger.debug(
            "coverage_threshold_passed",
            overall_coverage=report.overall_coverage,
            threshold=threshold,
        )
        return True, []

    # Determine severity based on coverage level
    severity: FindingSeverity
    if report.overall_coverage < 50.0:
        severity = "critical"
    elif report.overall_coverage < threshold:
        severity = "high"
    else:
        severity = "medium"

    # Create overall threshold finding
    gap = threshold - report.overall_coverage
    finding = Finding(
        finding_id="COV-THRESHOLD-001",
        category="test_coverage",
        severity=severity,
        description=(
            f"Overall coverage {report.overall_coverage:.1f}% is below threshold {threshold:.1f}%. "
            f"Gap: {gap:.1f}%."
        ),
        location="project",
        remediation=(
            f"Increase test coverage to at least {threshold:.1f}%. "
            f"Focus on files with lowest coverage to improve overall score."
        ),
    )
    findings.append(finding)

    # Add specific findings for under-covered files
    for result in report.results:
        if result.coverage_percentage < threshold:
            file_gap = threshold - result.coverage_percentage
            file_severity: FindingSeverity
            if result.coverage_percentage < 50.0:
                file_severity = "critical"
            elif result.coverage_percentage < threshold:
                file_severity = "high"
            else:
                file_severity = "medium"

            # Use hash of full path for unique finding IDs
            path_hash = hash(result.file_path) & 0xFFFFFFFF  # Ensure positive
            file_finding = Finding(
                finding_id=f"COV-FILE-{path_hash:08x}",
                category="test_coverage",
                severity=file_severity,
                description=(
                    f"File '{result.file_path}' has {result.coverage_percentage:.1f}% coverage "
                    f"(below {threshold:.1f}% threshold). "
                    f"Uncovered lines: {result.lines_total - result.lines_covered}."
                ),
                location=result.file_path,
                remediation=(
                    f"Add tests to cover {result.file_path}. "
                    f"Currently {result.lines_covered}/{result.lines_total} lines covered. "
                    f"Need to cover {file_gap:.1f}% more."
                ),
            )
            findings.append(file_finding)

    logger.warning(
        "coverage_threshold_failed",
        overall_coverage=report.overall_coverage,
        threshold=threshold,
        gap=gap,
        finding_count=len(findings),
    )

    return False, findings
