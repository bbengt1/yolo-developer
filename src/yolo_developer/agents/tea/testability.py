"""Testability audit types and functions (Story 9.6).

This module provides types and functions for auditing code testability:

- TestabilityPattern: Literal type for testability anti-patterns
- TestabilitySeverity: Literal type for issue severity levels
- TestabilityIssue: A testability issue found during audit
- TestabilityScore: Testability score with breakdown
- TestabilityMetrics: Aggregated testability metrics
- TestabilityReport: Complete testability audit report

The testability audit identifies hard-to-test patterns:
- Global state/singletons
- Tightly coupled dependencies
- Hidden dependencies (imports inside functions)
- Complex conditionals (cyclomatic complexity > 10)
- Long methods (> 50 lines)
- Deep nesting (> 4 levels)

Example:
    >>> from yolo_developer.agents.tea.testability import (
    ...     TestabilityIssue,
    ...     TestabilityReport,
    ...     audit_testability,
    ... )
    >>>
    >>> code_files = [{"artifact_id": "src/module.py", "content": "..."}]
    >>> report = audit_testability(code_files)
    >>> report.score.score
    85

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from yolo_developer.agents.tea.types import Finding

logger = structlog.get_logger(__name__)

# =============================================================================
# Literal Types
# =============================================================================

TestabilityPattern = Literal[
    "global_state",
    "tight_coupling",
    "hidden_dependency",
    "complex_conditional",
    "long_method",
    "deep_nesting",
]
"""Types of testability anti-patterns detected.

Values:
    global_state: Module-level mutable variables, singleton patterns
    tight_coupling: Direct instantiation in methods (no dependency injection)
    hidden_dependency: Imports inside functions, dynamic imports
    complex_conditional: Cyclomatic complexity > 10
    long_method: Functions > 50 lines
    deep_nesting: Nesting > 4 levels deep
"""

TestabilitySeverity = Literal[
    "critical",
    "high",
    "medium",
    "low",
]
"""Severity level for testability issues.

Values:
    critical: global_state - hardest to test, causes flaky tests
    high: tight_coupling, hidden_dependency - prevents mocking
    medium: complex_conditional, long_method - many test cases needed
    low: deep_nesting - complex but manageable
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TestabilityIssue:
    """A testability issue found during code audit.

    Attributes:
        issue_id: Unique identifier (e.g., "T-a1b2c3d4-001")
        pattern_type: The testability anti-pattern detected
        severity: Impact severity on testability
        location: File path where issue was found
        line_start: Starting line number
        line_end: Ending line number
        description: Human-readable description
        impact: How this affects testability
        remediation: Suggested fix
        created_at: ISO timestamp when issue was created

    Example:
        >>> issue = TestabilityIssue(
        ...     issue_id="T-a1b2c3d4-001",
        ...     pattern_type="global_state",
        ...     severity="critical",
        ...     location="src/module.py",
        ...     line_start=10,
        ...     line_end=15,
        ...     description="Module-level mutable variable",
        ...     impact="Cannot isolate tests",
        ...     remediation="Use dependency injection",
        ... )
        >>> issue.to_dict()
        {'issue_id': 'T-a1b2c3d4-001', ...}
    """

    issue_id: str
    pattern_type: TestabilityPattern
    severity: TestabilitySeverity
    location: str
    line_start: int
    line_end: int
    description: str
    impact: str
    remediation: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the issue.
        """
        return {
            "issue_id": self.issue_id,
            "pattern_type": self.pattern_type,
            "severity": self.severity,
            "location": self.location,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "description": self.description,
            "impact": self.impact,
            "remediation": self.remediation,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TestabilityScore:
    """Testability score with breakdown of penalties.

    Attributes:
        score: Final testability score (0-100)
        base_score: Starting score before penalties (always 100)
        breakdown: Penalties applied per severity category

    Example:
        >>> score = TestabilityScore(
        ...     score=75,
        ...     base_score=100,
        ...     breakdown={"critical_penalty": -20, "high_penalty": -5},
        ... )
        >>> score.to_dict()
        {'score': 75, 'base_score': 100, 'breakdown': {...}}
    """

    score: int
    base_score: int
    breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the score.
        """
        return {
            "score": self.score,
            "base_score": self.base_score,
            "breakdown": dict(self.breakdown),
        }


@dataclass(frozen=True)
class TestabilityMetrics:
    """Aggregated testability metrics.

    Attributes:
        total_issues: Count of all issues found
        issues_by_severity: Count of issues per severity level
        issues_by_pattern: Count of issues per pattern type
        files_analyzed: Number of files processed
        files_with_issues: Number of files with at least one issue

    Example:
        >>> metrics = TestabilityMetrics(
        ...     total_issues=5,
        ...     issues_by_severity={"critical": 1, "high": 2},
        ...     issues_by_pattern={"global_state": 1},
        ...     files_analyzed=10,
        ...     files_with_issues=3,
        ... )
        >>> metrics.to_dict()
        {'total_issues': 5, ...}
    """

    total_issues: int
    issues_by_severity: dict[str, int]
    issues_by_pattern: dict[str, int]
    files_analyzed: int
    files_with_issues: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the metrics.
        """
        return {
            "total_issues": self.total_issues,
            "issues_by_severity": dict(self.issues_by_severity),
            "issues_by_pattern": dict(self.issues_by_pattern),
            "files_analyzed": self.files_analyzed,
            "files_with_issues": self.files_with_issues,
        }


@dataclass(frozen=True)
class TestabilityReport:
    """Complete testability audit report.

    Attributes:
        issues: Tuple of all testability issues found
        score: Testability score with breakdown
        metrics: Aggregated metrics about issues
        recommendations: Prioritized improvement suggestions
        created_at: ISO timestamp when report was created

    Example:
        >>> report = TestabilityReport(
        ...     issues=(issue,),
        ...     score=score,
        ...     metrics=metrics,
        ...     recommendations=("Fix global state",),
        ... )
        >>> report.to_dict()
        {'issues': [...], 'score': {...}, ...}
    """

    issues: tuple[TestabilityIssue, ...]
    score: TestabilityScore
    metrics: TestabilityMetrics
    recommendations: tuple[str, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested objects.
        """
        return {
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score.to_dict(),
            "metrics": self.metrics.to_dict(),
            "recommendations": list(self.recommendations),
            "created_at": self.created_at,
        }


# =============================================================================
# Impact and Remediation Templates
# =============================================================================

IMPACT_TEMPLATES: dict[TestabilityPattern, str] = {
    "global_state": "Global state cannot be isolated between tests, causing flaky tests and making parallel test execution unsafe",
    "tight_coupling": "Tight coupling prevents mocking dependencies, requiring integration tests instead of fast unit tests",
    "hidden_dependency": "Hidden dependencies make test setup surprising and difficult to mock correctly",
    "complex_conditional": "Complex conditionals require many test cases to achieve coverage and are prone to untested edge cases",
    "long_method": "Long methods have unclear test boundaries, requiring overly complex test setups",
    "deep_nesting": "Deep nesting creates complex state combinations that are difficult to test exhaustively",
}

REMEDIATION_TEMPLATES: dict[TestabilityPattern, str] = {
    "global_state": "Extract global state into a class that can be injected, or use dependency injection pattern",
    "tight_coupling": "Accept dependencies as constructor parameters instead of creating them internally",
    "hidden_dependency": "Move imports to module level and accept dependencies as parameters",
    "complex_conditional": "Extract conditional branches into separate methods with clear responsibilities",
    "long_method": "Split into smaller, focused methods following single responsibility principle",
    "deep_nesting": "Use early returns, extract methods, or use guard clauses to reduce nesting",
}

# Severity mapping for patterns
PATTERN_SEVERITY_MAP: dict[TestabilityPattern, TestabilitySeverity] = {
    "global_state": "critical",
    "tight_coupling": "high",
    "hidden_dependency": "high",
    "complex_conditional": "medium",
    "long_method": "medium",
    "deep_nesting": "low",
}

# Penalty values per severity
SEVERITY_PENALTIES: dict[TestabilitySeverity, int] = {
    "critical": 20,
    "high": 10,
    "medium": 5,
    "low": 2,
}

# Maximum penalties per severity (caps)
SEVERITY_PENALTY_CAPS: dict[TestabilitySeverity, int] = {
    "critical": 60,
    "high": 40,
    "medium": 20,
    "low": 10,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_issue_id(file_path: str, sequence: int) -> str:
    """Generate unique issue ID from file path and sequence.

    Uses SHA256 with 12-character prefix for better collision resistance
    in large codebases (48 bits vs 32 bits with MD5/8 chars).

    Args:
        file_path: Path to the file containing the issue.
        sequence: Sequence number for issues in this file.

    Returns:
        Issue ID in format "T-{file_hash[:12]}-{seq:03d}".

    Example:
        >>> _generate_issue_id("src/module.py", 1)
        'T-a1b2c3d4e5f6-001'
    """
    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:12]
    return f"T-{file_hash}-{sequence:03d}"


def _map_pattern_to_severity(pattern: TestabilityPattern) -> TestabilitySeverity:
    """Map testability pattern to severity level.

    Args:
        pattern: The testability pattern detected.

    Returns:
        Severity level for the pattern.
    """
    return PATTERN_SEVERITY_MAP[pattern]


def _get_impact_description(pattern: TestabilityPattern) -> str:
    """Get impact description for a pattern.

    Args:
        pattern: The testability pattern.

    Returns:
        Human-readable impact description.
    """
    return IMPACT_TEMPLATES[pattern]


def _get_remediation_suggestion(pattern: TestabilityPattern) -> str:
    """Get remediation suggestion for a pattern.

    Args:
        pattern: The testability pattern.

    Returns:
        Actionable remediation suggestion.
    """
    return REMEDIATION_TEMPLATES[pattern]


# =============================================================================
# Pattern Detection Functions
# =============================================================================


def _detect_global_state(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect global state patterns in code.

    Identifies:
    - Module-level mutable variables (lists, dicts, sets)
    - Module-level dict(), list(), set() calls

    Note:
        Singleton patterns (e.g., _instance class variables, __new__ overrides)
        are not currently detected. This function focuses on obvious module-level
        mutable state that directly impacts test isolation.

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for global state patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0

    # Check module-level assignments
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check if it's a mutable type assignment
                    is_mutable = False
                    var_name = target.id

                    # Skip private constants (ALL_CAPS) and type annotations
                    if var_name.isupper() or var_name.startswith("_"):
                        continue

                    if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                        is_mutable = True
                    elif isinstance(node.value, ast.Call):
                        # Check for dict(), list(), set() calls
                        if isinstance(node.value.func, ast.Name):
                            if node.value.func.id in ("dict", "list", "set"):
                                is_mutable = True

                    if is_mutable:
                        seq += 1
                        line_start = node.lineno
                        line_end = node.end_lineno or node.lineno

                        issues.append(
                            TestabilityIssue(
                                issue_id=_generate_issue_id(file_path, seq),
                                pattern_type="global_state",
                                severity="critical",
                                location=file_path,
                                line_start=line_start,
                                line_end=line_end,
                                description=f"Module-level mutable variable '{var_name}'",
                                impact=_get_impact_description("global_state"),
                                remediation=_get_remediation_suggestion("global_state"),
                            )
                        )

    logger.debug(
        "global_state_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


def _detect_tight_coupling(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect tight coupling patterns in code.

    Identifies:
    - Direct instantiation in __init__ methods
    - Classes that create their own dependencies

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for tight coupling patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0

    # Find all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # Find __init__ method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # Look for self.x = SomeClass() patterns
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"
                                    and isinstance(stmt.value, ast.Call)
                                ):
                                    # Check if it's a class instantiation (capitalized name)
                                    if isinstance(stmt.value.func, ast.Name):
                                        func_name = stmt.value.func.id
                                        if func_name[0].isupper() and func_name not in (
                                            "Dict",
                                            "List",
                                            "Set",
                                            "Tuple",
                                            "Optional",
                                            "Any",
                                        ):
                                            seq += 1
                                            line_start = stmt.lineno
                                            line_end = stmt.end_lineno or stmt.lineno

                                            issues.append(
                                                TestabilityIssue(
                                                    issue_id=_generate_issue_id(file_path, seq),
                                                    pattern_type="tight_coupling",
                                                    severity="high",
                                                    location=file_path,
                                                    line_start=line_start,
                                                    line_end=line_end,
                                                    description=f"Class '{class_name}' creates its own dependency '{func_name}' in __init__",
                                                    impact=_get_impact_description(
                                                        "tight_coupling"
                                                    ),
                                                    remediation=_get_remediation_suggestion(
                                                        "tight_coupling"
                                                    ),
                                                )
                                            )

    logger.debug(
        "tight_coupling_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


def _detect_hidden_dependencies(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect hidden dependency patterns in code.

    Identifies:
    - Imports inside functions
    - Dynamic imports using __import__

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for hidden dependency patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0

    # Find imports inside functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name

            for stmt in ast.walk(node):
                # Check for import statements inside function
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    # Skip if it's at function level (not nested in conditionals)
                    seq += 1
                    line_start = stmt.lineno
                    line_end = stmt.end_lineno or stmt.lineno

                    if isinstance(stmt, ast.Import):
                        import_names = ", ".join(alias.name for alias in stmt.names)
                    else:
                        module = stmt.module or ""
                        import_names = f"{module}.{', '.join(alias.name for alias in stmt.names)}"

                    issues.append(
                        TestabilityIssue(
                            issue_id=_generate_issue_id(file_path, seq),
                            pattern_type="hidden_dependency",
                            severity="high",
                            location=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            description=f"Import '{import_names}' inside function '{func_name}'",
                            impact=_get_impact_description("hidden_dependency"),
                            remediation=_get_remediation_suggestion("hidden_dependency"),
                        )
                    )

    logger.debug(
        "hidden_dependency_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


def _calculate_cyclomatic_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Calculate McCabe cyclomatic complexity for a function.

    Complexity = 1 + number of decision points.
    Decision points: if, elif, for, while, except, with, and, or, assert, comprehension.

    Args:
        func_node: AST node for the function.

    Returns:
        Cyclomatic complexity score.
    """
    complexity = 1  # Base complexity

    for node in ast.walk(func_node):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Count number of additional boolean operations
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Assert):
            complexity += 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1

    return complexity


def _detect_complex_conditionals(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect complex conditional patterns in code.

    Identifies functions with cyclomatic complexity > 10.

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for complex conditional patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0
    complexity_threshold = 10

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = _calculate_cyclomatic_complexity(node)

            if complexity > complexity_threshold:
                seq += 1
                line_start = node.lineno
                line_end = node.end_lineno or node.lineno

                issues.append(
                    TestabilityIssue(
                        issue_id=_generate_issue_id(file_path, seq),
                        pattern_type="complex_conditional",
                        severity="medium",
                        location=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        description=f"Function '{node.name}' has cyclomatic complexity of {complexity} (> {complexity_threshold})",
                        impact=_get_impact_description("complex_conditional"),
                        remediation=_get_remediation_suggestion("complex_conditional"),
                    )
                )

    logger.debug(
        "complex_conditional_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


def _detect_long_methods(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect long method patterns in code.

    Identifies functions > 50 lines.

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for long method patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0
    length_threshold = 50

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.end_lineno and node.lineno:
                length = node.end_lineno - node.lineno + 1
                if length > length_threshold:
                    seq += 1

                    issues.append(
                        TestabilityIssue(
                            issue_id=_generate_issue_id(file_path, seq),
                            pattern_type="long_method",
                            severity="medium",
                            location=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            description=f"Function '{node.name}' is {length} lines long (> {length_threshold})",
                            impact=_get_impact_description("long_method"),
                            remediation=_get_remediation_suggestion("long_method"),
                        )
                    )

    logger.debug(
        "long_method_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


def _get_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth for a node.

    Args:
        node: AST node to analyze.
        current_depth: Current nesting level.

    Returns:
        Maximum nesting depth found.
    """
    max_depth = current_depth

    nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_nodes):
            child_depth = _get_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _get_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

    return max_depth


def _detect_deep_nesting(content: str, file_path: str) -> list[TestabilityIssue]:
    """Detect deep nesting patterns in code.

    Identifies nesting > 4 levels deep.

    Args:
        content: Python source code content.
        file_path: Path to the file being analyzed.

    Returns:
        List of TestabilityIssue for deep nesting patterns found.
    """
    issues: list[TestabilityIssue] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.debug("syntax_error_in_file", file_path=file_path)
        return issues

    seq = 0
    nesting_threshold = 4

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = _get_nesting_depth(node)

            if depth > nesting_threshold:
                seq += 1
                line_start = node.lineno
                line_end = node.end_lineno or node.lineno

                issues.append(
                    TestabilityIssue(
                        issue_id=_generate_issue_id(file_path, seq),
                        pattern_type="deep_nesting",
                        severity="low",
                        location=file_path,
                        line_start=line_start,
                        line_end=line_end,
                        description=f"Function '{node.name}' has nesting depth of {depth} (> {nesting_threshold})",
                        impact=_get_impact_description("deep_nesting"),
                        remediation=_get_remediation_suggestion("deep_nesting"),
                    )
                )

    logger.debug(
        "deep_nesting_detection_complete",
        file_path=file_path,
        issues_found=len(issues),
    )

    return issues


# =============================================================================
# Score Calculation
# =============================================================================


def calculate_testability_score(issues: tuple[TestabilityIssue, ...]) -> TestabilityScore:
    """Calculate testability score from issues.

    Score starts at 100 and penalties are applied per severity:
    - Critical: -20 per occurrence (capped at -60)
    - High: -10 per occurrence (capped at -40)
    - Medium: -5 per occurrence (capped at -20)
    - Low: -2 per occurrence (capped at -10)

    Args:
        issues: Tuple of testability issues found.

    Returns:
        TestabilityScore with score and breakdown.
    """
    base_score = 100

    # Count issues by severity
    severity_counts: dict[TestabilitySeverity, int] = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for issue in issues:
        severity_counts[issue.severity] += 1

    # Calculate penalties with caps
    breakdown: dict[str, int] = {}
    total_penalty = 0

    for severity, count in severity_counts.items():
        raw_penalty = count * SEVERITY_PENALTIES[severity]
        capped_penalty = min(raw_penalty, SEVERITY_PENALTY_CAPS[severity])
        breakdown[f"{severity}_penalty"] = -capped_penalty
        total_penalty += capped_penalty

    # Calculate final score (clamped to 0-100)
    final_score = max(0, min(100, base_score - total_penalty))

    logger.debug(
        "testability_score_calculated",
        base_score=base_score,
        total_penalty=total_penalty,
        final_score=final_score,
        breakdown=breakdown,
    )

    return TestabilityScore(
        score=final_score,
        base_score=base_score,
        breakdown=breakdown,
    )


# =============================================================================
# Metrics Collection
# =============================================================================


def collect_testability_metrics(
    issues: tuple[TestabilityIssue, ...],
    files_analyzed: int,
) -> TestabilityMetrics:
    """Collect testability metrics from issues.

    Args:
        issues: Tuple of testability issues found.
        files_analyzed: Number of files processed.

    Returns:
        TestabilityMetrics with aggregated data.
    """
    # Count by severity
    issues_by_severity: dict[str, int] = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    # Count by pattern
    issues_by_pattern: dict[str, int] = {}

    # Track unique files with issues
    files_with_issues_set: set[str] = set()

    for issue in issues:
        issues_by_severity[issue.severity] += 1

        if issue.pattern_type not in issues_by_pattern:
            issues_by_pattern[issue.pattern_type] = 0
        issues_by_pattern[issue.pattern_type] += 1

        files_with_issues_set.add(issue.location)

    return TestabilityMetrics(
        total_issues=len(issues),
        issues_by_severity=issues_by_severity,
        issues_by_pattern=issues_by_pattern,
        files_analyzed=files_analyzed,
        files_with_issues=len(files_with_issues_set),
    )


# =============================================================================
# Recommendation Generation
# =============================================================================


def generate_testability_recommendations(
    issues: tuple[TestabilityIssue, ...],
) -> tuple[str, ...]:
    """Generate prioritized recommendations from issues.

    Recommendations are sorted by severity (critical first) and grouped
    by pattern type for consolidated suggestions.

    Args:
        issues: Tuple of testability issues found.

    Returns:
        Tuple of prioritized recommendation strings.
    """
    if not issues:
        return ()

    # Sort by severity (critical > high > medium > low)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(issues, key=lambda i: severity_order[i.severity])

    recommendations: list[str] = []
    seen_patterns: set[tuple[str, str]] = set()  # (pattern, location) pairs

    for issue in sorted_issues:
        # Avoid duplicate recommendations for same pattern in same file
        key = (issue.pattern_type, issue.location)
        if key in seen_patterns:
            continue
        seen_patterns.add(key)

        severity_label = issue.severity.upper()
        recommendation = f"{severity_label}: {issue.description} - {issue.remediation}"
        recommendations.append(recommendation)

    return tuple(recommendations)


# =============================================================================
# Main Audit Function
# =============================================================================


def audit_testability(code_files: list[dict[str, Any]]) -> TestabilityReport:
    """Run testability audit on code files.

    Analyzes code for testability anti-patterns and generates a comprehensive report.

    Args:
        code_files: List of dictionaries with 'artifact_id' (file path) and 'content' keys.

    Returns:
        TestabilityReport with issues, score, metrics, and recommendations.

    Example:
        >>> code_files = [{"artifact_id": "src/module.py", "content": "..."}]
        >>> report = audit_testability(code_files)
        >>> report.score.score
        85
    """
    all_issues: list[TestabilityIssue] = []
    files_analyzed = 0

    for code_file in code_files:
        file_path = code_file.get("artifact_id", "unknown")
        content = code_file.get("content", "")

        if not content:
            continue

        files_analyzed += 1

        # Run all pattern detectors
        all_issues.extend(_detect_global_state(content, file_path))
        all_issues.extend(_detect_tight_coupling(content, file_path))
        all_issues.extend(_detect_hidden_dependencies(content, file_path))
        all_issues.extend(_detect_complex_conditionals(content, file_path))
        all_issues.extend(_detect_long_methods(content, file_path))
        all_issues.extend(_detect_deep_nesting(content, file_path))

    issues_tuple = tuple(all_issues)

    # Calculate score
    score = calculate_testability_score(issues_tuple)

    # Collect metrics
    metrics = collect_testability_metrics(issues_tuple, files_analyzed)

    # Generate recommendations
    recommendations = generate_testability_recommendations(issues_tuple)

    logger.info(
        "testability_audit_complete",
        files_analyzed=files_analyzed,
        total_issues=len(all_issues),
        score=score.score,
    )

    return TestabilityReport(
        issues=issues_tuple,
        score=score,
        metrics=metrics,
        recommendations=recommendations,
    )


# =============================================================================
# Finding Conversion
# =============================================================================


def convert_testability_issues_to_findings(
    issues: tuple[TestabilityIssue, ...],
) -> tuple[Finding, ...]:
    """Convert testability issues to TEA Findings.

    Maps TestabilityIssue to Finding for integration with TEA validation.

    Args:
        issues: Tuple of testability issues.

    Returns:
        Tuple of Finding objects.
    """
    from yolo_developer.agents.tea.types import Finding, FindingSeverity

    # Map testability severity to finding severity
    severity_map: dict[TestabilitySeverity, FindingSeverity] = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }

    findings: list[Finding] = []

    for issue in issues:
        finding = Finding(
            finding_id=issue.issue_id.replace("T-", "F-"),
            category="code_quality",
            severity=severity_map[issue.severity],
            description=f"Testability: {issue.description}",
            location=f"{issue.location}:{issue.line_start}-{issue.line_end}",
            remediation=issue.remediation,
        )
        findings.append(finding)

    return tuple(findings)
