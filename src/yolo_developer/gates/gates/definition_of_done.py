"""Definition of Done gate implementation.

This module implements the DoD quality gate that validates code against
a checklist before proceeding:
- Test presence verification
- Documentation presence check
- Code style compliance validation
- Acceptance criteria coverage

Example:
    >>> from yolo_developer.gates.gates.definition_of_done import (
    ...     definition_of_done_evaluator,
    ...     check_test_presence,
    ...     check_documentation,
    ... )
    >>> from yolo_developer.gates.types import GateContext
    >>>
    >>> context = GateContext(
    ...     state={"code": {...}, "story": {...}},
    ...     gate_name="definition_of_done",
    ... )
    >>> result = await definition_of_done_evaluator(context)

Security Note:
    This gate performs read-only validation of code structure.
    It does not execute any user code.
"""

from __future__ import annotations

import ast
import re
from enum import Enum
from typing import Any

import structlog

from yolo_developer.gates.evaluators import register_evaluator
from yolo_developer.gates.report_generator import format_report_text, generate_failure_report
from yolo_developer.gates.report_types import GateIssue, Severity
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Types and Constants
# =============================================================================


class DoDCategory(Enum):
    """Category for Definition of Done checklist items.

    Attributes:
        TESTS: Test-related checks (unit tests, coverage).
        DOCUMENTATION: Documentation checks (docstrings, comments).
        STYLE: Code style checks (naming, complexity, types).
        AC_COVERAGE: Acceptance criteria coverage checks.
    """

    TESTS = "tests"
    DOCUMENTATION = "documentation"
    STYLE = "style"
    AC_COVERAGE = "ac_coverage"


def _map_severity(original_severity: str) -> Severity:
    """Map original severity level to Severity enum.

    Args:
        original_severity: Original severity string (high, medium, low).

    Returns:
        Severity.BLOCKING for high, Severity.WARNING for medium/low.
    """
    if original_severity == "high":
        return Severity.BLOCKING
    return Severity.WARNING


# Checklist items for each category
DOD_CHECKLIST_ITEMS: dict[DoDCategory, list[str]] = {
    DoDCategory.TESTS: [
        "unit_tests_present",
        "public_functions_covered",
        "edge_cases_tested",
    ],
    DoDCategory.DOCUMENTATION: [
        "module_docstring",
        "public_api_docstrings",
        "complex_logic_comments",
    ],
    DoDCategory.STYLE: [
        "type_annotations",
        "naming_conventions",
        "function_complexity",
    ],
    DoDCategory.AC_COVERAGE: [
        "all_ac_addressed",
        "ac_tests_exist",
    ],
}

# Severity weights for score calculation
SEVERITY_WEIGHTS: dict[str, int] = {
    "high": 20,
    "medium": 10,
    "low": 3,
}

# Default compliance threshold (0.0-1.0 decimal format)
DEFAULT_DOD_THRESHOLD = 0.70


# =============================================================================
# Helper Functions for Code Analysis
# =============================================================================


def _extract_functions_from_content(content: str) -> list[dict[str, Any]]:
    """Extract function information from Python source code.

    Args:
        content: Python source code as string.

    Returns:
        List of dicts with function info (name, has_docstring, has_types, etc.).
    """
    functions: list[dict[str, Any]] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return functions

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_info: dict[str, Any] = {
                "name": node.name,
                "is_private": node.name.startswith("_"),
                "has_docstring": ast.get_docstring(node) is not None,
                "has_return_type": node.returns is not None,
                "has_arg_types": all(
                    arg.annotation is not None for arg in node.args.args if arg.arg != "self"
                ),
                "line_count": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                "nesting_depth": _calculate_nesting_depth(node),
            }
            functions.append(func_info)

    return functions


def _calculate_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth in a function.

    Args:
        node: AST node to analyze.
        current_depth: Current nesting level.

    Returns:
        Maximum nesting depth found.
    """
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If | ast.For | ast.While | ast.With | ast.Try):
            child_depth = _calculate_nesting_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _calculate_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
    return max_depth


def _has_module_docstring(content: str) -> bool:
    """Check if content has a module-level docstring.

    Args:
        content: Python source code as string.

    Returns:
        True if module has a docstring.
    """
    try:
        tree = ast.parse(content)
        return ast.get_docstring(tree) is not None
    except SyntaxError:
        return False


def _is_naming_convention_valid(name: str) -> bool:
    """Check if function name follows snake_case convention.

    Args:
        name: Function name to check.

    Returns:
        True if name follows snake_case.
    """
    # Private functions start with underscore
    if name.startswith("_"):
        name = name.lstrip("_")
    # snake_case: lowercase with underscores
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name))


def _extract_test_function_names(content: str) -> set[str]:
    """Extract test function names from test file content.

    Args:
        content: Python test file content.

    Returns:
        Set of test function names.
    """
    test_names: set[str] = set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return test_names

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("test_"):
                test_names.add(node.name)

    return test_names


# =============================================================================
# Task 2: Test Presence Detection
# =============================================================================


def check_test_presence(code: dict[str, Any], story: dict[str, Any]) -> list[GateIssue]:
    """Check for test presence in the code.

    Detects public functions and verifies corresponding tests exist.

    Args:
        code: Code artifact dict with 'files' key.
        story: Story dict with acceptance criteria.

    Returns:
        List of GateIssue for missing tests.
    """
    issues: list[GateIssue] = []
    files = code.get("files", [])

    # Separate source and test files
    source_files = [f for f in files if not f.get("path", "").startswith("tests/")]
    test_files = [f for f in files if f.get("path", "").startswith("tests/")]

    # Collect all test function names
    all_test_names: set[str] = set()
    for test_file in test_files:
        content = test_file.get("content", "")
        all_test_names.update(_extract_test_function_names(content))

    # Check each source file
    for source_file in source_files:
        path = source_file.get("path", "unknown")
        content = source_file.get("content", "")
        functions = _extract_functions_from_content(content)

        public_functions = [f for f in functions if not f["is_private"]]

        for func in public_functions:
            func_name = func["name"]
            # Check if a test exists for this function
            expected_test_name = f"test_{func_name}"
            has_test = any(
                expected_test_name in test_name or func_name in test_name
                for test_name in all_test_names
            )

            if not has_test:
                original_severity = "high"
                issues.append(
                    GateIssue(
                        location=func_name,
                        issue_type=f"test_missing_{func_name}",
                        description=f"Public function '{func_name}' in {path} has no corresponding test",
                        severity=_map_severity(original_severity),
                        context={
                            "original_severity": original_severity,
                            "category": DoDCategory.TESTS.value,
                            "remediation": f"Add a test function 'test_{func_name}' that covers the functionality of '{func_name}'",
                        },
                    )
                )

    # Check if test files exist at all
    if not test_files and source_files:
        original_severity = "high"
        issues.append(
            GateIssue(
                location="tests",
                issue_type="no_test_files",
                description="No test files found in the codebase",
                severity=_map_severity(original_severity),
                context={
                    "original_severity": original_severity,
                    "category": DoDCategory.TESTS.value,
                    "remediation": "Create test files in the tests/ directory to cover the implementation",
                },
            )
        )

    logger.debug("check_test_presence_complete", issue_count=len(issues))
    return issues


# =============================================================================
# Task 3: Documentation Check
# =============================================================================


def check_documentation(code: dict[str, Any]) -> list[GateIssue]:
    """Check for documentation presence in the code.

    Verifies module docstrings and function docstrings exist.

    Args:
        code: Code artifact dict with 'files' key.

    Returns:
        List of GateIssue for missing documentation.
    """
    issues: list[GateIssue] = []
    files = code.get("files", [])

    # Only check source files (not tests)
    source_files = [f for f in files if not f.get("path", "").startswith("tests/")]

    for source_file in source_files:
        path = source_file.get("path", "unknown")
        content = source_file.get("content", "")

        # Check module docstring
        if not _has_module_docstring(content):
            original_severity = "medium"
            issues.append(
                GateIssue(
                    location=path,
                    issue_type=f"missing_module_docstring_{path}",
                    description=f"Module '{path}' is missing a module-level docstring",
                    severity=_map_severity(original_severity),
                    context={
                        "original_severity": original_severity,
                        "category": DoDCategory.DOCUMENTATION.value,
                        "remediation": f"Add a module docstring at the top of {path} describing its purpose and usage",
                    },
                )
            )

        # Check function docstrings
        functions = _extract_functions_from_content(content)
        public_functions = [f for f in functions if not f["is_private"]]

        for func in public_functions:
            if not func["has_docstring"]:
                original_severity = "medium"
                issues.append(
                    GateIssue(
                        location=func["name"],
                        issue_type=f"missing_docstring_{func['name']}",
                        description=f"Public function '{func['name']}' in {path} is missing a docstring",
                        severity=_map_severity(original_severity),
                        context={
                            "original_severity": original_severity,
                            "category": DoDCategory.DOCUMENTATION.value,
                            "remediation": f"Add a docstring to '{func['name']}' describing its purpose, args, and return value",
                        },
                    )
                )

    logger.debug("check_documentation_complete", issue_count=len(issues))
    return issues


# =============================================================================
# Task 4: Code Style Validation
# =============================================================================


def check_code_style(code: dict[str, Any]) -> list[GateIssue]:
    """Check for code style compliance.

    Validates type annotations, naming conventions, and complexity.

    Args:
        code: Code artifact dict with 'files' key.

    Returns:
        List of GateIssue for style violations.
    """
    issues: list[GateIssue] = []
    files = code.get("files", [])

    # Only check source files
    source_files = [f for f in files if not f.get("path", "").startswith("tests/")]

    for source_file in source_files:
        path = source_file.get("path", "unknown")
        content = source_file.get("content", "")
        functions = _extract_functions_from_content(content)

        for func in functions:
            func_name = func["name"]

            # Check type annotations (skip private functions for this check)
            if not func["is_private"]:
                if not func["has_return_type"]:
                    original_severity = "medium"
                    issues.append(
                        GateIssue(
                            location=func_name,
                            issue_type=f"missing_return_type_{func_name}",
                            description=f"Function '{func_name}' in {path} is missing return type annotation",
                            severity=_map_severity(original_severity),
                            context={
                                "original_severity": original_severity,
                                "category": DoDCategory.STYLE.value,
                                "remediation": f"Add return type annotation to '{func_name}' (e.g., '-> ReturnType')",
                            },
                        )
                    )

                if not func["has_arg_types"]:
                    original_severity = "medium"
                    issues.append(
                        GateIssue(
                            location=func_name,
                            issue_type=f"missing_arg_types_{func_name}",
                            description=f"Function '{func_name}' in {path} is missing argument type annotations",
                            severity=_map_severity(original_severity),
                            context={
                                "original_severity": original_severity,
                                "category": DoDCategory.STYLE.value,
                                "remediation": f"Add type annotations to arguments of '{func_name}' (e.g., 'arg: Type')",
                            },
                        )
                    )

            # Check naming conventions
            if not _is_naming_convention_valid(func_name):
                original_severity = "medium"
                issues.append(
                    GateIssue(
                        location=func_name,
                        issue_type=f"naming_violation_{func_name}",
                        description=f"Function '{func_name}' in {path} violates snake_case naming convention",
                        severity=_map_severity(original_severity),
                        context={
                            "original_severity": original_severity,
                            "category": DoDCategory.STYLE.value,
                            "remediation": f"Rename '{func_name}' to follow snake_case convention (lowercase with underscores)",
                        },
                    )
                )

            # Check complexity (>20 lines or >4 nesting)
            if func["line_count"] > 20:
                original_severity = "low"
                issues.append(
                    GateIssue(
                        location=func_name,
                        issue_type=f"long_function_{func_name}",
                        description=f"Function '{func_name}' in {path} is too long ({func['line_count']} lines, max 20)",
                        severity=_map_severity(original_severity),
                        context={
                            "original_severity": original_severity,
                            "category": DoDCategory.STYLE.value,
                            "remediation": f"Refactor '{func_name}' into smaller functions to improve readability",
                        },
                    )
                )

            if func["nesting_depth"] > 4:
                original_severity = "medium"
                issues.append(
                    GateIssue(
                        location=func_name,
                        issue_type=f"deep_nesting_{func_name}",
                        description=f"Function '{func_name}' in {path} has excessive nesting (depth {func['nesting_depth']}, max 4)",
                        severity=_map_severity(original_severity),
                        context={
                            "original_severity": original_severity,
                            "category": DoDCategory.STYLE.value,
                            "remediation": f"Reduce nesting in '{func_name}' using early returns, guard clauses, or extraction",
                        },
                    )
                )

    logger.debug("check_code_style_complete", issue_count=len(issues))
    return issues


# =============================================================================
# Task 5: AC Coverage Check
# =============================================================================


def check_ac_coverage(code: dict[str, Any], story: dict[str, Any]) -> list[GateIssue]:
    """Check acceptance criteria coverage in the code.

    Matches ACs to implementation evidence in code (comments, function names, etc.).

    Args:
        code: Code artifact dict with 'files' key.
        story: Story dict with 'acceptance_criteria' key.

    Returns:
        List of GateIssue for unaddressed ACs.
    """
    issues: list[GateIssue] = []
    acceptance_criteria = story.get("acceptance_criteria", [])

    if not acceptance_criteria:
        logger.debug("check_ac_coverage_no_criteria")
        return issues

    files = code.get("files", [])

    # Combine all file content for searching
    all_content = " ".join(f.get("content", "") for f in files).lower()

    addressed_count = 0
    total_count = len(acceptance_criteria)

    for ac in acceptance_criteria:
        ac_text = ac if isinstance(ac, str) else str(ac)

        # Extract AC identifier (e.g., "AC1", "AC2")
        ac_match = re.search(r"AC\s*(\d+)", ac_text, re.IGNORECASE)
        ac_id = ac_match.group(0) if ac_match else ac_text[:20]

        # Extract key terms from AC for matching
        # Remove common words and extract meaningful terms
        ac_lower = ac_text.lower()
        key_terms = re.findall(r"\b[a-z_]{4,}\b", ac_lower)
        key_terms = [
            t
            for t in key_terms
            if t not in ("given", "when", "then", "that", "should", "must", "with", "from")
        ]

        # Check if AC is addressed in code
        ac_addressed = False

        # Check for AC reference in comments
        if ac_id.lower() in all_content:
            ac_addressed = True
        # Check for key terms presence (need at least 60% of them)
        elif key_terms:
            matched_terms = sum(1 for term in key_terms if term in all_content)
            if matched_terms >= len(key_terms) * 0.6:
                ac_addressed = True

        if ac_addressed:
            addressed_count += 1
        else:
            original_severity = "high"
            issues.append(
                GateIssue(
                    location=ac_id,
                    issue_type=f"unaddressed_ac_{ac_id}",
                    description=f"Acceptance criterion '{ac_id}' appears not addressed in the implementation",
                    severity=_map_severity(original_severity),
                    context={
                        "original_severity": original_severity,
                        "category": DoDCategory.AC_COVERAGE.value,
                        "remediation": f"Ensure implementation addresses: {ac_text[:100]}...",
                    },
                )
            )

    # Add coverage percentage info
    if total_count > 0:
        coverage_pct = int((addressed_count / total_count) * 100)
        if coverage_pct < 100:
            original_severity = "high" if coverage_pct < 50 else "medium"
            issues.append(
                GateIssue(
                    location="ac_coverage",
                    issue_type="ac_coverage_incomplete",
                    description=f"AC coverage is {coverage_pct}% ({addressed_count}/{total_count} criteria addressed)",
                    severity=_map_severity(original_severity),
                    context={
                        "original_severity": original_severity,
                        "category": DoDCategory.AC_COVERAGE.value,
                        "remediation": "Review and implement all acceptance criteria to achieve 100% coverage",
                    },
                )
            )

    logger.debug("check_ac_coverage_complete", addressed=addressed_count, total=total_count)
    return issues


# =============================================================================
# Task 6: Checklist Result Generation
# =============================================================================


def generate_dod_checklist(issues: list[GateIssue]) -> dict[str, Any]:
    """Generate DoD checklist from issues.

    Groups issues by category and calculates compliance score.

    Args:
        issues: List of GateIssue objects.

    Returns:
        Dict with score, breakdown, and categorized issues.
    """
    # Group issues by category
    categorized: dict[str, list[dict[str, Any]]] = {cat.value: [] for cat in DoDCategory}

    for issue in issues:
        category = issue.context.get("category", DoDCategory.TESTS.value)
        categorized[category].append(
            {
                "check_id": issue.issue_type,
                "description": issue.description,
                "severity": issue.context.get("original_severity", "medium"),
                "item_name": issue.location,
                "remediation": issue.context.get("remediation", ""),
            }
        )

    # Calculate score using original severity weights
    total_deductions = sum(
        SEVERITY_WEIGHTS.get(issue.context.get("original_severity", "medium"), 0)
        for issue in issues
    )
    score = max(0, 100 - total_deductions)

    # Calculate breakdown using original severity
    breakdown: dict[str, int] = {
        "high_count": sum(1 for i in issues if i.context.get("original_severity") == "high"),
        "medium_count": sum(1 for i in issues if i.context.get("original_severity") == "medium"),
        "low_count": sum(1 for i in issues if i.context.get("original_severity") == "low"),
        "total_deductions": total_deductions,
    }

    checklist = {
        "score": score,
        "breakdown": breakdown,
        **categorized,
    }

    logger.debug("generate_dod_checklist_complete", score=score, issue_count=len(issues))
    return checklist


# =============================================================================
# Task 7: DoD Evaluator
# =============================================================================


async def definition_of_done_evaluator(context: GateContext) -> GateResult:
    """Evaluate code against Definition of Done checklist.

    Main evaluator function that runs all DoD checks and returns a gate result.

    Args:
        context: Gate context with state containing 'code' and 'story'.

    Returns:
        GateResult indicating pass/fail with details.
    """
    state = context.state

    # Extract code from state (try 'code' first, then 'implementation')
    code = state.get("code") or state.get("implementation")
    if not code or not isinstance(code, dict):
        return GateResult(
            passed=False,
            gate_name=context.gate_name,
            reason="Missing or invalid 'code' in state - cannot evaluate Definition of Done",
        )

    # Extract story from state
    story = state.get("story", {})
    if not isinstance(story, dict):
        story = {}

    # Get threshold from config using threshold resolver (0.0-1.0 format)
    threshold_decimal = resolve_threshold(
        gate_name="definition_of_done",
        state=state,
        default=DEFAULT_DOD_THRESHOLD,
    )
    # Convert to 0-100 scale for internal score comparison
    threshold = int(threshold_decimal * 100)

    logger.info(
        "definition_of_done_evaluation_started",
        gate_name=context.gate_name,
        threshold=threshold,
    )

    # Run all checks
    all_issues: list[GateIssue] = []

    # Test presence (AC1)
    all_issues.extend(check_test_presence(code, story))

    # Documentation (AC2)
    all_issues.extend(check_documentation(code))

    # Code style (AC3)
    all_issues.extend(check_code_style(code))

    # AC coverage (AC4)
    all_issues.extend(check_ac_coverage(code, story))

    # Generate checklist (AC5)
    checklist = generate_dod_checklist(all_issues)
    score = checklist["score"]

    # Determine pass/fail
    passed = score >= threshold

    # Generate report using shared utilities
    score_decimal = score / 100.0
    threshold_decimal_report = threshold / 100.0
    report = generate_failure_report(
        gate_name=context.gate_name,
        issues=all_issues,
        score=score_decimal,
        threshold=threshold_decimal_report,
    )
    report_text = format_report_text(report)

    logger.info(
        "definition_of_done_evaluation_complete",
        gate_name=context.gate_name,
        score=score,
        threshold=threshold,
        passed=passed,
        issue_count=len(all_issues),
    )

    return GateResult(
        passed=passed,
        gate_name=context.gate_name,
        reason=f"DoD compliance score: {score}/100 - All checks passed."
        if passed
        else f"DoD compliance score: {score}/100 (threshold: {threshold}).\n{report_text}",
    )


# =============================================================================
# Task 9: Register Evaluator
# =============================================================================


# Register on module import
register_evaluator("definition_of_done", definition_of_done_evaluator)
