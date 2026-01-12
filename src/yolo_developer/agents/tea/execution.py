"""Test execution types and functions (Story 9.3).

This module provides the data types and functions used for test execution:

- FailureType: Literal type for test failure types
- ExecutionStatus: Literal type for test execution status
- TestFailure: Individual test failure details
- TestExecutionResult: Aggregate test execution results
- discover_tests: Extract test names from test file content
- execute_tests: Execute tests and return results

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.tea.execution import (
    ...     TestFailure,
    ...     TestExecutionResult,
    ...     discover_tests,
    ... )
    >>>
    >>> tests = discover_tests("def test_example(): assert True")
    >>> tests
    ['test_example']
    >>>
    >>> failure = TestFailure(
    ...     test_name="test_invalid_input",
    ...     file_path="tests/test_validation.py",
    ...     error_message="Missing assertion",
    ...     failure_type="no_assertion",
    ... )
    >>> failure.to_dict()
    {'test_name': 'test_invalid_input', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

import re
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

FailureType = Literal[
    "error",
    "failure",
    "no_assertion",
    "incomplete",
]
"""Type of test failure.

Values:
    error: Test couldn't run (syntax error, import failure)
    failure: Test ran but assertion failed
    no_assertion: Test exists but doesn't verify anything
    incomplete: Test marked as incomplete (TODO/FIXME)
"""

ExecutionStatus = Literal[
    "passed",
    "failed",
    "error",
]
"""Overall status of test execution.

Values:
    passed: All tests passed (or no tests found - vacuously true)
    failed: One or more tests failed but no errors
    error: Execution errors occurred (blocking)
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TestFailure:
    """Individual test failure with details.

    Represents a single test that failed or had issues during execution.

    Attributes:
        test_name: Name of the failing test function/method
        file_path: Path to the test file
        error_message: Human-readable description of the failure
        failure_type: Classification of the failure type
        created_at: ISO timestamp when failure was recorded

    Example:
        >>> failure = TestFailure(
        ...     test_name="test_invalid_input",
        ...     file_path="tests/test_validation.py",
        ...     error_message="Missing assertion - test does not verify anything",
        ...     failure_type="no_assertion",
        ... )
        >>> failure.to_dict()
        {'test_name': 'test_invalid_input', ...}
    """

    test_name: str
    file_path: str
    error_message: str
    failure_type: FailureType
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the test failure.
        """
        return {
            "test_name": self.test_name,
            "file_path": self.file_path,
            "error_message": self.error_message,
            "failure_type": self.failure_type,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TestExecutionResult:
    """Aggregate test execution results.

    Contains results from executing a test suite, including pass/fail counts,
    failures, and timing information.

    Attributes:
        status: Overall execution status (passed, failed, error)
        passed_count: Number of tests that passed
        failed_count: Number of tests that failed
        error_count: Number of execution errors
        failures: Tuple of TestFailure objects for each failure/error
        duration_ms: Total execution duration in milliseconds
        start_time: ISO timestamp when execution started
        end_time: ISO timestamp when execution ended

    Example:
        >>> result = TestExecutionResult(
        ...     status="failed",
        ...     passed_count=8,
        ...     failed_count=2,
        ...     error_count=0,
        ...     failures=(failure1, failure2),
        ...     duration_ms=150,
        ...     start_time="2026-01-12T10:00:00.000Z",
        ...     end_time="2026-01-12T10:00:00.150Z",
        ... )
        >>> result.to_dict()
        {'status': 'failed', 'passed_count': 8, ...}
    """

    status: ExecutionStatus
    passed_count: int
    failed_count: int
    error_count: int
    failures: tuple[TestFailure, ...] = field(default_factory=tuple)
    duration_ms: int = 0
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested failures.
        """
        return {
            "status": self.status,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "failures": [f.to_dict() for f in self.failures],
            "duration_ms": self.duration_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# =============================================================================
# Test Discovery Functions
# =============================================================================


def discover_tests(content: str) -> list[str]:
    """Discover test names from test file content.

    Extracts test function names and test class names from Python test content.
    Uses regex-based heuristics for MVP (AST parsing would be more accurate).

    Patterns discovered:
    - def test_*(...): - Standard test functions
    - async def test_*(...): - Async test functions
    - class Test*: - Test classes (pytest convention)

    Args:
        content: Python test file content as string.

    Returns:
        List of discovered test names. Empty list if no tests found.

    Example:
        >>> content = '''
        ... def test_example():
        ...     assert True
        ...
        ... class TestModule:
        ...     def test_method(self):
        ...         pass
        ... '''
        >>> discover_tests(content)
        ['test_example', 'TestModule']
    """
    if not content.strip():
        logger.debug("discover_tests_empty_content")
        return []

    tests: list[str] = []

    # Pattern for test functions: def test_* or async def test_*
    # Requires proper formatting (single space after def)
    test_func_pattern = r"(?:async\s+)?def\s+(test_[a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    func_matches = re.findall(test_func_pattern, content)
    tests.extend(func_matches)

    # Pattern for test classes: class Test*
    test_class_pattern = r"class\s+(Test[a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]"
    class_matches = re.findall(test_class_pattern, content)
    tests.extend(class_matches)

    logger.debug(
        "discover_tests_complete",
        test_count=len(tests),
        function_count=len(func_matches),
        class_count=len(class_matches),
    )

    return tests


def _extract_test_body(content: str, test_name: str) -> str:
    """Extract the body of a specific test function.

    Args:
        content: Full test file content.
        test_name: Name of the test function to extract.

    Returns:
        The test function body as a string, or empty string if not found.
    """
    lines = content.splitlines()
    in_function = False
    func_indent = 0
    body_lines: list[str] = []

    for line in lines:
        if not in_function:
            # Look for the function definition
            match = re.match(rf"^(\s*)(?:async\s+)?def\s+{re.escape(test_name)}\s*\(", line)
            if match:
                in_function = True
                func_indent = len(match.group(1))
        else:
            # Check if we're still in the function
            stripped = line.strip()

            # Empty lines are part of the function
            if not stripped:
                body_lines.append(line)
                continue

            # Calculate current indent
            current_indent = len(line) - len(line.lstrip())

            # If we're back at or before function indent level with content,
            # we've left the function
            if current_indent <= func_indent and stripped:
                break

            body_lines.append(line)

    return "\n".join(body_lines)


def detect_test_issues(content: str, file_path: str) -> list[TestFailure]:
    """Detect issues in test file content.

    Analyzes test content for common issues that would cause test failures:
    - Missing assertions (test does nothing)
    - TODO/FIXME markers indicating incomplete tests
    - Pass-only tests (stubs)

    Args:
        content: Test file content.
        file_path: Path to the test file (for error reporting).

    Returns:
        List of TestFailure objects for each detected issue.

    Example:
        >>> content = '''
        ... def test_no_assert():
        ...     x = 1
        ... '''
        >>> failures = detect_test_issues(content, "test.py")
        >>> len(failures)
        1
        >>> failures[0].failure_type
        'no_assertion'
    """
    if not content.strip():
        return []

    failures: list[TestFailure] = []
    tests = discover_tests(content)

    for test_name in tests:
        # Skip test classes - we analyze methods within them
        if test_name.startswith("Test") and not test_name.startswith("test_"):
            continue

        test_body = _extract_test_body(content, test_name)

        # Check for TODO/FIXME markers
        if re.search(r"\b(TODO|FIXME)\b", test_body, re.IGNORECASE):
            failures.append(
                TestFailure(
                    test_name=test_name,
                    file_path=file_path,
                    error_message="Test marked as incomplete (contains TODO/FIXME)",
                    failure_type="incomplete",
                )
            )
            logger.debug(
                "test_issue_detected",
                test_name=test_name,
                issue="incomplete",
                file_path=file_path,
            )
            continue

        # Check for pass-only tests (stub tests)
        body_stripped = test_body.strip()
        if body_stripped == "pass" or (
            body_stripped.startswith("pass") and not body_stripped.replace("pass", "").strip()
        ):
            failures.append(
                TestFailure(
                    test_name=test_name,
                    file_path=file_path,
                    error_message="Test is a stub (only contains pass)",
                    failure_type="incomplete",
                )
            )
            logger.debug(
                "test_issue_detected",
                test_name=test_name,
                issue="stub",
                file_path=file_path,
            )
            continue

        # Check for missing assertions - exclude comments
        # Remove comments from test body for assertion checking
        body_no_comments = "\n".join(
            line for line in test_body.splitlines() if not line.strip().startswith("#")
        )
        has_assertion = (
            "assert " in body_no_comments
            or "assert(" in body_no_comments
            or "pytest.raises" in body_no_comments
            or ".assert" in body_no_comments  # unittest style assertions
        )

        if not has_assertion:
            failures.append(
                TestFailure(
                    test_name=test_name,
                    file_path=file_path,
                    error_message="Test has no assertions - does not verify anything",
                    failure_type="no_assertion",
                )
            )
            logger.debug(
                "test_issue_detected",
                test_name=test_name,
                issue="no_assertion",
                file_path=file_path,
            )

    logger.info(
        "detect_test_issues_complete",
        file_path=file_path,
        test_count=len(tests),
        issue_count=len(failures),
    )

    return failures


def execute_tests(test_files: list[dict[str, Any]]) -> TestExecutionResult:
    """Execute tests and return results.

    For MVP, this is a heuristic-based simulation that analyzes test content
    rather than actually running pytest. It discovers tests and detects
    common issues that would cause failures.

    Args:
        test_files: List of test file dicts with 'artifact_id' and 'content' keys.

    Returns:
        TestExecutionResult with execution status and any failures.

    Example:
        >>> test_files = [{"artifact_id": "test.py", "content": "def test_x(): assert True"}]
        >>> result = execute_tests(test_files)
        >>> result.status
        'passed'
    """
    import time

    start_time = datetime.now(timezone.utc)
    start_mono = time.monotonic()

    if not test_files:
        logger.debug("execute_tests_no_files")
        end_time = datetime.now(timezone.utc)
        return TestExecutionResult(
            status="passed",
            passed_count=0,
            failed_count=0,
            error_count=0,
            failures=(),
            duration_ms=0,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

    all_tests: list[str] = []
    all_failures: list[TestFailure] = []

    for test_file in test_files:
        file_path = test_file.get("artifact_id", "unknown")
        content = test_file.get("content", "")

        # Discover tests in this file
        tests = discover_tests(content)
        all_tests.extend(tests)

        # Detect issues in this file
        failures = detect_test_issues(content, file_path)
        all_failures.extend(failures)

    end_mono = time.monotonic()
    end_time = datetime.now(timezone.utc)
    duration_ms = int((end_mono - start_mono) * 1000)

    # Calculate counts
    # Tests with issues are considered failed
    failed_test_names = {f.test_name for f in all_failures if f.failure_type != "error"}
    error_test_names = {f.test_name for f in all_failures if f.failure_type == "error"}

    # Filter out test class names from count (only count functions)
    test_functions = [t for t in all_tests if t.startswith("test_")]

    failed_count = len(failed_test_names)
    error_count = len(error_test_names)
    passed_count = len(test_functions) - failed_count - error_count
    passed_count = max(0, passed_count)  # Ensure non-negative

    # Determine status
    status: ExecutionStatus
    if error_count > 0:
        status = "error"
    elif failed_count > 0:
        status = "failed"
    else:
        status = "passed"

    logger.info(
        "execute_tests_complete",
        total_tests=len(test_functions),
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=error_count,
        status=status,
        duration_ms=duration_ms,
    )

    return TestExecutionResult(
        status=status,
        passed_count=passed_count,
        failed_count=failed_count,
        error_count=error_count,
        failures=tuple(all_failures),
        duration_ms=duration_ms,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
    )


# =============================================================================
# Finding Generation
# =============================================================================


def generate_test_findings(result: TestExecutionResult) -> list[Finding]:
    """Generate Finding objects from test execution result.

    Converts each TestFailure in the result to a Finding with appropriate
    severity and remediation guidance.

    Severity mapping:
    - error -> critical (test couldn't run)
    - failure -> high (test ran but assertion failed)
    - no_assertion -> medium (test exists but doesn't verify anything)
    - incomplete -> low (test marked as incomplete)

    Args:
        result: TestExecutionResult with failures to convert.

    Returns:
        List of Finding objects for each test failure.

    Example:
        >>> result = TestExecutionResult(...)
        >>> findings = generate_test_findings(result)
        >>> len(findings)
        2
    """
    # Import here to avoid circular imports
    from yolo_developer.agents.tea.types import Finding, FindingSeverity

    if not result.failures:
        logger.debug("generate_test_findings_no_failures")
        return []

    findings: list[Finding] = []
    finding_counter = 1

    # Severity mapping based on failure type
    severity_map: dict[str, FindingSeverity] = {
        "error": "critical",
        "failure": "high",
        "no_assertion": "medium",
        "incomplete": "low",
    }

    # Remediation templates based on failure type
    remediation_map = {
        "error": "Fix the error preventing the test from running. Check for syntax errors, import issues, or missing dependencies.",
        "failure": "Review and fix the assertion that is failing. Check expected vs actual values.",
        "no_assertion": "Add meaningful assertions to verify the test's expected behavior.",
        "incomplete": "Complete the test implementation and remove TODO/FIXME markers.",
    }

    for failure in result.failures:
        severity = severity_map.get(failure.failure_type, "medium")
        base_remediation = remediation_map.get(failure.failure_type, "Review and fix the test.")

        # Create unique finding ID using hash of test name
        test_hash = hash(f"{failure.file_path}:{failure.test_name}") & 0xFFFFFFFF
        finding_id = f"TEST-{finding_counter:03d}-{test_hash:08x}"

        finding = Finding(
            finding_id=finding_id,
            category="test_coverage",
            severity=severity,
            description=f"Test '{failure.test_name}' in {failure.file_path}: {failure.error_message}",
            location=failure.file_path,
            remediation=f"{base_remediation} Error: {failure.error_message}",
        )
        findings.append(finding)
        finding_counter += 1

        logger.debug(
            "test_finding_generated",
            finding_id=finding_id,
            test_name=failure.test_name,
            severity=severity,
            failure_type=failure.failure_type,
        )

    logger.info(
        "generate_test_findings_complete",
        finding_count=len(findings),
        critical_count=sum(1 for f in findings if f.severity == "critical"),
        high_count=sum(1 for f in findings if f.severity == "high"),
    )

    return findings
