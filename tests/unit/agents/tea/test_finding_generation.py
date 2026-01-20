"""Unit tests for test finding generation (Story 9.3).

Tests for the generate_test_findings function that converts test failures to Findings.
"""

from __future__ import annotations

from yolo_developer.agents.tea.execution import (
    TestExecutionResult,
    TestFailure,
    generate_test_findings,
)


class TestGenerateTestFindings:
    """Tests for generate_test_findings function."""

    def test_generate_findings_from_failures(self) -> None:
        """Test generating findings from test failures."""
        failures = (
            TestFailure(
                test_name="test_broken",
                file_path="tests/test_module.py",
                error_message="Test has no assertions",
                failure_type="no_assertion",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].category == "test_coverage"
        assert findings[0].location == "tests/test_module.py"
        assert "test_broken" in findings[0].description

    def test_error_failure_type_is_critical(self) -> None:
        """Test that error failure type maps to critical severity."""
        failures = (
            TestFailure(
                test_name="test_syntax_error",
                file_path="tests/test_broken.py",
                error_message="SyntaxError: invalid syntax",
                failure_type="error",
            ),
        )
        result = TestExecutionResult(
            status="error",
            passed_count=0,
            failed_count=0,
            error_count=1,
            failures=failures,
            duration_ms=5,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.005Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_failure_type_is_high(self) -> None:
        """Test that regular failure type maps to high severity."""
        failures = (
            TestFailure(
                test_name="test_assertion_failed",
                file_path="tests/test_module.py",
                error_message="AssertionError: 1 != 2",
                failure_type="failure",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_no_assertion_is_medium(self) -> None:
        """Test that no_assertion failure type maps to medium severity."""
        failures = (
            TestFailure(
                test_name="test_no_assert",
                file_path="tests/test_module.py",
                error_message="Test has no assertions",
                failure_type="no_assertion",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_incomplete_is_low(self) -> None:
        """Test that incomplete failure type maps to low severity."""
        failures = (
            TestFailure(
                test_name="test_todo",
                file_path="tests/test_module.py",
                error_message="Test marked as TODO",
                failure_type="incomplete",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].severity == "low"

    def test_no_failures_no_findings(self) -> None:
        """Test that no failures produces no findings."""
        result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
            failures=(),
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert findings == []

    def test_multiple_failures_multiple_findings(self) -> None:
        """Test that multiple failures produce multiple findings."""
        failures = (
            TestFailure(
                test_name="test_error",
                file_path="tests/test_a.py",
                error_message="Error 1",
                failure_type="error",
            ),
            TestFailure(
                test_name="test_fail",
                file_path="tests/test_b.py",
                error_message="Error 2",
                failure_type="failure",
            ),
            TestFailure(
                test_name="test_todo",
                file_path="tests/test_c.py",
                error_message="Error 3",
                failure_type="incomplete",
            ),
        )
        result = TestExecutionResult(
            status="error",
            passed_count=5,
            failed_count=2,
            error_count=1,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 3
        # Check severities
        severities = {f.severity for f in findings}
        assert "critical" in severities  # error
        assert "high" in severities  # failure
        assert "low" in severities  # incomplete

    def test_finding_has_remediation(self) -> None:
        """Test that findings include remediation guidance."""
        failures = (
            TestFailure(
                test_name="test_broken",
                file_path="tests/test_module.py",
                error_message="Test has no assertions",
                failure_type="no_assertion",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        assert len(findings) == 1
        assert findings[0].remediation != ""
        # Should include the error message or test name
        assert (
            "test_broken" in findings[0].description
            or "assertion" in findings[0].remediation.lower()
        )

    def test_finding_id_is_unique(self) -> None:
        """Test that each finding has a unique ID."""
        failures = (
            TestFailure(
                test_name="test_a",
                file_path="tests/test_module.py",
                error_message="Error A",
                failure_type="failure",
            ),
            TestFailure(
                test_name="test_b",
                file_path="tests/test_module.py",
                error_message="Error B",
                failure_type="failure",
            ),
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=2,
            error_count=0,
            failures=failures,
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )

        findings = generate_test_findings(result)

        finding_ids = [f.finding_id for f in findings]
        assert len(finding_ids) == len(set(finding_ids))  # All unique
