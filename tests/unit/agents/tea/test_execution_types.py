"""Unit tests for TEA agent test execution types (Story 9.3).

Tests for the type definitions used for test execution:
- TestFailure dataclass
- TestExecutionResult dataclass
- FailureType and ExecutionStatus literal types
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from yolo_developer.agents.tea.execution import (
    ExecutionStatus,
    FailureType,
    TestExecutionResult,
    TestFailure,
)


class TestFailureTypeLiteral:
    """Tests for FailureType literal type."""

    def test_valid_failure_type_values(self) -> None:
        """Test that FailureType has expected values."""
        expected = {"error", "failure", "no_assertion", "incomplete"}
        actual = set(get_args(FailureType))
        assert actual == expected

    def test_failure_type_count(self) -> None:
        """Test that FailureType has exactly 4 values."""
        assert len(get_args(FailureType)) == 4


class TestExecutionStatusLiteral:
    """Tests for ExecutionStatus literal type."""

    def test_valid_execution_status_values(self) -> None:
        """Test that ExecutionStatus has expected values."""
        expected = {"passed", "failed", "error"}
        actual = set(get_args(ExecutionStatus))
        assert actual == expected

    def test_execution_status_count(self) -> None:
        """Test that ExecutionStatus has exactly 3 values."""
        assert len(get_args(ExecutionStatus)) == 3


class TestTestFailure:
    """Tests for TestFailure frozen dataclass."""

    def test_test_failure_creation(self) -> None:
        """Test TestFailure can be created with required fields."""
        failure = TestFailure(
            test_name="test_invalid_input",
            file_path="tests/test_validation.py",
            error_message="Missing assertion - test does not verify anything",
            failure_type="no_assertion",
        )
        assert failure.test_name == "test_invalid_input"
        assert failure.file_path == "tests/test_validation.py"
        assert failure.error_message == "Missing assertion - test does not verify anything"
        assert failure.failure_type == "no_assertion"
        assert failure.created_at is not None

    def test_test_failure_to_dict(self) -> None:
        """Test TestFailure.to_dict() returns correct dictionary."""
        failure = TestFailure(
            test_name="test_edge_case",
            file_path="tests/test_module.py",
            error_message="Test marked as TODO",
            failure_type="incomplete",
        )
        result = failure.to_dict()

        assert isinstance(result, dict)
        assert result["test_name"] == "test_edge_case"
        assert result["file_path"] == "tests/test_module.py"
        assert result["error_message"] == "Test marked as TODO"
        assert result["failure_type"] == "incomplete"
        assert "created_at" in result

    def test_test_failure_is_frozen(self) -> None:
        """Test TestFailure is immutable."""
        failure = TestFailure(
            test_name="test_something",
            file_path="test.py",
            error_message="Error",
            failure_type="error",
        )
        with pytest.raises(FrozenInstanceError):
            failure.test_name = "other_test"  # type: ignore[misc]

    def test_test_failure_all_failure_types(self) -> None:
        """Test TestFailure accepts all failure_type values."""
        failure_types = get_args(FailureType)
        for failure_type in failure_types:
            failure = TestFailure(
                test_name=f"test_{failure_type}",
                file_path="test.py",
                error_message=f"Test {failure_type}",
                failure_type=failure_type,
            )
            assert failure.failure_type == failure_type


class TestTestExecutionResult:
    """Tests for TestExecutionResult frozen dataclass."""

    def test_test_execution_result_creation_passed(self) -> None:
        """Test TestExecutionResult can be created for passing tests."""
        result = TestExecutionResult(
            status="passed",
            passed_count=10,
            failed_count=0,
            error_count=0,
            failures=(),
            duration_ms=150,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.150Z",
        )
        assert result.status == "passed"
        assert result.passed_count == 10
        assert result.failed_count == 0
        assert result.error_count == 0
        assert result.failures == ()
        assert result.duration_ms == 150
        assert result.start_time == "2026-01-12T10:00:00.000Z"
        assert result.end_time == "2026-01-12T10:00:00.150Z"

    def test_test_execution_result_with_failures(self) -> None:
        """Test TestExecutionResult with test failures."""
        failure = TestFailure(
            test_name="test_invalid_input",
            file_path="tests/test_validation.py",
            error_message="Assertion failed",
            failure_type="failure",
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=8,
            failed_count=2,
            error_count=0,
            failures=(failure,),
            duration_ms=200,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.200Z",
        )
        assert result.status == "failed"
        assert result.passed_count == 8
        assert result.failed_count == 2
        assert len(result.failures) == 1
        assert result.failures[0].test_name == "test_invalid_input"

    def test_test_execution_result_with_errors(self) -> None:
        """Test TestExecutionResult with execution errors."""
        failure = TestFailure(
            test_name="test_syntax_error",
            file_path="tests/test_broken.py",
            error_message="SyntaxError: invalid syntax",
            failure_type="error",
        )
        result = TestExecutionResult(
            status="error",
            passed_count=0,
            failed_count=0,
            error_count=1,
            failures=(failure,),
            duration_ms=5,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.005Z",
        )
        assert result.status == "error"
        assert result.error_count == 1
        assert result.failures[0].failure_type == "error"

    def test_test_execution_result_to_dict(self) -> None:
        """Test TestExecutionResult.to_dict() returns correct dictionary."""
        failure = TestFailure(
            test_name="test_case",
            file_path="test.py",
            error_message="Failed",
            failure_type="failure",
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=5,
            failed_count=1,
            error_count=0,
            failures=(failure,),
            duration_ms=100,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.100Z",
        )
        dict_result = result.to_dict()

        assert isinstance(dict_result, dict)
        assert dict_result["status"] == "failed"
        assert dict_result["passed_count"] == 5
        assert dict_result["failed_count"] == 1
        assert dict_result["error_count"] == 0
        assert len(dict_result["failures"]) == 1
        assert dict_result["failures"][0]["test_name"] == "test_case"
        assert dict_result["duration_ms"] == 100
        assert dict_result["start_time"] == "2026-01-12T10:00:00.000Z"
        assert dict_result["end_time"] == "2026-01-12T10:00:00.100Z"

    def test_test_execution_result_is_frozen(self) -> None:
        """Test TestExecutionResult is immutable."""
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
        with pytest.raises(FrozenInstanceError):
            result.passed_count = 5  # type: ignore[misc]

    def test_test_execution_result_all_statuses(self) -> None:
        """Test TestExecutionResult accepts all status values."""
        statuses = get_args(ExecutionStatus)
        for status in statuses:
            result = TestExecutionResult(
                status=status,
                passed_count=0,
                failed_count=0,
                error_count=0,
                failures=(),
                duration_ms=0,
                start_time="2026-01-12T10:00:00.000Z",
                end_time="2026-01-12T10:00:00.000Z",
            )
            assert result.status == status

    def test_test_execution_result_zero_tests(self) -> None:
        """Test TestExecutionResult with zero tests (edge case)."""
        result = TestExecutionResult(
            status="passed",
            passed_count=0,
            failed_count=0,
            error_count=0,
            failures=(),
            duration_ms=0,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.000Z",
        )
        assert result.passed_count == 0
        assert result.failed_count == 0
        assert result.status == "passed"  # Vacuously true

    def test_test_execution_result_multiple_failures(self) -> None:
        """Test TestExecutionResult with multiple failures."""
        failures = tuple(
            TestFailure(
                test_name=f"test_case_{i}",
                file_path=f"tests/test_file{i}.py",
                error_message=f"Error {i}",
                failure_type="failure",
            )
            for i in range(3)
        )
        result = TestExecutionResult(
            status="failed",
            passed_count=7,
            failed_count=3,
            error_count=0,
            failures=failures,
            duration_ms=300,
            start_time="2026-01-12T10:00:00.000Z",
            end_time="2026-01-12T10:00:00.300Z",
        )
        assert len(result.failures) == 3
        dict_result = result.to_dict()
        assert len(dict_result["failures"]) == 3
