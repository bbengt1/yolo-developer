"""Unit tests for test execution simulation (Story 9.3).

Tests for the execute_tests function and failure detection heuristics.
"""

from __future__ import annotations

from yolo_developer.agents.tea.execution import (
    ExecutionStatus,
    TestExecutionResult,
    detect_test_issues,
    execute_tests,
)


class TestDetectTestIssues:
    """Tests for detect_test_issues function."""

    def test_detect_missing_assertions(self) -> None:
        """Test detection of tests without assertions."""
        content = '''
def test_no_assertion():
    x = 1 + 1
    y = x * 2
    # No assert statement

def test_with_assertion():
    assert 1 == 1
'''
        failures = detect_test_issues(content, "test.py")
        # Should detect test_no_assertion as having no assertion
        assert any(f.test_name == "test_no_assertion" for f in failures)
        assert any(f.failure_type == "no_assertion" for f in failures)
        # test_with_assertion should not be flagged
        assert not any(f.test_name == "test_with_assertion" for f in failures)

    def test_detect_incomplete_tests(self) -> None:
        """Test detection of incomplete tests with TODO markers."""
        content = '''
def test_incomplete():
    # TODO: implement this test
    pass

def test_fixme():
    # FIXME: fix assertion
    assert True

def test_complete():
    result = calculate()
    assert result == 42
'''
        failures = detect_test_issues(content, "test.py")
        # Should detect tests with TODO/FIXME
        assert any(f.test_name == "test_incomplete" and f.failure_type == "incomplete" for f in failures)
        assert any(f.test_name == "test_fixme" and f.failure_type == "incomplete" for f in failures)

    def test_detect_pass_only_tests(self) -> None:
        """Test detection of tests that only have pass statement."""
        content = '''
def test_stub():
    pass

def test_real():
    assert True
'''
        failures = detect_test_issues(content, "test.py")
        # test_stub should be detected as incomplete
        assert any(f.test_name == "test_stub" for f in failures)

    def test_detect_pytest_raises_as_assertion(self) -> None:
        """Test that pytest.raises is counted as an assertion."""
        content = '''
import pytest

def test_with_raises():
    with pytest.raises(ValueError):
        raise ValueError("error")

def test_no_assertion_no_raises():
    x = 1
'''
        failures = detect_test_issues(content, "test.py")
        # test_with_raises should NOT be flagged (pytest.raises counts as assertion)
        assert not any(f.test_name == "test_with_raises" for f in failures)
        # test_no_assertion_no_raises should be flagged
        assert any(f.test_name == "test_no_assertion_no_raises" for f in failures)

    def test_empty_content(self) -> None:
        """Test with empty content."""
        failures = detect_test_issues("", "test.py")
        assert failures == []

    def test_no_tests_in_content(self) -> None:
        """Test with no test functions."""
        content = '''
def helper():
    return 42

class UtilityClass:
    pass
'''
        failures = detect_test_issues(content, "test.py")
        assert failures == []


class TestExecuteTests:
    """Tests for execute_tests function."""

    def test_execute_passing_tests(self) -> None:
        """Test execution with all passing tests."""
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''
def test_one():
    assert 1 == 1

def test_two():
    assert True
''',
            }
        ]
        result = execute_tests(test_files)

        assert result.status == "passed"
        assert result.passed_count == 2
        assert result.failed_count == 0
        assert result.error_count == 0
        assert result.failures == ()
        assert result.duration_ms >= 0

    def test_execute_with_failures(self) -> None:
        """Test execution with test failures."""
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''
def test_pass():
    assert True

def test_no_assertion():
    x = 1  # Missing assertion

def test_incomplete():
    # TODO: implement
    pass
''',
            }
        ]
        result = execute_tests(test_files)

        assert result.status == "failed"
        assert result.passed_count >= 1  # At least test_pass
        assert result.failed_count >= 1  # At least one failure
        assert len(result.failures) >= 1

    def test_execute_empty_test_files(self) -> None:
        """Test execution with empty test file list."""
        result = execute_tests([])

        assert result.status == "passed"
        assert result.passed_count == 0
        assert result.failed_count == 0
        assert result.error_count == 0

    def test_execute_no_tests_in_files(self) -> None:
        """Test execution with files containing no tests."""
        test_files = [
            {
                "artifact_id": "tests/conftest.py",
                "content": '''
import pytest

@pytest.fixture
def sample_fixture():
    return 42
''',
            }
        ]
        result = execute_tests(test_files)

        # No tests found = passed (vacuously true)
        assert result.status == "passed"
        assert result.passed_count == 0
        assert result.failed_count == 0

    def test_execute_multiple_files(self) -> None:
        """Test execution across multiple test files."""
        test_files = [
            {
                "artifact_id": "tests/test_a.py",
                "content": '''
def test_a1():
    assert True

def test_a2():
    assert True
''',
            },
            {
                "artifact_id": "tests/test_b.py",
                "content": '''
def test_b1():
    assert True

def test_b2_incomplete():
    # TODO: implement
    pass
''',
            },
        ]
        result = execute_tests(test_files)

        # 3 passing tests, 1 incomplete
        assert result.passed_count >= 3
        assert result.failed_count >= 1 or len(result.failures) >= 1

    def test_execute_duration_tracking(self) -> None:
        """Test that duration is tracked."""
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''
def test_one():
    assert True
''',
            }
        ]
        result = execute_tests(test_files)

        assert result.duration_ms >= 0
        assert result.start_time is not None
        assert result.end_time is not None
        # end_time should be >= start_time
        assert result.end_time >= result.start_time

    def test_execute_status_determination_passed(self) -> None:
        """Test status is 'passed' when all tests pass."""
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''
def test_good():
    assert True
''',
            }
        ]
        result = execute_tests(test_files)
        assert result.status == "passed"

    def test_execute_status_determination_failed(self) -> None:
        """Test status is 'failed' when tests fail."""
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''
def test_missing_assert():
    x = 1
''',
            }
        ]
        result = execute_tests(test_files)
        assert result.status == "failed"

    def test_execute_file_path_in_failures(self) -> None:
        """Test that file path is included in failures."""
        test_files = [
            {
                "artifact_id": "tests/specific/test_module.py",
                "content": '''
def test_no_assert():
    pass
''',
            }
        ]
        result = execute_tests(test_files)

        if result.failures:
            assert any(f.file_path == "tests/specific/test_module.py" for f in result.failures)
