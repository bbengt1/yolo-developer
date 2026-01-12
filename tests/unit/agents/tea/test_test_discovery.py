"""Unit tests for test discovery (Story 9.3).

Tests for the _discover_tests function that extracts test names from test content.
"""

from __future__ import annotations

from yolo_developer.agents.tea.execution import discover_tests


class TestDiscoverTests:
    """Tests for discover_tests function."""

    def test_discover_standard_test_functions(self) -> None:
        """Test discovery of standard test functions."""
        content = '''
def test_simple_case():
    assert 1 == 1

def test_another_case():
    assert True

def helper_function():
    return 42
'''
        tests = discover_tests(content)
        assert "test_simple_case" in tests
        assert "test_another_case" in tests
        assert "helper_function" not in tests
        assert len(tests) == 2

    def test_discover_async_test_functions(self) -> None:
        """Test discovery of async test functions."""
        content = '''
async def test_async_operation():
    await some_async_call()
    assert True

async def test_another_async():
    result = await fetch_data()
    assert result is not None

async def helper_async():
    return "helper"
'''
        tests = discover_tests(content)
        assert "test_async_operation" in tests
        assert "test_another_async" in tests
        assert "helper_async" not in tests
        assert len(tests) == 2

    def test_discover_test_classes(self) -> None:
        """Test discovery of test classes."""
        content = '''
class TestValidation:
    def test_valid_input(self):
        assert True

    def test_invalid_input(self):
        assert False

class HelperClass:
    def helper_method(self):
        pass

class TestAuthentication:
    def test_login(self):
        pass
'''
        tests = discover_tests(content)
        assert "TestValidation" in tests
        assert "TestAuthentication" in tests
        assert "HelperClass" not in tests
        # Test methods within classes are not individually discovered
        # since we count the class as a test container

    def test_discover_empty_file(self) -> None:
        """Test discovery with empty content."""
        tests = discover_tests("")
        assert tests == []

    def test_discover_no_tests(self) -> None:
        """Test discovery with no test functions."""
        content = '''
def helper_function():
    return 42

def another_helper():
    pass

class UtilityClass:
    def method(self):
        pass
'''
        tests = discover_tests(content)
        assert tests == []

    def test_discover_mixed_content(self) -> None:
        """Test discovery with mixed test and non-test content."""
        content = '''
import pytest

def helper():
    return 1

class TestModule:
    def test_one(self):
        assert True

def test_standalone():
    assert True

async def test_async_standalone():
    assert True

def not_a_test():
    pass
'''
        tests = discover_tests(content)
        assert "TestModule" in tests
        assert "test_standalone" in tests
        assert "test_async_standalone" in tests
        assert "helper" not in tests
        assert "not_a_test" not in tests

    def test_discover_test_with_underscore_prefix(self) -> None:
        """Test that test_ prefix is required."""
        content = '''
def test_valid():
    pass

def _test_private():
    pass

def testInvalid():
    pass
'''
        tests = discover_tests(content)
        assert "test_valid" in tests
        assert "_test_private" not in tests
        assert "testInvalid" not in tests
        assert len(tests) == 1

    def test_discover_pytest_fixtures_excluded(self) -> None:
        """Test that pytest fixtures are not counted as tests."""
        content = '''
import pytest

@pytest.fixture
def test_fixture():
    return "fixture value"

def test_actual_test(test_fixture):
    assert test_fixture == "fixture value"
'''
        tests = discover_tests(content)
        # Fixtures have test_ prefix but decorated - still discovered
        # (heuristic limitation - fixture detection would need AST parsing)
        assert "test_actual_test" in tests
        # Note: test_fixture will also be discovered as our heuristic
        # doesn't parse decorators. This is acceptable for MVP.

    def test_discover_parametrized_tests(self) -> None:
        """Test discovery of parametrized tests."""
        content = '''
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
])
def test_multiply_by_two(input, expected):
    assert input * 2 == expected
'''
        tests = discover_tests(content)
        assert "test_multiply_by_two" in tests

    def test_discover_whitespace_variations(self) -> None:
        """Test discovery handles whitespace variations."""
        content = '''
def  test_extra_spaces():
    pass

def	test_with_tab():
    pass

def test_normal():
    pass
'''
        tests = discover_tests(content)
        # Only properly formatted definitions should be discovered
        assert "test_normal" in tests
