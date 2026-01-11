"""Unit tests for test generation prompt templates (Story 8.3 - Task 8).

Tests prompt template rendering, testing best practices inclusion,
and pytest conventions.
"""

from __future__ import annotations

from yolo_developer.agents.dev.prompts.test_generation import (
    PYTEST_CONVENTIONS,
    TEST_GENERATION_TEMPLATE,
    TESTING_BEST_PRACTICES,
    build_test_generation_prompt,
    build_test_retry_prompt,
)


class TestTestGenerationTemplate:
    """Tests for the TEST_GENERATION_TEMPLATE constant."""

    def test_template_has_implementation_code_placeholder(self) -> None:
        """Test that template has implementation_code placeholder."""
        assert "{implementation_code}" in TEST_GENERATION_TEMPLATE

    def test_template_has_function_list_placeholder(self) -> None:
        """Test that template has function_list placeholder."""
        assert "{function_list}" in TEST_GENERATION_TEMPLATE

    def test_template_has_module_name_placeholder(self) -> None:
        """Test that template has module_name placeholder."""
        assert "{module_name}" in TEST_GENERATION_TEMPLATE

    def test_template_has_testing_best_practices_placeholder(self) -> None:
        """Test that template has testing_best_practices placeholder."""
        assert "{testing_best_practices}" in TEST_GENERATION_TEMPLATE

    def test_template_has_pytest_conventions_placeholder(self) -> None:
        """Test that template has pytest_conventions placeholder."""
        assert "{pytest_conventions}" in TEST_GENERATION_TEMPLATE

    def test_template_has_additional_context_placeholder(self) -> None:
        """Test that template has additional_context placeholder."""
        assert "{additional_context}" in TEST_GENERATION_TEMPLATE

    def test_template_mentions_pytest(self) -> None:
        """Test that template mentions pytest framework."""
        assert "pytest" in TEST_GENERATION_TEMPLATE.lower()


class TestTestingBestPractices:
    """Tests for TESTING_BEST_PRACTICES constant (AC1, AC2, AC3)."""

    def test_includes_edge_case_coverage(self) -> None:
        """Test that best practices include edge case coverage (AC2)."""
        assert "edge case" in TESTING_BEST_PRACTICES.lower()

    def test_includes_empty_inputs(self) -> None:
        """Test that empty inputs testing is mentioned (AC2)."""
        assert "empty" in TESTING_BEST_PRACTICES.lower()

    def test_includes_none_values(self) -> None:
        """Test that None value testing is mentioned (AC2)."""
        assert "None" in TESTING_BEST_PRACTICES

    def test_includes_boundary_values(self) -> None:
        """Test that boundary value testing is mentioned (AC2)."""
        assert "boundary" in TESTING_BEST_PRACTICES.lower()

    def test_includes_test_isolation(self) -> None:
        """Test that test isolation is mentioned (AC3)."""
        assert "isolat" in TESTING_BEST_PRACTICES.lower()

    def test_includes_determinism(self) -> None:
        """Test that deterministic tests are mentioned (AC3)."""
        assert "deterministic" in TESTING_BEST_PRACTICES.lower()

    def test_includes_fixtures(self) -> None:
        """Test that fixtures are mentioned (AC3)."""
        assert "fixture" in TESTING_BEST_PRACTICES.lower()

    def test_includes_mocking(self) -> None:
        """Test that mocking is mentioned (AC3)."""
        assert "mock" in TESTING_BEST_PRACTICES.lower()

    def test_includes_assertions(self) -> None:
        """Test that proper assertions are mentioned."""
        assert "assert" in TESTING_BEST_PRACTICES.lower()


class TestPytestConventions:
    """Tests for PYTEST_CONVENTIONS constant (AC1, AC6)."""

    def test_includes_test_function_naming(self) -> None:
        """Test that test function naming convention is included (AC1)."""
        assert "test_" in PYTEST_CONVENTIONS.lower()

    def test_includes_test_class_naming(self) -> None:
        """Test that test class naming convention is included."""
        assert "Test" in PYTEST_CONVENTIONS

    def test_includes_docstrings_requirement(self) -> None:
        """Test that docstrings requirement is included (AC1)."""
        assert "docstring" in PYTEST_CONVENTIONS.lower()

    def test_includes_pytest_raises(self) -> None:
        """Test that pytest.raises is mentioned for exception testing (AC2)."""
        assert "pytest.raises" in PYTEST_CONVENTIONS

    def test_includes_parametrize(self) -> None:
        """Test that pytest.mark.parametrize is mentioned."""
        assert "parametrize" in PYTEST_CONVENTIONS.lower()

    def test_includes_fixture_decorator(self) -> None:
        """Test that @pytest.fixture is mentioned."""
        assert "fixture" in PYTEST_CONVENTIONS.lower()


class TestBuildTestGenerationPrompt:
    """Tests for build_test_generation_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = build_test_generation_prompt(
            implementation_code="def hello(): pass",
            function_list=["hello"],
            module_name="test_module",
        )
        assert isinstance(result, str)

    def test_includes_implementation_code(self) -> None:
        """Test that prompt includes the implementation code."""
        result = build_test_generation_prompt(
            implementation_code="def calculate_sum(a, b): return a + b",
            function_list=["calculate_sum"],
            module_name="math_utils",
        )
        assert "def calculate_sum(a, b): return a + b" in result

    def test_includes_function_list(self) -> None:
        """Test that prompt includes the function list."""
        result = build_test_generation_prompt(
            implementation_code="def foo(): pass\ndef bar(): pass",
            function_list=["foo", "bar"],
            module_name="test_module",
        )
        assert "foo" in result
        assert "bar" in result

    def test_includes_module_name(self) -> None:
        """Test that prompt includes the module name."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="my_awesome_module",
        )
        assert "my_awesome_module" in result

    def test_includes_testing_best_practices_by_default(self) -> None:
        """Test that testing best practices are included by default."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
        )
        assert "edge case" in result.lower()
        assert "isolat" in result.lower()

    def test_excludes_testing_best_practices_when_disabled(self) -> None:
        """Test that testing best practices can be excluded."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
            include_best_practices=False,
        )
        # Specific testing best practices content should not be present
        assert "deterministic" not in result.lower() or "best practices" not in result.lower()

    def test_includes_pytest_conventions_by_default(self) -> None:
        """Test that pytest conventions are included by default."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
        )
        assert "pytest" in result.lower()

    def test_excludes_pytest_conventions_when_disabled(self) -> None:
        """Test that pytest conventions can be excluded."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
            include_conventions=False,
        )
        # Pytest conventions section markers should not be present
        assert "## Pytest Conventions" not in result
        assert "AAA Pattern" not in result

    def test_includes_additional_context(self) -> None:
        """Test that additional context is included when provided."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
            additional_context="This module handles authentication",
        )
        assert "This module handles authentication" in result

    def test_handles_empty_additional_context(self) -> None:
        """Test that empty additional context is handled."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
            additional_context="",
        )
        # Should not break and should produce valid prompt
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_empty_function_list(self) -> None:
        """Test that empty function list is handled gracefully."""
        result = build_test_generation_prompt(
            implementation_code="# Empty module",
            function_list=[],
            module_name="empty_module",
        )
        assert isinstance(result, str)


class TestBuildTestRetryPrompt:
    """Tests for build_test_retry_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = build_test_retry_prompt(
            original_prompt="Generate tests...",
            error_message="Line 5: invalid syntax",
            previous_tests="def test_something():\n  pass",
        )
        assert isinstance(result, str)

    def test_includes_original_prompt(self) -> None:
        """Test that retry prompt includes original prompt."""
        result = build_test_retry_prompt(
            original_prompt="Generate unit tests for authentication",
            error_message="syntax error",
            previous_tests="test code",
        )
        assert "Generate unit tests for authentication" in result

    def test_includes_error_message(self) -> None:
        """Test that retry prompt includes error message."""
        result = build_test_retry_prompt(
            original_prompt="original",
            error_message="Line 42: expected ':'",
            previous_tests="code",
        )
        assert "Line 42: expected ':'" in result

    def test_includes_previous_tests_excerpt(self) -> None:
        """Test that retry prompt includes previous tests excerpt."""
        result = build_test_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_tests="def test_broken():\n    assert True",
        )
        assert "def test_broken():" in result

    def test_truncates_long_previous_tests(self) -> None:
        """Test that long previous tests are truncated."""
        long_tests = "def test_x(): pass\n" * 200  # Much longer than 500 chars
        result = build_test_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_tests=long_tests,
        )
        assert "truncated" in result

    def test_mentions_syntax_error_fix(self) -> None:
        """Test that retry prompt asks to fix syntax error."""
        result = build_test_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_tests="code",
        )
        assert "syntax error" in result.lower()


class TestPromptIntegration:
    """Integration tests for test generation prompt system."""

    def test_full_prompt_is_well_formed(self) -> None:
        """Test that a complete prompt has all expected sections."""
        result = build_test_generation_prompt(
            implementation_code='''
def calculate_total(items: list[float]) -> float:
    """Calculate the total of all items.

    Args:
        items: List of numeric values.

    Returns:
        Sum of all items.
    """
    return sum(items)


def validate_email(email: str) -> bool:
    """Validate email format.

    Args:
        email: Email address to validate.

    Returns:
        True if valid, False otherwise.
    """
    return "@" in email and "." in email
''',
            function_list=["calculate_total", "validate_email"],
            module_name="utils",
            additional_context="Part of the validation utilities package",
        )

        # Check all major sections are present
        assert "calculate_total" in result
        assert "validate_email" in result
        assert "utils" in result
        assert "Part of the validation utilities package" in result
        assert "pytest" in result.lower()
        assert "edge case" in result.lower()

    def test_prompt_format_for_llm(self) -> None:
        """Test that prompt format is suitable for LLM consumption."""
        result = build_test_generation_prompt(
            implementation_code="def test(): pass",
            function_list=["test"],
            module_name="test_module",
        )

        # Should mention pytest
        assert "pytest" in result.lower()
        # Should mention code block markers
        assert "```python" in result or "```" in result
        # Should have clear instruction markers
        assert "Generate" in result or "test" in result.lower()
