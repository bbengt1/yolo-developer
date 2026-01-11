"""Unit tests for integration test generation prompts (Story 8.4, AC6).

Tests for the integration test prompt templates that guide LLM-powered
generation of integration tests for cross-component functionality.
"""

from __future__ import annotations

import pytest


class TestIntegrationTestTemplate:
    """Tests for INTEGRATION_TEST_TEMPLATE constant."""

    def test_template_exists(self) -> None:
        """Test that template constant is exported."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
        )

        assert INTEGRATION_TEST_TEMPLATE is not None
        assert isinstance(INTEGRATION_TEST_TEMPLATE, str)
        assert len(INTEGRATION_TEST_TEMPLATE) > 100  # Should be substantial

    def test_template_includes_boundary_testing_guidance(self) -> None:
        """Test that template includes boundary testing requirements."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
        )

        assert "boundary" in INTEGRATION_TEST_TEMPLATE.lower()
        assert "component" in INTEGRATION_TEST_TEMPLATE.lower()
        assert "interaction" in INTEGRATION_TEST_TEMPLATE.lower()

    def test_template_includes_data_flow_verification(self) -> None:
        """Test that template includes data flow verification requirements."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
            INTEGRATION_TESTING_BEST_PRACTICES,
        )

        # Check main template has data flow mention
        assert "data flow" in INTEGRATION_TEST_TEMPLATE.lower()

        # Check best practices (included in prompt) has full guidance
        best_practices_lower = INTEGRATION_TESTING_BEST_PRACTICES.lower()
        assert "transformation" in best_practices_lower
        assert "integrity" in best_practices_lower

    def test_template_includes_error_handling_requirements(self) -> None:
        """Test that template includes error handling test requirements."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
        )

        assert "error" in INTEGRATION_TEST_TEMPLATE.lower()
        assert "graceful" in INTEGRATION_TEST_TEMPLATE.lower() or "degradation" in INTEGRATION_TEST_TEMPLATE.lower()

    def test_template_includes_pytest_structure(self) -> None:
        """Test that template includes pytest structure guidance."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
        )

        template_lower = INTEGRATION_TEST_TEMPLATE.lower()
        assert "pytest" in template_lower
        assert "fixture" in template_lower
        assert "mock" in template_lower

    def test_template_includes_async_guidance(self) -> None:
        """Test that template includes async test guidance."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            INTEGRATION_TEST_TEMPLATE,
        )

        # Should mention asyncio marker for async tests
        assert "asyncio" in INTEGRATION_TEST_TEMPLATE.lower()


class TestBuildIntegrationTestPrompt:
    """Tests for build_integration_test_prompt() function."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        assert callable(build_integration_test_prompt)

    def test_renders_code_files_content(self) -> None:
        """Test that prompt includes code file contents."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        code_content = "def process_data(x): return x * 2"

        prompt = build_integration_test_prompt(
            code_files_content=code_content,
            boundaries="Module A imports Module B",
            data_flows="Data enters at input, exits at output",
            error_scenarios="ValueError on invalid input",
        )

        assert code_content in prompt

    def test_renders_boundaries(self) -> None:
        """Test that prompt includes boundary information."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        boundaries = "Module A calls function B from Module B"

        prompt = build_integration_test_prompt(
            code_files_content="def foo(): pass",
            boundaries=boundaries,
            data_flows="",
            error_scenarios="",
        )

        assert boundaries in prompt

    def test_renders_data_flows(self) -> None:
        """Test that prompt includes data flow paths."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        data_flows = "Input -> Transform -> Validate -> Output"

        prompt = build_integration_test_prompt(
            code_files_content="def foo(): pass",
            boundaries="",
            data_flows=data_flows,
            error_scenarios="",
        )

        assert data_flows in prompt

    def test_renders_error_scenarios(self) -> None:
        """Test that prompt includes error scenarios."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        error_scenarios = "Invalid input raises ValueError"

        prompt = build_integration_test_prompt(
            code_files_content="def foo(): pass",
            boundaries="",
            data_flows="",
            error_scenarios=error_scenarios,
        )

        assert error_scenarios in prompt

    def test_includes_template_content(self) -> None:
        """Test that prompt includes template guidance."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_prompt,
        )

        prompt = build_integration_test_prompt(
            code_files_content="def foo(): pass",
            boundaries="A -> B",
            data_flows="In -> Out",
            error_scenarios="Error case",
        )

        # Should include testing requirements from template
        prompt_lower = prompt.lower()
        assert "integration" in prompt_lower
        assert "pytest" in prompt_lower or "test" in prompt_lower


class TestBuildIntegrationTestRetryPrompt:
    """Tests for build_integration_test_retry_prompt() function."""

    def test_function_exists(self) -> None:
        """Test that function is exported."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        assert callable(build_integration_test_retry_prompt)

    def test_includes_original_prompt(self) -> None:
        """Test that retry includes original prompt."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        original = "Generate integration tests for module A"
        error = "SyntaxError at line 5"
        previous = "def test_foo():\n  invalid"

        retry = build_integration_test_retry_prompt(original, error, previous)

        assert original in retry

    def test_includes_error_message(self) -> None:
        """Test that retry includes error message."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        original = "Generate tests"
        error = "IndentationError: unexpected indent"
        previous = "def test_foo():\n  invalid"

        retry = build_integration_test_retry_prompt(original, error, previous)

        assert error in retry

    def test_includes_previous_tests_excerpt(self) -> None:
        """Test that retry includes excerpt from previous attempt."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        original = "Generate tests"
        error = "SyntaxError"
        previous = "def test_foo():\n    pass\n    invalid syntax here"

        retry = build_integration_test_retry_prompt(original, error, previous)

        assert "test_foo" in retry

    def test_truncates_long_previous_tests(self) -> None:
        """Test that very long previous tests are truncated."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        original = "Generate tests"
        error = "SyntaxError"
        previous = "x" * 1000  # Long string

        retry = build_integration_test_retry_prompt(original, error, previous)

        # Should be truncated and have truncation indicator
        assert "truncated" in retry.lower() or len(retry) < len(previous) + 500

    def test_mentions_syntax_fix(self) -> None:
        """Test that retry prompt asks for syntax fix."""
        from yolo_developer.agents.dev.prompts.integration_test_generation import (
            build_integration_test_retry_prompt,
        )

        original = "Generate tests"
        error = "SyntaxError"
        previous = "def test_foo(): invalid"

        retry = build_integration_test_retry_prompt(original, error, previous)

        retry_lower = retry.lower()
        assert "fix" in retry_lower or "correct" in retry_lower or "syntax" in retry_lower
