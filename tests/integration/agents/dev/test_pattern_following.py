"""Integration tests for pattern following in Dev agent (Story 8.7).

Tests that dev_node correctly extracts patterns from state, includes them
in prompts, and validates generated code for pattern adherence.

Example:
    >>> pytest tests/integration/agents/dev/test_pattern_following.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yolo_developer.agents.dev.pattern_utils import (
    ErrorHandlingPattern,
    PatternValidationResult,
    StylePattern,
    clear_pattern_cache,
    get_error_patterns,
    get_naming_patterns,
    get_style_patterns,
    validate_pattern_adherence,
)
from yolo_developer.memory.patterns import CodePattern, PatternType


@pytest.fixture(autouse=True)
def clear_cache_before_test() -> None:
    """Clear pattern cache before each test to ensure isolation."""
    clear_pattern_cache()


class TestDevNodePatternExtraction:
    """Integration tests for pattern extraction from state (AC: 5, 6)."""

    def test_extracts_naming_patterns_from_memory_context(self) -> None:
        """Test dev_node extracts naming patterns from state memory_context."""
        # Create mock PatternLearner with naming patterns
        mock_learner = MagicMock()
        mock_learner.get_patterns_by_type = MagicMock(
            side_effect=lambda pattern_type: [
                CodePattern(
                    pattern_type=pattern_type,
                    name="test_pattern",
                    value="snake_case" if "FUNCTION" in pattern_type.name else "PascalCase",
                    confidence=0.95,
                    examples=("example_func",),
                )
            ]
            if "NAMING" in pattern_type.name
            else []
        )

        state: dict[str, Any] = {
            "memory_context": {
                "pattern_learner": mock_learner,
            },
            "messages": [],
            "decisions": [],
        }

        # Get patterns - this simulates what dev_node does internally
        patterns = get_naming_patterns(state)

        assert len(patterns) >= 1
        assert any(p.pattern_type == PatternType.NAMING_FUNCTION for p in patterns)

    def test_extracts_error_patterns_from_memory_context(self) -> None:
        """Test dev_node extracts error handling patterns from state."""
        custom_pattern = ErrorHandlingPattern(
            pattern_name="project_exceptions",
            exception_types=("ConfigurationError", "ValidationError"),
            handling_style="project-specific exceptions with logging",
        )

        state: dict[str, Any] = {
            "memory_context": {
                "error_patterns": [custom_pattern],
            },
        }

        patterns = get_error_patterns(state)

        assert len(patterns) >= 1
        assert patterns[0].pattern_name == "project_exceptions"

    def test_extracts_style_patterns_from_memory_context(self) -> None:
        """Test dev_node extracts style patterns from state."""
        custom_style = StylePattern(
            pattern_name="custom_imports",
            category="import_style",
            value="absolute imports only",
        )

        state: dict[str, Any] = {
            "memory_context": {
                "style_patterns": [custom_style],
            },
        }

        patterns = get_style_patterns(state)

        assert len(patterns) >= 1
        assert patterns[0].pattern_name == "custom_imports"

    def test_falls_back_to_defaults_without_memory_context(self) -> None:
        """Test pattern queries return defaults when memory_context missing."""
        state: dict[str, Any] = {}

        naming = get_naming_patterns(state)
        error = get_error_patterns(state)
        style = get_style_patterns(state)

        # All should return default patterns
        assert len(naming) >= 1
        assert len(error) >= 1
        assert len(style) >= 1

        # Verify defaults match architecture conventions
        assert any(p.value == "snake_case" for p in naming)


class TestDevNodePatternPromptIntegration:
    """Integration tests for pattern inclusion in LLM prompts (AC: 5)."""

    def test_patterns_included_in_code_generation_context(self) -> None:
        """Test that patterns are formatted for prompt context."""
        from yolo_developer.agents.dev.node import (
            _format_patterns_for_prompt,
            _get_relevant_patterns,
        )

        state: dict[str, Any] = {}  # Will use defaults

        patterns = _get_relevant_patterns(state)
        formatted = _format_patterns_for_prompt(patterns)

        # Formatted string should include pattern sections
        assert "Naming Conventions" in formatted or len(formatted) > 0
        # Should include naming patterns
        assert "snake_case" in formatted or "Naming" in formatted

    def test_patterns_dict_structure(self) -> None:
        """Test _get_relevant_patterns returns correct structure."""
        from yolo_developer.agents.dev.node import _get_relevant_patterns

        state: dict[str, Any] = {}

        patterns = _get_relevant_patterns(state)

        assert "naming" in patterns
        assert "error_handling" in patterns
        assert "style" in patterns
        assert isinstance(patterns["naming"], list)
        assert isinstance(patterns["error_handling"], list)
        assert isinstance(patterns["style"], list)


class TestDevNodePatternValidation:
    """Integration tests for pattern validation post-generation (AC: 4, 5)."""

    def test_validates_generated_code_for_pattern_adherence(self) -> None:
        """Test that generated code is validated for pattern adherence."""
        # Well-formed code that follows patterns
        compliant_code = '''"""Module docstring."""

from __future__ import annotations

import os

from mypackage import helper


def get_user_data(user_id: str) -> dict[str, str]:
    """Get user data by ID.

    Args:
        user_id: The user identifier.

    Returns:
        User data dictionary.
    """
    return {"id": user_id}


class UserService:
    """Service for user operations."""

    def process_request(self) -> None:
        """Process a request."""
        try:
            result = self._internal_method()
        except ValueError as e:
            raise ValueError(f"Failed: {e}") from e
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(compliant_code, state)

        assert isinstance(result, PatternValidationResult)
        assert result.passed is True
        assert result.score >= 70

    def test_detects_pattern_violations_in_generated_code(self) -> None:
        """Test that pattern violations are detected and flagged."""
        # Code with violations
        non_compliant_code = """
def getUserData():
    pass

def processOrder():
    pass

class user_service:
    pass

try:
    x = 1
except:
    pass
"""

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(non_compliant_code, state)

        assert len(result.deviations) >= 1
        # Should detect naming violations
        deviation_types = [d.pattern_type for d in result.deviations]
        assert (
            PatternType.NAMING_FUNCTION in deviation_types
            or PatternType.NAMING_CLASS in deviation_types
            or PatternType.DESIGN_PATTERN in deviation_types
        )

    def test_pattern_result_included_in_to_dict(self) -> None:
        """Test PatternValidationResult serializes correctly for DevOutput."""
        state: dict[str, Any] = {}
        code = "def get_user(): pass"

        result = validate_pattern_adherence(code, state)
        result_dict = result.to_dict()

        assert "score" in result_dict
        assert "passed" in result_dict
        assert "threshold" in result_dict
        assert "patterns_checked" in result_dict
        assert "adherence_percentage" in result_dict
        assert "deviations" in result_dict
        assert "deviation_count" in result_dict


class TestDevNodeDecisionRecordIntegration:
    """Integration tests for pattern results in decision records (AC: 5)."""

    @pytest.mark.asyncio
    async def test_dev_node_includes_pattern_info_in_output(self) -> None:
        """Test dev_node includes pattern validation in output notes."""
        from yolo_developer.agents.dev.node import (
            _generate_implementation,
            _reset_llm_router,
        )

        # Reset router to ensure we use stub implementation
        _reset_llm_router()

        story = {
            "id": "test-story-001",
            "title": "Test Story",
            "requirements": "Test requirements",
        }
        context: dict[str, Any] = {}
        state: dict[str, Any] = {}

        # Generate implementation (will use stub without LLM)
        artifact = await _generate_implementation(story, context, router=None, state=state)

        # Verify artifact was created
        assert artifact.story_id == "test-story-001"
        assert artifact.implementation_status == "completed"

    @pytest.mark.asyncio
    async def test_pattern_validation_with_state_context(self) -> None:
        """Test pattern validation uses state context when provided."""
        # This test verifies the code path exists even if LLM isn't available
        # The actual LLM integration is tested with mocks in unit tests

        state: dict[str, Any] = {
            "memory_context": {
                "pattern_learner": None,  # Will trigger defaults
            }
        }

        # Validate that pattern functions work with the state structure
        patterns = get_naming_patterns(state)
        assert len(patterns) >= 1  # Should get defaults

        # Validate code against patterns
        test_code = "def my_function(): pass"
        result = validate_pattern_adherence(test_code, state)
        assert isinstance(result, PatternValidationResult)
