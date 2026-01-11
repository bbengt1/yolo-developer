"""Unit tests for code generation prompt templates (Story 8.2 - Task 7).

Tests prompt template rendering, maintainability guidelines inclusion,
and project conventions.
"""

from __future__ import annotations

from yolo_developer.agents.dev.prompts import (
    CODE_GENERATION_TEMPLATE,
    MAINTAINABILITY_GUIDELINES,
    PROJECT_CONVENTIONS,
    build_code_generation_prompt,
)
from yolo_developer.agents.dev.prompts.code_generation import build_retry_prompt


class TestCodeGenerationTemplate:
    """Tests for the CODE_GENERATION_TEMPLATE constant."""

    def test_template_has_story_title_placeholder(self) -> None:
        """Test that template has story_title placeholder."""
        assert "{story_title}" in CODE_GENERATION_TEMPLATE

    def test_template_has_requirements_placeholder(self) -> None:
        """Test that template has requirements placeholder."""
        assert "{requirements}" in CODE_GENERATION_TEMPLATE

    def test_template_has_acceptance_criteria_placeholder(self) -> None:
        """Test that template has acceptance_criteria placeholder."""
        assert "{acceptance_criteria}" in CODE_GENERATION_TEMPLATE

    def test_template_has_design_decisions_placeholder(self) -> None:
        """Test that template has design_decisions placeholder."""
        assert "{design_decisions}" in CODE_GENERATION_TEMPLATE

    def test_template_has_maintainability_guidelines_placeholder(self) -> None:
        """Test that template has maintainability_guidelines placeholder."""
        assert "{maintainability_guidelines}" in CODE_GENERATION_TEMPLATE

    def test_template_has_project_conventions_placeholder(self) -> None:
        """Test that template has project_conventions placeholder."""
        assert "{project_conventions}" in CODE_GENERATION_TEMPLATE

    def test_template_has_additional_context_placeholder(self) -> None:
        """Test that template has additional_context placeholder."""
        assert "{additional_context}" in CODE_GENERATION_TEMPLATE


class TestMaintainabilityGuidelines:
    """Tests for MAINTAINABILITY_GUIDELINES constant (AC1, AC2, AC3, AC4)."""

    def test_includes_maintainability_first_hierarchy(self) -> None:
        """Test that guidelines include maintainability-first hierarchy."""
        assert "maintainability-first" in MAINTAINABILITY_GUIDELINES.lower()

    def test_includes_readability_priority(self) -> None:
        """Test that readability is mentioned as priority."""
        assert "Readability" in MAINTAINABILITY_GUIDELINES

    def test_includes_function_size_limits(self) -> None:
        """Test that function size limits are specified (AC1, AC3)."""
        assert "50 lines" in MAINTAINABILITY_GUIDELINES

    def test_includes_nesting_depth_limits(self) -> None:
        """Test that nesting depth limits are specified (AC3)."""
        assert "3 levels" in MAINTAINABILITY_GUIDELINES

    def test_includes_cyclomatic_complexity(self) -> None:
        """Test that cyclomatic complexity is mentioned (AC1)."""
        assert "Cyclomatic complexity" in MAINTAINABILITY_GUIDELINES

    def test_includes_single_responsibility(self) -> None:
        """Test that single responsibility is mentioned (AC3)."""
        assert "Single responsibility" in MAINTAINABILITY_GUIDELINES

    def test_includes_naming_conventions(self) -> None:
        """Test that naming conventions are included (AC2)."""
        assert "descriptive" in MAINTAINABILITY_GUIDELINES.lower()

    def test_includes_no_single_letter_vars(self) -> None:
        """Test that single letter variable restriction is mentioned (AC2)."""
        assert "single-letter" in MAINTAINABILITY_GUIDELINES.lower()

    def test_includes_yagni_principle(self) -> None:
        """Test that YAGNI principle is included (AC4)."""
        assert "YAGNI" in MAINTAINABILITY_GUIDELINES


class TestProjectConventions:
    """Tests for PROJECT_CONVENTIONS constant."""

    def test_includes_snake_case(self) -> None:
        """Test that snake_case convention is included."""
        assert "snake_case" in PROJECT_CONVENTIONS

    def test_includes_pascal_case(self) -> None:
        """Test that PascalCase convention is included."""
        assert "PascalCase" in PROJECT_CONVENTIONS

    def test_includes_type_annotations(self) -> None:
        """Test that type annotations requirement is included."""
        assert "type annotation" in PROJECT_CONVENTIONS.lower()

    def test_includes_async_patterns(self) -> None:
        """Test that async patterns are included."""
        assert "async/await" in PROJECT_CONVENTIONS

    def test_includes_docstring_requirement(self) -> None:
        """Test that docstring requirement is included."""
        assert "docstring" in PROJECT_CONVENTIONS.lower()

    def test_includes_future_annotations(self) -> None:
        """Test that future annotations import is mentioned."""
        assert "from __future__ import annotations" in PROJECT_CONVENTIONS


class TestBuildCodeGenerationPrompt:
    """Tests for build_code_generation_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = build_code_generation_prompt(
            story_title="Test Story",
            requirements="Test requirements",
        )
        assert isinstance(result, str)

    def test_includes_story_title(self) -> None:
        """Test that prompt includes the story title."""
        result = build_code_generation_prompt(
            story_title="User Authentication",
            requirements="Implement login",
        )
        assert "User Authentication" in result

    def test_includes_requirements(self) -> None:
        """Test that prompt includes the requirements."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Implement user login with email and password validation",
        )
        assert "Implement user login with email and password validation" in result

    def test_includes_acceptance_criteria(self) -> None:
        """Test that prompt includes acceptance criteria when provided."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            acceptance_criteria=["Users can login", "Invalid credentials rejected"],
        )
        assert "Users can login" in result
        assert "Invalid credentials rejected" in result

    def test_handles_empty_acceptance_criteria(self) -> None:
        """Test that prompt handles empty acceptance criteria."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            acceptance_criteria=None,
        )
        assert "No specific acceptance criteria provided" in result

    def test_includes_design_decisions(self) -> None:
        """Test that prompt includes design decisions when provided."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            design_decisions={"pattern": "Repository", "db": "PostgreSQL"},
        )
        assert "pattern" in result
        assert "Repository" in result
        assert "PostgreSQL" in result

    def test_handles_empty_design_decisions(self) -> None:
        """Test that prompt handles empty design decisions."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            design_decisions=None,
        )
        assert "No specific design decisions provided" in result

    def test_includes_maintainability_guidelines_by_default(self) -> None:
        """Test that maintainability guidelines are included by default."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
        )
        assert "50 lines" in result
        assert "Readability" in result

    def test_excludes_maintainability_guidelines_when_disabled(self) -> None:
        """Test that maintainability guidelines can be excluded."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            include_maintainability=False,
        )
        # Specific maintainability content should not be present
        assert "Cyclomatic complexity" not in result

    def test_includes_project_conventions_by_default(self) -> None:
        """Test that project conventions are included by default."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
        )
        assert "snake_case" in result
        assert "PascalCase" in result

    def test_excludes_project_conventions_when_disabled(self) -> None:
        """Test that project conventions can be excluded."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            include_conventions=False,
        )
        # Project conventions should not be present
        assert "Use snake_case for functions" not in result

    def test_includes_additional_context(self) -> None:
        """Test that additional context is included when provided."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            additional_context="This builds on the existing auth module",
        )
        assert "This builds on the existing auth module" in result

    def test_handles_empty_additional_context(self) -> None:
        """Test that empty additional context is handled."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test req",
            additional_context="",
        )
        assert "No additional context provided" in result


class TestBuildRetryPrompt:
    """Tests for build_retry_prompt function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        result = build_retry_prompt(
            original_prompt="Generate code...",
            error_message="Line 5: invalid syntax",
            previous_code="def test():\n  pass",
        )
        assert isinstance(result, str)

    def test_includes_original_prompt(self) -> None:
        """Test that retry prompt includes original prompt."""
        result = build_retry_prompt(
            original_prompt="Generate user login code",
            error_message="syntax error",
            previous_code="code",
        )
        assert "Generate user login code" in result

    def test_includes_error_message(self) -> None:
        """Test that retry prompt includes error message."""
        result = build_retry_prompt(
            original_prompt="original",
            error_message="Line 42: unexpected indent",
            previous_code="code",
        )
        assert "Line 42: unexpected indent" in result

    def test_includes_previous_code_excerpt(self) -> None:
        """Test that retry prompt includes previous code excerpt."""
        result = build_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_code="def broken_function():\n    return None",
        )
        assert "def broken_function():" in result

    def test_truncates_long_previous_code(self) -> None:
        """Test that long previous code is truncated."""
        long_code = "x = 1\n" * 200  # Much longer than 500 chars
        result = build_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_code=long_code,
        )
        assert "truncated" in result

    def test_mentions_syntax_error_fix(self) -> None:
        """Test that retry prompt asks to fix syntax error."""
        result = build_retry_prompt(
            original_prompt="original",
            error_message="error",
            previous_code="code",
        )
        assert "syntax error" in result.lower()


class TestPromptIntegration:
    """Integration tests for prompt template system."""

    def test_full_prompt_is_well_formed(self) -> None:
        """Test that a complete prompt has all expected sections."""
        result = build_code_generation_prompt(
            story_title="Complete User Profile Management",
            requirements="Allow users to update their profile information",
            acceptance_criteria=[
                "Users can update their name",
                "Users can update their email",
                "Changes are validated before saving",
            ],
            design_decisions={
                "pattern": "Repository Pattern",
                "validation": "Pydantic models",
            },
            additional_context="Builds on existing User model in models/user.py",
        )

        # Check all major sections are present
        assert "Complete User Profile Management" in result
        assert "Allow users to update their profile information" in result
        assert "Users can update their name" in result
        assert "Repository Pattern" in result
        assert "Builds on existing User model" in result
        assert "snake_case" in result
        assert "50 lines" in result

    def test_prompt_format_for_llm(self) -> None:
        """Test that prompt format is suitable for LLM consumption."""
        result = build_code_generation_prompt(
            story_title="Test",
            requirements="Test",
        )

        # Should have clear instruction markers
        assert "Instructions" in result or "Generate" in result
        # Should mention Python
        assert "Python" in result
        # Should mention code block markers
        assert "```python" in result
