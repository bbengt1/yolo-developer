"""Unit tests for documentation generation prompts (Story 8.5).

Tests the documentation prompt templates and builder functions used
for LLM-powered documentation enhancement.
"""

from __future__ import annotations


class TestDocumentationGuidelines:
    """Tests for DOCUMENTATION_GUIDELINES constant."""

    def test_guidelines_exist(self) -> None:
        """Test that DOCUMENTATION_GUIDELINES constant is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
        )

        assert DOCUMENTATION_GUIDELINES is not None
        assert len(DOCUMENTATION_GUIDELINES) > 0

    def test_guidelines_include_google_style(self) -> None:
        """Test that guidelines reference Google-style docstrings."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
        )

        assert "google" in DOCUMENTATION_GUIDELINES.lower()

    def test_guidelines_include_args_section(self) -> None:
        """Test that guidelines mention Args section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
        )

        assert "Args" in DOCUMENTATION_GUIDELINES

    def test_guidelines_include_returns_section(self) -> None:
        """Test that guidelines mention Returns section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
        )

        assert "Returns" in DOCUMENTATION_GUIDELINES

    def test_guidelines_include_example_section(self) -> None:
        """Test that guidelines mention Example section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
        )

        assert "Example" in DOCUMENTATION_GUIDELINES


class TestModuleDocstringTemplate:
    """Tests for MODULE_DOCSTRING_TEMPLATE constant."""

    def test_template_exists(self) -> None:
        """Test that MODULE_DOCSTRING_TEMPLATE constant is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            MODULE_DOCSTRING_TEMPLATE,
        )

        assert MODULE_DOCSTRING_TEMPLATE is not None
        assert len(MODULE_DOCSTRING_TEMPLATE) > 0

    def test_template_includes_one_line_summary(self) -> None:
        """Test that template mentions one-line summary."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            MODULE_DOCSTRING_TEMPLATE,
        )

        template_lower = MODULE_DOCSTRING_TEMPLATE.lower()
        assert "one-line" in template_lower or "summary" in template_lower

    def test_template_includes_example_section(self) -> None:
        """Test that template mentions Example section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            MODULE_DOCSTRING_TEMPLATE,
        )

        assert "Example" in MODULE_DOCSTRING_TEMPLATE


class TestFunctionDocstringTemplate:
    """Tests for FUNCTION_DOCSTRING_TEMPLATE constant."""

    def test_template_exists(self) -> None:
        """Test that FUNCTION_DOCSTRING_TEMPLATE constant is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            FUNCTION_DOCSTRING_TEMPLATE,
        )

        assert FUNCTION_DOCSTRING_TEMPLATE is not None
        assert len(FUNCTION_DOCSTRING_TEMPLATE) > 0

    def test_template_includes_args(self) -> None:
        """Test that template mentions Args section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            FUNCTION_DOCSTRING_TEMPLATE,
        )

        assert "Args" in FUNCTION_DOCSTRING_TEMPLATE

    def test_template_includes_returns(self) -> None:
        """Test that template mentions Returns section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            FUNCTION_DOCSTRING_TEMPLATE,
        )

        assert "Returns" in FUNCTION_DOCSTRING_TEMPLATE

    def test_template_includes_raises(self) -> None:
        """Test that template mentions Raises section."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            FUNCTION_DOCSTRING_TEMPLATE,
        )

        assert "Raises" in FUNCTION_DOCSTRING_TEMPLATE


class TestDocumentationGenerationTemplate:
    """Tests for DOCUMENTATION_GENERATION_TEMPLATE constant."""

    def test_template_exists(self) -> None:
        """Test that DOCUMENTATION_GENERATION_TEMPLATE constant is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GENERATION_TEMPLATE,
        )

        assert DOCUMENTATION_GENERATION_TEMPLATE is not None
        assert len(DOCUMENTATION_GENERATION_TEMPLATE) > 0

    def test_template_has_code_placeholder(self) -> None:
        """Test that template has code content placeholder."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GENERATION_TEMPLATE,
        )

        assert "{code_content}" in DOCUMENTATION_GENERATION_TEMPLATE

    def test_template_has_analysis_placeholder(self) -> None:
        """Test that template has documentation analysis placeholder."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GENERATION_TEMPLATE,
        )

        assert "{documentation_analysis}" in DOCUMENTATION_GENERATION_TEMPLATE

    def test_template_has_complex_sections_placeholder(self) -> None:
        """Test that template has complex sections placeholder."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GENERATION_TEMPLATE,
        )

        assert "{complex_sections}" in DOCUMENTATION_GENERATION_TEMPLATE

    def test_template_requests_python_output(self) -> None:
        """Test that template asks for Python code output."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GENERATION_TEMPLATE,
        )

        template_lower = DOCUMENTATION_GENERATION_TEMPLATE.lower()
        assert "python" in template_lower


class TestBuildDocumentationPrompt:
    """Tests for build_documentation_prompt function."""

    def test_function_exists(self) -> None:
        """Test that build_documentation_prompt function is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        assert callable(build_documentation_prompt)

    def test_builds_prompt_with_code(self) -> None:
        """Test that function builds prompt containing provided code."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        code = "def hello(): pass"
        prompt = build_documentation_prompt(
            code_content=code,
            documentation_analysis="No docstrings found",
            complex_sections="None detected",
        )

        assert code in prompt

    def test_includes_analysis(self) -> None:
        """Test that function includes documentation analysis."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        analysis = "Missing module docstring"
        prompt = build_documentation_prompt(
            code_content="def hello(): pass",
            documentation_analysis=analysis,
            complex_sections="None",
        )

        assert analysis in prompt

    def test_includes_complex_sections(self) -> None:
        """Test that function includes complex sections info."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        sections = "Lines 10-25: nested loop"
        prompt = build_documentation_prompt(
            code_content="def hello(): pass",
            documentation_analysis="None",
            complex_sections=sections,
        )

        assert sections in prompt

    def test_includes_guidelines_by_default(self) -> None:
        """Test that function includes guidelines by default."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        prompt = build_documentation_prompt(
            code_content="def hello(): pass",
            documentation_analysis="None",
            complex_sections="None",
        )

        assert "Args" in prompt
        assert "Returns" in prompt

    def test_can_exclude_guidelines(self) -> None:
        """Test that guidelines can be excluded."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            DOCUMENTATION_GUIDELINES,
            build_documentation_prompt,
        )

        prompt = build_documentation_prompt(
            code_content="def hello(): pass",
            documentation_analysis="None",
            complex_sections="None",
            include_guidelines=False,
        )

        # Guidelines section should not be fully included
        assert DOCUMENTATION_GUIDELINES not in prompt

    def test_includes_additional_context(self) -> None:
        """Test that additional context is included when provided."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_prompt,
        )

        context = "This module handles user authentication"
        prompt = build_documentation_prompt(
            code_content="def hello(): pass",
            documentation_analysis="None",
            complex_sections="None",
            additional_context=context,
        )

        assert context in prompt


class TestBuildDocumentationRetryPrompt:
    """Tests for build_documentation_retry_prompt function."""

    def test_function_exists(self) -> None:
        """Test that build_documentation_retry_prompt function is defined."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        assert callable(build_documentation_retry_prompt)

    def test_includes_original_prompt(self) -> None:
        """Test that retry prompt includes original prompt."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        original = "Generate documentation for this code"
        retry = build_documentation_retry_prompt(
            original_prompt=original,
            error_message="SyntaxError at line 5",
            previous_code="def broken(",
        )

        assert original in retry

    def test_includes_error_message(self) -> None:
        """Test that retry prompt includes error message."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        error = "IndentationError: unexpected indent"
        retry = build_documentation_retry_prompt(
            original_prompt="Generate docs",
            error_message=error,
            previous_code="def broken(",
        )

        assert error in retry

    def test_includes_previous_code_excerpt(self) -> None:
        """Test that retry prompt includes previous code excerpt."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        code = "def test(): pass"
        retry = build_documentation_retry_prompt(
            original_prompt="Generate docs",
            error_message="Error",
            previous_code=code,
        )

        assert code in retry

    def test_truncates_long_previous_code(self) -> None:
        """Test that long previous code is truncated."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        long_code = "x = 1\n" * 200  # Very long code
        retry = build_documentation_retry_prompt(
            original_prompt="Generate docs",
            error_message="Error",
            previous_code=long_code,
        )

        # Should be truncated
        assert "truncated" in retry.lower() or len(retry) < len(long_code) + 500

    def test_mentions_syntax_fix(self) -> None:
        """Test that retry prompt mentions fixing syntax."""
        from yolo_developer.agents.dev.prompts.documentation_generation import (
            build_documentation_retry_prompt,
        )

        retry = build_documentation_retry_prompt(
            original_prompt="Generate docs",
            error_message="SyntaxError",
            previous_code="def broken(",
        )

        retry_lower = retry.lower()
        assert "syntax" in retry_lower or "fix" in retry_lower
