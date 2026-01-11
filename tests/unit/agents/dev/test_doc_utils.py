"""Unit tests for documentation utilities (Story 8.5).

Tests the documentation analysis, generation, and validation utilities
used by the Dev agent for comprehensive documentation enhancement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from yolo_developer.llm.router import LLMRouter


class TestDocumentationInfo:
    """Tests for DocumentationInfo dataclass."""

    def test_dataclass_exists(self) -> None:
        """Test that DocumentationInfo dataclass is defined."""
        from yolo_developer.agents.dev.doc_utils import DocumentationInfo

        assert DocumentationInfo is not None

    def test_dataclass_is_frozen(self) -> None:
        """Test that DocumentationInfo is frozen."""
        from yolo_developer.agents.dev.doc_utils import DocumentationInfo

        info = DocumentationInfo(
            has_module_docstring=True,
            functions_missing_docstrings=(),
            functions_with_incomplete_docstrings=(),
            complex_sections=(),
            total_public_functions=1,
            documented_functions=1,
        )
        with pytest.raises(AttributeError):
            info.has_module_docstring = False  # type: ignore[misc]

    def test_documentation_coverage_full(self) -> None:
        """Test documentation_coverage property with full coverage."""
        from yolo_developer.agents.dev.doc_utils import DocumentationInfo

        info = DocumentationInfo(
            has_module_docstring=True,
            functions_missing_docstrings=(),
            functions_with_incomplete_docstrings=(),
            complex_sections=(),
            total_public_functions=5,
            documented_functions=5,
        )
        assert info.documentation_coverage == 100.0

    def test_documentation_coverage_partial(self) -> None:
        """Test documentation_coverage property with partial coverage."""
        from yolo_developer.agents.dev.doc_utils import DocumentationInfo

        info = DocumentationInfo(
            has_module_docstring=True,
            functions_missing_docstrings=("func1", "func2"),
            functions_with_incomplete_docstrings=(),
            complex_sections=(),
            total_public_functions=4,
            documented_functions=2,
        )
        assert info.documentation_coverage == 50.0

    def test_documentation_coverage_no_functions(self) -> None:
        """Test documentation_coverage with no functions returns 100."""
        from yolo_developer.agents.dev.doc_utils import DocumentationInfo

        info = DocumentationInfo(
            has_module_docstring=True,
            functions_missing_docstrings=(),
            functions_with_incomplete_docstrings=(),
            complex_sections=(),
            total_public_functions=0,
            documented_functions=0,
        )
        assert info.documentation_coverage == 100.0


class TestComplexSection:
    """Tests for ComplexSection dataclass."""

    def test_dataclass_exists(self) -> None:
        """Test that ComplexSection dataclass is defined."""
        from yolo_developer.agents.dev.doc_utils import ComplexSection

        assert ComplexSection is not None

    def test_dataclass_is_frozen(self) -> None:
        """Test that ComplexSection is frozen."""
        from yolo_developer.agents.dev.doc_utils import ComplexSection

        section = ComplexSection(
            start_line=10,
            end_line=25,
            complexity_type="nested_loop",
            function_name="process",
            description="Nested for loops",
        )
        with pytest.raises(AttributeError):
            section.start_line = 5  # type: ignore[misc]

    def test_complexity_types(self) -> None:
        """Test valid complexity types."""
        from yolo_developer.agents.dev.doc_utils import ComplexSection

        for complexity_type in ["nested_loop", "long_function", "complex_conditional", "deep_nesting"]:
            section = ComplexSection(
                start_line=1,
                end_line=10,
                complexity_type=complexity_type,  # type: ignore[arg-type]
                function_name=None,
                description="Test",
            )
            assert section.complexity_type == complexity_type


class TestDocumentationQualityReport:
    """Tests for DocumentationQualityReport dataclass."""

    def test_dataclass_exists(self) -> None:
        """Test that DocumentationQualityReport dataclass is defined."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        assert DocumentationQualityReport is not None

    def test_default_values(self) -> None:
        """Test default values for DocumentationQualityReport."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        report = DocumentationQualityReport()
        assert report.warnings == []
        assert report.has_module_docstring is False
        assert report.functions_with_args == 0
        assert report.functions_with_returns == 0
        assert report.functions_with_examples == 0
        assert report.total_functions == 0
        assert report.type_consistency_issues == []

    def test_is_acceptable_no_module_docstring(self) -> None:
        """Test is_acceptable returns False without module docstring."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        report = DocumentationQualityReport(
            has_module_docstring=False,
            functions_with_args=5,
            functions_with_returns=5,
            total_functions=5,
        )
        assert report.is_acceptable() is False

    def test_is_acceptable_low_coverage(self) -> None:
        """Test is_acceptable returns False with low coverage."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        report = DocumentationQualityReport(
            has_module_docstring=True,
            functions_with_args=2,
            functions_with_returns=2,
            total_functions=5,
        )
        # 40% coverage < 80% threshold
        assert report.is_acceptable() is False

    def test_is_acceptable_good_coverage(self) -> None:
        """Test is_acceptable returns True with good coverage."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        report = DocumentationQualityReport(
            has_module_docstring=True,
            functions_with_args=4,
            functions_with_returns=4,
            total_functions=5,
        )
        # 80% coverage >= 80% threshold
        assert report.is_acceptable() is True

    def test_is_acceptable_no_functions(self) -> None:
        """Test is_acceptable returns True with no functions."""
        from yolo_developer.agents.dev.doc_utils import DocumentationQualityReport

        report = DocumentationQualityReport(
            has_module_docstring=True,
            total_functions=0,
        )
        assert report.is_acceptable() is True


class TestExtractDocumentationInfo:
    """Tests for extract_documentation_info function."""

    def test_function_exists(self) -> None:
        """Test that extract_documentation_info function is defined."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        assert callable(extract_documentation_info)

    def test_detects_missing_module_docstring(self) -> None:
        """Test detection of missing module docstring."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        code = """
def hello():
    pass
"""
        info = extract_documentation_info(code)
        assert info.has_module_docstring is False

    def test_detects_present_module_docstring(self) -> None:
        """Test detection of present module docstring."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        code = '''"""Module docstring."""

def hello():
    pass
'''
        info = extract_documentation_info(code)
        assert info.has_module_docstring is True

    def test_detects_functions_missing_docstrings(self) -> None:
        """Test detection of functions without docstrings."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        code = '''"""Module."""

def func_without_doc():
    pass

def func_with_doc():
    """Has docstring."""
    pass
'''
        info = extract_documentation_info(code)
        assert "func_without_doc" in info.functions_missing_docstrings
        assert "func_with_doc" not in info.functions_missing_docstrings

    def test_counts_public_functions(self) -> None:
        """Test counting of public functions."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        code = '''"""Module."""

def public_one():
    """Doc."""
    pass

def public_two():
    pass

def _private():
    pass
'''
        info = extract_documentation_info(code)
        assert info.total_public_functions == 2
        assert info.documented_functions == 1

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.dev.doc_utils import extract_documentation_info

        code = "def broken("
        info = extract_documentation_info(code)
        # Should return empty/default info, not raise
        assert info.has_module_docstring is False
        assert info.total_public_functions == 0


class TestDetectComplexSections:
    """Tests for detect_complex_sections function."""

    def test_function_exists(self) -> None:
        """Test that detect_complex_sections function is defined."""
        from yolo_developer.agents.dev.doc_utils import detect_complex_sections

        assert callable(detect_complex_sections)

    def test_detects_nested_loops(self) -> None:
        """Test detection of nested loops."""
        from yolo_developer.agents.dev.doc_utils import detect_complex_sections

        code = '''"""Module."""

def process():
    for i in range(10):
        for j in range(10):
            print(i, j)
'''
        sections = detect_complex_sections(code)
        nested = [s for s in sections if s.complexity_type == "nested_loop"]
        assert len(nested) >= 1

    def test_detects_long_functions(self) -> None:
        """Test detection of long functions (>20 lines)."""
        from yolo_developer.agents.dev.doc_utils import detect_complex_sections

        # Create function with 25 lines
        lines = ['"""Module."""', "", "def long_function():"]
        for i in range(25):
            lines.append(f"    x{i} = {i}")
        code = "\n".join(lines)

        sections = detect_complex_sections(code)
        long_funcs = [s for s in sections if s.complexity_type == "long_function"]
        assert len(long_funcs) >= 1

    def test_detects_complex_conditionals(self) -> None:
        """Test detection of complex conditionals."""
        from yolo_developer.agents.dev.doc_utils import detect_complex_sections

        code = '''"""Module."""

def check(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return True
    return False
'''
        sections = detect_complex_sections(code)
        # Should detect nested conditionals
        complex_conds = [s for s in sections if s.complexity_type in ("complex_conditional", "deep_nesting")]
        assert len(complex_conds) >= 1

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.dev.doc_utils import detect_complex_sections

        code = "def broken("
        sections = detect_complex_sections(code)
        # Should return empty list, not raise
        assert sections == []


class TestValidateDocumentationQuality:
    """Tests for validate_documentation_quality function."""

    def test_function_exists(self) -> None:
        """Test that validate_documentation_quality function is defined."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        assert callable(validate_documentation_quality)

    def test_checks_module_docstring(self) -> None:
        """Test that module docstring presence is checked."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code_with = '''"""Module docstring."""

def func():
    pass
'''
        code_without = """
def func():
    pass
"""
        report_with = validate_documentation_quality(code_with)
        report_without = validate_documentation_quality(code_without)

        assert report_with.has_module_docstring is True
        assert report_without.has_module_docstring is False

    def test_counts_args_sections(self) -> None:
        """Test counting of Args sections in docstrings."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code = '''"""Module."""

def func_with_args(x: int) -> int:
    """Does something.

    Args:
        x: The input value.

    Returns:
        The result.
    """
    return x

def func_without_args(x: int) -> int:
    """No args section."""
    return x
'''
        report = validate_documentation_quality(code)
        assert report.functions_with_args >= 1

    def test_counts_returns_sections(self) -> None:
        """Test counting of Returns sections in docstrings."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code = '''"""Module."""

def func_with_returns() -> int:
    """Does something.

    Returns:
        The result.
    """
    return 1

def func_without_returns() -> int:
    """No returns section."""
    return 1
'''
        report = validate_documentation_quality(code)
        assert report.functions_with_returns >= 1

    def test_counts_example_sections(self) -> None:
        """Test counting of Example sections in docstrings."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code = '''"""Module."""

def func_with_example() -> int:
    """Does something.

    Example:
        >>> func_with_example()
        1
    """
    return 1
'''
        report = validate_documentation_quality(code)
        assert report.functions_with_examples >= 1

    def test_generates_warnings(self) -> None:
        """Test that warnings are generated for issues."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code = """
def func_no_doc():
    pass
"""
        report = validate_documentation_quality(code)
        # Should have warning about missing module docstring
        assert len(report.warnings) >= 1

    def test_handles_syntax_error(self) -> None:
        """Test graceful handling of syntax errors."""
        from yolo_developer.agents.dev.doc_utils import validate_documentation_quality

        code = "def broken("
        report = validate_documentation_quality(code)
        # Should return report with warning, not raise
        assert len(report.warnings) >= 1


class TestGenerateDocumentationWithLLM:
    """Tests for generate_documentation_with_llm function."""

    def test_function_exists(self) -> None:
        """Test that generate_documentation_with_llm function is defined."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        assert callable(generate_documentation_with_llm)

    @pytest.mark.asyncio
    async def test_calls_llm_router(self) -> None:
        """Test that function calls LLM router."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value='''```python
"""Module docstring."""

def hello():
    """Says hello."""
    pass
```'''
        )

        code = "def hello(): pass"
        result, is_valid = await generate_documentation_with_llm(
            code=code,
            context="Test module",
            router=mock_router,
        )

        mock_router.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_enhanced_code(self) -> None:
        """Test that function returns enhanced code."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value='''```python
"""Module docstring."""

def hello():
    """Says hello."""
    pass
```'''
        )

        code = "def hello(): pass"
        result, is_valid = await generate_documentation_with_llm(
            code=code,
            context="Test module",
            router=mock_router,
        )

        assert "Module docstring" in result
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validates_syntax(self) -> None:
        """Test that syntax is validated."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value='''```python
def broken(
```'''
        )

        code = "def hello(): pass"
        result, is_valid = await generate_documentation_with_llm(
            code=code,
            context="Test module",
            router=mock_router,
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_uses_complex_tier(self) -> None:
        """Test that complex tier is used per ADR-003."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        mock_router = MagicMock()
        mock_router.call = AsyncMock(
            return_value='''```python
"""Module."""

def hello():
    """Hello."""
    pass
```'''
        )

        await generate_documentation_with_llm(
            code="def hello(): pass",
            context="Test",
            router=mock_router,
        )

        # Verify tier was passed
        call_kwargs = mock_router.call.call_args
        assert call_kwargs is not None
        # Tier should be "complex"
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("tier") == "complex"

    @pytest.mark.asyncio
    async def test_returns_original_code_on_llm_error(self) -> None:
        """Test that original code is returned when LLM call fails."""
        from yolo_developer.agents.dev.doc_utils import generate_documentation_with_llm

        mock_router = MagicMock()
        mock_router.call = AsyncMock(side_effect=Exception("LLM unavailable"))

        original_code = "def hello(): pass"
        result, is_valid = await generate_documentation_with_llm(
            code=original_code,
            context="Test",
            router=mock_router,
            max_retries=0,  # No retries to speed up test
        )

        # Should return original code with is_valid=False
        assert result == original_code
        assert is_valid is False


class TestFormatDocumentationInfoForPrompt:
    """Tests for format_documentation_info_for_prompt function."""

    def test_function_exists(self) -> None:
        """Test that format_documentation_info_for_prompt function is defined."""
        from yolo_developer.agents.dev.doc_utils import format_documentation_info_for_prompt

        assert callable(format_documentation_info_for_prompt)

    def test_formats_info(self) -> None:
        """Test that function formats DocumentationInfo for prompt."""
        from yolo_developer.agents.dev.doc_utils import (
            DocumentationInfo,
            format_documentation_info_for_prompt,
        )

        info = DocumentationInfo(
            has_module_docstring=False,
            functions_missing_docstrings=("func1", "func2"),
            functions_with_incomplete_docstrings=("func3",),
            complex_sections=(),
            total_public_functions=3,
            documented_functions=1,
        )

        formatted = format_documentation_info_for_prompt(info)

        assert "module docstring" in formatted.lower()
        assert "func1" in formatted
        assert "func2" in formatted


class TestFormatComplexSectionsForPrompt:
    """Tests for format_complex_sections_for_prompt function."""

    def test_function_exists(self) -> None:
        """Test that format_complex_sections_for_prompt function is defined."""
        from yolo_developer.agents.dev.doc_utils import format_complex_sections_for_prompt

        assert callable(format_complex_sections_for_prompt)

    def test_formats_empty_list(self) -> None:
        """Test formatting of empty sections list."""
        from yolo_developer.agents.dev.doc_utils import format_complex_sections_for_prompt

        formatted = format_complex_sections_for_prompt([])
        assert "none" in formatted.lower() or "no complex" in formatted.lower()

    def test_formats_sections(self) -> None:
        """Test formatting of complex sections."""
        from yolo_developer.agents.dev.doc_utils import (
            ComplexSection,
            format_complex_sections_for_prompt,
        )

        sections = [
            ComplexSection(
                start_line=10,
                end_line=25,
                complexity_type="nested_loop",
                function_name="process",
                description="Nested for loops",
            ),
        ]

        formatted = format_complex_sections_for_prompt(sections)

        assert "10" in formatted
        assert "25" in formatted
        assert "nested" in formatted.lower()
