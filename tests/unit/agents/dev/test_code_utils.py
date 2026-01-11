"""Unit tests for Dev agent code utilities (Story 8.2 - Tasks 3, 5, 8, 9).

Tests syntax validation, code extraction, and maintainability checking.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yolo_developer.agents.dev.code_utils import (
    MaintainabilityReport,
    MaintainabilityWarning,
    check_maintainability,
    extract_code_from_response,
    validate_python_syntax,
)


class TestValidatePythonSyntax:
    """Tests for validate_python_syntax function (Task 3)."""

    def test_valid_simple_code(self) -> None:
        """Test that valid simple code passes validation."""
        code = "def hello(): pass"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_complex_code(self) -> None:
        """Test that valid complex code passes validation."""
        code = """
from __future__ import annotations

from typing import Any

def process_data(items: list[Any]) -> dict[str, int]:
    \"\"\"Process items and return counts.\"\"\"
    result: dict[str, int] = {}
    for item in items:
        key = str(item)
        result[key] = result.get(key, 0) + 1
    return result
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_async_code(self) -> None:
        """Test that valid async code passes validation."""
        code = """
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_invalid_syntax_error(self) -> None:
        """Test that syntax errors are detected."""
        code = "def broken("
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None
        assert "Line" in error

    def test_invalid_indentation_error(self) -> None:
        """Test that indentation errors are detected."""
        code = """
def hello():
pass  # Wrong indentation
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_empty_code(self) -> None:
        """Test that empty code is handled."""
        is_valid, error = validate_python_syntax("")
        assert is_valid is False
        assert error is not None
        assert "Empty" in error

    def test_whitespace_only_code(self) -> None:
        """Test that whitespace-only code is handled."""
        is_valid, error = validate_python_syntax("   \n\t\n  ")
        assert is_valid is False
        assert error is not None

    def test_non_python_content(self) -> None:
        """Test that non-Python content fails validation."""
        code = "This is not Python code at all"
        # This actually parses as a valid Python expression (bare name)
        # So we need a more complex example
        code = "function hello() { return 'world'; }"  # JavaScript
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None


class TestExtractCodeFromResponse:
    """Tests for extract_code_from_response function."""

    def test_extract_from_python_block(self) -> None:
        """Test extraction from ```python block."""
        response = """Here's the code:

```python
def hello():
    return "world"
```

This function returns a greeting."""
        code = extract_code_from_response(response)
        assert code == 'def hello():\n    return "world"'

    def test_extract_from_generic_block(self) -> None:
        """Test extraction from generic ``` block."""
        response = """Here's the code:

```
def hello():
    return "world"
```
"""
        code = extract_code_from_response(response)
        assert code == 'def hello():\n    return "world"'

    def test_extract_raw_code(self) -> None:
        """Test extraction when no blocks present."""
        response = """def hello():
    return "world"
"""
        code = extract_code_from_response(response)
        assert "def hello()" in code

    def test_extract_largest_block(self) -> None:
        """Test that largest code block is extracted."""
        response = """First block:
```python
x = 1
```

Second, larger block:
```python
def hello():
    return "world"

def goodbye():
    return "farewell"
```
"""
        code = extract_code_from_response(response)
        # Should extract the larger second block
        assert "def hello()" in code
        assert "def goodbye()" in code

    def test_empty_response(self) -> None:
        """Test that empty response returns empty string."""
        code = extract_code_from_response("")
        assert code == ""

    def test_none_response(self) -> None:
        """Test that None-like response is handled."""
        code = extract_code_from_response("")
        assert code == ""


class TestMaintainabilityWarning:
    """Tests for MaintainabilityWarning dataclass."""

    def test_create_warning(self) -> None:
        """Test creating a maintainability warning."""
        warning = MaintainabilityWarning(
            category="function_length",
            message="Function too long",
            line=10,
            severity="warning",
        )
        assert warning.category == "function_length"
        assert warning.message == "Function too long"
        assert warning.line == 10
        assert warning.severity == "warning"

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        warning = MaintainabilityWarning(
            category="test",
            message="test message",
        )
        assert warning.line is None
        assert warning.severity == "warning"

    def test_frozen_dataclass(self) -> None:
        """Test that warning is immutable."""
        from dataclasses import FrozenInstanceError

        warning = MaintainabilityWarning(category="test", message="test")
        with pytest.raises(FrozenInstanceError):
            warning.category = "modified"  # type: ignore


class TestMaintainabilityReport:
    """Tests for MaintainabilityReport dataclass."""

    def test_empty_report(self) -> None:
        """Test creating empty report."""
        report = MaintainabilityReport()
        assert report.warnings == []
        assert report.function_count == 0
        assert report.max_function_length == 0
        assert report.max_nesting_depth == 0
        assert report.has_warnings() is False

    def test_report_with_warnings(self) -> None:
        """Test report with warnings."""
        warning = MaintainabilityWarning(category="test", message="test")
        report = MaintainabilityReport(warnings=[warning])
        assert report.has_warnings() is True
        assert len(report.warnings) == 1

    def test_get_warnings_by_category(self) -> None:
        """Test filtering warnings by category."""
        warnings = [
            MaintainabilityWarning(category="function_length", message="too long"),
            MaintainabilityWarning(category="naming", message="bad name"),
            MaintainabilityWarning(category="function_length", message="also long"),
        ]
        report = MaintainabilityReport(warnings=warnings)

        length_warnings = report.get_warnings_by_category("function_length")
        assert len(length_warnings) == 2

        naming_warnings = report.get_warnings_by_category("naming")
        assert len(naming_warnings) == 1


class TestCheckMaintainability:
    """Tests for check_maintainability function (Task 5)."""

    def test_empty_code(self) -> None:
        """Test checking empty code."""
        report = check_maintainability("")
        assert report.function_count == 0
        assert report.has_warnings() is False

    def test_valid_simple_function(self) -> None:
        """Test checking simple valid function."""
        code = """
def hello():
    return "world"
"""
        report = check_maintainability(code)
        assert report.function_count == 1
        assert report.max_function_length > 0
        # No warnings for simple function
        length_warnings = report.get_warnings_by_category("function_length")
        assert len(length_warnings) == 0

    def test_long_function_warning(self) -> None:
        """Test that long functions generate warnings (AC1, AC3)."""
        # Generate a function with more than 50 lines
        lines = ["def long_function():"]
        for i in range(55):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x0")
        code = "\n".join(lines)

        report = check_maintainability(code)
        assert report.max_function_length > 50

        length_warnings = report.get_warnings_by_category("function_length")
        assert len(length_warnings) >= 1
        assert "long_function" in length_warnings[0].message

    def test_deep_nesting_warning(self) -> None:
        """Test that deep nesting generates warnings (AC3)."""
        code = """
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    return "too deep"
"""
        report = check_maintainability(code)
        assert report.max_nesting_depth > 3

        nesting_warnings = report.get_warnings_by_category("nesting_depth")
        assert len(nesting_warnings) >= 1
        assert "nesting depth" in nesting_warnings[0].message.lower()

    def test_acceptable_nesting(self) -> None:
        """Test that acceptable nesting doesn't warn."""
        code = """
def acceptable_nesting():
    if True:
        for i in range(10):
            if i > 5:
                return i
"""
        report = check_maintainability(code)
        assert report.max_nesting_depth <= 3

        nesting_warnings = report.get_warnings_by_category("nesting_depth")
        assert len(nesting_warnings) == 0

    def test_snake_case_function_names(self) -> None:
        """Test that snake_case function names are accepted (AC2)."""
        code = """
def process_user_data():
    pass

def _private_helper():
    pass

async def fetch_from_api():
    pass
"""
        report = check_maintainability(code)
        naming_warnings = report.get_warnings_by_category("naming")
        # Should have no warnings for snake_case names
        function_naming_warnings = [
            w for w in naming_warnings if "Function" in w.message
        ]
        assert len(function_naming_warnings) == 0

    def test_non_snake_case_warning(self) -> None:
        """Test that non-snake_case generates warning (AC2)."""
        code = """
def processUserData():
    pass
"""
        report = check_maintainability(code)
        naming_warnings = report.get_warnings_by_category("naming")
        function_warnings = [w for w in naming_warnings if "snake_case" in w.message]
        assert len(function_warnings) >= 1

    def test_single_letter_variable_warning(self) -> None:
        """Test that single-letter variables generate info (AC2)."""
        code = """
def example():
    a = 1  # Bad - single letter
    b = 2  # Bad - single letter
    return a + b
"""
        report = check_maintainability(code)
        naming_warnings = report.get_warnings_by_category("naming")
        # 'a' and 'b' should generate warnings
        single_letter_warnings = [
            w for w in naming_warnings if "Single-letter" in w.message
        ]
        assert len(single_letter_warnings) >= 2

    def test_allowed_single_letters(self) -> None:
        """Test that common iterators are allowed (AC2)."""
        code = """
def example():
    for i in range(10):
        for j in range(10):
            x = i + j
    return x
"""
        report = check_maintainability(code)
        naming_warnings = report.get_warnings_by_category("naming")
        # i, j, x should be allowed
        single_letter_warnings = [
            w for w in naming_warnings if "Single-letter" in w.message
        ]
        # Should have no warnings for i, j, x
        assert len(single_letter_warnings) == 0

    def test_invalid_syntax_returns_empty_report(self) -> None:
        """Test that invalid code returns empty report."""
        code = "def broken("
        report = check_maintainability(code)
        assert report.function_count == 0
        assert report.has_warnings() is False

    def test_async_function_analysis(self) -> None:
        """Test that async functions are analyzed."""
        code = """
async def async_handler():
    await something()
    return "done"
"""
        report = check_maintainability(code)
        assert report.function_count == 1

    def test_combined_analysis(self) -> None:
        """Test combined analysis of multiple issues."""
        # Function that's both long and deeply nested with bad names
        lines = ["def badFunctionName():"]
        lines.append("    if True:")
        lines.append("        if True:")
        lines.append("            if True:")
        lines.append("                if True:")
        for i in range(50):
            lines.append(f"                    a{i} = {i}")
        lines.append("                    return a0")
        code = "\n".join(lines)

        report = check_maintainability(code)
        assert report.has_warnings() is True
        # Should have function_length warning
        assert len(report.get_warnings_by_category("function_length")) >= 1
        # Should have nesting_depth warning
        assert len(report.get_warnings_by_category("nesting_depth")) >= 1


class TestCodeUtilsIntegration:
    """Integration tests for code utilities."""

    def test_full_workflow_valid_code(self) -> None:
        """Test full workflow with valid generated code."""
        response = """Here's the implementation:

```python
from __future__ import annotations


def calculate_average(numbers: list[float]) -> float:
    \"\"\"Calculate the average of a list of numbers.

    Args:
        numbers: List of numbers to average.

    Returns:
        Average value, or 0.0 for empty list.
    \"\"\"
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
```

This function handles empty lists gracefully.
"""
        # Extract code
        code = extract_code_from_response(response)
        assert "def calculate_average" in code

        # Validate syntax
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

        # Check maintainability
        report = check_maintainability(code)
        assert report.function_count == 1
        # Should have no serious warnings
        length_warnings = report.get_warnings_by_category("function_length")
        assert len(length_warnings) == 0

    def test_full_workflow_invalid_code(self) -> None:
        """Test full workflow with invalid generated code."""
        response = """```python
def broken_function(
    # Missing closing paren and colon
```"""
        # Extract code
        code = extract_code_from_response(response)

        # Validate syntax - should fail
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None


class TestCyclomaticComplexity:
    """Tests for cyclomatic complexity checking (AC1)."""

    def test_simple_function_low_complexity(self) -> None:
        """Test that simple function has low complexity."""
        code = """
def simple():
    return 1
"""
        report = check_maintainability(code)
        assert report.max_cyclomatic_complexity == 1
        # No warnings for low complexity
        complexity_warnings = report.get_warnings_by_category("cyclomatic_complexity")
        assert len(complexity_warnings) == 0

    def test_if_statements_increase_complexity(self) -> None:
        """Test that if statements increase complexity."""
        code = """
def with_ifs(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        report = check_maintainability(code)
        # Base 1 + 2 ifs (elif counts as if)
        assert report.max_cyclomatic_complexity >= 3

    def test_loops_increase_complexity(self) -> None:
        """Test that loops increase complexity."""
        code = """
def with_loops(items):
    result = 0
    for item in items:
        while item > 0:
            result += item
            item -= 1
    return result
"""
        report = check_maintainability(code)
        # Base 1 + 1 for + 1 while = 3
        assert report.max_cyclomatic_complexity >= 3

    def test_boolean_operators_increase_complexity(self) -> None:
        """Test that and/or operators increase complexity."""
        code = """
def with_boolean(a, b, c):
    if a and b and c:
        return True
    if a or b or c:
        return False
    return None
"""
        report = check_maintainability(code)
        # Boolean ops add (num_operands - 1) each
        # First if: 1 + (3-1 for and) = 3
        # Second if: 1 + (3-1 for or) = 3
        # Total: 1 base + 2 ifs + 2 + 2 = 7
        assert report.max_cyclomatic_complexity >= 5

    def test_high_complexity_warning(self) -> None:
        """Test that high complexity generates warning."""
        # Function with complexity > 10
        code = """
def complex_function(a, b, c, d, e, f, g, h, i, j, k):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    if i:
                                        if j:
                                            if k:
                                                return True
    return False
"""
        report = check_maintainability(code)
        assert report.max_cyclomatic_complexity > 10
        complexity_warnings = report.get_warnings_by_category("cyclomatic_complexity")
        assert len(complexity_warnings) >= 1
        assert "complexity" in complexity_warnings[0].message.lower()

    def test_comprehension_conditions(self) -> None:
        """Test that comprehension conditions are counted."""
        code = """
def with_comprehension(items):
    return [x for x in items if x > 0 if x < 100]
"""
        report = check_maintainability(code)
        # Base 1 + 2 if conditions in comprehension
        assert report.max_cyclomatic_complexity >= 3

    def test_try_except_complexity(self) -> None:
        """Test that except clauses increase complexity."""
        code = """
def with_try(x):
    try:
        return int(x)
    except ValueError:
        return 0
    except TypeError:
        return -1
"""
        report = check_maintainability(code)
        # Base 1 + 2 except handlers
        assert report.max_cyclomatic_complexity >= 3

    def test_acceptable_complexity(self) -> None:
        """Test that complexity <= 10 does not warn."""
        code = """
def acceptable(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
"""
        report = check_maintainability(code)
        # Should be around 5-7
        assert report.max_cyclomatic_complexity <= 10
        complexity_warnings = report.get_warnings_by_category("cyclomatic_complexity")
        assert len(complexity_warnings) == 0


class TestGenerateCodeWithValidation:
    """Tests for generate_code_with_validation function."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create mock LLM router."""
        from unittest.mock import AsyncMock, MagicMock

        router = MagicMock()
        router.call = AsyncMock()
        return router

    @pytest.mark.asyncio
    async def test_returns_valid_code_on_success(self, mock_router: MagicMock) -> None:
        """Test that valid code is returned on first success."""
        from yolo_developer.agents.dev.code_utils import generate_code_with_validation

        mock_router.call.return_value = """```python
def hello():
    return "world"
```"""

        code, is_valid = await generate_code_with_validation(
            router=mock_router,
            prompt="Generate hello function",
        )

        assert is_valid is True
        assert "def hello()" in code
        assert mock_router.call.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_syntax_error(self, mock_router: MagicMock) -> None:
        """Test that retry happens on syntax error."""
        from yolo_developer.agents.dev.code_utils import generate_code_with_validation

        # First call returns invalid, second returns valid
        mock_router.call.side_effect = [
            "```python\ndef broken(\n```",  # Invalid
            "```python\ndef fixed(): pass\n```",  # Valid
        ]

        code, is_valid = await generate_code_with_validation(
            router=mock_router,
            prompt="Generate function",
            max_retries=2,
        )

        assert is_valid is True
        assert "def fixed()" in code
        assert mock_router.call.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_invalid_after_max_retries(self, mock_router: MagicMock) -> None:
        """Test that invalid code is returned after max retries."""
        from yolo_developer.agents.dev.code_utils import generate_code_with_validation

        # All calls return invalid code
        mock_router.call.return_value = "```python\ndef broken(\n```"

        _code, is_valid = await generate_code_with_validation(
            router=mock_router,
            prompt="Generate function",
            max_retries=2,
        )

        assert is_valid is False
        # Original + 2 retries = 3 calls
        assert mock_router.call.call_count == 3

    @pytest.mark.asyncio
    async def test_uses_correct_tier(self, mock_router: MagicMock) -> None:
        """Test that specified tier is passed to router."""
        from yolo_developer.agents.dev.code_utils import generate_code_with_validation

        mock_router.call.return_value = "```python\ndef test(): pass\n```"

        await generate_code_with_validation(
            router=mock_router,
            prompt="Generate function",
            tier="critical",
        )

        call_kwargs = mock_router.call.call_args
        assert call_kwargs.kwargs["tier"] == "critical"
