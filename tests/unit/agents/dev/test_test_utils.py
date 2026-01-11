"""Unit tests for test analysis utilities (Story 8.3 - Task 9, 10).

Tests function extraction, edge case identification, and LLM test generation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from yolo_developer.agents.dev.test_utils import (
    FunctionInfo,
    QualityReport,
    calculate_coverage_estimate,
    check_coverage_threshold,
    extract_public_functions,
    generate_unit_tests_with_llm,
    identify_edge_cases,
    validate_test_quality,
)


class TestFunctionInfo:
    """Tests for the FunctionInfo dataclass."""

    def test_function_info_is_frozen(self) -> None:
        """Test that FunctionInfo is immutable."""
        info = FunctionInfo(
            name="test_func",
            signature="def test_func(x: int) -> int",
            docstring="Test function.",
            parameters=("x",),
            return_type="int",
        )
        with pytest.raises(AttributeError):
            info.name = "new_name"  # type: ignore[misc]

    def test_function_info_stores_all_attributes(self) -> None:
        """Test that FunctionInfo stores all provided attributes."""
        info = FunctionInfo(
            name="calculate",
            signature="def calculate(a: float, b: float) -> float",
            docstring="Calculate something.",
            parameters=("a", "b"),
            return_type="float",
        )
        assert info.name == "calculate"
        assert "calculate" in info.signature
        assert info.docstring == "Calculate something."
        assert info.parameters == ("a", "b")
        assert info.return_type == "float"

    def test_function_info_allows_none_docstring(self) -> None:
        """Test that FunctionInfo allows None docstring."""
        info = FunctionInfo(
            name="func",
            signature="def func()",
            docstring=None,
            parameters=(),
            return_type=None,
        )
        assert info.docstring is None

    def test_function_info_allows_none_return_type(self) -> None:
        """Test that FunctionInfo allows None return type."""
        info = FunctionInfo(
            name="func",
            signature="def func()",
            docstring=None,
            parameters=(),
            return_type=None,
        )
        assert info.return_type is None


class TestExtractPublicFunctions:
    """Tests for extract_public_functions function (AC1)."""

    def test_extracts_simple_function(self) -> None:
        """Test extraction of a simple function."""
        code = """
def hello():
    pass
"""
        result = extract_public_functions(code)
        assert len(result) == 1
        assert result[0].name == "hello"

    def test_extracts_function_with_parameters(self) -> None:
        """Test extraction of function with parameters."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        result = extract_public_functions(code)
        assert len(result) == 1
        assert result[0].name == "add"
        assert "a" in result[0].parameters
        assert "b" in result[0].parameters
        assert result[0].return_type == "int"

    def test_extracts_function_with_docstring(self) -> None:
        """Test extraction of function docstring."""
        code = '''
def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
'''
        result = extract_public_functions(code)
        assert len(result) == 1
        assert result[0].docstring is not None
        assert "Greet a person" in result[0].docstring

    def test_excludes_private_functions(self) -> None:
        """Test that private functions (starting with _) are excluded."""
        code = """
def public_func():
    pass

def _private_func():
    pass

def __dunder_func__():
    pass
"""
        result = extract_public_functions(code)
        names = [f.name for f in result]
        assert "public_func" in names
        assert "_private_func" not in names
        assert "__dunder_func__" not in names

    def test_extracts_async_functions(self) -> None:
        """Test extraction of async functions."""
        code = """
async def fetch_data(url: str) -> dict:
    pass
"""
        result = extract_public_functions(code)
        assert len(result) == 1
        assert result[0].name == "fetch_data"
        assert "url" in result[0].parameters

    def test_extracts_multiple_functions(self) -> None:
        """Test extraction of multiple functions."""
        code = """
def func1():
    pass

def func2(x):
    pass

def func3(a, b, c):
    pass
"""
        result = extract_public_functions(code)
        assert len(result) == 3
        names = [f.name for f in result]
        assert "func1" in names
        assert "func2" in names
        assert "func3" in names

    def test_handles_syntax_error_gracefully(self) -> None:
        """Test that syntax errors return empty list."""
        code = "def broken("
        result = extract_public_functions(code)
        assert result == []

    def test_handles_empty_code(self) -> None:
        """Test that empty code returns empty list."""
        result = extract_public_functions("")
        assert result == []

    def test_handles_code_with_no_functions(self) -> None:
        """Test code with only variables/classes returns empty list."""
        code = """
x = 1
y = 2

class MyClass:
    pass
"""
        result = extract_public_functions(code)
        assert result == []

    def test_extracts_complex_type_hints(self) -> None:
        """Test extraction with complex type hints."""
        code = """
def process(items: list[dict[str, Any]], callback: Callable[[int], None]) -> tuple[bool, str]:
    pass
"""
        result = extract_public_functions(code)
        assert len(result) == 1
        assert "items" in result[0].parameters
        assert "callback" in result[0].parameters

    def test_signature_includes_full_definition(self) -> None:
        """Test that signature includes the full function definition."""
        code = """
def validate(data: str, strict: bool = False) -> bool:
    pass
"""
        result = extract_public_functions(code)
        assert len(result) == 1
        assert "validate" in result[0].signature
        assert "data" in result[0].signature
        assert "strict" in result[0].signature


class TestIdentifyEdgeCases:
    """Tests for identify_edge_cases function (AC2)."""

    def test_identifies_string_parameter_edge_cases(self) -> None:
        """Test edge case identification for string parameters."""
        func_info = FunctionInfo(
            name="process_text",
            signature="def process_text(text: str) -> str",
            docstring=None,
            parameters=("text",),
            return_type="str",
        )
        edge_cases = identify_edge_cases(func_info)
        assert any("empty" in ec.lower() for ec in edge_cases)

    def test_identifies_list_parameter_edge_cases(self) -> None:
        """Test edge case identification for list parameters."""
        func_info = FunctionInfo(
            name="sum_items",
            signature="def sum_items(items: list[int]) -> int",
            docstring=None,
            parameters=("items",),
            return_type="int",
        )
        edge_cases = identify_edge_cases(func_info)
        assert any("empty" in ec.lower() for ec in edge_cases)

    def test_identifies_optional_parameter_edge_cases(self) -> None:
        """Test edge case identification for Optional parameters."""
        func_info = FunctionInfo(
            name="greet",
            signature="def greet(name: str | None) -> str",
            docstring=None,
            parameters=("name",),
            return_type="str",
        )
        edge_cases = identify_edge_cases(func_info)
        assert any("none" in ec.lower() for ec in edge_cases)

    def test_identifies_numeric_parameter_edge_cases(self) -> None:
        """Test edge case identification for numeric parameters."""
        func_info = FunctionInfo(
            name="divide",
            signature="def divide(a: int, b: int) -> float",
            docstring=None,
            parameters=("a", "b"),
            return_type="float",
        )
        edge_cases = identify_edge_cases(func_info)
        # Should suggest boundary cases like zero
        assert any("zero" in ec.lower() or "0" in ec for ec in edge_cases)

    def test_identifies_edge_cases_from_docstring(self) -> None:
        """Test that docstring hints influence edge case identification."""
        func_info = FunctionInfo(
            name="validate",
            signature="def validate(data: str) -> bool",
            docstring="Validate input. Raises ValueError for invalid data.",
            parameters=("data",),
            return_type="bool",
        )
        edge_cases = identify_edge_cases(func_info)
        assert any("invalid" in ec.lower() or "error" in ec.lower() for ec in edge_cases)

    def test_returns_list_of_strings(self) -> None:
        """Test that edge cases are returned as list of strings."""
        func_info = FunctionInfo(
            name="func",
            signature="def func(x: int) -> int",
            docstring=None,
            parameters=("x",),
            return_type="int",
        )
        edge_cases = identify_edge_cases(func_info)
        assert isinstance(edge_cases, list)
        assert all(isinstance(ec, str) for ec in edge_cases)

    def test_handles_function_with_no_parameters(self) -> None:
        """Test edge case identification for parameterless function."""
        func_info = FunctionInfo(
            name="get_timestamp",
            signature="def get_timestamp() -> float",
            docstring=None,
            parameters=(),
            return_type="float",
        )
        edge_cases = identify_edge_cases(func_info)
        # Should return at least something (basic test suggestions)
        assert isinstance(edge_cases, list)

    def test_identifies_dict_parameter_edge_cases(self) -> None:
        """Test edge case identification for dict parameters."""
        func_info = FunctionInfo(
            name="merge_configs",
            signature="def merge_configs(config: dict[str, Any]) -> dict",
            docstring=None,
            parameters=("config",),
            return_type="dict",
        )
        edge_cases = identify_edge_cases(func_info)
        assert any("empty" in ec.lower() for ec in edge_cases)


class TestGenerateUnitTestsWithLLM:
    """Tests for generate_unit_tests_with_llm function (AC5, AC6)."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create mock LLM router."""
        router = MagicMock()
        router.call = AsyncMock()
        return router

    @pytest.fixture
    def sample_code(self) -> str:
        """Sample implementation code to test."""
        return '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y
'''

    @pytest.fixture
    def sample_functions(self) -> list[FunctionInfo]:
        """Sample function info list."""
        return [
            FunctionInfo(
                name="add",
                signature="def add(a: int, b: int) -> int",
                docstring="Add two numbers.",
                parameters=("a", "b"),
                return_type="int",
            ),
            FunctionInfo(
                name="multiply",
                signature="def multiply(x: int, y: int) -> int",
                docstring="Multiply two numbers.",
                parameters=("x", "y"),
                return_type="int",
            ),
        ]

    @pytest.mark.asyncio
    async def test_returns_tuple_with_code_and_validity(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that function returns tuple of (code, is_valid)."""
        mock_router.call.return_value = '''```python
def test_add():
    assert add(1, 2) == 3
```'''
        result = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        code, is_valid = result
        assert isinstance(code, str)
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_calls_router_with_complex_tier(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that router is called with 'complex' tier per ADR-003."""
        mock_router.call.return_value = "```python\ndef test_x(): pass\n```"
        await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        # Verify call was made with complex tier
        mock_router.call.assert_called()
        call_kwargs = mock_router.call.call_args.kwargs
        assert call_kwargs.get("tier") == "complex"

    @pytest.mark.asyncio
    async def test_validates_generated_test_syntax(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that generated tests are validated for syntax."""
        # Return valid Python test code
        mock_router.call.return_value = '''```python
def test_add_positive_numbers():
    """Test adding positive numbers."""
    assert add(1, 2) == 3
```'''
        code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        assert is_valid is True
        assert "test_add" in code

    @pytest.mark.asyncio
    async def test_returns_invalid_for_syntax_error(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that syntax errors return is_valid=False."""
        # Return invalid Python code
        mock_router.call.return_value = "```python\ndef test_broken(\n```"
        _code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_retries_on_syntax_error(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that syntax errors trigger retry with corrected prompt."""
        # First call returns invalid, second returns valid
        mock_router.call.side_effect = [
            "```python\ndef test_broken(\n```",
            "```python\ndef test_fixed(): pass\n```",
        ]
        _code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        # Should have retried
        assert mock_router.call.call_count >= 2
        # Final result should be valid
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_includes_implementation_code_in_prompt(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that prompt includes the implementation code."""
        mock_router.call.return_value = "```python\ndef test_x(): pass\n```"
        await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        # Check the prompt content
        call_kwargs = mock_router.call.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        prompt_content = messages[0]["content"] if messages else ""
        assert "add" in prompt_content
        assert "multiply" in prompt_content

    @pytest.mark.asyncio
    async def test_handles_empty_function_list(
        self,
        mock_router: MagicMock,
        sample_code: str,
    ) -> None:
        """Test handling of empty function list."""
        mock_router.call.return_value = "```python\n# No tests\n```"
        code, _is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=[],
            module_name="empty_module",
            router=mock_router,
        )
        assert isinstance(code, str)

    @pytest.mark.asyncio
    async def test_handles_router_exception(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that router exceptions return empty code and is_valid=False."""
        mock_router.call.side_effect = Exception("API Error")
        code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        assert code == ""
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_extracts_code_from_markdown_blocks(
        self,
        mock_router: MagicMock,
        sample_code: str,
        sample_functions: list[FunctionInfo],
    ) -> None:
        """Test that code is properly extracted from markdown blocks."""
        mock_router.call.return_value = '''Here are the tests:

```python
def test_add():
    assert add(1, 2) == 3

def test_multiply():
    assert multiply(2, 3) == 6
```

These tests cover the basic functionality.'''
        code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=sample_code,
            functions=sample_functions,
            module_name="math_utils",
            router=mock_router,
        )
        assert is_valid is True
        assert "test_add" in code
        assert "test_multiply" in code
        # Should not include the markdown explanation text
        assert "Here are the tests" not in code


class TestCalculateCoverageEstimate:
    """Tests for calculate_coverage_estimate function (AC4)."""

    def test_returns_float_between_0_and_1(self) -> None:
        """Test that coverage estimate is between 0 and 1."""
        code = '''
def add(a, b):
    return a + b
'''
        tests = '''
def test_add():
    assert add(1, 2) == 3
'''
        coverage = calculate_coverage_estimate(code, tests)
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_higher_coverage_for_more_tests(self) -> None:
        """Test that more tests result in higher coverage estimate."""
        code = '''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
'''
        few_tests = '''
def test_add():
    assert add(1, 2) == 3
'''
        many_tests = '''
def test_add():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, -2) == -3

def test_subtract():
    assert subtract(5, 3) == 2

def test_subtract_negative():
    assert subtract(-5, -3) == -2
'''
        coverage_few = calculate_coverage_estimate(code, few_tests)
        coverage_many = calculate_coverage_estimate(code, many_tests)
        assert coverage_many >= coverage_few

    def test_empty_tests_return_zero(self) -> None:
        """Test that empty tests return zero coverage."""
        code = '''
def add(a, b):
    return a + b
'''
        coverage = calculate_coverage_estimate(code, "")
        assert coverage == 0.0

    def test_empty_code_return_one(self) -> None:
        """Test that empty code with tests returns 1.0 (nothing to cover)."""
        coverage = calculate_coverage_estimate("", "def test_x(): pass")
        assert coverage == 1.0

    def test_handles_syntax_error_in_code(self) -> None:
        """Test graceful handling of syntax errors in code."""
        code = "def broken("
        tests = "def test_x(): pass"
        coverage = calculate_coverage_estimate(code, tests)
        # Should return 0 for unparseable code
        assert coverage == 0.0

    def test_counts_functions_tested(self) -> None:
        """Test that coverage considers functions that have tests."""
        code = '''
def func1():
    pass

def func2():
    pass

def func3():
    pass
'''
        tests = '''
def test_func1():
    pass

def test_func2():
    pass
'''
        coverage = calculate_coverage_estimate(code, tests)
        # 2 out of 3 functions tested -> ~66%
        assert 0.5 <= coverage <= 0.8

    def test_detects_method_calls_on_objects(self) -> None:
        """Test that coverage detects method calls via obj.method()."""
        code = '''
def calculate(value):
    return value * 2

def process(data):
    return data.upper()
'''
        tests = '''
def test_calculate_via_method():
    calc = Calculator()
    result = calc.calculate(5)  # Method call
    assert result == 10

def test_process_via_method():
    proc = Processor()
    result = proc.process("hello")  # Method call
    assert result == "HELLO"
'''
        coverage = calculate_coverage_estimate(code, tests)
        # Both functions should be detected via method call pattern
        assert coverage >= 0.7

    def test_coverage_accuracy_for_full_coverage(self) -> None:
        """Test that full test coverage produces high coverage estimate."""
        code = '''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
'''
        tests = '''
def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -2) == -3

def test_subtract_positive():
    assert subtract(5, 3) == 2

def test_subtract_negative():
    assert subtract(-5, -3) == -2

def test_multiply_positive():
    assert multiply(2, 3) == 6

def test_multiply_by_zero():
    assert multiply(5, 0) == 0
'''
        coverage = calculate_coverage_estimate(code, tests)
        # All 3 functions tested with multiple tests -> should be high
        assert coverage >= 0.9


class TestCheckCoverageThreshold:
    """Tests for check_coverage_threshold function (AC4)."""

    def test_passes_when_above_threshold(self) -> None:
        """Test that coverage above threshold passes."""
        passes, message = check_coverage_threshold(
            coverage=0.85,
            threshold=0.80,
        )
        assert passes is True
        assert message == ""

    def test_passes_when_equal_to_threshold(self) -> None:
        """Test that coverage equal to threshold passes."""
        passes, _message = check_coverage_threshold(
            coverage=0.80,
            threshold=0.80,
        )
        assert passes is True

    def test_fails_when_below_threshold(self) -> None:
        """Test that coverage below threshold fails with warning."""
        passes, message = check_coverage_threshold(
            coverage=0.60,
            threshold=0.80,
        )
        assert passes is False
        assert "60" in message or "0.6" in message
        assert "80" in message or "0.8" in message

    def test_returns_meaningful_warning_message(self) -> None:
        """Test that warning message includes useful information."""
        passes, message = check_coverage_threshold(
            coverage=0.45,
            threshold=0.80,
        )
        assert passes is False
        assert "coverage" in message.lower()

    def test_handles_zero_threshold(self) -> None:
        """Test handling of zero threshold (always passes)."""
        passes, _message = check_coverage_threshold(
            coverage=0.0,
            threshold=0.0,
        )
        assert passes is True

    def test_handles_100_percent_threshold(self) -> None:
        """Test handling of 100% threshold."""
        passes, _message = check_coverage_threshold(
            coverage=0.99,
            threshold=1.0,
        )
        assert passes is False


class TestQualityReport:
    """Tests for QualityReport dataclass (AC3)."""

    def test_default_values(self) -> None:
        """Test default values of QualityReport."""
        report = QualityReport()
        assert report.warnings == []
        assert report.has_assertions is True
        assert report.is_deterministic is True
        assert report.uses_fixtures is False

    def test_is_acceptable_with_defaults(self) -> None:
        """Test is_acceptable with default values."""
        report = QualityReport()
        assert report.is_acceptable() is True

    def test_is_acceptable_fails_without_assertions(self) -> None:
        """Test is_acceptable fails when no assertions."""
        report = QualityReport(has_assertions=False)
        assert report.is_acceptable() is False

    def test_is_acceptable_fails_when_not_deterministic(self) -> None:
        """Test is_acceptable fails when not deterministic."""
        report = QualityReport(is_deterministic=False)
        assert report.is_acceptable() is False

    def test_stores_warnings(self) -> None:
        """Test that warnings are stored properly."""
        report = QualityReport(
            warnings=["Missing assertions", "Uses time.time()"]
        )
        assert len(report.warnings) == 2
        assert "Missing assertions" in report.warnings


class TestValidateTestQuality:
    """Tests for validate_test_quality function (AC3)."""

    def test_returns_test_quality_report(self) -> None:
        """Test that function returns QualityReport."""
        tests = '''
def test_add():
    assert add(1, 2) == 3
'''
        report = validate_test_quality(tests)
        assert isinstance(report, QualityReport)

    def test_detects_missing_assertions(self) -> None:
        """Test detection of tests without assertions."""
        tests = '''
def test_no_assert():
    x = 1 + 1
    print(x)
'''
        report = validate_test_quality(tests)
        assert report.has_assertions is False

    def test_passes_with_assertions(self) -> None:
        """Test that tests with assertions pass."""
        tests = '''
def test_with_assert():
    result = calculate()
    assert result == expected
'''
        report = validate_test_quality(tests)
        assert report.has_assertions is True

    def test_detects_random_without_seed(self) -> None:
        """Test detection of random usage without seeding."""
        tests = '''
import random

def test_random_usage():
    value = random.randint(1, 10)
    assert value > 0
'''
        report = validate_test_quality(tests)
        # Should warn about non-deterministic behavior
        assert report.is_deterministic is False or any(
            "random" in w.lower() for w in report.warnings
        )

    def test_detects_time_without_mocking(self) -> None:
        """Test detection of time.time() usage."""
        tests = '''
import time

def test_time_usage():
    start = time.time()
    assert start > 0
'''
        report = validate_test_quality(tests)
        # Should warn about potential non-determinism
        assert any("time" in w.lower() for w in report.warnings) or not report.is_deterministic

    def test_detects_fixture_usage(self) -> None:
        """Test detection of pytest fixture usage."""
        tests = '''
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
'''
        report = validate_test_quality(tests)
        assert report.uses_fixtures is True

    def test_handles_empty_tests(self) -> None:
        """Test handling of empty test code."""
        report = validate_test_quality("")
        assert isinstance(report, QualityReport)

    def test_handles_syntax_errors(self) -> None:
        """Test graceful handling of syntax errors."""
        tests = "def test_broken("
        report = validate_test_quality(tests)
        assert isinstance(report, QualityReport)

    def test_detects_global_state_mutation(self) -> None:
        """Test detection of potential global state mutation."""
        tests = '''
GLOBAL_VAR = []

def test_modifies_global():
    GLOBAL_VAR.append(1)
    assert len(GLOBAL_VAR) == 1
'''
        report = validate_test_quality(tests)
        # Should warn about global state mutation
        assert any("global" in w.lower() for w in report.warnings) or len(report.warnings) > 0
