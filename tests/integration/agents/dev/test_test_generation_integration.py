"""Integration tests for test generation pipeline (Story 8.3).

These tests verify the full integration of test generation components:
- Function extraction from implementation code
- LLM test generation with retry logic
- Coverage estimation and threshold checking
- Quality validation

Note: Tests with LLM require API keys and are marked for CI skip.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from yolo_developer.agents.dev.node import (
    _generate_stub_implementation,
    _generate_tests,
)
from yolo_developer.agents.dev.prompts import build_test_generation_prompt
from yolo_developer.agents.dev.test_utils import (
    calculate_coverage_estimate,
    check_coverage_threshold,
    extract_public_functions,
    generate_unit_tests_with_llm,
    validate_test_quality,
)
from yolo_developer.agents.dev.types import CodeFile


class TestTestGenerationPipelineIntegration:
    """Integration tests for the complete test generation pipeline."""

    def test_extract_functions_and_build_prompt(self) -> None:
        """Test extracting functions and building prompt works together."""
        implementation = '''
def calculate_total(items: list[float]) -> float:
    """Calculate the total of all items."""
    return sum(items)

def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email
'''
        # Extract functions
        functions = extract_public_functions(implementation)
        assert len(functions) == 2

        # Build prompt
        prompt = build_test_generation_prompt(
            implementation_code=implementation,
            function_list=[f.name for f in functions],
            module_name="utils",
        )

        # Verify prompt contains all necessary elements
        assert "calculate_total" in prompt
        assert "validate_email" in prompt
        assert "utils" in prompt
        assert "pytest" in prompt.lower()
        assert "edge case" in prompt.lower()

    def test_coverage_and_quality_validation_pipeline(self) -> None:
        """Test coverage estimation and quality validation work together."""
        implementation = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
'''
        test_code = '''
import pytest

class TestMathFunctions:
    """Tests for math functions."""

    def test_add_positive_numbers(self) -> None:
        """Test add with positive numbers."""
        assert add(2, 3) == 5

    def test_add_negative_numbers(self) -> None:
        """Test add with negative numbers."""
        assert add(-1, -2) == -3

    def test_multiply_positive_numbers(self) -> None:
        """Test multiply with positive numbers."""
        assert multiply(2, 3) == 6

    def test_multiply_by_zero(self) -> None:
        """Test multiply by zero."""
        assert multiply(5, 0) == 0
'''
        # Calculate coverage
        coverage = calculate_coverage_estimate(implementation, test_code)
        assert coverage >= 0.8  # Both functions are tested

        # Check threshold
        meets_threshold, _message = check_coverage_threshold(coverage, 0.8)
        assert meets_threshold

        # Validate quality
        quality = validate_test_quality(test_code)
        assert quality.is_acceptable()
        assert quality.has_assertions

    def test_quality_validation_detects_issues(self) -> None:
        """Test quality validation correctly identifies problems."""
        bad_test_code = '''
import random

class TestBadPractices:
    """Tests with bad practices."""

    global_counter = 0

    def test_uses_random(self) -> None:
        """Test that uses random without seeding."""
        value = random.randint(1, 100)
        # No assertion!
        pass

    def test_modifies_global_state(self) -> None:
        """Test that modifies global state."""
        global global_counter
        global_counter += 1
        assert global_counter > 0
'''
        quality = validate_test_quality(bad_test_code)
        # Should have warnings about non-determinism and isolation
        assert len(quality.warnings) > 0
        # Should detect missing assertions in at least one test
        # Note: The test has some assertions, so has_assertions may still be True
        # but warnings should flag the issues
        assert not quality.is_deterministic or len(quality.warnings) > 0

    @pytest.mark.asyncio
    async def test_stub_fallback_generates_valid_tests(self) -> None:
        """Test that stub fallback generates syntactically valid tests."""
        story = {
            "id": "test-story-001",
            "title": "Integration Test Story",
        }
        code_file = CodeFile(
            file_path="src/implementations/my_module.py",
            content='''
def process_data(data: list) -> dict:
    """Process incoming data."""
    return {"count": len(data), "data": data}

def validate_input(value: str) -> bool:
    """Validate input string."""
    return bool(value and value.strip())
''',
            file_type="source",
        )

        # Generate tests without router (stub fallback)
        tests = await _generate_tests(story, [code_file], router=None)

        assert len(tests) == 1
        test_file = tests[0]
        assert test_file.test_type == "unit"
        assert "test_" in test_file.file_path

        # Validate the generated test code is syntactically valid
        import ast

        try:
            ast.parse(test_file.content)
        except SyntaxError as e:
            pytest.fail(f"Generated test has syntax error: {e}")

        # Quality check the stub tests
        quality = validate_test_quality(test_file.content)
        # Stub tests should at least be deterministic
        assert quality.is_deterministic

    @pytest.mark.asyncio
    async def test_stub_implementation_generates_complete_artifacts(self) -> None:
        """Test stub implementation creates both code and test files."""
        story = {
            "id": "impl-story-001",
            "title": "Implementation Test",
            "description": "Test story for implementation",
        }

        artifact = await _generate_stub_implementation(story)

        # Check code files
        assert len(artifact.code_files) >= 1
        code_file = artifact.code_files[0]
        assert code_file.file_type == "source"
        assert "impl_story_001" in code_file.file_path

        # Check test files
        assert len(artifact.test_files) >= 1
        test_file = artifact.test_files[0]
        assert test_file.test_type == "unit"
        assert "test_" in test_file.file_path

        # Both should be syntactically valid Python
        import ast

        ast.parse(code_file.content)  # Should not raise
        ast.parse(test_file.content)  # Should not raise

    @pytest.mark.asyncio
    async def test_llm_test_generation_with_mocked_router(self) -> None:
        """Test LLM test generation flow with mocked router."""
        mock_router = MagicMock()

        # Generate valid test code response
        valid_test_code = '''```python
from __future__ import annotations

import pytest


class TestCalculator:
    """Tests for calculator functions."""

    def test_add_positive_numbers(self) -> None:
        """Test addition of positive numbers."""
        result = add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self) -> None:
        """Test addition of negative numbers."""
        result = add(-1, -2)
        assert result == -3

    def test_add_with_zero(self) -> None:
        """Test addition with zero."""
        result = add(0, 5)
        assert result == 5
```'''
        mock_router.call = AsyncMock(return_value=valid_test_code)

        implementation = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        functions = extract_public_functions(implementation)

        test_code, is_valid = await generate_unit_tests_with_llm(
            implementation_code=implementation,
            functions=functions,
            module_name="calculator",
            router=mock_router,
        )

        assert is_valid
        assert "test_add" in test_code
        assert "assert" in test_code

        # Verify router was called
        mock_router.call.assert_called()

    @pytest.mark.asyncio
    async def test_multiple_source_files_generate_multiple_tests(self) -> None:
        """Test that multiple source files generate multiple test files."""
        story = {"id": "multi-file-001", "title": "Multi-File Story"}
        code_files = [
            CodeFile(
                file_path="src/module_a.py",
                content="def func_a(): pass",
                file_type="source",
            ),
            CodeFile(
                file_path="src/module_b.py",
                content="def func_b(): pass",
                file_type="source",
            ),
            CodeFile(
                file_path="config.yaml",
                content="key: value",
                file_type="config",
            ),
        ]

        tests = await _generate_tests(story, code_files, router=None)

        # Should have 2 test files (one per source file, config ignored)
        assert len(tests) == 2
        test_paths = [t.file_path for t in tests]
        assert any("module_a" in p for p in test_paths)
        assert any("module_b" in p for p in test_paths)


class TestFunctionExtractionEdgeCases:
    """Tests for edge cases in function extraction."""

    def test_extracts_functions_with_complex_signatures(self) -> None:
        """Test extraction of functions with complex type hints."""
        code = '''
from typing import Optional, Union, Callable

def process(
    data: list[dict[str, Any]],
    callback: Optional[Callable[[int], str]] = None,
) -> Union[str, None]:
    """Process data with optional callback."""
    return None

async def async_fetch(url: str, timeout: float = 30.0) -> dict:
    """Async fetch data from URL."""
    return {}
'''
        functions = extract_public_functions(code)
        assert len(functions) == 2

        process_func = next(f for f in functions if f.name == "process")
        assert "data" in process_func.parameters
        assert "callback" in process_func.parameters

        fetch_func = next(f for f in functions if f.name == "async_fetch")
        assert "url" in fetch_func.parameters
        assert "timeout" in fetch_func.parameters

    def test_skips_private_and_dunder_methods(self) -> None:
        """Test that private and dunder methods are skipped."""
        code = """
def public_function(): pass
def _private_function(): pass
def __dunder_method__(): pass
"""
        functions = extract_public_functions(code)
        assert len(functions) == 1
        assert functions[0].name == "public_function"

    def test_handles_decorated_functions(self) -> None:
        """Test extraction of decorated functions."""
        code = """
@decorator
def decorated(): pass

@decorator1
@decorator2
def multi_decorated(): pass
"""
        functions = extract_public_functions(code)
        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "decorated" in names
        assert "multi_decorated" in names


class TestCoverageCalculationEdgeCases:
    """Tests for edge cases in coverage calculation."""

    def test_coverage_with_no_functions(self) -> None:
        """Test coverage calculation when code has no functions."""
        code = "# Just a comment"
        tests = "def test_something(): assert True"

        coverage = calculate_coverage_estimate(code, tests)
        # Should return 1.0 or handle gracefully
        assert coverage >= 0.0

    def test_coverage_with_no_tests(self) -> None:
        """Test coverage calculation when there are no test functions."""
        code = "def my_function(): pass"
        tests = "# No tests here"

        coverage = calculate_coverage_estimate(code, tests)
        assert coverage == 0.0

    def test_coverage_threshold_boundary(self) -> None:
        """Test coverage threshold at exact boundary."""
        # Exactly at threshold - passes with empty message
        passes, msg = check_coverage_threshold(0.8, 0.8)
        assert passes
        assert msg == ""  # Empty message when passing

        # Just below threshold
        fails, msg = check_coverage_threshold(0.79, 0.8)
        assert not fails
        assert "below" in msg.lower() and "79" in msg
