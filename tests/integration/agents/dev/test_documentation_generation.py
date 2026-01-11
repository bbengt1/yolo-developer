"""Integration tests for documentation generation (Story 8.5).

Tests the full flow from code analysis through documentation generation
to quality validation, verifying the end-to-end pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from yolo_developer.agents.dev.doc_utils import (
    ComplexSection,
    DocumentationInfo,
    detect_complex_sections,
    extract_documentation_info,
    format_complex_sections_for_prompt,
    format_documentation_info_for_prompt,
    generate_documentation_with_llm,
    validate_documentation_quality,
)

if TYPE_CHECKING:
    from yolo_developer.llm.router import LLMRouter


class TestDocumentationGenerationPipeline:
    """Integration tests for the full documentation generation pipeline."""

    @pytest.fixture
    def undocumented_code(self) -> str:
        """Sample code without documentation."""
        return '''
def calculate_total(items, tax_rate=0.1):
    total = 0
    for item in items:
        for discount in item.discounts:
            total -= discount.amount
        total += item.price
    return total * (1 + tax_rate)

def process_order(order):
    if order.status == "pending":
        for item in order.items:
            if item.quantity > 0:
                item.reserve()
    return order
'''

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Mock LLM router for testing."""
        router = MagicMock()
        router.call = AsyncMock(
            return_value='''```python
"""Order processing and calculation utilities.

This module provides functions for calculating order totals
and processing order items.

Key Functions:
    - calculate_total: Calculate total price with tax
    - process_order: Process and reserve order items

Example:
    >>> total = calculate_total([Item(price=10)], tax_rate=0.1)
    >>> total
    11.0
"""

def calculate_total(items, tax_rate=0.1):
    """Calculate total price including tax.

    Args:
        items: List of items with price attribute.
        tax_rate: Tax rate to apply. Defaults to 0.1.

    Returns:
        Total price including tax.

    Example:
        >>> calculate_total([Item(price=10)], 0.1)
        11.0
    """
    total = 0
    for item in items:
        total += item.price
    return total * (1 + tax_rate)

def process_order(order):
    """Process an order by reserving items.

    Args:
        order: Order object with status and items.

    Returns:
        Processed order object.

    Example:
        >>> order = process_order(pending_order)
        >>> order.status
        'pending'
    """
    # Reserve items only for pending orders
    if order.status == "pending":
        for item in order.items:
            if item.quantity > 0:
                item.reserve()
    return order
```'''
        )
        return router

    def test_extract_documentation_info_detects_missing_docs(
        self, undocumented_code: str
    ) -> None:
        """Test that documentation analysis detects missing docstrings."""
        info = extract_documentation_info(undocumented_code)

        assert info.has_module_docstring is False
        assert len(info.functions_missing_docstrings) == 2
        assert "calculate_total" in info.functions_missing_docstrings
        assert "process_order" in info.functions_missing_docstrings

    def test_detect_complex_sections_finds_nested_loops(
        self, undocumented_code: str
    ) -> None:
        """Test that complex section detection finds nested loops."""
        sections = detect_complex_sections(undocumented_code)

        nested = [s for s in sections if s.complexity_type == "nested_loop"]
        assert len(nested) >= 1

    def test_validate_documentation_quality_reports_issues(
        self, undocumented_code: str
    ) -> None:
        """Test that quality validation reports documentation issues."""
        report = validate_documentation_quality(undocumented_code)

        assert report.has_module_docstring is False
        assert len(report.warnings) >= 2  # Missing module + missing function docs
        assert report.is_acceptable() is False

    @pytest.mark.asyncio
    async def test_full_documentation_pipeline(
        self,
        undocumented_code: str,
        mock_router: MagicMock,
    ) -> None:
        """Test full pipeline from analysis to documented code."""
        # Step 1: Analyze current documentation
        info = extract_documentation_info(undocumented_code)
        assert info.has_module_docstring is False

        # Step 2: Detect complex sections
        sections = detect_complex_sections(undocumented_code)
        assert len(sections) >= 1

        # Step 3: Generate documentation with LLM
        documented_code, is_valid = await generate_documentation_with_llm(
            code=undocumented_code,
            context="Order processing module",
            router=mock_router,
        )

        # Should generate valid code
        assert is_valid is True
        assert len(documented_code) > 0

        # Step 4: Validate improved documentation
        report = validate_documentation_quality(documented_code)

        # Should now have module docstring
        assert report.has_module_docstring is True
        assert report.functions_with_args >= 2
        assert report.functions_with_returns >= 2
        assert report.is_acceptable() is True

    def test_format_functions_produce_valid_prompts(
        self, undocumented_code: str
    ) -> None:
        """Test that formatting functions produce usable prompt content."""
        info = extract_documentation_info(undocumented_code)
        sections = detect_complex_sections(undocumented_code)

        info_text = format_documentation_info_for_prompt(info)
        sections_text = format_complex_sections_for_prompt(sections)

        # Should produce readable text
        assert len(info_text) > 0
        assert len(sections_text) > 0
        assert "calculate_total" in info_text
        assert "process_order" in info_text


class TestDocumentationAnalysisEdgeCases:
    """Test edge cases in documentation analysis."""

    def test_handles_async_functions(self) -> None:
        """Test that async functions are properly analyzed."""
        code = '''"""Module doc."""

async def fetch_data(url: str) -> dict:
    """Fetches data from URL."""
    pass

async def process():
    pass
'''
        info = extract_documentation_info(code)

        assert info.has_module_docstring is True
        assert info.total_public_functions == 2
        assert info.documented_functions == 1
        assert "process" in info.functions_missing_docstrings

    def test_handles_class_methods(self) -> None:
        """Test that class methods are analyzed."""
        code = '''"""Module doc."""

class Calculator:
    """Calculator class."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
'''
        info = extract_documentation_info(code)

        assert info.has_module_docstring is True
        # Should count public methods
        assert info.total_public_functions >= 2

    def test_handles_decorated_functions(self) -> None:
        """Test that decorated functions are analyzed."""
        code = '''"""Module doc."""

def decorator(func):
    return func

@decorator
def decorated_func():
    pass
'''
        info = extract_documentation_info(code)

        assert "decorated_func" in info.functions_missing_docstrings


class TestDocumentationQualityValidation:
    """Test documentation quality validation."""

    def test_validates_args_section_completeness(self) -> None:
        """Test validation of Args section completeness."""
        code_with_args = '''"""Module."""

def func(x: int, y: str) -> bool:
    """Does something.

    Args:
        x: The x value.
        y: The y value.

    Returns:
        Result boolean.
    """
    return True
'''
        code_without_args = '''"""Module."""

def func(x: int, y: str) -> bool:
    """Does something without args section."""
    return True
'''
        report_with = validate_documentation_quality(code_with_args)
        report_without = validate_documentation_quality(code_without_args)

        assert report_with.functions_with_args == 1
        assert report_without.functions_with_args == 0

    def test_validates_returns_section(self) -> None:
        """Test validation of Returns section."""
        code = '''"""Module."""

def func_with_returns() -> int:
    """Function.

    Returns:
        An integer.
    """
    return 1

def func_without_returns() -> int:
    """Function."""
    return 1
'''
        report = validate_documentation_quality(code)

        assert report.functions_with_returns == 1
        assert report.total_functions == 2

    def test_validates_example_section(self) -> None:
        """Test validation of Example section."""
        code = '''"""Module."""

def func_with_example() -> int:
    """Function.

    Example:
        >>> func_with_example()
        1
    """
    return 1

def func_with_doctest() -> int:
    """Function.

    >>> func_with_doctest()
    1
    """
    return 1
'''
        report = validate_documentation_quality(code)

        # Both should count as having examples
        assert report.functions_with_examples == 2


class TestComplexSectionDetection:
    """Test complex section detection."""

    def test_detects_deep_nesting(self) -> None:
        """Test detection of deeply nested code."""
        code = '''"""Module."""

def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    return True
    return False
'''
        sections = detect_complex_sections(code)

        deep = [s for s in sections if s.complexity_type == "deep_nesting"]
        assert len(deep) >= 1

    def test_detects_while_loops(self) -> None:
        """Test detection of nested while loops."""
        code = '''"""Module."""

def nested_while():
    while True:
        while True:
            break
        break
'''
        sections = detect_complex_sections(code)

        nested = [s for s in sections if s.complexity_type == "nested_loop"]
        assert len(nested) >= 1

    def test_section_contains_function_name(self) -> None:
        """Test that detected sections include function name."""
        code = '''"""Module."""

def my_function():
    for i in range(10):
        for j in range(10):
            pass
'''
        sections = detect_complex_sections(code)

        assert len(sections) >= 1
        assert sections[0].function_name == "my_function"
