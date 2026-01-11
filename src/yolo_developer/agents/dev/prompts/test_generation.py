"""Test generation prompt templates for Dev agent (Story 8.3).

This module provides the core prompt template and builder function for
LLM-powered unit test generation. The template emphasizes testing best
practices, edge case coverage, and pytest conventions.

Key Concepts:
    - **Test Isolation:** Each test should be independent
    - **Determinism:** Tests must produce consistent results
    - **Edge Cases:** Cover empty inputs, None values, boundary conditions
    - **Proper Assertions:** Use descriptive assertions with clear messages

Example:
    >>> from yolo_developer.agents.dev.prompts.test_generation import (
    ...     build_test_generation_prompt,
    ... )
    >>>
    >>> prompt = build_test_generation_prompt(
    ...     implementation_code="def add(a, b): return a + b",
    ...     function_list=["add"],
    ...     module_name="math_utils",
    ... )
    >>> "add" in prompt
    True
"""

from __future__ import annotations

# =============================================================================
# Testing Best Practices (AC1, AC2, AC3)
# =============================================================================

TESTING_BEST_PRACTICES = """
## Testing Best Practices

### Coverage Requirements
- Test ALL public functions (not starting with _)
- Include happy path tests for expected behavior
- Include edge case tests for boundary conditions
- Include error handling tests for invalid inputs

### Edge Case Coverage (AC2)
- Test with empty inputs (empty lists, empty strings, etc.)
- Test with None values where applicable
- Test boundary values (0, -1, max values)
- Test with invalid/malformed inputs
- Verify error messages are meaningful

### Test Isolation and Determinism (AC3)
- Each test must be isolated - no shared mutable state between tests
- Tests must be deterministic - same result every run
- Use fixtures for setup/teardown of shared resources
- Use mocks for external dependencies (APIs, databases, file systems)
- Clean up any resources created during tests
- Never rely on test execution order
- Avoid using random without seeding
- Avoid using current time without mocking

### Assertion Best Practices
- Use clear, descriptive assertions
- Include assertion messages for complex checks
- Use pytest.raises for exception testing
- Use pytest.approx for floating point comparisons
- Prefer specific assertions (assertEqual) over generic (assertTrue)
"""

# =============================================================================
# Pytest Conventions (AC1, AC6)
# =============================================================================

PYTEST_CONVENTIONS = """
## Pytest Conventions

### Naming Conventions (AC1)
- Test functions: `test_<function_name>_<scenario>`
- Test classes: `Test<ModuleName>` or `Test<ClassName>`
- Test files: `test_<module_name>.py`
- All test functions must have docstrings explaining what they test

### Common Patterns
- Use `@pytest.fixture` for setup/teardown code
- Use `@pytest.mark.parametrize` for data-driven tests
- Use `pytest.raises` for exception testing:
  ```python
  with pytest.raises(ValueError) as exc_info:
      function_that_raises()
  assert "expected message" in str(exc_info.value)
  ```

### Test Structure (AAA Pattern)
- Arrange: Set up test data and preconditions
- Act: Execute the function under test
- Assert: Verify the expected outcome

### Example Test Function
```python
def test_calculate_sum_with_positive_numbers(self) -> None:
    \"\"\"Test that calculate_sum correctly sums positive numbers.\"\"\"
    # Arrange
    numbers = [1, 2, 3, 4, 5]

    # Act
    result = calculate_sum(numbers)

    # Assert
    assert result == 15
```
"""

# =============================================================================
# Test Generation Template
# =============================================================================

TEST_GENERATION_TEMPLATE = """You are a senior Python test engineer generating comprehensive unit tests.

# Implementation to Test

**Module:** {module_name}

```python
{implementation_code}
```

# Functions to Test

{function_list}

{testing_best_practices}

{pytest_conventions}

# Additional Context

{additional_context}

# Instructions

Generate comprehensive pytest unit tests for the implementation above.

Requirements for the generated tests:
1. Include `from __future__ import annotations` at the top
2. Test ALL public functions listed above
3. Include edge case tests (empty inputs, None, boundary values)
4. Include error/exception tests with pytest.raises
5. Use descriptive test names: `test_<function>_<scenario>`
6. Include docstrings for all test functions
7. Use fixtures for setup when appropriate
8. Keep tests isolated and deterministic
9. Use type annotations on all test functions

Output only the Python test code. Do not include explanations outside the code.
Wrap the code in ```python and ``` markers.
"""


def build_test_generation_prompt(
    implementation_code: str,
    function_list: list[str],
    module_name: str,
    additional_context: str = "",
    include_best_practices: bool = True,
    include_conventions: bool = True,
) -> str:
    """Build a complete test generation prompt for the LLM.

    Constructs a prompt that includes implementation code, functions to test,
    testing best practices, and pytest conventions.

    Args:
        implementation_code: The Python code to generate tests for.
        function_list: List of public function names to test.
        module_name: Name of the module being tested.
        additional_context: Any additional context for test generation. Defaults to empty.
        include_best_practices: Whether to include testing best practices.
            Defaults to True.
        include_conventions: Whether to include pytest conventions. Defaults to True.

    Returns:
        Formatted prompt string ready for LLM.

    Example:
        >>> prompt = build_test_generation_prompt(
        ...     implementation_code="def add(a, b): return a + b",
        ...     function_list=["add"],
        ...     module_name="math_utils",
        ... )
        >>> "add" in prompt
        True
        >>> "pytest" in prompt.lower()
        True
    """
    # Format function list
    if function_list:
        func_lines = [f"- `{func}`" for func in function_list]
        func_text = "\n".join(func_lines)
    else:
        func_text = "(No public functions identified - generate tests based on module analysis)"

    # Include or exclude guidelines
    best_practices_text = TESTING_BEST_PRACTICES if include_best_practices else ""
    conventions_text = PYTEST_CONVENTIONS if include_conventions else ""

    # Format additional context
    if additional_context:
        context_text = f"## Additional Context\n\n{additional_context}"
    else:
        context_text = "(No additional context provided)"

    # Build the complete prompt
    prompt = TEST_GENERATION_TEMPLATE.format(
        implementation_code=implementation_code,
        function_list=func_text,
        module_name=module_name,
        testing_best_practices=best_practices_text,
        pytest_conventions=conventions_text,
        additional_context=context_text,
    )

    return prompt


def build_test_retry_prompt(
    original_prompt: str,
    error_message: str,
    previous_tests: str,
) -> str:
    """Build a retry prompt when test generation fails syntax validation.

    Args:
        original_prompt: The original test generation prompt.
        error_message: The syntax error message from validation.
        previous_tests: The test code that failed validation.

    Returns:
        Modified prompt for retry with error context.

    Example:
        >>> retry_prompt = build_test_retry_prompt(
        ...     original_prompt="Generate tests...",
        ...     error_message="Line 5: unexpected indent",
        ...     previous_tests="def test_x():\\n  pass",
        ... )
        >>> "syntax error" in retry_prompt.lower()
        True
    """
    retry_template = """{original_prompt}

## Previous Attempt Failed

The previous test generation attempt failed with a syntax error:

**Error:** {error_message}

**Previous Tests (excerpt):**
```python
{previous_tests_excerpt}
```

Please fix the syntax error and generate valid Python test code.
Ensure all brackets, parentheses, and indentation are correct.
"""

    # Limit previous tests excerpt to first 500 chars
    tests_excerpt = previous_tests[:500]
    if len(previous_tests) > 500:
        tests_excerpt += "\n... (truncated)"

    return retry_template.format(
        original_prompt=original_prompt,
        error_message=error_message,
        previous_tests_excerpt=tests_excerpt,
    )
