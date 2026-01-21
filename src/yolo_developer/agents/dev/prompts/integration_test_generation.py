"""Integration test generation prompt templates for Dev agent (Story 8.4).

This module provides the core prompt template and builder functions for
LLM-powered integration test generation. Templates emphasize boundary
testing, data flow verification, and error handling across components.

Key Concepts:
    - **Component Boundaries:** Test interactions between modules
    - **Data Flow Verification:** Verify data transforms correctly across boundaries
    - **Error Handling:** Test graceful degradation and recovery
    - **Test Independence:** Each test should run independently with fixtures

Example:
    >>> from yolo_developer.agents.dev.prompts.integration_test_generation import (
    ...     build_integration_test_prompt,
    ... )
    >>>
    >>> prompt = build_integration_test_prompt(
    ...     code_files_content="def process(x): return x * 2",
    ...     boundaries="Module A imports Module B",
    ...     data_flows="Input -> Transform -> Output",
    ...     error_scenarios="ValueError on invalid input",
    ... )
    >>> "integration" in prompt.lower()
    True
"""

from __future__ import annotations

# =============================================================================
# Integration Testing Best Practices (AC1, AC3, AC4)
# =============================================================================

INTEGRATION_TESTING_BEST_PRACTICES = """
## Integration Testing Best Practices

### Boundary Testing (AC1)
- Test all detected component boundaries with realistic scenarios
- Verify data flows correctly between components
- Test state updates are properly propagated across boundaries
- Mock external dependencies at boundaries (LLM, filesystem, databases)

### Data Flow Verification (AC2)
- Verify data transformations at each step of the flow
- Confirm data integrity is maintained across component boundaries
- Test with valid inputs at boundary entry points
- Test with invalid inputs to verify rejection at appropriate boundaries
- Trace data from source to destination

### Error Condition Coverage (AC3)
- Test graceful degradation paths (fallback to stubs, default values)
- Verify errors propagate correctly and don't silently fail
- Test recovery mechanisms where they exist
- Verify error messages are meaningful and include context
- Test exception propagation across component boundaries

### Test Independence and Isolation (AC4)
- Each test must run independently (no order dependencies)
- Use fixtures for shared setup and teardown
- Mock all external dependencies (LLM APIs, filesystem, network)
- Clean up state after each test
- Never share mutable state between tests
"""

# =============================================================================
# Pytest Integration Test Conventions (AC4, AC5)
# =============================================================================

PYTEST_INTEGRATION_CONVENTIONS = """
## Pytest Integration Test Conventions

### Naming Conventions (AC5)
- Test files: `test_<component>_<scenario>_integration.py` or `test_<component>_integration.py`
- Test functions: `test_<component>_<scenario>_<expected_behavior>`
- Test classes: `Test<Component>Integration` or `Test<Component><Scenario>`
- All test functions must have descriptive docstrings

### Required Markers and Structure
- Use `@pytest.mark.asyncio` for all async tests
- Use `@pytest.fixture` for shared setup (mocks, sample data)
- Use `@pytest.fixture(autouse=True)` for automatic cleanup

### Fixture Patterns for Integration Tests
```python
@pytest.fixture
def mock_llm_router() -> MagicMock:
    \"\"\"Mock LLM router for integration tests.\"\"\"
    router = MagicMock(spec=LLMRouter)
    router.call_task = AsyncMock(return_value="response")
    return router

@pytest.fixture(autouse=True)
def cleanup_state() -> Generator[None, None, None]:
    \"\"\"Clean up any shared state after each test.\"\"\"
    yield
    # Cleanup code here
```

### Mock Patterns for External Dependencies
```python
from unittest.mock import AsyncMock, MagicMock, patch

# Mock LLM calls
mock_router.call_task.return_value = "generated response"
mock_router.call_task.side_effect = [response1, response2]  # Multiple calls

# Mock filesystem
with patch("pathlib.Path.read_text", return_value="content"):
    result = function_under_test()
```

### Assertion Patterns for Integration Tests
```python
# Verify data flow
assert output.source_file == input.file_path
assert len(result.boundaries) >= expected_count

# Verify state propagation
assert "update_key" in result
assert result["update_key"] == expected_value

# Verify error handling
with pytest.raises(ValueError) as exc_info:
    await function_that_should_fail(invalid_input)
assert "expected error context" in str(exc_info.value)
```

### Test Structure (AAA Pattern)
- Arrange: Set up mocks, fixtures, and test data
- Act: Call the function/method being tested
- Assert: Verify expected outcomes including side effects
"""

# =============================================================================
# Integration Test Generation Template
# =============================================================================

INTEGRATION_TEST_TEMPLATE = """You are a senior Python test engineer generating comprehensive integration tests.

# Implementation Files to Test

```python
{code_files_content}
```

# Component Boundaries Detected

{boundaries}

# Data Flow Paths

{data_flows}

# Error Scenarios to Test

{error_scenarios}

{integration_best_practices}

{pytest_conventions}

# Additional Context

{additional_context}

# Instructions

Generate comprehensive pytest integration tests for the components above.

Requirements for the generated tests:
1. Include `from __future__ import annotations` at the top
2. Use `@pytest.mark.asyncio` for all async test functions
3. Test ALL detected component boundaries with realistic scenarios
4. Verify data flows correctly across component interactions
5. Test all error scenarios with graceful degradation verification
6. Use fixtures for shared setup (mocks, sample data)
7. Mock external dependencies (LLM, filesystem, network, databases)
8. Each test must be independent - no shared mutable state
9. Clean up state after tests using fixtures with cleanup
10. Use descriptive test names: `test_<component>_<scenario>_<expected_behavior>`
11. Include docstrings explaining what each test verifies
12. Use type annotations on all test functions

Naming convention for test functions:
- `test_<component>_<interaction>_<expected_outcome>`
- Example: `test_dev_agent_code_generation_produces_valid_syntax`
- Example: `test_boundary_detection_identifies_imports_between_modules`

Output only the Python test code. Do not include explanations outside the code.
Wrap the code in ```python and ``` markers.
"""


def build_integration_test_prompt(
    code_files_content: str,
    boundaries: str,
    data_flows: str,
    error_scenarios: str,
    additional_context: str = "",
    include_best_practices: bool = True,
    include_conventions: bool = True,
) -> str:
    """Build a complete integration test generation prompt for the LLM.

    Constructs a prompt that includes component code, detected boundaries,
    data flow paths, error scenarios, and testing conventions.

    Args:
        code_files_content: Combined content of code files to test.
        boundaries: String describing detected component boundaries.
        data_flows: String describing data flow paths through components.
        error_scenarios: String describing error scenarios to test.
        additional_context: Any additional context for test generation.
            Defaults to empty.
        include_best_practices: Whether to include testing best practices.
            Defaults to True.
        include_conventions: Whether to include pytest conventions.
            Defaults to True.

    Returns:
        Formatted prompt string ready for LLM.

    Example:
        >>> prompt = build_integration_test_prompt(
        ...     code_files_content="def process(x): return x * 2",
        ...     boundaries="Module A imports Module B",
        ...     data_flows="Input -> Transform -> Output",
        ...     error_scenarios="ValueError on invalid input",
        ... )
        >>> "integration" in prompt.lower()
        True
        >>> "boundary" in prompt.lower()
        True
    """
    # Include or exclude guidelines
    best_practices_text = INTEGRATION_TESTING_BEST_PRACTICES if include_best_practices else ""
    conventions_text = PYTEST_INTEGRATION_CONVENTIONS if include_conventions else ""

    # Format additional context
    if additional_context:
        context_text = f"## Additional Context\n\n{additional_context}"
    else:
        context_text = "(No additional context provided)"

    # Build the complete prompt
    prompt = INTEGRATION_TEST_TEMPLATE.format(
        code_files_content=code_files_content,
        boundaries=boundaries,
        data_flows=data_flows,
        error_scenarios=error_scenarios,
        integration_best_practices=best_practices_text,
        pytest_conventions=conventions_text,
        additional_context=context_text,
    )

    return prompt


def build_integration_test_retry_prompt(
    original_prompt: str,
    error_message: str,
    previous_tests: str,
) -> str:
    """Build a retry prompt when integration test generation fails syntax validation.

    Args:
        original_prompt: The original test generation prompt.
        error_message: The syntax error message from validation.
        previous_tests: The test code that failed validation.

    Returns:
        Modified prompt for retry with error context.

    Example:
        >>> retry_prompt = build_integration_test_retry_prompt(
        ...     original_prompt="Generate integration tests...",
        ...     error_message="Line 5: unexpected indent",
        ...     previous_tests="def test_x():\\n  pass",
        ... )
        >>> "syntax" in retry_prompt.lower()
        True
    """
    retry_template = """{original_prompt}

## Previous Attempt Failed

The previous integration test generation attempt failed with a syntax error:

**Error:** {error_message}

**Previous Tests (excerpt):**
```python
{previous_tests_excerpt}
```

Please fix the syntax error and generate valid Python integration test code.
Ensure all brackets, parentheses, and indentation are correct.
Pay special attention to:
- Proper async/await usage
- Correct fixture definitions
- Proper mock setup with AsyncMock for async functions
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
