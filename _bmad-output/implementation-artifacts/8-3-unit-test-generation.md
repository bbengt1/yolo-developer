# Story 8.3: Unit Test Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want unit tests generated for all functionality,
So that code correctness is verified.

## Acceptance Criteria

1. **AC1: Public Function Test Coverage**
   - **Given** generated implementation code
   - **When** unit tests are created
   - **Then** all public functions have at least one test
   - **And** test functions are named descriptively (`test_<function_name>_<scenario>`)
   - **And** test functions have docstrings explaining what they test

2. **AC2: Edge Case Coverage**
   - **Given** a public function with edge cases
   - **When** unit tests are generated
   - **Then** tests include edge cases (empty inputs, None values, boundary values)
   - **And** tests include expected failure scenarios (invalid inputs, exceptions)
   - **And** tests verify error messages are meaningful

3. **AC3: Test Isolation and Determinism**
   - **Given** generated unit tests
   - **When** tests are executed
   - **Then** each test is independent (no shared mutable state)
   - **And** tests produce the same result every run (deterministic)
   - **And** tests use fixtures or mocks for external dependencies
   - **And** tests clean up any resources they create

4. **AC4: Coverage Threshold Validation**
   - **Given** generated tests and implementation code
   - **When** coverage is measured
   - **Then** coverage percentage is calculated accurately
   - **And** coverage meets configured threshold (from `quality.test_coverage_threshold`)
   - **And** uncovered lines are reported with warnings

5. **AC5: LLM Integration for Test Generation**
   - **Given** implementation code to test
   - **When** `_generate_unit_tests_with_llm()` is called
   - **Then** LLM is called to generate contextual test code
   - **And** LLM calls use tenacity retry with exponential backoff
   - **And** LLM tier is "complex" for test generation (per ADR-003)
   - **And** generated tests are validated for syntax before returning

6. **AC6: Prompt Engineering for Test Quality**
   - **Given** test generation prompts
   - **When** prompts are constructed for LLM
   - **Then** prompts explicitly include testing best practices
   - **And** prompts include the implementation code to test
   - **And** prompts include function signatures and docstrings
   - **And** prompts request pytest-style tests with proper assertions

## Tasks / Subtasks

- [x] Task 1: Create LLM Prompt Templates for Test Generation (AC: 6)
  - [x] Create `src/yolo_developer/agents/dev/prompts/test_generation.py`
  - [x] Create `TEST_GENERATION_TEMPLATE` with testing best practices
  - [x] Include pytest conventions in prompt template
  - [x] Include edge case identification guidelines
  - [x] Include isolation and determinism requirements in prompt
  - [x] Create `build_test_generation_prompt()` function
  - [x] Create `build_test_retry_prompt()` for syntax error recovery

- [x] Task 2: Implement Test Analysis Functions (AC: 1, 2)
  - [x] Create `extract_public_functions(code: str) -> list[FunctionInfo]`
  - [x] Use AST parsing to find public functions (not starting with `_`)
  - [x] Extract function name, signature, docstring, and parameters
  - [x] Create `identify_edge_cases(func_info: FunctionInfo) -> list[str]`
  - [x] Identify potential edge cases from type hints and docstrings

- [x] Task 3: Implement LLM Test Generation Function (AC: 5, 6)
  - [x] Create `generate_unit_tests_with_llm(code, functions, context) -> tuple[str, bool]`
  - [x] Integrate with LLMRouter (import from `yolo_developer.llm.router`)
  - [x] Use "complex" tier for test generation per ADR-003
  - [x] Apply tenacity retry pattern for resilience
  - [x] Include implementation code and function info in prompt
  - [x] Validate generated test syntax before returning

- [x] Task 4: Implement Coverage Calculation (AC: 4)
  - [x] Create `calculate_coverage_estimate(code: str, tests: str) -> float`
  - [x] Use AST to count testable statements in code
  - [x] Analyze test code to estimate coverage (heuristic-based)
  - [x] Create `check_coverage_threshold(coverage: float, threshold: float) -> tuple[bool, str]`
  - [x] Compare against configured threshold
  - [x] Return (passes, warning_message) tuple

- [x] Task 5: Update `_generate_tests` to Use LLM (AC: 1, 2, 3, 5)
  - [x] Replace stub implementation with LLM-powered generation
  - [x] Pass implementation code and extracted functions to LLM
  - [x] Validate generated test syntax before creating TestFile
  - [x] Retry with modified prompt if syntax validation fails
  - [x] Fall back to stub tests if LLM fails after retries
  - [x] Log coverage estimate and warnings

- [x] Task 6: Add Test Quality Validation (AC: 3, 4)
  - [x] Create `validate_test_quality(test_code: str) -> QualityReport`
  - [x] Check for test isolation patterns (no global state mutations)
  - [x] Check for determinism (no random without seed, no time.time())
  - [x] Check for proper assertions (assert statements exist)
  - [x] Check for fixture usage when appropriate
  - [x] Return report with quality warnings

- [x] Task 7: Export New Functions from Prompts Module (AC: 6)
  - [x] Update `src/yolo_developer/agents/dev/prompts/__init__.py`
  - [x] Export `TEST_GENERATION_TEMPLATE`
  - [x] Export `build_test_generation_prompt`
  - [x] Export `build_test_retry_prompt`

- [x] Task 8: Write Unit Tests for Prompt Templates (AC: 6)
  - [x] Create `tests/unit/agents/dev/prompts/test_test_generation.py`
  - [x] Test prompt template rendering with variables
  - [x] Test that testing best practices are included
  - [x] Test that implementation code is included
  - [x] Test prompt structure follows expected format

- [x] Task 9: Write Unit Tests for Test Analysis (AC: 1, 2)
  - [x] Create tests for `extract_public_functions()`
  - [x] Test extraction of functions with various signatures
  - [x] Test that private functions are excluded
  - [x] Test edge case identification from type hints

- [x] Task 10: Write Unit Tests for LLM Test Generation (AC: 5)
  - [x] Test LLM integration with mock responses
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to stub on persistent failures
  - [x] Test syntax validation of generated tests

- [x] Task 11: Write Unit Tests for Coverage Calculation (AC: 4)
  - [x] Test coverage estimation with simple functions
  - [x] Test threshold checking with various thresholds
  - [x] Test warning message generation for low coverage

- [x] Task 12: Write Integration Tests (AC: 1-6)
  - [x] Test full test generation flow with mocked LLM
  - [x] Test end-to-end from code to test file
  - [x] Test coverage validation integration
  - [x] Test quality validation integration

### Code Review Fixes (AI)

All issues identified during code review have been fixed:

- [x] [HIGH] AC5: Added tenacity @retry decorator with exponential backoff to `_call_llm_with_retry()` function per ADR-007
- [x] [HIGH] Added exports from test_utils.py to `agents/dev/__init__.py` for public API access
- [x] [HIGH] Updated `_generate_tests` to load coverage threshold from config via `config.quality.test_coverage_threshold` per AC4
- [x] [MED] Added documentation explaining why QualityReport is intentionally not frozen (incremental construction pattern)
- [x] [MED] Added note about heuristic nature of `identify_edge_cases()` function
- [x] [MED] Enhanced `calculate_coverage_estimate()` to detect method calls via `obj.method()` pattern
- [x] [MED] Added tests for coverage estimation accuracy and method call detection

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Continue using frozen dataclasses for TestFile
- **ADR-003 (LLM Provider):** Use LLMRouter with "complex" tier for test generation
- **ADR-005 (LangGraph Communication):** Maintain existing state update pattern
- **ADR-006 (Quality Gates):** DoD gate already integrated (Story 8.1)
- **ADR-007 (Error Handling):** Use tenacity for LLM retries
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` in all new files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/dev/node.py` (Story 8.1, 8.2)
- All dataclasses should be frozen (immutable)
- Use tenacity @retry decorator with exponential backoff for LLM calls

### Library Versions (from architecture.md)

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 1.0.5 | Orchestration framework |
| structlog | latest | Structured logging |
| tenacity | latest | Retry with backoff |
| LiteLLM | latest | Multi-provider LLM abstraction |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |

### LLM Router Usage Pattern

```python
from yolo_developer.llm.router import LLMRouter

# Use "complex" tier for test generation
response = await router.call(
    messages=[{"role": "user", "content": prompt}],
    tier="complex"  # Uses premium model per ADR-003
)
```

### Test Generation Prompt Structure

```python
TEST_GENERATION_TEMPLATE = """
Generate pytest unit tests for the following Python code:

```python
{implementation_code}
```

Functions to test:
{function_list}

Testing Requirements:
- Use pytest framework with pytest.mark decorators as needed
- Include docstrings for all test functions
- Test all public functions (not starting with _)
- Include edge cases: empty inputs, None values, boundary conditions
- Include expected failure tests with pytest.raises
- Use fixtures for setup/teardown when appropriate
- Keep tests isolated - no shared mutable state
- Make tests deterministic - no random without seeds

Generate clean, comprehensive pytest code that:
1. Covers all public functions
2. Tests happy path and edge cases
3. Uses descriptive test names
4. Includes proper assertions
"""
```

### Project Structure Notes

- **Prompt Templates:** `src/yolo_developer/agents/dev/prompts/test_generation.py`
- **Node Function:** `src/yolo_developer/agents/dev/node.py` (modify)
- **Test Location:** `tests/unit/agents/dev/`

### Previous Story Learnings Applied (Story 8.2)

From Story 8.2 implementation:
- LLM code generation with `_generate_code_with_llm()` pattern
- Syntax validation using `validate_python_syntax()` from code_utils.py
- Code extraction using `extract_code_from_response()` from code_utils.py
- Maintainability checking with `check_maintainability()`
- Retry prompt building with `build_retry_prompt()`
- LLMRouter initialization with `_get_llm_router()` pattern
- Coverage threshold in config: `quality.test_coverage_threshold`

### Git Commit Pattern (from recent commits)

Recent commits follow pattern:
```
feat: Implement <feature> with code review fixes (Story X.Y)
```

### Existing Dev Module Structure

```
src/yolo_developer/agents/dev/
├── __init__.py         # Exports: dev_node, DevOutput, ImplementationArtifact, CodeFile, TestFile
├── types.py            # Type definitions (CodeFile, TestFile, ImplementationArtifact, DevOutput)
├── node.py             # dev_node function and helpers (MODIFY _generate_tests)
├── code_utils.py       # Code validation and extraction utilities
└── prompts/
    ├── __init__.py     # Exports code generation prompts (ADD test generation)
    └── code_generation.py  # Code generation prompts
    └── test_generation.py  # NEW: Test generation prompts
```

### Key Imports for Implementation

```python
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.types import TestFile
from yolo_developer.agents.dev.code_utils import (
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)
```

### Function Extraction Pattern

```python
@dataclass(frozen=True)
class FunctionInfo:
    """Information about a public function to test.

    Attributes:
        name: Function name.
        signature: Full function signature string.
        docstring: Function docstring (if present).
        parameters: List of parameter names.
        return_type: Return type annotation (if present).
    """
    name: str
    signature: str
    docstring: str | None
    parameters: tuple[str, ...]
    return_type: str | None


def _extract_public_functions(code: str) -> list[FunctionInfo]:
    """Extract public function information from code using AST.

    Args:
        code: Python code string to analyze.

    Returns:
        List of FunctionInfo for each public function.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions: list[FunctionInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions
            if node.name.startswith("_"):
                continue
            # Extract function info...
    return functions
```

### Test Quality Report Pattern

```python
@dataclass
class TestQualityReport:
    """Report of test quality analysis.

    Attributes:
        warnings: List of quality warnings.
        has_assertions: Whether tests have proper assertions.
        is_deterministic: Whether tests appear deterministic.
        uses_fixtures: Whether tests use fixtures appropriately.
    """
    warnings: list[str] = field(default_factory=list)
    has_assertions: bool = True
    is_deterministic: bool = True
    uses_fixtures: bool = False

    def is_acceptable(self) -> bool:
        """Check if test quality is acceptable."""
        return self.has_assertions and self.is_deterministic
```

### Coverage Threshold from Config

```python
from yolo_developer.config import load_config

config = load_config()
threshold = config.quality.test_coverage_threshold  # Default 0.80
```

### Story Dependencies

This story builds on:
- Story 8.1 (Create Dev Agent Node) - dev_node foundation
- Story 8.2 (Maintainable Code Generation) - LLM integration patterns

This story enables:
- Story 8.4 (Integration Test Generation) - cross-component tests
- Story 9.2 (TEA Coverage Validation) - TEA validates coverage

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR58 | Dev Agent can write unit tests for implemented functionality | LLM-powered unit test generation with coverage validation |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.3] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-003] - LLM provider abstraction
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007] - Error handling patterns
- [Source: src/yolo_developer/agents/dev/node.py] - Existing dev node (Story 8.1, 8.2)
- [Source: src/yolo_developer/agents/dev/code_utils.py] - Code validation utilities
- [Source: src/yolo_developer/agents/dev/prompts/code_generation.py] - Code generation prompts (pattern)
- [Source: _bmad-output/implementation-artifacts/8-2-maintainable-code-generation.md] - Previous story learnings
- [FR58: Dev Agent can write unit tests for implemented functionality]
