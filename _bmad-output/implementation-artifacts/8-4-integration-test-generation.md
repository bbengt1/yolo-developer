# Story 8.4: Integration Test Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want integration tests for cross-component functionality,
So that components work together correctly.

## Acceptance Criteria

1. **AC1: Interaction Boundary Testing**
   - **Given** components that interact (e.g., Dev agent producing code consumed by TEA agent)
   - **When** integration tests are created
   - **Then** interaction boundaries are tested with realistic scenarios
   - **And** tests verify data flows correctly between components
   - **And** tests verify state updates are properly propagated

2. **AC2: Data Flow Verification**
   - **Given** a multi-component data flow (story -> code -> tests -> output)
   - **When** integration tests run
   - **Then** data transformations are verified at each step
   - **And** data integrity is maintained across boundaries
   - **And** invalid data is rejected at appropriate boundaries

3. **AC3: Error Condition Coverage**
   - **Given** potential error scenarios in component interactions
   - **When** integration tests cover error paths
   - **Then** graceful degradation is verified (e.g., fallback to stubs)
   - **And** error propagation is tested (errors don't silently fail)
   - **And** recovery mechanisms are tested where applicable

4. **AC4: Test Independence**
   - **Given** generated integration tests
   - **When** tests are executed
   - **Then** each test can run independently (no order dependency)
   - **And** tests clean up their state after execution
   - **And** tests use fixtures for shared setup
   - **And** tests mock external dependencies (LLM, filesystem)

5. **AC5: Integration Test Type Classification**
   - **Given** generated tests
   - **When** `TestFile` objects are created
   - **Then** `test_type` is set to `"integration"` (not "unit" or "e2e")
   - **And** test files are placed in appropriate directory structure
   - **And** tests follow naming convention `test_<component>_<scenario>.py`

6. **AC6: LLM-Powered Integration Test Generation**
   - **Given** implementation code with multiple interacting components
   - **When** `_generate_integration_tests_with_llm()` is called
   - **Then** LLM generates tests for component interactions
   - **And** LLM calls use "complex" tier per ADR-003
   - **And** LLM calls use tenacity retry with exponential backoff per ADR-007
   - **And** generated tests are syntax-validated before returning

7. **AC7: Component Boundary Detection**
   - **Given** implementation code files
   - **When** component analysis runs
   - **Then** interacting components are identified (imports, shared state)
   - **And** boundary points are detected (function calls, state updates)
   - **And** test scenarios are derived from detected boundaries

## Tasks / Subtasks

- [x] Task 1: Create Integration Test Prompt Templates (AC: 6, 7)
  - [x] Create `src/yolo_developer/agents/dev/prompts/integration_test_generation.py`
  - [x] Create `INTEGRATION_TEST_TEMPLATE` with boundary testing guidance
  - [x] Include component interaction patterns in prompt
  - [x] Include error handling test requirements in prompt
  - [x] Include data flow verification requirements
  - [x] Create `build_integration_test_prompt()` function
  - [x] Create `build_integration_test_retry_prompt()` for syntax recovery

- [x] Task 2: Implement Component Boundary Analysis (AC: 1, 7)
  - [x] Create `analyze_component_boundaries(code_files: list[CodeFile]) -> list[ComponentBoundary]`
  - [x] Use AST parsing to identify imports and dependencies between files
  - [x] Identify function calls that cross file boundaries
  - [x] Detect shared state access patterns
  - [x] Create `ComponentBoundary` dataclass with source, target, interaction_type
  - [x] Identify async boundaries (await calls to other modules)

- [x] Task 3: Implement Data Flow Analysis (AC: 2)
  - [x] Create `analyze_data_flow(code_files: list[CodeFile]) -> list[DataFlowPath]`
  - [x] Trace data from input to output through function calls
  - [x] Identify transformation points where data changes shape
  - [x] Detect data validation points (type checks, assertions)
  - [x] Create `DataFlowPath` dataclass with steps and transformations

- [x] Task 4: Implement LLM Integration Test Generation (AC: 6)
  - [x] Create `generate_integration_tests_with_llm(code_files, boundaries, flows, context) -> tuple[str, bool]`
  - [x] Integrate with LLMRouter (import from `yolo_developer.llm.router`)
  - [x] Use "complex" tier for integration test generation per ADR-003
  - [x] Apply tenacity retry pattern per ADR-007
  - [x] Include boundary and flow analysis in prompt
  - [x] Validate generated test syntax before returning

- [x] Task 5: Implement Error Scenario Detection (AC: 3)
  - [x] Create `detect_error_scenarios(code_files: list[CodeFile]) -> list[ErrorScenario]`
  - [x] Identify try/except blocks and exception types raised
  - [x] Identify boundary validation (type guards, assertions)
  - [x] Detect fallback patterns (if/else with fallback behavior)
  - [x] Create `ErrorScenario` dataclass with trigger, handling, recovery

- [x] Task 6: Update `_generate_tests` to Include Integration Tests (AC: 1-7)
  - [x] Modify `_generate_tests()` in node.py to generate both unit and integration tests
  - [x] Add conditional logic: if multiple code files, generate integration tests
  - [x] Analyze boundaries and flows before calling LLM
  - [x] Create separate TestFile with `test_type="integration"`
  - [x] Validate generated integration test syntax
  - [x] Fall back to stub integration tests if LLM fails

- [x] Task 7: Create Integration Test Quality Validation (AC: 3, 4)
  - [x] Create `validate_integration_test_quality(test_code: str) -> IntegrationTestQualityReport`
  - [x] Check for fixture usage (integration tests should use fixtures)
  - [x] Check for mock patterns (external dependencies should be mocked)
  - [x] Check for cleanup patterns (state restoration after tests)
  - [x] Check for async test markers (@pytest.mark.asyncio)
  - [x] Return report with quality warnings

- [x] Task 8: Export New Functions from Prompts Module (AC: 6)
  - [x] Update `src/yolo_developer/agents/dev/prompts/__init__.py`
  - [x] Export `INTEGRATION_TEST_TEMPLATE`
  - [x] Export `build_integration_test_prompt`
  - [x] Export `build_integration_test_retry_prompt`

- [x] Task 9: Write Unit Tests for Prompt Templates (AC: 6)
  - [x] Create `tests/unit/agents/dev/prompts/test_integration_test_generation.py`
  - [x] Test prompt template rendering with variables
  - [x] Test that boundary testing guidance is included
  - [x] Test that error scenario requirements are included
  - [x] Test prompt structure follows expected format

- [x] Task 10: Write Unit Tests for Component Analysis (AC: 1, 7)
  - [x] Create tests for `analyze_component_boundaries()`
  - [x] Test detection of imports between files
  - [x] Test detection of function calls across modules
  - [x] Test detection of shared state access
  - [x] Test with realistic multi-file scenarios

- [x] Task 11: Write Unit Tests for Data Flow Analysis (AC: 2)
  - [x] Create tests for `analyze_data_flow()`
  - [x] Test tracing data through function calls
  - [x] Test identification of transformation points
  - [x] Test with complex multi-step flows

- [x] Task 12: Write Unit Tests for Error Scenario Detection (AC: 3)
  - [x] Create tests for `detect_error_scenarios()`
  - [x] Test detection of try/except patterns
  - [x] Test detection of fallback patterns
  - [x] Test detection of validation points

- [x] Task 13: Write Unit Tests for LLM Integration Test Generation (AC: 6)
  - [x] Test LLM integration with mock responses
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to stub on persistent failures
  - [x] Test syntax validation of generated tests

- [x] Task 14: Write Integration Tests for Full Flow (AC: 1-7)
  - [x] Create `tests/unit/agents/dev/test_integration_utils.py` (covers all integration utility tests)
  - [x] Test full flow from code files to integration test files
  - [x] Test boundary detection to test generation pipeline
  - [x] Test error scenario coverage in generated tests
  - [x] Test quality validation integration

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for new types (ComponentBoundary, DataFlowPath, ErrorScenario)
- **ADR-003 (LLM Provider):** Use LLMRouter with "complex" tier for integration test generation
- **ADR-005 (LangGraph Communication):** Maintain existing state update pattern
- **ADR-006 (Quality Gates):** DoD gate already integrated (Story 8.1)
- **ADR-007 (Error Handling):** Use tenacity for LLM retries with exponential backoff
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` in all new files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/dev/node.py` (Stories 8.1, 8.2, 8.3)
- All dataclasses should be frozen (immutable)
- Use tenacity @retry decorator with exponential backoff for LLM calls
- Integration tests must work with pytest-asyncio for async function testing

### Library Versions (from architecture.md)

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 1.0.5 | Orchestration framework |
| structlog | latest | Structured logging |
| tenacity | latest | Retry with backoff |
| LiteLLM | latest | Multi-provider LLM abstraction |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |

### Project Structure Notes

**New Files to Create:**
- `src/yolo_developer/agents/dev/prompts/integration_test_generation.py` - Integration test prompt templates
- `src/yolo_developer/agents/dev/integration_utils.py` - Component boundary and data flow analysis utilities

**Files to Modify:**
- `src/yolo_developer/agents/dev/node.py` - Update `_generate_tests()` to include integration tests
- `src/yolo_developer/agents/dev/prompts/__init__.py` - Export new prompts
- `src/yolo_developer/agents/dev/__init__.py` - Export new functions

**Test Files:**
- `tests/unit/agents/dev/prompts/test_integration_test_generation.py`
- `tests/unit/agents/dev/test_integration_utils.py`
- `tests/integration/agents/dev/test_integration_test_generation.py`

### Key Type Definitions

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ComponentBoundary:
    """Represents an interaction boundary between components.

    Attributes:
        source_file: Path to the source component file.
        target_file: Path to the target component file.
        interaction_type: Type of interaction (import, call, state_access).
        boundary_point: Specific function/class at the boundary.
        is_async: Whether the interaction is async.
    """
    source_file: str
    target_file: str
    interaction_type: Literal["import", "function_call", "state_access", "class_instantiation"]
    boundary_point: str
    is_async: bool


@dataclass(frozen=True)
class DataFlowPath:
    """Represents a data flow path through components.

    Attributes:
        start_point: Where data enters the flow.
        end_point: Where data exits the flow.
        steps: Sequence of transformation steps.
        data_types: Types observed at each step.
    """
    start_point: str
    end_point: str
    steps: tuple[str, ...]
    data_types: tuple[str, ...]


@dataclass(frozen=True)
class ErrorScenario:
    """Represents an error scenario to test.

    Attributes:
        trigger: What triggers the error condition.
        handling: How the error is handled.
        recovery: Recovery mechanism if any.
        exception_type: Type of exception raised/caught.
    """
    trigger: str
    handling: str
    recovery: str | None
    exception_type: str | None


@dataclass
class IntegrationTestQualityReport:
    """Report of integration test quality analysis.

    Note: Not frozen because warnings are appended incrementally during analysis.

    Attributes:
        warnings: List of quality warnings.
        uses_fixtures: Whether tests use pytest fixtures.
        uses_mocks: Whether external dependencies are mocked.
        has_cleanup: Whether tests clean up state.
        is_async_compliant: Whether async tests have proper markers.
    """
    warnings: list[str]
    uses_fixtures: bool = False
    uses_mocks: bool = False
    has_cleanup: bool = False
    is_async_compliant: bool = True

    def is_acceptable(self) -> bool:
        """Check if test quality is acceptable."""
        return self.uses_fixtures and self.uses_mocks and self.has_cleanup
```

### Integration Test Prompt Structure

```python
INTEGRATION_TEST_TEMPLATE = """
Generate pytest integration tests for the following Python components:

## Source Files:
{code_files_content}

## Component Boundaries Detected:
{boundaries}

## Data Flow Paths:
{data_flows}

## Error Scenarios:
{error_scenarios}

## Integration Testing Requirements:

1. **Boundary Testing:**
   - Test all detected component boundaries
   - Verify data is correctly passed between components
   - Test state updates propagate correctly

2. **Data Flow Verification:**
   - Test data transformations at each step
   - Verify data integrity across boundaries
   - Test with valid and invalid inputs

3. **Error Handling:**
   - Test graceful degradation paths
   - Verify error propagation works correctly
   - Test recovery mechanisms where applicable

4. **Test Structure:**
   - Use @pytest.mark.asyncio for async tests
   - Use fixtures for shared setup
   - Mock external dependencies (LLM, filesystem)
   - Clean up state after each test
   - Each test should be independent

5. **Naming Convention:**
   - test_<component>_<scenario>_<expected_behavior>
   - Use descriptive docstrings

Generate comprehensive pytest integration tests that:
1. Cover all detected boundaries
2. Verify all data flow paths
3. Test all error scenarios
4. Use proper fixtures and mocks
5. Are independent and deterministic
"""
```

### Previous Story Learnings Applied (Stories 8.2, 8.3)

From Story 8.2 (Maintainable Code Generation):
- LLM code generation with `_generate_code_with_llm()` pattern
- Syntax validation using `validate_python_syntax()` from code_utils.py
- Code extraction using `extract_code_from_response()` from code_utils.py
- LLMRouter initialization with `_get_llm_router()` pattern
- Retry prompt building pattern with lower temperature on retry

From Story 8.3 (Unit Test Generation):
- Test generation prompt structure with best practices
- `extract_public_functions()` for AST-based function extraction
- `validate_test_quality()` for test quality checking
- `calculate_coverage_estimate()` for coverage heuristics
- TestFile creation with proper test_type classification
- Fallback to stub tests when LLM unavailable

### Existing Dev Module Structure (to extend)

```
src/yolo_developer/agents/dev/
├── __init__.py         # Exports: dev_node, DevOutput, ImplementationArtifact, CodeFile, TestFile
├── types.py            # Type definitions (CodeFile, TestFile, etc.)
├── node.py             # dev_node function and helpers
├── code_utils.py       # Code validation and extraction utilities
├── test_utils.py       # Unit test generation utilities (Story 8.3)
├── integration_utils.py # NEW: Integration test utilities
└── prompts/
    ├── __init__.py     # Exports all prompts
    ├── code_generation.py   # Code generation prompts
    ├── test_generation.py   # Unit test prompts (Story 8.3)
    └── integration_test_generation.py  # NEW: Integration test prompts
```

### Key Imports for Implementation

```python
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.types import CodeFile, TestFile, CodeFileType
from yolo_developer.agents.dev.code_utils import (
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.test_utils import (
    extract_public_functions,
    FunctionInfo,
)
from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)
```

### Component Boundary Detection Approach

```python
def analyze_component_boundaries(code_files: list[CodeFile]) -> list[ComponentBoundary]:
    """Analyze code files to detect component interaction boundaries.

    Uses AST parsing to identify:
    1. Import statements between files
    2. Function calls to imported modules
    3. Class instantiations from other modules
    4. Async calls (await expressions)

    Args:
        code_files: List of CodeFile objects to analyze.

    Returns:
        List of ComponentBoundary objects representing detected boundaries.
    """
    boundaries: list[ComponentBoundary] = []

    # Build map of file -> module name
    file_modules = {cf.file_path: _extract_module_name(cf.file_path) for cf in code_files}

    for code_file in code_files:
        if code_file.file_type != "source":
            continue

        try:
            tree = ast.parse(code_file.content)
        except SyntaxError:
            continue

        # Find imports from other files in our codebase
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import module
                for alias in node.names:
                    if alias.name in file_modules.values():
                        # Found internal import
                        target_file = _find_file_for_module(alias.name, file_modules)
                        if target_file:
                            boundaries.append(ComponentBoundary(
                                source_file=code_file.file_path,
                                target_file=target_file,
                                interaction_type="import",
                                boundary_point=alias.name,
                                is_async=False,
                            ))

            elif isinstance(node, ast.ImportFrom):
                # Handle: from module import ...
                ...

    return boundaries
```

### Git Commit Pattern

Recent commits follow pattern:
```
feat: Implement <feature> with code review fixes (Story X.Y)
```

### Story Dependencies

This story builds on:
- Story 8.1 (Create Dev Agent Node) - dev_node foundation
- Story 8.2 (Maintainable Code Generation) - LLM integration patterns
- Story 8.3 (Unit Test Generation) - test generation patterns, TestFile type

This story enables:
- Story 8.5 (Documentation Generation) - docs for tested code
- Story 9.2 (TEA Coverage Validation) - TEA validates test coverage including integration tests

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR59 | Dev Agent can write integration tests for cross-component functionality | LLM-powered integration test generation with boundary analysis |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.4] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-003] - LLM provider abstraction
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007] - Error handling patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#Test-Organization] - Test structure patterns
- [Source: src/yolo_developer/agents/dev/node.py] - Existing dev node (Stories 8.1-8.3)
- [Source: src/yolo_developer/agents/dev/test_utils.py] - Unit test utilities (Story 8.3)
- [Source: src/yolo_developer/agents/dev/prompts/test_generation.py] - Unit test prompts (pattern)
- [Source: tests/integration/agents/dev/] - Existing integration test examples
- [Source: _bmad-output/implementation-artifacts/8-3-unit-test-generation.md] - Previous story learnings
- [FR59: Dev Agent can write integration tests for cross-component functionality]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 14 tasks completed following red-green-refactor TDD cycle
- 50 new unit tests added (18 for prompts, 32 for integration utils)
- All 319 dev agent tests pass
- mypy type checking passes with strict mode
- ruff linting passes with no issues
- Integration tests generate automatically when multiple source files detected
- Stub fallback implemented for when LLM is unavailable
- Architecture compliance verified: ADR-001, ADR-003, ADR-005, ADR-006, ADR-007

### File List

**New Files Created:**
- `src/yolo_developer/agents/dev/prompts/integration_test_generation.py` - Integration test prompt templates
- `src/yolo_developer/agents/dev/integration_utils.py` - Component boundary analysis, data flow analysis, error detection, LLM generation
- `tests/unit/agents/dev/prompts/test_integration_test_generation.py` - 18 unit tests for prompts
- `tests/unit/agents/dev/test_integration_utils.py` - 32 unit tests for integration utilities

**Files Modified:**
- `src/yolo_developer/agents/dev/node.py` - Added `_generate_integration_tests()`, `_generate_stub_integration_test()`, updated `_generate_tests()`
- `src/yolo_developer/agents/dev/prompts/__init__.py` - Export new integration test prompts
