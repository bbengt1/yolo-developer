# Story 8.5: Documentation Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want code documentation generated,
So that future maintainers understand the code.

## Acceptance Criteria

1. **AC1: Public API Docstrings**
   - **Given** generated implementation code with public functions/classes
   - **When** documentation generation runs
   - **Then** all public functions have Google-style docstrings
   - **And** docstrings include Args, Returns, and Example sections
   - **And** docstrings accurately describe function behavior
   - **And** type information in docstrings matches type annotations

2. **AC2: Complex Logic Comments**
   - **Given** implementation code with complex logic (nested loops, algorithms, conditionals)
   - **When** documentation generation analyzes the code
   - **Then** explanatory comments are added above complex sections
   - **And** comments explain the "why" not just the "what"
   - **And** comments reference relevant acceptance criteria or requirements
   - **And** comments do not duplicate information obvious from code

3. **AC3: Module-Level Documentation**
   - **Given** a new module being implemented
   - **When** documentation generation completes
   - **Then** module has a comprehensive docstring at the top
   - **And** docstring includes one-line summary, purpose description, key concepts
   - **And** docstring includes Example section with runnable code
   - **And** docstring follows the codebase's existing module docstring pattern

4. **AC4: Project Convention Compliance**
   - **Given** the project's existing documentation patterns (Google-style docstrings)
   - **When** documentation is generated
   - **Then** docstrings follow Google docstring format (Args, Returns, Raises, Example)
   - **And** documentation is consistent with existing codebase documentation style
   - **And** docstrings use consistent terminology from the project
   - **And** generated docs pass the Definition of Done gate documentation checks

5. **AC5: LLM-Powered Documentation Enhancement**
   - **Given** implementation code that may have minimal or no documentation
   - **When** `generate_documentation_with_llm()` is called
   - **Then** LLM enhances code with comprehensive documentation
   - **And** LLM calls use "complex" tier per ADR-003
   - **And** LLM calls use tenacity retry with exponential backoff per ADR-007
   - **And** generated documentation is syntax-validated before returning

6. **AC6: Documentation Quality Validation**
   - **Given** generated documentation
   - **When** `validate_documentation_quality()` runs
   - **Then** docstring completeness is verified (Args, Returns, Example present)
   - **And** type consistency between annotations and docstrings is checked
   - **And** quality warnings are generated for incomplete documentation
   - **And** validation report indicates if documentation meets standards

## Tasks / Subtasks

- [x] Task 1: Create Documentation Prompt Templates (AC: 1, 3, 4, 5)
  - [x] Create `src/yolo_developer/agents/dev/prompts/documentation_generation.py`
  - [x] Create `DOCUMENTATION_GUIDELINES` constant with Google-style docstring requirements
  - [x] Create `MODULE_DOCSTRING_TEMPLATE` with module documentation guidance
  - [x] Create `FUNCTION_DOCSTRING_TEMPLATE` with function documentation guidance
  - [x] Create `DOCUMENTATION_GENERATION_TEMPLATE` for LLM documentation enhancement
  - [x] Create `build_documentation_prompt()` function
  - [x] Create `build_documentation_retry_prompt()` for syntax recovery

- [x] Task 2: Implement Documentation Analysis Utilities (AC: 1, 2, 6)
  - [x] Create `src/yolo_developer/agents/dev/doc_utils.py`
  - [x] Create `extract_documentation_info(code: str) -> DocumentationInfo` using AST
  - [x] Detect functions missing docstrings
  - [x] Detect module missing docstring
  - [x] Identify complex code sections needing comments (nesting depth, loop complexity)
  - [x] Create `DocumentationInfo` dataclass with analysis results

- [x] Task 3: Implement Docstring Generation (AC: 1, 3, 5)
  - [x] Create `generate_documentation_with_llm()` for comprehensive docstring generation
  - [x] Use LLM to generate comprehensive docstrings matching existing patterns
  - [x] Ensure generated docstrings include Args, Returns, Example sections
  - [x] Validate generated docstrings are syntactically correct

- [x] Task 4: Implement LLM Documentation Enhancement (AC: 5)
  - [x] Create `generate_documentation_with_llm(code: str, context: str, router: LLMRouter) -> tuple[str, bool]`
  - [x] Integrate with LLMRouter using "complex" tier per ADR-003
  - [x] Apply tenacity retry pattern per ADR-007
  - [x] Include documentation analysis in prompt
  - [x] Return enhanced code with documentation and validation status

- [x] Task 5: Implement Complex Logic Comment Detection (AC: 2)
  - [x] Create `detect_complex_sections(code: str) -> list[ComplexSection]`
  - [x] Identify nested loops (depth >= 2)
  - [x] Identify long functions (> 20 lines)
  - [x] Identify complex conditionals (multiple branches)
  - [x] Create `ComplexSection` dataclass with location and complexity type

- [x] Task 6: Implement Documentation Quality Validation (AC: 4, 6)
  - [x] Create `validate_documentation_quality(code: str) -> DocumentationQualityReport`
  - [x] Check for Args section presence in function docstrings
  - [x] Check for Returns section presence
  - [x] Check for Example section presence
  - [x] Verify type consistency between annotations and docstrings
  - [x] Create `DocumentationQualityReport` dataclass with warnings and metrics

- [x] Task 7: Update `_generate_implementation` to Include Documentation (AC: 1-6)
  - [x] Modify `node.py` to enhance generated code with documentation
  - [x] Add documentation generation step after code generation
  - [x] Ensure documentation is added before test generation
  - [x] Log documentation generation metrics
  - [x] Fall back to minimal docs if LLM enhancement fails

- [x] Task 8: Export New Functions from Prompts Module (AC: 5)
  - [x] Update `src/yolo_developer/agents/dev/prompts/__init__.py`
  - [x] Export `DOCUMENTATION_GUIDELINES`
  - [x] Export `DOCUMENTATION_GENERATION_TEMPLATE`
  - [x] Export `build_documentation_prompt`
  - [x] Export `build_documentation_retry_prompt`

- [x] Task 9: Export New Functions from Dev Module (AC: 1-6)
  - [x] Update `src/yolo_developer/agents/dev/__init__.py`
  - [x] Export documentation utilities
  - [x] Export `DocumentationInfo`, `DocumentationQualityReport`, `ComplexSection`
  - [x] Update module docstring to include Story 8.5

- [x] Task 10: Write Unit Tests for Prompt Templates (AC: 5)
  - [x] Create `tests/unit/agents/dev/prompts/test_documentation_generation.py`
  - [x] Test prompt template rendering with variables
  - [x] Test that Google-style docstring guidance is included
  - [x] Test that module docstring requirements are included
  - [x] Test prompt structure follows expected format

- [x] Task 11: Write Unit Tests for Documentation Analysis (AC: 1, 2, 6)
  - [x] Create `tests/unit/agents/dev/test_doc_utils.py`
  - [x] Test detection of functions missing docstrings
  - [x] Test detection of missing module docstring
  - [x] Test complex section detection (nested loops, long functions)
  - [x] Test with realistic multi-function code files

- [x] Task 12: Write Unit Tests for Documentation Generation (AC: 1, 3, 5)
  - [x] Test module docstring generation produces valid docstrings
  - [x] Test function docstring generation includes required sections
  - [x] Test generated docstrings are syntactically valid
  - [x] Test LLM integration with mock responses

- [x] Task 13: Write Unit Tests for Quality Validation (AC: 4, 6)
  - [x] Test quality validation detects missing Args sections
  - [x] Test quality validation detects missing Returns sections
  - [x] Test quality validation detects missing Examples
  - [x] Test quality validation verifies type consistency
  - [x] Test quality report structure and warnings

- [x] Task 14: Write Integration Tests for Full Flow (AC: 1-6)
  - [x] Create `tests/integration/agents/dev/test_documentation_generation.py`
  - [x] Test full flow from code to documented code
  - [x] Test documentation analysis to generation pipeline
  - [x] Test quality validation integration
  - [x] Test with multi-file scenarios

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for new types (DocumentationInfo, ComplexSection, DocumentationQualityReport)
- **ADR-003 (LLM Provider):** Use LLMRouter with "complex" tier for documentation generation
- **ADR-005 (LangGraph Communication):** Maintain existing state update pattern
- **ADR-006 (Quality Gates):** DoD gate already validates documentation (Story 8.1)
- **ADR-007 (Error Handling):** Use tenacity for LLM retries with exponential backoff
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` in all new files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/dev/node.py` (Stories 8.1-8.4)
- All dataclasses should be frozen (immutable)
- Use tenacity @retry decorator with exponential backoff for LLM calls
- Documentation must follow Google-style docstring format

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
- `src/yolo_developer/agents/dev/prompts/documentation_generation.py` - Documentation prompt templates
- `src/yolo_developer/agents/dev/doc_utils.py` - Documentation analysis and generation utilities

**Files to Modify:**
- `src/yolo_developer/agents/dev/node.py` - Add documentation enhancement to code generation flow
- `src/yolo_developer/agents/dev/prompts/__init__.py` - Export new prompts
- `src/yolo_developer/agents/dev/__init__.py` - Export new functions and types

**Test Files:**
- `tests/unit/agents/dev/prompts/test_documentation_generation.py`
- `tests/unit/agents/dev/test_doc_utils.py`
- `tests/integration/agents/dev/test_documentation_generation.py`

### Key Type Definitions

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class DocumentationInfo:
    """Analysis results for code documentation status.

    Attributes:
        has_module_docstring: Whether module has a docstring.
        functions_missing_docstrings: List of function names without docstrings.
        functions_with_incomplete_docstrings: Functions missing Args/Returns/Example.
        complex_sections: List of code sections needing explanatory comments.
        total_public_functions: Total count of public functions.
        documented_functions: Count of functions with docstrings.
    """
    has_module_docstring: bool
    functions_missing_docstrings: tuple[str, ...]
    functions_with_incomplete_docstrings: tuple[str, ...]
    complex_sections: tuple[ComplexSection, ...]
    total_public_functions: int
    documented_functions: int

    @property
    def documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage."""
        if self.total_public_functions == 0:
            return 100.0
        return (self.documented_functions / self.total_public_functions) * 100


@dataclass(frozen=True)
class ComplexSection:
    """Represents a complex code section that may need comments.

    Attributes:
        start_line: Line number where complex section starts.
        end_line: Line number where complex section ends.
        complexity_type: Type of complexity detected.
        function_name: Name of containing function, if any.
        description: Brief description of the complexity.
    """
    start_line: int
    end_line: int
    complexity_type: Literal["nested_loop", "long_function", "complex_conditional", "deep_nesting"]
    function_name: str | None
    description: str


@dataclass
class DocumentationQualityReport:
    """Report of documentation quality analysis.

    Note: Not frozen because warnings are appended incrementally during analysis.

    Attributes:
        warnings: List of quality warnings.
        has_module_docstring: Whether module docstring exists.
        functions_with_args: Count of functions with Args section.
        functions_with_returns: Count of functions with Returns section.
        functions_with_examples: Count of functions with Example section.
        total_functions: Total functions analyzed.
        type_consistency_issues: Functions where docstring types don't match annotations.
    """
    warnings: list[str] = field(default_factory=list)
    has_module_docstring: bool = False
    functions_with_args: int = 0
    functions_with_returns: int = 0
    functions_with_examples: int = 0
    total_functions: int = 0
    type_consistency_issues: list[str] = field(default_factory=list)

    def is_acceptable(self) -> bool:
        """Check if documentation quality is acceptable.

        Documentation is acceptable if module has docstring and
        at least 80% of functions have Args and Returns sections.
        """
        if not self.has_module_docstring:
            return False
        if self.total_functions == 0:
            return True
        args_coverage = self.functions_with_args / self.total_functions
        returns_coverage = self.functions_with_returns / self.total_functions
        return args_coverage >= 0.8 and returns_coverage >= 0.8
```

### Documentation Generation Prompt Structure

```python
DOCUMENTATION_GENERATION_TEMPLATE = """You are a senior Python developer adding comprehensive documentation.

## Source Code to Document

```python
{code_content}
```

## Documentation Analysis

{documentation_analysis}

## Complex Sections Needing Comments

{complex_sections}

## Documentation Requirements

### Module Docstring Requirements
- One-line summary describing module purpose
- Paragraph explaining functionality and use cases
- Key Concepts/Functions section listing main exports
- Example section with runnable doctest code

### Function Docstring Requirements (Google Style)
- One-line summary ending with period
- Extended description if behavior is non-obvious
- Args section with type and description for each parameter
- Returns section describing return value and type
- Raises section if function raises exceptions
- Example section with doctest-style code

### Comment Requirements for Complex Logic
- Add inline comments above complex sections
- Explain the "why" not just the "what"
- Reference acceptance criteria where applicable
- Keep comments concise but informative

## Instructions

Enhance the code with comprehensive documentation:
1. Add or improve module docstring following the pattern above
2. Add or improve function docstrings with Args, Returns, Example
3. Add explanatory comments for complex sections
4. Ensure type information in docstrings matches annotations
5. Use terminology consistent with existing codebase

Output the fully documented Python code. Do not include explanations outside the code.
Wrap the code in ```python and ``` markers.
"""
```

### Previous Story Learnings Applied (Stories 8.2, 8.3, 8.4)

From Story 8.2 (Maintainable Code Generation):
- LLM code generation with `_generate_code_with_llm()` pattern
- Syntax validation using `validate_python_syntax()` from code_utils.py
- Code extraction using `extract_code_from_response()` from code_utils.py
- LLMRouter initialization with `_get_llm_router()` pattern

From Story 8.3 (Unit Test Generation):
- Test generation prompt structure with best practices
- `extract_public_functions()` for AST-based function extraction
- Quality validation patterns (`validate_test_quality()`)
- Fallback to stub when LLM unavailable

From Story 8.4 (Integration Test Generation):
- Component analysis using AST parsing
- Data flow analysis patterns
- Quality report dataclass pattern
- LLM retry with tenacity pattern

### Existing Documentation Validation in DoD Gate

The Definition of Done gate (`gates/gates/definition_of_done.py`) already validates:
- Module-level docstring presence: `_has_module_docstring(content: str) -> bool`
- Function docstring presence via AST: `_extract_functions_from_content(content: str)`
- Uses `ast.get_docstring(tree)` and `ast.get_docstring(node)`

Story 8.5 documentation generation should ensure generated docs pass these existing checks.

### Existing Dev Module Structure (to extend)

```
src/yolo_developer/agents/dev/
├── __init__.py         # Exports: dev_node, DevOutput, CodeFile, TestFile, etc.
├── types.py            # Type definitions
├── node.py             # dev_node function and helpers
├── code_utils.py       # Code validation and extraction utilities
├── test_utils.py       # Unit test generation utilities (Story 8.3)
├── integration_utils.py # Integration test utilities (Story 8.4)
├── doc_utils.py        # NEW: Documentation utilities
└── prompts/
    ├── __init__.py     # Exports all prompts
    ├── code_generation.py
    ├── test_generation.py
    ├── integration_test_generation.py
    └── documentation_generation.py  # NEW: Documentation prompts
```

### Key Imports for Implementation

```python
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.types import CodeFile
from yolo_developer.agents.dev.code_utils import (
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.test_utils import extract_public_functions, FunctionInfo
from yolo_developer.llm.router import LLMRouter

logger = structlog.get_logger(__name__)
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
- Story 8.3 (Unit Test Generation) - AST extraction patterns
- Story 8.4 (Integration Test Generation) - Quality report patterns

This story enables:
- Story 8.6 (DoD Validation) - DoD validates documentation quality
- Story 9.x (TEA Agent) - TEA can assess documentation coverage

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR60 | Dev Agent can generate code documentation and comments | LLM-powered documentation enhancement with quality validation |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.5] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-003] - LLM provider abstraction
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007] - Error handling patterns
- [Source: src/yolo_developer/agents/dev/node.py] - Existing dev node (Stories 8.1-8.4)
- [Source: src/yolo_developer/agents/dev/code_utils.py] - Code utilities
- [Source: src/yolo_developer/agents/dev/test_utils.py] - Test utilities (pattern)
- [Source: src/yolo_developer/agents/dev/prompts/code_generation.py] - Code generation prompts (pattern)
- [Source: src/yolo_developer/gates/gates/definition_of_done.py] - DoD documentation checks
- [Source: _bmad-output/implementation-artifacts/8-4-integration-test-generation.md] - Previous story learnings
- [FR60: Dev Agent can generate code documentation and comments]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- All 14 tasks implemented following TDD red-green-refactor
- 86 tests pass (30 prompt tests, 42 doc_utils tests, 14 integration tests)
- mypy --strict passes
- ruff check passes
- Code review fixes applied: reraise=True added to retry decorator, retry params standardized

### File List

**New Files:**
- `src/yolo_developer/agents/dev/doc_utils.py` - Documentation analysis and generation utilities
- `src/yolo_developer/agents/dev/prompts/documentation_generation.py` - Documentation prompt templates
- `tests/unit/agents/dev/test_doc_utils.py` - Unit tests for doc_utils (42 tests)
- `tests/unit/agents/dev/prompts/test_documentation_generation.py` - Unit tests for prompts (30 tests)
- `tests/integration/agents/dev/test_documentation_generation.py` - Integration tests (14 tests)

**Modified Files:**
- `src/yolo_developer/agents/dev/node.py` - Added `_enhance_documentation()`, integrated into pipeline
- `src/yolo_developer/agents/dev/__init__.py` - Exported doc_utils types and functions
- `src/yolo_developer/agents/dev/prompts/__init__.py` - Exported documentation templates
