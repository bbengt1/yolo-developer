# Story 8.6: DoD Validation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want code validated against Definition of Done,
So that incomplete work doesn't proceed.

## Acceptance Criteria

1. **AC1: DoD Checklist Verification**
   - **Given** completed code for a story with tests and documentation
   - **When** DoD validation runs via `validate_dod()` function
   - **Then** all checklist items from `DOD_CHECKLIST_ITEMS` are verified
   - **And** each category (tests, documentation, style, ac_coverage) is evaluated
   - **And** verification results are returned in structured format
   - **And** results include pass/fail status for each checklist item

2. **AC2: Test Presence Confirmation**
   - **Given** story implementation with code files
   - **When** test presence is validated
   - **Then** `check_test_presence()` verifies test files exist for all source files
   - **And** public functions have corresponding test functions
   - **And** missing test issues have severity "high" (blocking)
   - **And** the validation reuses existing logic from `definition_of_done.py`

3. **AC3: Documentation Completeness Check**
   - **Given** story implementation code
   - **When** documentation completeness is validated
   - **Then** `check_documentation()` verifies module docstrings exist
   - **And** public function docstrings are verified
   - **And** missing documentation issues have severity "medium" (warning)
   - **And** the validation reuses existing logic from `definition_of_done.py`

4. **AC4: Style Compliance Validation**
   - **Given** story implementation code
   - **When** style compliance is validated
   - **Then** `check_code_style()` verifies type annotations exist
   - **And** naming conventions (snake_case) are verified
   - **And** function complexity (length, nesting) is checked
   - **And** the validation reuses existing logic from `definition_of_done.py`

5. **AC5: Dev Node DoD Gate Integration**
   - **Given** the dev agent node produces implementation
   - **When** the dev_node function completes
   - **Then** the `@quality_gate("definition_of_done", blocking=False)` decorator validates output
   - **And** gate results are logged to audit trail
   - **And** gate failure does NOT block dev_node (blocking=False per current implementation)
   - **And** gate failure reasons are included in the decision record

6. **AC6: Programmatic DoD Validation**
   - **Given** code that needs DoD validation outside of gate context
   - **When** `validate_implementation_dod()` is called with code and story data
   - **Then** all DoD checks are run: tests, docs, style, AC coverage
   - **And** results are returned as `DoDValidationResult` dataclass
   - **And** result includes overall pass/fail, score, and itemized checklist
   - **And** function is usable from dev agent and externally

7. **AC7: Dev Node Output Validation**
   - **Given** dev agent produces `DevOutput` with implementations
   - **When** DoD validation runs on the output
   - **Then** each `ImplementationArtifact` is validated
   - **And** validation uses code_files and test_files from artifact
   - **And** validation considers story acceptance criteria (from state)
   - **And** aggregate results are available for all artifacts

## Tasks / Subtasks

- [x] Task 1: Create DoD Validation Types (AC: 1, 6, 7)
  - [x] Create `src/yolo_developer/agents/dev/dod_utils.py`
  - [x] Create `DoDValidationResult` dataclass with fields: passed, score, checklist, issues
  - [x] Create `DoDChecklistItem` dataclass with fields: category, item_name, passed, severity, message
  - [x] Add type annotations for all new types

- [x] Task 2: Implement Programmatic DoD Validation (AC: 1, 6)
  - [x] Create `validate_implementation_dod(code: dict, story: dict) -> DoDValidationResult`
  - [x] Import and reuse `check_test_presence()` from `gates/gates/definition_of_done.py`
  - [x] Import and reuse `check_documentation()` from `gates/gates/definition_of_done.py`
  - [x] Import and reuse `check_code_style()` from `gates/gates/definition_of_done.py`
  - [x] Import and reuse `check_ac_coverage()` from `gates/gates/definition_of_done.py`
  - [x] Aggregate results into `DoDValidationResult`

- [x] Task 3: Implement Artifact-Level Validation (AC: 7)
  - [x] Create `validate_artifact_dod(artifact: ImplementationArtifact, story: dict) -> DoDValidationResult`
  - [x] Convert `ImplementationArtifact.code_files` to dict format expected by existing checks
  - [x] Convert `ImplementationArtifact.test_files` to dict format expected by existing checks
  - [x] Call `validate_implementation_dod()` with converted data

- [x] Task 4: Implement DevOutput Aggregate Validation (AC: 7)
  - [x] Create `validate_dev_output_dod(output: DevOutput, state: YoloState) -> list[DoDValidationResult]`
  - [x] Extract story data from state for AC coverage validation
  - [x] Validate each `ImplementationArtifact` in output
  - [x] Return list of validation results (one per artifact)

- [x] Task 5: Verify Gate Integration Works (AC: 5)
  - [x] Review existing `@quality_gate("definition_of_done", blocking=False)` on `dev_node`
  - [x] Verify gate receives appropriate state (code in `dev_output`)
  - [x] Add logging for gate evaluation in node
  - [x] Update decision record to include gate result summary if failed

- [x] Task 6: Export Functions from Dev Module (AC: 6)
  - [x] Update `src/yolo_developer/agents/dev/__init__.py`
  - [x] Export `validate_implementation_dod`
  - [x] Export `validate_artifact_dod`
  - [x] Export `validate_dev_output_dod`
  - [x] Export `DoDValidationResult`, `DoDChecklistItem`

- [x] Task 7: Write Unit Tests for DoD Validation Types (AC: 1, 6)
  - [x] Create `tests/unit/agents/dev/test_dod_utils.py`
  - [x] Test `DoDValidationResult` dataclass construction
  - [x] Test `DoDChecklistItem` dataclass construction
  - [x] Test `passed` property based on score threshold

- [x] Task 8: Write Unit Tests for Programmatic Validation (AC: 1, 2, 3, 4, 6)
  - [x] Test `validate_implementation_dod()` with complete implementation
  - [x] Test `validate_implementation_dod()` with missing tests
  - [x] Test `validate_implementation_dod()` with missing documentation
  - [x] Test `validate_implementation_dod()` with style violations
  - [x] Test `validate_implementation_dod()` with missing AC coverage

- [x] Task 9: Write Unit Tests for Artifact Validation (AC: 7)
  - [x] Test `validate_artifact_dod()` converts code_files correctly
  - [x] Test `validate_artifact_dod()` converts test_files correctly
  - [x] Test `validate_artifact_dod()` with valid artifact passes

- [x] Task 10: Write Unit Tests for DevOutput Validation (AC: 7)
  - [x] Test `validate_dev_output_dod()` validates all artifacts
  - [x] Test `validate_dev_output_dod()` extracts story from state
  - [x] Test `validate_dev_output_dod()` with multiple artifacts

- [x] Task 11: Write Integration Tests for Gate Flow (AC: 5)
  - [x] Create `tests/integration/agents/dev/test_dod_validation.py`
  - [x] Test dev_node gate evaluation with valid output
  - [x] Test dev_node gate evaluation with failing DoD
  - [x] Verify gate result in decision record

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses where immutability is required. Note: `DoDValidationResult` is intentionally NOT frozen to allow checklist items to be appended during validation, while `DoDChecklistItem` IS frozen for audit trail integrity.
- **ADR-005 (LangGraph Communication):** Gate decorator already integrated on dev_node
- **ADR-006 (Quality Gates):** Reuse existing gate infrastructure from Story 3.5
- **ADR-007 (Error Handling):** Use tenacity retry (already on dev_node)
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` in all new files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/dev/node.py` (Stories 8.1-8.5)
- All dataclasses should be frozen (immutable) except where incremental building needed
- REUSE existing `check_*` functions from `gates/gates/definition_of_done.py`
- Do NOT duplicate the validation logic - import and call existing functions

### Library Versions (from architecture.md)

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 1.0.5 | Orchestration framework |
| structlog | latest | Structured logging |
| tenacity | latest | Retry with backoff |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |

### Project Structure Notes

**New Files to Create:**
- `src/yolo_developer/agents/dev/dod_utils.py` - DoD validation utilities

**Files to Modify:**
- `src/yolo_developer/agents/dev/__init__.py` - Export new functions and types

**Test Files:**
- `tests/unit/agents/dev/test_dod_utils.py`
- `tests/integration/agents/dev/test_dod_validation.py`

### Key Type Definitions

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yolo_developer.gates.report_types import GateIssue


@dataclass(frozen=True)
class DoDChecklistItem:
    """Single item in DoD checklist.

    Attributes:
        category: Category from DOD_CHECKLIST_ITEMS (tests, documentation, style, ac_coverage).
        item_name: Specific checklist item name.
        passed: Whether this item passed validation.
        severity: Issue severity if failed (high=blocking, medium=warning, low=info).
        message: Human-readable description of result.
    """
    category: Literal["tests", "documentation", "style", "ac_coverage"]
    item_name: str
    passed: bool
    severity: Literal["high", "medium", "low"] | None
    message: str


@dataclass
class DoDValidationResult:
    """Result of DoD validation for an implementation.

    Note: Not frozen because checklist items are appended during validation.

    Attributes:
        score: Compliance score 0-100 using SEVERITY_WEIGHTS.
        passed: Whether score meets threshold (default 70).
        threshold: Score threshold used for pass/fail.
        checklist: List of individual checklist results.
        issues: List of GateIssue objects for failures.
        artifact_id: Optional story/artifact ID being validated.
    """
    score: int = 100
    passed: bool = True
    threshold: int = 70
    checklist: list[DoDChecklistItem] = field(default_factory=list)
    issues: list[GateIssue] = field(default_factory=list)
    artifact_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "checklist": [
                {
                    "category": item.category,
                    "item_name": item.item_name,
                    "passed": item.passed,
                    "severity": item.severity,
                    "message": item.message,
                }
                for item in self.checklist
            ],
            "issue_count": len(self.issues),
            "artifact_id": self.artifact_id,
        }
```

### Key Function Signature

```python
def validate_implementation_dod(
    code: dict[str, Any],
    story: dict[str, Any],
    threshold: int = 70,
) -> DoDValidationResult:
    """Validate implementation against Definition of Done checklist.

    Runs all DoD checks: test presence, documentation, style, AC coverage.
    Reuses existing check functions from gates/gates/definition_of_done.py.

    Args:
        code: Code artifact dict with 'files' key containing list of file dicts.
              Each file dict has 'path' and 'content' keys.
        story: Story dict with optional 'acceptance_criteria' key.
        threshold: Minimum score to pass (0-100, default 70).

    Returns:
        DoDValidationResult with score, pass/fail, and itemized checklist.

    Example:
        >>> code = {
        ...     "files": [
        ...         {"path": "src/impl.py", "content": "def foo(): pass"},
        ...         {"path": "tests/test_impl.py", "content": "def test_foo(): pass"},
        ...     ]
        ... }
        >>> story = {"acceptance_criteria": ["AC1: foo works"]}
        >>> result = validate_implementation_dod(code, story)
        >>> result.passed
        True
    """
```

### Reusing Existing DoD Gate Functions

The existing `gates/gates/definition_of_done.py` already has all the validation logic:

```python
# Import these existing functions - DO NOT REIMPLEMENT
from yolo_developer.gates.gates.definition_of_done import (
    check_test_presence,      # Returns list[GateIssue]
    check_documentation,      # Returns list[GateIssue]
    check_code_style,         # Returns list[GateIssue]
    check_ac_coverage,        # Returns list[GateIssue]
    generate_dod_checklist,   # Returns dict with score, breakdown, categorized issues
    SEVERITY_WEIGHTS,         # {"high": 20, "medium": 10, "low": 3}
    DEFAULT_DOD_THRESHOLD,    # 0.70
)
```

The `validate_implementation_dod()` function should:
1. Call each `check_*` function with appropriate arguments
2. Combine all issues
3. Call `generate_dod_checklist()` to get score
4. Build `DoDValidationResult` from the results

### Converting ImplementationArtifact to Code Dict

```python
def _artifact_to_code_dict(artifact: ImplementationArtifact) -> dict[str, Any]:
    """Convert ImplementationArtifact to dict format for DoD checks.

    Args:
        artifact: Implementation artifact with code_files and test_files.

    Returns:
        Dict with 'files' key containing list of file dicts.
    """
    files = []

    # Add code files
    for cf in artifact.code_files:
        files.append({
            "path": cf.file_path,
            "content": cf.content,
        })

    # Add test files
    for tf in artifact.test_files:
        files.append({
            "path": tf.file_path,
            "content": tf.content,
        })

    return {"files": files}
```

### Previous Story Learnings Applied (Stories 8.1-8.5)

From Story 8.1 (Create Dev Agent Node):
- `dev_node` already has `@quality_gate("definition_of_done", blocking=False)` decorator
- Gate decorator calls `definition_of_done_evaluator()` automatically
- Gate results are logged via structlog

From Story 8.5 (Documentation Generation):
- `DocumentationQualityReport` pattern for validation results
- `is_acceptable()` method pattern for pass/fail determination
- `to_dict()` method for serialization

From existing Definition of Done gate (Story 3.5):
- Full implementation exists in `gates/gates/definition_of_done.py`
- `check_test_presence()` - verifies test files and function coverage
- `check_documentation()` - verifies module and function docstrings
- `check_code_style()` - verifies types, naming, complexity
- `check_ac_coverage()` - verifies acceptance criteria addressed
- `generate_dod_checklist()` - aggregates issues into scored checklist

### Gate Integration Architecture

The dev_node already has the gate decorator:
```python
@retry(...)
@quality_gate("definition_of_done", blocking=False)
async def dev_node(state: YoloState) -> dict[str, Any]:
```

The gate decorator:
1. Calls `definition_of_done_evaluator(context)` with state
2. Evaluator extracts `code` from state (tries `state["code"]` then `state["implementation"]`)
3. Evaluator extracts `story` from state
4. Runs all checks and returns `GateResult`

For Story 8.6, we need to:
1. Ensure `dev_output` is accessible to the gate (it's returned in state update)
2. Create utility functions for validating artifacts programmatically
3. The gate already works - we're adding programmatic access outside gate context

### Existing Dev Module Structure (to extend)

```
src/yolo_developer/agents/dev/
├── __init__.py         # Exports: dev_node, DevOutput, CodeFile, TestFile, etc.
├── types.py            # Type definitions
├── node.py             # dev_node function and helpers
├── code_utils.py       # Code validation and extraction utilities
├── test_utils.py       # Unit test generation utilities (Story 8.3)
├── integration_utils.py # Integration test utilities (Story 8.4)
├── doc_utils.py        # Documentation utilities (Story 8.5)
├── dod_utils.py        # NEW: DoD validation utilities
└── prompts/
    ├── __init__.py
    └── ...
```

### Key Imports for Implementation

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from yolo_developer.agents.dev.types import (
    CodeFile,
    DevOutput,
    ImplementationArtifact,
    TestFile,
)
from yolo_developer.gates.gates.definition_of_done import (
    check_ac_coverage,
    check_code_style,
    check_documentation,
    check_test_presence,
    generate_dod_checklist,
    DEFAULT_DOD_THRESHOLD,
    SEVERITY_WEIGHTS,
)
from yolo_developer.gates.report_types import GateIssue
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)
```

### Git Commit Pattern

Recent commits follow pattern:
```
feat: Implement <feature> with code review fixes (Story X.Y)
```

### Story Dependencies

This story builds on:
- Story 3.5 (Definition of Done Gate) - provides all validation logic
- Story 8.1 (Create Dev Agent Node) - gate decorator already integrated
- Story 8.2-8.5 - dev_node produces DevOutput with artifacts

This story enables:
- Story 9.x (TEA Agent) - TEA can use DoD validation for quality assessment
- Dev workflow completion - programmatic DoD checks outside gate context

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR61 | Dev Agent can validate code against Definition of Done checklist | Programmatic `validate_implementation_dod()` + gate integration |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.6] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate pattern
- [Source: src/yolo_developer/gates/gates/definition_of_done.py] - Existing DoD gate implementation
- [Source: src/yolo_developer/agents/dev/node.py] - Existing dev node with gate decorator
- [Source: src/yolo_developer/agents/dev/types.py] - DevOutput, ImplementationArtifact types
- [Source: _bmad-output/implementation-artifacts/8-5-documentation-generation.md] - Previous story learnings
- [FR61: Dev Agent can validate code against Definition of Done checklist]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation completed successfully without blocking issues.

### Completion Notes List

- Created `dod_utils.py` with DoD validation types and programmatic validation functions
- Implemented `DoDChecklistItem` (frozen dataclass) and `DoDValidationResult` (mutable for appending)
- Created `validate_implementation_dod()` that REUSES existing `check_*` functions from `definition_of_done.py`
- Created `validate_artifact_dod()` for ImplementationArtifact validation
- Created `validate_dev_output_dod()` for DevOutput aggregate validation
- Updated `dev_node` to log gate warnings and include them in decision rationale (AC5)
- Exported all new types and functions from `agents/dev/__init__.py`
- 21 unit tests + 13 integration tests = 34 total tests, all passing
- All acceptance criteria satisfied through red-green-refactor TDD cycle
- Linting and type checking pass (ruff, mypy)

**Code Review Fixes Applied:**
- H1: Added `validate_dod` alias for API consistency with AC1 documentation
- H2: Updated `validate_implementation_dod()` to build complete checklist with ALL 11 DOD_CHECKLIST_ITEMS (not just failures)
- H3: Clarified Dev Notes that `DoDValidationResult` is intentionally NOT frozen
- M1: Added sprint-status.yaml to File List
- M2: Added error handling with try/except around all `check_*` function calls
- M3: Added explicit audit trail logging via `logger.info("dod_validation_audit", ...)`
- M4: Added 10 edge case tests for empty files, missing keys, alias verification, and complete checklist
- Final test count: 31 unit tests + 13 integration tests = 44 total tests, all passing

### File List

**New Files:**
- `src/yolo_developer/agents/dev/dod_utils.py` - DoD validation utilities
- `tests/unit/agents/dev/test_dod_utils.py` - Unit tests for DoD validation
- `tests/integration/agents/dev/test_dod_validation.py` - Integration tests for gate flow

**Modified Files:**
- `src/yolo_developer/agents/dev/__init__.py` - Added exports for DoD utilities (including `validate_dod` alias)
- `src/yolo_developer/agents/dev/node.py` - Added gate warning logging and decision record integration
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status through workflow

## Change Log

- 2026-01-11: Implemented Story 8.6 DoD Validation - Created programmatic DoD validation utilities reusing existing gate functions, updated dev_node with gate warning integration
- 2026-01-11: Code review fixes - Added validate_dod alias, complete checklist with all DOD items, error handling, audit logging, and 10 additional edge case tests

