# Story 8.1: Create Dev Agent Node

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system architect,
I want the Dev implemented as a LangGraph node,
So that it integrates properly with the orchestration system.

## Acceptance Criteria

1. **AC1: dev_node Function Signature**
   - **Given** the Dev agent module exists
   - **When** dev_node is called
   - **Then** it accepts YoloState TypedDict as input
   - **And** it returns dict[str, Any] with state updates (not full state)
   - **And** it follows async/await patterns (async def)
   - **And** the function is importable from `yolo_developer.agents.dev`

2. **AC2: Receives Stories with Designs**
   - **Given** the orchestration state contains stories with architectural designs
   - **When** dev_node processes the state
   - **Then** it extracts stories from state messages or architect_output
   - **And** it identifies stories that are ready for implementation
   - **And** it logs the number of stories received for processing

3. **AC3: Returns Implemented Code with Tests**
   - **Given** stories are processed by the dev
   - **When** dev_node completes processing
   - **Then** it returns implementation artifacts for each story
   - **And** each artifact includes: story_id, code_files, test_files, implementation_status
   - **And** implementation artifacts are stored in DevOutput dataclass
   - **And** the output is serializable via to_dict() method

4. **AC4: Follows Async Patterns**
   - **Given** dev_node performs I/O operations
   - **When** calling LLM or other services
   - **Then** all I/O uses async/await
   - **And** the function can be awaited in the orchestration graph
   - **And** retries use tenacity with exponential backoff

5. **AC5: Integrates with Definition of Done Gate**
   - **Given** the definition_of_done gate is registered
   - **When** dev_node is decorated with @quality_gate
   - **Then** the gate evaluates implementation completeness before handoff
   - **And** gate failures block processing in blocking mode
   - **And** gate results are logged for audit trail

6. **AC6: State Update Pattern**
   - **Given** dev_node completes processing
   - **When** returning results
   - **Then** it returns only state updates (messages, decisions, dev_output)
   - **And** it never mutates the input state
   - **And** messages are created using create_agent_message()
   - **And** Decision records are included with agent="dev"

## Tasks / Subtasks

- [x] Task 1: Create Dev Module Structure (AC: 1)
  - [x] Create `src/yolo_developer/agents/dev/` directory
  - [x] Create `__init__.py` with module docstring and exports
  - [x] Create `types.py` for type definitions
  - [x] Create `node.py` for dev_node function
  - [x] Follow existing pattern from `agents/analyst/`, `agents/pm/`, and `agents/architect/`

- [x] Task 2: Define Dev Type Definitions (AC: 3)
  - [x] Create `ImplementationStatus` Literal type ("pending", "in_progress", "completed", "failed")
  - [x] Create `CodeFile` frozen dataclass with: file_path, content, file_type ("source", "test", "config", "doc")
  - [x] Create `TestFile` frozen dataclass with: file_path, content, test_type ("unit", "integration", "e2e")
  - [x] Create `ImplementationArtifact` frozen dataclass with: story_id, code_files, test_files, implementation_status, notes
  - [x] Create `DevOutput` frozen dataclass with: implementations, processing_notes, to_dict() method
  - [x] Add type exports to `__init__.py`

- [x] Task 3: Implement dev_node Function Shell (AC: 1, 4, 6)
  - [x] Create async def dev_node(state: YoloState) -> dict[str, Any]
  - [x] Add function docstring following analyst_node/architect_node pattern
  - [x] Extract stories from state (from messages or architect_output)
  - [x] Add structured logging with structlog at start/complete
  - [x] Return state update dict with messages, decisions keys
  - [x] Create Decision record with agent="dev"
  - [x] Use create_agent_message() for output messages

- [x] Task 4: Implement Story Extraction (AC: 2)
  - [x] Create `_extract_stories_for_implementation(state: YoloState) -> list[Story]`
  - [x] Extract stories from architect_output if present in state
  - [x] Fallback to extracting from messages metadata
  - [x] Filter for stories ready for implementation
  - [x] Log story count and IDs extracted
  - [x] Return empty list if no stories found (graceful handling)

- [x] Task 5: Implement Code Generation Stub (AC: 3)
  - [x] Create `_generate_implementation(story: Story) -> ImplementationArtifact`
  - [x] For each story, generate stub implementation artifact
  - [x] Include placeholder code_files with story-based file paths
  - [x] Include placeholder test_files with corresponding test paths
  - [x] Set implementation_status to "completed" for stub
  - [x] Use mock/stub implementation initially (LLM integration in Story 8.2+)

- [x] Task 6: Implement Test Generation Stub (AC: 3)
  - [x] Create `_generate_tests(story: Story, code_files: list[CodeFile]) -> list[TestFile]`
  - [x] Generate stub test files for each code file
  - [x] Set appropriate test_type based on story requirements
  - [x] This is a stub - full implementation in Story 8.3 (unit) and 8.4 (integration)

- [x] Task 7: Register Definition of Done Gate (AC: 5)
  - [x] Reuse existing `definition_of_done_evaluator` from gates module
  - [x] Existing evaluator validates implementation completeness
  - [x] Evaluator already registered in gates/gates/definition_of_done.py
  - [x] Decorate dev_node with @quality_gate("definition_of_done")

- [x] Task 8: Write Unit Tests for Types (AC: 3)
  - [x] Test CodeFile dataclass creation and to_dict()
  - [x] Test TestFile dataclass creation and to_dict()
  - [x] Test ImplementationArtifact dataclass creation and to_dict()
  - [x] Test DevOutput dataclass creation and to_dict()
  - [x] Test immutability of frozen dataclasses
  - [x] Test all enum/literal values

- [x] Task 9: Write Unit Tests for Story Extraction (AC: 2)
  - [x] Test extraction from architect_output in state
  - [x] Test extraction from message metadata
  - [x] Test empty state returns empty list
  - [x] Test logging of extracted story count

- [x] Task 10: Write Unit Tests for dev_node (AC: 1, 4, 6)
  - [x] Test dev_node is async
  - [x] Test returns dict with messages and decisions
  - [x] Test Decision has agent="dev"
  - [x] Test message created with create_agent_message()
  - [x] Test input state not mutated
  - [x] Test handles empty state gracefully

- [x] Task 11: Write Integration Tests (AC: 5)
  - [x] Test quality gate integration (mock evaluator)
  - [x] Test gate blocking behavior
  - [x] Test gate advisory mode
  - [x] Test gate result logging

- [x] Task 12: Update Module Exports (AC: 1)
  - [x] Add dev_node and types to `agents/__init__.py`
  - [x] Ensure all public types are exported

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for CodeFile, TestFile, ImplementationArtifact, DevOutput (internal state)
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-006 (Quality Gates):** Integrate @quality_gate decorator with definition_of_done evaluator
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/analyst/`, `agents/pm/`, and `agents/architect/` modules
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- DevOutput should be serializable for state storage
- Use tenacity @retry decorator with exponential backoff for async I/O operations

### Library Versions (from architecture.md)

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 1.0.5 | Orchestration framework |
| structlog | latest | Structured logging |
| tenacity | latest | Retry with backoff |
| pydantic | v2.x | Type validation at boundaries |

### LangGraph Node Pattern (from web research)

Node functions should:
1. Accept the current state as input (YoloState TypedDict)
2. Return a dictionary with only the keys that need to be updated
3. Never mutate the input state directly

```python
async def dev_node(state: YoloState) -> dict[str, Any]:
    # Process state
    return {
        "messages": [message],
        "decisions": [decision],
        "dev_output": output.to_dict(),
    }
```

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/dev/`
- **Type Definitions:** `src/yolo_developer/agents/dev/types.py`
- **Node Function:** `src/yolo_developer/agents/dev/node.py`
- **Test Location:** `tests/unit/agents/dev/`

### Module Structure Pattern (from analyst/pm/architect)

```
src/yolo_developer/agents/dev/
├── __init__.py         # Exports: dev_node, DevOutput, ImplementationArtifact, CodeFile, TestFile
├── types.py            # Type definitions (dataclasses, enums, literals)
└── node.py             # dev_node function and helper functions
```

### Key Imports Pattern

```python
from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.dev.types import (
    CodeFile,
    DevOutput,
    ImplementationArtifact,
    TestFile,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)
```

### dev_node Return Structure

```python
return {
    "messages": [message],           # AIMessage from create_agent_message()
    "decisions": [decision],         # Decision dataclass
    "dev_output": output.to_dict(),  # DevOutput serialized
}
```

### Implementation Status Types

| Status | Description | When Used |
|--------|-------------|-----------|
| `pending` | Not yet started | Story received but not processed |
| `in_progress` | Currently being implemented | During LLM code generation |
| `completed` | Implementation finished | All code and tests generated |
| `failed` | Implementation failed | Error during generation |

### File Type Classifications

**Code Files:**
| Type | Description | Example |
|------|-------------|---------|
| `source` | Main implementation code | `src/module.py` |
| `test` | Test files | `tests/test_module.py` |
| `config` | Configuration files | `config.yaml` |
| `doc` | Documentation | `README.md` |

**Test Files:**
| Type | Description | Example |
|------|-------------|---------|
| `unit` | Unit tests | `tests/unit/test_func.py` |
| `integration` | Integration tests | `tests/integration/test_api.py` |
| `e2e` | End-to-end tests | `tests/e2e/test_flow.py` |

### Story Dependencies

This story establishes the foundation for:
- Story 8.2 (Maintainable Code Generation) - adds LLM-powered code generation
- Story 8.3 (Unit Test Generation) - full unit test implementation
- Story 8.4 (Integration Test Generation) - integration test implementation
- Story 8.5 (Documentation Generation) - code documentation
- Story 8.6 (DoD Validation) - full DoD checklist validation
- Story 8.7 (Pattern Following) - codebase pattern matching
- Story 8.8 (Communicative Commits) - commit message generation

### Previous Story Learnings Applied

From Epic 5, 6, & 7 patterns:
- Create dedicated module directory with types.py and node.py
- Use frozen dataclasses for all data structures
- Include comprehensive structlog logging
- Test both positive and negative cases
- Update `processing_notes` with activity summary
- Include changes in `Decision.rationale`
- Export public functions and types from `__init__.py`
- Add tenacity @retry decorator with exponential backoff for all I/O

### Git Commit Pattern (from recent commits)

Recent commits follow pattern:
```
feat: Implement <feature> with code review fixes (Story X.Y)
```

### Existing Gate Infrastructure

The `definition_of_done_evaluator` already exists in `gates/gates/definition_of_done.py` and:
- Validates that implementation artifacts are complete
- Checks for test coverage
- Verifies documentation presence
- Returns GateResult with pass/fail and remediation guidance

### Functional Requirements Addressed

| FR | Description | How Addressed |
|----|-------------|---------------|
| FR57 | Dev Agent can implement code following maintainability-first hierarchy | DevOutput with code files (stub in 8.1, full in 8.2) |
| FR58 | Dev Agent can write unit tests | TestFile with unit type (stub in 8.1, full in 8.3) |
| FR59 | Dev Agent can write integration tests | TestFile with integration type (stub in 8.1, full in 8.4) |
| FR60 | Dev Agent can generate code documentation | CodeFile with doc type (stub in 8.1, full in 8.5) |
| FR61 | Dev Agent can validate code against DoD checklist | Quality gate integration (full in 8.6) |
| FR62 | Dev Agent can follow existing codebase patterns | Pattern extraction (full in 8.7) |
| FR63 | Dev Agent can escalate to Architect | Decision with escalation context |
| FR64 | Dev Agent can produce communicative commits | Commit message generation (full in 8.8) |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic-8] - Epic definition
- [Source: _bmad-output/planning-artifacts/epics.md#Story-8.1] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005] - LangGraph communication patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] - State management patterns
- [Source: src/yolo_developer/agents/analyst/node.py] - Reference node implementation
- [Source: src/yolo_developer/agents/pm/node.py] - Reference node implementation
- [Source: src/yolo_developer/agents/architect/node.py] - Reference node implementation (most recent)
- [Source: src/yolo_developer/gates/gates/definition_of_done.py] - DoD gate evaluator
- [FR57: Dev Agent can implement code following maintainability-first hierarchy]
- [FR58: Dev Agent can write unit tests for implemented functionality]
- [LangGraph Node Patterns](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All tests pass: 72 unit tests in tests/unit/agents/dev/
- All agents tests pass: 1170 tests total
- Ruff check: All checks passed
- Mypy: Success: no issues found in 3 source files

### Completion Notes List

- Created dev agent module following analyst/pm/architect patterns
- Implemented frozen dataclasses for CodeFile, TestFile, ImplementationArtifact, DevOutput
- Implemented dev_node async function with structlog logging
- Story extraction supports architect_output, pm_output, and message metadata
- Stub implementation generates placeholder code and tests (full LLM in 8.2+)
- Quality gate decorator @quality_gate("definition_of_done") in advisory mode
- Tenacity @retry decorator with exponential backoff for async resilience
- All types exported from dev module and agents package
- Comprehensive test coverage: types, extraction, node, gate integration

### Code Review Fixes Applied (2026-01-10)

- **Fixed:** Generated test import path from `src.implementations.` to `yolo_developer.implementations.`
- **Fixed:** Added `repr=False` to content fields in CodeFile and TestFile for cleaner debugging
- **Fixed:** Added `__test__ = False` to TestFile class to prevent pytest collection warnings
- **Fixed:** Updated docstrings to document file_path validation expectations
- **Fixed:** Bug where _generate_tests created duplicate TestFile objects with same path - now generates one test per story (stub behavior)
- **Fixed:** Updated test_generates_one_test_per_story to accurately test stub behavior

### File List

**Implementation Files:**
- `src/yolo_developer/agents/dev/__init__.py` - Module exports (created)
- `src/yolo_developer/agents/dev/types.py` - Type definitions (CodeFile, TestFile, ImplementationArtifact, DevOutput) (created)
- `src/yolo_developer/agents/dev/node.py` - dev_node function and helpers (created)
- `src/yolo_developer/agents/__init__.py` - Updated to export dev module types (modified)

**Test Files:**
- `tests/unit/agents/dev/__init__.py` - Test package init (created)
- `tests/unit/agents/dev/test_module_structure.py` - Module structure tests (created)
- `tests/unit/agents/dev/test_types.py` - Type dataclass tests (created)
- `tests/unit/agents/dev/test_story_extraction.py` - Story extraction tests (created)
- `tests/unit/agents/dev/test_implementation.py` - Implementation generation tests (created)
- `tests/unit/agents/dev/test_dev_node.py` - dev_node function tests (created)
- `tests/unit/agents/dev/test_gate_integration.py` - Quality gate integration tests (created)

## Change Log

- 2026-01-10: Story implementation complete - Created Dev agent module with dev_node function, types, and comprehensive tests
- 2026-01-10: Code review fixes applied - Fixed test import path, added repr=False, __test__=False, docstring updates, and duplicate test generation bug
