# Story 7.1: Create Architect Agent Node

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system architect,
I want the Architect implemented as a LangGraph node,
So that it integrates properly with the orchestration system.

## Acceptance Criteria

1. **AC1: architect_node Function Signature**
   - **Given** the Architect agent module exists
   - **When** architect_node is called
   - **Then** it accepts YoloState TypedDict as input
   - **And** it returns dict[str, Any] with state updates (not full state)
   - **And** it follows async/await patterns (async def)
   - **And** the function is importable from `yolo_developer.agents.architect`

2. **AC2: Receives Stories Requiring Architectural Decisions**
   - **Given** the orchestration state contains stories from PM
   - **When** architect_node processes the state
   - **Then** it extracts stories from state messages or pm_output
   - **And** it identifies stories requiring architectural decisions
   - **And** it logs the number of stories received for processing

3. **AC3: Returns Design Decisions**
   - **Given** stories are processed by the architect
   - **When** architect_node completes processing
   - **Then** it returns design decisions for each story
   - **And** each decision includes: story_id, decision_type, rationale, alternatives_considered
   - **And** design decisions are stored in ArchitectOutput dataclass
   - **And** the output is serializable via to_dict() method

4. **AC4: Returns ADRs (Architecture Decision Records)**
   - **Given** significant architectural decisions are made
   - **When** architect_node completes processing
   - **Then** it returns ADR objects for major decisions
   - **And** each ADR includes: id, title, status, context, decision, consequences
   - **And** ADRs follow standard ADR format
   - **And** ADRs are linked to relevant stories

5. **AC5: Follows Async Patterns**
   - **Given** architect_node performs I/O operations
   - **When** calling LLM or other services
   - **Then** all I/O uses async/await
   - **And** the function can be awaited in the orchestration graph
   - **And** retries use tenacity with exponential backoff

6. **AC6: Integrates with Architecture Validation Gate**
   - **Given** the architecture_validation gate is registered
   - **When** architect_node is decorated with @quality_gate
   - **Then** the gate evaluates design quality before handoff
   - **And** gate failures block processing in blocking mode
   - **And** gate results are logged for audit trail

7. **AC7: State Update Pattern**
   - **Given** architect_node completes processing
   - **When** returning results
   - **Then** it returns only state updates (messages, decisions, architect_output)
   - **And** it never mutates the input state
   - **And** messages are created using create_agent_message()
   - **And** Decision records are included with agent="architect"

## Tasks / Subtasks

- [x] Task 1: Create Architect Module Structure (AC: 1)
  - [x] Create `src/yolo_developer/agents/architect/` directory
  - [x] Create `__init__.py` with module docstring and exports
  - [x] Create `types.py` for type definitions
  - [x] Create `node.py` for architect_node function
  - [x] Follow existing pattern from `agents/analyst/` and `agents/pm/`

- [x] Task 2: Define Architect Type Definitions (AC: 3, 4)
  - [x] Create `DesignDecisionType` Literal type ("pattern", "technology", "integration", "data", "security", "infrastructure")
  - [x] Create `DesignDecision` frozen dataclass with: id, story_id, decision_type, description, rationale, alternatives_considered, created_at
  - [x] Create `ADRStatus` Literal type ("proposed", "accepted", "deprecated", "superseded")
  - [x] Create `ADR` frozen dataclass with: id, title, status, context, decision, consequences, story_ids, created_at
  - [x] Create `ArchitectOutput` frozen dataclass with: design_decisions, adrs, processing_notes, to_dict() method
  - [x] Add type exports to `__init__.py`

- [x] Task 3: Implement architect_node Function Shell (AC: 1, 5, 7)
  - [x] Create async def architect_node(state: YoloState) -> dict[str, Any]
  - [x] Add function docstring following analyst_node pattern
  - [x] Extract stories from state (from messages or pm_output)
  - [x] Add structured logging with structlog at start/complete
  - [x] Return state update dict with messages, decisions keys
  - [x] Create Decision record with agent="architect"
  - [x] Use create_agent_message() for output messages

- [x] Task 4: Implement Story Extraction (AC: 2)
  - [x] Create `_extract_stories_from_state(state: YoloState) -> list[Story]`
  - [x] Extract stories from pm_output if present in state
  - [x] Fallback to extracting from messages metadata
  - [x] Log story count and IDs extracted
  - [x] Return empty list if no stories found (graceful handling)

- [x] Task 5: Implement Design Decision Generation (AC: 3)
  - [x] Create `_generate_design_decisions(stories: list[Story]) -> list[DesignDecision]`
  - [x] For each story, determine if architectural decision is needed
  - [x] Generate DesignDecision with unique ID (format: "design-{timestamp}-{counter}")
  - [x] Include decision_type based on story content analysis
  - [x] Add rationale and alternatives_considered fields
  - [x] Use mock/stub implementation initially (LLM integration in Story 7.2+)

- [x] Task 6: Implement ADR Generation Stub (AC: 4)
  - [x] Create `_generate_adrs(decisions: list[DesignDecision]) -> list[ADR]`
  - [x] Generate ADR for significant decisions (technology choices, patterns)
  - [x] ADR ID format: "ADR-{number:03d}"
  - [x] Include standard ADR fields per architecture doc
  - [x] Link ADRs to story_ids from decisions
  - [x] This is a stub - full implementation in Story 7.3

- [x] Task 7: Register Architecture Validation Gate (AC: 6)
  - [x] Reuse existing `architecture_validation_evaluator` from gates module
  - [x] Existing evaluator validates design decisions and ADRs
  - [x] Evaluator already registered in gates/gates/architecture_validation.py
  - [x] Decorate architect_node with @quality_gate("architecture_validation")

- [x] Task 8: Write Unit Tests for Types (AC: 3, 4)
  - [x] Test DesignDecision dataclass creation and to_dict()
  - [x] Test ADR dataclass creation and to_dict()
  - [x] Test ArchitectOutput dataclass creation and to_dict()
  - [x] Test immutability of frozen dataclasses
  - [x] Test all enum/literal values

- [x] Task 9: Write Unit Tests for Story Extraction (AC: 2)
  - [x] Test extraction from pm_output in state
  - [x] Test extraction from message metadata
  - [x] Test empty state returns empty list
  - [x] Test logging of extracted story count

- [x] Task 10: Write Unit Tests for architect_node (AC: 1, 5, 7)
  - [x] Test architect_node is async
  - [x] Test returns dict with messages and decisions
  - [x] Test Decision has agent="architect"
  - [x] Test message created with create_agent_message()
  - [x] Test input state not mutated
  - [x] Test handles empty state gracefully

- [x] Task 11: Write Integration Tests (AC: 6)
  - [x] Test quality gate integration (mock evaluator)
  - [x] Test gate blocking behavior
  - [x] Test gate advisory mode
  - [x] Test gate result logging

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for DesignDecision, ADR, ArchitectOutput (internal state)
- **ADR-005 (LangGraph Communication):** Return state update dict, don't mutate state directly
- **ADR-006 (Quality Gates):** Integrate @quality_gate decorator with architecture_validation evaluator
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- Follow existing patterns from `agents/analyst/` and `agents/pm/` modules
- All dataclasses should be frozen (immutable)
- Include `to_dict()` method on all output dataclasses
- ArchitectOutput should be serializable for state storage

### Project Structure Notes

- **Module Location:** `src/yolo_developer/agents/architect/`
- **Type Definitions:** `src/yolo_developer/agents/architect/types.py`
- **Node Function:** `src/yolo_developer/agents/architect/node.py`
- **Test Location:** `tests/unit/agents/architect/`

### Module Structure Pattern (from analyst/pm)

```
src/yolo_developer/agents/architect/
├── __init__.py         # Exports: architect_node, ArchitectOutput, DesignDecision, ADR, etc.
├── types.py            # Type definitions (dataclasses, enums, literals)
└── node.py             # architect_node function and helper functions
```

### Key Imports Pattern

```python
from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.types import (
    ADR,
    ArchitectOutput,
    DesignDecision,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)
```

### architect_node Return Structure

```python
return {
    "messages": [message],           # AIMessage from create_agent_message()
    "decisions": [decision],         # Decision dataclass
    "architect_output": output.to_dict(),  # ArchitectOutput serialized
}
```

### Design Decision Types

| Type | Description | Example |
|------|-------------|---------|
| `pattern` | Architectural pattern selection | "Repository pattern for data access" |
| `technology` | Technology choice | "Use PostgreSQL for persistence" |
| `integration` | Integration approach | "REST API for external services" |
| `data` | Data model decisions | "Event sourcing for audit history" |
| `security` | Security architecture | "OAuth2 for authentication" |
| `infrastructure` | Infrastructure choice | "Docker containers with Kubernetes" |

### ADR Format (Standard)

```
# ADR-001: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
Why this decision is needed.

## Decision
What was decided.

## Consequences
Positive and negative effects.
```

### Story Dependencies

This story establishes the foundation for:
- Story 7.2 (12-Factor Design Generation) - adds LLM-powered design
- Story 7.3 (ADR Generation) - full ADR implementation
- Story 7.4-7.8 - specialized architect capabilities

### Previous Story Learnings Applied

From Epic 5 & 6 patterns:
- Create dedicated module directory with types.py and node.py
- Use frozen dataclasses for all data structures
- Include comprehensive structlog logging
- Test both positive and negative cases
- Update `processing_notes` with activity summary
- Include changes in `Decision.rationale`
- Export public functions and types from `__init__.py`

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.1] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005] - LangGraph communication patterns
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-006] - Quality gate patterns
- [Source: src/yolo_developer/agents/analyst/node.py] - Reference node implementation
- [Source: src/yolo_developer/agents/pm/node.py] - Reference node implementation
- [FR49: Architect Agent can design system architecture following 12-Factor principles]
- [FR50: Architect Agent can produce Architecture Decision Records (ADRs)]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

(To be filled during implementation)

### Completion Notes List

- Created architect module with types.py (DesignDecision, ADR, ArchitectOutput dataclasses) and node.py (architect_node function)
- Implemented story extraction from pm_output and message metadata
- Implemented stub design decision generation with keyword-based type inference
- Implemented stub ADR generation for technology/pattern decisions
- Integrated @quality_gate("architecture_validation") decorator in advisory mode
- All 79 architect tests passing, mypy clean, ruff clean

**Code Review Fixes Applied:**
- [H1] Added tenacity @retry decorator with exponential backoff (AC5 compliance)
- [H2] Wired up helper functions (_extract_stories_from_state, _generate_design_decisions, _generate_adrs) to architect_node
- [H3] Updated Task 7 documentation to accurately reflect reuse of existing gate evaluator
- [M1/M2] Updated File List to include all modified files (agents/__init__.py, sprint-status.yaml)
- [L2] Fixed incorrect Story reference in architect/__init__.py (7.7 → 7.1)

### File List

**Implementation Files:**
- `src/yolo_developer/agents/architect/__init__.py` - Module exports
- `src/yolo_developer/agents/architect/types.py` - Type definitions (DesignDecision, ADR, ArchitectOutput)
- `src/yolo_developer/agents/architect/node.py` - architect_node function and helpers
- `src/yolo_developer/agents/__init__.py` - Updated to export architect module types

**Test Files:**
- `tests/unit/agents/architect/test_module_structure.py` - Module structure tests (6 tests)
- `tests/unit/agents/architect/test_types.py` - Type dataclass tests (18 tests)
- `tests/unit/agents/architect/test_story_extraction.py` - Story extraction tests (8 tests)
- `tests/unit/agents/architect/test_design_decisions.py` - Design decision generation tests (10 tests)
- `tests/unit/agents/architect/test_adr_generation.py` - ADR generation tests (11 tests)
- `tests/unit/agents/architect/test_architect_node.py` - architect_node function tests (16 tests)
- `tests/unit/agents/architect/test_gate_integration.py` - Quality gate integration tests (10 tests)

**Other Modified Files:**
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status
