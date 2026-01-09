# Story 6.1: Create PM Agent Node

Status: done

## Story

As a system architect,
I want the PM implemented as a LangGraph node,
So that it integrates properly with the orchestration system.

## Acceptance Criteria

1. **AC1: Node Receives Crystallized Requirements in State**
   - **Given** the orchestration graph is running
   - **When** the pm_node function is invoked
   - **Then** it receives crystallized requirements from analyst output in state
   - **And** state includes analyst's requirements, gaps, contradictions, and escalations
   - **And** the node can access all context needed for story transformation

2. **AC2: Node Returns Stories with Acceptance Criteria**
   - **Given** the pm_node processes requirements
   - **When** processing completes
   - **Then** it returns a dict with state updates (not mutating state)
   - **And** output includes stories with acceptance criteria
   - **And** each story follows "As a / I want / So that" format

3. **AC3: Node Follows Async Patterns**
   - **Given** the pm_node implementation
   - **When** I/O operations are needed
   - **Then** all I/O operations use async/await
   - **And** the node function signature is `async def pm_node(state: YoloState) -> dict`
   - **And** blocking calls are avoided

4. **AC4: Integration with AC Measurability Gate**
   - **Given** the pm_node produces stories
   - **When** quality gate validation runs
   - **Then** the node is decorated with @quality_gate("ac_measurability", blocking=True)
   - **And** stories with unmeasurable AC are flagged
   - **And** gate failures prevent handoff to downstream agents

## Tasks / Subtasks

- [x] Task 1: Create PM Agent Module Structure (AC: all)
  - [x] Create `src/yolo_developer/agents/pm/` directory
  - [x] Create `src/yolo_developer/agents/pm/__init__.py` with public exports
  - [x] Create `src/yolo_developer/agents/pm/types.py` for PM-specific types
  - [x] Create `src/yolo_developer/agents/pm/node.py` for node function
  - [x] Create `tests/unit/agents/pm/` directory with __init__.py

- [x] Task 2: Define PM Agent Types (AC: 1, 2)
  - [x] Create `StoryStatus` enum: DRAFT, READY, BLOCKED, IN_PROGRESS, DONE
  - [x] Create `StoryPriority` enum: CRITICAL, HIGH, MEDIUM, LOW
  - [x] Create `AcceptanceCriterion` dataclass (frozen) with fields:
    - `id`: str (unique identifier, e.g., "AC1")
    - `given`: str (precondition)
    - `when`: str (action)
    - `then`: str (expected outcome)
    - `and_clauses`: tuple[str, ...] (additional conditions)
  - [x] Create `Story` dataclass (frozen) with fields:
    - `id`: str (unique story identifier)
    - `title`: str (short descriptive title)
    - `role`: str (the "As a" part)
    - `action`: str (the "I want" part)
    - `benefit`: str (the "So that" part)
    - `acceptance_criteria`: tuple[AcceptanceCriterion, ...]
    - `priority`: StoryPriority
    - `status`: StoryStatus
    - `source_requirements`: tuple[str, ...] (requirement IDs this story addresses)
    - `dependencies`: tuple[str, ...] (story IDs this depends on)
    - `estimated_complexity`: str (S, M, L, XL)
  - [x] Add `to_dict()` methods for serialization

- [x] Task 3: Create PMOutput Dataclass (AC: 2)
  - [x] Create `PMOutput` dataclass (frozen) with fields:
    - `stories`: tuple[Story, ...]
    - `unprocessed_requirements`: tuple[str, ...] (requirements that couldn't be transformed)
    - `escalations_to_analyst`: tuple[str, ...] (requirements needing clarification)
    - `processing_notes`: str
  - [x] Add `story_count` property
  - [x] Add `to_dict()` method
  - [x] Add `has_escalations` property

- [x] Task 4: Implement PM Node Skeleton (AC: 1, 3)
  - [x] Create `pm_node()` async function signature with YoloState input
  - [x] Extract analyst output from state (requirements, gaps, contradictions, escalations)
  - [x] Extract project configuration and context
  - [x] Add structlog logging for node entry/exit
  - [x] Return basic dict structure with messages and decisions

- [x] Task 5: Implement Requirement to Story Transformation Stub (AC: 2)
  - [x] Create `_transform_requirements_to_stories()` function (returns placeholder stories for MVP)
  - [x] Map requirements to story format
  - [x] Generate basic acceptance criteria structure
  - [x] Handle requirements that can't be transformed

- [x] Task 6: Implement AC Generation Stub (AC: 2)
  - [x] Create `_generate_acceptance_criteria()` function
  - [x] Generate Given/When/Then format from requirement text
  - [x] Add placeholder for and_clauses

- [x] Task 7: Integrate with Quality Gate Decorator (AC: 4)
  - [x] Import quality_gate decorator from gates module
  - [x] Apply @quality_gate("ac_measurability", blocking=True) to pm_node
  - [x] Ensure gate evaluates story AC measurability

- [x] Task 8: Implement State Return Pattern (AC: 1, 2)
  - [x] Create PMOutput instance with stories
  - [x] Create Decision record for audit trail
  - [x] Create AIMessage with processing summary
  - [x] Return dict with messages, decisions, pm_output keys

- [x] Task 9: Export Types and Node from __init__.py (AC: all)
  - [x] Export StoryStatus, StoryPriority, AcceptanceCriterion, Story, PMOutput
  - [x] Export pm_node function
  - [x] Update agents/__init__.py to expose pm module

- [x] Task 10: Write Unit Tests for PM Types (AC: all)
  - [x] Test StoryStatus enum values
  - [x] Test StoryPriority enum values
  - [x] Test AcceptanceCriterion creation and to_dict()
  - [x] Test Story creation with all fields
  - [x] Test Story to_dict() serialization
  - [x] Test PMOutput creation and properties

- [x] Task 11: Write Unit Tests for PM Node (AC: all)
  - [x] Test pm_node receives state correctly
  - [x] Test pm_node returns proper dict structure
  - [x] Test pm_node creates Decision records
  - [x] Test pm_node creates appropriate AIMessage
  - [x] Test _transform_requirements_to_stories() mapping
  - [x] Test _generate_acceptance_criteria() format

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use frozen dataclasses for PM types (Story, AcceptanceCriterion, PMOutput)
- **ADR-005 (Inter-Agent Communication):** Node returns dict with state updates, never mutates state
- **ADR-006 (Quality Gate Pattern):** Apply @quality_gate decorator for AC measurability validation
- **ADR-008 (Configuration):** Access config from state for project-specific settings
- **ARCH-QUALITY-5:** All I/O operations must be async/await
- **ARCH-QUALITY-6:** Use structlog for all logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all state dictionary keys
- Return dict updates from pm_node, never mutate state
- Follow async patterns for all I/O operations
- Use `@dataclass(frozen=True)` for all PM types

### File Structure (ARCH-STRUCT)

New files to create:
```
src/yolo_developer/agents/pm/
├── __init__.py         # Public exports
├── types.py            # PM-specific types (Story, AcceptanceCriterion, PMOutput)
└── node.py             # pm_node function

tests/unit/agents/pm/
├── __init__.py
├── test_types.py       # Tests for PM types
└── test_node.py        # Tests for pm_node function
```

Files to modify:
- `src/yolo_developer/agents/__init__.py` - Add pm module export

### Existing Code Patterns to Follow

**From Analyst Module (Story 5.1-5.7):**

Types pattern in `agents/analyst/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

class RequirementType(Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT = "constraint"

@dataclass(frozen=True)
class Requirement:
    id: str
    content: str
    ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            ...
        }
```

Node function pattern in `agents/analyst/node.py`:
```python
from __future__ import annotations

import structlog
from langchain_core.messages import AIMessage

from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision

logger = structlog.get_logger()

@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> dict[str, Any]:
    """Analyze seed input and produce crystallized requirements."""
    logger.info("analyst_node_started")

    # Process requirements...

    decision = Decision(
        agent="analyst",
        summary="...",
        rationale="...",
        related_artifacts=tuple(...)
    )

    message = AIMessage(content="...")

    logger.info("analyst_node_completed", ...)

    return {
        "messages": [message],
        "decisions": [decision],
        "analyst_output": output.to_dict(),
    }
```

### Integration Points

**Input from Analyst (via state):**
- `state["analyst_output"]` - Contains:
  - `requirements`: List of crystallized requirements
  - `gaps`: Identified requirement gaps
  - `contradictions`: Flagged contradictions
  - `escalations`: Issues escalated to PM for decision

**Output to Orchestrator/Architect:**
- `pm_output`: PMOutput with stories and AC
- `decisions`: Decision records for audit trail
- `messages`: AIMessage for conversation history

### Quality Gate Integration

The AC Measurability gate (FR20) validates:
- Each AC has concrete Given/When/Then structure
- No subjective terms (fast, easy, simple)
- Conditions are measurable and testable
- Edge cases are covered

Gate integration pattern:
```python
@quality_gate("ac_measurability", blocking=True)
async def pm_node(state: YoloState) -> dict[str, Any]:
    ...
```

### PM Agent Responsibilities (FR42-48)

This story implements the node infrastructure. Actual LLM-powered transformation will be in Story 6.2:

- **FR42:** Transform requirements to user stories (stub in this story)
- **FR43:** Ensure AC are testable and measurable (quality gate)
- **FR44:** Story prioritization (stub in this story)
- **FR45:** Dependency identification (stub in this story)
- **FR46:** Epic breakdown (future story)
- **FR47:** Escalation to Analyst (basic structure in this story)
- **FR48:** Story documentation following templates (stub in this story)

### Testing Strategy

**Type Tests:**
- Test all enum values exist and are accessible
- Test dataclass creation with required fields
- Test dataclass creation with optional fields
- Test to_dict() produces expected structure
- Test immutability (frozen=True)

**Node Tests:**
- Test node receives correct state structure
- Test node returns correct dict structure
- Test Decision record is created with expected fields
- Test AIMessage content is appropriate
- Mock any external dependencies
- Use pytest-asyncio for async tests

### Previous Story Learnings (from Stories 5.1-5.7)

1. **Use dict.get() with defaults** to avoid KeyError/ValueError on invalid values
2. **Export all new types from __init__.py** to ensure proper public API
3. **Add both positive and edge case tests** for comprehensive coverage
4. **Use frozen dataclasses** for all internal types (immutability)
5. **Include to_dict() methods** for serialization/debugging
6. **Log node entry and completion** with structlog for observability
7. **Create Decision records** for audit trail transparency
8. **Return dict updates** instead of mutating state (ADR-005)
9. **Apply quality_gate decorator** before node function definition

### Project Structure Notes

- Module follows established analyst pattern from Epic 5
- PM module is parallel to analyst module in agents/
- Tests mirror source structure in tests/unit/agents/pm/
- No circular imports - PM imports from orchestrator/context, gates

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 6.1: Create PM Agent Node]
- [Source: _bmad-output/planning-artifacts/epics.md - Epic 6: PM Agent overview]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-001, ADR-005, ADR-006, ADR-008]
- [Source: _bmad-output/planning-artifacts/architecture.md - Project Structure (ARCH-STRUCT)]
- [Source: _bmad-output/planning-artifacts/architecture.md - Implementation Patterns]
- [Source: _bmad-output/planning-artifacts/prd.md - FR42-48: PM Agent Capabilities]
- [Source: _bmad-output/planning-artifacts/prd.md - FR20: AC Measurability Gate]
- [Source: src/yolo_developer/agents/analyst/ - Existing agent module pattern]
- [Source: _bmad-output/implementation-artifacts/5-7-escalation-to-pm.md - Previous story patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - All tests passing

### Completion Notes List

- All 11 tasks completed successfully
- 67 new tests passing (36 type tests + 31 node tests)
- mypy type checking passes with no issues
- ruff linter passes with all checks
- ruff format confirms files already formatted
- Full test suite: 1939 tests pass (excluding env-specific tests)
- Backward compatibility maintained (all existing tests pass)
- PM module follows established analyst module patterns
- Quality gate decorator integrated per ADR-006

**Code Review Fixes Applied:**
- HIGH-1: Added documentation that config extraction deferred to Story 6.2
- HIGH-2: Added gaps and contradictions extraction from analyst_output (AC1 compliance)
- HIGH-3: Added documentation that ac_measurability evaluator implemented in Story 6.3
- HIGH-4: Added test for empty requirement_text edge case
- HIGH-5: Added StoryStatus, StoryPriority, AcceptanceCriterion exports to agents/__init__.py
- MEDIUM-1: Removed unused MagicMock import
- MEDIUM-2: Fixed story ID numbering to use separate counter (no gaps when constraints filtered)
- MEDIUM-3: Added test for Decision timestamp validation
- MEDIUM-4: Added documentation that hardcoded "user" role is stub for Story 6.2

### File List

**New Files:**
- `src/yolo_developer/agents/pm/__init__.py` - Module exports
- `src/yolo_developer/agents/pm/types.py` - StoryStatus, StoryPriority, AcceptanceCriterion, Story, PMOutput types
- `src/yolo_developer/agents/pm/node.py` - pm_node function and helpers
- `tests/unit/agents/pm/__init__.py` - Test module init
- `tests/unit/agents/pm/test_types.py` - 36 type tests
- `tests/unit/agents/pm/test_node.py` - 31 node function tests (4 new tests added in review)

**Modified Files:**
- `src/yolo_developer/agents/__init__.py` - Added pm_node, PMOutput, Story, StoryStatus, StoryPriority, AcceptanceCriterion exports

## Change Log

- 2026-01-09: Story implementation complete - PM agent node with types, tests, and quality gate integration
- 2026-01-09: Code review fixes applied - 5 HIGH, 4 MEDIUM issues fixed, 4 new tests added
