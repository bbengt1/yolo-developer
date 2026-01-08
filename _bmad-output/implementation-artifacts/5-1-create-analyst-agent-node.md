# Story 5.1: Create Analyst Agent Node

Status: done

## Story

As a system architect,
I want the Analyst implemented as a LangGraph node,
So that it integrates properly with the orchestration system.

## Acceptance Criteria

1. **AC1: State Input via YoloState**
   - **Given** the orchestration graph is running
   - **When** the analyst_node function is invoked
   - **Then** it receives state via YoloState TypedDict
   - **And** state contains messages, current_agent, handoff_context, decisions

2. **AC2: Immutable State Updates**
   - **Given** the analyst_node is processing
   - **When** it completes processing
   - **Then** it returns a dict with state updates (not mutating state)
   - **And** messages are appended using create_agent_message helper
   - **And** decisions are appended to the decisions list
   - **And** current_agent is NOT modified (handoff does that)

3. **AC3: Async/Await Pattern**
   - **Given** the analyst_node needs to make LLM calls
   - **When** processing occurs
   - **Then** all I/O operations use async/await
   - **And** the function signature is `async def analyst_node(state: YoloState) -> dict[str, Any]`
   - **And** LLM calls use LiteLLM's async API (`acompletion`)

4. **AC4: Testability Gate Integration**
   - **Given** the analyst produces crystallized requirements
   - **When** handoff to PM is prepared
   - **Then** the testability quality gate decorator is applied
   - **And** the gate evaluates requirement testability
   - **And** blocking failures prevent handoff to PM

## Tasks / Subtasks

- [x] Task 1: Create Analyst Agent Module Structure (AC: 1, 2)
  - [x] Create `src/yolo_developer/agents/analyst/__init__.py`
  - [x] Create `src/yolo_developer/agents/analyst/node.py` with `analyst_node` function
  - [x] Create `src/yolo_developer/agents/analyst/types.py` for Analyst-specific types
  - [x] Export public API from `agents/analyst/__init__.py`

- [x] Task 2: Implement Analyst Node Function (AC: 1, 2, 3)
  - [x] Define `async def analyst_node(state: YoloState) -> dict[str, Any]`
  - [x] Extract seed content from state messages
  - [x] Create placeholder for LLM-based requirement crystallization
  - [x] Return state update dict with new messages and decisions
  - [x] Ensure no direct state mutation (return updates only)

- [x] Task 3: Create Analyst Types (AC: 2)
  - [x] Create `CrystallizedRequirement` frozen dataclass: id, original_text, refined_text, category, testable
  - [x] Create `AnalystOutput` frozen dataclass: requirements, identified_gaps, contradictions
  - [x] Add `to_dict()` methods for JSON serialization
  - [x] Type all attributes with full annotations

- [x] Task 4: Integrate Testability Gate (AC: 4)
  - [x] Apply `@quality_gate("testability", blocking=True)` decorator
  - [x] Ensure gate receives AnalystOutput for evaluation
  - [x] Handle gate failure gracefully (return error state, not exception)
  - [x] Log gate results via structlog

- [x] Task 5: Add LLM Integration Stub (AC: 3)
  - [x] Create `_call_llm` async helper using LiteLLM's `acompletion`
  - [x] Use config for model selection (cheap_model for routine analysis)
  - [x] Include retry logic with tenacity (3 retries, exponential backoff)
  - [x] Return structured output (parse JSON response)

- [x] Task 6: Create Agent Prompt Template (AC: 1, 2)
  - [x] Create `src/yolo_developer/agents/prompts/analyst.py`
  - [x] Define `ANALYST_SYSTEM_PROMPT` with role and instructions
  - [x] Define `ANALYST_USER_PROMPT_TEMPLATE` for seed analysis
  - [x] Include output format instructions (JSON schema)

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Test analyst_node receives YoloState correctly
  - [x] Test analyst_node returns dict (not YoloState)
  - [x] Test returned dict contains valid message updates
  - [x] Test returned dict contains valid decision updates
  - [x] Test async execution works correctly
  - [x] Test CrystallizedRequirement and AnalystOutput dataclasses
  - [x] Test to_dict() serialization

- [x] Task 8: Write Integration Tests (AC: 3, 4)
  - [x] Test analyst_node integrates with StateGraph
  - [x] Test analyst_node with mocked LLM responses
  - [x] Test testability gate decorator blocks on failure
  - [x] Test successful handoff context creation

- [x] Task 9: Update Exports and Orchestrator (AC: all)
  - [x] Export `analyst_node` from `agents/__init__.py`
  - [x] Export `CrystallizedRequirement`, `AnalystOutput` from types
  - [x] Register analyst_node in `orchestrator/graph.py` (add node, don't wire edges yet)
  - [x] Update module docstrings with usage examples

## Dev Notes

### Architecture Compliance

- **ADR-001 (TypedDict State):** Use YoloState TypedDict for internal state, Pydantic at boundaries only
- **ADR-003 (LiteLLM):** Use LiteLLM SDK for LLM calls with model tiering
- **ADR-005 (LangGraph):** Implement as LangGraph node with typed state transitions
- **ADR-006 (Quality Gates):** Apply decorator-based testability gate
- **FR36-41:** Analyst Agent capabilities (crystallize, identify gaps, categorize, validate, flag contradictions, escalate)
- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-005] - LangGraph message passing
- [Source: epics.md#Story-5.1] - Create Analyst Agent Node requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `CrystallizedRequirement`, `AnalystOutput`
- **Pure Node Function:** Node returns state update dict, no side effects except logging
- **Async Required:** All I/O (LLM calls) must be async/await
- **Structured Logging:** Use structlog for all logging
- **Type Annotations:** Full type hints on all functions and methods

### Previous Story Intelligence (Story 4.7)

**Files Created/Modified in Story 4.7:**
- `src/yolo_developer/seed/rejection.py` - Rejection types (pattern for agent types)
- Pattern: frozen dataclasses with `to_dict()` methods
- Pattern: `from __future__ import annotations` in all files
- Pattern: Export from `__init__.py` with explicit public API
- Tests: Unit tests for dataclasses, integration tests for CLI

**Key Patterns from Story 4.7:**

```python
# Data model pattern (reuse for analyst types)
@dataclass(frozen=True)
class CrystallizedRequirement:
    id: str
    original_text: str
    refined_text: str
    category: str  # "functional", "non-functional", "constraint"
    testable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "category": self.category,
            "testable": self.testable,
        }
```

### Existing Code to Reuse (CRITICAL)

**From `orchestrator/state.py` - YoloState and Message Creation:**
```python
from yolo_developer.orchestrator.state import YoloState, create_agent_message

# Node function signature
async def analyst_node(state: YoloState) -> dict[str, Any]:
    # Use create_agent_message for attributed messages
    msg = create_agent_message(
        content="Analysis complete: 5 requirements crystallized",
        agent="analyst",
        metadata={"requirement_count": 5},
    )
    return {"messages": [msg], "decisions": [decision]}
```

**From `orchestrator/context.py` - Decision Recording:**
```python
from yolo_developer.orchestrator.context import Decision

decision = Decision(
    agent="analyst",
    summary="Crystallized 5 requirements from seed",
    rationale="Seed contained 3 functional and 2 non-functional requirements",
    related_artifacts=("req-001", "req-002"),
)
```

**From `gates/types.py` - Gate Context:**
```python
from yolo_developer.gates.types import GateMode, GateResult, GateContext

# Apply gate decorator (implementation in gates/decorator.py)
@quality_gate("testability", mode=GateMode.BLOCKING)
async def analyst_node(state: YoloState) -> dict[str, Any]:
    ...
```

### Anti-Patterns to Avoid

- **DO NOT** mutate state directly - return update dict only
- **DO NOT** use sync LLM calls - use async `acompletion`
- **DO NOT** hardcode model names - use config.llm.cheap_model
- **DO NOT** skip structured logging - use structlog for audit
- **DO NOT** use Pydantic for internal types - use frozen dataclasses
- **DO NOT** create blocking sync code - everything async

### Project Structure Notes

**Files to Create:**
```
src/yolo_developer/agents/
├── analyst/
│   ├── __init__.py      # Public API exports
│   ├── node.py          # analyst_node function
│   └── types.py         # CrystallizedRequirement, AnalystOutput
└── prompts/
    └── analyst.py       # Prompt templates
```

**Files to Modify:**
```
src/yolo_developer/agents/
└── __init__.py          # Export analyst_node, types

src/yolo_developer/orchestrator/
└── graph.py             # Register analyst node (no edges yet)

tests/unit/agents/
└── analyst/
    ├── __init__.py
    ├── test_node.py     # Unit tests for node
    └── test_types.py    # Unit tests for types

tests/integration/
└── test_analyst_integration.py  # Integration tests
```

### Dependencies

**Depends On:**
- Epic 1 (Config) - ✅ Complete - for LLM model configuration
- Epic 3 (Gates) - Need testability gate decorator - implement stub if not complete
- `orchestrator/state.py` - YoloState, create_agent_message
- `orchestrator/context.py` - Decision

**Downstream Dependencies:**
- Story 5.2 (Requirement Crystallization) - will use this node
- Story 5.7 (Escalation to PM) - will extend this node
- Epic 10 (Orchestration) - will wire this node into graph

### External Dependencies

- **litellm** (installed) - LLM abstraction layer
- **tenacity** (installed) - Retry logic
- **structlog** (installed) - Structured logging
- No new dependencies required

### LLM Integration Pattern

```python
import structlog
from litellm import acompletion
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.config import load_config

logger = structlog.get_logger()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(prompt: str, system: str) -> str:
    """Call LLM with retry logic."""
    config = load_config()
    model = config.llm.cheap_model

    logger.info("calling_llm", model=model, prompt_length=len(prompt))

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content
```

### Node Implementation Pattern

```python
from __future__ import annotations

from typing import Any

import structlog

from yolo_developer.orchestrator.state import YoloState, create_agent_message
from yolo_developer.orchestrator.context import Decision

logger = structlog.get_logger()


async def analyst_node(state: YoloState) -> dict[str, Any]:
    """Analyst agent node for requirement crystallization.

    Receives seed requirements from state messages and produces
    crystallized, categorized requirements with testability assessment.

    Args:
        state: Current orchestration state with accumulated messages.

    Returns:
        State update dict with new messages and decisions.
        Never mutates the input state.
    """
    logger.info("analyst_node_start", current_agent=state.get("current_agent"))

    # Extract seed from messages
    seed_content = _extract_seed_from_messages(state["messages"])

    # Process requirements (LLM call)
    output = await _crystallize_requirements(seed_content)

    # Create decision record
    decision = Decision(
        agent="analyst",
        summary=f"Crystallized {len(output.requirements)} requirements",
        rationale="Parsed seed document and extracted structured requirements",
        related_artifacts=tuple(r.id for r in output.requirements),
    )

    # Create output message
    message = create_agent_message(
        content=f"Analysis complete: {len(output.requirements)} requirements crystallized",
        agent="analyst",
        metadata={"output": output.to_dict()},
    )

    logger.info("analyst_node_complete", requirement_count=len(output.requirements))

    return {
        "messages": [message],
        "decisions": [decision],
    }
```

### Testing Strategy

**Unit Tests:**
```python
import pytest
from yolo_developer.agents.analyst.types import CrystallizedRequirement, AnalystOutput

def test_crystallized_requirement_immutable() -> None:
    """CrystallizedRequirement should be immutable."""
    req = CrystallizedRequirement(
        id="req-001",
        original_text="The system should be fast",
        refined_text="Response time < 200ms for 95th percentile",
        category="non-functional",
        testable=True,
    )
    with pytest.raises(FrozenInstanceError):
        req.id = "new-id"

def test_analyst_output_to_dict() -> None:
    """to_dict should serialize all fields."""
    output = AnalystOutput(
        requirements=(CrystallizedRequirement(...),),
        identified_gaps=("Missing auth requirements",),
        contradictions=(),
    )
    d = output.to_dict()
    assert "requirements" in d
    assert "identified_gaps" in d
```

**Integration Tests:**
```python
import pytest
from langgraph.graph import StateGraph

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.orchestrator.state import YoloState

@pytest.mark.asyncio
async def test_analyst_node_returns_state_update() -> None:
    """analyst_node should return dict, not YoloState."""
    state: YoloState = {
        "messages": [HumanMessage(content="Build a todo app")],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }
    result = await analyst_node(state)

    assert isinstance(result, dict)
    assert "messages" in result
    assert "decisions" in result
    assert isinstance(result["messages"], list)
```

### Commit Message Pattern

```
feat: Implement Analyst agent node as LangGraph component (Story 5.1)

- Create analyst_node async function receiving YoloState
- Add CrystallizedRequirement, AnalystOutput frozen dataclasses
- Integrate with testability quality gate (stub if gate not ready)
- Add LLM integration with LiteLLM async API
- Add prompt templates in agents/prompts/analyst.py
- Write unit tests for types and node function
- Write integration tests for StateGraph integration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### References

- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-003] - LiteLLM SDK integration
- [Source: architecture.md#ADR-005] - LangGraph patterns
- [Source: architecture.md#ADR-006] - Quality gate decorators
- [Source: epics.md#Epic-5] - Analyst Agent epic context
- [Source: epics.md#Story-5.1] - Story requirements
- [Source: orchestrator/state.py] - YoloState, create_agent_message
- [Source: orchestrator/context.py] - Decision, HandoffContext
- [Source: gates/types.py] - GateMode, GateResult patterns

### Files to Consult (MUST READ Before Implementation)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `orchestrator/state.py` | YoloState TypedDict, create_agent_message | 52-139 |
| `orchestrator/context.py` | Decision dataclass pattern | 60-91 |
| `gates/types.py` | GateResult, GateContext patterns | 50-136 |
| `seed/rejection.py` | Frozen dataclass pattern from 4.7 | Full file |
| `config/schema.py` | Config access pattern | Full file |

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 9 tasks completed following red-green-refactor TDD cycle
- 17 unit tests + 9 integration tests all passing (26 new tests total)
- Full test suite: 1734 passed (10 pre-existing failures unrelated to this story)
- Applied `@quality_gate("testability", blocking=True)` decorator pattern
- LLM integration uses `_USE_LLM` flag for testing without actual API calls
- Used lazy imports in `get_agent_nodes()` to avoid circular dependencies
- Followed frozen dataclass pattern from Story 4.7 (seed/rejection.py)

### File List

**Created:**
- `src/yolo_developer/agents/analyst/__init__.py` - Public API exports for analyst module
- `src/yolo_developer/agents/analyst/node.py` - analyst_node async function with @quality_gate decorator
- `src/yolo_developer/agents/analyst/types.py` - CrystallizedRequirement and AnalystOutput frozen dataclasses
- `src/yolo_developer/agents/prompts/__init__.py` - Prompts package init
- `src/yolo_developer/agents/prompts/analyst.py` - ANALYST_SYSTEM_PROMPT and ANALYST_USER_PROMPT_TEMPLATE
- `tests/unit/agents/__init__.py` - Unit test package init (if created)
- `tests/unit/agents/analyst/__init__.py` - Test package init
- `tests/unit/agents/analyst/test_types.py` - 10 unit tests for types
- `tests/unit/agents/analyst/test_node.py` - 7 unit tests for node function
- `tests/integration/test_analyst_integration.py` - 9 integration tests

**Modified:**
- `src/yolo_developer/agents/__init__.py` - Added exports for analyst_node, AnalystOutput, CrystallizedRequirement
- `src/yolo_developer/orchestrator/graph.py` - Added get_agent_nodes() registry function

**Code Review Fixes Applied:**
- `src/yolo_developer/agents/analyst/__init__.py` - Sorted `__all__` per RUF022
- `tests/integration/test_analyst_integration.py` - Sorted imports, removed unused `Any` import, removed unused variable
- `tests/unit/agents/analyst/test_node.py` - Removed unused `Any` import
