# Story 10.2: SM Agent Node Implementation

Status: review

## Story

As a system architect,
I want the SM (Scrum Master) as the control plane agent,
So that orchestration decisions are centralized.

## Acceptance Criteria

1. **Given** the orchestration graph is running
   **When** the sm_node function is invoked
   **Then** it can route to any other agent

2. **Given** an agent completes execution
   **When** the sm_node evaluates the state
   **Then** it makes orchestration decisions based on state

3. **Given** the SM agent makes a routing decision
   **When** it logs the decision
   **Then** all routing decisions are logged with structured format

4. **Given** an edge case occurs (circular logic, blocked agent, escalation)
   **When** the sm_node handles it
   **Then** it handles edge cases gracefully with appropriate recovery

## Tasks / Subtasks

- [x] Task 1: Create SM agent module structure (AC: #1, #2)
  - [x] 1.1: Create `src/yolo_developer/agents/sm/__init__.py` module
  - [x] 1.2: Create `src/yolo_developer/agents/sm/types.py` with SM-specific types
  - [x] 1.3: Create `src/yolo_developer/agents/sm/node.py` with sm_node function
  - [x] 1.4: Export sm_node from agents package

- [x] Task 2: Implement SM state analysis (AC: #2)
  - [x] 2.1: Implement `_analyze_current_state()` to evaluate workflow progress
  - [x] 2.2: Implement `_get_next_agent()` routing decision function
  - [x] 2.3: Implement `_check_for_escalation()` to detect human escalation needs
  - [x] 2.4: Implement `_check_for_circular_logic()` to detect agent ping-pong

- [x] Task 3: Implement SM node function (AC: #1, #2, #3, #4)
  - [x] 3.1: Create async `sm_node(state: YoloState) -> dict[str, Any]` function
  - [x] 3.2: Add @quality_gate decorator for SM outputs
  - [x] 3.3: Add @retry decorator with tenacity
  - [x] 3.4: Implement state update return (messages, decisions, routing)

- [x] Task 4: Implement edge case handling (AC: #4)
  - [x] 4.1: Handle circular logic detection (>3 exchanges)
  - [x] 4.2: Handle gate_blocked state
  - [x] 4.3: Handle escalate_to_human flag
  - [x] 4.4: Handle agent errors/failures gracefully

- [x] Task 5: Update workflow to integrate SM node (AC: #1)
  - [x] 5.1: Add sm_node to workflow.py get_default_agent_nodes()
  - [x] 5.2: Update routing to use SM as control plane where appropriate

- [x] Task 6: Write tests (AC: all)
  - [x] 6.1: Create `tests/unit/agents/sm/` directory
  - [x] 6.2: Write `test_node.py` with unit tests for sm_node
  - [x] 6.3: Write `test_types.py` with tests for SM types
  - [x] 6.4: Write tests for edge case handling

## Dev Notes

### Architecture Requirements

This story implements **ADR-005: Inter-Agent Communication** and **ADR-007: Error Handling Strategy** which specify:

- SM as the control plane for orchestration decisions
- State-based routing with explicit handoff conditions
- SM-coordinated recovery for agent failures
- Checkpoint-based state recovery capability

Per the architecture document, the SM agent is the **central orchestrator** that:
- Plans sprints by prioritizing and sequencing stories (FR9)
- Delegates tasks to appropriate specialized agents (FR10)
- Monitors agent activity and health metrics (FR11)
- Detects circular logic between agents (FR12, >3 exchanges)
- Mediates conflicts between agents (FR13)

### Existing Infrastructure to Use

**Workflow Module** (`orchestrator/workflow.py` - Story 10.1):
```python
# Current routing functions exist but don't go through SM
# SM will be added as a control plane node
def route_after_analyst(state: YoloState) -> str
def route_after_pm(state: YoloState) -> str
def route_after_architect(state: YoloState) -> str
def route_after_dev(state: YoloState) -> str
def route_after_tea(state: YoloState) -> str
```

**State Management** (`orchestrator/state.py`):
```python
class YoloState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
```

**Context Preservation** (`orchestrator/context.py`):
```python
@dataclass(frozen=True)
class Decision:
    agent: str
    summary: str
    rationale: str
    timestamp: datetime
    related_artifacts: tuple[str, ...]

@dataclass(frozen=True)
class HandoffContext:
    source_agent: str
    target_agent: str
    decisions: tuple[Decision, ...]
    memory_refs: tuple[str, ...]
```

**Agent Node Pattern** (from `agents/tea/node.py`):
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("confidence_scoring", blocking=False)
async def tea_node(state: YoloState) -> dict[str, Any]:
    """Agent node following LangGraph pattern."""
    logger.info("node_start", current_agent=state.get("current_agent"))

    # Process state
    # ...

    # Create decision record
    decision = Decision(
        agent="tea",
        summary="...",
        rationale="...",
        related_artifacts=(),
    )

    # Create output message
    message = create_agent_message(
        content="...",
        agent="tea",
        metadata={"output": output.to_dict()},
    )

    # Return ONLY updates
    return {
        "messages": [message],
        "decisions": [decision],
        "some_output": output.to_dict(),
    }
```

### SM Node Responsibilities

Based on the architecture and epics, the SM node should handle:

1. **Routing Decisions**: Determine next agent based on state
   - Check `needs_architecture` flag
   - Check `gate_blocked` flag
   - Check `escalate_to_human` flag
   - Check for circular logic patterns

2. **Health Monitoring** (stub for Story 10.5):
   - Track agent exchange counts
   - Detect idle agents
   - Monitor cycle time

3. **Circular Logic Detection** (from FR12):
   - Track message exchanges between agents
   - Detect >3 exchanges on same issue
   - Trigger escalation when detected

4. **Conflict Mediation** (stub for Story 10.7):
   - Detect conflicting recommendations
   - Log conflict for resolution

### SM-Specific State Fields

The SM will need to track some additional state. Options:
1. Add fields to YoloState (requires state.py modification)
2. Track in sm_output dict (preferred for isolation)
3. Use message metadata for exchange tracking

Recommended approach: Track in sm_output to avoid state.py changes:
```python
class SMOutput(TypedDict):
    routing_decision: str
    routing_rationale: str
    exchange_count: int
    circular_logic_detected: bool
    escalation_triggered: bool
```

### Implementation Approach

The SM can be implemented in two modes:

**Mode 1: Passive Control Plane (Recommended for MVP)**
- SM is NOT in the main workflow path
- Routing functions in workflow.py handle normal flow
- SM is only invoked for escalation/conflict resolution
- Simpler integration, matches current workflow.py

**Mode 2: Active Control Plane (Full Architecture)**
- All agent-to-agent transitions go through SM
- SM makes all routing decisions
- More overhead but full control
- Would require significant workflow.py changes

For this story (10.2), implement **Mode 1** with the capability to support Mode 2:
- Create sm_node that CAN route to any agent
- Don't integrate into main workflow path yet
- Test routing capabilities standalone
- Story 10.8 (Agent Handoff Management) will integrate SM into workflow

### Routing Logic

SM routing should consider:
```python
def _get_next_agent(state: YoloState) -> str:
    """Determine next agent based on state analysis."""

    # Priority 1: Check for escalation
    if state.get("escalate_to_human", False):
        return "escalate"

    # Priority 2: Check for circular logic
    exchange_count = _count_recent_exchanges(state)
    if exchange_count > 3:
        return "escalate"

    # Priority 3: Check for blocked gates
    if state.get("gate_blocked", False):
        # Route back to agent that can fix it
        return _get_recovery_agent(state)

    # Priority 4: Normal flow based on current agent
    current = state.get("current_agent", "")
    return _get_natural_successor(current, state)
```

### Testing Strategy

**Unit Tests:**
- Test sm_node returns valid state update dict
- Test routing decision logic for each condition
- Test circular logic detection
- Test escalation triggering
- Test edge case handling

**Mocking:**
- Mock state to simulate various conditions
- No actual LLM calls needed for SM (pure routing logic)

### Dependencies

**Internal dependencies:**
- `yolo_developer.orchestrator.state` - YoloState TypedDict
- `yolo_developer.orchestrator.context` - Decision, create_agent_message
- `yolo_developer.gates` - quality_gate decorator
- `tenacity` - retry decorator
- `structlog` - logging

**No new external dependencies needed.**

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/agents/sm/__init__.py`
- `src/yolo_developer/agents/sm/types.py`
- `src/yolo_developer/agents/sm/node.py`
- `tests/unit/agents/sm/__init__.py`
- `tests/unit/agents/sm/test_node.py`
- `tests/unit/agents/sm/test_types.py`

**Files to modify:**
- `src/yolo_developer/agents/__init__.py` - Add sm_node export
- `src/yolo_developer/orchestrator/workflow.py` - Add sm_node to get_default_agent_nodes()

### Implementation Patterns

Per architecture document, follow these patterns:

1. **Async-first**: sm_node must be async
2. **State updates via dict**: Return dict updates, don't mutate state
3. **Structured logging**: Use structlog with key-value format
4. **Type annotations**: Full type hints on all functions
5. **Immutable outputs**: Use frozen dataclasses for types

```python
# CORRECT pattern for sm_node
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("sm_routing", blocking=False)
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node for orchestration control plane."""
    logger.info(
        "sm_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Analyze state and make routing decision
    routing_decision = _get_next_agent(state)
    routing_rationale = _get_routing_rationale(state, routing_decision)

    # Check for circular logic
    circular_detected = _check_for_circular_logic(state)

    # Create decision record
    decision = Decision(
        agent="sm",
        summary=f"Routing to {routing_decision}",
        rationale=routing_rationale,
    )

    # Create output message
    message = create_agent_message(
        content=f"SM decision: route to {routing_decision}. {routing_rationale}",
        agent="sm",
        metadata={"routing": routing_decision},
    )

    logger.info(
        "sm_node_complete",
        routing_decision=routing_decision,
        circular_detected=circular_detected,
    )

    return {
        "messages": [message],
        "decisions": [decision],
        "sm_output": {
            "routing_decision": routing_decision,
            "routing_rationale": routing_rationale,
            "circular_logic_detected": circular_detected,
        },
    }
```

### Previous Story Intelligence (Story 10.1)

From Story 10.1 implementation:
- Workflow module created at `orchestrator/workflow.py`
- Uses `build_workflow()` function returning compiled StateGraph
- Routing functions defined for each agent transition
- `get_default_agent_nodes()` returns dict of all agent nodes
- Checkpointing integrated via `create_workflow_with_checkpointing()`
- `run_workflow()` and `stream_workflow()` async functions available

**Key insight from 10.1:** The current workflow uses direct routing functions (route_after_analyst, etc.) rather than going through SM. For 10.2, create the SM node capability but don't force all routing through it yet.

### Git Intelligence

From recent commits:
- Story 10.1 implemented LangGraph workflow with conditional routing
- Stories 9.1-9.8 implemented TEA agent with coverage, confidence, risk, testability, deployment blocking, and gap analysis
- All agents follow same pattern: async node, structlog, quality_gate decorator, return dict updates

### Latest Technical Information

**LangGraph (v0.2+):**
- StateGraph takes TypedDict as state type
- Nodes return dict updates that get merged into state
- Conditional edges use routing functions
- Checkpointing via BaseCheckpointSaver interface

**structlog best practices:**
- Use key-value logging: `logger.info("event_name", key1=val1, key2=val2)`
- Get logger with `structlog.get_logger(__name__)`
- Use info for significant events, debug for verbose

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.2]
- [Source: src/yolo_developer/orchestrator/workflow.py]
- [Source: src/yolo_developer/orchestrator/state.py]
- [Source: src/yolo_developer/orchestrator/context.py]
- [Source: src/yolo_developer/agents/tea/node.py - pattern reference]
- [Source: _bmad-output/implementation-artifacts/10-1-create-langgraph-workflow.md]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation proceeded without issues

### Completion Notes List

- **Task 1**: Created SM agent module with `__init__.py`, `types.py`, and `node.py` files. Exported `sm_node` and related types from agents package.

- **Task 2**: Implemented state analysis functions:
  - `_analyze_current_state()`: Evaluates workflow state flags and metrics
  - `_get_next_agent()`: Priority-based routing decision (escalation > circular logic > gate blocked > natural flow)
  - `_check_for_escalation()`: Detects human_requested and agent_failure flags
  - `_check_for_circular_logic()`: Tracks agent-to-agent exchanges and detects ping-pong patterns

- **Task 3**: Implemented async `sm_node` function with:
  - `@retry` decorator with exponential backoff (3 attempts)
  - `@quality_gate("sm_routing", blocking=False)` decorator
  - Returns dict with messages, decisions, sm_output, and routing_decision

- **Task 4**: Implemented edge case handling:
  - Circular logic detection using `_detect_circular_pattern()` with >3 exchange threshold (FR12)
  - Gate blocked recovery via `_get_recovery_agent()`
  - Escalation flags via `_check_for_escalation()`
  - Graceful handling of invalid/unknown agent names

- **Task 5**: Updated workflow.py:
  - Added sm_node to `get_default_agent_nodes()`
  - SM is now available in the workflow but not yet in the main routing path (Mode 1 approach per story spec)

- **Task 6**: Wrote comprehensive tests:
  - 62 tests total covering all functionality
  - `test_types.py`: Tests for AgentExchange, SMOutput, and constants
  - `test_node.py`: Tests for all helper functions, sm_node, edge cases, and integration scenarios

### Change Log

- 2026-01-12: Implemented Story 10.2 - SM Agent Node Implementation
  - Created SM agent module with types and node implementation
  - Added comprehensive test coverage (62 tests)
  - Integrated SM node into workflow agent registry
  - All acceptance criteria satisfied

### File List

**New Files:**
- src/yolo_developer/agents/sm/__init__.py
- src/yolo_developer/agents/sm/types.py
- src/yolo_developer/agents/sm/node.py
- tests/unit/agents/sm/__init__.py
- tests/unit/agents/sm/test_types.py
- tests/unit/agents/sm/test_node.py

**Modified Files:**
- src/yolo_developer/agents/__init__.py (added sm module exports)
- src/yolo_developer/orchestrator/workflow.py (added sm_node to get_default_agent_nodes)
- tests/unit/orchestrator/test_workflow.py (updated expected agents to include sm)
