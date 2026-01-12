# Story 10.1: Create LangGraph Workflow

Status: done

## Story

As a system architect,
I want the orchestration built on LangGraph StateGraph,
So that agent execution follows a defined, predictable flow.

## Acceptance Criteria

1. **Given** all agent nodes are defined
   **When** the StateGraph is constructed
   **Then** nodes are connected via edges

2. **Given** the StateGraph is constructed
   **When** executing the workflow
   **Then** conditional routing is supported (e.g., route to specific agents based on state)

3. **Given** the StateGraph is defined
   **When** examining the state type
   **Then** it uses the existing YoloState TypedDict from `orchestrator/state.py`

4. **Given** the workflow executes
   **When** a node fails or the system restarts
   **Then** checkpointing is enabled for recovery

## Tasks / Subtasks

- [x] Task 1: Create workflow module structure (AC: #1, #3)
  - [x] 1.1: Create `src/yolo_developer/orchestrator/workflow.py` module
  - [x] 1.2: Define `WorkflowConfig` dataclass for workflow configuration
  - [x] 1.3: Define agent node registration system

- [x] Task 2: Implement StateGraph construction (AC: #1)
  - [x] 2.1: Create `build_workflow()` function that returns a compiled StateGraph
  - [x] 2.2: Add all existing agent nodes (analyst, pm, architect, dev, tea) to the graph
  - [x] 2.3: Define standard edges between agents for linear flow
  - [x] 2.4: Set entry point to analyst node

- [x] Task 3: Implement conditional routing (AC: #2)
  - [x] 3.1: Create routing functions for conditional edges
  - [x] 3.2: Add `route_after_analyst()` - routes to PM or escalates
  - [x] 3.3: Add `route_after_pm()` - routes to architect or dev based on needs
  - [x] 3.4: Add `route_after_architect()` - routes to dev
  - [x] 3.5: Add `route_after_dev()` - routes to TEA
  - [x] 3.6: Add `route_after_tea()` - routes to END or back to dev if issues found
  - [x] 3.7: Use `add_conditional_edges()` with routing functions

- [x] Task 4: Integrate checkpointing (AC: #4)
  - [x] 4.1: Integrate with LangGraph's `BaseCheckpointSaver` interface (compatible with project's Checkpointer)
  - [x] 4.2: Configure graph compilation with checkpointer parameter
  - [x] 4.3: Add `create_workflow_with_checkpointing()` function
  - [x] 4.4: Auto-create MemorySaver when `enable_checkpointing=True` and no checkpointer provided
  - [x] 4.5: Auto-generate thread_id when checkpointing enabled but no thread_id specified

- [x] Task 5: Add workflow execution interface (AC: #1, #2, #3, #4)
  - [x] 5.1: Create `run_workflow()` async function for single execution
  - [x] 5.2: Create `stream_workflow()` async generator for streaming execution
  - [x] 5.3: Add initial state creation helper

- [x] Task 6: Update exports and test (AC: all)
  - [x] 6.1: Export new functions from `orchestrator/__init__.py`
  - [x] 6.2: Write unit tests for graph construction
  - [x] 6.3: Write unit tests for routing functions
  - [x] 6.4: Write integration test for workflow execution with mocked agents

## Dev Notes

### Architecture Requirements

This story implements **ADR-005: Inter-Agent Communication** which specifies:
- LangGraph edges define explicit handoff conditions
- State machine transitions are predictable and replayable
- Message accumulation via reducers for audit trail
- Matches SM as control plane pattern

Per ADR-005, the StateGraph pattern:
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(YoloState)

# Add agent nodes
workflow.add_node("analyst", analyst_node)
workflow.add_node("pm", pm_node)
workflow.add_node("architect", architect_node)
workflow.add_node("dev", dev_node)
workflow.add_node("tea", tea_node)

# Define handoffs via edges
workflow.add_edge("analyst", "pm")
workflow.add_conditional_edges("pm", route_after_pm)
workflow.add_edge("architect", "dev")
```

### Existing Infrastructure to Use

**State Management** (already implemented in `orchestrator/state.py`):
```python
class YoloState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
```

**Checkpointing** (already implemented in `orchestrator/graph.py`):
```python
class Checkpointer:
    """Session checkpointer for graph execution."""
    async def checkpoint(self, state: YoloState) -> str
    async def resume(self) -> tuple[YoloState, SessionMetadata]
```

**Available Agent Nodes** (from `agents/__init__.py`):
- `analyst_node` - Requirement crystallization (Epic 5, complete)
- `pm_node` - Story transformation (Epic 6, complete)
- `architect_node` - Design decisions (Epic 7, complete)
- `dev_node` - Code implementation (Epic 8, complete)
- `tea_node` - Validation and QA (Epic 9, complete)

**Node Registry** (already started in `orchestrator/graph.py`):
```python
def get_agent_nodes() -> dict[str, AgentNode]:
    """Returns dict mapping agent name to node function."""
```

### Routing Decision Points

Based on the architecture, routing decisions should check:

1. **After Analyst → PM or Escalate**:
   - Check if `requirements` field has crystallized requirements
   - Check if `escalate_to_human` flag is set

2. **After PM → Architect or Dev**:
   - Check if `needs_architecture` flag is set (new design needed)
   - If stories don't require architectural decisions, go directly to dev

3. **After Architect → Dev**:
   - Always proceed to dev after design

4. **After Dev → TEA**:
   - Always proceed to TEA for validation

5. **After TEA → END or Dev**:
   - Check `deployment_decision.should_deploy` from TEA output
   - If blocking issues, can route back to dev for fixes
   - If validation passes, route to END

### State Fields for Routing

Add fields to track routing decisions (use existing YoloState or extend):
- `needs_architecture: bool` - Set by PM when design needed
- `escalate_to_human: bool` - Set by any agent when stuck
- `gate_blocked: bool` - Set by quality gates on failure
- `deployment_ready: bool` - Set by TEA when validation passes

### Testing Strategy

**Unit Tests:**
- Test graph construction builds correct nodes and edges
- Test each routing function returns correct next node
- Test conditional edges are added correctly

**Integration Tests:**
- Test workflow execution with mocked agent nodes
- Test checkpointing saves and restores state correctly
- Test conditional routing executes correct paths

### Dependencies

**Required packages (already in pyproject.toml):**
- `langgraph>=0.2.0` - StateGraph, conditional edges
- `langchain-core>=0.3.0` - Message types

**Internal dependencies:**
- `yolo_developer.orchestrator.state` - YoloState TypedDict
- `yolo_developer.orchestrator.graph` - Checkpointer, wrap_node
- `yolo_developer.orchestrator.session` - SessionManager
- `yolo_developer.agents` - All agent node functions

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/orchestrator/workflow.py` - Main workflow module (NEW)
- `tests/unit/orchestrator/test_workflow.py` - Unit tests including mocked execution tests (NEW)

**Files to modify:**
- `src/yolo_developer/orchestrator/__init__.py` - Add exports

### Implementation Patterns

Per architecture document, follow these patterns:

1. **Async-first**: All workflow execution functions must be async
2. **State updates via dict**: Nodes return dict updates, don't mutate state
3. **Structured logging**: Use structlog for all logging
4. **Type annotations**: Full type hints on all functions
5. **Error handling**: Use tenacity retry for LLM calls (handled in nodes)

```python
# CORRECT pattern for workflow module
from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger()

async def run_workflow(initial_state: YoloState) -> YoloState:
    """Execute the orchestration workflow."""
    logger.info("workflow_started", current_agent=initial_state.get("current_agent"))
    ...
```

### LangGraph API Reference (v0.2+)

```python
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Build graph
builder = StateGraph(YoloState)
builder.add_node("agent_name", agent_function)
builder.add_edge("source", "target")
builder.add_conditional_edges(
    "source",
    routing_function,
    {"result1": "target1", "result2": "target2"}
)
builder.set_entry_point("analyst")
builder.set_finish_point("tea")  # Or use END

# Compile with checkpointing
checkpointer = MemorySaver()  # Or use custom Checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Execute
config = {"configurable": {"thread_id": "session-123"}}
result = await graph.ainvoke(initial_state, config)

# Stream
async for event in graph.astream(initial_state, config):
    print(event)
```

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.1]
- [Source: src/yolo_developer/orchestrator/state.py]
- [Source: src/yolo_developer/orchestrator/graph.py]
- [LangGraph Application Structure](https://docs.langchain.com/langgraph-platform/application-structure)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- Created `workflow.py` module with LangGraph StateGraph implementation
- Implemented `WorkflowConfig` frozen dataclass with entry_point and enable_checkpointing options
- Implemented `get_default_agent_nodes()` returning all 5 agent nodes (analyst, pm, architect, dev, tea)
- Implemented 5 routing functions: `route_after_analyst`, `route_after_pm`, `route_after_architect`, `route_after_dev`, `route_after_tea`
- Routing uses state flags: `escalate_to_human`, `needs_architecture`, `gate_blocked` for conditional edges
- Implemented `build_workflow()` with full conditional edge configuration
- Implemented `create_workflow_with_checkpointing()` for checkpointer integration
- Implemented `run_workflow()` async execution function
- Implemented `stream_workflow()` async generator for streaming events
- Implemented `create_initial_state()` helper for workflow initialization
- Fixed `orchestrator/state.py` to import `HandoffContext` and `Decision` at runtime (required for LangGraph StateGraph introspection)
- Updated `orchestrator/__init__.py` with all new exports
- 44 unit tests covering all functionality
- All orchestrator tests pass with no regressions

**Code Review Fixes (2026-01-12):**
- H1: Changed from standard `logging` to `structlog` with key-value logging style
- M1: Updated task description to clarify use of LangGraph's `BaseCheckpointSaver` interface
- M2: Updated Dev Notes to reflect tests are in unit tests file, not separate integration file
- M3: Added `sprint-status.yaml` to File List
- M4: Made `enable_checkpointing` config actually functional - auto-creates MemorySaver when True
- Added auto-generation of thread_id when checkpointing enabled but no thread_id provided
- Added 2 new tests for checkpointing config behavior

### Change Log

- 2026-01-12: Initial implementation of LangGraph workflow module (Story 10.1)

### File List

**New files:**
- src/yolo_developer/orchestrator/workflow.py
- tests/unit/orchestrator/test_workflow.py

**Modified files:**
- src/yolo_developer/orchestrator/__init__.py
- src/yolo_developer/orchestrator/state.py
- _bmad-output/implementation-artifacts/sprint-status.yaml
