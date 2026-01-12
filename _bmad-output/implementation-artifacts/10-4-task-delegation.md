# Story 10.4: Task Delegation

Status: done

## Story

As a developer,
I want tasks delegated to appropriate agents,
So that each agent handles work within its expertise.

## Acceptance Criteria

1. **Given** work to be done
   **When** the SM delegates
   **Then** the appropriate agent receives the task

2. **Given** work is delegated
   **When** context is prepared
   **Then** context is passed with the delegation

3. **Given** a delegation occurs
   **When** the delegation is processed
   **Then** delegation is logged

4. **Given** an agent receives a delegation
   **When** the agent acknowledges
   **Then** acknowledgment is verified

## Tasks / Subtasks

- [x] Task 1: Create delegation module structure (AC: #1, #2)
  - [x] 1.1: Create `src/yolo_developer/agents/sm/delegation.py` module
  - [x] 1.2: Create `src/yolo_developer/agents/sm/delegation_types.py` for delegation types
  - [x] 1.3: Define `DelegationRequest`, `DelegationResult`, and `DelegationConfig` frozen dataclasses
  - [x] 1.4: Export delegation functions from SM agent package

- [x] Task 2: Implement task analysis and agent matching logic (AC: #1)
  - [x] 2.1: Implement `_analyze_task()` to extract task requirements and type
  - [x] 2.2: Implement `_match_agent()` to determine best agent for task based on expertise
  - [x] 2.3: Implement `_validate_agent_availability()` to check agent can accept work
  - [x] 2.4: Create agent expertise mapping (per FR10, each agent has defined expertise)

- [x] Task 3: Implement context preparation and handoff (AC: #2)
  - [x] 3.1: Implement `_prepare_delegation_context()` to gather relevant context
  - [x] 3.2: Extract relevant state fields for target agent
  - [x] 3.3: Include handoff notes and task-specific instructions
  - [x] 3.4: Use existing `HandoffContext` from `orchestrator/context.py`

- [x] Task 4: Implement delegation logging (AC: #3)
  - [x] 4.1: Log delegation events with structlog (source_agent, target_agent, task_type)
  - [x] 4.2: Create `Decision` record for delegation for audit trail
  - [x] 4.3: Track delegation metrics (count, success/failure)
  - [x] 4.4: Include delegation in `SMOutput.processing_notes`

- [x] Task 5: Implement acknowledgment verification (AC: #4)
  - [x] 5.1: Implement `_verify_acknowledgment()` to check agent accepted task
  - [x] 5.2: Define acknowledgment timeout and retry behavior
  - [x] 5.3: Handle unacknowledged delegations (re-route or escalate)
  - [x] 5.4: Create `DelegationResult` with acknowledgment status

- [x] Task 6: Implement main delegation function (AC: #1, #2, #3, #4)
  - [x] 6.1: Implement async `delegate_task()` main function
  - [x] 6.2: Orchestrate analysis → matching → context → delegation → acknowledgment
  - [x] 6.3: Return `DelegationResult` with full delegation details
  - [x] 6.4: Handle errors and edge cases gracefully

- [x] Task 7: Integrate with SM node (AC: all)
  - [x] 7.1: Add `delegate_task()` call in `sm_node` for task delegation scenarios
  - [x] 7.2: Update `SMOutput` type to include `delegation_result` field (optional)
  - [x] 7.3: Export `DelegationRequest`, `DelegationResult`, `DelegationConfig` from SM package
  - [x] 7.4: Wire delegation result into state updates

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1: Create `tests/unit/agents/sm/test_delegation.py`
  - [x] 8.2: Create `tests/unit/agents/sm/test_delegation_types.py`
  - [x] 8.3: Test task analysis for various task types
  - [x] 8.4: Test agent matching logic with expertise mapping
  - [x] 8.5: Test context preparation completeness
  - [x] 8.6: Test delegation logging and audit trail
  - [x] 8.7: Test acknowledgment verification flow
  - [x] 8.8: Test full delegation flow end-to-end

## Dev Notes

### Architecture Requirements

This story implements **FR10: SM Agent can delegate tasks to appropriate specialized agents**.

Per the architecture document and ADR-005/ADR-007:
- SM is the control plane for orchestration decisions
- State-based routing with explicit handoff conditions
- All operations should be async
- Return state updates, never mutate input state
- Use frozen dataclasses for immutable types

### Related FRs

- **FR10**: SM Agent can delegate tasks to appropriate specialized agents (PRIMARY)
- **FR15**: System can handle agent handoffs with context preservation
- **FR68**: SM Agent can trigger inter-agent sync protocols for blocking issues
- **FR69**: SM Agent can inject context when agents lack information

### Existing Infrastructure to Use

**SM Agent Module** (`agents/sm/` - Stories 10.2, 10.3):

```python
# types.py has:
RoutingDecision = Literal["analyst", "pm", "architect", "dev", "tea", "sm", "escalate"]
SMOutput  # Contains routing_decision, routing_rationale, sprint_plan, etc.
AgentExchange  # For tracking message exchanges
NATURAL_SUCCESSOR  # Mapping of agent to natural successor

# node.py has:
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node for orchestration control plane."""
    # Uses _get_next_agent() for routing
    # Uses _check_for_escalation() for escalation handling
    # Returns dict with messages, decisions, sm_output

# planning.py has:
async def plan_sprint(stories: list[dict], config: PlanningConfig) -> SprintPlan:
    """Generate sprint plan from stories."""
```

**Orchestrator Context** (`orchestrator/context.py`):

```python
@dataclass(frozen=True)
class HandoffContext:
    """Context passed during agent handoffs."""
    source_agent: str
    target_agent: str
    task_summary: str
    relevant_state_keys: tuple[str, ...]
    instructions: str = ""
    priority: Literal["low", "normal", "high", "critical"] = "normal"

@dataclass(frozen=True)
class Decision:
    agent: str
    summary: str
    rationale: str
    timestamp: datetime
    related_artifacts: tuple[str, ...]
```

**State Management** (`orchestrator/state.py`):

```python
class YoloState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
    gate_blocked: bool
    escalate_to_human: bool
    needs_architecture: bool
    # ... other fields
```

### Task Delegation Data Model

Per FR10 and workflow requirements, task delegation needs:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TaskType = Literal[
    "requirement_analysis",    # For Analyst
    "story_creation",          # For PM
    "architecture_design",     # For Architect
    "implementation",          # For Dev
    "validation",              # For TEA
    "orchestration",           # For SM (self)
]

@dataclass(frozen=True)
class DelegationRequest:
    """Request to delegate a task to an agent."""
    task_type: TaskType
    task_description: str
    source_agent: str
    target_agent: str
    context: dict[str, Any]
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "task_description": self.task_description,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "context": self.context,
            "priority": self.priority,
            "created_at": self.created_at,
        }

@dataclass(frozen=True)
class DelegationResult:
    """Result of a delegation attempt."""
    request: DelegationRequest
    success: bool
    acknowledged: bool
    acknowledgment_timestamp: str | None = None
    error_message: str | None = None
    handoff_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "success": self.success,
            "acknowledged": self.acknowledged,
            "acknowledgment_timestamp": self.acknowledgment_timestamp,
            "error_message": self.error_message,
            "handoff_context": self.handoff_context,
        }

@dataclass(frozen=True)
class DelegationConfig:
    """Configuration for delegation behavior."""
    acknowledgment_timeout_seconds: float = 30.0
    max_retry_attempts: int = 3
    allow_self_delegation: bool = False
```

### Agent Expertise Mapping

Per FR10, each agent has defined expertise:

```python
AGENT_EXPERTISE: dict[str, tuple[TaskType, ...]] = {
    "analyst": ("requirement_analysis",),
    "pm": ("story_creation",),
    "architect": ("architecture_design",),
    "dev": ("implementation",),
    "tea": ("validation",),
    "sm": ("orchestration",),
}

TASK_TO_AGENT: dict[TaskType, str] = {
    "requirement_analysis": "analyst",
    "story_creation": "pm",
    "architecture_design": "architect",
    "implementation": "dev",
    "validation": "tea",
    "orchestration": "sm",
}
```

### Delegation Flow Algorithm

```python
async def delegate_task(
    state: YoloState,
    task_type: TaskType,
    task_description: str,
    priority: Literal["low", "normal", "high", "critical"] = "normal",
    config: DelegationConfig | None = None,
) -> DelegationResult:
    """Delegate a task to the appropriate agent (FR10).

    Args:
        state: Current orchestration state
        task_type: Type of task to delegate
        task_description: Description of what needs to be done
        priority: Task priority level
        config: Delegation configuration

    Returns:
        DelegationResult with delegation details and acknowledgment status
    """
    config = config or DelegationConfig()
    source_agent = state.get("current_agent", "sm")

    # Step 1: Analyze task and match agent
    target_agent = _match_agent(task_type)

    # Step 2: Validate agent availability
    if not await _validate_agent_availability(target_agent, state):
        # Handle unavailable agent (escalate or fallback)
        ...

    # Step 3: Prepare delegation context
    context = _prepare_delegation_context(state, task_type, target_agent)

    # Step 4: Create delegation request
    request = DelegationRequest(
        task_type=task_type,
        task_description=task_description,
        source_agent=source_agent,
        target_agent=target_agent,
        context=context,
        priority=priority,
    )

    # Step 5: Log delegation
    logger.info(
        "task_delegation",
        source_agent=source_agent,
        target_agent=target_agent,
        task_type=task_type,
        priority=priority,
    )

    # Step 6: Verify acknowledgment (in this design, this is implicit via state update)
    acknowledged = True  # Will be true when agent picks up the task

    # Step 7: Create handoff context for state
    handoff = HandoffContext(
        source_agent=source_agent,
        target_agent=target_agent,
        task_summary=task_description,
        relevant_state_keys=_get_relevant_state_keys(task_type),
        instructions=f"Delegated {task_type} task with {priority} priority",
        priority=priority,
    )

    return DelegationResult(
        request=request,
        success=True,
        acknowledged=acknowledged,
        acknowledgment_timestamp=datetime.now(timezone.utc).isoformat(),
        handoff_context=handoff.to_dict() if isinstance(handoff, HandoffContext) else handoff,
    )
```

### Context Preparation

Per FR15 (context preservation) and FR69 (context injection):

```python
def _prepare_delegation_context(
    state: YoloState,
    task_type: TaskType,
    target_agent: str,
) -> dict[str, Any]:
    """Prepare context for delegation.

    Extracts relevant state information for the target agent
    based on task type and agent needs.
    """
    context: dict[str, Any] = {}

    # Always include core context
    context["message_count"] = len(state.get("messages", []))
    context["decision_count"] = len(state.get("decisions", []))

    # Task-specific context
    if task_type == "requirement_analysis":
        context["seed_input"] = state.get("seed_input")
    elif task_type == "story_creation":
        context["requirements"] = state.get("requirements", [])
    elif task_type == "architecture_design":
        context["stories"] = state.get("stories", [])
        context["requirements"] = state.get("requirements", [])
    elif task_type == "implementation":
        context["current_story"] = state.get("current_story")
        context["design"] = state.get("design")
    elif task_type == "validation":
        context["implementation"] = state.get("implementation")
        context["test_results"] = state.get("test_results")

    # Include existing handoff context if present
    if state.get("handoff_context"):
        context["previous_handoff"] = state.get("handoff_context")

    return context

def _get_relevant_state_keys(task_type: TaskType) -> tuple[str, ...]:
    """Get state keys relevant for a task type."""
    base_keys = ("messages", "decisions", "current_agent")

    task_keys: dict[TaskType, tuple[str, ...]] = {
        "requirement_analysis": ("seed_input", "sop_constraints"),
        "story_creation": ("requirements", "priorities"),
        "architecture_design": ("stories", "requirements", "tech_stack"),
        "implementation": ("current_story", "design", "patterns"),
        "validation": ("implementation", "test_results", "coverage"),
        "orchestration": ("sprint_plan", "health_metrics"),
    }

    return base_keys + task_keys.get(task_type, ())
```

### Integration with SM Node

The delegation function integrates with the existing `sm_node`:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("sm_routing", blocking=False)
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node with delegation support."""
    logger.info("sm_node_start", current_agent=state.get("current_agent"))

    # Existing routing logic...
    routing_decision, rationale = _get_next_agent(state)

    # NEW: If routing to an agent, create delegation
    delegation_result = None
    if routing_decision not in ("escalate", "sm"):
        # Determine task type from routing decision
        task_type = _routing_to_task_type(routing_decision)
        delegation_result = await delegate_task(
            state=state,
            task_type=task_type,
            task_description=rationale,
        )

    # Update SMOutput with delegation result
    output = SMOutput(
        routing_decision=routing_decision,
        routing_rationale=rationale,
        delegation_result=delegation_result.to_dict() if delegation_result else None,
        # ... existing fields
    )

    return {
        "messages": [message],
        "decisions": [decision],
        "sm_output": output.to_dict(),
        "routing_decision": routing_decision,
        "handoff_context": delegation_result.handoff_context if delegation_result else None,
    }
```

### Testing Strategy

**Unit Tests:**
- Test task analysis for each task type
- Test agent matching with expertise map
- Test agent availability validation
- Test context preparation for each task type
- Test delegation request creation
- Test delegation result creation
- Test acknowledgment verification

**Integration Tests:**
- Test full delegation flow with state
- Test delegation logging and audit trail
- Test integration with sm_node
- Test error handling for unavailable agents

### Previous Story Intelligence

From **Story 10.3** (Sprint Planning):
- Used frozen dataclasses with `to_dict()` serialization
- Created separate types module (`planning_types.py`) for clarity
- Exported all new types and functions from `__init__.py`
- Used structlog for consistent logging format
- All functions are async
- Comprehensive test coverage (47 tests for 2 files)
- Code review applied: removed unnecessary pass, added input validation

From **Story 10.2** (SM Agent Node):
- SM node uses `@retry` decorator with exponential backoff
- Uses `@quality_gate("sm_routing", blocking=False)`
- Returns dict with messages, decisions, sm_output
- `SMOutput` dataclass handles routing rationale and metadata
- Circular logic detection already implemented in `_check_for_circular_logic()`
- Uses `create_agent_message()` helper from `orchestrator/state.py`
- `Decision` dataclass from `orchestrator/context.py` for audit trail

**Key Pattern to Follow:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("sm_routing", blocking=False)
async def sm_node(state: YoloState) -> dict[str, Any]:
    # Return ONLY updates
    return {
        "messages": [message],
        "decisions": [decision],
        "sm_output": output.to_dict(),
    }
```

### Git Intelligence

Recent commits show consistent patterns:
- `9a54501`: Story 10.3 sprint planning with code review fixes
- `0073828`: Code review fixes for Story 10.2
- `dd711de`: SM agent node implementation
- `c859c80`: LangGraph workflow orchestration

Commit message pattern: `feat: <description> with code review fixes (Story X.Y)`

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/agents/sm/delegation.py` - Main delegation module (NEW)
- `src/yolo_developer/agents/sm/delegation_types.py` - Delegation types (NEW)
- `tests/unit/agents/sm/test_delegation.py` - Delegation tests (NEW)
- `tests/unit/agents/sm/test_delegation_types.py` - Types tests (NEW)

**Files to modify:**
- `src/yolo_developer/agents/sm/__init__.py` - Export delegation functions
- `src/yolo_developer/agents/sm/types.py` - Add `delegation_result` field to SMOutput
- `src/yolo_developer/agents/sm/node.py` - Integrate delegation into sm_node

### Implementation Patterns

Per architecture document, follow these patterns:

1. **Async-first**: `delegate_task()` must be async
2. **State updates via dict**: Return dict updates, don't mutate state
3. **Structured logging**: Use structlog with key-value format
4. **Type annotations**: Full type hints on all functions
5. **Immutable outputs**: Use frozen dataclasses for types
6. **snake_case**: All state dictionary keys use snake_case

```python
# CORRECT pattern for delegation module
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.sm.delegation_types import (
    DelegationConfig,
    DelegationRequest,
    DelegationResult,
    TaskType,
)

logger = structlog.get_logger(__name__)

async def delegate_task(
    state: YoloState,
    task_type: TaskType,
    task_description: str,
    priority: Literal["low", "normal", "high", "critical"] = "normal",
    config: DelegationConfig | None = None,
) -> DelegationResult:
    """Delegate task to appropriate agent (FR10)."""
    logger.info(
        "delegation_started",
        task_type=task_type,
        source_agent=state.get("current_agent", "sm"),
    )

    # ... implementation ...

    logger.info(
        "delegation_complete",
        target_agent=request.target_agent,
        acknowledged=result.acknowledged,
    )

    return result
```

### Dependencies

**Internal dependencies:**
- `yolo_developer.agents.sm.types` - SMOutput, RoutingDecision
- `yolo_developer.agents.sm.node` - sm_node function
- `yolo_developer.orchestrator.context` - Decision, HandoffContext
- `yolo_developer.orchestrator.state` - YoloState, create_agent_message
- `yolo_developer.gates` - quality_gate decorator
- `structlog` - logging
- `tenacity` - retry decorator

**No new external dependencies needed.**

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.4]
- [Source: _bmad-output/planning-artifacts/epics.md#FR10]
- [Source: _bmad-output/planning-artifacts/epics.md#FR15]
- [Source: _bmad-output/planning-artifacts/epics.md#FR68]
- [Source: _bmad-output/planning-artifacts/epics.md#FR69]
- [Source: src/yolo_developer/agents/sm/node.py - pattern reference]
- [Source: src/yolo_developer/agents/sm/types.py - type patterns]
- [Source: src/yolo_developer/agents/sm/planning.py - delegation pattern]
- [Source: src/yolo_developer/orchestrator/context.py - HandoffContext]
- [Source: _bmad-output/implementation-artifacts/10-3-sprint-planning.md]
- [Source: _bmad-output/implementation-artifacts/10-2-sm-agent-node-implementation.md]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation proceeded without issues

### Completion Notes List

- **Task 1**: Created delegation module structure with `delegation.py` and `delegation_types.py` files. Defined `DelegationRequest`, `DelegationResult`, and `DelegationConfig` frozen dataclasses with full type annotations and `to_dict()` serialization methods. Exported all delegation functions and types from SM agent package.

- **Task 2**: Implemented task analysis and agent matching per FR10:
  - `_analyze_task()`: Extracts keywords, estimates complexity, identifies context needs
  - `_match_agent()`: Maps TaskType to agent using TASK_TO_AGENT mapping
  - `_validate_agent_availability()`: Checks agent can accept work (not blocked by gate)
  - `AGENT_EXPERTISE` and `TASK_TO_AGENT` mappings for all 6 agents

- **Task 3**: Implemented context preparation per FR15/FR69:
  - `_prepare_delegation_context()`: Extracts task-specific state fields
  - `_get_relevant_state_keys()`: Returns state keys for each task type
  - Context includes core fields (messages, decisions) plus task-specific fields

- **Task 4**: Implemented delegation logging:
  - Uses structlog for structured logging (delegation_started, delegation_complete)
  - Logs source_agent, target_agent, task_type, priority
  - Includes audit trail via `Decision` record integration

- **Task 5**: Implemented acknowledgment verification:
  - `_verify_acknowledgment()`: Returns acknowledged=True with timestamp (implicit acknowledgment design)
  - `_handle_unacknowledged_delegation()`: Returns retry/escalate actions
  - `DelegationConfig` with `acknowledgment_timeout_seconds` and `max_retry_attempts`

- **Task 6**: Implemented async `delegate_task()` main function:
  - Orchestrates: analyze → match → validate → context → request → verify → result
  - Returns `DelegationResult` with full delegation details and handoff context
  - Handles unavailable agent gracefully with error message

- **Task 7**: Integrated with SM node:
  - Added `delegation_result` field to `SMOutput` dataclass
  - Exported all delegation types from SM package
  - Added `routing_to_task_type()` utility for SM node integration

- **Task 8**: Comprehensive test coverage:
  - `test_delegation_types.py`: 22 tests for types and constants
  - `test_delegation.py`: 53 tests for delegation logic
  - Total: 75 new tests, all passing
  - Tests cover: task analysis, agent matching, context preparation, acknowledgment, full delegation flow

### Change Log

- 2026-01-12: Implemented Story 10.4 - Task Delegation
  - Created delegation.py with delegate_task() and supporting functions
  - Created delegation_types.py with DelegationRequest, DelegationResult, DelegationConfig
  - Added delegation_result field to SMOutput for SM node integration
  - Exported all delegation types and functions from SM package
  - 75 new tests, all acceptance criteria satisfied

- 2026-01-12: Code Review Fixes Applied
  - **Issue 1 (HIGH)**: Integrated delegate_task() into sm_node - now calls delegation when routing to agents
  - **Issue 2 (HIGH)**: Wired delegation result into state updates - returns handoff_context in sm_node
  - **Issue 3 (HIGH)**: Updated acknowledgment documentation to explain implicit acknowledgment design
  - **Issue 4 (MEDIUM)**: Decision record now includes delegation info for audit trail
  - **Issue 5 (MEDIUM)**: _analyze_task() return value now used to enrich delegation context
  - **Issue 6 (MEDIUM)**: Added 5 integration tests for sm_node with delegation
  - **Issue 7 (LOW)**: delegate_task() signature now uses Priority type consistently
  - **Issue 8 (LOW)**: Updated module docstrings to clarify acknowledgment design
  - Test count: 75 → 80 (5 new integration tests)
  - All 192 SM agent tests pass

### File List

**New Files:**
- src/yolo_developer/agents/sm/delegation.py
- src/yolo_developer/agents/sm/delegation_types.py
- tests/unit/agents/sm/test_delegation.py
- tests/unit/agents/sm/test_delegation_types.py

**Modified Files:**
- src/yolo_developer/agents/sm/__init__.py (added delegation exports + Priority type)
- src/yolo_developer/agents/sm/types.py (added delegation_result field to SMOutput)
- src/yolo_developer/agents/sm/node.py (integrated delegate_task() - code review fix)

