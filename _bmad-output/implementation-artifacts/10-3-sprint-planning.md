# Story 10.3: Sprint Planning

Status: done

## Story

As a developer,
I want the SM to plan sprints automatically,
So that stories are properly sequenced for execution.

## Acceptance Criteria

1. **Given** stories with dependencies
   **When** sprint planning runs
   **Then** stories are prioritized by value and dependencies

2. **Given** a set of stories to plan
   **When** the sprint planning algorithm executes
   **Then** a feasible execution order is determined

3. **Given** sprint planning is invoked
   **When** the plan is generated
   **Then** sprint capacity is considered

4. **Given** the sprint plan is created
   **When** the plan is finalized
   **Then** the plan is logged for audit

## Tasks / Subtasks

- [x] Task 1: Create sprint planning module structure (AC: #1, #2)
  - [x] 1.1: Create `src/yolo_developer/agents/sm/planning.py` module
  - [x] 1.2: Create `src/yolo_developer/agents/sm/planning_types.py` for sprint planning types
  - [x] 1.3: Define `SprintPlan`, `SprintStory`, and `PlanningConfig` frozen dataclasses
  - [x] 1.4: Export planning functions from SM agent package

- [x] Task 2: Implement story prioritization logic (AC: #1)
  - [x] 2.1: Implement `_calculate_priority_score()` for weighted story scoring
  - [x] 2.2: Implement `_analyze_dependencies()` to build dependency graph
  - [x] 2.3: Implement `_topological_sort()` for dependency-aware ordering
  - [x] 2.4: Weight factors: value, dependencies, velocity, tech debt (per FR65)

- [x] Task 3: Implement sprint capacity management (AC: #3)
  - [x] 3.1: Implement capacity via `PlanningConfig` dataclass with `max_stories` and `max_points`
  - [x] 3.2: Story points via `SprintStory.estimated_points` field
  - [x] 3.3: Implement `_check_capacity()` to validate sprint fits capacity
  - [x] 3.4: Support configurable capacity limits via PlanningConfig

- [x] Task 4: Implement sprint plan generation (AC: #2, #4)
  - [x] 4.1: Implement async `plan_sprint()` main function
  - [x] 4.2: Generate feasible execution order respecting dependencies
  - [x] 4.3: Create SprintPlan output with ordered stories and rationale
  - [x] 4.4: Integrate with audit logging via `planning_rationale` and `created_at` fields

- [x] Task 5: Integrate with SM node (AC: #1, #2, #3, #4)
  - [x] 5.1: Add `plan_sprint()` exported from SM agent package
  - [x] 5.2: Update SMOutput type to include sprint_plan field (optional)
  - [x] 5.3: Export CircularDependencyError for error handling
  - [x] 5.4: Ensure plan can be persisted via `to_dict()` serialization

- [x] Task 6: Write comprehensive tests (AC: all)
  - [x] 6.1: Create `tests/unit/agents/sm/test_planning.py`
  - [x] 6.2: Test priority scoring with various story attributes
  - [x] 6.3: Test dependency analysis with circular detection
  - [x] 6.4: Test capacity constraints are respected
  - [x] 6.5: Test full sprint plan generation
  - [x] 6.6: Test audit logging of sprint plans

## Dev Notes

### Architecture Requirements

This story implements **FR9: SM Agent can plan sprints by prioritizing and sequencing stories** and **FR65: SM Agent can calculate weighted priority scores for story selection**.

Per the architecture document and ADR-005/ADR-007:
- SM is the control plane for orchestration decisions
- State-based routing with explicit handoff conditions
- All operations should be async
- Return state updates, never mutate input state

### Existing Infrastructure to Use

**SM Agent Module** (`agents/sm/` - Story 10.2):
```python
# types.py already has:
RoutingDecision = Literal["analyst", "pm", "architect", "dev", "tea", "sm", "escalate"]
SMOutput  # Contains routing_decision, routing_rationale, etc.
AgentExchange  # For tracking message exchanges

# node.py has:
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node for orchestration control plane."""
```

**Workflow Module** (`orchestrator/workflow.py` - Story 10.1):
```python
# Workflow already supports SM as a node
def get_default_agent_nodes() -> dict[str, AgentNode]:
    # Returns dict including sm_node

# Conditional routing infrastructure exists
def route_after_analyst(state: YoloState) -> str
def route_after_pm(state: YoloState) -> str
# etc.
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
```

### Sprint Planning Data Model

Per FR9, FR65, and PRD requirements, sprint planning needs:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

@dataclass(frozen=True)
class SprintStory:
    """A story prepared for sprint planning."""
    story_id: str
    title: str
    priority_score: float  # Composite weighted score
    dependencies: tuple[str, ...]  # Story IDs this depends on
    estimated_points: int  # Story points estimate
    value_score: float  # Business value component
    tech_debt_score: float  # Tech debt reduction component
    velocity_impact: float  # Expected velocity impact
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SprintPlan:
    """Complete sprint plan output."""
    sprint_id: str
    stories: tuple[SprintStory, ...]  # Ordered by execution sequence
    total_points: int
    capacity_used: float  # Percentage of capacity
    planning_rationale: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass(frozen=True)
class PlanningConfig:
    """Configuration for sprint planning."""
    max_stories: int = 10  # MVP: 5-10 stories per sprint
    max_points: int = 40  # Default sprint capacity
    value_weight: float = 0.4  # Priority scoring weights
    dependency_weight: float = 0.3
    velocity_weight: float = 0.2
    tech_debt_weight: float = 0.1
```

### Priority Scoring Algorithm (FR65)

Per FR65, weighted priority scores should consider:

1. **Value Score (40%)**: Business value of the story
   - High value features: 1.0
   - Medium value: 0.6
   - Low value: 0.3

2. **Dependency Score (30%)**: How many stories depend on this
   - Blockers for many: 1.0 (prioritize to unblock)
   - Some dependents: 0.5
   - No dependents: 0.2

3. **Velocity Impact (20%)**: Expected effect on team velocity
   - Quick wins: 1.0 (build momentum)
   - Standard: 0.5
   - Complex: 0.3

4. **Tech Debt Score (10%)**: Reduces tech debt
   - Significant reduction: 1.0
   - Some reduction: 0.5
   - No impact: 0.0

```python
def _calculate_priority_score(story: SprintStory, config: PlanningConfig) -> float:
    """Calculate weighted priority score per FR65."""
    return (
        story.value_score * config.value_weight +
        story.dependency_score * config.dependency_weight +
        story.velocity_impact * config.velocity_weight +
        story.tech_debt_score * config.tech_debt_weight
    )
```

### Dependency Analysis

Use topological sort to order stories respecting dependencies:

```python
from collections import defaultdict, deque

def _topological_sort(
    stories: list[SprintStory],
) -> list[SprintStory]:
    """Order stories respecting dependencies using Kahn's algorithm."""
    # Build adjacency list and in-degree count
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {s.story_id: 0 for s in stories}
    story_map = {s.story_id: s for s in stories}

    for story in stories:
        for dep in story.dependencies:
            if dep in story_map:
                graph[dep].append(story.story_id)
                in_degree[story.story_id] += 1

    # Process nodes with no dependencies first
    queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
    result: list[SprintStory] = []

    while queue:
        # Sort by priority among available stories
        available = sorted(queue, key=lambda sid: -story_map[sid].priority_score)
        current = available[0]
        queue = deque([x for x in queue if x != current])
        result.append(story_map[current])

        for dependent in graph[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Check for circular dependencies
    if len(result) != len(stories):
        raise ValueError("Circular dependency detected in stories")

    return result
```

### Capacity Management

Per NFR-SCALE-1, MVP supports 5-10 stories per sprint:

```python
def _check_capacity(
    stories: list[SprintStory],
    config: PlanningConfig,
) -> tuple[list[SprintStory], int]:
    """Select stories that fit within capacity."""
    selected: list[SprintStory] = []
    total_points = 0

    for story in stories:
        if len(selected) >= config.max_stories:
            break
        if total_points + story.estimated_points > config.max_points:
            continue
        selected.append(story)
        total_points += story.estimated_points

    return selected, total_points
```

### Integration with SM Node

The `plan_sprint()` function will be callable from `sm_node` when planning mode is active:

```python
async def plan_sprint(
    stories: list[dict[str, Any]],
    config: PlanningConfig | None = None,
) -> SprintPlan:
    """Generate a sprint plan from available stories (FR9)."""
    config = config or PlanningConfig()

    # Convert to SprintStory objects
    sprint_stories = [_dict_to_sprint_story(s) for s in stories]

    # Calculate priority scores
    for story in sprint_stories:
        story = story._replace(
            priority_score=_calculate_priority_score(story, config)
        )

    # Sort by dependencies (topological) then priority
    ordered = _topological_sort(sprint_stories)

    # Apply capacity constraints
    selected, total_points = _check_capacity(ordered, config)

    # Create plan
    return SprintPlan(
        sprint_id=f"sprint-{datetime.now().strftime('%Y%m%d')}",
        stories=tuple(selected),
        total_points=total_points,
        capacity_used=total_points / config.max_points,
        planning_rationale=_generate_planning_rationale(selected, config),
    )
```

### Testing Strategy

**Unit Tests:**
- Test priority score calculation with various weights
- Test dependency graph building
- Test topological sort with valid DAG
- Test circular dependency detection
- Test capacity constraints respected
- Test sprint plan generation

**Integration Tests:**
- Test `plan_sprint()` with realistic story data
- Test integration with sm_node
- Test audit logging of sprint plans

### Previous Story Intelligence

From **Story 10.1** (LangGraph Workflow):
- Workflow module uses `build_workflow()` returning compiled StateGraph
- Routing functions exist for conditional edges
- `get_default_agent_nodes()` returns all agent nodes including sm
- `run_workflow()` and `stream_workflow()` async functions available

From **Story 10.2** (SM Agent Node):
- SM node implemented with routing decision logic
- Uses `@retry` decorator with exponential backoff
- Uses `@quality_gate("sm_routing", blocking=False)`
- Returns dict with messages, decisions, sm_output, routing_decision
- SMOutput dataclass handles routing rationale and metadata
- Circular logic detection already implemented

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

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/agents/sm/planning.py` - Main planning module (NEW)
- `src/yolo_developer/agents/sm/planning_types.py` - Sprint planning types (NEW)
- `tests/unit/agents/sm/test_planning.py` - Planning tests (NEW)

**Files to modify:**
- `src/yolo_developer/agents/sm/__init__.py` - Export planning functions
- `src/yolo_developer/agents/sm/types.py` - Add sprint_plan field to SMOutput if needed
- `src/yolo_developer/agents/sm/node.py` - Integrate plan_sprint capability

### Implementation Patterns

Per architecture document, follow these patterns:

1. **Async-first**: `plan_sprint()` must be async
2. **State updates via dict**: Return dict updates, don't mutate state
3. **Structured logging**: Use structlog with key-value format
4. **Type annotations**: Full type hints on all functions
5. **Immutable outputs**: Use frozen dataclasses for types (SprintPlan, SprintStory)
6. **snake_case**: All state dictionary keys use snake_case

```python
# CORRECT pattern for planning module
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.sm.planning_types import (
    PlanningConfig,
    SprintPlan,
    SprintStory,
)

logger = structlog.get_logger(__name__)

async def plan_sprint(
    stories: list[dict[str, Any]],
    config: PlanningConfig | None = None,
) -> SprintPlan:
    """Generate sprint plan from stories (FR9)."""
    logger.info(
        "sprint_planning_started",
        story_count=len(stories),
    )

    # ... implementation ...

    logger.info(
        "sprint_planning_complete",
        sprint_id=plan.sprint_id,
        selected_stories=len(plan.stories),
        total_points=plan.total_points,
    )

    return plan
```

### Dependencies

**Internal dependencies:**
- `yolo_developer.agents.sm.types` - SMOutput, RoutingDecision
- `yolo_developer.agents.sm.node` - sm_node function
- `yolo_developer.orchestrator.context` - Decision dataclass
- `yolo_developer.orchestrator.state` - YoloState, create_agent_message
- `yolo_developer.gates` - quality_gate decorator
- `structlog` - logging
- `tenacity` - retry decorator

**No new external dependencies needed.**

### Git Intelligence

Recent commits show:
- Story 10.2 implemented SM agent node with routing decisions (0073828, dd711de)
- Story 10.1 implemented LangGraph workflow orchestration (c859c80)
- Code follows consistent patterns: async, structlog, quality_gate decorator, return dict updates

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.3]
- [Source: _bmad-output/planning-artifacts/epics.md#FR9]
- [Source: _bmad-output/planning-artifacts/epics.md#FR65]
- [Source: src/yolo_developer/agents/sm/node.py - pattern reference]
- [Source: src/yolo_developer/agents/sm/types.py - type patterns]
- [Source: src/yolo_developer/orchestrator/workflow.py]
- [Source: _bmad-output/implementation-artifacts/10-1-create-langgraph-workflow.md]
- [Source: _bmad-output/implementation-artifacts/10-2-sm-agent-node-implementation.md]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation proceeded without issues

### Completion Notes List

- **Task 1**: Created sprint planning module structure with `planning.py` and `planning_types.py` files. Defined `SprintStory`, `SprintPlan`, and `PlanningConfig` frozen dataclasses with full type annotations and `to_dict()` serialization methods. Exported all planning functions and types from SM agent package.

- **Task 2**: Implemented priority scoring algorithm per FR65:
  - `_calculate_priority_score()`: Weighted composite of value (40%), dependency (30%), velocity (20%), and tech debt (10%) scores
  - `_analyze_dependencies()`: Builds adjacency list graph and in-degree counts
  - `_topological_sort()`: Kahn's algorithm with priority-based tie-breaking
  - Default weights defined as module constants (DEFAULT_VALUE_WEIGHT, etc.)

- **Task 3**: Implemented capacity management:
  - `PlanningConfig` dataclass with `max_stories=10` and `max_points=40` defaults (per NFR-SCALE-1)
  - `_check_capacity()`: Selects stories within limits, skipping oversized stories to fit smaller ones

- **Task 4**: Implemented `plan_sprint()` async function:
  - Converts dict input to SprintStory objects
  - Recalculates priority scores
  - Applies topological sort respecting dependencies
  - Applies capacity constraints
  - Generates planning rationale for audit
  - Returns immutable SprintPlan with ISO timestamp

- **Task 5**: Integrated with SM agent:
  - Added `sprint_plan` optional field to SMOutput type
  - Exported `plan_sprint`, `CircularDependencyError`, and all planning types from SM package
  - Full `to_dict()` serialization for workflow state persistence

- **Task 6**: Comprehensive test coverage:
  - `test_planning_types.py`: 16 tests for types and constants
  - `test_planning.py`: 31 tests for planning logic
  - Total: 47 new tests, all passing
  - Tests cover: priority scoring, dependency analysis, topological sort, circular detection, capacity management, plan generation, audit logging

### Code Review Fixes Applied

- **M1**: Removed unnecessary `pass` statement from `CircularDependencyError` class
- **M2**: Added input validation in `_dict_to_sprint_story()` for required fields `story_id` and `title`
- **M3**: Moved `Sequence` import to `TYPE_CHECKING` block for better runtime performance
- **L3**: Added 2 new tests for input validation edge cases (`test_missing_story_id_raises_error`, `test_missing_title_raises_error`)

### Change Log

- 2026-01-12: Code review fixes applied (3 medium, 1 low issues fixed)
- 2026-01-12: Implemented Story 10.3 - Sprint Planning
  - Created planning.py with plan_sprint() and supporting functions
  - Created planning_types.py with SprintStory, SprintPlan, PlanningConfig
  - Added sprint_plan field to SMOutput for SM node integration
  - Exported all planning types and functions from SM package
  - 47 new tests, all acceptance criteria satisfied

### File List

**New Files:**
- src/yolo_developer/agents/sm/planning.py
- src/yolo_developer/agents/sm/planning_types.py
- tests/unit/agents/sm/test_planning.py
- tests/unit/agents/sm/test_planning_types.py

**Modified Files:**
- src/yolo_developer/agents/sm/__init__.py (added planning exports)
- src/yolo_developer/agents/sm/types.py (added sprint_plan field to SMOutput)
- _bmad-output/implementation-artifacts/sprint-status.yaml
