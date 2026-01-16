# Story 10.9: Sprint Progress Tracking

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want sprint progress visible,
So that I know how execution is proceeding.

## Acceptance Criteria

1. **Given** a sprint in progress
   **When** progress is queried
   **Then** completed stories are listed

2. **Given** a sprint in progress
   **When** progress is queried
   **Then** current story and agent are shown

3. **Given** a sprint in progress
   **When** progress is queried
   **Then** remaining work is displayed

4. **Given** a sprint in progress
   **When** progress is queried
   **Then** estimated completion is provided

## Tasks / Subtasks

- [x] Task 1: Create sprint progress types module (AC: #1, #2, #3, #4)
  - [x] 1.1: Create `src/yolo_developer/agents/sm/progress_types.py` module
  - [x] 1.2: Define `StoryStatus` Literal type (backlog, in_progress, completed, blocked, failed)
  - [x] 1.3: Define `StoryProgress` frozen dataclass (story_id, title, status, started_at, completed_at, agent_history, duration_ms)
  - [x] 1.4: Define `SprintProgressSnapshot` frozen dataclass (sprint_id, stories_completed, stories_in_progress, stories_remaining, stories_blocked, current_story, current_agent, total_stories, progress_percentage)
  - [x] 1.5: Define `CompletionEstimate` frozen dataclass (estimated_completion_time, estimated_remaining_ms, confidence, factors)
  - [x] 1.6: Define `SprintProgress` frozen dataclass (snapshot, completed_stories, in_progress_stories, remaining_stories, blocked_stories, completion_estimate, created_at)
  - [x] 1.7: Define `ProgressConfig` frozen dataclass (include_estimates, estimate_confidence_threshold, track_agent_history, include_blocked_details)
  - [x] 1.8: Add `to_dict()` method to all dataclasses for serialization
  - [x] 1.9: Define constants: DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD, VALID_STORY_STATUSES

- [x] Task 2: Implement story tracking functions (AC: #1, #2)
  - [x] 2.1: Create `src/yolo_developer/agents/sm/progress.py` module
  - [x] 2.2: Implement `_get_story_status()` to determine status from state
  - [x] 2.3: Implement `_get_completed_stories()` to list completed stories with metadata
  - [x] 2.4: Implement `_get_in_progress_stories()` to list currently in-progress stories
  - [x] 2.5: Implement `_get_remaining_stories()` to list stories not yet started
  - [x] 2.6: Implement `_get_blocked_stories()` to identify blocked stories
  - [x] 2.7: Implement `_get_current_story()` to identify the currently executing story
  - [x] 2.8: Implement `_get_current_agent()` to identify the currently executing agent

- [x] Task 3: Implement progress calculation functions (AC: #3)
  - [x] 3.1: Implement `_calculate_progress_percentage()` based on completed vs total
  - [x] 3.2: Implement `_calculate_story_duration()` from start to completion time
  - [x] 3.3: Implement `_build_progress_snapshot()` to create SprintProgressSnapshot
  - [x] 3.4: Implement `_categorize_stories_by_status()` to group stories

- [x] Task 4: Implement completion estimation functions (AC: #4)
  - [x] 4.1: Implement `_calculate_average_story_duration()` from completed stories
  - [x] 4.2: Implement `_estimate_remaining_time()` based on average duration and remaining count
  - [x] 4.3: Implement `_calculate_estimation_confidence()` based on sample size and variance
  - [x] 4.4: Implement `_build_completion_estimate()` to create CompletionEstimate
  - [x] 4.5: Implement `_get_estimation_factors()` to document what influenced the estimate

- [x] Task 5: Implement main progress tracking function (AC: all)
  - [x] 5.1: Implement async `track_progress()` main entry function
  - [x] 5.2: Orchestrate: categorize_stories -> build_snapshot -> estimate_completion
  - [x] 5.3: Return `SprintProgress` with full progress outcome
  - [x] 5.4: Make progress tracking configurable via `ProgressConfig`
  - [x] 5.5: Handle missing data gracefully with sensible defaults
  - [x] 5.6: Log progress queries with structlog

- [x] Task 6: Implement query helpers for CLI/SDK (AC: all)
  - [x] 6.1: Implement `get_progress_summary()` returning human-readable summary string
  - [x] 6.2: Implement `get_progress_for_display()` returning formatted dict for Rich rendering
  - [x] 6.3: Implement `get_stories_by_status()` helper for filtering

- [x] Task 7: Integrate with SM node (AC: all)
  - [x] 7.1: Update `types.py` to add `sprint_progress` field to SMOutput
  - [x] 7.2: Update `node.py` to optionally call `track_progress()` when progress is requested
  - [x] 7.3: Export progress functions from SM `__init__.py`

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1: Create `tests/unit/agents/sm/test_progress_types.py`
  - [x] 8.2: Create `tests/unit/agents/sm/test_progress.py`
  - [x] 8.3: Test story status determination for all status types
  - [x] 8.4: Test story categorization (completed, in_progress, remaining, blocked)
  - [x] 8.5: Test progress percentage calculation edge cases (0%, 50%, 100%)
  - [x] 8.6: Test completion estimation with various sample sizes
  - [x] 8.7: Test confidence scoring based on variance
  - [x] 8.8: Test configuration options
  - [x] 8.9: Add integration tests in test_node.py for SM node integration

## Dev Notes

### Architecture Requirements

This story implements:
- **FR16**: System can track sprint progress and completion status
- **FR66**: SM Agent can track burn-down velocity and cycle time metrics
- **NFR-PERF-3**: Real-time status updates <1 second refresh

Per the architecture document and ADR-001/ADR-005/ADR-007:
- State management uses TypedDict internally with Pydantic at boundaries
- LangGraph message passing with typed state transitions
- SM is the control plane for orchestration decisions
- All operations should be async
- Return state updates, never mutate input state
- Use frozen dataclasses for immutable types

**Key Concept**: Sprint progress tracking provides visibility into the current state of sprint execution, enabling developers to understand what has been completed, what is currently being worked on, and how much work remains. It also provides estimates for completion time based on historical data.

### Related FRs

- **FR16**: System can track sprint progress and completion status (PRIMARY)
- **FR66**: SM Agent can track burn-down velocity and cycle time metrics (PRIMARY)
- **FR65**: SM Agent can calculate weighted priority scores for story selection
- **FR9**: SM Agent can plan sprints by prioritizing and sequencing stories
- **FR11**: SM Agent can monitor agent activity and health metrics

### Existing Infrastructure to Use

**Planning Types** (`agents/sm/planning_types.py` - Story 10.3):

```python
# These types already exist for sprint/story tracking:
@dataclass(frozen=True)
class SprintStory:
    """A story prepared for sprint planning."""
    story_id: str
    title: str
    priority_score: float = 0.0
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    estimated_points: int = 1
    # ... more fields

@dataclass(frozen=True)
class SprintPlan:
    """Complete sprint plan output."""
    sprint_id: str
    stories: tuple[SprintStory, ...]
    total_points: int
    capacity_used: float
    planning_rationale: str
    created_at: str
```

**Health Types** (`agents/sm/health_types.py` - Story 10.5):

```python
# Can leverage health metrics for timing information:
@dataclass(frozen=True)
class HealthMetrics:
    """Aggregated health metrics for the sprint."""
    total_cycle_time_ms: float
    average_cycle_time_ms: float
    total_idle_time_ms: float
    average_idle_time_ms: float
    churn_count: int
    churn_rate: float
```

**State Module** (`orchestrator/state.py` - Already Implemented):

```python
class YoloState(TypedDict):
    """Main state for YOLO Developer orchestration."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
```

**SM Types** (`agents/sm/types.py` - Story 10.2):

```python
@dataclass(frozen=True)
class SMOutput:
    """Complete output from SM agent processing."""
    routing_decision: RoutingDecision
    routing_rationale: str
    # ... existing fields ...
    sprint_plan: dict[str, Any] | None = None
    delegation_result: dict[str, Any] | None = None
    health_status: dict[str, Any] | None = None
    # ADD: sprint_progress: dict[str, Any] | None = None
```

### Progress Data Model

Per existing patterns and new requirements:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StoryStatus = Literal["backlog", "in_progress", "completed", "blocked", "failed"]
"""Status of a story in the sprint.

Values:
    backlog: Story not yet started
    in_progress: Story currently being worked on
    completed: Story finished successfully
    blocked: Story cannot proceed (gate blocked, dependency issue)
    failed: Story failed and cannot be recovered
"""

# Constants
DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD: float = 0.7
"""Minimum confidence level for completion estimates to be considered reliable."""

VALID_STORY_STATUSES: frozenset[str] = frozenset({
    "backlog", "in_progress", "completed", "blocked", "failed"
})
"""Set of valid story status values."""

@dataclass(frozen=True)
class StoryProgress:
    """Progress information for a single story.

    Tracks the current status and timing of a story within the sprint.
    """
    story_id: str
    title: str
    status: StoryStatus
    started_at: str | None = None
    completed_at: str | None = None
    agent_history: tuple[str, ...] = field(default_factory=tuple)
    duration_ms: float | None = None
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "agent_history": list(self.agent_history),
            "duration_ms": self.duration_ms,
            "blocked_reason": self.blocked_reason,
        }

@dataclass(frozen=True)
class SprintProgressSnapshot:
    """Point-in-time snapshot of sprint progress.

    Provides counts and identifiers for current sprint state.
    """
    sprint_id: str
    total_stories: int
    stories_completed: int
    stories_in_progress: int
    stories_remaining: int
    stories_blocked: int
    current_story: str | None
    current_agent: str | None
    progress_percentage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "total_stories": self.total_stories,
            "stories_completed": self.stories_completed,
            "stories_in_progress": self.stories_in_progress,
            "stories_remaining": self.stories_remaining,
            "stories_blocked": self.stories_blocked,
            "current_story": self.current_story,
            "current_agent": self.current_agent,
            "progress_percentage": self.progress_percentage,
        }

@dataclass(frozen=True)
class CompletionEstimate:
    """Estimated completion information for the sprint.

    Based on historical data from completed stories.
    """
    estimated_completion_time: str | None  # ISO timestamp
    estimated_remaining_ms: float | None
    confidence: float  # 0.0-1.0
    factors: tuple[str, ...]  # Factors that influenced the estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_completion_time": self.estimated_completion_time,
            "estimated_remaining_ms": self.estimated_remaining_ms,
            "confidence": self.confidence,
            "factors": list(self.factors),
        }

@dataclass(frozen=True)
class SprintProgress:
    """Complete sprint progress information.

    Combines snapshot, detailed story lists, and completion estimates.
    """
    snapshot: SprintProgressSnapshot
    completed_stories: tuple[StoryProgress, ...]
    in_progress_stories: tuple[StoryProgress, ...]
    remaining_stories: tuple[StoryProgress, ...]
    blocked_stories: tuple[StoryProgress, ...]
    completion_estimate: CompletionEstimate | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "completed_stories": [s.to_dict() for s in self.completed_stories],
            "in_progress_stories": [s.to_dict() for s in self.in_progress_stories],
            "remaining_stories": [s.to_dict() for s in self.remaining_stories],
            "blocked_stories": [s.to_dict() for s in self.blocked_stories],
            "completion_estimate": self.completion_estimate.to_dict() if self.completion_estimate else None,
            "created_at": self.created_at,
        }

@dataclass(frozen=True)
class ProgressConfig:
    """Configuration for progress tracking.

    Controls what information is included in progress reports.
    """
    include_estimates: bool = True
    estimate_confidence_threshold: float = DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD
    track_agent_history: bool = True
    include_blocked_details: bool = True
```

### Main Progress Tracking Function

```python
import time
import structlog

from yolo_developer.agents.sm.progress_types import (
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
)
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

async def track_progress(
    state: YoloState,
    sprint_plan: dict[str, Any] | None = None,
    config: ProgressConfig | None = None,
) -> SprintProgress:
    """Track sprint progress and provide completion estimates (FR16, FR66).

    Analyzes the current sprint state to provide visibility into:
    - Completed, in-progress, remaining, and blocked stories
    - Current story and agent executing
    - Estimated completion time based on historical performance

    Args:
        state: Current orchestration state
        sprint_plan: Sprint plan containing stories (from Story 10.3)
        config: Progress tracking configuration

    Returns:
        SprintProgress with full progress information
    """
    config = config or ProgressConfig()

    logger.info("progress_tracking_started")

    # Step 1: Categorize stories by status
    completed = _get_completed_stories(state, sprint_plan)
    in_progress = _get_in_progress_stories(state, sprint_plan)
    remaining = _get_remaining_stories(state, sprint_plan)
    blocked = _get_blocked_stories(state, sprint_plan)

    # Step 2: Build progress snapshot
    snapshot = _build_progress_snapshot(
        sprint_id=sprint_plan.get("sprint_id", "unknown") if sprint_plan else "unknown",
        completed=completed,
        in_progress=in_progress,
        remaining=remaining,
        blocked=blocked,
        current_agent=state.get("current_agent"),
    )

    # Step 3: Calculate completion estimate if configured
    completion_estimate = None
    if config.include_estimates and completed:
        completion_estimate = _build_completion_estimate(
            completed=completed,
            remaining_count=len(remaining) + len(in_progress),
            confidence_threshold=config.estimate_confidence_threshold,
        )

    logger.info(
        "progress_tracking_complete",
        completed_count=len(completed),
        in_progress_count=len(in_progress),
        remaining_count=len(remaining),
        blocked_count=len(blocked),
        progress_percentage=snapshot.progress_percentage,
    )

    return SprintProgress(
        snapshot=snapshot,
        completed_stories=tuple(completed),
        in_progress_stories=tuple(in_progress),
        remaining_stories=tuple(remaining),
        blocked_stories=tuple(blocked),
        completion_estimate=completion_estimate,
    )
```

### Estimation Logic

```python
from datetime import datetime, timezone, timedelta
from statistics import mean, stdev

def _calculate_average_story_duration(completed: list[StoryProgress]) -> float | None:
    """Calculate average duration of completed stories in milliseconds."""
    durations = [s.duration_ms for s in completed if s.duration_ms is not None]
    if not durations:
        return None
    return mean(durations)

def _calculate_estimation_confidence(completed: list[StoryProgress]) -> float:
    """Calculate confidence based on sample size and variance.

    Confidence factors:
    - Sample size: More completed stories = higher confidence
    - Variance: Lower variance in durations = higher confidence
    """
    durations = [s.duration_ms for s in completed if s.duration_ms is not None]

    if len(durations) < 2:
        return 0.3  # Low confidence with minimal data

    # Factor 1: Sample size (max contribution 0.5)
    sample_factor = min(len(durations) / 10, 0.5)

    # Factor 2: Low variance (max contribution 0.5)
    avg = mean(durations)
    if avg > 0:
        variance_ratio = stdev(durations) / avg
        # Lower variance = higher confidence
        variance_factor = max(0, 0.5 - variance_ratio * 0.25)
    else:
        variance_factor = 0.25

    return min(sample_factor + variance_factor, 1.0)

def _build_completion_estimate(
    completed: list[StoryProgress],
    remaining_count: int,
    confidence_threshold: float,
) -> CompletionEstimate:
    """Build completion estimate from historical data."""
    avg_duration = _calculate_average_story_duration(completed)
    confidence = _calculate_estimation_confidence(completed)

    factors: list[str] = []

    if avg_duration is None or remaining_count == 0:
        return CompletionEstimate(
            estimated_completion_time=None,
            estimated_remaining_ms=None,
            confidence=0.0,
            factors=("insufficient_data",),
        )

    estimated_remaining_ms = avg_duration * remaining_count
    factors.append(f"based_on_{len(completed)}_completed_stories")
    factors.append(f"avg_duration_{avg_duration:.0f}ms")

    if confidence < confidence_threshold:
        factors.append("low_confidence_estimate")

    estimated_completion_time = (
        datetime.now(timezone.utc) + timedelta(milliseconds=estimated_remaining_ms)
    ).isoformat()

    return CompletionEstimate(
        estimated_completion_time=estimated_completion_time,
        estimated_remaining_ms=estimated_remaining_ms,
        confidence=confidence,
        factors=tuple(factors),
    )
```

### Integration with SM Node

Update sm_node() to optionally provide progress tracking:

```python
# In node.py - add progress tracking capability

from yolo_developer.agents.sm.progress import track_progress, SprintProgress

async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node with progress tracking (FR16)."""

    # ... existing analysis and routing logic ...

    # Step X: Track progress if sprint plan exists
    sprint_progress: SprintProgress | None = None
    if state.get("sprint_plan"):
        try:
            sprint_progress = await track_progress(
                state=state,
                sprint_plan=state.get("sprint_plan"),
            )
        except Exception as e:
            logger.warning("progress_tracking_failed", error=str(e))

    # ... rest of output creation ...

    # Include progress in output
    output = SMOutput(
        # ... existing fields ...
        sprint_progress=sprint_progress.to_dict() if sprint_progress else None,
    )
```

### Testing Strategy

**Unit Tests:**
- Test each progress type (type definitions and serialization)
- Test story status determination for all status types
- Test categorization of stories by status
- Test progress percentage calculation (0%, 50%, 100%, edge cases)
- Test average duration calculation with various inputs
- Test confidence calculation based on sample size and variance
- Test completion estimation with various scenarios
- Test configuration options

**Integration Tests:**
- Test full progress tracking flow with realistic state
- Test SM node integration with progress tracking
- Test progress updates as stories complete
- Test handling of blocked stories

### Previous Story Intelligence

From **Story 10.8** (Agent Handoff Management):
- Used frozen dataclasses with `to_dict()` serialization
- Created separate types module (`handoff_types.py`) for clarity
- Exported all new types and functions from `__init__.py`
- Used structlog for consistent logging format
- All functions are async
- Graceful degradation on failure (never block main workflow)

From **Story 10.5** (Health Monitoring):
- Pattern for metrics tracking and aggregation
- Uses HealthMetrics for timing information we can leverage
- Demonstrates calculation of averages and rates

From **Story 10.3** (Sprint Planning):
- SprintPlan and SprintStory types already exist
- Can reuse story information from sprint plan

**Key Pattern to Follow:**
```python
# New module structure
src/yolo_developer/agents/sm/
├── progress.py           # Main progress tracking logic (NEW)
├── progress_types.py     # Types only (NEW)
├── node.py               # Updated with progress tracking
├── types.py              # Add sprint_progress to SMOutput
└── __init__.py           # Export new types and functions
```

### Git Intelligence

Recent commits show consistent patterns:
- Latest: Story 10.8 agent handoff management with code review fixes
- `35752b6`: Story 10.6 circular logic detection with code review fixes
- `f16eff2`: Story 10.5 health monitoring with code review fixes
- `7764479`: Story 10.4 task delegation with code review fixes

Commit message pattern: `feat: Implement <description> with code review fixes (Story X.Y)`

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/agents/sm/progress.py` - Main progress tracking module (NEW)
- `src/yolo_developer/agents/sm/progress_types.py` - Type definitions (NEW)
- `tests/unit/agents/sm/test_progress.py` - Progress tests (NEW)
- `tests/unit/agents/sm/test_progress_types.py` - Types tests (NEW)

**Files to modify:**
- `src/yolo_developer/agents/sm/__init__.py` - Export progress functions
- `src/yolo_developer/agents/sm/types.py` - Add `sprint_progress` field to SMOutput
- `src/yolo_developer/agents/sm/node.py` - Integrate progress tracking

### Implementation Patterns

Per architecture document:

1. **Async-first**: `track_progress()` must be async
2. **State updates via dict**: Return dict updates, don't mutate state
3. **Structured logging**: Use structlog with key-value format
4. **Type annotations**: Full type hints on all functions
5. **Immutable outputs**: Use frozen dataclasses for types
6. **snake_case**: All state dictionary keys use snake_case
7. **Graceful degradation**: If progress tracking fails, don't block workflow
8. **Performance**: Target real-time updates <1s per NFR-PERF-3

```python
# CORRECT pattern for progress module
from __future__ import annotations

import structlog

from yolo_developer.agents.sm.progress_types import (
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
)
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

async def track_progress(
    state: YoloState,
    sprint_plan: dict[str, Any] | None = None,
    config: ProgressConfig | None = None,
) -> SprintProgress:
    """Track sprint progress and provide completion estimates (FR16, FR66).

    Args:
        state: Current orchestration state
        sprint_plan: Sprint plan containing stories
        config: Progress tracking configuration

    Returns:
        SprintProgress with full progress information
    """
    logger.info(
        "progress_tracking_started",
        sprint_id=sprint_plan.get("sprint_id") if sprint_plan else None,
    )

    # ... implementation ...

    logger.info(
        "progress_tracking_complete",
        completed_count=len(completed),
        progress_percentage=snapshot.progress_percentage,
    )

    return result
```

### Dependencies

**Internal dependencies:**
- `yolo_developer.agents.sm.types` - SMOutput (to be modified)
- `yolo_developer.agents.sm.node` - sm_node function (to be modified)
- `yolo_developer.agents.sm.planning_types` - SprintPlan, SprintStory
- `yolo_developer.agents.sm.health_types` - HealthMetrics (for timing reference)
- `yolo_developer.orchestrator.state` - YoloState
- `structlog` - logging
- `statistics` - mean, stdev for estimation

**No new external dependencies needed.**

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.9]
- [Source: _bmad-output/planning-artifacts/epics.md#FR16]
- [Source: _bmad-output/planning-artifacts/epics.md#FR66]
- [Source: src/yolo_developer/agents/sm/planning_types.py - SprintPlan, SprintStory]
- [Source: src/yolo_developer/agents/sm/health_types.py - HealthMetrics pattern]
- [Source: src/yolo_developer/agents/sm/types.py - SMOutput]
- [Source: src/yolo_developer/orchestrator/state.py - YoloState]
- [Source: _bmad-output/implementation-artifacts/10-8-agent-handoff-management.md - pattern reference]
- [Source: _bmad-output/implementation-artifacts/10-5-health-monitoring.md - metrics pattern]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None

### Completion Notes List

- All 8 tasks completed successfully
- 607 SM agent tests pass (including 10 new sprint progress tracking integration tests)
- 75 progress-specific tests pass (47 in test_progress.py, 28 in test_progress_types.py)
- mypy strict mode passes (17 source files in SM module)
- ruff check passes on all modified files
- Implementation follows all architectural patterns (ADR-001, ADR-005, ADR-007)
- All acceptance criteria met:
  - AC #1: Completed stories are listed via `completed_stories` tuple
  - AC #2: Current story and agent shown via `snapshot.current_story` and `snapshot.current_agent`
  - AC #3: Remaining work displayed via `remaining_stories` and `stories_remaining` count
  - AC #4: Estimated completion provided via `completion_estimate` with confidence scoring

### File List

**New Files:**
- `src/yolo_developer/agents/sm/progress_types.py` - Type definitions (StoryStatus, StoryProgress, SprintProgressSnapshot, CompletionEstimate, SprintProgress, ProgressConfig)
- `src/yolo_developer/agents/sm/progress.py` - Main progress tracking logic (track_progress, query helpers)
- `tests/unit/agents/sm/test_progress_types.py` - 28 type tests
- `tests/unit/agents/sm/test_progress.py` - 47 progress function tests

**Modified Files:**
- `src/yolo_developer/agents/sm/types.py` - Added `sprint_progress` field to SMOutput
- `src/yolo_developer/agents/sm/node.py` - Integrated progress tracking (Step 6e)
- `src/yolo_developer/agents/sm/__init__.py` - Exported progress functions and types
- `tests/unit/agents/sm/test_node.py` - Added 10 integration tests (TestSMNodeSprintProgressTracking)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

**Ruff Auto-Fixed Files (unused import cleanup):**
- `tests/unit/agents/sm/test_circular_detection.py`
- `tests/unit/agents/sm/test_circular_detection_types.py`
- `tests/unit/agents/sm/test_conflict_types.py`
- `tests/unit/agents/sm/test_delegation.py`
- `tests/unit/agents/sm/test_delegation_types.py`
- `tests/unit/agents/sm/test_handoff_types.py`
- `tests/unit/agents/sm/test_health.py`
- `tests/unit/agents/sm/test_health_types.py`

### Code Review Fixes Applied

Code review identified 6 issues (1 HIGH, 5 MEDIUM, 3 LOW). Fixed:
- **ISSUE #1 (HIGH)**: Updated File List to document all ruff auto-fixed test files
- **ISSUE #3 (MEDIUM)**: Refactored `track_progress()` to call `_categorize_stories_by_status()` once for efficiency
- **ISSUE #4 (MEDIUM)**: Removed unnecessary `elif` after `return` in `get_stories_by_status()`
- **ISSUE #5 (MEDIUM)**: Added test for unknown/invalid status handling (`test_unknown_status_treated_as_backlog`)
- **ISSUE #6 (MEDIUM)**: Added clarifying comment about `duration_ms` field behavior

Not fixed (by design):
- **ISSUE #2 (MEDIUM)**: `_build_progress_snapshot()` has 6 parameters - kept as-is since it's an internal helper and parameters are logically distinct

Test count increased from 607 to 608 with added edge case test.
