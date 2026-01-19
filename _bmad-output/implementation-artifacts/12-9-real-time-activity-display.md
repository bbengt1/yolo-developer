# Story 12.9: Real-Time Activity Display

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to see real-time agent activity during execution,
so that I know what the system is doing.

## Acceptance Criteria

### AC1: Current Agent Activity Shown
**Given** a sprint is running via `yolo run`
**When** I watch the CLI output
**Then** the currently executing agent name is displayed
**And** a description of what the agent is doing is shown
**And** the display updates as activity progresses

### AC2: Progress Updates in Real-Time
**Given** workflow execution is in progress
**When** agents transition or complete tasks
**Then** progress information updates without requiring page refresh
**And** elapsed time is visible
**And** completed steps are tracked

### AC3: Agent Transitions Visible
**Given** one agent completes and hands off to another
**When** the transition occurs
**Then** the transition is clearly indicated in the output
**And** the previous agent's completion is shown
**And** the new agent's start is announced

### AC4: Display Doesn't Overwhelm with Output
**Given** real-time activity display is running
**When** many events occur rapidly
**Then** the output remains readable
**And** verbose details are optional (controlled by --verbose flag)
**And** the display doesn't scroll excessively

## Tasks / Subtasks

- [x] Task 1: Create Activity Display Module (AC: #1, #2, #3, #4)
  - [x] Create `src/yolo_developer/cli/activity.py` module
  - [x] Implement `ActivityDisplay` class using Rich Live context
  - [x] Add support for agent status tracking (name, description, elapsed time)
  - [x] Implement event batching to prevent display overwhelming

- [x] Task 2: Define Activity Event Types (AC: #1, #2, #3)
  - [x] Add `ActivityEvent` dataclass for structured activity data (for future SDK/API use)
  - [x] CLI uses existing `stream_workflow` events (agent name as key)
  - [x] Track agent transitions with timing in ActivityDisplay
  - [x] Define event type literals (start, progress, complete, transition)

- [x] Task 3: Integrate Activity Display with Run Command (AC: #1, #2, #3, #4)
  - [x] Replace basic Progress spinner with ActivityDisplay in run.py
  - [x] Connect `stream_workflow` events to ActivityDisplay
  - [x] Handle verbose mode for detailed event output
  - [x] Ensure JSON output mode disables rich display

- [x] Task 4: Implement Visual Activity Panel (AC: #1, #4)
  - [x] Create Rich Panel with agent status display
  - [x] Show current agent with activity indicator (🔄 emoji)
  - [x] Display agent description/task
  - [x] Show elapsed time (total workflow time)

- [x] Task 5: Implement Agent Transition Display (AC: #3)
  - [x] Show clear visual separator on agent transitions (horizontal line)
  - [x] Display "Agent X → Agent Y" transition message with color styling
  - [x] Include transition reason from handoff context
  - [x] Transition updates bypass throttling for immediate visibility

- [x] Task 6: Implement Event Throttling/Batching (AC: #4)
  - [x] Add configurable refresh rate (default: 4 updates/sec)
  - [x] Throttle rapid updates via should_update() check
  - [x] Verbose mode shows total event count and last event description
  - [x] Ensure critical events (transitions) bypass throttle

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Test ActivityDisplay class initialization
  - [x] Test event handling and display updates
  - [x] Test agent transition visualization
  - [x] Test event throttling behavior
  - [x] Test verbose vs normal mode output
  - [x] Test integration with run command
  - [x] Mock Rich components for testing

## Dev Notes

### Existing Implementation Context

**Current run.py Progress Display (Story 12.4):**
```python
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    TimeElapsedColumn(),
    console=console,
    disable=json_output,
) as progress:
    task = progress.add_task(f"Running agent: {current_agent}", total=None)

    async for event in stream_workflow(...):
        progress.update(task, description=f"Running agent: {current_agent}")
```

This basic implementation needs enhancement with:
1. Rich Live context for full panel updates
2. More detailed activity information
3. Agent transition animations
4. Event batching for smooth display

**Orchestrator stream_workflow (Story 10.1):**
```python
async def stream_workflow(initial_state, config, ...) -> AsyncIterator[dict[str, Any]]:
    """Yields events as each node executes."""
    async for event in graph.astream(initial_state, ...):
        logger.debug("workflow_event", event_keys=list(event.keys()))
        yield event
```

The `event` dict contains the agent name as key with state updates as value.

### Rich Components to Use

**Rich Live Display:**
```python
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

class ActivityDisplay:
    def __init__(self, console: Console):
        self.console = console
        self.live = Live(console=console, refresh_per_second=4)
        self.current_agent = ""
        self.elapsed = 0.0
        self.transitions: list[str] = []

    def update(self, agent: str, description: str, elapsed: float):
        self.current_agent = agent
        self.elapsed = elapsed
        # Render updated panel

    def render(self) -> Panel:
        # Create activity panel with current state
```

**Rich Layout for Structured Display:**
```python
layout = Layout()
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="activity", size=8),
    Layout(name="transitions", size=6),
)
```

### Event Types and Structure

**Proposed ActivityEvent:**
```python
@dataclass
class ActivityEvent:
    event_type: Literal["start", "progress", "complete", "transition"]
    agent: str
    description: str
    timestamp: float
    details: dict[str, Any] | None = None
    previous_agent: str | None = None  # For transitions
```

### Display Patterns

**Normal Mode (default):**
```
┌─────────────────────────────────────────┐
│ 🔄 Running: analyst                     │
│                                         │
│ Analyzing seed requirements...          │
│ Elapsed: 00:12                          │
│                                         │
│ ─────────────────────────────────────── │
│ analyst → pm (requirements crystallized)│
└─────────────────────────────────────────┘
```

**Verbose Mode (--verbose):**
```
┌─────────────────────────────────────────┐
│ 🔄 Running: pm                          │
│                                         │
│ Transforming requirements to stories... │
│ Elapsed: 00:34                          │
│                                         │
│ Events: 42                              │
│ Last: Added story "user-authentication" │
│                                         │
│ ─────────────────────────────────────── │
│ Transitions:                            │
│   00:00 → analyst (started)             │
│   00:12 → pm (requirements ready)       │
└─────────────────────────────────────────┘
```

### Previous Story Learnings (Story 12.8)

1. Use `print()` instead of `console.print()` for JSON output to avoid Rich line-wrapping issues
2. Run `ruff check` and `mypy` before committing
3. Follow existing CLI patterns (structlog logging, typer.Exit for errors)
4. Test both normal and JSON output modes
5. Use constants for magic strings (refresh rates, colors)

### Architecture Compliance

Per ADR-009 (Typer + Rich):
- Use Rich components for all formatted output
- Support JSON output mode for automation
- Follow existing display.py patterns

Per ADR-005 (LangGraph Orchestration):
- Activity events should not interfere with workflow execution
- Display is purely observational, not blocking

### Dependencies

**Required imports:**
```python
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.console import Console, Group
```

**Existing dependencies (already in project):**
- Rich (latest) - already used throughout CLI
- structlog - for logging

### Test File Location

Tests: `tests/unit/cli/test_activity_display.py`

Mirror patterns from `test_run_command.py` for async testing.

### Key Implementation Considerations

1. **Thread Safety**: Display updates happen from async event loop, ensure Rich Live handles this correctly
2. **Graceful Interrupt**: Activity display should clean up properly on Ctrl+C
3. **Terminal Compatibility**: Rich Live requires terminal support, fallback for non-TTY
4. **Performance**: Event batching prevents CPU spikes from rapid updates

### References

- [Source: src/yolo_developer/cli/commands/run.py] - Current progress implementation
- [Source: src/yolo_developer/cli/display.py] - Display utilities
- [Source: src/yolo_developer/orchestrator/workflow.py:439-496] - stream_workflow function
- [Related: FR105 - CLI can display real-time agent activity during execution]
- [Related: ADR-009 - Typer + Rich framework selection]
- [Related: Story 12.4 (yolo run command) - Base implementation to enhance]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. **Created activity.py module** - Implemented ActivityDisplay class with Rich Live context support. Features include:
   - ActivityEvent dataclass for structured event data
   - format_elapsed_time helper for time display (MM:SS or HH:MM:SS)
   - Real-time updates with configurable refresh rate (default: 4/sec)
   - Event throttling via should_update() to prevent display overwhelming
   - Agent transition tracking with add_transition()
   - Context manager protocol (__enter__/__exit__) for clean resource management
   - Verbose mode showing event counts and last event details

2. **Integrated with run.py** - Replaced basic Progress spinner with ActivityDisplay:
   - ActivityDisplay used for non-JSON output modes
   - JSON mode continues to process events without display
   - Verbose flag passed to ActivityDisplay
   - Agent transitions detected and recorded automatically
   - Added AGENT_DESCRIPTIONS constant for human-readable agent descriptions

3. **Tests** - 53 tests total:
   - 25 tests for ActivityDisplay class (test_activity_display.py)
   - 28 tests for run command including 3 new integration tests
   - All tests pass with ruff and mypy clean

4. **Following Story 12.8 learnings**:
   - Used print() for JSON output (no Rich formatting)
   - Ran ruff check and mypy before completing
   - Used constants for magic strings (AGENT_DESCRIPTIONS, DEFAULT_REFRESH_RATE)

### File List

**New Files:**
- src/yolo_developer/cli/activity.py - ActivityDisplay class and ActivityEvent dataclass
- tests/unit/cli/test_activity_display.py - New test file with 25 tests

**Modified Files:**
- src/yolo_developer/cli/commands/run.py - Integrated ActivityDisplay, added AGENT_DESCRIPTIONS
- tests/unit/cli/test_run_command.py - Added 3 ActivityDisplay integration tests
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status

### Change Log

- 2026-01-19: Implemented Story 12.9 - Real-Time Activity Display with all 7 tasks complete
