# Story 13.6: Event Emission

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want events emitted for custom integrations,
so that I can build reactive systems.

## Acceptance Criteria

### AC1: Event Type Definition
**Given** event types are needed for integration
**When** I import from `yolo_developer.sdk.types`
**Then** EventType enum defines all event categories (AGENT_START, AGENT_END, WORKFLOW_START, WORKFLOW_END, GATE_PASS, GATE_FAIL, ERROR)
**And** EventData dataclass contains event_type, timestamp, agent, data
**And** types are documented with examples

### AC2: Event Subscription
**Given** a YoloClient instance
**When** I call `client.subscribe(callback, event_types=None)`
**Then** callback is registered for specified event types (or all if None)
**And** subscription returns a subscription_id for later unsubscription
**And** multiple callbacks can subscribe to the same event type
**And** callback signature is `(event: EventData) -> None`

### AC3: Event Emission During Workflow
**Given** registered event subscribers
**When** workflow execution occurs
**Then** WORKFLOW_START fires when `run_async()` begins
**And** AGENT_START fires before each agent executes
**And** AGENT_END fires after each agent completes
**And** GATE_PASS/GATE_FAIL fires after quality gate evaluation
**And** WORKFLOW_END fires when workflow completes
**And** ERROR fires on any exception

### AC4: Event Unsubscription
**Given** registered event subscriptions
**When** I call `client.unsubscribe(subscription_id)`
**Then** the callback is removed from the registry
**And** subsequent events don't fire for that callback
**And** `client.list_subscriptions()` reflects the removal

### AC5: Async Callback Support
**Given** async callback functions
**When** events are emitted
**Then** async callbacks are awaited properly
**And** sync callbacks are executed synchronously
**And** callback detection uses `asyncio.iscoroutinefunction()`

### AC6: Graceful Error Handling
**Given** callbacks that may raise exceptions
**When** a callback raises an error
**Then** the error is logged with full context
**And** other callbacks still receive the event
**And** workflow execution continues (callbacks don't block)
**And** EventCallbackError is available for inspection

## Tasks / Subtasks

- [x] Task 1: Design Event Types (AC: #1)
  - [x] Subtask 1.1: Create EventType enum in sdk/types.py
  - [x] Subtask 1.2: Create EventData frozen dataclass with event_type, timestamp, agent, data
  - [x] Subtask 1.3: Create EventCallbackError in sdk/exceptions.py
  - [x] Subtask 1.4: Create EventCallback Protocol for typed callbacks

- [x] Task 2: Implement Event Registry (AC: #2, #4)
  - [x] Subtask 2.1: Add _subscriptions dict to YoloClient to store callbacks
  - [x] Subtask 2.2: Implement subscribe(callback, event_types) returning subscription_id
  - [x] Subtask 2.3: Implement unsubscribe(subscription_id) method
  - [x] Subtask 2.4: Implement list_subscriptions() returning list of subscriptions

- [x] Task 3: Implement Event Emission (AC: #3, #5, #6)
  - [x] Subtask 3.1: Create _emit_event() internal method with sync/async callback handling
  - [x] Subtask 3.2: Add WORKFLOW_START/END emission to run_async()
  - [x] Subtask 3.3: Add AGENT_START/END emission around hook integration points
  - [x] Subtask 3.4: Add GATE_PASS/FAIL emission (placeholder for gate integration)
  - [x] Subtask 3.5: Add ERROR emission in exception handlers
  - [x] Subtask 3.6: Implement error handling that doesn't block other callbacks

- [x] Task 4: Write Unit Tests (AC: all)
  - [x] Subtask 4.1: Test event type definitions and EventData structure
  - [x] Subtask 4.2: Test subscription and unsubscription
  - [x] Subtask 4.3: Test event emission during workflow execution
  - [x] Subtask 4.4: Test async callback support
  - [x] Subtask 4.5: Test error handling in callbacks
  - [x] Subtask 4.6: Test filtering by event types

- [x] Task 5: Update Documentation (AC: all)
  - [x] Subtask 5.1: Update client.py docstrings
  - [x] Subtask 5.2: Add usage examples for common event patterns
  - [x] Subtask 5.3: Export new types from sdk/__init__.py

## Dev Notes

### Architecture Patterns

Per Stories 13.1-13.5 implementation and architecture.md:

1. **SDK Layer Position**: SDK sits between external consumers and the orchestrator
2. **Async/Sync Pattern**: Use `asyncio.iscoroutinefunction()` to detect async callbacks
3. **Result Types**: Use `@dataclass(frozen=True)` for immutable event data with timestamp
4. **Exception Chaining**: Always use `raise ... from e` pattern
5. **Hook Integration**: Events should fire alongside hooks in run_async()

### Event Types

```python
from enum import Enum, auto

class EventType(Enum):
    """Types of events emitted by YOLO Developer."""
    WORKFLOW_START = auto()   # Workflow execution begins
    WORKFLOW_END = auto()     # Workflow execution completes
    AGENT_START = auto()      # Agent begins execution
    AGENT_END = auto()        # Agent completes execution
    GATE_PASS = auto()        # Quality gate passes
    GATE_FAIL = auto()        # Quality gate fails
    ERROR = auto()            # Error occurred
```

### Proposed API Design

```python
from yolo_developer import YoloClient
from yolo_developer.sdk.types import EventData, EventType

client = YoloClient()

# Sync callback
def on_agent_start(event: EventData) -> None:
    print(f"Agent {event.agent} starting at {event.timestamp}")

# Async callback
async def on_workflow_end(event: EventData) -> None:
    await save_metrics(event.data)

# Subscribe to specific events
sub_id1 = client.subscribe(on_agent_start, event_types=[EventType.AGENT_START])

# Subscribe to all events
sub_id2 = client.subscribe(on_workflow_end)

# List subscriptions
subs = client.list_subscriptions()

# Unsubscribe
client.unsubscribe(sub_id1)

# Run workflow - events fire automatically
await client.run_async(seed_content="Build something")
```

### EventData Structure

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class EventData:
    """Immutable event data emitted during workflow execution.

    Attributes:
        event_type: The type of event that occurred.
        timestamp: When the event occurred (UTC).
        agent: The agent associated with the event, or None for workflow events.
        data: Additional event-specific data.
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
```

### Callback Protocol

```python
from typing import Protocol, Union

class EventCallback(Protocol):
    """Protocol for event callbacks (sync or async)."""
    def __call__(self, event: EventData) -> Union[None, Coroutine[Any, Any, None]]:
        ...
```

### Integration with Hooks (Story 13.5)

Events should fire at the same integration points as hooks in run_async():

```python
# In run_async(), alongside hook execution:
await self._emit_event(EventType.WORKFLOW_START, data={"seed_id": seed_id})

# Before pre-hooks fire:
await self._emit_event(EventType.AGENT_START, agent=entry_agent)

# Pre-hooks execute...
# Workflow executes...

# After post-hooks fire:
await self._emit_event(EventType.AGENT_END, agent=last_agent, data=workflow_output)

# At completion:
await self._emit_event(EventType.WORKFLOW_END, data={"run_id": run_id})
```

### Error Handling Pattern

```python
async def _emit_event(self, event_type: EventType, ...) -> None:
    event = EventData(event_type=event_type, ...)

    for subscription in self._get_matching_subscriptions(event_type):
        try:
            if asyncio.iscoroutinefunction(subscription.callback):
                await subscription.callback(event)
            else:
                subscription.callback(event)
        except Exception as e:
            logger.error(
                "event_callback_failed",
                subscription_id=subscription.id,
                event_type=event_type.name,
                error=str(e),
            )
            # Continue to next callback - don't block workflow
```

### Key Files to Touch

**Modify:**
- `src/yolo_developer/sdk/client.py` - Add subscription and emission methods
- `src/yolo_developer/sdk/types.py` - Add EventType, EventData, EventCallback
- `src/yolo_developer/sdk/__init__.py` - Export new types
- `src/yolo_developer/sdk/exceptions.py` - Add EventCallbackError
- `tests/unit/sdk/test_client.py` - Add event tests

### Previous Story Learnings (Stories 13.1-13.5)

1. Run `ruff check` and `mypy` before committing
2. Use `from __future__ import annotations` in all files
3. Use timezone-aware datetime: `datetime.now(timezone.utc)` per ruff DTZ005 rule
4. Use `_run_sync()` helper instead of deprecated `asyncio.get_event_loop()`
5. Frozen dataclasses for immutable results
6. Exception chaining with `raise ... from e`
7. Test both success and error paths
8. 98 tests currently passing for SDK module

### Testing Standards

Follow patterns from `tests/unit/sdk/test_client.py`:
- Use `pytest` with `pytest-asyncio` for async tests
- Mock orchestrator for unit tests
- Test file naming: `test_<module>.py`
- Test function naming: `test_<behavior>_<scenario>`
- Mark async tests with `@pytest.mark.asyncio`

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#Python SDK] - SDK structure
- [Source: _bmad-output/planning-artifacts/prd.md#FR111] - Event emission requirement
- [Source: _bmad-output/planning-artifacts/epics.md#Story 13.6] - Story definition
- [Source: src/yolo_developer/sdk/client.py] - Current SDK implementation
- [Related: Story 13.5 (Agent Hooks)] - Hook integration pattern
- [Related: Story 13.2 (Programmatic Init/Seed/Run)] - run_async() method

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- Implemented EventType enum with all 7 event categories (WORKFLOW_START, WORKFLOW_END, AGENT_START, AGENT_END, GATE_PASS, GATE_FAIL, ERROR)
- Implemented EventData frozen dataclass with event_type, timestamp, agent, and data fields
- Implemented EventSubscription dataclass for tracking subscriptions
- Implemented EventCallback Protocol supporting both sync and async callbacks
- Added EventCallbackError exception for callback failures
- Added subscribe(), unsubscribe(), and list_subscriptions() methods to YoloClient
- Implemented _emit_event() method with graceful error handling for callbacks
- Integrated event emission into run_async() at key workflow points
- **GATE_PASS emits on seed quality acceptance and workflow completion**
- **GATE_FAIL emits on seed quality rejection**
- All 153 SDK tests pass (134 client tests + 19 exception tests)
- ruff check and mypy pass on all SDK source files
- Exported all new types from sdk/__init__.py
- Updated docstrings with usage examples

### File List

**Modified:**
- src/yolo_developer/sdk/types.py - Added EventType enum, EventData, EventSubscription, EventCallback Protocol
- src/yolo_developer/sdk/exceptions.py - Added EventCallbackError exception
- src/yolo_developer/sdk/client.py - Added event subscription and emission methods, GATE_PASS/GATE_FAIL emissions
- src/yolo_developer/sdk/__init__.py - Exported new event types
- tests/unit/sdk/test_client.py - Added 36 new tests for event emission (including GATE_PASS/GATE_FAIL)
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status
