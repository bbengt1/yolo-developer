# Story 11.1: Decision Logging

Status: done

## Story

As a developer,
I want every agent decision logged with rationale,
So that I can understand why the system did what it did.

## Acceptance Criteria

1. **Given** an agent makes a decision
   **When** the decision is logged
   **Then** the decision content is captured

2. **Given** a decision is being logged
   **When** the log entry is created
   **Then** the rationale is recorded

3. **Given** a decision is being logged
   **When** the log entry is created
   **Then** the agent identity is included

4. **Given** a decision is being logged
   **When** the log entry is created
   **Then** timestamp and context are stored

5. **Given** decisions are logged
   **When** the audit trail is queried
   **Then** decisions are retrievable in chronological order

## Tasks / Subtasks

- [x] Task 1: Create type definitions (AC: #1, #2, #3, #4)
  - [x] 1.1 Create `src/yolo_developer/audit/types.py` with:
    - `DecisionType` Literal type: "requirement_analysis", "story_creation", "architecture_choice", "implementation_choice", "test_strategy", "orchestration", "quality_gate", "escalation"
    - `DecisionSeverity` Literal type: "info", "warning", "critical"
    - `AgentIdentity` frozen dataclass: agent_name, agent_type, session_id
    - `DecisionContext` frozen dataclass: sprint_id, story_id, artifact_id, parent_decision_id (optional)
    - `Decision` frozen dataclass: id, decision_type, content, rationale, agent, context, timestamp, metadata
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `VALID_DECISION_TYPES`, `VALID_DECISION_SEVERITIES`

- [x] Task 2: Create decision store protocol (AC: #1, #5)
  - [x] 2.1 Create `src/yolo_developer/audit/store.py`
  - [x] 2.2 Define `DecisionStore` Protocol with methods:
    - `async def log_decision(decision: Decision) -> str` - returns decision_id
    - `async def get_decision(decision_id: str) -> Decision | None`
    - `async def get_decisions(filters: DecisionFilters | None = None) -> list[Decision]`
    - `async def get_decision_count() -> int`
  - [x] 2.3 Create `DecisionFilters` frozen dataclass: agent_name, decision_type, start_time, end_time, sprint_id, story_id

- [x] Task 3: Implement in-memory decision store (AC: #1, #5)
  - [x] 3.1 Create `src/yolo_developer/audit/memory_store.py`
  - [x] 3.2 Implement `InMemoryDecisionStore` class implementing `DecisionStore` protocol
  - [x] 3.3 Use thread-safe storage with `threading.Lock` for concurrent access
  - [x] 3.4 Implement chronological ordering by timestamp
  - [x] 3.5 Implement filter matching for all `DecisionFilters` fields

- [x] Task 4: Implement decision logger (AC: #1, #2, #3, #4)
  - [x] 4.1 Create `src/yolo_developer/audit/logger.py`
  - [x] 4.2 Implement `DecisionLogger` class:
    - Constructor takes `DecisionStore` instance
    - `async def log(agent_name: str, agent_type: str, decision_type: DecisionType, content: str, rationale: str, context: DecisionContext | None = None, metadata: dict | None = None) -> str`
    - Auto-generates decision ID (UUID)
    - Auto-captures timestamp (UTC ISO format)
    - Auto-captures session_id from context or generates new
  - [x] 4.3 Add structured logging with structlog for each decision logged
  - [x] 4.4 Implement `get_logger(store: DecisionStore | None = None) -> DecisionLogger` factory function

- [x] Task 5: Create module initialization (AC: all)
  - [x] 5.1 Update `src/yolo_developer/audit/__init__.py`
  - [x] 5.2 Export all public types: Decision, DecisionType, DecisionSeverity, AgentIdentity, DecisionContext, DecisionFilters
  - [x] 5.3 Export store protocol and implementations: DecisionStore, InMemoryDecisionStore
  - [x] 5.4 Export logger: DecisionLogger, get_logger
  - [x] 5.5 Add module docstring documenting FR81 implementation

- [x] Task 6: Write comprehensive tests (AC: all)
  - [x] 6.1 Create `tests/unit/audit/test_types.py`:
    - Test type validation (valid/invalid values)
    - Test `to_dict()` produces JSON-serializable output
    - Test frozen dataclass immutability
  - [x] 6.2 Create `tests/unit/audit/test_memory_store.py`:
    - Test `log_decision` returns valid ID
    - Test `get_decision` retrieves correct decision
    - Test `get_decisions` returns chronological order
    - Test `get_decisions` with filters
    - Test concurrent access safety
  - [x] 6.3 Create `tests/unit/audit/test_logger.py`:
    - Test `log` creates valid Decision
    - Test auto-generated fields (id, timestamp, session_id)
    - Test `get_logger` factory function
    - Test structlog output with caplog

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `types.py` (frozen dataclasses per ADR-001)
- **Protocol Pattern**: Use Protocol for DecisionStore to allow future implementations (file-based, database)
- **Logging**: Use `structlog.get_logger(__name__)` pattern per architecture
- **State**: Frozen dataclasses for internal types, TypedDict not needed for audit (not graph state)
- **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Design Decisions

1. **In-Memory First**: Start with `InMemoryDecisionStore` for simplicity and testing. Story 11.4 (Audit Export) will add persistence.

2. **Protocol Pattern**: `DecisionStore` as Protocol enables:
   - Easy mocking in tests
   - Future implementations (JSON file, SQLite, etc.)
   - Dependency injection

3. **Decision ID**: Use UUID v4 for globally unique, collision-free IDs.

4. **Timestamps**: Use ISO 8601 format in UTC (e.g., "2026-01-17T12:34:56.789Z").

5. **Thread Safety**: `InMemoryDecisionStore` uses `threading.Lock` for concurrent agent access.

### Project Structure Notes

Module location: `src/yolo_developer/audit/`
```
audit/
├── __init__.py          # Module exports (update existing stub)
├── types.py             # Type definitions (NEW)
├── store.py             # DecisionStore protocol (NEW)
├── memory_store.py      # InMemoryDecisionStore implementation (NEW)
└── logger.py            # DecisionLogger class (NEW)
```

Test location: `tests/unit/audit/`
```
tests/unit/audit/
├── __init__.py          # Package init (NEW)
├── test_types.py        # Type tests (NEW)
├── test_memory_store.py # Store tests (NEW)
└── test_logger.py       # Logger tests (NEW)
```

### Integration Points

This is the foundation for Epic 11. Future stories will build on this:
- Story 11.2 (Requirement Traceability): Adds traceability links to Decision
- Story 11.3 (Human-Readable View): Formats Decision for display
- Story 11.4 (Audit Export): Adds file/JSON persistence
- Story 11.5 (Cross-Agent Correlation): Adds decision chain queries
- Story 11.6 (Token Tracking): Adds cost metadata to Decision
- Story 11.7 (Audit Filtering): Extends DecisionFilters

### Example Usage

```python
from yolo_developer.audit import (
    DecisionLogger,
    DecisionContext,
    InMemoryDecisionStore,
    get_logger,
)

# Create store and logger
store = InMemoryDecisionStore()
logger = get_logger(store)

# Log a decision
decision_id = await logger.log(
    agent_name="analyst",
    agent_type="analyst",
    decision_type="requirement_analysis",
    content="Crystallized requirement: User authentication via OAuth2",
    rationale="OAuth2 chosen for industry-standard security and third-party integration support",
    context=DecisionContext(
        sprint_id="sprint-1",
        story_id="1-2-user-authentication",
        artifact_id="req-001",
    ),
    metadata={"source_seed": "auth-requirements.md"},
)

# Retrieve decisions
decisions = await store.get_decisions()
for d in decisions:
    print(f"[{d.timestamp}] {d.agent.agent_name}: {d.content}")
```

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR81: System can log all agent decisions with rationale
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/planning-artifacts/architecture.md] structlog for structured logging
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11.1] Story definition and acceptance criteria

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- Implemented complete audit trail and decision logging module for FR81
- Used frozen dataclasses per ADR-001 for all internal types
- Protocol pattern for DecisionStore enables future implementations (file, SQLite, etc.)
- Thread-safe InMemoryDecisionStore with threading.Lock for concurrent agent access
- DecisionLogger with auto-generated UUID v4 IDs and ISO 8601 UTC timestamps
- Structured logging with structlog for decision events
- 70 comprehensive unit tests covering all types, store, logger, and module exports
- All tests pass (100% coverage), ruff clean, mypy clean

**Code Review Fixes Applied:**
- Added error handling in DecisionLogger.log() per ADR-007 (log errors, don't block callers)
- Fixed metadata type annotation from dict[str, object] to dict[str, Any] for consistency
- Added 5 validation edge case tests achieving 100% coverage (empty agent_type, empty session_id, empty id, empty timestamp, invalid severity)
- Added test for store failure handling (test_log_handles_store_failure_gracefully)

### File List

**Source Files:**
- src/yolo_developer/audit/__init__.py (updated with exports and FR81 docstring)
- src/yolo_developer/audit/types.py (NEW - Decision, AgentIdentity, DecisionContext, etc.)
- src/yolo_developer/audit/store.py (NEW - DecisionStore Protocol, DecisionFilters)
- src/yolo_developer/audit/memory_store.py (NEW - InMemoryDecisionStore)
- src/yolo_developer/audit/logger.py (NEW - DecisionLogger, get_logger)

**Test Files:**
- tests/unit/audit/__init__.py (NEW - package init)
- tests/unit/audit/test_types.py (NEW - 17 tests)
- tests/unit/audit/test_store.py (NEW - 6 tests)
- tests/unit/audit/test_memory_store.py (NEW - 12 tests)
- tests/unit/audit/test_logger.py (NEW - 14 tests)
- tests/unit/audit/test_init.py (NEW - 15 tests)
