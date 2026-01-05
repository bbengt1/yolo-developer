# Story 3.1: Create Quality Gate Decorator

Status: done

## Story

As a system architect,
I want a decorator that wraps agent nodes with quality validation,
So that gates are enforced consistently without cluttering agent code.

## Acceptance Criteria

1. **AC1: Basic Decorator Structure**
   - **Given** I have an agent node function
   - **When** I decorate it with @quality_gate("gate_name")
   - **Then** the decorator wraps the node function correctly
   - **And** the original function's metadata is preserved
   - **And** the gate name is captured for identification

2. **AC2: Blocking Gate Behavior**
   - **Given** a node decorated with @quality_gate("gate_name", blocking=True)
   - **When** the gate evaluation fails
   - **Then** the node execution is prevented
   - **And** the state is updated with gate_blocked=True
   - **And** the state includes gate_failure with the reason

3. **AC3: Advisory Gate Behavior**
   - **Given** a node decorated with @quality_gate("gate_name", blocking=False)
   - **When** the gate evaluation fails
   - **Then** a warning is logged
   - **And** the node execution continues
   - **And** the advisory failure is recorded in state

4. **AC4: Gate Result Logging**
   - **Given** any gate evaluation (pass or fail)
   - **When** the gate completes evaluation
   - **Then** the result is recorded in the audit trail
   - **And** the result includes gate_name, passed status, timestamp
   - **And** failure reasons are included when applicable

5. **AC5: State Management**
   - **Given** a decorated node function
   - **When** the gate passes
   - **Then** the original node function receives the state unchanged
   - **And** the node returns an updated state dict
   - **And** gate metadata is appended to state without mutating input

6. **AC6: Async Compatibility**
   - **Given** an async node function
   - **When** decorated with @quality_gate
   - **Then** the decorator handles async properly
   - **And** gate evaluation is awaited correctly
   - **And** the wrapped function remains async

## Tasks / Subtasks

- [x] Task 1: Define Gate Types and Results (AC: 1, 4)
  - [x] Create `src/yolo_developer/gates/types.py` module
  - [x] Define `GateResult` dataclass with passed, gate_name, reason, timestamp
  - [x] Define `GateMode` enum (BLOCKING, ADVISORY)
  - [x] Define `GateContext` dataclass for passing state to evaluators
  - [x] Export from `gates/__init__.py`

- [x] Task 2: Implement Core Decorator (AC: 1, 5, 6)
  - [x] Create `src/yolo_developer/gates/decorator.py` module
  - [x] Implement `quality_gate(gate_name: str, blocking: bool = True)` decorator
  - [x] Use `functools.wraps` to preserve function metadata
  - [x] Handle async node functions correctly (per AC6)
  - [x] Ensure state is passed through correctly on gate pass

- [x] Task 3: Implement Blocking Behavior (AC: 2)
  - [x] Add blocking gate logic to decorator
  - [x] Update state with `gate_blocked=True` on failure
  - [x] Update state with `gate_failure` containing reason
  - [x] Prevent node execution when blocked
  - [x] Return early with updated state containing gate info

- [x] Task 4: Implement Advisory Behavior (AC: 3)
  - [x] Add advisory gate logic to decorator
  - [x] Log warning on advisory failure using structlog
  - [x] Allow node execution to continue
  - [x] Record advisory failure in state (separate from blocking)
  - [x] Add `advisory_warnings` list to state for tracking

- [x] Task 5: Implement Gate Result Logging (AC: 4)
  - [x] Create gate result logging function
  - [x] Log gate_name, passed status, timestamp, reason
  - [x] Integrate with structlog for structured logging
  - [x] Prepare interface for future audit trail integration (Story 11.1)
  - [x] Add `gate_results` list to state for audit

- [x] Task 6: Create Gate Evaluator Protocol (AC: 1, 2, 3)
  - [x] Create `src/yolo_developer/gates/evaluators.py` module
  - [x] Define `GateEvaluator` Protocol with `async def evaluate(context: GateContext) -> GateResult`
  - [x] Create registry for named evaluators (gate_name -> evaluator mapping)
  - [x] Implement `register_evaluator(gate_name: str, evaluator: GateEvaluator)` function
  - [x] Implement `get_evaluator(gate_name: str) -> GateEvaluator | None` function

- [x] Task 7: Export from Gates Module (AC: 1)
  - [x] Update `src/yolo_developer/gates/__init__.py` with all exports
  - [x] Export `quality_gate` decorator
  - [x] Export `GateResult`, `GateMode`, `GateContext`
  - [x] Export `GateEvaluator` protocol and registry functions
  - [x] Update module docstring with usage example

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_types.py` for data structures
  - [x] Create `tests/unit/gates/test_decorator.py` for decorator
  - [x] Create `tests/unit/gates/test_evaluators.py` for evaluator registry
  - [x] Test blocking mode prevents execution on failure
  - [x] Test advisory mode logs warning but continues
  - [x] Test async function handling
  - [x] Test state management (gate_blocked, gate_failure, gate_results)

- [x] Task 9: Write Integration Tests (AC: 2, 3, 4)
  - [x] Create `tests/integration/test_quality_gates.py`
  - [x] Test full gate lifecycle: decorate -> evaluate -> block/pass
  - [x] Test multiple gates on same node
  - [x] Test gate result accumulation in state
  - [x] Test with mock LangGraph-style state dict

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Decorator-based gates per architecture specification
- **ADR-001 (State Management):** TypedDict for state, immutable data structures
- **FR18:** System can validate artifacts at each agent boundary before handoff
- **FR24:** System can block handoffs when quality gates fail

### Technical Requirements

- **Decorator Pattern:** Use `functools.wraps` to preserve function metadata
- **Async Support:** Handle async functions properly (LangGraph nodes are async)
- **State Keys:** Use snake_case for state dict keys per project conventions
- **Logging:** Use structlog for structured logging per ADR-008

### Architecture Pattern from ADR-006

```python
from functools import wraps

def quality_gate(gate_name: str, blocking: bool = True):
    def decorator(func):
        @wraps(func)
        async def wrapper(state: YoloState) -> YoloState:
            # Run gate evaluation
            result = await evaluate_gate(gate_name, state)

            if not result.passed and blocking:
                state["gate_blocked"] = True
                state["gate_failure"] = result.reason
                return state  # Don't proceed

            # Log gate result to audit
            await log_gate_result(gate_name, result)

            return await func(state)
        return wrapper
    return decorator

@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> YoloState:
    # Agent logic here
    ...
```

### Library/Framework Requirements

- **functools:** Use `wraps` for metadata preservation
- **structlog:** Structured logging for gate results
- **typing:** Use Protocol for GateEvaluator abstraction
- **dataclasses:** Frozen dataclasses for immutable gate types

### File Structure Requirements

```
src/yolo_developer/gates/
├── __init__.py         # UPDATE: Export gate components
├── types.py            # NEW: GateResult, GateMode, GateContext
├── decorator.py        # NEW: @quality_gate decorator
├── evaluators.py       # NEW: GateEvaluator protocol and registry
└── gates/
    └── __init__.py     # Placeholder for specific gate implementations
```

### Testing Standards

- Use pytest-asyncio for async tests
- Mock evaluators for unit tests
- Test both blocking and advisory modes
- Test state mutations are handled correctly
- Test decorator preserves function metadata

### Previous Story Intelligence (from Story 2.8)

**Learnings to Apply:**
1. Use frozen dataclasses for immutable data structures
2. Export validation functions from `__init__.py`
3. Use `asyncio.to_thread()` for any blocking operations
4. Follow snake_case naming convention for all state keys
5. Comprehensive unit tests before integration tests

**Files to Reference:**
- `src/yolo_developer/memory/decisions.py` - Dataclass patterns (frozen=True)
- `src/yolo_developer/memory/factory.py` - Factory pattern for registries
- `src/yolo_developer/memory/isolation.py` - Validation function patterns

### Git Intelligence (Recent Commits)

Recent implementation patterns from Stories 2.1-2.8:
- Consistent use of `@dataclass(frozen=True)` for immutable data
- Protocol-based abstractions for extensibility
- Export from `__init__.py` for public API
- Integration tests verify full lifecycle

### Project Structure Notes

- Alignment with `src/yolo_developer/gates/` module organization per architecture
- Gate evaluators will be implemented in subsequent stories (3.2-3.5)
- This story establishes the decorator infrastructure only
- Specific gates (testability, AC measurability, etc.) are separate stories

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: epics.md#Story-3.1] - Create Quality Gate Decorator requirements
- [Source: prd.md#FR18] - System can validate artifacts at each agent boundary before handoff
- [Source: prd.md#FR24] - System can block handoffs when quality gates fail
- [Story 2.7/2.8 Implementation] - Patterns for dataclasses and factory registration

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. **Implemented full quality gate decorator framework** following ADR-006 pattern
2. **GateResult dataclass** uses `@dataclass(frozen=True)` for immutability with `to_dict()` method for state serialization
3. **GateContext dataclass** provides state access and metadata to evaluators
4. **GateEvaluator Protocol** uses `@runtime_checkable` for isinstance checking - callable signature `(GateContext) -> Awaitable[GateResult]`
5. **Evaluator registry** implemented with global dict `_evaluators` and functions: `register_evaluator`, `get_evaluator`, `list_evaluators`, `clear_evaluators`
6. **Decorator behavior**: blocking mode sets `gate_blocked=True` and `gate_failure`; advisory mode adds to `advisory_warnings` list
7. **State management**: decorator copies input state to prevent mutation, appends results to `gate_results` list
8. **Fail-open behavior**: unregistered gate names pass by default with warning log
9. **51 gate-specific tests** (42 unit + 9 integration), 736 total tests pass
10. **Fixed decorator execution order test**: outer decorator (topmost @) runs first, then inner decorators
11. **GateMode enum**: Defined for future extensibility and type documentation; decorator uses simpler `blocking: bool` parameter for ergonomics
12. **Code review fixes**: Updated _log_gate_result to sync (no async ops), corrected task description to match AC6 async-only requirement

### File List

**New Source Files:**
- `src/yolo_developer/gates/types.py` - GateMode enum, GateResult dataclass, GateContext dataclass
- `src/yolo_developer/gates/evaluators.py` - GateEvaluator protocol and registry functions
- `src/yolo_developer/gates/decorator.py` - @quality_gate decorator implementation

**Modified Source Files:**
- `src/yolo_developer/gates/__init__.py` - Updated exports and module docstring

**New Test Files:**
- `tests/unit/gates/test_types.py` - 15 unit tests for data structures
- `tests/unit/gates/test_evaluators.py` - 7 unit tests for evaluator registry
- `tests/unit/gates/test_decorator.py` - 20 unit tests for decorator behavior
- `tests/integration/test_quality_gates.py` - 9 integration tests for full lifecycle

**Modified Test Files (formatting via ruff):**
- `tests/unit/gates/__init__.py` - Test package (existed from Story 1.2)
- `tests/unit/memory/test_factory.py` - Formatting changes
- `tests/integration/test_project_isolation.py` - Formatting changes, removed unused JSONGraphStore import
