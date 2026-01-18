# Story 11.2: Requirement Traceability

Status: review

## Story

As a developer,
I want to trace any line of code back to its requirement,
So that I can verify coverage and understand purpose.

## Acceptance Criteria

1. **Given** generated code
   **When** traceability is queried
   **Then** the originating requirement is identified

2. **Given** a code artifact with traceability
   **When** the trace is examined
   **Then** the story is linked

3. **Given** a traced artifact
   **When** the full trace is requested
   **Then** the design decision is referenced

4. **Given** a requirement or code artifact
   **When** the traceability chain is navigated
   **Then** the full chain is navigable (requirement → story → design → code)

5. **Given** traceability data exists
   **When** coverage is queried
   **Then** requirements without implementations are identifiable

## Tasks / Subtasks

- [x] Task 1: Create traceability type definitions (AC: #1, #2, #3, #4)
  - [x] 1.1 Create `src/yolo_developer/audit/traceability_types.py` with:
    - `ArtifactType` Literal type: "requirement", "story", "design_decision", "code", "test"
    - `TraceLink` frozen dataclass: id, source_id, source_type, target_id, target_type, link_type, created_at, metadata
    - `LinkType` Literal type: "derives_from", "implements", "tests", "documents"
    - `TraceableArtifact` frozen dataclass: id, artifact_type, name, description, created_at, metadata
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging (per ADR-007)
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `VALID_ARTIFACT_TYPES`, `VALID_LINK_TYPES`

- [x] Task 2: Create traceability store protocol (AC: #1, #4, #5)
  - [x] 2.1 Create `src/yolo_developer/audit/traceability_store.py`
  - [x] 2.2 Define `TraceabilityStore` Protocol with methods:
    - `async def register_artifact(artifact: TraceableArtifact) -> str` - returns artifact_id
    - `async def create_link(link: TraceLink) -> str` - returns link_id
    - `async def get_artifact(artifact_id: str) -> TraceableArtifact | None`
    - `async def get_links_from(source_id: str) -> list[TraceLink]` - get all links where source_id matches
    - `async def get_links_to(target_id: str) -> list[TraceLink]` - get all links where target_id matches
    - `async def get_trace_chain(artifact_id: str, direction: Literal["upstream", "downstream"]) -> list[TraceableArtifact]` - traverse the full trace chain
    - `async def get_unlinked_artifacts(artifact_type: ArtifactType) -> list[TraceableArtifact]` - find artifacts with no outgoing links

- [x] Task 3: Implement in-memory traceability store (AC: #1, #4, #5)
  - [x] 3.1 Create `src/yolo_developer/audit/traceability_memory_store.py`
  - [x] 3.2 Implement `InMemoryTraceabilityStore` class implementing `TraceabilityStore` protocol
  - [x] 3.3 Use thread-safe storage with `threading.Lock` for concurrent access
  - [x] 3.4 Implement BFS/DFS traversal for trace chain navigation
  - [x] 3.5 Implement efficient reverse index for `get_links_to` queries

- [x] Task 4: Create traceability service (AC: #1, #2, #3, #4, #5)
  - [x] 4.1 Create `src/yolo_developer/audit/traceability.py`
  - [x] 4.2 Implement `TraceabilityService` class:
    - Constructor takes `TraceabilityStore` instance
    - `async def trace_requirement(requirement_id: str, name: str, description: str) -> str` - register a requirement
    - `async def trace_story(story_id: str, requirement_id: str, name: str, description: str) -> str` - register story linked to requirement
    - `async def trace_design(design_id: str, story_id: str, name: str, description: str) -> str` - register design linked to story
    - `async def trace_code(code_id: str, design_id: str, name: str, description: str) -> str` - register code linked to design
    - `async def trace_test(test_id: str, code_id: str, name: str, description: str) -> str` - register test linked to code
    - `async def get_requirement_for_code(code_id: str) -> TraceableArtifact | None` - navigate upstream to requirement
    - `async def get_code_for_requirement(requirement_id: str) -> list[TraceableArtifact]` - navigate downstream to code
    - `async def get_coverage_report() -> dict[str, Any]` - return coverage statistics
  - [x] 4.3 Add structured logging with structlog for each trace operation
  - [x] 4.4 Implement `get_traceability_service(store: TraceabilityStore | None = None) -> TraceabilityService` factory function

- [x] Task 5: Integrate with decision logging (AC: #3)
  - [x] 5.1 Update `src/yolo_developer/audit/types.py`:
    - Add `trace_links: list[str]` field to `DecisionContext` (list of TraceLink IDs)
  - [x] 5.2 Update `src/yolo_developer/audit/logger.py`:
    - Add optional `trace_links` parameter to `log()` method
    - Log trace link references with decisions

- [x] Task 6: Update module exports (AC: all)
  - [x] 6.1 Update `src/yolo_developer/audit/__init__.py`
  - [x] 6.2 Export all new public types: TraceableArtifact, TraceLink, ArtifactType, LinkType
  - [x] 6.3 Export store protocol and implementations: TraceabilityStore, InMemoryTraceabilityStore
  - [x] 6.4 Export service: TraceabilityService, get_traceability_service
  - [x] 6.5 Export constants: VALID_ARTIFACT_TYPES, VALID_LINK_TYPES
  - [x] 6.6 Update module docstring documenting FR82 implementation

- [x] Task 7: Write comprehensive tests (AC: all)
  - [x] 7.1 Create `tests/unit/audit/test_traceability_types.py`:
    - Test type validation (valid/invalid values)
    - Test `to_dict()` produces JSON-serializable output
    - Test frozen dataclass immutability
  - [x] 7.2 Create `tests/unit/audit/test_traceability_store.py`:
    - Test protocol definition
  - [x] 7.3 Create `tests/unit/audit/test_traceability_memory_store.py`:
    - Test `register_artifact` returns valid ID
    - Test `create_link` returns valid ID
    - Test `get_links_from` retrieves correct links
    - Test `get_links_to` retrieves correct links
    - Test `get_trace_chain` upstream navigation
    - Test `get_trace_chain` downstream navigation
    - Test `get_unlinked_artifacts` finds orphans
    - Test concurrent access safety
  - [x] 7.4 Create `tests/unit/audit/test_traceability_service.py`:
    - Test `trace_requirement` creates artifact
    - Test `trace_story` creates artifact and link
    - Test `trace_design` creates artifact and link
    - Test `trace_code` creates artifact and link
    - Test `trace_test` creates artifact and link
    - Test `get_requirement_for_code` navigates upstream
    - Test `get_code_for_requirement` navigates downstream
    - Test `get_coverage_report` returns statistics
    - Test `get_traceability_service` factory function
  - [x] 7.5 Update `tests/unit/audit/test_init.py`:
    - Add tests for new exports

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `traceability_types.py` (frozen dataclasses per ADR-001)
- **Protocol Pattern**: Use Protocol for TraceabilityStore to allow future implementations (file-based, database, Neo4j)
- **Logging**: Use `structlog.get_logger(__name__)` pattern per architecture
- **State**: Frozen dataclasses for internal types
- **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Design Decisions

1. **Bidirectional Navigation**: The traceability system must support both upstream (code → requirement) and downstream (requirement → code) navigation efficiently.

2. **Graph Structure**: The trace links form a directed acyclic graph (DAG):
   ```
   Requirement → Story → Design Decision → Code → Test
   ```

3. **Link Types**: Use semantic link types to describe relationships:
   - `derives_from`: Story derives from requirement
   - `implements`: Code implements design
   - `tests`: Test validates code
   - `documents`: ADR documents design decision

4. **Coverage Tracking**: Track which requirements have implementations and which are orphaned.

5. **Integration with Decision Logging**: Design decisions from Story 11.1 can reference trace links for full traceability.

### Project Structure Notes

Module location: `src/yolo_developer/audit/`
```
audit/
├── __init__.py              # Module exports (update)
├── types.py                 # Type definitions (update - add trace_links to DecisionContext)
├── store.py                 # DecisionStore protocol (existing)
├── memory_store.py          # InMemoryDecisionStore implementation (existing)
├── logger.py                # DecisionLogger class (update - add trace_links parameter)
├── traceability_types.py    # Traceability type definitions (NEW)
├── traceability_store.py    # TraceabilityStore protocol (NEW)
├── traceability_memory_store.py  # InMemoryTraceabilityStore implementation (NEW)
└── traceability.py          # TraceabilityService class (NEW)
```

Test location: `tests/unit/audit/`
```
tests/unit/audit/
├── __init__.py                      # Package init (existing)
├── test_types.py                    # Type tests (existing)
├── test_memory_store.py             # Store tests (existing)
├── test_logger.py                   # Logger tests (existing)
├── test_traceability_types.py       # Traceability type tests (NEW)
├── test_traceability_store.py       # Protocol tests (NEW)
├── test_traceability_memory_store.py # Memory store tests (NEW)
├── test_traceability_service.py     # Service tests (NEW)
└── test_init.py                     # Module export tests (update)
```

### Previous Story Intelligence (11.1)

Story 11.1 established the following patterns that MUST be followed:

1. **Frozen Dataclasses**: All types use `@dataclass(frozen=True)` with:
   - `__post_init__` for validation with warning logging
   - `to_dict()` for JSON serialization

2. **Protocol Pattern**: `DecisionStore` Protocol enables pluggable backends

3. **Thread Safety**: `InMemoryDecisionStore` uses `threading.Lock`

4. **Structured Logging**: Uses `structlog.get_logger(__name__)`

5. **Factory Function**: `get_logger(store)` pattern for dependency injection

6. **Error Handling**: Per ADR-007 - log errors, don't block callers

### Integration Points

This story builds on Story 11.1 and enables future stories:
- Story 11.1 (Decision Logging): Already implemented - integrate trace links
- Story 11.3 (Human-Readable View): Will format trace chains for display
- Story 11.4 (Audit Export): Will export trace data
- Story 11.5 (Cross-Agent Correlation): Will use trace links for correlation
- Story 11.8 (Auto ADR Generation): Will create trace links to generated ADRs

### Example Usage

```python
from yolo_developer.audit import (
    TraceabilityService,
    InMemoryTraceabilityStore,
    get_traceability_service,
)

# Create store and service
store = InMemoryTraceabilityStore()
service = get_traceability_service(store)

# Register requirement
req_id = await service.trace_requirement(
    requirement_id="FR81",
    name="Decision Logging",
    description="System can log all agent decisions with rationale",
)

# Register story derived from requirement
story_id = await service.trace_story(
    story_id="11-1-decision-logging",
    requirement_id="FR81",
    name="Story 11.1: Decision Logging",
    description="Implement decision logging for audit trail",
)

# Register design decision for the story
design_id = await service.trace_design(
    design_id="design-001",
    story_id="11-1-decision-logging",
    name="In-Memory Decision Store",
    description="Use in-memory store with protocol for future backends",
)

# Register code implementing the design
code_id = await service.trace_code(
    code_id="src/yolo_developer/audit/logger.py",
    design_id="design-001",
    name="DecisionLogger",
    description="High-level decision logging with auto-generated IDs",
)

# Navigate upstream: code → requirement
requirement = await service.get_requirement_for_code(code_id)
print(f"Code implements requirement: {requirement.name}")

# Navigate downstream: requirement → code
code_artifacts = await service.get_code_for_requirement(req_id)
print(f"Requirement has {len(code_artifacts)} code artifacts")

# Get coverage report
report = await service.get_coverage_report()
print(f"Requirements covered: {report['covered_requirements']}/{report['total_requirements']}")
```

### Technical Constraints

1. **Async/Await**: All I/O operations must be async per ADR patterns
2. **Type Hints**: Full type annotations required (mypy strict mode)
3. **Import Order**: Standard library → Third-party → Local (per architecture)
4. **snake_case**: All field names use snake_case
5. **Test Coverage**: Target 100% coverage matching Story 11.1

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR82: System can generate decision traceability from requirement to code
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/planning-artifacts/architecture.md] structlog for structured logging
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11.2] Story definition and acceptance criteria
- [Source: _bmad-output/implementation-artifacts/11-1-decision-logging.md] Previous story patterns and integration points

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debugging required

### Completion Notes List

1. Implemented full traceability system following Story 11.1 patterns (frozen dataclasses, Protocol pattern, factory functions)
2. All 161 audit tests pass with 100% code coverage
3. Integrated trace_links into DecisionContext and DecisionLogger for Story 11.1/11.2 integration
4. BFS traversal used for efficient trace chain navigation (upstream/downstream)
5. Thread-safe InMemoryTraceabilityStore with reverse index for efficient get_links_to queries
6. All acceptance criteria met:
   - AC1: Code can be traced to originating requirement via get_requirement_for_code()
   - AC2: Code artifacts are linked to stories via trace chain
   - AC3: Design decisions referenced via trace chain (requirement → story → design → code)
   - AC4: Full chain navigable in both directions (upstream/downstream)
   - AC5: Coverage report identifies requirements without implementations

### File List

**New Files Created:**
- src/yolo_developer/audit/traceability_types.py - ArtifactType, LinkType, TraceableArtifact, TraceLink
- src/yolo_developer/audit/traceability_store.py - TraceabilityStore Protocol
- src/yolo_developer/audit/traceability_memory_store.py - InMemoryTraceabilityStore implementation
- src/yolo_developer/audit/traceability.py - TraceabilityService and get_traceability_service factory
- tests/unit/audit/test_traceability_types.py - 32 tests for type definitions
- tests/unit/audit/test_traceability_store.py - 9 tests for protocol
- tests/unit/audit/test_traceability_memory_store.py - 16 tests for memory store
- tests/unit/audit/test_traceability_service.py - 18 tests for service

**Modified Files:**
- src/yolo_developer/audit/types.py - Added trace_links field to DecisionContext
- src/yolo_developer/audit/logger.py - Added trace_links parameter to log() method
- src/yolo_developer/audit/__init__.py - Added all new traceability exports
- tests/unit/audit/test_init.py - Added tests for new traceability exports
- tests/unit/audit/test_logger.py - Added trace_links integration tests
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status
