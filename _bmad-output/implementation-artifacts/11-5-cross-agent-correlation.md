# Story 11.5: Cross-Agent Correlation

Status: done

## Story

As a developer,
I want decisions correlated across agents,
So that I can see how decisions flow through the system.

## Acceptance Criteria

1. **Given** related decisions across agents
   **When** correlation is performed
   **Then** decision chains are identified

2. **Given** a decision from one agent
   **When** the correlation is examined
   **Then** cause-effect relationships are shown

3. **Given** correlated decisions
   **When** a timeline view is requested
   **Then** decisions are displayed chronologically with agent transitions

4. **Given** a correlation query
   **When** searching for related decisions
   **Then** correlations are searchable by agent, time, or session

5. **Given** a workflow execution
   **When** the full correlation is traced
   **Then** the complete decision flow from seed to code is navigable

## Tasks / Subtasks

- [x] Task 1: Create correlation type definitions (AC: #1, #2)
  - [x] 1.1 Create `src/yolo_developer/audit/correlation_types.py` with:
    - `CorrelationType` Literal type: "causal", "temporal", "session", "artifact"
    - `DecisionChain` frozen dataclass: id, decisions (tuple of decision IDs), chain_type, created_at, metadata
    - `CausalRelation` frozen dataclass: id, cause_decision_id, effect_decision_id, relation_type, evidence, created_at
    - `AgentTransition` frozen dataclass: id, from_agent, to_agent, decision_id, timestamp, context
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging (per ADR-007)
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `VALID_CORRELATION_TYPES`

- [x] Task 2: Create correlation store protocol (AC: #1, #4)
  - [x] 2.1 Create `src/yolo_developer/audit/correlation_store.py`
  - [x] 2.2 Define `CorrelationStore` Protocol with methods:
    - `async def store_chain(chain: DecisionChain) -> str` - returns chain_id
    - `async def store_causal_relation(relation: CausalRelation) -> str` - returns relation_id
    - `async def store_transition(transition: AgentTransition) -> str` - returns transition_id
    - `async def get_chain(chain_id: str) -> DecisionChain | None`
    - `async def get_chains_for_decision(decision_id: str) -> list[DecisionChain]`
    - `async def get_causal_relations(decision_id: str) -> list[CausalRelation]`
    - `async def get_transitions_by_session(session_id: str) -> list[AgentTransition]`
    - `async def search_correlations(filters: CorrelationFilters | None = None) -> list[DecisionChain]`
  - [x] 2.3 Create `CorrelationFilters` frozen dataclass: agent_name, session_id, start_time, end_time, chain_type

- [x] Task 3: Implement in-memory correlation store (AC: #1, #4)
  - [x] 3.1 Create `src/yolo_developer/audit/correlation_memory_store.py`
  - [x] 3.2 Implement `InMemoryCorrelationStore` class implementing `CorrelationStore` protocol
  - [x] 3.3 Use thread-safe storage with `threading.Lock` for concurrent access
  - [x] 3.4 Implement efficient indexing for decision_id → chains lookup
  - [x] 3.5 Implement search with filter matching

- [x] Task 4: Create correlation service (AC: #1, #2, #3, #4, #5)
  - [x] 4.1 Create `src/yolo_developer/audit/correlation.py`
  - [x] 4.2 Implement `CorrelationService` class:
    - Constructor takes `DecisionStore`, `CorrelationStore` instances
    - `async def correlate_decisions(decision_ids: list[str], chain_type: CorrelationType = "session") -> DecisionChain` - create correlation chain
    - `async def add_causal_relation(cause_id: str, effect_id: str, relation_type: str, evidence: str = "") -> CausalRelation` - record cause-effect
    - `async def record_transition(from_agent: str, to_agent: str, decision_id: str, context: dict | None = None) -> AgentTransition` - record agent handoff
    - `async def get_decision_chain(decision_id: str) -> list[Decision]` - get all correlated decisions
    - `async def get_timeline(filters: CorrelationFilters | None = None) -> list[tuple[Decision, str | None]]` - chronological decisions with agent info
    - `async def get_workflow_flow(session_id: str) -> dict[str, Any]` - complete workflow trace
    - `async def search(query: str, filters: CorrelationFilters | None = None) -> list[Decision]` - full-text search on content/rationale
  - [x] 4.3 Add structured logging with structlog for each correlation operation
  - [x] 4.4 Implement `get_correlation_service(decision_store: DecisionStore, correlation_store: CorrelationStore | None = None) -> CorrelationService` factory

- [x] Task 5: Implement timeline view (AC: #3)
  - [x] 5.1 Add `TimelineEntry` frozen dataclass in `correlation_types.py`:
    - decision: Decision
    - agent_transition: AgentTransition | None
    - previous_agent: str | None
    - sequence_number: int
  - [x] 5.2 Implement `async def get_timeline_view(session_id: str | None = None, start_time: str | None = None, end_time: str | None = None) -> list[TimelineEntry]` in CorrelationService
  - [x] 5.3 Timeline entries ordered by timestamp with sequence numbers
  - [x] 5.4 Include agent transition markers between decisions

- [x] Task 6: Implement correlation detection (AC: #1, #2)
  - [x] 6.1 Implement `async def auto_correlate_session(session_id: str) -> DecisionChain` - auto-detect correlations within a session
  - [x] 6.2 Implement `async def detect_causal_relations(decision_id: str) -> list[CausalRelation]` - infer cause-effect from parent_decision_id in context
  - [x] 6.3 Use `DecisionContext.parent_decision_id` to build causal chains
  - [x] 6.4 Use `DecisionContext.trace_links` to correlate via traceability

- [x] Task 7: Update module exports (AC: all)
  - [x] 7.1 Update `src/yolo_developer/audit/__init__.py`
  - [x] 7.2 Export all new public types: DecisionChain, CausalRelation, AgentTransition, TimelineEntry, CorrelationType
  - [x] 7.3 Export store protocol and implementations: CorrelationStore, InMemoryCorrelationStore
  - [x] 7.4 Export service: CorrelationService, get_correlation_service
  - [x] 7.5 Export constants: VALID_CORRELATION_TYPES
  - [x] 7.6 Export filter type: CorrelationFilters
  - [x] 7.7 Update module docstring documenting FR85 implementation

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1 Create `tests/unit/audit/test_correlation_types.py`:
    - Test type validation (valid/invalid values)
    - Test `to_dict()` produces JSON-serializable output
    - Test frozen dataclass immutability
    - Test DecisionChain with multiple decisions
    - Test CausalRelation with evidence
    - Test AgentTransition with context
  - [x] 8.2 Create `tests/unit/audit/test_correlation_store.py`:
    - Test protocol definition
  - [x] 8.3 Create `tests/unit/audit/test_correlation_memory_store.py`:
    - Test `store_chain` returns valid ID
    - Test `store_causal_relation` returns valid ID
    - Test `store_transition` returns valid ID
    - Test `get_chain` retrieves correct chain
    - Test `get_chains_for_decision` retrieves chains containing decision
    - Test `get_causal_relations` retrieves relations for decision
    - Test `get_transitions_by_session` retrieves transitions
    - Test `search_correlations` with filters
    - Test concurrent access safety
  - [x] 8.4 Create `tests/unit/audit/test_correlation_service.py`:
    - Test `correlate_decisions` creates chain
    - Test `add_causal_relation` creates relation
    - Test `record_transition` creates transition
    - Test `get_decision_chain` returns correlated decisions
    - Test `get_timeline` returns chronological decisions
    - Test `get_timeline_view` returns TimelineEntry list
    - Test `get_workflow_flow` returns complete trace
    - Test `search` finds decisions by content
    - Test `auto_correlate_session` detects correlations
    - Test `detect_causal_relations` infers from parent_decision_id
    - Test `get_correlation_service` factory function
  - [x] 8.5 Update `tests/unit/audit/test_init.py`:
    - Add tests for new correlation exports

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `correlation_types.py` (frozen dataclasses per ADR-001)
- **Protocol Pattern**: Use Protocol for CorrelationStore to allow future implementations (file-based, database, Neo4j)
- **Logging**: Use `structlog.get_logger(__name__)` pattern per architecture
- **State**: Frozen dataclasses for internal types
- **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Design Decisions

1. **Correlation Types**: Support multiple correlation mechanisms:
   - `causal`: Direct cause-effect relationships (A caused B)
   - `temporal`: Time-based correlation (same time window)
   - `session`: Same session correlation (same session_id)
   - `artifact`: Related to same artifact (via trace_links)

2. **Decision Chains**: Group related decisions for navigation:
   - Chains have an ID, list of decision IDs, and chain type
   - A decision can belong to multiple chains
   - Chains support efficient lookup by any member decision

3. **Causal Relations**: Explicit cause-effect tracking:
   - Uses `parent_decision_id` from `DecisionContext` (Story 11.1)
   - Supports evidence field for explaining relationship
   - Auto-detection from parent_decision_id links

4. **Agent Transitions**: Track workflow handoffs:
   - Captures from_agent → to_agent with triggering decision
   - Essential for timeline visualization
   - Links to session for workflow reconstruction

5. **Timeline View**: Chronological visualization:
   - Decisions ordered by timestamp
   - Agent transitions marked between decisions
   - Sequence numbers for ordering
   - Supports filtering by session, time range

6. **Search**: Full-text search on decisions:
   - Searches content and rationale fields
   - Combines with filters (agent, session, time)
   - Returns correlated decisions together

### Project Structure Notes

Module location: `src/yolo_developer/audit/`
```
audit/
├── __init__.py                  # Module exports (update)
├── types.py                     # Decision types (existing)
├── store.py                     # DecisionStore protocol (existing)
├── memory_store.py              # InMemoryDecisionStore (existing)
├── logger.py                    # DecisionLogger (existing)
├── traceability_types.py        # Traceability types (existing)
├── traceability_store.py        # TraceabilityStore protocol (existing)
├── traceability_memory_store.py # InMemoryTraceabilityStore (existing)
├── traceability.py              # TraceabilityService (existing)
├── formatter_types.py           # Formatter types (existing)
├── formatter_protocol.py        # AuditFormatter protocol (existing)
├── rich_formatter.py            # RichAuditFormatter (existing)
├── plain_formatter.py           # PlainAuditFormatter (existing)
├── view.py                      # AuditViewService (existing)
├── export_types.py              # Export types (existing)
├── export_protocol.py           # AuditExporter protocol (existing)
├── json_exporter.py             # JsonAuditExporter (existing)
├── csv_exporter.py              # CsvAuditExporter (existing)
├── pdf_exporter.py              # PdfAuditExporter (existing)
├── export.py                    # AuditExportService (existing)
├── correlation_types.py         # Correlation type definitions (NEW)
├── correlation_store.py         # CorrelationStore protocol (NEW)
├── correlation_memory_store.py  # InMemoryCorrelationStore (NEW)
└── correlation.py               # CorrelationService (NEW)
```

Test location: `tests/unit/audit/`
```
tests/unit/audit/
├── __init__.py                       # Package init (existing)
├── conftest.py                       # Shared fixtures (existing)
├── test_types.py                     # Decision type tests (existing)
├── test_store.py                     # Protocol tests (existing)
├── test_memory_store.py              # Store tests (existing)
├── test_logger.py                    # Logger tests (existing)
├── test_traceability_types.py        # Traceability type tests (existing)
├── test_traceability_store.py        # Protocol tests (existing)
├── test_traceability_memory_store.py # Memory store tests (existing)
├── test_traceability_service.py      # Service tests (existing)
├── test_formatter_types.py           # Formatter type tests (existing)
├── test_formatter_protocol.py        # Protocol tests (existing)
├── test_rich_formatter.py            # Rich formatter tests (existing)
├── test_plain_formatter.py           # Plain formatter tests (existing)
├── test_view_service.py              # View service tests (existing)
├── test_export_types.py              # Export type tests (existing)
├── test_export_protocol.py           # Protocol tests (existing)
├── test_json_exporter.py             # JSON exporter tests (existing)
├── test_csv_exporter.py              # CSV exporter tests (existing)
├── test_pdf_exporter.py              # PDF exporter tests (existing)
├── test_export.py                    # Export service tests (existing)
├── test_init.py                      # Module export tests (update)
├── test_correlation_types.py         # Correlation type tests (NEW)
├── test_correlation_store.py         # Protocol tests (NEW)
├── test_correlation_memory_store.py  # Memory store tests (NEW)
└── test_correlation_service.py       # Service tests (NEW)
```

### Previous Story Intelligence

Stories 11.1-11.4 established the following patterns that MUST be followed:

1. **Frozen Dataclasses**: All types use `@dataclass(frozen=True)` with:
   - `__post_init__` for validation with warning logging
   - `to_dict()` for JSON serialization

2. **Protocol Pattern**: `DecisionStore`, `TraceabilityStore`, `AuditExporter`, `AuditFormatter` Protocols enable pluggable backends

3. **Thread Safety**: In-memory stores use `threading.Lock`

4. **Structured Logging**: Uses `structlog.get_logger(__name__)`

5. **Factory Function**: `get_*_service(store)` pattern for dependency injection

6. **Error Handling**: Per ADR-007 - log errors, don't block callers

7. **Test Fixtures**: Use shared fixtures from `conftest.py` (create_test_decision, create_test_artifact, create_test_link)

### Integration Points

This story builds on Stories 11.1-11.4:
- **Story 11.1 (Decision Logging)**: Uses `Decision`, `DecisionContext`, `parent_decision_id` for correlation
- **Story 11.2 (Requirement Traceability)**: Uses `trace_links` for artifact-based correlation
- **Story 11.3 (Human-Readable View)**: Timeline view complements existing view service
- **Story 11.4 (Audit Export)**: Correlation data should be exportable

Future stories that will use this:
- **Story 11.7 (Audit Filtering)**: Will extend correlation filters
- **Story 11.8 (Auto ADR Generation)**: Will use causal relations for ADR context

### Example Usage

```python
from yolo_developer.audit import (
    CorrelationService,
    InMemoryDecisionStore,
    InMemoryCorrelationStore,
    get_correlation_service,
    DecisionLogger,
    get_logger,
)

# Create stores and services
decision_store = InMemoryDecisionStore()
correlation_store = InMemoryCorrelationStore()
logger = get_logger(decision_store)
correlation_service = get_correlation_service(decision_store, correlation_store)

# Log some decisions
dec1_id = await logger.log(
    agent_name="analyst",
    agent_type="analyst",
    decision_type="requirement_analysis",
    content="Requirement crystallized",
    rationale="Clear and testable",
)

dec2_id = await logger.log(
    agent_name="pm",
    agent_type="pm",
    decision_type="story_creation",
    content="Story created from requirement",
    rationale="Derived from analyst requirement",
    context=DecisionContext(parent_decision_id=dec1_id),
)

# Record the agent transition
await correlation_service.record_transition(
    from_agent="analyst",
    to_agent="pm",
    decision_id=dec2_id,
)

# Auto-correlate decisions in a session
chain = await correlation_service.auto_correlate_session("session-123")
print(f"Found {len(chain.decisions)} correlated decisions")

# Get timeline view
timeline = await correlation_service.get_timeline_view(session_id="session-123")
for entry in timeline:
    print(f"[{entry.sequence_number}] {entry.decision.agent.agent_name}: {entry.decision.content}")
    if entry.agent_transition:
        print(f"  → Transition from {entry.previous_agent}")

# Get workflow flow
flow = await correlation_service.get_workflow_flow("session-123")
print(f"Workflow: {flow['agent_sequence']}")
print(f"Total decisions: {flow['total_decisions']}")

# Search for related decisions
results = await correlation_service.search("authentication", filters=CorrelationFilters(agent_name="architect"))
print(f"Found {len(results)} decisions about authentication")
```

### Technical Constraints

1. **Async/Await**: All I/O operations must be async per ADR patterns
2. **Type Hints**: Full type annotations required (mypy strict mode)
3. **Import Order**: Standard library → Third-party → Local (per architecture)
4. **snake_case**: All field names use snake_case
5. **Test Coverage**: Target 100% coverage matching Stories 11.1-11.4
6. **Shared Fixtures**: Use conftest.py fixtures for test data

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR85: System can correlate decisions across agent boundaries
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/planning-artifacts/architecture.md] structlog for structured logging
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11.5] Story definition and acceptance criteria
- [Source: _bmad-output/implementation-artifacts/11-1-decision-logging.md] Decision types and parent_decision_id
- [Source: _bmad-output/implementation-artifacts/11-2-requirement-traceability.md] Traceability and trace_links integration
- [Source: _bmad-output/implementation-artifacts/11-3-human-readable-audit-view.md] View service patterns
- [Source: _bmad-output/implementation-artifacts/11-4-audit-export.md] Export service patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debug issues encountered during implementation

### Completion Notes List

1. **Task 1-3 (completed prior)**: Correlation type definitions, store protocol, and in-memory store were implemented in previous session
2. **Task 4**: Created `correlation.py` with `CorrelationService` class implementing all correlation operations:
   - `correlate_decisions()` - creates DecisionChain from decision IDs
   - `add_causal_relation()` - records cause-effect relationships
   - `record_transition()` - tracks agent handoffs
   - `get_decision_chain()` - retrieves correlated decisions
   - `get_timeline()` - returns chronological decisions with chains
   - `get_workflow_flow()` - returns complete session workflow trace
   - `search()` - full-text search on decision content
3. **Task 5**: Added `TimelineEntry` dataclass with `agent_transition` and `previous_agent` fields, `get_timeline_view()` method with optional `session_id`, `start_time`, and `end_time` parameters for timeline visualization
4. **Task 6**: Implemented `auto_correlate_session()` and `detect_causal_relations()` with both `parent_decision_id` and `trace_links` correlation support
5. **Task 7**: Updated `__init__.py` with all correlation exports and FR85 documentation
6. **Task 8**: All tests written and passing (105 correlation tests, 517 total audit tests)

### Code Review Fixes Applied

After adversarial code review, the following issues were fixed:
- **HIGH #1-2**: Added `agent_transition: AgentTransition | None` and `previous_agent: str | None` fields to `TimelineEntry`, updated `get_timeline_view()` to populate these fields by looking up transitions
- **HIGH #3**: Implemented `trace_links` correlation in `detect_causal_relations()` to create artifact-based causal relations between decisions sharing trace links
- **MEDIUM #5**: Added `start_time` and `end_time` optional parameters to `get_timeline_view()`, made `session_id` optional
- **MEDIUM #6**: Added 14 new tests for `TimelineEntry` (validation warnings, new fields, to_dict)
- **LOW #7-8**: Updated File List with all 5 source files, corrected test counts

### File List

Source files created/modified:
- `src/yolo_developer/audit/correlation_types.py` (NEW - DecisionChain, CausalRelation, AgentTransition, TimelineEntry dataclasses)
- `src/yolo_developer/audit/correlation_store.py` (NEW - CorrelationStore protocol, CorrelationFilters)
- `src/yolo_developer/audit/correlation_memory_store.py` (NEW - InMemoryCorrelationStore implementation)
- `src/yolo_developer/audit/correlation.py` (NEW - CorrelationService)
- `src/yolo_developer/audit/__init__.py` (updated exports for FR85)

Test files created/modified:
- `tests/unit/audit/test_correlation_types.py` (40 tests including TimelineEntry validation)
- `tests/unit/audit/test_correlation_store.py` (15 tests)
- `tests/unit/audit/test_correlation_memory_store.py` (23 tests)
- `tests/unit/audit/test_correlation_service.py` (32 tests including trace_links and timeline view)
- `tests/unit/audit/test_init.py` (added 12 correlation export tests)
