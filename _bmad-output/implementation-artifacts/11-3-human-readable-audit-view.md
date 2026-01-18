# Story 11.3: Human-Readable Audit View

Status: done

## Story

As a developer,
I want to view the audit trail in human-readable format,
So that I can review system behavior easily.

## Acceptance Criteria

1. **Given** audit data
   **When** human-readable view is requested
   **Then** events are displayed chronologically

2. **Given** audit data with multiple decision types
   **When** human-readable view is rendered
   **Then** formatting aids readability (colors, sections, indentation)

3. **Given** audit data with technical details
   **When** human-readable view is requested
   **Then** technical details are expandable/collapsible

4. **Given** audit data with various decision severities
   **When** human-readable view is rendered
   **Then** key decisions are highlighted based on severity

## Tasks / Subtasks

- [x] Task 1: Create audit formatter type definitions (AC: #1, #2, #4)
  - [x] 1.1 Create `src/yolo_developer/audit/formatter_types.py` with:
    - `FormatterStyle` Literal type: "minimal", "standard", "verbose"
    - `ColorScheme` frozen dataclass: severity colors, agent colors, section colors
    - `FormatOptions` frozen dataclass: style, show_metadata, show_trace_links, max_content_length, highlight_severity
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging (per ADR-007)
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `DEFAULT_COLOR_SCHEME`, `DEFAULT_FORMAT_OPTIONS`

- [x] Task 2: Create audit formatter protocol (AC: #1, #2, #3, #4)
  - [x] 2.1 Create `src/yolo_developer/audit/formatter_protocol.py`
  - [x] 2.2 Define `AuditFormatter` Protocol with methods:
    - `def format_decision(decision: Decision) -> str` - format single decision
    - `def format_decisions(decisions: list[Decision]) -> str` - format list chronologically
    - `def format_trace_chain(artifacts: list[TraceableArtifact], links: list[TraceLink]) -> str` - format trace chain
    - `def format_coverage_report(report: dict[str, Any]) -> str` - format coverage statistics
    - `def format_summary(decisions: list[Decision]) -> str` - format summary statistics

- [x] Task 3: Implement Rich-based terminal formatter (AC: #1, #2, #3, #4)
  - [x] 3.1 Create `src/yolo_developer/audit/rich_formatter.py`
  - [x] 3.2 Implement `RichAuditFormatter` class implementing `AuditFormatter` protocol
  - [x] 3.3 Use Rich library for:
    - `Console` for output rendering
    - `Panel` for decision display with borders
    - `Table` for tabular data (decisions list, coverage report)
    - `Tree` for trace chain visualization
    - `Syntax` for code/JSON highlighting in technical details
    - `Markdown` for rendering decision content
  - [x] 3.4 Implement color coding:
    - Severity: critical=red, high=yellow, low=green
    - Agent types: analyst=blue, pm=cyan, architect=magenta, dev=green, sm=yellow, tea=red
  - [x] 3.5 Implement expandable sections using `Collapse` or manual expand/collapse state
  - [x] 3.6 Add chronological sorting with timestamp formatting

- [x] Task 4: Implement plain text formatter (AC: #1, #2)
  - [x] 4.1 Create `src/yolo_developer/audit/plain_formatter.py`
  - [x] 4.2 Implement `PlainAuditFormatter` class implementing `AuditFormatter` protocol
  - [x] 4.3 Use ASCII-based formatting for non-terminal contexts:
    - Indentation for hierarchy
    - Dashes/equals for separators
    - Bracketed labels for sections

- [x] Task 5: Create audit view service (AC: #1, #2, #3, #4)
  - [x] 5.1 Create `src/yolo_developer/audit/view.py`
  - [x] 5.2 Implement `AuditViewService` class:
    - Constructor takes `DecisionStore`, `TraceabilityStore`, and optional `AuditFormatter`
    - `async def view_decisions(filters: DecisionFilters | None = None, options: FormatOptions | None = None) -> str`
    - `async def view_decision(decision_id: str, options: FormatOptions | None = None) -> str`
    - `async def view_trace_chain(artifact_id: str, direction: Literal["upstream", "downstream"], options: FormatOptions | None = None) -> str`
    - `async def view_coverage(options: FormatOptions | None = None) -> str`
    - `async def view_summary(filters: DecisionFilters | None = None, options: FormatOptions | None = None) -> str`
  - [x] 5.3 Add structured logging with structlog for view operations
  - [x] 5.4 Implement `get_audit_view_service(decision_store, traceability_store, formatter) -> AuditViewService` factory function

- [x] Task 6: Update module exports (AC: all)
  - [x] 6.1 Update `src/yolo_developer/audit/__init__.py`
  - [x] 6.2 Export new public types: FormatterStyle, ColorScheme, FormatOptions
  - [x] 6.3 Export formatter protocol: AuditFormatter
  - [x] 6.4 Export formatter implementations: RichAuditFormatter, PlainAuditFormatter
  - [x] 6.5 Export service: AuditViewService, get_audit_view_service
  - [x] 6.6 Export constants: DEFAULT_COLOR_SCHEME, DEFAULT_FORMAT_OPTIONS
  - [x] 6.7 Update module docstring documenting FR83 implementation

- [x] Task 7: Write comprehensive tests (AC: all)
  - [x] 7.1 Create `tests/unit/audit/test_formatter_types.py`:
    - Test FormatOptions validation (valid/invalid values)
    - Test ColorScheme validation
    - Test `to_dict()` produces JSON-serializable output
    - Test frozen dataclass immutability
    - Test DEFAULT_COLOR_SCHEME and DEFAULT_FORMAT_OPTIONS constants
  - [x] 7.2 Create `tests/unit/audit/test_formatter_protocol.py`:
    - Test protocol definition
  - [x] 7.3 Create `tests/unit/audit/test_rich_formatter.py`:
    - Test `format_decision` produces Rich-formatted output
    - Test `format_decisions` orders chronologically
    - Test `format_trace_chain` produces tree visualization
    - Test `format_coverage_report` produces table output
    - Test `format_summary` produces summary statistics
    - Test severity color coding
    - Test agent type color coding
    - Test different FormatterStyle options (minimal, standard, verbose)
  - [x] 7.4 Create `tests/unit/audit/test_plain_formatter.py`:
    - Test `format_decision` produces plain text output
    - Test `format_decisions` orders chronologically
    - Test `format_trace_chain` produces ASCII tree
    - Test all format methods produce valid plain text
  - [x] 7.5 Create `tests/unit/audit/test_view_service.py`:
    - Test `view_decisions` retrieves and formats decisions
    - Test `view_decision` retrieves single decision
    - Test `view_trace_chain` navigates and formats trace chain
    - Test `view_coverage` generates coverage report
    - Test `view_summary` generates summary statistics
    - Test `get_audit_view_service` factory function
    - Test with filters applied
    - Test with different format options
  - [x] 7.6 Update `tests/unit/audit/test_init.py`:
    - Add tests for new exports

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `formatter_types.py` (frozen dataclasses per ADR-001)
- **Protocol Pattern**: Use Protocol for AuditFormatter to allow future implementations (HTML, JSON, etc.)
- **Logging**: Use `structlog.get_logger(__name__)` pattern per architecture
- **State**: Frozen dataclasses for internal types
- **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Design Decisions

1. **Rich Library Integration**: Use Rich (already in project dependencies) for beautiful terminal output
   - Panels, Tables, Trees, Syntax highlighting
   - Color support with fallback for non-terminal contexts

2. **Formatter Protocol**: Allows swapping formatters for different output contexts
   - Rich for interactive terminal
   - Plain for logging/files
   - Future: HTML, JSON

3. **Expandable Details**: Technical details like metadata, trace links, and full rationale are initially collapsed
   - In Rich: Use Collapse or render with expand flag
   - In Plain: Use indentation with "..." indicator for truncation

4. **Severity Highlighting**: Decisions with critical/high severity are visually distinct
   - Colors in Rich (red/yellow)
   - Markers in Plain ([CRITICAL], [HIGH])

5. **Integration with Existing Types**: Builds on Story 11.1 (Decision) and Story 11.2 (TraceableArtifact)

### Project Structure Notes

Module location: `src/yolo_developer/audit/`
```
audit/
├── __init__.py                  # Module exports (update)
├── types.py                     # Decision types (existing)
├── store.py                     # DecisionStore protocol (existing)
├── memory_store.py              # InMemoryDecisionStore (existing)
├── logger.py                    # DecisionLogger class (existing)
├── traceability_types.py        # Traceability types (existing)
├── traceability_store.py        # TraceabilityStore protocol (existing)
├── traceability_memory_store.py # InMemoryTraceabilityStore (existing)
├── traceability.py              # TraceabilityService class (existing)
├── formatter_types.py           # Formatter type definitions (NEW)
├── formatter_protocol.py        # AuditFormatter Protocol (NEW)
├── rich_formatter.py            # RichAuditFormatter implementation (NEW)
├── plain_formatter.py           # PlainAuditFormatter implementation (NEW)
└── view.py                      # AuditViewService class (NEW)
```

Test location: `tests/unit/audit/`
```
tests/unit/audit/
├── __init__.py                  # Package init (existing)
├── test_types.py                # Type tests (existing)
├── test_memory_store.py         # Store tests (existing)
├── test_logger.py               # Logger tests (existing)
├── test_traceability_*.py       # Traceability tests (existing)
├── test_formatter_types.py      # Formatter type tests (NEW)
├── test_formatter_protocol.py   # Protocol tests (NEW)
├── test_rich_formatter.py       # Rich formatter tests (NEW)
├── test_plain_formatter.py      # Plain formatter tests (NEW)
├── test_view_service.py         # View service tests (NEW)
└── test_init.py                 # Module export tests (update)
```

### Previous Story Intelligence (11.1, 11.2)

Stories 11.1 and 11.2 established the following patterns that MUST be followed:

1. **Frozen Dataclasses**: All types use `@dataclass(frozen=True)` with:
   - `__post_init__` for validation with warning logging
   - `to_dict()` for JSON serialization

2. **Protocol Pattern**: `DecisionStore`, `TraceabilityStore` Protocols enable pluggable backends

3. **Thread Safety**: `InMemoryDecisionStore` and `InMemoryTraceabilityStore` use `threading.Lock`

4. **Structured Logging**: Uses `structlog.get_logger(__name__)`

5. **Factory Function**: `get_logger(store)`, `get_traceability_service(store)` pattern for dependency injection

6. **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Types to Work With

From Story 11.1 (`types.py`):
```python
@dataclass(frozen=True)
class Decision:
    id: str
    agent: AgentIdentity
    decision_type: DecisionType
    severity: DecisionSeverity
    content: str
    rationale: str
    timestamp: str
    context: DecisionContext
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DecisionContext:
    sprint_id: str | None = None
    story_id: str | None = None
    artifact_id: str | None = None
    parent_decision_id: str | None = None
    trace_links: list[str] = field(default_factory=list)
```

From Story 11.2 (`traceability_types.py`):
```python
@dataclass(frozen=True)
class TraceableArtifact:
    id: str
    artifact_type: ArtifactType
    name: str
    description: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TraceLink:
    id: str
    source_id: str
    source_type: ArtifactType
    target_id: str
    target_type: ArtifactType
    link_type: LinkType
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Integration Points

This story builds on Stories 11.1 and 11.2 and enables future stories:
- Story 11.1 (Decision Logging): Uses Decision, DecisionStore
- Story 11.2 (Requirement Traceability): Uses TraceableArtifact, TraceLink, TraceabilityService
- Story 11.4 (Audit Export): Will use formatters for export
- Story 11.5 (Cross-Agent Correlation): Will display correlation data
- Story 11.7 (Audit Filtering): Will integrate with filters

### Example Usage

```python
from rich.console import Console
from yolo_developer.audit import (
    AuditViewService,
    RichAuditFormatter,
    FormatOptions,
    DecisionFilters,
    InMemoryDecisionStore,
    InMemoryTraceabilityStore,
    get_audit_view_service,
)

# Create stores and formatter
decision_store = InMemoryDecisionStore()
traceability_store = InMemoryTraceabilityStore()
formatter = RichAuditFormatter(Console())

# Create view service
view_service = get_audit_view_service(
    decision_store=decision_store,
    traceability_store=traceability_store,
    formatter=formatter,
)

# View all decisions chronologically
output = await view_service.view_decisions()
print(output)

# View decisions with filters
filters = DecisionFilters(
    agent_types=["analyst", "pm"],
    severities=["critical", "high"],
)
output = await view_service.view_decisions(filters=filters)
print(output)

# View single decision with verbose details
options = FormatOptions(style="verbose", show_metadata=True)
output = await view_service.view_decision("decision-123", options=options)
print(output)

# View trace chain as tree
output = await view_service.view_trace_chain(
    artifact_id="code-123",
    direction="upstream",
)
print(output)

# View coverage report
output = await view_service.view_coverage()
print(output)

# View summary statistics
output = await view_service.view_summary()
print(output)
```

### Technical Constraints

1. **Async/Await**: All I/O operations must be async per ADR patterns
2. **Type Hints**: Full type annotations required (mypy strict mode)
3. **Import Order**: Standard library → Third-party → Local (per architecture)
4. **snake_case**: All field names use snake_case
5. **Test Coverage**: Target 100% coverage matching Story 11.1, 11.2
6. **Rich Library**: Already installed (used by Typer CLI per ARCH-DEP-5)

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR83: Users can view audit trail in human-readable format
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/planning-artifacts/architecture.md] structlog for structured logging
- [Source: _bmad-output/planning-artifacts/architecture.md] Typer + Rich for CLI
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11.3] Story definition and acceptance criteria
- [Source: _bmad-output/implementation-artifacts/11-1-decision-logging.md] Decision types and store patterns
- [Source: _bmad-output/implementation-artifacts/11-2-requirement-traceability.md] Traceability types and service patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- All 7 tasks completed using red-green-refactor TDD cycle
- Task 1: Created `formatter_types.py` with FormatterStyle, ColorScheme, FormatOptions frozen dataclasses
- Task 2: Created `formatter_protocol.py` with AuditFormatter Protocol (runtime checkable)
- Task 3: Created `rich_formatter.py` with RichAuditFormatter using Rich library (Panel, Table, Tree)
- Task 4: Created `plain_formatter.py` with PlainAuditFormatter for ASCII output
- Task 5: Created `view.py` with AuditViewService orchestrating view operations
- Task 6: Updated `__init__.py` with all new exports and FR83 documentation
- Task 7: Created comprehensive tests (270 tests total in audit module after code review fixes)
- Fixed protocol signatures to include optional FormatOptions parameter
- All tests passing (270 passed)
- mypy type checking passing
- ruff linting passing

### Code Review Fixes

Code review identified the following issues which have been fixed:

1. **HIGH #2 - view_coverage() incomplete**: Fixed coverage calculation in `view.py` to properly calculate:
   - `total_requirements`: Count of all requirement artifacts
   - `covered_requirements`: Requirements with incoming links (something implements them)
   - `coverage_percentage`: Proper percentage calculation
   - `unlinked_requirements`: Requirements with no incoming links

2. **MEDIUM #5 - Protocol consistency**: Added `options: FormatOptions | None = None` parameter to:
   - `format_trace_chain` in `formatter_protocol.py`
   - `format_coverage_report` in `formatter_protocol.py`
   - Updated implementations in `rich_formatter.py` and `plain_formatter.py`

3. **MEDIUM #6 - Missing tests**: Added tests for options parameter acceptance in:
   - `test_rich_formatter.py`: `test_format_trace_chain_accepts_options`, `test_format_coverage_report_accepts_options`
   - `test_plain_formatter.py`: `test_format_trace_chain_accepts_options`, `test_format_coverage_report_accepts_options`

4. **MEDIUM #7 - Unused options parameter**: Fixed `view.py` to pass options to `format_trace_chain` and `format_coverage_report` calls

### File List

**New Source Files:**
- `src/yolo_developer/audit/formatter_types.py`
- `src/yolo_developer/audit/formatter_protocol.py`
- `src/yolo_developer/audit/rich_formatter.py`
- `src/yolo_developer/audit/plain_formatter.py`
- `src/yolo_developer/audit/view.py`

**Modified Source Files:**
- `src/yolo_developer/audit/__init__.py`

**New Test Files:**
- `tests/unit/audit/test_formatter_types.py`
- `tests/unit/audit/test_formatter_protocol.py`
- `tests/unit/audit/test_rich_formatter.py`
- `tests/unit/audit/test_plain_formatter.py`
- `tests/unit/audit/test_view_service.py`

**Modified Test Files:**
- `tests/unit/audit/test_init.py`
