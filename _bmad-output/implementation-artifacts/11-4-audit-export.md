# Story 11.4: Audit Export

Status: done

## Story

As a developer,
I want to export audit trails for compliance,
So that I can demonstrate process adherence.

## Acceptance Criteria

1. **Given** audit data
   **When** export is requested
   **Then** data is exported in requested format (JSON, CSV, PDF)

2. **Given** audit data with various decision types
   **When** export is generated
   **Then** all relevant fields are included (id, type, content, rationale, agent, timestamp, severity, context)

3. **Given** audit data spanning multiple decisions and traces
   **When** export is completed
   **Then** export is complete and accurate (no data loss)

4. **Given** audit data with potentially sensitive information
   **When** export is requested with redaction enabled
   **Then** sensitive data can be redacted (metadata fields, session_ids)

## Tasks / Subtasks

- [x] Task 1: Create export type definitions (AC: #1, #2, #4)
  - [x] 1.1 Create `src/yolo_developer/audit/export_types.py` with:
    - `ExportFormat` Literal type: "json", "csv", "pdf"
    - `RedactionConfig` frozen dataclass: redact_metadata, redact_session_ids, redact_fields (list of field paths)
    - `ExportOptions` frozen dataclass: format, include_decisions, include_traces, include_coverage, redaction_config
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging (per ADR-007)
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `VALID_EXPORT_FORMATS`, `DEFAULT_REDACTION_CONFIG`, `DEFAULT_EXPORT_OPTIONS`

- [x] Task 2: Create export protocol (AC: #1, #2, #3)
  - [x] 2.1 Create `src/yolo_developer/audit/export_protocol.py`
  - [x] 2.2 Define `AuditExporter` Protocol with methods:
    - `def export_decisions(decisions: list[Decision], options: ExportOptions | None = None) -> bytes` - export decisions
    - `def export_traces(artifacts: list[TraceableArtifact], links: list[TraceLink], options: ExportOptions | None = None) -> bytes` - export traceability data
    - `def export_full_audit(decisions: list[Decision], artifacts: list[TraceableArtifact], links: list[TraceLink], options: ExportOptions | None = None) -> bytes` - export complete audit trail
    - `def get_file_extension() -> str` - return appropriate file extension for format
    - `def get_content_type() -> str` - return MIME content type for HTTP responses

- [x] Task 3: Implement JSON exporter (AC: #1, #2, #3, #4)
  - [x] 3.1 Create `src/yolo_developer/audit/json_exporter.py`
  - [x] 3.2 Implement `JsonAuditExporter` class implementing `AuditExporter` protocol
  - [x] 3.3 Use standard library `json` module with:
    - `json.dumps()` with `indent=2` for readable output
    - Consistent key ordering via `sort_keys=True`
    - UTF-8 encoding for bytes output
  - [x] 3.4 Implement structure:
    - decisions array with full Decision.to_dict() data
    - artifacts array with TraceableArtifact.to_dict() data
    - links array with TraceLink.to_dict() data
    - metadata object with export_timestamp, total_counts
  - [x] 3.5 Implement redaction logic:
    - Remove/replace specified fields based on RedactionConfig
    - Replace sensitive values with "[REDACTED]" placeholder

- [x] Task 4: Implement CSV exporter (AC: #1, #2, #3, #4)
  - [x] 4.1 Create `src/yolo_developer/audit/csv_exporter.py`
  - [x] 4.2 Implement `CsvAuditExporter` class implementing `AuditExporter` protocol
  - [x] 4.3 Use standard library `csv` module with:
    - `csv.DictWriter` for structured output
    - Consistent column ordering
    - UTF-8 BOM for Excel compatibility
  - [x] 4.4 Implement CSV structure:
    - Flatten nested objects (agent.agent_name -> agent_name, context.sprint_id -> sprint_id)
    - Separate sheets/sections for decisions, artifacts, links (use markers or multiple CSV output)
  - [x] 4.5 Implement redaction logic matching JSON exporter

- [x] Task 5: Implement PDF exporter (AC: #1, #2, #3, #4)
  - [x] 5.1 Create `src/yolo_developer/audit/pdf_exporter.py`
  - [x] 5.2 Implement `PdfAuditExporter` class implementing `AuditExporter` protocol
  - [x] 5.3 Use `reportlab` library for PDF generation:
    - Title page with export metadata
    - Table of contents
    - Decision section with formatted entries
    - Traceability section with artifact/link tables
    - Coverage summary section
  - [x] 5.4 Implement styling:
    - Color-coded severity (similar to Rich formatter)
    - Tables for structured data
    - Page numbers and headers
  - [x] 5.5 Implement redaction logic matching other exporters
  - [x] 5.6 Add reportlab to dependencies if not present

- [x] Task 6: Create export service (AC: #1, #2, #3, #4)
  - [x] 6.1 Create `src/yolo_developer/audit/export.py`
  - [x] 6.2 Implement `AuditExportService` class:
    - Constructor takes `DecisionStore`, `TraceabilityStore`, and optional exporter map
    - `async def export(format: ExportFormat, filters: DecisionFilters | None = None, options: ExportOptions | None = None) -> bytes`
    - `async def export_to_file(path: str, format: ExportFormat | None = None, filters: DecisionFilters | None = None, options: ExportOptions | None = None) -> str` - returns file path
    - `def get_supported_formats() -> list[ExportFormat]`
  - [x] 6.3 Implement format detection from file extension when format not specified
  - [x] 6.4 Add structured logging with structlog for export operations
  - [x] 6.5 Implement `get_audit_export_service(decision_store, traceability_store, exporters) -> AuditExportService` factory function

- [x] Task 7: Update module exports (AC: all)
  - [x] 7.1 Update `src/yolo_developer/audit/__init__.py`
  - [x] 7.2 Export new public types: ExportFormat, RedactionConfig, ExportOptions
  - [x] 7.3 Export exporter protocol: AuditExporter
  - [x] 7.4 Export exporter implementations: JsonAuditExporter, CsvAuditExporter, PdfAuditExporter
  - [x] 7.5 Export service: AuditExportService, get_audit_export_service
  - [x] 7.6 Export constants: VALID_EXPORT_FORMATS, DEFAULT_REDACTION_CONFIG, DEFAULT_EXPORT_OPTIONS
  - [x] 7.7 Update module docstring documenting FR84 implementation

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1 Create `tests/unit/audit/test_export_types.py`:
    - Test ExportOptions validation (valid/invalid format values)
    - Test RedactionConfig validation
    - Test `to_dict()` produces JSON-serializable output
    - Test frozen dataclass immutability
    - Test DEFAULT_REDACTION_CONFIG and DEFAULT_EXPORT_OPTIONS constants
  - [x] 8.2 Create `tests/unit/audit/test_export_protocol.py`:
    - Test protocol definition
  - [x] 8.3 Create `tests/unit/audit/test_json_exporter.py`:
    - Test `export_decisions` produces valid JSON
    - Test `export_traces` produces valid JSON
    - Test `export_full_audit` produces complete structure
    - Test `get_file_extension()` returns ".json"
    - Test `get_content_type()` returns "application/json"
    - Test redaction removes specified fields
    - Test redaction replaces sensitive values
    - Test empty input produces valid empty export
  - [x] 8.4 Create `tests/unit/audit/test_csv_exporter.py`:
    - Test `export_decisions` produces valid CSV
    - Test `export_traces` produces valid CSV
    - Test `export_full_audit` produces complete structure
    - Test `get_file_extension()` returns ".csv"
    - Test `get_content_type()` returns "text/csv"
    - Test redaction works correctly
    - Test column ordering is consistent
    - Test nested fields are flattened correctly
  - [x] 8.5 Create `tests/unit/audit/test_pdf_exporter.py`:
    - Test `export_decisions` produces valid PDF bytes
    - Test `export_traces` produces valid PDF bytes
    - Test `export_full_audit` produces complete PDF
    - Test `get_file_extension()` returns ".pdf"
    - Test `get_content_type()` returns "application/pdf"
    - Test redaction works correctly
    - Test PDF contains expected sections
  - [x] 8.6 Create `tests/unit/audit/test_export.py`:
    - Test `export()` retrieves data and exports in requested format
    - Test `export_to_file()` writes file and returns path
    - Test format detection from file extension
    - Test with filters applied
    - Test with different export options
    - Test `get_supported_formats()` returns all formats
    - Test `get_audit_export_service()` factory function
  - [x] 8.7 Update `tests/unit/audit/test_init.py`:
    - Add tests for new exports

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `export_types.py` (frozen dataclasses per ADR-001)
- **Protocol Pattern**: Use Protocol for AuditExporter to allow future export format implementations
- **Logging**: Use `structlog.get_logger(__name__)` pattern per architecture
- **State**: Frozen dataclasses for internal types
- **Error Handling**: Per ADR-007 - log errors, don't block callers

### Key Design Decisions

1. **Three Export Formats**: JSON, CSV, PDF cover most compliance requirements
   - JSON: Machine-readable, API integration, full fidelity
   - CSV: Spreadsheet analysis, simple imports
   - PDF: Human-readable reports, formal compliance documentation

2. **Exporter Protocol**: Allows adding new formats (XML, HTML, etc.) without changing service

3. **Redaction Support**: Critical for compliance exports
   - Configurable field redaction (metadata, session_ids, custom fields)
   - Replace sensitive values with "[REDACTED]" marker
   - Consistent redaction across all export formats

4. **Export Options**: Control what data is included
   - include_decisions: Include decision log
   - include_traces: Include traceability data
   - include_coverage: Include coverage statistics

5. **File Extension Detection**: Service auto-detects format from file path when not specified

6. **Integration with Existing Types**: Builds on Stories 11.1, 11.2, 11.3

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
├── formatter_types.py           # Formatter types (existing from 11.3)
├── formatter_protocol.py        # AuditFormatter Protocol (existing from 11.3)
├── rich_formatter.py            # RichAuditFormatter (existing from 11.3)
├── plain_formatter.py           # PlainAuditFormatter (existing from 11.3)
├── view.py                      # AuditViewService (existing from 11.3)
├── export_types.py              # Export type definitions (NEW)
├── export_protocol.py           # AuditExporter Protocol (NEW)
├── json_exporter.py             # JsonAuditExporter implementation (NEW)
├── csv_exporter.py              # CsvAuditExporter implementation (NEW)
├── pdf_exporter.py              # PdfAuditExporter implementation (NEW)
└── export.py                    # AuditExportService class (NEW)
```

Test location: `tests/unit/audit/`
```
tests/unit/audit/
├── __init__.py                  # Package init (existing)
├── test_types.py                # Type tests (existing)
├── test_memory_store.py         # Store tests (existing)
├── test_logger.py               # Logger tests (existing)
├── test_traceability_*.py       # Traceability tests (existing)
├── test_formatter_*.py          # Formatter tests (existing from 11.3)
├── test_view_service.py         # View service tests (existing from 11.3)
├── test_export_types.py         # Export type tests (NEW)
├── test_export_protocol.py      # Protocol tests (NEW)
├── test_json_exporter.py        # JSON exporter tests (NEW)
├── test_csv_exporter.py         # CSV exporter tests (NEW)
├── test_pdf_exporter.py         # PDF exporter tests (NEW)
├── test_export_service.py       # Export service tests (NEW)
└── test_init.py                 # Module export tests (update)
```

### Previous Story Intelligence (11.1, 11.2, 11.3)

Stories 11.1, 11.2, and 11.3 established the following patterns that MUST be followed:

1. **Frozen Dataclasses**: All types use `@dataclass(frozen=True)` with:
   - `__post_init__` for validation with warning logging
   - `to_dict()` for JSON serialization

2. **Protocol Pattern**: `DecisionStore`, `TraceabilityStore`, `AuditFormatter` Protocols enable pluggable backends

3. **Thread Safety**: `InMemoryDecisionStore` and `InMemoryTraceabilityStore` use `threading.Lock`

4. **Structured Logging**: Uses `structlog.get_logger(__name__)`

5. **Factory Function**: `get_logger(store)`, `get_traceability_service(store)`, `get_audit_view_service()` pattern for dependency injection

6. **Error Handling**: Per ADR-007 - log errors, don't block callers

7. **Story 11.3 Key Learnings**:
   - Protocol methods should include optional `options` parameter for consistency
   - Coverage calculation needs to properly count requirements and covered items
   - Tests should verify options parameter acceptance

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

    def to_dict(self) -> dict[str, Any]: ...
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

    def to_dict(self) -> dict[str, Any]: ...

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

    def to_dict(self) -> dict[str, Any]: ...
```

From Story 11.1 (`store.py`):
```python
@dataclass(frozen=True)
class DecisionFilters:
    agent_types: tuple[str, ...] | None = None
    decision_types: tuple[str, ...] | None = None
    severities: tuple[str, ...] | None = None
    start_time: str | None = None
    end_time: str | None = None
    sprint_id: str | None = None
    story_id: str | None = None
```

### Integration Points

This story builds on Stories 11.1, 11.2, 11.3 and enables future stories:
- Story 11.1 (Decision Logging): Uses Decision, DecisionStore, DecisionFilters
- Story 11.2 (Requirement Traceability): Uses TraceableArtifact, TraceLink, TraceabilityStore
- Story 11.3 (Human-Readable View): Similar formatter protocol pattern
- Story 11.5 (Cross-Agent Correlation): Will export correlation data
- Story 11.7 (Audit Filtering): Will use same DecisionFilters

### Example Usage

```python
from yolo_developer.audit import (
    AuditExportService,
    JsonAuditExporter,
    CsvAuditExporter,
    PdfAuditExporter,
    ExportOptions,
    RedactionConfig,
    DecisionFilters,
    InMemoryDecisionStore,
    InMemoryTraceabilityStore,
    get_audit_export_service,
)

# Create stores
decision_store = InMemoryDecisionStore()
traceability_store = InMemoryTraceabilityStore()

# Create export service
export_service = get_audit_export_service(
    decision_store=decision_store,
    traceability_store=traceability_store,
)

# Export to JSON (full audit)
json_bytes = await export_service.export("json")
with open("audit.json", "wb") as f:
    f.write(json_bytes)

# Export to CSV with filters
filters = DecisionFilters(
    agent_types=("analyst", "pm"),
    severities=("critical", "high"),
)
csv_bytes = await export_service.export("csv", filters=filters)

# Export to PDF with redaction
redaction = RedactionConfig(
    redact_metadata=True,
    redact_session_ids=True,
)
options = ExportOptions(
    format="pdf",
    include_decisions=True,
    include_traces=True,
    include_coverage=True,
    redaction_config=redaction,
)
await export_service.export_to_file("compliance_report.pdf", options=options)

# Auto-detect format from extension
await export_service.export_to_file("audit_trail.csv")  # Uses CSV exporter
await export_service.export_to_file("report.json")      # Uses JSON exporter
```

### Technical Constraints

1. **Async/Await**: All I/O operations must be async per ADR patterns
2. **Type Hints**: Full type annotations required (mypy strict mode)
3. **Import Order**: Standard library -> Third-party -> Local (per architecture)
4. **snake_case**: All field names use snake_case
5. **Test Coverage**: Target 100% coverage matching Stories 11.1, 11.2, 11.3
6. **PDF Library**: Use reportlab (add to dependencies if needed)
7. **Bytes Output**: All exporters return bytes for consistent handling
8. **UTF-8 Encoding**: All text exports use UTF-8 encoding

### Dependency Note

Check if `reportlab` is in `pyproject.toml`. If not, add it:
```bash
uv add reportlab
```

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR84: System can export audit trail for compliance reporting
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/planning-artifacts/architecture.md] structlog for structured logging
- [Source: _bmad-output/planning-artifacts/epics.md#Story-11.4] Story definition and acceptance criteria
- [Source: _bmad-output/implementation-artifacts/11-1-decision-logging.md] Decision types and store patterns
- [Source: _bmad-output/implementation-artifacts/11-2-requirement-traceability.md] Traceability types and store patterns
- [Source: _bmad-output/implementation-artifacts/11-3-human-readable-audit-view.md] Formatter protocol pattern and code review learnings

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- All 8 tasks completed successfully using TDD (red-green-refactor) approach
- 394 tests in the audit module pass (including 132 new tests for this story)
- All code passes mypy strict mode and ruff checks
- reportlab dependency added for PDF export
- Module exports updated with full documentation

### File List

**New Source Files:**
- `src/yolo_developer/audit/export_types.py` - Export type definitions (ExportFormat, RedactionConfig, ExportOptions)
- `src/yolo_developer/audit/export_protocol.py` - AuditExporter Protocol definition
- `src/yolo_developer/audit/json_exporter.py` - JsonAuditExporter implementation
- `src/yolo_developer/audit/csv_exporter.py` - CsvAuditExporter implementation
- `src/yolo_developer/audit/pdf_exporter.py` - PdfAuditExporter implementation
- `src/yolo_developer/audit/export.py` - AuditExportService and factory function

**Modified Source Files:**
- `src/yolo_developer/audit/__init__.py` - Added all new exports and updated documentation

**Modified Config Files:**
- `pyproject.toml` - Added reportlab dependency for PDF export
- `uv.lock` - Updated lockfile with new dependency

**Sprint Tracking (auto-updated):**
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Story status updated to done

**New Test Files:**
- `tests/unit/audit/test_export_types.py` - 19 tests
- `tests/unit/audit/test_export_protocol.py` - 9 tests
- `tests/unit/audit/test_json_exporter.py` - 23 tests
- `tests/unit/audit/test_csv_exporter.py` - 21 tests
- `tests/unit/audit/test_pdf_exporter.py` - 19 tests
- `tests/unit/audit/test_export.py` - 20 tests

**Modified Test Files:**
- `tests/unit/audit/test_init.py` - Added 13 tests for new exports

### Code Review Fixes Applied (2026-01-18)

**Issue #1 (MEDIUM): PDF exporter `_create_artifacts_table` unused redaction parameter**
- Added Metadata column to artifacts table with redaction support
- File: `src/yolo_developer/audit/pdf_exporter.py:370-422`

**Issue #2 (MEDIUM): PDF exporter `_create_links_table` unused redaction parameter**
- Added Metadata column to links table with redaction support
- File: `src/yolo_developer/audit/pdf_exporter.py:424-470`

**Issue #3 (MEDIUM): Duplicated test helper functions**
- Created `tests/unit/audit/conftest.py` with shared fixtures:
  - `create_test_decision()`, `create_test_artifact()`, `create_test_link()`
- Updated 4 test files to use shared fixtures

**Issue #4 (LOW): Improved PDF redaction test coverage**
- Added `TestPdfRedactionContent` class with 4 comprehensive tests
- File: `tests/unit/audit/test_pdf_exporter.py:259-335`

**Post-Review Files:**
- `tests/unit/audit/conftest.py` (NEW) - Shared test fixtures
- `tests/unit/audit/test_pdf_exporter.py` (MODIFIED) - 4 new redaction tests
- `tests/unit/audit/test_json_exporter.py` (MODIFIED) - Use shared fixtures
- `tests/unit/audit/test_csv_exporter.py` (MODIFIED) - Use shared fixtures
- `tests/unit/audit/test_export.py` (MODIFIED) - Use shared fixtures
- `src/yolo_developer/audit/pdf_exporter.py` (MODIFIED) - Metadata columns with redaction

