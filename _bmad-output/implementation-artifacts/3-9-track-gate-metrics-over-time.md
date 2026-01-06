# Story 3.9: Track Gate Metrics Over Time

Status: complete

## Story

As a developer,
I want to see gate pass/fail metrics over time,
So that I can identify quality trends in my projects.

## Acceptance Criteria

1. **AC1: Pass/Fail Rates by Gate Type**
   - **Given** gates have been evaluated over multiple sprints
   - **When** I query gate metrics
   - **Then** pass/fail rates by gate type are available
   - **And** rates are calculated as percentages
   - **And** total evaluation counts are included

2. **AC2: Trends Over Time**
   - **Given** historical gate evaluation data exists
   - **When** I query trends
   - **Then** trends over time are calculable
   - **And** time-based aggregation is supported (daily, weekly, sprint)
   - **And** trend direction (improving/declining) is identifiable

3. **AC3: Per-Agent Breakdown**
   - **Given** gates are associated with agent boundaries
   - **When** I query metrics with agent filter
   - **Then** per-agent breakdown is available
   - **And** I can see which agents have the most gate failures
   - **And** agent-specific trends are calculable

4. **AC4: Persistent Storage**
   - **Given** gate evaluations occur
   - **When** metrics are recorded
   - **Then** metrics are stored persistently
   - **And** metrics survive application restarts
   - **And** metrics are queryable across sessions

5. **AC5: Structured Metrics Data**
   - **Given** gate metrics are collected
   - **When** metrics are accessed
   - **Then** data is available in structured format (dict/JSON)
   - **And** metrics integrate with existing audit trail system
   - **And** metrics support structured logging output

## Tasks / Subtasks

- [x] Task 1: Create Metrics Data Models (AC: 1, 2, 5)
  - [x] Create `src/yolo_developer/gates/metrics_types.py` module
  - [x] Define `GateMetricRecord` dataclass with fields: `gate_name`, `passed`, `score`, `threshold`, `timestamp`, `agent_name`, `sprint_id`
  - [x] Define `GateMetricsSummary` dataclass for aggregated metrics
  - [x] Define `GateTrend` dataclass with fields: `gate_name`, `period`, `pass_rate`, `avg_score`, `evaluation_count`, `direction`
  - [x] Add `to_dict()` methods for structured logging compatibility
  - [x] Use frozen dataclasses for immutability

- [x] Task 2: Create Metrics Storage Protocol (AC: 4)
  - [x] Create `src/yolo_developer/gates/metrics_store.py` module
  - [x] Define `GateMetricsStore` protocol with methods: `record_evaluation()`, `get_metrics()`, `get_trends()`, `get_agent_breakdown()`
  - [x] Define async interface for all storage operations
  - [x] Support filtering by gate_name, agent_name, time_range, sprint_id

- [x] Task 3: Implement JSON File Storage Backend (AC: 4)
  - [x] Implement `JsonGateMetricsStore` class implementing the protocol
  - [x] Store metrics in `.yolo/metrics/gate_metrics.json`
  - [x] Support atomic writes with file locking for concurrent access
  - [x] Implement efficient append-only storage pattern
  - [x] Add cleanup/rotation for old metrics (configurable retention)

- [x] Task 4: Implement Metrics Recording Integration (AC: 1, 4, 5)
  - [x] Update `decorator.py` to record metrics after gate evaluation
  - [x] Extract agent name from state or context
  - [x] Extract sprint_id from state if available
  - [x] Use structlog to log metric recording events
  - [x] Ensure recording is non-blocking (async)

- [x] Task 5: Implement Pass/Fail Rate Calculations (AC: 1)
  - [x] Create `src/yolo_developer/gates/metrics_calculator.py` module
  - [x] Implement `calculate_pass_rates(records: list[GateMetricRecord]) -> dict[str, float]`
  - [x] Implement `calculate_gate_summary(gate_name: str, records: list) -> GateMetricsSummary`
  - [x] Include total evaluations, pass count, fail count, pass rate percentage
  - [x] Support filtering by time range

- [x] Task 6: Implement Trend Analysis (AC: 2)
  - [x] Implement `calculate_trends(records: list, period: str) -> list[GateTrend]`
  - [x] Support period options: "daily", "weekly", "sprint"
  - [x] Calculate trend direction: "improving", "stable", "declining"
  - [x] Use rolling window comparison for direction detection
  - [x] Handle edge cases (insufficient data, single data point)

- [x] Task 7: Implement Agent Breakdown (AC: 3)
  - [x] Implement `get_agent_breakdown(records: list) -> dict[str, GateMetricsSummary]`
  - [x] Group metrics by agent_name
  - [x] Calculate per-agent pass rates and trends
  - [x] Identify agents with highest failure rates
  - [x] Support ranking by failure count or rate

- [x] Task 8: Create Metrics Query API (AC: 1, 2, 3, 5)
  - [x] API implemented via metrics_calculator.py functions
  - [x] `calculate_pass_rates()` provides pass rate queries
  - [x] `calculate_trends()` provides trend queries
  - [x] `get_agent_breakdown()` provides agent summary
  - [x] All functions work with async store.get_metrics()

- [x] Task 9: Integrate with Audit Trail (AC: 5)
  - [x] Metrics events logged via structlog in decorator.py
  - [x] GateMetricRecord.to_dict() supports structured logging
  - [x] Timestamps use ISO 8601 format via datetime.isoformat()
  - [x] Error logging for failed metric recording

- [x] Task 10: Write Unit Tests (AC: all)
  - [x] Created `tests/unit/gates/test_metrics_types.py` (26 tests)
  - [x] Created `tests/unit/gates/test_metrics_store.py` (14 tests)
  - [x] Created `tests/unit/gates/test_metrics_calculator.py` (38 tests)
  - [x] Test metric recording and retrieval
  - [x] Test pass rate calculations with various scenarios
  - [x] Test trend calculations with mock time series data
  - [x] Test agent breakdown grouping

- [x] Task 11: Write Integration Tests (AC: all)
  - [x] Created `tests/integration/test_gate_metrics_tracking.py`
  - [x] Test end-to-end metric recording via decorator
  - [x] Test persistence across simulated restarts
  - [x] Test concurrent metric recording
  - [x] Test metrics query with real gate evaluations

- [x] Task 12: Update Exports and Documentation (AC: 5)
  - [x] Export metrics types from `gates/__init__.py`
  - [x] Export metrics API functions from `gates/__init__.py`
  - [x] Update module docstring with metrics usage examples
  - [x] Add inline documentation for metrics patterns

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Metrics extend the gate infrastructure without modifying core evaluation logic
- **ADR-002 (Memory Persistence):** JSON file storage follows the ChromaDB embedded pattern for local persistence
- **FR27:** System can track quality gate pass/fail metrics over time
- **FR81-88 (Audit Trail):** Metrics integrate with structured logging and audit trail

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses for `GateMetricRecord`, `GateMetricsSummary`, `GateTrend`
- **Async Operations:** All storage and query operations must be async
- **Structured Logging:** All metric events logged via structlog
- **Non-Blocking:** Metric recording should not slow down gate evaluation
- **Thread Safety:** JSON storage must handle concurrent writes safely

### Existing Pattern References

**From Story 3.8 (Report Types Pattern):**
```python
@dataclass(frozen=True)
class GateIssue:
    location: str
    issue_type: str
    description: str
    severity: Severity
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity.value,
            "context": self.context,
        }
```

**From decorator.py (Gate Result Recording):**
```python
# Record result in state for audit trail
working_state["gate_results"].append(result.to_dict())
```

### Proposed Metrics Data Models

```python
# src/yolo_developer/gates/metrics_types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrendDirection(Enum):
    """Direction of quality trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass(frozen=True)
class GateMetricRecord:
    """Single gate evaluation metric record.

    Attributes:
        gate_name: Name of the evaluated gate
        passed: Whether the gate passed
        score: Numeric score achieved (0.0-1.0)
        threshold: Required threshold for passing
        timestamp: When the evaluation occurred
        agent_name: Agent that triggered the gate (optional)
        sprint_id: Sprint identifier (optional)
    """
    gate_name: str
    passed: bool
    score: float
    threshold: float
    timestamp: datetime
    agent_name: str | None = None
    sprint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "sprint_id": self.sprint_id,
        }


@dataclass(frozen=True)
class GateMetricsSummary:
    """Aggregated metrics summary for a gate or agent.

    Attributes:
        gate_name: Gate name (or "all" for aggregate)
        total_evaluations: Total number of evaluations
        pass_count: Number of passes
        fail_count: Number of failures
        pass_rate: Pass rate as percentage (0.0-100.0)
        avg_score: Average score across evaluations
        period_start: Start of measurement period
        period_end: End of measurement period
    """
    gate_name: str
    total_evaluations: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_score: float
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "total_evaluations": self.total_evaluations,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


@dataclass(frozen=True)
class GateTrend:
    """Trend analysis for a gate over time.

    Attributes:
        gate_name: Name of the gate
        period: Time period type (daily, weekly, sprint)
        pass_rate: Pass rate for the period
        avg_score: Average score for the period
        evaluation_count: Number of evaluations in period
        direction: Trend direction compared to previous period
        period_label: Human-readable period label (e.g., "2026-01-05")
    """
    gate_name: str
    period: str
    pass_rate: float
    avg_score: float
    evaluation_count: int
    direction: TrendDirection
    period_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "period": self.period,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "evaluation_count": self.evaluation_count,
            "direction": self.direction.value,
            "period_label": self.period_label,
        }
```

### Proposed Storage Protocol

```python
# src/yolo_developer/gates/metrics_store.py

from typing import Protocol
from datetime import datetime
from yolo_developer.gates.metrics_types import GateMetricRecord, GateMetricsSummary, GateTrend


class GateMetricsStore(Protocol):
    """Protocol for gate metrics storage backends."""

    async def record_evaluation(self, record: GateMetricRecord) -> None:
        """Record a gate evaluation metric."""
        ...

    async def get_metrics(
        self,
        gate_name: str | None = None,
        agent_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        sprint_id: str | None = None,
    ) -> list[GateMetricRecord]:
        """Query metrics with optional filters."""
        ...

    async def get_summary(
        self,
        gate_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> GateMetricsSummary:
        """Get aggregated summary for a gate."""
        ...

    async def get_agent_breakdown(self) -> dict[str, GateMetricsSummary]:
        """Get metrics breakdown by agent."""
        ...
```

### Decorator Integration Point

```python
# In decorator.py wrapper function, after gate evaluation:

# Record metric for tracking (non-blocking)
if metrics_store is not None:
    metric_record = GateMetricRecord(
        gate_name=gate_name,
        passed=result.passed,
        score=result.score if result.score is not None else (1.0 if result.passed else 0.0),
        threshold=context.threshold if hasattr(context, 'threshold') else 0.0,
        timestamp=datetime.now(UTC),
        agent_name=working_state.get("current_agent"),
        sprint_id=working_state.get("sprint_id"),
    )
    # Fire and forget - don't block on metric recording
    asyncio.create_task(metrics_store.record_evaluation(metric_record))
```

### File Structure

```
src/yolo_developer/gates/
├── __init__.py              # UPDATE: Export metrics types and API
├── metrics_types.py         # NEW: GateMetricRecord, GateMetricsSummary, GateTrend
├── metrics_store.py         # NEW: GateMetricsStore protocol, JsonGateMetricsStore
├── metrics_calculator.py    # NEW: Pass rate, trend calculations
├── metrics_api.py           # NEW: High-level metrics query API
├── decorator.py             # UPDATE: Add metric recording integration
├── report_types.py          # No changes
├── report_generator.py      # No changes
├── remediation.py           # No changes
├── evaluators.py            # No changes
├── threshold_resolver.py    # No changes
├── types.py                 # No changes
└── gates/
    └── ...                  # No changes to individual gates
```

### Previous Story Intelligence (from Story 3.8)

**Patterns to Apply:**
1. Use frozen dataclasses for all data types (immutable)
2. All operations async with proper error handling
3. Use structlog for all logging operations
4. Export new types and functions from `__init__.py`
5. Include `to_dict()` methods for JSON serialization
6. Use tuple for immutable collections in frozen dataclasses
7. Validate input types before processing
8. All thresholds use 0.0-1.0 decimal format

**Key Files to Reference:**
- `src/yolo_developer/gates/report_types.py` - Frozen dataclass patterns
- `src/yolo_developer/gates/decorator.py` - Gate evaluation lifecycle
- `src/yolo_developer/gates/types.py` - GateResult structure
- `src/yolo_developer/memory/` - Storage patterns (ChromaDB, JSON graph)
- `tests/integration/test_gate_failure_reports.py` - Integration test patterns

**Code Review Learnings from Story 3.8:**
- Export new types from `__init__.py`
- Remove unused imports
- Use `()` literal instead of `tuple()` for empty tuples
- Rename test imports to avoid pytest collection issues (use `eval_*` prefix)

### Web Research Findings (2026-01-05)

**OpenTelemetry Metrics Best Practices:**
- Use Counter for pass/fail counts (monotonic increasing)
- Use Gauge for current pass rate percentages
- Use Histogram for score distributions
- Support OTLP export for external observability platforms

**ChromaDB Metadata Storage:**
- Metadata supports string, int, float, boolean values
- Can store metrics alongside embeddings for correlation
- Supports metadata filtering for efficient queries

**Structlog Integration:**
- Enrich logs with metric data for correlation
- Use processor pipeline for metric extraction
- Support both human-readable and JSON output formats

### Testing Standards

- Use pytest with pytest-asyncio for async tests
- Create fixtures for metric record generation
- Test storage persistence with temp directories
- Test concurrent writes with asyncio.gather
- Mock datetime for deterministic trend tests
- Verify structured logging output contains metric data
- Test edge cases: empty data, single record, large datasets

### Implementation Approach

1. **Data Models First:** Create metrics_types.py with all dataclasses
2. **Storage Layer:** Implement JSON file storage with protocol
3. **Calculator Module:** Implement rate and trend calculations
4. **API Layer:** Create high-level query interface
5. **Decorator Integration:** Add metric recording to gate decorator
6. **Audit Integration:** Ensure metrics flow to audit trail
7. **Testing:** Unit tests per module, then integration tests

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: architecture.md#ADR-002] - Memory Persistence Strategy
- [Source: prd.md#FR27] - Track quality gate pass/fail metrics over time
- [Source: prd.md#FR81-88] - Audit Trail & Observability requirements
- [Source: epics.md#Story-3.9] - Track Gate Metrics Over Time requirements
- [Story 3.8 Implementation] - Report types and generator patterns
- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/languages/python/) - Metrics collection patterns
- [ChromaDB Cookbook](https://cookbook.chromadb.dev/core/collections/) - Metadata storage patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 466 unit/integration tests for gates module pass
- Ruff check and format pass with no errors
- Mypy type checking passes

### Completion Notes List

1. **Task 1-3**: Created metrics data models (GateMetricRecord, GateMetricsSummary, GateTrend, TrendDirection) and JsonGateMetricsStore with atomic writes and asyncio.Lock for concurrency
2. **Task 4**: Integrated metrics recording in decorator.py using asyncio.create_task() with proper task reference management (RUF006 compliance)
3. **Task 5-7**: Implemented calculator functions for pass rates, trends (daily/weekly/sprint with 5% significance threshold), and agent breakdown
4. **Task 8-9**: API functionality delivered via calculator functions; audit trail integration via structlog and to_dict() methods
5. **Task 10-11**: Created comprehensive unit tests (78 new tests) and integration tests (8 tests) covering all acceptance criteria
6. **Task 12**: Exported all metrics types and functions from gates/__init__.py with updated docstrings

### File List

**New Files:**
- `src/yolo_developer/gates/metrics_types.py` - GateMetricRecord, GateMetricsSummary, GateTrend, TrendDirection
- `src/yolo_developer/gates/metrics_store.py` - GateMetricsStore protocol, JsonGateMetricsStore implementation
- `src/yolo_developer/gates/metrics_calculator.py` - calculate_pass_rates, calculate_gate_summary, calculate_trends, get_agent_breakdown, filter_records_by_time_range
- `tests/unit/gates/test_metrics_types.py` - 26 unit tests
- `tests/unit/gates/test_metrics_store.py` - 14 unit tests
- `tests/unit/gates/test_metrics_calculator.py` - 38 unit tests
- `tests/integration/test_gate_metrics_tracking.py` - 8 integration tests

**Modified Files:**
- `src/yolo_developer/gates/decorator.py` - Added set_metrics_store(), get_metrics_store(), _record_metric(), _async_record_metric()
- `src/yolo_developer/gates/__init__.py` - Added exports for all metrics types and functions
