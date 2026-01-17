# Story 10.16: Health Telemetry Dashboard Data

Status: done

## Story

As a developer,
I want health telemetry data for dashboard display,
So that system health is always visible.

## Acceptance Criteria

1. **Given** health metrics being collected
   **When** telemetry is queried
   **Then** burn-down velocity is available

2. **Given** telemetry data structures exist
   **When** dashboard requests metrics
   **Then** cycle time is available

3. **Given** churn rate tracking is active
   **When** telemetry is queried
   **Then** churn rate is available

4. **Given** agent activity is tracked
   **When** telemetry is queried
   **Then** agent idle time is available

5. **Given** telemetry data exists
   **When** data is retrieved
   **Then** data is formatted for display:
   - Human-readable summaries
   - Serializable to JSON via `to_dict()` methods
   - Timestamps in ISO format

## Tasks / Subtasks

- [x] Task 1: Create type definitions (AC: #1, #2, #3, #4, #5)
  - [x] 1.1 Create `telemetry_types.py` with:
    - `TelemetrySnapshot` frozen dataclass aggregating all dashboard metrics
    - `TelemetryConfig` frozen dataclass for telemetry collection settings
    - `DashboardMetrics` frozen dataclass for display-optimized metrics
    - `MetricSummary` frozen dataclass for individual metric summaries
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging
  - [x] 1.3 Add `to_dict()` methods for JSON serialization
  - [x] 1.4 Export constants: `DEFAULT_TELEMETRY_INTERVAL_SECONDS`, `VALID_METRIC_CATEGORIES`

- [x] Task 2: Implement telemetry collection (AC: #1, #2, #3, #4)
  - [x] 2.1 Create `telemetry.py` module
  - [x] 2.2 Implement `collect_telemetry(state: YoloState, health_status: HealthStatus | None, velocity_metrics: VelocityMetrics | None) -> TelemetrySnapshot`
  - [x] 2.3 Implement `_aggregate_burn_down_velocity(velocity: VelocityMetrics | None) -> MetricSummary`
  - [x] 2.4 Implement `_aggregate_cycle_time(health: HealthStatus | None, velocity: VelocityMetrics | None) -> MetricSummary`
  - [x] 2.5 Implement `_aggregate_churn_rate(health: HealthStatus | None) -> MetricSummary`
  - [x] 2.6 Implement `_aggregate_idle_times(health: HealthStatus | None) -> dict[str, MetricSummary]`

- [x] Task 3: Implement dashboard data formatting (AC: #5)
  - [x] 3.1 Implement `format_for_dashboard(snapshot: TelemetrySnapshot) -> DashboardMetrics`
  - [x] 3.2 Implement `_format_velocity_display(velocity: MetricSummary) -> str` - human-readable velocity string
  - [x] 3.3 Implement `_format_cycle_time_display(cycle_time: MetricSummary) -> str` - human-readable cycle time
  - [x] 3.4 Implement `_format_health_summary(snapshot: TelemetrySnapshot) -> str` - overall health sentence
  - [x] 3.5 Implement `_format_agent_status_table(idle_times: dict[str, MetricSummary]) -> list[dict[str, str]]` - tabular agent data

- [x] Task 4: Implement main telemetry function (AC: #1, #2, #3, #4, #5)
  - [x] 4.1 Implement `async def get_dashboard_telemetry(state: YoloState, config: TelemetryConfig | None = None) -> DashboardMetrics`
  - [x] 4.2 Orchestrate: collect health → collect velocity → aggregate telemetry → format for dashboard
  - [x] 4.3 Add structured logging with structlog for telemetry events
  - [x] 4.4 Ensure proper error handling - return safe defaults on failure

- [x] Task 5: Integrate with SM node (AC: all)
  - [x] 5.1 Update `sm_node` in `node.py` to optionally collect telemetry each cycle
  - [x] 5.2 Add `telemetry_snapshot` field to SMOutput in `types.py`
  - [x] 5.3 Update state with telemetry when collected
  - [x] 5.4 Add telemetry logging to SM node output

- [x] Task 6: Update __init__.py exports (AC: all)
  - [x] 6.1 Add imports from `telemetry` and `telemetry_types`
  - [x] 6.2 Update `__all__` list with new exports (alphabetically sorted)
  - [x] 6.3 Update module docstring with telemetry references

- [x] Task 7: Write comprehensive tests (AC: all)
  - [x] 7.1 Test type validation in `telemetry_types.py` (valid/invalid values, edge cases)
  - [x] 7.2 Test `collect_telemetry` with various input combinations
  - [x] 7.3 Test `collect_telemetry` with None health/velocity (graceful handling)
  - [x] 7.4 Test `format_for_dashboard` produces human-readable output
  - [x] 7.5 Test `get_dashboard_telemetry` full flow
  - [x] 7.6 Test `to_dict()` methods produce valid JSON-serializable output
  - [x] 7.7 Test SM node integration with telemetry collection
  - [x] 7.8 Test logging output with structlog capture

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `telemetry_types.py` (frozen dataclasses per ADR-001)
- **Implementation**: Create `telemetry.py` (async functions per ADR-005)
- **Logging**: Use `structlog.get_logger(__name__)` pattern
- **State**: YoloState TypedDict for graph state, frozen dataclasses for internal
- **Error Handling**: Per ADR-007 - return safe defaults, never block main workflow

### Integration Points

From existing code analysis:
- `health.py:monitor_health()` - returns HealthStatus with HealthMetrics
- `health_types.py:HealthStatus` - contains metrics, alerts, and severity
- `health_types.py:HealthMetrics` - agent_idle_times, agent_cycle_times, agent_churn_rates
- `velocity.py:calculate_velocity_metrics()` - returns VelocityMetrics
- `velocity_types.py:VelocityMetrics` - average_stories_per_sprint, trend, cycle times
- `node.py:sm_node()` - integrate telemetry collection

### Key Types from Related Modules

```python
# From health_types.py
@dataclass(frozen=True)
class HealthMetrics:
    agent_idle_times: dict[str, float]
    agent_cycle_times: dict[str, float]
    agent_churn_rates: dict[str, float]
    overall_cycle_time: float
    overall_churn_rate: float
    cycle_time_percentiles: dict[str, float]  # p50, p90, p95

@dataclass(frozen=True)
class HealthStatus:
    status: HealthSeverity  # healthy, warning, degraded, critical
    metrics: HealthMetrics
    alerts: tuple[HealthAlert, ...]
    is_healthy: bool

# From velocity_types.py
@dataclass(frozen=True)
class VelocityMetrics:
    average_stories_per_sprint: float
    average_points_per_sprint: float
    average_cycle_time_ms: float
    cycle_time_p50_ms: float
    cycle_time_p90_ms: float
    sprints_analyzed: int
    trend: VelocityTrend  # improving, stable, declining
```

### Telemetry Data Structure Design

```python
# Proposed telemetry types for dashboard display
@dataclass(frozen=True)
class MetricSummary:
    """Summary of a single metric for dashboard display."""
    name: str
    value: float
    unit: str
    display_value: str  # Human-readable: "5.2 stories/sprint"
    trend: str | None  # "improving", "stable", "declining" or None
    status: str  # "healthy", "warning", "critical"

@dataclass(frozen=True)
class TelemetrySnapshot:
    """Complete telemetry snapshot for dashboard."""
    burn_down_velocity: MetricSummary
    cycle_time: MetricSummary
    churn_rate: MetricSummary
    agent_idle_times: dict[str, MetricSummary]
    health_status: str  # Overall health
    collected_at: str  # ISO timestamp

@dataclass(frozen=True)
class DashboardMetrics:
    """Display-optimized metrics for dashboard rendering."""
    snapshot: TelemetrySnapshot
    velocity_display: str  # "5.2 stories/sprint (stable)"
    cycle_time_display: str  # "45 min avg (p90: 1.2h)"
    health_summary: str  # "System healthy, 0 alerts"
    agent_status_table: list[dict[str, str]]  # [{agent, idle_time, status}]
```

### Dashboard Display Requirements (FR72)

Per FR72, the dashboard data should include:
1. **Burn-down velocity** - stories/sprint with trend indicator
2. **Cycle time** - average and percentiles (p50, p90, p95)
3. **Churn rate** - exchanges per minute (overall and per-agent)
4. **Agent idle time** - per-agent idle time with warnings

### Project Structure Notes

- Module location: `src/yolo_developer/agents/sm/telemetry.py`
- Types location: `src/yolo_developer/agents/sm/telemetry_types.py`
- Tests location: `tests/unit/agents/sm/test_telemetry.py`
- Test types: `tests/unit/agents/sm/test_telemetry_types.py`

### Previous Story Learnings (Story 10.15)

From Story 10.15 implementation:
1. Use `cast()` for type narrowing with union types
2. Include comprehensive tests for all async functions
3. Test all input combinations including None values
4. Keep File List section updated in story file
5. SM node integration: add field to SMOutput, update return dict, add logging

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR72: SM Agent can maintain system health telemetry dashboard data
- [Source: _bmad-output/planning-artifacts/prd.md] FR66: SM Agent can track burn-down velocity and cycle time metrics
- [Source: _bmad-output/planning-artifacts/prd.md] FR67: SM Agent can detect agent churn rate and idle time
- [Source: _bmad-output/planning-artifacts/prd.md] FR11: SM Agent can monitor agent activity and health metrics
- [Source: src/yolo_developer/agents/sm/health_types.py] HealthMetrics, HealthStatus types
- [Source: src/yolo_developer/agents/sm/velocity_types.py] VelocityMetrics type
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001] TypedDict for graph state, frozen dataclasses for internal
- [Source: _bmad-output/implementation-artifacts/10-15-rollback-coordination.md] Previous story patterns

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

All 7 tasks completed successfully:
- Created telemetry_types.py with MetricSummary, TelemetrySnapshot, TelemetryConfig, DashboardMetrics frozen dataclasses
- Implemented telemetry.py with collect_telemetry, format_for_dashboard, get_dashboard_telemetry functions
- Integrated telemetry collection into sm_node with telemetry_snapshot in SMOutput
- Updated __init__.py with all new exports
- 46 new tests for telemetry module + 8 SM node integration tests (1167 total SM tests pass)
- All ruff and mypy checks pass

### Code Review Fixes

The following issues were identified and fixed during adversarial code review:

1. **MEDIUM-3 (Type Safety)**: Replaced 4 `# type: ignore[arg-type]` comments in telemetry.py with proper `MetricStatus` type annotations. This ensures type safety for the `status` field in MetricSummary and TelemetrySnapshot.

2. **MEDIUM-2 (Weak Logging Test)**: Improved Task 7.8 logging test from a smoke test to proper structlog verification. Added second test for collect_telemetry logging. Tests now verify actual log capture through properly configured structlog.

3. **Code Quality**: Updated health status mapping in `collect_telemetry` to properly map HealthSeverity values ("degraded", "critical", "warning", "healthy") to MetricStatus values ("critical", "warning", "healthy").

### File List

**New Files:**
- src/yolo_developer/agents/sm/telemetry.py - Telemetry collection and formatting
- src/yolo_developer/agents/sm/telemetry_types.py - Type definitions
- tests/unit/agents/sm/test_telemetry.py - Telemetry module tests (46 tests)
- tests/unit/agents/sm/test_telemetry_types.py - Type definition tests (37 tests)

**Modified Files:**
- src/yolo_developer/agents/sm/node.py - Added telemetry collection (Step 6f) and logging
- src/yolo_developer/agents/sm/types.py - Added telemetry_snapshot field to SMOutput
- src/yolo_developer/agents/sm/__init__.py - Added telemetry exports and docstring
- tests/unit/agents/sm/test_node.py - Added TestTelemetryIntegration class (8 tests)
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status to review

