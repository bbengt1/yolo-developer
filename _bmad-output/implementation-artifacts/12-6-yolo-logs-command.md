# Story 12.6: yolo logs Command

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want to view logs via `yolo logs`,
so that I can see decision history.

## Acceptance Criteria

### AC1: Recent Decisions Display
**Given** audit data exists
**When** I run `yolo logs`
**Then** recent decisions are displayed
**And** decisions show timestamp, agent, type, and content
**And** output is formatted with Rich styling

### AC2: Agent Filter
**Given** audit data exists from multiple agents
**When** I run `yolo logs --agent analyst`
**Then** only decisions from the specified agent are displayed
**And** the filter is case-insensitive

### AC3: Since Filter
**Given** audit data exists spanning multiple time periods
**When** I run `yolo logs --since 1h` (or `--since 2024-01-15`)
**Then** only decisions after the specified time are displayed
**And** relative times (1h, 30m, 7d) and ISO timestamps are supported

### AC4: Pagination for Long Logs
**Given** audit data contains more than 20 decisions
**When** I run `yolo logs`
**Then** output is paginated showing 20 entries by default
**And** `--limit N` overrides the default page size
**And** `--all` shows all entries without pagination

## Tasks / Subtasks

- [x] Task 1: Add CLI Flags to main.py (AC: #1, #2, #3, #4)
  - [x] Add `--agent/-a` flag to filter by agent name
  - [x] Add `--since/-s` flag for time-based filtering
  - [x] Add `--type/-t` flag to filter by decision type (bonus)
  - [x] Add `--limit/-l` flag to control pagination (default 20)
  - [x] Add `--all` flag to disable pagination
  - [x] Add `--verbose/-v` flag for detailed output
  - [x] Add `--json/-j` flag for machine-readable output
  - [x] Update logs command help text with examples

- [x] Task 2: Implement Time Parsing Utility (AC: #3)
  - [x] Create `_parse_since()` function for relative times (1h, 30m, 7d, 1w)
  - [x] Support ISO 8601 timestamp parsing
  - [x] Return None for invalid input with warning
  - [x] Add unit tests for time parsing

- [x] Task 3: Implement logs_command Core Logic (AC: #1, #2, #3)
  - [x] Load project configuration via load_config()
  - [x] Create InMemoryDecisionStore instance (or load from session)
  - [x] Build AuditFilters from CLI flags
  - [x] Query decisions via AuditFilterService
  - [x] Handle no audit data gracefully with helpful message

- [x] Task 4: Implement Rich Display Output (AC: #1, #4)
  - [x] Build Rich table for decisions list
  - [x] Show columns: Timestamp, Agent, Type, Summary
  - [x] Apply decision severity colors (info=dim, warning=yellow, critical=red)
  - [x] Truncate long content with ellipsis
  - [x] Implement pagination with "Showing X-Y of Z entries" message

- [x] Task 5: Implement Detailed/Verbose Output (AC: #1)
  - [x] Show full decision content in verbose mode
  - [x] Include rationale and context
  - [x] Show metadata if present
  - [x] Display trace links if available

- [x] Task 6: Implement JSON Output (AC: #1)
  - [x] Create structured dict with decisions array
  - [x] Include filter metadata
  - [x] Output via json.dumps with proper formatting
  - [x] Ensure all datetime fields are ISO formatted

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Test CLI flag parsing
  - [x] Test time parsing utility (relative and ISO)
  - [x] Test agent filter application
  - [x] Test since filter application
  - [x] Test pagination logic
  - [x] Test --all flag behavior
  - [x] Test JSON output structure
  - [x] Test verbose mode detail level
  - [x] Test no audit data handling

## Dev Notes

### Existing Implementation

The logs command exists as a placeholder at `src/yolo_developer/cli/commands/logs.py` (28 lines) that currently just displays "coming soon".

**CLI wiring already exists in main.py:**
```python
@app.command("logs")
def logs() -> None:
    """Browse decision audit trail.

    View the history of agent decisions with filtering and
    search capabilities.

    This command will be fully implemented in Story 12.6.
    """
    from yolo_developer.cli.commands.logs import logs_command

    logs_command()
```

### Key Dependencies

**Audit Filtering (audit/filter_service.py, filter_types.py):**
- `AuditFilterService(decision_store, traceability_store, cost_store?, adr_store?)` - Coordinates filtering
- `AuditFilters(agent_name?, decision_type?, start_time?, end_time?, ...)` - Unified filter dataclass
- `filter_decisions(filters)` - Returns list of Decision records
- `filter_all(filters)` - Returns dict with decisions, artifacts, costs, adrs

**Decision Types (audit/types.py):**
- `Decision` - Complete decision record with id, decision_type, content, rationale, agent, context, timestamp, metadata, severity
- `AgentIdentity` - Agent name, type, session_id
- `DecisionContext` - Sprint/story/artifact/parent_decision info
- `DecisionType` - Literal["requirement_analysis", "story_creation", "architecture_choice", "implementation_choice", "test_strategy", "orchestration", "quality_gate", "escalation"]
- `DecisionSeverity` - Literal["info", "warning", "critical"]

**Audit View (audit/view.py):**
- `AuditViewService` - Human-readable output orchestration
- `view_decisions(filters?, options?)` - Formatted decision output
- `view_summary(filters?, options?)` - Summary statistics

**Decision Store (audit/store.py, memory_store.py):**
- `DecisionStore` - Protocol for decision persistence
- `InMemoryDecisionStore` - Testing/single-session implementation
- `DecisionFilters` - Store-specific filter type

**Rich Formatter (audit/rich_formatter.py):**
- `RichAuditFormatter` - Rich terminal output formatter
- `format_decisions(decisions, options?)` - Formatted decision list

**Configuration (config/):**
- `load_config()` - Load project configuration
- Default audit directory: `.yolo/audit`

### Display Patterns

Use Rich for logs display following existing patterns:
```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from yolo_developer.cli.display import create_table, info_panel

# Severity colors
SEVERITY_COLORS = {
    "info": "dim",
    "warning": "yellow",
    "critical": "red",
}

# Agent colors (match status command patterns)
AGENT_COLORS = {
    "analyst": "cyan",
    "pm": "blue",
    "architect": "magenta",
    "dev": "green",
    "tea": "yellow",
    "sm": "white",
}
```

Example table pattern:
```python
table = create_table("Decision Log", [
    ("Timestamp", "dim"),
    ("Agent", "cyan"),
    ("Type", "yellow"),
    ("Summary", "white"),
])
table.add_row(
    decision.timestamp[:19],  # Truncate to seconds
    decision.agent.agent_name,
    decision.decision_type,
    decision.content[:60] + "..." if len(decision.content) > 60 else decision.content,
)
```

### Time Parsing Implementation

```python
import re
from datetime import datetime, timedelta, timezone

def _parse_since(since_str: str) -> str | None:
    """Parse relative time or ISO timestamp to ISO 8601 string.

    Supports:
    - Relative: 30m, 1h, 2d, 1w
    - ISO 8601: 2026-01-15T10:00:00Z or 2026-01-15

    Returns:
        ISO 8601 timestamp string, or None if invalid.
    """
    # Try relative time pattern
    match = re.match(r'^(\d+)([mhdw])$', since_str.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        now = datetime.now(timezone.utc)
        deltas = {
            'm': timedelta(minutes=value),
            'h': timedelta(hours=value),
            'd': timedelta(days=value),
            'w': timedelta(weeks=value),
        }
        result = now - deltas[unit]
        return result.isoformat()

    # Try ISO timestamp (various formats)
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(since_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    return None
```

### Architecture Patterns

Per ADR-005 and existing CLI patterns:
- Use async/await for store queries
- Use structlog for logging
- Use typer.Exit() for error exits
- Follow same flag patterns as run/status commands (--verbose, --json)
- Handle missing audit data gracefully

### Project Structure Notes

- CLI command: `src/yolo_developer/cli/commands/logs.py`
- CLI wiring: `src/yolo_developer/cli/main.py:323-334`
- Tests: `tests/unit/cli/test_logs_command.py` (new)
- Display utilities: `src/yolo_developer/cli/display.py`
- Audit module: `src/yolo_developer/audit/` (existing)

### JSON Output Structure

```json
{
  "decisions": [
    {
      "id": "dec-001",
      "timestamp": "2026-01-19T10:00:00+00:00",
      "agent": {
        "agent_name": "analyst",
        "agent_type": "analyst",
        "session_id": "session-123"
      },
      "decision_type": "requirement_analysis",
      "content": "OAuth2 authentication required",
      "rationale": "Industry standard security",
      "severity": "info",
      "context": {
        "sprint_id": "sprint-1",
        "story_id": "1-2-user-auth"
      },
      "metadata": {}
    }
  ],
  "filters_applied": {
    "agent_name": "analyst",
    "start_time": "2026-01-19T09:00:00+00:00",
    "end_time": null
  },
  "total_count": 42,
  "showing": 20,
  "page": 1
}
```

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-12.6]
- [Source: src/yolo_developer/cli/commands/logs.py]
- [Source: src/yolo_developer/cli/main.py:323-334]
- [Source: src/yolo_developer/audit/__init__.py]
- [Source: src/yolo_developer/audit/filter_service.py]
- [Source: src/yolo_developer/audit/filter_types.py]
- [Source: src/yolo_developer/audit/types.py]
- [Source: src/yolo_developer/audit/view.py]
- [Source: src/yolo_developer/audit/rich_formatter.py]
- [Related: Story 11.1 (Decision Logging)]
- [Related: Story 11.7 (Audit Filtering)]
- [Related: Story 12.5 (yolo status command)]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- Implemented full `yolo logs` CLI command with 7 flags: --agent, --since, --type, --limit, --all, --verbose, --json
- Time parsing utility supports relative times (30m, 1h, 7d, 1w) and ISO 8601 formats
- Rich table display with severity coloring and agent colors matching status command
- Verbose mode shows full decision details including rationale, context, and trace links
- JSON output includes decisions array, filters_applied, and pagination info
- Pagination defaults to 20 entries, --all shows everything
- Graceful handling of no audit data with helpful tips
- 43 unit tests covering all acceptance criteria
- All tests pass, ruff and mypy clean

**Code Review Fixes Applied:**
- Added decision_type input validation against VALID_DECISION_TYPES
- Added limit validation (must be >= 1)
- Consolidated case-sensitivity normalization to single entry point (normalized_agent)
- Removed dead code: verbose parameter from _display_decisions_table (never used)
- Added constants for magic numbers: TABLE_SUMMARY_MAX_LENGTH, DEFAULT_TRUNCATE_LENGTH
- Added 3 new tests for validation edge cases

### Change Log

- 2026-01-19: Implemented Story 12.6 - yolo logs command (Date: 2026-01-19)
- 2026-01-19: Applied code review fixes (validation, dead code removal, constants)

### File List

- src/yolo_developer/cli/commands/logs.py (modified - complete rewrite + code review fixes)
- src/yolo_developer/cli/main.py (modified - added CLI flags at lines 323-397)
- tests/unit/cli/test_logs_command.py (new - 43 tests including validation tests)
- tests/unit/cli/test_placeholder_commands.py (modified - removed logs tests)
