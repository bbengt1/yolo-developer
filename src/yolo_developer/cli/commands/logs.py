"""YOLO logs command implementation (Story 12.6).

This module provides the yolo logs command which displays:
- Recent agent decisions from the audit trail
- Filtering by agent, time range, or decision type
- Pagination with configurable limits
- Verbose mode for full decision details

The command supports multiple output modes:
- Default: Rich formatted tables with truncated content
- JSON: Machine-readable JSON output
- Verbose: Full decision content with rationale and context

Example:
    >>> from yolo_developer.cli.commands.logs import logs_command
    >>>
    >>> logs_command()  # Show recent decisions (default 20)
    >>> logs_command(agent="analyst")  # Filter by agent
    >>> logs_command(since="1h")  # Decisions from last hour
    >>> logs_command(json_output=True)  # Output as JSON

References:
    - FR98-FR105: CLI command requirements
    - Story 12.6: yolo logs command
    - Story 11.1: Decision logging
    - Story 11.7: Audit filtering
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import structlog
from rich.panel import Panel

from yolo_developer.cli.display import (
    console,
    create_table,
    info_panel,
    warning_panel,
)

if TYPE_CHECKING:
    from yolo_developer.audit.types import Decision

logger = structlog.get_logger(__name__)

# Severity colors for Rich display
SEVERITY_COLORS: dict[str, str] = {
    "info": "dim",
    "warning": "yellow",
    "critical": "red",
}

# Agent colors for Rich display
AGENT_COLORS: dict[str, str] = {
    "analyst": "cyan",
    "pm": "blue",
    "architect": "magenta",
    "dev": "green",
    "tea": "yellow",
    "sm": "white",
}

# Decision type display names (shorter for table)
DECISION_TYPE_DISPLAY: dict[str, str] = {
    "requirement_analysis": "Requirement",
    "story_creation": "Story",
    "architecture_choice": "Architecture",
    "implementation_choice": "Implementation",
    "test_strategy": "Test",
    "orchestration": "Orchestration",
    "quality_gate": "Gate",
    "escalation": "Escalation",
}

# Valid decision types (must match audit/types.py VALID_DECISION_TYPES)
VALID_DECISION_TYPES: frozenset[str] = frozenset(DECISION_TYPE_DISPLAY.keys())

# Display constants
TABLE_SUMMARY_MAX_LENGTH = 50
DEFAULT_TRUNCATE_LENGTH = 60


def _parse_since(since_str: str) -> str | None:
    """Parse relative time or ISO timestamp to ISO 8601 string.

    Supports:
    - Relative: 30m, 1h, 2d, 1w
    - ISO 8601: 2026-01-15T10:00:00Z or 2026-01-15

    Args:
        since_str: Time string to parse.

    Returns:
        ISO 8601 timestamp string, or None if invalid.
    """
    # Try relative time pattern
    match = re.match(r"^(\d+)([mhdw])$", since_str.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        now = datetime.now(timezone.utc)
        deltas = {
            "m": timedelta(minutes=value),
            "h": timedelta(hours=value),
            "d": timedelta(days=value),
            "w": timedelta(weeks=value),
        }
        result = now - deltas[unit]
        return result.isoformat()

    # Try ISO timestamp (various formats)
    # Formats with timezone info (%z handles the timezone)
    for fmt in ["%Y-%m-%dT%H:%M:%S%z"]:
        try:
            dt = datetime.strptime(since_str, fmt)  # noqa: DTZ007
            return dt.isoformat()
        except ValueError:
            continue

    # Formats without timezone - add UTC
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            # Parse naive datetime and immediately make it timezone-aware
            naive_dt = datetime.strptime(since_str, fmt)  # noqa: DTZ007
            dt = naive_dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    return None


def _truncate(text: str, max_length: int = DEFAULT_TRUNCATE_LENGTH) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: Text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp for display.

    Args:
        timestamp: ISO 8601 timestamp string.

    Returns:
        Formatted timestamp (YYYY-MM-DD HH:MM:SS).
    """
    # Take first 19 characters to get YYYY-MM-DDTHH:MM:SS
    if len(timestamp) >= 19:
        return timestamp[:10] + " " + timestamp[11:19]
    return timestamp


async def _load_decisions(
    agent_name: str | None,
    decision_type: str | None,
    start_time: str | None,
) -> list[Decision]:
    """Load decisions from the audit store with filters.

    Args:
        agent_name: Filter by agent name (case-insensitive).
        decision_type: Filter by decision type.
        start_time: Filter by start time (ISO 8601).

    Returns:
        List of Decision records matching filters.
    """
    from yolo_developer.audit import (
        AuditFilters,
        AuditFilterService,
        InMemoryDecisionStore,
        InMemoryTraceabilityStore,
    )

    # Create stores - in a real implementation, these would be loaded from disk
    # For now, we use in-memory stores which will be empty
    decision_store = InMemoryDecisionStore()
    traceability_store = InMemoryTraceabilityStore()

    # Create filter service
    filter_service = AuditFilterService(
        decision_store=decision_store,
        traceability_store=traceability_store,
    )

    # Build filters - agent_name is already normalized to lowercase by caller
    filters = AuditFilters(
        agent_name=agent_name,
        decision_type=decision_type,
        start_time=start_time,
    )

    # Query decisions
    return await filter_service.filter_decisions(filters)


def _display_decisions_table(
    decisions: list[Decision],
    offset: int,
    total: int,
) -> None:
    """Display decisions in a Rich table.

    Args:
        decisions: List of decisions to display.
        offset: Starting offset for pagination message.
        total: Total number of decisions.
    """
    if not decisions:
        info_panel(
            "No decisions found matching your criteria.\n\n"
            "[dim]Tips:[/dim]\n"
            "  - Try removing filters to see all decisions\n"
            "  - Ensure audit logging is enabled during runs",
            title="Decision Log",
        )
        return

    # Create table
    columns = [
        ("Timestamp", "dim"),
        ("Agent", "cyan"),
        ("Type", "yellow"),
        ("Summary", "white"),
    ]

    table = create_table("Decision Log", columns)

    # Add rows
    for decision in decisions:
        agent_name = decision.agent.agent_name
        agent_color = AGENT_COLORS.get(agent_name, "white")

        type_display = DECISION_TYPE_DISPLAY.get(
            decision.decision_type, decision.decision_type
        )

        row = [
            _format_timestamp(decision.timestamp),
            f"[{agent_color}]{agent_name}[/{agent_color}]",
            type_display,
            _truncate(decision.content, TABLE_SUMMARY_MAX_LENGTH),
        ]

        table.add_row(*row)

    console.print()
    console.print(table)

    # Pagination info
    showing_end = offset + len(decisions)
    console.print()
    console.print(
        f"[dim]Showing {offset + 1}-{showing_end} of {total} entries[/dim]"
    )
    console.print()


def _display_decisions_verbose(decisions: list[Decision]) -> None:
    """Display decisions with full details.

    Args:
        decisions: List of decisions to display.
    """
    if not decisions:
        info_panel(
            "No decisions found matching your criteria.",
            title="Decision Log",
        )
        return

    console.print()
    console.print("[bold cyan]Decision Log (Verbose)[/bold cyan]")
    console.print("=" * 50)

    for i, decision in enumerate(decisions, 1):
        agent_name = decision.agent.agent_name
        agent_color = AGENT_COLORS.get(agent_name, "white")
        severity_color = SEVERITY_COLORS.get(decision.severity, "white")

        # Decision header
        console.print()
        console.print(
            Panel(
                f"[bold]Decision {i}[/bold]  |  "
                f"[{agent_color}]{agent_name}[/{agent_color}]  |  "
                f"{decision.decision_type}  |  "
                f"[{severity_color}]{decision.severity}[/{severity_color}]",
                title=_format_timestamp(decision.timestamp),
                border_style="cyan",
            )
        )

        # Content
        console.print()
        console.print("[bold]Content:[/bold]")
        console.print(f"  {decision.content}")

        # Rationale
        if decision.rationale:
            console.print()
            console.print("[bold]Rationale:[/bold]")
            console.print(f"  {decision.rationale}")

        # Context
        ctx = decision.context
        if ctx.sprint_id or ctx.story_id or ctx.artifact_id:
            console.print()
            console.print("[bold]Context:[/bold]")
            if ctx.sprint_id:
                console.print(f"  Sprint: {ctx.sprint_id}")
            if ctx.story_id:
                console.print(f"  Story: {ctx.story_id}")
            if ctx.artifact_id:
                console.print(f"  Artifact: {ctx.artifact_id}")

        # Trace links
        if ctx.trace_links:
            console.print()
            console.print("[bold]Trace Links:[/bold]")
            for link in ctx.trace_links:
                console.print(f"  - {link}")

        # Metadata
        if decision.metadata:
            console.print()
            console.print("[bold]Metadata:[/bold]")
            for key, value in decision.metadata.items():
                console.print(f"  {key}: {value}")

        console.print()
        console.print("[dim]" + "-" * 50 + "[/dim]")


def _build_json_output(
    decisions: list[Decision],
    filters_applied: dict[str, Any],
    total_count: int,
    showing: int,
) -> dict[str, Any]:
    """Build JSON output structure.

    Args:
        decisions: List of decisions to include.
        filters_applied: Dictionary of applied filters.
        total_count: Total number of matching decisions.
        showing: Number of decisions shown.

    Returns:
        Dictionary with all output data.
    """
    return {
        "decisions": [d.to_dict() for d in decisions],
        "filters_applied": filters_applied,
        "total_count": total_count,
        "showing": showing,
    }


def logs_command(
    agent: str | None = None,
    since: str | None = None,
    decision_type: str | None = None,
    limit: int = 20,
    show_all: bool = False,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """Execute the logs command.

    Displays agent decisions from the audit trail with filtering options.

    Args:
        agent: Filter by agent name (case-insensitive).
        since: Filter by time (relative like "1h" or ISO timestamp).
        decision_type: Filter by decision type.
        limit: Maximum entries to display (default 20).
        show_all: Show all entries without pagination.
        verbose: Show detailed output including rationale.
        json_output: Output results as JSON.
    """
    logger.debug(
        "logs_command_invoked",
        agent=agent,
        since=since,
        decision_type=decision_type,
        limit=limit,
        show_all=show_all,
        verbose=verbose,
        json_output=json_output,
    )

    # Normalize agent name to lowercase for case-insensitive filtering
    normalized_agent = agent.lower() if agent else None

    # Validate decision_type filter
    if decision_type and decision_type not in VALID_DECISION_TYPES:
        valid_types = ", ".join(sorted(VALID_DECISION_TYPES))
        warning_panel(
            f"Invalid decision type: '{decision_type}'\n\n"
            f"[dim]Valid types:[/dim]\n  {valid_types}",
            title="Invalid Filter",
        )
        return

    # Validate limit
    if limit < 1:
        warning_panel(
            f"Invalid limit: {limit}\n\n"
            "[dim]Limit must be a positive integer.[/dim]",
            title="Invalid Parameter",
        )
        return

    # Parse since filter
    start_time: str | None = None
    if since:
        start_time = _parse_since(since)
        if start_time is None:
            warning_panel(
                f"Invalid time format: '{since}'\n\n"
                "[dim]Supported formats:[/dim]\n"
                "  Relative: 30m, 1h, 2d, 1w\n"
                "  ISO: 2026-01-15 or 2026-01-15T10:00:00Z",
                title="Invalid Filter",
            )
            return

    # Load decisions
    try:
        all_decisions = asyncio.run(
            _load_decisions(
                agent_name=normalized_agent,
                decision_type=decision_type,
                start_time=start_time,
            )
        )
    except (ValueError, RuntimeError) as e:
        logger.warning("failed_to_load_decisions", error=str(e))
        warning_panel(
            f"Failed to load decisions: {e}",
            title="Error",
        )
        return

    # Sort by timestamp (most recent first)
    all_decisions.sort(key=lambda d: d.timestamp, reverse=True)

    total_count = len(all_decisions)

    # Apply pagination
    if show_all:
        decisions = all_decisions
    else:
        decisions = all_decisions[:limit]

    # Build filters dict for JSON output
    filters_applied = {
        "agent_name": normalized_agent,
        "decision_type": decision_type,
        "start_time": start_time,
        "end_time": None,
    }

    # JSON output
    if json_output:
        output = _build_json_output(
            decisions=decisions,
            filters_applied=filters_applied,
            total_count=total_count,
            showing=len(decisions),
        )
        console.print_json(json.dumps(output, default=str))
        return

    # Verbose output with full details
    if verbose and decisions:
        _display_decisions_verbose(decisions)
        console.print(f"[dim]Showing {len(decisions)} of {total_count} entries[/dim]")
        if not show_all and total_count > limit:
            console.print(
                f"[dim]Use --all to show all {total_count} entries[/dim]"
            )
        console.print()
        return

    # Standard table output
    _display_decisions_table(
        decisions=decisions,
        offset=0,
        total=total_count,
    )

    # Help text for more entries
    if not show_all and total_count > limit:
        console.print(
            f"[dim]Use --all to show all {total_count} entries, "
            f"or --limit N to adjust page size[/dim]"
        )
        console.print()
