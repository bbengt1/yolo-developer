"""YOLO status command implementation (Story 12.5).

This module provides the yolo status command which displays:
- Sprint progress: Completed stories, in-progress work, current agent
- Health metrics: Agent idle times, cycle times, churn rates, alerts
- Session information: Active session, available sessions, resume instructions

The command supports multiple output modes:
- Default: Rich formatted tables and panels
- JSON: Machine-readable JSON output
- Verbose: Detailed metrics including agent snapshots
- Health-only: Focus on health metrics only
- Sessions-only: Focus on session list only

Example:
    >>> from yolo_developer.cli.commands.status import status_command
    >>>
    >>> status_command()  # Show all status info
    >>> status_command(json_output=True)  # Output as JSON
    >>> status_command(verbose=True)  # Show detailed health metrics
    >>> status_command(health_only=True)  # Show only health metrics

References:
    - FR98-FR105: CLI command requirements
    - Story 12.5: yolo status command
    - Story 10.5: Health monitoring
    - Story 10.4: Session persistence
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from rich.panel import Panel

from yolo_developer.cli.display import (
    console,
    create_table,
    info_panel,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.health_types import HealthStatus
    from yolo_developer.orchestrator.session import SessionMetadata

logger = structlog.get_logger(__name__)

# Health status color mapping for Rich display
STATUS_COLORS: dict[str, str] = {
    "healthy": "green",
    "warning": "yellow",
    "degraded": "orange1",
    "critical": "red",
}

# Alert severity color mapping
ALERT_COLORS: dict[str, str] = {
    "info": "blue",
    "warning": "yellow",
    "critical": "red",
}


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration string (e.g., "2m 30s", "1h 5m").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _format_datetime(dt: datetime) -> str:
    """Format datetime for display.

    Args:
        dt: Datetime object.

    Returns:
        Formatted datetime string.
    """
    # If datetime is recent (within 24 hours), show relative time
    now = datetime.now(timezone.utc)
    delta = now - dt
    if delta.total_seconds() < 60:
        return "just now"
    elif delta.total_seconds() < 3600:
        minutes = int(delta.total_seconds() // 60)
        return f"{minutes}m ago"
    elif delta.total_seconds() < 86400:
        hours = int(delta.total_seconds() // 3600)
        return f"{hours}h ago"
    else:
        return dt.strftime("%Y-%m-%d %H:%M")


async def _get_session_data(
    sessions_dir: str,
) -> tuple[str | None, SessionMetadata | None, list[SessionMetadata]]:
    """Get session data from SessionManager.

    Args:
        sessions_dir: Path to sessions directory.

    Returns:
        Tuple of (active_session_id, active_metadata, all_sessions).
    """
    from yolo_developer.orchestrator.session import SessionManager

    manager = SessionManager(sessions_dir)

    active_id = await manager.get_active_session_id()
    active_metadata = None
    if active_id:
        try:
            _, active_metadata = await manager.load_session(active_id)
        except (FileNotFoundError, OSError, ValueError) as e:
            logger.warning("failed_to_load_active_session", error=str(e))

    all_sessions = await manager.list_sessions()

    return active_id, active_metadata, all_sessions


async def _get_health_data(
    state: dict[str, object] | None,
) -> HealthStatus | None:
    """Get health status from monitor_health.

    Args:
        state: Current orchestration state (or None if no active session).

    Returns:
        HealthStatus object or None if health monitoring unavailable.
    """
    if state is None:
        return None

    try:
        from yolo_developer.agents.sm.health import monitor_health

        return await monitor_health(state)  # type: ignore[arg-type]
    except (ValueError, KeyError, AttributeError) as e:
        logger.warning("failed_to_get_health_data", error=str(e))
        return None


def _display_sprint_progress(
    metadata: SessionMetadata | None,
    verbose: bool,
) -> None:
    """Display sprint progress section.

    Args:
        metadata: Session metadata with progress info.
        verbose: Whether to show detailed output.
    """
    if metadata is None:
        info_panel("No active sprint found.", title="Sprint Progress")
        console.print(
            "\n[dim]Start a sprint with:[/dim] [cyan]yolo run[/cyan]\n"
        )
        return

    completed = metadata.stories_completed
    total = metadata.stories_total
    percentage = (completed / total * 100) if total > 0 else 0.0

    # Create progress display
    console.print()
    console.print(
        Panel(
            f"[bold]Stories:[/bold] {completed}/{total} "
            f"([green]{percentage:.0f}%[/green] complete)",
            title="Sprint Progress",
            border_style="cyan",
        )
    )

    # Progress bar
    bar_width = 40
    filled = int(bar_width * percentage / 100)
    bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_width - filled)
    console.print(f"  Progress: [{bar}]")
    console.print()


def _display_health_metrics(
    health: HealthStatus | None,
    verbose: bool,
) -> None:
    """Display health metrics section.

    Args:
        health: HealthStatus object from monitor_health.
        verbose: Whether to show detailed output.
    """
    if health is None:
        info_panel(
            "No health data available. Start a session to monitor health.",
            title="System Health",
        )
        return

    # Overall status
    status_color = STATUS_COLORS.get(health.status, "white")
    console.print()
    console.print(
        Panel(
            f"[{status_color}][bold]{health.status.upper()}[/bold][/{status_color}]\n\n"
            f"{health.summary}",
            title="System Health",
            border_style=status_color,
        )
    )

    # Alerts
    if health.alerts:
        console.print()
        console.print("[bold]Active Alerts:[/bold]")
        for alert in health.alerts:
            alert_color = ALERT_COLORS.get(alert.severity, "white")
            console.print(
                f"  [{alert_color}]● {alert.severity.upper()}[/{alert_color}]: "
                f"{alert.message}"
            )
    else:
        console.print("\n  [green]✓ No active alerts[/green]")

    # Verbose: Show detailed metrics
    if verbose and health.metrics:
        console.print()
        metrics = health.metrics

        # Agent health table
        if metrics.agent_idle_times:
            table = create_table(
                "Agent Health",
                [
                    ("Agent", "cyan"),
                    ("Idle Time", "white"),
                    ("Cycle Time", "white"),
                    ("Churn Rate", "white"),
                ],
            )

            for agent in sorted(metrics.agent_idle_times.keys()):
                idle = metrics.agent_idle_times.get(agent, 0.0)
                cycle = metrics.agent_cycle_times.get(agent)
                churn = metrics.agent_churn_rates.get(agent, 0.0)

                table.add_row(
                    agent,
                    _format_duration(idle),
                    _format_duration(cycle) if cycle else "-",
                    f"{churn:.2f}/min",
                )

            console.print(table)

        # Overall metrics
        console.print()
        console.print("[bold]Overall Metrics:[/bold]")
        console.print(
            f"  Cycle Time: {_format_duration(metrics.overall_cycle_time)}"
        )
        console.print(f"  Churn Rate: {metrics.overall_churn_rate:.2f}/min")
        if metrics.unproductive_churn_rate > 0:
            console.print(
                f"  Unproductive Churn: {metrics.unproductive_churn_rate:.2f}/min"
            )

    console.print()


def _display_sessions(
    active_id: str | None,
    all_sessions: list[SessionMetadata],
    verbose: bool,
) -> None:
    """Display sessions section.

    Args:
        active_id: Currently active session ID.
        all_sessions: List of all available sessions.
        verbose: Whether to show detailed output.
    """
    if not all_sessions:
        info_panel("No sessions found.", title="Sessions")
        console.print(
            "\n[dim]Start a session with:[/dim] [cyan]yolo run[/cyan]\n"
        )
        return

    console.print()
    table = create_table(
        "Sessions",
        [
            ("Session", "cyan"),
            ("Status", "white"),
            ("Agent", "white"),
            ("Progress", "white"),
            ("Last Active", "white"),
        ],
    )

    for session in all_sessions[:10]:  # Limit to 10 sessions
        is_active = session.session_id == active_id
        status = "[green]● Active[/green]" if is_active else "[dim]Paused[/dim]"

        progress = (
            f"{session.stories_completed}/{session.stories_total}"
            if session.stories_total > 0
            else "-"
        )

        table.add_row(
            session.session_id[:20] + "..." if len(session.session_id) > 20 else session.session_id,
            status,
            session.current_agent or "-",
            progress,
            _format_datetime(session.last_checkpoint),
        )

    console.print(table)

    # Resume instructions
    if active_id:
        console.print()
        console.print("[dim]Resume active session:[/dim] [cyan]yolo run --resume[/cyan]")
    elif all_sessions:
        console.print()
        console.print("[dim]Resume a session:[/dim] [cyan]yolo run --resume --thread-id <session-id>[/cyan]")

    console.print()


def _build_json_output(
    active_id: str | None,
    metadata: SessionMetadata | None,
    all_sessions: list[SessionMetadata],
    health: HealthStatus | None,
) -> dict[str, Any]:
    """Build JSON output structure.

    Args:
        active_id: Currently active session ID.
        metadata: Active session metadata.
        all_sessions: List of all available sessions.
        health: HealthStatus object.

    Returns:
        Dictionary with all status data.
    """
    result: dict[str, Any] = {}

    # Sprint progress
    if metadata:
        completed = metadata.stories_completed
        total = metadata.stories_total
        result["sprint"] = {
            "stories_completed": completed,
            "stories_total": total,
            "completion_percentage": (completed / total * 100) if total > 0 else 0.0,
            "status": "in_progress" if completed < total else "completed",
        }
    else:
        result["sprint"] = None

    # Health metrics
    if health:
        result["health"] = health.to_dict()
    else:
        result["health"] = None

    # Active session
    if metadata:
        result["session"] = {
            "session_id": metadata.session_id,
            "created_at": metadata.created_at.isoformat(),
            "last_checkpoint": metadata.last_checkpoint.isoformat(),
            "current_agent": metadata.current_agent,
        }
    else:
        result["session"] = None

    # Available sessions
    result["available_sessions"] = [
        {
            "session_id": s.session_id,
            "last_checkpoint": s.last_checkpoint.isoformat(),
            "current_agent": s.current_agent,
            "stories_completed": s.stories_completed,
            "stories_total": s.stories_total,
        }
        for s in all_sessions
    ]

    return result


async def _gather_all_status_data(
    sessions_dir: str,
) -> tuple[
    str | None,
    SessionMetadata | None,
    list[SessionMetadata],
    HealthStatus | None,
]:
    """Gather all status data in a single async context.

    This avoids multiple asyncio.run() calls by doing all async work
    in one coroutine.

    Args:
        sessions_dir: Path to sessions directory.

    Returns:
        Tuple of (active_id, metadata, all_sessions, health).
    """
    # Get session data
    active_id, metadata, all_sessions = await _get_session_data(sessions_dir)

    # Get health data if we have an active session
    health = None
    if metadata and active_id:
        try:
            from yolo_developer.orchestrator.session import SessionManager

            manager = SessionManager(sessions_dir)
            state, _ = await manager.load_session(active_id)
            health = await _get_health_data(state)  # type: ignore[arg-type]
        except (FileNotFoundError, OSError, ValueError) as e:
            logger.warning("failed_to_get_health_data", error=str(e))

    return active_id, metadata, all_sessions, health


def status_command(
    verbose: bool = False,
    json_output: bool = False,
    health_only: bool = False,
    sessions_only: bool = False,
) -> None:
    """Execute the status command.

    Displays sprint progress, health metrics, and session information.

    Args:
        verbose: Show detailed output including agent health snapshots.
        json_output: Output results as JSON instead of formatted display.
        health_only: Show only health metrics.
        sessions_only: Show only session list.
    """
    logger.debug(
        "status_command_invoked",
        verbose=verbose,
        json_output=json_output,
        health_only=health_only,
        sessions_only=sessions_only,
    )

    # Sessions directory is always .yolo/sessions (matches run command)
    sessions_dir = ".yolo/sessions"

    # Gather all async data in single event loop run
    try:
        active_id, metadata, all_sessions, health = asyncio.run(
            _gather_all_status_data(sessions_dir)
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("failed_to_get_status_data", error=str(e))
        active_id, metadata, all_sessions, health = None, None, [], None

    # JSON output
    if json_output:
        output = _build_json_output(active_id, metadata, all_sessions, health)
        console.print_json(json.dumps(output, default=str))
        return

    # Display header
    console.print()
    console.print("[bold cyan]YOLO Developer Status[/bold cyan]")
    console.print("=" * 40)

    # Display sections based on flags
    if health_only:
        _display_health_metrics(health, verbose)
    elif sessions_only:
        _display_sessions(active_id, all_sessions, verbose)
    else:
        # Show all sections
        _display_sprint_progress(metadata, verbose)
        _display_health_metrics(health, verbose)
        _display_sessions(active_id, all_sessions, verbose)
