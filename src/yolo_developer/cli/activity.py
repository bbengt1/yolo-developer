"""Real-time activity display module (Story 12.9).

This module provides the ActivityDisplay class for showing real-time
agent activity during workflow execution using Rich Live context.

The display supports:
- Current agent status tracking
- Agent transition visualization
- Event throttling to prevent display overwhelming
- Verbose mode for detailed output
- Elapsed time tracking

Example:
    >>> from yolo_developer.cli.activity import ActivityDisplay
    >>>
    >>> with ActivityDisplay(verbose=True) as display:
    ...     display.update("analyst", "Analyzing requirements", 5.0)
    ...     display.add_transition("analyst", "pm", "ready", 10.0)

References:
    - FR105: CLI can display real-time agent activity during execution
    - ADR-009: Typer + Rich framework selection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Default refresh rate (updates per second)
DEFAULT_REFRESH_RATE = 4


@dataclass
class ActivityEvent:
    """Structured activity event data for API consumers.

    This dataclass provides a typed interface for activity events that can
    be used by external consumers (SDK, MCP tools) to receive structured
    workflow events. The CLI display uses internal dict representation
    for simplicity, but this class is exported for programmatic access.

    Note:
        Currently used for type definitions and future SDK integration.
        The ActivityDisplay class uses dicts internally for transitions.

    Attributes:
        event_type: Type of event (start, progress, complete, transition).
        agent: Name of the agent associated with the event.
        description: Human-readable description of what's happening.
        timestamp: Unix timestamp of when the event occurred.
        details: Optional additional details about the event.
        previous_agent: For transitions, the agent that completed.

    Example:
        >>> event = ActivityEvent(
        ...     event_type="transition",
        ...     agent="pm",
        ...     description="Starting PM agent",
        ...     timestamp=time.time(),
        ...     previous_agent="analyst",
        ... )
    """

    event_type: Literal["start", "progress", "complete", "transition"]
    agent: str
    description: str
    timestamp: float
    details: dict[str, Any] | None = None
    previous_agent: str | None = None


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time as MM:SS or HH:MM:SS.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string.

    Example:
        >>> format_elapsed_time(65.5)
        '01:05'
        >>> format_elapsed_time(3665.0)
        '01:01:05'
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ActivityDisplay:
    """Real-time activity display using Rich Live context.

    Provides a visual panel showing current agent activity, elapsed time,
    and agent transitions during workflow execution.

    Attributes:
        verbose: Show detailed event information.
        refresh_rate: Display updates per second.
        current_agent: Currently executing agent name.
        current_description: Description of current activity.
        elapsed_time: Total elapsed time in seconds.
        event_count: Total number of events processed.
        transitions: List of agent transitions.

    Example:
        >>> display = ActivityDisplay(verbose=True)
        >>> with display:
        ...     display.update("analyst", "Analyzing...", 5.0)
    """

    verbose: bool = False
    refresh_rate: int = DEFAULT_REFRESH_RATE
    current_agent: str = ""
    current_description: str = ""
    elapsed_time: float = 0.0
    event_count: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)
    _console: Console | None = field(default=None, repr=False)
    _live: Live | None = field(default=None, repr=False)
    _last_update: float = field(default=0.0, repr=False)
    _last_event: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Initialize console if not provided."""
        if self._console is None:
            self._console = Console()

    def update(self, agent: str, description: str, elapsed: float) -> None:
        """Update the display with new agent status.

        Args:
            agent: Name of the currently executing agent.
            description: Description of what the agent is doing.
            elapsed: Total elapsed time in seconds.
        """
        self.current_agent = agent
        self.current_description = description
        self.elapsed_time = elapsed
        self.event_count += 1
        self._last_event = description

        # Update the live display if running
        if self._live is not None and self.should_update():
            self._live.update(self.render())
            self._last_update = time.time()

    def add_transition(
        self,
        from_agent: str,
        to_agent: str,
        reason: str,
        elapsed: float,
    ) -> None:
        """Record an agent transition.

        Args:
            from_agent: Agent that completed.
            to_agent: Agent that is starting.
            reason: Reason for the transition.
            elapsed: Elapsed time when transition occurred.
        """
        self.transitions.append(
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "elapsed": elapsed,
            }
        )

        # Transition updates always bypass throttling
        if self._live is not None:
            self._live.update(self.render())
            self._last_update = time.time()

    def should_update(self, is_transition: bool = False) -> bool:
        """Check if display should update based on throttling.

        Args:
            is_transition: Whether this is a transition event (bypasses throttle).

        Returns:
            True if update should proceed, False if throttled.
        """
        # Transitions always bypass throttle
        if is_transition:
            return True

        # First update always allowed
        if self._last_update == 0.0:
            return True

        # Check throttle interval
        interval = 1.0 / self.refresh_rate
        return (time.time() - self._last_update) >= interval

    def render(self) -> Panel:
        """Render the activity display panel.

        Returns:
            Rich Panel containing the current activity status.
        """
        elements: list[Text | str] = []

        # Header with spinner and agent name
        if self.current_agent:
            header = Text()
            header.append("🔄 Running: ", style="bold cyan")
            header.append(self.current_agent, style="bold green")
            elements.append(header)
            elements.append("")

            # Description
            if self.current_description:
                elements.append(Text(self.current_description, style="dim"))

            # Elapsed time
            time_text = Text()
            time_text.append("Elapsed: ", style="cyan")
            time_text.append(format_elapsed_time(self.elapsed_time), style="bold")
            elements.append(time_text)

        # Verbose mode additions
        if self.verbose and self.event_count > 0:
            elements.append("")
            events_text = Text()
            events_text.append(f"Events: {self.event_count}", style="dim")
            elements.append(events_text)

            if self._last_event:
                last_text = Text()
                last_text.append(f"Last: {self._last_event[:40]}", style="dim italic")
                elements.append(last_text)

        # Transitions section
        if self.transitions:
            elements.append("")
            elements.append(Text("─" * 35, style="dim"))

            # Show most recent transition
            last_transition = self.transitions[-1]
            trans_text = Text()
            trans_text.append(last_transition["from_agent"], style="yellow")
            trans_text.append(" → ", style="dim")
            trans_text.append(last_transition["to_agent"], style="green")
            trans_text.append(f" ({last_transition['reason']})", style="dim")
            elements.append(trans_text)

        # Build group from elements
        group = Group(*elements) if elements else Group(Text("Waiting...", style="dim"))

        return Panel(
            group,
            title="[bold]Agent Activity[/bold]",
            border_style="blue",
        )

    def start(self) -> None:
        """Start the live display."""
        if self._console is None:
            self._console = Console()

        self._live = Live(
            self.render(),
            console=self._console,
            refresh_per_second=self.refresh_rate,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> ActivityDisplay:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()
