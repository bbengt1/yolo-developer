"""Rich-based terminal formatter for audit views (Story 11.3).

This module provides a Rich library-based implementation of the AuditFormatter
protocol for beautiful terminal output with colors, panels, tables, and trees.

Example:
    >>> from rich.console import Console
    >>> from yolo_developer.audit.rich_formatter import RichAuditFormatter
    >>>
    >>> console = Console()
    >>> formatter = RichAuditFormatter(console)
    >>>
    >>> output = formatter.format_decision(decision)
    >>> print(output)

References:
    - FR83: Users can view audit trail in human-readable format
    - Rich library: https://rich.readthedocs.io/
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from yolo_developer.audit.formatter_types import (
    DEFAULT_COLOR_SCHEME,
    DEFAULT_FORMAT_OPTIONS,
    ColorScheme,
    FormatOptions,
)

if TYPE_CHECKING:
    from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
    from yolo_developer.audit.types import Decision


class RichAuditFormatter:
    """Rich library-based formatter for audit views.

    Uses Rich for beautiful terminal output with colors, panels, tables,
    and tree visualizations.

    Attributes:
        _console: Rich Console instance for rendering
        _color_scheme: Color configuration for severity and agent coloring

    Example:
        >>> console = Console()
        >>> formatter = RichAuditFormatter(console)
        >>> output = formatter.format_decision(decision)
    """

    def __init__(
        self,
        console: Console,
        color_scheme: ColorScheme | None = None,
    ) -> None:
        """Initialize the Rich formatter.

        Args:
            console: Rich Console instance for output rendering.
            color_scheme: Optional color scheme (uses DEFAULT_COLOR_SCHEME if None).
        """
        self._console = console
        self._color_scheme = color_scheme or DEFAULT_COLOR_SCHEME

    def _get_severity_color(self, severity: str) -> str:
        """Get color for a severity level."""
        colors = {
            "critical": self._color_scheme.severity_critical,
            "warning": self._color_scheme.severity_warning,
            "info": self._color_scheme.severity_info,
        }
        return colors.get(severity, self._color_scheme.severity_info)

    def _get_agent_color(self, agent_type: str) -> str:
        """Get color for an agent type."""
        colors = {
            "analyst": self._color_scheme.agent_analyst,
            "pm": self._color_scheme.agent_pm,
            "architect": self._color_scheme.agent_architect,
            "dev": self._color_scheme.agent_dev,
            "sm": self._color_scheme.agent_sm,
            "tea": self._color_scheme.agent_tea,
        }
        return colors.get(agent_type, "white")

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[: max_length - 3] + "..."

    def _render_to_string(self, renderable: Any) -> str:
        """Render a Rich object to string."""
        string_io = StringIO()
        temp_console = Console(
            file=string_io,
            force_terminal=True,
            no_color=self._console.no_color,
            width=self._console.width,
        )
        temp_console.print(renderable)
        return string_io.getvalue()

    def format_decision(self, decision: Decision, options: FormatOptions | None = None) -> str:
        """Format a single decision for display.

        Args:
            decision: The decision to format.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of the decision.
        """
        opts = options or DEFAULT_FORMAT_OPTIONS
        severity_color = self._get_severity_color(decision.severity)
        agent_color = self._get_agent_color(decision.agent.agent_type)

        # Build content based on style
        if opts.style == "minimal":
            content = (
                f"[{severity_color}]{decision.severity.upper()}[/{severity_color}] "
                f"[{agent_color}]{decision.agent.agent_type}[/{agent_color}] "
                f"| {decision.timestamp} | {decision.id}"
            )
            return self._render_to_string(content)

        # Standard or verbose
        lines = [
            f"[bold]ID:[/bold] {decision.id}",
            f"[bold]Type:[/bold] {decision.decision_type}",
            f"[bold]Severity:[/bold] [{severity_color}]{decision.severity}[/{severity_color}]",
            f"[bold]Agent:[/bold] [{agent_color}]{decision.agent.agent_name}[/{agent_color}] ({decision.agent.agent_type})",
            f"[bold]Timestamp:[/bold] {decision.timestamp}",
            "",
            "[bold]Content:[/bold]",
            self._truncate_content(decision.content, opts.max_content_length),
            "",
            "[bold]Rationale:[/bold]",
            self._truncate_content(decision.rationale, opts.max_content_length),
        ]

        # Add context for standard/verbose
        if decision.context.sprint_id or decision.context.story_id:
            lines.append("")
            lines.append("[bold]Context:[/bold]")
            if decision.context.sprint_id:
                lines.append(f"  Sprint: {decision.context.sprint_id}")
            if decision.context.story_id:
                lines.append(f"  Story: {decision.context.story_id}")

        # Add metadata for verbose
        if opts.style == "verbose" and opts.show_metadata and decision.metadata:
            lines.append("")
            lines.append("[bold]Metadata:[/bold]")
            for key, value in decision.metadata.items():
                lines.append(f"  {key}: {value}")

        # Add trace links for verbose
        if opts.style == "verbose" and opts.show_trace_links:
            if decision.context.trace_links:
                lines.append("")
                lines.append("[bold]Trace Links:[/bold]")
                for link_id in decision.context.trace_links:
                    lines.append(f"  - {link_id}")

        content = "\n".join(lines)
        panel = Panel(
            content,
            title=f"Decision: {decision.id}",
            border_style=severity_color,
        )
        return self._render_to_string(panel)

    def format_decisions(
        self, decisions: list[Decision], options: FormatOptions | None = None
    ) -> str:
        """Format a list of decisions chronologically.

        Args:
            decisions: List of decisions to format.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of all decisions.
        """
        if not decisions:
            return self._render_to_string("[dim]No decisions to display.[/dim]")

        # Sort by timestamp (oldest first)
        sorted_decisions = sorted(decisions, key=lambda d: d.timestamp)

        parts = []
        for decision in sorted_decisions:
            parts.append(self.format_decision(decision, options))

        return "\n".join(parts)

    def format_trace_chain(
        self,
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: FormatOptions | None = None,
    ) -> str:
        """Format a trace chain of artifacts and links as a tree.

        Args:
            artifacts: List of artifacts in the trace chain.
            links: List of links connecting the artifacts.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of the trace chain.
        """
        # Note: options parameter available for future style customization
        _ = options  # Currently unused but part of protocol interface
        if not artifacts:
            return self._render_to_string("[dim]No artifacts in trace chain.[/dim]")

        # Build artifact lookup
        artifact_map = {a.id: a for a in artifacts}

        # Build adjacency list from links
        children: dict[str, list[str]] = {}
        has_parent: set[str] = set()

        for link in links:
            if link.source_id not in children:
                children[link.source_id] = []
            children[link.source_id].append(link.target_id)
            has_parent.add(link.target_id)

        # Find root artifacts (no incoming links)
        roots = [a for a in artifacts if a.id not in has_parent]
        if not roots:
            roots = artifacts[:1]  # Fallback to first artifact

        # Build tree
        tree = Tree("[bold]Trace Chain[/bold]")

        def add_children(parent_tree: Tree, artifact_id: str, visited: set[str]) -> None:
            if artifact_id in visited:
                return
            visited.add(artifact_id)

            artifact = artifact_map.get(artifact_id)
            if artifact:
                artifact_node = parent_tree.add(
                    f"[bold]{artifact.artifact_type}:[/bold] {artifact.name} ({artifact.id})"
                )
                for child_id in children.get(artifact_id, []):
                    add_children(artifact_node, child_id, visited)

        visited: set[str] = set()
        for root in roots:
            add_children(tree, root.id, visited)

        # Add any orphan artifacts
        for artifact in artifacts:
            if artifact.id not in visited:
                tree.add(
                    f"[dim][bold]{artifact.artifact_type}:[/bold] {artifact.name} ({artifact.id})[/dim]"
                )

        return self._render_to_string(tree)

    def format_coverage_report(
        self, report: dict[str, Any], options: FormatOptions | None = None
    ) -> str:
        """Format coverage statistics as a table.

        Args:
            report: Coverage report dictionary.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of coverage statistics.
        """
        # Note: options parameter available for future style customization
        _ = options  # Currently unused but part of protocol interface
        table = Table(title="Coverage Report")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        # Add standard metrics
        if "total_requirements" in report:
            table.add_row("Total Requirements", str(report["total_requirements"]))
        if "covered_requirements" in report:
            table.add_row("Covered Requirements", str(report["covered_requirements"]))
        if "coverage_percentage" in report:
            pct = report["coverage_percentage"]
            color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
            table.add_row("Coverage", f"[{color}]{pct:.1f}%[/{color}]")

        # Add unlinked requirements if present
        if report.get("unlinked_requirements"):
            unlinked = report["unlinked_requirements"]
            table.add_row("Unlinked Requirements", str(len(unlinked)))

        return self._render_to_string(table)

    def format_summary(
        self, decisions: list[Decision], options: FormatOptions | None = None
    ) -> str:
        """Format summary statistics for decisions.

        Args:
            decisions: List of decisions to summarize.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string with counts by type, severity, agent, etc.
        """
        if not decisions:
            return self._render_to_string("[dim]No decisions to summarize.[/dim]")

        # Count by type
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_agent: dict[str, int] = {}

        for decision in decisions:
            by_type[decision.decision_type] = by_type.get(decision.decision_type, 0) + 1
            by_severity[decision.severity] = by_severity.get(decision.severity, 0) + 1
            by_agent[decision.agent.agent_type] = by_agent.get(decision.agent.agent_type, 0) + 1

        # Create summary table
        table = Table(title=f"Decision Summary (Total: {len(decisions)})")
        table.add_column("Category", style="bold")
        table.add_column("Breakdown")

        # Type breakdown
        type_parts = [f"{t}: {c}" for t, c in sorted(by_type.items())]
        table.add_row("By Type", ", ".join(type_parts))

        # Severity breakdown
        severity_parts = []
        for sev, count in sorted(by_severity.items()):
            color = self._get_severity_color(sev)
            severity_parts.append(f"[{color}]{sev}: {count}[/{color}]")
        table.add_row("By Severity", ", ".join(severity_parts))

        # Agent breakdown
        agent_parts = []
        for agent, count in sorted(by_agent.items()):
            color = self._get_agent_color(agent)
            agent_parts.append(f"[{color}]{agent}: {count}[/{color}]")
        table.add_row("By Agent", ", ".join(agent_parts))

        return self._render_to_string(table)
