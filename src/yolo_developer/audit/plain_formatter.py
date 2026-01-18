"""Plain text formatter for audit views (Story 11.3).

This module provides a plain text implementation of the AuditFormatter
protocol for non-terminal contexts like logging and file output.

Example:
    >>> from yolo_developer.audit.plain_formatter import PlainAuditFormatter
    >>>
    >>> formatter = PlainAuditFormatter()
    >>> output = formatter.format_decision(decision)
    >>> print(output)

References:
    - FR83: Users can view audit trail in human-readable format
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yolo_developer.audit.formatter_types import (
    DEFAULT_FORMAT_OPTIONS,
    FormatOptions,
)

if TYPE_CHECKING:
    from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
    from yolo_developer.audit.types import Decision


class PlainAuditFormatter:
    """Plain text formatter for audit views.

    Uses ASCII formatting for non-terminal contexts like logging and files.
    Uses indentation, dashes, and bracketed labels for structure.

    Example:
        >>> formatter = PlainAuditFormatter()
        >>> output = formatter.format_decision(decision)
    """

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[: max_length - 3] + "..."

    def _get_severity_marker(self, severity: str) -> str:
        """Get ASCII marker for severity level."""
        markers = {
            "critical": "[CRITICAL]",
            "warning": "[WARNING]",
            "info": "[INFO]",
        }
        return markers.get(severity, "[INFO]")

    def format_decision(self, decision: Decision, options: FormatOptions | None = None) -> str:
        """Format a single decision for display.

        Args:
            decision: The decision to format.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted plain text string representation of the decision.
        """
        opts = options or DEFAULT_FORMAT_OPTIONS
        severity_marker = self._get_severity_marker(decision.severity)

        lines = [
            "=" * 60,
            f"Decision: {decision.id}",
            "=" * 60,
            f"Type:      {decision.decision_type}",
            f"Severity:  {severity_marker}",
            f"Agent:     {decision.agent.agent_name} ({decision.agent.agent_type})",
            f"Timestamp: {decision.timestamp}",
            "",
            "Content:",
            f"  {self._truncate_content(decision.content, opts.max_content_length)}",
            "",
            "Rationale:",
            f"  {self._truncate_content(decision.rationale, opts.max_content_length)}",
        ]

        # Add context
        if decision.context.sprint_id or decision.context.story_id:
            lines.append("")
            lines.append("Context:")
            if decision.context.sprint_id:
                lines.append(f"  Sprint: {decision.context.sprint_id}")
            if decision.context.story_id:
                lines.append(f"  Story: {decision.context.story_id}")

        # Add metadata for verbose
        if opts.style == "verbose" and opts.show_metadata and decision.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in decision.metadata.items():
                lines.append(f"  {key}: {value}")

        # Add trace links for verbose
        if opts.style == "verbose" and opts.show_trace_links:
            if decision.context.trace_links:
                lines.append("")
                lines.append("Trace Links:")
                for link_id in decision.context.trace_links:
                    lines.append(f"  - {link_id}")

        lines.append("")
        return "\n".join(lines)

    def format_decisions(
        self, decisions: list[Decision], options: FormatOptions | None = None
    ) -> str:
        """Format a list of decisions chronologically.

        Args:
            decisions: List of decisions to format.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted plain text string representation of all decisions.
        """
        if not decisions:
            return "No decisions to display."

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
        """Format a trace chain of artifacts and links.

        Args:
            artifacts: List of artifacts in the trace chain.
            links: List of links connecting the artifacts.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted plain text tree representation of the trace chain.
        """
        # Note: options parameter available for future style customization
        _ = options  # Currently unused but part of protocol interface
        if not artifacts:
            return "No artifacts in trace chain."

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
            roots = artifacts[:1]

        lines = ["Trace Chain", "-" * 40]

        def add_children(artifact_id: str, indent: int, visited: set[str]) -> None:
            if artifact_id in visited:
                return
            visited.add(artifact_id)

            artifact = artifact_map.get(artifact_id)
            if artifact:
                prefix = "  " * indent + "+-- " if indent > 0 else ""
                lines.append(f"{prefix}[{artifact.artifact_type}] {artifact.name} ({artifact.id})")
                for child_id in children.get(artifact_id, []):
                    add_children(child_id, indent + 1, visited)

        visited: set[str] = set()
        for root in roots:
            add_children(root.id, 0, visited)

        # Add any orphan artifacts
        for artifact in artifacts:
            if artifact.id not in visited:
                lines.append(
                    f"(unlinked) [{artifact.artifact_type}] {artifact.name} ({artifact.id})"
                )

        return "\n".join(lines)

    def format_coverage_report(
        self, report: dict[str, Any], options: FormatOptions | None = None
    ) -> str:
        """Format coverage statistics.

        Args:
            report: Coverage report dictionary.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted plain text representation of coverage statistics.
        """
        # Note: options parameter available for future style customization
        _ = options  # Currently unused but part of protocol interface
        lines = ["Coverage Report", "=" * 40]

        if "total_requirements" in report:
            lines.append(f"Total Requirements:   {report['total_requirements']}")
        if "covered_requirements" in report:
            lines.append(f"Covered Requirements: {report['covered_requirements']}")
        if "coverage_percentage" in report:
            pct = report["coverage_percentage"]
            lines.append(f"Coverage:             {pct:.1f}%")

        if report.get("unlinked_requirements"):
            lines.append("")
            lines.append("Unlinked Requirements:")
            for req_id in report["unlinked_requirements"]:
                lines.append(f"  - {req_id}")

        return "\n".join(lines)

    def format_summary(
        self, decisions: list[Decision], options: FormatOptions | None = None
    ) -> str:
        """Format summary statistics for decisions.

        Args:
            decisions: List of decisions to summarize.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted plain text string with counts by type, severity, agent, etc.
        """
        if not decisions:
            return "No decisions to summarize."

        # Count by type
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_agent: dict[str, int] = {}

        for decision in decisions:
            by_type[decision.decision_type] = by_type.get(decision.decision_type, 0) + 1
            by_severity[decision.severity] = by_severity.get(decision.severity, 0) + 1
            by_agent[decision.agent.agent_type] = by_agent.get(decision.agent.agent_type, 0) + 1

        lines = [
            f"Decision Summary (Total: {len(decisions)})",
            "=" * 40,
            "",
            "By Type:",
        ]

        for t, c in sorted(by_type.items()):
            lines.append(f"  {t}: {c}")

        lines.append("")
        lines.append("By Severity:")
        for sev, count in sorted(by_severity.items()):
            marker = self._get_severity_marker(sev)
            lines.append(f"  {marker} {sev}: {count}")

        lines.append("")
        lines.append("By Agent:")
        for agent, count in sorted(by_agent.items()):
            lines.append(f"  {agent}: {count}")

        return "\n".join(lines)
