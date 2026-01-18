"""Protocol definition for audit formatters (Story 11.3).

This module defines the AuditFormatter Protocol that all formatters must implement.

The Protocol pattern allows swapping formatters for different output contexts:
- Rich for interactive terminal
- Plain for logging/files
- Future: HTML, JSON

Example:
    >>> from yolo_developer.audit.formatter_protocol import AuditFormatter
    >>>
    >>> class CustomFormatter:
    ...     def format_decision(self, decision: Decision) -> str:
    ...         return f"Decision: {decision.content}"
    ...     # ... other methods
    >>>
    >>> formatter = CustomFormatter()
    >>> isinstance(formatter, AuditFormatter)
    True

References:
    - FR83: Users can view audit trail in human-readable format
    - Story 11.1: Decision types used for formatting
    - Story 11.2: Traceability types used for trace chain formatting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.formatter_types import FormatOptions
    from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
    from yolo_developer.audit.types import Decision


@runtime_checkable
class AuditFormatter(Protocol):
    """Protocol for audit trail formatters.

    Defines the interface for formatting audit data into human-readable output.
    Implementations can target different output contexts (terminal, file, HTML).

    Methods:
        format_decision: Format a single decision
        format_decisions: Format a list of decisions chronologically
        format_trace_chain: Format a trace chain of artifacts and links
        format_coverage_report: Format coverage statistics
        format_summary: Format summary statistics for decisions

    Example:
        >>> class RichFormatter:
        ...     def format_decision(self, decision: Decision) -> str:
        ...         # Rich terminal formatting
        ...         ...
        >>>
        >>> formatter: AuditFormatter = RichFormatter()
    """

    def format_decision(self, decision: Decision, options: FormatOptions | None = None) -> str:
        """Format a single decision for display.

        Args:
            decision: The decision to format.
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of the decision.
        """
        ...

    def format_decisions(
        self, decisions: list[Decision], options: FormatOptions | None = None
    ) -> str:
        """Format a list of decisions chronologically.

        Args:
            decisions: List of decisions to format (will be sorted by timestamp).
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of all decisions.
        """
        ...

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
            Formatted string representation of the trace chain (e.g., tree view).
        """
        ...

    def format_coverage_report(
        self, report: dict[str, Any], options: FormatOptions | None = None
    ) -> str:
        """Format coverage statistics.

        Args:
            report: Coverage report dictionary from TraceabilityService.get_coverage_report().
            options: Optional format options (uses defaults if None).

        Returns:
            Formatted string representation of coverage statistics.
        """
        ...

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
        ...
