"""Protocol definition for audit exporters (Story 11.4).

This module defines the AuditExporter Protocol that all exporters must implement.

The Protocol pattern allows swapping exporters for different output formats:
- JSON for machine-readable, API integration
- CSV for spreadsheet analysis
- PDF for human-readable compliance documentation
- Future: XML, HTML, etc.

Example:
    >>> from yolo_developer.audit.export_protocol import AuditExporter
    >>>
    >>> class CustomExporter:
    ...     def export_decisions(self, decisions: list[Decision], options=None) -> bytes:
    ...         return b'{"decisions": []}'
    ...     # ... other methods
    >>>
    >>> exporter = CustomExporter()
    >>> isinstance(exporter, AuditExporter)
    True

References:
    - FR84: System can export audit trail for compliance reporting
    - Story 11.1: Decision types used for export
    - Story 11.2: Traceability types used for trace export
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from yolo_developer.audit.export_types import ExportOptions
    from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
    from yolo_developer.audit.types import Decision


@runtime_checkable
class AuditExporter(Protocol):
    """Protocol for audit trail exporters.

    Defines the interface for exporting audit data into various formats.
    Implementations target different output formats (JSON, CSV, PDF).

    Methods:
        export_decisions: Export a list of decisions
        export_traces: Export traceability data (artifacts and links)
        export_full_audit: Export complete audit trail (decisions + traces)
        get_file_extension: Return the appropriate file extension for the format
        get_content_type: Return the MIME content type for HTTP responses

    Example:
        >>> class JsonExporter:
        ...     def export_decisions(self, decisions, options=None) -> bytes:
        ...         # JSON export implementation
        ...         ...
        >>>
        >>> exporter: AuditExporter = JsonExporter()
    """

    def export_decisions(
        self, decisions: list[Decision], options: ExportOptions | None = None
    ) -> bytes:
        """Export a list of decisions.

        Args:
            decisions: List of decisions to export.
            options: Optional export options (uses defaults if None).

        Returns:
            Bytes containing the exported data in the appropriate format.
        """
        ...

    def export_traces(
        self,
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export traceability data (artifacts and links).

        Args:
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            Bytes containing the exported traceability data.
        """
        ...

    def export_full_audit(
        self,
        decisions: list[Decision],
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export complete audit trail (decisions + traces).

        Args:
            decisions: List of decisions to export.
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            Bytes containing the complete exported audit trail.
        """
        ...

    def get_file_extension(self) -> str:
        """Return the appropriate file extension for the export format.

        Returns:
            File extension including the dot (e.g., ".json", ".csv", ".pdf").
        """
        ...

    def get_content_type(self) -> str:
        """Return the MIME content type for HTTP responses.

        Returns:
            MIME content type string (e.g., "application/json", "text/csv").
        """
        ...
