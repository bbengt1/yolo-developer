"""Export service for audit data (Story 11.4).

This module provides the AuditExportService for exporting audit trails
to various formats (JSON, CSV, PDF).

The service integrates with DecisionStore and TraceabilityStore to:
- Query decisions and traceability data
- Apply filters for targeted exports
- Support redaction for compliance requirements
- Export to files with format auto-detection

Example:
    >>> from yolo_developer.audit.export import get_audit_export_service
    >>> from yolo_developer.audit.memory_store import InMemoryDecisionStore
    >>> from yolo_developer.audit.traceability_memory_store import (
    ...     InMemoryTraceabilityStore,
    ... )
    >>>
    >>> decision_store = InMemoryDecisionStore()
    >>> traceability_store = InMemoryTraceabilityStore()
    >>> service = get_audit_export_service(decision_store, traceability_store)
    >>>
    >>> # Export all audit data as JSON
    >>> json_bytes = await service.export(format="json")
    >>>
    >>> # Export with filters and redaction
    >>> from yolo_developer.audit.store import DecisionFilters
    >>> from yolo_developer.audit.export_types import ExportOptions, RedactionConfig
    >>> filters = DecisionFilters(agent_name="analyst")
    >>> options = ExportOptions(redaction_config=RedactionConfig(redact_metadata=True))
    >>> json_bytes = await service.export(format="json", filters=filters, options=options)
    >>>
    >>> # Export to file (format auto-detected from extension)
    >>> await service.export_to_file("/path/to/audit_export.csv")

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (JSON, CSV, PDF)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import structlog

from yolo_developer.audit.csv_exporter import CsvAuditExporter
from yolo_developer.audit.export_protocol import AuditExporter
from yolo_developer.audit.export_types import (
    DEFAULT_EXPORT_OPTIONS,
    VALID_EXPORT_FORMATS,
    ExportFormat,
    ExportOptions,
)
from yolo_developer.audit.json_exporter import JsonAuditExporter
from yolo_developer.audit.pdf_exporter import PdfAuditExporter
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink

if TYPE_CHECKING:
    from yolo_developer.audit.store import DecisionFilters, DecisionStore
    from yolo_developer.audit.traceability_store import TraceabilityStore

_logger = structlog.get_logger(__name__)

# File extension to format mapping
EXTENSION_TO_FORMAT: dict[str, ExportFormat] = {
    ".json": "json",
    ".csv": "csv",
    ".pdf": "pdf",
}


class AuditExportService:
    """Service for exporting audit trail data.

    Provides high-level methods for exporting decisions and traceability
    data to various formats. Supports filtering, redaction, and file output.

    Attributes:
        _decision_store: Store for decision records.
        _traceability_store: Store for traceability data.
        _exporters: Mapping of format names to exporter instances.

    Example:
        >>> service = AuditExportService(decision_store, traceability_store)
        >>> json_bytes = await service.export(format="json")
        >>> await service.export_to_file("/path/to/export.csv")
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        traceability_store: TraceabilityStore,
        exporters: dict[str, AuditExporter] | None = None,
    ) -> None:
        """Initialize the export service.

        Args:
            decision_store: Store for retrieving decisions.
            traceability_store: Store for retrieving traceability data.
            exporters: Optional custom exporter mapping. If None, uses
                default exporters for JSON, CSV, and PDF.
        """
        self._decision_store = decision_store
        self._traceability_store = traceability_store

        # Initialize default exporters if not provided
        if exporters is not None:
            self._exporters: dict[str, AuditExporter] = exporters
        else:
            self._exporters = {
                "json": JsonAuditExporter(),
                "csv": CsvAuditExporter(),
                "pdf": PdfAuditExporter(),
            }

        _logger.debug(
            "AuditExportService initialized",
            supported_formats=list(self._exporters.keys()),
        )

    def get_supported_formats(self) -> list[ExportFormat]:
        """Get list of supported export formats.

        Returns:
            List of supported format names.
        """
        return [f for f in self._exporters.keys() if f in VALID_EXPORT_FORMATS]  # type: ignore[misc]

    async def export(
        self,
        format: ExportFormat,
        filters: DecisionFilters | None = None,
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export audit trail data to the specified format.

        Retrieves decisions and traceability data from stores, applies
        filters if provided, and exports to the requested format.

        Args:
            format: Export format ("json", "csv", or "pdf").
            filters: Optional filters for decision queries.
            options: Optional export options including redaction config.

        Returns:
            Exported data as bytes.

        Raises:
            ValueError: If format is not supported.
        """
        if format not in self._exporters:
            supported = ", ".join(self._exporters.keys())
            raise ValueError(
                f"Unsupported export format: {format}. Supported formats: {supported}"
            )

        options = options or DEFAULT_EXPORT_OPTIONS
        exporter = self._exporters[format]

        _logger.info(
            "Starting audit export",
            format=format,
            has_filters=filters is not None,
            redact_metadata=options.redaction_config.redact_metadata,
            redact_session_ids=options.redaction_config.redact_session_ids,
        )

        # Fetch data from stores
        decisions = await self._decision_store.get_decisions(filters)
        artifacts, links = await self._get_all_traceability_data()

        _logger.debug(
            "Fetched audit data",
            decision_count=len(decisions),
            artifact_count=len(artifacts),
            link_count=len(links),
        )

        # Export using full audit method
        result = exporter.export_full_audit(decisions, artifacts, links, options)

        _logger.info(
            "Audit export completed",
            format=format,
            output_size=len(result),
        )

        return result

    async def export_to_file(
        self,
        path: str,
        format: ExportFormat | None = None,
        filters: DecisionFilters | None = None,
        options: ExportOptions | None = None,
    ) -> str:
        """Export audit trail data to a file.

        If format is not specified, it is auto-detected from the file extension.

        Args:
            path: Output file path.
            format: Optional export format. If None, detected from extension.
            filters: Optional filters for decision queries.
            options: Optional export options including redaction config.

        Returns:
            The path to the created file.

        Raises:
            ValueError: If format cannot be determined from extension.
        """
        # Detect format from extension if not specified
        if format is None:
            _, ext = os.path.splitext(path)
            ext_lower = ext.lower()
            if ext_lower not in EXTENSION_TO_FORMAT:
                supported = ", ".join(EXTENSION_TO_FORMAT.keys())
                raise ValueError(
                    f"Cannot determine export format from extension: {ext}. "
                    f"Supported extensions: {supported}"
                )
            format = EXTENSION_TO_FORMAT[ext_lower]

        _logger.info(
            "Exporting audit trail to file",
            path=path,
            format=format,
        )

        # Export data
        data = await self.export(format=format, filters=filters, options=options)

        # Write to file
        with open(path, "wb") as f:
            f.write(data)

        _logger.info(
            "Audit export file created",
            path=path,
            size=len(data),
        )

        return path

    async def _get_all_traceability_data(
        self,
    ) -> tuple[list[TraceableArtifact], list[TraceLink]]:
        """Retrieve all traceability data from the store.

        Returns:
            Tuple of (artifacts, links).
        """
        # Get all artifacts by checking for common artifact types
        # Note: This is a simple implementation - in production, the store
        # might have a get_all_artifacts method
        artifacts: list[TraceableArtifact] = []
        links: list[TraceLink] = []

        # For InMemoryTraceabilityStore, we can access internal data
        # For protocol compliance, we use the public interface
        store = self._traceability_store

        # Attempt to get artifacts by type
        artifact_types = ["requirement", "story", "design_decision", "code", "test"]
        seen_artifact_ids: set[str] = set()

        for artifact_type in artifact_types:
            # Get unlinked artifacts of this type
            type_artifacts = await store.get_unlinked_artifacts(artifact_type)  # type: ignore[arg-type]
            for artifact in type_artifacts:
                if artifact.id not in seen_artifact_ids:
                    seen_artifact_ids.add(artifact.id)
                    artifacts.append(artifact)

        # Also try to get linked artifacts by traversing from unlinked ones
        # This ensures we capture all artifacts in the graph
        all_artifact_ids = set(seen_artifact_ids)
        for artifact in list(artifacts):
            # Get upstream and downstream artifacts
            upstream = await store.get_trace_chain(artifact.id, "upstream")
            downstream = await store.get_trace_chain(artifact.id, "downstream")

            for chain_artifact in upstream + downstream:
                if chain_artifact.id not in all_artifact_ids:
                    all_artifact_ids.add(chain_artifact.id)
                    artifacts.append(chain_artifact)

        # Get all links by checking links from each artifact
        seen_link_ids: set[str] = set()
        for artifact_id in all_artifact_ids:
            from_links = await store.get_links_from(artifact_id)
            to_links = await store.get_links_to(artifact_id)

            for link in from_links + to_links:
                if link.id not in seen_link_ids:
                    seen_link_ids.add(link.id)
                    links.append(link)

        return artifacts, links


def get_audit_export_service(
    decision_store: DecisionStore,
    traceability_store: TraceabilityStore,
    exporters: dict[str, AuditExporter] | None = None,
) -> AuditExportService:
    """Factory function to create an AuditExportService.

    Args:
        decision_store: Store for retrieving decisions.
        traceability_store: Store for retrieving traceability data.
        exporters: Optional custom exporter mapping.

    Returns:
        Configured AuditExportService instance.

    Example:
        >>> from yolo_developer.audit.memory_store import InMemoryDecisionStore
        >>> from yolo_developer.audit.traceability_memory_store import (
        ...     InMemoryTraceabilityStore,
        ... )
        >>>
        >>> decision_store = InMemoryDecisionStore()
        >>> traceability_store = InMemoryTraceabilityStore()
        >>> service = get_audit_export_service(decision_store, traceability_store)
    """
    return AuditExportService(
        decision_store=decision_store,
        traceability_store=traceability_store,
        exporters=exporters,
    )
