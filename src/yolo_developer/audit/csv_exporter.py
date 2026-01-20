"""CSV exporter for audit data (Story 11.4).

This module provides CSV export functionality for audit trails.

The CsvAuditExporter produces spreadsheet-compatible CSV with:
- UTF-8 BOM for Excel compatibility
- Flattened nested objects into separate columns
- Consistent column ordering
- Support for redaction of sensitive data

Example:
    >>> from yolo_developer.audit.csv_exporter import CsvAuditExporter
    >>> from yolo_developer.audit.export_types import ExportOptions, RedactionConfig
    >>>
    >>> exporter = CsvAuditExporter()
    >>> decisions = [...]  # list of Decision objects
    >>> csv_bytes = exporter.export_decisions(decisions)
    >>>
    >>> # With redaction
    >>> redaction = RedactionConfig(redact_session_ids=True)
    >>> options = ExportOptions(redaction_config=redaction)
    >>> csv_bytes = exporter.export_decisions(decisions, options=options)

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (CSV)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import csv
import io
import json

from yolo_developer.audit.export_types import (
    DEFAULT_EXPORT_OPTIONS,
    ExportOptions,
    RedactionConfig,
)
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import Decision

REDACTED_PLACEHOLDER = "[REDACTED]"

# Consistent column ordering for decisions
DECISION_COLUMNS = [
    "id",
    "decision_type",
    "content",
    "rationale",
    "timestamp",
    "severity",
    "agent_name",
    "agent_type",
    "session_id",
    "sprint_id",
    "story_id",
    "artifact_id",
    "parent_decision_id",
    "trace_links",
    "metadata",
]

# Consistent column ordering for artifacts
ARTIFACT_COLUMNS = [
    "id",
    "artifact_type",
    "name",
    "description",
    "created_at",
    "metadata",
]

# Consistent column ordering for links
LINK_COLUMNS = [
    "id",
    "source_id",
    "source_type",
    "target_id",
    "target_type",
    "link_type",
    "created_at",
    "metadata",
]


class CsvAuditExporter:
    """CSV exporter for audit data.

    Exports audit trail data to CSV format with:
    - UTF-8 BOM for Excel compatibility
    - Flattened nested structures
    - Consistent column ordering
    - Configurable redaction support

    Example:
        >>> exporter = CsvAuditExporter()
        >>> decisions = [decision1, decision2]
        >>> csv_bytes = exporter.export_decisions(decisions)
    """

    def export_decisions(
        self, decisions: list[Decision], options: ExportOptions | None = None
    ) -> bytes:
        """Export decisions to CSV format.

        Args:
            decisions: List of decisions to export.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded CSV bytes with BOM.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        rows = [self._flatten_decision(d, redaction) for d in decisions]

        return self._to_csv_bytes(rows, DECISION_COLUMNS)

    def export_traces(
        self,
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export traceability data to CSV format.

        Creates a combined CSV with sections for artifacts and links.

        Args:
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded CSV bytes with BOM.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        output = io.StringIO()

        # Write artifacts section
        output.write("# ARTIFACTS\n")
        artifact_rows = [self._flatten_artifact(a, redaction) for a in artifacts]
        self._write_csv_section(output, artifact_rows, ARTIFACT_COLUMNS)

        output.write("\n# LINKS\n")
        link_rows = [self._flatten_link(link, redaction) for link in links]
        self._write_csv_section(output, link_rows, LINK_COLUMNS)

        # Convert to bytes with BOM
        csv_str = output.getvalue()
        return ("\ufeff" + csv_str).encode("utf-8")

    def export_full_audit(
        self,
        decisions: list[Decision],
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export complete audit trail to CSV format.

        Creates a combined CSV with sections for decisions, artifacts, and links.

        Args:
            decisions: List of decisions to export.
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded CSV bytes with BOM.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        output = io.StringIO()

        # Write decisions section
        output.write("# DECISIONS\n")
        decision_rows = [self._flatten_decision(d, redaction) for d in decisions]
        self._write_csv_section(output, decision_rows, DECISION_COLUMNS)

        # Write artifacts section
        output.write("\n# ARTIFACTS\n")
        artifact_rows = [self._flatten_artifact(a, redaction) for a in artifacts]
        self._write_csv_section(output, artifact_rows, ARTIFACT_COLUMNS)

        # Write links section
        output.write("\n# LINKS\n")
        link_rows = [self._flatten_link(link, redaction) for link in links]
        self._write_csv_section(output, link_rows, LINK_COLUMNS)

        # Convert to bytes with BOM
        csv_str = output.getvalue()
        return ("\ufeff" + csv_str).encode("utf-8")

    def get_file_extension(self) -> str:
        """Return the file extension for CSV format.

        Returns:
            File extension ".csv".
        """
        return ".csv"

    def get_content_type(self) -> str:
        """Return the MIME content type for CSV.

        Returns:
            MIME type "text/csv".
        """
        return "text/csv"

    def _to_csv_bytes(self, rows: list[dict[str, str]], columns: list[str]) -> bytes:
        """Convert rows to CSV bytes with BOM.

        Args:
            rows: List of row dictionaries.
            columns: Column names in order.

        Returns:
            UTF-8 encoded CSV bytes with BOM.
        """
        output = io.StringIO()
        self._write_csv_section(output, rows, columns)
        csv_str = output.getvalue()
        # Add UTF-8 BOM for Excel compatibility
        return ("\ufeff" + csv_str).encode("utf-8")

    def _write_csv_section(
        self,
        output: io.StringIO,
        rows: list[dict[str, str]],
        columns: list[str],
    ) -> None:
        """Write a CSV section to the output.

        Args:
            output: StringIO to write to.
            rows: List of row dictionaries.
            columns: Column names in order.
        """
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def _flatten_decision(self, decision: Decision, redaction: RedactionConfig) -> dict[str, str]:
        """Flatten a Decision into a flat dictionary for CSV.

        Args:
            decision: Decision to flatten.
            redaction: Redaction configuration.

        Returns:
            Flat dictionary with string values.
        """
        d = decision.to_dict()

        # Flatten agent
        agent = d.get("agent", {})
        session_id = agent.get("session_id", "")
        if redaction.redact_session_ids:
            session_id = REDACTED_PLACEHOLDER

        # Flatten context
        context = d.get("context", {})
        sprint_id = context.get("sprint_id", "")
        story_id = context.get("story_id", "")

        # Apply custom field redaction to context
        for field_path in redaction.redact_fields:
            if field_path == "context.sprint_id":
                sprint_id = REDACTED_PLACEHOLDER
            elif field_path == "context.story_id":
                story_id = REDACTED_PLACEHOLDER

        # Handle metadata
        metadata = d.get("metadata", {})
        if redaction.redact_metadata:
            metadata_str = REDACTED_PLACEHOLDER
        else:
            metadata_str = json.dumps(metadata) if metadata else ""

        # Handle trace_links
        trace_links = context.get("trace_links", [])
        trace_links_str = ",".join(trace_links) if trace_links else ""

        return {
            "id": d.get("id", ""),
            "decision_type": d.get("decision_type", ""),
            "content": d.get("content", ""),
            "rationale": d.get("rationale", ""),
            "timestamp": d.get("timestamp", ""),
            "severity": d.get("severity", ""),
            "agent_name": agent.get("agent_name", ""),
            "agent_type": agent.get("agent_type", ""),
            "session_id": session_id,
            "sprint_id": sprint_id,
            "story_id": story_id,
            "artifact_id": context.get("artifact_id", "") or "",
            "parent_decision_id": context.get("parent_decision_id", "") or "",
            "trace_links": trace_links_str,
            "metadata": metadata_str,
        }

    def _flatten_artifact(
        self, artifact: TraceableArtifact, redaction: RedactionConfig
    ) -> dict[str, str]:
        """Flatten a TraceableArtifact into a flat dictionary for CSV.

        Args:
            artifact: Artifact to flatten.
            redaction: Redaction configuration.

        Returns:
            Flat dictionary with string values.
        """
        d = artifact.to_dict()

        metadata = d.get("metadata", {})
        if redaction.redact_metadata:
            metadata_str = REDACTED_PLACEHOLDER
        else:
            metadata_str = json.dumps(metadata) if metadata else ""

        return {
            "id": d.get("id", ""),
            "artifact_type": d.get("artifact_type", ""),
            "name": d.get("name", ""),
            "description": d.get("description", ""),
            "created_at": d.get("created_at", ""),
            "metadata": metadata_str,
        }

    def _flatten_link(self, link: TraceLink, redaction: RedactionConfig) -> dict[str, str]:
        """Flatten a TraceLink into a flat dictionary for CSV.

        Args:
            link: Link to flatten.
            redaction: Redaction configuration.

        Returns:
            Flat dictionary with string values.
        """
        d = link.to_dict()

        metadata = d.get("metadata", {})
        if redaction.redact_metadata:
            metadata_str = REDACTED_PLACEHOLDER
        else:
            metadata_str = json.dumps(metadata) if metadata else ""

        return {
            "id": d.get("id", ""),
            "source_id": d.get("source_id", ""),
            "source_type": d.get("source_type", ""),
            "target_id": d.get("target_id", ""),
            "target_type": d.get("target_type", ""),
            "link_type": d.get("link_type", ""),
            "created_at": d.get("created_at", ""),
            "metadata": metadata_str,
        }
