"""JSON exporter for audit data (Story 11.4).

This module provides JSON export functionality for audit trails.

The JsonAuditExporter produces machine-readable JSON with:
- Full fidelity data preservation
- Consistent key ordering
- UTF-8 encoding
- Support for redaction of sensitive data

Example:
    >>> from yolo_developer.audit.json_exporter import JsonAuditExporter
    >>> from yolo_developer.audit.export_types import ExportOptions, RedactionConfig
    >>>
    >>> exporter = JsonAuditExporter()
    >>> decisions = [...]  # list of Decision objects
    >>> json_bytes = exporter.export_decisions(decisions)
    >>>
    >>> # With redaction
    >>> redaction = RedactionConfig(redact_metadata=True)
    >>> options = ExportOptions(redaction_config=redaction)
    >>> json_bytes = exporter.export_decisions(decisions, options=options)

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (JSON)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from yolo_developer.audit.export_types import (
    DEFAULT_EXPORT_OPTIONS,
    ExportOptions,
    RedactionConfig,
)
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import Decision

REDACTED_PLACEHOLDER = "[REDACTED]"


class JsonAuditExporter:
    """JSON exporter for audit data.

    Exports audit trail data to JSON format with:
    - Readable indentation (2 spaces)
    - Consistent key ordering
    - UTF-8 encoding
    - Configurable redaction support

    Example:
        >>> exporter = JsonAuditExporter()
        >>> decisions = [decision1, decision2]
        >>> json_bytes = exporter.export_decisions(decisions)
    """

    def export_decisions(
        self, decisions: list[Decision], options: ExportOptions | None = None
    ) -> bytes:
        """Export decisions to JSON format.

        Args:
            decisions: List of decisions to export.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded JSON bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        exported_decisions = [
            self._apply_decision_redaction(d.to_dict(), redaction) for d in decisions
        ]

        data = {
            "decisions": exported_decisions,
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_decisions": len(decisions),
                "format": "json",
            },
        }

        return self._to_json_bytes(data)

    def export_traces(
        self,
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export traceability data to JSON format.

        Args:
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded JSON bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        exported_artifacts = [
            self._apply_artifact_redaction(a.to_dict(), redaction) for a in artifacts
        ]
        exported_links = [
            self._apply_link_redaction(link.to_dict(), redaction) for link in links
        ]

        data = {
            "artifacts": exported_artifacts,
            "links": exported_links,
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_artifacts": len(artifacts),
                "total_links": len(links),
                "format": "json",
            },
        }

        return self._to_json_bytes(data)

    def export_full_audit(
        self,
        decisions: list[Decision],
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export complete audit trail to JSON format.

        Args:
            decisions: List of decisions to export.
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            UTF-8 encoded JSON bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        exported_decisions = [
            self._apply_decision_redaction(d.to_dict(), redaction) for d in decisions
        ]
        exported_artifacts = [
            self._apply_artifact_redaction(a.to_dict(), redaction) for a in artifacts
        ]
        exported_links = [
            self._apply_link_redaction(link.to_dict(), redaction) for link in links
        ]

        data = {
            "decisions": exported_decisions,
            "artifacts": exported_artifacts,
            "links": exported_links,
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_decisions": len(decisions),
                "total_artifacts": len(artifacts),
                "total_links": len(links),
                "format": "json",
            },
        }

        return self._to_json_bytes(data)

    def get_file_extension(self) -> str:
        """Return the file extension for JSON format.

        Returns:
            File extension ".json".
        """
        return ".json"

    def get_content_type(self) -> str:
        """Return the MIME content type for JSON.

        Returns:
            MIME type "application/json".
        """
        return "application/json"

    def _to_json_bytes(self, data: dict[str, Any]) -> bytes:
        """Convert data to JSON bytes with consistent formatting.

        Args:
            data: Dictionary to serialize.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        json_str = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        return json_str.encode("utf-8")

    def _apply_decision_redaction(
        self, decision_dict: dict[str, Any], redaction: RedactionConfig
    ) -> dict[str, Any]:
        """Apply redaction to a decision dictionary.

        Args:
            decision_dict: Decision as dictionary.
            redaction: Redaction configuration.

        Returns:
            Decision dictionary with redactions applied.
        """
        result = decision_dict.copy()

        # Redact metadata
        if redaction.redact_metadata and "metadata" in result:
            result["metadata"] = REDACTED_PLACEHOLDER

        # Redact session_id
        if redaction.redact_session_ids and "agent" in result:
            result["agent"] = result["agent"].copy()
            result["agent"]["session_id"] = REDACTED_PLACEHOLDER

        # Redact custom field paths
        for field_path in redaction.redact_fields:
            result = self._redact_field_path(result, field_path)

        return result

    def _apply_artifact_redaction(
        self, artifact_dict: dict[str, Any], redaction: RedactionConfig
    ) -> dict[str, Any]:
        """Apply redaction to an artifact dictionary.

        Args:
            artifact_dict: Artifact as dictionary.
            redaction: Redaction configuration.

        Returns:
            Artifact dictionary with redactions applied.
        """
        result = artifact_dict.copy()

        if redaction.redact_metadata and "metadata" in result:
            result["metadata"] = REDACTED_PLACEHOLDER

        for field_path in redaction.redact_fields:
            result = self._redact_field_path(result, field_path)

        return result

    def _apply_link_redaction(
        self, link_dict: dict[str, Any], redaction: RedactionConfig
    ) -> dict[str, Any]:
        """Apply redaction to a link dictionary.

        Args:
            link_dict: Link as dictionary.
            redaction: Redaction configuration.

        Returns:
            Link dictionary with redactions applied.
        """
        result = link_dict.copy()

        if redaction.redact_metadata and "metadata" in result:
            result["metadata"] = REDACTED_PLACEHOLDER

        for field_path in redaction.redact_fields:
            result = self._redact_field_path(result, field_path)

        return result

    def _redact_field_path(
        self, data: dict[str, Any], field_path: str
    ) -> dict[str, Any]:
        """Redact a specific field path in the data.

        Args:
            data: Dictionary to redact.
            field_path: Dot-separated path (e.g., "context.sprint_id").

        Returns:
            Dictionary with field redacted.
        """
        if not field_path:
            return data

        parts = field_path.split(".")
        result = data.copy()

        if len(parts) == 1:
            if parts[0] in result:
                result[parts[0]] = REDACTED_PLACEHOLDER
        elif len(parts) == 2:
            parent, child = parts
            if parent in result and isinstance(result[parent], dict):
                result[parent] = result[parent].copy()
                if child in result[parent]:
                    result[parent][child] = REDACTED_PLACEHOLDER

        return result
