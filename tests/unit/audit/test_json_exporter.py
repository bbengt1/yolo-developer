"""Tests for JSON exporter (Story 11.4 - Task 3).

Tests the JsonAuditExporter for exporting audit data to JSON format.

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (JSON)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import json

from yolo_developer.audit.export_types import ExportOptions, RedactionConfig

from .conftest import create_test_artifact, create_test_decision, create_test_link


class TestJsonAuditExporter:
    """Tests for JsonAuditExporter class."""

    def test_exporter_exists(self) -> None:
        """Test JsonAuditExporter class exists."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        assert JsonAuditExporter is not None

    def test_exporter_implements_protocol(self) -> None:
        """Test JsonAuditExporter implements AuditExporter protocol."""
        from yolo_developer.audit.export_protocol import AuditExporter
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        assert isinstance(exporter, AuditExporter)

    def test_get_file_extension(self) -> None:
        """Test get_file_extension returns .json."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        assert exporter.get_file_extension() == ".json"

    def test_get_content_type(self) -> None:
        """Test get_content_type returns application/json."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        assert exporter.get_content_type() == "application/json"


class TestJsonExportDecisions:
    """Tests for export_decisions method."""

    def test_export_decisions_returns_bytes(self) -> None:
        """Test export_decisions returns bytes."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])

        assert isinstance(result, bytes)

    def test_export_decisions_produces_valid_json(self) -> None:
        """Test export_decisions produces valid JSON."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        data = json.loads(result.decode("utf-8"))

        assert isinstance(data, dict)
        assert "decisions" in data

    def test_export_decisions_includes_all_fields(self) -> None:
        """Test export includes all relevant decision fields (AC #2)."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        data = json.loads(result.decode("utf-8"))

        assert len(data["decisions"]) == 1
        dec = data["decisions"][0]
        assert dec["id"] == "dec-001"
        assert dec["decision_type"] == "requirement_analysis"
        assert dec["content"] == "Test decision"
        assert dec["rationale"] == "Test rationale"
        assert dec["timestamp"] == "2026-01-18T12:00:00Z"
        assert dec["severity"] == "info"
        assert "agent" in dec
        assert "context" in dec

    def test_export_decisions_includes_metadata_object(self) -> None:
        """Test export includes metadata with export info."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        data = json.loads(result.decode("utf-8"))

        assert "metadata" in data
        assert "export_timestamp" in data["metadata"]
        assert "total_decisions" in data["metadata"]
        assert data["metadata"]["total_decisions"] == 1

    def test_export_decisions_empty_list(self) -> None:
        """Test export with empty list produces valid JSON."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()

        result = exporter.export_decisions([])
        data = json.loads(result.decode("utf-8"))

        assert data["decisions"] == []
        assert data["metadata"]["total_decisions"] == 0

    def test_export_decisions_multiple(self) -> None:
        """Test export with multiple decisions."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decisions = [
            create_test_decision(id="dec-001", content="First"),
            create_test_decision(id="dec-002", content="Second"),
            create_test_decision(id="dec-003", content="Third"),
        ]

        result = exporter.export_decisions(decisions)
        data = json.loads(result.decode("utf-8"))

        assert len(data["decisions"]) == 3
        assert data["metadata"]["total_decisions"] == 3


class TestJsonExportTraces:
    """Tests for export_traces method."""

    def test_export_traces_returns_bytes(self) -> None:
        """Test export_traces returns bytes."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_traces([artifact], [link])

        assert isinstance(result, bytes)

    def test_export_traces_produces_valid_json(self) -> None:
        """Test export_traces produces valid JSON."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_traces([artifact], [link])
        data = json.loads(result.decode("utf-8"))

        assert isinstance(data, dict)
        assert "artifacts" in data
        assert "links" in data

    def test_export_traces_includes_all_artifact_fields(self) -> None:
        """Test export includes all artifact fields."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        artifact = create_test_artifact()

        result = exporter.export_traces([artifact], [])
        data = json.loads(result.decode("utf-8"))

        art = data["artifacts"][0]
        assert art["id"] == "art-001"
        assert art["artifact_type"] == "requirement"
        assert art["name"] == "Test Artifact"
        assert art["description"] == "Test description"
        assert art["created_at"] == "2026-01-18T12:00:00Z"

    def test_export_traces_includes_all_link_fields(self) -> None:
        """Test export includes all link fields."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        link = create_test_link()

        result = exporter.export_traces([], [link])
        data = json.loads(result.decode("utf-8"))

        lnk = data["links"][0]
        assert lnk["id"] == "link-001"
        assert lnk["source_id"] == "art-001"
        assert lnk["target_id"] == "art-002"
        assert lnk["link_type"] == "traces_to"


class TestJsonExportFullAudit:
    """Tests for export_full_audit method."""

    def test_export_full_audit_returns_bytes(self) -> None:
        """Test export_full_audit returns bytes."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_full_audit([decision], [artifact], [link])

        assert isinstance(result, bytes)

    def test_export_full_audit_includes_all_sections(self) -> None:
        """Test export_full_audit includes decisions, artifacts, and links."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_full_audit([decision], [artifact], [link])
        data = json.loads(result.decode("utf-8"))

        assert "decisions" in data
        assert "artifacts" in data
        assert "links" in data
        assert "metadata" in data
        assert len(data["decisions"]) == 1
        assert len(data["artifacts"]) == 1
        assert len(data["links"]) == 1


class TestJsonExportRedaction:
    """Tests for redaction functionality."""

    def test_redact_metadata_replaces_with_redacted(self) -> None:
        """Test redact_metadata replaces metadata contents."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()
        redaction = RedactionConfig(redact_metadata=True)
        options = ExportOptions(redaction_config=redaction)

        result = exporter.export_decisions([decision], options=options)
        data = json.loads(result.decode("utf-8"))

        dec = data["decisions"][0]
        assert dec["metadata"] == "[REDACTED]"

    def test_redact_session_ids_replaces_with_redacted(self) -> None:
        """Test redact_session_ids replaces session IDs."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()
        redaction = RedactionConfig(redact_session_ids=True)
        options = ExportOptions(redaction_config=redaction)

        result = exporter.export_decisions([decision], options=options)
        data = json.loads(result.decode("utf-8"))

        dec = data["decisions"][0]
        assert dec["agent"]["session_id"] == "[REDACTED]"

    def test_redact_custom_fields(self) -> None:
        """Test redacting custom field paths."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()
        redaction = RedactionConfig(redact_fields=("context.sprint_id", "context.story_id"))
        options = ExportOptions(redaction_config=redaction)

        result = exporter.export_decisions([decision], options=options)
        data = json.loads(result.decode("utf-8"))

        dec = data["decisions"][0]
        assert dec["context"]["sprint_id"] == "[REDACTED]"
        assert dec["context"]["story_id"] == "[REDACTED]"

    def test_no_redaction_by_default(self) -> None:
        """Test no redaction occurs with default options."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        data = json.loads(result.decode("utf-8"))

        dec = data["decisions"][0]
        assert dec["metadata"] == {"key": "value"}
        assert dec["agent"]["session_id"] == "session-123"
        assert dec["context"]["sprint_id"] == "sprint-1"


class TestJsonExportOptions:
    """Tests for options parameter handling."""

    def test_export_decisions_accepts_options(self) -> None:
        """Test export_decisions accepts options parameter."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        options = ExportOptions(format="json")

        result = exporter.export_decisions([], options=options)
        assert isinstance(result, bytes)

    def test_export_traces_accepts_options(self) -> None:
        """Test export_traces accepts options parameter."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        options = ExportOptions(format="json")

        result = exporter.export_traces([], [], options=options)
        assert isinstance(result, bytes)

    def test_export_full_audit_accepts_options(self) -> None:
        """Test export_full_audit accepts options parameter."""
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        exporter = JsonAuditExporter()
        options = ExportOptions(format="json")

        result = exporter.export_full_audit([], [], [], options=options)
        assert isinstance(result, bytes)
