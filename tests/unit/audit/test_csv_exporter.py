"""Tests for CSV exporter (Story 11.4 - Task 4).

Tests the CsvAuditExporter for exporting audit data to CSV format.

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

from yolo_developer.audit.export_types import ExportOptions, RedactionConfig

from .conftest import create_test_artifact, create_test_decision, create_test_link


def parse_csv(data: bytes) -> list[dict[str, str]]:
    """Parse CSV bytes into list of dictionaries."""
    text = data.decode("utf-8-sig")  # Handle BOM
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


class TestCsvAuditExporter:
    """Tests for CsvAuditExporter class."""

    def test_exporter_exists(self) -> None:
        """Test CsvAuditExporter class exists."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        assert CsvAuditExporter is not None

    def test_exporter_implements_protocol(self) -> None:
        """Test CsvAuditExporter implements AuditExporter protocol."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter
        from yolo_developer.audit.export_protocol import AuditExporter

        exporter = CsvAuditExporter()
        assert isinstance(exporter, AuditExporter)

    def test_get_file_extension(self) -> None:
        """Test get_file_extension returns .csv."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        assert exporter.get_file_extension() == ".csv"

    def test_get_content_type(self) -> None:
        """Test get_content_type returns text/csv."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        assert exporter.get_content_type() == "text/csv"


class TestCsvExportDecisions:
    """Tests for export_decisions method."""

    def test_export_decisions_returns_bytes(self) -> None:
        """Test export_decisions returns bytes."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])

        assert isinstance(result, bytes)

    def test_export_decisions_produces_valid_csv(self) -> None:
        """Test export_decisions produces valid CSV."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        rows = parse_csv(result)

        assert len(rows) == 1

    def test_export_decisions_flattens_nested_fields(self) -> None:
        """Test export flattens nested objects into columns."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        rows = parse_csv(result)

        row = rows[0]
        # Flattened agent fields
        assert "agent_name" in row
        assert row["agent_name"] == "analyst"
        assert "agent_type" in row
        assert "session_id" in row
        # Flattened context fields
        assert "sprint_id" in row
        assert row["sprint_id"] == "sprint-1"
        assert "story_id" in row

    def test_export_decisions_includes_all_fields(self) -> None:
        """Test export includes all relevant decision fields (AC #2)."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()

        result = exporter.export_decisions([decision])
        rows = parse_csv(result)

        row = rows[0]
        assert row["id"] == "dec-001"
        assert row["decision_type"] == "requirement_analysis"
        assert row["content"] == "Test decision"
        assert row["rationale"] == "Test rationale"
        assert row["timestamp"] == "2026-01-18T12:00:00Z"
        assert row["severity"] == "info"

    def test_export_decisions_consistent_column_order(self) -> None:
        """Test column ordering is consistent."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decisions = [
            create_test_decision(id="dec-001"),
            create_test_decision(id="dec-002"),
        ]

        result1 = exporter.export_decisions(decisions)
        result2 = exporter.export_decisions(decisions)

        # Headers should be the same
        header1 = result1.decode("utf-8-sig").split("\n")[0]
        header2 = result2.decode("utf-8-sig").split("\n")[0]
        assert header1 == header2

    def test_export_decisions_empty_list(self) -> None:
        """Test export with empty list produces only header."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()

        result = exporter.export_decisions([])
        text = result.decode("utf-8-sig")
        lines = [line for line in text.strip().split("\n") if line]

        # Should have at least a header line
        assert len(lines) >= 1

    def test_export_decisions_multiple(self) -> None:
        """Test export with multiple decisions."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decisions = [
            create_test_decision(id="dec-001", content="First"),
            create_test_decision(id="dec-002", content="Second"),
            create_test_decision(id="dec-003", content="Third"),
        ]

        result = exporter.export_decisions(decisions)
        rows = parse_csv(result)

        assert len(rows) == 3


class TestCsvExportTraces:
    """Tests for export_traces method."""

    def test_export_traces_returns_bytes(self) -> None:
        """Test export_traces returns bytes."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_traces([artifact], [link])

        assert isinstance(result, bytes)

    def test_export_traces_produces_valid_csv(self) -> None:
        """Test export_traces produces valid CSV."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_traces([artifact], [link])
        text = result.decode("utf-8-sig")

        # Should contain sections for artifacts and links
        assert "ARTIFACTS" in text or "id" in text

    def test_export_traces_includes_artifact_fields(self) -> None:
        """Test export includes artifact fields."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        artifact = create_test_artifact()

        result = exporter.export_traces([artifact], [])
        text = result.decode("utf-8-sig")

        assert "art-001" in text
        assert "requirement" in text
        assert "Test Artifact" in text


class TestCsvExportFullAudit:
    """Tests for export_full_audit method."""

    def test_export_full_audit_returns_bytes(self) -> None:
        """Test export_full_audit returns bytes."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_full_audit([decision], [artifact], [link])

        assert isinstance(result, bytes)

    def test_export_full_audit_includes_all_sections(self) -> None:
        """Test export_full_audit includes decisions, artifacts, and links."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()
        artifact = create_test_artifact()
        link = create_test_link()

        result = exporter.export_full_audit([decision], [artifact], [link])
        text = result.decode("utf-8-sig")

        # Should contain data from all sections
        assert "dec-001" in text
        assert "art-001" in text
        assert "link-001" in text


class TestCsvExportRedaction:
    """Tests for redaction functionality."""

    def test_redact_metadata_replaces_with_redacted(self) -> None:
        """Test redact_metadata replaces metadata contents."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()
        redaction = RedactionConfig(redact_metadata=True)
        options = ExportOptions(redaction_config=redaction)

        result = exporter.export_decisions([decision], options=options)
        rows = parse_csv(result)

        row = rows[0]
        assert row.get("metadata", "") == "[REDACTED]"

    def test_redact_session_ids_replaces_with_redacted(self) -> None:
        """Test redact_session_ids replaces session IDs."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        decision = create_test_decision()
        redaction = RedactionConfig(redact_session_ids=True)
        options = ExportOptions(redaction_config=redaction)

        result = exporter.export_decisions([decision], options=options)
        rows = parse_csv(result)

        row = rows[0]
        assert row.get("session_id", "") == "[REDACTED]"


class TestCsvExportOptions:
    """Tests for options parameter handling."""

    def test_export_decisions_accepts_options(self) -> None:
        """Test export_decisions accepts options parameter."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        options = ExportOptions(format="csv")

        result = exporter.export_decisions([], options=options)
        assert isinstance(result, bytes)

    def test_export_traces_accepts_options(self) -> None:
        """Test export_traces accepts options parameter."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        options = ExportOptions(format="csv")

        result = exporter.export_traces([], [], options=options)
        assert isinstance(result, bytes)

    def test_export_full_audit_accepts_options(self) -> None:
        """Test export_full_audit accepts options parameter."""
        from yolo_developer.audit.csv_exporter import CsvAuditExporter

        exporter = CsvAuditExporter()
        options = ExportOptions(format="csv")

        result = exporter.export_full_audit([], [], [], options=options)
        assert isinstance(result, bytes)
