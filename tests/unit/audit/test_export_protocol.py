"""Tests for export protocol definition (Story 11.4 - Task 2).

Tests the AuditExporter Protocol definition for the audit export functionality.

References:
    - FR84: System can export audit trail for compliance reporting
    - ADR-001: Protocol pattern for pluggable implementations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from yolo_developer.audit.export_types import ExportOptions
    from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
    from yolo_developer.audit.types import Decision


class TestAuditExporterProtocol:
    """Tests for AuditExporter Protocol definition."""

    def test_audit_exporter_protocol_exists(self) -> None:
        """Test AuditExporter Protocol is defined."""
        from yolo_developer.audit.export_protocol import AuditExporter

        assert AuditExporter is not None

    def test_audit_exporter_protocol_is_runtime_checkable(self) -> None:
        """Test AuditExporter Protocol is runtime checkable."""
        from yolo_developer.audit.export_protocol import AuditExporter

        # Runtime checkable protocols can be used with isinstance
        assert hasattr(AuditExporter, "__protocol_attrs__") or hasattr(
            AuditExporter, "_is_runtime_protocol"
        )

    def test_conforming_class_is_instance_of_protocol(self) -> None:
        """Test class implementing protocol methods is recognized."""
        from yolo_developer.audit.export_protocol import AuditExporter
        from yolo_developer.audit.export_types import ExportOptions
        from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
        from yolo_developer.audit.types import Decision

        class MockExporter:
            """Mock exporter implementing all protocol methods."""

            def export_decisions(
                self, decisions: list[Decision], options: ExportOptions | None = None
            ) -> bytes:
                return b"decisions"

            def export_traces(
                self,
                artifacts: list[TraceableArtifact],
                links: list[TraceLink],
                options: ExportOptions | None = None,
            ) -> bytes:
                return b"traces"

            def export_full_audit(
                self,
                decisions: list[Decision],
                artifacts: list[TraceableArtifact],
                links: list[TraceLink],
                options: ExportOptions | None = None,
            ) -> bytes:
                return b"full"

            def get_file_extension(self) -> str:
                return ".json"

            def get_content_type(self) -> str:
                return "application/json"

        exporter = MockExporter()
        assert isinstance(exporter, AuditExporter)

    def test_non_conforming_class_is_not_instance(self) -> None:
        """Test class missing methods is not recognized as protocol."""
        from yolo_developer.audit.export_protocol import AuditExporter

        class IncompleteExporter:
            """Exporter missing required methods."""

            def export_decisions(self, decisions: list) -> bytes:
                return b"decisions"

            # Missing other required methods

        exporter = IncompleteExporter()
        # Should NOT be recognized as AuditExporter due to missing methods
        assert not isinstance(exporter, AuditExporter)

    def test_protocol_has_export_decisions_method(self) -> None:
        """Test protocol defines export_decisions method."""
        from yolo_developer.audit.export_protocol import AuditExporter

        # Check protocol has expected method
        assert hasattr(AuditExporter, "export_decisions")

    def test_protocol_has_export_traces_method(self) -> None:
        """Test protocol defines export_traces method."""
        from yolo_developer.audit.export_protocol import AuditExporter

        assert hasattr(AuditExporter, "export_traces")

    def test_protocol_has_export_full_audit_method(self) -> None:
        """Test protocol defines export_full_audit method."""
        from yolo_developer.audit.export_protocol import AuditExporter

        assert hasattr(AuditExporter, "export_full_audit")

    def test_protocol_has_get_file_extension_method(self) -> None:
        """Test protocol defines get_file_extension method."""
        from yolo_developer.audit.export_protocol import AuditExporter

        assert hasattr(AuditExporter, "get_file_extension")

    def test_protocol_has_get_content_type_method(self) -> None:
        """Test protocol defines get_content_type method."""
        from yolo_developer.audit.export_protocol import AuditExporter

        assert hasattr(AuditExporter, "get_content_type")
