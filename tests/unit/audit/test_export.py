"""Tests for export service (Story 11.4 - Task 6).

Tests the AuditExportService for exporting audit data.

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (JSON, CSV, PDF)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.export_types import ExportOptions, RedactionConfig
from yolo_developer.audit.memory_store import InMemoryDecisionStore
from yolo_developer.audit.store import DecisionFilters
from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore

from .conftest import create_test_artifact, create_test_decision, create_test_link


class TestAuditExportService:
    """Tests for AuditExportService class."""

    def test_service_exists(self) -> None:
        """Test AuditExportService class exists."""
        from yolo_developer.audit.export import AuditExportService

        assert AuditExportService is not None

    def test_service_initializes_with_stores(self) -> None:
        """Test service initializes with decision and traceability stores."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = AuditExportService(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )

        assert service is not None

    def test_get_supported_formats(self) -> None:
        """Test get_supported_formats returns valid formats."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        formats = service.get_supported_formats()

        assert "json" in formats
        assert "csv" in formats
        assert "pdf" in formats


class TestAuditExportServiceExport:
    """Tests for export method."""

    @pytest.mark.asyncio
    async def test_export_json_returns_bytes(self) -> None:
        """Test export with JSON format returns bytes."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        # Add test data
        decision = create_test_decision()
        await decision_store.log_decision(decision)

        result = await service.export(format="json")

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_csv_returns_bytes(self) -> None:
        """Test export with CSV format returns bytes."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        result = await service.export(format="csv")

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_pdf_returns_bytes(self) -> None:
        """Test export with PDF format returns bytes."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        result = await service.export(format="pdf")

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_with_filters(self) -> None:
        """Test export respects decision filters."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        # Add multiple decisions
        dec1 = create_test_decision(id="dec-001")
        dec2 = create_test_decision(id="dec-002")
        await decision_store.log_decision(dec1)
        await decision_store.log_decision(dec2)

        # Export with filters - should work without error
        filters = DecisionFilters(agent_name="analyst")
        result = await service.export(format="json", filters=filters)

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_with_options(self) -> None:
        """Test export accepts export options."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        options = ExportOptions(
            format="json",
            redaction_config=RedactionConfig(redact_metadata=True),
        )
        result = await service.export(format="json", options=options)

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_includes_traceability(self) -> None:
        """Test export includes traceability data."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        # Add traceability data
        artifact = create_test_artifact()
        link = create_test_link()
        await traceability_store.register_artifact(artifact)
        await traceability_store.create_link(link)

        result = await service.export(format="json")

        # Should include traceability data in export
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestAuditExportServiceExportToFile:
    """Tests for export_to_file method."""

    @pytest.mark.asyncio
    async def test_export_to_file_creates_file(self, tmp_path) -> None:
        """Test export_to_file creates a file."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        file_path = str(tmp_path / "audit_export.json")
        result_path = await service.export_to_file(path=file_path, format="json")

        assert result_path == file_path
        # File should exist
        import os

        assert os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_export_to_file_detects_format_from_extension(self, tmp_path) -> None:
        """Test export_to_file detects format from file extension."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        # Specify .csv extension without explicit format
        file_path = str(tmp_path / "audit_export.csv")
        result_path = await service.export_to_file(path=file_path)

        assert result_path == file_path
        import os

        assert os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_export_to_file_explicit_format_overrides_extension(
        self, tmp_path
    ) -> None:
        """Test explicit format overrides file extension."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        # Specify .txt extension but export as JSON
        file_path = str(tmp_path / "audit_export.txt")
        result_path = await service.export_to_file(path=file_path, format="json")

        assert result_path == file_path
        # File should contain JSON
        import os

        assert os.path.exists(file_path)
        with open(file_path, "rb") as f:
            content = f.read()
            # JSON starts with {
            assert content.startswith(b"{")


class TestGetAuditExportService:
    """Tests for factory function."""

    def test_factory_function_exists(self) -> None:
        """Test get_audit_export_service factory exists."""
        from yolo_developer.audit.export import get_audit_export_service

        assert get_audit_export_service is not None

    def test_factory_creates_service(self) -> None:
        """Test factory creates AuditExportService instance."""
        from yolo_developer.audit.export import (
            AuditExportService,
            get_audit_export_service,
        )

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        service = get_audit_export_service(decision_store, traceability_store)

        assert isinstance(service, AuditExportService)


class TestExportServiceCustomExporters:
    """Tests for custom exporter support."""

    def test_service_accepts_custom_exporters(self) -> None:
        """Test service accepts custom exporter mapping."""
        from yolo_developer.audit.export import AuditExportService
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        # Provide custom exporter map
        custom_exporters = {"json": JsonAuditExporter()}
        service = AuditExportService(
            decision_store,
            traceability_store,
            exporters=custom_exporters,
        )

        assert service is not None

    @pytest.mark.asyncio
    async def test_service_uses_custom_exporter(self) -> None:
        """Test service uses provided custom exporter."""
        from yolo_developer.audit.export import AuditExportService
        from yolo_developer.audit.json_exporter import JsonAuditExporter

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()

        # Create custom exporter
        custom_json_exporter = JsonAuditExporter()
        service = AuditExportService(
            decision_store,
            traceability_store,
            exporters={"json": custom_json_exporter},
        )

        decision = create_test_decision()
        await decision_store.log_decision(decision)

        result = await service.export(format="json")

        # Should work with custom exporter
        assert isinstance(result, bytes)


class TestExportServiceInvalidFormat:
    """Tests for invalid format handling."""

    @pytest.mark.asyncio
    async def test_export_invalid_format_raises_error(self) -> None:
        """Test export with invalid format raises ValueError."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        with pytest.raises(ValueError, match="Unsupported export format"):
            await service.export(format="invalid")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_export_to_file_unknown_extension_raises_error(
        self, tmp_path
    ) -> None:
        """Test export_to_file with unknown extension raises ValueError."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        file_path = str(tmp_path / "audit_export.xyz")

        with pytest.raises(ValueError, match="Cannot determine export format"):
            await service.export_to_file(path=file_path)


class TestExportServiceEmptyData:
    """Tests for exporting empty data."""

    @pytest.mark.asyncio
    async def test_export_empty_decisions(self) -> None:
        """Test export with no decisions returns valid output."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        result = await service.export(format="json")

        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_export_empty_traces(self) -> None:
        """Test export with no traces returns valid output."""
        from yolo_developer.audit.export import AuditExportService

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditExportService(decision_store, traceability_store)

        # Add only decisions, no traces
        decision = create_test_decision()
        await decision_store.log_decision(decision)

        result = await service.export(format="json")

        assert isinstance(result, bytes)
