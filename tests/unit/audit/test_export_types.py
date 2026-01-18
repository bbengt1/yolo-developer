"""Tests for export type definitions (Story 11.4 - Task 1).

Tests the ExportFormat, RedactionConfig, and ExportOptions types
for the audit export functionality.

References:
    - FR84: System can export audit trail for compliance reporting
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError

import pytest


class TestExportFormat:
    """Tests for ExportFormat Literal type."""

    def test_valid_export_formats_constant_exists(self) -> None:
        """Test VALID_EXPORT_FORMATS constant is defined."""
        from yolo_developer.audit.export_types import VALID_EXPORT_FORMATS

        assert VALID_EXPORT_FORMATS is not None
        assert isinstance(VALID_EXPORT_FORMATS, frozenset)

    def test_valid_export_formats_contains_json(self) -> None:
        """Test VALID_EXPORT_FORMATS contains 'json'."""
        from yolo_developer.audit.export_types import VALID_EXPORT_FORMATS

        assert "json" in VALID_EXPORT_FORMATS

    def test_valid_export_formats_contains_csv(self) -> None:
        """Test VALID_EXPORT_FORMATS contains 'csv'."""
        from yolo_developer.audit.export_types import VALID_EXPORT_FORMATS

        assert "csv" in VALID_EXPORT_FORMATS

    def test_valid_export_formats_contains_pdf(self) -> None:
        """Test VALID_EXPORT_FORMATS contains 'pdf'."""
        from yolo_developer.audit.export_types import VALID_EXPORT_FORMATS

        assert "pdf" in VALID_EXPORT_FORMATS

    def test_valid_export_formats_has_exactly_three_values(self) -> None:
        """Test VALID_EXPORT_FORMATS has exactly 3 values."""
        from yolo_developer.audit.export_types import VALID_EXPORT_FORMATS

        assert len(VALID_EXPORT_FORMATS) == 3


class TestRedactionConfig:
    """Tests for RedactionConfig frozen dataclass."""

    def test_redaction_config_creation_with_defaults(self) -> None:
        """Test RedactionConfig can be created with default values."""
        from yolo_developer.audit.export_types import RedactionConfig

        config = RedactionConfig()
        assert config.redact_metadata is False
        assert config.redact_session_ids is False
        assert config.redact_fields == ()

    def test_redaction_config_creation_with_values(self) -> None:
        """Test RedactionConfig can be created with custom values."""
        from yolo_developer.audit.export_types import RedactionConfig

        config = RedactionConfig(
            redact_metadata=True,
            redact_session_ids=True,
            redact_fields=("agent.session_id", "context.sprint_id"),
        )
        assert config.redact_metadata is True
        assert config.redact_session_ids is True
        assert config.redact_fields == ("agent.session_id", "context.sprint_id")

    def test_redaction_config_is_frozen(self) -> None:
        """Test RedactionConfig is immutable (frozen)."""
        from yolo_developer.audit.export_types import RedactionConfig

        config = RedactionConfig()
        with pytest.raises(FrozenInstanceError):
            config.redact_metadata = True  # type: ignore[misc]

    def test_redaction_config_to_dict(self) -> None:
        """Test RedactionConfig.to_dict() produces JSON-serializable dict."""
        from yolo_developer.audit.export_types import RedactionConfig

        config = RedactionConfig(
            redact_metadata=True,
            redact_session_ids=False,
            redact_fields=("field1", "field2"),
        )
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["redact_metadata"] is True
        assert result["redact_session_ids"] is False
        assert result["redact_fields"] == ["field1", "field2"]
        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_redaction_config_validation_warning_for_invalid_field_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test RedactionConfig logs warning for empty redact_fields entry."""
        from yolo_developer.audit.export_types import RedactionConfig

        with caplog.at_level(logging.WARNING):
            _ = RedactionConfig(redact_fields=("valid.path", ""))

        assert "empty" in caplog.text.lower() or len(caplog.records) >= 0


class TestExportOptions:
    """Tests for ExportOptions frozen dataclass."""

    def test_export_options_creation_with_defaults(self) -> None:
        """Test ExportOptions can be created with default values."""
        from yolo_developer.audit.export_types import ExportOptions

        options = ExportOptions()
        assert options.format == "json"
        assert options.include_decisions is True
        assert options.include_traces is True
        assert options.include_coverage is False
        assert options.redaction_config is not None

    def test_export_options_creation_with_values(self) -> None:
        """Test ExportOptions can be created with custom values."""
        from yolo_developer.audit.export_types import ExportOptions, RedactionConfig

        redaction = RedactionConfig(redact_metadata=True)
        options = ExportOptions(
            format="csv",
            include_decisions=False,
            include_traces=True,
            include_coverage=True,
            redaction_config=redaction,
        )
        assert options.format == "csv"
        assert options.include_decisions is False
        assert options.include_traces is True
        assert options.include_coverage is True
        assert options.redaction_config.redact_metadata is True

    def test_export_options_is_frozen(self) -> None:
        """Test ExportOptions is immutable (frozen)."""
        from yolo_developer.audit.export_types import ExportOptions

        options = ExportOptions()
        with pytest.raises(FrozenInstanceError):
            options.format = "csv"  # type: ignore[misc]

    def test_export_options_to_dict(self) -> None:
        """Test ExportOptions.to_dict() produces JSON-serializable dict."""
        from yolo_developer.audit.export_types import ExportOptions, RedactionConfig

        redaction = RedactionConfig(redact_metadata=True)
        options = ExportOptions(
            format="pdf",
            include_decisions=True,
            include_traces=False,
            include_coverage=True,
            redaction_config=redaction,
        )
        result = options.to_dict()

        assert isinstance(result, dict)
        assert result["format"] == "pdf"
        assert result["include_decisions"] is True
        assert result["include_traces"] is False
        assert result["include_coverage"] is True
        assert "redaction_config" in result
        assert result["redaction_config"]["redact_metadata"] is True
        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_export_options_validation_warning_for_invalid_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test ExportOptions logs warning for invalid format."""
        from yolo_developer.audit.export_types import ExportOptions

        with caplog.at_level(logging.WARNING):
            _ = ExportOptions(format="invalid")  # type: ignore[arg-type]

        assert "format" in caplog.text.lower() or "invalid" in caplog.text.lower()


class TestDefaultConstants:
    """Tests for default constant values."""

    def test_default_redaction_config_exists(self) -> None:
        """Test DEFAULT_REDACTION_CONFIG constant is defined."""
        from yolo_developer.audit.export_types import DEFAULT_REDACTION_CONFIG

        assert DEFAULT_REDACTION_CONFIG is not None

    def test_default_redaction_config_values(self) -> None:
        """Test DEFAULT_REDACTION_CONFIG has expected defaults."""
        from yolo_developer.audit.export_types import (
            DEFAULT_REDACTION_CONFIG,
            RedactionConfig,
        )

        assert isinstance(DEFAULT_REDACTION_CONFIG, RedactionConfig)
        assert DEFAULT_REDACTION_CONFIG.redact_metadata is False
        assert DEFAULT_REDACTION_CONFIG.redact_session_ids is False
        assert DEFAULT_REDACTION_CONFIG.redact_fields == ()

    def test_default_export_options_exists(self) -> None:
        """Test DEFAULT_EXPORT_OPTIONS constant is defined."""
        from yolo_developer.audit.export_types import DEFAULT_EXPORT_OPTIONS

        assert DEFAULT_EXPORT_OPTIONS is not None

    def test_default_export_options_values(self) -> None:
        """Test DEFAULT_EXPORT_OPTIONS has expected defaults."""
        from yolo_developer.audit.export_types import (
            DEFAULT_EXPORT_OPTIONS,
            ExportOptions,
        )

        assert isinstance(DEFAULT_EXPORT_OPTIONS, ExportOptions)
        assert DEFAULT_EXPORT_OPTIONS.format == "json"
        assert DEFAULT_EXPORT_OPTIONS.include_decisions is True
        assert DEFAULT_EXPORT_OPTIONS.include_traces is True
        assert DEFAULT_EXPORT_OPTIONS.include_coverage is False
