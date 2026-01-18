"""Type definitions for audit export (Story 11.4).

This module provides the data types used by the audit export functionality:

- ExportFormat: Literal type for export format options
- RedactionConfig: Configuration for redacting sensitive data
- ExportOptions: Complete export configuration options

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.export_types import (
    ...     ExportOptions,
    ...     RedactionConfig,
    ...     DEFAULT_EXPORT_OPTIONS,
    ...     DEFAULT_REDACTION_CONFIG,
    ... )
    >>>
    >>> # Use default options
    >>> options = DEFAULT_EXPORT_OPTIONS
    >>> options.format
    'json'
    >>>
    >>> # Create custom options with redaction
    >>> redaction = RedactionConfig(redact_metadata=True)
    >>> custom = ExportOptions(format="pdf", redaction_config=redaction)
    >>> custom.to_dict()
    {'format': 'pdf', 'include_decisions': True, ...}

References:
    - FR84: System can export audit trail for compliance reporting
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

ExportFormat = Literal["json", "csv", "pdf"]
"""Format for audit trail export.

Values:
    json: Machine-readable JSON format with full fidelity
    csv: Spreadsheet-compatible CSV format with flattened structure
    pdf: Human-readable PDF format for compliance documentation
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

VALID_EXPORT_FORMATS: frozenset[str] = frozenset({"json", "csv", "pdf"})
"""Set of valid export format values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class RedactionConfig:
    """Configuration for redacting sensitive data in exports.

    Controls which fields are redacted when exporting audit data
    for compliance or external sharing purposes.

    Attributes:
        redact_metadata: Whether to redact metadata dict contents (default: False)
        redact_session_ids: Whether to redact session IDs (default: False)
        redact_fields: Tuple of field paths to redact (e.g., "agent.session_id")

    Example:
        >>> config = RedactionConfig(redact_metadata=True)
        >>> config.redact_metadata
        True
        >>> config.to_dict()
        {'redact_metadata': True, 'redact_session_ids': False, 'redact_fields': []}
    """

    redact_metadata: bool = False
    redact_session_ids: bool = False
    redact_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate redaction config and log warnings for issues."""
        for field_path in self.redact_fields:
            if not field_path:
                _logger.warning(
                    "RedactionConfig redact_fields contains empty field path"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the redaction config.
        """
        return {
            "redact_metadata": self.redact_metadata,
            "redact_session_ids": self.redact_session_ids,
            "redact_fields": list(self.redact_fields),
        }


@dataclass(frozen=True)
class ExportOptions:
    """Complete export configuration options.

    Controls the export format, what data to include, and redaction settings.

    Attributes:
        format: Export format (json/csv/pdf) (default: "json")
        include_decisions: Include decision log in export (default: True)
        include_traces: Include traceability data in export (default: True)
        include_coverage: Include coverage statistics in export (default: False)
        redaction_config: Configuration for redacting sensitive data

    Example:
        >>> options = ExportOptions(format="pdf", include_coverage=True)
        >>> options.format
        'pdf'
        >>> options.include_coverage
        True
    """

    format: ExportFormat = "json"
    include_decisions: bool = True
    include_traces: bool = True
    include_coverage: bool = False
    redaction_config: RedactionConfig = field(default_factory=RedactionConfig)

    def __post_init__(self) -> None:
        """Validate export options and log warnings for issues."""
        if self.format not in VALID_EXPORT_FORMATS:
            _logger.warning(
                "ExportOptions format='%s' is not a valid format",
                self.format,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the export options.
        """
        return {
            "format": self.format,
            "include_decisions": self.include_decisions,
            "include_traces": self.include_traces,
            "include_coverage": self.include_coverage,
            "redaction_config": self.redaction_config.to_dict(),
        }


# =============================================================================
# Default Constants (Subtask 1.4)
# =============================================================================

DEFAULT_REDACTION_CONFIG: RedactionConfig = RedactionConfig()
"""Default redaction config with no redaction enabled."""

DEFAULT_EXPORT_OPTIONS: ExportOptions = ExportOptions()
"""Default export options with JSON format and decisions + traces included."""
