"""Type definitions for audit formatter (Story 11.3).

This module provides the data types used by the audit formatter:

- FormatterStyle: Literal type for output verbosity levels
- ColorScheme: Color configuration for terminal output
- FormatOptions: Formatting configuration options

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.formatter_types import (
    ...     ColorScheme,
    ...     FormatOptions,
    ...     DEFAULT_COLOR_SCHEME,
    ...     DEFAULT_FORMAT_OPTIONS,
    ... )
    >>>
    >>> # Use default options
    >>> options = DEFAULT_FORMAT_OPTIONS
    >>> options.style
    'standard'
    >>>
    >>> # Create custom options
    >>> custom = FormatOptions(style="verbose", show_metadata=True)
    >>> custom.to_dict()
    {'style': 'verbose', 'show_metadata': True, ...}

References:
    - FR83: Users can view audit trail in human-readable format
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

FormatterStyle = Literal["minimal", "standard", "verbose"]
"""Style of output formatting.

Values:
    minimal: Bare minimum information (ID, type, timestamp)
    standard: Normal level of detail (content, rationale, agent)
    verbose: Full details including metadata and trace links
"""

# =============================================================================
# Constants (Subtask 1.4)
# =============================================================================

VALID_FORMATTER_STYLES: frozenset[str] = frozenset({"minimal", "standard", "verbose"})
"""Set of valid formatter style values."""


# =============================================================================
# Data Classes (Subtasks 1.1, 1.2, 1.3)
# =============================================================================


@dataclass(frozen=True)
class ColorScheme:
    """Color configuration for terminal output.

    Defines colors for severity levels, agent types, and section formatting.
    Uses Rich color names compatible with the Rich library.

    Attributes:
        severity_critical: Color for critical severity (default: red)
        severity_warning: Color for warning severity (default: yellow)
        severity_info: Color for info severity (default: green)
        agent_analyst: Color for analyst agent (default: blue)
        agent_pm: Color for PM agent (default: cyan)
        agent_architect: Color for architect agent (default: magenta)
        agent_dev: Color for dev agent (default: green)
        agent_sm: Color for SM agent (default: yellow)
        agent_tea: Color for TEA agent (default: red)
        section_header: Style for section headers (default: bold)
        section_border: Style for section borders (default: dim)

    Example:
        >>> scheme = ColorScheme()
        >>> scheme.severity_critical
        'red'
        >>> scheme.agent_analyst
        'blue'
    """

    # Severity colors
    severity_critical: str = "red"
    severity_warning: str = "yellow"
    severity_info: str = "green"

    # Agent colors
    agent_analyst: str = "blue"
    agent_pm: str = "cyan"
    agent_architect: str = "magenta"
    agent_dev: str = "green"
    agent_sm: str = "yellow"
    agent_tea: str = "red"

    # Section formatting
    section_header: str = "bold"
    section_border: str = "dim"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the color scheme.
        """
        return {
            "severity_critical": self.severity_critical,
            "severity_warning": self.severity_warning,
            "severity_info": self.severity_info,
            "agent_analyst": self.agent_analyst,
            "agent_pm": self.agent_pm,
            "agent_architect": self.agent_architect,
            "agent_dev": self.agent_dev,
            "agent_sm": self.agent_sm,
            "agent_tea": self.agent_tea,
            "section_header": self.section_header,
            "section_border": self.section_border,
        }


@dataclass(frozen=True)
class FormatOptions:
    """Formatting configuration options for audit views.

    Controls verbosity, metadata display, and content truncation
    for formatted audit output.

    Attributes:
        style: Output verbosity level (minimal/standard/verbose)
        show_metadata: Whether to display metadata dict (default: False)
        show_trace_links: Whether to display trace links (default: False)
        max_content_length: Max chars for content before truncation (default: 500)
        highlight_severity: Whether to color-code by severity (default: True)

    Example:
        >>> options = FormatOptions(style="verbose", show_metadata=True)
        >>> options.style
        'verbose'
        >>> options.show_metadata
        True
    """

    style: FormatterStyle = "standard"
    show_metadata: bool = False
    show_trace_links: bool = False
    max_content_length: int = 500
    highlight_severity: bool = True

    def __post_init__(self) -> None:
        """Validate format options and log warnings for issues."""
        if self.style not in VALID_FORMATTER_STYLES:
            _logger.warning(
                "FormatOptions style='%s' is not a valid style",
                self.style,
            )
        if self.max_content_length < 0:
            _logger.warning(
                "FormatOptions max_content_length=%d is negative",
                self.max_content_length,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the format options.
        """
        return {
            "style": self.style,
            "show_metadata": self.show_metadata,
            "show_trace_links": self.show_trace_links,
            "max_content_length": self.max_content_length,
            "highlight_severity": self.highlight_severity,
        }


# =============================================================================
# Default Constants (Subtask 1.4)
# =============================================================================

DEFAULT_COLOR_SCHEME: ColorScheme = ColorScheme()
"""Default color scheme for terminal output."""

DEFAULT_FORMAT_OPTIONS: FormatOptions = FormatOptions()
"""Default format options with standard verbosity."""
