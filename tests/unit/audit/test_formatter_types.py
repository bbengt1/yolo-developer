"""Tests for audit formatter type definitions (Story 11.3).

Tests cover:
- FormatterStyle Literal type validation
- ColorScheme frozen dataclass
- FormatOptions frozen dataclass with validation
- to_dict() serialization methods
- Frozen dataclass immutability
- DEFAULT_COLOR_SCHEME and DEFAULT_FORMAT_OPTIONS constants
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from yolo_developer.audit.formatter_types import ColorScheme, FormatOptions


class TestFormatterStyle:
    """Tests for FormatterStyle Literal type."""

    def test_valid_formatter_styles(self) -> None:
        """Test that valid formatter styles are accepted."""
        from yolo_developer.audit.formatter_types import VALID_FORMATTER_STYLES

        assert "minimal" in VALID_FORMATTER_STYLES
        assert "standard" in VALID_FORMATTER_STYLES
        assert "verbose" in VALID_FORMATTER_STYLES
        assert len(VALID_FORMATTER_STYLES) == 3

    def test_invalid_style_not_in_valid_styles(self) -> None:
        """Test that invalid styles are not in valid styles set."""
        from yolo_developer.audit.formatter_types import VALID_FORMATTER_STYLES

        assert "invalid" not in VALID_FORMATTER_STYLES
        assert "full" not in VALID_FORMATTER_STYLES


class TestColorScheme:
    """Tests for ColorScheme frozen dataclass."""

    def test_create_color_scheme_with_defaults(self) -> None:
        """Test creating ColorScheme with default values."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme()
        assert scheme.severity_critical == "red"
        assert scheme.severity_warning == "yellow"
        assert scheme.severity_info == "green"
        assert scheme.agent_analyst == "blue"
        assert scheme.agent_pm == "cyan"
        assert scheme.agent_architect == "magenta"
        assert scheme.agent_dev == "green"
        assert scheme.agent_sm == "yellow"
        assert scheme.agent_tea == "red"
        assert scheme.section_header == "bold"
        assert scheme.section_border == "dim"

    def test_create_color_scheme_with_custom_values(self) -> None:
        """Test creating ColorScheme with custom values."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme(
            severity_critical="bright_red",
            severity_warning="bright_yellow",
            agent_analyst="bright_blue",
        )
        assert scheme.severity_critical == "bright_red"
        assert scheme.severity_warning == "bright_yellow"
        assert scheme.agent_analyst == "bright_blue"
        # Other values should be defaults
        assert scheme.severity_info == "green"

    def test_color_scheme_is_frozen(self) -> None:
        """Test that ColorScheme is immutable."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme()
        with pytest.raises(AttributeError):
            scheme.severity_critical = "blue"  # type: ignore[misc]

    def test_color_scheme_to_dict(self) -> None:
        """Test ColorScheme to_dict serialization."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme()
        result = scheme.to_dict()

        assert isinstance(result, dict)
        assert result["severity_critical"] == "red"
        assert result["severity_warning"] == "yellow"
        assert result["severity_info"] == "green"
        assert result["agent_analyst"] == "blue"
        assert result["section_header"] == "bold"

    def test_color_scheme_to_dict_is_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme()
        result = scheme.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed == result


class TestFormatOptions:
    """Tests for FormatOptions frozen dataclass."""

    def test_create_format_options_with_defaults(self) -> None:
        """Test creating FormatOptions with default values."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions()
        assert options.style == "standard"
        assert options.show_metadata is False
        assert options.show_trace_links is False
        assert options.max_content_length == 500
        assert options.highlight_severity is True

    def test_create_format_options_with_custom_values(self) -> None:
        """Test creating FormatOptions with custom values."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions(
            style="verbose",
            show_metadata=True,
            show_trace_links=True,
            max_content_length=1000,
            highlight_severity=False,
        )
        assert options.style == "verbose"
        assert options.show_metadata is True
        assert options.show_trace_links is True
        assert options.max_content_length == 1000
        assert options.highlight_severity is False

    def test_format_options_is_frozen(self) -> None:
        """Test that FormatOptions is immutable."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions()
        with pytest.raises(AttributeError):
            options.style = "verbose"  # type: ignore[misc]

    def test_format_options_to_dict(self) -> None:
        """Test FormatOptions to_dict serialization."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions(style="verbose", show_metadata=True)
        result = options.to_dict()

        assert isinstance(result, dict)
        assert result["style"] == "verbose"
        assert result["show_metadata"] is True
        assert result["show_trace_links"] is False
        assert result["max_content_length"] == 500
        assert result["highlight_severity"] is True

    def test_format_options_to_dict_is_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions()
        result = options.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed == result

    def test_format_options_invalid_style_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid style logs a warning."""
        from yolo_developer.audit.formatter_types import FormatOptions

        # Invalid style should log a warning but not raise
        options = FormatOptions(style="invalid")  # type: ignore[arg-type]
        assert options.style == "invalid"
        assert "is not a valid style" in caplog.text

    def test_format_options_negative_max_content_length_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative max_content_length logs a warning."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions(max_content_length=-1)
        assert options.max_content_length == -1
        assert "max_content_length" in caplog.text


class TestDefaultConstants:
    """Tests for DEFAULT_COLOR_SCHEME and DEFAULT_FORMAT_OPTIONS constants."""

    def test_default_color_scheme_exists(self) -> None:
        """Test that DEFAULT_COLOR_SCHEME constant exists."""
        from yolo_developer.audit.formatter_types import DEFAULT_COLOR_SCHEME

        assert DEFAULT_COLOR_SCHEME is not None

    def test_default_color_scheme_is_color_scheme(self) -> None:
        """Test that DEFAULT_COLOR_SCHEME is a ColorScheme instance."""
        from yolo_developer.audit.formatter_types import (
            DEFAULT_COLOR_SCHEME,
            ColorScheme,
        )

        assert isinstance(DEFAULT_COLOR_SCHEME, ColorScheme)

    def test_default_format_options_exists(self) -> None:
        """Test that DEFAULT_FORMAT_OPTIONS constant exists."""
        from yolo_developer.audit.formatter_types import DEFAULT_FORMAT_OPTIONS

        assert DEFAULT_FORMAT_OPTIONS is not None

    def test_default_format_options_is_format_options(self) -> None:
        """Test that DEFAULT_FORMAT_OPTIONS is a FormatOptions instance."""
        from yolo_developer.audit.formatter_types import (
            DEFAULT_FORMAT_OPTIONS,
            FormatOptions,
        )

        assert isinstance(DEFAULT_FORMAT_OPTIONS, FormatOptions)

    def test_default_format_options_has_standard_style(self) -> None:
        """Test that DEFAULT_FORMAT_OPTIONS uses standard style."""
        from yolo_developer.audit.formatter_types import DEFAULT_FORMAT_OPTIONS

        assert DEFAULT_FORMAT_OPTIONS.style == "standard"

    def test_valid_formatter_styles_constant(self) -> None:
        """Test VALID_FORMATTER_STYLES constant."""
        from yolo_developer.audit.formatter_types import VALID_FORMATTER_STYLES

        assert isinstance(VALID_FORMATTER_STYLES, frozenset)
        assert VALID_FORMATTER_STYLES == frozenset({"minimal", "standard", "verbose"})


class TestColorSchemeEquality:
    """Tests for ColorScheme equality and hashing."""

    def test_color_scheme_equality(self) -> None:
        """Test that equal ColorSchemes are equal."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme1 = ColorScheme()
        scheme2 = ColorScheme()
        assert scheme1 == scheme2

    def test_color_scheme_inequality(self) -> None:
        """Test that different ColorSchemes are not equal."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme1 = ColorScheme()
        scheme2 = ColorScheme(severity_critical="blue")
        assert scheme1 != scheme2

    def test_color_scheme_hashable(self) -> None:
        """Test that ColorScheme is hashable."""
        from yolo_developer.audit.formatter_types import ColorScheme

        scheme = ColorScheme()
        hash_value = hash(scheme)
        assert isinstance(hash_value, int)


class TestFormatOptionsEquality:
    """Tests for FormatOptions equality and hashing."""

    def test_format_options_equality(self) -> None:
        """Test that equal FormatOptions are equal."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options1 = FormatOptions()
        options2 = FormatOptions()
        assert options1 == options2

    def test_format_options_inequality(self) -> None:
        """Test that different FormatOptions are not equal."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options1 = FormatOptions()
        options2 = FormatOptions(style="verbose")
        assert options1 != options2

    def test_format_options_hashable(self) -> None:
        """Test that FormatOptions is hashable."""
        from yolo_developer.audit.formatter_types import FormatOptions

        options = FormatOptions()
        hash_value = hash(options)
        assert isinstance(hash_value, int)
