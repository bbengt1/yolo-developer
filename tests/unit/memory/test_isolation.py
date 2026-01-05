"""Unit tests for project isolation validation and types.

Tests for project ID validation, error handling, and type definitions
used for isolating memory between different projects.
"""

from __future__ import annotations

import pytest

from yolo_developer.memory.isolation import (
    DEFAULT_PROJECT_ID,
    PROJECT_ID_MAX_LENGTH,
    PROJECT_ID_MIN_LENGTH,
    PROJECT_ID_PATTERN,
    InvalidProjectIdError,
    validate_project_id,
)


class TestProjectIdValidation:
    """Tests for validate_project_id function."""

    def test_validate_project_id_simple_alphanumeric(self) -> None:
        """Valid simple alphanumeric project ID passes validation."""
        result = validate_project_id("myproject123")
        assert result == "myproject123"

    def test_validate_project_id_with_hyphens(self) -> None:
        """Valid project ID with hyphens passes validation."""
        result = validate_project_id("my-project-name")
        assert result == "my-project-name"

    def test_validate_project_id_with_underscores(self) -> None:
        """Valid project ID with underscores passes validation."""
        result = validate_project_id("my_project_name")
        assert result == "my_project_name"

    def test_validate_project_id_mixed_valid_chars(self) -> None:
        """Valid project ID with mixed valid characters passes validation."""
        result = validate_project_id("my-project_123")
        assert result == "my-project_123"

    def test_validate_project_id_single_char(self) -> None:
        """Single character project ID passes validation."""
        result = validate_project_id("a")
        assert result == "a"

    def test_validate_project_id_max_length(self) -> None:
        """Project ID at max length (64 chars) passes validation."""
        project_id = "a" * 64
        result = validate_project_id(project_id)
        assert result == project_id

    def test_validate_project_id_empty_raises_error(self) -> None:
        """Empty project ID raises InvalidProjectIdError."""
        with pytest.raises(InvalidProjectIdError) as exc_info:
            validate_project_id("")
        assert "empty" in str(exc_info.value).lower() or "1-64" in str(exc_info.value)

    def test_validate_project_id_too_long_raises_error(self) -> None:
        """Project ID exceeding max length raises InvalidProjectIdError."""
        project_id = "a" * 65
        with pytest.raises(InvalidProjectIdError) as exc_info:
            validate_project_id(project_id)
        assert "64" in str(exc_info.value) or "length" in str(exc_info.value).lower()

    def test_validate_project_id_special_chars_raises_error(self) -> None:
        """Project ID with special characters raises InvalidProjectIdError."""
        with pytest.raises(InvalidProjectIdError):
            validate_project_id("my@project!")

    def test_validate_project_id_spaces_raises_error(self) -> None:
        """Project ID with spaces raises InvalidProjectIdError."""
        with pytest.raises(InvalidProjectIdError):
            validate_project_id("my project")

    def test_validate_project_id_dots_raises_error(self) -> None:
        """Project ID with dots raises InvalidProjectIdError."""
        with pytest.raises(InvalidProjectIdError):
            validate_project_id("my.project")

    def test_validate_project_id_slashes_raises_error(self) -> None:
        """Project ID with slashes raises InvalidProjectIdError."""
        with pytest.raises(InvalidProjectIdError):
            validate_project_id("my/project")

    def test_validate_project_id_leading_hyphen_allowed(self) -> None:
        """Project ID with leading hyphen is allowed."""
        result = validate_project_id("-myproject")
        assert result == "-myproject"

    def test_validate_project_id_leading_underscore_allowed(self) -> None:
        """Project ID with leading underscore is allowed."""
        result = validate_project_id("_myproject")
        assert result == "_myproject"

    def test_validate_project_id_uppercase_allowed(self) -> None:
        """Project ID with uppercase letters passes validation."""
        result = validate_project_id("MyProject")
        assert result == "MyProject"


class TestInvalidProjectIdError:
    """Tests for InvalidProjectIdError exception."""

    def test_invalid_project_id_error_message(self) -> None:
        """InvalidProjectIdError contains descriptive message."""
        error = InvalidProjectIdError("test@id", "contains invalid characters")
        assert "test@id" in str(error)
        assert "invalid" in str(error).lower()

    def test_invalid_project_id_error_has_project_id_attribute(self) -> None:
        """InvalidProjectIdError stores the invalid project_id."""
        error = InvalidProjectIdError("bad-id!", "reason")
        assert error.project_id == "bad-id!"

    def test_invalid_project_id_error_has_reason_attribute(self) -> None:
        """InvalidProjectIdError stores the reason for invalidity."""
        error = InvalidProjectIdError("bad-id!", "contains special chars")
        assert error.reason == "contains special chars"

    def test_invalid_project_id_error_inherits_from_value_error(self) -> None:
        """InvalidProjectIdError inherits from ValueError."""
        error = InvalidProjectIdError("bad", "reason")
        assert isinstance(error, ValueError)


class TestConstants:
    """Tests for module constants."""

    def test_default_project_id_value(self) -> None:
        """DEFAULT_PROJECT_ID has expected value."""
        assert DEFAULT_PROJECT_ID == "default"

    def test_project_id_min_length(self) -> None:
        """PROJECT_ID_MIN_LENGTH is 1."""
        assert PROJECT_ID_MIN_LENGTH == 1

    def test_project_id_max_length(self) -> None:
        """PROJECT_ID_MAX_LENGTH is 64."""
        assert PROJECT_ID_MAX_LENGTH == 64

    def test_project_id_pattern_matches_valid(self) -> None:
        """PROJECT_ID_PATTERN regex matches valid project IDs."""
        import re

        pattern = re.compile(PROJECT_ID_PATTERN)
        assert pattern.match("myproject")
        assert pattern.match("my-project")
        assert pattern.match("my_project")
        assert pattern.match("MyProject123")

    def test_project_id_pattern_rejects_invalid(self) -> None:
        """PROJECT_ID_PATTERN regex rejects invalid project IDs."""
        import re

        pattern = re.compile(PROJECT_ID_PATTERN)
        assert not pattern.match("my@project")
        assert not pattern.match("my project")
        assert not pattern.match("my.project")
