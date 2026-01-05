"""Project isolation types and validation for YOLO Developer memory layer.

This module provides types and validation functions for project IDs used to
isolate memory between different projects. Project IDs are used in collection
naming for ChromaDB and file paths for JSON graph storage.

Example:
    >>> from yolo_developer.memory.isolation import validate_project_id
    >>>
    >>> # Valid project IDs
    >>> validate_project_id("my-project")
    'my-project'
    >>> validate_project_id("MyProject_123")
    'MyProject_123'
    >>>
    >>> # Invalid project IDs raise errors
    >>> validate_project_id("my@project!")
    Raises InvalidProjectIdError

Security Note:
    Project IDs are used in file paths and ChromaDB collection names.
    Validation prevents directory traversal and injection attacks by
    restricting characters to alphanumeric, hyphens, and underscores.
"""

from __future__ import annotations

import re

# Default project ID used when none is specified
DEFAULT_PROJECT_ID = "default"

# Project ID constraints
PROJECT_ID_MIN_LENGTH = 1
PROJECT_ID_MAX_LENGTH = 64

# Regex pattern for valid project IDs:
# - Alphanumeric characters (a-z, A-Z, 0-9)
# - Hyphens (-)
# - Underscores (_)
# - Length 1-64 characters
PROJECT_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

# Compiled regex for performance
_PROJECT_ID_REGEX = re.compile(PROJECT_ID_PATTERN)


class InvalidProjectIdError(ValueError):
    """Raised when a project ID is invalid.

    Provides detailed information about why the project ID was rejected,
    including the invalid value and the specific reason.

    Attributes:
        project_id: The invalid project ID that was provided.
        reason: Explanation of why the project ID is invalid.

    Example:
        >>> error = InvalidProjectIdError("my@project", "contains invalid characters")
        >>> error.project_id
        'my@project'
        >>> error.reason
        'contains invalid characters'
    """

    def __init__(self, project_id: str, reason: str) -> None:
        """Initialize InvalidProjectIdError with context.

        Args:
            project_id: The invalid project ID value.
            reason: Explanation of why validation failed.
        """
        self.project_id = project_id
        self.reason = reason
        message = (
            f"Invalid project ID '{project_id}': {reason}. "
            f"Valid project IDs must be {PROJECT_ID_MIN_LENGTH}-{PROJECT_ID_MAX_LENGTH} characters "
            f"and contain only alphanumeric characters, hyphens, and underscores."
        )
        super().__init__(message)


def validate_project_id(project_id: str) -> str:
    """Validate a project ID and return it if valid.

    Validates that the project ID meets the following criteria:
    - Length between 1 and 64 characters
    - Contains only alphanumeric characters, hyphens, and underscores

    Args:
        project_id: The project ID string to validate.

    Returns:
        The validated project ID (unchanged if valid).

    Raises:
        InvalidProjectIdError: If the project ID is invalid, with details
            about why validation failed.

    Example:
        >>> validate_project_id("my-project")
        'my-project'
        >>> validate_project_id("")
        Raises InvalidProjectIdError
        >>> validate_project_id("my@project!")
        Raises InvalidProjectIdError
    """
    # Check for empty project ID
    if not project_id:
        raise InvalidProjectIdError(
            project_id,
            "project ID cannot be empty",
        )

    # Check length constraints
    if len(project_id) > PROJECT_ID_MAX_LENGTH:
        raise InvalidProjectIdError(
            project_id,
            f"project ID exceeds maximum length of {PROJECT_ID_MAX_LENGTH} characters "
            f"(got {len(project_id)} characters)",
        )

    # Check character pattern
    if not _PROJECT_ID_REGEX.match(project_id):
        raise InvalidProjectIdError(
            project_id,
            "project ID contains invalid characters",
        )

    return project_id
