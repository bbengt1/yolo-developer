"""Type definitions for Dev agent (Story 8.1).

This module provides the data types used by the Dev agent:

- ImplementationStatus: Literal type for implementation lifecycle status
- CodeFileType: Literal type for code file classifications
- TestFileType: Literal type for test file classifications
- CodeFile: A code file generated during implementation
- TestFile: A test file generated during implementation
- ImplementationArtifact: Complete implementation output for a story
- DevOutput: Complete output from dev processing

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.dev.types import (
    ...     CodeFile,
    ...     TestFile,
    ...     ImplementationArtifact,
    ...     DevOutput,
    ... )
    >>>
    >>> code_file = CodeFile(
    ...     file_path="src/auth/handler.py",
    ...     content="def authenticate(): pass",
    ...     file_type="source",
    ... )
    >>> code_file.to_dict()
    {'file_path': 'src/auth/handler.py', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

# =============================================================================
# Literal Types
# =============================================================================

ImplementationStatus = Literal[
    "pending",
    "in_progress",
    "completed",
    "failed",
]
"""Lifecycle status of an implementation.

Values:
    pending: Story received but not yet processed
    in_progress: Currently being implemented
    completed: Implementation finished successfully
    failed: Implementation failed with errors
"""

CodeFileType = Literal[
    "source",
    "test",
    "config",
    "doc",
]
"""Type classification for code files.

Values:
    source: Main implementation code
    test: Test files
    config: Configuration files
    doc: Documentation files
"""

TestFileType = Literal[
    "unit",
    "integration",
    "e2e",
]
"""Type classification for test files.

Values:
    unit: Unit tests for isolated functionality
    integration: Integration tests for component interactions
    e2e: End-to-end tests for complete user flows
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CodeFile:
    """A code file generated during implementation.

    Represents a single file produced by the Dev agent, including
    source code, tests, configuration, or documentation.

    Attributes:
        file_path: Path to the file relative to project root. Should be a valid
            relative path without null bytes or absolute path markers.
        content: The file content (code, config, etc.). Excluded from repr for
            readability when debugging.
        file_type: Classification of the file type
        created_at: ISO timestamp when file was created

    Example:
        >>> code_file = CodeFile(
        ...     file_path="src/auth/handler.py",
        ...     content="def authenticate(user): pass",
        ...     file_type="source",
        ... )
        >>> code_file.to_dict()
        {'file_path': 'src/auth/handler.py', ...}
    """

    file_path: str
    content: str = field(repr=False)
    file_type: CodeFileType
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the code file.
        """
        return {
            "file_path": self.file_path,
            "content": self.content,
            "file_type": self.file_type,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TestFile:
    """A test file generated during implementation.

    Represents a test file produced by the Dev agent, with
    classification for the type of testing.

    Note:
        This class is named TestFile for domain clarity. The __test__ = False
        attribute prevents pytest from attempting to collect it as a test class.

    Attributes:
        file_path: Path to the test file relative to project root. Should be a valid
            relative path without null bytes or absolute path markers.
        content: The test file content. Excluded from repr for readability.
        test_type: Classification of test type (unit, integration, e2e)
        created_at: ISO timestamp when file was created

    Example:
        >>> test_file = TestFile(
        ...     file_path="tests/unit/test_auth.py",
        ...     content="def test_authenticate(): pass",
        ...     test_type="unit",
        ... )
        >>> test_file.to_dict()
        {'file_path': 'tests/unit/test_auth.py', ...}
    """

    __test__ = False  # Prevent pytest collection

    file_path: str
    content: str = field(repr=False)
    test_type: TestFileType
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the test file.
        """
        return {
            "file_path": self.file_path,
            "content": self.content,
            "test_type": self.test_type,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ImplementationArtifact:
    """Complete implementation output for a single story.

    Contains all code files, test files, and metadata for a story
    implementation.

    Attributes:
        story_id: ID of the story being implemented
        code_files: Tuple of generated code files
        test_files: Tuple of generated test files
        implementation_status: Current status of implementation
        notes: Implementation notes and decisions
        created_at: ISO timestamp when artifact was created

    Example:
        >>> artifact = ImplementationArtifact(
        ...     story_id="story-001",
        ...     code_files=(code_file,),
        ...     test_files=(test_file,),
        ...     implementation_status="completed",
        ...     notes="Implemented with Repository pattern",
        ... )
        >>> artifact.to_dict()
        {'story_id': 'story-001', ...}
    """

    story_id: str
    code_files: tuple[CodeFile, ...] = field(default_factory=tuple)
    test_files: tuple[TestFile, ...] = field(default_factory=tuple)
    implementation_status: ImplementationStatus = "pending"
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested code and test files.
        """
        return {
            "story_id": self.story_id,
            "code_files": [f.to_dict() for f in self.code_files],
            "test_files": [f.to_dict() for f in self.test_files],
            "implementation_status": self.implementation_status,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DevOutput:
    """Complete output from Dev agent processing.

    Contains all implementation artifacts generated during
    dev_node execution, plus processing notes.

    Attributes:
        implementations: Tuple of implementation artifacts per story
        processing_notes: Notes about the processing (stats, issues, etc.)

    Example:
        >>> output = DevOutput(
        ...     implementations=(artifact1, artifact2),
        ...     processing_notes="Processed 2 stories, generated 4 files",
        ... )
        >>> output.to_dict()
        {'implementations': [...], 'processing_notes': '...'}
    """

    implementations: tuple[ImplementationArtifact, ...] = field(default_factory=tuple)
    processing_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with nested implementation artifacts.
        """
        return {
            "implementations": [i.to_dict() for i in self.implementations],
            "processing_notes": self.processing_notes,
        }
