"""Type dataclass tests for Dev agent (Story 8.1, AC3).

Tests for CodeFile, TestFile, ImplementationArtifact, and DevOutput
dataclasses including creation, serialization, and immutability.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.dev.types import (
    CodeFile,
    CodeFileType,
    DevOutput,
    ImplementationArtifact,
    ImplementationStatus,
    TestFile,
    TestFileType,
)


class TestImplementationStatusLiteral:
    """Tests for ImplementationStatus literal type."""

    def test_valid_status_pending(self) -> None:
        """Test 'pending' is a valid status."""
        status: ImplementationStatus = "pending"
        assert status == "pending"

    def test_valid_status_in_progress(self) -> None:
        """Test 'in_progress' is a valid status."""
        status: ImplementationStatus = "in_progress"
        assert status == "in_progress"

    def test_valid_status_completed(self) -> None:
        """Test 'completed' is a valid status."""
        status: ImplementationStatus = "completed"
        assert status == "completed"

    def test_valid_status_failed(self) -> None:
        """Test 'failed' is a valid status."""
        status: ImplementationStatus = "failed"
        assert status == "failed"


class TestCodeFileTypeLiteral:
    """Tests for CodeFileType literal type."""

    def test_valid_type_source(self) -> None:
        """Test 'source' is a valid type."""
        file_type: CodeFileType = "source"
        assert file_type == "source"

    def test_valid_type_test(self) -> None:
        """Test 'test' is a valid type."""
        file_type: CodeFileType = "test"
        assert file_type == "test"

    def test_valid_type_config(self) -> None:
        """Test 'config' is a valid type."""
        file_type: CodeFileType = "config"
        assert file_type == "config"

    def test_valid_type_doc(self) -> None:
        """Test 'doc' is a valid type."""
        file_type: CodeFileType = "doc"
        assert file_type == "doc"


class TestTestFileTypeLiteral:
    """Tests for TestFileType literal type."""

    def test_valid_type_unit(self) -> None:
        """Test 'unit' is a valid type."""
        test_type: TestFileType = "unit"
        assert test_type == "unit"

    def test_valid_type_integration(self) -> None:
        """Test 'integration' is a valid type."""
        test_type: TestFileType = "integration"
        assert test_type == "integration"

    def test_valid_type_e2e(self) -> None:
        """Test 'e2e' is a valid type."""
        test_type: TestFileType = "e2e"
        assert test_type == "e2e"


class TestCodeFile:
    """Tests for CodeFile dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test CodeFile creation with required fields."""
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        assert code_file.file_path == "src/module.py"
        assert code_file.content == "def hello(): pass"
        assert code_file.file_type == "source"
        assert code_file.created_at is not None

    def test_to_dict_returns_all_fields(self) -> None:
        """Test to_dict() returns all fields."""
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        result = code_file.to_dict()
        assert result["file_path"] == "src/module.py"
        assert result["content"] == "def hello(): pass"
        assert result["file_type"] == "source"
        assert "created_at" in result

    def test_immutability(self) -> None:
        """Test CodeFile is immutable (frozen)."""
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        with pytest.raises(AttributeError):
            code_file.file_path = "other.py"  # type: ignore[misc]


class TestTestFile:
    """Tests for TestFile dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test TestFile creation with required fields."""
        test_file = TestFile(
            file_path="tests/test_module.py",
            content="def test_hello(): pass",
            test_type="unit",
        )
        assert test_file.file_path == "tests/test_module.py"
        assert test_file.content == "def test_hello(): pass"
        assert test_file.test_type == "unit"
        assert test_file.created_at is not None

    def test_to_dict_returns_all_fields(self) -> None:
        """Test to_dict() returns all fields."""
        test_file = TestFile(
            file_path="tests/test_module.py",
            content="def test_hello(): pass",
            test_type="unit",
        )
        result = test_file.to_dict()
        assert result["file_path"] == "tests/test_module.py"
        assert result["content"] == "def test_hello(): pass"
        assert result["test_type"] == "unit"
        assert "created_at" in result

    def test_immutability(self) -> None:
        """Test TestFile is immutable (frozen)."""
        test_file = TestFile(
            file_path="tests/test_module.py",
            content="def test_hello(): pass",
            test_type="unit",
        )
        with pytest.raises(AttributeError):
            test_file.file_path = "other.py"  # type: ignore[misc]


class TestImplementationArtifact:
    """Tests for ImplementationArtifact dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test ImplementationArtifact creation with story_id only."""
        artifact = ImplementationArtifact(story_id="story-001")
        assert artifact.story_id == "story-001"
        assert artifact.code_files == ()
        assert artifact.test_files == ()
        assert artifact.implementation_status == "pending"
        assert artifact.notes == ""

    def test_creation_with_all_fields(self) -> None:
        """Test ImplementationArtifact creation with all fields."""
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        test_file = TestFile(
            file_path="tests/test_module.py",
            content="def test_hello(): pass",
            test_type="unit",
        )
        artifact = ImplementationArtifact(
            story_id="story-001",
            code_files=(code_file,),
            test_files=(test_file,),
            implementation_status="completed",
            notes="Implementation complete",
        )
        assert artifact.story_id == "story-001"
        assert len(artifact.code_files) == 1
        assert len(artifact.test_files) == 1
        assert artifact.implementation_status == "completed"
        assert artifact.notes == "Implementation complete"

    def test_to_dict_returns_nested_structure(self) -> None:
        """Test to_dict() returns nested code and test files."""
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        artifact = ImplementationArtifact(
            story_id="story-001",
            code_files=(code_file,),
            implementation_status="completed",
        )
        result = artifact.to_dict()
        assert result["story_id"] == "story-001"
        assert len(result["code_files"]) == 1
        assert result["code_files"][0]["file_path"] == "src/module.py"
        assert result["implementation_status"] == "completed"

    def test_immutability(self) -> None:
        """Test ImplementationArtifact is immutable (frozen)."""
        artifact = ImplementationArtifact(story_id="story-001")
        with pytest.raises(AttributeError):
            artifact.story_id = "story-002"  # type: ignore[misc]


class TestDevOutput:
    """Tests for DevOutput dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test DevOutput creation with defaults."""
        output = DevOutput()
        assert output.implementations == ()
        assert output.processing_notes == ""

    def test_creation_with_implementations(self) -> None:
        """Test DevOutput creation with implementations."""
        artifact = ImplementationArtifact(story_id="story-001")
        output = DevOutput(
            implementations=(artifact,),
            processing_notes="Processed 1 story",
        )
        assert len(output.implementations) == 1
        assert output.processing_notes == "Processed 1 story"

    def test_to_dict_returns_nested_structure(self) -> None:
        """Test to_dict() returns nested implementations."""
        artifact = ImplementationArtifact(story_id="story-001")
        output = DevOutput(
            implementations=(artifact,),
            processing_notes="Test notes",
        )
        result = output.to_dict()
        assert len(result["implementations"]) == 1
        assert result["implementations"][0]["story_id"] == "story-001"
        assert result["processing_notes"] == "Test notes"

    def test_immutability(self) -> None:
        """Test DevOutput is immutable (frozen)."""
        output = DevOutput()
        with pytest.raises(AttributeError):
            output.processing_notes = "new notes"  # type: ignore[misc]
