"""Module structure tests for Dev agent (Story 8.1).

Tests that the dev module is properly structured and exports
the expected functions and types.
"""

from __future__ import annotations


class TestModuleExists:
    """Test that dev module exists and is importable."""

    def test_dev_module_importable(self) -> None:
        """Test dev module can be imported."""
        from yolo_developer.agents import dev

        assert dev is not None

    def test_dev_node_importable_from_module(self) -> None:
        """Test dev_node is importable from dev module."""
        from yolo_developer.agents.dev import dev_node

        assert dev_node is not None
        assert callable(dev_node)

    def test_dev_node_importable_from_agents(self) -> None:
        """Test dev_node is importable from agents package."""
        from yolo_developer.agents import dev_node

        assert dev_node is not None
        assert callable(dev_node)


class TestTypeExports:
    """Test that type definitions are properly exported."""

    def test_dev_output_exported(self) -> None:
        """Test DevOutput is exported from dev module."""
        from yolo_developer.agents.dev import DevOutput

        assert DevOutput is not None

    def test_implementation_artifact_exported(self) -> None:
        """Test ImplementationArtifact is exported from dev module."""
        from yolo_developer.agents.dev import ImplementationArtifact

        assert ImplementationArtifact is not None

    def test_code_file_exported(self) -> None:
        """Test CodeFile is exported from dev module."""
        from yolo_developer.agents.dev import CodeFile

        assert CodeFile is not None

    def test_test_file_exported(self) -> None:
        """Test TestFile is exported from dev module."""
        from yolo_developer.agents.dev import TestFile

        assert TestFile is not None

    def test_implementation_status_exported(self) -> None:
        """Test ImplementationStatus is exported from dev module."""
        from yolo_developer.agents.dev import ImplementationStatus

        assert ImplementationStatus is not None

    def test_code_file_type_exported(self) -> None:
        """Test CodeFileType is exported from dev module."""
        from yolo_developer.agents.dev import CodeFileType

        assert CodeFileType is not None

    def test_test_file_type_exported(self) -> None:
        """Test TestFileType is exported from dev module."""
        from yolo_developer.agents.dev import TestFileType

        assert TestFileType is not None
