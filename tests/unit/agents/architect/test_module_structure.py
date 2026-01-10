"""Tests for Architect agent module structure (Story 7.1, Task 1).

Tests verify that the Architect agent module is properly structured
following the patterns from analyst and pm agent modules.
"""

from __future__ import annotations


class TestArchitectModuleStructure:
    """Test suite for module structure verification."""

    def test_architect_module_importable(self) -> None:
        """Test that architect module can be imported."""
        from yolo_developer.agents import architect

        assert architect is not None

    def test_architect_node_function_importable(self) -> None:
        """Test that architect_node function can be imported from module."""
        from yolo_developer.agents.architect import architect_node

        assert architect_node is not None
        assert callable(architect_node)

    def test_types_module_importable(self) -> None:
        """Test that types module can be imported."""
        from yolo_developer.agents.architect import types

        assert types is not None

    def test_architect_output_importable(self) -> None:
        """Test that ArchitectOutput can be imported."""
        from yolo_developer.agents.architect import ArchitectOutput

        assert ArchitectOutput is not None

    def test_design_decision_importable(self) -> None:
        """Test that DesignDecision can be imported."""
        from yolo_developer.agents.architect import DesignDecision

        assert DesignDecision is not None

    def test_adr_importable(self) -> None:
        """Test that ADR can be imported."""
        from yolo_developer.agents.architect import ADR

        assert ADR is not None
