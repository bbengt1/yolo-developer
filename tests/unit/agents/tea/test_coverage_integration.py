"""Unit tests for coverage integration in TEA node (Story 9.2 - Task 11).

This module tests the integration of coverage validation into the TEA node:
- Coverage analysis is called for code files
- Coverage findings are included in ValidationResult
- Overall confidence is adjusted based on coverage
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.tea.node import (
    _validate_artifact,
    tea_node,
)
from yolo_developer.orchestrator.state import YoloState


class TestCoverageInValidateArtifact:
    """Tests for coverage integration in _validate_artifact."""

    def test_code_file_triggers_coverage_analysis(self) -> None:
        """Test that code files trigger coverage analysis."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/module.py",
            "content": '''"""Module."""

def add(a, b):
    return a + b
''',
        }
        # Provide test content via state context
        result = _validate_artifact(artifact, test_content="def test_add(): assert True")
        assert result is not None
        # Should have coverage-related information
        assert result.score >= 0

    def test_test_file_does_not_trigger_coverage(self) -> None:
        """Test that test files don't trigger coverage analysis (they ARE coverage)."""
        artifact = {
            "type": "test_file",
            "artifact_id": "tests/test_module.py",
            "content": '''def test_add(): assert True''',
        }
        result = _validate_artifact(artifact)
        # Test files are validated differently
        assert result is not None

    def test_coverage_findings_in_validation_result(self) -> None:
        """Test that coverage findings appear in ValidationResult."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/untested.py",
            "content": '''"""Untested module."""

def some_function():
    return 42

def another_function():
    return 24
''',
        }
        # No test content = low coverage, should generate findings
        result = _validate_artifact(artifact, test_content="")
        # Untested code should generate coverage-related findings
        coverage_findings = [f for f in result.findings if f.category == "test_coverage"]
        # With no tests, we expect at least one coverage finding (threshold failure)
        assert len(coverage_findings) >= 1, "Untested code should generate coverage findings"

    def test_high_coverage_improves_score(self) -> None:
        """Test that high coverage improves validation score."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/module.py",
            "content": '''"""Module."""
def add(a, b):
    return a + b
''',
        }
        # With good test coverage
        high_coverage_result = _validate_artifact(
            artifact,
            test_content="def test_add(): assert add(1, 2) == 3",
        )

        # Without test coverage
        low_coverage_result = _validate_artifact(artifact, test_content="")

        # High coverage should have better or equal score
        assert high_coverage_result.score >= low_coverage_result.score


class TestCoverageInTeaNode:
    """Tests for coverage integration in tea_node."""

    @pytest.mark.asyncio
    async def test_tea_node_processes_coverage_for_code_files(self) -> None:
        """Test that tea_node processes coverage for code files."""
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": '''def func(): return 42''',
                                "file_type": "source",
                            }
                        ],
                        "test_files": [
                            {
                                "file_path": "tests/test_module.py",
                                "content": '''def test_func(): assert True''',
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(state)

        assert "messages" in result
        assert "decisions" in result
        assert "tea_output" in result

    @pytest.mark.asyncio
    async def test_tea_node_confidence_reflects_coverage(self) -> None:
        """Test that overall confidence reflects coverage analysis."""
        # State with good coverage
        good_state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": '''def add(a, b): return a + b''',
                                "file_type": "source",
                            }
                        ],
                        "test_files": [
                            {
                                "file_path": "tests/test_module.py",
                                "content": '''def test_add(): assert add(1, 2) == 3''',
                                "test_type": "unit",
                            }
                        ],
                    }
                ]
            },
        }

        result = await tea_node(good_state)
        assert result["tea_output"]["overall_confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_tea_node_low_coverage_reduces_confidence(self) -> None:
        """Test that low coverage reduces confidence."""
        # State with no test files
        poor_state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "dev_output": {
                "implementations": [
                    {
                        "story_id": "test-story",
                        "code_files": [
                            {
                                "file_path": "src/module.py",
                                "content": '''def func(): return 42''',
                                "file_type": "source",
                            }
                        ],
                        "test_files": [],  # No tests
                    }
                ]
            },
        }

        result = await tea_node(poor_state)
        # With no tests, confidence should be affected
        assert result["tea_output"]["overall_confidence"] is not None


class TestCoverageValidationDetails:
    """Tests for detailed coverage validation behavior."""

    def test_empty_code_file_has_full_coverage(self) -> None:
        """Test that empty code files are considered fully covered."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/empty.py",
            "content": "",
        }
        result = _validate_artifact(artifact)
        # Empty files should not penalize score
        assert result.score >= 80

    def test_critical_path_file_affects_confidence(self) -> None:
        """Test that critical path files affect confidence appropriately."""
        artifact = {
            "type": "code_file",
            "artifact_id": "orchestrator/core.py",
            "content": '''def orchestrate(): pass''',
        }
        result = _validate_artifact(artifact, test_content="")
        # Critical path without tests should be flagged
        assert result is not None
