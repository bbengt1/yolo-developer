"""Unit tests for TEA agent node (Story 9.1).

Tests for the tea_node function and related helpers.
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from yolo_developer.agents.tea.node import (
    _calculate_overall_confidence,
    _validate_artifact,
    tea_node,
)
from yolo_developer.agents.tea.types import ValidationResult
from yolo_developer.orchestrator.state import YoloState


@pytest.fixture
def empty_state() -> YoloState:
    """Create an empty state for testing."""
    return {
        "messages": [],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
    }


@pytest.fixture
def state_with_artifacts() -> YoloState:
    """Create a state with dev_output artifacts for testing."""
    return {
        "messages": [],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
        "dev_output": {
            "implementations": [
                {
                    "story_id": "story-001",
                    "code_files": [
                        {
                            "file_path": "src/main.py",
                            "content": '"""Module docstring."""\n\ndef main() -> None:\n    pass',
                            "file_type": "source",
                        },
                    ],
                    "test_files": [
                        {
                            "file_path": "tests/test_main.py",
                            "content": "def test_main():\n    assert True",
                            "test_type": "unit",
                        },
                    ],
                },
            ],
        },
    }


class TestTeaNodeSignature:
    """Tests for tea_node function signature (AC1)."""

    def test_tea_node_is_async(self) -> None:
        """Test that tea_node is an async function."""
        # The wrapped function has an inner async function
        # We can check if calling it returns a coroutine
        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
        }
        result = tea_node(state)
        assert asyncio.iscoroutine(result)
        # Clean up the coroutine to avoid warning
        result.close()

    def test_tea_node_accepts_yolo_state(self, empty_state: YoloState) -> None:
        """Test that tea_node accepts YoloState as input."""
        result = tea_node(empty_state)
        assert asyncio.iscoroutine(result)
        result.close()

    def test_tea_node_is_importable(self) -> None:
        """Test that tea_node is importable from yolo_developer.agents.tea."""
        from yolo_developer.agents.tea import tea_node as imported_tea_node

        assert imported_tea_node is not None
        assert callable(imported_tea_node)


class TestTeaNodeReturns:
    """Tests for tea_node return values (AC6)."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, empty_state: YoloState) -> None:
        """Test that tea_node returns a dictionary."""
        result = await tea_node(empty_state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_messages(self, empty_state: YoloState) -> None:
        """Test that tea_node returns messages key."""
        result = await tea_node(empty_state)
        assert "messages" in result
        assert isinstance(result["messages"], list)

    @pytest.mark.asyncio
    async def test_returns_decisions(self, empty_state: YoloState) -> None:
        """Test that tea_node returns decisions key."""
        result = await tea_node(empty_state)
        assert "decisions" in result
        assert isinstance(result["decisions"], list)

    @pytest.mark.asyncio
    async def test_returns_tea_output(self, empty_state: YoloState) -> None:
        """Test that tea_node returns tea_output key."""
        result = await tea_node(empty_state)
        assert "tea_output" in result
        assert isinstance(result["tea_output"], dict)

    @pytest.mark.asyncio
    async def test_decision_has_tea_agent(self, state_with_artifacts: YoloState) -> None:
        """Test that Decision has agent='tea'."""
        result = await tea_node(state_with_artifacts)
        decisions = result["decisions"]
        assert len(decisions) >= 1
        assert decisions[0].agent == "tea"

    @pytest.mark.asyncio
    async def test_message_created_correctly(self, state_with_artifacts: YoloState) -> None:
        """Test that message is created with proper attributes."""
        result = await tea_node(state_with_artifacts)
        messages = result["messages"]
        assert len(messages) == 1
        msg = messages[0]
        assert hasattr(msg, "content")
        assert "TEA validation complete" in msg.content


class TestTeaNodeStateImmutability:
    """Tests for state immutability (AC6)."""

    @pytest.mark.asyncio
    async def test_input_state_not_mutated(self, state_with_artifacts: YoloState) -> None:
        """Test that input state is not mutated."""
        original_state = copy.deepcopy(state_with_artifacts)
        await tea_node(state_with_artifacts)

        # Compare key values
        assert state_with_artifacts["messages"] == original_state["messages"]
        assert state_with_artifacts["decisions"] == original_state["decisions"]
        assert state_with_artifacts["current_agent"] == original_state["current_agent"]

    @pytest.mark.asyncio
    async def test_returns_only_updates(self, empty_state: YoloState) -> None:
        """Test that only state updates are returned, not full state."""
        result = await tea_node(empty_state)

        # Should not contain keys that weren't changed
        assert "current_agent" not in result
        assert "handoff_context" not in result


class TestTeaNodeEmptyState:
    """Tests for handling empty state."""

    @pytest.mark.asyncio
    async def test_handles_empty_state_gracefully(self, empty_state: YoloState) -> None:
        """Test that tea_node handles empty state without errors."""
        result = await tea_node(empty_state)
        assert result is not None
        assert "messages" in result
        assert "decisions" in result
        assert "tea_output" in result

    @pytest.mark.asyncio
    async def test_empty_state_confidence_is_full(self, empty_state: YoloState) -> None:
        """Test that empty state results in full confidence."""
        result = await tea_node(empty_state)
        tea_output = result["tea_output"]
        assert tea_output["overall_confidence"] == 1.0
        assert tea_output["deployment_recommendation"] == "deploy"


class TestValidateArtifact:
    """Tests for _validate_artifact helper function."""

    def test_validates_code_file(self) -> None:
        """Test validation of a code file artifact."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/main.py",
            "content": '"""Module docstring."""\n\ndef main() -> None:\n    pass',
            "file_type": "source",
        }
        result = _validate_artifact(artifact)
        assert isinstance(result, ValidationResult)
        assert result.artifact_id == "src/main.py"

    def test_validates_test_file(self) -> None:
        """Test validation of a test file artifact."""
        artifact = {
            "type": "test_file",
            "artifact_id": "tests/test_main.py",
            "content": "def test_main():\n    assert True",
            "test_type": "unit",
        }
        result = _validate_artifact(artifact)
        assert isinstance(result, ValidationResult)
        assert result.artifact_id == "tests/test_main.py"

    def test_detects_missing_docstring(self) -> None:
        """Test that missing docstring is detected."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/no_doc.py",
            "content": "def main():\n    pass",
            "file_type": "source",
        }
        result = _validate_artifact(artifact)
        # Should have documentation finding
        doc_findings = [f for f in result.findings if f.category == "documentation"]
        assert len(doc_findings) >= 1

    def test_detects_missing_type_hints(self) -> None:
        """Test that missing return type hints are detected."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/no_types.py",
            "content": '"""Module."""\n\ndef main():\n    pass',  # No -> annotation
            "file_type": "source",
        }
        result = _validate_artifact(artifact)
        # Should have code_quality finding
        quality_findings = [f for f in result.findings if f.category == "code_quality"]
        assert len(quality_findings) >= 1

    def test_detects_missing_assertions_in_test(self) -> None:
        """Test that missing assertions in tests are detected."""
        artifact = {
            "type": "test_file",
            "artifact_id": "tests/empty_test.py",
            "content": "def test_empty():\n    pass",  # No assert
            "test_type": "unit",
        }
        result = _validate_artifact(artifact)
        # Should have test_coverage finding
        coverage_findings = [f for f in result.findings if f.category == "test_coverage"]
        assert len(coverage_findings) >= 1

    def test_returns_passed_for_good_code(self) -> None:
        """Test that good code returns passed status."""
        artifact = {
            "type": "code_file",
            "artifact_id": "src/good.py",
            "content": '"""Good module."""\n\ndef main() -> None:\n    """Main function."""\n    pass',
            "file_type": "source",
        }
        result = _validate_artifact(artifact)
        # Should pass with score of 100
        assert result.validation_status == "passed"
        assert result.score == 100


class TestCalculateOverallConfidence:
    """Tests for _calculate_overall_confidence helper function."""

    def test_empty_results_returns_full_confidence(self) -> None:
        """Test that empty results return 1.0 confidence."""
        confidence, recommendation = _calculate_overall_confidence([])
        assert confidence == 1.0
        assert recommendation == "deploy"

    def test_all_passed_returns_high_confidence(self) -> None:
        """Test that all passed results return high confidence."""
        results = [
            ValidationResult(
                artifact_id="file1.py",
                validation_status="passed",
                score=100,
            ),
            ValidationResult(
                artifact_id="file2.py",
                validation_status="passed",
                score=100,
            ),
        ]
        confidence, recommendation = _calculate_overall_confidence(results)
        assert confidence == 1.0
        assert recommendation == "deploy"

    def test_warnings_return_deploy_with_warnings(self) -> None:
        """Test that warnings return deploy_with_warnings."""
        results = [
            ValidationResult(
                artifact_id="file1.py",
                validation_status="warning",
                score=80,
            ),
        ]
        confidence, recommendation = _calculate_overall_confidence(results)
        assert confidence == 0.8
        assert recommendation == "deploy_with_warnings"

    def test_failures_return_block(self) -> None:
        """Test that failures return block recommendation."""
        results = [
            ValidationResult(
                artifact_id="file1.py",
                validation_status="failed",
                score=50,
            ),
        ]
        confidence, recommendation = _calculate_overall_confidence(results)
        assert confidence == 0.5
        assert recommendation == "block"

    def test_mixed_results_calculate_average(self) -> None:
        """Test that mixed results calculate weighted average."""
        results = [
            ValidationResult(
                artifact_id="file1.py",
                validation_status="passed",
                score=100,
            ),
            ValidationResult(
                artifact_id="file2.py",
                validation_status="warning",
                score=80,
            ),
            ValidationResult(
                artifact_id="file3.py",
                validation_status="failed",
                score=60,
            ),
        ]
        confidence, recommendation = _calculate_overall_confidence(results)
        # Average: (100 + 80 + 60) / 3 = 80
        assert confidence == 0.8
        # Failed takes precedence
        assert recommendation == "block"


class TestTeaNodeIntegration:
    """Integration tests for tea_node with artifacts."""

    @pytest.mark.asyncio
    async def test_processes_all_artifacts(self, state_with_artifacts: YoloState) -> None:
        """Test that all artifacts are processed."""
        result = await tea_node(state_with_artifacts)
        tea_output = result["tea_output"]

        # Should have validation results for 2 artifacts (1 code, 1 test)
        assert len(tea_output["validation_results"]) == 2

    @pytest.mark.asyncio
    async def test_processing_notes_include_stats(
        self, state_with_artifacts: YoloState
    ) -> None:
        """Test that processing notes include statistics."""
        result = await tea_node(state_with_artifacts)
        tea_output = result["tea_output"]

        notes = tea_output["processing_notes"]
        assert "Validated 2 artifacts" in notes
        assert "confidence" in notes.lower()

    @pytest.mark.asyncio
    async def test_decision_includes_rationale(
        self, state_with_artifacts: YoloState
    ) -> None:
        """Test that decision includes rationale."""
        result = await tea_node(state_with_artifacts)
        decisions = result["decisions"]

        assert len(decisions) >= 1
        decision = decisions[0]
        assert decision.rationale is not None
        assert len(decision.rationale) > 0

    @pytest.mark.asyncio
    async def test_decision_includes_related_artifacts(
        self, state_with_artifacts: YoloState
    ) -> None:
        """Test that decision includes related artifacts."""
        result = await tea_node(state_with_artifacts)
        decisions = result["decisions"]

        assert len(decisions) >= 1
        decision = decisions[0]
        assert len(decision.related_artifacts) == 2
        assert "src/main.py" in decision.related_artifacts
        assert "tests/test_main.py" in decision.related_artifacts


class TestTeaNodeRetryBehavior:
    """Tests for retry decorator behavior (AC4)."""

    def test_tea_node_has_retry_decorator(self) -> None:
        """Test that tea_node has retry decorator applied."""
        # The retry decorator adds __wrapped__ attribute
        # We can verify by checking the function has tenacity retry statistics
        assert hasattr(tea_node, "retry")

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry_needed(
        self, empty_state: YoloState
    ) -> None:
        """Test that successful execution doesn't trigger retries."""
        # Normal execution should work without retry
        result = await tea_node(empty_state)
        assert result is not None
        assert "tea_output" in result

    @pytest.mark.asyncio
    async def test_retry_statistics_accessible(self, empty_state: YoloState) -> None:
        """Test that retry statistics are accessible after execution."""
        # Execute the function
        await tea_node(empty_state)

        # The retry decorator should have statistics attribute
        # This verifies the decorator is properly configured
        assert hasattr(tea_node, "retry")
        retry_state = tea_node.retry
        assert retry_state is not None
