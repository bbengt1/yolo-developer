"""Integration tests for DoD validation gate flow (Story 8.6 - Task 11).

Tests the integration between:
- dev_node function
- definition_of_done gate decorator
- Gate evaluation flow with advisory warnings

These tests verify the end-to-end gate integration behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.dev.node import dev_node

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_state() -> dict[str, Any]:
    """Create a minimal valid state for dev_node."""
    return {
        "messages": [],
        "current_agent": "dev",
        "handoff_context": None,
        "decisions": [],
        "gate_results": [],
        "advisory_warnings": [],
    }


@pytest.fixture
def state_with_stories() -> dict[str, Any]:
    """Create state with architect output containing stories."""
    return {
        "messages": [],
        "current_agent": "dev",
        "handoff_context": None,
        "decisions": [],
        "gate_results": [],
        "advisory_warnings": [],
        "architect_output": {
            "design_decisions": [
                {"story_id": "story-001", "pattern": "Repository"},
                {"story_id": "story-002", "pattern": "Factory"},
            ]
        },
    }


@pytest.fixture
def state_with_gate_warning() -> dict[str, Any]:
    """Create state with advisory gate warning."""
    return {
        "messages": [],
        "current_agent": "dev",
        "handoff_context": None,
        "decisions": [],
        "gate_results": [
            {
                "gate_name": "definition_of_done",
                "passed": False,
                "reason": "DoD score 50/100 below threshold 70",
            }
        ],
        "advisory_warnings": [
            {
                "gate_name": "definition_of_done",
                "reason": "DoD score 50/100 below threshold 70",
                "timestamp": "2026-01-11T12:00:00Z",
            }
        ],
        "architect_output": {
            "design_decisions": [
                {"story_id": "story-001", "pattern": "Repository"},
            ]
        },
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestDevNodeGateIntegration:
    """Integration tests for dev_node with definition_of_done gate."""

    @pytest.mark.asyncio
    async def test_dev_node_has_gate_decorator(self) -> None:
        """Test that dev_node has the quality_gate decorator applied."""
        # Check decorator metadata
        assert hasattr(dev_node, "_gate_name")
        assert dev_node._gate_name == "definition_of_done"
        assert hasattr(dev_node, "_gate_blocking")
        assert dev_node._gate_blocking is False  # Advisory mode

    @pytest.mark.asyncio
    async def test_dev_node_runs_with_empty_state(
        self, mock_state: dict[str, Any]
    ) -> None:
        """Test dev_node runs and returns valid output with empty state."""
        # Reset LLM router to avoid external calls
        from yolo_developer.agents.dev.node import _reset_llm_router

        _reset_llm_router()

        result = await dev_node(mock_state)

        # Should return state update dict
        assert "messages" in result
        assert "decisions" in result
        assert "dev_output" in result

    @pytest.mark.asyncio
    async def test_dev_node_processes_gate_warnings(
        self, state_with_gate_warning: dict[str, Any]
    ) -> None:
        """Test dev_node includes gate warnings in decision rationale."""
        from yolo_developer.agents.dev.node import _reset_llm_router

        _reset_llm_router()

        result = await dev_node(state_with_gate_warning)

        # Should have decision with gate warning info
        assert len(result["decisions"]) > 0
        decision = result["decisions"][0]

        # Decision rationale should include gate warning
        assert "Gate warnings" in decision.rationale

    @pytest.mark.asyncio
    async def test_dev_node_continues_on_advisory_failure(
        self, state_with_gate_warning: dict[str, Any]
    ) -> None:
        """Test dev_node continues execution despite advisory gate failure."""
        from yolo_developer.agents.dev.node import _reset_llm_router

        _reset_llm_router()

        result = await dev_node(state_with_gate_warning)

        # Should still produce output (advisory doesn't block)
        assert "dev_output" in result
        assert result["dev_output"]["implementations"] is not None

    @pytest.mark.asyncio
    async def test_gate_result_in_decision_record(
        self, state_with_gate_warning: dict[str, Any]
    ) -> None:
        """Test that gate failure is recorded in decision."""
        from yolo_developer.agents.dev.node import _reset_llm_router

        _reset_llm_router()

        result = await dev_node(state_with_gate_warning)

        # Extract decision
        decision = result["decisions"][0]

        # Should mention definition_of_done gate
        assert "definition_of_done" in decision.rationale.lower() or "gate" in decision.rationale.lower()


class TestGateEvaluationFlow:
    """Tests for the full gate evaluation flow."""

    @pytest.mark.asyncio
    async def test_gate_evaluator_is_registered(self) -> None:
        """Test that definition_of_done evaluator is registered."""
        from yolo_developer.gates.evaluators import get_evaluator

        evaluator = get_evaluator("definition_of_done")
        assert evaluator is not None

    @pytest.mark.asyncio
    async def test_gate_context_receives_state(self) -> None:
        """Test that gate context receives the full state."""
        from yolo_developer.gates.evaluators import get_evaluator
        from yolo_developer.gates.types import GateContext

        evaluator = get_evaluator("definition_of_done")
        assert evaluator is not None

        state = {
            "code": {
                "files": [
                    {"path": "src/test.py", "content": "def foo(): pass"},
                ]
            },
            "story": {"acceptance_criteria": []},
        }

        context = GateContext(state=state, gate_name="definition_of_done")
        result = await evaluator(context)

        # Should return a gate result
        assert hasattr(result, "passed")
        assert hasattr(result, "gate_name")
        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_gate_handles_missing_code_gracefully(self) -> None:
        """Test gate returns failure when code is missing from state."""
        from yolo_developer.gates.evaluators import get_evaluator
        from yolo_developer.gates.types import GateContext

        evaluator = get_evaluator("definition_of_done")
        assert evaluator is not None

        state: dict[str, Any] = {}  # No code key

        context = GateContext(state=state, gate_name="definition_of_done")
        result = await evaluator(context)

        # Should fail gracefully
        assert result.passed is False
        assert "Missing" in result.reason or "invalid" in result.reason.lower()


class TestDoDValidationIntegration:
    """Tests for DoD validation utility integration."""

    @pytest.mark.asyncio
    async def test_validate_implementation_dod_import(self) -> None:
        """Test validate_implementation_dod can be imported from dev module."""
        from yolo_developer.agents.dev import validate_implementation_dod

        # Should be callable
        assert callable(validate_implementation_dod)

    @pytest.mark.asyncio
    async def test_validate_artifact_dod_import(self) -> None:
        """Test validate_artifact_dod can be imported from dev module."""
        from yolo_developer.agents.dev import validate_artifact_dod

        assert callable(validate_artifact_dod)

    @pytest.mark.asyncio
    async def test_validate_dev_output_dod_import(self) -> None:
        """Test validate_dev_output_dod can be imported from dev module."""
        from yolo_developer.agents.dev import validate_dev_output_dod

        assert callable(validate_dev_output_dod)

    @pytest.mark.asyncio
    async def test_dod_types_import(self) -> None:
        """Test DoD types can be imported from dev module."""
        from yolo_developer.agents.dev import DoDChecklistItem, DoDValidationResult

        # Should be able to construct
        item = DoDChecklistItem(
            category="tests",
            item_name="test",
            passed=True,
            severity=None,
            message="test",
        )
        assert item.passed is True

        result = DoDValidationResult()
        assert result.score == 100

    @pytest.mark.asyncio
    async def test_full_validation_flow(self) -> None:
        """Test the full validation flow from DevOutput to results."""
        from yolo_developer.agents.dev import (
            CodeFile,
            DevOutput,
            ImplementationArtifact,
            TestFile,
            validate_dev_output_dod,
        )

        code_file = CodeFile(
            file_path="src/impl.py",
            content='''"""Module docstring."""

def public_function(arg: str) -> str:
    """Function docstring."""
    return arg
''',
            file_type="source",
        )

        test_file = TestFile(
            file_path="tests/test_impl.py",
            content='''"""Test module."""

def test_public_function() -> None:
    """Test function."""
    pass
''',
            test_type="unit",
        )

        artifact = ImplementationArtifact(
            story_id="story-001",
            code_files=(code_file,),
            test_files=(test_file,),
            implementation_status="completed",
        )

        output = DevOutput(
            implementations=(artifact,),
            processing_notes="Test output",
        )

        state = {"story": {"acceptance_criteria": []}}

        results = validate_dev_output_dod(output, state)

        assert len(results) == 1
        assert results[0].artifact_id == "story-001"
        assert isinstance(results[0].score, int)
