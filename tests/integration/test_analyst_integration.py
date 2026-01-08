"""Integration tests for Analyst agent (Story 5.1).

Tests the analyst_node integration with:
- LangGraph StateGraph
- Quality gate decorator
- YoloState message handling
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.agents.analyst.node import _parse_llm_response
from yolo_developer.agents.analyst.types import AnalystOutput, CrystallizedRequirement
from yolo_developer.gates import GateContext, GateResult, clear_evaluators, register_evaluator
from yolo_developer.orchestrator.state import YoloState


class TestAnalystNodeIntegration:
    """Integration tests for analyst_node with orchestration."""

    @pytest.fixture(autouse=True)
    def cleanup_evaluators(self) -> None:
        """Clean up gate evaluators after each test."""
        yield
        clear_evaluators()

    @pytest.fixture
    def sample_state(self) -> YoloState:
        """Create a sample YoloState for testing."""
        return {
            "messages": [
                HumanMessage(content="Build a todo application with user authentication")
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

    @pytest.mark.asyncio
    async def test_analyst_node_with_testability_gate_passing(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should work with testability gate when gate passes."""
        # Register a passing testability gate
        async def passing_gate(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="testability")

        register_evaluator("testability", passing_gate)

        # Run analyst node
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="test",
                    refined_text="test",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Should have messages and decisions
        assert "messages" in result
        assert len(result["messages"]) >= 1
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_analyst_node_with_testability_gate_blocking(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should be blocked when testability gate fails."""
        # Register a failing testability gate
        async def failing_gate(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name="testability",
                reason="Requirements not testable",
            )

        register_evaluator("testability", failing_gate)

        # Run analyst node - gate should block
        result = await analyst_node(sample_state)

        # Gate should have blocked execution
        assert result.get("gate_blocked") is True
        assert "Requirements not testable" in result.get("gate_failure", "")

    @pytest.mark.asyncio
    async def test_analyst_node_state_update_format(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should return proper state update dict format."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="orig",
                    refined_text="refined",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Should have gate_results from decorator
        assert "gate_results" in result or "messages" in result

        # Check message format if present
        if "messages" in result:
            for msg in result["messages"]:
                # Should have content
                assert hasattr(msg, "content")
                assert msg.content

    @pytest.mark.asyncio
    async def test_analyst_node_with_mocked_llm(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should work with mocked LLM responses."""
        # Mock the crystallized output that would come from LLM processing
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build a todo application",
                    refined_text="Create a web-based task management application",
                    category="functional",
                    testable=True,
                ),
                CrystallizedRequirement(
                    id="req-002",
                    original_text="with user authentication",
                    refined_text="Implement user authentication with email/password",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=("No mention of data persistence",),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Verify result structure
        assert isinstance(result, dict)
        if "messages" in result:
            assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_analyst_node_decision_recording(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should record decisions with correct agent attribution."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="test",
                    refined_text="test",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Verify decisions
        if "decisions" in result:
            for decision in result["decisions"]:
                assert decision.agent == "analyst"
                assert "Crystallized" in decision.summary


class TestLLMResponseParsing:
    """Tests for LLM response parsing functionality."""

    def test_parse_valid_json_response(self) -> None:
        """_parse_llm_response should parse valid JSON correctly."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "Original",
                    "refined_text": "Refined",
                    "category": "functional",
                    "testable": True,
                }
            ],
            "identified_gaps": ["Gap 1"],
            "contradictions": ["Contradiction 1"],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        assert result.requirements[0].id == "req-001"
        assert result.identified_gaps == ("Gap 1",)
        assert result.contradictions == ("Contradiction 1",)

    def test_parse_invalid_json_response(self) -> None:
        """_parse_llm_response should handle invalid JSON gracefully."""
        response = "This is not valid JSON"

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 0
        assert "Failed to parse LLM response" in result.identified_gaps

    def test_parse_empty_requirements(self) -> None:
        """_parse_llm_response should handle empty requirements list."""
        response = json.dumps({
            "requirements": [],
            "identified_gaps": [],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 0
        assert len(result.identified_gaps) == 0

    def test_parse_missing_fields(self) -> None:
        """_parse_llm_response should handle missing optional fields."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "Test",
                    "refined_text": "Test refined",
                    # Missing category and testable - should use defaults
                }
            ],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        # Should use default values
        assert result.requirements[0].category == "functional"
        assert result.requirements[0].testable is True
