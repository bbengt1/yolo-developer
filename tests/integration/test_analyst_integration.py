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


class TestCrystallizationIntegration:
    """Integration tests for crystallization flow (Story 5.2 Task 9)."""

    @pytest.mark.asyncio
    async def test_full_crystallization_flow_with_vague_content(self) -> None:
        """Test full crystallization flow detects vague terms and adds metadata."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a fast, scalable API")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        assert "messages" in result
        assert "decisions" in result
        # The output should include crystallized requirements with new fields
        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]
        assert len(requirements) >= 1

        # Check new fields are present in output
        req = requirements[0]
        assert "scope_notes" in req
        assert "implementation_hints" in req
        assert "confidence" in req

    @pytest.mark.asyncio
    async def test_crystallization_with_specific_content(self) -> None:
        """Test crystallization with specific, non-vague content."""
        state: YoloState = {
            "messages": [HumanMessage(content="Response time MUST be under 200ms")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]
        req = requirements[0]

        # Non-vague content should have full confidence
        assert req["confidence"] == 1.0
        assert req["scope_notes"] is None

    @pytest.mark.asyncio
    async def test_decision_includes_transformation_info(self) -> None:
        """Test that decisions capture transformation information."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a fast API endpoint")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        decisions = result["decisions"]
        assert len(decisions) >= 1
        decision = decisions[0]

        # Decision should have analyst attribution
        assert decision.agent == "analyst"
        # Decision should include requirement count in summary
        assert "requirement" in decision.summary.lower() or "crystallized" in decision.summary.lower()

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_mocked_llm(self) -> None:
        """Test that existing test patterns still work with enhanced output."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build an app",
                    refined_text="Create an application",
                    category="functional",
                    testable=True,
                    # New fields with defaults
                    scope_notes=None,
                    implementation_hints=(),
                    confidence=1.0,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        state: YoloState = {
            "messages": [HumanMessage(content="Build an app")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        # Should work exactly as before
        assert isinstance(result, dict)
        assert "messages" in result
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_scope_boundaries_in_output(self) -> None:
        """Test that scope boundaries are properly included in output."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a simple, easy-to-use REST API")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]

        # Vague content should have scope_notes
        assert len(requirements) >= 1
        # At least some requirements should have scope clarification
        has_scope_notes = any(req.get("scope_notes") for req in requirements)
        assert has_scope_notes, "Vague content should generate scope_notes"

    @pytest.mark.asyncio
    async def test_llm_response_with_new_fields_parses_correctly(self) -> None:
        """Test parsing LLM response that includes new fields."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "System should be fast",
                    "refined_text": "Response time < 200ms at 95th percentile",
                    "category": "non-functional",
                    "testable": True,
                    "scope_notes": "Applies to GET endpoints; POST excluded",
                    "implementation_hints": ["Use async handlers", "Add response caching"],
                    "confidence": 0.9,
                }
            ],
            "identified_gaps": [],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        req = result.requirements[0]
        assert req.scope_notes == "Applies to GET endpoints; POST excluded"
        assert req.implementation_hints == ("Use async handlers", "Add response caching")
        assert req.confidence == 0.9
