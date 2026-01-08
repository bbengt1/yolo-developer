"""Unit tests for Analyst agent node (Story 5.1 Task 2).

Tests for analyst_node function and state management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.agents.analyst.types import AnalystOutput, CrystallizedRequirement
from yolo_developer.orchestrator.state import YoloState


class TestAnalystNode:
    """Tests for analyst_node function."""

    @pytest.fixture
    def mock_llm_output(self) -> AnalystOutput:
        """Mock AnalystOutput for testing."""
        return AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build a todo app",
                    refined_text="Create a web-based task management application",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

    @pytest.fixture
    def sample_state(self) -> YoloState:
        """Create a sample YoloState for testing."""
        return {
            "messages": [HumanMessage(content="Build a todo app with user authentication")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

    @pytest.mark.asyncio
    async def test_analyst_node_receives_yolo_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should receive YoloState TypedDict correctly."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        # Should return a dict, not raise any errors
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyst_node_returns_dict_not_yolo_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return dict with state updates, not YoloState."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        # Should be a plain dict
        assert isinstance(result, dict)

        # Should NOT have all YoloState fields (we only return updates)
        # Should NOT include current_agent (handoff does that)
        assert "current_agent" not in result

    @pytest.mark.asyncio
    async def test_analyst_node_returns_valid_messages(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return valid message updates."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 1

        # Message should have agent attribution
        msg = result["messages"][0]
        assert msg.additional_kwargs.get("agent") == "analyst"

    @pytest.mark.asyncio
    async def test_analyst_node_returns_valid_decisions(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return valid decision updates."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) >= 1

        # Decision should have analyst attribution
        decision = result["decisions"][0]
        assert decision.agent == "analyst"

    @pytest.mark.asyncio
    async def test_analyst_node_does_not_mutate_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should not mutate the input state."""
        original_messages_len = len(sample_state["messages"])
        original_decisions_len = len(sample_state["decisions"])

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            await analyst_node(sample_state)

        # Original state should be unchanged
        assert len(sample_state["messages"]) == original_messages_len
        assert len(sample_state["decisions"]) == original_decisions_len

    @pytest.mark.asyncio
    async def test_analyst_node_is_async(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should be an async function."""
        import asyncio
        import inspect

        from yolo_developer.agents.analyst import analyst_node

        # Verify it's a coroutine function
        assert inspect.iscoroutinefunction(analyst_node)

        # Verify it returns an awaitable
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            coro = analyst_node(sample_state)
            assert asyncio.iscoroutine(coro)
            await coro  # Clean up

    @pytest.mark.asyncio
    async def test_analyst_node_handles_empty_messages(
        self,
    ) -> None:
        """analyst_node should handle empty messages gracefully."""
        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        mock_output = AnalystOutput(
            requirements=(),
            identified_gaps=("No seed content provided",),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        assert isinstance(result, dict)
        assert "messages" in result
