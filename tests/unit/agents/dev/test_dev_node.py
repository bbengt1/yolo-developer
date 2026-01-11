"""dev_node function tests for Dev agent (Story 8.1, AC1, AC4, AC6).

Tests for the dev_node LangGraph node function including async patterns,
state updates, and Decision/message creation.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from yolo_developer.agents.dev.node import dev_node
from yolo_developer.orchestrator.state import YoloState


class TestDevNodeIsAsync:
    """Tests for dev_node async patterns (AC1, AC4)."""

    def test_dev_node_is_async_function(self) -> None:
        """Test dev_node is an async function."""
        assert asyncio.iscoroutinefunction(dev_node)

    @pytest.mark.asyncio
    async def test_dev_node_can_be_awaited(self) -> None:
        """Test dev_node can be awaited."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert result is not None


class TestDevNodeReturnStructure:
    """Tests for dev_node return structure (AC6)."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_messages(self) -> None:
        """Test returns dict with messages key."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert "messages" in result
        assert isinstance(result["messages"], list)

    @pytest.mark.asyncio
    async def test_returns_dict_with_decisions(self) -> None:
        """Test returns dict with decisions key."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert "decisions" in result
        assert isinstance(result["decisions"], list)

    @pytest.mark.asyncio
    async def test_returns_dict_with_dev_output(self) -> None:
        """Test returns dict with dev_output key."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert "dev_output" in result
        assert isinstance(result["dev_output"], dict)


class TestDevNodeDecision:
    """Tests for Decision record creation (AC6)."""

    @pytest.mark.asyncio
    async def test_decision_has_agent_dev(self) -> None:
        """Test Decision has agent='dev'."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert len(result["decisions"]) == 1
        decision = result["decisions"][0]
        assert decision.agent == "dev"

    @pytest.mark.asyncio
    async def test_decision_has_summary(self) -> None:
        """Test Decision has summary."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        decision = result["decisions"][0]
        assert decision.summary is not None
        assert len(decision.summary) > 0

    @pytest.mark.asyncio
    async def test_decision_has_rationale(self) -> None:
        """Test Decision has rationale."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        decision = result["decisions"][0]
        assert decision.rationale is not None


class TestDevNodeMessage:
    """Tests for message creation (AC6)."""

    @pytest.mark.asyncio
    async def test_message_created_with_agent_metadata(self) -> None:
        """Test message has agent metadata."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert message.additional_kwargs.get("agent") == "dev"


class TestDevNodeStateMutation:
    """Tests for state immutability (AC6)."""

    @pytest.mark.asyncio
    async def test_input_state_not_mutated(self) -> None:
        """Test input state is not mutated."""
        original_messages: list[Any] = []
        original_decisions: list[Any] = []
        state: YoloState = {
            "messages": original_messages,
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": original_decisions,
        }
        state_copy = copy.deepcopy(state)

        await dev_node(state)

        # Original state should be unchanged
        assert state["messages"] == state_copy["messages"]
        assert state["decisions"] == state_copy["decisions"]
        assert state["current_agent"] == state_copy["current_agent"]


class TestDevNodeEmptyState:
    """Tests for empty state handling."""

    @pytest.mark.asyncio
    async def test_handles_empty_state_gracefully(self) -> None:
        """Test handles state with no stories."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = await dev_node(state)
        assert "messages" in result
        assert "decisions" in result
        assert "dev_output" in result


class TestDevNodeWithStories:
    """Tests for dev_node with stories in state."""

    @pytest.mark.asyncio
    async def test_processes_stories_from_architect_output(self) -> None:
        """Test processes stories from architect_output."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                ],
            },
        }
        result = await dev_node(state)
        dev_output = result["dev_output"]
        assert len(dev_output["implementations"]) == 1
        assert dev_output["implementations"][0]["story_id"] == "story-001"

    @pytest.mark.asyncio
    async def test_generates_code_files(self) -> None:
        """Test generates code files for stories."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                ],
            },
        }
        result = await dev_node(state)
        dev_output = result["dev_output"]
        impl = dev_output["implementations"][0]
        assert len(impl["code_files"]) >= 1

    @pytest.mark.asyncio
    async def test_generates_test_files(self) -> None:
        """Test generates test files for stories."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                ],
            },
        }
        result = await dev_node(state)
        dev_output = result["dev_output"]
        impl = dev_output["implementations"][0]
        assert len(impl["test_files"]) >= 1
