"""Tests for architect_node function (Story 7.1, Task 10).

Tests verify that the architect_node function follows the correct patterns:
- Async function signature
- Returns state update dict (not full state)
- Never mutates input state
- Includes Decision with agent="architect"
- Uses create_agent_message for output
"""

from __future__ import annotations

import asyncio
import copy
import inspect

import pytest

from yolo_developer.agents.architect.node import architect_node


class TestArchitectNodeSignature:
    """Test architect_node function signature and async behavior."""

    def test_architect_node_is_coroutine_function(self) -> None:
        """Test that architect_node is an async function."""
        # The decorator wraps it, but the underlying function should be async
        # After decoration, it's still a coroutine function
        assert asyncio.iscoroutinefunction(architect_node)

    def test_architect_node_is_callable(self) -> None:
        """Test that architect_node is callable."""
        assert callable(architect_node)


class TestArchitectNodeReturnValue:
    """Test architect_node return value structure."""

    @pytest.mark.asyncio
    async def test_returns_dict(self) -> None:
        """Test that architect_node returns a dict."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_messages_key(self) -> None:
        """Test that result contains messages key."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_returns_decisions_key(self) -> None:
        """Test that result contains decisions key."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) >= 1

    @pytest.mark.asyncio
    async def test_returns_architect_output_key(self) -> None:
        """Test that result contains architect_output key."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        assert "architect_output" in result
        assert isinstance(result["architect_output"], dict)


class TestDecisionRecord:
    """Test that Decision record is properly created."""

    @pytest.mark.asyncio
    async def test_decision_has_architect_agent(self) -> None:
        """Test that Decision has agent='architect'."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        decisions = result["decisions"]
        assert len(decisions) >= 1
        # Decision is a dataclass, check its agent attribute
        decision = decisions[0]
        assert hasattr(decision, "agent")
        assert decision.agent == "architect"

    @pytest.mark.asyncio
    async def test_decision_has_summary(self) -> None:
        """Test that Decision has a summary."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        decision = result["decisions"][0]
        assert hasattr(decision, "summary")
        assert decision.summary  # Non-empty

    @pytest.mark.asyncio
    async def test_decision_has_rationale(self) -> None:
        """Test that Decision has a rationale."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        decision = result["decisions"][0]
        assert hasattr(decision, "rationale")
        assert decision.rationale  # Non-empty


class TestMessageCreation:
    """Test that messages are created with create_agent_message."""

    @pytest.mark.asyncio
    async def test_message_has_architect_agent(self) -> None:
        """Test that message has architect agent in additional_kwargs."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        message = result["messages"][0]
        assert hasattr(message, "additional_kwargs")
        assert message.additional_kwargs.get("agent") == "architect"

    @pytest.mark.asyncio
    async def test_message_has_output_metadata(self) -> None:
        """Test that message includes output in metadata."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        message = result["messages"][0]
        assert "output" in message.additional_kwargs

    @pytest.mark.asyncio
    async def test_message_has_content(self) -> None:
        """Test that message has non-empty content."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        message = result["messages"][0]
        assert hasattr(message, "content")
        assert message.content  # Non-empty


class TestStateImmutability:
    """Test that architect_node does not mutate input state."""

    @pytest.mark.asyncio
    async def test_input_state_not_mutated(self) -> None:
        """Test that input state is not modified."""
        original_state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }
        state_copy = copy.deepcopy(original_state)

        await architect_node(state_copy)

        # The original state should not be modified
        # Note: The decorator may modify the working copy, but we pass a copy
        assert state_copy["messages"] == original_state["messages"]
        assert state_copy["decisions"] == original_state["decisions"]

    @pytest.mark.asyncio
    async def test_returns_only_updates(self) -> None:
        """Test that result contains only updates, not full state."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
            "some_other_key": "value",
        }

        result = await architect_node(state)

        # Result should not include keys from input state
        assert "current_agent" not in result
        assert "handoff_context" not in result
        assert "some_other_key" not in result


class TestEmptyStateHandling:
    """Test graceful handling of empty or minimal state."""

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self) -> None:
        """Test that empty messages list is handled."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        result = await architect_node(state)

        assert "messages" in result
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_handles_minimal_state(self) -> None:
        """Test that minimal state is handled gracefully."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "architecture": {},
        }

        # Should not raise
        result = await architect_node(state)
        assert result is not None
