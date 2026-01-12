"""Tests for SM agent node (Story 10.2).

Tests the sm_node function and its helper functions:
- _analyze_current_state
- _get_next_agent
- _check_for_escalation
- _check_for_circular_logic
- sm_node (main entry point)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yolo_developer.agents.sm.node import (
    _analyze_current_state,
    _check_for_circular_logic,
    _check_for_escalation,
    _count_recent_exchanges,
    _detect_circular_pattern,
    _get_natural_successor,
    _get_next_agent,
    _get_recovery_agent,
    _get_routing_rationale,
    sm_node,
)
from yolo_developer.agents.sm.types import AgentExchange
from yolo_developer.orchestrator.state import YoloState


def create_test_state(**overrides: Any) -> YoloState:
    """Create a test YoloState with optional overrides."""
    default_state: YoloState = {
        "messages": [],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }
    return {**default_state, **overrides}  # type: ignore[return-value]


def create_agent_message(content: str, agent: str) -> AIMessage:
    """Create an AIMessage with agent attribution."""
    return AIMessage(
        content=content,
        additional_kwargs={"agent": agent},
    )


class TestAnalyzeCurrentState:
    """Tests for _analyze_current_state function."""

    def test_analyze_empty_state(self) -> None:
        """Test analyzing state with minimal data."""
        state = create_test_state()
        analysis = _analyze_current_state(state)

        assert analysis["current_agent"] == "analyst"
        assert analysis["message_count"] == 0
        assert analysis["has_handoff_context"] is False
        assert analysis["needs_architecture"] is False
        assert analysis["gate_blocked"] is False
        assert analysis["escalate_to_human"] is False
        assert analysis["decision_count"] == 0

    def test_analyze_state_with_messages(self) -> None:
        """Test analyzing state with messages."""
        messages = [
            HumanMessage(content="Build an app"),
            create_agent_message("Requirements analyzed", "analyst"),
        ]
        state = create_test_state(messages=messages)
        analysis = _analyze_current_state(state)

        assert analysis["message_count"] == 2

    def test_analyze_state_with_flags(self) -> None:
        """Test analyzing state with various flags set."""
        state = create_test_state(
            current_agent="pm",
            needs_architecture=True,
            gate_blocked=True,
            escalate_to_human=True,
        )
        analysis = _analyze_current_state(state)

        assert analysis["current_agent"] == "pm"
        assert analysis["needs_architecture"] is True
        assert analysis["gate_blocked"] is True
        assert analysis["escalate_to_human"] is True


class TestCountRecentExchanges:
    """Tests for _count_recent_exchanges function."""

    def test_count_no_exchanges(self) -> None:
        """Test counting exchanges with no messages."""
        state = create_test_state()
        count, exchanges = _count_recent_exchanges(state)

        assert count == 0
        assert exchanges == []

    def test_count_single_agent_messages(self) -> None:
        """Test counting when all messages are from same agent."""
        messages = [
            create_agent_message("Message 1", "analyst"),
            create_agent_message("Message 2", "analyst"),
        ]
        state = create_test_state(messages=messages)
        count, exchanges = _count_recent_exchanges(state)

        assert count == 0  # No handoffs between different agents

    def test_count_multiple_exchanges(self) -> None:
        """Test counting multiple agent exchanges."""
        messages = [
            create_agent_message("Requirements", "analyst"),
            create_agent_message("Stories created", "pm"),
            create_agent_message("Design ready", "architect"),
            create_agent_message("Implementation done", "dev"),
        ]
        state = create_test_state(messages=messages)
        count, exchanges = _count_recent_exchanges(state)

        assert count == 3  # analyst->pm, pm->architect, architect->dev
        assert len(exchanges) == 3


class TestDetectCircularPattern:
    """Tests for _detect_circular_pattern function."""

    def test_no_pattern_with_few_exchanges(self) -> None:
        """Test no circular pattern with few exchanges."""
        exchanges = [
            AgentExchange(source_agent="analyst", target_agent="pm", exchange_type="handoff"),
            AgentExchange(source_agent="pm", target_agent="architect", exchange_type="handoff"),
        ]

        assert _detect_circular_pattern(exchanges) is False

    def test_no_pattern_with_different_pairs(self) -> None:
        """Test no circular pattern when different agent pairs communicate."""
        exchanges = [
            AgentExchange(source_agent="analyst", target_agent="pm", exchange_type="handoff"),
            AgentExchange(source_agent="pm", target_agent="architect", exchange_type="handoff"),
            AgentExchange(source_agent="architect", target_agent="dev", exchange_type="handoff"),
            AgentExchange(source_agent="dev", target_agent="tea", exchange_type="handoff"),
        ]

        assert _detect_circular_pattern(exchanges) is False

    def test_detect_circular_pattern(self) -> None:
        """Test detecting circular pattern when same pair exchanges >3 times."""
        # Dev and TEA ping-ponging 4 times
        exchanges = [
            AgentExchange(source_agent="dev", target_agent="tea", exchange_type="handoff"),
            AgentExchange(source_agent="tea", target_agent="dev", exchange_type="response"),
            AgentExchange(source_agent="dev", target_agent="tea", exchange_type="handoff"),
            AgentExchange(source_agent="tea", target_agent="dev", exchange_type="response"),
        ]

        assert _detect_circular_pattern(exchanges) is True


class TestCheckForEscalation:
    """Tests for _check_for_escalation function."""

    def test_no_escalation_default(self) -> None:
        """Test no escalation with default state."""
        state = create_test_state()
        should_escalate, reason = _check_for_escalation(state)

        assert should_escalate is False
        assert reason is None

    def test_escalation_human_requested(self) -> None:
        """Test escalation when human escalation flag is set."""
        state = create_test_state(escalate_to_human=True)
        should_escalate, reason = _check_for_escalation(state)

        assert should_escalate is True
        assert reason == "human_requested"

    def test_escalation_agent_failure(self) -> None:
        """Test escalation when agent failure flag is set."""
        state = create_test_state(agent_failure=True)
        should_escalate, reason = _check_for_escalation(state)

        assert should_escalate is True
        assert reason == "agent_failure"


class TestCheckForCircularLogic:
    """Tests for _check_for_circular_logic function."""

    def test_no_circular_logic_default(self) -> None:
        """Test no circular logic with default state."""
        state = create_test_state()
        is_circular, exchanges = _check_for_circular_logic(state)

        assert is_circular is False
        assert exchanges == []

    def test_circular_logic_detected(self) -> None:
        """Test circular logic detection with ping-pong messages."""
        messages = [
            create_agent_message("Try fix", "dev"),
            create_agent_message("Still failing", "tea"),
            create_agent_message("Try again", "dev"),
            create_agent_message("Still failing", "tea"),
            create_agent_message("Another try", "dev"),
            create_agent_message("Still failing", "tea"),
            create_agent_message("One more", "dev"),
            create_agent_message("Nope", "tea"),
        ]
        state = create_test_state(messages=messages)
        is_circular, exchanges = _check_for_circular_logic(state)

        # Should have detected circular pattern
        assert is_circular is True


class TestGetRecoveryAgent:
    """Tests for _get_recovery_agent function."""

    def test_tea_blocked_routes_to_dev(self) -> None:
        """Test that TEA blocked routes back to dev."""
        state = create_test_state(current_agent="tea", gate_blocked=True)
        recovery = _get_recovery_agent(state)

        assert recovery == "dev"

    def test_architect_blocked_routes_to_pm(self) -> None:
        """Test that architect blocked routes back to PM."""
        state = create_test_state(current_agent="architect", gate_blocked=True)
        recovery = _get_recovery_agent(state)

        assert recovery == "pm"

    def test_default_routes_to_analyst(self) -> None:
        """Test that default blocked routes to analyst."""
        state = create_test_state(current_agent="unknown", gate_blocked=True)
        recovery = _get_recovery_agent(state)

        assert recovery == "analyst"


class TestGetNaturalSuccessor:
    """Tests for _get_natural_successor function."""

    def test_analyst_to_pm(self) -> None:
        """Test analyst naturally goes to PM."""
        state = create_test_state()
        successor = _get_natural_successor("analyst", state)

        assert successor == "pm"

    def test_pm_to_architect_when_needed(self) -> None:
        """Test PM goes to architect when architecture is needed."""
        state = create_test_state(needs_architecture=True)
        successor = _get_natural_successor("pm", state)

        assert successor == "architect"

    def test_pm_to_dev_when_no_architecture_needed(self) -> None:
        """Test PM goes directly to dev when no architecture needed."""
        state = create_test_state(needs_architecture=False)
        successor = _get_natural_successor("pm", state)

        assert successor == "dev"

    def test_architect_to_dev(self) -> None:
        """Test architect naturally goes to dev."""
        state = create_test_state()
        successor = _get_natural_successor("architect", state)

        assert successor == "dev"

    def test_dev_to_tea(self) -> None:
        """Test dev naturally goes to TEA."""
        state = create_test_state()
        successor = _get_natural_successor("dev", state)

        assert successor == "tea"

    def test_tea_to_dev(self) -> None:
        """Test TEA naturally goes to dev (for fixes or completion)."""
        state = create_test_state()
        successor = _get_natural_successor("tea", state)

        assert successor == "dev"


class TestGetNextAgent:
    """Tests for _get_next_agent function."""

    def test_escalate_when_human_requested(self) -> None:
        """Test escalation when human explicitly requested."""
        state = create_test_state(escalate_to_human=True)
        decision, rationale = _get_next_agent(state)

        assert decision == "escalate"
        assert "human_requested" in rationale

    def test_recovery_when_gate_blocked(self) -> None:
        """Test recovery routing when gate is blocked."""
        state = create_test_state(current_agent="tea", gate_blocked=True)
        decision, rationale = _get_next_agent(state)

        assert decision == "dev"  # TEA blocked routes to dev
        assert "Gate blocked" in rationale or "recovery" in rationale.lower()

    def test_natural_progression_analyst_to_pm(self) -> None:
        """Test natural progression from analyst to PM."""
        state = create_test_state(current_agent="analyst")
        decision, rationale = _get_next_agent(state)

        assert decision == "pm"
        assert "Natural" in rationale or "progression" in rationale.lower()

    def test_unknown_agent_defaults_to_analyst(self) -> None:
        """Test that unknown agent defaults to analyst."""
        state = create_test_state(current_agent="unknown_agent")
        decision, rationale = _get_next_agent(state)

        assert decision == "analyst"
        assert "Unknown" in rationale or "defaulting" in rationale.lower()


class TestGetRoutingRationale:
    """Tests for _get_routing_rationale function."""

    def test_escalation_rationale(self) -> None:
        """Test rationale for escalation routing."""
        state = create_test_state(escalate_to_human=True)
        analysis = _analyze_current_state(state)
        rationale = _get_routing_rationale(state, "escalate", analysis)

        assert "escalation" in rationale.lower()

    def test_gate_blocked_rationale(self) -> None:
        """Test rationale for gate blocked routing."""
        state = create_test_state(gate_blocked=True)
        analysis = {"current_agent": "tea", "message_count": 5, "decision_count": 2, "gate_blocked": True}
        rationale = _get_routing_rationale(state, "dev", analysis)

        assert "Gate blocked" in rationale

    def test_normal_progression_rationale(self) -> None:
        """Test rationale for normal progression."""
        state = create_test_state()
        analysis = {"current_agent": "analyst", "message_count": 1, "decision_count": 0, "gate_blocked": False}
        rationale = _get_routing_rationale(state, "pm", analysis)

        assert "progression" in rationale.lower() or "pm" in rationale.lower()


class TestSMNode:
    """Tests for sm_node function (AC #1, #2, #3, #4)."""

    @pytest.mark.asyncio
    async def test_sm_node_returns_state_update(self) -> None:
        """Test that sm_node returns a state update dict (AC #1)."""
        state = create_test_state(current_agent="analyst")
        result = await sm_node(state)

        assert isinstance(result, dict)
        assert "messages" in result
        assert "decisions" in result
        assert "sm_output" in result
        assert "routing_decision" in result

    @pytest.mark.asyncio
    async def test_sm_node_routes_to_any_agent(self) -> None:
        """Test that sm_node can route to any agent (AC #1)."""
        # Test routing to PM
        state = create_test_state(current_agent="analyst")
        result = await sm_node(state)
        assert result["routing_decision"] == "pm"

        # Test routing to architect
        state = create_test_state(current_agent="pm", needs_architecture=True)
        result = await sm_node(state)
        assert result["routing_decision"] == "architect"

        # Test routing to dev
        state = create_test_state(current_agent="architect")
        result = await sm_node(state)
        assert result["routing_decision"] == "dev"

        # Test routing to TEA
        state = create_test_state(current_agent="dev")
        result = await sm_node(state)
        assert result["routing_decision"] == "tea"

        # Test routing to escalate
        state = create_test_state(current_agent="analyst", escalate_to_human=True)
        result = await sm_node(state)
        assert result["routing_decision"] == "escalate"

    @pytest.mark.asyncio
    async def test_sm_node_makes_decisions_based_on_state(self) -> None:
        """Test that sm_node evaluates state for decisions (AC #2)."""
        # Test needs_architecture flag
        state = create_test_state(current_agent="pm", needs_architecture=True)
        result = await sm_node(state)
        assert result["routing_decision"] == "architect"

        # Test gate_blocked flag
        state = create_test_state(current_agent="tea", gate_blocked=True)
        result = await sm_node(state)
        assert result["routing_decision"] == "dev"

    @pytest.mark.asyncio
    async def test_sm_node_logs_with_structured_format(self) -> None:
        """Test that sm_node logs routing decisions with structured format (AC #3)."""
        state = create_test_state(current_agent="analyst")
        result = await sm_node(state)

        # Check sm_output has structured format
        sm_output = result["sm_output"]
        assert "routing_decision" in sm_output
        assert "routing_rationale" in sm_output
        assert "circular_logic_detected" in sm_output
        assert "escalation_triggered" in sm_output
        assert "exchange_count" in sm_output
        assert "recent_exchanges" in sm_output
        assert "processing_notes" in sm_output
        assert "created_at" in sm_output

    @pytest.mark.asyncio
    async def test_sm_node_handles_escalation_gracefully(self) -> None:
        """Test that sm_node handles escalation edge case (AC #4)."""
        state = create_test_state(escalate_to_human=True)
        result = await sm_node(state)

        assert result["routing_decision"] == "escalate"
        sm_output = result["sm_output"]
        assert sm_output["escalation_triggered"] is True
        assert sm_output["escalation_reason"] == "human_requested"

    @pytest.mark.asyncio
    async def test_sm_node_handles_gate_blocked(self) -> None:
        """Test that sm_node handles gate_blocked edge case (AC #4)."""
        state = create_test_state(current_agent="tea", gate_blocked=True)
        result = await sm_node(state)

        assert result["routing_decision"] == "dev"  # Recovery agent for TEA
        sm_output = result["sm_output"]
        assert sm_output["gate_blocked"] is True
        assert sm_output["recovery_agent"] == "dev"

    @pytest.mark.asyncio
    async def test_sm_node_creates_decision_record(self) -> None:
        """Test that sm_node creates a decision record."""
        state = create_test_state(current_agent="analyst")
        result = await sm_node(state)

        decisions = result["decisions"]
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.agent == "sm"
        assert "pm" in decision.summary  # Routing to PM

    @pytest.mark.asyncio
    async def test_sm_node_creates_message(self) -> None:
        """Test that sm_node creates an output message."""
        state = create_test_state(current_agent="analyst")
        result = await sm_node(state)

        messages = result["messages"]
        assert len(messages) == 1
        message = messages[0]
        assert message.additional_kwargs.get("agent") == "sm"
        assert "pm" in message.content.lower()


class TestSMNodeEdgeCases:
    """Additional edge case tests for sm_node (AC #4)."""

    @pytest.mark.asyncio
    async def test_handles_empty_state(self) -> None:
        """Test handling of minimal state."""
        state = create_test_state()
        result = await sm_node(state)

        assert result["routing_decision"] in ["pm", "analyst"]

    @pytest.mark.asyncio
    async def test_handles_agent_failure(self) -> None:
        """Test handling of agent failure flag."""
        state = create_test_state(agent_failure=True)
        result = await sm_node(state)

        assert result["routing_decision"] == "escalate"
        assert result["sm_output"]["escalation_reason"] == "agent_failure"

    @pytest.mark.asyncio
    async def test_preserves_input_state(self) -> None:
        """Test that sm_node does not mutate input state."""
        original_messages: list[Any] = []
        state = create_test_state(messages=original_messages)

        await sm_node(state)

        # Original state should be unchanged
        assert state["messages"] == []

    @pytest.mark.asyncio
    async def test_sm_node_can_route_to_self(self) -> None:
        """Test that SM defaults to analyst for its own successor."""
        state = create_test_state(current_agent="sm")
        result = await sm_node(state)

        # SM's natural successor is analyst
        assert result["routing_decision"] == "analyst"

    @pytest.mark.asyncio
    async def test_handles_invalid_current_agent(self) -> None:
        """Test handling of invalid current agent."""
        state = create_test_state(current_agent="invalid_agent_xyz")
        result = await sm_node(state)

        # Should default to analyst (natural fallback)
        assert result["routing_decision"] == "analyst"
        # The rationale includes the invalid agent name and routes to analyst
        assert "invalid_agent_xyz" in result["sm_output"]["routing_rationale"]
        assert "analyst" in result["sm_output"]["routing_rationale"]


class TestSMNodeIntegration:
    """Integration tests for sm_node with workflow."""

    @pytest.mark.asyncio
    async def test_workflow_analyst_to_pm_flow(self) -> None:
        """Test complete flow from analyst to PM."""
        # Simulate analyst completing work
        messages = [
            HumanMessage(content="Build a login feature"),
            create_agent_message("Requirements crystallized: authentication needed", "analyst"),
        ]
        state = create_test_state(
            messages=messages,
            current_agent="analyst",
        )

        result = await sm_node(state)

        assert result["routing_decision"] == "pm"

    @pytest.mark.asyncio
    async def test_workflow_pm_to_architect_with_flag(self) -> None:
        """Test PM to architect when needs_architecture is set."""
        state = create_test_state(
            current_agent="pm",
            needs_architecture=True,
        )

        result = await sm_node(state)

        assert result["routing_decision"] == "architect"

    @pytest.mark.asyncio
    async def test_workflow_pm_skip_architect(self) -> None:
        """Test PM directly to dev when no architecture needed."""
        state = create_test_state(
            current_agent="pm",
            needs_architecture=False,
        )

        result = await sm_node(state)

        assert result["routing_decision"] == "dev"
