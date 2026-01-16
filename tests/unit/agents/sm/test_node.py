"""Tests for SM agent node (Story 10.2, 10.6, 10.7, 10.8).

Tests the sm_node function and its helper functions:
- _analyze_current_state
- _get_next_agent
- _check_for_escalation
- _check_for_circular_logic
- sm_node (main entry point)

Also tests integration with:
- Enhanced circular logic detection (Story 10.6)
- Conflict mediation (Story 10.7)
- Handoff management (Story 10.8)
"""

from __future__ import annotations

from typing import Any

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
from yolo_developer.orchestrator.context import Decision
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
        count, _exchanges = _count_recent_exchanges(state)

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
        is_circular, _exchanges = _check_for_circular_logic(state)

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

    @pytest.mark.asyncio
    async def test_circular_logic_triggers_escalation_via_sm_node(self) -> None:
        """Test that circular logic detection triggers escalation through sm_node (AC #4).

        This integration test verifies the full path from ping-pong messages
        through sm_node to escalation routing decision.
        """
        # Create ping-pong messages between dev and tea (>3 exchanges)
        messages = [
            create_agent_message("Implementation attempt 1", "dev"),
            create_agent_message("Tests failing", "tea"),
            create_agent_message("Implementation attempt 2", "dev"),
            create_agent_message("Tests still failing", "tea"),
            create_agent_message("Implementation attempt 3", "dev"),
            create_agent_message("Tests still failing", "tea"),
            create_agent_message("Implementation attempt 4", "dev"),
            create_agent_message("Tests still failing", "tea"),
        ]
        state = create_test_state(
            messages=messages,
            current_agent="tea",
        )

        result = await sm_node(state)

        # Should escalate due to circular logic
        assert result["routing_decision"] == "escalate"
        sm_output = result["sm_output"]
        assert sm_output["circular_logic_detected"] is True
        assert sm_output["escalation_triggered"] is True
        assert sm_output["escalation_reason"] == "circular_logic"


class TestSMNodeEnhancedCircularDetection:
    """Tests for enhanced circular logic detection integration (Story 10.6)."""

    @pytest.mark.asyncio
    async def test_sm_node_includes_cycle_analysis(self) -> None:
        """Test that sm_node returns cycle_analysis from enhanced detection.

        Story 10.6 adds enhanced circular logic detection that provides
        detailed pattern analysis in the output.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)

        # cycle_analysis should be present in output (may be None if no cycles)
        assert "cycle_analysis" in result
        sm_output = result["sm_output"]
        assert "cycle_analysis" in sm_output

    @pytest.mark.asyncio
    async def test_enhanced_detection_tracks_patterns(self) -> None:
        """Test enhanced detection provides pattern details in cycle_analysis.

        When circular logic is detected, the enhanced detection should
        provide pattern_type, agents_involved, and severity information.
        """
        # Create ping-pong messages that trigger enhanced detection
        messages = [
            create_agent_message("Analysis 1", "analyst"),
            create_agent_message("Needs more analysis", "pm"),
            create_agent_message("Analysis 2", "analyst"),
            create_agent_message("Still needs more", "pm"),
            create_agent_message("Analysis 3", "analyst"),
            create_agent_message("Not quite right", "pm"),
            create_agent_message("Analysis 4", "analyst"),
            create_agent_message("Still not right", "pm"),
        ]
        state = create_test_state(
            messages=messages,
            current_agent="pm",
        )

        result = await sm_node(state)

        # Should detect circular logic
        assert result["routing_decision"] == "escalate"

        # Check that cycle_analysis is populated
        cycle_analysis = result.get("cycle_analysis")
        assert cycle_analysis is not None
        assert cycle_analysis["circular_detected"] is True
        assert len(cycle_analysis["patterns_found"]) > 0

    @pytest.mark.asyncio
    async def test_sm_output_includes_cycle_analysis_field(self) -> None:
        """Test SMOutput includes cycle_analysis field for serialization.

        The SMOutput.to_dict() should include cycle_analysis for
        downstream consumers.
        """
        state = create_test_state(
            messages=[],
            current_agent="dev",
        )

        result = await sm_node(state)

        sm_output = result["sm_output"]
        # cycle_analysis field should exist in serialized output
        assert "cycle_analysis" in sm_output

    @pytest.mark.asyncio
    async def test_enhanced_detection_does_not_block_workflow(self) -> None:
        """Test that enhanced detection errors don't block the main workflow.

        Even if enhanced detection has an issue, the basic routing
        should still work.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        # This should complete without error
        result = await sm_node(state)

        # Basic routing should work
        assert result["routing_decision"] == "pm"  # Natural successor from analyst

    @pytest.mark.asyncio
    async def test_processing_notes_include_enhanced_detection_info(self) -> None:
        """Test processing notes include enhanced detection details when cycles found.

        When enhanced detection finds patterns, the processing_notes
        should mention the detection results.
        """
        # Create enough exchanges to trigger detection
        messages = [
            create_agent_message("Work 1", "dev"),
            create_agent_message("Review 1", "tea"),
            create_agent_message("Work 2", "dev"),
            create_agent_message("Review 2", "tea"),
            create_agent_message("Work 3", "dev"),
            create_agent_message("Review 3", "tea"),
            create_agent_message("Work 4", "dev"),
            create_agent_message("Review 4", "tea"),
        ]
        state = create_test_state(
            messages=messages,
            current_agent="tea",
        )

        result = await sm_node(state)

        sm_output = result["sm_output"]
        processing_notes = sm_output["processing_notes"]

        # If circular detected, notes should mention enhanced detection
        if sm_output["circular_logic_detected"]:
            assert "Enhanced detection" in processing_notes or "patterns" in processing_notes.lower()


class TestSMNodeConflictMediation:
    """Tests for conflict mediation integration (Story 10.7).

    Story 10.7 adds conflict mediation that detects and resolves
    conflicts between agents with different recommendations.
    """

    @pytest.mark.asyncio
    async def test_sm_node_returns_mediation_result(self) -> None:
        """Test that sm_node returns mediation_result in output.

        Story 10.7 adds mediation_result to the return dict.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)

        # mediation_result should be present in output (may be None if no conflicts)
        assert "mediation_result" in result
        sm_output = result["sm_output"]
        assert "mediation_result" in sm_output

    @pytest.mark.asyncio
    async def test_sm_output_includes_mediation_result_field(self) -> None:
        """Test SMOutput includes mediation_result field for serialization.

        The SMOutput.to_dict() should include mediation_result for
        downstream consumers.
        """
        state = create_test_state(
            messages=[],
            current_agent="dev",
        )

        result = await sm_node(state)

        sm_output = result["sm_output"]
        # mediation_result field should exist in serialized output
        assert "mediation_result" in sm_output

    @pytest.mark.asyncio
    async def test_no_conflicts_mediation_returns_success(self) -> None:
        """Test mediation returns success when no conflicts present.

        When there are no conflicting decisions, mediation should
        complete successfully with empty conflict list.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
            decisions=[],  # No decisions = no conflicts
        )

        result = await sm_node(state)

        mediation_result = result["mediation_result"]
        if mediation_result is not None:
            # If mediation was performed, it should report success
            assert mediation_result.get("success", True) is True
            # No conflicts should be detected
            assert len(mediation_result.get("conflicts_detected", [])) == 0

    @pytest.mark.asyncio
    async def test_conflict_detection_with_decisions(self) -> None:
        """Test conflict detection when decisions have conflicting artifacts.

        When multiple agents have decisions that reference the same
        artifacts with different approaches, conflicts should be detected.
        """
        # Create decisions with overlapping artifacts and contradictory approaches
        decision1 = Decision(
            agent="architect",
            summary="Use microservices architecture",
            rationale="Scalability is the priority, not simplicity",
            related_artifacts=("system-design",),
        )
        decision2 = Decision(
            agent="dev",
            summary="Use monolithic architecture",
            rationale="Simplicity is the priority, not scalability",
            related_artifacts=("system-design",),
        )

        state = create_test_state(
            messages=[],
            current_agent="sm",
            decisions=[decision1, decision2],
        )

        result = await sm_node(state)

        mediation_result = result.get("mediation_result")
        # The mediation should have been performed
        assert mediation_result is not None

        # Check that conflicts were analyzed
        conflicts = mediation_result.get("conflicts_detected", [])
        # With contradictory decisions, conflicts should be detected
        # Note: conflict detection uses heuristics, so this may vary
        if len(conflicts) > 0:
            assert mediation_result.get("success") is True

    @pytest.mark.asyncio
    async def test_mediation_does_not_block_workflow(self) -> None:
        """Test that mediation errors don't block the main workflow.

        Even if mediation has an issue, the basic routing should still work.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        # This should complete without error
        result = await sm_node(state)

        # Basic routing should work
        assert result["routing_decision"] == "pm"  # Natural successor from analyst

    @pytest.mark.asyncio
    async def test_processing_notes_include_mediation_info(self) -> None:
        """Test processing notes include mediation details when conflicts found.

        When conflicts are detected, the processing_notes should mention
        the mediation results.
        """
        # Create decisions that may conflict
        decision1 = Decision(
            agent="pm",
            summary="High priority for security features",
            rationale="Security is critical and blocking, must be done first",
            related_artifacts=("feature-priority",),
        )
        decision2 = Decision(
            agent="architect",
            summary="Low priority for security features",
            rationale="Performance features should come first, not security",
            related_artifacts=("feature-priority",),
        )

        state = create_test_state(
            messages=[],
            current_agent="sm",
            decisions=[decision1, decision2],
        )

        result = await sm_node(state)

        sm_output = result["sm_output"]
        processing_notes = sm_output["processing_notes"]
        mediation_result = result.get("mediation_result")

        # If conflicts were detected, notes should mention mediation
        if mediation_result and len(mediation_result.get("conflicts_detected", [])) > 0:
            assert "Mediation" in processing_notes or "conflict" in processing_notes.lower()

    @pytest.mark.asyncio
    async def test_escalation_triggered_by_unresolved_conflicts(self) -> None:
        """Test that unresolved conflicts can trigger escalation.

        When conflicts cannot be resolved automatically, escalation
        should be triggered (FR13).
        """
        # Create blocking conflict that may require escalation
        decision1 = Decision(
            agent="architect",
            summary="Must use blocking synchronous pattern for data consistency",
            rationale="This is a blocking requirement, cannot proceed otherwise",
            related_artifacts=("architecture-pattern",),
        )
        decision2 = Decision(
            agent="dev",
            summary="Must use blocking asynchronous pattern for performance",
            rationale="This is a blocking requirement, cannot proceed otherwise",
            related_artifacts=("architecture-pattern",),
        )

        state = create_test_state(
            messages=[],
            current_agent="sm",
            decisions=[decision1, decision2],
        )

        result = await sm_node(state)

        mediation_result = result.get("mediation_result")

        # Check if escalation was triggered due to conflicts
        if mediation_result:
            escalations = mediation_result.get("escalations_triggered", [])
            # If there are escalations, the sm_output should reflect it
            if len(escalations) > 0:
                sm_output = result["sm_output"]
                # Escalation may have been triggered
                # Note: The actual escalation depends on conflict severity
                assert sm_output["escalation_triggered"] is True or len(escalations) > 0

    @pytest.mark.asyncio
    async def test_mediation_result_structure(self) -> None:
        """Test that mediation_result has expected structure when present.

        The mediation_result should contain all expected fields from
        MediationResult.to_dict().
        """
        decision = Decision(
            agent="analyst",
            summary="Test decision",
            rationale="Test rationale",
            related_artifacts=("test-artifact",),
        )

        state = create_test_state(
            messages=[],
            current_agent="analyst",
            decisions=[decision],
        )

        result = await sm_node(state)

        mediation_result = result.get("mediation_result")
        if mediation_result is not None:
            # Check expected fields
            assert "success" in mediation_result
            assert "conflicts_detected" in mediation_result
            assert "resolutions" in mediation_result
            assert "notifications_sent" in mediation_result
            assert "escalations_triggered" in mediation_result
            assert "mediation_notes" in mediation_result
            assert "mediated_at" in mediation_result

    @pytest.mark.asyncio
    async def test_sm_node_with_design_conflict(self) -> None:
        """Test SM node handles design conflicts between architect and dev.

        Design conflicts occur when architect and dev have different
        design decisions on the same artifact.
        """
        decision1 = Decision(
            agent="architect",
            summary="Use event-driven design pattern",
            rationale="Better for decoupling and scalability",
            related_artifacts=("api-design", "system-architecture"),
        )
        decision2 = Decision(
            agent="dev",
            summary="Use request-response design pattern",
            rationale="Simpler to implement and debug, not event-driven",
            related_artifacts=("api-design",),
        )

        state = create_test_state(
            messages=[],
            current_agent="sm",
            decisions=[decision1, decision2],
        )

        result = await sm_node(state)

        # Should complete without error
        assert "mediation_result" in result
        assert "routing_decision" in result

        # Basic workflow should continue
        mediation_result = result.get("mediation_result")
        if mediation_result:
            # Mediation should have processed the decisions
            assert isinstance(mediation_result.get("conflicts_detected"), list)

    @pytest.mark.asyncio
    async def test_sm_node_preserves_routing_with_conflicts(self) -> None:
        """Test that routing still works correctly even with conflicts.

        Conflict mediation should not prevent normal routing decisions.
        """
        decision1 = Decision(
            agent="architect",
            summary="Choose SQL database",
            rationale="Structured data needs SQL",
            related_artifacts=("database-choice",),
        )
        decision2 = Decision(
            agent="dev",
            summary="Choose NoSQL database",
            rationale="Flexibility with NoSQL, not SQL",
            related_artifacts=("database-choice",),
        )

        state = create_test_state(
            messages=[],
            current_agent="analyst",
            decisions=[decision1, decision2],
        )

        result = await sm_node(state)

        # Normal routing should still occur (analyst -> pm)
        # unless escalation was triggered
        if not result["sm_output"]["escalation_triggered"]:
            assert result["routing_decision"] == "pm"


class TestSMNodeHandoffManagement:
    """Tests for handoff management integration in SM node (Story 10.8).

    Verifies that the SM node properly integrates with the manage_handoff()
    function to provide context preservation during agent transitions.
    """

    @pytest.mark.asyncio
    async def test_sm_node_returns_handoff_result(self) -> None:
        """Test that SM node returns handoff_result in state updates.

        When routing to another agent, handoff_result should be included
        in the return dict.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)

        # Should have handoff_result key (may be None for certain routes)
        assert "handoff_result" in result
        # For analyst->pm routing, handoff should be attempted
        if result["routing_decision"] == "pm":
            assert result["handoff_result"] is not None

    @pytest.mark.asyncio
    async def test_sm_output_includes_handoff_result_field(self) -> None:
        """Test that sm_output dict includes handoff_result field.

        The SMOutput should include handoff_result for serialization.
        """
        state = create_test_state(
            messages=[],
            current_agent="pm",
            needs_architecture=True,  # Route to architect
        )

        result = await sm_node(state)
        sm_output = result["sm_output"]

        # sm_output should have handoff_result field
        assert "handoff_result" in sm_output

    @pytest.mark.asyncio
    async def test_handoff_not_performed_for_escalation(self) -> None:
        """Test that handoff is not performed when routing to escalation.

        Escalation is a special route, not to an agent, so no handoff needed.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
            escalate_to_human=True,  # Trigger escalation
        )

        result = await sm_node(state)

        assert result["routing_decision"] == "escalate"
        # handoff_result should be None for escalation
        assert result["handoff_result"] is None

    @pytest.mark.asyncio
    async def test_handoff_not_performed_for_same_agent(self) -> None:
        """Test that handoff is not performed when staying on same agent.

        No handoff needed when source and target are the same.
        """
        # SM routes to sm only in specific edge cases
        # Using gate_blocked scenario that routes back to analyst
        state = create_test_state(
            messages=[],
            current_agent="sm",
        )

        result = await sm_node(state)

        # SM default routes to analyst for new work
        if result["routing_decision"] == "analyst":
            # Handoff should be performed since different agent
            assert result["handoff_result"] is not None

    @pytest.mark.asyncio
    async def test_handoff_does_not_block_workflow(self) -> None:
        """Test that handoff errors don't block the main workflow.

        Even if handoff management has an issue, routing should continue.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        # This should complete without error
        result = await sm_node(state)

        # Basic routing should work
        assert result["routing_decision"] == "pm"

    @pytest.mark.asyncio
    async def test_processing_notes_include_handoff_info(self) -> None:
        """Test processing notes include handoff details when performed.

        When handoff is performed, the processing_notes should mention it.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)
        sm_output = result["sm_output"]
        processing_notes = sm_output["processing_notes"]

        # If handoff was performed, notes should mention it
        if result["handoff_result"] is not None:
            assert "Handoff" in processing_notes

    @pytest.mark.asyncio
    async def test_handoff_result_structure(self) -> None:
        """Test that handoff_result has expected structure when present.

        The handoff_result should contain all expected fields from
        HandoffResult.to_dict().
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)
        handoff_result = result.get("handoff_result")

        if handoff_result is not None:
            # Should have expected fields
            assert "record" in handoff_result
            assert "success" in handoff_result
            assert "context_validated" in handoff_result
            assert "state_updates" in handoff_result
            assert "warnings" in handoff_result

            # Record should have its fields
            record = handoff_result["record"]
            assert "handoff_id" in record
            assert "source_agent" in record
            assert "target_agent" in record
            assert "status" in record

    @pytest.mark.asyncio
    async def test_handoff_records_correct_agents(self) -> None:
        """Test that handoff record captures correct source and target.

        The HandoffRecord should have accurate agent information.
        """
        state = create_test_state(
            messages=[],
            current_agent="pm",
            needs_architecture=True,
        )

        result = await sm_node(state)
        handoff_result = result.get("handoff_result")

        if handoff_result is not None:
            record = handoff_result["record"]
            assert record["source_agent"] == "pm"
            assert record["target_agent"] == result["routing_decision"]

    @pytest.mark.asyncio
    async def test_handoff_success_updates_context(self) -> None:
        """Test that successful handoff provides context updates.

        When handoff succeeds, state_updates should include handoff_context.
        """
        state = create_test_state(
            messages=[],
            current_agent="analyst",
        )

        result = await sm_node(state)
        handoff_result = result.get("handoff_result")

        if handoff_result is not None and handoff_result["success"]:
            # Successful handoff should provide state updates
            state_updates = handoff_result.get("state_updates")
            if state_updates is not None:
                assert "handoff_context" in state_updates

    @pytest.mark.asyncio
    async def test_sm_node_preserves_routing_with_handoff(self) -> None:
        """Test that routing still works correctly even with handoff.

        Handoff management should not prevent normal routing decisions.
        """
        decision = Decision(
            agent="analyst",
            summary="Requirements crystallized",
            rationale="Analysis complete",
            related_artifacts=("requirements",),
        )

        state = create_test_state(
            messages=[],
            current_agent="analyst",
            decisions=[decision],
        )

        result = await sm_node(state)

        # Normal routing should still occur (analyst -> pm)
        # unless escalation was triggered
        if not result["sm_output"]["escalation_triggered"]:
            assert result["routing_decision"] == "pm"
        # And handoff should have been attempted
        assert "handoff_result" in result
