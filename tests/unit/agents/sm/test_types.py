"""Tests for SM agent types (Story 10.2).

Tests the type definitions used by the SM agent:
- AgentExchange
- SMOutput
- Constants (CIRCULAR_LOGIC_THRESHOLD, VALID_AGENTS, NATURAL_SUCCESSOR)
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.sm.types import (
    CIRCULAR_LOGIC_THRESHOLD,
    NATURAL_SUCCESSOR,
    VALID_AGENTS,
    AgentExchange,
    SMOutput,
)


class TestAgentExchange:
    """Tests for AgentExchange dataclass."""

    def test_create_basic_exchange(self) -> None:
        """Test creating a basic agent exchange."""
        exchange = AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
        )

        assert exchange.source_agent == "analyst"
        assert exchange.target_agent == "pm"
        assert exchange.exchange_type == "handoff"
        assert exchange.topic == ""  # Default empty
        assert exchange.timestamp  # Should have default timestamp

    def test_create_exchange_with_topic(self) -> None:
        """Test creating an exchange with topic."""
        exchange = AgentExchange(
            source_agent="pm",
            target_agent="architect",
            exchange_type="query",
            topic="architecture_clarification",
        )

        assert exchange.source_agent == "pm"
        assert exchange.target_agent == "architect"
        assert exchange.exchange_type == "query"
        assert exchange.topic == "architecture_clarification"

    def test_exchange_is_frozen(self) -> None:
        """Test that AgentExchange is immutable."""
        exchange = AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
        )

        with pytest.raises(AttributeError):
            exchange.source_agent = "dev"  # type: ignore[misc]

    def test_exchange_to_dict(self) -> None:
        """Test converting exchange to dictionary."""
        exchange = AgentExchange(
            source_agent="dev",
            target_agent="tea",
            exchange_type="response",
            topic="implementation_complete",
        )

        result = exchange.to_dict()

        assert result["source_agent"] == "dev"
        assert result["target_agent"] == "tea"
        assert result["exchange_type"] == "response"
        assert result["topic"] == "implementation_complete"
        assert "timestamp" in result


class TestSMOutput:
    """Tests for SMOutput dataclass."""

    def test_create_basic_output(self) -> None:
        """Test creating basic SM output."""
        output = SMOutput(
            routing_decision="pm",
            routing_rationale="Natural workflow progression",
        )

        assert output.routing_decision == "pm"
        assert output.routing_rationale == "Natural workflow progression"
        assert output.circular_logic_detected is False
        assert output.escalation_triggered is False
        assert output.escalation_reason is None
        assert output.exchange_count == 0
        assert output.recent_exchanges == ()
        assert output.gate_blocked is False
        assert output.recovery_agent is None
        assert output.processing_notes == ""

    def test_create_output_with_escalation(self) -> None:
        """Test creating SM output with escalation."""
        output = SMOutput(
            routing_decision="escalate",
            routing_rationale="Human escalation requested",
            escalation_triggered=True,
            escalation_reason="human_requested",
        )

        assert output.routing_decision == "escalate"
        assert output.escalation_triggered is True
        assert output.escalation_reason == "human_requested"

    def test_create_output_with_circular_logic(self) -> None:
        """Test creating SM output with circular logic detection."""
        exchanges = (
            AgentExchange(source_agent="dev", target_agent="tea", exchange_type="handoff"),
            AgentExchange(source_agent="tea", target_agent="dev", exchange_type="response"),
        )

        output = SMOutput(
            routing_decision="escalate",
            routing_rationale="Circular logic detected",
            circular_logic_detected=True,
            escalation_triggered=True,
            escalation_reason="circular_logic",
            exchange_count=4,
            recent_exchanges=exchanges,
        )

        assert output.circular_logic_detected is True
        assert output.exchange_count == 4
        assert len(output.recent_exchanges) == 2

    def test_create_output_with_gate_blocked(self) -> None:
        """Test creating SM output with gate blocked state."""
        output = SMOutput(
            routing_decision="dev",
            routing_rationale="Gate blocked, routing back to dev",
            gate_blocked=True,
            recovery_agent="dev",
        )

        assert output.gate_blocked is True
        assert output.recovery_agent == "dev"

    def test_output_is_frozen(self) -> None:
        """Test that SMOutput is immutable."""
        output = SMOutput(
            routing_decision="pm",
            routing_rationale="Test",
        )

        with pytest.raises(AttributeError):
            output.routing_decision = "dev"  # type: ignore[misc]

    def test_output_to_dict(self) -> None:
        """Test converting output to dictionary."""
        exchange = AgentExchange(
            source_agent="analyst",
            target_agent="pm",
            exchange_type="handoff",
        )

        output = SMOutput(
            routing_decision="architect",
            routing_rationale="Architecture review needed",
            circular_logic_detected=False,
            escalation_triggered=False,
            exchange_count=1,
            recent_exchanges=(exchange,),
            processing_notes="Analyzed state successfully",
        )

        result = output.to_dict()

        assert result["routing_decision"] == "architect"
        assert result["routing_rationale"] == "Architecture review needed"
        assert result["circular_logic_detected"] is False
        assert result["escalation_triggered"] is False
        assert result["exchange_count"] == 1
        assert len(result["recent_exchanges"]) == 1
        assert result["processing_notes"] == "Analyzed state successfully"
        assert "created_at" in result


class TestConstants:
    """Tests for SM module constants."""

    def test_circular_logic_threshold(self) -> None:
        """Test circular logic threshold value."""
        assert CIRCULAR_LOGIC_THRESHOLD == 3

    def test_valid_agents_contains_all_agents(self) -> None:
        """Test that VALID_AGENTS contains all expected agents."""
        expected_agents = {"analyst", "pm", "architect", "dev", "tea", "sm", "escalate"}
        assert VALID_AGENTS == expected_agents

    def test_natural_successor_mapping(self) -> None:
        """Test natural successor mapping for workflow."""
        assert NATURAL_SUCCESSOR["analyst"] == "pm"
        assert NATURAL_SUCCESSOR["pm"] == "architect"
        assert NATURAL_SUCCESSOR["architect"] == "dev"
        assert NATURAL_SUCCESSOR["dev"] == "tea"
        assert NATURAL_SUCCESSOR["tea"] == "dev"  # TEA routes back to dev
        assert NATURAL_SUCCESSOR["sm"] == "analyst"  # SM defaults to analyst

    def test_natural_successor_all_agents_except_escalate(self) -> None:
        """Test that all agents (except escalate) have natural successors."""
        agents_with_successors = set(NATURAL_SUCCESSOR.keys())
        expected_agents = VALID_AGENTS - {"escalate"}

        # All agents except escalate should have a natural successor
        assert "escalate" not in agents_with_successors
        # Verify all expected agents have successors defined
        assert agents_with_successors == expected_agents


class TestRoutingDecisionType:
    """Tests for RoutingDecision literal type values."""

    def test_valid_routing_decisions(self) -> None:
        """Test that valid routing decisions are accepted."""
        # These should all be valid RoutingDecision values
        valid_decisions = [
            "analyst",
            "pm",
            "architect",
            "dev",
            "tea",
            "sm",
            "escalate",
        ]

        for decision in valid_decisions:
            output = SMOutput(
                routing_decision=decision,  # type: ignore[arg-type]
                routing_rationale="Test",
            )
            assert output.routing_decision == decision


class TestEscalationReasonType:
    """Tests for EscalationReason literal type values."""

    def test_valid_escalation_reasons(self) -> None:
        """Test that valid escalation reasons are accepted."""
        valid_reasons = [
            "human_requested",
            "circular_logic",
            "gate_blocked_unresolvable",
            "conflict_unresolved",
            "agent_failure",
            "unknown",
        ]

        for reason in valid_reasons:
            output = SMOutput(
                routing_decision="escalate",
                routing_rationale="Test",
                escalation_triggered=True,
                escalation_reason=reason,  # type: ignore[arg-type]
            )
            assert output.escalation_reason == reason
