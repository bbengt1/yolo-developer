"""Integration tests for orchestrator handoff scenarios.

Tests cover:
- Node wrapper creates HandoffContext on exit
- State integrity validated during handoffs
- Message accumulation works across agents
- Multi-agent chain with context preservation
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yolo_developer.orchestrator import (
    Decision,
    HandoffContext,
    YoloState,
    create_agent_message,
)


class TestNodeWrapper:
    """Tests for node wrapper functionality."""

    @pytest.mark.asyncio
    async def test_wrap_node_creates_handoff_context(self) -> None:
        """wrap_node should create HandoffContext when agent completes."""
        from yolo_developer.orchestrator.graph import wrap_node

        # Define a simple agent node function
        async def analyst_node(state: YoloState) -> dict[str, Any]:
            msg = create_agent_message("Analysis complete", agent="analyst")
            return {"messages": [msg]}

        # Wrap the node
        wrapped = wrap_node(analyst_node, agent_name="analyst", target_agent="pm")

        # Create initial state
        state: YoloState = {
            "messages": [HumanMessage(content="Start")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        # Execute wrapped node
        result = await wrapped(state)

        # Should have handoff_context in result
        assert "handoff_context" in result
        assert isinstance(result["handoff_context"], HandoffContext)
        assert result["handoff_context"].source_agent == "analyst"
        assert result["handoff_context"].target_agent == "pm"

    @pytest.mark.asyncio
    async def test_wrap_node_preserves_original_output(self) -> None:
        """wrap_node should preserve the original node's output."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def pm_node(state: YoloState) -> dict[str, Any]:
            msg = create_agent_message("Story created", agent="pm")
            decision = Decision(
                agent="pm",
                summary="Created user story",
                rationale="Based on requirements",
            )
            return {
                "messages": [msg],
                "decisions": [decision],
            }

        wrapped = wrap_node(pm_node, agent_name="pm", target_agent="architect")

        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Original output preserved
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "decisions" in result
        assert len(result["decisions"]) == 1

    @pytest.mark.asyncio
    async def test_wrap_node_includes_decisions_in_context(self) -> None:
        """wrap_node should include decisions made during processing in context."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def architect_node(state: YoloState) -> dict[str, Any]:
            decisions = [
                Decision(agent="architect", summary="Chose microservices", rationale="Scalability"),
                Decision(
                    agent="architect", summary="Selected PostgreSQL", rationale="ACID compliance"
                ),
            ]
            msg = create_agent_message("Architecture complete", agent="architect")
            return {"messages": [msg], "decisions": decisions}

        wrapped = wrap_node(architect_node, agent_name="architect", target_agent="dev")

        state: YoloState = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Decisions should be in handoff context
        context = result["handoff_context"]
        assert len(context.decisions) == 2

    @pytest.mark.asyncio
    async def test_wrap_node_updates_current_agent(self) -> None:
        """wrap_node should update current_agent to target agent."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def dev_node(state: YoloState) -> dict[str, Any]:
            return {"messages": [create_agent_message("Done", agent="dev")]}

        wrapped = wrap_node(dev_node, agent_name="dev", target_agent="tester")

        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        assert result["current_agent"] == "tester"

    @pytest.mark.asyncio
    async def test_wrap_node_includes_memory_refs_in_context(self) -> None:
        """wrap_node should include memory_refs from node output in context."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def analyst_node(state: YoloState) -> dict[str, Any]:
            # Node returns memory refs for stored embeddings
            return {
                "messages": [create_agent_message("Analysis done", agent="analyst")],
                "memory_refs": ["req-001", "req-002", "decision-analyst-2024"],
            }

        wrapped = wrap_node(analyst_node, agent_name="analyst", target_agent="pm")

        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Memory refs should be in handoff context
        context = result["handoff_context"]
        assert len(context.memory_refs) == 3
        assert "req-001" in context.memory_refs
        assert "req-002" in context.memory_refs
        assert "decision-analyst-2024" in context.memory_refs

    @pytest.mark.asyncio
    async def test_wrap_node_handles_empty_memory_refs(self) -> None:
        """wrap_node should handle nodes that don't return memory_refs."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def simple_node(state: YoloState) -> dict[str, Any]:
            # Node doesn't return memory_refs
            return {"messages": [create_agent_message("Done", agent="simple")]}

        wrapped = wrap_node(simple_node, agent_name="simple", target_agent="next")

        state: YoloState = {
            "messages": [],
            "current_agent": "simple",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Should have empty memory_refs tuple
        context = result["handoff_context"]
        assert context.memory_refs == ()

    @pytest.mark.asyncio
    async def test_wrap_node_filters_invalid_decisions(self, caplog: Any) -> None:
        """wrap_node should filter out invalid decision types and log warning."""
        import logging

        from yolo_developer.orchestrator.graph import wrap_node

        async def bad_node(state: YoloState) -> dict[str, Any]:
            # Node returns mix of valid and invalid decisions
            valid_decision = Decision(agent="test", summary="Valid", rationale="OK")
            return {
                "messages": [],
                "decisions": [
                    valid_decision,
                    {"agent": "fake", "summary": "Invalid dict"},  # Invalid
                    "not a decision",  # Invalid
                ],
            }

        wrapped = wrap_node(bad_node, agent_name="test", target_agent="next")

        state: YoloState = {
            "messages": [],
            "current_agent": "test",
            "handoff_context": None,
            "decisions": [],
        }

        with caplog.at_level(logging.WARNING):
            result = await wrapped(state)

        # Only valid decision should be in context
        context = result["handoff_context"]
        assert len(context.decisions) == 1
        assert context.decisions[0].summary == "Valid"

        # Should have logged warnings for invalid types
        assert "invalid decision type" in caplog.text.lower()


class TestStateIntegrityOnHandoff:
    """Tests for state integrity validation during handoffs."""

    @pytest.mark.asyncio
    async def test_validated_handoff_passes_for_preserved_state(self) -> None:
        """validated_handoff should pass when state is preserved."""
        from yolo_developer.orchestrator.graph import validated_handoff

        before_state: dict[str, Any] = {
            "data": "important",
            "current_agent": "analyst",
            "messages": [],
        }
        after_state: dict[str, Any] = {
            "data": "important",
            "current_agent": "pm",
            "messages": [AIMessage(content="test")],
        }

        # Should not raise
        validated_handoff(before_state, after_state)

    @pytest.mark.asyncio
    async def test_validated_handoff_logs_on_integrity_violation(self, caplog: Any) -> None:
        """validated_handoff should log warning on integrity violation."""
        import logging

        from yolo_developer.orchestrator.graph import validated_handoff

        before_state: dict[str, Any] = {
            "data": "original",
        }
        after_state: dict[str, Any] = {
            "data": "modified",  # Changed!
        }

        with caplog.at_level(logging.WARNING):
            validated_handoff(before_state, after_state)

        assert "integrity" in caplog.text.lower()


class TestMessageAccumulationAcrossAgents:
    """Tests for message accumulation across multiple agents."""

    def test_messages_accumulate_through_reducer(self) -> None:
        """Messages from multiple agents should accumulate via reducer."""
        from langchain_core.messages import BaseMessage

        from yolo_developer.orchestrator import get_messages_reducer

        reducer = get_messages_reducer()

        # Simulate messages from multiple agents
        messages: list[BaseMessage] = []
        messages = reducer(messages, [HumanMessage(content="User request")])
        messages = reducer(
            messages, [AIMessage(content="Analysis", additional_kwargs={"agent": "analyst"})]
        )
        messages = reducer(
            messages, [AIMessage(content="Story", additional_kwargs={"agent": "pm"})]
        )
        messages = reducer(
            messages, [AIMessage(content="Design", additional_kwargs={"agent": "architect"})]
        )

        assert len(messages) == 4
        assert messages[0].content == "User request"
        assert messages[1].content == "Analysis"
        assert messages[2].content == "Story"
        assert messages[3].content == "Design"


class TestMultiAgentHandoff:
    """Integration tests for multi-agent handoff chains."""

    @pytest.mark.asyncio
    async def test_two_agent_handoff_preserves_context(self) -> None:
        """Context should be preserved in handoff between two agents."""
        from yolo_developer.orchestrator.graph import wrap_node

        # First agent
        async def agent_a(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("A complete", agent="agent_a")],
                "decisions": [
                    Decision(agent="agent_a", summary="Decision A", rationale="Reason A")
                ],
            }

        # Second agent
        async def agent_b(state: YoloState) -> dict[str, Any]:
            # Should have access to handoff context from A
            context = state.get("handoff_context")
            if context and context.source_agent == "agent_a":
                return {
                    "messages": [create_agent_message("B received A's context", agent="agent_b")]
                }
            return {"messages": [create_agent_message("B without context", agent="agent_b")]}

        wrapped_a = wrap_node(agent_a, agent_name="agent_a", target_agent="agent_b")
        wrapped_b = wrap_node(agent_b, agent_name="agent_b", target_agent="done")

        # Initial state
        state: YoloState = {
            "messages": [],
            "current_agent": "agent_a",
            "handoff_context": None,
            "decisions": [],
        }

        # Execute agent A
        result_a = await wrapped_a(state)

        # Update state with A's output (simulating graph transition)
        state["handoff_context"] = result_a["handoff_context"]
        state["current_agent"] = result_a["current_agent"]
        state["decisions"] = result_a.get("decisions", [])

        # Execute agent B
        result_b = await wrapped_b(state)

        # B should have received A's context
        assert "received A's context" in result_b["messages"][0].content

    @pytest.mark.asyncio
    async def test_three_agent_chain_accumulates_decisions(self) -> None:
        """Decisions should accumulate through multi-agent chain."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def agent1(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("1 done", agent="agent1")],
                "decisions": [Decision(agent="agent1", summary="D1", rationale="R1")],
            }

        async def agent2(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("2 done", agent="agent2")],
                "decisions": [Decision(agent="agent2", summary="D2", rationale="R2")],
            }

        async def agent3(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("3 done", agent="agent3")],
                "decisions": [Decision(agent="agent3", summary="D3", rationale="R3")],
            }

        w1 = wrap_node(agent1, "agent1", "agent2")
        w2 = wrap_node(agent2, "agent2", "agent3")
        w3 = wrap_node(agent3, "agent3", "done")

        # Run chain
        state: YoloState = {
            "messages": [],
            "current_agent": "agent1",
            "handoff_context": None,
            "decisions": [],
        }

        r1 = await w1(state)
        state["handoff_context"] = r1["handoff_context"]
        state["decisions"].extend(r1.get("decisions", []))

        r2 = await w2(state)
        state["handoff_context"] = r2["handoff_context"]
        state["decisions"].extend(r2.get("decisions", []))

        r3 = await w3(state)
        state["decisions"].extend(r3.get("decisions", []))

        # All decisions should be accumulated
        assert len(state["decisions"]) == 3
        agents = [d.agent for d in state["decisions"]]
        assert "agent1" in agents
        assert "agent2" in agents
        assert "agent3" in agents


class TestContextQueriesByReceivingAgent:
    """Tests for receiving agent's ability to query context."""

    @pytest.mark.asyncio
    async def test_receiving_agent_can_access_source_agent(self) -> None:
        """Receiving agent should be able to identify the source agent."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def source_agent(state: YoloState) -> dict[str, Any]:
            return {"messages": [create_agent_message("Done", agent="source")]}

        wrapped = wrap_node(source_agent, agent_name="source", target_agent="receiver")

        state: YoloState = {
            "messages": [],
            "current_agent": "source",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Receiving agent can query source
        context = result["handoff_context"]
        assert context.source_agent == "source"

    @pytest.mark.asyncio
    async def test_receiving_agent_can_query_decisions_by_summary(self) -> None:
        """Receiving agent should be able to find decisions by summary content."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def architect(state: YoloState) -> dict[str, Any]:
            decisions = [
                Decision(
                    agent="architect",
                    summary="Chose microservices architecture",
                    rationale="For scalability",
                ),
                Decision(
                    agent="architect",
                    summary="Selected PostgreSQL database",
                    rationale="For ACID compliance",
                ),
            ]
            return {"messages": [], "decisions": decisions}

        wrapped = wrap_node(architect, "architect", "dev")

        state: YoloState = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Receiving agent can search decisions
        context = result["handoff_context"]
        db_decisions = [d for d in context.decisions if "database" in d.summary.lower()]
        assert len(db_decisions) == 1
        assert "PostgreSQL" in db_decisions[0].summary

    @pytest.mark.asyncio
    async def test_receiving_agent_can_query_timestamp(self) -> None:
        """Receiving agent should be able to query handoff timestamp."""
        from datetime import datetime, timezone

        from yolo_developer.orchestrator.graph import wrap_node

        async def agent_node(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        wrapped = wrap_node(agent_node, "agent", "next")

        state: YoloState = {
            "messages": [],
            "current_agent": "agent",
            "handoff_context": None,
            "decisions": [],
        }

        before_time = datetime.now(timezone.utc)
        result = await wrapped(state)
        after_time = datetime.now(timezone.utc)

        # Handoff timestamp should be queryable and within bounds
        context = result["handoff_context"]
        assert context.timestamp >= before_time
        assert context.timestamp <= after_time

    @pytest.mark.asyncio
    async def test_receiving_agent_knows_it_is_target(self) -> None:
        """Receiving agent should be identified as target in context."""
        from yolo_developer.orchestrator.graph import wrap_node

        async def pm_agent(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        wrapped = wrap_node(pm_agent, "pm", "developer")

        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
        }

        result = await wrapped(state)

        # Developer (receiver) should be the target
        context = result["handoff_context"]
        assert context.target_agent == "developer"
        assert result["current_agent"] == "developer"
