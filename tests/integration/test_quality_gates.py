"""Integration tests for quality gate framework.

Tests the full gate lifecycle including decoration, evaluation,
state management, and result accumulation.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.gates import (
    GateContext,
    GateResult,
    clear_evaluators,
    quality_gate,
    register_evaluator,
)


class TestQualityGateLifecycle:
    """Integration tests for full gate lifecycle."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self) -> None:
        """Clear evaluators before and after each test."""
        clear_evaluators()
        yield
        clear_evaluators()

    @pytest.mark.asyncio
    async def test_full_gate_lifecycle_passing(self) -> None:
        """Test complete lifecycle: decorate -> evaluate -> pass -> execute."""

        # Register evaluator
        async def passing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("lifecycle_test", passing_evaluator)

        # Decorate node
        @quality_gate("lifecycle_test", blocking=True)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            state["processed"] = True
            state["agent_output"] = "Analysis complete"
            return state

        # Execute
        initial_state = {"messages": [], "current_step": "analysis"}
        result = await agent_node(initial_state)

        # Verify
        assert result["processed"] is True
        assert result["agent_output"] == "Analysis complete"
        assert "gate_results" in result
        assert len(result["gate_results"]) == 1
        assert result["gate_results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_full_gate_lifecycle_blocking(self) -> None:
        """Test complete lifecycle: decorate -> evaluate -> fail -> block."""

        # Register evaluator
        async def failing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Quality requirements not met",
            )

        register_evaluator("blocking_lifecycle", failing_evaluator)

        # Decorate node
        @quality_gate("blocking_lifecycle", blocking=True)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            state["processed"] = True  # Should not execute
            return state

        # Execute
        initial_state = {"messages": []}
        result = await agent_node(initial_state)

        # Verify blocked
        assert "processed" not in result
        assert result["gate_blocked"] is True
        assert result["gate_failure"] == "Quality requirements not met"
        assert result["gate_results"][0]["passed"] is False

    @pytest.mark.asyncio
    async def test_full_gate_lifecycle_advisory(self) -> None:
        """Test complete lifecycle: decorate -> evaluate -> fail -> warn -> execute."""

        # Register evaluator
        async def failing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Minor style issue",
            )

        register_evaluator("advisory_lifecycle", failing_evaluator)

        # Decorate node
        @quality_gate("advisory_lifecycle", blocking=False)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            state["processed"] = True
            return state

        # Execute
        initial_state = {}
        result = await agent_node(initial_state)

        # Verify executed despite failure
        assert result["processed"] is True
        assert "gate_blocked" not in result
        assert "advisory_warnings" in result
        assert len(result["advisory_warnings"]) == 1


class TestMultipleGates:
    """Integration tests for multiple gates on same node."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self) -> None:
        """Clear evaluators before and after each test."""
        clear_evaluators()
        yield
        clear_evaluators()

    @pytest.mark.asyncio
    async def test_multiple_gates_all_pass(self) -> None:
        """Multiple gates all passing allows execution."""

        # Register multiple evaluators
        async def gate1_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        async def gate2_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("gate1", gate1_eval)
        register_evaluator("gate2", gate2_eval)

        # Apply multiple decorators
        @quality_gate("gate2", blocking=True)
        @quality_gate("gate1", blocking=True)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            state["executed"] = True
            return state

        result = await agent_node({})

        assert result["executed"] is True
        assert len(result["gate_results"]) == 2

    @pytest.mark.asyncio
    async def test_outer_gate_fails_blocks_inner_gate(self) -> None:
        """Outer blocking gate failure prevents inner gate and node execution.

        Note: Decorators are applied bottom-up, so @outer wraps @inner wraps func.
        When called, outer runs first, then inner, then func.
        """
        execution_order: list[str] = []

        async def outer_eval(ctx: GateContext) -> GateResult:
            execution_order.append("outer")
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Outer gate failed",
            )

        async def inner_eval(ctx: GateContext) -> GateResult:
            execution_order.append("inner")  # Should not execute
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("outer_gate", outer_eval)
        register_evaluator("inner_gate", inner_eval)

        @quality_gate("outer_gate", blocking=True)  # Runs first (outer)
        @quality_gate("inner_gate", blocking=True)  # Runs second (inner)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            execution_order.append("node")
            return state

        result = await agent_node({})

        # Outer gate should have executed and blocked
        assert "outer" in execution_order
        assert "inner" not in execution_order
        assert "node" not in execution_order
        assert result["gate_blocked"] is True


class TestGateResultAccumulation:
    """Integration tests for gate result accumulation in state."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self) -> None:
        """Clear evaluators before and after each test."""
        clear_evaluators()
        yield
        clear_evaluators()

    @pytest.mark.asyncio
    async def test_gate_results_accumulate(self) -> None:
        """Gate results from multiple gates accumulate in state."""

        async def gate1_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="first_gate")

        async def gate2_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="second_gate")

        register_evaluator("first_gate", gate1_eval)
        register_evaluator("second_gate", gate2_eval)

        @quality_gate("second_gate", blocking=True)
        @quality_gate("first_gate", blocking=True)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await agent_node({})

        assert len(result["gate_results"]) == 2
        gate_names = [r["gate_name"] for r in result["gate_results"]]
        assert "first_gate" in gate_names
        assert "second_gate" in gate_names

    @pytest.mark.asyncio
    async def test_existing_gate_results_preserved(self) -> None:
        """Existing gate_results in state are preserved."""

        async def gate_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("preserve_test", gate_eval)

        @quality_gate("preserve_test", blocking=True)
        async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        # Start with existing gate results
        initial_state = {
            "gate_results": [{"gate_name": "previous_gate", "passed": True, "reason": None}]
        }
        result = await agent_node(initial_state)

        assert len(result["gate_results"]) == 2
        assert result["gate_results"][0]["gate_name"] == "previous_gate"
        assert result["gate_results"][1]["gate_name"] == "preserve_test"


class TestLangGraphStyleState:
    """Integration tests with LangGraph-style state dictionaries."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self) -> None:
        """Clear evaluators before and after each test."""
        clear_evaluators()
        yield
        clear_evaluators()

    @pytest.mark.asyncio
    async def test_langgraph_style_state(self) -> None:
        """Gate works with LangGraph-style nested state."""

        async def context_aware_eval(ctx: GateContext) -> GateResult:
            # Check state for required fields
            messages = ctx.state.get("messages", [])
            if len(messages) == 0:
                return GateResult(
                    passed=False,
                    gate_name=ctx.gate_name,
                    reason="No messages in state",
                )
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("context_gate", context_aware_eval)

        @quality_gate("context_gate", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            state["analysis_complete"] = True
            return state

        # Test with valid state
        valid_state = {
            "messages": [{"role": "user", "content": "Analyze this"}],
            "current_agent": "analyst",
        }
        result = await analyst_node(valid_state)

        assert result["analysis_complete"] is True
        assert result["gate_results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_langgraph_style_state_blocked(self) -> None:
        """Gate blocks with invalid LangGraph-style state."""

        async def context_aware_eval(ctx: GateContext) -> GateResult:
            messages = ctx.state.get("messages", [])
            if len(messages) == 0:
                return GateResult(
                    passed=False,
                    gate_name=ctx.gate_name,
                    reason="No messages in state",
                )
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("context_block_gate", context_aware_eval)

        @quality_gate("context_block_gate", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            state["analysis_complete"] = True
            return state

        # Test with invalid state (empty messages)
        invalid_state = {"messages": [], "current_agent": "analyst"}
        result = await analyst_node(invalid_state)

        assert "analysis_complete" not in result
        assert result["gate_blocked"] is True
        assert result["gate_failure"] == "No messages in state"
