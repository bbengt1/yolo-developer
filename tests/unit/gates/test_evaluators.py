"""Unit tests for gate evaluator protocol and registry.

Tests GateEvaluator protocol and evaluator registration functions.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.types import GateContext, GateResult


class TestGateEvaluatorProtocol:
    """Tests for GateEvaluator protocol compliance."""

    @pytest.mark.asyncio
    async def test_evaluator_protocol_callable(self) -> None:
        """GateEvaluator is a callable Protocol."""
        from yolo_developer.gates.evaluators import GateEvaluator

        async def my_evaluator(context: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=context.gate_name)

        # Should satisfy the protocol
        evaluator: GateEvaluator = my_evaluator
        ctx = GateContext(state={}, gate_name="test")
        result = await evaluator(ctx)
        assert result.passed is True


class TestEvaluatorRegistry:
    """Tests for evaluator registration and retrieval."""

    def test_register_evaluator(self) -> None:
        """Can register an evaluator by name."""
        from yolo_developer.gates.evaluators import (
            clear_evaluators,
            get_evaluator,
            register_evaluator,
        )

        clear_evaluators()

        async def my_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("my_gate", my_eval)
        retrieved = get_evaluator("my_gate")
        assert retrieved is my_eval

    def test_get_unregistered_evaluator_returns_none(self) -> None:
        """Getting unregistered evaluator returns None."""
        from yolo_developer.gates.evaluators import clear_evaluators, get_evaluator

        clear_evaluators()
        assert get_evaluator("nonexistent") is None

    def test_register_overwrites_existing(self) -> None:
        """Registering same name overwrites previous evaluator."""
        from yolo_developer.gates.evaluators import (
            clear_evaluators,
            get_evaluator,
            register_evaluator,
        )

        clear_evaluators()

        async def eval1(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="1")

        async def eval2(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="2")

        register_evaluator("same_name", eval1)
        register_evaluator("same_name", eval2)

        retrieved = get_evaluator("same_name")
        assert retrieved is eval2

    def test_clear_evaluators(self) -> None:
        """clear_evaluators removes all registered evaluators."""
        from yolo_developer.gates.evaluators import (
            clear_evaluators,
            get_evaluator,
            register_evaluator,
        )

        async def eval1(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="test")

        register_evaluator("test_gate", eval1)
        clear_evaluators()
        assert get_evaluator("test_gate") is None

    def test_list_evaluators(self) -> None:
        """list_evaluators returns all registered gate names."""
        from yolo_developer.gates.evaluators import (
            clear_evaluators,
            list_evaluators,
            register_evaluator,
        )

        clear_evaluators()

        async def eval1(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="test")

        register_evaluator("gate_a", eval1)
        register_evaluator("gate_b", eval1)

        names = list_evaluators()
        assert "gate_a" in names
        assert "gate_b" in names


class TestEvaluatorExecution:
    """Tests for evaluator execution."""

    @pytest.mark.asyncio
    async def test_evaluator_receives_context(self) -> None:
        """Evaluator receives GateContext with state."""
        from yolo_developer.gates.evaluators import (
            clear_evaluators,
            get_evaluator,
            register_evaluator,
        )

        clear_evaluators()

        received_context: GateContext | None = None

        async def capture_eval(ctx: GateContext) -> GateResult:
            nonlocal received_context
            received_context = ctx
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("capture_gate", capture_eval)

        ctx = GateContext(
            state={"key": "value"},
            gate_name="capture_gate",
            artifact_id="art-001",
        )

        evaluator = get_evaluator("capture_gate")
        assert evaluator is not None
        await evaluator(ctx)

        assert received_context is not None
        assert received_context.state["key"] == "value"
        assert received_context.artifact_id == "art-001"
