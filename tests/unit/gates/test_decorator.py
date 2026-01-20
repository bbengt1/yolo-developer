"""Unit tests for quality_gate decorator.

Tests decorator functionality including metadata preservation,
async handling, and state management.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


class TestQualityGateDecoratorBasics:
    """Tests for basic decorator functionality."""

    def test_decorator_preserves_function_name(self) -> None:
        """Decorated function preserves original __name__."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("test_gate")
        async def my_function(state: dict[str, Any]) -> dict[str, Any]:
            """My docstring."""
            return state

        assert my_function.__name__ == "my_function"

    def test_decorator_preserves_docstring(self) -> None:
        """Decorated function preserves original __doc__."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("test_gate")
        async def my_function(state: dict[str, Any]) -> dict[str, Any]:
            """My docstring."""
            return state

        assert my_function.__doc__ == "My docstring."

    def test_decorator_with_gate_name(self) -> None:
        """Decorator captures gate name."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("testability")
        async def node_func(state: dict[str, Any]) -> dict[str, Any]:
            return state

        # Gate name should be accessible via attribute
        assert hasattr(node_func, "_gate_name")
        assert node_func._gate_name == "testability"

    def test_decorator_default_blocking_true(self) -> None:
        """Decorator defaults to blocking=True."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("test")
        async def node_func(state: dict[str, Any]) -> dict[str, Any]:
            return state

        assert hasattr(node_func, "_gate_blocking")
        assert node_func._gate_blocking is True

    def test_decorator_explicit_blocking_false(self) -> None:
        """Decorator accepts blocking=False."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("test", blocking=False)
        async def node_func(state: dict[str, Any]) -> dict[str, Any]:
            return state

        assert node_func._gate_blocking is False


class TestQualityGateAsyncHandling:
    """Tests for async function handling."""

    @pytest.mark.asyncio
    async def test_async_function_remains_async(self) -> None:
        """Decorated async function remains awaitable."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        # Register a passing evaluator
        async def passing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("async_test", passing_evaluator)

        @quality_gate("async_test")
        async def async_node(state: dict[str, Any]) -> dict[str, Any]:
            state["executed"] = True
            return state

        result = await async_node({"initial": "state"})
        assert result["executed"] is True

    @pytest.mark.asyncio
    async def test_gate_evaluation_is_awaited(self) -> None:
        """Gate evaluation is properly awaited."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        eval_called = False

        async def tracking_evaluator(ctx: GateContext) -> GateResult:
            nonlocal eval_called
            await asyncio.sleep(0.001)  # Simulate async work
            eval_called = True
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("tracking_test", tracking_evaluator)

        @quality_gate("tracking_test")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        await node({})
        assert eval_called is True


class TestQualityGateStateManagement:
    """Tests for state management through the decorator."""

    @pytest.mark.asyncio
    async def test_state_passed_to_node_on_gate_pass(self) -> None:
        """State is passed unchanged to node when gate passes."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        async def pass_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("pass_state_test", pass_eval)

        received_state: dict[str, Any] = {}

        @quality_gate("pass_state_test")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal received_state
            received_state = state.copy()
            state["node_ran"] = True
            return state

        input_state = {"original_key": "original_value"}
        result = await node(input_state)

        assert received_state["original_key"] == "original_value"
        assert result["node_ran"] is True

    @pytest.mark.asyncio
    async def test_gate_results_appended_to_state(self) -> None:
        """Gate results are recorded in state.gate_results."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        async def pass_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("results_test", pass_eval)

        @quality_gate("results_test")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await node({})
        assert "gate_results" in result
        assert len(result["gate_results"]) == 1
        assert result["gate_results"][0]["gate_name"] == "results_test"
        assert result["gate_results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_input_state_not_mutated(self) -> None:
        """Original input state is not mutated."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        async def pass_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("no_mutate_test", pass_eval)

        @quality_gate("no_mutate_test")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            state["new_key"] = "new_value"
            return state

        original = {"original": "value"}
        original_copy = original.copy()
        await node(original)

        assert original == original_copy


class TestQualityGateWithoutEvaluator:
    """Tests for decorator when no evaluator is registered."""

    @pytest.mark.asyncio
    async def test_unregistered_gate_passes_by_default(self) -> None:
        """Unregistered gate name passes by default (fail-open)."""
        from yolo_developer.gates.decorator import quality_gate

        @quality_gate("nonexistent_gate")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            state["executed"] = True
            return state

        result = await node({})
        assert result["executed"] is True

    @pytest.mark.asyncio
    async def test_unregistered_gate_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unregistered gate logs a warning."""
        import structlog

        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators

        # Ensure no evaluator is registered
        clear_evaluators()

        # Configure structlog to use standard logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        @quality_gate("missing_evaluator_gate")
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        with caplog.at_level("WARNING"):
            await node({})

        # Verify warning was logged about missing evaluator
        log_output = caplog.text
        assert "No evaluator registered for gate" in log_output
        assert "missing_evaluator_gate" in log_output


class TestBlockingGateBehavior:
    """Tests for blocking gate mode (AC2)."""

    @pytest.mark.asyncio
    async def test_blocking_gate_failure_prevents_execution(self) -> None:
        """Blocking gate failure prevents node execution."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Test failure",
            )

        register_evaluator("blocking_test", failing_eval)

        node_executed = False

        @quality_gate("blocking_test", blocking=True)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal node_executed
            node_executed = True
            return state

        await node({})
        assert node_executed is False

    @pytest.mark.asyncio
    async def test_blocking_gate_sets_gate_blocked_true(self) -> None:
        """Blocking gate failure sets gate_blocked=True in state."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Test failure",
            )

        register_evaluator("blocked_test", failing_eval)

        @quality_gate("blocked_test", blocking=True)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await node({})
        assert result["gate_blocked"] is True

    @pytest.mark.asyncio
    async def test_blocking_gate_sets_gate_failure_reason(self) -> None:
        """Blocking gate failure sets gate_failure with reason in state."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Missing required tests",
            )

        register_evaluator("failure_reason_test", failing_eval)

        @quality_gate("failure_reason_test", blocking=True)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await node({})
        assert result["gate_failure"] == "Missing required tests"

    @pytest.mark.asyncio
    async def test_blocking_gate_pass_allows_execution(self) -> None:
        """Blocking gate pass allows node execution."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def passing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("blocking_pass_test", passing_eval)

        @quality_gate("blocking_pass_test", blocking=True)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            state["node_executed"] = True
            return state

        result = await node({})
        assert result["node_executed"] is True
        assert "gate_blocked" not in result


class TestAdvisoryGateBehavior:
    """Tests for advisory gate mode (AC3)."""

    @pytest.mark.asyncio
    async def test_advisory_gate_failure_allows_execution(self) -> None:
        """Advisory gate failure allows node execution to continue."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Style issue",
            )

        register_evaluator("advisory_test", failing_eval)

        @quality_gate("advisory_test", blocking=False)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            state["node_executed"] = True
            return state

        result = await node({})
        assert result["node_executed"] is True

    @pytest.mark.asyncio
    async def test_advisory_gate_does_not_set_gate_blocked(self) -> None:
        """Advisory gate failure does not set gate_blocked."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Minor issue",
            )

        register_evaluator("advisory_no_block_test", failing_eval)

        @quality_gate("advisory_no_block_test", blocking=False)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await node({})
        assert "gate_blocked" not in result

    @pytest.mark.asyncio
    async def test_advisory_gate_records_warning(self) -> None:
        """Advisory gate failure records warning in advisory_warnings list."""
        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Style violation",
            )

        register_evaluator("advisory_warning_test", failing_eval)

        @quality_gate("advisory_warning_test", blocking=False)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        result = await node({})
        assert "advisory_warnings" in result
        assert len(result["advisory_warnings"]) == 1
        assert result["advisory_warnings"][0]["gate_name"] == "advisory_warning_test"
        assert result["advisory_warnings"][0]["reason"] == "Style violation"

    @pytest.mark.asyncio
    async def test_advisory_gate_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Advisory gate failure logs a warning."""
        import structlog

        from yolo_developer.gates.decorator import quality_gate
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        # Configure structlog to use standard logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Advisory failure reason",
            )

        register_evaluator("advisory_log_test", failing_eval)

        @quality_gate("advisory_log_test", blocking=False)
        async def node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        with caplog.at_level("WARNING"):
            await node({})

        log_output = caplog.text
        assert "Advisory gate failure" in log_output
        assert "advisory_log_test" in log_output


class TestMetricsRecordingIntegration:
    """Tests for metrics recording in decorator (Story 3.9 - Task 4)."""

    def test_set_metrics_store_is_available(self) -> None:
        """set_metrics_store function should be importable."""
        from yolo_developer.gates.decorator import set_metrics_store

        assert callable(set_metrics_store)

    def test_get_metrics_store_is_available(self) -> None:
        """get_metrics_store function should be importable."""
        from yolo_developer.gates.decorator import get_metrics_store

        assert callable(get_metrics_store)

    def test_set_and_get_metrics_store(self) -> None:
        """Should be able to set and get metrics store."""
        import tempfile
        from pathlib import Path

        from yolo_developer.gates.decorator import (
            get_metrics_store,
            set_metrics_store,
        )
        from yolo_developer.gates.metrics_store import JsonGateMetricsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            retrieved = get_metrics_store()
            assert retrieved is store

            # Clean up
            set_metrics_store(None)

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_gate_pass(self) -> None:
        """Metrics should be recorded when gate passes and store is configured."""
        import asyncio
        import tempfile
        from pathlib import Path

        from yolo_developer.gates.decorator import (
            quality_gate,
            set_metrics_store,
        )
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.metrics_store import JsonGateMetricsStore
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def passing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("metrics_pass_test", passing_eval)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("metrics_pass_test")
                async def node(state: dict) -> dict:
                    return state

                await node({})

                # Allow async recording to complete
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) >= 1
                assert metrics[0].gate_name == "metrics_pass_test"
                assert metrics[0].passed is True
            finally:
                set_metrics_store(None)

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_gate_fail(self) -> None:
        """Metrics should be recorded when gate fails."""
        import asyncio
        import tempfile
        from pathlib import Path

        from yolo_developer.gates.decorator import (
            quality_gate,
            set_metrics_store,
        )
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.metrics_store import JsonGateMetricsStore
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def failing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=False, gate_name=ctx.gate_name, reason="Test fail")

        register_evaluator("metrics_fail_test", failing_eval)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("metrics_fail_test", blocking=False)
                async def node(state: dict) -> dict:
                    return state

                await node({})
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) >= 1
                assert metrics[0].passed is False
            finally:
                set_metrics_store(None)

    @pytest.mark.asyncio
    async def test_metrics_extracts_agent_name_from_state(self) -> None:
        """Metrics should extract agent_name from state.current_agent."""
        import asyncio
        import tempfile
        from pathlib import Path

        from yolo_developer.gates.decorator import (
            quality_gate,
            set_metrics_store,
        )
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.metrics_store import JsonGateMetricsStore
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def passing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("metrics_agent_test", passing_eval)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("metrics_agent_test")
                async def node(state: dict) -> dict:
                    return state

                await node({"current_agent": "analyst"})
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) >= 1
                assert metrics[0].agent_name == "analyst"
            finally:
                set_metrics_store(None)

    @pytest.mark.asyncio
    async def test_metrics_extracts_sprint_id_from_state(self) -> None:
        """Metrics should extract sprint_id from state."""
        import asyncio
        import tempfile
        from pathlib import Path

        from yolo_developer.gates.decorator import (
            quality_gate,
            set_metrics_store,
        )
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.metrics_store import JsonGateMetricsStore
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()

        async def passing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("metrics_sprint_test", passing_eval)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("metrics_sprint_test")
                async def node(state: dict) -> dict:
                    return state

                await node({"sprint_id": "sprint-3"})
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) >= 1
                assert metrics[0].sprint_id == "sprint-3"
            finally:
                set_metrics_store(None)

    @pytest.mark.asyncio
    async def test_no_metrics_when_store_not_configured(self) -> None:
        """No errors when metrics store is not configured."""
        from yolo_developer.gates.decorator import (
            quality_gate,
            set_metrics_store,
        )
        from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
        from yolo_developer.gates.types import GateContext, GateResult

        clear_evaluators()
        set_metrics_store(None)

        async def passing_eval(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("no_store_test", passing_eval)

        @quality_gate("no_store_test")
        async def node(state: dict) -> dict:
            state["executed"] = True
            return state

        # Should not raise and should execute normally
        result = await node({})
        assert result["executed"] is True
