"""Integration tests for gate metrics tracking (Story 3.9 - Task 11).

These tests verify the end-to-end metrics tracking functionality:
- Metrics recording via decorator
- Persistence across simulated restarts
- Concurrent metric recording
- Metrics query with real gate evaluations
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yolo_developer.gates.decorator import (
    quality_gate,
    set_metrics_store,
)
from yolo_developer.gates.evaluators import clear_evaluators, register_evaluator
from yolo_developer.gates.metrics_calculator import (
    calculate_pass_rates,
    calculate_trends,
    get_agent_breakdown,
)
from yolo_developer.gates.metrics_store import JsonGateMetricsStore
from yolo_developer.gates.metrics_types import GateMetricRecord, TrendDirection
from yolo_developer.gates.types import GateContext, GateResult


class TestEndToEndMetricsRecording:
    """Integration tests for end-to-end metric recording via decorator."""

    @pytest.mark.asyncio
    async def test_decorator_records_metrics_on_gate_pass(self) -> None:
        """Decorated function should record metrics when gate passes."""
        clear_evaluators()

        async def passing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name=ctx.gate_name)

        register_evaluator("integration_test_gate", passing_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("integration_test_gate")
                async def test_node(state: dict) -> dict:
                    return state

                await test_node({})

                # Allow async recording to complete
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) == 1
                assert metrics[0].gate_name == "integration_test_gate"
                assert metrics[0].passed is True
                assert metrics[0].score == 1.0

            finally:
                set_metrics_store(None)
                clear_evaluators()

    @pytest.mark.asyncio
    async def test_decorator_records_metrics_on_gate_fail(self) -> None:
        """Decorated function should record metrics when gate fails."""
        clear_evaluators()

        async def failing_evaluator(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name=ctx.gate_name,
                reason="Test failure reason",
            )

        register_evaluator("integration_fail_gate", failing_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("integration_fail_gate", blocking=False)
                async def test_node(state: dict) -> dict:
                    return state

                await test_node({})

                # Allow async recording to complete
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                assert len(metrics) == 1
                assert metrics[0].passed is False
                assert metrics[0].score == 0.0

            finally:
                set_metrics_store(None)
                clear_evaluators()


class TestMetricsPersistence:
    """Integration tests for metrics persistence."""

    @pytest.mark.asyncio
    async def test_metrics_persist_across_store_instances(self) -> None:
        """Metrics should persist when creating new store instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            timestamp = datetime.now(timezone.utc)

            # First store instance - write records
            store1 = JsonGateMetricsStore(base_path=base_path)
            await store1.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=timestamp,
                )
            )

            # Simulate restart - new store instance
            store2 = JsonGateMetricsStore(base_path=base_path)
            await store2.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=False,
                    score=0.50,
                    threshold=0.80,
                    timestamp=timestamp + timedelta(minutes=1),
                )
            )

            # Third instance - verify all records present
            store3 = JsonGateMetricsStore(base_path=base_path)
            metrics = await store3.get_metrics()

            assert len(metrics) == 2
            assert metrics[0].passed is True
            assert metrics[1].passed is False

    @pytest.mark.asyncio
    async def test_metrics_summary_accurate_after_persistence(self) -> None:
        """Summary calculations should be accurate on reloaded data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            timestamp = datetime.now(timezone.utc)

            store = JsonGateMetricsStore(base_path=base_path)

            # Add 10 records: 7 pass, 3 fail
            for i in range(10):
                await store.record_evaluation(
                    GateMetricRecord(
                        gate_name="testability",
                        passed=i < 7,
                        score=0.85 if i < 7 else 0.50,
                        threshold=0.80,
                        timestamp=timestamp,
                    )
                )

            # New store instance
            store2 = JsonGateMetricsStore(base_path=base_path)
            summary = await store2.get_summary()

            assert summary.total_evaluations == 10
            assert summary.pass_count == 7
            assert summary.fail_count == 3
            assert summary.pass_rate == 70.0


class TestConcurrentMetricsRecording:
    """Integration tests for concurrent metrics recording."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_maintain_data_integrity(self) -> None:
        """Concurrent writes should not corrupt data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            timestamp = datetime.now(timezone.utc)

            async def write_records(start_index: int) -> None:
                for i in range(10):
                    await store.record_evaluation(
                        GateMetricRecord(
                            gate_name=f"gate_{start_index}_{i}",
                            passed=True,
                            score=0.85,
                            threshold=0.80,
                            timestamp=timestamp,
                        )
                    )

            # Write 50 records concurrently (5 tasks x 10 records each)
            await asyncio.gather(
                write_records(0),
                write_records(10),
                write_records(20),
                write_records(30),
                write_records(40),
            )

            metrics = await store.get_metrics()
            assert len(metrics) == 50


class TestMetricsQueryWithRealEvaluations:
    """Integration tests for metrics query with real gate evaluations."""

    @pytest.mark.asyncio
    async def test_pass_rates_from_real_evaluations(self) -> None:
        """Pass rates should be accurate from real evaluations."""
        clear_evaluators()

        call_count = [0]

        async def alternating_evaluator(ctx: GateContext) -> GateResult:
            call_count[0] += 1
            # Alternate pass/fail
            passed = call_count[0] % 2 == 1
            return GateResult(passed=passed, gate_name=ctx.gate_name)

        register_evaluator("alternating_gate", alternating_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("alternating_gate", blocking=False)
                async def test_node(state: dict) -> dict:
                    return state

                # Run 4 times: should be pass, fail, pass, fail
                for _ in range(4):
                    await test_node({})
                    await asyncio.sleep(0.05)

                # Allow final recording
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                rates = calculate_pass_rates(metrics)

                assert len(metrics) == 4
                assert rates["alternating_gate"] == 50.0  # 2 pass, 2 fail

            finally:
                set_metrics_store(None)
                clear_evaluators()

    @pytest.mark.asyncio
    async def test_trends_from_time_series_data(self) -> None:
        """Trend calculations should work with time-series data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))

            # Day 1: 2 pass (100%)
            # Day 2: 1 pass, 1 fail (50%)
            base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=base_time,
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=base_time + timedelta(hours=1),
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=True,
                    score=0.85,
                    threshold=0.80,
                    timestamp=base_time + timedelta(days=1),
                )
            )
            await store.record_evaluation(
                GateMetricRecord(
                    gate_name="testability",
                    passed=False,
                    score=0.50,
                    threshold=0.80,
                    timestamp=base_time + timedelta(days=1, hours=1),
                )
            )

            metrics = await store.get_metrics()
            trends = calculate_trends(metrics, "daily")

            assert len(trends) == 2
            assert trends[0].pass_rate == 100.0
            assert trends[0].period_label == "2026-01-01"
            assert trends[1].pass_rate == 50.0
            assert trends[1].period_label == "2026-01-02"
            assert trends[1].direction == TrendDirection.DECLINING

    @pytest.mark.asyncio
    async def test_agent_breakdown_from_evaluation_data(self) -> None:
        """Agent breakdown should work with evaluation data."""
        clear_evaluators()

        async def simple_evaluator(ctx: GateContext) -> GateResult:
            # Pass if agent is "analyst", fail if "dev"
            agent = ctx.state.get("current_agent")
            passed = agent == "analyst"
            return GateResult(passed=passed, gate_name=ctx.gate_name)

        register_evaluator("agent_test_gate", simple_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonGateMetricsStore(base_path=Path(tmpdir))
            set_metrics_store(store)

            try:

                @quality_gate("agent_test_gate", blocking=False)
                async def test_node(state: dict) -> dict:
                    return state

                # Run with different agents
                await test_node({"current_agent": "analyst"})
                await asyncio.sleep(0.05)
                await test_node({"current_agent": "analyst"})
                await asyncio.sleep(0.05)
                await test_node({"current_agent": "dev"})
                await asyncio.sleep(0.05)
                await test_node({"current_agent": "dev"})
                await asyncio.sleep(0.1)

                metrics = await store.get_metrics()
                breakdown = get_agent_breakdown(metrics)

                assert "analyst" in breakdown
                assert "dev" in breakdown
                assert breakdown["analyst"].pass_rate == 100.0
                assert breakdown["dev"].pass_rate == 0.0

            finally:
                set_metrics_store(None)
                clear_evaluators()
