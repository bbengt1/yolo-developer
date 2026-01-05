"""Integration tests for testability gate with decorator.

Tests the full integration of the testability gate evaluator
with the @quality_gate decorator framework.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.gates import quality_gate
from yolo_developer.gates.gates import testability  # noqa: F401 - registers evaluator


@pytest.fixture(autouse=True)
def setup_evaluators() -> None:
    """Ensure testability evaluator is registered for each test.

    The import of testability module registers the evaluator,
    but we need to ensure it's available after any clear_evaluators() calls.
    """
    # Import registers the evaluator
    # Ensure it's registered (in case previous test cleared it)
    from yolo_developer.gates.evaluators import get_evaluator
    from yolo_developer.gates.gates.testability import (
        register_evaluator,
        testability_evaluator,
    )

    if get_evaluator("testability") is None:
        register_evaluator("testability", testability_evaluator)


class TestTestabilityGateIntegration:
    """Integration tests for testability gate with decorator."""

    @pytest.mark.asyncio
    async def test_gate_decorator_blocks_untestable_requirements(self) -> None:
        """Gate decorator blocks execution when requirements are untestable."""
        node_executed = False

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal node_executed
            node_executed = True
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The system should be fast and intuitive"},
            ]
        }

        result = await analyst_node(state)

        assert node_executed is False
        assert result.get("gate_blocked") is True
        assert "testability" in result.get("gate_failure", "").lower()
        assert "processed" not in result

    @pytest.mark.asyncio
    async def test_gate_decorator_passes_testable_requirements(self) -> None:
        """Gate decorator allows execution when requirements are testable."""
        node_executed = False

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal node_executed
            node_executed = True
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The API responds within 500ms for 95% of requests"},
                {
                    "id": "req-002",
                    "content": "Given valid credentials, When user logs in, Then session is created",
                },
            ]
        }

        result = await analyst_node(state)

        assert node_executed is True
        assert result.get("gate_blocked") is not True
        assert result.get("processed") is True

    @pytest.mark.asyncio
    async def test_gate_reads_requirements_from_state(self) -> None:
        """Gate evaluator reads requirements from state['requirements']."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        # Test with requirements in state
        state_with_reqs: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "API responds within 500ms for 95% of requests"},
            ]
        }

        result = await analyst_node(state_with_reqs)
        assert result.get("gate_blocked") is not True

        # Test with no requirements key - should pass (nothing to validate)
        state_no_reqs: dict[str, Any] = {"other_data": "value"}
        result = await analyst_node(state_no_reqs)
        assert result.get("gate_blocked") is not True

    @pytest.mark.asyncio
    async def test_gate_result_recorded_in_state(self) -> None:
        """Gate evaluation result is recorded in state['gate_results']."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The API responds within 500ms"},
            ]
        }

        result = await analyst_node(state)

        assert "gate_results" in result
        assert len(result["gate_results"]) == 1
        assert result["gate_results"][0]["gate_name"] == "testability"
        assert result["gate_results"][0]["passed"] is True

    @pytest.mark.asyncio
    async def test_gate_blocking_records_failure_details(self) -> None:
        """Blocking gate failure records detailed failure information."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        state: dict[str, Any] = {
            "requirements": [
                {
                    "id": "req-bad",
                    "content": "The user interface should be beautiful and intuitive",
                },
            ]
        }

        result = await analyst_node(state)

        assert result.get("gate_blocked") is True
        assert result.get("gate_failure") is not None
        # Failure should mention the specific requirement
        assert "req-bad" in result["gate_failure"]
        # Failure should mention the vague terms found
        assert (
            "beautiful" in result["gate_failure"].lower()
            or "intuitive" in result["gate_failure"].lower()
        )

    @pytest.mark.asyncio
    async def test_gate_advisory_mode_continues_on_failure(self) -> None:
        """Advisory gate mode logs warning but continues execution."""
        node_executed = False

        @quality_gate("testability", blocking=False)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal node_executed
            node_executed = True
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The system should be user-friendly"},
            ]
        }

        result = await analyst_node(state)

        assert node_executed is True
        assert result.get("processed") is True
        assert result.get("gate_blocked") is not True
        # Advisory warnings should be recorded
        assert "advisory_warnings" in result
        assert len(result["advisory_warnings"]) == 1
        assert result["advisory_warnings"][0]["gate_name"] == "testability"

    @pytest.mark.asyncio
    async def test_multiple_requirements_partial_failure(self) -> None:
        """Gate fails if any requirement is untestable."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            return state

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-good", "content": "API responds within 500ms"},
                {"id": "req-bad", "content": "The UI should be intuitive"},
                {"id": "req-good2", "content": "System handles at least 1000 concurrent users"},
            ]
        }

        result = await analyst_node(state)

        assert result.get("gate_blocked") is True
        # Failure should mention the bad requirement
        assert "req-bad" in result["gate_failure"]
        # Good requirements should not be in failure (req-good and req-good2 pass)
        failure_text = result["gate_failure"]
        # Extract just the failing requirements line
        failing_line = next(line for line in failure_text.split("\n") if "requirement(s)" in line)
        assert "req-good," not in failing_line  # req-good shouldn't be in the list
        assert "req-bad" in failing_line

    @pytest.mark.asyncio
    async def test_empty_requirements_passes(self) -> None:
        """Empty requirements list passes the gate."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            state["processed"] = True
            return state

        state: dict[str, Any] = {"requirements": []}

        result = await analyst_node(state)

        assert result.get("gate_blocked") is not True
        assert result.get("processed") is True

    @pytest.mark.asyncio
    async def test_requirement_with_explicit_success_criteria(self) -> None:
        """Requirement with explicit success_criteria field passes."""

        @quality_gate("testability", blocking=True)
        async def analyst_node(state: dict[str, Any]) -> dict[str, Any]:
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "requirements": [
                {
                    "id": "req-001",
                    "content": "The system should perform well",
                    "success_criteria": "Response time < 200ms for 99th percentile",
                },
            ]
        }

        result = await analyst_node(state)

        assert result.get("gate_blocked") is not True
        assert result.get("processed") is True
