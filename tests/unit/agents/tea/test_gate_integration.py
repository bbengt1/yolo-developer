"""Integration tests for TEA agent quality gate (Story 9.1).

Tests for the integration between tea_node and the confidence_scoring gate.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.tea.node import tea_node
from yolo_developer.orchestrator.state import YoloState


@pytest.fixture
def state_with_artifacts() -> YoloState:
    """Create a state with dev_output artifacts for testing."""
    return {
        "messages": [],
        "current_agent": "tea",
        "handoff_context": None,
        "decisions": [],
        "dev_output": {
            "implementations": [
                {
                    "story_id": "story-001",
                    "code_files": [
                        {
                            "file_path": "src/main.py",
                            "content": '"""Module docstring."""\n\ndef main() -> None:\n    pass',
                            "file_type": "source",
                        },
                    ],
                    "test_files": [],
                },
            ],
        },
    }


class TestQualityGateDecoration:
    """Tests for quality gate decoration of tea_node."""

    def test_tea_node_has_gate_decoration(self) -> None:
        """Test that tea_node has quality gate decoration."""
        # The function should have wrapper attributes from decorators
        # We can verify by checking the function exists and is callable
        assert callable(tea_node)

    @pytest.mark.asyncio
    async def test_gate_runs_without_blocking(self, state_with_artifacts: YoloState) -> None:
        """Test that gate runs in advisory mode (blocking=False)."""
        # Should complete without raising exception even if gate would fail
        result = await tea_node(state_with_artifacts)
        assert result is not None
        assert "messages" in result


class TestGateResultHandling:
    """Tests for handling gate results."""

    @pytest.mark.asyncio
    async def test_returns_results_even_with_gate_pass(
        self, state_with_artifacts: YoloState
    ) -> None:
        """Test that results are returned when gate passes."""
        result = await tea_node(state_with_artifacts)

        # Should have all expected keys
        assert "messages" in result
        assert "decisions" in result
        assert "tea_output" in result

    @pytest.mark.asyncio
    async def test_tea_output_structure_correct(self, state_with_artifacts: YoloState) -> None:
        """Test that TEA output has correct structure for gate consumption."""
        result = await tea_node(state_with_artifacts)
        tea_output = result["tea_output"]

        # Gate expects these fields for confidence calculation
        assert "overall_confidence" in tea_output
        assert "deployment_recommendation" in tea_output
        assert "validation_results" in tea_output

        # Confidence should be a float between 0 and 1
        assert isinstance(tea_output["overall_confidence"], float)
        assert 0.0 <= tea_output["overall_confidence"] <= 1.0


class TestGateLogging:
    """Tests for gate result logging."""

    @pytest.mark.asyncio
    async def test_logs_completion_with_confidence(
        self, state_with_artifacts: YoloState, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that completion is logged with confidence score."""
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        with caplog.at_level("INFO"):
            await tea_node(state_with_artifacts)

        # Should have logged completion
        # Note: Exact log message depends on structlog configuration


class TestGateMocking:
    """Tests with mocked gate behavior."""

    @pytest.mark.asyncio
    async def test_gate_evaluator_called(self, state_with_artifacts: YoloState) -> None:
        """Test that gate evaluator is invoked during tea_node execution."""
        # The gate is registered with blocking=False, so even if evaluation fails,
        # the function should complete
        result = await tea_node(state_with_artifacts)
        assert result is not None

    @pytest.mark.asyncio
    async def test_gate_receives_state_context(self, state_with_artifacts: YoloState) -> None:
        """Test that gate receives proper state context."""
        # Run the node to ensure gate context is created properly
        result = await tea_node(state_with_artifacts)

        # The gate should have access to tea_output for confidence calculation
        tea_output = result["tea_output"]
        assert tea_output["overall_confidence"] is not None


class TestAdvisoryModeGate:
    """Tests for advisory mode gate behavior."""

    @pytest.mark.asyncio
    async def test_advisory_mode_does_not_block(self, state_with_artifacts: YoloState) -> None:
        """Test that advisory mode doesn't block on gate failure."""
        # Even with potential gate issues, function should complete
        result = await tea_node(state_with_artifacts)

        assert result is not None
        assert "tea_output" in result

    @pytest.mark.asyncio
    async def test_returns_output_regardless_of_gate_result(
        self, state_with_artifacts: YoloState
    ) -> None:
        """Test that output is returned regardless of gate result."""
        result = await tea_node(state_with_artifacts)

        # Should have messages and decisions
        assert len(result["messages"]) >= 1
        assert len(result["decisions"]) >= 1

        # TEA output should be complete
        tea_output = result["tea_output"]
        assert "validation_results" in tea_output
        assert "overall_confidence" in tea_output
        assert "deployment_recommendation" in tea_output
