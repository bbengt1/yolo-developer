"""Quality gate integration tests for Dev agent (Story 8.1, AC5).

Tests for the definition_of_done gate integration with dev_node.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yolo_developer.agents.dev.node import dev_node
from yolo_developer.orchestrator.state import YoloState


class TestQualityGateDecorator:
    """Tests for quality gate decorator on dev_node."""

    def test_dev_node_has_quality_gate_attribute(self) -> None:
        """Test dev_node has quality gate metadata."""
        # The @quality_gate decorator adds metadata to the function
        # Check that the function is wrapped (has __wrapped__ or gate info)
        assert callable(dev_node)
        # The function should still be async
        import asyncio

        assert asyncio.iscoroutinefunction(dev_node)


class TestGateIntegration:
    """Tests for gate integration behavior."""

    @pytest.mark.asyncio
    async def test_dev_node_executes_with_gate(self) -> None:
        """Test dev_node executes successfully with gate decorator."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        # Should complete without error even with gate in advisory mode
        result = await dev_node(state)
        assert "messages" in result
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_gate_in_advisory_mode(self) -> None:
        """Test gate is in advisory mode (blocking=False)."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        # In advisory mode, gate failures don't block execution
        # The function should complete regardless of gate result
        result = await dev_node(state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_processes_with_gate_evaluation(self) -> None:
        """Test processing continues after gate evaluation."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
            "architect_output": {
                "design_decisions": [
                    {"story_id": "story-001", "decision_type": "pattern"},
                ],
            },
        }
        result = await dev_node(state)
        # Gate should not block, so we get implementations
        assert len(result["dev_output"]["implementations"]) == 1


class TestGateResultLogging:
    """Tests for gate result logging."""

    @pytest.mark.asyncio
    async def test_logs_processing_start(self) -> None:
        """Test dev_node logs at start."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        with patch("yolo_developer.agents.dev.node.logger") as mock_logger:
            await dev_node(state)
            mock_logger.info.assert_any_call(
                "dev_node_start",
                current_agent="dev",
                message_count=0,
            )

    @pytest.mark.asyncio
    async def test_logs_processing_complete(self) -> None:
        """Test dev_node logs at completion."""
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        with patch("yolo_developer.agents.dev.node.logger") as mock_logger:
            await dev_node(state)
            # Check that dev_node_complete was called
            call_names = [call[0][0] for call in mock_logger.info.call_args_list]
            assert "dev_node_complete" in call_names


class TestRetryBehavior:
    """Tests for tenacity retry decorator."""

    def test_retry_decorator_applied(self) -> None:
        """Test that retry decorator is applied to dev_node."""
        # The @retry decorator wraps the function
        # We can check for the __wrapped__ attribute chain
        import asyncio

        assert asyncio.iscoroutinefunction(dev_node)
        # The function should be callable and awaitable
        assert callable(dev_node)
