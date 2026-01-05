"""Unit tests for gate types and data structures.

Tests GateResult, GateMode, and GateContext dataclasses and enums.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest


class TestGateMode:
    """Tests for GateMode enum."""

    def test_blocking_mode_exists(self) -> None:
        """GateMode has BLOCKING variant."""
        from yolo_developer.gates.types import GateMode

        assert GateMode.BLOCKING.value == "blocking"

    def test_advisory_mode_exists(self) -> None:
        """GateMode has ADVISORY variant."""
        from yolo_developer.gates.types import GateMode

        assert GateMode.ADVISORY.value == "advisory"

    def test_gate_mode_is_enum(self) -> None:
        """GateMode is an Enum class."""
        from enum import Enum

        from yolo_developer.gates.types import GateMode

        assert issubclass(GateMode, Enum)


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_gate_result_passed_true(self) -> None:
        """GateResult can be created with passed=True."""
        from yolo_developer.gates.types import GateResult

        result = GateResult(
            passed=True,
            gate_name="testability",
        )
        assert result.passed is True
        assert result.gate_name == "testability"
        assert result.reason is None

    def test_gate_result_passed_false_with_reason(self) -> None:
        """GateResult can be created with passed=False and reason."""
        from yolo_developer.gates.types import GateResult

        result = GateResult(
            passed=False,
            gate_name="architecture",
            reason="Missing required dependency diagram",
        )
        assert result.passed is False
        assert result.reason == "Missing required dependency diagram"

    def test_gate_result_has_timestamp(self) -> None:
        """GateResult has auto-generated timestamp."""
        from yolo_developer.gates.types import GateResult

        before = datetime.now(timezone.utc)
        result = GateResult(passed=True, gate_name="test")
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    def test_gate_result_custom_timestamp(self) -> None:
        """GateResult accepts custom timestamp."""
        from yolo_developer.gates.types import GateResult

        custom_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = GateResult(passed=True, gate_name="test", timestamp=custom_time)
        assert result.timestamp == custom_time

    def test_gate_result_is_frozen(self) -> None:
        """GateResult is immutable (frozen dataclass)."""
        from yolo_developer.gates.types import GateResult

        result = GateResult(passed=True, gate_name="test")
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_gate_result_to_dict(self) -> None:
        """GateResult can be converted to dictionary."""
        from yolo_developer.gates.types import GateResult

        result = GateResult(
            passed=False,
            gate_name="testability",
            reason="Tests missing",
        )
        result_dict = result.to_dict()

        assert result_dict["passed"] is False
        assert result_dict["gate_name"] == "testability"
        assert result_dict["reason"] == "Tests missing"
        assert "timestamp" in result_dict


class TestGateContext:
    """Tests for GateContext dataclass."""

    def test_gate_context_with_state(self) -> None:
        """GateContext holds state dictionary."""
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"messages": [], "current_agent": "analyst"}
        context = GateContext(state=state, gate_name="testability")

        assert context.state == state
        assert context.gate_name == "testability"

    def test_gate_context_with_artifact_id(self) -> None:
        """GateContext can include artifact_id."""
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"key": "value"},
            gate_name="architecture",
            artifact_id="story-001",
        )
        assert context.artifact_id == "story-001"

    def test_gate_context_default_artifact_id_is_none(self) -> None:
        """GateContext artifact_id defaults to None."""
        from yolo_developer.gates.types import GateContext

        context = GateContext(state={}, gate_name="test")
        assert context.artifact_id is None

    def test_gate_context_is_frozen(self) -> None:
        """GateContext is immutable (frozen dataclass)."""
        from yolo_developer.gates.types import GateContext

        context = GateContext(state={}, gate_name="test")
        with pytest.raises(AttributeError):
            context.gate_name = "other"  # type: ignore[misc]

    def test_gate_context_with_metadata(self) -> None:
        """GateContext can include arbitrary metadata."""
        from yolo_developer.gates.types import GateContext

        metadata = {"severity": "high", "category": "security"}
        context = GateContext(
            state={},
            gate_name="security",
            metadata=metadata,
        )
        assert context.metadata == metadata

    def test_gate_context_metadata_defaults_to_empty(self) -> None:
        """GateContext metadata defaults to empty dict."""
        from yolo_developer.gates.types import GateContext

        context = GateContext(state={}, gate_name="test")
        assert context.metadata == {}
