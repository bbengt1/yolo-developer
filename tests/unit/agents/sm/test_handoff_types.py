"""Tests for handoff management types (Story 10.8).

Tests all type definitions for the agent handoff management system:
- HandoffStatus Literal type
- HandoffMetrics dataclass
- HandoffRecord dataclass
- HandoffResult dataclass
- HandoffConfig dataclass
- Constants and validation sets
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, get_args

import pytest


class TestHandoffStatusType:
    """Tests for HandoffStatus Literal type (Task 1.2)."""

    def test_valid_statuses(self) -> None:
        """Should define exactly four valid handoff statuses."""
        from yolo_developer.agents.sm.handoff_types import HandoffStatus

        valid_statuses = get_args(HandoffStatus)
        assert len(valid_statuses) == 4
        assert "pending" in valid_statuses
        assert "in_progress" in valid_statuses
        assert "completed" in valid_statuses
        assert "failed" in valid_statuses

    def test_valid_handoff_statuses_constant(self) -> None:
        """Should have VALID_HANDOFF_STATUSES frozenset matching Literal."""
        from yolo_developer.agents.sm.handoff_types import (
            VALID_HANDOFF_STATUSES,
            HandoffStatus,
        )

        valid_statuses = get_args(HandoffStatus)
        assert VALID_HANDOFF_STATUSES == frozenset(valid_statuses)


class TestHandoffMetrics:
    """Tests for HandoffMetrics dataclass (Task 1.3)."""

    def test_create_metrics(self) -> None:
        """Should create metrics with all required fields."""
        from yolo_developer.agents.sm.handoff_types import HandoffMetrics

        metrics = HandoffMetrics(
            duration_ms=150.5,
            context_size_bytes=1024,
            messages_transferred=10,
            decisions_transferred=3,
        )

        assert metrics.duration_ms == 150.5
        assert metrics.context_size_bytes == 1024
        assert metrics.messages_transferred == 10
        assert metrics.decisions_transferred == 3
        assert metrics.memory_refs_transferred == 0  # Default

    def test_metrics_with_memory_refs(self) -> None:
        """Should allow specifying memory_refs_transferred."""
        from yolo_developer.agents.sm.handoff_types import HandoffMetrics

        metrics = HandoffMetrics(
            duration_ms=100.0,
            context_size_bytes=512,
            messages_transferred=5,
            decisions_transferred=2,
            memory_refs_transferred=7,
        )

        assert metrics.memory_refs_transferred == 7

    def test_metrics_is_frozen(self) -> None:
        """Should be immutable (frozen dataclass)."""
        from yolo_developer.agents.sm.handoff_types import HandoffMetrics

        metrics = HandoffMetrics(
            duration_ms=100.0,
            context_size_bytes=512,
            messages_transferred=5,
            decisions_transferred=2,
        )

        with pytest.raises(AttributeError):
            metrics.duration_ms = 200.0  # type: ignore[misc]

    def test_metrics_to_dict(self) -> None:
        """Should serialize to dictionary correctly."""
        from yolo_developer.agents.sm.handoff_types import HandoffMetrics

        metrics = HandoffMetrics(
            duration_ms=150.5,
            context_size_bytes=1024,
            messages_transferred=10,
            decisions_transferred=3,
            memory_refs_transferred=5,
        )

        result = metrics.to_dict()

        assert result == {
            "duration_ms": 150.5,
            "context_size_bytes": 1024,
            "messages_transferred": 10,
            "decisions_transferred": 3,
            "memory_refs_transferred": 5,
        }


class TestHandoffRecord:
    """Tests for HandoffRecord dataclass (Task 1.4)."""

    def test_create_record_minimal(self) -> None:
        """Should create record with minimal required fields."""
        from yolo_developer.agents.sm.handoff_types import HandoffRecord

        record = HandoffRecord(
            handoff_id="handoff_analyst_pm_123",
            source_agent="analyst",
            target_agent="pm",
            status="pending",
        )

        assert record.handoff_id == "handoff_analyst_pm_123"
        assert record.source_agent == "analyst"
        assert record.target_agent == "pm"
        assert record.status == "pending"
        assert record.started_at is not None  # Auto-generated
        assert record.completed_at is None
        assert record.metrics is None
        assert record.context_checksum is None
        assert record.error_message is None

    def test_create_record_complete(self) -> None:
        """Should create record with all fields."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffMetrics,
            HandoffRecord,
        )

        metrics = HandoffMetrics(
            duration_ms=100.0,
            context_size_bytes=512,
            messages_transferred=5,
            decisions_transferred=2,
        )

        record = HandoffRecord(
            handoff_id="handoff_pm_architect_456",
            source_agent="pm",
            target_agent="architect",
            status="completed",
            started_at="2026-01-16T10:00:00+00:00",
            completed_at="2026-01-16T10:00:01+00:00",
            metrics=metrics,
            context_checksum="abc123",
            error_message=None,
        )

        assert record.status == "completed"
        assert record.completed_at == "2026-01-16T10:00:01+00:00"
        assert record.metrics is not None
        assert record.context_checksum == "abc123"

    def test_record_with_error(self) -> None:
        """Should allow creating failed record with error message."""
        from yolo_developer.agents.sm.handoff_types import HandoffRecord

        record = HandoffRecord(
            handoff_id="handoff_dev_tea_789",
            source_agent="dev",
            target_agent="tea",
            status="failed",
            error_message="Context validation failed",
        )

        assert record.status == "failed"
        assert record.error_message == "Context validation failed"

    def test_record_is_frozen(self) -> None:
        """Should be immutable (frozen dataclass)."""
        from yolo_developer.agents.sm.handoff_types import HandoffRecord

        record = HandoffRecord(
            handoff_id="test",
            source_agent="analyst",
            target_agent="pm",
            status="pending",
        )

        with pytest.raises(AttributeError):
            record.status = "completed"  # type: ignore[misc]

    def test_record_to_dict(self) -> None:
        """Should serialize to dictionary correctly."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffMetrics,
            HandoffRecord,
        )

        metrics = HandoffMetrics(
            duration_ms=100.0,
            context_size_bytes=512,
            messages_transferred=5,
            decisions_transferred=2,
        )

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="completed",
            started_at="2026-01-16T10:00:00+00:00",
            completed_at="2026-01-16T10:00:01+00:00",
            metrics=metrics,
            context_checksum="abc123",
        )

        result = record.to_dict()

        assert result["handoff_id"] == "handoff_123"
        assert result["source_agent"] == "analyst"
        assert result["target_agent"] == "pm"
        assert result["status"] == "completed"
        assert result["metrics"] is not None
        assert result["metrics"]["duration_ms"] == 100.0
        assert result["context_checksum"] == "abc123"

    def test_record_to_dict_without_metrics(self) -> None:
        """Should handle None metrics in serialization."""
        from yolo_developer.agents.sm.handoff_types import HandoffRecord

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="pending",
        )

        result = record.to_dict()

        assert result["metrics"] is None


class TestHandoffResult:
    """Tests for HandoffResult dataclass (Task 1.5)."""

    def test_create_result_success(self) -> None:
        """Should create successful handoff result."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffRecord,
            HandoffResult,
        )

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="completed",
        )

        result = HandoffResult(
            record=record,
            success=True,
            context_validated=True,
            state_updates={"current_agent": "pm"},
        )

        assert result.success is True
        assert result.context_validated is True
        assert result.state_updates == {"current_agent": "pm"}
        assert result.warnings == ()  # Default empty tuple

    def test_create_result_with_warnings(self) -> None:
        """Should allow creating result with warnings."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffRecord,
            HandoffResult,
        )

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="completed",
        )

        result = HandoffResult(
            record=record,
            success=True,
            context_validated=True,
            warnings=("Context size exceeded recommendation", "Missing optional field"),
        )

        assert len(result.warnings) == 2
        assert "Context size exceeded recommendation" in result.warnings

    def test_create_result_failure(self) -> None:
        """Should allow creating failed handoff result."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffRecord,
            HandoffResult,
        )

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="failed",
            error_message="Validation failed",
        )

        result = HandoffResult(
            record=record,
            success=False,
            context_validated=False,
            state_updates=None,
        )

        assert result.success is False
        assert result.context_validated is False
        assert result.state_updates is None

    def test_result_is_frozen(self) -> None:
        """Should be immutable (frozen dataclass)."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffRecord,
            HandoffResult,
        )

        record = HandoffRecord(
            handoff_id="test",
            source_agent="analyst",
            target_agent="pm",
            status="pending",
        )

        result = HandoffResult(
            record=record,
            success=True,
            context_validated=True,
        )

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_result_to_dict(self) -> None:
        """Should serialize to dictionary correctly."""
        from yolo_developer.agents.sm.handoff_types import (
            HandoffRecord,
            HandoffResult,
        )

        record = HandoffRecord(
            handoff_id="handoff_123",
            source_agent="analyst",
            target_agent="pm",
            status="completed",
        )

        result = HandoffResult(
            record=record,
            success=True,
            context_validated=True,
            state_updates={"current_agent": "pm"},
            warnings=("warning1",),
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["context_validated"] is True
        assert data["record"]["handoff_id"] == "handoff_123"
        assert data["state_updates"] == {"current_agent": "pm"}
        assert data["warnings"] == ["warning1"]


class TestHandoffConfig:
    """Tests for HandoffConfig dataclass (Task 1.6)."""

    def test_create_config_defaults(self) -> None:
        """Should create config with sensible defaults."""
        from yolo_developer.agents.sm.handoff_types import (
            DEFAULT_MAX_CONTEXT_SIZE,
            DEFAULT_TIMEOUT_SECONDS,
            HandoffConfig,
        )

        config = HandoffConfig()

        assert config.validate_context_integrity is True
        assert config.log_timing is True
        assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert config.max_context_size_bytes == DEFAULT_MAX_CONTEXT_SIZE
        assert config.include_all_messages is False
        assert config.max_messages_to_transfer == 50

    def test_create_config_custom(self) -> None:
        """Should allow customizing all config options."""
        from yolo_developer.agents.sm.handoff_types import HandoffConfig

        config = HandoffConfig(
            validate_context_integrity=False,
            log_timing=False,
            timeout_seconds=10.0,
            max_context_size_bytes=2_000_000,
            include_all_messages=True,
            max_messages_to_transfer=100,
        )

        assert config.validate_context_integrity is False
        assert config.log_timing is False
        assert config.timeout_seconds == 10.0
        assert config.max_context_size_bytes == 2_000_000
        assert config.include_all_messages is True
        assert config.max_messages_to_transfer == 100

    def test_config_is_frozen(self) -> None:
        """Should be immutable (frozen dataclass)."""
        from yolo_developer.agents.sm.handoff_types import HandoffConfig

        config = HandoffConfig()

        with pytest.raises(AttributeError):
            config.timeout_seconds = 20.0  # type: ignore[misc]

    def test_config_to_dict(self) -> None:
        """Should serialize to dictionary correctly."""
        from yolo_developer.agents.sm.handoff_types import HandoffConfig

        config = HandoffConfig(
            validate_context_integrity=True,
            log_timing=True,
            timeout_seconds=5.0,
            max_context_size_bytes=1_000_000,
            include_all_messages=False,
            max_messages_to_transfer=50,
        )

        result = config.to_dict()

        assert result == {
            "validate_context_integrity": True,
            "log_timing": True,
            "timeout_seconds": 5.0,
            "max_context_size_bytes": 1_000_000,
            "include_all_messages": False,
            "max_messages_to_transfer": 50,
        }


class TestConstants:
    """Tests for module constants (Task 1.8)."""

    def test_default_timeout_seconds(self) -> None:
        """Should have DEFAULT_TIMEOUT_SECONDS of 5 per NFR-PERF-1."""
        from yolo_developer.agents.sm.handoff_types import DEFAULT_TIMEOUT_SECONDS

        assert DEFAULT_TIMEOUT_SECONDS == 5

    def test_default_max_context_size(self) -> None:
        """Should have DEFAULT_MAX_CONTEXT_SIZE of 1MB."""
        from yolo_developer.agents.sm.handoff_types import DEFAULT_MAX_CONTEXT_SIZE

        assert DEFAULT_MAX_CONTEXT_SIZE == 1_000_000

    def test_valid_handoff_statuses_is_frozenset(self) -> None:
        """Should be a frozenset for immutability."""
        from yolo_developer.agents.sm.handoff_types import VALID_HANDOFF_STATUSES

        assert isinstance(VALID_HANDOFF_STATUSES, frozenset)

    def test_valid_handoff_statuses_content(self) -> None:
        """Should contain exactly the four valid statuses."""
        from yolo_developer.agents.sm.handoff_types import VALID_HANDOFF_STATUSES

        assert VALID_HANDOFF_STATUSES == frozenset(
            {"pending", "in_progress", "completed", "failed"}
        )
