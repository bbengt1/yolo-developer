"""Tests for cost tracking types (Story 11.6).

Tests for TokenUsage, CostRecord, and CostAggregation dataclasses.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.cost_types import (
    VALID_GROUPBY_VALUES,
    VALID_TIER_VALUES,
    CostAggregation,
    CostGroupBy,
    CostRecord,
    TokenUsage,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_create_token_usage(self) -> None:
        """Test creating a TokenUsage instance."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_is_frozen(self) -> None:
        """Test that TokenUsage is immutable."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        with pytest.raises(AttributeError):
            usage.prompt_tokens = 200  # type: ignore[misc]

    def test_token_usage_to_dict(self) -> None:
        """Test TokenUsage to_dict serialization."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        result = usage.to_dict()

        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_token_usage_warns_on_negative_prompt_tokens(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative prompt_tokens triggers a warning."""
        TokenUsage(
            prompt_tokens=-1,
            completion_tokens=50,
            total_tokens=49,
        )

        assert "prompt_tokens is negative" in caplog.text

    def test_token_usage_warns_on_negative_completion_tokens(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative completion_tokens triggers a warning."""
        TokenUsage(
            prompt_tokens=100,
            completion_tokens=-5,
            total_tokens=95,
        )

        assert "completion_tokens is negative" in caplog.text

    def test_token_usage_warns_on_negative_total_tokens(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative total_tokens triggers a warning."""
        TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=-10,
        )

        assert "total_tokens is negative" in caplog.text


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_create_cost_record(self) -> None:
        """Test creating a CostRecord instance."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        record = CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        assert record.id == "cost-001"
        assert record.model == "gpt-4o-mini"
        assert record.tier == "routine"
        assert record.cost_usd == 0.0015
        assert record.agent_name == "analyst"
        assert record.session_id == "session-123"
        assert record.story_id is None
        assert record.sprint_id is None
        assert record.metadata == {}

    def test_cost_record_with_optional_fields(self) -> None:
        """Test CostRecord with optional fields populated."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        record = CostRecord(
            id="cost-002",
            timestamp="2026-01-18T12:00:00Z",
            model="claude-sonnet-4-20250514",
            tier="complex",
            token_usage=usage,
            cost_usd=0.015,
            agent_name="pm",
            session_id="session-456",
            story_id="1-2-user-auth",
            sprint_id="sprint-1",
            metadata={"feature": "authentication"},
        )

        assert record.story_id == "1-2-user-auth"
        assert record.sprint_id == "sprint-1"
        assert record.metadata == {"feature": "authentication"}

    def test_cost_record_is_frozen(self) -> None:
        """Test that CostRecord is immutable."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        record = CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        with pytest.raises(AttributeError):
            record.cost_usd = 999.0  # type: ignore[misc]

    def test_cost_record_to_dict(self) -> None:
        """Test CostRecord to_dict serialization."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        record = CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
            story_id="1-2-user-auth",
        )

        result = record.to_dict()

        assert result["id"] == "cost-001"
        assert result["timestamp"] == "2026-01-18T12:00:00Z"
        assert result["model"] == "gpt-4o-mini"
        assert result["tier"] == "routine"
        assert result["token_usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        assert result["cost_usd"] == 0.0015
        assert result["agent_name"] == "analyst"
        assert result["session_id"] == "session-123"
        assert result["story_id"] == "1-2-user-auth"
        assert result["sprint_id"] is None
        assert result["metadata"] == {}

    def test_cost_record_warns_on_empty_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty id triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        assert "CostRecord id is empty" in caplog.text

    def test_cost_record_warns_on_invalid_tier(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid tier triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="invalid_tier",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        assert "tier='invalid_tier' is not a valid tier" in caplog.text

    def test_cost_record_warns_on_negative_cost(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative cost triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=-0.001,
            agent_name="analyst",
            session_id="session-123",
        )

        assert "cost_usd is negative" in caplog.text

    def test_cost_record_warns_on_empty_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty model triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="session-123",
        )

        assert "CostRecord model is empty" in caplog.text

    def test_cost_record_warns_on_empty_agent_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty agent_name triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="",
            session_id="session-123",
        )

        assert "CostRecord agent_name is empty" in caplog.text

    def test_cost_record_warns_on_empty_session_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty session_id triggers a warning."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        CostRecord(
            id="cost-001",
            timestamp="2026-01-18T12:00:00Z",
            model="gpt-4o-mini",
            tier="routine",
            token_usage=usage,
            cost_usd=0.0015,
            agent_name="analyst",
            session_id="",
        )

        assert "CostRecord session_id is empty" in caplog.text


class TestCostAggregation:
    """Tests for CostAggregation dataclass."""

    def test_create_cost_aggregation(self) -> None:
        """Test creating a CostAggregation instance."""
        agg = CostAggregation(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            total_cost_usd=0.015,
            call_count=10,
            models=("gpt-4o-mini", "claude-sonnet-4-20250514"),
        )

        assert agg.total_prompt_tokens == 1000
        assert agg.total_completion_tokens == 500
        assert agg.total_tokens == 1500
        assert agg.total_cost_usd == 0.015
        assert agg.call_count == 10
        assert agg.models == ("gpt-4o-mini", "claude-sonnet-4-20250514")
        assert agg.period_start is None
        assert agg.period_end is None

    def test_cost_aggregation_with_period(self) -> None:
        """Test CostAggregation with period fields."""
        agg = CostAggregation(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            total_cost_usd=0.015,
            call_count=10,
            models=("gpt-4o-mini",),
            period_start="2026-01-01T00:00:00Z",
            period_end="2026-01-18T23:59:59Z",
        )

        assert agg.period_start == "2026-01-01T00:00:00Z"
        assert agg.period_end == "2026-01-18T23:59:59Z"

    def test_cost_aggregation_is_frozen(self) -> None:
        """Test that CostAggregation is immutable."""
        agg = CostAggregation(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            total_cost_usd=0.015,
            call_count=10,
            models=("gpt-4o-mini",),
        )

        with pytest.raises(AttributeError):
            agg.total_cost_usd = 999.0  # type: ignore[misc]

    def test_cost_aggregation_to_dict(self) -> None:
        """Test CostAggregation to_dict serialization."""
        agg = CostAggregation(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            total_cost_usd=0.015,
            call_count=10,
            models=("gpt-4o-mini", "claude-sonnet-4-20250514"),
            period_start="2026-01-01T00:00:00Z",
            period_end="2026-01-18T23:59:59Z",
        )

        result = agg.to_dict()

        assert result == {
            "total_prompt_tokens": 1000,
            "total_completion_tokens": 500,
            "total_tokens": 1500,
            "total_cost_usd": 0.015,
            "call_count": 10,
            "models": ["gpt-4o-mini", "claude-sonnet-4-20250514"],
            "period_start": "2026-01-01T00:00:00Z",
            "period_end": "2026-01-18T23:59:59Z",
        }

    def test_cost_aggregation_empty(self) -> None:
        """Test CostAggregation with zero values (empty aggregation)."""
        agg = CostAggregation(
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            call_count=0,
            models=(),
        )

        assert agg.call_count == 0
        assert agg.total_cost_usd == 0.0
        assert agg.models == ()


class TestConstants:
    """Tests for module constants."""

    def test_valid_groupby_values(self) -> None:
        """Test VALID_GROUPBY_VALUES contains expected values."""
        assert "agent" in VALID_GROUPBY_VALUES
        assert "story" in VALID_GROUPBY_VALUES
        assert "sprint" in VALID_GROUPBY_VALUES
        assert "model" in VALID_GROUPBY_VALUES
        assert "tier" in VALID_GROUPBY_VALUES
        assert len(VALID_GROUPBY_VALUES) == 5

    def test_valid_tier_values(self) -> None:
        """Test VALID_TIER_VALUES contains expected values."""
        assert "routine" in VALID_TIER_VALUES
        assert "complex" in VALID_TIER_VALUES
        assert "critical" in VALID_TIER_VALUES
        assert len(VALID_TIER_VALUES) == 3


class TestCostGroupByType:
    """Tests for CostGroupBy Literal type."""

    def test_cost_groupby_accepts_valid_values(self) -> None:
        """Test that CostGroupBy type accepts valid values."""
        # This is a type-level test - if it compiles, it passes
        valid_values: list[CostGroupBy] = ["agent", "story", "sprint", "model", "tier"]
        assert len(valid_values) == 5
