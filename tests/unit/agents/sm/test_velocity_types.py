"""Tests for velocity tracking types (Story 10.12).

Tests the velocity tracking data types:
- SprintVelocity: Velocity data for a single sprint
- VelocityMetrics: Aggregated velocity metrics
- VelocityConfig: Configuration for velocity calculations
- VelocityForecast: Forecasted velocity for planning
- Constants and literal types

References:
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging

import pytest

from yolo_developer.agents.sm.velocity_types import (
    DEFAULT_MIN_SPRINTS_FOR_FORECAST,
    DEFAULT_MIN_SPRINTS_FOR_TREND,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_TREND_THRESHOLD,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    VALID_TRENDS,
    SprintVelocity,
    VelocityConfig,
    VelocityForecast,
    VelocityMetrics,
)


class TestSprintVelocity:
    """Tests for SprintVelocity dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Test creating SprintVelocity with required fields."""
        velocity = SprintVelocity(
            sprint_id="sprint-1",
            stories_completed=5,
            points_completed=5.0,
            total_cycle_time_ms=18000000.0,
            avg_cycle_time_ms=3600000.0,
        )
        assert velocity.sprint_id == "sprint-1"
        assert velocity.stories_completed == 5
        assert velocity.points_completed == 5.0
        assert velocity.total_cycle_time_ms == 18000000.0
        assert velocity.avg_cycle_time_ms == 3600000.0
        assert velocity.completed_at is not None

    def test_immutable(self) -> None:
        """Test that SprintVelocity is frozen/immutable."""
        velocity = SprintVelocity(
            sprint_id="sprint-1",
            stories_completed=5,
            points_completed=5.0,
            total_cycle_time_ms=18000000.0,
            avg_cycle_time_ms=3600000.0,
        )
        with pytest.raises(AttributeError):
            velocity.stories_completed = 10  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        velocity = SprintVelocity(
            sprint_id="sprint-2",
            stories_completed=3,
            points_completed=6.0,
            total_cycle_time_ms=10800000.0,
            avg_cycle_time_ms=3600000.0,
        )
        d = velocity.to_dict()
        assert d["sprint_id"] == "sprint-2"
        assert d["stories_completed"] == 3
        assert d["points_completed"] == 6.0
        assert d["total_cycle_time_ms"] == 10800000.0
        assert d["avg_cycle_time_ms"] == 3600000.0
        assert "completed_at" in d

    def test_post_init_warning_negative_stories(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative stories_completed logs a warning."""
        with caplog.at_level(logging.WARNING):
            SprintVelocity(
                sprint_id="sprint-bad",
                stories_completed=-1,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            )
        assert "stories_completed=-1 is negative" in caplog.text

    def test_post_init_warning_negative_points(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative points_completed logs a warning."""
        with caplog.at_level(logging.WARNING):
            SprintVelocity(
                sprint_id="sprint-bad",
                stories_completed=5,
                points_completed=-5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            )
        assert "points_completed=-5.00 is negative" in caplog.text

    def test_post_init_warning_negative_cycle_time(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative total_cycle_time_ms logs a warning."""
        with caplog.at_level(logging.WARNING):
            SprintVelocity(
                sprint_id="sprint-bad",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=-1000.0,
                avg_cycle_time_ms=0.0,
            )
        assert "total_cycle_time_ms=-1000.00 is negative" in caplog.text


class TestVelocityMetrics:
    """Tests for VelocityMetrics dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Test creating VelocityMetrics with required fields."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.5,
            average_points_per_sprint=5.5,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="stable",
        )
        assert metrics.average_stories_per_sprint == 5.5
        assert metrics.trend == "stable"
        assert metrics.variance_stories == 0.0  # default
        assert metrics.variance_points == 0.0  # default

    def test_with_optional_fields(self) -> None:
        """Test creating VelocityMetrics with optional fields."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=3,
            trend="improving",
            variance_stories=1.5,
            variance_points=1.2,
        )
        assert metrics.variance_stories == 1.5
        assert metrics.variance_points == 1.2

    def test_immutable(self) -> None:
        """Test that VelocityMetrics is frozen/immutable."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="stable",
        )
        with pytest.raises(AttributeError):
            metrics.trend = "improving"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=6.0,
            average_points_per_sprint=12.0,
            average_cycle_time_ms=4000000.0,
            cycle_time_p50_ms=3800000.0,
            cycle_time_p90_ms=5500000.0,
            sprints_analyzed=10,
            trend="declining",
            variance_stories=2.0,
        )
        d = metrics.to_dict()
        assert d["average_stories_per_sprint"] == 6.0
        assert d["trend"] == "declining"
        assert d["variance_stories"] == 2.0
        assert "calculated_at" in d

    def test_post_init_warning_negative_avg_stories(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative average_stories_per_sprint logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityMetrics(
                average_stories_per_sprint=-1.0,
                average_points_per_sprint=5.0,
                average_cycle_time_ms=3600000.0,
                cycle_time_p50_ms=3500000.0,
                cycle_time_p90_ms=5000000.0,
                sprints_analyzed=5,
                trend="stable",
            )
        assert "average_stories_per_sprint=-1.00 is negative" in caplog.text

    def test_post_init_warning_negative_sprints_analyzed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative sprints_analyzed logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityMetrics(
                average_stories_per_sprint=5.0,
                average_points_per_sprint=5.0,
                average_cycle_time_ms=3600000.0,
                cycle_time_p50_ms=3500000.0,
                cycle_time_p90_ms=5000000.0,
                sprints_analyzed=-1,
                trend="stable",
            )
        assert "sprints_analyzed=-1 is negative" in caplog.text

    def test_post_init_warning_invalid_trend(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid trend logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityMetrics(
                average_stories_per_sprint=5.0,
                average_points_per_sprint=5.0,
                average_cycle_time_ms=3600000.0,
                cycle_time_p50_ms=3500000.0,
                cycle_time_p90_ms=5000000.0,
                sprints_analyzed=5,
                trend="invalid_trend",  # type: ignore[arg-type]
            )
        assert "trend='invalid_trend' is not a valid trend value" in caplog.text


class TestVelocityConfig:
    """Tests for VelocityConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values match constants."""
        config = VelocityConfig()
        assert config.rolling_window == DEFAULT_ROLLING_WINDOW
        assert config.trend_threshold == DEFAULT_TREND_THRESHOLD
        assert config.min_sprints_for_trend == DEFAULT_MIN_SPRINTS_FOR_TREND
        assert config.min_sprints_for_forecast == DEFAULT_MIN_SPRINTS_FOR_FORECAST

    def test_custom_values(self) -> None:
        """Test creating with custom values."""
        config = VelocityConfig(
            rolling_window=3,
            trend_threshold=0.15,
            min_sprints_for_trend=2,
            min_sprints_for_forecast=1,
        )
        assert config.rolling_window == 3
        assert config.trend_threshold == 0.15
        assert config.min_sprints_for_trend == 2
        assert config.min_sprints_for_forecast == 1

    def test_immutable(self) -> None:
        """Test that VelocityConfig is frozen/immutable."""
        config = VelocityConfig()
        with pytest.raises(AttributeError):
            config.rolling_window = 10  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = VelocityConfig(rolling_window=7)
        d = config.to_dict()
        assert d["rolling_window"] == 7
        assert "trend_threshold" in d
        assert "min_sprints_for_trend" in d
        assert "min_sprints_for_forecast" in d

    def test_post_init_warning_invalid_rolling_window(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid rolling_window logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityConfig(rolling_window=0)
        assert "rolling_window=0 should be at least 1" in caplog.text

    def test_post_init_warning_invalid_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid trend_threshold logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityConfig(trend_threshold=1.5)
        assert "trend_threshold=1.500 should be between 0.0 and 1.0" in caplog.text

    def test_post_init_warning_negative_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative trend_threshold logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityConfig(trend_threshold=-0.1)
        assert "trend_threshold=-0.100 should be between 0.0 and 1.0" in caplog.text


class TestVelocityForecast:
    """Tests for VelocityForecast dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Test creating VelocityForecast with required fields."""
        forecast = VelocityForecast(
            expected_stories_next_sprint=5,
            expected_points_next_sprint=5.0,
            confidence=0.85,
            forecast_factors=("based_on_5_sprints", "stable_trend"),
        )
        assert forecast.expected_stories_next_sprint == 5
        assert forecast.expected_points_next_sprint == 5.0
        assert forecast.confidence == 0.85
        assert len(forecast.forecast_factors) == 2

    def test_immutable(self) -> None:
        """Test that VelocityForecast is frozen/immutable."""
        forecast = VelocityForecast(
            expected_stories_next_sprint=5,
            expected_points_next_sprint=5.0,
            confidence=0.85,
            forecast_factors=(),
        )
        with pytest.raises(AttributeError):
            forecast.confidence = 0.5  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        forecast = VelocityForecast(
            expected_stories_next_sprint=4,
            expected_points_next_sprint=8.0,
            confidence=0.7,
            forecast_factors=("low_data", "improving_trend"),
        )
        d = forecast.to_dict()
        assert d["expected_stories_next_sprint"] == 4
        assert d["expected_points_next_sprint"] == 8.0
        assert d["confidence"] == 0.7
        # Tuple converted to list for JSON
        assert d["forecast_factors"] == ["low_data", "improving_trend"]
        assert "generated_at" in d

    def test_post_init_warning_invalid_confidence(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid confidence logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityForecast(
                expected_stories_next_sprint=5,
                expected_points_next_sprint=5.0,
                confidence=1.5,
                forecast_factors=(),
            )
        assert "confidence=1.500 is outside valid range" in caplog.text

    def test_post_init_warning_negative_stories(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative stories logs a warning."""
        with caplog.at_level(logging.WARNING):
            VelocityForecast(
                expected_stories_next_sprint=-1,
                expected_points_next_sprint=5.0,
                confidence=0.5,
                forecast_factors=(),
            )
        assert "expected_stories_next_sprint=-1 is negative" in caplog.text


class TestConstants:
    """Tests for module constants."""

    def test_valid_trends(self) -> None:
        """Test VALID_TRENDS contains expected values."""
        assert VALID_TRENDS == frozenset({"improving", "stable", "declining"})
        assert "improving" in VALID_TRENDS
        assert "stable" in VALID_TRENDS
        assert "declining" in VALID_TRENDS

    def test_default_rolling_window(self) -> None:
        """Test DEFAULT_ROLLING_WINDOW is reasonable."""
        assert DEFAULT_ROLLING_WINDOW >= 1
        assert DEFAULT_ROLLING_WINDOW <= 20

    def test_default_trend_threshold(self) -> None:
        """Test DEFAULT_TREND_THRESHOLD is valid."""
        assert 0.0 <= DEFAULT_TREND_THRESHOLD <= 1.0

    def test_min_sprints_constants(self) -> None:
        """Test minimum sprint constants are positive."""
        assert DEFAULT_MIN_SPRINTS_FOR_TREND >= 1
        assert DEFAULT_MIN_SPRINTS_FOR_FORECAST >= 1

    def test_confidence_bounds(self) -> None:
        """Test confidence bound constants."""
        assert MIN_CONFIDENCE == 0.0
        assert MAX_CONFIDENCE == 1.0
        assert MIN_CONFIDENCE < MAX_CONFIDENCE
