"""Tests for velocity tracking module (Story 10.12).

Tests the velocity tracking functions:
- calculate_sprint_velocity: Calculate velocity for a completed sprint
- calculate_velocity_metrics: Aggregate velocity across multiple sprints
- get_velocity_trend: Determine velocity trend (improving/stable/declining)
- forecast_velocity: Forecast velocity for planning
- track_sprint_velocity: Track sprint velocity and update metrics
- _calculate_percentile: Calculate percentile values
- _calculate_std_dev: Calculate variance (std dev)

References:
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - Story 10.9: Sprint Progress Tracking (StoryProgress input)
    - Story 10.11: Priority Scoring (module patterns)
"""

from __future__ import annotations

from yolo_developer.agents.sm.progress_types import StoryProgress
from yolo_developer.agents.sm.velocity import (
    _calculate_percentile,
    _calculate_std_dev,
    calculate_sprint_velocity,
    calculate_velocity_metrics,
    forecast_velocity,
    get_velocity_trend,
    track_sprint_velocity,
)
from yolo_developer.agents.sm.velocity_types import (
    DEFAULT_MIN_SPRINTS_FOR_FORECAST,
    DEFAULT_MIN_SPRINTS_FOR_TREND,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_TREND_THRESHOLD,
    VALID_TRENDS,
    SprintVelocity,
    VelocityConfig,
    VelocityForecast,
    VelocityMetrics,
)


class TestVelocityTypes:
    """Tests for velocity tracking types (Task 1)."""

    def test_sprint_velocity_defaults(self) -> None:
        """Test SprintVelocity with required fields."""
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
        assert velocity.completed_at is not None  # Auto-generated

    def test_sprint_velocity_to_dict(self) -> None:
        """Test SprintVelocity serialization."""
        velocity = SprintVelocity(
            sprint_id="sprint-2",
            stories_completed=3,
            points_completed=3.0,
            total_cycle_time_ms=10800000.0,
            avg_cycle_time_ms=3600000.0,
        )
        d = velocity.to_dict()
        assert d["sprint_id"] == "sprint-2"
        assert d["stories_completed"] == 3
        assert d["points_completed"] == 3.0
        assert "completed_at" in d

    def test_velocity_metrics_defaults(self) -> None:
        """Test VelocityMetrics with required fields."""
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
        assert metrics.variance_stories == 0.0  # Default
        assert metrics.variance_points == 0.0  # Default

    def test_velocity_metrics_to_dict(self) -> None:
        """Test VelocityMetrics serialization."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=3,
            trend="improving",
            variance_stories=1.2,
        )
        d = metrics.to_dict()
        assert d["trend"] == "improving"
        assert d["variance_stories"] == 1.2
        assert "calculated_at" in d

    def test_velocity_config_defaults(self) -> None:
        """Test VelocityConfig defaults match constants."""
        config = VelocityConfig()
        assert config.rolling_window == DEFAULT_ROLLING_WINDOW
        assert config.trend_threshold == DEFAULT_TREND_THRESHOLD
        assert config.min_sprints_for_trend == DEFAULT_MIN_SPRINTS_FOR_TREND
        assert config.min_sprints_for_forecast == DEFAULT_MIN_SPRINTS_FOR_FORECAST

    def test_velocity_config_custom(self) -> None:
        """Test VelocityConfig with custom values."""
        config = VelocityConfig(
            rolling_window=3,
            trend_threshold=0.15,
            min_sprints_for_trend=2,
        )
        assert config.rolling_window == 3
        assert config.trend_threshold == 0.15
        assert config.min_sprints_for_trend == 2

    def test_velocity_config_to_dict(self) -> None:
        """Test VelocityConfig serialization."""
        config = VelocityConfig(rolling_window=7)
        d = config.to_dict()
        assert d["rolling_window"] == 7
        assert "trend_threshold" in d

    def test_velocity_forecast_defaults(self) -> None:
        """Test VelocityForecast with required fields."""
        forecast = VelocityForecast(
            expected_stories_next_sprint=5,
            expected_points_next_sprint=5.0,
            confidence=0.85,
            forecast_factors=("based_on_5_sprints", "stable_trend"),
        )
        assert forecast.expected_stories_next_sprint == 5
        assert forecast.confidence == 0.85
        assert len(forecast.forecast_factors) == 2

    def test_velocity_forecast_to_dict(self) -> None:
        """Test VelocityForecast serialization."""
        forecast = VelocityForecast(
            expected_stories_next_sprint=4,
            expected_points_next_sprint=4.0,
            confidence=0.7,
            forecast_factors=("low_data",),
        )
        d = forecast.to_dict()
        assert d["expected_stories_next_sprint"] == 4
        assert d["confidence"] == 0.7
        assert d["forecast_factors"] == ["low_data"]  # Converted to list

    def test_constants_valid(self) -> None:
        """Test velocity constants are valid."""
        assert DEFAULT_ROLLING_WINDOW >= 1
        assert 0.0 <= DEFAULT_TREND_THRESHOLD <= 1.0
        assert DEFAULT_MIN_SPRINTS_FOR_TREND >= 1
        assert DEFAULT_MIN_SPRINTS_FOR_FORECAST >= 1
        assert VALID_TRENDS == frozenset({"improving", "stable", "declining"})


class TestCalculateSprintVelocity:
    """Tests for calculate_sprint_velocity function (Task 5.1)."""

    def test_basic_velocity(self) -> None:
        """Test basic velocity calculation."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-2", title="Story 2", status="completed", duration_ms=7200000.0
            ),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-20260116")
        assert velocity.sprint_id == "sprint-20260116"
        assert velocity.stories_completed == 2
        assert velocity.points_completed == 2.0  # Default 1.0 per story
        assert velocity.total_cycle_time_ms == 10800000.0
        assert velocity.avg_cycle_time_ms == 5400000.0

    def test_empty_stories(self) -> None:
        """Test velocity with no completed stories."""
        velocity = calculate_sprint_velocity([], "sprint-empty")
        assert velocity.stories_completed == 0
        assert velocity.points_completed == 0.0
        assert velocity.total_cycle_time_ms == 0.0
        assert velocity.avg_cycle_time_ms == 0.0

    def test_custom_story_points(self) -> None:
        """Test velocity with custom story points per story."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0,
                story_points=2.0,
            ),
            StoryProgress(
                story_id="1-2", title="Story 2", status="completed", duration_ms=3600000.0,
                story_points=3.0,
            ),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.stories_completed == 2
        assert velocity.points_completed == 5.0  # 2.0 + 3.0

    def test_story_without_duration(self) -> None:
        """Test velocity calculation handles None duration."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(story_id="1-2", title="Story 2", status="completed", duration_ms=None),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.stories_completed == 2
        assert velocity.total_cycle_time_ms == 3600000.0
        assert velocity.avg_cycle_time_ms == 1800000.0  # 3600000 / 2

    def test_single_story(self) -> None:
        """Test velocity with single story."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=5000000.0
            ),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-single")
        assert velocity.stories_completed == 1
        assert velocity.avg_cycle_time_ms == 5000000.0


class TestGetVelocityTrend:
    """Tests for get_velocity_trend function (Task 5.3)."""

    def test_improving_trend(self) -> None:
        """Test detection of improving velocity trend."""
        # Start low, end high - need more data than rolling window to show trend
        # Overall avg = (3+3+3+5+6+7) / 6 = 4.5
        # Recent (last 3): (5+6+7) / 3 = 6.0
        # Change ratio = (6.0 - 4.5) / 4.5 = 0.33 > 0.1 threshold
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s4",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s5",
                stories_completed=6,
                points_completed=6.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s6",
                stories_completed=7,
                points_completed=7.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        config = VelocityConfig(rolling_window=3)
        trend = get_velocity_trend(velocities, config)
        assert trend == "improving"

    def test_declining_trend(self) -> None:
        """Test detection of declining velocity trend."""
        # Start high, end low - need more data than rolling window to show trend
        # Overall avg = (7+7+7+5+4+3) / 6 = 5.5
        # Recent (last 3): (5+4+3) / 3 = 4.0
        # Change ratio = (4.0 - 5.5) / 5.5 = -0.27 < -0.1 threshold
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=7,
                points_completed=7.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=7,
                points_completed=7.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=7,
                points_completed=7.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s4",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s5",
                stories_completed=4,
                points_completed=4.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s6",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        config = VelocityConfig(rolling_window=3)
        trend = get_velocity_trend(velocities, config)
        assert trend == "declining"

    def test_stable_trend(self) -> None:
        """Test detection of stable velocity trend."""
        # All similar values
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        trend = get_velocity_trend(velocities)
        assert trend == "stable"

    def test_insufficient_data(self) -> None:
        """Test trend returns stable with insufficient data."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        trend = get_velocity_trend(velocities)
        assert trend == "stable"

    def test_empty_velocities(self) -> None:
        """Test trend with empty velocities."""
        trend = get_velocity_trend([])
        assert trend == "stable"

    def test_zero_average(self) -> None:
        """Test trend when average is zero."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=0,
                points_completed=0.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=0,
                points_completed=0.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=0,
                points_completed=0.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        trend = get_velocity_trend(velocities)
        assert trend == "stable"

    def test_custom_config(self) -> None:
        """Test trend with custom config."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=6,
                points_completed=6.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        # With min_sprints_for_trend=2, this should work
        config = VelocityConfig(min_sprints_for_trend=2)
        trend = get_velocity_trend(velocities, config)
        # Recent avg (6) vs overall avg (5.5) = +9% (improving with threshold 10%)
        # Actually it's stable since change is below 10%
        assert trend == "stable"


class TestCalculateVelocityMetrics:
    """Tests for calculate_velocity_metrics function (Task 5.2)."""

    def test_basic_metrics(self) -> None:
        """Test basic velocity metrics calculation."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=18000000.0,
                avg_cycle_time_ms=3600000.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=6,
                points_completed=6.0,
                total_cycle_time_ms=21600000.0,
                avg_cycle_time_ms=3600000.0,
            ),
        ]
        metrics = calculate_velocity_metrics(velocities)
        assert metrics.average_stories_per_sprint == 5.5
        assert metrics.average_points_per_sprint == 5.5
        assert metrics.sprints_analyzed == 2

    def test_empty_velocities(self) -> None:
        """Test metrics with no velocities."""
        metrics = calculate_velocity_metrics([])
        assert metrics.average_stories_per_sprint == 0.0
        assert metrics.average_points_per_sprint == 0.0
        assert metrics.sprints_analyzed == 0
        assert metrics.trend == "stable"

    def test_percentile_calculation(self) -> None:
        """Test that percentiles are calculated."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=3000000.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=4000000.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=5000000.0,
            ),
        ]
        metrics = calculate_velocity_metrics(velocities)
        assert metrics.cycle_time_p50_ms > 0
        assert metrics.cycle_time_p90_ms > 0
        assert metrics.cycle_time_p90_ms >= metrics.cycle_time_p50_ms

    def test_variance_calculation(self) -> None:
        """Test that variance is calculated."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=7,
                points_completed=7.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        metrics = calculate_velocity_metrics(velocities)
        assert metrics.variance_stories > 0

    def test_rolling_window(self) -> None:
        """Test that rolling window limits averages."""
        # Create 10 velocities, rolling window default is 5
        velocities = [
            SprintVelocity(
                sprint_id=f"s{i}",
                stories_completed=i,
                points_completed=float(i),
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            )
            for i in range(1, 11)  # 1 through 10
        ]
        config = VelocityConfig(rolling_window=3)
        metrics = calculate_velocity_metrics(velocities, config)
        # Last 3: 8, 9, 10 -> avg = 9
        assert metrics.average_stories_per_sprint == 9.0


class TestForecastVelocity:
    """Tests for forecast_velocity function (Task 5.4)."""

    def test_basic_forecast(self) -> None:
        """Test basic velocity forecast."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.5,
            average_points_per_sprint=5.5,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="stable",
        )
        forecast = forecast_velocity(metrics)
        assert forecast.expected_stories_next_sprint == 6  # Rounded from 5.5
        assert forecast.confidence > 0.0
        assert "stable_trend" in forecast.forecast_factors

    def test_insufficient_data(self) -> None:
        """Test forecast with insufficient data."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=1,  # Less than min required
            trend="stable",
        )
        forecast = forecast_velocity(metrics)
        assert forecast.expected_stories_next_sprint == 0
        assert forecast.confidence == 0.0
        assert "insufficient_data" in forecast.forecast_factors[0]

    def test_improving_trend_boost(self) -> None:
        """Test forecast boost for improving trend."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=10.0,
            average_points_per_sprint=10.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="improving",
        )
        forecast = forecast_velocity(metrics)
        # 10.0 * 1.05 = 10.5 -> rounds to 10 or 11
        assert forecast.expected_stories_next_sprint >= 10
        assert "improving_trend" in forecast.forecast_factors[1]

    def test_declining_trend_reduction(self) -> None:
        """Test forecast reduction for declining trend."""
        metrics = VelocityMetrics(
            average_stories_per_sprint=10.0,
            average_points_per_sprint=10.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="declining",
        )
        forecast = forecast_velocity(metrics)
        # 10.0 * 0.95 = 9.5 -> rounds to 10 or 9
        assert forecast.expected_stories_next_sprint <= 10
        assert "declining_trend" in forecast.forecast_factors[1]

    def test_confidence_scales_with_data(self) -> None:
        """Test that confidence increases with more data."""
        metrics_few = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=3,
            trend="stable",
        )
        metrics_many = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=10,
            trend="stable",
        )
        forecast_few = forecast_velocity(metrics_few)
        forecast_many = forecast_velocity(metrics_many)
        assert forecast_many.confidence >= forecast_few.confidence

    def test_high_variance_penalty(self) -> None:
        """Test that high variance reduces confidence."""
        metrics_low_var = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="stable",
            variance_stories=0.5,
        )
        metrics_high_var = VelocityMetrics(
            average_stories_per_sprint=5.0,
            average_points_per_sprint=5.0,
            average_cycle_time_ms=3600000.0,
            cycle_time_p50_ms=3500000.0,
            cycle_time_p90_ms=5000000.0,
            sprints_analyzed=5,
            trend="stable",
            variance_stories=5.0,  # High variance (100% of mean)
        )
        forecast_low = forecast_velocity(metrics_low_var)
        forecast_high = forecast_velocity(metrics_high_var)
        assert forecast_low.confidence >= forecast_high.confidence


class TestTrackSprintVelocity:
    """Tests for track_sprint_velocity function (Task 5.5)."""

    def test_first_sprint(self) -> None:
        """Test tracking velocity for first sprint."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-2", title="Story 2", status="completed", duration_ms=3600000.0
            ),
        ]
        velocity, metrics = track_sprint_velocity("sprint-1", stories, [])
        assert velocity.stories_completed == 2
        assert metrics.sprints_analyzed == 1

    def test_with_history(self) -> None:
        """Test tracking velocity with existing history."""
        history = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=3600000.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=4,
                points_completed=4.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=3600000.0,
            ),
        ]
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-2", title="Story 2", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-3", title="Story 3", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-4", title="Story 4", status="completed", duration_ms=3600000.0
            ),
            StoryProgress(
                story_id="1-5", title="Story 5", status="completed", duration_ms=3600000.0
            ),
        ]
        velocity, metrics = track_sprint_velocity("sprint-3", stories, history)
        assert velocity.stories_completed == 5
        assert metrics.sprints_analyzed == 3
        # Average: (3 + 4 + 5) / 3 = 4
        assert metrics.average_stories_per_sprint == 4.0

    def test_custom_story_points(self) -> None:
        """Test tracking with custom story points."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0,
                story_points=3.0,
            ),
        ]
        velocity, _ = track_sprint_velocity("sprint-1", stories, [])
        assert velocity.points_completed == 3.0


class TestHelperFunctions:
    """Tests for helper functions (Task 5.6)."""

    def test_percentile_basic(self) -> None:
        """Test basic percentile calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = _calculate_percentile(values, 50.0)
        assert p50 == 3.0

    def test_percentile_empty(self) -> None:
        """Test percentile with empty values."""
        p50 = _calculate_percentile([], 50.0)
        assert p50 == 0.0

    def test_percentile_single_value(self) -> None:
        """Test percentile with single value."""
        p50 = _calculate_percentile([5.0], 50.0)
        assert p50 == 5.0

    def test_percentile_p90(self) -> None:
        """Test 90th percentile calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p90 = _calculate_percentile(values, 90.0)
        assert p90 >= 9.0

    def test_variance_basic(self) -> None:
        """Test basic variance calculation."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        variance = _calculate_std_dev(values)
        # Standard deviation should be around 2.0
        assert 1.5 < variance < 2.5

    def test_variance_empty(self) -> None:
        """Test variance with empty values."""
        variance = _calculate_std_dev([])
        assert variance == 0.0

    def test_variance_single_value(self) -> None:
        """Test variance with single value."""
        variance = _calculate_std_dev([5.0])
        assert variance == 0.0

    def test_variance_identical_values(self) -> None:
        """Test variance with identical values."""
        variance = _calculate_std_dev([5.0, 5.0, 5.0, 5.0])
        assert variance == 0.0


class TestRollingWindowCalculations:
    """Tests for rolling window behavior (Task 5.7)."""

    def test_window_smaller_than_data(self) -> None:
        """Test rolling window smaller than available data."""
        velocities = [
            SprintVelocity(
                sprint_id=f"s{i}",
                stories_completed=i,
                points_completed=float(i),
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            )
            for i in range(1, 11)  # 1 through 10
        ]
        config = VelocityConfig(rolling_window=3)
        metrics = calculate_velocity_metrics(velocities, config)
        # Last 3 sprints: 8, 9, 10 -> avg = 9
        assert metrics.average_stories_per_sprint == 9.0

    def test_window_larger_than_data(self) -> None:
        """Test rolling window larger than available data."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=3,
                points_completed=3.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        config = VelocityConfig(rolling_window=10)
        metrics = calculate_velocity_metrics(velocities, config)
        # Uses all available: (3 + 5) / 2 = 4
        assert metrics.average_stories_per_sprint == 4.0

    def test_window_equals_data(self) -> None:
        """Test rolling window equals available data."""
        velocities = [
            SprintVelocity(
                sprint_id="s1",
                stories_completed=2,
                points_completed=2.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s2",
                stories_completed=4,
                points_completed=4.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
            SprintVelocity(
                sprint_id="s3",
                stories_completed=6,
                points_completed=6.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=0.0,
            ),
        ]
        config = VelocityConfig(rolling_window=3)
        metrics = calculate_velocity_metrics(velocities, config)
        # All 3: (2 + 4 + 6) / 3 = 4
        assert metrics.average_stories_per_sprint == 4.0


class TestEdgeCases:
    """Tests for edge cases (Task 5.6)."""

    def test_all_zero_durations(self) -> None:
        """Test handling of all zero durations."""
        stories = [
            StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=0.0),
            StoryProgress(story_id="1-2", title="Story 2", status="completed", duration_ms=0.0),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.stories_completed == 2
        assert velocity.total_cycle_time_ms == 0.0
        assert velocity.avg_cycle_time_ms == 0.0

    def test_mixed_none_durations(self) -> None:
        """Test handling of mixed None and non-None durations."""
        stories = [
            StoryProgress(
                story_id="1-1", title="Story 1", status="completed", duration_ms=1000000.0
            ),
            StoryProgress(story_id="1-2", title="Story 2", status="completed", duration_ms=None),
            StoryProgress(
                story_id="1-3", title="Story 3", status="completed", duration_ms=2000000.0
            ),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.stories_completed == 3
        assert velocity.total_cycle_time_ms == 3000000.0
        assert velocity.avg_cycle_time_ms == 1000000.0  # 3000000 / 3

    def test_large_number_of_sprints(self) -> None:
        """Test handling of large number of sprints."""
        velocities = [
            SprintVelocity(
                sprint_id=f"s{i}",
                stories_completed=5,
                points_completed=5.0,
                total_cycle_time_ms=0.0,
                avg_cycle_time_ms=3600000.0,
            )
            for i in range(100)
        ]
        metrics = calculate_velocity_metrics(velocities)
        assert metrics.sprints_analyzed == 100
        assert metrics.average_stories_per_sprint == 5.0

    def test_very_small_values(self) -> None:
        """Test handling of very small duration values."""
        stories = [
            StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=0.001),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.total_cycle_time_ms == 0.001
        assert velocity.avg_cycle_time_ms == 0.001

    def test_very_large_values(self) -> None:
        """Test handling of very large duration values."""
        stories = [
            StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=1e15),
        ]
        velocity = calculate_sprint_velocity(stories, "sprint-1")
        assert velocity.total_cycle_time_ms == 1e15
