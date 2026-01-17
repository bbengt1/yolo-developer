"""Velocity tracking functions for sprint planning (Story 10.12).

This module provides functions for tracking and analyzing velocity metrics:

- calculate_sprint_velocity: Calculate velocity for a completed sprint
- calculate_velocity_metrics: Aggregate velocity across multiple sprints
- get_velocity_trend: Determine velocity trend (improving/stable/declining)
- forecast_velocity: Forecast velocity for planning
- track_sprint_velocity: Track sprint velocity and update metrics

Example:
    >>> from yolo_developer.agents.sm.velocity import (
    ...     calculate_sprint_velocity,
    ...     calculate_velocity_metrics,
    ...     get_velocity_trend,
    ...     forecast_velocity,
    ... )
    >>> from yolo_developer.agents.sm.progress_types import StoryProgress
    >>>
    >>> # Calculate velocity for completed stories
    >>> stories = [
    ...     StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0),
    ...     StoryProgress(story_id="1-2", title="Story 2", status="completed", duration_ms=7200000.0),
    ... ]
    >>> velocity = calculate_sprint_velocity(stories, "sprint-20260116")
    >>> velocity.stories_completed
    2

References:
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - Story 10.9: Sprint Progress Tracking (StoryProgress input)
    - Story 10.11: Priority Scoring (module patterns)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import structlog

from yolo_developer.agents.sm.velocity_types import (
    CONFIDENCE_DECIMAL_PLACES,
    SprintVelocity,
    VelocityConfig,
    VelocityForecast,
    VelocityMetrics,
    VelocityTrend,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.progress_types import StoryProgress

logger = structlog.get_logger(__name__)


# =============================================================================
# Core Velocity Functions
# =============================================================================


def calculate_sprint_velocity(
    completed_stories: Sequence[StoryProgress],
    sprint_id: str,
) -> SprintVelocity:
    """Calculate velocity for a completed sprint.

    Computes velocity metrics from completed stories, including story count,
    points completed (sum of individual story_points), and cycle time statistics.

    Args:
        completed_stories: Sequence of completed StoryProgress objects.
        sprint_id: Unique identifier for the sprint.

    Returns:
        SprintVelocity with calculated metrics.

    Example:
        >>> from yolo_developer.agents.sm.progress_types import StoryProgress
        >>> stories = [
        ...     StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0),
        ... ]
        >>> velocity = calculate_sprint_velocity(stories, "sprint-20260116")
        >>> velocity.stories_completed
        1

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
    """
    logger.debug(
        "calculating_sprint_velocity",
        sprint_id=sprint_id,
        story_count=len(completed_stories),
    )

    if not completed_stories:
        logger.info(
            "empty_sprint_velocity",
            sprint_id=sprint_id,
            message="No completed stories for velocity calculation",
        )
        return SprintVelocity(
            sprint_id=sprint_id,
            stories_completed=0,
            points_completed=0.0,
            total_cycle_time_ms=0.0,
            avg_cycle_time_ms=0.0,
        )

    stories_count = len(completed_stories)
    # Sum individual story points (each story has story_points field, default 1.0)
    points_completed = sum(story.story_points for story in completed_stories)

    # Calculate cycle time from story durations
    total_cycle_time = sum(
        story.duration_ms if story.duration_ms is not None else 0.0 for story in completed_stories
    )
    avg_cycle_time = total_cycle_time / stories_count if stories_count > 0 else 0.0

    logger.info(
        "sprint_velocity_calculated",
        sprint_id=sprint_id,
        stories_completed=stories_count,
        points_completed=points_completed,
        total_cycle_time_ms=total_cycle_time,
        avg_cycle_time_ms=avg_cycle_time,
    )

    return SprintVelocity(
        sprint_id=sprint_id,
        stories_completed=stories_count,
        points_completed=points_completed,
        total_cycle_time_ms=total_cycle_time,
        avg_cycle_time_ms=avg_cycle_time,
    )


def get_velocity_trend(
    velocities: Sequence[SprintVelocity],
    config: VelocityConfig | None = None,
) -> VelocityTrend:
    """Determine velocity trend: improving, stable, or declining.

    Compares recent sprints (within rolling window) to overall average
    and determines if velocity is improving, declining, or stable.

    Args:
        velocities: Sequence of SprintVelocity objects in chronological order.
        config: Configuration for trend detection (uses defaults if None).

    Returns:
        VelocityTrend indicating trend direction.

    Example:
        >>> velocities = [
        ...     SprintVelocity(sprint_id="s1", stories_completed=3, points_completed=3.0,
        ...                    total_cycle_time_ms=0.0, avg_cycle_time_ms=0.0),
        ...     SprintVelocity(sprint_id="s2", stories_completed=4, points_completed=4.0,
        ...                    total_cycle_time_ms=0.0, avg_cycle_time_ms=0.0),
        ...     SprintVelocity(sprint_id="s3", stories_completed=5, points_completed=5.0,
        ...                    total_cycle_time_ms=0.0, avg_cycle_time_ms=0.0),
        ... ]
        >>> get_velocity_trend(velocities)
        'improving'

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
    """
    if config is None:
        config = VelocityConfig()

    logger.debug(
        "calculating_velocity_trend",
        sprint_count=len(velocities),
        rolling_window=config.rolling_window,
        trend_threshold=config.trend_threshold,
    )

    # Not enough data for trend analysis
    if len(velocities) < config.min_sprints_for_trend:
        logger.debug(
            "insufficient_data_for_trend",
            sprint_count=len(velocities),
            min_required=config.min_sprints_for_trend,
        )
        return "stable"

    # Calculate overall average
    all_stories = [v.stories_completed for v in velocities]
    all_avg = sum(all_stories) / len(all_stories) if all_stories else 0.0

    # Avoid division by zero
    if all_avg == 0:
        return "stable"

    # Calculate recent average (within rolling window)
    window_size = min(config.rolling_window, len(velocities))
    recent_velocities = list(velocities)[-window_size:]
    recent_avg = sum(v.stories_completed for v in recent_velocities) / len(recent_velocities)

    # Calculate change ratio
    change_ratio = (recent_avg - all_avg) / all_avg

    logger.debug(
        "trend_calculation",
        all_avg=all_avg,
        recent_avg=recent_avg,
        change_ratio=change_ratio,
        threshold=config.trend_threshold,
    )

    if change_ratio > config.trend_threshold:
        trend: VelocityTrend = "improving"
    elif change_ratio < -config.trend_threshold:
        trend = "declining"
    else:
        trend = "stable"

    logger.info(
        "velocity_trend_determined",
        trend=trend,
        change_ratio=change_ratio,
        all_avg=all_avg,
        recent_avg=recent_avg,
    )

    return trend


def _calculate_percentile(values: Sequence[float], percentile: float) -> float:
    """Calculate percentile value from a sequence.

    Args:
        values: Sequence of numeric values.
        percentile: Percentile to calculate (0.0-100.0).

    Returns:
        Percentile value, or 0.0 if no values.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Calculate percentile index
    index = (percentile / 100.0) * (n - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, n - 1)
    fraction = index - lower_idx

    # Linear interpolation
    return sorted_values[lower_idx] + fraction * (
        sorted_values[upper_idx] - sorted_values[lower_idx]
    )


def _calculate_std_dev(values: Sequence[float]) -> float:
    """Calculate standard deviation of values.

    Uses population standard deviation (divides by N, not N-1).

    Args:
        values: Sequence of numeric values.

    Returns:
        Standard deviation, or 0.0 if insufficient values.
    """
    if not values or len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def calculate_velocity_metrics(
    sprint_velocities: Sequence[SprintVelocity],
    config: VelocityConfig | None = None,
) -> VelocityMetrics:
    """Calculate aggregated velocity metrics across sprints.

    Computes rolling averages, percentiles, variance, and trend for
    velocity tracking and planning decisions.

    Args:
        sprint_velocities: Sequence of SprintVelocity objects.
        config: Configuration for metric calculations (uses defaults if None).

    Returns:
        VelocityMetrics with aggregated data.

    Example:
        >>> velocities = [
        ...     SprintVelocity(sprint_id="s1", stories_completed=5, points_completed=5.0,
        ...                    total_cycle_time_ms=18000000.0, avg_cycle_time_ms=3600000.0),
        ...     SprintVelocity(sprint_id="s2", stories_completed=6, points_completed=6.0,
        ...                    total_cycle_time_ms=21600000.0, avg_cycle_time_ms=3600000.0),
        ... ]
        >>> metrics = calculate_velocity_metrics(velocities)
        >>> metrics.average_stories_per_sprint
        5.5

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
    """
    if config is None:
        config = VelocityConfig()

    logger.debug(
        "calculating_velocity_metrics",
        sprint_count=len(sprint_velocities),
        rolling_window=config.rolling_window,
    )

    if not sprint_velocities:
        logger.info("empty_velocity_metrics", message="No sprints for metrics calculation")
        return VelocityMetrics(
            average_stories_per_sprint=0.0,
            average_points_per_sprint=0.0,
            average_cycle_time_ms=0.0,
            cycle_time_p50_ms=0.0,
            cycle_time_p90_ms=0.0,
            sprints_analyzed=0,
            trend="stable",
            variance_stories=0.0,
            variance_points=0.0,
        )

    # Use rolling window for averages
    window_size = min(config.rolling_window, len(sprint_velocities))
    recent_velocities = list(sprint_velocities)[-window_size:]

    # Calculate averages
    stories_values = [v.stories_completed for v in recent_velocities]
    points_values = [v.points_completed for v in recent_velocities]
    cycle_time_values = [v.avg_cycle_time_ms for v in recent_velocities if v.avg_cycle_time_ms > 0]

    avg_stories = sum(stories_values) / len(stories_values) if stories_values else 0.0
    avg_points = sum(points_values) / len(points_values) if points_values else 0.0
    avg_cycle_time = sum(cycle_time_values) / len(cycle_time_values) if cycle_time_values else 0.0

    # Calculate percentiles from all sprints with cycle time data
    all_cycle_times = [v.avg_cycle_time_ms for v in sprint_velocities if v.avg_cycle_time_ms > 0]
    p50 = _calculate_percentile(all_cycle_times, 50.0)
    p90 = _calculate_percentile(all_cycle_times, 90.0)

    # Calculate variance
    variance_stories = _calculate_std_dev([float(s) for s in stories_values])
    variance_points = _calculate_std_dev(points_values)

    # Determine trend
    trend = get_velocity_trend(sprint_velocities, config)

    logger.info(
        "velocity_metrics_calculated",
        sprints_analyzed=len(sprint_velocities),
        avg_stories=avg_stories,
        avg_points=avg_points,
        avg_cycle_time_ms=avg_cycle_time,
        trend=trend,
    )

    return VelocityMetrics(
        average_stories_per_sprint=avg_stories,
        average_points_per_sprint=avg_points,
        average_cycle_time_ms=avg_cycle_time,
        cycle_time_p50_ms=p50,
        cycle_time_p90_ms=p90,
        sprints_analyzed=len(sprint_velocities),
        trend=trend,
        variance_stories=variance_stories,
        variance_points=variance_points,
    )


def forecast_velocity(
    metrics: VelocityMetrics,
    config: VelocityConfig | None = None,
) -> VelocityForecast:
    """Forecast velocity for the next sprint.

    Generates a prediction based on historical velocity metrics,
    with confidence adjusted by data quality and variance.

    Args:
        metrics: Aggregated velocity metrics from historical data.
        config: Configuration for forecasting (uses defaults if None).

    Returns:
        VelocityForecast with predictions and confidence.

    Example:
        >>> metrics = VelocityMetrics(
        ...     average_stories_per_sprint=5.5,
        ...     average_points_per_sprint=5.5,
        ...     average_cycle_time_ms=3600000.0,
        ...     cycle_time_p50_ms=3500000.0,
        ...     cycle_time_p90_ms=5000000.0,
        ...     sprints_analyzed=5,
        ...     trend="stable",
        ... )
        >>> forecast = forecast_velocity(metrics)
        >>> forecast.expected_stories_next_sprint
        5

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
    """
    if config is None:
        config = VelocityConfig()

    logger.debug(
        "generating_velocity_forecast",
        sprints_analyzed=metrics.sprints_analyzed,
        trend=metrics.trend,
    )

    factors: list[str] = []

    # Handle insufficient data
    if metrics.sprints_analyzed < config.min_sprints_for_forecast:
        logger.info(
            "insufficient_data_for_forecast",
            sprints_analyzed=metrics.sprints_analyzed,
            min_required=config.min_sprints_for_forecast,
        )
        factors.append(f"insufficient_data_{metrics.sprints_analyzed}_sprints")
        return VelocityForecast(
            expected_stories_next_sprint=0,
            expected_points_next_sprint=0.0,
            confidence=0.0,
            forecast_factors=tuple(factors),
        )

    # Base prediction from averages
    expected_stories = round(metrics.average_stories_per_sprint)
    expected_points = metrics.average_points_per_sprint

    factors.append(f"based_on_{metrics.sprints_analyzed}_sprints")

    # Adjust for trend
    if metrics.trend == "improving":
        # Add a small boost for improving trend
        expected_stories = round(metrics.average_stories_per_sprint * 1.05)
        expected_points = metrics.average_points_per_sprint * 1.05
        factors.append("improving_trend_+5%")
    elif metrics.trend == "declining":
        # Reduce for declining trend
        expected_stories = round(metrics.average_stories_per_sprint * 0.95)
        expected_points = metrics.average_points_per_sprint * 0.95
        factors.append("declining_trend_-5%")
    else:
        factors.append("stable_trend")

    # Calculate confidence based on data quality
    base_confidence = min(metrics.sprints_analyzed / 10.0, 1.0)  # More data = more confidence

    # Reduce confidence for high variance
    if metrics.average_stories_per_sprint > 0:
        coefficient_of_variation = metrics.variance_stories / metrics.average_stories_per_sprint
        variance_penalty = min(coefficient_of_variation * 0.5, 0.3)  # Max 30% penalty
        base_confidence = max(base_confidence - variance_penalty, 0.1)
        if variance_penalty > 0.1:
            factors.append("high_variance_penalty")

    # Ensure non-negative predictions
    expected_stories = max(expected_stories, 0)
    expected_points = max(expected_points, 0.0)

    logger.info(
        "velocity_forecast_generated",
        expected_stories=expected_stories,
        expected_points=expected_points,
        confidence=base_confidence,
        factors=factors,
    )

    return VelocityForecast(
        expected_stories_next_sprint=expected_stories,
        expected_points_next_sprint=expected_points,
        confidence=round(base_confidence, CONFIDENCE_DECIMAL_PLACES),
        forecast_factors=tuple(factors),
    )


def track_sprint_velocity(
    sprint_id: str,
    completed_stories: Sequence[StoryProgress],
    history: Sequence[SprintVelocity],
    config: VelocityConfig | None = None,
) -> tuple[SprintVelocity, VelocityMetrics]:
    """Track sprint velocity and update aggregated metrics.

    Calculates velocity for the current sprint, adds it to history,
    and recalculates aggregated metrics. Story points are taken from
    individual story's story_points field.

    Args:
        sprint_id: Unique identifier for the current sprint.
        completed_stories: Sequence of completed StoryProgress objects.
        history: Sequence of previous SprintVelocity records.
        config: Configuration for velocity calculations (uses defaults if None).

    Returns:
        Tuple of (current SprintVelocity, updated VelocityMetrics).

    Example:
        >>> from yolo_developer.agents.sm.progress_types import StoryProgress
        >>> stories = [
        ...     StoryProgress(story_id="1-1", title="Story 1", status="completed", duration_ms=3600000.0),
        ... ]
        >>> velocity, metrics = track_sprint_velocity("sprint-20260116", stories, [])
        >>> velocity.stories_completed
        1

    References:
        - FR66: SM Agent can track burn-down velocity and cycle time metrics
    """
    if config is None:
        config = VelocityConfig()

    logger.debug(
        "tracking_sprint_velocity",
        sprint_id=sprint_id,
        completed_stories=len(completed_stories),
        history_size=len(history),
    )

    # Calculate current sprint velocity
    current_velocity = calculate_sprint_velocity(
        completed_stories,
        sprint_id,
    )

    # Combine with history for metrics calculation
    all_velocities = [*list(history), current_velocity]

    # Calculate updated metrics
    updated_metrics = calculate_velocity_metrics(all_velocities, config)

    logger.info(
        "sprint_velocity_tracked",
        sprint_id=sprint_id,
        stories_completed=current_velocity.stories_completed,
        total_sprints=len(all_velocities),
        trend=updated_metrics.trend,
    )

    return current_velocity, updated_metrics
