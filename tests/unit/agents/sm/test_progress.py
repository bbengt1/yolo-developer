"""Unit tests for sprint progress tracking functions (Story 10.9).

Tests the sprint progress tracking module:
- Story categorization by status
- Progress percentage calculation
- Completion estimation
- Query helpers for CLI/SDK
- Main track_progress() function

References:
    - FR16: System can track sprint progress and completion status
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.agents.sm.progress import (
    _build_completion_estimate,
    _build_progress_snapshot,
    _calculate_average_story_duration,
    _calculate_estimation_confidence,
    _calculate_progress_percentage,
    _calculate_story_duration,
    _categorize_stories_by_status,
    _get_current_agent,
    _get_current_story,
    _get_estimation_factors,
    get_progress_for_display,
    get_progress_summary,
    get_stories_by_status,
    track_progress,
)
from yolo_developer.agents.sm.progress_types import (
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_story_progress(
    story_id: str,
    title: str,
    status: str = "backlog",
    duration_ms: float | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> StoryProgress:
    """Create a StoryProgress for testing."""
    return StoryProgress(
        story_id=story_id,
        title=title,
        status=status,  # type: ignore[arg-type]
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
    )


def make_sprint_plan(
    stories: list[dict[str, Any]], sprint_id: str = "sprint-test"
) -> dict[str, Any]:
    """Create a sprint plan dict for testing."""
    return {
        "sprint_id": sprint_id,
        "stories": [
            {
                "story_id": s.get("story_id", f"story-{i}"),
                "title": s.get("title", f"Story {i}"),
                "status": s.get("status", "backlog"),
                "started_at": s.get("started_at"),
                "completed_at": s.get("completed_at"),
                "duration_ms": s.get("duration_ms"),
            }
            for i, s in enumerate(stories)
        ],
    }


def make_state(current_agent: str = "dev") -> dict[str, Any]:
    """Create a minimal state dict for testing."""
    return {
        "current_agent": current_agent,
        "messages": [],
        "handoff_context": None,
        "decisions": [],
    }


# =============================================================================
# Story Categorization Tests
# =============================================================================


class TestCategorizeStoriesByStatus:
    """Tests for _categorize_stories_by_status()."""

    def test_empty_sprint_plan(self) -> None:
        """Empty sprint plan returns empty categories."""
        result = _categorize_stories_by_status(None)
        assert result["completed"] == []
        assert result["in_progress"] == []
        assert result["remaining"] == []
        assert result["blocked"] == []

    def test_all_backlog(self) -> None:
        """All stories in backlog."""
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "backlog"},
                {"story_id": "1-2", "title": "Second", "status": "backlog"},
            ]
        )
        result = _categorize_stories_by_status(plan)
        assert len(result["remaining"]) == 2
        assert len(result["completed"]) == 0
        assert len(result["in_progress"]) == 0
        assert len(result["blocked"]) == 0

    def test_mixed_statuses(self) -> None:
        """Stories with mixed statuses are categorized correctly."""
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "completed", "duration_ms": 1000.0},
                {"story_id": "1-2", "title": "Second", "status": "in_progress"},
                {"story_id": "1-3", "title": "Third", "status": "backlog"},
                {"story_id": "1-4", "title": "Fourth", "status": "blocked"},
                {"story_id": "1-5", "title": "Fifth", "status": "failed"},
            ]
        )
        result = _categorize_stories_by_status(plan)
        assert len(result["completed"]) == 1
        assert len(result["in_progress"]) == 1
        assert len(result["remaining"]) == 1
        assert len(result["blocked"]) == 2  # blocked + failed

    def test_completed_stories_have_duration(self) -> None:
        """Completed stories retain duration information."""
        plan = make_sprint_plan(
            [
                {
                    "story_id": "1-1",
                    "title": "First",
                    "status": "completed",
                    "duration_ms": 3600000.0,
                },
            ]
        )
        result = _categorize_stories_by_status(plan)
        assert len(result["completed"]) == 1
        assert result["completed"][0].duration_ms == 3600000.0

    def test_unknown_status_treated_as_backlog(self) -> None:
        """Unknown/invalid status values are treated as backlog.

        This ensures graceful handling of unexpected status values
        that might come from external data sources.
        """
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "unknown_status"},
                {"story_id": "1-2", "title": "Second", "status": "invalid"},
                {"story_id": "1-3", "title": "Third", "status": ""},
                {"story_id": "1-4", "title": "Fourth"},  # Missing status entirely
            ]
        )
        result = _categorize_stories_by_status(plan)
        # All should be in "remaining" (treated as backlog)
        assert len(result["remaining"]) == 4
        assert len(result["completed"]) == 0
        assert len(result["in_progress"]) == 0
        assert len(result["blocked"]) == 0
        # Verify all have "backlog" status in the StoryProgress
        for story in result["remaining"]:
            assert story.status == "backlog"


class TestGetCurrentStory:
    """Tests for _get_current_story()."""

    def test_no_sprint_plan(self) -> None:
        """Returns None when no sprint plan."""
        assert _get_current_story(None) is None

    def test_no_in_progress_story(self) -> None:
        """Returns None when no story is in progress."""
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "completed"},
                {"story_id": "1-2", "title": "Second", "status": "backlog"},
            ]
        )
        assert _get_current_story(plan) is None

    def test_finds_in_progress_story(self) -> None:
        """Finds the in-progress story."""
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "completed"},
                {"story_id": "1-2", "title": "Second", "status": "in_progress"},
                {"story_id": "1-3", "title": "Third", "status": "backlog"},
            ]
        )
        assert _get_current_story(plan) == "1-2"


class TestGetCurrentAgent:
    """Tests for _get_current_agent()."""

    def test_returns_current_agent(self) -> None:
        """Returns current agent from state."""
        state = make_state("architect")
        assert _get_current_agent(state) == "architect"

    def test_returns_none_if_missing(self) -> None:
        """Returns None if current_agent not in state."""
        state: dict[str, Any] = {"messages": []}
        assert _get_current_agent(state) is None


# =============================================================================
# Progress Calculation Tests
# =============================================================================


class TestCalculateProgressPercentage:
    """Tests for _calculate_progress_percentage()."""

    def test_zero_total(self) -> None:
        """Zero total stories returns 0%."""
        assert _calculate_progress_percentage(0, 0) == 0.0

    def test_zero_completed(self) -> None:
        """Zero completed returns 0%."""
        assert _calculate_progress_percentage(0, 10) == 0.0

    def test_half_completed(self) -> None:
        """Half completed returns 50%."""
        assert _calculate_progress_percentage(5, 10) == 50.0

    def test_all_completed(self) -> None:
        """All completed returns 100%."""
        assert _calculate_progress_percentage(10, 10) == 100.0

    def test_one_of_three(self) -> None:
        """One of three returns 33.33...%."""
        result = _calculate_progress_percentage(1, 3)
        assert 33.3 <= result <= 33.4


class TestCalculateStoryDuration:
    """Tests for _calculate_story_duration()."""

    def test_missing_start(self) -> None:
        """Returns None if no start time."""
        assert _calculate_story_duration(None, "2026-01-16T12:00:00+00:00") is None

    def test_missing_end(self) -> None:
        """Returns None if no end time."""
        assert _calculate_story_duration("2026-01-16T10:00:00+00:00", None) is None

    def test_calculates_duration(self) -> None:
        """Calculates duration in milliseconds."""
        # 2 hours = 7,200,000 ms
        result = _calculate_story_duration(
            "2026-01-16T10:00:00+00:00",
            "2026-01-16T12:00:00+00:00",
        )
        assert result == 7200000.0

    def test_handles_different_timezones(self) -> None:
        """Handles timestamps with different timezones."""
        result = _calculate_story_duration(
            "2026-01-16T10:00:00+00:00",
            "2026-01-16T13:00:00+03:00",  # Same instant as 10:00 UTC
        )
        assert result == 0.0


class TestBuildProgressSnapshot:
    """Tests for _build_progress_snapshot()."""

    def test_creates_snapshot(self) -> None:
        """Creates a valid snapshot from categorized stories."""
        completed = [make_story_progress("1-1", "First", "completed")]
        in_progress = [make_story_progress("1-2", "Second", "in_progress")]
        remaining = [make_story_progress("1-3", "Third", "backlog")]
        blocked = [make_story_progress("1-4", "Fourth", "blocked")]

        snapshot = _build_progress_snapshot(
            sprint_id="sprint-test",
            completed=completed,
            in_progress=in_progress,
            remaining=remaining,
            blocked=blocked,
            current_agent="dev",
        )

        assert snapshot.sprint_id == "sprint-test"
        assert snapshot.total_stories == 4
        assert snapshot.stories_completed == 1
        assert snapshot.stories_in_progress == 1
        assert snapshot.stories_remaining == 1
        assert snapshot.stories_blocked == 1
        assert snapshot.progress_percentage == 25.0
        assert snapshot.current_agent == "dev"

    def test_empty_sprint(self) -> None:
        """Handles empty sprint."""
        snapshot = _build_progress_snapshot(
            sprint_id="sprint-empty",
            completed=[],
            in_progress=[],
            remaining=[],
            blocked=[],
            current_agent=None,
        )
        assert snapshot.total_stories == 0
        assert snapshot.progress_percentage == 0.0


# =============================================================================
# Completion Estimation Tests
# =============================================================================


class TestCalculateAverageStoryDuration:
    """Tests for _calculate_average_story_duration()."""

    def test_no_completed_stories(self) -> None:
        """Returns None with no completed stories."""
        assert _calculate_average_story_duration([]) is None

    def test_no_durations(self) -> None:
        """Returns None when stories have no duration data."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=None),
            make_story_progress("1-2", "Second", "completed", duration_ms=None),
        ]
        assert _calculate_average_story_duration(stories) is None

    def test_calculates_average(self) -> None:
        """Calculates average of durations."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=1000.0),
            make_story_progress("1-2", "Second", "completed", duration_ms=2000.0),
            make_story_progress("1-3", "Third", "completed", duration_ms=3000.0),
        ]
        assert _calculate_average_story_duration(stories) == 2000.0

    def test_ignores_none_durations(self) -> None:
        """Ignores stories without duration."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=1000.0),
            make_story_progress("1-2", "Second", "completed", duration_ms=None),
            make_story_progress("1-3", "Third", "completed", duration_ms=3000.0),
        ]
        assert _calculate_average_story_duration(stories) == 2000.0


class TestCalculateEstimationConfidence:
    """Tests for _calculate_estimation_confidence()."""

    def test_single_story_low_confidence(self) -> None:
        """Single story gives low confidence."""
        stories = [make_story_progress("1-1", "First", "completed", duration_ms=1000.0)]
        confidence = _calculate_estimation_confidence(stories)
        assert confidence == 0.3  # Minimal data

    def test_two_stories_with_variance(self) -> None:
        """Two stories with variance."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=1000.0),
            make_story_progress("1-2", "Second", "completed", duration_ms=3000.0),
        ]
        confidence = _calculate_estimation_confidence(stories)
        assert 0.3 < confidence < 0.6  # Low-medium confidence

    def test_many_stories_consistent_high_confidence(self) -> None:
        """Many consistent stories give higher confidence."""
        # 10 stories with very similar durations
        stories = [
            make_story_progress(f"1-{i}", f"Story {i}", "completed", duration_ms=1000.0 + i)
            for i in range(10)
        ]
        confidence = _calculate_estimation_confidence(stories)
        assert confidence > 0.7  # High confidence

    def test_no_duration_data(self) -> None:
        """No duration data gives minimal confidence."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=None),
        ]
        confidence = _calculate_estimation_confidence(stories)
        assert confidence == 0.3


class TestGetEstimationFactors:
    """Tests for _get_estimation_factors()."""

    def test_includes_sample_size(self) -> None:
        """Factors include sample size."""
        factors = _get_estimation_factors(5, 2000.0, 0.8, 0.7)
        assert any("5" in f and "completed" in f for f in factors)

    def test_includes_average_duration(self) -> None:
        """Factors include average duration."""
        factors = _get_estimation_factors(5, 2000.0, 0.8, 0.7)
        assert any("2000" in f and "duration" in f for f in factors)

    def test_flags_low_confidence(self) -> None:
        """Flags low confidence estimates."""
        factors = _get_estimation_factors(2, 1000.0, 0.3, 0.7)
        assert any("low_confidence" in f for f in factors)

    def test_no_flag_for_high_confidence(self) -> None:
        """No low confidence flag when above threshold."""
        factors = _get_estimation_factors(10, 1000.0, 0.9, 0.7)
        assert not any("low_confidence" in f for f in factors)


class TestBuildCompletionEstimate:
    """Tests for _build_completion_estimate()."""

    def test_no_completed_stories(self) -> None:
        """Returns insufficient data estimate with no completed stories."""
        estimate = _build_completion_estimate([], 5, 0.7)
        assert estimate.estimated_completion_time is None
        assert estimate.estimated_remaining_ms is None
        assert estimate.confidence == 0.0
        assert "insufficient_data" in estimate.factors

    def test_no_remaining_stories(self) -> None:
        """Returns insufficient data when no remaining work."""
        stories = [make_story_progress("1-1", "First", "completed", duration_ms=1000.0)]
        estimate = _build_completion_estimate(stories, 0, 0.7)
        assert estimate.estimated_completion_time is None
        assert "insufficient_data" in estimate.factors

    def test_calculates_estimate(self) -> None:
        """Calculates completion estimate from historical data."""
        stories = [
            make_story_progress("1-1", "First", "completed", duration_ms=3600000.0),  # 1 hour
            make_story_progress("1-2", "Second", "completed", duration_ms=3600000.0),
        ]
        estimate = _build_completion_estimate(stories, 3, 0.7)  # 3 remaining

        assert estimate.estimated_completion_time is not None
        assert estimate.estimated_remaining_ms == 10800000.0  # 3 hours
        assert estimate.confidence > 0


# =============================================================================
# Query Helper Tests
# =============================================================================


class TestGetProgressSummary:
    """Tests for get_progress_summary()."""

    def test_returns_string(self) -> None:
        """Returns a human-readable summary string."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=10,
            stories_completed=5,
            stories_in_progress=2,
            stories_remaining=2,
            stories_blocked=1,
            current_story="5-3",
            current_agent="dev",
            progress_percentage=50.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        summary = get_progress_summary(progress)
        assert isinstance(summary, str)
        assert "50.0%" in summary
        assert "5" in summary  # completed count
        assert "10" in summary  # total count

    def test_includes_current_story(self) -> None:
        """Summary includes current story when available."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=5,
            stories_completed=2,
            stories_in_progress=1,
            stories_remaining=2,
            stories_blocked=0,
            current_story="3-1-feature",
            current_agent="dev",
            progress_percentage=40.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        summary = get_progress_summary(progress)
        assert "3-1-feature" in summary


class TestGetProgressForDisplay:
    """Tests for get_progress_for_display()."""

    def test_returns_dict(self) -> None:
        """Returns a dictionary formatted for display."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=5,
            stories_completed=2,
            stories_in_progress=1,
            stories_remaining=1,
            stories_blocked=1,
            current_story="3-1",
            current_agent="dev",
            progress_percentage=40.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        result = get_progress_for_display(progress)
        assert isinstance(result, dict)
        assert "sprint_id" in result
        assert "progress_percentage" in result
        assert "current_story" in result

    def test_includes_estimate_when_available(self) -> None:
        """Includes completion estimate when available."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=5,
            stories_completed=3,
            stories_in_progress=1,
            stories_remaining=1,
            stories_blocked=0,
            current_story="4-1",
            current_agent="dev",
            progress_percentage=60.0,
        )
        estimate = CompletionEstimate(
            estimated_completion_time="2026-01-16T15:00:00+00:00",
            estimated_remaining_ms=7200000.0,
            confidence=0.8,
            factors=("based_on_3_completed_stories",),
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=estimate,
        )

        result = get_progress_for_display(progress)
        assert "estimated_completion_time" in result
        assert result["confidence"] == 0.8


class TestGetStoriesByStatus:
    """Tests for get_stories_by_status()."""

    def test_filters_completed(self) -> None:
        """Filters to completed stories."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=3,
            stories_completed=1,
            stories_in_progress=1,
            stories_remaining=1,
            stories_blocked=0,
            current_story="2-1",
            current_agent="dev",
            progress_percentage=33.3,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(make_story_progress("1-1", "First", "completed"),),
            in_progress_stories=(make_story_progress("2-1", "Second", "in_progress"),),
            remaining_stories=(make_story_progress("3-1", "Third", "backlog"),),
            blocked_stories=(),
            completion_estimate=None,
        )

        result = get_stories_by_status(progress, "completed")
        assert len(result) == 1
        assert result[0].story_id == "1-1"

    def test_filters_in_progress(self) -> None:
        """Filters to in-progress stories."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=2,
            stories_completed=0,
            stories_in_progress=2,
            stories_remaining=0,
            stories_blocked=0,
            current_story="1-1",
            current_agent="dev",
            progress_percentage=0.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(
                make_story_progress("1-1", "First", "in_progress"),
                make_story_progress("1-2", "Second", "in_progress"),
            ),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        result = get_stories_by_status(progress, "in_progress")
        assert len(result) == 2


# =============================================================================
# Main Function Tests
# =============================================================================


class TestTrackProgress:
    """Tests for track_progress() main function."""

    @pytest.mark.asyncio
    async def test_no_sprint_plan(self) -> None:
        """Returns progress with empty data when no sprint plan."""
        state = make_state()
        result = await track_progress(state, None)

        assert result.snapshot.total_stories == 0
        assert result.snapshot.progress_percentage == 0.0
        assert result.completion_estimate is None

    @pytest.mark.asyncio
    async def test_with_sprint_plan(self) -> None:
        """Tracks progress from sprint plan."""
        state = make_state("dev")
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "completed", "duration_ms": 1000.0},
                {"story_id": "1-2", "title": "Second", "status": "in_progress"},
                {"story_id": "1-3", "title": "Third", "status": "backlog"},
            ]
        )

        result = await track_progress(state, plan)

        assert result.snapshot.total_stories == 3
        assert result.snapshot.stories_completed == 1
        assert result.snapshot.stories_in_progress == 1
        assert result.snapshot.stories_remaining == 1
        assert result.snapshot.current_agent == "dev"

    @pytest.mark.asyncio
    async def test_respects_config_no_estimates(self) -> None:
        """Respects config to disable estimates."""
        state = make_state()
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "completed", "duration_ms": 1000.0},
                {"story_id": "1-2", "title": "Second", "status": "backlog"},
            ]
        )
        config = ProgressConfig(include_estimates=False)

        result = await track_progress(state, plan, config)

        assert result.completion_estimate is None

    @pytest.mark.asyncio
    async def test_generates_estimate_when_enabled(self) -> None:
        """Generates completion estimate when configured."""
        state = make_state()
        plan = make_sprint_plan(
            [
                {
                    "story_id": "1-1",
                    "title": "First",
                    "status": "completed",
                    "duration_ms": 3600000.0,
                },
                {
                    "story_id": "1-2",
                    "title": "Second",
                    "status": "completed",
                    "duration_ms": 3600000.0,
                },
                {"story_id": "1-3", "title": "Third", "status": "backlog"},
            ]
        )
        config = ProgressConfig(include_estimates=True)

        result = await track_progress(state, plan, config)

        assert result.completion_estimate is not None
        assert result.completion_estimate.estimated_remaining_ms is not None

    @pytest.mark.asyncio
    async def test_returns_sprint_progress(self) -> None:
        """Returns a valid SprintProgress object."""
        state = make_state()
        plan = make_sprint_plan([{"story_id": "1-1", "title": "First", "status": "backlog"}])

        result = await track_progress(state, plan)

        assert isinstance(result, SprintProgress)
        assert isinstance(result.snapshot, SprintProgressSnapshot)
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_categorizes_blocked_and_failed(self) -> None:
        """Blocked and failed stories are both in blocked_stories."""
        state = make_state()
        plan = make_sprint_plan(
            [
                {"story_id": "1-1", "title": "First", "status": "blocked"},
                {"story_id": "1-2", "title": "Second", "status": "failed"},
            ]
        )

        result = await track_progress(state, plan)

        assert result.snapshot.stories_blocked == 2
        assert len(result.blocked_stories) == 2
