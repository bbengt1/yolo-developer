"""Unit tests for sprint progress tracking types (Story 10.9).

Tests the data types used by the sprint progress tracking module:
- StoryProgress: Progress information for a single story
- SprintProgressSnapshot: Point-in-time snapshot of sprint progress
- CompletionEstimate: Estimated completion information
- SprintProgress: Complete sprint progress information
- ProgressConfig: Configuration for progress tracking

All types are frozen dataclasses per ADR-001 for internal state.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from yolo_developer.agents.sm.progress_types import (
    DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD,
    VALID_STORY_STATUSES,
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
    StoryStatus,
)


class TestStoryStatus:
    """Tests for StoryStatus literal type."""

    def test_valid_statuses_constant(self) -> None:
        """Valid statuses constant contains all expected values."""
        expected = {"backlog", "in_progress", "completed", "blocked", "failed"}
        assert VALID_STORY_STATUSES == frozenset(expected)

    def test_all_statuses_are_valid(self) -> None:
        """All defined status values are in VALID_STORY_STATUSES."""
        # The Literal type defines these values
        statuses: list[StoryStatus] = [
            "backlog",
            "in_progress",
            "completed",
            "blocked",
            "failed",
        ]
        for status in statuses:
            assert status in VALID_STORY_STATUSES


class TestStoryProgress:
    """Tests for StoryProgress dataclass."""

    def test_create_minimal(self) -> None:
        """Create StoryProgress with minimal fields."""
        progress = StoryProgress(
            story_id="1-1-test",
            title="Test Story",
            status="backlog",
        )
        assert progress.story_id == "1-1-test"
        assert progress.title == "Test Story"
        assert progress.status == "backlog"
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.agent_history == ()
        assert progress.duration_ms is None
        assert progress.blocked_reason is None

    def test_create_complete(self) -> None:
        """Create StoryProgress with all fields."""
        progress = StoryProgress(
            story_id="1-2-auth",
            title="User Authentication",
            status="completed",
            started_at="2026-01-16T10:00:00+00:00",
            completed_at="2026-01-16T11:00:00+00:00",
            agent_history=("analyst", "pm", "dev", "tea"),
            duration_ms=3600000.0,
            blocked_reason=None,
        )
        assert progress.story_id == "1-2-auth"
        assert progress.status == "completed"
        assert progress.duration_ms == 3600000.0
        assert len(progress.agent_history) == 4

    def test_create_blocked(self) -> None:
        """Create blocked StoryProgress with reason."""
        progress = StoryProgress(
            story_id="1-3-blocked",
            title="Blocked Story",
            status="blocked",
            started_at="2026-01-16T10:00:00+00:00",
            blocked_reason="Gate blocked: testability",
        )
        assert progress.status == "blocked"
        assert progress.blocked_reason == "Gate blocked: testability"

    def test_frozen(self) -> None:
        """StoryProgress is immutable."""
        progress = StoryProgress(
            story_id="1-1-test",
            title="Test",
            status="backlog",
        )
        with pytest.raises(AttributeError):
            progress.status = "completed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        progress = StoryProgress(
            story_id="1-1-test",
            title="Test Story",
            status="in_progress",
            started_at="2026-01-16T10:00:00+00:00",
            agent_history=("analyst", "pm"),
            duration_ms=1000.0,
        )
        result = progress.to_dict()

        assert isinstance(result, dict)
        assert result["story_id"] == "1-1-test"
        assert result["title"] == "Test Story"
        assert result["status"] == "in_progress"
        assert result["started_at"] == "2026-01-16T10:00:00+00:00"
        assert result["completed_at"] is None
        assert result["agent_history"] == ["analyst", "pm"]  # Converted to list
        assert result["duration_ms"] == 1000.0
        assert result["blocked_reason"] is None


class TestSprintProgressSnapshot:
    """Tests for SprintProgressSnapshot dataclass."""

    def test_create(self) -> None:
        """Create SprintProgressSnapshot with all fields."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-20260116",
            total_stories=10,
            stories_completed=5,
            stories_in_progress=2,
            stories_remaining=2,
            stories_blocked=1,
            current_story="5-3-feature",
            current_agent="dev",
            progress_percentage=50.0,
        )
        assert snapshot.sprint_id == "sprint-20260116"
        assert snapshot.total_stories == 10
        assert snapshot.stories_completed == 5
        assert snapshot.progress_percentage == 50.0

    def test_create_empty_sprint(self) -> None:
        """Create snapshot for sprint with no progress."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-new",
            total_stories=5,
            stories_completed=0,
            stories_in_progress=0,
            stories_remaining=5,
            stories_blocked=0,
            current_story=None,
            current_agent=None,
            progress_percentage=0.0,
        )
        assert snapshot.stories_completed == 0
        assert snapshot.current_story is None
        assert snapshot.progress_percentage == 0.0

    def test_frozen(self) -> None:
        """SprintProgressSnapshot is immutable."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=5,
            stories_completed=0,
            stories_in_progress=0,
            stories_remaining=5,
            stories_blocked=0,
            current_story=None,
            current_agent=None,
            progress_percentage=0.0,
        )
        with pytest.raises(AttributeError):
            snapshot.stories_completed = 1  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-20260116",
            total_stories=10,
            stories_completed=5,
            stories_in_progress=2,
            stories_remaining=2,
            stories_blocked=1,
            current_story="5-3-feature",
            current_agent="dev",
            progress_percentage=50.0,
        )
        result = snapshot.to_dict()

        assert isinstance(result, dict)
        assert result["sprint_id"] == "sprint-20260116"
        assert result["total_stories"] == 10
        assert result["stories_completed"] == 5
        assert result["stories_in_progress"] == 2
        assert result["stories_remaining"] == 2
        assert result["stories_blocked"] == 1
        assert result["current_story"] == "5-3-feature"
        assert result["current_agent"] == "dev"
        assert result["progress_percentage"] == 50.0


class TestCompletionEstimate:
    """Tests for CompletionEstimate dataclass."""

    def test_create_with_estimate(self) -> None:
        """Create CompletionEstimate with valid estimate."""
        estimate = CompletionEstimate(
            estimated_completion_time="2026-01-16T15:00:00+00:00",
            estimated_remaining_ms=18000000.0,  # 5 hours
            confidence=0.85,
            factors=("based_on_5_completed_stories", "avg_duration_3600000ms"),
        )
        assert estimate.estimated_completion_time == "2026-01-16T15:00:00+00:00"
        assert estimate.estimated_remaining_ms == 18000000.0
        assert estimate.confidence == 0.85
        assert len(estimate.factors) == 2

    def test_create_no_estimate(self) -> None:
        """Create CompletionEstimate with insufficient data."""
        estimate = CompletionEstimate(
            estimated_completion_time=None,
            estimated_remaining_ms=None,
            confidence=0.0,
            factors=("insufficient_data",),
        )
        assert estimate.estimated_completion_time is None
        assert estimate.estimated_remaining_ms is None
        assert estimate.confidence == 0.0
        assert "insufficient_data" in estimate.factors

    def test_frozen(self) -> None:
        """CompletionEstimate is immutable."""
        estimate = CompletionEstimate(
            estimated_completion_time=None,
            estimated_remaining_ms=None,
            confidence=0.0,
            factors=(),
        )
        with pytest.raises(AttributeError):
            estimate.confidence = 0.5  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        estimate = CompletionEstimate(
            estimated_completion_time="2026-01-16T15:00:00+00:00",
            estimated_remaining_ms=18000000.0,
            confidence=0.85,
            factors=("factor1", "factor2"),
        )
        result = estimate.to_dict()

        assert isinstance(result, dict)
        assert result["estimated_completion_time"] == "2026-01-16T15:00:00+00:00"
        assert result["estimated_remaining_ms"] == 18000000.0
        assert result["confidence"] == 0.85
        assert result["factors"] == ["factor1", "factor2"]  # Converted to list


class TestSprintProgress:
    """Tests for SprintProgress dataclass."""

    def test_create_complete(self) -> None:
        """Create SprintProgress with all components."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-20260116",
            total_stories=5,
            stories_completed=2,
            stories_in_progress=1,
            stories_remaining=1,
            stories_blocked=1,
            current_story="3-1-feature",
            current_agent="dev",
            progress_percentage=40.0,
        )
        completed = (
            StoryProgress(story_id="1-1", title="First", status="completed"),
            StoryProgress(story_id="2-1", title="Second", status="completed"),
        )
        in_progress = (
            StoryProgress(story_id="3-1", title="Third", status="in_progress"),
        )
        remaining = (StoryProgress(story_id="4-1", title="Fourth", status="backlog"),)
        blocked = (
            StoryProgress(
                story_id="5-1",
                title="Fifth",
                status="blocked",
                blocked_reason="Dependency",
            ),
        )
        estimate = CompletionEstimate(
            estimated_completion_time="2026-01-16T15:00:00+00:00",
            estimated_remaining_ms=7200000.0,
            confidence=0.75,
            factors=("based_on_2_completed_stories",),
        )

        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=completed,
            in_progress_stories=in_progress,
            remaining_stories=remaining,
            blocked_stories=blocked,
            completion_estimate=estimate,
        )

        assert progress.snapshot.sprint_id == "sprint-20260116"
        assert len(progress.completed_stories) == 2
        assert len(progress.in_progress_stories) == 1
        assert len(progress.remaining_stories) == 1
        assert len(progress.blocked_stories) == 1
        assert progress.completion_estimate is not None
        assert progress.completion_estimate.confidence == 0.75

    def test_create_without_estimate(self) -> None:
        """Create SprintProgress without completion estimate."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-new",
            total_stories=5,
            stories_completed=0,
            stories_in_progress=1,
            stories_remaining=4,
            stories_blocked=0,
            current_story="1-1",
            current_agent="analyst",
            progress_percentage=0.0,
        )

        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(
                StoryProgress(story_id="1-1", title="First", status="in_progress"),
            ),
            remaining_stories=(
                StoryProgress(story_id="2-1", title="Second", status="backlog"),
            ),
            blocked_stories=(),
            completion_estimate=None,
        )

        assert progress.completion_estimate is None

    def test_created_at_auto_generated(self) -> None:
        """created_at is auto-generated if not provided."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=1,
            stories_completed=0,
            stories_in_progress=0,
            stories_remaining=1,
            stories_blocked=0,
            current_story=None,
            current_agent=None,
            progress_percentage=0.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        assert progress.created_at is not None
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(progress.created_at.replace("Z", "+00:00"))

    def test_frozen(self) -> None:
        """SprintProgress is immutable."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=1,
            stories_completed=0,
            stories_in_progress=0,
            stories_remaining=1,
            stories_blocked=0,
            current_story=None,
            current_agent=None,
            progress_percentage=0.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )
        with pytest.raises(AttributeError):
            progress.snapshot = snapshot  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict returns fully serializable dictionary."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-20260116",
            total_stories=2,
            stories_completed=1,
            stories_in_progress=1,
            stories_remaining=0,
            stories_blocked=0,
            current_story="2-1",
            current_agent="dev",
            progress_percentage=50.0,
        )
        estimate = CompletionEstimate(
            estimated_completion_time="2026-01-16T15:00:00+00:00",
            estimated_remaining_ms=3600000.0,
            confidence=0.8,
            factors=("factor1",),
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(
                StoryProgress(story_id="1-1", title="First", status="completed"),
            ),
            in_progress_stories=(
                StoryProgress(story_id="2-1", title="Second", status="in_progress"),
            ),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=estimate,
            created_at="2026-01-16T10:00:00+00:00",
        )

        result = progress.to_dict()

        assert isinstance(result, dict)
        assert result["snapshot"]["sprint_id"] == "sprint-20260116"
        assert len(result["completed_stories"]) == 1
        assert result["completed_stories"][0]["story_id"] == "1-1"
        assert len(result["in_progress_stories"]) == 1
        assert result["remaining_stories"] == []
        assert result["blocked_stories"] == []
        assert result["completion_estimate"]["confidence"] == 0.8
        assert result["created_at"] == "2026-01-16T10:00:00+00:00"

    def test_to_dict_without_estimate(self) -> None:
        """to_dict handles None completion_estimate."""
        snapshot = SprintProgressSnapshot(
            sprint_id="sprint-test",
            total_stories=1,
            stories_completed=0,
            stories_in_progress=0,
            stories_remaining=1,
            stories_blocked=0,
            current_story=None,
            current_agent=None,
            progress_percentage=0.0,
        )
        progress = SprintProgress(
            snapshot=snapshot,
            completed_stories=(),
            in_progress_stories=(),
            remaining_stories=(),
            blocked_stories=(),
            completion_estimate=None,
        )

        result = progress.to_dict()
        assert result["completion_estimate"] is None


class TestProgressConfig:
    """Tests for ProgressConfig dataclass."""

    def test_defaults(self) -> None:
        """ProgressConfig has sensible defaults."""
        config = ProgressConfig()
        assert config.include_estimates is True
        assert config.estimate_confidence_threshold == DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD
        assert config.track_agent_history is True
        assert config.include_blocked_details is True

    def test_custom_values(self) -> None:
        """ProgressConfig accepts custom values."""
        config = ProgressConfig(
            include_estimates=False,
            estimate_confidence_threshold=0.9,
            track_agent_history=False,
            include_blocked_details=False,
        )
        assert config.include_estimates is False
        assert config.estimate_confidence_threshold == 0.9
        assert config.track_agent_history is False
        assert config.include_blocked_details is False

    def test_frozen(self) -> None:
        """ProgressConfig is immutable."""
        config = ProgressConfig()
        with pytest.raises(AttributeError):
            config.include_estimates = False  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """to_dict returns serializable dictionary."""
        config = ProgressConfig(
            include_estimates=True,
            estimate_confidence_threshold=0.7,
            track_agent_history=True,
            include_blocked_details=False,
        )
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["include_estimates"] is True
        assert result["estimate_confidence_threshold"] == 0.7
        assert result["track_agent_history"] is True
        assert result["include_blocked_details"] is False


class TestConstants:
    """Tests for module constants."""

    def test_default_confidence_threshold(self) -> None:
        """Default confidence threshold is reasonable."""
        assert DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD == 0.7
        assert 0.0 <= DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD <= 1.0

    def test_valid_story_statuses_is_frozenset(self) -> None:
        """VALID_STORY_STATUSES is a frozenset."""
        assert isinstance(VALID_STORY_STATUSES, frozenset)

    def test_valid_story_statuses_count(self) -> None:
        """VALID_STORY_STATUSES has expected count."""
        assert len(VALID_STORY_STATUSES) == 5
