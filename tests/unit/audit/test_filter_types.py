"""Tests for unified audit filter types (Story 11.7).

Tests for AuditFilters dataclass including creation, validation,
serialization, and conversion to store-specific filter types.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.filter_types import AuditFilters


class TestAuditFiltersCreation:
    """Tests for AuditFilters dataclass creation."""

    def test_create_empty_filters(self) -> None:
        """Test creating AuditFilters with all defaults."""
        filters = AuditFilters()

        assert filters.agent_name is None
        assert filters.decision_type is None
        assert filters.artifact_type is None
        assert filters.start_time is None
        assert filters.end_time is None
        assert filters.sprint_id is None
        assert filters.story_id is None
        assert filters.session_id is None
        assert filters.severity is None

    def test_create_filters_with_agent_name(self) -> None:
        """Test creating AuditFilters with agent_name."""
        filters = AuditFilters(agent_name="analyst")

        assert filters.agent_name == "analyst"

    def test_create_filters_with_time_range(self) -> None:
        """Test creating AuditFilters with time range."""
        filters = AuditFilters(
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
        )

        assert filters.start_time == "2026-01-01T00:00:00Z"
        assert filters.end_time == "2026-01-31T23:59:59Z"

    def test_create_filters_with_artifact_type(self) -> None:
        """Test creating AuditFilters with valid artifact_type."""
        filters = AuditFilters(artifact_type="requirement")

        assert filters.artifact_type == "requirement"

    def test_create_filters_with_all_fields(self) -> None:
        """Test creating AuditFilters with all fields populated."""
        filters = AuditFilters(
            agent_name="analyst",
            decision_type="requirement_analysis",
            artifact_type="story",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
            sprint_id="sprint-1",
            story_id="1-2-auth",
            session_id="session-123",
            severity="high",
        )

        assert filters.agent_name == "analyst"
        assert filters.decision_type == "requirement_analysis"
        assert filters.artifact_type == "story"
        assert filters.start_time == "2026-01-01T00:00:00Z"
        assert filters.end_time == "2026-01-31T23:59:59Z"
        assert filters.sprint_id == "sprint-1"
        assert filters.story_id == "1-2-auth"
        assert filters.session_id == "session-123"
        assert filters.severity == "high"

    def test_filters_is_frozen(self) -> None:
        """Test that AuditFilters is immutable."""
        filters = AuditFilters(agent_name="analyst")

        with pytest.raises(AttributeError):
            filters.agent_name = "pm"  # type: ignore[misc]


class TestAuditFiltersValidation:
    """Tests for AuditFilters validation."""

    def test_warns_on_invalid_artifact_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid artifact_type triggers a warning."""
        AuditFilters(artifact_type="invalid_type")

        assert "artifact_type='invalid_type' is not a valid artifact type" in caplog.text

    def test_no_warning_for_valid_artifact_types(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that valid artifact types don't trigger warnings."""
        valid_types = ["requirement", "story", "design_decision", "code", "test"]

        for artifact_type in valid_types:
            caplog.clear()
            AuditFilters(artifact_type=artifact_type)
            assert "is not a valid artifact type" not in caplog.text

    def test_no_warning_for_none_artifact_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that None artifact_type doesn't trigger a warning."""
        AuditFilters(artifact_type=None)

        assert "is not a valid artifact type" not in caplog.text


class TestAuditFiltersToDict:
    """Tests for AuditFilters.to_dict() method."""

    def test_to_dict_empty_filters(self) -> None:
        """Test to_dict with empty filters."""
        filters = AuditFilters()
        result = filters.to_dict()

        assert result == {
            "agent_name": None,
            "decision_type": None,
            "artifact_type": None,
            "start_time": None,
            "end_time": None,
            "sprint_id": None,
            "story_id": None,
            "session_id": None,
            "severity": None,
        }

    def test_to_dict_with_values(self) -> None:
        """Test to_dict with populated filters."""
        filters = AuditFilters(
            agent_name="analyst",
            artifact_type="requirement",
            start_time="2026-01-01T00:00:00Z",
        )
        result = filters.to_dict()

        assert result["agent_name"] == "analyst"
        assert result["artifact_type"] == "requirement"
        assert result["start_time"] == "2026-01-01T00:00:00Z"
        assert result["decision_type"] is None


class TestAuditFiltersToDecisionFilters:
    """Tests for AuditFilters.to_decision_filters() conversion."""

    def test_to_decision_filters_basic(self) -> None:
        """Test basic conversion to DecisionFilters."""
        from yolo_developer.audit.store import DecisionFilters

        filters = AuditFilters(
            agent_name="analyst",
            decision_type="requirement_analysis",
        )
        result = filters.to_decision_filters()

        assert isinstance(result, DecisionFilters)
        assert result.agent_name == "analyst"
        assert result.decision_type == "requirement_analysis"

    def test_to_decision_filters_with_time_range(self) -> None:
        """Test conversion with time range."""
        filters = AuditFilters(
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
        )
        result = filters.to_decision_filters()

        assert result.start_time == "2026-01-01T00:00:00Z"
        assert result.end_time == "2026-01-31T23:59:59Z"

    def test_to_decision_filters_with_story_and_sprint(self) -> None:
        """Test conversion with story_id and sprint_id."""
        filters = AuditFilters(
            sprint_id="sprint-1",
            story_id="1-2-auth",
        )
        result = filters.to_decision_filters()

        assert result.sprint_id == "sprint-1"
        assert result.story_id == "1-2-auth"

    def test_to_decision_filters_ignores_irrelevant_fields(self) -> None:
        """Test that artifact_type and session_id are not in DecisionFilters."""
        filters = AuditFilters(
            artifact_type="requirement",
            session_id="session-123",
            agent_name="analyst",
        )
        result = filters.to_decision_filters()

        # DecisionFilters doesn't have artifact_type or session_id
        assert result.agent_name == "analyst"
        assert not hasattr(result, "artifact_type")
        assert not hasattr(result, "session_id")


class TestAuditFiltersToTraceabilityFilters:
    """Tests for AuditFilters.to_traceability_filters() conversion."""

    def test_to_traceability_filters_basic(self) -> None:
        """Test basic conversion to traceability filter dict."""
        filters = AuditFilters(
            artifact_type="requirement",
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
        )
        result = filters.to_traceability_filters()

        assert isinstance(result, dict)
        assert result["artifact_type"] == "requirement"
        assert result["created_after"] == "2026-01-01T00:00:00Z"
        assert result["created_before"] == "2026-01-31T23:59:59Z"

    def test_to_traceability_filters_maps_time_fields(self) -> None:
        """Test that start_time maps to created_after and end_time to created_before."""
        filters = AuditFilters(
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
        )
        result = filters.to_traceability_filters()

        assert result["created_after"] == "2026-01-01T00:00:00Z"
        assert result["created_before"] == "2026-01-31T23:59:59Z"

    def test_to_traceability_filters_none_values(self) -> None:
        """Test that None values are passed through correctly."""
        filters = AuditFilters()
        result = filters.to_traceability_filters()

        assert result["artifact_type"] is None
        assert result["created_after"] is None
        assert result["created_before"] is None

    def test_to_traceability_filters_ignores_irrelevant_fields(self) -> None:
        """Test that agent_name and other fields are not in result."""
        filters = AuditFilters(
            agent_name="analyst",
            decision_type="requirement_analysis",
            artifact_type="requirement",
        )
        result = filters.to_traceability_filters()

        assert "agent_name" not in result
        assert "decision_type" not in result
        assert result["artifact_type"] == "requirement"


class TestAuditFiltersToCostFilters:
    """Tests for AuditFilters.to_cost_filters() conversion."""

    def test_to_cost_filters_basic(self) -> None:
        """Test basic conversion to CostFilters."""
        from yolo_developer.audit.cost_store import CostFilters

        filters = AuditFilters(
            agent_name="analyst",
            session_id="session-123",
        )
        result = filters.to_cost_filters()

        assert isinstance(result, CostFilters)
        assert result.agent_name == "analyst"
        assert result.session_id == "session-123"

    def test_to_cost_filters_with_time_range(self) -> None:
        """Test conversion with time range."""
        filters = AuditFilters(
            start_time="2026-01-01T00:00:00Z",
            end_time="2026-01-31T23:59:59Z",
        )
        result = filters.to_cost_filters()

        assert result.start_time == "2026-01-01T00:00:00Z"
        assert result.end_time == "2026-01-31T23:59:59Z"

    def test_to_cost_filters_with_story_and_sprint(self) -> None:
        """Test conversion with story_id and sprint_id."""
        filters = AuditFilters(
            sprint_id="sprint-1",
            story_id="1-2-auth",
        )
        result = filters.to_cost_filters()

        assert result.sprint_id == "sprint-1"
        assert result.story_id == "1-2-auth"

    def test_to_cost_filters_ignores_irrelevant_fields(self) -> None:
        """Test that decision_type and artifact_type are not in CostFilters."""
        filters = AuditFilters(
            decision_type="requirement_analysis",
            artifact_type="requirement",
            agent_name="analyst",
        )
        result = filters.to_cost_filters()

        # CostFilters doesn't have decision_type or artifact_type
        assert result.agent_name == "analyst"
        assert not hasattr(result, "decision_type")
        assert not hasattr(result, "artifact_type")
