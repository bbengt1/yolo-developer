"""Tests for ADR types (Story 11.8).

Tests for AutoADR dataclass including creation, validation,
serialization, and immutability.
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.adr_types import (
    VALID_ADR_STATUSES,
    AutoADR,
)


class TestAutoADRCreation:
    """Tests for AutoADR dataclass creation."""

    def test_create_adr_with_required_fields(self) -> None:
        """Test creating AutoADR with all required fields."""
        adr = AutoADR(
            id="ADR-001",
            title="Use PostgreSQL for Data Storage",
            status="proposed",
            context="Database selection needed for persistence layer.",
            decision="Selected PostgreSQL for ACID compliance.",
            consequences="Positive: Strong consistency. Trade-off: Operational complexity.",
            source_decision_id="dec-123",
        )

        assert adr.id == "ADR-001"
        assert adr.title == "Use PostgreSQL for Data Storage"
        assert adr.status == "proposed"
        assert adr.context == "Database selection needed for persistence layer."
        assert adr.decision == "Selected PostgreSQL for ACID compliance."
        assert "Strong consistency" in adr.consequences
        assert adr.source_decision_id == "dec-123"
        assert adr.story_ids == ()
        assert adr.created_at  # Should have default value

    def test_create_adr_with_story_ids(self) -> None:
        """Test creating AutoADR with story_ids populated."""
        adr = AutoADR(
            id="ADR-002",
            title="Use Redis for Caching",
            status="proposed",
            context="Need caching layer.",
            decision="Selected Redis.",
            consequences="Fast lookups.",
            source_decision_id="dec-456",
            story_ids=("1-2-caching", "1-3-performance"),
        )

        assert adr.story_ids == ("1-2-caching", "1-3-performance")

    def test_create_adr_with_custom_created_at(self) -> None:
        """Test creating AutoADR with custom created_at timestamp."""
        custom_time = "2026-01-15T10:30:00+00:00"
        adr = AutoADR(
            id="ADR-003",
            title="Test ADR",
            status="accepted",
            context="Test context.",
            decision="Test decision.",
            consequences="Test consequences.",
            source_decision_id="dec-789",
            created_at=custom_time,
        )

        assert adr.created_at == custom_time

    def test_create_adr_with_all_statuses(self) -> None:
        """Test creating AutoADR with each valid status."""
        for status in VALID_ADR_STATUSES:
            adr = AutoADR(
                id=f"ADR-{status}",
                title=f"Test {status}",
                status=status,  # type: ignore[arg-type]
                context="Context",
                decision="Decision",
                consequences="Consequences",
                source_decision_id="dec-test",
            )
            assert adr.status == status


class TestAutoADRImmutability:
    """Tests for AutoADR frozen dataclass immutability."""

    def test_adr_is_frozen(self) -> None:
        """Test that AutoADR is immutable."""
        adr = AutoADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        with pytest.raises(AttributeError):
            adr.title = "New Title"  # type: ignore[misc]

    def test_adr_id_is_frozen(self) -> None:
        """Test that ADR id cannot be changed."""
        adr = AutoADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        with pytest.raises(AttributeError):
            adr.id = "ADR-999"  # type: ignore[misc]

    def test_adr_story_ids_tuple_is_immutable(self) -> None:
        """Test that story_ids tuple cannot be modified."""
        adr = AutoADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
            story_ids=("story-1", "story-2"),
        )

        # Tuple itself is immutable
        with pytest.raises(TypeError):
            adr.story_ids[0] = "new-story"  # type: ignore[index]


class TestAutoADRValidation:
    """Tests for AutoADR validation warnings."""

    def test_warns_on_empty_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty id triggers a warning."""
        AutoADR(
            id="",
            title="Test ADR",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        assert "AutoADR id is empty" in caplog.text

    def test_warns_on_invalid_id_format(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that non-ADR-XXX id format triggers a warning."""
        AutoADR(
            id="invalid-001",
            title="Test ADR",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        assert "does not follow ADR-XXX format" in caplog.text

    def test_warns_on_empty_title(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty title triggers a warning."""
        AutoADR(
            id="ADR-001",
            title="",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        assert "AutoADR title is empty" in caplog.text

    def test_warns_on_invalid_status(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid status triggers a warning."""
        AutoADR(
            id="ADR-001",
            title="Test",
            status="invalid_status",  # type: ignore[arg-type]
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
        )

        assert "is not a valid status" in caplog.text

    def test_warns_on_empty_source_decision_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that empty source_decision_id triggers a warning."""
        AutoADR(
            id="ADR-001",
            title="Test",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="",
        )

        assert "source_decision_id is empty" in caplog.text

    def test_no_warnings_for_valid_adr(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that valid ADR doesn't trigger warnings."""
        AutoADR(
            id="ADR-001",
            title="Valid ADR",
            status="proposed",
            context="Valid context",
            decision="Valid decision",
            consequences="Valid consequences",
            source_decision_id="dec-123",
        )

        # Check no warning messages about ADR validation
        assert "AutoADR" not in caplog.text or "is empty" not in caplog.text


class TestAutoADRToDict:
    """Tests for AutoADR.to_dict() serialization."""

    def test_to_dict_includes_all_fields(self) -> None:
        """Test that to_dict includes all ADR fields."""
        adr = AutoADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
            source_decision_id="dec-123",
            story_ids=("story-1", "story-2"),
            created_at="2026-01-15T10:00:00+00:00",
        )

        result = adr.to_dict()

        assert result["id"] == "ADR-001"
        assert result["title"] == "Test ADR"
        assert result["status"] == "proposed"
        assert result["context"] == "Test context"
        assert result["decision"] == "Test decision"
        assert result["consequences"] == "Test consequences"
        assert result["source_decision_id"] == "dec-123"
        assert result["story_ids"] == ["story-1", "story-2"]
        assert result["created_at"] == "2026-01-15T10:00:00+00:00"

    def test_to_dict_converts_story_ids_to_list(self) -> None:
        """Test that to_dict converts story_ids tuple to list."""
        adr = AutoADR(
            id="ADR-001",
            title="Test",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
            story_ids=("a", "b", "c"),
        )

        result = adr.to_dict()

        assert isinstance(result["story_ids"], list)
        assert result["story_ids"] == ["a", "b", "c"]

    def test_to_dict_empty_story_ids(self) -> None:
        """Test that to_dict handles empty story_ids."""
        adr = AutoADR(
            id="ADR-001",
            title="Test",
            status="proposed",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            source_decision_id="dec-123",
            story_ids=(),
        )

        result = adr.to_dict()

        assert result["story_ids"] == []

    def test_to_dict_is_json_serializable(self) -> None:
        """Test that to_dict output is JSON-serializable."""
        import json

        adr = AutoADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
            source_decision_id="dec-123",
            story_ids=("story-1",),
        )

        result = adr.to_dict()

        # Should not raise
        json_str = json.dumps(result)
        assert "ADR-001" in json_str


class TestAutoADRConstants:
    """Tests for ADR type constants."""

    def test_valid_adr_statuses_contains_expected_values(self) -> None:
        """Test that VALID_ADR_STATUSES contains all expected values."""
        expected = {"proposed", "accepted", "deprecated", "superseded"}
        assert VALID_ADR_STATUSES == expected

    def test_valid_adr_statuses_is_frozenset(self) -> None:
        """Test that VALID_ADR_STATUSES is a frozenset."""
        assert isinstance(VALID_ADR_STATUSES, frozenset)
