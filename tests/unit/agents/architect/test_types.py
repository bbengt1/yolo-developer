"""Tests for Architect agent type definitions (Story 7.1, Task 8).

Tests verify that the dataclasses (DesignDecision, ADR, ArchitectOutput) are
properly defined with correct fields, immutability, and serialization.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.architect.types import (
    ADR,
    ADRStatus,
    ArchitectOutput,
    DesignDecision,
    DesignDecisionType,
)


class TestDesignDecisionDataclass:
    """Test DesignDecision dataclass."""

    def test_design_decision_creation(self) -> None:
        """Test creating a DesignDecision with required fields."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",
            description="Use PostgreSQL for persistence",
            rationale="Reliable, scalable, well-supported",
            alternatives_considered=("MySQL", "MongoDB"),
        )

        assert decision.id == "design-001"
        assert decision.story_id == "story-001"
        assert decision.decision_type == "technology"
        assert decision.description == "Use PostgreSQL for persistence"
        assert decision.rationale == "Reliable, scalable, well-supported"
        assert decision.alternatives_considered == ("MySQL", "MongoDB")

    def test_design_decision_has_created_at(self) -> None:
        """Test that DesignDecision has created_at timestamp."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="pattern",
            description="Test",
            rationale="Test rationale",
        )

        assert decision.created_at is not None
        assert "T" in decision.created_at  # ISO format

    def test_design_decision_to_dict(self) -> None:
        """Test DesignDecision serialization to dict."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="integration",
            description="REST API integration",
            rationale="Standard approach",
            alternatives_considered=("GraphQL", "gRPC"),
        )

        d = decision.to_dict()

        assert d["id"] == "design-001"
        assert d["story_id"] == "story-001"
        assert d["decision_type"] == "integration"
        assert d["description"] == "REST API integration"
        assert d["rationale"] == "Standard approach"
        assert d["alternatives_considered"] == ["GraphQL", "gRPC"]
        assert "created_at" in d

    def test_design_decision_immutable(self) -> None:
        """Test that DesignDecision is frozen (immutable)."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="data",
            description="Test",
            rationale="Test",
        )

        with pytest.raises(AttributeError):
            decision.id = "new-id"  # type: ignore[misc]

    def test_design_decision_default_alternatives(self) -> None:
        """Test that alternatives_considered defaults to empty tuple."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="security",
            description="Test",
            rationale="Test",
        )

        assert decision.alternatives_considered == ()


class TestDesignDecisionType:
    """Test DesignDecisionType literal values."""

    def test_valid_decision_types(self) -> None:
        """Test all valid DesignDecisionType values."""
        valid_types: list[DesignDecisionType] = [
            "pattern",
            "technology",
            "integration",
            "data",
            "security",
            "infrastructure",
        ]

        for dtype in valid_types:
            decision = DesignDecision(
                id="test",
                story_id="test",
                decision_type=dtype,
                description="Test",
                rationale="Test",
            )
            assert decision.decision_type == dtype


class TestADRDataclass:
    """Test ADR dataclass."""

    def test_adr_creation(self) -> None:
        """Test creating an ADR with required fields."""
        adr = ADR(
            id="ADR-001",
            title="Use PostgreSQL for persistence",
            status="proposed",
            context="We need a reliable database for user data.",
            decision="We will use PostgreSQL.",
            consequences="Need PostgreSQL expertise on team.",
            story_ids=("story-001", "story-002"),
        )

        assert adr.id == "ADR-001"
        assert adr.title == "Use PostgreSQL for persistence"
        assert adr.status == "proposed"
        assert adr.context == "We need a reliable database for user data."
        assert adr.decision == "We will use PostgreSQL."
        assert adr.consequences == "Need PostgreSQL expertise on team."
        assert adr.story_ids == ("story-001", "story-002")

    def test_adr_has_created_at(self) -> None:
        """Test that ADR has created_at timestamp."""
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="accepted",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
        )

        assert adr.created_at is not None
        assert "T" in adr.created_at  # ISO format

    def test_adr_to_dict(self) -> None:
        """Test ADR serialization to dict."""
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="accepted",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
            story_ids=("story-001",),
        )

        d = adr.to_dict()

        assert d["id"] == "ADR-001"
        assert d["title"] == "Test ADR"
        assert d["status"] == "accepted"
        assert d["context"] == "Test context"
        assert d["decision"] == "Test decision"
        assert d["consequences"] == "Test consequences"
        assert d["story_ids"] == ["story-001"]
        assert "created_at" in d

    def test_adr_immutable(self) -> None:
        """Test that ADR is frozen (immutable)."""
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
        )

        with pytest.raises(AttributeError):
            adr.status = "accepted"  # type: ignore[misc]

    def test_adr_default_story_ids(self) -> None:
        """Test that story_ids defaults to empty tuple."""
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences",
        )

        assert adr.story_ids == ()


class TestADRStatus:
    """Test ADRStatus literal values."""

    def test_valid_adr_statuses(self) -> None:
        """Test all valid ADRStatus values."""
        valid_statuses: list[ADRStatus] = [
            "proposed",
            "accepted",
            "deprecated",
            "superseded",
        ]

        for status in valid_statuses:
            adr = ADR(
                id="test",
                title="Test",
                status=status,
                context="Test",
                decision="Test",
                consequences="Test",
            )
            assert adr.status == status


class TestArchitectOutputDataclass:
    """Test ArchitectOutput dataclass."""

    def test_architect_output_creation(self) -> None:
        """Test creating ArchitectOutput with defaults."""
        output = ArchitectOutput()

        assert output.design_decisions == ()
        assert output.adrs == ()
        assert output.processing_notes == ""

    def test_architect_output_with_decisions(self) -> None:
        """Test ArchitectOutput with design decisions."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",
            description="Test",
            rationale="Test",
        )
        output = ArchitectOutput(
            design_decisions=(decision,),
            processing_notes="Processed 1 decision",
        )

        assert len(output.design_decisions) == 1
        assert output.design_decisions[0].id == "design-001"
        assert output.processing_notes == "Processed 1 decision"

    def test_architect_output_with_adrs(self) -> None:
        """Test ArchitectOutput with ADRs."""
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="proposed",
            context="Test",
            decision="Test",
            consequences="Test",
        )
        output = ArchitectOutput(adrs=(adr,))

        assert len(output.adrs) == 1
        assert output.adrs[0].id == "ADR-001"

    def test_architect_output_to_dict(self) -> None:
        """Test ArchitectOutput serialization to dict."""
        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="pattern",
            description="Test",
            rationale="Test",
        )
        adr = ADR(
            id="ADR-001",
            title="Test ADR",
            status="accepted",
            context="Test",
            decision="Test",
            consequences="Test",
        )
        output = ArchitectOutput(
            design_decisions=(decision,),
            adrs=(adr,),
            processing_notes="Test notes",
        )

        d = output.to_dict()

        assert "design_decisions" in d
        assert len(d["design_decisions"]) == 1
        assert d["design_decisions"][0]["id"] == "design-001"
        assert "adrs" in d
        assert len(d["adrs"]) == 1
        assert d["adrs"][0]["id"] == "ADR-001"
        assert d["processing_notes"] == "Test notes"

    def test_architect_output_immutable(self) -> None:
        """Test that ArchitectOutput is frozen (immutable)."""
        output = ArchitectOutput()

        with pytest.raises(AttributeError):
            output.processing_notes = "new notes"  # type: ignore[misc]

    def test_architect_output_empty_to_dict(self) -> None:
        """Test ArchitectOutput to_dict with empty fields."""
        output = ArchitectOutput()

        d = output.to_dict()

        assert d["design_decisions"] == []
        assert d["adrs"] == []
        assert d["processing_notes"] == ""
