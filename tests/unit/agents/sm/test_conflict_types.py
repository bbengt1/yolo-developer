"""Unit tests for conflict mediation types (Story 10.7).

Tests for all type definitions in conflict_types.py:
- ConflictType literal
- ConflictSeverity literal
- ResolutionStrategy literal
- ConflictParty dataclass
- Conflict dataclass
- ConflictResolution dataclass
- MediationResult dataclass
- ConflictMediationConfig dataclass
- Constants and defaults
"""

from __future__ import annotations

from datetime import datetime

import pytest

from yolo_developer.agents.sm.conflict_types import (
    DEFAULT_MAX_MEDIATION_ROUNDS,
    DEFAULT_PRINCIPLES_HIERARCHY,
    RESOLUTION_PRINCIPLES,
    VALID_CONFLICT_SEVERITIES,
    VALID_CONFLICT_TYPES,
    VALID_RESOLUTION_STRATEGIES,
    Conflict,
    ConflictMediationConfig,
    ConflictParty,
    ConflictResolution,
    MediationResult,
)

# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_conflict_types(self) -> None:
        """Valid conflict types should contain expected values."""
        assert "design_conflict" in VALID_CONFLICT_TYPES
        assert "priority_conflict" in VALID_CONFLICT_TYPES
        assert "approach_conflict" in VALID_CONFLICT_TYPES
        assert "scope_conflict" in VALID_CONFLICT_TYPES
        assert len(VALID_CONFLICT_TYPES) == 4

    def test_valid_conflict_severities(self) -> None:
        """Valid conflict severities should contain expected values."""
        assert "minor" in VALID_CONFLICT_SEVERITIES
        assert "moderate" in VALID_CONFLICT_SEVERITIES
        assert "major" in VALID_CONFLICT_SEVERITIES
        assert "blocking" in VALID_CONFLICT_SEVERITIES
        assert len(VALID_CONFLICT_SEVERITIES) == 4

    def test_valid_resolution_strategies(self) -> None:
        """Valid resolution strategies should contain expected values."""
        assert "accept_first" in VALID_RESOLUTION_STRATEGIES
        assert "accept_second" in VALID_RESOLUTION_STRATEGIES
        assert "compromise" in VALID_RESOLUTION_STRATEGIES
        assert "defer" in VALID_RESOLUTION_STRATEGIES
        assert "escalate_human" in VALID_RESOLUTION_STRATEGIES
        assert len(VALID_RESOLUTION_STRATEGIES) == 5

    def test_resolution_principles_structure(self) -> None:
        """Resolution principles should have correct structure."""
        assert "safety" in RESOLUTION_PRINCIPLES
        assert "correctness" in RESOLUTION_PRINCIPLES
        assert "simplicity" in RESOLUTION_PRINCIPLES
        assert "performance" in RESOLUTION_PRINCIPLES
        assert "speed" in RESOLUTION_PRINCIPLES

        for _principle, info in RESOLUTION_PRINCIPLES.items():
            assert "description" in info
            assert "weight" in info
            assert "keywords" in info
            assert isinstance(info["weight"], float)
            assert isinstance(info["keywords"], tuple)

    def test_resolution_principles_weights_ordered(self) -> None:
        """Principle weights should be in descending order."""
        assert RESOLUTION_PRINCIPLES["safety"]["weight"] == 1.0
        assert RESOLUTION_PRINCIPLES["correctness"]["weight"] == 0.9
        assert RESOLUTION_PRINCIPLES["simplicity"]["weight"] == 0.7
        assert RESOLUTION_PRINCIPLES["performance"]["weight"] == 0.5
        assert RESOLUTION_PRINCIPLES["speed"]["weight"] == 0.3

    def test_default_principles_hierarchy(self) -> None:
        """Default principles hierarchy should match expected order."""
        assert DEFAULT_PRINCIPLES_HIERARCHY == (
            "safety",
            "correctness",
            "simplicity",
            "performance",
            "speed",
        )

    def test_default_max_mediation_rounds(self) -> None:
        """Default max mediation rounds should be 3."""
        assert DEFAULT_MAX_MEDIATION_ROUNDS == 3


# =============================================================================
# ConflictParty Tests
# =============================================================================


class TestConflictParty:
    """Tests for ConflictParty dataclass."""

    def test_create_conflict_party(self) -> None:
        """Should create ConflictParty with all fields."""
        party = ConflictParty(
            agent="architect",
            position="Use microservices",
            rationale="Better scalability",
            artifacts=("adr-001",),
        )

        assert party.agent == "architect"
        assert party.position == "Use microservices"
        assert party.rationale == "Better scalability"
        assert party.artifacts == ("adr-001",)

    def test_conflict_party_default_artifacts(self) -> None:
        """Should create ConflictParty with empty artifacts by default."""
        party = ConflictParty(
            agent="dev",
            position="Keep it simple",
            rationale="KISS principle",
        )

        assert party.artifacts == ()

    def test_conflict_party_frozen(self) -> None:
        """ConflictParty should be immutable."""
        party = ConflictParty(
            agent="pm",
            position="Prioritize feature A",
            rationale="Higher business value",
        )

        with pytest.raises(AttributeError):
            party.agent = "analyst"  # type: ignore[misc]

    def test_conflict_party_to_dict(self) -> None:
        """Should convert ConflictParty to dictionary."""
        party = ConflictParty(
            agent="architect",
            position="Use microservices",
            rationale="Better scalability",
            artifacts=("adr-001", "req-123"),
        )

        result = party.to_dict()

        assert result["agent"] == "architect"
        assert result["position"] == "Use microservices"
        assert result["rationale"] == "Better scalability"
        assert result["artifacts"] == ["adr-001", "req-123"]

    def test_conflict_party_to_dict_converts_tuple_to_list(self) -> None:
        """to_dict should convert artifacts tuple to list."""
        party = ConflictParty(
            agent="tea",
            position="Add more tests",
            rationale="Improve coverage",
            artifacts=("test-plan-1",),
        )

        result = party.to_dict()

        assert isinstance(result["artifacts"], list)


# =============================================================================
# Conflict Tests
# =============================================================================


class TestConflict:
    """Tests for Conflict dataclass."""

    def test_create_conflict(self) -> None:
        """Should create Conflict with all fields."""
        party1 = ConflictParty(agent="architect", position="Microservices", rationale="Scalability")
        party2 = ConflictParty(agent="dev", position="Monolith", rationale="Simplicity")

        conflict = Conflict(
            conflict_id="conflict-001",
            conflict_type="design_conflict",
            severity="major",
            parties=(party1, party2),
            topic="service_architecture",
            blocking_progress=True,
        )

        assert conflict.conflict_id == "conflict-001"
        assert conflict.conflict_type == "design_conflict"
        assert conflict.severity == "major"
        assert len(conflict.parties) == 2
        assert conflict.topic == "service_architecture"
        assert conflict.blocking_progress is True

    def test_conflict_default_values(self) -> None:
        """Should create Conflict with default values."""
        party = ConflictParty(agent="pm", position="Test", rationale="Test")

        conflict = Conflict(
            conflict_id="conflict-002",
            conflict_type="priority_conflict",
            severity="minor",
            parties=(party,),
            topic="feature_priority",
        )

        assert conflict.blocking_progress is False
        # detected_at should be a valid ISO timestamp
        datetime.fromisoformat(conflict.detected_at.replace("Z", "+00:00"))

    def test_conflict_frozen(self) -> None:
        """Conflict should be immutable."""
        party = ConflictParty(agent="analyst", position="Test", rationale="Test")
        conflict = Conflict(
            conflict_id="conflict-003",
            conflict_type="scope_conflict",
            severity="moderate",
            parties=(party,),
            topic="scope_boundaries",
        )

        with pytest.raises(AttributeError):
            conflict.severity = "blocking"  # type: ignore[misc]

    def test_conflict_to_dict(self) -> None:
        """Should convert Conflict to dictionary."""
        party1 = ConflictParty(agent="architect", position="A", rationale="R1")
        party2 = ConflictParty(agent="dev", position="B", rationale="R2")

        conflict = Conflict(
            conflict_id="conflict-004",
            conflict_type="approach_conflict",
            severity="major",
            parties=(party1, party2),
            topic="implementation_approach",
            detected_at="2026-01-16T10:00:00+00:00",
            blocking_progress=False,
        )

        result = conflict.to_dict()

        assert result["conflict_id"] == "conflict-004"
        assert result["conflict_type"] == "approach_conflict"
        assert result["severity"] == "major"
        assert len(result["parties"]) == 2
        assert result["parties"][0]["agent"] == "architect"
        assert result["parties"][1]["agent"] == "dev"
        assert result["topic"] == "implementation_approach"
        assert result["detected_at"] == "2026-01-16T10:00:00+00:00"
        assert result["blocking_progress"] is False


# =============================================================================
# ConflictResolution Tests
# =============================================================================


class TestConflictResolution:
    """Tests for ConflictResolution dataclass."""

    def test_create_conflict_resolution_accept_first(self) -> None:
        """Should create resolution with accept_first strategy."""
        resolution = ConflictResolution(
            conflict_id="conflict-001",
            strategy="accept_first",
            resolution_rationale="Security concerns take precedence",
            winning_position="Use secure API gateway",
            principles_applied=("safety", "correctness"),
        )

        assert resolution.conflict_id == "conflict-001"
        assert resolution.strategy == "accept_first"
        assert resolution.winning_position == "Use secure API gateway"
        assert resolution.principles_applied == ("safety", "correctness")

    def test_create_conflict_resolution_compromise(self) -> None:
        """Should create resolution with compromise strategy."""
        resolution = ConflictResolution(
            conflict_id="conflict-002",
            strategy="compromise",
            resolution_rationale="Both positions have merit",
            compromises=("Use modular monolith", "Plan migration path"),
        )

        assert resolution.strategy == "compromise"
        assert resolution.winning_position is None
        assert resolution.compromises == ("Use modular monolith", "Plan migration path")

    def test_conflict_resolution_default_values(self) -> None:
        """Should create resolution with default values."""
        resolution = ConflictResolution(
            conflict_id="conflict-003",
            strategy="defer",
            resolution_rationale="Need more context",
        )

        assert resolution.winning_position is None
        assert resolution.compromises == ()
        assert resolution.principles_applied == ()
        # documented_at should be a valid ISO timestamp
        datetime.fromisoformat(resolution.documented_at.replace("Z", "+00:00"))

    def test_conflict_resolution_frozen(self) -> None:
        """ConflictResolution should be immutable."""
        resolution = ConflictResolution(
            conflict_id="conflict-004",
            strategy="escalate_human",
            resolution_rationale="Cannot determine automatically",
        )

        with pytest.raises(AttributeError):
            resolution.strategy = "accept_first"  # type: ignore[misc]

    def test_conflict_resolution_to_dict(self) -> None:
        """Should convert ConflictResolution to dictionary."""
        resolution = ConflictResolution(
            conflict_id="conflict-005",
            strategy="accept_second",
            resolution_rationale="Simpler approach preferred",
            winning_position="Use straightforward implementation",
            compromises=(),
            principles_applied=("simplicity",),
            documented_at="2026-01-16T11:00:00+00:00",
        )

        result = resolution.to_dict()

        assert result["conflict_id"] == "conflict-005"
        assert result["strategy"] == "accept_second"
        assert result["resolution_rationale"] == "Simpler approach preferred"
        assert result["winning_position"] == "Use straightforward implementation"
        assert result["compromises"] == []
        assert result["principles_applied"] == ["simplicity"]
        assert result["documented_at"] == "2026-01-16T11:00:00+00:00"


# =============================================================================
# MediationResult Tests
# =============================================================================


class TestMediationResult:
    """Tests for MediationResult dataclass."""

    def test_create_mediation_result_success(self) -> None:
        """Should create successful MediationResult."""
        party = ConflictParty(agent="architect", position="A", rationale="R")
        conflict = Conflict(
            conflict_id="c-001",
            conflict_type="design_conflict",
            severity="major",
            parties=(party,),
            topic="arch",
        )
        resolution = ConflictResolution(
            conflict_id="c-001",
            strategy="accept_first",
            resolution_rationale="Clear winner",
        )

        result = MediationResult(
            conflicts_detected=(conflict,),
            resolutions=(resolution,),
            notifications_sent=("architect", "dev"),
            escalations_triggered=(),
            success=True,
            mediation_notes="Resolved via safety principle",
        )

        assert len(result.conflicts_detected) == 1
        assert len(result.resolutions) == 1
        assert result.notifications_sent == ("architect", "dev")
        assert result.escalations_triggered == ()
        assert result.success is True
        assert result.mediation_notes == "Resolved via safety principle"

    def test_create_mediation_result_with_escalation(self) -> None:
        """Should create MediationResult with escalation."""
        party = ConflictParty(agent="pm", position="B", rationale="R")
        conflict = Conflict(
            conflict_id="c-002",
            conflict_type="priority_conflict",
            severity="blocking",
            parties=(party,),
            topic="priority",
            blocking_progress=True,
        )
        resolution = ConflictResolution(
            conflict_id="c-002",
            strategy="escalate_human",
            resolution_rationale="Cannot resolve automatically",
        )

        result = MediationResult(
            conflicts_detected=(conflict,),
            resolutions=(resolution,),
            notifications_sent=("pm", "analyst"),
            escalations_triggered=("c-002",),
            success=False,
            mediation_notes="Escalated due to blocking severity",
        )

        assert result.escalations_triggered == ("c-002",)
        assert result.success is False

    def test_mediation_result_no_conflicts(self) -> None:
        """Should create MediationResult with no conflicts."""
        result = MediationResult(
            conflicts_detected=(),
            resolutions=(),
            notifications_sent=(),
            escalations_triggered=(),
            success=True,
        )

        assert len(result.conflicts_detected) == 0
        assert len(result.resolutions) == 0
        assert result.success is True
        assert result.mediation_notes == ""

    def test_mediation_result_frozen(self) -> None:
        """MediationResult should be immutable."""
        result = MediationResult(
            conflicts_detected=(),
            resolutions=(),
            notifications_sent=(),
            escalations_triggered=(),
            success=True,
        )

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_mediation_result_to_dict(self) -> None:
        """Should convert MediationResult to dictionary."""
        party = ConflictParty(agent="tea", position="C", rationale="R")
        conflict = Conflict(
            conflict_id="c-003",
            conflict_type="approach_conflict",
            severity="moderate",
            parties=(party,),
            topic="testing",
        )
        resolution = ConflictResolution(
            conflict_id="c-003",
            strategy="compromise",
            resolution_rationale="Balanced approach",
        )

        result = MediationResult(
            conflicts_detected=(conflict,),
            resolutions=(resolution,),
            notifications_sent=("tea",),
            escalations_triggered=(),
            success=True,
            mediation_notes="Compromise achieved",
            mediated_at="2026-01-16T12:00:00+00:00",
        )

        result_dict = result.to_dict()

        assert len(result_dict["conflicts_detected"]) == 1
        assert result_dict["conflicts_detected"][0]["conflict_id"] == "c-003"
        assert len(result_dict["resolutions"]) == 1
        assert result_dict["resolutions"][0]["strategy"] == "compromise"
        assert result_dict["notifications_sent"] == ["tea"]
        assert result_dict["escalations_triggered"] == []
        assert result_dict["success"] is True
        assert result_dict["mediation_notes"] == "Compromise achieved"
        assert result_dict["mediated_at"] == "2026-01-16T12:00:00+00:00"


# =============================================================================
# ConflictMediationConfig Tests
# =============================================================================


class TestConflictMediationConfig:
    """Tests for ConflictMediationConfig dataclass."""

    def test_create_config_defaults(self) -> None:
        """Should create config with default values."""
        config = ConflictMediationConfig()

        assert config.auto_resolve_minor is True
        assert config.escalate_blocking is True
        assert config.max_mediation_rounds == 3
        assert config.principles_hierarchy == DEFAULT_PRINCIPLES_HIERARCHY
        assert config.score_threshold == 0.1

    def test_create_config_custom(self) -> None:
        """Should create config with custom values."""
        config = ConflictMediationConfig(
            auto_resolve_minor=False,
            escalate_blocking=False,
            max_mediation_rounds=5,
            principles_hierarchy=("correctness", "safety"),
            score_threshold=0.2,
        )

        assert config.auto_resolve_minor is False
        assert config.escalate_blocking is False
        assert config.max_mediation_rounds == 5
        assert config.principles_hierarchy == ("correctness", "safety")
        assert config.score_threshold == 0.2

    def test_config_frozen(self) -> None:
        """ConflictMediationConfig should be immutable."""
        config = ConflictMediationConfig()

        with pytest.raises(AttributeError):
            config.max_mediation_rounds = 10  # type: ignore[misc]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_conflict_with_many_parties(self) -> None:
        """Should handle conflict with more than 2 parties."""
        parties = tuple(
            ConflictParty(agent=f"agent-{i}", position=f"Position {i}", rationale="R")
            for i in range(5)
        )

        conflict = Conflict(
            conflict_id="multi-party",
            conflict_type="scope_conflict",
            severity="major",
            parties=parties,
            topic="scope_boundaries",
        )

        assert len(conflict.parties) == 5
        result = conflict.to_dict()
        assert len(result["parties"]) == 5

    def test_resolution_with_many_compromises(self) -> None:
        """Should handle resolution with many compromises."""
        compromises = tuple(f"Compromise {i}" for i in range(10))

        resolution = ConflictResolution(
            conflict_id="many-comp",
            strategy="compromise",
            resolution_rationale="Many compromises needed",
            compromises=compromises,
        )

        assert len(resolution.compromises) == 10
        result = resolution.to_dict()
        assert len(result["compromises"]) == 10

    def test_mediation_result_with_many_conflicts(self) -> None:
        """Should handle mediation result with many conflicts."""
        conflicts = []
        resolutions = []

        for i in range(10):
            party = ConflictParty(agent=f"agent-{i}", position=f"Pos {i}", rationale="R")
            conflict = Conflict(
                conflict_id=f"c-{i}",
                conflict_type="design_conflict",
                severity="minor",
                parties=(party,),
                topic=f"topic-{i}",
            )
            resolution = ConflictResolution(
                conflict_id=f"c-{i}",
                strategy="accept_first",
                resolution_rationale="Auto-resolved",
            )
            conflicts.append(conflict)
            resolutions.append(resolution)

        result = MediationResult(
            conflicts_detected=tuple(conflicts),
            resolutions=tuple(resolutions),
            notifications_sent=(),
            escalations_triggered=(),
            success=True,
        )

        assert len(result.conflicts_detected) == 10
        assert len(result.resolutions) == 10
        result_dict = result.to_dict()
        assert len(result_dict["conflicts_detected"]) == 10

    def test_empty_strings_in_party(self) -> None:
        """Should allow empty strings in ConflictParty fields."""
        party = ConflictParty(
            agent="",
            position="",
            rationale="",
            artifacts=(),
        )

        assert party.agent == ""
        result = party.to_dict()
        assert result["agent"] == ""

    def test_unicode_in_conflict_fields(self) -> None:
        """Should handle unicode in conflict fields."""
        party = ConflictParty(
            agent="架构师",  # Chinese for "architect"
            position="使用微服务",  # Chinese for "use microservices"
            rationale="更好的可扩展性",  # Chinese for "better scalability"
        )

        conflict = Conflict(
            conflict_id="unicode-test",
            conflict_type="design_conflict",
            severity="minor",
            parties=(party,),
            topic="国际化测试",  # Chinese for "internationalization test"
        )

        result = conflict.to_dict()
        assert result["topic"] == "国际化测试"
        assert result["parties"][0]["agent"] == "架构师"
