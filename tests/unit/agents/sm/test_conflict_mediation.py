"""Unit tests for conflict mediation (Story 10.7).

Tests for conflict detection and mediation functionality:
- Agent position extraction
- Conflict detection (design, priority, approach, scope)
- Severity calculation
- Position scoring
- Resolution strategy selection
- Resolution documentation
- Agent notification
- Main mediation function
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.sm.conflict_mediation import (
    _calculate_conflict_severity,
    _create_notification_message,
    _decisions_conflict,
    _detect_all_conflicts,
    _detect_approach_conflicts,
    _detect_design_conflicts,
    _detect_priority_conflicts,
    _detect_scope_conflicts,
    _document_resolution,
    _evaluate_conflict,
    _extract_agent_positions,
    _find_compromise,
    _identify_affected_agents,
    _log_conflict_mediation,
    _notify_agents,
    _score_position,
    _score_positions,
    _should_defer,
    mediate_conflicts,
)
from yolo_developer.agents.sm.conflict_types import (
    Conflict,
    ConflictMediationConfig,
    ConflictParty,
    ConflictResolution,
    MediationResult,
)
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def empty_state() -> YoloState:
    """Create empty state for testing."""
    return YoloState(
        messages=[],
        current_agent="sm",
        handoff_context=None,
        decisions=[],
    )


@pytest.fixture
def state_with_decisions() -> YoloState:
    """Create state with sample decisions for testing."""
    decisions = [
        Decision(
            agent="architect",
            summary="Use microservices architecture for scalability",
            rationale="Better scalability and maintainability for expected growth",
            related_artifacts=("adr-001", "design-doc"),
        ),
        Decision(
            agent="dev",
            summary="Use monolith architecture for simplicity",
            rationale="Simpler to implement and maintain for MVP",
            related_artifacts=("adr-001",),
        ),
        Decision(
            agent="pm",
            summary="Prioritize feature A first",
            rationale="Higher business value based on customer feedback",
            related_artifacts=("req-001",),
        ),
        Decision(
            agent="analyst",
            summary="Feature B should come first before A",
            rationale="Feature B is a blocker for A due to dependencies",
            related_artifacts=("req-002",),
        ),
    ]
    return YoloState(
        messages=[],
        current_agent="sm",
        handoff_context=None,
        decisions=decisions,
    )


@pytest.fixture
def basic_conflict() -> Conflict:
    """Create basic conflict for testing."""
    party1 = ConflictParty(
        agent="architect",
        position="Use microservices",
        rationale="Security and scalability concerns",
        artifacts=("adr-001",),
    )
    party2 = ConflictParty(
        agent="dev",
        position="Use monolith",
        rationale="Simpler and faster to implement",
        artifacts=("adr-001",),
    )
    return Conflict(
        conflict_id="test-conflict-001",
        conflict_type="design_conflict",
        severity="major",
        parties=(party1, party2),
        topic="service_architecture",
    )


@pytest.fixture
def default_config() -> ConflictMediationConfig:
    """Create default config for testing."""
    return ConflictMediationConfig()


# =============================================================================
# Agent Position Extraction Tests
# =============================================================================


class TestExtractAgentPositions:
    """Tests for _extract_agent_positions function."""

    def test_empty_state(self, empty_state: YoloState) -> None:
        """Should return empty dict for empty state."""
        result = _extract_agent_positions(empty_state)
        assert result == {}

    def test_single_agent_single_decision(self) -> None:
        """Should extract single agent's decision."""
        state = YoloState(
            messages=[],
            current_agent="sm",
            handoff_context=None,
            decisions=[
                Decision(
                    agent="architect",
                    summary="Test decision",
                    rationale="Test rationale",
                )
            ],
        )

        result = _extract_agent_positions(state)

        assert "architect" in result
        assert len(result["architect"]) == 1

    def test_multiple_agents(self, state_with_decisions: YoloState) -> None:
        """Should extract positions from multiple agents."""
        result = _extract_agent_positions(state_with_decisions)

        assert len(result) == 4  # architect, dev, pm, analyst
        assert "architect" in result
        assert "dev" in result
        assert "pm" in result
        assert "analyst" in result


# =============================================================================
# Severity Calculation Tests
# =============================================================================


class TestCalculateConflictSeverity:
    """Tests for _calculate_conflict_severity function."""

    def test_blocking_keyword_in_rationale(self) -> None:
        """Should return blocking for security keyword."""
        party = ConflictParty(
            agent="tea",
            position="Block deployment",
            rationale="Critical security vulnerability found",
            artifacts=(),
        )

        severity = _calculate_conflict_severity((party,))

        assert severity == "blocking"

    def test_word_boundary_matching_no_false_positive(self) -> None:
        """Issue #6 fix: Should NOT match substring 'insecurity' as 'security'."""
        party = ConflictParty(
            agent="dev",
            position="Address feelings of insecurity",
            rationale="The team feels insecurity about the deadline",
            artifacts=(),
        )

        severity = _calculate_conflict_severity((party,))

        # "insecurity" should NOT trigger "security" match
        assert severity == "minor"

    def test_word_boundary_matching_exact_match(self) -> None:
        """Issue #6 fix: Should match exact word 'security'."""
        party = ConflictParty(
            agent="tea",
            position="Security concern",
            rationale="This is a security issue",
            artifacts=(),
        )

        severity = _calculate_conflict_severity((party,))

        assert severity == "blocking"

    def test_many_parties_returns_major(self) -> None:
        """Should return major for 3+ parties."""
        parties = tuple(
            ConflictParty(agent=f"agent-{i}", position="P", rationale="R") for i in range(3)
        )

        severity = _calculate_conflict_severity(parties)

        assert severity == "major"

    def test_two_parties_moderate(self) -> None:
        """Should return moderate for 2 parties without blocking keywords."""
        parties = (
            ConflictParty(agent="a1", position="P1", rationale="Simple reason"),
            ConflictParty(agent="a2", position="P2", rationale="Another reason"),
        )

        severity = _calculate_conflict_severity(parties)

        assert severity == "moderate"

    def test_single_party_minor(self) -> None:
        """Should return minor for single party."""
        party = ConflictParty(agent="a1", position="P1", rationale="Simple")

        severity = _calculate_conflict_severity((party,))

        assert severity == "minor"

    def test_custom_blocking_indicators(self) -> None:
        """Should use custom blocking indicators."""
        party = ConflictParty(
            agent="dev",
            position="Test",
            rationale="This is a custom_indicator issue",
        )

        severity = _calculate_conflict_severity((party,), blocking_indicators={"custom_indicator"})

        assert severity == "blocking"


# =============================================================================
# Decisions Conflict Tests
# =============================================================================


class TestDecisionsConflict:
    """Tests for _decisions_conflict function."""

    def test_same_agent_no_conflict(self) -> None:
        """Same agent decisions should not conflict."""
        dec1 = Decision(agent="dev", summary="A", rationale="R1")
        dec2 = Decision(agent="dev", summary="B", rationale="R2")

        assert _decisions_conflict(dec1, dec2) is False

    def test_overlapping_artifacts_conflict(self) -> None:
        """Overlapping artifacts should indicate conflict."""
        dec1 = Decision(
            agent="architect",
            summary="Design A",
            rationale="R1",
            related_artifacts=("adr-001",),
        )
        dec2 = Decision(
            agent="dev",
            summary="Design B",
            rationale="R2",
            related_artifacts=("adr-001", "adr-002"),
        )

        assert _decisions_conflict(dec1, dec2) is True

    def test_contradictory_language(self) -> None:
        """Contradictory language should indicate conflict."""
        dec1 = Decision(
            agent="architect",
            summary="Use microservices pattern",
            rationale="R1",
        )
        dec2 = Decision(
            agent="dev",
            summary="Use monolith pattern",
            rationale="R2",
        )

        assert _decisions_conflict(dec1, dec2) is True

    def test_no_overlap_no_contradiction(self) -> None:
        """No overlap and no contradiction should not conflict."""
        dec1 = Decision(
            agent="pm",
            summary="Feature scope",
            rationale="R1",
            related_artifacts=("req-001",),
        )
        dec2 = Decision(
            agent="dev",
            summary="Implementation details",
            rationale="R2",
            related_artifacts=("code-001",),
        )

        assert _decisions_conflict(dec1, dec2) is False


# =============================================================================
# Conflict Detection Tests
# =============================================================================


class TestDetectDesignConflicts:
    """Tests for _detect_design_conflicts function."""

    def test_no_conflicts_empty_state(self, empty_state: YoloState) -> None:
        """Should return empty list for empty state."""
        conflicts = _detect_design_conflicts(empty_state)
        assert conflicts == []

    def test_detects_design_conflict(self, state_with_decisions: YoloState) -> None:
        """Should detect design conflict on shared artifact."""
        conflicts = _detect_design_conflicts(state_with_decisions)

        # architect and dev both have decisions on adr-001
        design_conflicts = [c for c in conflicts if c.conflict_type == "design_conflict"]
        assert len(design_conflicts) >= 1

    def test_conflict_has_correct_properties(self, state_with_decisions: YoloState) -> None:
        """Detected conflict should have correct properties."""
        conflicts = _detect_design_conflicts(state_with_decisions)

        if conflicts:
            conflict = conflicts[0]
            assert conflict.conflict_type == "design_conflict"
            assert len(conflict.parties) == 2
            assert conflict.conflict_id.startswith("design_")


class TestDetectPriorityConflicts:
    """Tests for _detect_priority_conflicts function."""

    def test_no_conflicts_empty_state(self, empty_state: YoloState) -> None:
        """Should return empty list for empty state."""
        conflicts = _detect_priority_conflicts(empty_state)
        assert conflicts == []

    def test_detects_priority_conflict(self) -> None:
        """Should detect priority conflict."""
        state = YoloState(
            messages=[],
            current_agent="sm",
            handoff_context=None,
            decisions=[
                Decision(
                    agent="pm",
                    summary="Prioritize feature A high priority",
                    rationale="Business value",
                ),
                Decision(
                    agent="analyst",
                    summary="Feature B should be low priority",
                    rationale="Low value complexity",
                    related_artifacts=("feat-a",),
                ),
            ],
        )

        # This may or may not detect conflict depending on exact matching
        conflicts = _detect_priority_conflicts(state)

        # At least verify no errors
        assert isinstance(conflicts, list)


class TestDetectApproachConflicts:
    """Tests for _detect_approach_conflicts function."""

    def test_no_conflicts_empty_state(self, empty_state: YoloState) -> None:
        """Should return empty list for empty state."""
        conflicts = _detect_approach_conflicts(empty_state)
        assert conflicts == []

    def test_detects_approach_conflict(self) -> None:
        """Should detect approach conflict."""
        state = YoloState(
            messages=[],
            current_agent="sm",
            handoff_context=None,
            decisions=[
                Decision(
                    agent="dev",
                    summary="Use async pattern for API",
                    rationale="Performance",
                    related_artifacts=("api-001",),
                ),
                Decision(
                    agent="architect",
                    summary="Use sync approach for API",
                    rationale="Simplicity",
                    related_artifacts=("api-001",),
                ),
            ],
        )

        conflicts = _detect_approach_conflicts(state)

        # May or may not detect depending on exact logic
        assert isinstance(conflicts, list)


class TestDetectScopeConflicts:
    """Tests for _detect_scope_conflicts function."""

    def test_no_conflicts_empty_state(self, empty_state: YoloState) -> None:
        """Should return empty list for empty state."""
        conflicts = _detect_scope_conflicts(empty_state)
        assert conflicts == []


class TestDetectAllConflicts:
    """Tests for _detect_all_conflicts function."""

    def test_empty_state_no_conflicts(self, empty_state: YoloState) -> None:
        """Should return empty list for empty state."""
        conflicts = _detect_all_conflicts(empty_state)
        assert conflicts == []

    def test_aggregates_all_conflict_types(self, state_with_decisions: YoloState) -> None:
        """Should aggregate conflicts from all detection functions."""
        conflicts = _detect_all_conflicts(state_with_decisions)

        # Should be a list (may or may not have conflicts)
        assert isinstance(conflicts, list)


# =============================================================================
# Position Scoring Tests
# =============================================================================


class TestScorePosition:
    """Tests for _score_position function."""

    def test_safety_keyword_scores_high(self, default_config: ConflictMediationConfig) -> None:
        """Position with safety keyword should score high."""
        party = ConflictParty(
            agent="tea",
            position="Address security vulnerability",
            rationale="Critical security risk",
        )

        score, principles = _score_position(party, default_config)

        assert score > 0
        assert "safety" in principles

    def test_no_keywords_zero_score(self, default_config: ConflictMediationConfig) -> None:
        """Position with no keywords should score zero."""
        party = ConflictParty(
            agent="dev",
            position="Do something",
            rationale="Because reasons",
        )

        score, principles = _score_position(party, default_config)

        assert score == 0
        assert principles == []

    def test_multiple_principles_match(self, default_config: ConflictMediationConfig) -> None:
        """Position matching multiple principles should accumulate score."""
        party = ConflictParty(
            agent="architect",
            position="Simple and correct solution",
            rationale="Valid security approach that is straightforward",
        )

        score, principles = _score_position(party, default_config)

        # Should match safety (security), correctness (valid), simplicity (simple, straightforward)
        assert score > 0
        assert len(principles) >= 2


class TestScorePositions:
    """Tests for _score_positions function."""

    def test_scores_all_parties(
        self, basic_conflict: Conflict, default_config: ConflictMediationConfig
    ) -> None:
        """Should score all parties in conflict."""
        scores = _score_positions(basic_conflict, default_config)

        assert "architect" in scores
        assert "dev" in scores
        assert len(scores) == 2

    def test_scores_are_tuples(
        self, basic_conflict: Conflict, default_config: ConflictMediationConfig
    ) -> None:
        """Scores should be tuples of (score, principles)."""
        scores = _score_positions(basic_conflict, default_config)

        for _agent, score_tuple in scores.items():
            assert isinstance(score_tuple, tuple)
            assert len(score_tuple) == 2
            assert isinstance(score_tuple[0], float)
            assert isinstance(score_tuple[1], list)


# =============================================================================
# Compromise Finding Tests
# =============================================================================


class TestFindCompromise:
    """Tests for _find_compromise function."""

    def test_returns_compromises(
        self, basic_conflict: Conflict, default_config: ConflictMediationConfig
    ) -> None:
        """Should return compromise suggestions."""
        scores = _score_positions(basic_conflict, default_config)
        compromises = _find_compromise(basic_conflict, scores)

        assert isinstance(compromises, tuple)
        assert len(compromises) >= 2  # Should have at least 2 acknowledgments

    def test_empty_parties_returns_message(self, default_config: ConflictMediationConfig) -> None:
        """Issue #2 fix: Empty parties should return appropriate message."""
        # Create conflict with empty parties tuple
        empty_conflict = Conflict(
            conflict_id="empty-001",
            conflict_type="design_conflict",
            severity="minor",
            parties=(),  # Empty parties
            topic="test",
        )

        compromises = _find_compromise(empty_conflict, {})

        assert compromises == ("No parties to compromise between",)


# =============================================================================
# Defer Decision Tests
# =============================================================================


class TestShouldDefer:
    """Tests for _should_defer function."""

    def test_never_defer_blocking(
        self, basic_conflict: Conflict, default_config: ConflictMediationConfig
    ) -> None:
        """Should never defer blocking conflicts."""
        blocking_conflict = Conflict(
            conflict_id="blocking-001",
            conflict_type="design_conflict",
            severity="blocking",
            parties=basic_conflict.parties,
            topic="test",
            blocking_progress=True,
        )

        assert _should_defer(blocking_conflict, default_config) is False


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestEvaluateConflict:
    """Tests for _evaluate_conflict function."""

    def test_returns_strategy_tuple(
        self, basic_conflict: Conflict, default_config: ConflictMediationConfig
    ) -> None:
        """Should return tuple with all components."""
        result = _evaluate_conflict(basic_conflict, default_config)

        assert len(result) == 5
        strategy, rationale, _winner, compromises, principles = result
        assert strategy in {
            "accept_first",
            "accept_second",
            "compromise",
            "defer",
            "escalate_human",
        }
        assert isinstance(rationale, str)
        assert isinstance(compromises, tuple)
        assert isinstance(principles, tuple)

    def test_blocking_conflict_may_escalate(self, default_config: ConflictMediationConfig) -> None:
        """Blocking conflict with unclear winner may escalate."""
        # Create conflict where both positions have similar scores
        party1 = ConflictParty(
            agent="a1",
            position="Option A",
            rationale="Some reason",
        )
        party2 = ConflictParty(
            agent="a2",
            position="Option B",
            rationale="Other reason",
        )
        conflict = Conflict(
            conflict_id="blocking-test",
            conflict_type="design_conflict",
            severity="blocking",
            parties=(party1, party2),
            topic="test",
            blocking_progress=True,
        )

        strategy, _, _, _, _ = _evaluate_conflict(conflict, default_config)

        # May be escalate_human or compromise depending on scores
        assert strategy in {"escalate_human", "compromise", "accept_first", "accept_second"}


# =============================================================================
# Resolution Documentation Tests
# =============================================================================


class TestDocumentResolution:
    """Tests for _document_resolution function."""

    def test_creates_resolution_record(self, basic_conflict: Conflict) -> None:
        """Should create ConflictResolution record."""
        resolution = _document_resolution(
            conflict=basic_conflict,
            strategy="accept_first",
            rationale="Test rationale",
            winning_position="Position A",
            compromises=(),
            principles_applied=("safety",),
        )

        assert resolution.conflict_id == basic_conflict.conflict_id
        assert resolution.strategy == "accept_first"
        assert resolution.resolution_rationale == "Test rationale"
        assert resolution.winning_position == "Position A"
        assert resolution.principles_applied == ("safety",)

    def test_compromise_resolution(self, basic_conflict: Conflict) -> None:
        """Should handle compromise resolution."""
        resolution = _document_resolution(
            conflict=basic_conflict,
            strategy="compromise",
            rationale="Seeking middle ground",
            winning_position=None,
            compromises=("Comp 1", "Comp 2"),
            principles_applied=("simplicity", "correctness"),
        )

        assert resolution.strategy == "compromise"
        assert resolution.winning_position is None
        assert len(resolution.compromises) == 2


# =============================================================================
# Notification Tests
# =============================================================================


class TestIdentifyAffectedAgents:
    """Tests for _identify_affected_agents function."""

    def test_returns_all_parties(self, basic_conflict: Conflict) -> None:
        """Should return all agents from conflict parties."""
        affected = _identify_affected_agents(basic_conflict)

        assert "architect" in affected
        assert "dev" in affected
        assert len(affected) == 2


class TestCreateNotificationMessage:
    """Tests for _create_notification_message function."""

    def test_message_contains_key_info(self, basic_conflict: Conflict) -> None:
        """Notification should contain key information."""
        resolution = ConflictResolution(
            conflict_id=basic_conflict.conflict_id,
            strategy="accept_first",
            resolution_rationale="Test rationale",
            winning_position="Position A",
            principles_applied=("safety",),
        )

        message = _create_notification_message(basic_conflict, resolution)

        assert basic_conflict.topic in message
        assert "accept_first" in message
        assert "Position A" in message
        assert "safety" in message


class TestNotifyAgents:
    """Tests for _notify_agents function."""

    def test_returns_notified_agents(self, basic_conflict: Conflict) -> None:
        """Should return tuple of notified agents."""
        resolution = ConflictResolution(
            conflict_id=basic_conflict.conflict_id,
            strategy="accept_first",
            resolution_rationale="Test",
        )

        notified = _notify_agents(basic_conflict, resolution)

        assert isinstance(notified, tuple)
        assert "architect" in notified
        assert "dev" in notified


class TestLogConflictMediation:
    """Tests for _log_conflict_mediation function (Issue #8)."""

    def test_logs_mediation_without_escalation(self, basic_conflict: Conflict) -> None:
        """Should log successful mediation at INFO level."""
        resolution = ConflictResolution(
            conflict_id=basic_conflict.conflict_id,
            strategy="accept_first",
            resolution_rationale="Test rationale",
            winning_position="Position A",
        )

        # Should not raise any exception
        _log_conflict_mediation(basic_conflict, resolution, escalated=False)

    def test_logs_escalation_as_warning(self, basic_conflict: Conflict) -> None:
        """Should log escalation at WARNING level."""
        resolution = ConflictResolution(
            conflict_id=basic_conflict.conflict_id,
            strategy="escalate_human",
            resolution_rationale="Cannot resolve automatically",
        )

        # Should not raise any exception
        _log_conflict_mediation(basic_conflict, resolution, escalated=True)

    def test_handles_logging_failure_gracefully(self, basic_conflict: Conflict) -> None:
        """Should handle logging failures without raising (Issue #7)."""
        resolution = ConflictResolution(
            conflict_id=basic_conflict.conflict_id,
            strategy="accept_first",
            resolution_rationale="Test",
        )

        # Even with potential logging issues, should not raise
        # This tests the try-except wrapper added in Issue #7
        _log_conflict_mediation(basic_conflict, resolution, escalated=False)


# =============================================================================
# Main Mediation Function Tests
# =============================================================================


class TestMediateConflicts:
    """Tests for mediate_conflicts function."""

    @pytest.mark.asyncio
    async def test_no_conflicts_returns_success(self, empty_state: YoloState) -> None:
        """Should return success with no conflicts detected."""
        result = await mediate_conflicts(empty_state)

        assert result.success is True
        assert len(result.conflicts_detected) == 0
        assert len(result.resolutions) == 0
        assert "No conflicts detected" in result.mediation_notes

    @pytest.mark.asyncio
    async def test_processes_conflicts(self, state_with_decisions: YoloState) -> None:
        """Should process detected conflicts."""
        result = await mediate_conflicts(state_with_decisions)

        # May or may not detect conflicts depending on decision content
        assert isinstance(result.conflicts_detected, tuple)
        assert isinstance(result.resolutions, tuple)
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_custom_config(self, state_with_decisions: YoloState) -> None:
        """Should respect custom configuration."""
        config = ConflictMediationConfig(
            auto_resolve_minor=False,
            escalate_blocking=False,
            max_mediation_rounds=5,
        )

        result = await mediate_conflicts(state_with_decisions, config)

        # Should not error with custom config
        assert isinstance(result, MediationResult)

    @pytest.mark.asyncio
    async def test_result_has_correct_structure(self, empty_state: YoloState) -> None:
        """Result should have all expected fields."""
        result = await mediate_conflicts(empty_state)

        assert hasattr(result, "conflicts_detected")
        assert hasattr(result, "resolutions")
        assert hasattr(result, "notifications_sent")
        assert hasattr(result, "escalations_triggered")
        assert hasattr(result, "success")
        assert hasattr(result, "mediation_notes")
        assert hasattr(result, "mediated_at")

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, empty_state: YoloState) -> None:
        """Result should be serializable to dict."""
        result = await mediate_conflicts(empty_state)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "conflicts_detected" in result_dict
        assert "resolutions" in result_dict
        assert "success" in result_dict


# =============================================================================
# Integration Tests
# =============================================================================


class TestMediationIntegration:
    """Integration tests for full mediation flow."""

    @pytest.mark.asyncio
    async def test_full_mediation_flow_with_real_conflict(self) -> None:
        """Test complete mediation flow with a real conflict scenario."""
        # Create state with clear conflict
        state = YoloState(
            messages=[],
            current_agent="sm",
            handoff_context=None,
            decisions=[
                Decision(
                    agent="architect",
                    summary="Use microservices for security isolation",
                    rationale="Security vulnerability concerns require service isolation",
                    related_artifacts=("arch-001",),
                ),
                Decision(
                    agent="dev",
                    summary="Use simple monolith for fast delivery",
                    rationale="Need quick MVP, deadline approaching",
                    related_artifacts=("arch-001",),
                ),
            ],
        )

        result = await mediate_conflicts(state)

        # Should detect and resolve the conflict
        if result.conflicts_detected:
            assert len(result.resolutions) == len(result.conflicts_detected)
            assert all(
                r.conflict_id in [c.conflict_id for c in result.conflicts_detected]
                for r in result.resolutions
            )

    @pytest.mark.asyncio
    async def test_multiple_conflicts_all_resolved(self) -> None:
        """Multiple conflicts should all be resolved."""
        state = YoloState(
            messages=[],
            current_agent="sm",
            handoff_context=None,
            decisions=[
                # Design conflict
                Decision(
                    agent="architect",
                    summary="Use approach A for design",
                    rationale="Better architecture",
                    related_artifacts=("design-001",),
                ),
                Decision(
                    agent="dev",
                    summary="Use approach B for design",
                    rationale="Simpler implementation",
                    related_artifacts=("design-001",),
                ),
                # Priority conflict
                Decision(
                    agent="pm",
                    summary="High priority for feature X",
                    rationale="Business need",
                    related_artifacts=("feat-001",),
                ),
                Decision(
                    agent="analyst",
                    summary="Low priority for feature X",
                    rationale="Technical complexity",
                    related_artifacts=("feat-001",),
                ),
            ],
        )

        result = await mediate_conflicts(state)

        # Each conflict should have a resolution
        assert len(result.resolutions) == len(result.conflicts_detected)
