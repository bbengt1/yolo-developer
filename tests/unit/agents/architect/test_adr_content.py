"""Tests for ADR content generation (Story 7.3, Tasks 1-3).

Tests verify enhanced ADR content generation with 12-Factor analysis integration.
"""

from __future__ import annotations

from yolo_developer.agents.architect.types import DesignDecision, FactorResult, TwelveFactorAnalysis


def _create_test_decision(
    decision_id: str = "design-001",
    story_id: str = "story-001",
    decision_type: str = "technology",
    description: str = "Use PostgreSQL for persistence",
    rationale: str = "ACID compliance required",
    alternatives: tuple[str, ...] = ("MySQL", "MongoDB"),
) -> DesignDecision:
    """Create a test design decision."""
    return DesignDecision(
        id=decision_id,
        story_id=story_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        description=description,
        rationale=rationale,
        alternatives_considered=alternatives,
    )


def _create_test_twelve_factor_analysis(
    overall_compliance: float = 0.85,
    recommendations: tuple[str, ...] = ("Externalize config", "Use env vars"),
) -> TwelveFactorAnalysis:
    """Create a test 12-Factor analysis."""
    return TwelveFactorAnalysis(
        factor_results={
            "config": FactorResult(
                factor_name="config",
                applies=True,
                compliant=False,
                finding="Hardcoded config detected",
                recommendation="Use environment variables",
            )
        },
        applicable_factors=("config",),
        overall_compliance=overall_compliance,
        recommendations=recommendations,
    )


class TestGenerateAdrContext:
    """Test _generate_adr_context function."""

    def test_context_includes_story_reference(self) -> None:
        """Test that context mentions the story ID."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_context

        decision = _create_test_decision(story_id="story-001")
        analysis = _create_test_twelve_factor_analysis()

        context = _generate_adr_context(decision, analysis)

        assert "story-001" in context

    def test_context_includes_decision_description(self) -> None:
        """Test that context includes the decision description."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_context

        decision = _create_test_decision(description="Use PostgreSQL for persistence")
        analysis = _create_test_twelve_factor_analysis()

        context = _generate_adr_context(decision, analysis)

        assert "PostgreSQL" in context or "persistence" in context

    def test_context_includes_twelve_factor_compliance(self) -> None:
        """Test that context includes 12-Factor compliance percentage."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_context

        decision = _create_test_decision()
        analysis = _create_test_twelve_factor_analysis(overall_compliance=0.85)

        context = _generate_adr_context(decision, analysis)

        assert "85%" in context or "12-Factor" in context.lower() or "twelve" in context.lower()

    def test_context_without_analysis(self) -> None:
        """Test that context works without 12-Factor analysis."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_context

        decision = _create_test_decision()

        context = _generate_adr_context(decision, None)

        assert len(context) > 0
        assert decision.story_id in context


class TestGenerateAdrDecision:
    """Test _generate_adr_decision function."""

    def test_decision_includes_rationale(self) -> None:
        """Test that decision includes the rationale."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_decision

        decision = _create_test_decision(rationale="ACID compliance required")

        adr_decision = _generate_adr_decision(decision)

        assert "ACID" in adr_decision or "compliance" in adr_decision

    def test_decision_includes_chosen_approach(self) -> None:
        """Test that decision clearly states what was chosen."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_decision

        decision = _create_test_decision(description="Use PostgreSQL for persistence")

        adr_decision = _generate_adr_decision(decision)

        assert len(adr_decision) > 0

    def test_decision_formatted_properly(self) -> None:
        """Test that decision is properly formatted."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_decision

        decision = _create_test_decision()

        adr_decision = _generate_adr_decision(decision)

        # Should be non-empty and properly formatted
        assert isinstance(adr_decision, str)
        assert len(adr_decision) > 10


class TestGenerateAdrConsequences:
    """Test _generate_adr_consequences function."""

    def test_consequences_includes_positive_effects(self) -> None:
        """Test that consequences mention positive effects."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_consequences

        decision = _create_test_decision()
        analysis = _create_test_twelve_factor_analysis()

        consequences = _generate_adr_consequences(decision, analysis)

        # Should mention positive or pro or benefit
        assert any(word in consequences.lower() for word in ["positive", "pro", "benefit", "good", "advantage"])

    def test_consequences_includes_negative_effects(self) -> None:
        """Test that consequences mention negative effects or trade-offs."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_consequences

        decision = _create_test_decision()
        analysis = _create_test_twelve_factor_analysis()

        consequences = _generate_adr_consequences(decision, analysis)

        # Should mention negative or con or trade-off
        assert any(word in consequences.lower() for word in ["negative", "con", "trade-off", "tradeoff", "cost", "complexity"])

    def test_consequences_includes_recommendations_when_not_compliant(self) -> None:
        """Test that consequences include 12-Factor recommendations when compliance < 100%."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_consequences

        decision = _create_test_decision()
        analysis = _create_test_twelve_factor_analysis(
            overall_compliance=0.75,
            recommendations=("Use environment variables",),
        )

        consequences = _generate_adr_consequences(decision, analysis)

        assert "environment" in consequences.lower() or "recommendation" in consequences.lower()

    def test_consequences_without_analysis(self) -> None:
        """Test that consequences work without 12-Factor analysis."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_consequences

        decision = _create_test_decision()

        consequences = _generate_adr_consequences(decision, None)

        assert len(consequences) > 0


class TestDocumentAlternatives:
    """Test _document_alternatives function (Task 2)."""

    def test_alternatives_listed(self) -> None:
        """Test that alternatives are listed in output."""
        from yolo_developer.agents.architect.adr_generator import _document_alternatives

        decision = _create_test_decision(alternatives=("MySQL", "MongoDB"))

        alternatives_doc = _document_alternatives(decision)

        assert "MySQL" in alternatives_doc
        assert "MongoDB" in alternatives_doc

    def test_alternatives_include_pros_cons(self) -> None:
        """Test that alternatives documentation includes pros/cons analysis."""
        from yolo_developer.agents.architect.adr_generator import _document_alternatives

        decision = _create_test_decision(alternatives=("MySQL", "MongoDB"))

        alternatives_doc = _document_alternatives(decision)

        # Should have some analysis structure
        assert len(alternatives_doc) > len("MySQL, MongoDB")

    def test_empty_alternatives(self) -> None:
        """Test handling of empty alternatives."""
        from yolo_developer.agents.architect.adr_generator import _document_alternatives

        decision = _create_test_decision(alternatives=())

        alternatives_doc = _document_alternatives(decision)

        # Should indicate no alternatives or be empty
        assert "no alternatives" in alternatives_doc.lower() or alternatives_doc == ""


class TestGenerateAdrTitle:
    """Test _generate_adr_title function."""

    def test_title_is_descriptive(self) -> None:
        """Test that title is descriptive."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_title

        decision = _create_test_decision(
            decision_type="technology",
            description="Use PostgreSQL for persistence",
        )

        title = _generate_adr_title(decision)

        assert len(title) > 5
        assert "PostgreSQL" in title or "persistence" in title.lower()

    def test_title_includes_decision_type(self) -> None:
        """Test that title reflects decision type."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_title

        decision = _create_test_decision(decision_type="pattern")

        title = _generate_adr_title(decision)

        assert len(title) > 0

    def test_title_strips_redundant_prefix(self) -> None:
        """Test that title doesn't duplicate verb prefix."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_title

        # Description starts with "Use" which would be stripped
        decision = _create_test_decision(
            decision_type="technology",
            description="Use Redis for caching",
        )

        title = _generate_adr_title(decision)

        # Should be "Use Redis for caching", not "Use Use Redis for caching"
        assert title.count("Use") == 1
        assert "Redis" in title

    def test_title_handles_lowercase_prefix(self) -> None:
        """Test that title handles lowercase verb prefixes in description."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_title

        decision = _create_test_decision(
            decision_type="infrastructure",
            description="deploy containers to kubernetes",
        )

        title = _generate_adr_title(decision)

        # Should strip "deploy " and add "Deploy" prefix
        assert title.startswith("Deploy")
        assert "kubernetes" in title.lower()

    def test_title_unknown_decision_type(self) -> None:
        """Test title generation with unknown decision type."""
        from yolo_developer.agents.architect.adr_generator import _generate_adr_title

        decision = _create_test_decision(
            decision_type="unknown_type",  # type: ignore[arg-type]
            description="Custom approach for problem",
        )

        title = _generate_adr_title(decision)

        # Should use "Decide" as fallback prefix
        assert title.startswith("Decide")
        assert "Custom" in title
