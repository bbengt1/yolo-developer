"""Unit tests for Analyst agent node (Story 5.1 Task 2, Story 5.2, Story 5.3, 5.4, 5.5).

Tests for analyst_node function, state management, vague term detection,
gap analysis functions, requirement categorization, and implementability validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.agents.analyst.node import (
    CONSTRAINT_KEYWORDS,
    FUNCTIONAL_KEYWORDS,
    NON_FUNCTIONAL_KEYWORDS,
    _assess_complexity,
    _assign_sub_category,
    _calculate_category_confidence,
    _categorize_all_requirements,
    _categorize_requirement,
    _check_impossibility,
    _count_keyword_matches,
    _detect_vague_terms,
    _enhance_with_gap_analysis,
    _generate_remediation,
    _identify_edge_cases,
    _identify_external_dependencies,
    _identify_implied_requirements,
    _suggest_from_patterns,
    _validate_all_requirements,
    _validate_implementability,
)
from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    ComplexityLevel,
    CrystallizedRequirement,
    DependencyType,
    GapType,
    ImplementabilityStatus,
    RequirementCategory,
    Severity,
)
from yolo_developer.orchestrator.state import YoloState


class TestAnalystNode:
    """Tests for analyst_node function."""

    @pytest.fixture
    def mock_llm_output(self) -> AnalystOutput:
        """Mock AnalystOutput for testing."""
        return AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build a todo app",
                    refined_text="Create a web-based task management application",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

    @pytest.fixture
    def sample_state(self) -> YoloState:
        """Create a sample YoloState for testing."""
        return {
            "messages": [HumanMessage(content="Build a todo app with user authentication")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

    @pytest.mark.asyncio
    async def test_analyst_node_receives_yolo_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should receive YoloState TypedDict correctly."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        # Should return a dict, not raise any errors
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyst_node_returns_dict_not_yolo_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return dict with state updates, not YoloState."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        # Should be a plain dict
        assert isinstance(result, dict)

        # Should NOT have all YoloState fields (we only return updates)
        # Should NOT include current_agent (handoff does that)
        assert "current_agent" not in result

    @pytest.mark.asyncio
    async def test_analyst_node_returns_valid_messages(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return valid message updates."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 1

        # Message should have agent attribution
        msg = result["messages"][0]
        assert msg.additional_kwargs.get("agent") == "analyst"

    @pytest.mark.asyncio
    async def test_analyst_node_returns_valid_decisions(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should return valid decision updates."""
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            result = await analyst_node(sample_state)

        assert "decisions" in result
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) >= 1

        # Decision should have analyst attribution
        decision = result["decisions"][0]
        assert decision.agent == "analyst"

    @pytest.mark.asyncio
    async def test_analyst_node_does_not_mutate_state(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should not mutate the input state."""
        original_messages_len = len(sample_state["messages"])
        original_decisions_len = len(sample_state["decisions"])

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            await analyst_node(sample_state)

        # Original state should be unchanged
        assert len(sample_state["messages"]) == original_messages_len
        assert len(sample_state["decisions"]) == original_decisions_len

    @pytest.mark.asyncio
    async def test_analyst_node_is_async(
        self,
        sample_state: YoloState,
        mock_llm_output: AnalystOutput,
    ) -> None:
        """analyst_node should be an async function."""
        import asyncio
        import inspect

        from yolo_developer.agents.analyst import analyst_node

        # Verify it's a coroutine function
        assert inspect.iscoroutinefunction(analyst_node)

        # Verify it returns an awaitable
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_llm_output),
        ):
            coro = analyst_node(sample_state)
            assert asyncio.iscoroutine(coro)
            await coro  # Clean up

    @pytest.mark.asyncio
    async def test_analyst_node_handles_empty_messages(
        self,
    ) -> None:
        """analyst_node should handle empty messages gracefully."""
        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        mock_output = AnalystOutput(
            requirements=(),
            identified_gaps=("No seed content provided",),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        assert isinstance(result, dict)
        assert "messages" in result


class TestVagueTermDetection:
    """Tests for vague term detection (Story 5.2 Task 2)."""

    def test_detect_common_vague_terms(self) -> None:
        """Should detect common vague terms like fast, easy, simple."""
        text = "The system should be fast and easy to use"
        result = _detect_vague_terms(text)

        assert "should" in result
        assert "fast" in result
        assert "easy" in result

    def test_detect_quantifier_vagueness(self) -> None:
        """Should detect vague quantifiers like scalable, efficient."""
        text = "Build a scalable and efficient API that is performant"
        result = _detect_vague_terms(text)

        assert "scalable" in result
        assert "efficient" in result
        assert "performant" in result

    def test_detect_certainty_vagueness(self) -> None:
        """Should detect uncertainty terms like might, could, may."""
        text = "The feature might be useful and could possibly help users"
        result = _detect_vague_terms(text)

        assert "might" in result
        assert "could" in result
        assert "possibly" in result

    def test_detect_scope_vagueness(self) -> None:
        """Should detect scope vagueness like etc, various, several."""
        text = "Support various formats, several options, etc"
        result = _detect_vague_terms(text)

        assert "various" in result
        assert "several" in result
        assert "etc" in result

    def test_detect_quality_vagueness(self) -> None:
        """Should detect quality vagueness like good, clean, robust."""
        text = "Create a clean, robust solution with good performance"
        result = _detect_vague_terms(text)

        assert "clean" in result
        assert "robust" in result
        assert "good" in result

    def test_no_vague_terms_returns_empty(self) -> None:
        """Should return empty set when no vague terms present."""
        text = "API response time MUST be under 200ms at 95th percentile"
        result = _detect_vague_terms(text)

        assert result == set()

    def test_empty_text_returns_empty(self) -> None:
        """Should return empty set for empty text."""
        result = _detect_vague_terms("")
        assert result == set()

    def test_case_insensitive_detection(self) -> None:
        """Should detect vague terms case-insensitively."""
        text = "FAST response, Easy interface, SHOULD work"
        result = _detect_vague_terms(text)

        assert "fast" in result
        assert "easy" in result
        assert "should" in result

    def test_returns_set_type(self) -> None:
        """Should return a set of detected vague terms."""
        text = "The system should be fast"
        result = _detect_vague_terms(text)

        assert isinstance(result, set)

    def test_detects_compound_vague_phrases(self) -> None:
        """Should detect multi-word vague phrases."""
        text = "Make it user-friendly and straightforward"
        result = _detect_vague_terms(text)

        assert "user-friendly" in result
        assert "straightforward" in result

    def test_detects_real_time_vagueness(self) -> None:
        """Should detect 'real-time' as vague unless quantified."""
        text = "Provide real-time updates"
        result = _detect_vague_terms(text)

        assert "real-time" in result


class TestLLMResponseParsing:
    """Tests for _parse_llm_response with enhanced fields (Story 5.2 Task 8)."""

    def test_parse_response_with_new_fields(self) -> None:
        """Should parse response with scope_notes, hints, and confidence."""
        from yolo_developer.agents.analyst.node import _parse_llm_response

        response = """{
            "requirements": [{
                "id": "req-001",
                "original_text": "System should be fast",
                "refined_text": "Response time < 200ms at 95th percentile",
                "category": "non-functional",
                "testable": true,
                "scope_notes": "Applies to GET endpoints only",
                "implementation_hints": ["Use async handlers", "Add caching"],
                "confidence": 0.9
            }],
            "identified_gaps": [],
            "contradictions": []
        }"""

        result = _parse_llm_response(response)

        assert len(result.requirements) == 1
        req = result.requirements[0]
        assert req.scope_notes == "Applies to GET endpoints only"
        assert req.implementation_hints == ("Use async handlers", "Add caching")
        assert req.confidence == 0.9

    def test_parse_response_without_new_fields_uses_defaults(self) -> None:
        """Should use defaults when new fields are missing (backward compat)."""
        from yolo_developer.agents.analyst.node import _parse_llm_response

        response = """{
            "requirements": [{
                "id": "req-001",
                "original_text": "original",
                "refined_text": "refined",
                "category": "functional",
                "testable": true
            }],
            "identified_gaps": [],
            "contradictions": []
        }"""

        result = _parse_llm_response(response)

        req = result.requirements[0]
        assert req.scope_notes is None
        assert req.implementation_hints == ()
        assert req.confidence == 1.0

    def test_parse_response_with_null_scope_notes(self) -> None:
        """Should handle null scope_notes correctly."""
        from yolo_developer.agents.analyst.node import _parse_llm_response

        response = """{
            "requirements": [{
                "id": "req-001",
                "original_text": "orig",
                "refined_text": "ref",
                "category": "functional",
                "testable": true,
                "scope_notes": null,
                "implementation_hints": [],
                "confidence": 1.0
            }],
            "identified_gaps": [],
            "contradictions": []
        }"""

        result = _parse_llm_response(response)

        assert result.requirements[0].scope_notes is None

    def test_parse_response_confidence_conversion(self) -> None:
        """Should convert confidence to float correctly."""
        from yolo_developer.agents.analyst.node import _parse_llm_response

        response = """{
            "requirements": [{
                "id": "req-001",
                "original_text": "orig",
                "refined_text": "ref",
                "category": "functional",
                "testable": true,
                "confidence": 0.75
            }],
            "identified_gaps": [],
            "contradictions": []
        }"""

        result = _parse_llm_response(response)

        assert result.requirements[0].confidence == 0.75
        assert isinstance(result.requirements[0].confidence, float)


class TestCrystallizeRequirementsPlaceholder:
    """Tests for placeholder crystallization behavior (Story 5.2 Task 8)."""

    @pytest.mark.asyncio
    async def test_placeholder_includes_scope_notes_for_vague_content(self) -> None:
        """Placeholder should add scope_notes when vague terms detected."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Build a fast and scalable API")

        assert len(result.requirements) == 1
        req = result.requirements[0]
        assert req.scope_notes is not None
        assert "fast" in req.scope_notes.lower() or "scalable" in req.scope_notes.lower()

    @pytest.mark.asyncio
    async def test_placeholder_reduces_confidence_for_vague_content(self) -> None:
        """Placeholder should reduce confidence when vague terms detected."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements(
            "Build a fast, easy, scalable, robust system"
        )

        assert len(result.requirements) == 1
        # Multiple vague terms should reduce confidence
        assert result.requirements[0].confidence < 1.0

    @pytest.mark.asyncio
    async def test_placeholder_full_confidence_for_specific_content(self) -> None:
        """Placeholder should have full confidence for non-vague content."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Response time MUST be under 200ms")

        assert len(result.requirements) == 1
        assert result.requirements[0].confidence == 1.0
        assert result.requirements[0].scope_notes is None

    @pytest.mark.asyncio
    async def test_placeholder_generates_api_hints(self) -> None:
        """Placeholder should generate hints for API-related content."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Create an API endpoint for users")

        assert len(result.requirements) == 1
        assert len(result.requirements[0].implementation_hints) > 0
        assert "async" in result.requirements[0].implementation_hints[0].lower()

    @pytest.mark.asyncio
    async def test_placeholder_generates_ui_hints(self) -> None:
        """Placeholder should generate hints for UI-related content."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Build a user interface for settings")

        assert len(result.requirements) == 1
        assert len(result.requirements[0].implementation_hints) > 0
        assert "component" in result.requirements[0].implementation_hints[0].lower()

    @pytest.mark.asyncio
    async def test_placeholder_generates_data_hints(self) -> None:
        """Placeholder should generate hints for data-related content."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Store data in the database")

        assert len(result.requirements) == 1
        assert len(result.requirements[0].implementation_hints) > 0
        assert "repository" in result.requirements[0].implementation_hints[0].lower()

    @pytest.mark.asyncio
    async def test_placeholder_confidence_clamped_to_minimum(self) -> None:
        """Placeholder should clamp confidence to minimum 0.3 even with many vague terms."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        # Use many vague terms to push confidence below the floor
        # 10+ vague terms would try to set confidence to 0.0 or negative
        result = await _crystallize_requirements(
            "Build a fast, easy, simple, scalable, efficient, robust, "
            "clean, good, nice, beautiful, modern system"
        )

        assert len(result.requirements) == 1
        # Confidence should be clamped to minimum 0.3, not go below
        assert result.requirements[0].confidence >= 0.3
        assert result.requirements[0].confidence <= 1.0


class TestPromptAliases:
    """Tests for backward-compatible prompt aliases (Story 5.2)."""

    def test_refinement_aliases_exist(self) -> None:
        """REFINEMENT_ aliases should exist for backward compatibility."""
        from yolo_developer.agents.prompts.analyst import (
            ANALYST_SYSTEM_PROMPT,
            ANALYST_USER_PROMPT_TEMPLATE,
            REFINEMENT_SYSTEM_PROMPT,
            REFINEMENT_USER_PROMPT_TEMPLATE,
        )

        # Aliases should point to the same prompts
        assert REFINEMENT_SYSTEM_PROMPT is ANALYST_SYSTEM_PROMPT
        assert REFINEMENT_USER_PROMPT_TEMPLATE is ANALYST_USER_PROMPT_TEMPLATE


class TestEdgeCaseDetection:
    """Tests for _identify_edge_cases function (Story 5.3)."""

    def test_detects_input_validation_edge_cases(self) -> None:
        """Should detect edge cases for input-related requirements."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User can submit form data",
                refined_text="User submits form with required fields",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_edge_cases(reqs)

        assert len(gaps) > 0
        assert all(gap.gap_type == GapType.EDGE_CASE for gap in gaps)
        # Should detect empty input handling
        descriptions = [g.description.lower() for g in gaps]
        assert any("empty" in d or "null" in d for d in descriptions)

    def test_detects_api_error_edge_cases(self) -> None:
        """Should detect edge cases for API-related requirements."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="Call external API",
                refined_text="Integrate with payment service API",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_edge_cases(reqs)

        assert len(gaps) > 0
        # Should detect network/timeout error handling
        descriptions = [g.description.lower() for g in gaps]
        assert any("network" in d or "timeout" in d for d in descriptions)

    def test_edge_cases_have_source_requirements(self) -> None:
        """Each edge case should link to source requirement."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User enters data",
                refined_text="User provides input",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_edge_cases(reqs)

        for gap in gaps:
            assert len(gap.source_requirements) > 0
            assert "req-001" in gap.source_requirements

    def test_edge_cases_have_severity(self) -> None:
        """Each edge case should have a severity assigned."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User submits request",
                refined_text="API request handling",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_edge_cases(reqs)

        for gap in gaps:
            assert gap.severity in [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
            ]

    def test_empty_requirements_returns_empty(self) -> None:
        """Empty requirements should return empty gaps."""
        gaps = _identify_edge_cases(())
        assert gaps == ()


class TestImpliedRequirementDetection:
    """Tests for _identify_implied_requirements function (Story 5.3)."""

    def test_login_implies_logout(self) -> None:
        """Login requirement should imply logout functionality."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User can login to the system",
                refined_text="User authenticates with email/password",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_implied_requirements(reqs)

        assert len(gaps) > 0
        assert all(gap.gap_type == GapType.IMPLIED_REQUIREMENT for gap in gaps)
        descriptions = [g.description.lower() for g in gaps]
        assert any("logout" in d for d in descriptions)

    def test_save_implies_failure_handling(self) -> None:
        """Save operations should imply failure handling."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User can save their work",
                refined_text="Data is persisted to storage",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_implied_requirements(reqs)

        descriptions = [g.description.lower() for g in gaps]
        assert any("failure" in d or "warning" in d for d in descriptions)

    def test_delete_implies_confirmation(self) -> None:
        """Delete operations should imply confirmation."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User can delete items",
                refined_text="Items can be removed from the system",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_implied_requirements(reqs)

        descriptions = [g.description.lower() for g in gaps]
        assert any("confirmation" in d or "undo" in d for d in descriptions)

    def test_implied_requirements_include_rationale(self) -> None:
        """Implied requirements should have rationale."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login feature",
                refined_text="Authentication system",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_implied_requirements(reqs)

        for gap in gaps:
            assert gap.rationale
            assert len(gap.rationale) > 10  # Non-trivial rationale

    def test_no_duplicates_across_requirements(self) -> None:
        """Same implied requirement should not be duplicated."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login",
                refined_text="Login functionality",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Admin login",
                refined_text="Admin authentication",
                category="functional",
                testable=True,
            ),
        )

        gaps = _identify_implied_requirements(reqs)

        descriptions = [g.description.lower() for g in gaps]
        # Logout should only appear once
        logout_count = sum(1 for d in descriptions if "logout" in d)
        assert logout_count <= 1


class TestPatternSuggestions:
    """Tests for _suggest_from_patterns function (Story 5.3)."""

    def test_auth_domain_suggests_patterns(self) -> None:
        """Authentication requirements should suggest auth patterns."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User authentication",
                refined_text="Users authenticate with credentials",
                category="functional",
                testable=True,
            ),
        )

        gaps = _suggest_from_patterns(reqs)

        assert len(gaps) > 0
        assert all(gap.gap_type == GapType.PATTERN_SUGGESTION for gap in gaps)
        descriptions = [g.description.lower() for g in gaps]
        # Should suggest common auth patterns
        assert any(
            pattern in " ".join(descriptions)
            for pattern in ["registration", "mfa", "lockout", "reset"]
        )

    def test_crud_domain_suggests_patterns(self) -> None:
        """CRUD operations should suggest CRUD patterns."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="Create new records",
                refined_text="Users can create data entries",
                category="functional",
                testable=True,
            ),
        )

        gaps = _suggest_from_patterns(reqs)

        descriptions = [g.description.lower() for g in gaps]
        # Should suggest CRUD patterns like pagination, filtering
        assert any(
            pattern in " ".join(descriptions)
            for pattern in ["pagination", "filter", "delete", "update"]
        )

    def test_api_domain_suggests_patterns(self) -> None:
        """API requirements should suggest API patterns."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="Build REST API",
                refined_text="Implement RESTful endpoints",
                category="functional",
                testable=True,
            ),
        )

        gaps = _suggest_from_patterns(reqs)

        descriptions = [g.description.lower() for g in gaps]
        # Should suggest API patterns
        assert any(
            pattern in " ".join(descriptions)
            for pattern in ["rate", "version", "error", "validation"]
        )

    def test_pattern_suggestions_include_domain(self) -> None:
        """Pattern suggestions should reference domain in rationale."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login",
                refined_text="Authentication feature",
                category="functional",
                testable=True,
            ),
        )

        gaps = _suggest_from_patterns(reqs)

        for gap in gaps:
            assert "domain" in gap.rationale.lower() or "pattern" in gap.rationale.lower()


class TestGapAnalysisIntegration:
    """Integration tests for gap analysis in crystallize_requirements (Story 5.3)."""

    @pytest.mark.asyncio
    async def test_placeholder_generates_structured_gaps(self) -> None:
        """Placeholder crystallization should generate structured gaps."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Build user login system")

        assert len(result.structured_gaps) > 0
        # Should have different gap types
        gap_types = {g.gap_type for g in result.structured_gaps}
        assert len(gap_types) >= 1

    @pytest.mark.asyncio
    async def test_gaps_are_sorted_by_severity(self) -> None:
        """Structured gaps should be sorted by severity (critical first)."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements(
            "Build user authentication system with login and API endpoints"
        )

        if len(result.structured_gaps) > 1:
            severity_order = {
                Severity.CRITICAL: 0,
                Severity.HIGH: 1,
                Severity.MEDIUM: 2,
                Severity.LOW: 3,
            }
            for i in range(len(result.structured_gaps) - 1):
                current = severity_order[result.structured_gaps[i].severity]
                next_severity = severity_order[result.structured_gaps[i + 1].severity]
                assert current <= next_severity

    @pytest.mark.asyncio
    async def test_gaps_have_valid_ids(self) -> None:
        """Structured gaps should have valid gap IDs."""
        from yolo_developer.agents.analyst.node import _crystallize_requirements

        result = await _crystallize_requirements("Build API with user data")

        # All gaps should have IDs matching gap-XXX pattern
        for gap in result.structured_gaps:
            assert gap.id.startswith("gap-")
            assert len(gap.id) == 7  # gap-001, gap-002, etc.

    @pytest.mark.asyncio
    async def test_analyst_node_includes_gap_analysis(self) -> None:
        """analyst_node should include gap analysis in output."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build user login feature")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        # Check message additional_kwargs contains structured gaps in output
        # Metadata is spread directly into additional_kwargs, so "output" key exists
        msg = result["messages"][0]
        output_data = msg.additional_kwargs["output"]
        assert "structured_gaps" in output_data
        assert len(output_data["structured_gaps"]) > 0

    @pytest.mark.asyncio
    async def test_decision_includes_gap_severity_counts(self) -> None:
        """Decision rationale should include gap severity counts."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build user authentication")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        decision = result["decisions"][0]
        # Rationale should mention severity counts
        assert "critical" in decision.rationale.lower()
        assert "high" in decision.rationale.lower()


class TestEnhanceWithGapAnalysis:
    """Tests for _enhance_with_gap_analysis function (Story 5.3, 5.4)."""

    def test_enhances_output_with_gaps(self) -> None:
        """_enhance_with_gap_analysis should add structured_gaps to output.

        Story 5.4: Also verifies requirements are categorized.
        """
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates with email/password",
            category="functional",
            testable=True,
        )
        initial_output = AnalystOutput(
            requirements=(req,),
            identified_gaps=(),
            contradictions=(),
        )

        result = _enhance_with_gap_analysis(initial_output)

        assert len(result.structured_gaps) > 0
        # Story 5.4: Requirements are categorized, verify core fields match
        assert len(result.requirements) == 1
        enhanced_req = result.requirements[0]
        assert enhanced_req.id == req.id
        assert enhanced_req.original_text == req.original_text
        # Story 5.4: Categorization fields should be populated
        assert enhanced_req.category_rationale is not None
        assert enhanced_req.sub_category is not None  # "user_management" for login

    def test_empty_requirements_returns_unchanged(self) -> None:
        """Empty requirements should return output unchanged."""
        initial_output = AnalystOutput(
            requirements=(),
            identified_gaps=("legacy gap",),
            contradictions=(),
        )

        result = _enhance_with_gap_analysis(initial_output)

        assert result == initial_output
        assert len(result.structured_gaps) == 0

    def test_gaps_are_sorted_by_severity_before_numbering(self) -> None:
        """Gap IDs should be sequential by severity (gap-001 = highest severity)."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Build API with user authentication and database",
            refined_text="REST API with auth and data storage",
            category="functional",
            testable=True,
        )
        initial_output = AnalystOutput(
            requirements=(req,),
            identified_gaps=(),
            contradictions=(),
        )

        result = _enhance_with_gap_analysis(initial_output)

        # Verify gaps are sorted by severity
        if len(result.structured_gaps) > 1:
            severity_order = {
                Severity.CRITICAL: 0,
                Severity.HIGH: 1,
                Severity.MEDIUM: 2,
                Severity.LOW: 3,
            }
            for i in range(len(result.structured_gaps) - 1):
                current = severity_order[result.structured_gaps[i].severity]
                next_sev = severity_order[result.structured_gaps[i + 1].severity]
                assert current <= next_sev, (
                    f"Gap {result.structured_gaps[i].id} ({result.structured_gaps[i].severity}) "
                    f"should come before {result.structured_gaps[i + 1].id} ({result.structured_gaps[i + 1].severity})"
                )

    def test_gap_ids_are_sequential_after_sorting(self) -> None:
        """Gap IDs should be gap-001, gap-002, etc. in severity order."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User login with API integration",
            refined_text="Authentication system with external API",
            category="functional",
            testable=True,
        )
        initial_output = AnalystOutput(
            requirements=(req,),
            identified_gaps=(),
            contradictions=(),
        )

        result = _enhance_with_gap_analysis(initial_output)

        # Verify IDs are sequential
        for i, gap in enumerate(result.structured_gaps, start=1):
            expected_id = f"gap-{i:03d}"
            assert gap.id == expected_id, f"Expected {expected_id}, got {gap.id}"

    def test_preserves_existing_fields(self) -> None:
        """Should preserve core requirement fields, identified_gaps, and contradictions.

        Note: Story 5.4 adds categorization fields (sub_category, category_confidence,
        category_rationale) to requirements, so we verify core fields match.
        """
        req = CrystallizedRequirement(
            id="req-001",
            original_text="test",
            refined_text="test",
            category="functional",
            testable=True,
        )
        initial_output = AnalystOutput(
            requirements=(req,),
            identified_gaps=("legacy gap 1", "legacy gap 2"),
            contradictions=("contradiction 1",),
        )

        result = _enhance_with_gap_analysis(initial_output)

        # Story 5.4: Requirements are now categorized, so verify core fields match
        assert len(result.requirements) == len(initial_output.requirements)
        for orig, enhanced in zip(
            initial_output.requirements, result.requirements, strict=True
        ):
            assert enhanced.id == orig.id
            assert enhanced.original_text == orig.original_text
            assert enhanced.refined_text == orig.refined_text
            assert enhanced.category == orig.category  # May be updated by categorization
            assert enhanced.testable == orig.testable
            # Story 5.4: New categorization fields should be populated
            assert enhanced.category_rationale is not None  # Should have rationale
            assert 0.0 <= enhanced.category_confidence <= 1.0  # Valid confidence

        # Legacy fields should be preserved
        assert result.identified_gaps == initial_output.identified_gaps
        assert result.contradictions == initial_output.contradictions


# =============================================================================
# Story 5.4: Requirement Categorization Tests
# =============================================================================


class TestCountKeywordMatches:
    """Tests for _count_keyword_matches function (Story 5.4)."""

    def test_empty_text_returns_zero(self) -> None:
        """Empty text should return zero matches."""
        assert _count_keyword_matches("", FUNCTIONAL_KEYWORDS) == 0

    def test_single_keyword_match(self) -> None:
        """Should count a single keyword match."""
        count = _count_keyword_matches("User can create account", FUNCTIONAL_KEYWORDS)
        assert count >= 1  # "create" is a functional keyword

    def test_multiple_keyword_matches(self) -> None:
        """Should count multiple keywords."""
        text = "User can create, edit, and delete items"
        count = _count_keyword_matches(text, FUNCTIONAL_KEYWORDS)
        assert count >= 3  # "create", "edit", "delete"

    def test_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        count1 = _count_keyword_matches("User can LOGIN", FUNCTIONAL_KEYWORDS)
        count2 = _count_keyword_matches("user can login", FUNCTIONAL_KEYWORDS)
        assert count1 == count2

    def test_phrase_matching(self) -> None:
        """Should match multi-word phrases like 'user can'."""
        count = _count_keyword_matches("User can do something", FUNCTIONAL_KEYWORDS)
        assert count >= 1  # "user can" is a phrase keyword

    def test_non_functional_keywords(self) -> None:
        """Should detect non-functional keywords."""
        count = _count_keyword_matches(
            "Response time < 200ms for performance",
            NON_FUNCTIONAL_KEYWORDS,
        )
        assert count >= 2  # "response time", "performance"

    def test_constraint_keywords(self) -> None:
        """Should detect constraint keywords."""
        count = _count_keyword_matches(
            "Must use Python and AWS",
            CONSTRAINT_KEYWORDS,
        )
        assert count >= 2  # "must use", "python", "aws"


class TestCalculateCategoryConfidence:
    """Tests for _calculate_category_confidence function (Story 5.4)."""

    def test_no_keywords_low_confidence(self) -> None:
        """No keywords should result in low confidence."""
        confidence = _calculate_category_confidence("random text here", 0, 0, 0)
        assert confidence == 0.3

    def test_clear_functional_high_confidence(self) -> None:
        """Clear functional keywords should have high confidence."""
        confidence = _calculate_category_confidence(
            "User can create and delete",
            5, 0, 0,  # Clear functional winner
        )
        assert confidence >= 0.7

    def test_ambiguous_lower_confidence_than_clear_winner(self) -> None:
        """Ambiguous (tie) categories should have lower confidence than clear winner."""
        tie_confidence = _calculate_category_confidence(
            "Some text",
            2, 2, 0,  # Tie between functional and non-functional
        )
        clear_winner_confidence = _calculate_category_confidence(
            "Some text",
            5, 0, 0,  # Clear functional winner
        )
        # Tie should have lower confidence than clear winner (differentiation bonus)
        assert tie_confidence <= clear_winner_confidence

    def test_vague_terms_reduce_confidence(self) -> None:
        """Vague terms should reduce confidence."""
        # "fast" and "should" are vague terms
        confidence = _calculate_category_confidence(
            "System should be fast",
            2, 1, 0,
        )
        # Should be reduced due to vague terms
        assert confidence < 0.9

    def test_confidence_bounded(self) -> None:
        """Confidence should always be between 0.0 and 1.0."""
        # Very strong signal
        confidence = _calculate_category_confidence(
            "Clear requirement",
            10, 0, 0,
        )
        assert 0.0 <= confidence <= 1.0


class TestAssignSubCategory:
    """Tests for _assign_sub_category function (Story 5.4)."""

    def test_empty_text_returns_none(self) -> None:
        """Empty text should return None."""
        result = _assign_sub_category("", RequirementCategory.FUNCTIONAL)
        assert result is None

    def test_functional_user_management(self) -> None:
        """Login/user text should get user_management sub-category."""
        result = _assign_sub_category(
            "User can login with password",
            RequirementCategory.FUNCTIONAL,
        )
        assert result == "user_management"

    def test_functional_data_operations(self) -> None:
        """CRUD text should get data_operations sub-category."""
        result = _assign_sub_category(
            "Create and store records in database",
            RequirementCategory.FUNCTIONAL,
        )
        assert result == "data_operations"

    def test_functional_integration(self) -> None:
        """API text should get integration sub-category."""
        result = _assign_sub_category(
            "REST API endpoint for external service",
            RequirementCategory.FUNCTIONAL,
        )
        assert result == "integration"

    def test_non_functional_performance(self) -> None:
        """Response time text should get performance sub-category."""
        result = _assign_sub_category(
            "Fast response time under load",
            RequirementCategory.NON_FUNCTIONAL,
        )
        assert result == "performance"

    def test_non_functional_security(self) -> None:
        """Security text should get security sub-category."""
        result = _assign_sub_category(
            "Encrypt sensitive data securely",
            RequirementCategory.NON_FUNCTIONAL,
        )
        assert result == "security"

    def test_constraint_technical(self) -> None:
        """Tech stack text should get technical sub-category."""
        result = _assign_sub_category(
            "Must use Python and PostgreSQL",
            RequirementCategory.CONSTRAINT,
        )
        assert result == "technical"

    def test_constraint_regulatory(self) -> None:
        """Compliance text should get regulatory sub-category."""
        result = _assign_sub_category(
            "GDPR compliance required",
            RequirementCategory.CONSTRAINT,
        )
        assert result == "regulatory"

    def test_no_match_returns_none(self) -> None:
        """Text with no matching keywords should return None."""
        result = _assign_sub_category(
            "random words here",
            RequirementCategory.FUNCTIONAL,
        )
        assert result is None


class TestCategorizeRequirement:
    """Tests for _categorize_requirement function (Story 5.4)."""

    def test_functional_requirement_detection(self) -> None:
        """Should detect functional requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can create account",
            refined_text="User submits registration form",
            category="functional",
            testable=True,
        )
        result = _categorize_requirement(req)

        assert result.category == "functional"
        assert result.category_confidence > 0.0
        assert result.category_rationale is not None

    def test_non_functional_requirement_detection(self) -> None:
        """Should detect non-functional requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="System should be fast",
            refined_text="Response time < 200ms",
            category="non-functional",
            testable=True,
        )
        result = _categorize_requirement(req)

        assert result.category == "non_functional"
        assert result.sub_category == "performance"

    def test_constraint_requirement_detection(self) -> None:
        """Should detect constraint requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Must use Python",
            refined_text="Implementation required in Python 3.10+",
            category="constraint",
            testable=True,
        )
        result = _categorize_requirement(req)

        assert result.category == "constraint"
        assert result.sub_category == "technical"

    def test_preserves_original_fields(self) -> None:
        """Should preserve id, original_text, refined_text, etc."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Original",
            refined_text="Refined",
            category="functional",
            testable=True,
            scope_notes="Some notes",
            implementation_hints=("hint1",),
            confidence=0.9,
        )
        result = _categorize_requirement(req)

        assert result.id == req.id
        assert result.original_text == req.original_text
        assert result.refined_text == req.refined_text
        assert result.testable == req.testable
        assert result.scope_notes == req.scope_notes
        assert result.implementation_hints == req.implementation_hints
        assert result.confidence == req.confidence

    def test_rationale_includes_keywords(self) -> None:
        """Rationale should mention detected keywords."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates",
            category="functional",
            testable=True,
        )
        result = _categorize_requirement(req)

        assert "Keywords:" in result.category_rationale
        assert "Scores:" in result.category_rationale


class TestCategorizeAllRequirements:
    """Tests for _categorize_all_requirements function (Story 5.4)."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty input should return empty tuple."""
        result = _categorize_all_requirements(())
        assert result == ()

    def test_processes_all_requirements(self) -> None:
        """Should process all requirements in input."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login",
                refined_text="User authenticates",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Fast response",
                refined_text="< 200ms response",
                category="non-functional",
                testable=True,
            ),
        )
        result = _categorize_all_requirements(reqs)

        assert len(result) == 2
        assert all(r.category_rationale is not None for r in result)

    def test_maintains_order(self) -> None:
        """Output order should match input order."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="First",
                refined_text="First",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Second",
                refined_text="Second",
                category="functional",
                testable=True,
            ),
        )
        result = _categorize_all_requirements(reqs)

        assert result[0].id == "req-001"
        assert result[1].id == "req-002"

    def test_mixed_categories(self) -> None:
        """Should correctly categorize mixed requirement types."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User can login",
                refined_text="Authentication required",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Must be secure",
                refined_text="Encrypt all data",
                category="non-functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-003",
                original_text="Use Python",
                refined_text="Python 3.10+ required",
                category="constraint",
                testable=True,
            ),
        )
        result = _categorize_all_requirements(reqs)

        categories = {r.category for r in result}
        # All three category types should be present
        assert len(categories) == 3


# =============================================================================
# Story 5.5: Implementability Validation Tests
# =============================================================================


class TestCheckImpossibility:
    """Tests for _check_impossibility function."""

    def test_detects_100_percent_uptime(self) -> None:
        """Should detect 100% uptime guarantees as impossible."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="System must have 100% uptime",
            refined_text="The system shall guarantee 100% uptime",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert len(issues) >= 1
        assert any("absolute_guarantee" in issue for issue in issues)

    def test_detects_hundred_percent_spelled_out(self) -> None:
        """Should detect 'hundred percent' spelled out."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="We need hundred percent reliability",
            refined_text="The system requires hundred percent reliability",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert any("absolute_guarantee" in issue for issue in issues)

    def test_detects_zero_latency(self) -> None:
        """Should detect zero latency requirements as impossible."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Response must have zero latency",
            refined_text="API responses shall have zero latency",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert any("zero_guarantee" in issue for issue in issues)

    def test_detects_unlimited_resources(self) -> None:
        """Should detect unlimited resource requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Support unlimited users",
            refined_text="The system must support unlimited concurrent users",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert any("unbounded" in issue for issue in issues)

    def test_detects_infinite_storage(self) -> None:
        """Should detect infinite storage requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Provide infinite storage",
            refined_text="Users shall have infinite storage capacity",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert any("unbounded" in issue for issue in issues)

    def test_no_issues_for_reasonable_requirements(self) -> None:
        """Should not flag reasonable requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="99.9% uptime SLA",
            refined_text="System shall maintain 99.9% uptime",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is False
        assert len(issues) == 0

    def test_case_insensitive_matching(self) -> None:
        """Should detect patterns regardless of case."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="ZERO DOWNTIME required",
            refined_text="System must have ZERO DOWNTIME",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert len(issues) >= 1

    def test_multiple_issues_detected(self) -> None:
        """Should detect multiple impossible requirements."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="100% uptime and zero latency with unlimited users",
            refined_text="Guarantee 100% uptime, zero latency, unlimited concurrent users",
            category="non-functional",
            testable=True,
        )
        has_issues, issues, _ = _check_impossibility(req)

        assert has_issues is True
        assert len(issues) >= 2


class TestIdentifyExternalDependencies:
    """Tests for _identify_external_dependencies function."""

    def test_detects_api_dependencies(self) -> None:
        """Should detect API-related dependencies."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Integrate with REST API",
            refined_text="System shall integrate with external REST API for data",
            category="functional",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        assert len(deps) >= 1
        assert any(d.dependency_type == DependencyType.API for d in deps)

    def test_detects_database_dependencies(self) -> None:
        """Should detect database dependencies (infrastructure type)."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Store data in PostgreSQL",
            refined_text="Use PostgreSQL database for persistence",
            category="constraint",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        assert len(deps) >= 1
        # PostgreSQL maps to INFRASTRUCTURE in DEPENDENCY_KEYWORDS
        assert any(d.dependency_type == DependencyType.INFRASTRUCTURE for d in deps)

    def test_detects_cloud_service_dependencies(self) -> None:
        """Should detect cloud service dependencies (service type)."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Deploy to AWS",
            refined_text="Application shall be deployed on AWS infrastructure",
            category="constraint",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        assert len(deps) >= 1
        # AWS maps to SERVICE in DEPENDENCY_KEYWORDS
        assert any(d.dependency_type == DependencyType.SERVICE for d in deps)

    def test_detects_payment_service_dependencies(self) -> None:
        """Should detect payment integration dependencies."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Process payments with Stripe",
            refined_text="Integrate Stripe for payment processing",
            category="functional",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        assert len(deps) >= 1
        assert any("stripe" in d.name.lower() for d in deps)

    def test_deduplicates_by_type(self) -> None:
        """Should deduplicate dependencies of the same type."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Use REST API and GraphQL API",
            refined_text="System shall use REST API and GraphQL API",
            category="functional",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        # Should only have one API dependency (deduped by type)
        api_deps = [d for d in deps if d.dependency_type == DependencyType.API]
        assert len(api_deps) <= 1

    def test_no_dependencies_for_simple_requirement(self) -> None:
        """Should return empty tuple for requirements without dependencies."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Button should be blue",
            refined_text="Login button shall be blue colored",
            category="ui",
            testable=True,
        )
        deps = _identify_external_dependencies(req)

        assert len(deps) == 0


class TestAssessComplexity:
    """Tests for _assess_complexity function."""

    def test_low_complexity_for_simple_requirements(self) -> None:
        """Simple requirements should have LOW complexity."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Add a button",
            refined_text="Add a submit button to the form",
            category="ui",
            testable=True,
        )
        complexity, rationale = _assess_complexity(req, ())

        assert complexity == ComplexityLevel.LOW
        assert rationale is not None

    def test_medium_complexity_with_validation(self) -> None:
        """Requirements with validation should have at least MEDIUM complexity."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Validate user input",
            refined_text="Validate email format and required fields",
            category="functional",
            testable=True,
        )
        complexity, _ = _assess_complexity(req, ())

        assert complexity in (ComplexityLevel.MEDIUM, ComplexityLevel.HIGH)

    def test_high_complexity_with_distributed_keywords(self) -> None:
        """Requirements with distributed keywords should have HIGH complexity."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Build distributed concurrent system",
            refined_text="Implement distributed async processing with concurrent workers",
            category="functional",
            testable=True,
        )
        complexity, _ = _assess_complexity(req, ())

        # "distributed", "concurrent", "async" are HIGH complexity keywords
        assert complexity in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)

    def test_very_high_complexity_with_many_dependencies(self) -> None:
        """Many dependencies should contribute to HIGH or VERY_HIGH complexity."""
        from yolo_developer.agents.analyst.types import ExternalDependency

        req = CrystallizedRequirement(
            id="req-001",
            original_text="Complex distributed system",
            refined_text="Build distributed real-time system with ML",
            category="functional",
            testable=True,
        )
        deps = (
            ExternalDependency(
                name="API 1",
                dependency_type=DependencyType.API,
                description="External API",
                availability_notes="Available",
                criticality="required",
            ),
            ExternalDependency(
                name="API 2",
                dependency_type=DependencyType.API,
                description="Another API",
                availability_notes="Available",
                criticality="required",
            ),
            ExternalDependency(
                name="Database",
                dependency_type=DependencyType.DATA_SOURCE,
                description="Database",
                availability_notes="Available",
                criticality="required",
            ),
        )
        complexity, rationale = _assess_complexity(req, deps)

        # With "distributed", "real-time", "ML" and 3 dependencies, should be HIGH or VERY_HIGH
        assert complexity in (ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH)
        assert "dependencies" in rationale.lower() or "distributed" in rationale.lower()

    def test_weighted_scoring_considers_all_indicators(self) -> None:
        """Complexity should use weighted scoring across all levels."""
        # Requirement with indicators from multiple levels
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Simple CRUD with basic validation",
            refined_text="Create basic CRUD operations with validation",
            category="functional",
            testable=True,
        )
        complexity, rationale = _assess_complexity(req, ())

        # Should consider both CRUD (lower) and validation (medium)
        assert complexity in (
            ComplexityLevel.LOW,
            ComplexityLevel.MEDIUM,
        )
        assert rationale is not None


class TestGenerateRemediation:
    """Tests for _generate_remediation function."""

    def test_generates_suggestions_for_absolute_guarantee(self) -> None:
        """Should generate specific suggestions for absolute guarantee issues."""
        issues = ["[absolute_guarantee] 100% guarantees are impossible"]
        suggestions = _generate_remediation(issues, ComplexityLevel.HIGH, ())

        assert len(suggestions) >= 1
        # Should suggest SLA or percentage-based alternative
        assert any("SLA" in s or "99" in s or "threshold" in s.lower() for s in suggestions)

    def test_generates_suggestions_for_zero_guarantee(self) -> None:
        """Should generate suggestions for zero guarantee issues."""
        issues = ["[zero_guarantee] Zero latency is impossible"]
        suggestions = _generate_remediation(issues, ComplexityLevel.MEDIUM, ())

        assert len(suggestions) >= 1
        # Should suggest SLA or realistic alternatives
        assert any("SLA" in s or "realistic" in s.lower() or "99" in s for s in suggestions)

    def test_generates_suggestions_for_unbounded(self) -> None:
        """Should generate suggestions for unbounded resource issues."""
        issues = ["[unbounded_resource] Unlimited resources not possible"]
        suggestions = _generate_remediation(issues, ComplexityLevel.HIGH, ())

        assert len(suggestions) >= 1
        # Should suggest defining limits
        assert any("limit" in s.lower() or "bound" in s.lower() or "cap" in s.lower() for s in suggestions)

    def test_generates_suggestions_for_high_complexity(self) -> None:
        """Should suggest breaking down high complexity requirements."""
        issues: list[str] = []
        suggestions = _generate_remediation(issues, ComplexityLevel.VERY_HIGH, ())

        assert len(suggestions) >= 1
        # Should suggest breaking down or phased approach
        assert any("break" in s.lower() or "phase" in s.lower() or "piece" in s.lower() for s in suggestions)

    def test_generates_suggestions_for_dependencies(self) -> None:
        """Should suggest contingency plans for many dependencies."""
        from yolo_developer.agents.analyst.types import ExternalDependency

        deps = (
            ExternalDependency(
                name="API",
                dependency_type=DependencyType.API,
                description="External API",
                availability_notes="Requires account",
                criticality="required",
            ),
            ExternalDependency(
                name="Database",
                dependency_type=DependencyType.DATA_SOURCE,
                description="Database",
                availability_notes="Cloud hosted",
                criticality="required",
            ),
            ExternalDependency(
                name="Service",
                dependency_type=DependencyType.SERVICE,
                description="External service",
                availability_notes="Third party",
                criticality="required",
            ),
        )
        suggestions = _generate_remediation([], ComplexityLevel.HIGH, deps)

        assert len(suggestions) >= 1


class TestValidateImplementability:
    """Tests for _validate_implementability function."""

    def test_returns_implementable_for_valid_requirement(self) -> None:
        """Valid requirements should be marked IMPLEMENTABLE."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="Users can authenticate with email and password",
            category="functional",
            testable=True,
        )
        result = _validate_implementability(req)

        assert result.status == ImplementabilityStatus.IMPLEMENTABLE
        assert len(result.issues) == 0
        assert result.rationale is not None

    def test_returns_not_implementable_for_impossible_requirement(self) -> None:
        """Impossible requirements should be marked NOT_IMPLEMENTABLE."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="100% uptime guaranteed",
            refined_text="System must guarantee 100% uptime always",
            category="non-functional",
            testable=True,
        )
        result = _validate_implementability(req)

        assert result.status == ImplementabilityStatus.NOT_IMPLEMENTABLE
        assert len(result.issues) >= 1
        assert len(result.remediation_suggestions) >= 1

    def test_high_complexity_marked_correctly(self) -> None:
        """High complexity requirements should have appropriate status and remediation."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Build AI system",
            refined_text="Implement machine learning with deep learning neural network",
            category="functional",
            testable=True,
        )
        result = _validate_implementability(req)

        # High complexity requirements are still implementable but should have
        # remediation suggestions to help break them down
        assert result.complexity == ComplexityLevel.VERY_HIGH
        assert len(result.remediation_suggestions) >= 1

    def test_includes_complexity_assessment(self) -> None:
        """Result should include complexity level."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Add button",
            refined_text="Add a submit button",
            category="ui",
            testable=True,
        )
        result = _validate_implementability(req)

        assert result.complexity in (
            ComplexityLevel.LOW,
            ComplexityLevel.MEDIUM,
            ComplexityLevel.HIGH,
            ComplexityLevel.VERY_HIGH,
        )

    def test_includes_dependencies(self) -> None:
        """Result should include identified dependencies."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="Integrate with Stripe API",
            refined_text="Process payments via Stripe API integration",
            category="functional",
            testable=True,
        )
        result = _validate_implementability(req)

        assert isinstance(result.dependencies, tuple)


class TestValidateAllRequirements:
    """Tests for _validate_all_requirements function."""

    def test_validates_multiple_requirements(self) -> None:
        """Should validate all requirements in the tuple."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login",
                refined_text="User can login with email",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Fast response",
                refined_text="API responds in < 200ms",
                category="non-functional",
                testable=True,
            ),
        )
        result, score = _validate_all_requirements(reqs)

        assert len(result) == 2
        assert 0.0 <= score <= 1.0

    def test_updates_requirement_fields(self) -> None:
        """Should update requirement fields with validation results."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="User login",
                refined_text="User can login with email",
                category="functional",
                testable=True,
            ),
        )
        result, _ = _validate_all_requirements(reqs)

        assert result[0].implementability_status is not None
        assert result[0].complexity is not None
        assert result[0].implementability_rationale is not None

    def test_score_reflects_implementability(self) -> None:
        """Score should reflect overall implementability."""
        # All implementable
        good_reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="Add button",
                refined_text="Add submit button",
                category="ui",
                testable=True,
            ),
        )
        _, good_score = _validate_all_requirements(good_reqs)

        # Has impossible requirement
        bad_reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="100% uptime",
                refined_text="Guarantee 100% uptime",
                category="non-functional",
                testable=True,
            ),
        )
        _, bad_score = _validate_all_requirements(bad_reqs)

        assert good_score > bad_score

    def test_empty_tuple_returns_perfect_score(self) -> None:
        """Empty requirements should return 1.0 score."""
        result, score = _validate_all_requirements(())

        assert len(result) == 0
        assert score == 1.0

    def test_maintains_order(self) -> None:
        """Output order should match input order."""
        reqs = (
            CrystallizedRequirement(
                id="req-001",
                original_text="First",
                refined_text="First req",
                category="functional",
                testable=True,
            ),
            CrystallizedRequirement(
                id="req-002",
                original_text="Second",
                refined_text="Second req",
                category="functional",
                testable=True,
            ),
        )
        result, _ = _validate_all_requirements(reqs)

        assert result[0].id == "req-001"
        assert result[1].id == "req-002"
