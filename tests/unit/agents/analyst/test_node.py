"""Unit tests for Analyst agent node (Story 5.1 Task 2, Story 5.2).

Tests for analyst_node function, state management, and vague term detection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.agents.analyst.node import _detect_vague_terms
from yolo_developer.agents.analyst.types import AnalystOutput, CrystallizedRequirement
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
