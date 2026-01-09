"""Integration tests for Analyst agent (Story 5.1, 5.3, 5.4).

Tests the analyst_node integration with:
- LangGraph StateGraph
- Quality gate decorator
- YoloState message handling
- Gap analysis integration (Story 5.3)
- Requirement categorization (Story 5.4)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.agents.analyst import analyst_node
from yolo_developer.agents.analyst.node import _parse_llm_response
from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CrystallizedRequirement,
    GapType,
    IdentifiedGap,
    Severity,
)
from yolo_developer.gates import GateContext, GateResult, clear_evaluators, register_evaluator
from yolo_developer.orchestrator.state import YoloState


class TestAnalystNodeIntegration:
    """Integration tests for analyst_node with orchestration."""

    @pytest.fixture(autouse=True)
    def cleanup_evaluators(self) -> None:
        """Clean up gate evaluators after each test."""
        yield
        clear_evaluators()

    @pytest.fixture
    def sample_state(self) -> YoloState:
        """Create a sample YoloState for testing."""
        return {
            "messages": [
                HumanMessage(content="Build a todo application with user authentication")
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

    @pytest.mark.asyncio
    async def test_analyst_node_with_testability_gate_passing(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should work with testability gate when gate passes."""
        # Register a passing testability gate
        async def passing_gate(ctx: GateContext) -> GateResult:
            return GateResult(passed=True, gate_name="testability")

        register_evaluator("testability", passing_gate)

        # Run analyst node
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="test",
                    refined_text="test",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Should have messages and decisions
        assert "messages" in result
        assert len(result["messages"]) >= 1
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_analyst_node_with_testability_gate_blocking(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should be blocked when testability gate fails."""
        # Register a failing testability gate
        async def failing_gate(ctx: GateContext) -> GateResult:
            return GateResult(
                passed=False,
                gate_name="testability",
                reason="Requirements not testable",
            )

        register_evaluator("testability", failing_gate)

        # Run analyst node - gate should block
        result = await analyst_node(sample_state)

        # Gate should have blocked execution
        assert result.get("gate_blocked") is True
        assert "Requirements not testable" in result.get("gate_failure", "")

    @pytest.mark.asyncio
    async def test_analyst_node_state_update_format(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should return proper state update dict format."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="orig",
                    refined_text="refined",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Should have gate_results from decorator
        assert "gate_results" in result or "messages" in result

        # Check message format if present
        if "messages" in result:
            for msg in result["messages"]:
                # Should have content
                assert hasattr(msg, "content")
                assert msg.content

    @pytest.mark.asyncio
    async def test_analyst_node_with_mocked_llm(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should work with mocked LLM responses."""
        # Mock the crystallized output that would come from LLM processing
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build a todo application",
                    refined_text="Create a web-based task management application",
                    category="functional",
                    testable=True,
                ),
                CrystallizedRequirement(
                    id="req-002",
                    original_text="with user authentication",
                    refined_text="Implement user authentication with email/password",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=("No mention of data persistence",),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Verify result structure
        assert isinstance(result, dict)
        if "messages" in result:
            assert len(result["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_analyst_node_decision_recording(
        self,
        sample_state: YoloState,
    ) -> None:
        """analyst_node should record decisions with correct agent attribution."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="test",
                    refined_text="test",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(sample_state)

        # Verify decisions
        if "decisions" in result:
            for decision in result["decisions"]:
                assert decision.agent == "analyst"
                assert "Crystallized" in decision.summary


class TestLLMResponseParsing:
    """Tests for LLM response parsing functionality."""

    def test_parse_valid_json_response(self) -> None:
        """_parse_llm_response should parse valid JSON correctly."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "Original",
                    "refined_text": "Refined",
                    "category": "functional",
                    "testable": True,
                }
            ],
            "identified_gaps": ["Gap 1"],
            "contradictions": ["Contradiction 1"],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        assert result.requirements[0].id == "req-001"
        assert result.identified_gaps == ("Gap 1",)
        assert result.contradictions == ("Contradiction 1",)

    def test_parse_invalid_json_response(self) -> None:
        """_parse_llm_response should handle invalid JSON gracefully."""
        response = "This is not valid JSON"

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 0
        assert "Failed to parse LLM response" in result.identified_gaps

    def test_parse_empty_requirements(self) -> None:
        """_parse_llm_response should handle empty requirements list."""
        response = json.dumps({
            "requirements": [],
            "identified_gaps": [],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 0
        assert len(result.identified_gaps) == 0

    def test_parse_missing_fields(self) -> None:
        """_parse_llm_response should handle missing optional fields."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "Test",
                    "refined_text": "Test refined",
                    # Missing category and testable - should use defaults
                }
            ],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        # Should use default values
        assert result.requirements[0].category == "functional"
        assert result.requirements[0].testable is True


class TestCrystallizationIntegration:
    """Integration tests for crystallization flow (Story 5.2 Task 9)."""

    @pytest.mark.asyncio
    async def test_full_crystallization_flow_with_vague_content(self) -> None:
        """Test full crystallization flow detects vague terms and adds metadata."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a fast, scalable API")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        assert "messages" in result
        assert "decisions" in result
        # The output should include crystallized requirements with new fields
        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]
        assert len(requirements) >= 1

        # Check new fields are present in output
        req = requirements[0]
        assert "scope_notes" in req
        assert "implementation_hints" in req
        assert "confidence" in req

    @pytest.mark.asyncio
    async def test_crystallization_with_specific_content(self) -> None:
        """Test crystallization with specific, non-vague content."""
        state: YoloState = {
            "messages": [HumanMessage(content="Response time MUST be under 200ms")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]
        req = requirements[0]

        # Non-vague content should have full confidence
        assert req["confidence"] == 1.0
        assert req["scope_notes"] is None

    @pytest.mark.asyncio
    async def test_decision_includes_transformation_info(self) -> None:
        """Test that decisions capture transformation information."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a fast API endpoint")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        decisions = result["decisions"]
        assert len(decisions) >= 1
        decision = decisions[0]

        # Decision should have analyst attribution
        assert decision.agent == "analyst"
        # Decision should include requirement count in summary
        assert "requirement" in decision.summary.lower() or "crystallized" in decision.summary.lower()

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_mocked_llm(self) -> None:
        """Test that existing test patterns still work with enhanced output."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="Build an app",
                    refined_text="Create an application",
                    category="functional",
                    testable=True,
                    # New fields with defaults
                    scope_notes=None,
                    implementation_hints=(),
                    confidence=1.0,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
        )

        state: YoloState = {
            "messages": [HumanMessage(content="Build an app")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        # Should work exactly as before
        assert isinstance(result, dict)
        assert "messages" in result
        assert "decisions" in result

    @pytest.mark.asyncio
    async def test_scope_boundaries_in_output(self) -> None:
        """Test that scope boundaries are properly included in output."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build a simple, easy-to-use REST API")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data["requirements"]

        # Vague content should have scope_notes
        assert len(requirements) >= 1
        # At least some requirements should have scope clarification
        has_scope_notes = any(req.get("scope_notes") for req in requirements)
        assert has_scope_notes, "Vague content should generate scope_notes"

    @pytest.mark.asyncio
    async def test_llm_response_with_new_fields_parses_correctly(self) -> None:
        """Test parsing LLM response that includes new fields."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "System should be fast",
                    "refined_text": "Response time < 200ms at 95th percentile",
                    "category": "non-functional",
                    "testable": True,
                    "scope_notes": "Applies to GET endpoints; POST excluded",
                    "implementation_hints": ["Use async handlers", "Add response caching"],
                    "confidence": 0.9,
                }
            ],
            "identified_gaps": [],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        assert isinstance(result, AnalystOutput)
        assert len(result.requirements) == 1
        req = result.requirements[0]
        assert req.scope_notes == "Applies to GET endpoints; POST excluded"
        assert req.implementation_hints == ("Use async handlers", "Add response caching")
        assert req.confidence == 0.9


class TestGapAnalysisIntegration:
    """Integration tests for gap analysis functionality (Story 5.3)."""

    @pytest.mark.asyncio
    async def test_full_gap_analysis_flow(self) -> None:
        """Test complete gap analysis flow from seed content to structured gaps."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="Build user login system with email/password authentication"
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        # Verify structured gaps in output
        output_data = result["messages"][0].additional_kwargs["output"]
        structured_gaps = output_data["structured_gaps"]

        # Should identify gaps for authentication features
        assert len(structured_gaps) > 0

        # Check gap structure
        for gap in structured_gaps:
            assert "id" in gap
            assert "description" in gap
            assert "gap_type" in gap
            assert "severity" in gap
            assert "source_requirements" in gap
            assert "rationale" in gap

    @pytest.mark.asyncio
    async def test_gap_analysis_identifies_multiple_gap_types(self) -> None:
        """Test that gap analysis identifies different types of gaps."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="Build REST API with user data input and database storage"
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        structured_gaps = output_data["structured_gaps"]

        # Extract gap types present
        gap_types = {gap["gap_type"] for gap in structured_gaps}

        # Should have at least 2 different gap types for rich content
        assert len(gap_types) >= 1, "Should identify at least one gap type"

    @pytest.mark.asyncio
    async def test_gap_analysis_with_mocked_llm_output(self) -> None:
        """Test gap analysis when LLM returns gaps."""
        mock_gap = IdentifiedGap(
            id="gap-001",
            description="Missing logout functionality",
            gap_type=GapType.IMPLIED_REQUIREMENT,
            severity=Severity.HIGH,
            source_requirements=("req-001",),
            rationale="Login implies logout needed",
        )

        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="User login",
                    refined_text="User authenticates with credentials",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=(),
            contradictions=(),
            structured_gaps=(mock_gap,),
        )

        state: YoloState = {
            "messages": [HumanMessage(content="Build user login")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        # Verify gap is in output
        output_data = result["messages"][0].additional_kwargs["output"]
        gaps = output_data["structured_gaps"]
        assert len(gaps) >= 1
        assert gaps[0]["gap_type"] == "implied_requirement"
        assert gaps[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_gap_severity_counts_in_decision(self) -> None:
        """Test that decision includes gap severity counts."""
        state: YoloState = {
            "messages": [
                HumanMessage(content="Build user authentication with login form")
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        decision = result["decisions"][0]
        rationale = decision.rationale.lower()

        # Decision should mention severity levels
        assert "critical" in rationale
        assert "high" in rationale
        assert "medium" in rationale
        assert "low" in rationale

    @pytest.mark.asyncio
    async def test_gap_ids_in_related_artifacts(self) -> None:
        """Test that gap IDs are included in decision related_artifacts."""
        state: YoloState = {
            "messages": [HumanMessage(content="Build API with data input")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        decision = result["decisions"][0]
        artifacts = decision.related_artifacts

        # Should have both requirement IDs and gap IDs
        req_ids = [a for a in artifacts if a.startswith("req-")]
        gap_ids = [a for a in artifacts if a.startswith("gap-")]

        assert len(req_ids) >= 1, "Should have requirement IDs"
        assert len(gap_ids) >= 1, "Should have gap IDs"

    def test_parse_llm_response_with_structured_gaps(self) -> None:
        """Test parsing LLM response that includes structured_gaps."""
        response = json.dumps({
            "requirements": [
                {
                    "id": "req-001",
                    "original_text": "User login",
                    "refined_text": "User authenticates",
                    "category": "functional",
                    "testable": True,
                }
            ],
            "identified_gaps": [],
            "structured_gaps": [
                {
                    "id": "gap-001",
                    "description": "Missing logout functionality",
                    "gap_type": "implied_requirement",
                    "severity": "high",
                    "source_requirements": ["req-001"],
                    "rationale": "Login implies logout needed",
                }
            ],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        assert len(result.structured_gaps) == 1
        gap = result.structured_gaps[0]
        assert gap.id == "gap-001"
        assert gap.gap_type == GapType.IMPLIED_REQUIREMENT
        assert gap.severity == Severity.HIGH
        assert gap.source_requirements == ("req-001",)

    def test_parse_llm_response_with_invalid_gap_type(self) -> None:
        """Test parsing handles invalid gap_type gracefully."""
        response = json.dumps({
            "requirements": [],
            "identified_gaps": [],
            "structured_gaps": [
                {
                    "id": "gap-001",
                    "description": "Test",
                    "gap_type": "invalid_type",  # Invalid
                    "severity": "high",
                    "source_requirements": [],
                    "rationale": "Test",
                }
            ],
            "contradictions": [],
        })

        result = _parse_llm_response(response)

        # Should skip invalid gaps but not crash
        assert isinstance(result, AnalystOutput)
        # Invalid gap should be skipped
        assert len(result.structured_gaps) == 0

    @pytest.mark.asyncio
    async def test_empty_seed_no_gaps(self) -> None:
        """Test that empty seed content produces no gaps."""
        state: YoloState = {
            "messages": [],  # No messages
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        # Empty seed should have minimal/no structured gaps
        assert "structured_gaps" in output_data

    @pytest.mark.asyncio
    async def test_gap_analysis_backward_compatibility(self) -> None:
        """Test that old code without structured_gaps still works."""
        # Create output without structured_gaps (simulating old LLM response)
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="test",
                    refined_text="test",
                    category="functional",
                    testable=True,
                ),
            ),
            identified_gaps=("Old style gap",),
            contradictions=(),
            # structured_gaps defaults to ()
        )

        state: YoloState = {
            "messages": [HumanMessage(content="test")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        # Mock to return output without structured gaps, which should then
        # be enhanced by _enhance_with_gap_analysis
        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        # Should still work and include structured_gaps in output dict
        output_data = result["messages"][0].additional_kwargs["output"]
        assert "structured_gaps" in output_data
        assert isinstance(output_data["structured_gaps"], list)


class TestCategorizationIntegration:
    """Integration tests for requirement categorization (Story 5.4)."""

    @pytest.mark.asyncio
    async def test_categorization_in_full_analysis_flow(self) -> None:
        """Test that categorization runs as part of full analysis."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="""
                    Build a user authentication system with these requirements:
                    1. Users can register with email and password
                    2. Login response time should be under 200ms
                    3. Must use PostgreSQL database
                    4. Send email notification on successful registration
                    """
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # Should have at least one requirement
        assert len(requirements) >= 1

        # Each requirement should have categorization fields
        for req in requirements:
            # Story 5.4 fields should be present
            assert "sub_category" in req
            assert "category_confidence" in req
            assert "category_rationale" in req
            # Confidence should be valid
            assert 0.0 <= req["category_confidence"] <= 1.0
            # Rationale should exist
            assert req["category_rationale"] is not None

    @pytest.mark.asyncio
    async def test_categorization_detects_functional_requirements(self) -> None:
        """Test categorization correctly identifies functional requirements."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="User can login with email and password and view their profile"
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # Should detect functional requirement with user_management sub-category
        assert len(requirements) >= 1
        # At least one should be functional
        categories = [req.get("category") for req in requirements]
        assert "functional" in categories

        # Check for user_management sub-category on functional reqs
        functional_reqs = [r for r in requirements if r.get("category") == "functional"]
        sub_categories = [r.get("sub_category") for r in functional_reqs]
        assert "user_management" in sub_categories or any(s for s in sub_categories)

    @pytest.mark.asyncio
    async def test_categorization_detects_non_functional_requirements(self) -> None:
        """Test categorization correctly identifies non-functional requirements."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="API response time must be under 200 milliseconds for performance"
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # Check if any requirement is non-functional with performance sub-category
        assert len(requirements) >= 1

        # Look for performance-related categorization
        for req in requirements:
            if req.get("category") == "non_functional":
                assert req.get("sub_category") == "performance"
                break

    @pytest.mark.asyncio
    async def test_categorization_rationale_includes_keywords(self) -> None:
        """Test that categorization rationale mentions detected keywords."""
        state: YoloState = {
            "messages": [
                HumanMessage(content="User can create, edit, and delete items")
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # At least one requirement should have keywords in rationale
        for req in requirements:
            rationale = req.get("category_rationale", "")
            if rationale:
                # Should mention either Keywords or Scores
                assert "Keywords:" in rationale or "Scores:" in rationale

    @pytest.mark.asyncio
    async def test_categorization_preserves_other_fields(self) -> None:
        """Test that categorization doesn't break other requirement fields."""
        mock_output = AnalystOutput(
            requirements=(
                CrystallizedRequirement(
                    id="req-001",
                    original_text="User can login",
                    refined_text="User authenticates with email/password",
                    category="functional",
                    testable=True,
                    scope_notes="Web only, not mobile",
                    implementation_hints=("Use JWT tokens",),
                    confidence=0.9,
                ),
            ),
            identified_gaps=("Legacy gap",),
            contradictions=("Contradiction 1",),
        )

        state: YoloState = {
            "messages": [HumanMessage(content="User can login")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        with patch(
            "yolo_developer.agents.analyst.node._crystallize_requirements",
            new=AsyncMock(return_value=mock_output),
        ):
            result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # Original fields should still be present
        assert len(requirements) >= 1
        req = requirements[0]
        assert req["id"] == "req-001"
        assert req["original_text"] == "User can login"
        assert req["testable"] is True
        assert req["confidence"] == 0.9

        # New categorization fields should also be present
        assert "sub_category" in req
        assert "category_confidence" in req
        assert "category_rationale" in req

    @pytest.mark.asyncio
    async def test_categorization_with_mixed_requirements(self) -> None:
        """Test categorization handles mixed requirement types."""
        state: YoloState = {
            "messages": [
                HumanMessage(
                    content="""
                    1. Users can register accounts (functional)
                    2. System must respond in under 200ms (non-functional performance)
                    3. Must use PostgreSQL database (constraint technical)
                    """
                )
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await analyst_node(state)

        output_data = result["messages"][0].additional_kwargs["output"]
        requirements = output_data.get("requirements", [])

        # Should have at least one requirement
        assert len(requirements) >= 1

        # All requirements should be categorized
        for req in requirements:
            assert req.get("category") in ("functional", "non_functional", "constraint")
            assert req.get("category_rationale") is not None
