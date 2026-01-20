"""Unit tests for PM agent escalation to Analyst (Story 6.7).

This module tests the escalation functionality that allows the PM agent
to escalate unclear requirements back to the Analyst for clarification.

Test Organization:
- TestEscalationTypes: Type definitions and structure tests
- TestAmbiguityDetection: Vague term and ambiguity detection tests
- TestQuestionGeneration: Escalation question generation tests
- TestEscalationCreation: Escalation object creation tests
- TestCheckForEscalation: Main escalation check function tests
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Task 1: Escalation Types Tests
# =============================================================================


class TestEscalationTypes:
    """Test escalation type definitions (Task 1)."""

    def test_escalation_reason_literal_exists(self) -> None:
        """Test EscalationReason Literal type is defined with correct values."""
        from yolo_developer.agents.pm.types import EscalationReason

        # Verify all expected values are valid
        valid_reasons = [
            "ambiguous_terms",
            "missing_criteria",
            "contradictory",
            "technical_question",
        ]
        for reason in valid_reasons:
            assert reason in EscalationReason.__args__

    def test_escalation_question_typeddict_structure(self) -> None:
        """Test EscalationQuestion TypedDict has correct fields."""
        from yolo_developer.agents.pm.types import EscalationQuestion

        # Create a valid EscalationQuestion
        question: EscalationQuestion = {
            "question_text": "What does 'fast' mean?",
            "source_requirement_id": "req-001",
            "ambiguity_type": "vague_term",
            "context": "Requirement mentions fast response",
        }

        assert question["question_text"] == "What does 'fast' mean?"
        assert question["source_requirement_id"] == "req-001"
        assert question["ambiguity_type"] == "vague_term"
        assert question["context"] == "Requirement mentions fast response"

    def test_escalation_typeddict_structure(self) -> None:
        """Test Escalation TypedDict has correct fields."""
        from yolo_developer.agents.pm.types import PMEscalation

        # Create a valid Escalation
        escalation: PMEscalation = {
            "id": "esc-123-001",
            "source_agent": "pm",
            "target_agent": "analyst",
            "requirement_id": "req-001",
            "questions": [],
            "partial_work": None,
            "reason": "ambiguous_terms",
            "created_at": "2026-01-09T12:00:00Z",
        }

        assert escalation["id"] == "esc-123-001"
        assert escalation["source_agent"] == "pm"
        assert escalation["target_agent"] == "analyst"
        assert escalation["requirement_id"] == "req-001"
        assert escalation["reason"] == "ambiguous_terms"

    def test_escalation_result_typeddict_structure(self) -> None:
        """Test EscalationResult TypedDict has correct fields."""
        from yolo_developer.agents.pm.types import EscalationResult

        # Create a valid EscalationResult
        result: EscalationResult = {
            "escalations": [],
            "escalation_count": 0,
        }

        assert result["escalations"] == []
        assert result["escalation_count"] == 0

    def test_types_exported_from_init(self) -> None:
        """Test escalation types are exported from __init__.py."""
        from yolo_developer.agents.pm import (
            EscalationQuestion,
            EscalationReason,
            EscalationResult,
            PMEscalation,
        )

        # Just verify imports work
        assert EscalationReason is not None
        assert EscalationQuestion is not None
        assert PMEscalation is not None
        assert EscalationResult is not None


# =============================================================================
# Task 7: Ambiguity Detection Tests
# =============================================================================


class TestAmbiguityDetection:
    """Test ambiguity detection in requirements (Task 2, 7)."""

    def test_clear_requirement_returns_empty_list(self) -> None:
        """Test that a clear requirement returns no ambiguities."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        # Clear requirement with specific metrics
        req = {
            "id": "req-001",
            "refined_text": "API response time must be under 200ms at the 95th percentile",
            "category": "non_functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert ambiguities == []

    def test_vague_term_fast_detected(self) -> None:
        """Test that 'fast' is detected as a vague term."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "The system should be fast",
            "category": "non_functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        assert any("fast" in amb.lower() for amb in ambiguities)

    def test_vague_term_easy_detected(self) -> None:
        """Test that 'easy' is detected as a vague term."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "The interface should be easy to use",
            "category": "functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        assert any("easy" in amb.lower() for amb in ambiguities)

    def test_vague_term_intuitive_detected(self) -> None:
        """Test that 'intuitive' is detected as a vague term."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "The UI must be intuitive",
            "category": "functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        assert any("intuitive" in amb.lower() for amb in ambiguities)

    def test_missing_criteria_detected(self) -> None:
        """Test that requirements with no measurable criteria are flagged."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "Users should be satisfied with the experience",
            "category": "non_functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        # Should mention missing criteria or satisfaction
        assert any("criteria" in amb.lower() or "measur" in amb.lower() for amb in ambiguities)

    def test_contradictory_statement_detected(self) -> None:
        """Test that contradictory statements are detected."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "The system should be simple but comprehensive",
            "category": "functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        # Should detect the contradiction pattern
        assert any("contradict" in amb.lower() or "but" in amb.lower() for amb in ambiguities)

    def test_multiple_ambiguities_returned(self) -> None:
        """Test that multiple ambiguities are detected and returned."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "The system should be fast and easy to use with intuitive navigation",
            "category": "functional",
        }

        ambiguities = _detect_ambiguity(req)
        # Should detect at least fast, easy, intuitive
        assert len(ambiguities) >= 3

    def test_technical_question_detected(self) -> None:
        """Test that technical questions are detected."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "Is it possible to integrate with legacy systems?",
            "category": "functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        assert any("technical_question" in amb for amb in ambiguities)

    def test_technical_question_feasibility_detected(self) -> None:
        """Test that feasibility questions are detected."""
        from yolo_developer.agents.pm.escalation import _detect_ambiguity

        req = {
            "id": "req-001",
            "refined_text": "Check the feasibility of real-time sync across regions",
            "category": "non_functional",
        }

        ambiguities = _detect_ambiguity(req)
        assert len(ambiguities) > 0
        assert any("technical_question" in amb for amb in ambiguities)


# =============================================================================
# Task 8: Question Generation Tests
# =============================================================================


class TestQuestionGeneration:
    """Test escalation question generation (Task 3, 8)."""

    def test_question_generated_for_vague_term(self) -> None:
        """Test that a question is generated for a vague term."""
        from yolo_developer.agents.pm.escalation import _generate_escalation_questions

        req = {
            "id": "req-001",
            "refined_text": "The system should be fast",
            "category": "non_functional",
        }
        ambiguities = ["vague_term:fast"]

        questions = _generate_escalation_questions(req, ambiguities)
        assert len(questions) == 1
        assert "fast" in questions[0]["question_text"].lower()
        assert questions[0]["ambiguity_type"] == "vague_term"

    def test_question_generated_for_missing_criteria(self) -> None:
        """Test that a question is generated for missing criteria."""
        from yolo_developer.agents.pm.escalation import _generate_escalation_questions

        req = {
            "id": "req-001",
            "refined_text": "Users should be happy",
            "category": "non_functional",
        }
        ambiguities = ["missing_criteria"]

        questions = _generate_escalation_questions(req, ambiguities)
        assert len(questions) == 1
        assert "criteria" in questions[0]["question_text"].lower()
        assert questions[0]["ambiguity_type"] == "missing_criteria"

    def test_question_references_source_requirement(self) -> None:
        """Test that questions include source requirement reference."""
        from yolo_developer.agents.pm.escalation import _generate_escalation_questions

        req = {
            "id": "req-099",
            "refined_text": "The system should be fast",
            "category": "non_functional",
        }
        ambiguities = ["vague_term:fast"]

        questions = _generate_escalation_questions(req, ambiguities)
        assert questions[0]["source_requirement_id"] == "req-099"

    def test_multiple_questions_for_multiple_ambiguities(self) -> None:
        """Test that multiple questions are generated for multiple ambiguities."""
        from yolo_developer.agents.pm.escalation import _generate_escalation_questions

        req = {
            "id": "req-001",
            "refined_text": "System should be fast and easy",
            "category": "functional",
        }
        ambiguities = ["vague_term:fast", "vague_term:easy"]

        questions = _generate_escalation_questions(req, ambiguities)
        assert len(questions) == 2
        # Each question should target different term
        question_texts = [q["question_text"].lower() for q in questions]
        assert any("fast" in qt for qt in question_texts)
        assert any("easy" in qt for qt in question_texts)


# =============================================================================
# Task 9: Escalation Creation Tests
# =============================================================================


class TestEscalationCreation:
    """Test escalation object creation (Task 4, 9)."""

    def test_escalation_has_unique_id(self) -> None:
        """Test that escalation has a unique ID."""
        from yolo_developer.agents.pm.escalation import _create_escalation

        req = {"id": "req-001", "refined_text": "Test", "category": "functional"}
        questions = [
            {
                "question_text": "Test?",
                "source_requirement_id": "req-001",
                "ambiguity_type": "vague_term",
                "context": "",
            }
        ]

        esc1 = _create_escalation(req, questions, "ambiguous_terms")
        esc2 = _create_escalation(req, questions, "ambiguous_terms")

        # IDs should be unique
        assert esc1["id"] != esc2["id"]
        # ID should match format esc-{timestamp}-{counter}
        assert re.match(r"esc-\d+-\d+", esc1["id"])

    def test_source_agent_is_pm(self) -> None:
        """Test that source_agent is set to 'pm'."""
        from yolo_developer.agents.pm.escalation import _create_escalation

        req = {"id": "req-001", "refined_text": "Test", "category": "functional"}
        questions = [
            {
                "question_text": "Test?",
                "source_requirement_id": "req-001",
                "ambiguity_type": "vague_term",
                "context": "",
            }
        ]

        esc = _create_escalation(req, questions, "ambiguous_terms")
        assert esc["source_agent"] == "pm"

    def test_target_agent_is_analyst(self) -> None:
        """Test that target_agent is set to 'analyst'."""
        from yolo_developer.agents.pm.escalation import _create_escalation

        req = {"id": "req-001", "refined_text": "Test", "category": "functional"}
        questions = [
            {
                "question_text": "Test?",
                "source_requirement_id": "req-001",
                "ambiguity_type": "vague_term",
                "context": "",
            }
        ]

        esc = _create_escalation(req, questions, "ambiguous_terms")
        assert esc["target_agent"] == "analyst"

    def test_original_requirement_preserved(self) -> None:
        """Test that original requirement is preserved in escalation."""
        from yolo_developer.agents.pm.escalation import _create_escalation

        req = {
            "id": "req-007",
            "refined_text": "System should be fast",
            "category": "non_functional",
        }
        questions = [
            {
                "question_text": "Test?",
                "source_requirement_id": "req-007",
                "ambiguity_type": "vague_term",
                "context": "",
            }
        ]

        esc = _create_escalation(req, questions, "ambiguous_terms")
        assert esc["requirement_id"] == "req-007"

    def test_escalation_reason_is_correct(self) -> None:
        """Test that escalation reason is set correctly."""
        from yolo_developer.agents.pm.escalation import _create_escalation

        req = {"id": "req-001", "refined_text": "Test", "category": "functional"}
        questions = [
            {
                "question_text": "Test?",
                "source_requirement_id": "req-001",
                "ambiguity_type": "vague_term",
                "context": "",
            }
        ]

        esc = _create_escalation(req, questions, "missing_criteria")
        assert esc["reason"] == "missing_criteria"


# =============================================================================
# Task 5: Main Escalation Check Function Tests
# =============================================================================


class TestCheckForEscalation:
    """Test main escalation check function (Task 5)."""

    def test_clear_requirement_returns_none(self) -> None:
        """Test that clear requirements return None (no escalation)."""
        from yolo_developer.agents.pm.escalation import check_for_escalation

        req = {
            "id": "req-001",
            "refined_text": "API response time must be under 200ms",
            "category": "non_functional",
        }

        result = check_for_escalation(req)
        assert result is None

    def test_unclear_requirement_returns_escalation(self) -> None:
        """Test that unclear requirements return an Escalation."""
        from yolo_developer.agents.pm.escalation import check_for_escalation

        req = {
            "id": "req-001",
            "refined_text": "The system should be fast and easy",
            "category": "functional",
        }

        result = check_for_escalation(req)
        assert result is not None
        assert result["source_agent"] == "pm"
        assert result["target_agent"] == "analyst"
        assert len(result["questions"]) > 0

    def test_escalation_includes_all_questions(self) -> None:
        """Test that escalation includes questions for all ambiguities."""
        from yolo_developer.agents.pm.escalation import check_for_escalation

        req = {
            "id": "req-001",
            "refined_text": "The system should be fast, easy, and intuitive",
            "category": "functional",
        }

        result = check_for_escalation(req)
        assert result is not None
        # Should have at least 3 questions (fast, easy, intuitive)
        assert len(result["questions"]) >= 3


# =============================================================================
# Task 10: Integration Tests (pm_node integration)
# =============================================================================


class TestPMNodeEscalationIntegration:
    """Test pm_node integration with escalation (Task 6, 10)."""

    @pytest.mark.asyncio
    async def test_pm_node_returns_escalations_in_state_update(self) -> None:
        """Test that pm_node returns escalations in the state update dict."""
        from yolo_developer.agents.pm import pm_node

        # Create state with an unclear requirement
        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "The system should be fast and easy to use",
                        "category": "functional",
                    }
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # Should include escalations in result
        assert "escalations" in result
        assert len(result["escalations"]) > 0
        # Each escalation should have correct structure
        esc = result["escalations"][0]
        assert esc["source_agent"] == "pm"
        assert esc["target_agent"] == "analyst"

    @pytest.mark.asyncio
    async def test_pm_node_continues_processing_clear_requirements(self) -> None:
        """Test that pm_node processes clear requirements even when escalations exist."""
        from yolo_developer.agents.pm import pm_node

        # Mix of clear and unclear requirements
        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "API response time must be under 200ms at 95th percentile",
                        "category": "non_functional",
                    },
                    {
                        "id": "req-002",
                        "refined_text": "The system should be fast",
                        "category": "non_functional",
                    },
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # Should have at least one story from clear requirement
        assert result["pm_output"]["story_count"] >= 1
        # Should have escalation for unclear requirement
        assert len(result["escalations"]) >= 1

    @pytest.mark.asyncio
    async def test_processing_notes_includes_escalation_count(self) -> None:
        """Test that processing_notes includes escalation count."""
        from yolo_developer.agents.pm import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "The system should be fast and easy",
                        "category": "functional",
                    }
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # processing_notes should mention escalations
        notes = result["pm_output"]["processing_notes"]
        assert "escalation" in notes.lower()

    @pytest.mark.asyncio
    async def test_decision_includes_escalation_summary(self) -> None:
        """Test that Decision record includes escalation summary."""
        from yolo_developer.agents.pm import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "The system should be intuitive",
                        "category": "functional",
                    }
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # Decision rationale should mention escalations
        decision = result["decisions"][0]
        assert "escalation" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_no_escalations_for_clear_requirements_batch(self) -> None:
        """Test that no escalations are returned for clear requirements."""
        from yolo_developer.agents.pm import pm_node

        # All clear requirements
        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "API response time must be under 200ms at 95th percentile",
                        "category": "non_functional",
                    },
                    {
                        "id": "req-002",
                        "refined_text": "User login requires valid email and password matching SHA-256 hash",
                        "category": "functional",
                    },
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # Should have no escalations
        assert len(result["escalations"]) == 0
        # Should have stories for all requirements
        assert result["pm_output"]["story_count"] == 2

    @pytest.mark.asyncio
    async def test_escalation_pending_flag_true_when_escalations_exist(self) -> None:
        """Test that escalation_pending is True when escalations exist (AC3)."""
        from yolo_developer.agents.pm import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "The system should be fast",
                        "category": "functional",
                    }
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # AC3: state indicates escalation_pending = True
        assert "escalation_pending" in result
        assert result["escalation_pending"] is True

    @pytest.mark.asyncio
    async def test_escalation_pending_flag_false_when_no_escalations(self) -> None:
        """Test that escalation_pending is False when no escalations."""
        from yolo_developer.agents.pm import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": {
                "requirements": [
                    {
                        "id": "req-001",
                        "refined_text": "API response time must be under 200ms",
                        "category": "non_functional",
                    }
                ],
                "escalations": [],
                "gaps": [],
                "contradictions": [],
            },
        }

        result = await pm_node(state)

        # escalation_pending should be False
        assert "escalation_pending" in result
        assert result["escalation_pending"] is False
