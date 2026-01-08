"""Unit tests for Analyst agent types (Story 5.1 Task 3).

Tests for CrystallizedRequirement and AnalystOutput frozen dataclasses.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CrystallizedRequirement,
)


class TestCrystallizedRequirement:
    """Tests for CrystallizedRequirement dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """CrystallizedRequirement should be creatable with all fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="The system should be fast",
            refined_text="Response time < 200ms for 95th percentile",
            category="non-functional",
            testable=True,
        )

        assert req.id == "req-001"
        assert req.original_text == "The system should be fast"
        assert req.refined_text == "Response time < 200ms for 95th percentile"
        assert req.category == "non-functional"
        assert req.testable is True

    def test_immutability(self) -> None:
        """CrystallizedRequirement should be immutable (frozen dataclass)."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="original",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        with pytest.raises(AttributeError):
            req.id = "new-id"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        req = CrystallizedRequirement(
            id="req-002",
            original_text="User can log in",
            refined_text="User can authenticate with email/password",
            category="functional",
            testable=True,
        )

        result = req.to_dict()

        assert result == {
            "id": "req-002",
            "original_text": "User can log in",
            "refined_text": "User can authenticate with email/password",
            "category": "functional",
            "testable": True,
            "scope_notes": None,
            "implementation_hints": [],
            "confidence": 1.0,
        }

    def test_equality(self) -> None:
        """CrystallizedRequirement equality should be based on field values."""
        req1 = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
        )
        req2 = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        assert req1 == req2

    def test_hashability(self) -> None:
        """CrystallizedRequirement should be hashable (frozen)."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        # Should be usable in sets and as dict keys
        s = {req}
        assert req in s


class TestAnalystOutput:
    """Tests for AnalystOutput dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """AnalystOutput should be creatable with all fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="original",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        output = AnalystOutput(
            requirements=(req,),
            identified_gaps=("Missing authentication details",),
            contradictions=("Requirement A conflicts with B",),
        )

        assert len(output.requirements) == 1
        assert output.requirements[0] == req
        assert output.identified_gaps == ("Missing authentication details",)
        assert output.contradictions == ("Requirement A conflicts with B",)

    def test_empty_tuples_default(self) -> None:
        """AnalystOutput should allow empty tuples."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        assert output.requirements == ()
        assert output.identified_gaps == ()
        assert output.contradictions == ()

    def test_immutability(self) -> None:
        """AnalystOutput should be immutable (frozen dataclass)."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        with pytest.raises(AttributeError):
            output.requirements = ()  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields including nested requirements."""
        req1 = CrystallizedRequirement(
            id="req-001",
            original_text="orig1",
            refined_text="ref1",
            category="functional",
            testable=True,
        )
        req2 = CrystallizedRequirement(
            id="req-002",
            original_text="orig2",
            refined_text="ref2",
            category="non-functional",
            testable=False,
        )

        output = AnalystOutput(
            requirements=(req1, req2),
            identified_gaps=("gap1", "gap2"),
            contradictions=("conflict1",),
        )

        result = output.to_dict()

        assert "requirements" in result
        assert len(result["requirements"]) == 2
        assert result["requirements"][0]["id"] == "req-001"
        assert result["requirements"][1]["id"] == "req-002"
        assert result["identified_gaps"] == ["gap1", "gap2"]
        assert result["contradictions"] == ["conflict1"]

    def test_hashability(self) -> None:
        """AnalystOutput should be hashable (frozen)."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        s = {output}
        assert output in s


class TestCrystallizedRequirementEnhanced:
    """Tests for enhanced CrystallizedRequirement fields (Story 5.2 Task 1)."""

    def test_new_fields_have_correct_defaults(self) -> None:
        """New optional fields should have correct default values."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="System should be fast",
            refined_text="Response time < 200ms",
            category="non-functional",
            testable=True,
        )

        assert req.scope_notes is None
        assert req.implementation_hints == ()
        assert req.confidence == 1.0

    def test_creation_with_all_new_fields(self) -> None:
        """CrystallizedRequirement should accept all new fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="System should be fast",
            refined_text="Response time < 200ms at 95th percentile",
            category="non-functional",
            testable=True,
            scope_notes="Applies to GET endpoints only; POST excluded",
            implementation_hints=("Use async handlers", "Add response caching"),
            confidence=0.85,
        )

        assert req.scope_notes == "Applies to GET endpoints only; POST excluded"
        assert req.implementation_hints == ("Use async handlers", "Add response caching")
        assert req.confidence == 0.85

    def test_confidence_boundary_values(self) -> None:
        """Confidence should accept boundary values 0.0 and 1.0."""
        req_min = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
            confidence=0.0,
        )
        req_max = CrystallizedRequirement(
            id="req-002",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
            confidence=1.0,
        )

        assert req_min.confidence == 0.0
        assert req_max.confidence == 1.0

    def test_to_dict_includes_new_fields(self) -> None:
        """to_dict should serialize all new fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="System should be fast",
            refined_text="Response time < 200ms",
            category="non-functional",
            testable=True,
            scope_notes="GET endpoints only",
            implementation_hints=("Use caching", "Async handlers"),
            confidence=0.9,
        )

        result = req.to_dict()

        assert result["scope_notes"] == "GET endpoints only"
        assert result["implementation_hints"] == ["Use caching", "Async handlers"]
        assert result["confidence"] == 0.9

    def test_to_dict_with_none_scope_notes(self) -> None:
        """to_dict should handle None scope_notes correctly."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        result = req.to_dict()

        assert result["scope_notes"] is None
        assert result["implementation_hints"] == []
        assert result["confidence"] == 1.0

    def test_backward_compatibility_existing_code(self) -> None:
        """Existing code creating CrystallizedRequirement should still work."""
        # This mimics existing code from Story 5.1 that doesn't use new fields
        req = CrystallizedRequirement(
            id="req-001",
            original_text="original",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        # Should work without new fields
        assert req.id == "req-001"
        result = req.to_dict()
        assert "id" in result
        assert "original_text" in result

    def test_immutability_of_new_fields(self) -> None:
        """New fields should also be immutable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
            scope_notes="notes",
            implementation_hints=("hint1",),
            confidence=0.8,
        )

        with pytest.raises(AttributeError):
            req.scope_notes = "new notes"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            req.confidence = 0.5  # type: ignore[misc]

    def test_hashability_with_new_fields(self) -> None:
        """CrystallizedRequirement with new fields should remain hashable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="refined",
            category="functional",
            testable=True,
            scope_notes="notes",
            implementation_hints=("hint1", "hint2"),
            confidence=0.75,
        )

        s = {req}
        assert req in s
