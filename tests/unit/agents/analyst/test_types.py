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
