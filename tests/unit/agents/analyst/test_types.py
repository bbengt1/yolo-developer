"""Unit tests for Analyst agent types (Story 5.1 Task 3, Story 5.3).

Tests for CrystallizedRequirement, AnalystOutput, and gap analysis types.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CrystallizedRequirement,
    GapType,
    IdentifiedGap,
    Severity,
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


class TestGapType:
    """Tests for GapType enum (Story 5.3)."""

    def test_gap_type_values(self) -> None:
        """GapType should have expected string values."""
        assert GapType.EDGE_CASE.value == "edge_case"
        assert GapType.IMPLIED_REQUIREMENT.value == "implied_requirement"
        assert GapType.PATTERN_SUGGESTION.value == "pattern_suggestion"

    def test_gap_type_is_str_enum(self) -> None:
        """GapType should be a string enum."""
        assert isinstance(GapType.EDGE_CASE, str)
        assert GapType.EDGE_CASE.value == "edge_case"

    def test_gap_type_from_string(self) -> None:
        """GapType should be creatable from string value."""
        gap_type = GapType("edge_case")
        assert gap_type == GapType.EDGE_CASE

    def test_gap_type_invalid_value(self) -> None:
        """GapType should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            GapType("invalid_type")


class TestSeverity:
    """Tests for Severity enum (Story 5.3)."""

    def test_severity_values(self) -> None:
        """Severity should have expected string values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"

    def test_severity_is_str_enum(self) -> None:
        """Severity should be a string enum."""
        assert isinstance(Severity.CRITICAL, str)
        assert Severity.HIGH.value == "high"

    def test_severity_from_string(self) -> None:
        """Severity should be creatable from string value."""
        severity = Severity("critical")
        assert severity == Severity.CRITICAL

    def test_severity_invalid_value(self) -> None:
        """Severity should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            Severity("invalid_severity")


class TestIdentifiedGap:
    """Tests for IdentifiedGap dataclass (Story 5.3)."""

    def test_creation_with_all_fields(self) -> None:
        """IdentifiedGap should be creatable with all fields."""
        gap = IdentifiedGap(
            id="gap-001",
            description="Missing error handling for invalid input",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.HIGH,
            source_requirements=("req-001", "req-002"),
            rationale="Input validation requires error response",
        )

        assert gap.id == "gap-001"
        assert gap.description == "Missing error handling for invalid input"
        assert gap.gap_type == GapType.EDGE_CASE
        assert gap.severity == Severity.HIGH
        assert gap.source_requirements == ("req-001", "req-002")
        assert gap.rationale == "Input validation requires error response"

    def test_immutability(self) -> None:
        """IdentifiedGap should be immutable (frozen dataclass)."""
        gap = IdentifiedGap(
            id="gap-001",
            description="desc",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.MEDIUM,
            source_requirements=("req-001",),
            rationale="reason",
        )

        with pytest.raises(AttributeError):
            gap.id = "new-id"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            gap.severity = Severity.HIGH  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly with enum values."""
        gap = IdentifiedGap(
            id="gap-001",
            description="Missing logout functionality",
            gap_type=GapType.IMPLIED_REQUIREMENT,
            severity=Severity.MEDIUM,
            source_requirements=("req-001",),
            rationale="Login implies logout needed",
        )

        result = gap.to_dict()

        assert result == {
            "id": "gap-001",
            "description": "Missing logout functionality",
            "gap_type": "implied_requirement",
            "severity": "medium",
            "source_requirements": ["req-001"],
            "rationale": "Login implies logout needed",
        }

    def test_to_dict_converts_enums_to_strings(self) -> None:
        """to_dict should convert enums to their string values."""
        gap = IdentifiedGap(
            id="gap-001",
            description="Pattern suggestion",
            gap_type=GapType.PATTERN_SUGGESTION,
            severity=Severity.LOW,
            source_requirements=(),
            rationale="Domain pattern",
        )

        result = gap.to_dict()

        assert result["gap_type"] == "pattern_suggestion"
        assert result["severity"] == "low"
        assert isinstance(result["gap_type"], str)
        assert isinstance(result["severity"], str)

    def test_source_requirements_is_list_in_dict(self) -> None:
        """to_dict should convert source_requirements tuple to list."""
        gap = IdentifiedGap(
            id="gap-001",
            description="desc",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.MEDIUM,
            source_requirements=("req-001", "req-002", "req-003"),
            rationale="reason",
        )

        result = gap.to_dict()

        assert isinstance(result["source_requirements"], list)
        assert result["source_requirements"] == ["req-001", "req-002", "req-003"]

    def test_equality(self) -> None:
        """IdentifiedGap equality should be based on field values."""
        gap1 = IdentifiedGap(
            id="gap-001",
            description="desc",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.MEDIUM,
            source_requirements=("req-001",),
            rationale="reason",
        )
        gap2 = IdentifiedGap(
            id="gap-001",
            description="desc",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.MEDIUM,
            source_requirements=("req-001",),
            rationale="reason",
        )

        assert gap1 == gap2

    def test_hashability(self) -> None:
        """IdentifiedGap should be hashable (frozen)."""
        gap = IdentifiedGap(
            id="gap-001",
            description="desc",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.MEDIUM,
            source_requirements=("req-001",),
            rationale="reason",
        )

        s = {gap}
        assert gap in s


class TestAnalystOutputWithStructuredGaps:
    """Tests for AnalystOutput with structured_gaps field (Story 5.3)."""

    def test_structured_gaps_default_empty(self) -> None:
        """structured_gaps should default to empty tuple."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        assert output.structured_gaps == ()

    def test_creation_with_structured_gaps(self) -> None:
        """AnalystOutput should accept structured_gaps field."""
        gap = IdentifiedGap(
            id="gap-001",
            description="Missing logout",
            gap_type=GapType.IMPLIED_REQUIREMENT,
            severity=Severity.HIGH,
            source_requirements=("req-001",),
            rationale="Login implies logout",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            structured_gaps=(gap,),
        )

        assert len(output.structured_gaps) == 1
        assert output.structured_gaps[0] == gap

    def test_to_dict_includes_structured_gaps(self) -> None:
        """to_dict should serialize structured_gaps."""
        gap = IdentifiedGap(
            id="gap-001",
            description="Missing feature",
            gap_type=GapType.PATTERN_SUGGESTION,
            severity=Severity.LOW,
            source_requirements=("req-001",),
            rationale="Pattern suggests this",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=("legacy gap",),
            contradictions=(),
            structured_gaps=(gap,),
        )

        result = output.to_dict()

        assert "structured_gaps" in result
        assert len(result["structured_gaps"]) == 1
        assert result["structured_gaps"][0]["id"] == "gap-001"
        assert result["structured_gaps"][0]["gap_type"] == "pattern_suggestion"

    def test_backward_compatibility_without_structured_gaps(self) -> None:
        """Existing code without structured_gaps should still work."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="refined",
            category="functional",
            testable=True,
        )

        # Create output the old way (without structured_gaps)
        output = AnalystOutput(
            requirements=(req,),
            identified_gaps=("gap1",),
            contradictions=(),
        )

        # Should work and have empty structured_gaps
        assert output.structured_gaps == ()
        result = output.to_dict()
        assert "structured_gaps" in result
        assert result["structured_gaps"] == []

    def test_to_dict_with_multiple_structured_gaps(self) -> None:
        """to_dict should serialize multiple structured_gaps correctly."""
        gap1 = IdentifiedGap(
            id="gap-001",
            description="Edge case 1",
            gap_type=GapType.EDGE_CASE,
            severity=Severity.HIGH,
            source_requirements=("req-001",),
            rationale="reason 1",
        )
        gap2 = IdentifiedGap(
            id="gap-002",
            description="Implied req",
            gap_type=GapType.IMPLIED_REQUIREMENT,
            severity=Severity.MEDIUM,
            source_requirements=("req-002",),
            rationale="reason 2",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            structured_gaps=(gap1, gap2),
        )

        result = output.to_dict()

        assert len(result["structured_gaps"]) == 2
        assert result["structured_gaps"][0]["severity"] == "high"
        assert result["structured_gaps"][1]["severity"] == "medium"
