"""Unit tests for Analyst agent types (Story 5.1 Task 3, Story 5.3, Story 5.4, Story 5.5, Story 5.7).

Tests for CrystallizedRequirement, AnalystOutput, gap analysis types,
categorization types, and implementability types.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CategorizationResult,
    ComplexityLevel,
    ConstraintSubCategory,
    Contradiction,
    ContradictionType,
    CrystallizedRequirement,
    DependencyType,
    Escalation,
    EscalationPriority,
    EscalationReason,
    ExternalDependency,
    FunctionalSubCategory,
    GapType,
    IdentifiedGap,
    ImplementabilityResult,
    ImplementabilityStatus,
    NonFunctionalSubCategory,
    RequirementCategory,
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
            # Story 5.4 categorization fields
            "sub_category": None,
            "category_confidence": 1.0,
            "category_rationale": None,
            # Story 5.5 implementability fields
            "implementability_status": None,
            "complexity": None,
            "external_dependencies": [],
            "implementability_issues": [],
            "implementability_rationale": None,
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


class TestRequirementCategory:
    """Tests for RequirementCategory enum (Story 5.4)."""

    def test_requirement_category_values(self) -> None:
        """RequirementCategory should have expected string values."""
        assert RequirementCategory.FUNCTIONAL.value == "functional"
        assert RequirementCategory.NON_FUNCTIONAL.value == "non_functional"
        assert RequirementCategory.CONSTRAINT.value == "constraint"

    def test_requirement_category_is_str_enum(self) -> None:
        """RequirementCategory should be a string enum."""
        assert isinstance(RequirementCategory.FUNCTIONAL, str)
        assert RequirementCategory.FUNCTIONAL == "functional"

    def test_requirement_category_from_string(self) -> None:
        """RequirementCategory should be creatable from string value."""
        cat = RequirementCategory("functional")
        assert cat == RequirementCategory.FUNCTIONAL

    def test_requirement_category_invalid_value(self) -> None:
        """RequirementCategory should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            RequirementCategory("invalid_category")

    def test_requirement_category_all_members(self) -> None:
        """RequirementCategory should have exactly 3 members."""
        members = list(RequirementCategory)
        assert len(members) == 3
        assert RequirementCategory.FUNCTIONAL in members
        assert RequirementCategory.NON_FUNCTIONAL in members
        assert RequirementCategory.CONSTRAINT in members


class TestFunctionalSubCategory:
    """Tests for FunctionalSubCategory enum (Story 5.4)."""

    def test_functional_subcategory_values(self) -> None:
        """FunctionalSubCategory should have expected string values."""
        assert FunctionalSubCategory.USER_MANAGEMENT.value == "user_management"
        assert FunctionalSubCategory.DATA_OPERATIONS.value == "data_operations"
        assert FunctionalSubCategory.INTEGRATION.value == "integration"
        assert FunctionalSubCategory.REPORTING.value == "reporting"
        assert FunctionalSubCategory.WORKFLOW.value == "workflow"
        assert FunctionalSubCategory.COMMUNICATION.value == "communication"

    def test_functional_subcategory_is_str_enum(self) -> None:
        """FunctionalSubCategory should be a string enum."""
        assert isinstance(FunctionalSubCategory.USER_MANAGEMENT, str)
        assert FunctionalSubCategory.USER_MANAGEMENT == "user_management"

    def test_functional_subcategory_from_string(self) -> None:
        """FunctionalSubCategory should be creatable from string value."""
        subcat = FunctionalSubCategory("user_management")
        assert subcat == FunctionalSubCategory.USER_MANAGEMENT

    def test_functional_subcategory_all_members(self) -> None:
        """FunctionalSubCategory should have exactly 6 members."""
        members = list(FunctionalSubCategory)
        assert len(members) == 6


class TestNonFunctionalSubCategory:
    """Tests for NonFunctionalSubCategory enum (Story 5.4)."""

    def test_non_functional_subcategory_values(self) -> None:
        """NonFunctionalSubCategory should have expected string values."""
        assert NonFunctionalSubCategory.PERFORMANCE.value == "performance"
        assert NonFunctionalSubCategory.SECURITY.value == "security"
        assert NonFunctionalSubCategory.USABILITY.value == "usability"
        assert NonFunctionalSubCategory.RELIABILITY.value == "reliability"
        assert NonFunctionalSubCategory.SCALABILITY.value == "scalability"
        assert NonFunctionalSubCategory.MAINTAINABILITY.value == "maintainability"
        assert NonFunctionalSubCategory.ACCESSIBILITY.value == "accessibility"

    def test_non_functional_subcategory_is_str_enum(self) -> None:
        """NonFunctionalSubCategory should be a string enum."""
        assert isinstance(NonFunctionalSubCategory.PERFORMANCE, str)
        assert NonFunctionalSubCategory.PERFORMANCE == "performance"

    def test_non_functional_subcategory_from_string(self) -> None:
        """NonFunctionalSubCategory should be creatable from string value."""
        subcat = NonFunctionalSubCategory("security")
        assert subcat == NonFunctionalSubCategory.SECURITY

    def test_non_functional_subcategory_all_members(self) -> None:
        """NonFunctionalSubCategory should have exactly 7 members."""
        members = list(NonFunctionalSubCategory)
        assert len(members) == 7


class TestConstraintSubCategory:
    """Tests for ConstraintSubCategory enum (Story 5.4)."""

    def test_constraint_subcategory_values(self) -> None:
        """ConstraintSubCategory should have expected string values."""
        assert ConstraintSubCategory.TECHNICAL.value == "technical"
        assert ConstraintSubCategory.BUSINESS.value == "business"
        assert ConstraintSubCategory.REGULATORY.value == "regulatory"
        assert ConstraintSubCategory.RESOURCE.value == "resource"
        assert ConstraintSubCategory.TIMELINE.value == "timeline"

    def test_constraint_subcategory_is_str_enum(self) -> None:
        """ConstraintSubCategory should be a string enum."""
        assert isinstance(ConstraintSubCategory.TECHNICAL, str)
        assert ConstraintSubCategory.TECHNICAL == "technical"

    def test_constraint_subcategory_from_string(self) -> None:
        """ConstraintSubCategory should be creatable from string value."""
        subcat = ConstraintSubCategory("regulatory")
        assert subcat == ConstraintSubCategory.REGULATORY

    def test_constraint_subcategory_all_members(self) -> None:
        """ConstraintSubCategory should have exactly 5 members."""
        members = list(ConstraintSubCategory)
        assert len(members) == 5


class TestCategorizationResult:
    """Tests for CategorizationResult dataclass (Story 5.4)."""

    def test_creation_with_all_fields(self) -> None:
        """CategorizationResult should be creatable with all fields."""
        result = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category="user_management",
            confidence=0.95,
            rationale="Contains 'login', 'user' - clear functional requirement",
        )

        assert result.category == RequirementCategory.FUNCTIONAL
        assert result.sub_category == "user_management"
        assert result.confidence == 0.95
        assert "login" in result.rationale

    def test_creation_with_none_sub_category(self) -> None:
        """CategorizationResult should allow None sub_category."""
        result = CategorizationResult(
            category=RequirementCategory.CONSTRAINT,
            sub_category=None,
            confidence=0.6,
            rationale="Ambiguous constraint type",
        )

        assert result.sub_category is None

    def test_immutability(self) -> None:
        """CategorizationResult should be immutable (frozen dataclass)."""
        result = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category="data_operations",
            confidence=0.8,
            rationale="CRUD keywords",
        )

        with pytest.raises(AttributeError):
            result.category = RequirementCategory.CONSTRAINT  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        result = CategorizationResult(
            category=RequirementCategory.NON_FUNCTIONAL,
            sub_category="performance",
            confidence=0.9,
            rationale="Response time mentioned",
        )

        d = result.to_dict()

        assert d == {
            "category": "non_functional",
            "sub_category": "performance",
            "confidence": 0.9,
            "rationale": "Response time mentioned",
        }

    def test_to_dict_with_none_sub_category(self) -> None:
        """to_dict should handle None sub_category correctly."""
        result = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category=None,
            confidence=0.7,
            rationale="No clear sub-category",
        )

        d = result.to_dict()

        assert d["sub_category"] is None

    def test_to_dict_converts_enum_to_string(self) -> None:
        """to_dict should convert category enum to string value."""
        result = CategorizationResult(
            category=RequirementCategory.CONSTRAINT,
            sub_category="technical",
            confidence=0.85,
            rationale="Tech stack mentioned",
        )

        d = result.to_dict()

        assert d["category"] == "constraint"
        assert isinstance(d["category"], str)

    def test_hashability(self) -> None:
        """CategorizationResult should be hashable (frozen)."""
        result = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category="integration",
            confidence=0.75,
            rationale="API keywords",
        )

        s = {result}
        assert result in s

    def test_equality(self) -> None:
        """CategorizationResult equality should be based on field values."""
        result1 = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category="reporting",
            confidence=0.8,
            rationale="Report export",
        )
        result2 = CategorizationResult(
            category=RequirementCategory.FUNCTIONAL,
            sub_category="reporting",
            confidence=0.8,
            rationale="Report export",
        )

        assert result1 == result2


class TestCrystallizedRequirementCategorization:
    """Tests for CrystallizedRequirement categorization fields (Story 5.4 Task 3)."""

    def test_new_categorization_fields_have_defaults(self) -> None:
        """New categorization fields should have correct default values."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates with email",
            category="functional",
            testable=True,
        )

        assert req.sub_category is None
        assert req.category_confidence == 1.0
        assert req.category_rationale is None

    def test_creation_with_all_categorization_fields(self) -> None:
        """CrystallizedRequirement should accept all new categorization fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates with email",
            category="functional",
            testable=True,
            sub_category="user_management",
            category_confidence=0.95,
            category_rationale="Contains 'login', 'user' keywords",
        )

        assert req.sub_category == "user_management"
        assert req.category_confidence == 0.95
        assert req.category_rationale == "Contains 'login', 'user' keywords"

    def test_to_dict_includes_categorization_fields(self) -> None:
        """to_dict should include new categorization fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="non_functional",
            testable=True,
            sub_category="performance",
            category_confidence=0.85,
            category_rationale="Response time mentioned",
        )

        d = req.to_dict()

        assert "sub_category" in d
        assert "category_confidence" in d
        assert "category_rationale" in d
        assert d["sub_category"] == "performance"
        assert d["category_confidence"] == 0.85
        assert d["category_rationale"] == "Response time mentioned"

    def test_to_dict_with_none_categorization_fields(self) -> None:
        """to_dict should handle None categorization fields correctly."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="functional",
            testable=True,
        )

        d = req.to_dict()

        assert d["sub_category"] is None
        assert d["category_confidence"] == 1.0
        assert d["category_rationale"] is None

    def test_category_confidence_boundary_values(self) -> None:
        """category_confidence should accept 0.0 and 1.0 boundary values."""
        req_min = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="ref",
            category="functional",
            testable=True,
            category_confidence=0.0,
        )
        req_max = CrystallizedRequirement(
            id="req-002",
            original_text="text",
            refined_text="ref",
            category="functional",
            testable=True,
            category_confidence=1.0,
        )

        assert req_min.category_confidence == 0.0
        assert req_max.category_confidence == 1.0

    def test_backward_compatibility_existing_to_dict(self) -> None:
        """Existing to_dict format should still include original fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="functional",
            testable=True,
            scope_notes="scope",
            implementation_hints=("hint1",),
            confidence=0.9,
        )

        d = req.to_dict()

        # Original fields still present
        assert d["id"] == "req-001"
        assert d["original_text"] == "orig"
        assert d["refined_text"] == "ref"
        assert d["category"] == "functional"
        assert d["testable"] is True
        assert d["scope_notes"] == "scope"
        assert d["implementation_hints"] == ["hint1"]
        assert d["confidence"] == 0.9

    def test_immutability_of_categorization_fields(self) -> None:
        """New categorization fields should also be immutable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="ref",
            category="functional",
            testable=True,
            sub_category="data_operations",
            category_confidence=0.8,
            category_rationale="CRUD keywords",
        )

        with pytest.raises(AttributeError):
            req.sub_category = "user_management"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            req.category_confidence = 0.5  # type: ignore[misc]

        with pytest.raises(AttributeError):
            req.category_rationale = "new rationale"  # type: ignore[misc]

    def test_hashability_with_categorization_fields(self) -> None:
        """CrystallizedRequirement with categorization fields should be hashable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="ref",
            category="constraint",
            testable=True,
            sub_category="technical",
            category_confidence=0.7,
            category_rationale="Tech stack specified",
        )

        s = {req}
        assert req in s


# =============================================================================
# Story 5.5: Implementability Types Tests
# =============================================================================


class TestImplementabilityStatus:
    """Tests for ImplementabilityStatus enum (Story 5.5)."""

    def test_implementability_status_values(self) -> None:
        """ImplementabilityStatus should have expected string values."""
        assert ImplementabilityStatus.IMPLEMENTABLE.value == "implementable"
        assert ImplementabilityStatus.NEEDS_CLARIFICATION.value == "needs_clarification"
        assert ImplementabilityStatus.NOT_IMPLEMENTABLE.value == "not_implementable"

    def test_implementability_status_is_str_enum(self) -> None:
        """ImplementabilityStatus should be a string enum."""
        assert isinstance(ImplementabilityStatus.IMPLEMENTABLE, str)
        assert ImplementabilityStatus.IMPLEMENTABLE == "implementable"

    def test_implementability_status_from_string(self) -> None:
        """ImplementabilityStatus should be creatable from string value."""
        status = ImplementabilityStatus("implementable")
        assert status == ImplementabilityStatus.IMPLEMENTABLE

    def test_implementability_status_invalid_value(self) -> None:
        """ImplementabilityStatus should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            ImplementabilityStatus("invalid_status")

    def test_implementability_status_all_members(self) -> None:
        """ImplementabilityStatus should have exactly 3 members."""
        members = list(ImplementabilityStatus)
        assert len(members) == 3


class TestComplexityLevel:
    """Tests for ComplexityLevel enum (Story 5.5)."""

    def test_complexity_level_values(self) -> None:
        """ComplexityLevel should have expected string values."""
        assert ComplexityLevel.LOW.value == "low"
        assert ComplexityLevel.MEDIUM.value == "medium"
        assert ComplexityLevel.HIGH.value == "high"
        assert ComplexityLevel.VERY_HIGH.value == "very_high"

    def test_complexity_level_is_str_enum(self) -> None:
        """ComplexityLevel should be a string enum."""
        assert isinstance(ComplexityLevel.LOW, str)
        assert ComplexityLevel.LOW == "low"

    def test_complexity_level_from_string(self) -> None:
        """ComplexityLevel should be creatable from string value."""
        complexity = ComplexityLevel("medium")
        assert complexity == ComplexityLevel.MEDIUM

    def test_complexity_level_invalid_value(self) -> None:
        """ComplexityLevel should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            ComplexityLevel("invalid_level")

    def test_complexity_level_all_members(self) -> None:
        """ComplexityLevel should have exactly 4 members."""
        members = list(ComplexityLevel)
        assert len(members) == 4


class TestDependencyType:
    """Tests for DependencyType enum (Story 5.5)."""

    def test_dependency_type_values(self) -> None:
        """DependencyType should have expected string values."""
        assert DependencyType.API.value == "api"
        assert DependencyType.LIBRARY.value == "library"
        assert DependencyType.SERVICE.value == "service"
        assert DependencyType.INFRASTRUCTURE.value == "infrastructure"
        assert DependencyType.DATA_SOURCE.value == "data_source"

    def test_dependency_type_is_str_enum(self) -> None:
        """DependencyType should be a string enum."""
        assert isinstance(DependencyType.API, str)
        assert DependencyType.API == "api"

    def test_dependency_type_from_string(self) -> None:
        """DependencyType should be creatable from string value."""
        dep_type = DependencyType("service")
        assert dep_type == DependencyType.SERVICE

    def test_dependency_type_invalid_value(self) -> None:
        """DependencyType should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            DependencyType("invalid_type")

    def test_dependency_type_all_members(self) -> None:
        """DependencyType should have exactly 5 members."""
        members = list(DependencyType)
        assert len(members) == 5


class TestExternalDependency:
    """Tests for ExternalDependency dataclass (Story 5.5)."""

    def test_creation_with_all_fields(self) -> None:
        """ExternalDependency should be creatable with all fields."""
        dep = ExternalDependency(
            name="PostgreSQL",
            dependency_type=DependencyType.INFRASTRUCTURE,
            description="Relational database for persistent storage",
            availability_notes="Widely available, managed services exist",
            criticality="required",
        )

        assert dep.name == "PostgreSQL"
        assert dep.dependency_type == DependencyType.INFRASTRUCTURE
        assert dep.description == "Relational database for persistent storage"
        assert dep.availability_notes == "Widely available, managed services exist"
        assert dep.criticality == "required"

    def test_immutability(self) -> None:
        """ExternalDependency should be immutable (frozen dataclass)."""
        dep = ExternalDependency(
            name="Redis",
            dependency_type=DependencyType.INFRASTRUCTURE,
            description="Cache",
            availability_notes="Easy to provision",
            criticality="optional",
        )

        with pytest.raises(AttributeError):
            dep.name = "Memcached"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            dep.criticality = "required"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        dep = ExternalDependency(
            name="Stripe",
            dependency_type=DependencyType.API,
            description="Payment processing API",
            availability_notes="Requires API key",
            criticality="required",
        )

        result = dep.to_dict()

        assert result == {
            "name": "Stripe",
            "dependency_type": "api",
            "description": "Payment processing API",
            "availability_notes": "Requires API key",
            "criticality": "required",
        }

    def test_to_dict_converts_enum_to_string(self) -> None:
        """to_dict should convert dependency_type enum to string value."""
        dep = ExternalDependency(
            name="AWS",
            dependency_type=DependencyType.SERVICE,
            description="Cloud provider",
            availability_notes="Account needed",
            criticality="required",
        )

        result = dep.to_dict()

        assert result["dependency_type"] == "service"
        assert isinstance(result["dependency_type"], str)

    def test_hashability(self) -> None:
        """ExternalDependency should be hashable (frozen)."""
        dep = ExternalDependency(
            name="Redis",
            dependency_type=DependencyType.INFRASTRUCTURE,
            description="Cache",
            availability_notes="notes",
            criticality="optional",
        )

        s = {dep}
        assert dep in s


class TestImplementabilityResult:
    """Tests for ImplementabilityResult dataclass (Story 5.5)."""

    def test_creation_with_all_fields(self) -> None:
        """ImplementabilityResult should be creatable with all fields."""
        dep = ExternalDependency(
            name="PostgreSQL",
            dependency_type=DependencyType.INFRASTRUCTURE,
            description="Database",
            availability_notes="notes",
            criticality="required",
        )

        result = ImplementabilityResult(
            status=ImplementabilityStatus.IMPLEMENTABLE,
            complexity=ComplexityLevel.MEDIUM,
            dependencies=(dep,),
            issues=(),
            remediation_suggestions=(),
            rationale="Standard CRUD requirement",
        )

        assert result.status == ImplementabilityStatus.IMPLEMENTABLE
        assert result.complexity == ComplexityLevel.MEDIUM
        assert len(result.dependencies) == 1
        assert result.issues == ()
        assert result.rationale == "Standard CRUD requirement"

    def test_creation_with_issues(self) -> None:
        """ImplementabilityResult should handle issues and remediations."""
        result = ImplementabilityResult(
            status=ImplementabilityStatus.NOT_IMPLEMENTABLE,
            complexity=ComplexityLevel.HIGH,
            dependencies=(),
            issues=("100% uptime is impossible",),
            remediation_suggestions=("Use 99.9% SLA instead",),
            rationale="Absolute guarantees are infeasible",
        )

        assert result.status == ImplementabilityStatus.NOT_IMPLEMENTABLE
        assert len(result.issues) == 1
        assert len(result.remediation_suggestions) == 1

    def test_immutability(self) -> None:
        """ImplementabilityResult should be immutable (frozen dataclass)."""
        result = ImplementabilityResult(
            status=ImplementabilityStatus.IMPLEMENTABLE,
            complexity=ComplexityLevel.LOW,
            dependencies=(),
            issues=(),
            remediation_suggestions=(),
            rationale="Simple",
        )

        with pytest.raises(AttributeError):
            result.status = ImplementabilityStatus.NOT_IMPLEMENTABLE  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.complexity = ComplexityLevel.HIGH  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        dep = ExternalDependency(
            name="Redis",
            dependency_type=DependencyType.INFRASTRUCTURE,
            description="Cache",
            availability_notes="notes",
            criticality="optional",
        )

        result = ImplementabilityResult(
            status=ImplementabilityStatus.IMPLEMENTABLE,
            complexity=ComplexityLevel.MEDIUM,
            dependencies=(dep,),
            issues=("minor issue",),
            remediation_suggestions=("fix it",),
            rationale="Needs caching",
        )

        d = result.to_dict()

        assert d["status"] == "implementable"
        assert d["complexity"] == "medium"
        assert len(d["dependencies"]) == 1
        assert d["dependencies"][0]["name"] == "Redis"
        assert d["issues"] == ["minor issue"]
        assert d["remediation_suggestions"] == ["fix it"]
        assert d["rationale"] == "Needs caching"

    def test_to_dict_converts_enums_to_strings(self) -> None:
        """to_dict should convert all enums to string values."""
        result = ImplementabilityResult(
            status=ImplementabilityStatus.NEEDS_CLARIFICATION,
            complexity=ComplexityLevel.HIGH,
            dependencies=(),
            issues=(),
            remediation_suggestions=(),
            rationale="reason",
        )

        d = result.to_dict()

        assert d["status"] == "needs_clarification"
        assert d["complexity"] == "high"
        assert isinstance(d["status"], str)
        assert isinstance(d["complexity"], str)

    def test_hashability(self) -> None:
        """ImplementabilityResult should be hashable (frozen)."""
        result = ImplementabilityResult(
            status=ImplementabilityStatus.IMPLEMENTABLE,
            complexity=ComplexityLevel.LOW,
            dependencies=(),
            issues=(),
            remediation_suggestions=(),
            rationale="simple",
        )

        s = {result}
        assert result in s


class TestCrystallizedRequirementImplementability:
    """Tests for CrystallizedRequirement implementability fields (Story 5.5)."""

    def test_new_implementability_fields_have_defaults(self) -> None:
        """New implementability fields should have correct default values."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates with email",
            category="functional",
            testable=True,
        )

        assert req.implementability_status is None
        assert req.complexity is None
        assert req.external_dependencies == ()
        assert req.implementability_issues == ()
        assert req.implementability_rationale is None

    def test_creation_with_all_implementability_fields(self) -> None:
        """CrystallizedRequirement should accept all implementability fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="User can login",
            refined_text="User authenticates with email",
            category="functional",
            testable=True,
            implementability_status="implementable",
            complexity="low",
            external_dependencies=({"name": "Auth0", "dependency_type": "service"},),
            implementability_issues=(),
            implementability_rationale="Standard auth pattern",
        )

        assert req.implementability_status == "implementable"
        assert req.complexity == "low"
        assert len(req.external_dependencies) == 1
        assert req.implementability_issues == ()
        assert req.implementability_rationale == "Standard auth pattern"

    def test_to_dict_includes_implementability_fields(self) -> None:
        """to_dict should include new implementability fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="functional",
            testable=True,
            implementability_status="implementable",
            complexity="medium",
            external_dependencies=({"name": "PostgreSQL"},),
            implementability_issues=("minor issue",),
            implementability_rationale="Needs database",
        )

        d = req.to_dict()

        assert d["implementability_status"] == "implementable"
        assert d["complexity"] == "medium"
        assert d["external_dependencies"] == [{"name": "PostgreSQL"}]
        assert d["implementability_issues"] == ["minor issue"]
        assert d["implementability_rationale"] == "Needs database"

    def test_to_dict_with_none_implementability_fields(self) -> None:
        """to_dict should handle None implementability fields correctly."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="functional",
            testable=True,
        )

        d = req.to_dict()

        assert d["implementability_status"] is None
        assert d["complexity"] is None
        assert d["external_dependencies"] == []
        assert d["implementability_issues"] == []
        assert d["implementability_rationale"] is None

    def test_immutability_of_implementability_fields(self) -> None:
        """New implementability fields should also be immutable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="ref",
            category="functional",
            testable=True,
            implementability_status="implementable",
            complexity="low",
            implementability_rationale="reason",
        )

        with pytest.raises(AttributeError):
            req.implementability_status = "not_implementable"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            req.complexity = "high"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            req.implementability_rationale = "new reason"  # type: ignore[misc]

    def test_hashability_with_implementability_fields(self) -> None:
        """CrystallizedRequirement with implementability fields should be hashable."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="text",
            refined_text="ref",
            category="functional",
            testable=True,
            implementability_status="implementable",
            complexity="medium",
            implementability_rationale="reason",
        )

        s = {req}
        assert req in s

    def test_backward_compatibility_with_all_previous_fields(self) -> None:
        """CrystallizedRequirement should work with all previous story fields."""
        req = CrystallizedRequirement(
            id="req-001",
            original_text="orig",
            refined_text="ref",
            category="functional",
            testable=True,
            # Story 5.2 fields
            scope_notes="scope",
            implementation_hints=("hint1",),
            confidence=0.9,
            # Story 5.4 fields
            sub_category="user_management",
            category_confidence=0.85,
            category_rationale="login keywords",
            # Story 5.5 fields
            implementability_status="implementable",
            complexity="low",
            external_dependencies=(),
            implementability_issues=(),
            implementability_rationale="standard pattern",
        )

        d = req.to_dict()

        # All fields should be present
        assert d["id"] == "req-001"
        assert d["scope_notes"] == "scope"
        assert d["sub_category"] == "user_management"
        assert d["implementability_status"] == "implementable"


# =============================================================================
# Story 5.6: Contradiction Flagging Types
# =============================================================================


class TestContradictionType:
    """Tests for ContradictionType enum (Story 5.6)."""

    def test_contradiction_type_values(self) -> None:
        """ContradictionType should have expected string values."""
        assert ContradictionType.DIRECT.value == "direct"
        assert ContradictionType.IMPLICIT_RESOURCE.value == "implicit_resource"
        assert ContradictionType.IMPLICIT_BEHAVIOR.value == "implicit_behavior"
        assert ContradictionType.SEMANTIC.value == "semantic"

    def test_contradiction_type_is_str_enum(self) -> None:
        """ContradictionType should be a string enum."""
        assert isinstance(ContradictionType.DIRECT, str)
        assert ContradictionType.DIRECT == "direct"

    def test_contradiction_type_from_string(self) -> None:
        """ContradictionType should be creatable from string value."""
        ct = ContradictionType("semantic")
        assert ct == ContradictionType.SEMANTIC

    def test_contradiction_type_invalid_value(self) -> None:
        """ContradictionType should raise ValueError for invalid values."""
        with pytest.raises(ValueError):
            ContradictionType("invalid_type")

    def test_contradiction_type_all_members(self) -> None:
        """ContradictionType should have exactly 4 members."""
        members = list(ContradictionType)
        assert len(members) == 4


class TestContradiction:
    """Tests for Contradiction dataclass (Story 5.6)."""

    def test_creation_with_all_fields(self) -> None:
        """Contradiction should be creatable with all fields."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001", "req-002"),
            description="Encryption vs plaintext conflict",
            explanation="req-001 requires encryption, req-002 requires plaintext",
            severity=Severity.CRITICAL,
            resolution_suggestions=("Clarify security requirements",),
        )

        assert conflict.id == "conflict-001"
        assert conflict.contradiction_type == ContradictionType.DIRECT
        assert conflict.requirement_ids == ("req-001", "req-002")
        assert conflict.description == "Encryption vs plaintext conflict"
        assert conflict.explanation == "req-001 requires encryption, req-002 requires plaintext"
        assert conflict.severity == Severity.CRITICAL
        assert conflict.resolution_suggestions == ("Clarify security requirements",)

    def test_immutability(self) -> None:
        """Contradiction should be immutable (frozen dataclass)."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.SEMANTIC,
            requirement_ids=("req-001", "req-002"),
            description="desc",
            explanation="expl",
            severity=Severity.HIGH,
            resolution_suggestions=("suggestion",),
        )

        with pytest.raises(AttributeError):
            conflict.id = "new-id"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize all fields correctly."""
        conflict = Contradiction(
            id="conflict-002",
            contradiction_type=ContradictionType.IMPLICIT_RESOURCE,
            requirement_ids=("req-003", "req-004"),
            description="Memory conflict",
            explanation="One needs high memory, other needs low memory",
            severity=Severity.MEDIUM,
            resolution_suggestions=("Define resource constraints", "Split tiers"),
        )

        result = conflict.to_dict()

        assert result == {
            "id": "conflict-002",
            "contradiction_type": "implicit_resource",
            "requirement_ids": ["req-003", "req-004"],
            "description": "Memory conflict",
            "explanation": "One needs high memory, other needs low memory",
            "severity": "medium",
            "resolution_suggestions": ["Define resource constraints", "Split tiers"],
        }

    def test_to_dict_converts_enums_to_strings(self) -> None:
        """to_dict should convert enums to their string values."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.IMPLICIT_BEHAVIOR,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.LOW,
            resolution_suggestions=(),
        )

        d = conflict.to_dict()

        assert d["contradiction_type"] == "implicit_behavior"
        assert d["severity"] == "low"

    def test_requirement_ids_is_list_in_dict(self) -> None:
        """requirement_ids should be converted to list in to_dict."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001", "req-002", "req-003"),
            description="desc",
            explanation="expl",
            severity=Severity.HIGH,
            resolution_suggestions=("s1",),
        )

        d = conflict.to_dict()

        assert isinstance(d["requirement_ids"], list)
        assert d["requirement_ids"] == ["req-001", "req-002", "req-003"]

    def test_resolution_suggestions_is_list_in_dict(self) -> None:
        """resolution_suggestions should be converted to list in to_dict."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.SEMANTIC,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.MEDIUM,
            resolution_suggestions=("suggestion1", "suggestion2"),
        )

        d = conflict.to_dict()

        assert isinstance(d["resolution_suggestions"], list)
        assert d["resolution_suggestions"] == ["suggestion1", "suggestion2"]

    def test_equality(self) -> None:
        """Contradiction equality should be based on field values."""
        conflict1 = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.HIGH,
            resolution_suggestions=(),
        )
        conflict2 = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.HIGH,
            resolution_suggestions=(),
        )

        assert conflict1 == conflict2

    def test_hashability(self) -> None:
        """Contradiction should be hashable for use in sets/dicts."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.SEMANTIC,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.LOW,
            resolution_suggestions=("s1",),
        )

        s = {conflict}
        assert conflict in s


class TestAnalystOutputWithStructuredContradictions:
    """Tests for AnalystOutput structured_contradictions field (Story 5.6)."""

    def test_structured_contradictions_default_empty(self) -> None:
        """structured_contradictions should default to empty tuple."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        assert output.structured_contradictions == ()

    def test_creation_with_structured_contradictions(self) -> None:
        """AnalystOutput should accept structured_contradictions."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001", "req-002"),
            description="Encryption conflict",
            explanation="Conflicting encryption requirements",
            severity=Severity.CRITICAL,
            resolution_suggestions=("Clarify requirements",),
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=("legacy conflict",),
            structured_contradictions=(conflict,),
        )

        assert len(output.structured_contradictions) == 1
        assert output.structured_contradictions[0].id == "conflict-001"

    def test_to_dict_includes_structured_contradictions(self) -> None:
        """to_dict should include structured_contradictions."""
        conflict = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.SEMANTIC,
            requirement_ids=("req-001",),
            description="desc",
            explanation="expl",
            severity=Severity.HIGH,
            resolution_suggestions=(),
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            structured_contradictions=(conflict,),
        )

        d = output.to_dict()

        assert "structured_contradictions" in d
        assert len(d["structured_contradictions"]) == 1
        assert d["structured_contradictions"][0]["id"] == "conflict-001"
        assert d["structured_contradictions"][0]["contradiction_type"] == "semantic"

    def test_backward_compatibility_without_structured_contradictions(self) -> None:
        """AnalystOutput should work without structured_contradictions (backward compat)."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=("gap1",),
            contradictions=("conflict1",),
        )

        d = output.to_dict()

        assert d["identified_gaps"] == ["gap1"]
        assert d["contradictions"] == ["conflict1"]
        assert d["structured_contradictions"] == []

    def test_to_dict_with_multiple_structured_contradictions(self) -> None:
        """to_dict should serialize multiple structured contradictions."""
        conflict1 = Contradiction(
            id="conflict-001",
            contradiction_type=ContradictionType.DIRECT,
            requirement_ids=("req-001", "req-002"),
            description="Direct conflict",
            explanation="Must vs must not",
            severity=Severity.CRITICAL,
            resolution_suggestions=("Clarify",),
        )
        conflict2 = Contradiction(
            id="conflict-002",
            contradiction_type=ContradictionType.IMPLICIT_RESOURCE,
            requirement_ids=("req-003", "req-004"),
            description="Memory conflict",
            explanation="High vs low memory",
            severity=Severity.MEDIUM,
            resolution_suggestions=("Define limits",),
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            structured_contradictions=(conflict1, conflict2),
        )

        d = output.to_dict()

        assert len(d["structured_contradictions"]) == 2
        assert d["structured_contradictions"][0]["id"] == "conflict-001"
        assert d["structured_contradictions"][1]["id"] == "conflict-002"


# =============================================================================
# Story 5.7: Escalation to PM Types Tests
# =============================================================================


class TestEscalationReason:
    """Tests for EscalationReason enum (Story 5.7)."""

    def test_escalation_reason_values(self) -> None:
        """EscalationReason should have all expected values."""
        assert EscalationReason.UNRESOLVABLE_AMBIGUITY.value == "unresolvable_ambiguity"
        assert EscalationReason.CONFLICTING_REQUIREMENTS.value == "conflicting_requirements"
        assert EscalationReason.MISSING_DOMAIN_KNOWLEDGE.value == "missing_domain_knowledge"
        assert EscalationReason.STAKEHOLDER_DECISION_NEEDED.value == "stakeholder_decision_needed"
        assert EscalationReason.SCOPE_CLARIFICATION.value == "scope_clarification"

    def test_escalation_reason_is_str_enum(self) -> None:
        """EscalationReason should be a str enum."""
        assert isinstance(EscalationReason.UNRESOLVABLE_AMBIGUITY, str)
        assert isinstance(EscalationReason.CONFLICTING_REQUIREMENTS, str)

    def test_escalation_reason_from_string(self) -> None:
        """EscalationReason should be constructible from string value."""
        assert EscalationReason("unresolvable_ambiguity") == EscalationReason.UNRESOLVABLE_AMBIGUITY
        assert EscalationReason("conflicting_requirements") == EscalationReason.CONFLICTING_REQUIREMENTS
        assert EscalationReason("missing_domain_knowledge") == EscalationReason.MISSING_DOMAIN_KNOWLEDGE

    def test_escalation_reason_invalid_value(self) -> None:
        """EscalationReason should reject invalid values."""
        with pytest.raises(ValueError):
            EscalationReason("invalid_reason")

    def test_escalation_reason_all_members(self) -> None:
        """EscalationReason should have exactly 5 members."""
        members = list(EscalationReason)
        assert len(members) == 5


class TestEscalationPriority:
    """Tests for EscalationPriority enum (Story 5.7)."""

    def test_escalation_priority_values(self) -> None:
        """EscalationPriority should have all expected values."""
        assert EscalationPriority.URGENT.value == "urgent"
        assert EscalationPriority.HIGH.value == "high"
        assert EscalationPriority.NORMAL.value == "normal"

    def test_escalation_priority_is_str_enum(self) -> None:
        """EscalationPriority should be a str enum."""
        assert isinstance(EscalationPriority.URGENT, str)
        assert isinstance(EscalationPriority.HIGH, str)
        assert isinstance(EscalationPriority.NORMAL, str)

    def test_escalation_priority_from_string(self) -> None:
        """EscalationPriority should be constructible from string value."""
        assert EscalationPriority("urgent") == EscalationPriority.URGENT
        assert EscalationPriority("high") == EscalationPriority.HIGH
        assert EscalationPriority("normal") == EscalationPriority.NORMAL

    def test_escalation_priority_invalid_value(self) -> None:
        """EscalationPriority should reject invalid values."""
        with pytest.raises(ValueError):
            EscalationPriority("low")  # "low" is not a valid priority

    def test_escalation_priority_all_members(self) -> None:
        """EscalationPriority should have exactly 3 members."""
        members = list(EscalationPriority)
        assert len(members) == 3


class TestEscalation:
    """Tests for Escalation dataclass (Story 5.7)."""

    def test_creation_with_all_fields(self) -> None:
        """Escalation should be creatable with all fields."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.CONFLICTING_REQUIREMENTS,
            priority=EscalationPriority.HIGH,
            summary="Real-time vs batch processing conflict",
            context="req-001 requires real-time, req-002 requires batch",
            original_requirements=("req-001", "req-002"),
            analysis_attempts=("Tried scope separation", "Checked priority hints"),
            decision_requested="Should we prioritize real-time or batch?",
            related_gaps=("gap-001",),
            related_contradictions=("conflict-001",),
        )

        assert esc.id == "esc-001"
        assert esc.reason == EscalationReason.CONFLICTING_REQUIREMENTS
        assert esc.priority == EscalationPriority.HIGH
        assert esc.summary == "Real-time vs batch processing conflict"
        assert esc.context == "req-001 requires real-time, req-002 requires batch"
        assert esc.original_requirements == ("req-001", "req-002")
        assert esc.analysis_attempts == ("Tried scope separation", "Checked priority hints")
        assert esc.decision_requested == "Should we prioritize real-time or batch?"
        assert esc.related_gaps == ("gap-001",)
        assert esc.related_contradictions == ("conflict-001",)
        assert esc.timestamp is not None  # Auto-generated

    def test_creation_with_minimal_fields(self) -> None:
        """Escalation should be creatable with only required fields."""
        esc = Escalation(
            id="esc-002",
            reason=EscalationReason.SCOPE_CLARIFICATION,
            priority=EscalationPriority.NORMAL,
            summary="Is feature X in scope?",
            context="Feature mentioned but not detailed",
            original_requirements=("req-005",),
            analysis_attempts=("Searched for scope hints",),
            decision_requested="Is feature X in scope?",
        )

        assert esc.id == "esc-002"
        assert esc.related_gaps == ()  # Default empty
        assert esc.related_contradictions == ()  # Default empty

    def test_immutability(self) -> None:
        """Escalation should be immutable (frozen)."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.UNRESOLVABLE_AMBIGUITY,
            priority=EscalationPriority.NORMAL,
            summary="Ambiguous requirement",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=("Tried analysis",),
            decision_requested="Please clarify",
        )

        with pytest.raises(AttributeError):
            esc.id = "esc-002"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """Escalation.to_dict should serialize all fields correctly."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.MISSING_DOMAIN_KNOWLEDGE,
            priority=EscalationPriority.HIGH,
            summary="Missing business rules",
            context="Domain context needed",
            original_requirements=("req-001", "req-002"),
            analysis_attempts=("Attempted inference",),
            decision_requested="What are the business rules?",
            related_gaps=("gap-001",),
            related_contradictions=(),
        )

        d = esc.to_dict()

        assert d["id"] == "esc-001"
        assert d["reason"] == "missing_domain_knowledge"  # String value
        assert d["priority"] == "high"  # String value
        assert d["summary"] == "Missing business rules"
        assert d["context"] == "Domain context needed"
        assert d["original_requirements"] == ["req-001", "req-002"]  # List
        assert d["analysis_attempts"] == ["Attempted inference"]  # List
        assert d["decision_requested"] == "What are the business rules?"
        assert d["related_gaps"] == ["gap-001"]  # List
        assert d["related_contradictions"] == []  # List
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)  # ISO format string

    def test_to_dict_converts_enums_to_strings(self) -> None:
        """to_dict should convert enum values to their string representations."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.STAKEHOLDER_DECISION_NEEDED,
            priority=EscalationPriority.URGENT,
            summary="Test",
            context="Test context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Test?",
        )

        d = esc.to_dict()

        assert d["reason"] == "stakeholder_decision_needed"
        assert d["priority"] == "urgent"
        assert not isinstance(d["reason"], EscalationReason)
        assert not isinstance(d["priority"], EscalationPriority)

    def test_original_requirements_is_list_in_dict(self) -> None:
        """original_requirements tuple should become list in to_dict."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.CONFLICTING_REQUIREMENTS,
            priority=EscalationPriority.HIGH,
            summary="Test",
            context="Test context",
            original_requirements=("req-001", "req-002", "req-003"),
            analysis_attempts=("Attempt 1", "Attempt 2"),
            decision_requested="Test?",
        )

        d = esc.to_dict()

        assert isinstance(d["original_requirements"], list)
        assert d["original_requirements"] == ["req-001", "req-002", "req-003"]
        assert isinstance(d["analysis_attempts"], list)
        assert d["analysis_attempts"] == ["Attempt 1", "Attempt 2"]

    def test_equality(self) -> None:
        """Two Escalation objects with same values should be equal."""
        from datetime import datetime, timezone

        fixed_time = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)

        esc1 = Escalation(
            id="esc-001",
            reason=EscalationReason.SCOPE_CLARIFICATION,
            priority=EscalationPriority.NORMAL,
            summary="Test",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Test?",
            timestamp=fixed_time,
        )
        esc2 = Escalation(
            id="esc-001",
            reason=EscalationReason.SCOPE_CLARIFICATION,
            priority=EscalationPriority.NORMAL,
            summary="Test",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Test?",
            timestamp=fixed_time,
        )

        assert esc1 == esc2

    def test_hashability(self) -> None:
        """Escalation should be hashable (for use in sets/dicts)."""
        from datetime import datetime, timezone

        fixed_time = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)

        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.CONFLICTING_REQUIREMENTS,
            priority=EscalationPriority.HIGH,
            summary="Test",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Test?",
            timestamp=fixed_time,
        )

        # Should not raise
        hash_value = hash(esc)
        assert isinstance(hash_value, int)

        # Should be usable in set
        escalation_set = {esc}
        assert len(escalation_set) == 1


class TestAnalystOutputWithEscalations:
    """Tests for AnalystOutput with escalations field (Story 5.7)."""

    def test_escalations_default_empty(self) -> None:
        """AnalystOutput.escalations should default to empty tuple."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        assert output.escalations == ()

    def test_escalation_needed_false_when_empty(self) -> None:
        """escalation_needed should be False when no escalations."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        assert output.escalation_needed is False

    def test_escalation_needed_true_when_has_escalations(self) -> None:
        """escalation_needed should be True when escalations exist."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.CONFLICTING_REQUIREMENTS,
            priority=EscalationPriority.HIGH,
            summary="Test",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Test?",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            escalations=(esc,),
        )

        assert output.escalation_needed is True

    def test_creation_with_escalations(self) -> None:
        """AnalystOutput should accept escalations tuple."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.UNRESOLVABLE_AMBIGUITY,
            priority=EscalationPriority.NORMAL,
            summary="Ambiguous requirement",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=("Tried analysis",),
            decision_requested="Please clarify",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            escalations=(esc,),
        )

        assert len(output.escalations) == 1
        assert output.escalations[0].id == "esc-001"

    def test_to_dict_includes_escalations(self) -> None:
        """to_dict should include escalations and escalation_needed."""
        esc = Escalation(
            id="esc-001",
            reason=EscalationReason.MISSING_DOMAIN_KNOWLEDGE,
            priority=EscalationPriority.HIGH,
            summary="Missing business rules",
            context="Context",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="What are the rules?",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            escalations=(esc,),
        )

        d = output.to_dict()

        assert "escalations" in d
        assert "escalation_needed" in d
        assert len(d["escalations"]) == 1
        assert d["escalations"][0]["id"] == "esc-001"
        assert d["escalation_needed"] is True

    def test_to_dict_escalations_empty_when_none(self) -> None:
        """to_dict should show escalation_needed=False when no escalations."""
        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
        )

        d = output.to_dict()

        assert d["escalations"] == []
        assert d["escalation_needed"] is False

    def test_backward_compatibility_without_escalations(self) -> None:
        """AnalystOutput should work without escalations parameter."""
        # This should work exactly as before Story 5.7
        output = AnalystOutput(
            requirements=(),
            identified_gaps=("Gap 1",),
            contradictions=("Contradiction 1",),
            structured_gaps=(),
            structured_contradictions=(),
        )

        assert output.escalations == ()
        assert output.escalation_needed is False
        d = output.to_dict()
        assert d["escalations"] == []
        assert d["escalation_needed"] is False

    def test_to_dict_with_multiple_escalations(self) -> None:
        """to_dict should serialize multiple escalations correctly."""
        esc1 = Escalation(
            id="esc-001",
            reason=EscalationReason.CONFLICTING_REQUIREMENTS,
            priority=EscalationPriority.URGENT,
            summary="Critical conflict",
            context="Context 1",
            original_requirements=("req-001",),
            analysis_attempts=(),
            decision_requested="Question 1?",
        )
        esc2 = Escalation(
            id="esc-002",
            reason=EscalationReason.SCOPE_CLARIFICATION,
            priority=EscalationPriority.NORMAL,
            summary="Scope unclear",
            context="Context 2",
            original_requirements=("req-002",),
            analysis_attempts=(),
            decision_requested="Question 2?",
        )

        output = AnalystOutput(
            requirements=(),
            identified_gaps=(),
            contradictions=(),
            escalations=(esc1, esc2),
        )

        d = output.to_dict()

        assert len(d["escalations"]) == 2
        assert d["escalations"][0]["id"] == "esc-001"
        assert d["escalations"][0]["priority"] == "urgent"
        assert d["escalations"][1]["id"] == "esc-002"
        assert d["escalations"][1]["priority"] == "normal"
        assert d["escalation_needed"] is True
