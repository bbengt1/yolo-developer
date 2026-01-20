"""Unit tests for PM agent testability validation (Story 6.3).

Tests for the testability validation functions that check acceptance criteria
for testability issues including vague terms, structure validation,
edge case coverage, and AC count.

Test Categories:
- TestVagueTermDetection: Tests for _detect_vague_terms (AC: 2)
- TestACStructureValidation: Tests for _validate_ac_structure (AC: 1)
- TestEdgeCaseDetection: Tests for _check_edge_cases (AC: 3)
- TestACCountValidation: Tests for _validate_ac_count (AC: 4)
- TestMainValidation: Tests for validate_story_testability (AC: 5)
"""

from __future__ import annotations

from yolo_developer.agents.pm.testability import (
    BOUNDARY_PATTERNS,
    EMPTY_PATTERNS,
    ERROR_PATTERNS,
    VAGUE_TERMS,
    _check_edge_cases,
    _detect_vague_terms,
    _validate_ac_count,
    _validate_ac_structure,
    validate_story_testability,
)
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    Story,
    StoryPriority,
    StoryStatus,
    TestabilityResult,
)


# Helper function to create test stories
def _create_story(
    acceptance_criteria: tuple[AcceptanceCriterion, ...],
    story_id: str = "story-001",
) -> Story:
    """Create a test story with the given acceptance criteria."""
    return Story(
        id=story_id,
        title="Test Story",
        role="user",
        action="test action",
        benefit="test benefit",
        acceptance_criteria=acceptance_criteria,
        priority=StoryPriority.MEDIUM,
        status=StoryStatus.DRAFT,
        source_requirements=("req-001",),
    )


def _create_ac(
    ac_id: str = "AC1",
    given: str = "valid precondition",
    when: str = "valid action",
    then: str = "valid outcome",
    and_clauses: tuple[str, ...] = (),
) -> AcceptanceCriterion:
    """Create a test acceptance criterion."""
    return AcceptanceCriterion(
        id=ac_id,
        given=given,
        when=when,
        then=then,
        and_clauses=and_clauses,
    )


class TestVagueTermDetection:
    """Tests for _detect_vague_terms function (AC: 2)."""

    def test_clean_ac_returns_empty_list(self) -> None:
        """Test that AC without vague terms returns empty list."""
        ac = _create_ac(
            given="the user is authenticated",
            when="they click the submit button",
            then="the form is saved to the database",
        )
        result = _detect_vague_terms(ac)
        assert result == []

    def test_detects_quantifier_vague_terms(self) -> None:
        """Test detection of quantifier vagueness (fast, slow, efficient)."""
        ac = _create_ac(
            given="the system is running",
            when="user requests data",
            then="the response is fast and efficient",
        )
        result = _detect_vague_terms(ac)
        assert "fast" in result
        assert "efficient" in result

    def test_detects_ease_vague_terms(self) -> None:
        """Test detection of ease vagueness (easy, simple, intuitive)."""
        ac = _create_ac(
            given="the user interface is loaded",
            when="user navigates the system",
            then="the experience is simple and intuitive",
        )
        result = _detect_vague_terms(ac)
        assert "simple" in result
        assert "intuitive" in result

    def test_detects_certainty_vague_terms(self) -> None:
        """Test detection of certainty vagueness (should, might, could)."""
        ac = _create_ac(
            given="the feature is enabled",
            when="user triggers the feature",
            then="the system should respond appropriately",
        )
        result = _detect_vague_terms(ac)
        assert "should" in result

    def test_detects_quality_vague_terms(self) -> None:
        """Test detection of quality vagueness (good, better, best, robust)."""
        ac = _create_ac(
            given="the system is configured",
            when="errors occur",
            then="the error handling is robust and clean",
        )
        result = _detect_vague_terms(ac)
        assert "robust" in result
        assert "clean" in result

    def test_detects_multiple_vague_terms_in_single_ac(self) -> None:
        """Test detection of multiple vague terms in one AC."""
        ac = _create_ac(
            given="system is fast",
            when="user performs easy action",
            then="result is good and beautiful",
        )
        result = _detect_vague_terms(ac)
        assert len(result) >= 3
        assert "fast" in result
        assert "easy" in result
        assert "good" in result

    def test_detects_vague_terms_in_and_clauses(self) -> None:
        """Test detection of vague terms in and_clauses field."""
        ac = _create_ac(
            given="system is ready",
            when="user acts",
            then="result is returned",
            and_clauses=("response is fast", "interface is intuitive"),
        )
        result = _detect_vague_terms(ac)
        assert "fast" in result
        assert "intuitive" in result

    def test_detection_is_case_insensitive(self) -> None:
        """Test that vague term detection is case insensitive."""
        ac = _create_ac(
            given="system is FAST",
            when="user does EASY task",
            then="result is Good",
        )
        result = _detect_vague_terms(ac)
        assert "fast" in result or "FAST" in [r.upper() for r in result]

    def test_all_vague_terms_in_frozenset(self) -> None:
        """Test that VAGUE_TERMS contains all expected categories."""
        # Quantifier
        assert "fast" in VAGUE_TERMS
        assert "efficient" in VAGUE_TERMS
        assert "scalable" in VAGUE_TERMS
        # Ease
        assert "easy" in VAGUE_TERMS
        assert "simple" in VAGUE_TERMS
        assert "intuitive" in VAGUE_TERMS
        # Certainty
        assert "should" in VAGUE_TERMS
        assert "might" in VAGUE_TERMS
        # Quality
        assert "good" in VAGUE_TERMS
        assert "robust" in VAGUE_TERMS


class TestACStructureValidation:
    """Tests for _validate_ac_structure function (AC: 1)."""

    def test_valid_ac_structure_returns_empty_list(self) -> None:
        """Test that valid AC structure returns empty list."""
        ac = _create_ac(
            given="valid precondition",
            when="valid action occurs",
            then="valid outcome happens",
        )
        result = _validate_ac_structure(ac)
        assert result == []

    def test_empty_given_field_fails(self) -> None:
        """Test that empty given field returns error."""
        ac = _create_ac(given="", when="action", then="outcome")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("given" in issue.lower() for issue in result)

    def test_empty_when_field_fails(self) -> None:
        """Test that empty when field returns error."""
        ac = _create_ac(given="precondition", when="", then="outcome")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("when" in issue.lower() for issue in result)

    def test_empty_then_field_fails(self) -> None:
        """Test that empty then field returns error."""
        ac = _create_ac(given="precondition", when="action", then="")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("then" in issue.lower() for issue in result)

    def test_whitespace_only_given_fails(self) -> None:
        """Test that whitespace-only given field fails."""
        ac = _create_ac(given="   ", when="action", then="outcome")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("given" in issue.lower() for issue in result)

    def test_whitespace_only_when_fails(self) -> None:
        """Test that whitespace-only when field fails."""
        ac = _create_ac(given="precondition", when="\t\n", then="outcome")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("when" in issue.lower() for issue in result)

    def test_whitespace_only_then_fails(self) -> None:
        """Test that whitespace-only then field fails."""
        ac = _create_ac(given="precondition", when="action", then="  \n  ")
        result = _validate_ac_structure(ac)
        assert len(result) > 0
        assert any("then" in issue.lower() for issue in result)

    def test_multiple_empty_fields_returns_multiple_issues(self) -> None:
        """Test that multiple empty fields returns multiple issues."""
        ac = _create_ac(given="", when="", then="")
        result = _validate_ac_structure(ac)
        assert len(result) == 3


class TestEdgeCaseDetection:
    """Tests for _check_edge_cases function (AC: 3)."""

    def test_story_with_error_handling_passes(self) -> None:
        """Test that story with error handling AC is detected."""
        ac = _create_ac(
            given="invalid input is provided",
            when="the system processes the input",
            then="an error message is displayed",
        )
        story = _create_story(acceptance_criteria=(ac,))
        result = _check_edge_cases(story)
        # Should NOT have error_handling in missing list
        assert "error_handling" not in result

    def test_story_with_empty_input_handling_passes(self) -> None:
        """Test that story with empty input AC is detected."""
        ac = _create_ac(
            given="the input field is empty",
            when="user submits the form",
            then="a validation message is shown",
        )
        story = _create_story(acceptance_criteria=(ac,))
        result = _check_edge_cases(story)
        assert "empty_input" not in result

    def test_story_with_boundary_handling_passes(self) -> None:
        """Test that story with boundary condition AC is detected."""
        ac = _create_ac(
            given="the value is at maximum limit",
            when="user tries to increase it",
            then="the system prevents overflow",
        )
        story = _create_story(acceptance_criteria=(ac,))
        result = _check_edge_cases(story)
        assert "boundary" not in result

    def test_story_missing_all_edge_cases_returns_suggestions(self) -> None:
        """Test that story missing all edge cases returns all suggestions."""
        ac = _create_ac(
            given="user is logged in",
            when="they click submit",
            then="data is saved",
        )
        story = _create_story(acceptance_criteria=(ac,))
        result = _check_edge_cases(story)
        assert "error_handling" in result
        assert "empty_input" in result
        assert "boundary" in result

    def test_partial_edge_case_coverage(self) -> None:
        """Test that partial coverage returns only missing categories."""
        ac1 = _create_ac(
            given="an error occurs",
            when="system handles it",
            then="user sees message",
        )
        ac2 = _create_ac(
            given="user is logged in",
            when="they act",
            then="result happens",
        )
        story = _create_story(acceptance_criteria=(ac1, ac2))
        result = _check_edge_cases(story)
        assert "error_handling" not in result
        assert "empty_input" in result
        assert "boundary" in result

    def test_edge_case_detection_is_case_insensitive(self) -> None:
        """Test that edge case detection is case insensitive."""
        ac = _create_ac(
            given="an ERROR occurs",
            when="EMPTY input is given",
            then="MAXIMUM value is reached",
        )
        story = _create_story(acceptance_criteria=(ac,))
        result = _check_edge_cases(story)
        # All edge cases should be covered
        assert "error_handling" not in result
        assert "empty_input" not in result
        assert "boundary" not in result

    def test_edge_case_patterns_defined(self) -> None:
        """Test that edge case pattern constants are defined."""
        assert "error" in ERROR_PATTERNS
        assert "fail" in ERROR_PATTERNS
        assert "invalid" in ERROR_PATTERNS
        assert "empty" in EMPTY_PATTERNS
        assert "null" in EMPTY_PATTERNS
        assert "maximum" in BOUNDARY_PATTERNS
        assert "minimum" in BOUNDARY_PATTERNS


class TestACCountValidation:
    """Tests for _validate_ac_count function (AC: 4)."""

    def test_zero_acs_returns_warning(self) -> None:
        """Test that 0 ACs returns warning."""
        story = _create_story(acceptance_criteria=())
        result = _validate_ac_count(story)
        assert result is not None
        assert "0" in result or "only" in result.lower()

    def test_one_ac_returns_warning(self) -> None:
        """Test that 1 AC returns warning."""
        ac = _create_ac()
        story = _create_story(acceptance_criteria=(ac,))
        result = _validate_ac_count(story)
        assert result is not None
        assert "1" in result or "only" in result.lower()

    def test_two_acs_returns_none(self) -> None:
        """Test that 2 ACs returns None (acceptable)."""
        acs = tuple(_create_ac(ac_id=f"AC{i}") for i in range(1, 3))
        story = _create_story(acceptance_criteria=acs)
        result = _validate_ac_count(story)
        assert result is None

    def test_five_acs_returns_none(self) -> None:
        """Test that 5 ACs returns None (acceptable)."""
        acs = tuple(_create_ac(ac_id=f"AC{i}") for i in range(1, 6))
        story = _create_story(acceptance_criteria=acs)
        result = _validate_ac_count(story)
        assert result is None

    def test_eight_acs_returns_none(self) -> None:
        """Test that 8 ACs returns None (acceptable upper bound)."""
        acs = tuple(_create_ac(ac_id=f"AC{i}") for i in range(1, 9))
        story = _create_story(acceptance_criteria=acs)
        result = _validate_ac_count(story)
        assert result is None

    def test_nine_acs_returns_splitting_warning(self) -> None:
        """Test that 9+ ACs returns splitting warning."""
        acs = tuple(_create_ac(ac_id=f"AC{i}") for i in range(1, 10))
        story = _create_story(acceptance_criteria=acs)
        result = _validate_ac_count(story)
        assert result is not None
        assert "9" in result or "split" in result.lower()

    def test_fifteen_acs_returns_splitting_warning(self) -> None:
        """Test that 15 ACs returns splitting warning."""
        acs = tuple(_create_ac(ac_id=f"AC{i}") for i in range(1, 16))
        story = _create_story(acceptance_criteria=acs)
        result = _validate_ac_count(story)
        assert result is not None
        assert "split" in result.lower() or "15" in result


class TestMainValidation:
    """Tests for validate_story_testability function (AC: 5)."""

    def test_fully_valid_story_returns_is_valid_true(self) -> None:
        """Test that fully valid story returns is_valid=True."""
        ac1 = _create_ac(
            ac_id="AC1",
            given="an error occurs",
            when="system processes it",
            then="error message is displayed",
        )
        ac2 = _create_ac(
            ac_id="AC2",
            given="input is empty",
            when="user submits",
            then="validation message shown",
        )
        ac3 = _create_ac(
            ac_id="AC3",
            given="value at maximum limit",
            when="increase attempted",
            then="overflow prevented",
        )
        story = _create_story(acceptance_criteria=(ac1, ac2, ac3))
        result = validate_story_testability(story)

        assert result["is_valid"] is True
        assert result["vague_terms_found"] == []

    def test_story_with_vague_terms_returns_is_valid_false(self) -> None:
        """Test that story with vague terms returns is_valid=False."""
        ac = _create_ac(
            given="the system is fast",
            when="user does easy task",
            then="result is good",
        )
        story = _create_story(acceptance_criteria=(ac, ac))
        result = validate_story_testability(story)

        assert result["is_valid"] is False
        assert len(result["vague_terms_found"]) > 0

    def test_story_with_structural_issues_returns_is_valid_false(self) -> None:
        """Test that story with structural issues returns is_valid=False."""
        ac1 = _create_ac(given="", when="action", then="outcome")
        ac2 = _create_ac(given="valid", when="valid", then="valid")
        story = _create_story(acceptance_criteria=(ac1, ac2))
        result = validate_story_testability(story)

        assert result["is_valid"] is False

    def test_all_result_fields_populated(self) -> None:
        """Test that all TestabilityResult fields are populated."""
        ac = _create_ac()
        story = _create_story(acceptance_criteria=(ac, ac))
        result = validate_story_testability(story)

        # Check all expected fields exist
        assert "is_valid" in result
        assert "vague_terms_found" in result
        assert "missing_edge_cases" in result
        assert "ac_count_warning" in result
        assert "validation_notes" in result

        # Check types
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["vague_terms_found"], list)
        assert isinstance(result["missing_edge_cases"], list)
        assert result["ac_count_warning"] is None or isinstance(result["ac_count_warning"], str)
        assert isinstance(result["validation_notes"], list)

    def test_vague_terms_found_includes_ac_id(self) -> None:
        """Test that vague_terms_found includes (ac_id, term) tuples."""
        ac1 = _create_ac(ac_id="AC1", given="system is fast", when="action", then="result")
        ac2 = _create_ac(ac_id="AC2", given="valid", when="valid", then="valid")
        story = _create_story(acceptance_criteria=(ac1, ac2))
        result = validate_story_testability(story)

        # Should have at least one vague term from AC1
        assert len(result["vague_terms_found"]) > 0
        # Check it's a tuple of (ac_id, term)
        first_finding = result["vague_terms_found"][0]
        assert len(first_finding) == 2
        assert first_finding[0] == "AC1"
        assert first_finding[1] == "fast"

    def test_validation_notes_contains_details(self) -> None:
        """Test that validation_notes contains useful details."""
        ac = _create_ac()
        story = _create_story(acceptance_criteria=(ac, ac))
        result = validate_story_testability(story)

        # Should have at least one note
        assert len(result["validation_notes"]) > 0

    def test_missing_edge_cases_contains_categories(self) -> None:
        """Test that missing_edge_cases contains category names."""
        ac = _create_ac(
            given="normal condition",
            when="normal action",
            then="normal result",
        )
        story = _create_story(acceptance_criteria=(ac, ac))
        result = validate_story_testability(story)

        # Should have missing edge case categories
        missing = result["missing_edge_cases"]
        assert "error_handling" in missing or "empty_input" in missing or "boundary" in missing

    def test_result_is_typed_dict(self) -> None:
        """Test that result conforms to TestabilityResult type."""
        ac = _create_ac()
        story = _create_story(acceptance_criteria=(ac, ac))
        result = validate_story_testability(story)

        # Type check by accessing known keys
        _: TestabilityResult = result  # Type checker validates this
        assert result["is_valid"] is not None
