"""Unit tests for AC measurability gate implementation.

Tests for data structures, subjective term detection, GWT structure validation,
concrete condition detection, evaluator functionality, and failure report generation.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def ensure_ac_measurability_evaluator_registered() -> None:
    """Ensure ac_measurability evaluator is registered for each test.

    Other tests may call clear_evaluators(), so we need to
    re-register the ac_measurability evaluator before tests that need it.
    """
    from yolo_developer.gates.evaluators import get_evaluator
    from yolo_developer.gates.gates.ac_measurability import (
        ac_measurability_evaluator,
        register_evaluator,
    )

    if get_evaluator("ac_measurability") is None:
        register_evaluator("ac_measurability", ac_measurability_evaluator)


class TestACMeasurabilityTypes:
    """Tests for AC measurability gate data structures (Task 1)."""

    def test_ac_measurability_issue_creation(self) -> None:
        """ACMeasurabilityIssue can be created with required fields."""
        from yolo_developer.gates.gates.ac_measurability import ACMeasurabilityIssue

        issue = ACMeasurabilityIssue(
            story_id="story-001",
            ac_index=0,
            issue_type="missing_gwt",
            description="Missing 'Given' clause",
            severity="blocking",
        )

        assert issue.story_id == "story-001"
        assert issue.ac_index == 0
        assert issue.issue_type == "missing_gwt"
        assert issue.description == "Missing 'Given' clause"
        assert issue.severity == "blocking"

    def test_ac_measurability_issue_is_frozen(self) -> None:
        """ACMeasurabilityIssue is immutable."""
        from yolo_developer.gates.gates.ac_measurability import ACMeasurabilityIssue

        issue = ACMeasurabilityIssue(
            story_id="story-001",
            ac_index=0,
            issue_type="missing_gwt",
            description="Test",
            severity="blocking",
        )

        with pytest.raises(AttributeError):
            issue.story_id = "new-id"  # type: ignore[misc]

    def test_ac_measurability_issue_to_dict(self) -> None:
        """ACMeasurabilityIssue can be converted to dict."""
        from yolo_developer.gates.gates.ac_measurability import ACMeasurabilityIssue

        issue = ACMeasurabilityIssue(
            story_id="story-001",
            ac_index=2,
            issue_type="subjective_term",
            description="Contains subjective term",
            severity="warning",
        )

        d = issue.to_dict()
        assert d["story_id"] == "story-001"
        assert d["ac_index"] == 2
        assert d["issue_type"] == "subjective_term"
        assert d["description"] == "Contains subjective term"
        assert d["severity"] == "warning"

    def test_subjective_terms_constant_exists(self) -> None:
        """SUBJECTIVE_TERMS constant is defined with expected terms."""
        from yolo_developer.gates.gates.ac_measurability import SUBJECTIVE_TERMS

        assert isinstance(SUBJECTIVE_TERMS, (list, tuple, frozenset))
        # Check some expected subjective terms
        subjective_terms_lower = [t.lower() for t in SUBJECTIVE_TERMS]
        assert "intuitive" in subjective_terms_lower
        assert "clean" in subjective_terms_lower
        assert (
            "user-friendly" in subjective_terms_lower or "user friendly" in subjective_terms_lower
        )
        assert "appropriate" in subjective_terms_lower

    def test_subjective_terms_has_minimum_coverage(self) -> None:
        """SUBJECTIVE_TERMS has comprehensive coverage."""
        from yolo_developer.gates.gates.ac_measurability import SUBJECTIVE_TERMS

        # Should have at least 25 subjective terms per dev notes
        assert len(SUBJECTIVE_TERMS) >= 25

    def test_gwt_patterns_constant_exists(self) -> None:
        """GWT_PATTERNS constant is defined with required patterns."""
        from yolo_developer.gates.gates.ac_measurability import GWT_PATTERNS

        assert isinstance(GWT_PATTERNS, dict)
        assert "given" in GWT_PATTERNS
        assert "when" in GWT_PATTERNS
        assert "then" in GWT_PATTERNS

    def test_concrete_condition_patterns_constant_exists(self) -> None:
        """CONCRETE_CONDITION_PATTERNS constant is defined with regex patterns."""
        from yolo_developer.gates.gates.ac_measurability import CONCRETE_CONDITION_PATTERNS

        assert isinstance(CONCRETE_CONDITION_PATTERNS, (list, tuple))
        # Should have multiple patterns for different condition types
        assert len(CONCRETE_CONDITION_PATTERNS) >= 5
        # Each pattern should be a compiled regex
        import re

        for pattern in CONCRETE_CONDITION_PATTERNS:
            assert isinstance(pattern, re.Pattern)

    def test_concrete_condition_patterns_covers_key_scenarios(self) -> None:
        """CONCRETE_CONDITION_PATTERNS detects key measurable outcome types."""
        from yolo_developer.gates.gates.ac_measurability import CONCRETE_CONDITION_PATTERNS

        # Test that patterns can match expected concrete conditions
        test_cases = [
            "user sees the dashboard",  # UI outcome
            "record is created",  # State change
            "redirected to login page",  # Navigation
            "error message is displayed",  # Error handling
            "'Invalid credentials'",  # Quoted text
            "operation succeeds",  # Boolean outcome
            "at least 10 results",  # Numeric condition
        ]

        for test_text in test_cases:
            matched = any(pattern.search(test_text) for pattern in CONCRETE_CONDITION_PATTERNS)
            assert matched, f"Expected to match concrete condition in: '{test_text}'"


class TestSubjectiveTermDetection:
    """Tests for subjective term detection (Task 2)."""

    def test_detect_single_subjective_term(self) -> None:
        """Detects single subjective term in text."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        result = detect_subjective_terms("The interface should be intuitive")
        assert len(result) == 1
        assert result[0][0].lower() == "intuitive"

    def test_detect_multiple_subjective_terms(self) -> None:
        """Detects multiple subjective terms in text."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        result = detect_subjective_terms("The layout should be clean and user-friendly")
        terms = [t[0].lower() for t in result]
        assert "clean" in terms
        assert "user-friendly" in terms or "user friendly" in terms

    def test_detect_subjective_term_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        result = detect_subjective_terms("The system should be INTUITIVE")
        assert len(result) == 1
        assert result[0][0].lower() == "intuitive"

    def test_detect_no_subjective_terms(self) -> None:
        """Returns empty list when no subjective terms found."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        result = detect_subjective_terms(
            "Given a user, When they click login, Then they see dashboard"
        )
        assert len(result) == 0

    def test_detect_multi_word_subjective_phrase(self) -> None:
        """Detects multi-word subjective phrases like 'user friendly'."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        result = detect_subjective_terms("The interface should be user friendly")
        terms = [t[0].lower() for t in result]
        # Should detect "user friendly"
        assert any("user" in t and "friendly" in t for t in terms)

    def test_detect_returns_position(self) -> None:
        """Detection returns position of subjective term."""
        from yolo_developer.gates.gates.ac_measurability import detect_subjective_terms

        text = "The system should be intuitive"
        result = detect_subjective_terms(text)
        assert len(result) == 1
        term, position = result[0]
        # Position should be where the term starts
        assert text[position : position + len(term)].lower() == term.lower()


class TestGWTStructureValidation:
    """Tests for Given/When/Then structure validation (Task 3)."""

    def test_has_gwt_structure_complete(self) -> None:
        """Complete GWT structure passes validation."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = (
            "Given a logged-in user, When they click logout, Then they are redirected to login page"
        )
        passed, missing = has_gwt_structure(ac_text)
        assert passed is True
        assert missing == []

    def test_has_gwt_structure_missing_given(self) -> None:
        """Missing 'Given' is detected."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = "When the user clicks submit, Then a success message is displayed"
        passed, missing = has_gwt_structure(ac_text)
        assert passed is False
        assert "Given" in missing

    def test_has_gwt_structure_missing_when(self) -> None:
        """Missing 'When' is detected."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = "Given a user is logged in, Then they see the dashboard"
        passed, missing = has_gwt_structure(ac_text)
        assert passed is False
        assert "When" in missing

    def test_has_gwt_structure_missing_then(self) -> None:
        """Missing 'Then' is detected."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = "Given a user, When they click the button"
        passed, missing = has_gwt_structure(ac_text)
        assert passed is False
        assert "Then" in missing

    def test_has_gwt_structure_missing_all(self) -> None:
        """All missing parts detected when no GWT structure."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = "The user interface should be intuitive"
        passed, missing = has_gwt_structure(ac_text)
        assert passed is False
        assert "Given" in missing
        assert "When" in missing
        assert "Then" in missing

    def test_has_gwt_structure_case_insensitive(self) -> None:
        """GWT detection is case-insensitive."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = "GIVEN a user, WHEN they login, THEN they see dashboard"
        passed, missing = has_gwt_structure(ac_text)
        assert passed is True
        assert missing == []

    def test_has_gwt_structure_multi_line(self) -> None:
        """GWT detection works with multi-line text."""
        from yolo_developer.gates.gates.ac_measurability import has_gwt_structure

        ac_text = """Given a user with admin privileges
        When they access the admin panel
        Then they see all user management options"""
        passed, missing = has_gwt_structure(ac_text)
        assert passed is True
        assert missing == []


class TestConcreteConditionDetection:
    """Tests for concrete condition detection (Task 4)."""

    def test_has_concrete_condition_with_redirect(self) -> None:
        """Detects concrete redirect condition."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a user, When they login, Then they are redirected to the dashboard"
        assert has_concrete_condition(ac_text) is True

    def test_has_concrete_condition_with_displays(self) -> None:
        """Detects concrete display condition."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given invalid input, When submitted, Then an error message is displayed"
        assert has_concrete_condition(ac_text) is True

    def test_has_concrete_condition_with_specific_text(self) -> None:
        """Detects concrete condition with quoted text."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a login failure, When error occurs, Then 'Invalid credentials' appears"
        assert has_concrete_condition(ac_text) is True

    def test_has_concrete_condition_with_state_change(self) -> None:
        """Detects concrete state change condition."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a form, When submitted, Then the record is created in the database"
        assert has_concrete_condition(ac_text) is True

    def test_has_concrete_condition_with_boolean_outcome(self) -> None:
        """Detects concrete boolean outcome."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given valid data, When processed, Then the operation succeeds"
        assert has_concrete_condition(ac_text) is True

    def test_no_concrete_condition_vague(self) -> None:
        """Vague outcome fails concrete condition check."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a user, When they use the app, Then the experience is good"
        assert has_concrete_condition(ac_text) is False

    def test_no_concrete_condition_subjective_only(self) -> None:
        """Subjective-only outcome fails concrete condition check."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a user, When they view the page, Then the interface is intuitive"
        assert has_concrete_condition(ac_text) is False

    def test_no_concrete_condition_abstract_outcome(self) -> None:
        """Abstract outcome without observable action fails."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a developer, When they read the code, Then it is maintainable"
        assert has_concrete_condition(ac_text) is False

    def test_no_concrete_condition_vague_improvement(self) -> None:
        """Vague improvement statement fails."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given the system, When updated, Then performance is better"
        assert has_concrete_condition(ac_text) is False

    def test_no_concrete_condition_future_tense(self) -> None:
        """Future tense without concrete action fails."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a user, When they click, Then everything will work properly"
        assert has_concrete_condition(ac_text) is False

    def test_no_concrete_condition_empty_then(self) -> None:
        """Empty 'Then' clause fails."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a user, When they login, Then"
        assert has_concrete_condition(ac_text) is False

    def test_has_concrete_condition_with_numbers(self) -> None:
        """Detects concrete condition with numbers."""
        from yolo_developer.gates.gates.ac_measurability import has_concrete_condition

        ac_text = "Given a search, When submitted, Then at least 10 results appear"
        assert has_concrete_condition(ac_text) is True


class TestImprovementSuggestions:
    """Tests for improvement suggestion generation (Task 5)."""

    def test_generate_suggestions_for_missing_given(self) -> None:
        """Generates suggestion for missing 'Given' clause."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_improvement_suggestions,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="s-1",
                ac_index=0,
                issue_type="missing_gwt",
                description="Missing 'Given' clause",
                severity="blocking",
            )
        ]

        suggestions = generate_improvement_suggestions(issues)
        assert len(suggestions) >= 1
        suggestion_text = next(iter(suggestions.values()))
        assert "Given" in suggestion_text

    def test_generate_suggestions_for_subjective_term(self) -> None:
        """Generates suggestion for subjective terms."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_improvement_suggestions,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="s-1",
                ac_index=0,
                issue_type="subjective_term",
                description="Contains subjective term 'intuitive'",
                severity="warning",
            )
        ]

        suggestions = generate_improvement_suggestions(issues)
        assert len(suggestions) >= 1
        suggestion_text = next(iter(suggestions.values()))
        assert "measurable" in suggestion_text.lower() or "specific" in suggestion_text.lower()

    def test_generate_suggestions_for_vague_outcome(self) -> None:
        """Generates suggestion for vague outcomes."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_improvement_suggestions,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="s-1",
                ac_index=0,
                issue_type="vague_outcome",
                description="'Then' clause lacks concrete condition",
                severity="warning",
            )
        ]

        suggestions = generate_improvement_suggestions(issues)
        assert len(suggestions) >= 1


class TestACMeasurabilityEvaluator:
    """Tests for AC measurability evaluator (Task 6)."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_with_measurable_acs(self) -> None:
        """Evaluator passes when all ACs are measurable."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a user with valid credentials, When they submit login, Then they are redirected to dashboard"
                        },
                        {
                            "content": "Given invalid password, When login submitted, Then error message 'Invalid credentials' is displayed"
                        },
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is True
        assert result.gate_name == "ac_measurability"

    @pytest.mark.asyncio
    async def test_evaluator_fails_missing_gwt_structure(self) -> None:
        """Evaluator fails when ACs lack GWT structure."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-002",
                    "acceptance_criteria": [
                        {"content": "The user interface should be intuitive"},
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert result.gate_name == "ac_measurability"
        assert "Given" in result.reason or "When" in result.reason or "Then" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_passes_with_warnings_subjective_terms(self) -> None:
        """Evaluator passes with warnings when subjective terms present but GWT valid."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-003",
                    "acceptance_criteria": [
                        {
                            "content": "Given a logged-in user, When they view dashboard, Then the layout is clean and user-friendly"
                        },
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Should pass because GWT structure is present (subjective terms are warnings only)
        assert result.passed is True
        # But should have warnings in reason
        assert result.reason is not None
        assert "warning" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluator_handles_empty_stories(self) -> None:
        """Evaluator handles empty stories list."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"stories": []}
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Empty stories should pass (nothing to validate)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_handles_missing_stories_key(self) -> None:
        """Evaluator handles missing stories key in state."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {}
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Missing stories should pass (nothing to validate)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_identifies_failing_stories(self) -> None:
        """Evaluator reason identifies which stories failed."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "Given a user, When they login, Then they see dashboard"},
                    ],
                },
                {
                    "id": "story-002",
                    "acceptance_criteria": [
                        {"content": "The system should be fast"},  # Missing GWT
                    ],
                },
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert "story-002" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_rejects_non_list_stories(self) -> None:
        """Evaluator fails gracefully when stories is not a list."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"stories": "not a list"}
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert "must be a list" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_handles_non_dict_story(self) -> None:
        """Evaluator handles story items that are not dicts."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [{"content": "Given a, When b, Then c succeeds"}],
                },
                None,  # Invalid: not a dict
                "also invalid",  # Invalid: string instead of dict
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert "index-1" in result.reason  # None at index 1
        assert "index-2" in result.reason  # string at index 2

    @pytest.mark.asyncio
    async def test_evaluator_handles_missing_acceptance_criteria(self) -> None:
        """Evaluator handles story missing acceptance_criteria key."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {"id": "story-001"},  # Missing acceptance_criteria key
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Empty acceptance_criteria defaults to empty list, so should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_handles_non_dict_ac(self) -> None:
        """Evaluator handles AC items that are not dicts."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "Given a, When b, Then c succeeds"},
                        "not a dict",  # Invalid
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert "AC at index 1" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_handles_missing_content_key(self) -> None:
        """Evaluator handles AC missing content key."""
        from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"id": "ac-1"},  # Missing content key
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Should fail due to missing GWT structure (empty content)
        assert result.passed is False
        assert "story-001" in result.reason


class TestFailureReportGeneration:
    """Tests for failure report generation (Task 7)."""

    def test_generate_report_with_single_issue(self) -> None:
        """Report is generated for single issue."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_ac_measurability_report,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=0,
                issue_type="missing_gwt",
                description="Missing 'Given' clause",
                severity="blocking",
            )
        ]

        report = generate_ac_measurability_report(issues)
        assert "story-001" in report
        assert "Given" in report
        assert "blocking" in report.lower()

    def test_generate_report_with_multiple_issues(self) -> None:
        """Report handles multiple issues."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_ac_measurability_report,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=0,
                issue_type="missing_gwt",
                description="Missing structure",
                severity="blocking",
            ),
            ACMeasurabilityIssue(
                story_id="story-002",
                ac_index=1,
                issue_type="subjective_term",
                description="Contains 'intuitive'",
                severity="warning",
            ),
        ]

        report = generate_ac_measurability_report(issues)
        assert "story-001" in report
        assert "story-002" in report

    def test_generate_report_includes_ac_index(self) -> None:
        """Report includes AC index."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_ac_measurability_report,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=2,
                issue_type="missing_gwt",
                description="Missing structure",
                severity="blocking",
            )
        ]

        report = generate_ac_measurability_report(issues)
        assert "AC #2" in report or "ac_index" in report.lower() or "#2" in report

    def test_generate_report_includes_remediation(self) -> None:
        """Report includes remediation guidance."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_ac_measurability_report,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=0,
                issue_type="missing_gwt",
                description="Missing 'Given' clause",
                severity="blocking",
            )
        ]

        report = generate_ac_measurability_report(issues)
        # Report should include some form of remediation guidance
        assert any(word in report.lower() for word in ["example", "add", "suggest", "given"])

    def test_generate_report_empty_issues(self) -> None:
        """Report handles empty issues list."""
        from yolo_developer.gates.gates.ac_measurability import generate_ac_measurability_report

        report = generate_ac_measurability_report([])
        # Empty report
        assert report == ""

    def test_generate_report_shows_severity_summary(self) -> None:
        """Report shows blocking and warning counts."""
        from yolo_developer.gates.gates.ac_measurability import (
            ACMeasurabilityIssue,
            generate_ac_measurability_report,
        )

        issues = [
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=0,
                issue_type="missing_gwt",
                description="Missing structure",
                severity="blocking",
            ),
            ACMeasurabilityIssue(
                story_id="story-001",
                ac_index=0,
                issue_type="subjective_term",
                description="Contains 'clean'",
                severity="warning",
            ),
        ]

        report = generate_ac_measurability_report(issues)
        # Should have summary with blocking and warning counts
        assert "1 blocking" in report.lower() or "blocking" in report.lower()
        assert "1 warning" in report.lower() or "warning" in report.lower()


class TestEvaluatorRegistration:
    """Tests for evaluator registration (Task 8)."""

    def test_ac_measurability_evaluator_registered(self) -> None:
        """AC measurability evaluator is registered on module import."""
        # Import the ac_measurability module to trigger registration
        from yolo_developer.gates.evaluators import get_evaluator
        from yolo_developer.gates.gates import ac_measurability  # noqa: F401

        evaluator = get_evaluator("ac_measurability")
        assert evaluator is not None

    def test_ac_measurability_evaluator_follows_protocol(self) -> None:
        """AC measurability evaluator follows GateEvaluator protocol."""
        from yolo_developer.gates.evaluators import GateEvaluator, get_evaluator
        from yolo_developer.gates.gates import ac_measurability  # noqa: F401

        evaluator = get_evaluator("ac_measurability")
        assert evaluator is not None
        # Check it's callable (Protocol uses __call__)
        assert callable(evaluator)
        # Check it's a runtime checkable protocol instance
        assert isinstance(evaluator, GateEvaluator)
