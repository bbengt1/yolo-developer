"""Unit tests for testability gate implementation.

Tests for data structures, vague term detection, success criteria detection,
evaluator functionality, and failure report generation.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def ensure_testability_evaluator_registered() -> None:
    """Ensure testability evaluator is registered for each test.

    Other tests may call clear_evaluators(), so we need to
    re-register the testability evaluator before tests that need it.
    """
    from yolo_developer.gates.evaluators import get_evaluator
    from yolo_developer.gates.gates.testability import (
        register_evaluator,
        testability_evaluator,
    )

    if get_evaluator("testability") is None:
        register_evaluator("testability", testability_evaluator)


class TestTestabilityTypes:
    """Tests for testability gate data structures (Task 1)."""

    def test_testability_issue_creation(self) -> None:
        """TestabilityIssue can be created with required fields."""
        from yolo_developer.gates.gates.testability import TestabilityIssue

        issue = TestabilityIssue(
            requirement_id="req-001",
            issue_type="vague_term",
            description="Contains vague term 'fast'",
            severity="blocking",
        )

        assert issue.requirement_id == "req-001"
        assert issue.issue_type == "vague_term"
        assert issue.description == "Contains vague term 'fast'"
        assert issue.severity == "blocking"

    def test_testability_issue_is_frozen(self) -> None:
        """TestabilityIssue is immutable."""
        from yolo_developer.gates.gates.testability import TestabilityIssue

        issue = TestabilityIssue(
            requirement_id="req-001",
            issue_type="vague_term",
            description="Test",
            severity="blocking",
        )

        with pytest.raises(AttributeError):
            issue.requirement_id = "new-id"  # type: ignore[misc]

    def test_testability_issue_to_dict(self) -> None:
        """TestabilityIssue can be converted to dict."""
        from yolo_developer.gates.gates.testability import TestabilityIssue

        issue = TestabilityIssue(
            requirement_id="req-001",
            issue_type="vague_term",
            description="Contains vague term",
            severity="warning",
        )

        d = issue.to_dict()
        assert d["requirement_id"] == "req-001"
        assert d["issue_type"] == "vague_term"
        assert d["description"] == "Contains vague term"
        assert d["severity"] == "warning"

    def test_vague_terms_constant_exists(self) -> None:
        """VAGUE_TERMS constant is defined with expected terms."""
        from yolo_developer.gates.gates.testability import VAGUE_TERMS

        assert isinstance(VAGUE_TERMS, (list, tuple, frozenset))
        # Check some expected vague terms
        vague_terms_lower = [t.lower() for t in VAGUE_TERMS]
        assert "fast" in vague_terms_lower
        assert "easy" in vague_terms_lower
        assert "intuitive" in vague_terms_lower
        assert "user-friendly" in vague_terms_lower or "user friendly" in vague_terms_lower

    def test_vague_terms_has_minimum_coverage(self) -> None:
        """VAGUE_TERMS has comprehensive coverage."""
        from yolo_developer.gates.gates.testability import VAGUE_TERMS

        # Should have at least 15 vague terms
        assert len(VAGUE_TERMS) >= 15


class TestVagueTermDetection:
    """Tests for vague term detection (Task 2)."""

    def test_detect_single_vague_term(self) -> None:
        """Detects single vague term in text."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        result = detect_vague_terms("The system should be fast")
        assert len(result) == 1
        assert result[0][0].lower() == "fast"

    def test_detect_multiple_vague_terms(self) -> None:
        """Detects multiple vague terms in text."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        result = detect_vague_terms("The system should be fast and easy to use")
        terms = [t[0].lower() for t in result]
        assert "fast" in terms
        # "easy to use" is matched as a phrase, which is correct behavior
        assert "easy to use" in terms or "easy" in terms

    def test_detect_vague_term_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        result = detect_vague_terms("The system should be FAST")
        assert len(result) == 1
        assert result[0][0].lower() == "fast"

    def test_detect_no_vague_terms(self) -> None:
        """Returns empty list when no vague terms found."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        result = detect_vague_terms("The API responds within 500ms")
        assert len(result) == 0

    def test_detect_multi_word_vague_phrase(self) -> None:
        """Detects multi-word vague phrases like 'user friendly'."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        result = detect_vague_terms("The interface should be user friendly")
        terms = [t[0].lower() for t in result]
        # Should detect "user friendly" or "user-friendly"
        assert any("user" in t and "friendly" in t for t in terms) or "user friendly" in terms

    def test_detect_returns_position(self) -> None:
        """Detection returns position of vague term."""
        from yolo_developer.gates.gates.testability import detect_vague_terms

        text = "The system should be fast"
        result = detect_vague_terms(text)
        assert len(result) == 1
        term, position = result[0]
        # Position should be where "fast" starts
        assert text[position : position + len(term)].lower() == term.lower()


class TestSuccessCriteriaDetection:
    """Tests for success criteria detection (Task 3)."""

    def test_has_success_criteria_with_numbers(self) -> None:
        """Requirement with numbers has success criteria."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-001",
            "content": "The API responds within 500ms for 95% of requests",
        }
        assert has_success_criteria(requirement) is True

    def test_has_success_criteria_with_percentage(self) -> None:
        """Requirement with percentage has success criteria."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-002",
            "content": "Test coverage must be at least 80%",
        }
        assert has_success_criteria(requirement) is True

    def test_has_success_criteria_with_given_when_then(self) -> None:
        """Requirement with Given/When/Then has success criteria."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-003",
            "content": "Given a user is logged in, When they click logout, Then they are redirected to login page",
        }
        assert has_success_criteria(requirement) is True

    def test_no_success_criteria_vague_requirement(self) -> None:
        """Vague requirement without measurable outcome fails."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-004",
            "content": "The system should be fast and responsive",
        }
        assert has_success_criteria(requirement) is False

    def test_has_success_criteria_with_explicit_field(self) -> None:
        """Requirement with explicit success_criteria field passes."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-005",
            "content": "The system should perform well",
            "success_criteria": "Response time < 200ms",
        }
        assert has_success_criteria(requirement) is True

    def test_has_success_criteria_with_timeframe(self) -> None:
        """Requirement with timeframe has success criteria."""
        from yolo_developer.gates.gates.testability import has_success_criteria

        requirement = {
            "id": "req-006",
            "content": "The batch job completes within 2 hours",
        }
        assert has_success_criteria(requirement) is True


class TestTestabilityEvaluator:
    """Tests for testability evaluator (Task 4)."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_with_testable_requirements(self) -> None:
        """Evaluator passes when all requirements are testable."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The API responds within 500ms"},
                {"id": "req-002", "content": "User login succeeds with valid credentials"},
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is True
        assert result.gate_name == "testability"

    @pytest.mark.asyncio
    async def test_evaluator_fails_with_vague_requirements(self) -> None:
        """Evaluator fails when requirements contain vague terms."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The system should be fast and easy to use"},
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is False
        assert result.gate_name == "testability"
        assert result.reason is not None
        assert "fast" in result.reason.lower() or "easy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluator_fails_without_success_criteria(self) -> None:
        """Evaluator fails when requirements lack success criteria."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The system handles errors gracefully"},
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is False
        assert "success criteria" in result.reason.lower() or "measurable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluator_handles_empty_requirements(self) -> None:
        """Evaluator handles empty requirements list."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"requirements": []}
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        # Empty requirements should pass (nothing to validate)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_handles_missing_requirements_key(self) -> None:
        """Evaluator handles missing requirements key in state."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {}
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        # Missing requirements should pass (nothing to validate)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_identifies_failing_requirements(self) -> None:
        """Evaluator reason identifies which requirements failed."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "The API responds within 500ms"},
                {"id": "req-002", "content": "The system should be intuitive"},
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is False
        assert "req-002" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_rejects_non_list_requirements(self) -> None:
        """Evaluator fails gracefully when requirements is not a list."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {"requirements": "not a list"}
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is False
        assert "must be a list" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_handles_non_dict_requirement(self) -> None:
        """Evaluator handles requirement items that are not dicts."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001", "content": "Valid requirement with 500ms response"},
                None,  # Invalid: not a dict
                "also invalid",  # Invalid: string instead of dict
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        assert result.passed is False
        assert "index-1" in result.reason  # None at index 1
        assert "index-2" in result.reason  # string at index 2

    @pytest.mark.asyncio
    async def test_evaluator_handles_missing_content_key(self) -> None:
        """Evaluator handles requirement missing content key."""
        from yolo_developer.gates.gates.testability import testability_evaluator
        from yolo_developer.gates.types import GateContext

        state: dict[str, Any] = {
            "requirements": [
                {"id": "req-001"},  # Missing content key
            ]
        }
        context = GateContext(state=state, gate_name="testability")

        result = await testability_evaluator(context)
        # Should fail due to no success criteria (empty content)
        assert result.passed is False
        assert "req-001" in result.reason


class TestFailureReportGeneration:
    """Tests for failure report generation (Task 5)."""

    def test_generate_report_with_single_issue(self) -> None:
        """Report is generated for single issue."""
        from yolo_developer.gates.gates.testability import (
            TestabilityIssue,
            generate_testability_report,
        )

        issues = [
            TestabilityIssue(
                requirement_id="req-001",
                issue_type="vague_term",
                description="Contains vague term 'fast'",
                severity="blocking",
            )
        ]

        report = generate_testability_report(issues)
        assert "req-001" in report
        assert "fast" in report.lower()
        assert "blocking" in report.lower()

    def test_generate_report_with_multiple_issues(self) -> None:
        """Report handles multiple issues."""
        from yolo_developer.gates.gates.testability import (
            TestabilityIssue,
            generate_testability_report,
        )

        issues = [
            TestabilityIssue(
                requirement_id="req-001",
                issue_type="vague_term",
                description="Contains vague term 'fast'",
                severity="blocking",
            ),
            TestabilityIssue(
                requirement_id="req-002",
                issue_type="no_success_criteria",
                description="No measurable outcome",
                severity="warning",
            ),
        ]

        report = generate_testability_report(issues)
        assert "req-001" in report
        assert "req-002" in report

    def test_generate_report_includes_remediation(self) -> None:
        """Report includes remediation guidance."""
        from yolo_developer.gates.gates.testability import (
            TestabilityIssue,
            generate_testability_report,
        )

        issues = [
            TestabilityIssue(
                requirement_id="req-001",
                issue_type="vague_term",
                description="Contains vague term 'fast'",
                severity="blocking",
            )
        ]

        report = generate_testability_report(issues)
        # Report should include some form of remediation guidance
        assert any(
            word in report.lower()
            for word in ["instead", "specific", "quantif", "measur", "suggest", "example"]
        )

    def test_generate_report_empty_issues(self) -> None:
        """Report handles empty issues list."""
        from yolo_developer.gates.gates.testability import generate_testability_report

        report = generate_testability_report([])
        # Empty report or success message
        assert report == "" or "pass" in report.lower() or "no issues" in report.lower()


class TestEvaluatorRegistration:
    """Tests for evaluator registration (Task 6)."""

    def test_testability_evaluator_registered(self) -> None:
        """Testability evaluator is registered on module import."""
        # Import the testability module to trigger registration
        from yolo_developer.gates.evaluators import get_evaluator
        from yolo_developer.gates.gates import testability  # noqa: F401

        evaluator = get_evaluator("testability")
        assert evaluator is not None

    def test_testability_evaluator_follows_protocol(self) -> None:
        """Testability evaluator follows GateEvaluator protocol."""
        from yolo_developer.gates.evaluators import GateEvaluator, get_evaluator
        from yolo_developer.gates.gates import testability  # noqa: F401

        evaluator = get_evaluator("testability")
        assert evaluator is not None
        # Check it's callable (Protocol uses __call__)
        assert callable(evaluator)
        # Check it's a runtime checkable protocol instance
        assert isinstance(evaluator, GateEvaluator)
