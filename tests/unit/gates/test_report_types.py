"""Unit tests for gate failure report types (Story 3.8 - Task 1).

Tests the GateIssue, Severity, and GateFailureReport data models
for creating structured failure reports.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.report_types import (
    GateFailureReport,
    GateIssue,
    Severity,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_blocking_value(self) -> None:
        """Severity.BLOCKING should have value 'blocking'."""
        assert Severity.BLOCKING.value == "blocking"

    def test_warning_value(self) -> None:
        """Severity.WARNING should have value 'warning'."""
        assert Severity.WARNING.value == "warning"

    def test_enum_members(self) -> None:
        """Severity should have exactly two members."""
        assert len(Severity) == 2
        assert Severity.BLOCKING in Severity
        assert Severity.WARNING in Severity


class TestGateIssue:
    """Tests for GateIssue dataclass."""

    def test_create_issue_with_required_fields(self) -> None:
        """GateIssue should be created with required fields."""
        issue = GateIssue(
            location="req-001",
            issue_type="vague_term",
            description="Contains vague term 'fast'",
            severity=Severity.BLOCKING,
        )

        assert issue.location == "req-001"
        assert issue.issue_type == "vague_term"
        assert issue.description == "Contains vague term 'fast'"
        assert issue.severity == Severity.BLOCKING
        assert issue.context == {}

    def test_create_issue_with_context(self) -> None:
        """GateIssue should accept optional context dict."""
        issue = GateIssue(
            location="file.py:42",
            issue_type="coverage_gap",
            description="Missing branch coverage",
            severity=Severity.WARNING,
            context={"line": 42, "branch": "else"},
        )

        assert issue.context == {"line": 42, "branch": "else"}

    def test_issue_is_frozen(self) -> None:
        """GateIssue should be immutable (frozen)."""
        issue = GateIssue(
            location="req-001",
            issue_type="vague_term",
            description="Test",
            severity=Severity.BLOCKING,
        )

        with pytest.raises(AttributeError):
            issue.location = "changed"  # type: ignore[misc]

    def test_to_dict_returns_expected_structure(self) -> None:
        """to_dict() should return properly structured dict."""
        issue = GateIssue(
            location="req-001",
            issue_type="vague_term",
            description="Contains vague term",
            severity=Severity.BLOCKING,
            context={"term": "fast"},
        )

        result = issue.to_dict()

        assert result == {
            "location": "req-001",
            "issue_type": "vague_term",
            "description": "Contains vague term",
            "severity": "blocking",
            "context": {"term": "fast"},
        }

    def test_to_dict_severity_uses_value(self) -> None:
        """to_dict() should convert severity enum to string value."""
        issue = GateIssue(
            location="test",
            issue_type="test",
            description="test",
            severity=Severity.WARNING,
        )

        result = issue.to_dict()

        assert result["severity"] == "warning"
        assert isinstance(result["severity"], str)

    def test_to_dict_empty_context(self) -> None:
        """to_dict() should include empty context dict."""
        issue = GateIssue(
            location="test",
            issue_type="test",
            description="test",
            severity=Severity.BLOCKING,
        )

        result = issue.to_dict()

        assert result["context"] == {}


class TestGateFailureReport:
    """Tests for GateFailureReport dataclass."""

    def test_create_report_with_required_fields(self) -> None:
        """GateFailureReport should be created with required fields."""
        issues = (
            GateIssue(
                location="req-001",
                issue_type="vague_term",
                description="Contains 'fast'",
                severity=Severity.BLOCKING,
            ),
        )

        report = GateFailureReport(
            gate_name="testability",
            issues=issues,
            score=0.75,
            threshold=0.80,
            summary="Testability score 75% below threshold 80%.",
        )

        assert report.gate_name == "testability"
        assert report.issues == issues
        assert report.score == 0.75
        assert report.threshold == 0.80
        assert report.summary == "Testability score 75% below threshold 80%."

    def test_report_issues_is_tuple(self) -> None:
        """GateFailureReport.issues should be a tuple (immutable)."""
        report = GateFailureReport(
            gate_name="testability",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="Passed",
        )

        assert isinstance(report.issues, tuple)

    def test_report_is_frozen(self) -> None:
        """GateFailureReport should be immutable (frozen)."""
        report = GateFailureReport(
            gate_name="testability",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="Passed",
        )

        with pytest.raises(AttributeError):
            report.gate_name = "changed"  # type: ignore[misc]

    def test_to_dict_returns_expected_structure(self) -> None:
        """to_dict() should return properly structured dict."""
        issue = GateIssue(
            location="req-001",
            issue_type="vague_term",
            description="Contains vague term",
            severity=Severity.BLOCKING,
        )

        report = GateFailureReport(
            gate_name="testability",
            issues=(issue,),
            score=0.75,
            threshold=0.80,
            summary="Test summary",
        )

        result = report.to_dict()

        assert result == {
            "gate_name": "testability",
            "issues": [
                {
                    "location": "req-001",
                    "issue_type": "vague_term",
                    "description": "Contains vague term",
                    "severity": "blocking",
                    "context": {},
                }
            ],
            "score": 0.75,
            "threshold": 0.80,
            "summary": "Test summary",
        }

    def test_to_dict_with_multiple_issues(self) -> None:
        """to_dict() should serialize all issues."""
        issues = (
            GateIssue(
                location="req-001",
                issue_type="vague_term",
                description="Issue 1",
                severity=Severity.BLOCKING,
            ),
            GateIssue(
                location="req-002",
                issue_type="no_success_criteria",
                description="Issue 2",
                severity=Severity.WARNING,
            ),
        )

        report = GateFailureReport(
            gate_name="testability",
            issues=issues,
            score=0.50,
            threshold=0.80,
            summary="Multiple issues",
        )

        result = report.to_dict()

        assert len(result["issues"]) == 2
        assert result["issues"][0]["location"] == "req-001"
        assert result["issues"][1]["location"] == "req-002"

    def test_to_dict_with_empty_issues(self) -> None:
        """to_dict() should handle empty issues tuple."""
        report = GateFailureReport(
            gate_name="testability",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="All passed",
        )

        result = report.to_dict()

        assert result["issues"] == []

    def test_report_with_decimal_score_threshold(self) -> None:
        """Report should preserve decimal precision."""
        report = GateFailureReport(
            gate_name="confidence_scoring",
            issues=(),
            score=0.8567,
            threshold=0.90,
            summary="Score below threshold",
        )

        assert report.score == 0.8567
        assert report.threshold == 0.90

        result = report.to_dict()
        assert result["score"] == 0.8567
        assert result["threshold"] == 0.90
