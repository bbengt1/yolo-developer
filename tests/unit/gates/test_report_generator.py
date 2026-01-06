"""Unit tests for report generator utility (Story 3.8 - Task 2).

Tests the generate_failure_report() and format_report_text() functions.
"""

from __future__ import annotations

from yolo_developer.gates.report_generator import (
    format_report_text,
    generate_failure_report,
)
from yolo_developer.gates.report_types import GateFailureReport, GateIssue, Severity


class TestGenerateFailureReport:
    """Tests for generate_failure_report function."""

    def test_generates_report_with_issues(self) -> None:
        """Should generate a GateFailureReport from issues."""
        issues = [
            GateIssue(
                location="req-001",
                issue_type="vague_term",
                description="Contains 'fast'",
                severity=Severity.BLOCKING,
            ),
        ]

        report = generate_failure_report(
            gate_name="testability",
            issues=issues,
            score=0.75,
            threshold=0.80,
        )

        assert isinstance(report, GateFailureReport)
        assert report.gate_name == "testability"
        assert report.score == 0.75
        assert report.threshold == 0.80
        assert len(report.issues) == 1

    def test_report_issues_is_tuple(self) -> None:
        """Report issues should be converted to immutable tuple."""
        issues = [
            GateIssue(
                location="test",
                issue_type="test",
                description="test",
                severity=Severity.WARNING,
            )
        ]

        report = generate_failure_report("test", issues, 0.5, 0.8)

        assert isinstance(report.issues, tuple)

    def test_empty_issues_list(self) -> None:
        """Should handle empty issues list."""
        report = generate_failure_report(
            gate_name="testability",
            issues=[],
            score=1.0,
            threshold=0.80,
        )

        assert report.issues == ()
        assert "0 blocking" in report.summary
        assert "0 warning" in report.summary

    def test_summary_counts_blocking_issues(self) -> None:
        """Summary should count blocking issues correctly."""
        issues = [
            GateIssue("r1", "t1", "d1", Severity.BLOCKING),
            GateIssue("r2", "t2", "d2", Severity.BLOCKING),
            GateIssue("r3", "t3", "d3", Severity.WARNING),
        ]

        report = generate_failure_report("test", issues, 0.5, 0.8)

        assert "2 blocking" in report.summary
        assert "1 warning" in report.summary

    def test_summary_includes_score_and_threshold(self) -> None:
        """Summary should include percentage scores."""
        report = generate_failure_report(
            gate_name="testability",
            issues=[],
            score=0.75,
            threshold=0.80,
        )

        assert "75%" in report.summary
        assert "80%" in report.summary

    def test_summary_includes_gate_name(self) -> None:
        """Summary should include the gate name."""
        report = generate_failure_report(
            gate_name="architecture_validation",
            issues=[],
            score=0.60,
            threshold=0.70,
        )

        assert "architecture_validation" in report.summary

    def test_preserves_issue_order(self) -> None:
        """Issues should preserve their order in the report."""
        issues = [
            GateIssue("first", "t1", "d1", Severity.BLOCKING),
            GateIssue("second", "t2", "d2", Severity.WARNING),
            GateIssue("third", "t3", "d3", Severity.BLOCKING),
        ]

        report = generate_failure_report("test", issues, 0.5, 0.8)

        assert report.issues[0].location == "first"
        assert report.issues[1].location == "second"
        assert report.issues[2].location == "third"


class TestFormatReportText:
    """Tests for format_report_text function."""

    def test_includes_gate_name_header(self) -> None:
        """Report text should include formatted gate name as header."""
        report = GateFailureReport(
            gate_name="testability",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="Passed",
        )

        text = format_report_text(report)

        assert "Testability Gate Report" in text

    def test_includes_summary(self) -> None:
        """Report text should include the summary."""
        report = GateFailureReport(
            gate_name="test",
            issues=(),
            score=0.75,
            threshold=0.80,
            summary="Test summary message",
        )

        text = format_report_text(report)

        assert "Test summary message" in text

    def test_formats_issues_by_location(self) -> None:
        """Report text should group issues by location."""
        issues = (
            GateIssue("req-001", "type1", "Issue 1", Severity.BLOCKING),
            GateIssue("req-001", "type2", "Issue 2", Severity.WARNING),
            GateIssue("req-002", "type3", "Issue 3", Severity.BLOCKING),
        )
        report = GateFailureReport(
            gate_name="test",
            issues=issues,
            score=0.5,
            threshold=0.8,
            summary="Summary",
        )

        text = format_report_text(report)

        assert "Location: req-001" in text
        assert "Location: req-002" in text
        assert "Issue 1" in text
        assert "Issue 2" in text
        assert "Issue 3" in text

    def test_includes_severity_markers(self) -> None:
        """Report text should include [BLOCKING] and [WARNING] markers."""
        issues = (
            GateIssue("loc", "type", "Blocking issue", Severity.BLOCKING),
            GateIssue("loc", "type", "Warning issue", Severity.WARNING),
        )
        report = GateFailureReport(
            gate_name="test",
            issues=issues,
            score=0.5,
            threshold=0.8,
            summary="Summary",
        )

        text = format_report_text(report)

        assert "[BLOCKING]" in text
        assert "[WARNING]" in text

    def test_includes_remediation_suggestions(self) -> None:
        """Report text should include remediation suggestions for known issue types."""
        issues = (GateIssue("req-001", "vague_term", "Contains 'fast'", Severity.BLOCKING),)
        report = GateFailureReport(
            gate_name="testability",
            issues=issues,
            score=0.5,
            threshold=0.8,
            summary="Summary",
        )

        text = format_report_text(report)

        assert "Suggestion:" in text
        # Should include part of the vague_term remediation
        assert "measurable" in text.lower()

    def test_handles_unknown_issue_type(self) -> None:
        """Report text should handle unknown issue types gracefully."""
        issues = (GateIssue("loc", "unknown_type_xyz", "Unknown issue", Severity.BLOCKING),)
        report = GateFailureReport(
            gate_name="test",
            issues=issues,
            score=0.5,
            threshold=0.8,
            summary="Summary",
        )

        text = format_report_text(report)

        # Should not crash, should still show the issue
        assert "Unknown issue" in text
        assert "[BLOCKING]" in text

    def test_empty_report(self) -> None:
        """Report text should handle empty issues gracefully."""
        report = GateFailureReport(
            gate_name="test",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="All passed",
        )

        text = format_report_text(report)

        assert "Test Gate Report" in text
        assert "All passed" in text

    def test_formats_gate_name_with_underscores(self) -> None:
        """Gate names with underscores should be formatted nicely."""
        report = GateFailureReport(
            gate_name="architecture_validation",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="Passed",
        )

        text = format_report_text(report)

        assert "Architecture Validation Gate Report" in text

    def test_includes_separator_lines(self) -> None:
        """Report text should include visual separator lines."""
        report = GateFailureReport(
            gate_name="test",
            issues=(),
            score=1.0,
            threshold=0.80,
            summary="Passed",
        )

        text = format_report_text(report)

        assert "=" in text  # Header separator

    def test_sorted_locations(self) -> None:
        """Locations should be sorted alphabetically."""
        issues = (
            GateIssue("z-req", "type", "Z issue", Severity.BLOCKING),
            GateIssue("a-req", "type", "A issue", Severity.BLOCKING),
            GateIssue("m-req", "type", "M issue", Severity.BLOCKING),
        )
        report = GateFailureReport(
            gate_name="test",
            issues=issues,
            score=0.5,
            threshold=0.8,
            summary="Summary",
        )

        text = format_report_text(report)

        # a-req should appear before m-req, which appears before z-req
        a_pos = text.find("a-req")
        m_pos = text.find("m-req")
        z_pos = text.find("z-req")

        assert a_pos < m_pos < z_pos
