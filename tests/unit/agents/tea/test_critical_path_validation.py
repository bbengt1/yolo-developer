"""Unit tests for critical path validation (Story 9.2 - Task 4).

This module tests the critical path validation functions that ensure
critical code paths (orchestrator/, gates/, agents/) have 100% coverage.
"""

from __future__ import annotations

from yolo_developer.agents.tea.coverage import (
    CoverageReport,
    CoverageResult,
    validate_critical_paths,
)


class TestValidateCriticalPaths:
    """Tests for validate_critical_paths function."""

    def test_critical_path_full_coverage(self) -> None:
        """Test that critical paths with 100% coverage generate no findings."""
        result = CoverageResult(
            file_path="orchestrator/core.py",
            lines_total=100,
            lines_covered=100,
            coverage_percentage=100.0,
            uncovered_lines=(),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={"orchestrator/core.py": 100.0},
        )
        findings = validate_critical_paths(report)
        assert len(findings) == 0

    def test_critical_path_low_coverage(self) -> None:
        """Test that critical paths with <100% coverage generate critical findings."""
        result = CoverageResult(
            file_path="orchestrator/core.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=((90, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=80.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={"orchestrator/core.py": 80.0},
        )
        findings = validate_critical_paths(report)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert findings[0].category == "test_coverage"
        assert "orchestrator/core.py" in findings[0].location

    def test_multiple_critical_paths_some_failing(self) -> None:
        """Test multiple critical paths with mixed coverage."""
        results = (
            CoverageResult(
                file_path="orchestrator/core.py",
                lines_total=100,
                lines_covered=100,
                coverage_percentage=100.0,
                uncovered_lines=(),
            ),
            CoverageResult(
                file_path="gates/gate.py",
                lines_total=50,
                lines_covered=40,
                coverage_percentage=80.0,
                uncovered_lines=((45, 50),),
            ),
            CoverageResult(
                file_path="agents/analyst/node.py",
                lines_total=200,
                lines_covered=180,
                coverage_percentage=90.0,
                uncovered_lines=((180, 200),),
            ),
        )
        report = CoverageReport(
            results=results,
            overall_coverage=90.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={
                "orchestrator/core.py": 100.0,
                "gates/gate.py": 80.0,
                "agents/analyst/node.py": 90.0,
            },
        )
        findings = validate_critical_paths(report)
        # Two critical paths below 100%
        assert len(findings) == 2
        assert all(f.severity == "critical" for f in findings)

    def test_non_critical_path_not_validated(self) -> None:
        """Test that non-critical paths don't generate critical path findings."""
        result = CoverageResult(
            file_path="utils/helper.py",
            lines_total=100,
            lines_covered=50,
            coverage_percentage=50.0,
            uncovered_lines=((50, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=50.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},  # No critical files
        )
        findings = validate_critical_paths(report)
        # No critical path findings for non-critical files
        assert len(findings) == 0

    def test_finding_includes_coverage_details(self) -> None:
        """Test that finding includes useful coverage details."""
        result = CoverageResult(
            file_path="gates/quality.py",
            lines_total=100,
            lines_covered=75,
            coverage_percentage=75.0,
            uncovered_lines=((76, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=75.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={"gates/quality.py": 75.0},
        )
        findings = validate_critical_paths(report)
        assert len(findings) == 1
        finding = findings[0]
        # Description should mention current coverage and required
        assert "75" in finding.description or "75.0" in finding.description
        assert "100" in finding.description
        # Remediation should be actionable
        assert "test" in finding.remediation.lower() or "coverage" in finding.remediation.lower()

    def test_empty_report(self) -> None:
        """Test handling of empty report."""
        report = CoverageReport(
            results=(),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        findings = validate_critical_paths(report)
        assert len(findings) == 0

    def test_finding_has_valid_id(self) -> None:
        """Test that each finding has a unique ID."""
        results = (
            CoverageResult(
                file_path="orchestrator/a.py",
                lines_total=100,
                lines_covered=80,
                coverage_percentage=80.0,
                uncovered_lines=(),
            ),
            CoverageResult(
                file_path="gates/b.py",
                lines_total=100,
                lines_covered=90,
                coverage_percentage=90.0,
                uncovered_lines=(),
            ),
        )
        report = CoverageReport(
            results=results,
            overall_coverage=85.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={
                "orchestrator/a.py": 80.0,
                "gates/b.py": 90.0,
            },
        )
        findings = validate_critical_paths(report)
        assert len(findings) == 2
        # All findings should have unique IDs
        ids = [f.finding_id for f in findings]
        assert len(set(ids)) == len(ids)
