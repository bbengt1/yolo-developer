"""Unit tests for threshold checking (Story 9.2 - Task 10).

This module tests the threshold checking functions that validate
coverage against configured thresholds.
"""

from __future__ import annotations

from yolo_developer.agents.tea.coverage import (
    CoverageReport,
    CoverageResult,
    check_coverage_threshold,
)


class TestCheckCoverageThreshold:
    """Tests for check_coverage_threshold function."""

    def test_threshold_pass(self) -> None:
        """Test coverage above threshold passes."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=85,
            coverage_percentage=85.0,
            uncovered_lines=((86, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=85.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is True
        assert len(findings) == 0

    def test_threshold_fail(self) -> None:
        """Test coverage below threshold fails with findings."""
        result = CoverageResult(
            file_path="src/module.py",
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
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is False
        assert len(findings) >= 1
        # Finding should indicate the gap
        assert any("75" in f.description for f in findings)
        assert any("80" in f.description for f in findings)

    def test_threshold_boundary_exact(self) -> None:
        """Test coverage exactly at threshold passes."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=((81, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=80.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is True
        assert len(findings) == 0

    def test_threshold_custom_value(self) -> None:
        """Test custom threshold value is respected."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=85,
            coverage_percentage=85.0,
            uncovered_lines=(),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=85.0,
            threshold=90.0,
            passed=False,
            critical_files_coverage={},
        )
        # 85% passes at 80% threshold
        passed, _findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is True

        # 85% fails at 90% threshold
        passed, _findings2 = check_coverage_threshold(report, threshold=90.0)
        assert passed is False

    def test_severity_mapping_critical(self) -> None:
        """Test severity is critical when coverage < 50%."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=40,
            coverage_percentage=40.0,
            uncovered_lines=((41, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=40.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is False
        assert any(f.severity == "critical" for f in findings)

    def test_severity_mapping_high(self) -> None:
        """Test severity is high when 50% <= coverage < 80%."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=70,
            coverage_percentage=70.0,
            uncovered_lines=((71, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=70.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is False
        assert any(f.severity == "high" for f in findings)

    def test_finding_includes_uncovered_files(self) -> None:
        """Test that finding lists files that need more coverage."""
        results = (
            CoverageResult(
                file_path="src/module_a.py",
                lines_total=100,
                lines_covered=60,
                coverage_percentage=60.0,
                uncovered_lines=((61, 100),),
            ),
            CoverageResult(
                file_path="src/module_b.py",
                lines_total=100,
                lines_covered=90,
                coverage_percentage=90.0,
                uncovered_lines=((91, 100),),
            ),
        )
        report = CoverageReport(
            results=results,
            overall_coverage=75.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is False
        # Should have specific findings for under-covered files
        assert len(findings) >= 1

    def test_empty_report_passes(self) -> None:
        """Test empty report (no files) passes."""
        report = CoverageReport(
            results=(),
            overall_coverage=100.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert passed is True
        assert len(findings) == 0

    def test_finding_has_remediation(self) -> None:
        """Test that findings include actionable remediation."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=70,
            coverage_percentage=70.0,
            uncovered_lines=((71, 100),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=70.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        _passed, findings = check_coverage_threshold(report, threshold=80.0)
        assert len(findings) >= 1
        for finding in findings:
            # Remediation should be non-empty and actionable
            assert finding.remediation
            assert len(finding.remediation) > 10
