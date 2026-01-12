"""Unit tests for coverage types (Story 9.2 - Task 8).

This module tests the frozen dataclasses used for coverage validation:
- CoverageResult: Individual file coverage result
- CoverageReport: Aggregate coverage report with threshold checking
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.tea.coverage import (
    CoverageReport,
    CoverageResult,
)


class TestCoverageResult:
    """Tests for CoverageResult dataclass."""

    def test_coverage_result_creation(self) -> None:
        """Test creating a CoverageResult with all fields."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=((10, 15), (30, 35)),
        )
        assert result.file_path == "src/module.py"
        assert result.lines_total == 100
        assert result.lines_covered == 80
        assert result.coverage_percentage == 80.0
        assert result.uncovered_lines == ((10, 15), (30, 35))

    def test_coverage_result_to_dict(self) -> None:
        """Test to_dict() serialization."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=((10, 15),),
        )
        data = result.to_dict()
        assert data["file_path"] == "src/module.py"
        assert data["lines_total"] == 100
        assert data["lines_covered"] == 80
        assert data["coverage_percentage"] == 80.0
        assert data["uncovered_lines"] == [[10, 15]]  # Lists for JSON serialization

    def test_coverage_result_immutability(self) -> None:
        """Test that CoverageResult is frozen (immutable)."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=(),
        )
        with pytest.raises(AttributeError):
            result.file_path = "other.py"  # type: ignore[misc]

    def test_coverage_result_zero_coverage(self) -> None:
        """Test edge case: 0% coverage."""
        result = CoverageResult(
            file_path="src/untested.py",
            lines_total=50,
            lines_covered=0,
            coverage_percentage=0.0,
            uncovered_lines=((1, 50),),
        )
        assert result.coverage_percentage == 0.0
        assert result.lines_covered == 0

    def test_coverage_result_full_coverage(self) -> None:
        """Test edge case: 100% coverage."""
        result = CoverageResult(
            file_path="src/fully_tested.py",
            lines_total=100,
            lines_covered=100,
            coverage_percentage=100.0,
            uncovered_lines=(),
        )
        assert result.coverage_percentage == 100.0
        assert result.uncovered_lines == ()

    def test_coverage_result_empty_file(self) -> None:
        """Test edge case: empty file (0 lines)."""
        result = CoverageResult(
            file_path="src/empty.py",
            lines_total=0,
            lines_covered=0,
            coverage_percentage=100.0,  # Empty files are considered 100% covered
            uncovered_lines=(),
        )
        assert result.lines_total == 0
        assert result.coverage_percentage == 100.0


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""

    def test_coverage_report_creation(self) -> None:
        """Test creating a CoverageReport with all fields."""
        result1 = CoverageResult(
            file_path="src/a.py",
            lines_total=100,
            lines_covered=90,
            coverage_percentage=90.0,
            uncovered_lines=((10, 15),),
        )
        result2 = CoverageResult(
            file_path="src/b.py",
            lines_total=100,
            lines_covered=80,
            coverage_percentage=80.0,
            uncovered_lines=((20, 30),),
        )
        report = CoverageReport(
            results=(result1, result2),
            overall_coverage=85.0,
            threshold=80.0,
            passed=True,
            critical_files_coverage={"src/a.py": 90.0},
        )
        assert len(report.results) == 2
        assert report.overall_coverage == 85.0
        assert report.threshold == 80.0
        assert report.passed is True
        assert report.critical_files_coverage == {"src/a.py": 90.0}

    def test_coverage_report_to_dict(self) -> None:
        """Test to_dict() serialization."""
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
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        data = report.to_dict()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["overall_coverage"] == 85.0
        assert data["threshold"] == 80.0
        assert data["passed"] is True
        assert data["critical_files_coverage"] == {}

    def test_coverage_report_immutability(self) -> None:
        """Test that CoverageReport is frozen (immutable)."""
        report = CoverageReport(
            results=(),
            overall_coverage=0.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        with pytest.raises(AttributeError):
            report.overall_coverage = 100.0  # type: ignore[misc]

    def test_coverage_report_empty_results(self) -> None:
        """Test report with no results (empty project)."""
        report = CoverageReport(
            results=(),
            overall_coverage=100.0,  # No files = 100% covered by convention
            threshold=80.0,
            passed=True,
            critical_files_coverage={},
        )
        assert len(report.results) == 0
        assert report.passed is True

    def test_coverage_report_failed_threshold(self) -> None:
        """Test report that fails threshold check."""
        result = CoverageResult(
            file_path="src/module.py",
            lines_total=100,
            lines_covered=70,
            coverage_percentage=70.0,
            uncovered_lines=((1, 30),),
        )
        report = CoverageReport(
            results=(result,),
            overall_coverage=70.0,
            threshold=80.0,
            passed=False,
            critical_files_coverage={},
        )
        assert report.passed is False
        assert report.overall_coverage < report.threshold

    def test_coverage_report_boundary_threshold(self) -> None:
        """Test report at exactly the threshold boundary."""
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
            passed=True,  # Exactly at threshold should pass
            critical_files_coverage={},
        )
        assert report.passed is True
        assert report.overall_coverage == report.threshold
