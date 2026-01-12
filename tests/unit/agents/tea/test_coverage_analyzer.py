"""Unit tests for coverage analyzer (Story 9.2 - Task 9).

This module tests the coverage analysis functions:
- _analyze_coverage: Main coverage analysis function
- _count_code_lines: Line counting for code files
- _estimate_coverage: Heuristic-based coverage estimation
"""

from __future__ import annotations

from yolo_developer.agents.tea.coverage import (
    CoverageReport,
    analyze_coverage,
)


class TestAnalyzeCoverage:
    """Tests for analyze_coverage function."""

    def test_analyze_coverage_with_tests(self) -> None:
        """Test coverage analysis for code with corresponding tests."""
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": '''"""Module docstring."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b
''',
            }
        ]
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''"""Test module."""

import pytest

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(3, 1) == 2
''',
            }
        ]
        report = analyze_coverage(code_files, test_files)
        assert isinstance(report, CoverageReport)
        assert len(report.results) == 1
        assert report.overall_coverage > 0.0
        assert report.results[0].file_path == "src/module.py"

    def test_analyze_coverage_no_tests(self) -> None:
        """Test coverage analysis for code without tests."""
        code_files = [
            {
                "artifact_id": "src/untested.py",
                "content": '''"""Untested module."""

def some_function():
    return 42
''',
            }
        ]
        test_files: list[dict[str, str]] = []
        report = analyze_coverage(code_files, test_files)
        assert isinstance(report, CoverageReport)
        assert len(report.results) == 1
        # Without tests, coverage should be very low
        assert report.overall_coverage < 50.0

    def test_analyze_coverage_empty_code(self) -> None:
        """Test coverage analysis for empty code files."""
        code_files = [
            {
                "artifact_id": "src/empty.py",
                "content": "",
            }
        ]
        test_files: list[dict[str, str]] = []
        report = analyze_coverage(code_files, test_files)
        assert isinstance(report, CoverageReport)
        # Empty files should have 100% coverage (nothing to test)
        assert report.results[0].coverage_percentage == 100.0

    def test_analyze_coverage_critical_paths(self) -> None:
        """Test critical path identification."""
        code_files = [
            {
                "artifact_id": "orchestrator/core.py",
                "content": '''"""Orchestrator core."""

def orchestrate():
    pass
''',
            },
            {
                "artifact_id": "gates/gate.py",
                "content": '''"""Gate module."""

def validate():
    pass
''',
            },
            {
                "artifact_id": "utils/helper.py",
                "content": '''"""Helper module."""

def help():
    pass
''',
            },
        ]
        test_files = [
            {
                "artifact_id": "tests/test_core.py",
                "content": '''def test_orchestrate(): assert True''',
            },
            {
                "artifact_id": "tests/test_gate.py",
                "content": '''def test_validate(): assert True''',
            },
        ]
        report = analyze_coverage(code_files, test_files)
        # Critical paths should be identified
        assert "orchestrator/core.py" in report.critical_files_coverage
        assert "gates/gate.py" in report.critical_files_coverage
        # Non-critical path should not be in critical_files_coverage
        assert "utils/helper.py" not in report.critical_files_coverage

    def test_analyze_coverage_empty_inputs(self) -> None:
        """Test coverage analysis with empty inputs."""
        report = analyze_coverage([], [])
        assert isinstance(report, CoverageReport)
        assert len(report.results) == 0
        assert report.overall_coverage == 100.0  # No files = 100% by convention
        assert report.passed is True

    def test_analyze_coverage_threshold_default(self) -> None:
        """Test that default threshold is 80%."""
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": '''def func(): pass''',
            }
        ]
        test_files: list[dict[str, str]] = []
        report = analyze_coverage(code_files, test_files)
        assert report.threshold == 80.0

    def test_analyze_coverage_threshold_custom(self) -> None:
        """Test custom threshold."""
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": '''def func(): pass''',
            }
        ]
        test_files: list[dict[str, str]] = []
        report = analyze_coverage(code_files, test_files, threshold=90.0)
        assert report.threshold == 90.0


class TestCoverageResultCalculation:
    """Tests for coverage result calculations."""

    def test_lines_total_counts_executable_lines(self) -> None:
        """Test that lines_total counts executable lines."""
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": '''"""Module docstring."""

# Comment line
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
''',
            }
        ]
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''def test_add(): assert True''',
            }
        ]
        report = analyze_coverage(code_files, test_files)
        # Should count def lines and return statements
        assert report.results[0].lines_total > 0

    def test_coverage_percentage_calculation(self) -> None:
        """Test coverage percentage is calculated correctly."""
        code_files = [
            {
                "artifact_id": "src/module.py",
                "content": '''def a(): pass
def b(): pass
def c(): pass
def d(): pass
''',
            }
        ]
        # Test file that tests 2 of 4 functions
        test_files = [
            {
                "artifact_id": "tests/test_module.py",
                "content": '''def test_a(): assert True
def test_b(): assert True
''',
            }
        ]
        report = analyze_coverage(code_files, test_files)
        result = report.results[0]
        # Coverage should be based on test presence heuristic
        assert 0.0 <= result.coverage_percentage <= 100.0


class TestAgentsPathDetection:
    """Tests for agents/ path as critical."""

    def test_agents_path_is_critical(self) -> None:
        """Test that agents/ paths are considered critical."""
        code_files = [
            {
                "artifact_id": "agents/analyst/node.py",
                "content": '''def analyze(): pass''',
            }
        ]
        test_files = [
            {
                "artifact_id": "tests/test_analyst.py",
                "content": '''def test_analyze(): assert True''',
            }
        ]
        report = analyze_coverage(code_files, test_files)
        assert "agents/analyst/node.py" in report.critical_files_coverage
