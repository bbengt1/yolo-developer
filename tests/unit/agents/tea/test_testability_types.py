"""Unit tests for testability audit types (Story 9.6).

Tests for TestabilityIssue, TestabilityScore, TestabilityMetrics, and TestabilityReport
frozen dataclasses.
"""

from __future__ import annotations

import pytest


class TestTestabilityPattern:
    """Tests for TestabilityPattern literal type."""

    def test_valid_patterns(self) -> None:
        """Test that all expected pattern values are valid."""
        from yolo_developer.agents.tea.testability import TestabilityPattern

        # These should type check correctly
        patterns: list[TestabilityPattern] = [
            "global_state",
            "tight_coupling",
            "hidden_dependency",
            "complex_conditional",
            "long_method",
            "deep_nesting",
        ]
        assert len(patterns) == 6


class TestTestabilitySeverity:
    """Tests for TestabilitySeverity literal type."""

    def test_valid_severities(self) -> None:
        """Test that all expected severity values are valid."""
        from yolo_developer.agents.tea.testability import TestabilitySeverity

        severities: list[TestabilitySeverity] = [
            "critical",
            "high",
            "medium",
            "low",
        ]
        assert len(severities) == 4


class TestTestabilityIssue:
    """Tests for TestabilityIssue frozen dataclass."""

    def test_creation(self) -> None:
        """Test TestabilityIssue can be created with required fields."""
        from yolo_developer.agents.tea.testability import TestabilityIssue

        issue = TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/module.py",
            line_start=10,
            line_end=15,
            description="Module-level mutable variable",
            impact="Cannot isolate tests",
            remediation="Use dependency injection",
        )

        assert issue.issue_id == "T-a1b2c3d4-001"
        assert issue.pattern_type == "global_state"
        assert issue.severity == "critical"
        assert issue.location == "src/module.py"
        assert issue.line_start == 10
        assert issue.line_end == 15
        assert issue.description == "Module-level mutable variable"
        assert issue.impact == "Cannot isolate tests"
        assert issue.remediation == "Use dependency injection"
        assert issue.created_at  # Should have default timestamp

    def test_to_dict(self) -> None:
        """Test TestabilityIssue.to_dict() serialization."""
        from yolo_developer.agents.tea.testability import TestabilityIssue

        issue = TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/module.py",
            line_start=10,
            line_end=15,
            description="Module-level mutable variable",
            impact="Cannot isolate tests",
            remediation="Use dependency injection",
        )

        result = issue.to_dict()

        assert result["issue_id"] == "T-a1b2c3d4-001"
        assert result["pattern_type"] == "global_state"
        assert result["severity"] == "critical"
        assert result["location"] == "src/module.py"
        assert result["line_start"] == 10
        assert result["line_end"] == 15
        assert result["description"] == "Module-level mutable variable"
        assert result["impact"] == "Cannot isolate tests"
        assert result["remediation"] == "Use dependency injection"
        assert "created_at" in result

    def test_immutability(self) -> None:
        """Test TestabilityIssue is immutable (frozen)."""
        from yolo_developer.agents.tea.testability import TestabilityIssue

        issue = TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/module.py",
            line_start=10,
            line_end=15,
            description="Test",
            impact="Test",
            remediation="Test",
        )

        with pytest.raises(AttributeError):
            issue.issue_id = "changed"  # type: ignore[misc]


class TestTestabilityScore:
    """Tests for TestabilityScore frozen dataclass."""

    def test_creation(self) -> None:
        """Test TestabilityScore can be created with required fields."""
        from yolo_developer.agents.tea.testability import TestabilityScore

        score = TestabilityScore(
            score=75,
            base_score=100,
            breakdown={
                "critical_penalty": -20,
                "high_penalty": 0,
                "medium_penalty": -5,
                "low_penalty": 0,
            },
        )

        assert score.score == 75
        assert score.base_score == 100
        assert score.breakdown["critical_penalty"] == -20
        assert score.breakdown["high_penalty"] == 0

    def test_to_dict(self) -> None:
        """Test TestabilityScore.to_dict() serialization."""
        from yolo_developer.agents.tea.testability import TestabilityScore

        score = TestabilityScore(
            score=75,
            base_score=100,
            breakdown={
                "critical_penalty": -20,
                "high_penalty": 0,
                "medium_penalty": -5,
                "low_penalty": 0,
            },
        )

        result = score.to_dict()

        assert result["score"] == 75
        assert result["base_score"] == 100
        assert result["breakdown"]["critical_penalty"] == -20

    def test_immutability(self) -> None:
        """Test TestabilityScore is immutable (frozen)."""
        from yolo_developer.agents.tea.testability import TestabilityScore

        score = TestabilityScore(
            score=75,
            base_score=100,
            breakdown={},
        )

        with pytest.raises(AttributeError):
            score.score = 50  # type: ignore[misc]


class TestTestabilityMetrics:
    """Tests for TestabilityMetrics frozen dataclass."""

    def test_creation(self) -> None:
        """Test TestabilityMetrics can be created with required fields."""
        from yolo_developer.agents.tea.testability import TestabilityMetrics

        metrics = TestabilityMetrics(
            total_issues=5,
            issues_by_severity={"critical": 1, "high": 2, "medium": 1, "low": 1},
            issues_by_pattern={"global_state": 1, "tight_coupling": 2, "long_method": 2},
            files_analyzed=10,
            files_with_issues=3,
        )

        assert metrics.total_issues == 5
        assert metrics.issues_by_severity["critical"] == 1
        assert metrics.issues_by_pattern["global_state"] == 1
        assert metrics.files_analyzed == 10
        assert metrics.files_with_issues == 3

    def test_to_dict(self) -> None:
        """Test TestabilityMetrics.to_dict() serialization."""
        from yolo_developer.agents.tea.testability import TestabilityMetrics

        metrics = TestabilityMetrics(
            total_issues=5,
            issues_by_severity={"critical": 1, "high": 2, "medium": 1, "low": 1},
            issues_by_pattern={"global_state": 1, "tight_coupling": 2, "long_method": 2},
            files_analyzed=10,
            files_with_issues=3,
        )

        result = metrics.to_dict()

        assert result["total_issues"] == 5
        assert result["issues_by_severity"]["critical"] == 1
        assert result["files_analyzed"] == 10

    def test_immutability(self) -> None:
        """Test TestabilityMetrics is immutable (frozen)."""
        from yolo_developer.agents.tea.testability import TestabilityMetrics

        metrics = TestabilityMetrics(
            total_issues=5,
            issues_by_severity={},
            issues_by_pattern={},
            files_analyzed=10,
            files_with_issues=3,
        )

        with pytest.raises(AttributeError):
            metrics.total_issues = 10  # type: ignore[misc]


class TestTestabilityReport:
    """Tests for TestabilityReport frozen dataclass."""

    def test_creation(self) -> None:
        """Test TestabilityReport can be created with required fields."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )

        issue = TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/module.py",
            line_start=10,
            line_end=15,
            description="Test",
            impact="Test",
            remediation="Test",
        )

        score = TestabilityScore(
            score=80,
            base_score=100,
            breakdown={"critical_penalty": -20},
        )

        metrics = TestabilityMetrics(
            total_issues=1,
            issues_by_severity={"critical": 1},
            issues_by_pattern={"global_state": 1},
            files_analyzed=5,
            files_with_issues=1,
        )

        report = TestabilityReport(
            issues=(issue,),
            score=score,
            metrics=metrics,
            recommendations=("Fix global state in src/module.py",),
        )

        assert len(report.issues) == 1
        assert report.score.score == 80
        assert report.metrics.total_issues == 1
        assert len(report.recommendations) == 1
        assert report.created_at  # Should have default timestamp

    def test_to_dict(self) -> None:
        """Test TestabilityReport.to_dict() serialization."""
        from yolo_developer.agents.tea.testability import (
            TestabilityIssue,
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )

        issue = TestabilityIssue(
            issue_id="T-a1b2c3d4-001",
            pattern_type="global_state",
            severity="critical",
            location="src/module.py",
            line_start=10,
            line_end=15,
            description="Test",
            impact="Test",
            remediation="Test",
        )

        score = TestabilityScore(
            score=80,
            base_score=100,
            breakdown={"critical_penalty": -20},
        )

        metrics = TestabilityMetrics(
            total_issues=1,
            issues_by_severity={"critical": 1},
            issues_by_pattern={"global_state": 1},
            files_analyzed=5,
            files_with_issues=1,
        )

        report = TestabilityReport(
            issues=(issue,),
            score=score,
            metrics=metrics,
            recommendations=("Fix global state",),
        )

        result = report.to_dict()

        assert len(result["issues"]) == 1
        assert result["issues"][0]["issue_id"] == "T-a1b2c3d4-001"
        assert result["score"]["score"] == 80
        assert result["metrics"]["total_issues"] == 1
        assert result["recommendations"] == ["Fix global state"]
        assert "created_at" in result

    def test_immutability(self) -> None:
        """Test TestabilityReport is immutable (frozen)."""
        from yolo_developer.agents.tea.testability import (
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )

        score = TestabilityScore(score=80, base_score=100, breakdown={})
        metrics = TestabilityMetrics(
            total_issues=0,
            issues_by_severity={},
            issues_by_pattern={},
            files_analyzed=0,
            files_with_issues=0,
        )

        report = TestabilityReport(
            issues=(),
            score=score,
            metrics=metrics,
            recommendations=(),
        )

        with pytest.raises(AttributeError):
            report.issues = ()  # type: ignore[misc]

    def test_empty_report(self) -> None:
        """Test TestabilityReport with no issues."""
        from yolo_developer.agents.tea.testability import (
            TestabilityMetrics,
            TestabilityReport,
            TestabilityScore,
        )

        score = TestabilityScore(score=100, base_score=100, breakdown={})
        metrics = TestabilityMetrics(
            total_issues=0,
            issues_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0},
            issues_by_pattern={},
            files_analyzed=5,
            files_with_issues=0,
        )

        report = TestabilityReport(
            issues=(),
            score=score,
            metrics=metrics,
            recommendations=(),
        )

        assert len(report.issues) == 0
        assert report.score.score == 100
        assert report.metrics.files_with_issues == 0
