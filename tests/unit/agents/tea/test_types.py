"""Unit tests for TEA agent types (Story 9.1).

Tests for the type definitions used by the TEA agent:
- Finding dataclass
- ValidationResult dataclass
- TEAOutput dataclass
- Literal types (ValidationStatus, FindingSeverity, FindingCategory)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from yolo_developer.agents.tea.types import (
    DeploymentRecommendation,
    Finding,
    FindingCategory,
    FindingSeverity,
    TEAOutput,
    ValidationResult,
    ValidationStatus,
)


class TestValidationStatusLiteral:
    """Tests for ValidationStatus literal type."""

    def test_valid_status_values(self) -> None:
        """Test that ValidationStatus has expected values."""
        expected = {"pending", "passed", "failed", "warning"}
        actual = set(get_args(ValidationStatus))
        assert actual == expected

    def test_status_count(self) -> None:
        """Test that ValidationStatus has exactly 4 values."""
        assert len(get_args(ValidationStatus)) == 4


class TestFindingSeverityLiteral:
    """Tests for FindingSeverity literal type."""

    def test_valid_severity_values(self) -> None:
        """Test that FindingSeverity has expected values."""
        expected = {"critical", "high", "medium", "low", "info"}
        actual = set(get_args(FindingSeverity))
        assert actual == expected

    def test_severity_count(self) -> None:
        """Test that FindingSeverity has exactly 5 values."""
        assert len(get_args(FindingSeverity)) == 5


class TestFindingCategoryLiteral:
    """Tests for FindingCategory literal type."""

    def test_valid_category_values(self) -> None:
        """Test that FindingCategory has expected values."""
        expected = {
            "test_coverage",
            "code_quality",
            "documentation",
            "security",
            "performance",
            "architecture",
        }
        actual = set(get_args(FindingCategory))
        assert actual == expected

    def test_category_count(self) -> None:
        """Test that FindingCategory has exactly 6 values."""
        assert len(get_args(FindingCategory)) == 6


class TestDeploymentRecommendationLiteral:
    """Tests for DeploymentRecommendation literal type."""

    def test_valid_recommendation_values(self) -> None:
        """Test that DeploymentRecommendation has expected values."""
        expected = {"deploy", "deploy_with_warnings", "block"}
        actual = set(get_args(DeploymentRecommendation))
        assert actual == expected

    def test_recommendation_count(self) -> None:
        """Test that DeploymentRecommendation has exactly 3 values."""
        assert len(get_args(DeploymentRecommendation)) == 3


class TestFinding:
    """Tests for Finding frozen dataclass."""

    def test_finding_creation(self) -> None:
        """Test Finding can be created with required fields."""
        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="high",
            description="Missing unit tests",
            location="src/auth.py",
            remediation="Add unit tests",
        )
        assert finding.finding_id == "F001"
        assert finding.category == "test_coverage"
        assert finding.severity == "high"
        assert finding.description == "Missing unit tests"
        assert finding.location == "src/auth.py"
        assert finding.remediation == "Add unit tests"
        assert finding.created_at is not None

    def test_finding_to_dict(self) -> None:
        """Test Finding.to_dict() returns correct dictionary."""
        finding = Finding(
            finding_id="F002",
            category="security",
            severity="critical",
            description="SQL injection risk",
            location="src/db.py:42",
            remediation="Use parameterized queries",
        )
        result = finding.to_dict()

        assert isinstance(result, dict)
        assert result["finding_id"] == "F002"
        assert result["category"] == "security"
        assert result["severity"] == "critical"
        assert result["description"] == "SQL injection risk"
        assert result["location"] == "src/db.py:42"
        assert result["remediation"] == "Use parameterized queries"
        assert "created_at" in result

    def test_finding_is_frozen(self) -> None:
        """Test Finding is immutable."""
        finding = Finding(
            finding_id="F003",
            category="documentation",
            severity="low",
            description="Missing docstring",
            location="src/utils.py",
            remediation="Add docstring",
        )
        with pytest.raises(FrozenInstanceError):
            finding.severity = "high"  # type: ignore[misc]

    def test_finding_all_severities(self) -> None:
        """Test Finding accepts all severity values."""
        severities = get_args(FindingSeverity)
        for severity in severities:
            finding = Finding(
                finding_id=f"F-{severity}",
                category="code_quality",
                severity=severity,
                description=f"Test {severity}",
                location="test.py",
                remediation="Fix it",
            )
            assert finding.severity == severity

    def test_finding_all_categories(self) -> None:
        """Test Finding accepts all category values."""
        categories = get_args(FindingCategory)
        for category in categories:
            finding = Finding(
                finding_id=f"F-{category}",
                category=category,
                severity="info",
                description=f"Test {category}",
                location="test.py",
                remediation="Fix it",
            )
            assert finding.category == category


class TestValidationResult:
    """Tests for ValidationResult frozen dataclass."""

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult can be created with required fields."""
        result = ValidationResult(
            artifact_id="src/main.py",
            validation_status="passed",
        )
        assert result.artifact_id == "src/main.py"
        assert result.validation_status == "passed"
        assert result.findings == ()
        assert result.recommendations == ()
        assert result.score == 100
        assert result.created_at is not None

    def test_validation_result_with_findings(self) -> None:
        """Test ValidationResult with findings."""
        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="medium",
            description="Low coverage",
            location="src/main.py",
            remediation="Add tests",
        )
        result = ValidationResult(
            artifact_id="src/main.py",
            validation_status="warning",
            findings=(finding,),
            recommendations=("Add unit tests",),
            score=80,
        )
        assert len(result.findings) == 1
        assert result.findings[0].finding_id == "F001"
        assert len(result.recommendations) == 1
        assert result.score == 80

    def test_validation_result_to_dict(self) -> None:
        """Test ValidationResult.to_dict() returns correct dictionary."""
        finding = Finding(
            finding_id="F001",
            category="code_quality",
            severity="low",
            description="Long function",
            location="src/utils.py:100",
            remediation="Refactor function",
        )
        result = ValidationResult(
            artifact_id="src/utils.py",
            validation_status="warning",
            findings=(finding,),
            recommendations=("Consider refactoring",),
            score=90,
        )
        dict_result = result.to_dict()

        assert isinstance(dict_result, dict)
        assert dict_result["artifact_id"] == "src/utils.py"
        assert dict_result["validation_status"] == "warning"
        assert len(dict_result["findings"]) == 1
        assert dict_result["findings"][0]["finding_id"] == "F001"
        assert dict_result["recommendations"] == ["Consider refactoring"]
        assert dict_result["score"] == 90
        assert "created_at" in dict_result

    def test_validation_result_is_frozen(self) -> None:
        """Test ValidationResult is immutable."""
        result = ValidationResult(
            artifact_id="test.py",
            validation_status="passed",
        )
        with pytest.raises(FrozenInstanceError):
            result.score = 50  # type: ignore[misc]

    def test_validation_result_all_statuses(self) -> None:
        """Test ValidationResult accepts all status values."""
        statuses = get_args(ValidationStatus)
        for status in statuses:
            result = ValidationResult(
                artifact_id=f"test_{status}.py",
                validation_status=status,
            )
            assert result.validation_status == status


class TestTEAOutput:
    """Tests for TEAOutput frozen dataclass."""

    def test_tea_output_creation_empty(self) -> None:
        """Test TEAOutput can be created with defaults."""
        output = TEAOutput()
        assert output.validation_results == ()
        assert output.processing_notes == ""
        assert output.overall_confidence == 1.0
        assert output.deployment_recommendation == "deploy"
        assert output.created_at is not None

    def test_tea_output_with_results(self) -> None:
        """Test TEAOutput with validation results."""
        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="high",
            description="Missing tests",
            location="src/main.py",
            remediation="Add tests",
        )
        result = ValidationResult(
            artifact_id="src/main.py",
            validation_status="warning",
            findings=(finding,),
            score=75,
        )
        output = TEAOutput(
            validation_results=(result,),
            processing_notes="Validated 1 artifact",
            overall_confidence=0.75,
            deployment_recommendation="deploy_with_warnings",
        )
        assert len(output.validation_results) == 1
        assert output.validation_results[0].artifact_id == "src/main.py"
        assert output.overall_confidence == 0.75
        assert output.deployment_recommendation == "deploy_with_warnings"

    def test_tea_output_to_dict(self) -> None:
        """Test TEAOutput.to_dict() returns correct dictionary."""
        result = ValidationResult(
            artifact_id="src/main.py",
            validation_status="passed",
            score=100,
        )
        output = TEAOutput(
            validation_results=(result,),
            processing_notes="All good",
            overall_confidence=0.95,
            deployment_recommendation="deploy",
        )
        dict_output = output.to_dict()

        assert isinstance(dict_output, dict)
        assert len(dict_output["validation_results"]) == 1
        assert dict_output["validation_results"][0]["artifact_id"] == "src/main.py"
        assert dict_output["processing_notes"] == "All good"
        assert dict_output["overall_confidence"] == 0.95
        assert dict_output["deployment_recommendation"] == "deploy"
        assert "created_at" in dict_output

    def test_tea_output_is_frozen(self) -> None:
        """Test TEAOutput is immutable."""
        output = TEAOutput()
        with pytest.raises(FrozenInstanceError):
            output.overall_confidence = 0.5  # type: ignore[misc]

    def test_tea_output_all_recommendations(self) -> None:
        """Test TEAOutput accepts all recommendation values."""
        recommendations = get_args(DeploymentRecommendation)
        for recommendation in recommendations:
            output = TEAOutput(
                deployment_recommendation=recommendation,
            )
            assert output.deployment_recommendation == recommendation

    def test_tea_output_multiple_results(self) -> None:
        """Test TEAOutput with multiple validation results."""
        results = tuple(
            ValidationResult(
                artifact_id=f"src/file{i}.py",
                validation_status="passed",
                score=90 + i,
            )
            for i in range(3)
        )
        output = TEAOutput(
            validation_results=results,
            processing_notes="Validated 3 artifacts",
        )
        assert len(output.validation_results) == 3
        dict_output = output.to_dict()
        assert len(dict_output["validation_results"]) == 3
