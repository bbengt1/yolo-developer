"""Unit tests for technical risk type definitions (Story 7.5, Task 1).

Tests verify:
- TechnicalRisk dataclass creation and serialization
- TechnicalRiskReport dataclass creation and serialization
- overall_risk_level calculation from risk list
- Immutability of frozen dataclasses
"""

from __future__ import annotations

import pytest


class TestTechnicalRiskCategory:
    """Test TechnicalRiskCategory Literal type."""

    def test_valid_categories(self) -> None:
        """Test that valid categories are accepted."""
        from yolo_developer.agents.architect.types import TechnicalRiskCategory

        valid_categories: list[TechnicalRiskCategory] = [
            "technology",
            "integration",
            "scalability",
            "compatibility",
            "operational",
        ]
        assert len(valid_categories) == 5

    def test_category_type_is_literal(self) -> None:
        """Test that TechnicalRiskCategory is a Literal type."""
        from typing import get_args

        from yolo_developer.agents.architect.types import TechnicalRiskCategory

        args = get_args(TechnicalRiskCategory)
        assert "technology" in args
        assert "integration" in args
        assert "scalability" in args
        assert "compatibility" in args
        assert "operational" in args


class TestMitigationPriority:
    """Test MitigationPriority Literal type."""

    def test_valid_priorities(self) -> None:
        """Test that valid priorities are accepted."""
        from yolo_developer.agents.architect.types import MitigationPriority

        valid_priorities: list[MitigationPriority] = ["P1", "P2", "P3", "P4"]
        assert len(valid_priorities) == 4

    def test_priority_type_is_literal(self) -> None:
        """Test that MitigationPriority is a Literal type."""
        from typing import get_args

        from yolo_developer.agents.architect.types import MitigationPriority

        args = get_args(MitigationPriority)
        assert "P1" in args
        assert "P2" in args
        assert "P3" in args
        assert "P4" in args


class TestTechnicalRisk:
    """Test TechnicalRisk frozen dataclass."""

    def test_create_technical_risk(self) -> None:
        """Test TechnicalRisk dataclass creation."""
        from yolo_developer.agents.architect.types import TechnicalRisk

        risk = TechnicalRisk(
            category="technology",
            description="Using experimental library",
            severity="high",
            affected_components=("AuthService", "UserAPI"),
            mitigation="Switch to stable version",
            mitigation_effort="medium",
            mitigation_priority="P2",
        )

        assert risk.category == "technology"
        assert risk.description == "Using experimental library"
        assert risk.severity == "high"
        assert risk.affected_components == ("AuthService", "UserAPI")
        assert risk.mitigation == "Switch to stable version"
        assert risk.mitigation_effort == "medium"
        assert risk.mitigation_priority == "P2"

    def test_technical_risk_is_frozen(self) -> None:
        """Test that TechnicalRisk is immutable."""
        from yolo_developer.agents.architect.types import TechnicalRisk

        risk = TechnicalRisk(
            category="integration",
            description="API rate limiting",
            severity="medium",
            affected_components=("ExternalAPI",),
            mitigation="Implement caching",
            mitigation_effort="low",
            mitigation_priority="P2",
        )

        with pytest.raises(AttributeError):
            risk.description = "New description"  # type: ignore[misc]

    def test_technical_risk_to_dict(self) -> None:
        """Test TechnicalRisk serialization to dict."""
        from yolo_developer.agents.architect.types import TechnicalRisk

        risk = TechnicalRisk(
            category="scalability",
            description="Single point of failure",
            severity="critical",
            affected_components=("Database", "CacheLayer"),
            mitigation="Add replication",
            mitigation_effort="high",
            mitigation_priority="P1",
        )

        result = risk.to_dict()

        assert isinstance(result, dict)
        assert result["category"] == "scalability"
        assert result["description"] == "Single point of failure"
        assert result["severity"] == "critical"
        assert result["affected_components"] == ["Database", "CacheLayer"]
        assert result["mitigation"] == "Add replication"
        assert result["mitigation_effort"] == "high"
        assert result["mitigation_priority"] == "P1"

    def test_technical_risk_default_affected_components(self) -> None:
        """Test TechnicalRisk with default empty affected_components."""
        from yolo_developer.agents.architect.types import TechnicalRisk

        risk = TechnicalRisk(
            category="operational",
            description="Missing monitoring",
            severity="low",
            mitigation="Add observability",
            mitigation_effort="medium",
            mitigation_priority="P3",
        )

        assert risk.affected_components == ()


class TestTechnicalRiskReport:
    """Test TechnicalRiskReport frozen dataclass."""

    def test_create_technical_risk_report(self) -> None:
        """Test TechnicalRiskReport dataclass creation."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            TechnicalRiskReport,
        )

        risks = (
            TechnicalRisk(
                category="technology",
                description="Deprecated API",
                severity="high",
                affected_components=("LegacyService",),
                mitigation="Migrate to new API",
                mitigation_effort="high",
                mitigation_priority="P2",
            ),
        )

        report = TechnicalRiskReport(
            risks=risks,
            overall_risk_level="high",
            summary="One high-severity technology risk identified",
        )

        assert len(report.risks) == 1
        assert report.overall_risk_level == "high"
        assert "high-severity" in report.summary

    def test_technical_risk_report_is_frozen(self) -> None:
        """Test that TechnicalRiskReport is immutable."""
        from yolo_developer.agents.architect.types import TechnicalRiskReport

        report = TechnicalRiskReport(
            risks=(),
            overall_risk_level="low",
            summary="No risks identified",
        )

        with pytest.raises(AttributeError):
            report.summary = "New summary"  # type: ignore[misc]

    def test_technical_risk_report_to_dict(self) -> None:
        """Test TechnicalRiskReport serialization to dict."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            TechnicalRiskReport,
        )

        risks = (
            TechnicalRisk(
                category="integration",
                description="Vendor lock-in",
                severity="medium",
                affected_components=("CloudProvider",),
                mitigation="Use abstraction layer",
                mitigation_effort="medium",
                mitigation_priority="P2",
            ),
        )

        report = TechnicalRiskReport(
            risks=risks,
            overall_risk_level="medium",
            summary="Medium integration risk",
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert "risks" in result
        assert len(result["risks"]) == 1
        assert result["risks"][0]["category"] == "integration"
        assert result["overall_risk_level"] == "medium"
        assert result["summary"] == "Medium integration risk"

    def test_empty_risks_report(self) -> None:
        """Test TechnicalRiskReport with no risks."""
        from yolo_developer.agents.architect.types import TechnicalRiskReport

        report = TechnicalRiskReport(
            risks=(),
            overall_risk_level="low",
            summary="No technical risks identified",
        )

        assert len(report.risks) == 0
        assert report.overall_risk_level == "low"


class TestCalculateOverallRiskLevel:
    """Test overall_risk_level calculation from risk list."""

    def test_critical_risk_sets_overall_critical(self) -> None:
        """Test that any critical risk sets overall level to critical."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            calculate_overall_risk_level,
        )

        risks = [
            TechnicalRisk(
                category="technology",
                description="Low risk",
                severity="low",
                mitigation="Fix",
                mitigation_effort="low",
                mitigation_priority="P4",
            ),
            TechnicalRisk(
                category="scalability",
                description="Critical risk",
                severity="critical",
                mitigation="Fix now",
                mitigation_effort="high",
                mitigation_priority="P1",
            ),
        ]

        assert calculate_overall_risk_level(risks) == "critical"

    def test_high_risk_with_no_critical(self) -> None:
        """Test that high is returned when no critical risks exist."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            calculate_overall_risk_level,
        )

        risks = [
            TechnicalRisk(
                category="technology",
                description="High risk",
                severity="high",
                mitigation="Fix",
                mitigation_effort="medium",
                mitigation_priority="P2",
            ),
            TechnicalRisk(
                category="integration",
                description="Medium risk",
                severity="medium",
                mitigation="Fix",
                mitigation_effort="low",
                mitigation_priority="P3",
            ),
        ]

        assert calculate_overall_risk_level(risks) == "high"

    def test_medium_risk_with_no_high_or_critical(self) -> None:
        """Test that medium is returned when no high/critical risks exist."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            calculate_overall_risk_level,
        )

        risks = [
            TechnicalRisk(
                category="operational",
                description="Medium risk",
                severity="medium",
                mitigation="Fix",
                mitigation_effort="low",
                mitigation_priority="P3",
            ),
        ]

        assert calculate_overall_risk_level(risks) == "medium"

    def test_low_risk_only(self) -> None:
        """Test that low is returned when only low risks exist."""
        from yolo_developer.agents.architect.types import (
            TechnicalRisk,
            calculate_overall_risk_level,
        )

        risks = [
            TechnicalRisk(
                category="compatibility",
                description="Low risk",
                severity="low",
                mitigation="Fix",
                mitigation_effort="low",
                mitigation_priority="P4",
            ),
        ]

        assert calculate_overall_risk_level(risks) == "low"

    def test_empty_risks_returns_low(self) -> None:
        """Test that empty risk list returns low."""
        from yolo_developer.agents.architect.types import calculate_overall_risk_level

        assert calculate_overall_risk_level([]) == "low"


class TestCalculateMitigationPriority:
    """Test mitigation priority calculation from severity and effort."""

    def test_critical_severity_always_p1(self) -> None:
        """Test that critical severity always results in P1 priority."""
        from yolo_developer.agents.architect.types import calculate_mitigation_priority

        assert calculate_mitigation_priority("critical", "high") == "P1"
        assert calculate_mitigation_priority("critical", "medium") == "P1"
        assert calculate_mitigation_priority("critical", "low") == "P1"

    def test_high_severity_priorities(self) -> None:
        """Test high severity priority calculation."""
        from yolo_developer.agents.architect.types import calculate_mitigation_priority

        assert calculate_mitigation_priority("high", "high") == "P2"
        assert calculate_mitigation_priority("high", "medium") == "P1"
        assert calculate_mitigation_priority("high", "low") == "P1"

    def test_medium_severity_priorities(self) -> None:
        """Test medium severity priority calculation."""
        from yolo_developer.agents.architect.types import calculate_mitigation_priority

        assert calculate_mitigation_priority("medium", "high") == "P3"
        assert calculate_mitigation_priority("medium", "medium") == "P2"
        assert calculate_mitigation_priority("medium", "low") == "P2"

    def test_low_severity_priorities(self) -> None:
        """Test low severity priority calculation."""
        from yolo_developer.agents.architect.types import calculate_mitigation_priority

        assert calculate_mitigation_priority("low", "high") == "P4"
        assert calculate_mitigation_priority("low", "medium") == "P3"
        assert calculate_mitigation_priority("low", "low") == "P3"


class TestTypeExports:
    """Test that types are properly exported from architect module."""

    def test_technical_risk_importable(self) -> None:
        """Test TechnicalRisk is importable from architect module."""
        from yolo_developer.agents.architect import TechnicalRisk

        assert TechnicalRisk is not None

    def test_technical_risk_report_importable(self) -> None:
        """Test TechnicalRiskReport is importable from architect module."""
        from yolo_developer.agents.architect import TechnicalRiskReport

        assert TechnicalRiskReport is not None

    def test_technical_risk_category_importable(self) -> None:
        """Test TechnicalRiskCategory is importable from architect module."""
        from yolo_developer.agents.architect import TechnicalRiskCategory

        assert TechnicalRiskCategory is not None

    def test_mitigation_priority_importable(self) -> None:
        """Test MitigationPriority is importable from architect module."""
        from yolo_developer.agents.architect import MitigationPriority

        assert MitigationPriority is not None

    def test_calculate_functions_importable(self) -> None:
        """Test calculation functions are importable from architect module."""
        from yolo_developer.agents.architect import (
            calculate_mitigation_priority,
            calculate_overall_risk_level,
        )

        assert calculate_overall_risk_level is not None
        assert calculate_mitigation_priority is not None
