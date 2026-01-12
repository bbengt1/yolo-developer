"""Unit tests for risk categorization types (Story 9.5 - Task 1, 9).

Tests for the risk categorization data types:
- RiskLevel: Literal type for risk levels (critical/high/low)
- OverallRiskLevel: Literal type for overall risk (critical/high/low/none)
- CategorizedRisk: A categorized risk derived from a validation finding
- RiskReport: Complete risk categorization report

All types should be frozen dataclasses (immutable) per ADR-001.
"""

from __future__ import annotations

import pytest


class TestRiskLevel:
    """Tests for RiskLevel literal type."""

    def test_risk_level_critical(self) -> None:
        """Test that 'critical' is a valid RiskLevel."""
        from yolo_developer.agents.tea.risk import RiskLevel

        level: RiskLevel = "critical"
        assert level == "critical"

    def test_risk_level_high(self) -> None:
        """Test that 'high' is a valid RiskLevel."""
        from yolo_developer.agents.tea.risk import RiskLevel

        level: RiskLevel = "high"
        assert level == "high"

    def test_risk_level_low(self) -> None:
        """Test that 'low' is a valid RiskLevel."""
        from yolo_developer.agents.tea.risk import RiskLevel

        level: RiskLevel = "low"
        assert level == "low"


class TestOverallRiskLevel:
    """Tests for OverallRiskLevel literal type."""

    def test_overall_risk_level_critical(self) -> None:
        """Test that 'critical' is a valid OverallRiskLevel."""
        from yolo_developer.agents.tea.risk import OverallRiskLevel

        level: OverallRiskLevel = "critical"
        assert level == "critical"

    def test_overall_risk_level_high(self) -> None:
        """Test that 'high' is a valid OverallRiskLevel."""
        from yolo_developer.agents.tea.risk import OverallRiskLevel

        level: OverallRiskLevel = "high"
        assert level == "high"

    def test_overall_risk_level_low(self) -> None:
        """Test that 'low' is a valid OverallRiskLevel."""
        from yolo_developer.agents.tea.risk import OverallRiskLevel

        level: OverallRiskLevel = "low"
        assert level == "low"

    def test_overall_risk_level_none(self) -> None:
        """Test that 'none' is a valid OverallRiskLevel."""
        from yolo_developer.agents.tea.risk import OverallRiskLevel

        level: OverallRiskLevel = "none"
        assert level == "none"


class TestCategorizedRisk:
    """Tests for CategorizedRisk frozen dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test creating CategorizedRisk with all fields."""
        from yolo_developer.agents.tea.risk import CategorizedRisk
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Missing unit tests for auth module",
            location="src/auth/handler.py",
            remediation="Add unit tests for authenticate() function",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="critical",
            impact_description="Critical test coverage gap may allow undetected bugs in production",
            requires_acknowledgment=False,
        )

        assert risk.risk_id == "R-F001"
        assert risk.finding == finding
        assert risk.risk_level == "critical"
        assert risk.impact_description == "Critical test coverage gap may allow undetected bugs in production"
        assert risk.requires_acknowledgment is False

    def test_create_high_risk_requires_acknowledgment(self) -> None:
        """Test that high risks set requires_acknowledgment=True."""
        from yolo_developer.agents.tea.risk import CategorizedRisk
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F002",
            category="code_quality",
            severity="high",
            description="Complex function needs refactoring",
            location="src/processor.py",
            remediation="Break down function into smaller units",
        )

        risk = CategorizedRisk(
            risk_id="R-F002",
            finding=finding,
            risk_level="high",
            impact_description="Code quality concern could impact maintainability",
            requires_acknowledgment=True,
        )

        assert risk.risk_level == "high"
        assert risk.requires_acknowledgment is True

    def test_create_low_risk(self) -> None:
        """Test creating a low-level risk."""
        from yolo_developer.agents.tea.risk import CategorizedRisk
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F003",
            category="documentation",
            severity="low",
            description="Missing docstring for helper function",
            location="src/utils.py",
            remediation="Add docstring explaining function purpose",
        )

        risk = CategorizedRisk(
            risk_id="R-F003",
            finding=finding,
            risk_level="low",
            impact_description="Documentation enhancement would improve clarity",
            requires_acknowledgment=False,
        )

        assert risk.risk_level == "low"
        assert risk.requires_acknowledgment is False

    def test_is_frozen_dataclass(self) -> None:
        """Test that CategorizedRisk is immutable (frozen)."""
        from yolo_developer.agents.tea.risk import CategorizedRisk
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Test issue",
            location="src/test.py",
            remediation="Fix it",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="critical",
            impact_description="Critical impact",
            requires_acknowledgment=False,
        )

        with pytest.raises(AttributeError):
            risk.risk_level = "low"  # type: ignore[misc]

    def test_to_dict_method(self) -> None:
        """Test to_dict() serialization method."""
        from yolo_developer.agents.tea.risk import CategorizedRisk
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="security",
            severity="high",
            description="Security concern",
            location="src/auth.py",
            remediation="Review security",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="high",
            impact_description="Security concern requires review before deployment",
            requires_acknowledgment=True,
        )

        result = risk.to_dict()

        assert result["risk_id"] == "R-F001"
        assert result["risk_level"] == "high"
        assert result["impact_description"] == "Security concern requires review before deployment"
        assert result["requires_acknowledgment"] is True
        assert "finding" in result
        assert result["finding"]["finding_id"] == "F001"


class TestRiskReport:
    """Tests for RiskReport frozen dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test creating RiskReport with all fields."""
        from yolo_developer.agents.tea.risk import CategorizedRisk, RiskReport
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Test issue",
            location="src/test.py",
            remediation="Fix it",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="critical",
            impact_description="Critical impact",
            requires_acknowledgment=False,
        )

        report = RiskReport(
            risks=(risk,),
            critical_count=1,
            high_count=0,
            low_count=0,
            overall_risk_level="critical",
            deployment_blocked=True,
            blocking_reasons=("Critical risk R-F001: Critical impact",),
            acknowledgment_required=(),
        )

        assert report.risks == (risk,)
        assert report.critical_count == 1
        assert report.high_count == 0
        assert report.low_count == 0
        assert report.overall_risk_level == "critical"
        assert report.deployment_blocked is True
        assert report.blocking_reasons == ("Critical risk R-F001: Critical impact",)
        assert report.acknowledgment_required == ()

    def test_create_with_mixed_risks(self) -> None:
        """Test creating RiskReport with multiple risk levels."""
        from yolo_developer.agents.tea.risk import CategorizedRisk, RiskReport
        from yolo_developer.agents.tea.types import Finding

        critical_finding = Finding(
            finding_id="F001",
            category="security",
            severity="critical",
            description="Critical security issue",
            location="src/auth.py",
            remediation="Fix immediately",
        )

        high_finding = Finding(
            finding_id="F002",
            category="code_quality",
            severity="high",
            description="Code quality issue",
            location="src/processor.py",
            remediation="Refactor code",
        )

        low_finding = Finding(
            finding_id="F003",
            category="documentation",
            severity="low",
            description="Missing docs",
            location="src/utils.py",
            remediation="Add docs",
        )

        critical_risk = CategorizedRisk(
            risk_id="R-F001",
            finding=critical_finding,
            risk_level="critical",
            impact_description="Critical impact",
            requires_acknowledgment=False,
        )

        high_risk = CategorizedRisk(
            risk_id="R-F002",
            finding=high_finding,
            risk_level="high",
            impact_description="High impact",
            requires_acknowledgment=True,
        )

        low_risk = CategorizedRisk(
            risk_id="R-F003",
            finding=low_finding,
            risk_level="low",
            impact_description="Low impact",
            requires_acknowledgment=False,
        )

        report = RiskReport(
            risks=(critical_risk, high_risk, low_risk),
            critical_count=1,
            high_count=1,
            low_count=1,
            overall_risk_level="critical",
            deployment_blocked=True,
            blocking_reasons=("Critical risk R-F001: Critical impact",),
            acknowledgment_required=("R-F002: High impact at src/processor.py",),
        )

        assert len(report.risks) == 3
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.low_count == 1
        assert report.overall_risk_level == "critical"
        assert report.deployment_blocked is True

    def test_create_empty_report(self) -> None:
        """Test creating RiskReport with no risks."""
        from yolo_developer.agents.tea.risk import RiskReport

        report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
            blocking_reasons=(),
            acknowledgment_required=(),
        )

        assert report.risks == ()
        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.low_count == 0
        assert report.overall_risk_level == "none"
        assert report.deployment_blocked is False

    def test_is_frozen_dataclass(self) -> None:
        """Test that RiskReport is immutable (frozen)."""
        from yolo_developer.agents.tea.risk import RiskReport

        report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        with pytest.raises(AttributeError):
            report.deployment_blocked = True  # type: ignore[misc]

    def test_to_dict_method(self) -> None:
        """Test to_dict() serialization method."""
        from yolo_developer.agents.tea.risk import CategorizedRisk, RiskReport
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="high",
            description="Test issue",
            location="src/test.py",
            remediation="Fix it",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="high",
            impact_description="High impact",
            requires_acknowledgment=True,
        )

        report = RiskReport(
            risks=(risk,),
            critical_count=0,
            high_count=1,
            low_count=0,
            overall_risk_level="high",
            deployment_blocked=False,
            blocking_reasons=(),
            acknowledgment_required=("R-F001: High impact at src/test.py",),
        )

        result = report.to_dict()

        assert result["critical_count"] == 0
        assert result["high_count"] == 1
        assert result["low_count"] == 0
        assert result["overall_risk_level"] == "high"
        assert result["deployment_blocked"] is False
        assert result["blocking_reasons"] == []
        assert result["acknowledgment_required"] == ["R-F001: High impact at src/test.py"]
        assert len(result["risks"]) == 1
        assert result["risks"][0]["risk_id"] == "R-F001"
        assert "created_at" in result

    def test_default_blocking_reasons_empty(self) -> None:
        """Test that blocking_reasons defaults to empty tuple."""
        from yolo_developer.agents.tea.risk import RiskReport

        report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        assert report.blocking_reasons == ()

    def test_default_acknowledgment_required_empty(self) -> None:
        """Test that acknowledgment_required defaults to empty tuple."""
        from yolo_developer.agents.tea.risk import RiskReport

        report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        assert report.acknowledgment_required == ()

    def test_created_at_has_default(self) -> None:
        """Test that created_at is automatically set."""
        from yolo_developer.agents.tea.risk import RiskReport

        report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        assert report.created_at is not None
        assert len(report.created_at) > 0
        # Should be ISO format
        assert "T" in report.created_at

    def test_only_high_risks_no_blocking(self) -> None:
        """Test that only high risks don't block deployment."""
        from yolo_developer.agents.tea.risk import CategorizedRisk, RiskReport
        from yolo_developer.agents.tea.types import Finding

        finding = Finding(
            finding_id="F001",
            category="code_quality",
            severity="high",
            description="High severity issue",
            location="src/code.py",
            remediation="Fix it",
        )

        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="high",
            impact_description="High impact",
            requires_acknowledgment=True,
        )

        report = RiskReport(
            risks=(risk,),
            critical_count=0,
            high_count=1,
            low_count=0,
            overall_risk_level="high",
            deployment_blocked=False,  # High doesn't block
            acknowledgment_required=("R-F001: High impact at src/code.py",),
        )

        assert report.deployment_blocked is False
        assert report.overall_risk_level == "high"
        assert len(report.acknowledgment_required) == 1
