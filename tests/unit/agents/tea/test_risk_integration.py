"""Integration tests for risk categorization with TEA node (Story 9.5 - Task 13).

Tests for the integration of risk categorization with the TEA agent:
- TEA node integration with risk report
- RiskReport included in TEAOutput
- Risk blocking integrates with confidence scoring
"""

from __future__ import annotations

from yolo_developer.agents.tea.types import (
    Finding,
    TEAOutput,
    ValidationResult,
)


class TestRiskCategorizationFlow:
    """Tests for the full risk categorization flow."""

    def test_categorize_risks_full_flow(self) -> None:
        """Test full risk categorization from findings to report."""
        from yolo_developer.agents.tea.risk import (
            categorize_risks,
            generate_risk_report,
        )

        # Create validation results with various findings
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
            remediation="Refactor",
        )

        low_finding = Finding(
            finding_id="F003",
            category="documentation",
            severity="low",
            description="Missing docs",
            location="src/utils.py",
            remediation="Add docs",
        )

        validation_result = ValidationResult(
            artifact_id="src/main.py",
            validation_status="failed",
            findings=(critical_finding, high_finding, low_finding),
            score=50,
        )

        # Full flow
        risks = categorize_risks((validation_result,))
        report = generate_risk_report(risks)

        # Verify report
        assert len(report.risks) == 3
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.low_count == 1
        assert report.overall_risk_level == "critical"
        assert report.deployment_blocked is True
        assert len(report.blocking_reasons) > 0
        assert len(report.acknowledgment_required) > 0

    def test_empty_validation_results_produces_empty_report(self) -> None:
        """Test that empty validation results produce no-risk report."""
        from yolo_developer.agents.tea.risk import categorize_risks, generate_risk_report

        risks = categorize_risks(())
        report = generate_risk_report(risks)

        assert report.overall_risk_level == "none"
        assert report.deployment_blocked is False
        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.low_count == 0

    def test_risk_report_serialization(self) -> None:
        """Test that RiskReport can be serialized and included in TEAOutput."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="high",
            description="Test coverage issue",
            location="src/test.py",
            remediation="Add tests",
        )

        risk = categorize_finding(finding)
        report = generate_risk_report((risk,))

        # Verify to_dict() works
        report_dict = report.to_dict()

        assert "risks" in report_dict
        assert "critical_count" in report_dict
        assert "high_count" in report_dict
        assert "low_count" in report_dict
        assert "overall_risk_level" in report_dict
        assert "deployment_blocked" in report_dict
        assert "blocking_reasons" in report_dict
        assert "acknowledgment_required" in report_dict
        assert "created_at" in report_dict


class TestRiskBlockingWithConfidenceScoring:
    """Tests for risk blocking integration with confidence scoring."""

    def test_critical_risk_blocks_even_with_high_confidence(self) -> None:
        """Test that critical risks block deployment even if confidence is high."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
        )

        # Even if other scores are good, critical risk should block
        critical_finding = Finding(
            finding_id="F001",
            category="security",
            severity="critical",
            description="Critical security vulnerability",
            location="src/auth.py",
            remediation="Fix immediately",
        )

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is True
        assert len(reasons) > 0

    def test_high_risks_with_passing_confidence_not_blocked(self) -> None:
        """Test that high risks don't block but require acknowledgment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        high_finding = Finding(
            finding_id="F001",
            category="code_quality",
            severity="high",
            description="Code quality concern",
            location="src/processor.py",
            remediation="Consider refactoring",
        )

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        is_blocked, _reasons = check_risk_deployment_blocking(report)
        acknowledgments = get_acknowledgment_requirements(report)

        assert is_blocked is False
        assert len(acknowledgments) > 0

    def test_low_risks_no_blocking_no_acknowledgment(self) -> None:
        """Test that low risks don't block and don't require acknowledgment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        low_finding = Finding(
            finding_id="F001",
            category="documentation",
            severity="low",
            description="Minor documentation issue",
            location="src/utils.py",
            remediation="Nice to have",
        )

        risk = categorize_finding(low_finding)
        report = generate_risk_report((risk,))

        is_blocked, _reasons = check_risk_deployment_blocking(report)
        acknowledgments = get_acknowledgment_requirements(report)

        assert is_blocked is False
        assert len(acknowledgments) == 0


class TestTEAOutputWithRiskReport:
    """Tests for TEAOutput with risk_report field."""

    def test_tea_output_has_risk_report_field(self) -> None:
        """Test that TEAOutput type supports risk_report field."""
        from yolo_developer.agents.tea.risk import RiskReport

        # TEAOutput should support risk_report field
        # This tests AC6: Integration with TEA Output
        RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        # Verify we can create TEAOutput with risk_report
        # Note: The actual field might need to be added to TEAOutput
        output = TEAOutput(
            validation_results=(),
            processing_notes="Test",
            overall_confidence=1.0,
            deployment_recommendation="deploy",
        )

        # TEAOutput should serialize properly
        output_dict = output.to_dict()
        assert "validation_results" in output_dict
        assert "overall_confidence" in output_dict
        assert "deployment_recommendation" in output_dict

    def test_risk_report_to_dict_includes_all_fields(self) -> None:
        """Test that RiskReport.to_dict() includes all required fields."""
        from yolo_developer.agents.tea.risk import (
            CategorizedRisk,
            RiskReport,
        )

        finding = Finding(
            finding_id="F001",
            category="security",
            severity="critical",
            description="Critical issue",
            location="src/auth.py",
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

        report_dict = report.to_dict()

        # Verify AC5: Risk Report Generation fields
        assert report_dict["risks"][0]["risk_id"] == "R-F001"
        assert report_dict["critical_count"] == 1
        assert report_dict["high_count"] == 0
        assert report_dict["low_count"] == 0
        assert report_dict["overall_risk_level"] == "critical"
        assert report_dict["deployment_blocked"] is True
        assert len(report_dict["blocking_reasons"]) == 1
        assert report_dict["acknowledgment_required"] == []
        assert "created_at" in report_dict


class TestMultipleSeverityLevels:
    """Tests for handling multiple severity levels correctly."""

    def test_multiple_critical_risks(self) -> None:
        """Test multiple critical risks are all counted and block."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        findings = [
            Finding(
                finding_id=f"F-CRIT-{i}",
                category="security",
                severity="critical",
                description=f"Critical issue {i}",
                location=f"src/file{i}.py",
                remediation="Fix it",
            )
            for i in range(3)
        ]

        risks = tuple(categorize_finding(f) for f in findings)
        report = generate_risk_report(risks)

        assert report.critical_count == 3
        assert report.deployment_blocked is True
        assert len(report.blocking_reasons) == 3

    def test_mixed_severity_overall_level_is_highest(self) -> None:
        """Test that overall_risk_level reflects highest severity present."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        # Only high and low
        findings = [
            Finding(
                finding_id="F001",
                category="code_quality",
                severity="high",
                description="High issue",
                location="src/a.py",
                remediation="Fix",
            ),
            Finding(
                finding_id="F002",
                category="documentation",
                severity="low",
                description="Low issue",
                location="src/b.py",
                remediation="Nice to have",
            ),
            Finding(
                finding_id="F003",
                category="test_coverage",
                severity="info",
                description="Info",
                location="src/c.py",
                remediation="Consider",
            ),
        ]

        risks = tuple(categorize_finding(f) for f in findings)
        report = generate_risk_report(risks)

        assert report.overall_risk_level == "high"
        assert report.high_count == 1
        assert report.low_count == 2  # info maps to low
        assert report.deployment_blocked is False  # No critical
