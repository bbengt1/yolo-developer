"""Unit tests for risk categorization functions (Story 9.5 - Tasks 2-7, 10-12).

Tests for the risk categorization functions:
- _map_severity_to_risk_level: Map finding severity to risk level
- categorize_finding: Categorize a single finding into a risk
- categorize_risks: Categorize all findings from validation results
- generate_risk_report: Generate a complete risk report
- check_risk_deployment_blocking: Check if risks should block deployment
- get_acknowledgment_requirements: Get items requiring acknowledgment
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.tea.types import Finding, ValidationResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def critical_finding() -> Finding:
    """Create a critical severity finding."""
    return Finding(
        finding_id="F001",
        category="security",
        severity="critical",
        description="Critical security vulnerability",
        location="src/auth.py",
        remediation="Fix immediately",
    )


@pytest.fixture
def high_finding() -> Finding:
    """Create a high severity finding."""
    return Finding(
        finding_id="F002",
        category="code_quality",
        severity="high",
        description="High code quality issue",
        location="src/processor.py",
        remediation="Refactor code",
    )


@pytest.fixture
def medium_finding() -> Finding:
    """Create a medium severity finding."""
    return Finding(
        finding_id="F003",
        category="test_coverage",
        severity="medium",
        description="Medium test coverage issue",
        location="src/utils.py",
        remediation="Add tests",
    )


@pytest.fixture
def low_finding() -> Finding:
    """Create a low severity finding."""
    return Finding(
        finding_id="F004",
        category="documentation",
        severity="low",
        description="Missing docstring",
        location="src/helpers.py",
        remediation="Add docstring",
    )


@pytest.fixture
def info_finding() -> Finding:
    """Create an info severity finding."""
    return Finding(
        finding_id="F005",
        category="architecture",
        severity="info",
        description="Architecture note",
        location="src/config.py",
        remediation="Consider refactoring",
    )


# =============================================================================
# Task 2: Severity-to-Risk Mapping Tests
# =============================================================================


class TestSeverityToRiskMapping:
    """Tests for _map_severity_to_risk_level function (Task 2)."""

    def test_critical_severity_maps_to_critical_risk(self) -> None:
        """Test that critical severity maps to critical risk level."""
        from yolo_developer.agents.tea.risk import _map_severity_to_risk_level

        result = _map_severity_to_risk_level("critical")
        assert result == "critical"

    def test_high_severity_maps_to_high_risk(self) -> None:
        """Test that high severity maps to high risk level."""
        from yolo_developer.agents.tea.risk import _map_severity_to_risk_level

        result = _map_severity_to_risk_level("high")
        assert result == "high"

    def test_medium_severity_maps_to_low_risk(self) -> None:
        """Test that medium severity maps to low risk level."""
        from yolo_developer.agents.tea.risk import _map_severity_to_risk_level

        result = _map_severity_to_risk_level("medium")
        assert result == "low"

    def test_low_severity_maps_to_low_risk(self) -> None:
        """Test that low severity maps to low risk level."""
        from yolo_developer.agents.tea.risk import _map_severity_to_risk_level

        result = _map_severity_to_risk_level("low")
        assert result == "low"

    def test_info_severity_maps_to_low_risk(self) -> None:
        """Test that info severity maps to low risk level."""
        from yolo_developer.agents.tea.risk import _map_severity_to_risk_level

        result = _map_severity_to_risk_level("info")
        assert result == "low"


# =============================================================================
# Task 2b: Impact Description Tests (Code Review Fix M2)
# =============================================================================


class TestGetImpactDescription:
    """Tests for _get_impact_description function (Code Review Fix M2)."""

    def test_known_category_critical_level(self) -> None:
        """Test impact description for known category with critical level."""
        from yolo_developer.agents.tea.risk import _get_impact_description

        result = _get_impact_description("security", "critical")
        assert "security" in result.lower() or "vulnerability" in result.lower()
        assert "critical" in result.lower() or "immediately" in result.lower()

    def test_known_category_high_level(self) -> None:
        """Test impact description for known category with high level."""
        from yolo_developer.agents.tea.risk import _get_impact_description

        result = _get_impact_description("test_coverage", "high")
        assert len(result) > 0
        assert "coverage" in result.lower() or "test" in result.lower()

    def test_known_category_low_level(self) -> None:
        """Test impact description for known category with low level."""
        from yolo_developer.agents.tea.risk import _get_impact_description

        result = _get_impact_description("documentation", "low")
        assert len(result) > 0

    def test_unknown_category_fallback(self) -> None:
        """Test that unknown category falls back to code_quality templates."""
        from yolo_developer.agents.tea.risk import _get_impact_description

        # Use a category that doesn't exist in IMPACT_TEMPLATES
        result = _get_impact_description("unknown_category", "critical")  # type: ignore[arg-type]

        # Should fall back to code_quality critical template
        assert len(result) > 0
        # code_quality critical says "runtime failures"
        assert "runtime" in result.lower() or "quality" in result.lower()

    def test_all_categories_have_descriptions(self) -> None:
        """Test that all known categories return non-empty descriptions."""
        from yolo_developer.agents.tea.risk import _get_impact_description

        categories = ["test_coverage", "code_quality", "documentation", "security", "performance", "architecture"]
        risk_levels = ["critical", "high", "low"]

        for category in categories:
            for level in risk_levels:
                result = _get_impact_description(category, level)  # type: ignore[arg-type]
                assert len(result) > 0, f"Empty description for {category}/{level}"


# =============================================================================
# Task 3: Risk Categorization Logic Tests
# =============================================================================


class TestCategorizeFinding:
    """Tests for categorize_finding function (Task 3)."""

    def test_categorize_critical_finding(self, critical_finding: Finding) -> None:
        """Test categorizing a critical severity finding."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(critical_finding)

        assert risk.risk_id == "R-F001"
        assert risk.finding == critical_finding
        assert risk.risk_level == "critical"
        assert risk.requires_acknowledgment is False
        assert "critical" in risk.impact_description.lower() or "security" in risk.impact_description.lower()

    def test_categorize_high_finding_requires_acknowledgment(self, high_finding: Finding) -> None:
        """Test that high severity findings require acknowledgment."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(high_finding)

        assert risk.risk_id == "R-F002"
        assert risk.risk_level == "high"
        assert risk.requires_acknowledgment is True

    def test_categorize_medium_finding_as_low_risk(self, medium_finding: Finding) -> None:
        """Test that medium severity findings become low risk."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(medium_finding)

        assert risk.risk_id == "R-F003"
        assert risk.risk_level == "low"
        assert risk.requires_acknowledgment is False

    def test_categorize_low_finding_as_low_risk(self, low_finding: Finding) -> None:
        """Test that low severity findings become low risk."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(low_finding)

        assert risk.risk_id == "R-F004"
        assert risk.risk_level == "low"
        assert risk.requires_acknowledgment is False

    def test_categorize_info_finding_as_low_risk(self, info_finding: Finding) -> None:
        """Test that info severity findings become low risk."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(info_finding)

        assert risk.risk_id == "R-F005"
        assert risk.risk_level == "low"
        assert risk.requires_acknowledgment is False

    def test_risk_id_format(self, critical_finding: Finding) -> None:
        """Test that risk_id follows the R-{finding_id} format."""
        from yolo_developer.agents.tea.risk import categorize_finding

        risk = categorize_finding(critical_finding)

        assert risk.risk_id.startswith("R-")
        assert critical_finding.finding_id in risk.risk_id

    def test_impact_description_based_on_category(self) -> None:
        """Test that impact description is based on finding category."""
        from yolo_developer.agents.tea.risk import categorize_finding

        # Test different categories
        categories = ["test_coverage", "code_quality", "documentation", "security", "performance", "architecture"]

        for category in categories:
            finding = Finding(
                finding_id="F-TEST",
                category=category,  # type: ignore[arg-type]
                severity="high",
                description="Test finding",
                location="src/test.py",
                remediation="Fix it",
            )
            risk = categorize_finding(finding)
            # Impact description should not be empty
            assert len(risk.impact_description) > 0


# =============================================================================
# Task 4: Bulk Risk Categorization Tests
# =============================================================================


class TestCategorizeRisks:
    """Tests for categorize_risks function (Task 4)."""

    def test_categorize_risks_from_validation_results(
        self, critical_finding: Finding, high_finding: Finding
    ) -> None:
        """Test categorizing risks from validation results."""
        from yolo_developer.agents.tea.risk import categorize_risks

        validation_result = ValidationResult(
            artifact_id="src/auth.py",
            validation_status="failed",
            findings=(critical_finding, high_finding),
            score=50,
        )

        risks = categorize_risks((validation_result,))

        assert len(risks) == 2
        risk_ids = {r.risk_id for r in risks}
        assert "R-F001" in risk_ids
        assert "R-F002" in risk_ids

    def test_categorize_risks_from_multiple_validation_results(
        self, critical_finding: Finding, low_finding: Finding
    ) -> None:
        """Test categorizing risks from multiple validation results."""
        from yolo_developer.agents.tea.risk import categorize_risks

        result1 = ValidationResult(
            artifact_id="src/auth.py",
            validation_status="failed",
            findings=(critical_finding,),
            score=60,
        )

        result2 = ValidationResult(
            artifact_id="src/helpers.py",
            validation_status="warning",
            findings=(low_finding,),
            score=90,
        )

        risks = categorize_risks((result1, result2))

        assert len(risks) == 2

    def test_categorize_risks_empty_validation_results(self) -> None:
        """Test categorizing risks with empty validation results."""
        from yolo_developer.agents.tea.risk import categorize_risks

        risks = categorize_risks(())

        assert risks == ()

    def test_categorize_risks_no_findings(self) -> None:
        """Test categorizing risks when validation results have no findings."""
        from yolo_developer.agents.tea.risk import categorize_risks

        result = ValidationResult(
            artifact_id="src/clean.py",
            validation_status="passed",
            findings=(),
            score=100,
        )

        risks = categorize_risks((result,))

        assert risks == ()

    def test_categorize_risks_preserves_order(
        self, critical_finding: Finding, high_finding: Finding, low_finding: Finding
    ) -> None:
        """Test that categorize_risks preserves finding order."""
        from yolo_developer.agents.tea.risk import categorize_risks

        result = ValidationResult(
            artifact_id="src/test.py",
            validation_status="warning",
            findings=(critical_finding, high_finding, low_finding),
            score=70,
        )

        risks = categorize_risks((result,))

        assert len(risks) == 3
        assert risks[0].risk_id == "R-F001"
        assert risks[1].risk_id == "R-F002"
        assert risks[2].risk_id == "R-F004"


# =============================================================================
# Task 5: Risk Report Generation Tests
# =============================================================================


class TestGenerateRiskReport:
    """Tests for generate_risk_report function (Task 5)."""

    def test_generate_report_with_critical_risk(self, critical_finding: Finding) -> None:
        """Test generating report with critical risk."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        assert report.critical_count == 1
        assert report.high_count == 0
        assert report.low_count == 0
        assert report.overall_risk_level == "critical"
        assert report.deployment_blocked is True

    def test_generate_report_with_high_risk_only(self, high_finding: Finding) -> None:
        """Test generating report with only high risk (no critical)."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        assert report.critical_count == 0
        assert report.high_count == 1
        assert report.low_count == 0
        assert report.overall_risk_level == "high"
        assert report.deployment_blocked is False

    def test_generate_report_with_low_risk_only(self, low_finding: Finding) -> None:
        """Test generating report with only low risk."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risk = categorize_finding(low_finding)
        report = generate_risk_report((risk,))

        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.low_count == 1
        assert report.overall_risk_level == "low"
        assert report.deployment_blocked is False

    def test_generate_report_empty_risks(self) -> None:
        """Test generating report with no risks."""
        from yolo_developer.agents.tea.risk import generate_risk_report

        report = generate_risk_report(())

        assert report.critical_count == 0
        assert report.high_count == 0
        assert report.low_count == 0
        assert report.overall_risk_level == "none"
        assert report.deployment_blocked is False
        assert report.blocking_reasons == ()
        assert report.acknowledgment_required == ()

    def test_generate_report_mixed_risks(
        self, critical_finding: Finding, high_finding: Finding, low_finding: Finding
    ) -> None:
        """Test generating report with mixed risk levels."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risks = (
            categorize_finding(critical_finding),
            categorize_finding(high_finding),
            categorize_finding(low_finding),
        )
        report = generate_risk_report(risks)

        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.low_count == 1
        assert report.overall_risk_level == "critical"  # Highest level
        assert report.deployment_blocked is True  # Critical blocks

    def test_generate_report_overall_risk_level_hierarchy(
        self, high_finding: Finding, low_finding: Finding
    ) -> None:
        """Test that overall_risk_level reflects highest risk."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        # High and low - should be "high"
        risks = (
            categorize_finding(high_finding),
            categorize_finding(low_finding),
        )
        report = generate_risk_report(risks)

        assert report.overall_risk_level == "high"

    def test_generate_report_includes_blocking_reasons(self, critical_finding: Finding) -> None:
        """Test that blocking reasons are included when blocked."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        assert len(report.blocking_reasons) > 0
        assert "R-F001" in report.blocking_reasons[0] or "critical" in report.blocking_reasons[0].lower()

    def test_generate_report_includes_acknowledgment_required(self, high_finding: Finding) -> None:
        """Test that acknowledgment requirements are included for high risks."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        assert len(report.acknowledgment_required) > 0

    def test_generate_report_counts_accuracy(self) -> None:
        """Test that risk counts are accurate."""
        from yolo_developer.agents.tea.risk import categorize_finding, generate_risk_report

        # Create multiple findings of each severity
        findings = [
            Finding(finding_id=f"F-CRIT-{i}", category="security", severity="critical",
                    description="Critical", location="src/a.py", remediation="Fix")
            for i in range(2)
        ] + [
            Finding(finding_id=f"F-HIGH-{i}", category="code_quality", severity="high",
                    description="High", location="src/b.py", remediation="Fix")
            for i in range(3)
        ] + [
            Finding(finding_id=f"F-LOW-{i}", category="documentation", severity="low",
                    description="Low", location="src/c.py", remediation="Fix")
            for i in range(4)
        ]

        risks = tuple(categorize_finding(f) for f in findings)
        report = generate_risk_report(risks)

        assert report.critical_count == 2
        assert report.high_count == 3
        assert report.low_count == 4


# =============================================================================
# Task 6: Deployment Blocking Logic Tests
# =============================================================================


class TestCheckRiskDeploymentBlocking:
    """Tests for check_risk_deployment_blocking function (Task 6)."""

    def test_blocking_with_critical_risks(self, critical_finding: Finding) -> None:
        """Test that critical risks block deployment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
        )

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is True
        assert len(reasons) > 0

    def test_no_blocking_with_high_risks_only(self, high_finding: Finding) -> None:
        """Test that high risks alone don't block deployment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
        )

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is False
        assert len(reasons) == 0

    def test_no_blocking_with_low_risks_only(self, low_finding: Finding) -> None:
        """Test that low risks don't block deployment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
        )

        risk = categorize_finding(low_finding)
        report = generate_risk_report((risk,))

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is False
        assert len(reasons) == 0

    def test_no_blocking_with_empty_report(self) -> None:
        """Test that empty report doesn't block deployment."""
        from yolo_developer.agents.tea.risk import check_risk_deployment_blocking, generate_risk_report

        report = generate_risk_report(())

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is False
        assert len(reasons) == 0

    def test_blocking_reasons_include_critical_descriptions(self, critical_finding: Finding) -> None:
        """Test that blocking reasons include critical risk descriptions."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            check_risk_deployment_blocking,
            generate_risk_report,
        )

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        is_blocked, reasons = check_risk_deployment_blocking(report)

        assert is_blocked is True
        # Reason should contain risk ID or impact description
        reason_text = " ".join(reasons).lower()
        assert "r-f001" in reason_text or "critical" in reason_text


# =============================================================================
# Task 7: Acknowledgment Requirements Tests
# =============================================================================


class TestGetAcknowledgmentRequirements:
    """Tests for get_acknowledgment_requirements function (Task 7)."""

    def test_acknowledgment_for_high_risks(self, high_finding: Finding) -> None:
        """Test that high risks generate acknowledgment requirements."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        requirements = get_acknowledgment_requirements(report)

        assert len(requirements) > 0

    def test_no_acknowledgment_for_critical_risks(self, critical_finding: Finding) -> None:
        """Test that critical risks don't generate acknowledgment (they block instead)."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        risk = categorize_finding(critical_finding)
        report = generate_risk_report((risk,))

        requirements = get_acknowledgment_requirements(report)

        # Critical risks block, they don't require acknowledgment
        # The report should have blocking_reasons but not acknowledgment_required
        assert report.deployment_blocked is True
        # Acknowledgment is for high risks, not critical
        for req in requirements:
            assert "critical" not in req.lower() or "R-F001" not in req

    def test_no_acknowledgment_for_low_risks(self, low_finding: Finding) -> None:
        """Test that low risks don't require acknowledgment."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        risk = categorize_finding(low_finding)
        report = generate_risk_report((risk,))

        requirements = get_acknowledgment_requirements(report)

        assert len(requirements) == 0

    def test_empty_acknowledgment_for_no_high_risks(self) -> None:
        """Test empty acknowledgment when no high risks."""
        from yolo_developer.agents.tea.risk import generate_risk_report, get_acknowledgment_requirements

        report = generate_risk_report(())

        requirements = get_acknowledgment_requirements(report)

        assert requirements == ()

    def test_acknowledgment_format_includes_risk_id(self, high_finding: Finding) -> None:
        """Test that acknowledgment includes risk_id."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        requirements = get_acknowledgment_requirements(report)

        assert len(requirements) > 0
        # Should include risk_id
        assert any("R-F002" in req for req in requirements)

    def test_acknowledgment_format_includes_location(self, high_finding: Finding) -> None:
        """Test that acknowledgment includes location."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        risk = categorize_finding(high_finding)
        report = generate_risk_report((risk,))

        requirements = get_acknowledgment_requirements(report)

        assert len(requirements) > 0
        # Should include location
        assert any(high_finding.location in req for req in requirements)

    def test_multiple_high_risks_generate_multiple_requirements(self) -> None:
        """Test that multiple high risks generate multiple acknowledgment requirements."""
        from yolo_developer.agents.tea.risk import (
            categorize_finding,
            generate_risk_report,
            get_acknowledgment_requirements,
        )

        findings = [
            Finding(finding_id=f"F-HIGH-{i}", category="code_quality", severity="high",
                    description=f"High issue {i}", location=f"src/file{i}.py", remediation="Fix")
            for i in range(3)
        ]

        risks = tuple(categorize_finding(f) for f in findings)
        report = generate_risk_report(risks)

        requirements = get_acknowledgment_requirements(report)

        assert len(requirements) == 3
