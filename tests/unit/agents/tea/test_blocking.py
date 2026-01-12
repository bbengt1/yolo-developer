"""Unit tests for deployment blocking types and functions (Story 9.7).

Tests for:
- BlockingReason: Type creation and serialization
- RemediationStep: Type creation and serialization
- DeploymentDecision: Type creation and serialization
- DeploymentOverride: Type creation and serialization
- DeploymentDecisionReport: Type creation and serialization
- Immutability (frozen dataclass)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class TestBlockingReasonType:
    """Tests for BlockingReason dataclass (AC: 1, 2)."""

    def test_blocking_reason_creation(self) -> None:
        """BlockingReason can be created with all required fields."""
        from yolo_developer.agents.tea.blocking import BlockingReason

        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Confidence score 75 is below deployment threshold 90",
            threshold_value=90,
            actual_value=75,
            related_findings=("F-CONFIDENCE-75",),
        )

        assert reason.reason_id == "BR-LOW-001"
        assert reason.reason_type == "low_confidence"
        assert reason.description == "Confidence score 75 is below deployment threshold 90"
        assert reason.threshold_value == 90
        assert reason.actual_value == 75
        assert reason.related_findings == ("F-CONFIDENCE-75",)

    def test_blocking_reason_optional_fields(self) -> None:
        """BlockingReason handles optional fields correctly."""
        from yolo_developer.agents.tea.blocking import BlockingReason

        reason = BlockingReason(
            reason_id="BR-CRI-001",
            reason_type="critical_risk",
            description="Critical risk finding(s) present",
        )

        assert reason.threshold_value is None
        assert reason.actual_value is None
        assert reason.related_findings == ()

    def test_blocking_reason_to_dict(self) -> None:
        """BlockingReason.to_dict() produces correct dictionary."""
        from yolo_developer.agents.tea.blocking import BlockingReason

        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Test description",
            threshold_value=90,
            actual_value=75,
            related_findings=("F001", "F002"),
        )

        result = reason.to_dict()

        assert result["reason_id"] == "BR-LOW-001"
        assert result["reason_type"] == "low_confidence"
        assert result["description"] == "Test description"
        assert result["threshold_value"] == 90
        assert result["actual_value"] == 75
        assert result["related_findings"] == ["F001", "F002"]  # List in output

    def test_blocking_reason_immutable(self) -> None:
        """BlockingReason is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.blocking import BlockingReason

        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Test",
        )

        with pytest.raises(FrozenInstanceError):
            reason.reason_id = "BR-LOW-002"  # type: ignore[misc]


class TestRemediationStepType:
    """Tests for RemediationStep dataclass (AC: 3)."""

    def test_remediation_step_creation(self) -> None:
        """RemediationStep can be created with all required fields."""
        from yolo_developer.agents.tea.blocking import RemediationStep

        step = RemediationStep(
            step_id="RS-001",
            priority=1,
            action="Address critical finding in src/auth/handler.py",
            expected_impact="Resolving critical findings removes deployment blocker",
            related_reason_id="BR-CRI-001",
        )

        assert step.step_id == "RS-001"
        assert step.priority == 1
        assert step.action == "Address critical finding in src/auth/handler.py"
        assert step.expected_impact == "Resolving critical findings removes deployment blocker"
        assert step.related_reason_id == "BR-CRI-001"

    def test_remediation_step_to_dict(self) -> None:
        """RemediationStep.to_dict() produces correct dictionary."""
        from yolo_developer.agents.tea.blocking import RemediationStep

        step = RemediationStep(
            step_id="RS-002",
            priority=2,
            action="Increase test coverage",
            expected_impact="Higher coverage increases confidence",
            related_reason_id="BR-LOW-001",
        )

        result = step.to_dict()

        assert result["step_id"] == "RS-002"
        assert result["priority"] == 2
        assert result["action"] == "Increase test coverage"
        assert result["expected_impact"] == "Higher coverage increases confidence"
        assert result["related_reason_id"] == "BR-LOW-001"

    def test_remediation_step_immutable(self) -> None:
        """RemediationStep is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.blocking import RemediationStep

        step = RemediationStep(
            step_id="RS-001",
            priority=1,
            action="Test",
            expected_impact="Test",
            related_reason_id="BR-001",
        )

        with pytest.raises(FrozenInstanceError):
            step.priority = 2  # type: ignore[misc]


class TestDeploymentDecisionType:
    """Tests for DeploymentDecision dataclass (AC: 1)."""

    def test_deployment_decision_blocked(self) -> None:
        """DeploymentDecision can represent blocked state."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision

        decision = DeploymentDecision(
            is_blocked=True,
            recommendation="block",
        )

        assert decision.is_blocked is True
        assert decision.recommendation == "block"
        assert decision.evaluated_at is not None

    def test_deployment_decision_allowed(self) -> None:
        """DeploymentDecision can represent allowed state."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision

        decision = DeploymentDecision(
            is_blocked=False,
            recommendation="deploy",
        )

        assert decision.is_blocked is False
        assert decision.recommendation == "deploy"

    def test_deployment_decision_with_warnings(self) -> None:
        """DeploymentDecision can represent deploy_with_warnings state."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision

        decision = DeploymentDecision(
            is_blocked=False,
            recommendation="deploy_with_warnings",
        )

        assert decision.is_blocked is False
        assert decision.recommendation == "deploy_with_warnings"

    def test_deployment_decision_to_dict(self) -> None:
        """DeploymentDecision.to_dict() produces correct dictionary."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision

        decision = DeploymentDecision(
            is_blocked=True,
            recommendation="block",
        )

        result = decision.to_dict()

        assert result["is_blocked"] is True
        assert result["recommendation"] == "block"
        assert "evaluated_at" in result

    def test_deployment_decision_immutable(self) -> None:
        """DeploymentDecision is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision

        decision = DeploymentDecision(
            is_blocked=True,
            recommendation="block",
        )

        with pytest.raises(FrozenInstanceError):
            decision.is_blocked = False  # type: ignore[misc]


class TestDeploymentOverrideType:
    """Tests for DeploymentOverride dataclass (AC: 4)."""

    def test_deployment_override_creation(self) -> None:
        """DeploymentOverride can be created with all required fields."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Critical hotfix for production outage",
            acknowledged_risks=(
                "Confidence score 75 is below threshold 90",
                "2 critical risk findings present",
            ),
        )

        assert override.acknowledged_by == "user@example.com"
        assert override.acknowledged_at == "2026-01-12T10:30:00Z"
        assert override.acknowledgment_reason == "Critical hotfix for production outage"
        assert len(override.acknowledged_risks) == 2

    def test_deployment_override_to_dict(self) -> None:
        """DeploymentOverride.to_dict() produces correct dictionary."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test reason",
            acknowledged_risks=("Risk 1", "Risk 2"),
        )

        result = override.to_dict()

        assert result["acknowledged_by"] == "user@example.com"
        assert result["acknowledged_at"] == "2026-01-12T10:30:00Z"
        assert result["acknowledgment_reason"] == "Test reason"
        assert result["acknowledged_risks"] == ["Risk 1", "Risk 2"]  # List in output

    def test_deployment_override_immutable(self) -> None:
        """DeploymentOverride is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test",
            acknowledged_risks=(),
        )

        with pytest.raises(FrozenInstanceError):
            override.acknowledged_by = "other@example.com"  # type: ignore[misc]


class TestDeploymentDecisionReportType:
    """Tests for DeploymentDecisionReport dataclass (AC: 5)."""

    def test_deployment_decision_report_blocked(self) -> None:
        """DeploymentDecisionReport can represent blocked state with reasons."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            DeploymentDecision,
            DeploymentDecisionReport,
            RemediationStep,
        )

        decision = DeploymentDecision(is_blocked=True, recommendation="block")
        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Low confidence",
        )
        step = RemediationStep(
            step_id="RS-001",
            priority=1,
            action="Increase coverage",
            expected_impact="Improves score",
            related_reason_id="BR-LOW-001",
        )

        report = DeploymentDecisionReport(
            decision=decision,
            blocking_reasons=(reason,),
            remediation_steps=(step,),
        )

        assert report.decision.is_blocked is True
        assert len(report.blocking_reasons) == 1
        assert len(report.remediation_steps) == 1
        assert report.override is None

    def test_deployment_decision_report_allowed(self) -> None:
        """DeploymentDecisionReport can represent allowed state."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision, DeploymentDecisionReport

        decision = DeploymentDecision(is_blocked=False, recommendation="deploy")

        report = DeploymentDecisionReport(
            decision=decision,
        )

        assert report.decision.is_blocked is False
        assert report.blocking_reasons == ()
        assert report.remediation_steps == ()

    def test_deployment_decision_report_with_override(self) -> None:
        """DeploymentDecisionReport can include override."""
        from yolo_developer.agents.tea.blocking import (
            DeploymentDecision,
            DeploymentDecisionReport,
            DeploymentOverride,
        )

        decision = DeploymentDecision(is_blocked=True, recommendation="block")
        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Hotfix",
            acknowledged_risks=("Risk 1",),
        )

        report = DeploymentDecisionReport(
            decision=decision,
            override=override,
        )

        assert report.override is not None
        assert report.override.acknowledged_by == "user@example.com"

    def test_deployment_decision_report_to_dict(self) -> None:
        """DeploymentDecisionReport.to_dict() produces correct dictionary."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            DeploymentDecision,
            DeploymentDecisionReport,
            RemediationStep,
        )

        decision = DeploymentDecision(is_blocked=True, recommendation="block")
        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Low confidence",
        )
        step = RemediationStep(
            step_id="RS-001",
            priority=1,
            action="Fix it",
            expected_impact="Better",
            related_reason_id="BR-LOW-001",
        )

        report = DeploymentDecisionReport(
            decision=decision,
            blocking_reasons=(reason,),
            remediation_steps=(step,),
        )

        result = report.to_dict()

        assert result["decision"]["is_blocked"] is True
        assert len(result["blocking_reasons"]) == 1
        assert len(result["remediation_steps"]) == 1
        assert result["override"] is None
        assert "created_at" in result

    def test_deployment_decision_report_immutable(self) -> None:
        """DeploymentDecisionReport is immutable (frozen dataclass)."""
        from yolo_developer.agents.tea.blocking import DeploymentDecision, DeploymentDecisionReport

        decision = DeploymentDecision(is_blocked=False, recommendation="deploy")
        report = DeploymentDecisionReport(decision=decision)

        with pytest.raises(FrozenInstanceError):
            report.decision = DeploymentDecision(is_blocked=True, recommendation="block")  # type: ignore[misc]


class TestBlockingReasonTypeLiteral:
    """Tests for BlockingReasonType literal type."""

    def test_valid_blocking_reason_types(self) -> None:
        """BlockingReasonType accepts valid values."""
        from yolo_developer.agents.tea.blocking import BlockingReason

        # Test all valid types
        valid_types = ["low_confidence", "critical_risk", "validation_failed", "high_risk_count"]

        for reason_type in valid_types:
            reason = BlockingReason(
                reason_id="BR-001",
                reason_type=reason_type,  # type: ignore[arg-type]
                description="Test",
            )
            assert reason.reason_type == reason_type


# =============================================================================
# Tests for Blocking Reason Generation (Task 10, AC: 2)
# =============================================================================


class TestBlockingReasonGeneration:
    """Tests for blocking reason generation functions."""

    def test_generate_reason_id_format(self) -> None:
        """Reason ID generation follows format BR-XXX-NNN."""
        from yolo_developer.agents.tea.blocking import _generate_reason_id

        result = _generate_reason_id("low_confidence", 1)
        assert result == "BR-LOW-001"

        result = _generate_reason_id("critical_risk", 2)
        assert result == "BR-CRI-002"

        result = _generate_reason_id("validation_failed", 10)
        assert result == "BR-VAL-010"

    def test_generate_low_confidence_reason(self) -> None:
        """Low confidence reason is generated correctly."""
        from yolo_developer.agents.tea.blocking import _generate_low_confidence_reason

        reason = _generate_low_confidence_reason(score=75, threshold=90)

        assert reason.reason_type == "low_confidence"
        assert reason.threshold_value == 90
        assert reason.actual_value == 75
        assert "75" in reason.description
        assert "90" in reason.description

    def test_generate_critical_risk_reason(self) -> None:
        """Critical risk reason is generated correctly."""
        from yolo_developer.agents.tea.blocking import _generate_critical_risk_reason
        from yolo_developer.agents.tea.risk import CategorizedRisk, RiskReport
        from yolo_developer.agents.tea.types import Finding

        # Create a mock risk report with critical findings
        finding = Finding(
            finding_id="F001",
            category="security",
            severity="critical",
            description="SQL injection",
            location="src/db.py",
            remediation="Use parameterized queries",
        )
        risk = CategorizedRisk(
            risk_id="R-F001",
            finding=finding,
            risk_level="critical",
            impact_description="Critical security vulnerability",
            requires_acknowledgment=False,
        )
        risk_report = RiskReport(
            risks=(risk,),
            critical_count=1,
            high_count=0,
            low_count=0,
            overall_risk_level="critical",
            deployment_blocked=True,
        )

        reason = _generate_critical_risk_reason(risk_report)

        assert reason.reason_type == "critical_risk"
        assert reason.actual_value == 1
        assert "R-F001" in reason.related_findings

    def test_generate_validation_failed_reason(self) -> None:
        """Validation failed reason is generated correctly."""
        from yolo_developer.agents.tea.blocking import _generate_validation_failed_reason
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Missing tests",
            location="src/auth.py",
            remediation="Add tests",
        )
        result = ValidationResult(
            artifact_id="src/auth.py",
            validation_status="failed",
            findings=(finding,),
        )

        reason = _generate_validation_failed_reason((result,))

        assert reason.reason_type == "validation_failed"
        assert reason.actual_value == 1
        assert "src/auth.py" in reason.description
        assert "F001" in reason.related_findings

    def test_generate_high_risk_count_reason_below_threshold(self) -> None:
        """High risk count reason returns None when below threshold."""
        from yolo_developer.agents.tea.blocking import _generate_high_risk_count_reason
        from yolo_developer.agents.tea.risk import RiskReport

        # 5 high risks is at threshold, not above
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=5,
            low_count=0,
            overall_risk_level="high",
            deployment_blocked=False,
        )

        reason = _generate_high_risk_count_reason(risk_report)
        assert reason is None

    def test_generate_high_risk_count_reason_above_threshold(self) -> None:
        """High risk count reason returns reason when above threshold."""
        from yolo_developer.agents.tea.blocking import _generate_high_risk_count_reason
        from yolo_developer.agents.tea.risk import RiskReport

        # 6 high risks is above threshold
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=6,
            low_count=0,
            overall_risk_level="high",
            deployment_blocked=False,
        )

        reason = _generate_high_risk_count_reason(risk_report)

        assert reason is not None
        assert reason.reason_type == "high_risk_count"
        assert reason.actual_value == 6
        assert reason.threshold_value == 5

    def test_generate_blocking_reasons_combines_all(self) -> None:
        """generate_blocking_reasons combines multiple reason types."""
        from yolo_developer.agents.tea.blocking import generate_blocking_reasons
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )
        from yolo_developer.agents.tea.types import Finding, ValidationResult

        # Create low confidence result
        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            base_score=50.0,
            final_score=50,
        )
        confidence_result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            deployment_recommendation="block",
        )

        # Create critical risk report
        risk_report = RiskReport(
            risks=(),
            critical_count=1,
            high_count=0,
            low_count=0,
            overall_risk_level="critical",
            deployment_blocked=True,
        )

        # Create failed validation
        finding = Finding(
            finding_id="F001",
            category="test_coverage",
            severity="critical",
            description="Test",
            location="test.py",
            remediation="Fix",
        )
        validation_results = (
            ValidationResult(
                artifact_id="test.py",
                validation_status="failed",
                findings=(finding,),
            ),
        )

        reasons = generate_blocking_reasons(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=validation_results,
        )

        # Should have 3 reasons: low_confidence, critical_risk, validation_failed
        assert len(reasons) == 3
        reason_types = {r.reason_type for r in reasons}
        assert "low_confidence" in reason_types
        assert "critical_risk" in reason_types
        assert "validation_failed" in reason_types


# =============================================================================
# Tests for Remediation Step Generation (Task 11, AC: 3)
# =============================================================================


class TestRemediationStepGeneration:
    """Tests for remediation step generation functions."""

    def test_generate_step_id_format(self) -> None:
        """Step ID generation follows format RS-NNN."""
        from yolo_developer.agents.tea.blocking import _generate_step_id

        assert _generate_step_id(1) == "RS-001"
        assert _generate_step_id(10) == "RS-010"
        assert _generate_step_id(100) == "RS-100"

    def test_generate_confidence_remediation(self) -> None:
        """Confidence remediation generates appropriate steps."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            _generate_confidence_remediation,
        )

        reason = BlockingReason(
            reason_id="BR-LOW-001",
            reason_type="low_confidence",
            description="Low confidence",
            threshold_value=90,
            actual_value=75,
        )

        steps = _generate_confidence_remediation(reason)

        assert len(steps) == 3  # Based on REMEDIATION_TEMPLATES
        assert all(s.related_reason_id == "BR-LOW-001" for s in steps)
        assert any("coverage" in s.action.lower() for s in steps)

    def test_generate_risk_remediation(self) -> None:
        """Risk remediation generates appropriate steps."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            _generate_risk_remediation,
        )

        reason = BlockingReason(
            reason_id="BR-CRI-001",
            reason_type="critical_risk",
            description="Critical risk",
        )

        steps = _generate_risk_remediation(reason)

        assert len(steps) == 3  # Based on REMEDIATION_TEMPLATES
        assert all(s.related_reason_id == "BR-CRI-001" for s in steps)

    def test_generate_validation_remediation(self) -> None:
        """Validation remediation generates appropriate steps."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            _generate_validation_remediation,
        )
        from yolo_developer.agents.tea.types import ValidationResult

        reason = BlockingReason(
            reason_id="BR-VAL-001",
            reason_type="validation_failed",
            description="Validation failed",
        )

        validation_results = (
            ValidationResult(
                artifact_id="src/auth.py",
                validation_status="failed",
            ),
        )

        steps = _generate_validation_remediation(reason, validation_results)

        # Should have specific step for artifact + general templates
        assert len(steps) >= 4
        assert any("src/auth.py" in s.action for s in steps)

    def test_generate_remediation_steps_sorts_by_priority(self) -> None:
        """generate_remediation_steps sorts steps by priority."""
        from yolo_developer.agents.tea.blocking import (
            BlockingReason,
            generate_remediation_steps,
        )

        reasons = (
            BlockingReason(
                reason_id="BR-LOW-001",
                reason_type="low_confidence",
                description="Low confidence",
            ),
            BlockingReason(
                reason_id="BR-CRI-001",
                reason_type="critical_risk",
                description="Critical risk",
            ),
        )

        steps = generate_remediation_steps(reasons, ())

        # Steps should be sorted by priority (1, 2, 3, ...)
        priorities = [s.priority for s in steps]
        assert priorities == sorted(priorities)
        assert steps[0].priority == 1


# =============================================================================
# Tests for Deployment Decision Evaluation (Task 12, AC: 1)
# =============================================================================


class TestDeploymentDecisionEvaluation:
    """Tests for deployment decision evaluation functions."""

    def test_deployment_blocked_low_confidence(self) -> None:
        """Deployment is blocked for low confidence score."""
        from yolo_developer.agents.tea.blocking import evaluate_deployment_decision
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            base_score=50.0,
            final_score=50,
        )
        confidence_result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            deployment_recommendation="block",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        decision = evaluate_deployment_decision(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
        )

        assert decision.is_blocked is True
        assert decision.recommendation == "block"

    def test_deployment_blocked_critical_risks(self) -> None:
        """Deployment is blocked for critical risks."""
        from yolo_developer.agents.tea.blocking import evaluate_deployment_decision
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )
        confidence_result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=1,
            high_count=0,
            low_count=0,
            overall_risk_level="critical",
            deployment_blocked=True,
        )

        decision = evaluate_deployment_decision(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
        )

        assert decision.is_blocked is True
        assert decision.recommendation == "block"

    def test_deployment_blocked_failed_validations(self) -> None:
        """Deployment is blocked for failed validations."""
        from yolo_developer.agents.tea.blocking import evaluate_deployment_decision
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )
        from yolo_developer.agents.tea.types import ValidationResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )
        confidence_result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )
        validation_results = (
            ValidationResult(
                artifact_id="test.py",
                validation_status="failed",
            ),
        )

        decision = evaluate_deployment_decision(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=validation_results,
        )

        assert decision.is_blocked is True
        assert decision.recommendation == "block"

    def test_deployment_allowed_all_pass(self) -> None:
        """Deployment is allowed when all checks pass."""
        from yolo_developer.agents.tea.blocking import evaluate_deployment_decision
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )
        from yolo_developer.agents.tea.types import ValidationResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )
        confidence_result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )
        validation_results = (
            ValidationResult(
                artifact_id="test.py",
                validation_status="passed",
            ),
        )

        decision = evaluate_deployment_decision(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=validation_results,
        )

        assert decision.is_blocked is False
        assert decision.recommendation == "deploy"

    def test_deploy_with_warnings_borderline(self) -> None:
        """deploy_with_warnings for borderline cases (warnings or high risks)."""
        from yolo_developer.agents.tea.blocking import evaluate_deployment_decision
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )
        from yolo_developer.agents.tea.types import ValidationResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )
        confidence_result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=3,  # Has high risks but not blocking
            low_count=0,
            overall_risk_level="high",
            deployment_blocked=False,
        )
        validation_results = (
            ValidationResult(
                artifact_id="test.py",
                validation_status="warning",
            ),
        )

        decision = evaluate_deployment_decision(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=validation_results,
        )

        assert decision.is_blocked is False
        assert decision.recommendation == "deploy_with_warnings"


# =============================================================================
# Tests for Override Handling (Task 13, AC: 4)
# =============================================================================


class TestOverrideHandling:
    """Tests for override handling functions."""

    def test_create_override(self) -> None:
        """create_override creates valid override."""
        from yolo_developer.agents.tea.blocking import BlockingReason, create_override

        reasons = (
            BlockingReason(
                reason_id="BR-LOW-001",
                reason_type="low_confidence",
                description="Low confidence score",
            ),
        )

        override = create_override(
            acknowledged_by="user@example.com",
            acknowledgment_reason="Critical hotfix",
            blocking_reasons=reasons,
        )

        assert override.acknowledged_by == "user@example.com"
        assert override.acknowledgment_reason == "Critical hotfix"
        assert "Low confidence score" in override.acknowledged_risks

    def test_validate_override_valid(self) -> None:
        """validate_override returns True for valid override."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test reason",
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is True

    def test_validate_override_empty_acknowledged_by(self) -> None:
        """validate_override returns False for empty acknowledged_by."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test reason",
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is False

    def test_validate_override_empty_reason(self) -> None:
        """validate_override returns False for empty acknowledgment_reason."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="",
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is False

    def test_validate_override_empty_risks(self) -> None:
        """validate_override returns False for empty acknowledged_risks."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test reason",
            acknowledged_risks=(),
        )

        assert validate_override(override) is False

    def test_validate_override_whitespace_only_acknowledged_at(self) -> None:
        """validate_override returns False for whitespace-only acknowledged_at."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="   ",  # Whitespace only
            acknowledgment_reason="Test reason",
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is False

    def test_validate_override_whitespace_only_acknowledged_by(self) -> None:
        """validate_override returns False for whitespace-only acknowledged_by."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="   ",  # Whitespace only
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Test reason",
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is False

    def test_validate_override_whitespace_only_reason(self) -> None:
        """validate_override returns False for whitespace-only acknowledgment_reason."""
        from yolo_developer.agents.tea.blocking import DeploymentOverride, validate_override

        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="   ",  # Whitespace only
            acknowledged_risks=("Risk 1",),
        )

        assert validate_override(override) is False


# =============================================================================
# Tests for Report Generation (Task 14, AC: 5)
# =============================================================================


class TestReportGeneration:
    """Tests for report generation functions."""

    def test_generate_report_when_blocked(self) -> None:
        """generate_deployment_decision_report generates complete report when blocked."""
        from yolo_developer.agents.tea.blocking import generate_deployment_decision_report
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            base_score=50.0,
            final_score=50,
        )
        confidence_result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            deployment_recommendation="block",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        report = generate_deployment_decision_report(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
        )

        assert report.decision.is_blocked is True
        assert len(report.blocking_reasons) >= 1
        assert len(report.remediation_steps) >= 1

    def test_generate_report_when_not_blocked(self) -> None:
        """generate_deployment_decision_report generates empty reasons when not blocked."""
        from yolo_developer.agents.tea.blocking import generate_deployment_decision_report
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            base_score=100.0,
            final_score=100,
        )
        confidence_result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )

        report = generate_deployment_decision_report(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
        )

        assert report.decision.is_blocked is False
        assert len(report.blocking_reasons) == 0
        assert len(report.remediation_steps) == 0

    def test_generate_report_with_override(self) -> None:
        """generate_deployment_decision_report includes override."""
        from yolo_developer.agents.tea.blocking import (
            DeploymentOverride,
            generate_deployment_decision_report,
        )
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            base_score=50.0,
            final_score=50,
        )
        confidence_result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            deployment_recommendation="block",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=0,
            high_count=0,
            low_count=0,
            overall_risk_level="none",
            deployment_blocked=False,
        )
        override = DeploymentOverride(
            acknowledged_by="user@example.com",
            acknowledged_at="2026-01-12T10:30:00Z",
            acknowledgment_reason="Hotfix",
            acknowledged_risks=("Risk 1",),
        )

        report = generate_deployment_decision_report(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
            override=override,
        )

        assert report.override is not None
        assert report.override.acknowledged_by == "user@example.com"


# =============================================================================
# Tests for Configuration Integration (Task 7, AC: 7)
# =============================================================================


class TestConfigurationIntegration:
    """Tests for configuration integration functions."""

    def test_get_deployment_threshold_default(self) -> None:
        """get_deployment_threshold returns 90 as default."""
        from yolo_developer.agents.tea.blocking import get_deployment_threshold

        # Without proper config, should return default 90
        threshold = get_deployment_threshold()
        assert threshold == 90

    def test_is_deployment_blocking_enabled_default(self) -> None:
        """is_deployment_blocking_enabled returns True by default."""
        from yolo_developer.agents.tea.blocking import is_deployment_blocking_enabled

        assert is_deployment_blocking_enabled() is True

    def test_get_high_risk_count_threshold_default(self) -> None:
        """get_high_risk_count_threshold returns 5 as default."""
        from yolo_developer.agents.tea.blocking import get_high_risk_count_threshold

        assert get_high_risk_count_threshold() == 5


class TestBlockingReasonGenerationEdgeCases:
    """Additional edge case tests for blocking reason generation."""

    def test_generate_blocking_reasons_empty_validation_results(self) -> None:
        """generate_blocking_reasons handles empty validation_results with other blockers."""
        from yolo_developer.agents.tea.blocking import generate_blocking_reasons
        from yolo_developer.agents.tea.risk import RiskReport
        from yolo_developer.agents.tea.scoring import (
            ConfidenceBreakdown,
            ConfidenceResult,
        )

        # Low confidence + critical risk, but no validation results
        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            base_score=50.0,
            final_score=50,
        )
        confidence_result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            deployment_recommendation="block",
        )
        risk_report = RiskReport(
            risks=(),
            critical_count=1,
            high_count=0,
            low_count=0,
            overall_risk_level="critical",
            deployment_blocked=True,
        )

        # Empty validation_results
        reasons = generate_blocking_reasons(
            confidence_result=confidence_result,
            risk_report=risk_report,
            validation_results=(),
        )

        # Should have 2 reasons: low_confidence, critical_risk (no validation_failed)
        assert len(reasons) == 2
        reason_types = {r.reason_type for r in reasons}
        assert "low_confidence" in reason_types
        assert "critical_risk" in reason_types
        assert "validation_failed" not in reason_types
