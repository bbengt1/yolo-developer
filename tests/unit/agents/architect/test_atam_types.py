"""Tests for ATAM Review type definitions (Story 7.7, Task 1, Task 9).

Tests verify the ATAM-related frozen dataclasses for architectural review.
"""

from __future__ import annotations

import pytest


class TestATAMScenarioDataclass:
    """Test ATAMScenario frozen dataclass."""

    def test_atam_scenario_creation(self) -> None:
        """Test creating an ATAMScenario with all fields."""
        from yolo_developer.agents.architect.types import ATAMScenario

        scenario = ATAMScenario(
            scenario_id="ATAM-001",
            quality_attribute="performance",
            stimulus="100 concurrent API requests",
            response="95th percentile < 500ms",
            analysis="Design supports async processing",
        )

        assert scenario.scenario_id == "ATAM-001"
        assert scenario.quality_attribute == "performance"
        assert scenario.stimulus == "100 concurrent API requests"
        assert scenario.response == "95th percentile < 500ms"
        assert scenario.analysis == "Design supports async processing"

    def test_atam_scenario_is_frozen(self) -> None:
        """Test that ATAMScenario is immutable."""
        from yolo_developer.agents.architect.types import ATAMScenario

        scenario = ATAMScenario(
            scenario_id="ATAM-001",
            quality_attribute="performance",
            stimulus="Test stimulus",
            response="Test response",
            analysis="Test analysis",
        )

        with pytest.raises(AttributeError):
            scenario.scenario_id = "ATAM-002"  # type: ignore[misc]

    def test_atam_scenario_to_dict(self) -> None:
        """Test ATAMScenario.to_dict() serialization."""
        from yolo_developer.agents.architect.types import ATAMScenario

        scenario = ATAMScenario(
            scenario_id="ATAM-001",
            quality_attribute="security",
            stimulus="Unauthorized access attempt",
            response="Request blocked with 401",
            analysis="Authentication required for all endpoints",
        )

        result = scenario.to_dict()

        assert result == {
            "scenario_id": "ATAM-001",
            "quality_attribute": "security",
            "stimulus": "Unauthorized access attempt",
            "response": "Request blocked with 401",
            "analysis": "Authentication required for all endpoints",
        }


class TestATAMTradeOffConflictDataclass:
    """Test ATAMTradeOffConflict frozen dataclass."""

    def test_atam_trade_off_conflict_creation(self) -> None:
        """Test creating an ATAMTradeOffConflict with all fields."""
        from yolo_developer.agents.architect.types import ATAMTradeOffConflict

        conflict = ATAMTradeOffConflict(
            attribute_a="performance",
            attribute_b="security",
            description="Encryption adds latency to all requests",
            severity="medium",
            resolution_strategy="Use async encryption with result caching",
        )

        assert conflict.attribute_a == "performance"
        assert conflict.attribute_b == "security"
        assert conflict.description == "Encryption adds latency to all requests"
        assert conflict.severity == "medium"
        assert conflict.resolution_strategy == "Use async encryption with result caching"

    def test_atam_trade_off_conflict_is_frozen(self) -> None:
        """Test that ATAMTradeOffConflict is immutable."""
        from yolo_developer.agents.architect.types import ATAMTradeOffConflict

        conflict = ATAMTradeOffConflict(
            attribute_a="performance",
            attribute_b="security",
            description="Test conflict",
            severity="low",
            resolution_strategy="Test strategy",
        )

        with pytest.raises(AttributeError):
            conflict.severity = "high"  # type: ignore[misc]

    def test_atam_trade_off_conflict_to_dict(self) -> None:
        """Test ATAMTradeOffConflict.to_dict() serialization."""
        from yolo_developer.agents.architect.types import ATAMTradeOffConflict

        conflict = ATAMTradeOffConflict(
            attribute_a="scalability",
            attribute_b="maintainability",
            description="Distributed system adds complexity",
            severity="high",
            resolution_strategy="Start simple, document scaling path",
        )

        result = conflict.to_dict()

        assert result == {
            "attribute_a": "scalability",
            "attribute_b": "maintainability",
            "description": "Distributed system adds complexity",
            "severity": "high",
            "resolution_strategy": "Start simple, document scaling path",
        }

    def test_atam_trade_off_conflict_severity_values(self) -> None:
        """Test valid severity values for ATAMTradeOffConflict."""
        from yolo_developer.agents.architect.types import ATAMTradeOffConflict

        for severity in ["critical", "high", "medium", "low"]:
            conflict = ATAMTradeOffConflict(
                attribute_a="a",
                attribute_b="b",
                description="desc",
                severity=severity,
                resolution_strategy="strategy",
            )
            assert conflict.severity == severity


class TestATAMRiskAssessmentDataclass:
    """Test ATAMRiskAssessment frozen dataclass."""

    def test_atam_risk_assessment_creation(self) -> None:
        """Test creating an ATAMRiskAssessment with all fields."""
        from yolo_developer.agents.architect.types import ATAMRiskAssessment

        assessment = ATAMRiskAssessment(
            risk_id="RISK-001",
            quality_impact=("reliability", "performance"),
            mitigation_feasibility="high",
            unmitigated=False,
        )

        assert assessment.risk_id == "RISK-001"
        assert assessment.quality_impact == ("reliability", "performance")
        assert assessment.mitigation_feasibility == "high"
        assert assessment.unmitigated is False

    def test_atam_risk_assessment_unmitigated(self) -> None:
        """Test ATAMRiskAssessment with unmitigated risk."""
        from yolo_developer.agents.architect.types import ATAMRiskAssessment

        assessment = ATAMRiskAssessment(
            risk_id="RISK-002",
            quality_impact=("security",),
            mitigation_feasibility="low",
            unmitigated=True,
        )

        assert assessment.unmitigated is True
        assert assessment.mitigation_feasibility == "low"

    def test_atam_risk_assessment_is_frozen(self) -> None:
        """Test that ATAMRiskAssessment is immutable."""
        from yolo_developer.agents.architect.types import ATAMRiskAssessment

        assessment = ATAMRiskAssessment(
            risk_id="RISK-001",
            quality_impact=("performance",),
            mitigation_feasibility="medium",
            unmitigated=False,
        )

        with pytest.raises(AttributeError):
            assessment.unmitigated = True  # type: ignore[misc]

    def test_atam_risk_assessment_to_dict(self) -> None:
        """Test ATAMRiskAssessment.to_dict() serialization."""
        from yolo_developer.agents.architect.types import ATAMRiskAssessment

        assessment = ATAMRiskAssessment(
            risk_id="RISK-003",
            quality_impact=("scalability", "reliability"),
            mitigation_feasibility="medium",
            unmitigated=False,
        )

        result = assessment.to_dict()

        assert result == {
            "risk_id": "RISK-003",
            "quality_impact": ["scalability", "reliability"],
            "mitigation_feasibility": "medium",
            "unmitigated": False,
        }


class TestATAMReviewResultDataclass:
    """Test ATAMReviewResult frozen dataclass."""

    def test_atam_review_result_pass(self) -> None:
        """Test ATAMReviewResult with passing review."""
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            ATAMRiskAssessment,
            ATAMScenario,
            ATAMTradeOffConflict,
        )

        scenario = ATAMScenario(
            scenario_id="ATAM-001",
            quality_attribute="performance",
            stimulus="Load test",
            response="Meets SLA",
            analysis="Good",
        )

        result = ATAMReviewResult(
            overall_pass=True,
            confidence=0.85,
            scenarios_evaluated=(scenario,),
            trade_off_conflicts=(),
            risk_assessments=(),
            failure_reasons=(),
            summary="Design passes ATAM review with 85% confidence",
        )

        assert result.overall_pass is True
        assert result.confidence == 0.85
        assert len(result.scenarios_evaluated) == 1
        assert len(result.failure_reasons) == 0

    def test_atam_review_result_fail(self) -> None:
        """Test ATAMReviewResult with failing review."""
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            ATAMRiskAssessment,
        )

        risk = ATAMRiskAssessment(
            risk_id="RISK-CRIT-001",
            quality_impact=("security",),
            mitigation_feasibility="low",
            unmitigated=True,
        )

        result = ATAMReviewResult(
            overall_pass=False,
            confidence=0.45,
            scenarios_evaluated=(),
            trade_off_conflicts=(),
            risk_assessments=(risk,),
            failure_reasons=(
                "Critical unmitigated risk: RISK-CRIT-001",
                "Confidence below threshold (0.45 < 0.6)",
            ),
            summary="Design fails ATAM review due to critical risks",
        )

        assert result.overall_pass is False
        assert result.confidence == 0.45
        assert len(result.failure_reasons) == 2
        assert "Critical unmitigated risk" in result.failure_reasons[0]

    def test_atam_review_result_is_frozen(self) -> None:
        """Test that ATAMReviewResult is immutable."""
        from yolo_developer.agents.architect.types import ATAMReviewResult

        result = ATAMReviewResult(
            overall_pass=True,
            confidence=0.9,
            scenarios_evaluated=(),
            trade_off_conflicts=(),
            risk_assessments=(),
            failure_reasons=(),
            summary="Test",
        )

        with pytest.raises(AttributeError):
            result.overall_pass = False  # type: ignore[misc]

    def test_atam_review_result_to_dict(self) -> None:
        """Test ATAMReviewResult.to_dict() serialization."""
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            ATAMRiskAssessment,
            ATAMScenario,
            ATAMTradeOffConflict,
        )

        scenario = ATAMScenario(
            scenario_id="ATAM-001",
            quality_attribute="reliability",
            stimulus="Service failure",
            response="Automatic recovery",
            analysis="Retry logic implemented",
        )

        conflict = ATAMTradeOffConflict(
            attribute_a="performance",
            attribute_b="reliability",
            description="Retries add latency",
            severity="low",
            resolution_strategy="Use circuit breaker",
        )

        risk = ATAMRiskAssessment(
            risk_id="RISK-001",
            quality_impact=("integration",),
            mitigation_feasibility="high",
            unmitigated=False,
        )

        result = ATAMReviewResult(
            overall_pass=True,
            confidence=0.78,
            scenarios_evaluated=(scenario,),
            trade_off_conflicts=(conflict,),
            risk_assessments=(risk,),
            failure_reasons=(),
            summary="Design passes with minor concerns",
        )

        d = result.to_dict()

        assert d["overall_pass"] is True
        assert d["confidence"] == 0.78
        assert len(d["scenarios_evaluated"]) == 1
        assert d["scenarios_evaluated"][0]["scenario_id"] == "ATAM-001"
        assert len(d["trade_off_conflicts"]) == 1
        assert d["trade_off_conflicts"][0]["attribute_a"] == "performance"
        assert len(d["risk_assessments"]) == 1
        assert d["risk_assessments"][0]["risk_id"] == "RISK-001"
        assert d["failure_reasons"] == []
        assert d["summary"] == "Design passes with minor concerns"

    def test_atam_review_result_with_all_components(self) -> None:
        """Test ATAMReviewResult with multiple scenarios, conflicts, and risks."""
        from yolo_developer.agents.architect.types import (
            ATAMReviewResult,
            ATAMRiskAssessment,
            ATAMScenario,
            ATAMTradeOffConflict,
        )

        scenarios = (
            ATAMScenario("ATAM-001", "performance", "s1", "r1", "a1"),
            ATAMScenario("ATAM-002", "security", "s2", "r2", "a2"),
            ATAMScenario("ATAM-003", "reliability", "s3", "r3", "a3"),
        )

        conflicts = (
            ATAMTradeOffConflict("a", "b", "d1", "medium", "s1"),
            ATAMTradeOffConflict("c", "d", "d2", "low", "s2"),
        )

        risks = (
            ATAMRiskAssessment("R1", ("performance",), "high", False),
            ATAMRiskAssessment("R2", ("security", "reliability"), "medium", False),
        )

        result = ATAMReviewResult(
            overall_pass=True,
            confidence=0.82,
            scenarios_evaluated=scenarios,
            trade_off_conflicts=conflicts,
            risk_assessments=risks,
            failure_reasons=(),
            summary="Comprehensive review complete",
        )

        assert len(result.scenarios_evaluated) == 3
        assert len(result.trade_off_conflicts) == 2
        assert len(result.risk_assessments) == 2
        assert result.overall_pass is True


class TestATAMTypeExports:
    """Test that ATAM types are properly exported from architect module."""

    def test_atam_scenario_importable(self) -> None:
        """Test ATAMScenario is importable from yolo_developer.agents.architect."""
        from yolo_developer.agents.architect import ATAMScenario

        assert ATAMScenario is not None

    def test_atam_trade_off_conflict_importable(self) -> None:
        """Test ATAMTradeOffConflict is importable from yolo_developer.agents.architect."""
        from yolo_developer.agents.architect import ATAMTradeOffConflict

        assert ATAMTradeOffConflict is not None

    def test_atam_risk_assessment_importable(self) -> None:
        """Test ATAMRiskAssessment is importable from yolo_developer.agents.architect."""
        from yolo_developer.agents.architect import ATAMRiskAssessment

        assert ATAMRiskAssessment is not None

    def test_atam_review_result_importable(self) -> None:
        """Test ATAMReviewResult is importable from yolo_developer.agents.architect."""
        from yolo_developer.agents.architect import ATAMReviewResult

        assert ATAMReviewResult is not None
