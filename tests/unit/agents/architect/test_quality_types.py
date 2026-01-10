"""Tests for Quality Attribute type definitions (Story 7.4, Task 1).

Tests verify QualityAttribute, QualityRisk, QualityTradeOff, and
QualityAttributeEvaluation dataclasses.
"""

from __future__ import annotations

import pytest


class TestQualityAttributeLiteral:
    """Test QualityAttribute Literal type."""

    def test_quality_attributes_defined(self) -> None:
        """Test that QUALITY_ATTRIBUTES constant is defined."""
        from yolo_developer.agents.architect.types import QUALITY_ATTRIBUTES

        assert isinstance(QUALITY_ATTRIBUTES, tuple)
        assert len(QUALITY_ATTRIBUTES) >= 5

    def test_quality_attributes_include_core_attributes(self) -> None:
        """Test that core quality attributes are included."""
        from yolo_developer.agents.architect.types import QUALITY_ATTRIBUTES

        core_attrs = ("performance", "security", "reliability", "scalability", "maintainability")
        for attr in core_attrs:
            assert attr in QUALITY_ATTRIBUTES


class TestQualityRisk:
    """Test QualityRisk frozen dataclass."""

    def test_quality_risk_creation(self) -> None:
        """Test creating a QualityRisk instance."""
        from yolo_developer.agents.architect.types import QualityRisk

        risk = QualityRisk(
            attribute="performance",
            description="High latency in database queries",
            severity="high",
            mitigation="Add database indexing and connection pooling",
            mitigation_effort="medium",
        )

        assert risk.attribute == "performance"
        assert risk.description == "High latency in database queries"
        assert risk.severity == "high"
        assert risk.mitigation == "Add database indexing and connection pooling"
        assert risk.mitigation_effort == "medium"

    def test_quality_risk_to_dict(self) -> None:
        """Test QualityRisk.to_dict() serialization."""
        from yolo_developer.agents.architect.types import QualityRisk

        risk = QualityRisk(
            attribute="security",
            description="Missing input validation",
            severity="critical",
            mitigation="Add validation layer",
            mitigation_effort="low",
        )

        result = risk.to_dict()

        assert isinstance(result, dict)
        assert result["attribute"] == "security"
        assert result["description"] == "Missing input validation"
        assert result["severity"] == "critical"
        assert result["mitigation"] == "Add validation layer"
        assert result["mitigation_effort"] == "low"

    def test_quality_risk_immutable(self) -> None:
        """Test that QualityRisk is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from yolo_developer.agents.architect.types import QualityRisk

        risk = QualityRisk(
            attribute="reliability",
            description="No retry logic",
            severity="medium",
            mitigation="Add tenacity retry",
            mitigation_effort="low",
        )

        with pytest.raises(FrozenInstanceError):
            risk.severity = "low"  # type: ignore[misc]

    def test_quality_risk_severity_values(self) -> None:
        """Test that severity accepts valid values."""
        from yolo_developer.agents.architect.types import QualityRisk, RiskSeverity

        # Just verify the type exists and accepts valid values
        severities: list[RiskSeverity] = ["critical", "high", "medium", "low"]
        for severity in severities:
            risk = QualityRisk(
                attribute="performance",
                description="Test",
                severity=severity,
                mitigation="Fix it",
                mitigation_effort="low",
            )
            assert risk.severity == severity


class TestQualityTradeOff:
    """Test QualityTradeOff frozen dataclass."""

    def test_quality_tradeoff_creation(self) -> None:
        """Test creating a QualityTradeOff instance."""
        from yolo_developer.agents.architect.types import QualityTradeOff

        tradeoff = QualityTradeOff(
            attribute_a="performance",
            attribute_b="security",
            description="Encryption adds 50ms latency per request",
            resolution="Use async encryption and cache encrypted results",
        )

        assert tradeoff.attribute_a == "performance"
        assert tradeoff.attribute_b == "security"
        assert "Encryption" in tradeoff.description
        assert "async" in tradeoff.resolution

    def test_quality_tradeoff_to_dict(self) -> None:
        """Test QualityTradeOff.to_dict() serialization."""
        from yolo_developer.agents.architect.types import QualityTradeOff

        tradeoff = QualityTradeOff(
            attribute_a="scalability",
            attribute_b="maintainability",
            description="Distributed architecture adds complexity",
            resolution="Start simple, document scaling path",
        )

        result = tradeoff.to_dict()

        assert isinstance(result, dict)
        assert result["attribute_a"] == "scalability"
        assert result["attribute_b"] == "maintainability"
        assert result["description"] == "Distributed architecture adds complexity"
        assert result["resolution"] == "Start simple, document scaling path"

    def test_quality_tradeoff_immutable(self) -> None:
        """Test that QualityTradeOff is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from yolo_developer.agents.architect.types import QualityTradeOff

        tradeoff = QualityTradeOff(
            attribute_a="performance",
            attribute_b="reliability",
            description="Caching may serve stale data",
            resolution="TTL tuning",
        )

        with pytest.raises(FrozenInstanceError):
            tradeoff.resolution = "New resolution"  # type: ignore[misc]


class TestQualityAttributeEvaluation:
    """Test QualityAttributeEvaluation frozen dataclass."""

    def test_quality_evaluation_creation(self) -> None:
        """Test creating a QualityAttributeEvaluation instance."""
        from yolo_developer.agents.architect.types import (
            QualityAttributeEvaluation,
            QualityRisk,
            QualityTradeOff,
        )

        risk = QualityRisk(
            attribute="security",
            description="Test risk",
            severity="medium",
            mitigation="Fix it",
            mitigation_effort="low",
        )
        tradeoff = QualityTradeOff(
            attribute_a="performance",
            attribute_b="security",
            description="Test tradeoff",
            resolution="Balance both",
        )

        evaluation = QualityAttributeEvaluation(
            attribute_scores={
                "performance": 0.8,
                "security": 0.7,
                "reliability": 0.9,
                "scalability": 0.6,
                "maintainability": 0.85,
            },
            trade_offs=(tradeoff,),
            risks=(risk,),
            overall_score=0.77,
        )

        assert evaluation.attribute_scores["performance"] == 0.8
        assert len(evaluation.trade_offs) == 1
        assert len(evaluation.risks) == 1
        assert evaluation.overall_score == 0.77

    def test_quality_evaluation_to_dict(self) -> None:
        """Test QualityAttributeEvaluation.to_dict() serialization."""
        from yolo_developer.agents.architect.types import (
            QualityAttributeEvaluation,
            QualityRisk,
            QualityTradeOff,
        )

        risk = QualityRisk(
            attribute="reliability",
            description="No failover",
            severity="high",
            mitigation="Add redundancy",
            mitigation_effort="high",
        )
        tradeoff = QualityTradeOff(
            attribute_a="performance",
            attribute_b="reliability",
            description="Sync writes slower",
            resolution="Use async with confirmation",
        )

        evaluation = QualityAttributeEvaluation(
            attribute_scores={"performance": 0.75, "reliability": 0.65},
            trade_offs=(tradeoff,),
            risks=(risk,),
            overall_score=0.70,
        )

        result = evaluation.to_dict()

        assert isinstance(result, dict)
        assert result["attribute_scores"]["performance"] == 0.75
        assert len(result["trade_offs"]) == 1
        assert result["trade_offs"][0]["attribute_a"] == "performance"
        assert len(result["risks"]) == 1
        assert result["risks"][0]["severity"] == "high"
        assert result["overall_score"] == 0.70

    def test_quality_evaluation_immutable(self) -> None:
        """Test that QualityAttributeEvaluation is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from yolo_developer.agents.architect.types import QualityAttributeEvaluation

        evaluation = QualityAttributeEvaluation(
            attribute_scores={"performance": 0.8},
            trade_offs=(),
            risks=(),
            overall_score=0.8,
        )

        with pytest.raises(FrozenInstanceError):
            evaluation.overall_score = 0.9  # type: ignore[misc]

    def test_quality_evaluation_empty_collections(self) -> None:
        """Test QualityAttributeEvaluation with empty trade_offs and risks."""
        from yolo_developer.agents.architect.types import QualityAttributeEvaluation

        evaluation = QualityAttributeEvaluation(
            attribute_scores={"performance": 0.9},
            trade_offs=(),
            risks=(),
            overall_score=0.9,
        )

        result = evaluation.to_dict()
        assert result["trade_offs"] == []
        assert result["risks"] == []

    def test_quality_evaluation_score_range(self) -> None:
        """Test that overall_score is in valid range 0.0-1.0."""
        from yolo_developer.agents.architect.types import QualityAttributeEvaluation

        # Score should accept values in range
        evaluation = QualityAttributeEvaluation(
            attribute_scores={"performance": 0.5},
            trade_offs=(),
            risks=(),
            overall_score=0.5,
        )
        assert 0.0 <= evaluation.overall_score <= 1.0


class TestMitigationEffort:
    """Test MitigationEffort Literal type."""

    def test_mitigation_effort_values(self) -> None:
        """Test that MitigationEffort accepts valid values."""
        from yolo_developer.agents.architect.types import MitigationEffort, QualityRisk

        efforts: list[MitigationEffort] = ["high", "medium", "low"]
        for effort in efforts:
            risk = QualityRisk(
                attribute="performance",
                description="Test",
                severity="low",
                mitigation="Fix",
                mitigation_effort=effort,
            )
            assert risk.mitigation_effort == effort
