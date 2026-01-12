"""Unit tests for confidence scoring types (Story 9.4 - Task 1, 2, 10).

Tests for the confidence scoring data types:
- ConfidenceWeight: Weight configuration for score components
- ConfidenceBreakdown: Detailed breakdown of score components
- ConfidenceResult: Complete confidence scoring result
- Weight configuration functions (Task 2)

All types should be frozen dataclasses (immutable) per ADR-001.
"""

from __future__ import annotations

import pytest


class TestWeightConfiguration:
    """Tests for weight configuration functions (Task 2)."""

    def test_get_default_weights(self) -> None:
        """Test get_default_weights returns correct values."""
        from yolo_developer.agents.tea.scoring import get_default_weights

        weights = get_default_weights()

        assert weights.coverage_weight == 0.4
        assert weights.test_execution_weight == 0.3
        assert weights.validation_weight == 0.3

    def test_default_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        from yolo_developer.agents.tea.scoring import get_default_weights

        weights = get_default_weights()
        total = weights.coverage_weight + weights.test_execution_weight + weights.validation_weight

        assert abs(total - 1.0) < 0.001

    def test_validate_weights_valid(self) -> None:
        """Test validate_weights with valid weights."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight, validate_weights

        weights = ConfidenceWeight(
            coverage_weight=0.4,
            test_execution_weight=0.3,
            validation_weight=0.3,
        )

        assert validate_weights(weights) is True

    def test_validate_weights_invalid(self) -> None:
        """Test validate_weights with invalid weights."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight, validate_weights

        weights = ConfidenceWeight(
            coverage_weight=0.5,
            test_execution_weight=0.5,
            validation_weight=0.5,
        )

        assert validate_weights(weights) is False

    def test_validate_weights_floating_point_tolerance(self) -> None:
        """Test validate_weights handles floating point precision."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight, validate_weights

        # Values that should sum to 1.0 with minor floating point issues
        weights = ConfidenceWeight(
            coverage_weight=0.33333333,
            test_execution_weight=0.33333333,
            validation_weight=0.33333334,
        )

        assert validate_weights(weights) is True

    def test_get_weights_from_config_returns_defaults(self) -> None:
        """Test get_weights_from_config falls back to defaults."""
        from yolo_developer.agents.tea.scoring import get_weights_from_config

        weights = get_weights_from_config()

        # Should return default weights when config not available
        assert weights.coverage_weight == 0.4
        assert weights.test_execution_weight == 0.3
        assert weights.validation_weight == 0.3


class TestConfidenceWeight:
    """Tests for ConfidenceWeight frozen dataclass."""

    def test_create_with_valid_weights(self) -> None:
        """Test creating ConfidenceWeight with valid weights summing to 1.0."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight

        weight = ConfidenceWeight(
            coverage_weight=0.4,
            test_execution_weight=0.3,
            validation_weight=0.3,
        )

        assert weight.coverage_weight == 0.4
        assert weight.test_execution_weight == 0.3
        assert weight.validation_weight == 0.3

    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to approximately 1.0."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight

        weight = ConfidenceWeight(
            coverage_weight=0.4,
            test_execution_weight=0.3,
            validation_weight=0.3,
        )

        total = weight.coverage_weight + weight.test_execution_weight + weight.validation_weight
        assert abs(total - 1.0) < 0.001  # Allow floating point tolerance

    def test_is_frozen_dataclass(self) -> None:
        """Test that ConfidenceWeight is immutable (frozen)."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight

        weight = ConfidenceWeight(
            coverage_weight=0.4,
            test_execution_weight=0.3,
            validation_weight=0.3,
        )

        with pytest.raises(AttributeError):
            weight.coverage_weight = 0.5  # type: ignore[misc]

    def test_to_dict_method(self) -> None:
        """Test to_dict() serialization method."""
        from yolo_developer.agents.tea.scoring import ConfidenceWeight

        weight = ConfidenceWeight(
            coverage_weight=0.4,
            test_execution_weight=0.3,
            validation_weight=0.3,
        )

        result = weight.to_dict()

        assert result == {
            "coverage_weight": 0.4,
            "test_execution_weight": 0.3,
            "validation_weight": 0.3,
        }


class TestConfidenceBreakdown:
    """Tests for ConfidenceBreakdown frozen dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test creating ConfidenceBreakdown with all fields."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            penalties=("-10 for 1 high finding",),
            bonuses=(),
            base_score=82.0,
            final_score=82,
        )

        assert breakdown.coverage_score == 85.0
        assert breakdown.test_execution_score == 90.0
        assert breakdown.validation_score == 70.0
        assert breakdown.weighted_coverage == 34.0
        assert breakdown.weighted_test_execution == 27.0
        assert breakdown.weighted_validation == 21.0
        assert breakdown.penalties == ("-10 for 1 high finding",)
        assert breakdown.bonuses == ()
        assert breakdown.base_score == 82.0
        assert breakdown.final_score == 82

    def test_is_frozen_dataclass(self) -> None:
        """Test that ConfidenceBreakdown is immutable (frozen)."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            penalties=(),
            bonuses=(),
            base_score=82.0,
            final_score=82,
        )

        with pytest.raises(AttributeError):
            breakdown.coverage_score = 100.0  # type: ignore[misc]

    def test_to_dict_method(self) -> None:
        """Test to_dict() serialization method."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            penalties=("-10 for high finding",),
            bonuses=("+5 for perfect tests",),
            base_score=82.0,
            final_score=87,
        )

        result = breakdown.to_dict()

        assert result == {
            "coverage_score": 85.0,
            "test_execution_score": 90.0,
            "validation_score": 70.0,
            "weighted_coverage": 34.0,
            "weighted_test_execution": 27.0,
            "weighted_validation": 21.0,
            "penalties": ["-10 for high finding"],
            "bonuses": ["+5 for perfect tests"],
            "base_score": 82.0,
            "final_score": 87,
        }

    def test_default_empty_tuples(self) -> None:
        """Test that penalties and bonuses default to empty tuples."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            base_score=82.0,
            final_score=82,
        )

        assert breakdown.penalties == ()
        assert breakdown.bonuses == ()


class TestConfidenceResult:
    """Tests for ConfidenceResult frozen dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test creating ConfidenceResult with all fields."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            penalties=(),
            bonuses=(),
            base_score=82.0,
            final_score=82,
        )

        result = ConfidenceResult(
            score=82,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            blocking_reasons=("Score 82 is below threshold 90",),
            deployment_recommendation="block",
        )

        assert result.score == 82
        assert result.breakdown == breakdown
        assert result.passed_threshold is False
        assert result.threshold_value == 90
        assert result.blocking_reasons == ("Score 82 is below threshold 90",)
        assert result.deployment_recommendation == "block"

    def test_is_frozen_dataclass(self) -> None:
        """Test that ConfidenceResult is immutable (frozen)."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            penalties=(),
            bonuses=(),
            base_score=100.0,
            final_score=100,
        )

        result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            blocking_reasons=(),
            deployment_recommendation="deploy",
        )

        with pytest.raises(AttributeError):
            result.score = 50  # type: ignore[misc]

    def test_to_dict_method(self) -> None:
        """Test to_dict() serialization method."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        breakdown = ConfidenceBreakdown(
            coverage_score=85.0,
            test_execution_score=90.0,
            validation_score=70.0,
            weighted_coverage=34.0,
            weighted_test_execution=27.0,
            weighted_validation=21.0,
            penalties=("-10 for high finding",),
            bonuses=(),
            base_score=82.0,
            final_score=82,
        )

        result = ConfidenceResult(
            score=82,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            blocking_reasons=("Score 82 is below threshold 90",),
            deployment_recommendation="block",
        )

        serialized = result.to_dict()

        assert serialized["score"] == 82
        assert serialized["passed_threshold"] is False
        assert serialized["threshold_value"] == 90
        assert serialized["blocking_reasons"] == ["Score 82 is below threshold 90"]
        assert serialized["deployment_recommendation"] == "block"
        assert "breakdown" in serialized
        assert serialized["breakdown"]["coverage_score"] == 85.0
        assert "created_at" in serialized
        assert "blocking_finding" in serialized

    def test_blocking_finding_field(self) -> None:
        """Test that blocking_finding field is included (AC4)."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult
        from yolo_developer.agents.tea.types import Finding

        breakdown = ConfidenceBreakdown(
            coverage_score=50.0,
            test_execution_score=50.0,
            validation_score=50.0,
            weighted_coverage=20.0,
            weighted_test_execution=15.0,
            weighted_validation=15.0,
            penalties=(),
            bonuses=(),
            base_score=50.0,
            final_score=50,
        )

        blocking_finding = Finding(
            finding_id="F-CONFIDENCE-50",
            category="test_coverage",
            severity="critical",
            description="Confidence score 50 is below deployment threshold 90",
            location="confidence_scoring",
            remediation="Increase test coverage",
        )

        result = ConfidenceResult(
            score=50,
            breakdown=breakdown,
            passed_threshold=False,
            threshold_value=90,
            blocking_reasons=("Score 50 is below threshold 90",),
            deployment_recommendation="block",
            blocking_finding=blocking_finding,
        )

        assert result.blocking_finding is not None
        assert result.blocking_finding.severity == "critical"

        serialized = result.to_dict()
        assert serialized["blocking_finding"] is not None
        assert serialized["blocking_finding"]["severity"] == "critical"

    def test_created_at_has_default(self) -> None:
        """Test that created_at is automatically set."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            penalties=(),
            bonuses=(),
            base_score=100.0,
            final_score=100,
        )

        result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            blocking_reasons=(),
            deployment_recommendation="deploy",
        )

        assert result.created_at is not None
        assert len(result.created_at) > 0
        # Should be ISO format
        assert "T" in result.created_at

    def test_default_blocking_reasons_empty(self) -> None:
        """Test that blocking_reasons defaults to empty tuple."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        breakdown = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            penalties=(),
            bonuses=(),
            base_score=100.0,
            final_score=100,
        )

        result = ConfidenceResult(
            score=100,
            breakdown=breakdown,
            passed_threshold=True,
            threshold_value=90,
            deployment_recommendation="deploy",
        )

        assert result.blocking_reasons == ()

    def test_score_range_0_to_100(self) -> None:
        """Test that score values work within 0-100 range."""
        from yolo_developer.agents.tea.scoring import ConfidenceBreakdown, ConfidenceResult

        # Test score = 0
        breakdown_zero = ConfidenceBreakdown(
            coverage_score=0.0,
            test_execution_score=0.0,
            validation_score=0.0,
            weighted_coverage=0.0,
            weighted_test_execution=0.0,
            weighted_validation=0.0,
            penalties=("-75 for critical findings",),
            bonuses=(),
            base_score=0.0,
            final_score=0,
        )

        result_zero = ConfidenceResult(
            score=0,
            breakdown=breakdown_zero,
            passed_threshold=False,
            threshold_value=90,
            blocking_reasons=("Score 0 is below threshold 90",),
            deployment_recommendation="block",
        )

        assert result_zero.score == 0

        # Test score = 100
        breakdown_perfect = ConfidenceBreakdown(
            coverage_score=100.0,
            test_execution_score=100.0,
            validation_score=100.0,
            weighted_coverage=40.0,
            weighted_test_execution=30.0,
            weighted_validation=30.0,
            penalties=(),
            bonuses=("+5 for perfect tests",),
            base_score=100.0,
            final_score=100,
        )

        result_perfect = ConfidenceResult(
            score=100,
            breakdown=breakdown_perfect,
            passed_threshold=True,
            threshold_value=90,
            blocking_reasons=(),
            deployment_recommendation="deploy",
        )

        assert result_perfect.score == 100
