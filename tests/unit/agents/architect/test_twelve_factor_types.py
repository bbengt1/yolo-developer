"""Tests for Twelve-Factor type definitions (Story 7.2, Task 1, Task 9).

Tests verify the FactorResult and TwelveFactorAnalysis frozen dataclasses
following the patterns established in Story 7.1 for DesignDecision and ADR.
"""

from __future__ import annotations

import pytest


class TestFactorResult:
    """Test FactorResult dataclass."""

    def test_factor_result_creation(self) -> None:
        """Test basic FactorResult creation."""
        from yolo_developer.agents.architect.types import FactorResult

        result = FactorResult(
            factor_name="config",
            applies=True,
            compliant=True,
            finding="Configuration is externalized",
            recommendation="",
        )

        assert result.factor_name == "config"
        assert result.applies is True
        assert result.compliant is True
        assert result.finding == "Configuration is externalized"
        assert result.recommendation == ""

    def test_factor_result_non_applicable(self) -> None:
        """Test FactorResult when factor doesn't apply."""
        from yolo_developer.agents.architect.types import FactorResult

        result = FactorResult(
            factor_name="port_binding",
            applies=False,
            compliant=None,
            finding="Story does not involve service exposure",
            recommendation="",
        )

        assert result.applies is False
        assert result.compliant is None

    def test_factor_result_non_compliant(self) -> None:
        """Test FactorResult when non-compliant."""
        from yolo_developer.agents.architect.types import FactorResult

        result = FactorResult(
            factor_name="config",
            applies=True,
            compliant=False,
            finding="Hardcoded database URL detected",
            recommendation="Use environment variable DATABASE_URL",
        )

        assert result.compliant is False
        assert result.recommendation == "Use environment variable DATABASE_URL"

    def test_factor_result_to_dict(self) -> None:
        """Test FactorResult serialization."""
        from yolo_developer.agents.architect.types import FactorResult

        result = FactorResult(
            factor_name="processes",
            applies=True,
            compliant=True,
            finding="Stateless process pattern used",
            recommendation="",
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["factor_name"] == "processes"
        assert d["applies"] is True
        assert d["compliant"] is True
        assert d["finding"] == "Stateless process pattern used"
        assert d["recommendation"] == ""

    def test_factor_result_is_frozen(self) -> None:
        """Test that FactorResult is immutable."""
        from yolo_developer.agents.architect.types import FactorResult

        result = FactorResult(
            factor_name="config",
            applies=True,
            compliant=True,
            finding="OK",
            recommendation="",
        )

        with pytest.raises(AttributeError):
            result.factor_name = "other"  # type: ignore[misc]


class TestTwelveFactorAnalysis:
    """Test TwelveFactorAnalysis dataclass."""

    def test_twelve_factor_analysis_creation(self) -> None:
        """Test basic TwelveFactorAnalysis creation."""
        from yolo_developer.agents.architect.types import (
            FactorResult,
            TwelveFactorAnalysis,
        )

        factor_results = {
            "config": FactorResult(
                factor_name="config",
                applies=True,
                compliant=True,
                finding="OK",
                recommendation="",
            ),
        }

        analysis = TwelveFactorAnalysis(
            factor_results=factor_results,
            applicable_factors=("config",),
            overall_compliance=1.0,
            recommendations=(),
        )

        assert analysis.factor_results == factor_results
        assert analysis.applicable_factors == ("config",)
        assert analysis.overall_compliance == 1.0
        assert analysis.recommendations == ()

    def test_twelve_factor_analysis_with_recommendations(self) -> None:
        """Test TwelveFactorAnalysis with recommendations."""
        from yolo_developer.agents.architect.types import (
            FactorResult,
            TwelveFactorAnalysis,
        )

        factor_results = {
            "config": FactorResult(
                factor_name="config",
                applies=True,
                compliant=False,
                finding="Hardcoded values",
                recommendation="Use env vars",
            ),
        }

        analysis = TwelveFactorAnalysis(
            factor_results=factor_results,
            applicable_factors=("config",),
            overall_compliance=0.0,
            recommendations=("Use env vars for all config",),
        )

        assert analysis.overall_compliance == 0.0
        assert len(analysis.recommendations) == 1

    def test_twelve_factor_analysis_to_dict(self) -> None:
        """Test TwelveFactorAnalysis serialization."""
        from yolo_developer.agents.architect.types import (
            FactorResult,
            TwelveFactorAnalysis,
        )

        factor_results = {
            "config": FactorResult(
                factor_name="config",
                applies=True,
                compliant=True,
                finding="OK",
                recommendation="",
            ),
            "processes": FactorResult(
                factor_name="processes",
                applies=True,
                compliant=True,
                finding="Stateless",
                recommendation="",
            ),
        }

        analysis = TwelveFactorAnalysis(
            factor_results=factor_results,
            applicable_factors=("config", "processes"),
            overall_compliance=1.0,
            recommendations=(),
        )

        d = analysis.to_dict()

        assert isinstance(d, dict)
        assert "factor_results" in d
        assert "applicable_factors" in d
        assert "overall_compliance" in d
        assert "recommendations" in d
        assert isinstance(d["factor_results"], dict)
        assert d["overall_compliance"] == 1.0
        assert d["applicable_factors"] == ["config", "processes"]

    def test_twelve_factor_analysis_is_frozen(self) -> None:
        """Test that TwelveFactorAnalysis is immutable."""
        from yolo_developer.agents.architect.types import TwelveFactorAnalysis

        analysis = TwelveFactorAnalysis(
            factor_results={},
            applicable_factors=(),
            overall_compliance=0.5,
            recommendations=(),
        )

        with pytest.raises(AttributeError):
            analysis.overall_compliance = 1.0  # type: ignore[misc]

    def test_twelve_factor_analysis_default_values(self) -> None:
        """Test TwelveFactorAnalysis with minimal fields."""
        from yolo_developer.agents.architect.types import TwelveFactorAnalysis

        analysis = TwelveFactorAnalysis(
            factor_results={},
            applicable_factors=(),
            overall_compliance=0.0,
            recommendations=(),
        )

        assert analysis.factor_results == {}
        assert analysis.applicable_factors == ()
        assert analysis.overall_compliance == 0.0


class TestTwelveFactorsConstant:
    """Test TWELVE_FACTORS constant."""

    def test_twelve_factors_exists(self) -> None:
        """Test TWELVE_FACTORS constant is defined."""
        from yolo_developer.agents.architect.types import TWELVE_FACTORS

        assert TWELVE_FACTORS is not None
        assert len(TWELVE_FACTORS) == 12

    def test_twelve_factors_contains_all_factors(self) -> None:
        """Test TWELVE_FACTORS contains all 12 factor names."""
        from yolo_developer.agents.architect.types import TWELVE_FACTORS

        expected_factors = {
            "codebase",
            "dependencies",
            "config",
            "backing_services",
            "build_release_run",
            "processes",
            "port_binding",
            "concurrency",
            "disposability",
            "dev_prod_parity",
            "logs",
            "admin_processes",
        }

        assert set(TWELVE_FACTORS) == expected_factors

    def test_twelve_factors_is_tuple(self) -> None:
        """Test TWELVE_FACTORS is a tuple (immutable)."""
        from yolo_developer.agents.architect.types import TWELVE_FACTORS

        assert isinstance(TWELVE_FACTORS, tuple)


class TestTypeExports:
    """Test type exports from architect module."""

    def test_factor_result_importable_from_architect(self) -> None:
        """Test FactorResult is exported from architect __init__.py."""
        from yolo_developer.agents.architect import FactorResult

        assert FactorResult is not None

    def test_twelve_factor_analysis_importable_from_architect(self) -> None:
        """Test TwelveFactorAnalysis is exported from architect __init__.py."""
        from yolo_developer.agents.architect import TwelveFactorAnalysis

        assert TwelveFactorAnalysis is not None

    def test_twelve_factors_importable_from_architect(self) -> None:
        """Test TWELVE_FACTORS is exported from architect __init__.py."""
        from yolo_developer.agents.architect import TWELVE_FACTORS

        assert TWELVE_FACTORS is not None
