"""Unit tests for quality threshold rejection (Story 4.7).

Tests cover:
- QualityThreshold, RejectionReason, RejectionResult dataclasses
- validate_quality_thresholds() function
- generate_remediation_steps() function
- CLI display functions (_display_rejection, _display_threshold_warning)
- Edge cases: exact threshold, all pass, all fail
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from yolo_developer.seed.ambiguity import Ambiguity, AmbiguitySeverity, AmbiguityType
from yolo_developer.seed.rejection import (
    DEFAULT_AMBIGUITY_THRESHOLD,
    DEFAULT_OVERALL_THRESHOLD,
    DEFAULT_SOP_THRESHOLD,
    QualityThreshold,
    RejectionReason,
    RejectionResult,
    create_rejection_with_remediation,
    generate_remediation_steps,
    validate_quality_thresholds,
)
from yolo_developer.seed.report import QualityMetrics, generate_validation_report
from yolo_developer.seed.sop import (
    ConflictSeverity,
    SOPCategory,
    SOPConflict,
    SOPConstraint,
    SOPValidationResult,
)
from yolo_developer.seed.types import SeedFeature, SeedGoal, SeedParseResult, SeedSource

# =============================================================================
# QualityThreshold Tests
# =============================================================================


class TestQualityThreshold:
    """Tests for QualityThreshold dataclass."""

    def test_default_values(self) -> None:
        """Test default threshold values are set correctly."""
        thresholds = QualityThreshold()
        assert thresholds.overall == DEFAULT_OVERALL_THRESHOLD
        assert thresholds.ambiguity == DEFAULT_AMBIGUITY_THRESHOLD
        assert thresholds.sop == DEFAULT_SOP_THRESHOLD

    def test_custom_values(self) -> None:
        """Test custom threshold values can be set."""
        thresholds = QualityThreshold(overall=0.85, ambiguity=0.75, sop=0.90)
        assert thresholds.overall == 0.85
        assert thresholds.ambiguity == 0.75
        assert thresholds.sop == 0.90

    def test_immutable(self) -> None:
        """Test that QualityThreshold is immutable."""
        thresholds = QualityThreshold()
        with pytest.raises(AttributeError):
            thresholds.overall = 0.5  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test to_dict returns correct structure."""
        thresholds = QualityThreshold(overall=0.8, ambiguity=0.7, sop=0.9)
        result = thresholds.to_dict()
        assert result == {"overall": 0.8, "ambiguity": 0.7, "sop": 0.9}

    def test_validation_below_zero(self) -> None:
        """Test that negative thresholds raise ValueError."""
        with pytest.raises(ValueError, match="overall"):
            QualityThreshold(overall=-0.1)

    def test_validation_above_one(self) -> None:
        """Test that thresholds above 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="sop"):
            QualityThreshold(sop=1.5)

    def test_validation_exact_boundaries(self) -> None:
        """Test that 0.0 and 1.0 are valid threshold values."""
        low = QualityThreshold(overall=0.0, ambiguity=0.0, sop=0.0)
        assert low.overall == 0.0

        high = QualityThreshold(overall=1.0, ambiguity=1.0, sop=1.0)
        assert high.overall == 1.0


# =============================================================================
# RejectionReason Tests
# =============================================================================


class TestRejectionReason:
    """Tests for RejectionReason dataclass."""

    def test_creation(self) -> None:
        """Test RejectionReason can be created."""
        reason = RejectionReason(
            threshold_name="overall",
            actual_score=0.52,
            required_score=0.70,
        )
        assert reason.threshold_name == "overall"
        assert reason.actual_score == 0.52
        assert reason.required_score == 0.70

    def test_auto_description(self) -> None:
        """Test description is auto-generated if not provided."""
        reason = RejectionReason(
            threshold_name="ambiguity",
            actual_score=0.45,
            required_score=0.60,
        )
        assert "Ambiguity" in reason.description
        assert "0.45" in reason.description
        assert "0.60" in reason.description

    def test_custom_description(self) -> None:
        """Test custom description is preserved."""
        reason = RejectionReason(
            threshold_name="sop",
            actual_score=0.75,
            required_score=0.80,
            description="Custom SOP failure message",
        )
        assert reason.description == "Custom SOP failure message"

    def test_immutable(self) -> None:
        """Test that RejectionReason is immutable."""
        reason = RejectionReason(
            threshold_name="overall",
            actual_score=0.5,
            required_score=0.7,
        )
        with pytest.raises(AttributeError):
            reason.actual_score = 0.8  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test to_dict returns correct structure."""
        reason = RejectionReason(
            threshold_name="overall",
            actual_score=0.52,
            required_score=0.70,
            description="Test description",
        )
        result = reason.to_dict()
        assert result["threshold_name"] == "overall"
        assert result["actual_score"] == 0.52
        assert result["required_score"] == 0.70
        assert result["description"] == "Test description"


# =============================================================================
# RejectionResult Tests
# =============================================================================


class TestRejectionResult:
    """Tests for RejectionResult dataclass."""

    def test_passed_result(self) -> None:
        """Test creating a passing result."""
        result = RejectionResult(passed=True)
        assert result.passed is True
        assert result.failure_count == 0
        assert result.reasons == ()
        assert result.recommendations == ()

    def test_failed_result(self) -> None:
        """Test creating a failed result with reasons."""
        reason = RejectionReason(
            threshold_name="overall",
            actual_score=0.52,
            required_score=0.70,
        )
        result = RejectionResult(
            passed=False,
            reasons=(reason,),
            recommendations=("Fix issues",),
        )
        assert result.passed is False
        assert result.failure_count == 1
        assert len(result.reasons) == 1
        assert len(result.recommendations) == 1

    def test_has_overall_failure(self) -> None:
        """Test has_overall_failure property."""
        reason = RejectionReason(
            threshold_name="overall",
            actual_score=0.5,
            required_score=0.7,
        )
        result = RejectionResult(passed=False, reasons=(reason,))
        assert result.has_overall_failure is True
        assert result.has_ambiguity_failure is False
        assert result.has_sop_failure is False

    def test_has_ambiguity_failure(self) -> None:
        """Test has_ambiguity_failure property."""
        reason = RejectionReason(
            threshold_name="ambiguity",
            actual_score=0.4,
            required_score=0.6,
        )
        result = RejectionResult(passed=False, reasons=(reason,))
        assert result.has_ambiguity_failure is True
        assert result.has_overall_failure is False

    def test_has_sop_failure(self) -> None:
        """Test has_sop_failure property."""
        reason = RejectionReason(
            threshold_name="sop",
            actual_score=0.7,
            required_score=0.8,
        )
        result = RejectionResult(passed=False, reasons=(reason,))
        assert result.has_sop_failure is True

    def test_multiple_failures(self) -> None:
        """Test result with multiple failures."""
        reasons = (
            RejectionReason("overall", 0.5, 0.7),
            RejectionReason("ambiguity", 0.4, 0.6),
            RejectionReason("sop", 0.7, 0.8),
        )
        result = RejectionResult(passed=False, reasons=reasons)
        assert result.failure_count == 3
        assert result.has_overall_failure
        assert result.has_ambiguity_failure
        assert result.has_sop_failure

    def test_immutable(self) -> None:
        """Test that RejectionResult is immutable."""
        result = RejectionResult(passed=True)
        with pytest.raises(AttributeError):
            result.passed = False  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test to_dict returns correct structure."""
        reason = RejectionReason("overall", 0.5, 0.7)
        result = RejectionResult(
            passed=False,
            reasons=(reason,),
            recommendations=("Fix issues", "Review document"),
        )
        output = result.to_dict()
        assert output["passed"] is False
        assert output["failure_count"] == 1
        assert len(output["reasons"]) == 1
        assert output["recommendations"] == ["Fix issues", "Review document"]


# =============================================================================
# validate_quality_thresholds Tests
# =============================================================================


class TestValidateQualityThresholds:
    """Tests for validate_quality_thresholds function."""

    def test_all_pass(self) -> None:
        """Test that all scores above thresholds pass."""
        metrics = QualityMetrics(
            ambiguity_score=0.90,
            sop_score=0.95,
            extraction_score=0.85,
            overall_score=0.90,
        )
        result = validate_quality_thresholds(metrics)
        assert result.passed is True
        assert result.failure_count == 0

    def test_overall_fail(self) -> None:
        """Test that below-threshold overall score fails."""
        metrics = QualityMetrics(
            ambiguity_score=0.90,
            sop_score=0.95,
            extraction_score=0.50,
            overall_score=0.65,
        )
        result = validate_quality_thresholds(metrics)
        assert result.passed is False
        assert result.has_overall_failure
        assert not result.has_ambiguity_failure
        assert not result.has_sop_failure

    def test_ambiguity_fail(self) -> None:
        """Test that below-threshold ambiguity score fails."""
        metrics = QualityMetrics(
            ambiguity_score=0.50,
            sop_score=0.95,
            extraction_score=0.85,
            overall_score=0.80,
        )
        result = validate_quality_thresholds(metrics)
        assert result.passed is False
        assert result.has_ambiguity_failure
        assert not result.has_overall_failure

    def test_sop_fail(self) -> None:
        """Test that below-threshold SOP score fails."""
        metrics = QualityMetrics(
            ambiguity_score=0.90,
            sop_score=0.70,
            extraction_score=0.85,
            overall_score=0.80,
        )
        result = validate_quality_thresholds(metrics)
        assert result.passed is False
        assert result.has_sop_failure
        assert not result.has_ambiguity_failure

    def test_all_fail(self) -> None:
        """Test that all below-threshold scores fail."""
        metrics = QualityMetrics(
            ambiguity_score=0.40,
            sop_score=0.50,
            extraction_score=0.30,
            overall_score=0.40,
        )
        result = validate_quality_thresholds(metrics)
        assert result.passed is False
        assert result.failure_count == 3
        assert result.has_overall_failure
        assert result.has_ambiguity_failure
        assert result.has_sop_failure

    def test_exact_threshold_passes(self) -> None:
        """Test that exact threshold values pass."""
        thresholds = QualityThreshold(overall=0.70, ambiguity=0.60, sop=0.80)
        metrics = QualityMetrics(
            ambiguity_score=0.60,
            sop_score=0.80,
            extraction_score=0.70,
            overall_score=0.70,
        )
        result = validate_quality_thresholds(metrics, thresholds)
        assert result.passed is True

    def test_just_below_threshold_fails(self) -> None:
        """Test that just below threshold fails."""
        thresholds = QualityThreshold(overall=0.70)
        metrics = QualityMetrics(
            ambiguity_score=0.90,
            sop_score=0.90,
            extraction_score=0.90,
            overall_score=0.6999,
        )
        result = validate_quality_thresholds(metrics, thresholds)
        assert result.passed is False
        assert result.has_overall_failure

    def test_custom_thresholds(self) -> None:
        """Test with custom threshold values."""
        strict = QualityThreshold(overall=0.95, ambiguity=0.90, sop=0.95)
        metrics = QualityMetrics(
            ambiguity_score=0.85,
            sop_score=0.90,
            extraction_score=0.90,
            overall_score=0.88,
        )
        result = validate_quality_thresholds(metrics, strict)
        assert result.passed is False
        assert result.failure_count == 3  # overall, ambiguity, and sop (0.90 < 0.95)

    def test_default_thresholds_when_none(self) -> None:
        """Test that None thresholds use defaults."""
        metrics = QualityMetrics(
            ambiguity_score=0.70,
            sop_score=0.85,
            extraction_score=0.80,
            overall_score=0.75,
        )
        result = validate_quality_thresholds(metrics, None)
        assert result.passed is True


# =============================================================================
# generate_remediation_steps Tests
# =============================================================================


class TestGenerateRemediationSteps:
    """Tests for generate_remediation_steps function."""

    def _create_parse_result(
        self,
        *,
        ambiguities: tuple[Ambiguity, ...] = (),
        sop_validation: SOPValidationResult | None = None,
    ) -> SeedParseResult:
        """Helper to create a SeedParseResult with ambiguities/conflicts."""
        return SeedParseResult(
            goals=(SeedGoal(title="G1", description="D1", priority=1),),
            features=(SeedFeature(name="F1", description="D1"),),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            ambiguities=ambiguities,
            ambiguity_confidence=0.85 if ambiguities else 1.0,
            sop_validation=sop_validation,
        )

    def test_ambiguity_failure_steps(self) -> None:
        """Test remediation steps for ambiguity failures."""
        # Create ambiguities
        ambiguities = (
            Ambiguity(
                ambiguity_type=AmbiguityType.SCOPE,
                severity=AmbiguitySeverity.HIGH,
                source_text="test",
                location="line 1",
                description="test",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.TECHNICAL,
                severity=AmbiguitySeverity.MEDIUM,
                source_text="test",
                location="line 2",
                description="test",
            ),
        )
        result = self._create_parse_result(ambiguities=ambiguities)
        report = generate_validation_report(result)

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("ambiguity", 0.5, 0.6),),
        )

        steps = generate_remediation_steps(rejection, report)
        assert any("high-severity" in s.lower() for s in steps)
        assert any("medium-severity" in s.lower() for s in steps)

    def test_sop_failure_steps(self) -> None:
        """Test remediation steps for SOP failures."""
        # Create SOP conflicts
        constraint = SOPConstraint(
            id="test-001",
            rule_text="Rule 1",
            category=SOPCategory.ARCHITECTURE,
            source="test",
            severity=ConflictSeverity.HARD,
        )
        conflict = SOPConflict(
            constraint=constraint,
            seed_text="Conflict text",
            severity=ConflictSeverity.HARD,
            description="Test conflict",
        )
        sop_result = SOPValidationResult(conflicts=[conflict], passed=False)
        result = self._create_parse_result(sop_validation=sop_result)
        report = generate_validation_report(result)

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("sop", 0.7, 0.8),),
        )

        steps = generate_remediation_steps(rejection, report)
        assert any("hard sop" in s.lower() for s in steps)

    def test_overall_only_failure_steps(self) -> None:
        """Test remediation for overall-only failures (extraction issue)."""
        result = self._create_parse_result()
        report = generate_validation_report(result)

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("overall", 0.6, 0.7),),
        )

        steps = generate_remediation_steps(rejection, report)
        assert any("clarity" in s.lower() or "structure" in s.lower() for s in steps)

    def test_always_includes_review_step(self) -> None:
        """Test that review step is always included."""
        result = self._create_parse_result()
        report = generate_validation_report(result)

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("overall", 0.6, 0.7),),
        )

        steps = generate_remediation_steps(rejection, report)
        assert any("review" in s.lower() for s in steps)

    def test_edge_case_no_specific_failures(self) -> None:
        """Test edge case where rejection has no specific failure reasons.

        This covers the defensive 'else' branch in generate_remediation_steps
        that handles the case where a RejectionResult is passed but none of
        the has_*_failure properties are True (shouldn't happen but handled).
        """
        result = self._create_parse_result()
        report = generate_validation_report(result)

        # Create rejection with passed=False but a reason that doesn't match
        # any of the known threshold types (edge case)
        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("unknown_type", 0.5, 0.7),),
        )

        steps = generate_remediation_steps(rejection, report)
        # Should still return at least one step (the fallback)
        assert len(steps) >= 1
        # Should include generic guidance since no specific failures matched
        assert any("review" in s.lower() or "address" in s.lower() for s in steps)


# =============================================================================
# create_rejection_with_remediation Tests
# =============================================================================


class TestCreateRejectionWithRemediation:
    """Tests for create_rejection_with_remediation function."""

    def test_passing_result_no_recommendations(self) -> None:
        """Test that passing results have no recommendations."""
        result = SeedParseResult(
            goals=(SeedGoal(title="G1", description="D1", priority=1),),
            features=(SeedFeature(name="F1", description="D1"),),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
        )
        report = generate_validation_report(result)
        metrics = report.quality_metrics

        rejection = create_rejection_with_remediation(metrics, report)
        assert rejection.passed is True
        assert len(rejection.recommendations) == 0

    def test_failing_result_has_recommendations(self) -> None:
        """Test that failing results have recommendations."""
        # Create result with ambiguities for low score
        ambiguities = tuple(
            Ambiguity(
                ambiguity_type=AmbiguityType.SCOPE,
                severity=AmbiguitySeverity.HIGH,
                source_text=f"test {i}",
                location=f"line {i}",
                description=f"test {i}",
            )
            for i in range(5)
        )
        result = SeedParseResult(
            goals=(SeedGoal(title="G1", description="D1", priority=1),),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            ambiguities=ambiguities,
            ambiguity_confidence=0.3,
        )
        report = generate_validation_report(result)

        # Use strict thresholds to force failure
        thresholds = QualityThreshold(ambiguity=0.9)
        rejection = create_rejection_with_remediation(report.quality_metrics, report, thresholds)

        assert rejection.passed is False
        assert len(rejection.recommendations) > 0


# =============================================================================
# CLI Display Functions Tests
# =============================================================================


class TestCLIDisplayFunctions:
    """Tests for CLI display functions (Story 4.7).

    Tests _display_rejection and _display_threshold_warning functions
    to ensure they output the expected content.
    """

    def test_display_rejection_shows_failed_thresholds(self) -> None:
        """Test that _display_rejection shows failed thresholds."""
        from rich.console import Console

        from yolo_developer.cli.commands.seed import _display_rejection

        # Create a rejection result with failures
        reasons = (
            RejectionReason("overall", 0.52, 0.70),
            RejectionReason("ambiguity", 0.45, 0.60),
        )
        rejection = RejectionResult(
            passed=False,
            reasons=reasons,
            recommendations=("Fix high-severity ambiguities", "Review document"),
        )

        # Capture console output
        output = StringIO()
        console = Console(file=output, force_terminal=True)

        with patch("yolo_developer.cli.commands.seed.console", console):
            _display_rejection(rejection)

        result = output.getvalue()
        assert "Seed Rejected" in result
        assert "Overall" in result
        assert "Ambiguity" in result
        assert "0.52" in result or "0.70" in result  # Score values
        assert "--force" in result  # Tip about force flag

    def test_display_rejection_shows_remediation(self) -> None:
        """Test that _display_rejection shows remediation steps."""
        from rich.console import Console

        from yolo_developer.cli.commands.seed import _display_rejection

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("overall", 0.5, 0.7),),
            recommendations=("Step 1: Do this", "Step 2: Do that"),
        )

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        with patch("yolo_developer.cli.commands.seed.console", console):
            _display_rejection(rejection)

        result = output.getvalue()
        assert "Remediation" in result
        assert "Step 1" in result or "1." in result

    def test_display_threshold_warning_shows_bypass_message(self) -> None:
        """Test that _display_threshold_warning shows bypass warning."""
        from rich.console import Console

        from yolo_developer.cli.commands.seed import _display_threshold_warning

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("sop", 0.75, 0.80),),
            recommendations=("Review soft conflicts",),
        )

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        with patch("yolo_developer.cli.commands.seed.console", console):
            _display_threshold_warning(rejection)

        result = output.getvalue()
        assert "Warning" in result or "Bypass" in result
        assert "--force" in result or "force" in result.lower()
        assert "Sop" in result

    def test_display_threshold_warning_shows_recommendations(self) -> None:
        """Test that _display_threshold_warning shows recommendations."""
        from rich.console import Console

        from yolo_developer.cli.commands.seed import _display_threshold_warning

        rejection = RejectionResult(
            passed=False,
            reasons=(RejectionReason("overall", 0.6, 0.7),),
            recommendations=("Improve clarity", "Add constraints"),
        )

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        with patch("yolo_developer.cli.commands.seed.console", console):
            _display_threshold_warning(rejection)

        result = output.getvalue()
        # Should show recommendations as suggestions
        assert "Improve" in result or "clarity" in result or "production" in result.lower()


class TestLoadThresholdsFromConfig:
    """Tests for _load_thresholds_from_config function (Story 4.7)."""

    def test_returns_defaults_when_file_not_found(self) -> None:
        """Test that defaults are returned when config file not found."""
        from yolo_developer.cli.commands.seed import _load_thresholds_from_config

        # Mock load_config to raise FileNotFoundError
        # Patch at the source module since it's imported inside the function
        with patch(
            "yolo_developer.config.load_config",
            side_effect=FileNotFoundError("No config"),
        ):
            thresholds = _load_thresholds_from_config()

        # Should return defaults
        assert thresholds.overall == DEFAULT_OVERALL_THRESHOLD
        assert thresholds.ambiguity == DEFAULT_AMBIGUITY_THRESHOLD
        assert thresholds.sop == DEFAULT_SOP_THRESHOLD

    def test_returns_defaults_when_configuration_error(self) -> None:
        """Test that defaults are returned when config is invalid."""
        from yolo_developer.cli.commands.seed import _load_thresholds_from_config
        from yolo_developer.config import ConfigurationError

        # Mock load_config to raise ConfigurationError
        with patch(
            "yolo_developer.config.load_config",
            side_effect=ConfigurationError("Invalid config"),
        ):
            thresholds = _load_thresholds_from_config()

        # Should return defaults
        assert thresholds.overall == DEFAULT_OVERALL_THRESHOLD
        assert thresholds.ambiguity == DEFAULT_AMBIGUITY_THRESHOLD
        assert thresholds.sop == DEFAULT_SOP_THRESHOLD

    def test_returns_config_values_when_available(self) -> None:
        """Test that config values are used when available."""
        from unittest.mock import MagicMock

        from yolo_developer.cli.commands.seed import _load_thresholds_from_config
        from yolo_developer.config.schema import QualityConfig, SeedThresholdConfig

        # Create mock config with custom thresholds
        mock_config = MagicMock()
        mock_config.quality = QualityConfig(
            seed_thresholds=SeedThresholdConfig(
                overall=0.85,
                ambiguity=0.75,
                sop=0.90,
            )
        )

        with patch(
            "yolo_developer.config.load_config",
            return_value=mock_config,
        ):
            thresholds = _load_thresholds_from_config()

        assert thresholds.overall == 0.85
        assert thresholds.ambiguity == 0.75
        assert thresholds.sop == 0.90
