"""Unit tests for per-gate quality threshold configuration (Story 3.7 - Task 1).

Tests the GateThreshold model and extended QualityConfig with gate_thresholds field.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestGateThreshold:
    """Tests for GateThreshold nested model (Task 1)."""

    def test_gate_threshold_exists(self) -> None:
        """Verify GateThreshold class exists and can be imported."""
        from yolo_developer.config.schema import GateThreshold

        assert GateThreshold is not None

    def test_gate_threshold_default_min_score(self) -> None:
        """Verify GateThreshold has min_score with 0.80 default."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold()
        assert hasattr(config, "min_score")
        assert config.min_score == 0.80

    def test_gate_threshold_default_blocking(self) -> None:
        """Verify GateThreshold has blocking with True default."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold()
        assert hasattr(config, "blocking")
        assert config.blocking is True

    def test_gate_threshold_accepts_custom_min_score(self) -> None:
        """Verify GateThreshold accepts custom min_score value."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold(min_score=0.75)
        assert config.min_score == 0.75

    def test_gate_threshold_accepts_advisory_mode(self) -> None:
        """Verify GateThreshold accepts blocking=False for advisory mode."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold(blocking=False)
        assert config.blocking is False

    def test_gate_threshold_rejects_min_score_above_one(self) -> None:
        """Verify min_score values above 1.0 are rejected."""
        from yolo_developer.config.schema import GateThreshold

        with pytest.raises(ValidationError) as exc_info:
            GateThreshold(min_score=1.5)
        assert "min_score" in str(exc_info.value)

    def test_gate_threshold_rejects_negative_min_score(self) -> None:
        """Verify negative min_score values are rejected."""
        from yolo_developer.config.schema import GateThreshold

        with pytest.raises(ValidationError) as exc_info:
            GateThreshold(min_score=-0.1)
        assert "min_score" in str(exc_info.value)

    def test_gate_threshold_accepts_zero_min_score(self) -> None:
        """Verify min_score of 0.0 is accepted (edge case)."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold(min_score=0.0)
        assert config.min_score == 0.0

    def test_gate_threshold_accepts_one_min_score(self) -> None:
        """Verify min_score of 1.0 is accepted (edge case)."""
        from yolo_developer.config.schema import GateThreshold

        config = GateThreshold(min_score=1.0)
        assert config.min_score == 1.0

    def test_gate_threshold_fields_have_descriptions(self) -> None:
        """Verify GateThreshold fields have Field descriptions."""
        from yolo_developer.config.schema import GateThreshold

        assert GateThreshold.model_fields["min_score"].description is not None
        assert GateThreshold.model_fields["blocking"].description is not None


class TestQualityConfigGateThresholds:
    """Tests for QualityConfig gate_thresholds field (Task 1)."""

    def test_quality_config_has_gate_thresholds_field(self) -> None:
        """Verify QualityConfig has gate_thresholds field."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        assert hasattr(config, "gate_thresholds")

    def test_quality_config_gate_thresholds_default_empty(self) -> None:
        """Verify gate_thresholds defaults to empty dict."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        assert config.gate_thresholds == {}

    def test_quality_config_accepts_gate_thresholds_dict(self) -> None:
        """Verify QualityConfig accepts gate_thresholds dict."""
        from yolo_developer.config.schema import GateThreshold, QualityConfig

        config = QualityConfig(
            gate_thresholds={
                "testability": GateThreshold(min_score=0.85),
            }
        )
        assert "testability" in config.gate_thresholds
        assert config.gate_thresholds["testability"].min_score == 0.85

    def test_quality_config_accepts_multiple_gate_thresholds(self) -> None:
        """Verify QualityConfig accepts multiple gate configurations."""
        from yolo_developer.config.schema import GateThreshold, QualityConfig

        config = QualityConfig(
            gate_thresholds={
                "testability": GateThreshold(min_score=0.80),
                "ac_measurability": GateThreshold(min_score=0.75),
                "architecture_validation": GateThreshold(min_score=0.70, blocking=False),
                "definition_of_done": GateThreshold(min_score=0.85),
                "confidence_scoring": GateThreshold(min_score=0.90),
            }
        )
        assert len(config.gate_thresholds) == 5
        assert config.gate_thresholds["architecture_validation"].blocking is False

    def test_quality_config_gate_thresholds_has_description(self) -> None:
        """Verify gate_thresholds field has description."""
        from yolo_developer.config.schema import QualityConfig

        assert QualityConfig.model_fields["gate_thresholds"].description is not None


class TestQualityConfigValidateThresholds:
    """Tests for QualityConfig.validate_thresholds() method (Task 1)."""

    def test_validate_thresholds_method_exists(self) -> None:
        """Verify QualityConfig has validate_thresholds method."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        assert hasattr(config, "validate_thresholds")
        assert callable(config.validate_thresholds)

    def test_validate_thresholds_returns_empty_for_valid_config(self) -> None:
        """Verify validate_thresholds returns empty list for valid config."""
        from yolo_developer.config.schema import GateThreshold, QualityConfig

        config = QualityConfig(
            gate_thresholds={
                "testability": GateThreshold(min_score=0.80),
            }
        )
        errors = config.validate_thresholds()
        assert errors == []

    def test_validate_thresholds_returns_empty_for_empty_gate_thresholds(self) -> None:
        """Verify validate_thresholds returns empty list for empty gate_thresholds."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        errors = config.validate_thresholds()
        assert errors == []


class TestGateThresholdImports:
    """Tests for GateThreshold module exports."""

    def test_gate_threshold_importable_from_config_module(self) -> None:
        """Verify GateThreshold can be imported from yolo_developer.config."""
        from yolo_developer.config import GateThreshold

        assert GateThreshold is not None

    def test_gate_threshold_importable_from_schema(self) -> None:
        """Verify GateThreshold can be imported from schema module."""
        from yolo_developer.config.schema import GateThreshold

        assert GateThreshold is not None


class TestYoloConfigGateThresholdsEnvVars:
    """Tests for gate thresholds environment variable overrides."""

    def test_env_override_gate_threshold_min_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_QUALITY__GATE_THRESHOLDS__testability__MIN_SCORE env var works."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        # Note: Pydantic Settings nested dict env vars use JSON format
        monkeypatch.setenv(
            "YOLO_QUALITY__GATE_THRESHOLDS",
            '{"testability": {"min_score": 0.85, "blocking": true}}',
        )
        config = YoloConfig()
        assert "testability" in config.quality.gate_thresholds
        assert config.quality.gate_thresholds["testability"].min_score == 0.85

    def test_env_override_multiple_gate_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify multiple gate thresholds can be set via env var."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv(
            "YOLO_QUALITY__GATE_THRESHOLDS",
            '{"testability": {"min_score": 0.80}, "confidence_scoring": {"min_score": 0.95}}',
        )
        config = YoloConfig()
        assert len(config.quality.gate_thresholds) == 2
        assert config.quality.gate_thresholds["confidence_scoring"].min_score == 0.95
