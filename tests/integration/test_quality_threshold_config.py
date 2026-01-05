"""Integration tests for quality threshold configuration (Story 3.7).

Tests the full integration of configurable quality thresholds across gates,
including per-gate overrides, fallback to defaults, and YoloConfig integration.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.types import GateContext


class TestGatesUsingConfiguredThresholds:
    """Test that gates correctly use configured thresholds (AC: 1)."""

    @pytest.mark.asyncio
    async def test_testability_uses_configured_threshold(self) -> None:
        """Testability gate should use gate-specific threshold from config."""
        from yolo_developer.gates.gates.testability import testability_evaluator

        # Create context with gate-specific threshold configuration
        context = GateContext(
            state={
                "requirements": [
                    {"id": "req-1", "content": "API responds in 500ms"},  # Measurable
                    {"id": "req-2", "content": "System must be fast"},  # Vague
                ],
                "config": {
                    "quality": {
                        "gate_thresholds": {
                            "testability": {"min_score": 0.50},  # 50% threshold
                        }
                    }
                },
            },
            gate_name="testability",
        )

        result = await testability_evaluator(context)
        # 1 of 2 requirements is testable = 50%, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_confidence_scoring_uses_configured_threshold(self) -> None:
        """Confidence scoring gate should use gate-specific threshold from config."""
        from yolo_developer.gates.gates.confidence_scoring import (
            confidence_scoring_evaluator,
        )

        # Create context with gate-specific threshold configuration
        context = GateContext(
            state={
                "code": {
                    "files": [
                        {"path": "src/module.py", "content": "def func(): pass"},
                        {"path": "tests/test_module.py", "content": "def test_func(): pass"},
                    ]
                },
                "gate_results": [
                    {"gate_name": "testability", "passed": True},
                    {"gate_name": "ac_measurability", "passed": True},
                ],
                "config": {
                    "quality": {
                        "gate_thresholds": {
                            "confidence_scoring": {"min_score": 0.50},  # 50% threshold
                        }
                    }
                },
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)
        # Should pass with 50% threshold
        assert result.passed is True


class TestPerGateThresholdOverrides:
    """Test per-gate threshold overrides work correctly (AC: 4)."""

    @pytest.mark.asyncio
    async def test_gate_specific_threshold_overrides_global(self) -> None:
        """Gate-specific threshold should override global threshold."""
        from yolo_developer.gates.gates.testability import testability_evaluator

        # Config has both global and gate-specific thresholds
        context = GateContext(
            state={
                "requirements": [
                    {"id": "req-1", "content": "API responds in 500ms"},  # Measurable
                ],
                "config": {
                    "quality": {
                        "test_coverage_threshold": 0.95,  # Global: 95% (would fail)
                        "gate_thresholds": {
                            "testability": {"min_score": 0.80},  # Gate-specific: 80% (should pass)
                        },
                    }
                },
            },
            gate_name="testability",
        )

        result = await testability_evaluator(context)
        # Gate-specific 80% threshold should be used, not global 95%
        # 1/1 = 100% testable, should pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_different_gates_use_different_thresholds(self) -> None:
        """Different gates should use their own configured thresholds."""
        from yolo_developer.gates.gates.definition_of_done import (
            definition_of_done_evaluator,
        )
        from yolo_developer.gates.gates.testability import testability_evaluator

        # Configure different thresholds for different gates
        shared_config = {
            "quality": {
                "gate_thresholds": {
                    "testability": {"min_score": 0.50},  # 50%
                    "definition_of_done": {"min_score": 0.70},  # 70%
                }
            }
        }

        # Test testability gate
        testability_context = GateContext(
            state={
                "requirements": [
                    {"id": "req-1", "content": "API responds in 500ms"},
                    {"id": "req-2", "content": "System must be fast"},  # Vague
                ],
                "config": shared_config,
            },
            gate_name="testability",
        )
        testability_result = await testability_evaluator(testability_context)
        assert testability_result.passed is True  # 50% pass rate meets 50% threshold

        # Test definition_of_done gate
        dod_context = GateContext(
            state={
                "code": {
                    "files": [
                        {
                            "path": "src/module.py",
                            "content": '"""Module."""\n\ndef func(x: int) -> int:\n    """Doc."""\n    return x',
                        },
                        {
                            "path": "tests/test_module.py",
                            "content": '"""Test."""\n\ndef test_func():\n    pass',
                        },
                    ]
                },
                "story": {},
                "config": shared_config,
            },
            gate_name="definition_of_done",
        )
        dod_result = await definition_of_done_evaluator(dod_context)
        # Good code should pass with 70% threshold
        assert dod_result.passed is True


class TestFallbackToDefaults:
    """Test fallback to default thresholds when not configured (AC: 1)."""

    @pytest.mark.asyncio
    async def test_testability_uses_default_when_no_config(self) -> None:
        """Testability gate should use default threshold when no config."""
        from yolo_developer.gates.gates.testability import (
            DEFAULT_TESTABILITY_THRESHOLD,
            testability_evaluator,
        )

        # Context without any config
        context = GateContext(
            state={
                "requirements": [
                    {"id": "req-1", "content": "API responds in 500ms"},  # Measurable
                ],
                # No config key
            },
            gate_name="testability",
        )

        result = await testability_evaluator(context)
        # Should use default threshold (0.80)
        assert result.passed is True
        assert DEFAULT_TESTABILITY_THRESHOLD == 0.80

    @pytest.mark.asyncio
    async def test_uses_global_threshold_when_no_gate_specific(self) -> None:
        """Gate should use global threshold when no gate-specific config."""
        from yolo_developer.gates.gates.confidence_scoring import (
            confidence_scoring_evaluator,
        )

        # Config with global threshold only
        context = GateContext(
            state={
                "code": {
                    "files": [
                        {"path": "src/module.py", "content": "def func(): pass"},
                        {"path": "tests/test_module.py", "content": "def test_func(): pass"},
                    ]
                },
                "gate_results": [
                    {"gate_name": "testability", "passed": True},
                    {"gate_name": "ac_measurability", "passed": True},
                ],
                "config": {
                    "quality": {
                        "confidence_threshold": 0.50,  # Global, no gate-specific
                    }
                },
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)
        # Should use global 50% threshold
        assert result.passed is True
        assert "50" in result.reason


class TestYoloConfigIntegration:
    """Test integration with YoloConfig (AC: 6)."""

    def test_yoloconfig_accepts_gate_thresholds(self) -> None:
        """YoloConfig should accept gate_thresholds in quality config."""
        from yolo_developer.config.schema import GateThreshold, YoloConfig

        config = YoloConfig(
            project_name="test-project",
            quality={
                "test_coverage_threshold": 0.85,
                "confidence_threshold": 0.90,
                "gate_thresholds": {
                    "testability": GateThreshold(min_score=0.80),
                    "architecture_validation": GateThreshold(min_score=0.70, blocking=False),
                },
            },
        )

        assert config.quality.test_coverage_threshold == 0.85
        assert config.quality.confidence_threshold == 0.90
        assert "testability" in config.quality.gate_thresholds
        assert config.quality.gate_thresholds["testability"].min_score == 0.80
        assert config.quality.gate_thresholds["architecture_validation"].blocking is False

    def test_yoloconfig_validates_thresholds(self) -> None:
        """YoloConfig should validate threshold ranges."""
        from pydantic import ValidationError

        from yolo_developer.config.schema import YoloConfig

        # Should raise validation error for threshold > 1.0
        with pytest.raises(ValidationError):
            YoloConfig(
                project_name="test-project",
                quality={"test_coverage_threshold": 1.5},  # Invalid
            )

        # Should raise validation error for negative threshold
        with pytest.raises(ValidationError):
            YoloConfig(
                project_name="test-project",
                quality={"confidence_threshold": -0.5},  # Invalid
            )

    def test_quality_config_validate_thresholds_method(self) -> None:
        """QualityConfig.validate_thresholds() should return errors for invalid values."""
        from yolo_developer.config.schema import GateThreshold, QualityConfig

        # Valid config
        valid_config = QualityConfig(
            gate_thresholds={
                "testability": GateThreshold(min_score=0.80),
            }
        )
        errors = valid_config.validate_thresholds()
        assert errors == []

        # Note: GateThreshold with invalid min_score would raise ValidationError
        # at construction time due to Pydantic ge/le constraints, so we test
        # the validate_thresholds method with valid GateThreshold instances


class TestThresholdResolutionPriority:
    """Test threshold resolution priority order."""

    @pytest.mark.asyncio
    async def test_priority_gate_specific_over_global_over_default(self) -> None:
        """Priority should be: gate-specific > global > default."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        # Test 1: Gate-specific wins over global
        state_with_both = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.90,  # Global
                    "gate_thresholds": {
                        "testability": {"min_score": 0.75},  # Gate-specific
                    },
                }
            }
        }
        result = resolve_threshold("testability", state_with_both, default=0.80)
        assert result == 0.75  # Gate-specific wins

        # Test 2: Global wins over default
        state_with_global = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.90,  # Global (no gate-specific)
                }
            }
        }
        result = resolve_threshold("testability", state_with_global, default=0.80)
        assert result == 0.90  # Global wins

        # Test 3: Default when nothing configured
        state_empty: dict = {}
        result = resolve_threshold("testability", state_empty, default=0.80)
        assert result == 0.80  # Default used
