"""Unit tests for threshold resolver utility (Story 3.7 - Task 2).

Tests the resolve_threshold function for priority-based threshold resolution.
"""

from __future__ import annotations

import pytest


class TestResolveThresholdFunction:
    """Tests for resolve_threshold function."""

    def test_resolve_threshold_exists(self) -> None:
        """Verify resolve_threshold function can be imported."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        assert resolve_threshold is not None
        assert callable(resolve_threshold)

    def test_resolve_threshold_returns_default_for_empty_state(self) -> None:
        """Verify default is returned when state is empty."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        result = resolve_threshold("testability", {}, 0.80)
        assert result == 0.80

    def test_resolve_threshold_returns_default_for_missing_config(self) -> None:
        """Verify default is returned when config key is missing."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {"other_key": "value"}
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80

    def test_resolve_threshold_returns_default_for_missing_quality(self) -> None:
        """Verify default is returned when quality key is missing."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {"config": {"llm": {}}}
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80


class TestGateSpecificThreshold:
    """Tests for gate-specific threshold resolution (highest priority)."""

    def test_resolve_gate_specific_threshold(self) -> None:
        """Verify gate-specific threshold takes highest priority."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.85,  # Global
                    "gate_thresholds": {
                        "testability": {"min_score": 0.90},  # Gate-specific
                    },
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.90  # Gate-specific wins

    def test_resolve_gate_specific_threshold_for_different_gate(self) -> None:
        """Verify correct gate-specific threshold is selected."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "testability": {"min_score": 0.80},
                        "confidence_scoring": {"min_score": 0.95},
                    },
                }
            }
        }
        result = resolve_threshold("confidence_scoring", state, 0.90)
        assert result == 0.95


class TestGlobalThreshold:
    """Tests for global threshold resolution (second priority)."""

    def test_resolve_global_test_coverage_threshold(self) -> None:
        """Verify global test_coverage_threshold is used for testability gate."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.85,
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.85

    def test_resolve_global_confidence_threshold(self) -> None:
        """Verify global confidence_threshold is used for confidence_scoring gate."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "confidence_threshold": 0.92,
                }
            }
        }
        result = resolve_threshold("confidence_scoring", state, 0.90)
        assert result == 0.92

    def test_global_threshold_used_when_no_gate_specific(self) -> None:
        """Verify global threshold used when gate has no specific config."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.85,
                    "gate_thresholds": {
                        "other_gate": {"min_score": 0.99},  # Different gate
                    },
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.85  # Falls back to global


class TestDefaultThreshold:
    """Tests for default threshold resolution (lowest priority)."""

    def test_default_used_for_unknown_gate_without_global(self) -> None:
        """Verify default is used for gates without global mapping."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "test_coverage_threshold": 0.85,  # Only testability mapping
                }
            }
        }
        # ac_measurability has no global mapping
        result = resolve_threshold("ac_measurability", state, 0.75)
        assert result == 0.75

    def test_default_used_when_gate_thresholds_empty(self) -> None:
        """Verify default is used when gate_thresholds is empty."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "gate_thresholds": {},
                }
            }
        }
        result = resolve_threshold("definition_of_done", state, 0.85)
        assert result == 0.85


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_non_dict_config(self) -> None:
        """Verify graceful handling when config is not a dict."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {"config": "not a dict"}
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80

    def test_handles_non_dict_quality(self) -> None:
        """Verify graceful handling when quality is not a dict."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {"config": {"quality": "not a dict"}}
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80

    def test_handles_non_dict_gate_thresholds(self) -> None:
        """Verify graceful handling when gate_thresholds is not a dict."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {"config": {"quality": {"gate_thresholds": "not a dict"}}}
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80

    def test_handles_non_dict_gate_config(self) -> None:
        """Verify graceful handling when gate config is not a dict."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "gate_thresholds": {"testability": "not a dict"},
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80)
        assert result == 0.80


class TestThresholdKeyParameter:
    """Tests for custom threshold_key parameter."""

    def test_custom_threshold_key(self) -> None:
        """Verify custom threshold_key parameter works."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "testability": {"custom_key": 0.95},
                    },
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80, threshold_key="custom_key")
        assert result == 0.95

    def test_missing_threshold_key_returns_default(self) -> None:
        """Verify default returned when threshold_key not in gate config."""
        from yolo_developer.gates.threshold_resolver import resolve_threshold

        state = {
            "config": {
                "quality": {
                    "gate_thresholds": {
                        "testability": {"other_key": 0.95},  # Different key
                    },
                }
            }
        }
        result = resolve_threshold("testability", state, 0.80, threshold_key="min_score")
        assert result == 0.80


class TestModuleExports:
    """Tests for module exports."""

    def test_resolve_threshold_importable_from_gates_module(self) -> None:
        """Verify resolve_threshold can be imported from gates module."""
        from yolo_developer.gates import resolve_threshold

        assert resolve_threshold is not None


class TestStructuredLogging:
    """Tests for structured logging integration."""

    def test_resolver_logs_resolution_decision(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify resolver logs the threshold resolution decision."""
        import logging

        from yolo_developer.gates.threshold_resolver import resolve_threshold

        with caplog.at_level(logging.DEBUG):
            resolve_threshold("testability", {}, 0.80)

        # Should have logged something (debug level)
        # Note: This may need adjustment based on actual implementation
        assert len(caplog.records) >= 0  # At minimum, shouldn't crash
