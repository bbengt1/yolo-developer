"""Threshold resolver utility for quality gates (Story 3.7 - Task 2).

This module provides centralized threshold resolution with priority-based lookup:
1. Gate-specific threshold (highest priority)
2. Global threshold (from quality config)
3. Default value (lowest priority)

Example:
    >>> from yolo_developer.gates.threshold_resolver import resolve_threshold
    >>> state = {
    ...     "config": {
    ...         "quality": {
    ...             "test_coverage_threshold": 0.85,
    ...             "gate_thresholds": {"testability": {"min_score": 0.90}},
    ...         }
    ...     }
    ... }
    >>> resolve_threshold("testability", state, default=0.80)
    0.90  # Gate-specific wins
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# Global threshold key mapping: gate_name -> quality config key
GLOBAL_THRESHOLD_MAPPING: dict[str, str] = {
    "testability": "test_coverage_threshold",
    "confidence_scoring": "confidence_threshold",
    "architecture_validation": "architecture_threshold",
    "definition_of_done": "dod_threshold",
    "ac_measurability": "ac_measurability_threshold",
}


def resolve_threshold(
    gate_name: str,
    state: dict[str, Any],
    default: float,
    threshold_key: str = "min_score",
) -> float:
    """Resolve threshold value with priority: gate-specific → global → default.

    Args:
        gate_name: Name of the gate (e.g., "testability", "confidence_scoring")
        state: State dict containing optional "config" key with quality settings
        default: Default threshold if not configured
        threshold_key: Key within gate config (default "min_score")

    Returns:
        Resolved threshold value (0.0-1.0)

    Example:
        >>> state = {"config": {"quality": {"test_coverage_threshold": 0.85}}}
        >>> resolve_threshold("testability", state, 0.80)
        0.85
    """
    config = state.get("config", {})
    if not isinstance(config, dict):
        logger.debug(
            "threshold_resolution",
            gate_name=gate_name,
            source="default",
            reason="config not a dict",
            threshold=default,
        )
        return default

    quality = config.get("quality", {})
    if not isinstance(quality, dict):
        logger.debug(
            "threshold_resolution",
            gate_name=gate_name,
            source="default",
            reason="quality not a dict",
            threshold=default,
        )
        return default

    # 1. Check gate-specific threshold (highest priority)
    gate_thresholds = quality.get("gate_thresholds", {})
    if isinstance(gate_thresholds, dict):
        gate_config = gate_thresholds.get(gate_name, {})
        if isinstance(gate_config, dict) and threshold_key in gate_config:
            threshold: float = float(gate_config[threshold_key])
            logger.debug(
                "threshold_resolution",
                gate_name=gate_name,
                source="gate_specific",
                threshold_key=threshold_key,
                threshold=threshold,
            )
            return threshold

    # 2. Check global threshold (map gate name to global key)
    global_key = GLOBAL_THRESHOLD_MAPPING.get(gate_name)
    if global_key and global_key in quality:
        threshold = float(quality[global_key])
        logger.debug(
            "threshold_resolution",
            gate_name=gate_name,
            source="global",
            global_key=global_key,
            threshold=threshold,
        )
        return threshold

    # 3. Return default (lowest priority)
    logger.debug(
        "threshold_resolution",
        gate_name=gate_name,
        source="default",
        threshold=default,
    )
    return default
