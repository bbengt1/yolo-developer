"""Quality Gate Framework for YOLO Developer.

This module provides the quality gate infrastructure for validating
agent artifacts at handoff boundaries. Gates can be blocking (prevent
execution on failure) or advisory (log warning and continue).

Exports:
    quality_gate: Decorator for wrapping agent nodes with gates.
    GateResult: Result of a gate evaluation.
    GateMode: Enum for blocking/advisory mode.
    GateContext: Context passed to gate evaluators.
    GateEvaluator: Protocol for gate evaluator functions.
    register_evaluator: Register a gate evaluator by name.
    get_evaluator: Get a registered evaluator by name.
    list_evaluators: List all registered gate names.
    clear_evaluators: Clear all registered evaluators.
    resolve_threshold: Resolve threshold from config with priority order.

Example - Basic Gate Usage:
    >>> from yolo_developer.gates import quality_gate, register_evaluator
    >>> from yolo_developer.gates import GateContext, GateResult
    >>>
    >>> # Define a gate evaluator
    >>> async def testability_gate(ctx: GateContext) -> GateResult:
    ...     has_tests = "test_coverage" in ctx.state
    ...     return GateResult(
    ...         passed=has_tests,
    ...         gate_name=ctx.gate_name,
    ...         reason=None if has_tests else "Missing test coverage",
    ...     )
    >>>
    >>> # Register the evaluator
    >>> register_evaluator("testability", testability_gate)
    >>>
    >>> # Use the decorator on an agent node
    >>> @quality_gate("testability", blocking=True)
    ... async def analyst_node(state: dict) -> dict:
    ...     # Agent logic here
    ...     return state

Example - Configurable Thresholds:
    Gates support configurable thresholds via the state config. Use
    resolve_threshold() to get the threshold with priority order:
    gate-specific > global > default.

    >>> from yolo_developer.gates import resolve_threshold
    >>>
    >>> # In a gate evaluator:
    >>> async def my_gate(ctx: GateContext) -> GateResult:
    ...     threshold = resolve_threshold(
    ...         gate_name="my_gate",
    ...         state=ctx.state,
    ...         default=0.80,  # Default threshold (0.0-1.0)
    ...     )
    ...     # Use threshold in evaluation logic
    ...     passed = score >= threshold
    ...     return GateResult(passed=passed, gate_name=ctx.gate_name, reason=None)

    Configuration example (yolo.yaml):

    .. code-block:: yaml

        quality:
          test_coverage_threshold: 0.85  # Global
          confidence_threshold: 0.90     # Global

          # Per-gate overrides
          gate_thresholds:
            testability:
              min_score: 0.80
              blocking: true
            architecture_validation:
              min_score: 0.70
              blocking: false  # Advisory mode
"""

from __future__ import annotations

from yolo_developer.gates.decorator import quality_gate
from yolo_developer.gates.evaluators import (
    GateEvaluator,
    clear_evaluators,
    get_evaluator,
    list_evaluators,
    register_evaluator,
)
from yolo_developer.gates.threshold_resolver import resolve_threshold
from yolo_developer.gates.types import GateContext, GateMode, GateResult

__all__ = [
    "GateContext",
    "GateEvaluator",
    "GateMode",
    "GateResult",
    "clear_evaluators",
    "get_evaluator",
    "list_evaluators",
    "quality_gate",
    "register_evaluator",
    "resolve_threshold",
]
