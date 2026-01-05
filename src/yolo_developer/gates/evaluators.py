"""Gate evaluator protocol and registry.

This module defines the GateEvaluator protocol and provides a registry
for managing named evaluators. The registry allows gates to be looked up
by name at runtime.

Example:
    >>> from yolo_developer.gates.evaluators import register_evaluator, get_evaluator
    >>> from yolo_developer.gates.types import GateContext, GateResult
    >>>
    >>> async def my_evaluator(ctx: GateContext) -> GateResult:
    ...     # Evaluate something
    ...     return GateResult(passed=True, gate_name=ctx.gate_name)
    >>>
    >>> register_evaluator("my_gate", my_evaluator)
    >>> evaluator = get_evaluator("my_gate")
    >>> # evaluator is now the registered function

Security Note:
    Gate evaluators should not perform destructive operations.
    They are meant for validation and should only read state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from yolo_developer.gates.types import GateContext, GateResult


@runtime_checkable
class GateEvaluator(Protocol):
    """Protocol for gate evaluator functions.

    Gate evaluators are async callables that receive a GateContext
    and return a GateResult indicating whether the gate passed.

    Example:
        >>> async def testability_evaluator(ctx: GateContext) -> GateResult:
        ...     # Check if tests exist
        ...     has_tests = "tests" in ctx.state
        ...     return GateResult(
        ...         passed=has_tests,
        ...         gate_name=ctx.gate_name,
        ...         reason=None if has_tests else "No tests found",
        ...     )
    """

    def __call__(self, context: GateContext) -> Awaitable[GateResult]:
        """Evaluate the gate.

        Args:
            context: Gate context containing state and metadata.

        Returns:
            Awaitable that resolves to GateResult.
        """
        ...


# Global registry for gate evaluators
_evaluators: dict[str, GateEvaluator] = {}


def register_evaluator(gate_name: str, evaluator: GateEvaluator) -> None:
    """Register a gate evaluator by name.

    Registers an evaluator function that will be called when the
    @quality_gate decorator with the given gate_name is executed.

    Args:
        gate_name: Unique name for the gate.
        evaluator: Async callable implementing GateEvaluator protocol.

    Example:
        >>> async def my_eval(ctx: GateContext) -> GateResult:
        ...     return GateResult(passed=True, gate_name=ctx.gate_name)
        >>> register_evaluator("my_gate", my_eval)
    """
    _evaluators[gate_name] = evaluator


def get_evaluator(gate_name: str) -> GateEvaluator | None:
    """Get a registered evaluator by name.

    Args:
        gate_name: Name of the gate to look up.

    Returns:
        The registered evaluator, or None if not found.

    Example:
        >>> evaluator = get_evaluator("testability")
        >>> if evaluator:
        ...     result = await evaluator(context)
    """
    return _evaluators.get(gate_name)


def list_evaluators() -> list[str]:
    """List all registered gate names.

    Returns:
        List of registered gate names.

    Example:
        >>> names = list_evaluators()
        >>> "testability" in names
        True
    """
    return list(_evaluators.keys())


def clear_evaluators() -> None:
    """Clear all registered evaluators.

    Primarily used for testing to reset state between tests.

    Example:
        >>> clear_evaluators()
        >>> list_evaluators()
        []
    """
    _evaluators.clear()
