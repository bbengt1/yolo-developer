"""Quality gate decorator for agent node functions.

This module provides the @quality_gate decorator that wraps agent node
functions with quality validation. Gates can be blocking (prevent execution)
or advisory (log warning and continue).

Example:
    >>> from yolo_developer.gates.decorator import quality_gate
    >>>
    >>> @quality_gate("testability", blocking=True)
    ... async def analyst_node(state: dict) -> dict:
    ...     # Agent logic here
    ...     return state

Metrics Recording:
    Gate evaluations can be recorded for tracking via set_metrics_store().

    >>> from pathlib import Path
    >>> from yolo_developer.gates.decorator import set_metrics_store
    >>> from yolo_developer.gates.metrics_store import JsonGateMetricsStore
    >>>
    >>> store = JsonGateMetricsStore(base_path=Path(".yolo"))
    >>> set_metrics_store(store)  # Enables metrics recording

Security Note:
    Gate evaluation should not modify state destructively.
    The decorator copies state to prevent accidental mutations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from yolo_developer.gates.evaluators import get_evaluator
from yolo_developer.gates.metrics_types import GateMetricRecord
from yolo_developer.gates.types import GateContext, GateResult

if TYPE_CHECKING:
    from yolo_developer.gates.metrics_store import GateMetricsStore

logger = structlog.get_logger(__name__)

# Module-level metrics store (optional)
_metrics_store: GateMetricsStore | None = None

# Background tasks set to prevent garbage collection
_background_tasks: set[asyncio.Task[None]] = set()


def set_metrics_store(store: GateMetricsStore | None) -> None:
    """Set the metrics store for recording gate evaluations.

    When a store is set, all gate evaluations will be recorded
    for metrics tracking. Set to None to disable recording.

    Args:
        store: The metrics store to use, or None to disable.

    Example:
        >>> from pathlib import Path
        >>> from yolo_developer.gates.metrics_store import JsonGateMetricsStore
        >>> store = JsonGateMetricsStore(base_path=Path(".yolo"))
        >>> set_metrics_store(store)
    """
    global _metrics_store
    _metrics_store = store
    if store is not None:
        logger.info("Metrics store configured for gate tracking")
    else:
        logger.debug("Metrics store disabled")


def get_metrics_store() -> GateMetricsStore | None:
    """Get the currently configured metrics store.

    Returns:
        The configured metrics store, or None if not set.
    """
    return _metrics_store


# Type variable for the decorated function
F = TypeVar("F", bound=Callable[..., Awaitable[dict[str, Any]]])


def quality_gate(gate_name: str, blocking: bool = True) -> Callable[[F], F]:
    """Decorator to wrap agent nodes with quality gate validation.

    Evaluates a quality gate before allowing the node function to execute.
    In blocking mode, gate failures prevent execution. In advisory mode,
    failures log a warning but allow execution to continue.

    Args:
        gate_name: Name of the gate to evaluate (must be registered).
        blocking: If True, gate failure prevents node execution.
                  If False, gate failure logs warning and continues.

    Returns:
        Decorator function that wraps the node.

    Example:
        >>> @quality_gate("testability", blocking=True)
        ... async def analyst_node(state: dict) -> dict:
        ...     # This only runs if testability gate passes
        ...     return state

        >>> @quality_gate("style_check", blocking=False)
        ... async def dev_node(state: dict) -> dict:
        ...     # Runs even if style_check fails (advisory)
        ...     return state
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            # Create a working copy of state to avoid mutating input
            working_state = state.copy()

            # Initialize gate tracking lists if not present
            if "gate_results" not in working_state:
                working_state["gate_results"] = []
            if "advisory_warnings" not in working_state:
                working_state["advisory_warnings"] = []

            # Get the evaluator for this gate
            evaluator = get_evaluator(gate_name)

            if evaluator is None:
                # No evaluator registered - log warning and pass by default
                logger.warning(
                    "No evaluator registered for gate",
                    gate_name=gate_name,
                    action="passing_by_default",
                )
                # Create a pass result for audit trail
                result = GateResult(
                    passed=True,
                    gate_name=gate_name,
                    reason="No evaluator registered - passed by default",
                )
            else:
                # Create context for evaluation
                context = GateContext(
                    state=working_state,
                    gate_name=gate_name,
                )

                # Run the gate evaluation
                result = await evaluator(context)

            # Log the gate result
            _log_gate_result(gate_name, result, blocking)

            # Record metric for tracking (non-blocking)
            _record_metric(
                gate_name=gate_name,
                result=result,
                working_state=working_state,
            )

            # Record result in state for audit trail
            working_state["gate_results"].append(result.to_dict())

            # Handle gate failure based on mode
            if not result.passed:
                if blocking:
                    # Blocking mode: prevent execution
                    logger.error(
                        "Gate blocked execution",
                        gate_name=gate_name,
                        reason=result.reason,
                    )
                    working_state["gate_blocked"] = True
                    working_state["gate_failure"] = result.reason
                    return working_state

                # Advisory mode: log warning and continue
                logger.warning(
                    "Advisory gate failure",
                    gate_name=gate_name,
                    reason=result.reason,
                )
                working_state["advisory_warnings"].append(
                    {
                        "gate_name": gate_name,
                        "reason": result.reason,
                        "timestamp": result.timestamp.isoformat(),
                    }
                )

            # Gate passed (or advisory) - execute the node
            return await func(working_state)

        # Attach gate metadata to wrapper for introspection
        wrapper._gate_name = gate_name  # type: ignore[attr-defined]
        wrapper._gate_blocking = blocking  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def _log_gate_result(gate_name: str, result: GateResult, blocking: bool) -> None:
    """Log gate evaluation result for audit trail.

    Logs structured information about the gate evaluation including
    gate name, pass/fail status, timestamp, and reason.

    Args:
        gate_name: Name of the evaluated gate.
        result: The gate evaluation result.
        blocking: Whether the gate was in blocking mode.
    """
    log_data = {
        "gate_name": gate_name,
        "passed": result.passed,
        "blocking": blocking,
        "timestamp": result.timestamp.isoformat(),
    }

    if result.reason:
        log_data["reason"] = result.reason

    if result.passed:
        logger.info("Gate evaluation passed", **log_data)
    else:
        if blocking:
            logger.error("Gate evaluation failed (blocking)", **log_data)
        else:
            logger.warning("Gate evaluation failed (advisory)", **log_data)


def _record_metric(
    gate_name: str,
    result: GateResult,
    working_state: dict[str, Any],
) -> None:
    """Record gate evaluation metric for tracking.

    Creates a metric record and stores it asynchronously without
    blocking the gate evaluation flow.

    Args:
        gate_name: Name of the evaluated gate.
        result: The gate evaluation result.
        working_state: Current workflow state dictionary.
    """
    if _metrics_store is None:
        return

    # Create metric record
    # Use actual score/threshold from GateResult if available, fallback to binary
    score = result.score if result.score is not None else (1.0 if result.passed else 0.0)
    threshold = result.threshold if result.threshold is not None else 0.0
    metric_record = GateMetricRecord(
        gate_name=gate_name,
        passed=result.passed,
        score=score,
        threshold=threshold,
        timestamp=datetime.now(timezone.utc),
        agent_name=working_state.get("current_agent"),
        sprint_id=working_state.get("sprint_id"),
    )

    # Fire and forget - don't block on metric recording
    # Store task reference to prevent garbage collection (RUF006)
    task = asyncio.create_task(_async_record_metric(metric_record))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.debug(
        "Gate metric recorded",
        gate_name=gate_name,
        passed=result.passed,
        score=metric_record.score,
    )


async def _async_record_metric(record: GateMetricRecord) -> None:
    """Asynchronously record a metric to the store.

    This is called via asyncio.create_task() to avoid blocking.

    Args:
        record: The metric record to store.
    """
    if _metrics_store is None:
        return

    try:
        await _metrics_store.record_evaluation(record)
    except Exception as e:
        logger.error(
            "Failed to record gate metric",
            gate_name=record.gate_name,
            error=str(e),
        )
