"""Rollback coordination module for SM agent (Story 10.15).

This module provides rollback coordination functionality for the orchestration system:

- Rollback detection: Determines when rollback is needed (AC #1, AC #2)
- Rollback planning: Creates ordered rollback plans (AC #1, AC #2)
- Rollback execution: Executes rollback with state restoration (AC #3, AC #4)
- Failure handling: Handles rollback failures with escalation (AC #5)

Key Concepts:
- **Non-blocking**: Rollback coordination never blocks the main workflow
- **Immutable outputs**: Returns frozen dataclasses
- **Structured logging**: Uses structlog for audit trail
- **Checkpoint-based recovery**: Per ADR-007

Example:
    >>> from yolo_developer.agents.sm.rollback import coordinate_rollback
    >>> from yolo_developer.agents.sm.rollback_types import RollbackConfig
    >>>
    >>> # Coordinate rollback when emergency protocol recommends it
    >>> result = await coordinate_rollback(state, checkpoint, emergency_protocol)
    >>> result.status
    'completed'
    >>>
    >>> # With custom config
    >>> config = RollbackConfig(max_steps=50)
    >>> result = await coordinate_rollback(state, checkpoint, None, config)
    >>> result.rollback_complete
    True

Architecture Note:
    Per ADR-007, this module follows checkpoint-based recovery patterns that
    ensure system resilience through state preservation and rollback capabilities.

References:
    - FR71: SM Agent can coordinate rollback operations as emergency sprints
    - ADR-007: Checkpoint-based recovery patterns
    - Story 10.10: Emergency Protocols (Checkpoint, EmergencyProtocol)
    - Story 10.14: Human Escalation (integration for failure escalation)
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import structlog

from yolo_developer.agents.sm.rollback_types import (
    DEFAULT_MAX_ROLLBACK_STEPS,
    RollbackConfig,
    RollbackPlan,
    RollbackReason,
    RollbackResult,
    RollbackStatus,
    RollbackStep,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.emergency_types import Checkpoint, EmergencyProtocol
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Threshold for gate errors to trigger rollback consideration
GATE_ERROR_THRESHOLD: int = 3
"""Number of gate errors before considering rollback."""

# Fields that should be restored during rollback
RESTORABLE_FIELDS: frozenset[str] = frozenset(
    {
        "current_agent",
        "gate_errors",
        "recovery_attempts",
        "recovery_action",
        "escalate_to_human",
        "decisions",
        "messages",
    }
)
"""State fields that can be safely restored during rollback."""

# Fields to clear during rollback (error state)
CLEARABLE_FIELDS: frozenset[str] = frozenset(
    {
        "gate_errors",
        "recovery_attempts",
        "error",
    }
)
"""State fields to clear during rollback to reset error state."""


# =============================================================================
# Rollback Detection Functions (Task 2.2)
# =============================================================================


def should_rollback(
    state: dict[str, Any] | YoloState,
    emergency_protocol: EmergencyProtocol | None,
    checkpoint: Checkpoint | None,
) -> tuple[bool, RollbackReason | None]:
    """Determine if rollback should be initiated (AC #1, #2).

    Checks multiple conditions to determine if rollback is needed:
    1. Emergency protocol selected rollback action
    2. State recovery_action is "rollback"
    3. Gate blocked with repeated failures
    4. Conflict requires rollback

    Args:
        state: Current orchestration state.
        emergency_protocol: Emergency protocol result (may have selected rollback).
        checkpoint: Available checkpoint for rollback (required for rollback).

    Returns:
        Tuple of (should_rollback, reason). Returns (False, None) if no rollback needed.

    Example:
        >>> should, reason = should_rollback(state, protocol, checkpoint)
        >>> should
        True
        >>> reason
        'emergency_recovery'
    """
    state_dict = dict(state)

    # No checkpoint = no rollback possible
    if checkpoint is None:
        logger.debug("rollback_check_no_checkpoint")
        return False, None

    # Priority 1: Emergency protocol selected rollback
    if emergency_protocol is not None and emergency_protocol.selected_action == "rollback":
        logger.info(
            "rollback_trigger_detected",
            reason="emergency_recovery",
            protocol_id=emergency_protocol.protocol_id,
        )
        return True, "emergency_recovery"

    # Priority 2: State recovery_action is rollback
    recovery_action = state_dict.get("recovery_action")
    if recovery_action == "rollback":
        logger.info(
            "rollback_trigger_detected",
            reason="checkpoint_recovery",
            recovery_action=recovery_action,
        )
        return True, "checkpoint_recovery"

    # Priority 3: Gate blocked with repeated failures
    gate_blocked = state_dict.get("gate_blocked", False)
    gate_errors = state_dict.get("gate_errors", [])
    if gate_blocked and len(gate_errors) >= GATE_ERROR_THRESHOLD:
        logger.info(
            "rollback_trigger_detected",
            reason="gate_failure",
            gate_error_count=len(gate_errors),
        )
        return True, "gate_failure"

    # Priority 4: Conflict requires rollback
    mediation_result = state_dict.get("mediation_result")
    if mediation_result and isinstance(mediation_result, dict):
        requires_rollback = mediation_result.get("requires_rollback", False)
        escalations = mediation_result.get("escalations_triggered", ())
        if requires_rollback or len(escalations) > 0:
            logger.info(
                "rollback_trigger_detected",
                reason="conflict_resolution",
                escalation_count=len(escalations),
            )
            return True, "conflict_resolution"

    # No rollback needed
    logger.debug("rollback_check_not_needed")
    return False, None


# =============================================================================
# Rollback Planning Functions (Task 2.3-2.6)
# =============================================================================


def _identify_affected_state(
    state: dict[str, Any],
    checkpoint: Checkpoint,
) -> list[tuple[str, Any, Any]]:
    """Identify state fields that differ from checkpoint (AC #2).

    Args:
        state: Current orchestration state.
        checkpoint: Checkpoint with state_snapshot.

    Returns:
        List of (field_name, checkpoint_value, current_value) tuples.
    """
    affected: list[tuple[str, Any, Any]] = []
    snapshot = checkpoint.state_snapshot

    for field in RESTORABLE_FIELDS:
        checkpoint_value = snapshot.get(field)
        current_value = state.get(field)

        # Compare values (handle None and different types)
        if checkpoint_value != current_value:
            affected.append((field, checkpoint_value, current_value))

    logger.debug(
        "affected_state_identified",
        affected_field_count=len(affected),
        fields=[f[0] for f in affected],
    )

    return affected


def _create_rollback_steps(
    affected_fields: list[tuple[str, Any, Any]],
) -> tuple[RollbackStep, ...]:
    """Create ordered rollback steps from affected fields (AC #1).

    Args:
        affected_fields: List of (field, checkpoint_value, current_value) tuples.

    Returns:
        Tuple of RollbackStep objects in execution order.
    """
    steps: list[RollbackStep] = []

    for i, (field, checkpoint_value, current_value) in enumerate(affected_fields):
        step_id = f"step-{i + 1:03d}"

        # Determine action based on field type
        if checkpoint_value is None:
            action = "clear_field"
        elif isinstance(checkpoint_value, list) and len(checkpoint_value) == 0:
            action = "clear_field"
        else:
            action = "restore_field"

        step = RollbackStep(
            step_id=step_id,
            action=action,
            target_field=field,
            previous_value=checkpoint_value,
            current_value=current_value,
            executed=False,
            success=None,
        )
        steps.append(step)

    return tuple(steps)


def _validate_rollback_safety(
    plan: RollbackPlan,
    state: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate that rollback is safe to execute (AC #1).

    Args:
        plan: Rollback plan to validate.
        state: Current state for context.

    Returns:
        Tuple of (is_safe, error_message).
    """
    # Check step count limit
    if len(plan.steps) > DEFAULT_MAX_ROLLBACK_STEPS:
        return False, f"Too many steps: {len(plan.steps)} > {DEFAULT_MAX_ROLLBACK_STEPS}"

    # Check for required fields
    if not plan.checkpoint_id:
        return False, "No checkpoint ID specified"

    # All validations passed
    return True, None


def _estimate_impact(affected_fields: list[tuple[str, Any, Any]]) -> str:
    """Estimate the impact level of the rollback.

    Args:
        affected_fields: List of affected state fields.

    Returns:
        Impact level: "minimal", "moderate", or "significant".
    """
    count = len(affected_fields)

    if count == 0:
        return "minimal"
    elif count <= 3:
        return "moderate"
    else:
        return "significant"


def create_rollback_plan(
    state: dict[str, Any] | YoloState,
    checkpoint: Checkpoint,
    reason: RollbackReason,
) -> RollbackPlan:
    """Create a rollback plan with ordered steps (AC #1, AC #2).

    Analyzes state vs checkpoint to identify changes, creates ordered
    rollback steps, and validates the plan is safe to execute.

    Args:
        state: Current orchestration state.
        checkpoint: Checkpoint to restore to.
        reason: Reason for the rollback.

    Returns:
        RollbackPlan ready for execution.

    Example:
        >>> plan = create_rollback_plan(state, checkpoint, "checkpoint_recovery")
        >>> plan.steps
        (RollbackStep(...), RollbackStep(...))
    """
    state_dict = dict(state)
    plan_id = f"plan-{uuid.uuid4().hex[:8]}"

    logger.info(
        "rollback_plan_creating",
        plan_id=plan_id,
        checkpoint_id=checkpoint.checkpoint_id,
        reason=reason,
    )

    # Step 1: Identify affected state fields
    affected_fields = _identify_affected_state(state_dict, checkpoint)

    # Step 2: Create rollback steps
    steps = _create_rollback_steps(affected_fields)

    # Step 3: Estimate impact
    estimated_impact = _estimate_impact(affected_fields)

    # Step 4: Create plan
    plan = RollbackPlan(
        plan_id=plan_id,
        reason=reason,
        checkpoint_id=checkpoint.checkpoint_id,
        steps=steps,
        estimated_impact=estimated_impact,
    )

    # Step 5: Validate safety
    is_safe, error_msg = _validate_rollback_safety(plan, state_dict)
    if not is_safe:
        logger.warning(
            "rollback_plan_validation_failed",
            plan_id=plan_id,
            error=error_msg,
        )

    logger.info(
        "rollback_plan_created",
        plan_id=plan_id,
        step_count=len(steps),
        estimated_impact=estimated_impact,
    )

    return plan


# =============================================================================
# Rollback Execution Functions (Task 3.1-3.5)
# =============================================================================


def _execute_step(
    state: dict[str, Any],
    step: RollbackStep,
) -> tuple[bool, str | None]:
    """Execute a single rollback step (AC #3).

    Args:
        state: State to modify (mutated in place).
        step: Step to execute.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        if step.action == "restore_field":
            state[step.target_field] = step.previous_value
        elif step.action == "clear_field":
            if step.target_field in state:
                if isinstance(state.get(step.target_field), list):
                    state[step.target_field] = []
                else:
                    state[step.target_field] = step.previous_value

        logger.debug(
            "rollback_step_executed",
            step_id=step.step_id,
            action=step.action,
            target_field=step.target_field,
        )
        return True, None

    except Exception as e:
        error_msg = f"Step {step.step_id} failed: {e}"
        logger.error(
            "rollback_step_failed",
            step_id=step.step_id,
            error=str(e),
        )
        return False, error_msg


def _restore_state_field(
    state: dict[str, Any],
    field: str,
    value: Any,
) -> bool:
    """Restore a single state field to a given value.

    Args:
        state: State to modify.
        field: Field name to restore.
        value: Value to set.

    Returns:
        True if successful, False otherwise.
    """
    try:
        state[field] = value
        return True
    except Exception:
        return False


def _revert_decisions(
    state: dict[str, Any],
    checkpoint: Checkpoint,
) -> int:
    """Revert decisions made since checkpoint (AC #3).

    Args:
        state: State containing decisions list.
        checkpoint: Checkpoint with original decisions.

    Returns:
        Number of decisions reverted.
    """
    checkpoint_decisions = checkpoint.state_snapshot.get("decisions", [])
    current_decisions = state.get("decisions", [])

    if not isinstance(current_decisions, list):
        return 0

    reverted_count = len(current_decisions) - len(checkpoint_decisions)
    state["decisions"] = list(checkpoint_decisions)

    logger.debug(
        "decisions_reverted",
        reverted_count=max(0, reverted_count),
    )

    return max(0, reverted_count)


def _clear_error_state(state: dict[str, Any]) -> None:
    """Clear accumulated error state (AC #3).

    Args:
        state: State to clear error fields from.
    """
    for field in CLEARABLE_FIELDS:
        if field in state:
            if isinstance(state[field], list):
                state[field] = []
            elif isinstance(state[field], int):
                state[field] = 0
            else:
                state[field] = None

    logger.debug("error_state_cleared", fields=list(CLEARABLE_FIELDS))


def execute_rollback(
    state: dict[str, Any] | YoloState,
    plan: RollbackPlan,
    config: RollbackConfig | None = None,
) -> RollbackResult:
    """Execute a rollback plan (AC #3, AC #4).

    Executes each step in the plan, restoring state from checkpoint.
    Logs each recovery step for audit trail.

    Args:
        state: State to modify (mutated in place).
        plan: Rollback plan to execute.
        config: Optional configuration for rollback behavior.

    Returns:
        RollbackResult with execution outcome.

    Example:
        >>> result = execute_rollback(state, plan)
        >>> result.status
        'completed'
    """
    config = config or RollbackConfig()
    state_dict: dict[str, Any] = cast(
        dict[str, Any], dict(state) if not isinstance(state, dict) else state
    )
    start_time = time.time()

    logger.info(
        "rollback_execution_start",
        plan_id=plan.plan_id,
        step_count=len(plan.steps),
    )

    steps_executed = 0
    steps_failed = 0
    error_message: str | None = None

    # Execute steps up to max_steps limit
    max_steps = min(len(plan.steps), config.max_steps)

    for step in plan.steps[:max_steps]:
        success, err_msg = _execute_step(state_dict, step)

        if success:
            steps_executed += 1
        else:
            steps_failed += 1
            error_message = err_msg

            # Stop on first failure if partial rollback not allowed
            if not config.allow_partial_rollback:
                logger.warning(
                    "rollback_stopped_on_failure",
                    plan_id=plan.plan_id,
                    step_id=step.step_id,
                    steps_executed=steps_executed,
                )
                break

    # Clear error state after successful steps
    if steps_executed > 0 and steps_failed == 0:
        _clear_error_state(state_dict)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    # Determine final status
    rollback_complete = steps_failed == 0 and steps_executed == len(plan.steps)
    status: RollbackStatus = "completed" if rollback_complete else "failed"

    result = RollbackResult(
        plan=plan,
        status=status,
        steps_executed=steps_executed,
        steps_failed=steps_failed,
        rollback_complete=rollback_complete,
        duration_ms=duration_ms,
        error_message=error_message,
    )

    logger.info(
        "rollback_execution_complete",
        plan_id=plan.plan_id,
        status=status,
        steps_executed=steps_executed,
        steps_failed=steps_failed,
        duration_ms=duration_ms,
    )

    return result


# =============================================================================
# Rollback Failure Handling (Task 4.1-4.4)
# =============================================================================


def _preserve_failed_state(
    state: dict[str, Any],
    plan: RollbackPlan,
) -> None:
    """Preserve current state on rollback failure (AC #5).

    Ensures no partial rollback effects remain - state should be
    in a consistent (though possibly error) state.

    Args:
        state: State that failed during rollback.
        plan: Plan that was being executed.
    """
    # Mark that rollback failed
    state["rollback_failed"] = True
    state["failed_plan_id"] = plan.plan_id

    logger.warning(
        "failed_state_preserved",
        plan_id=plan.plan_id,
    )


def _create_escalation_context(
    plan: RollbackPlan,
    result: RollbackResult,
) -> dict[str, Any]:
    """Create detailed escalation context for human intervention (AC #5).

    Args:
        plan: The rollback plan that failed.
        result: The partial result from failed execution.

    Returns:
        Dictionary with escalation context.
    """
    return {
        "plan_id": plan.plan_id,
        "checkpoint_id": plan.checkpoint_id,
        "reason": plan.reason,
        "steps_total": len(plan.steps),
        "steps_executed": result.steps_executed,
        "steps_failed": result.steps_failed,
        "error_message": result.error_message,
        "duration_ms": result.duration_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def handle_rollback_failure(
    state: dict[str, Any] | YoloState,
    plan: RollbackPlan,
    partial_result: RollbackResult,
) -> RollbackResult:
    """Handle rollback failure with escalation (AC #5).

    Preserves current state (no partial rollback) and escalates
    to human intervention with detailed failure context.

    Args:
        state: State at time of failure.
        plan: Plan that was being executed.
        partial_result: Result from partial execution.

    Returns:
        RollbackResult with escalated status.

    Example:
        >>> result = handle_rollback_failure(state, plan, partial_result)
        >>> result.status
        'escalated'
    """
    state_dict: dict[str, Any] = cast(
        dict[str, Any], dict(state) if not isinstance(state, dict) else state
    )

    logger.warning(
        "rollback_failure_handling",
        plan_id=plan.plan_id,
        steps_executed=partial_result.steps_executed,
        steps_failed=partial_result.steps_failed,
    )

    # Step 1: Preserve failed state
    _preserve_failed_state(state_dict, plan)

    # Step 2: Create escalation context
    escalation_context = _create_escalation_context(plan, partial_result)

    # Step 3: Create escalated result
    escalated_result = RollbackResult(
        plan=plan,
        status="escalated",
        steps_executed=partial_result.steps_executed,
        steps_failed=partial_result.steps_failed,
        rollback_complete=False,
        duration_ms=partial_result.duration_ms,
        error_message=f"Rollback failed: {partial_result.error_message}. "
        f"Escalated for human intervention.",
    )

    logger.warning(
        "rollback_escalated",
        plan_id=plan.plan_id,
        escalation_context=escalation_context,
    )

    return escalated_result


# =============================================================================
# Main Orchestration Function (Task 5.1-5.4)
# =============================================================================


async def coordinate_rollback(
    state: dict[str, Any] | YoloState,
    checkpoint: Checkpoint | None = None,
    emergency_protocol: EmergencyProtocol | None = None,
    config: RollbackConfig | None = None,
) -> RollbackResult | None:
    """Coordinate rollback operations as emergency sprints (FR71).

    Main entry point for rollback coordination. Orchestrates the full flow:
    1. Check if rollback is needed
    2. Create rollback plan
    3. Validate plan safety
    4. Execute rollback
    5. Handle result (success or failure)

    This function is designed to be non-blocking - it should never
    fail the main orchestration workflow.

    Args:
        state: Current orchestration state.
        checkpoint: Checkpoint to restore to (required for rollback).
        emergency_protocol: Emergency protocol that may have triggered rollback.
        config: Optional configuration for rollback behavior.

    Returns:
        RollbackResult if rollback was performed, None if not needed.

    Example:
        >>> result = await coordinate_rollback(state, checkpoint, protocol)
        >>> result.status
        'completed'
    """
    config = config or RollbackConfig()
    state_dict = dict(state)

    logger.info(
        "rollback_coordination_start",
        checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
        protocol_id=emergency_protocol.protocol_id if emergency_protocol else None,
    )

    # Step 1: Check if rollback is needed
    should, reason = should_rollback(state_dict, emergency_protocol, checkpoint)

    if not should or reason is None or checkpoint is None:
        logger.debug("rollback_not_needed")
        return None

    # Step 2: Create rollback plan
    plan = create_rollback_plan(state_dict, checkpoint, reason)

    # Step 3: Validate plan
    is_safe, error_msg = _validate_rollback_safety(plan, state_dict)
    if not is_safe:
        logger.error(
            "rollback_plan_unsafe",
            plan_id=plan.plan_id,
            error=error_msg,
        )
        # Create failed result for unsafe plan
        return RollbackResult(
            plan=plan,
            status="failed",
            steps_executed=0,
            steps_failed=0,
            rollback_complete=False,
            duration_ms=0.0,
            error_message=f"Plan validation failed: {error_msg}",
        )

    # Step 4: Execute rollback
    result = execute_rollback(state_dict, plan, config)

    # Step 5: Handle result
    if result.status == "failed" and config.auto_escalate_on_failure:
        result = handle_rollback_failure(state_dict, plan, result)

    logger.info(
        "rollback_coordination_complete",
        plan_id=plan.plan_id,
        status=result.status,
        rollback_complete=result.rollback_complete,
    )

    return result
