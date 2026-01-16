"""Emergency protocol module for SM agent (Story 10.10).

This module provides emergency protocol functionality for the orchestration system:

- Emergency detection: Detects health degradation, circular logic, blocked gates,
  stuck agents, and system errors (AC #1, FR17)
- State checkpointing: Captures state snapshots for recovery (AC #2)
- Recovery evaluation: Evaluates recovery options with confidence scores (AC #3)
- Escalation: Escalates to human when recovery cannot proceed (AC #4, FR70)

Key Concepts:
- **Non-blocking**: Emergency protocols never block the main workflow
- **Immutable outputs**: Returns frozen dataclasses
- **Structured logging**: Uses structlog for audit trail
- **Checkpoint-based recovery**: Per ADR-007

Example:
    >>> from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
    >>> from yolo_developer.agents.sm.emergency_types import EmergencyConfig
    >>>
    >>> # Trigger emergency protocol when health is critical
    >>> protocol = await trigger_emergency_protocol(state, health_status)
    >>> protocol.status
    'recovering'
    >>>
    >>> # With custom config
    >>> config = EmergencyConfig(enable_auto_recovery=False)
    >>> protocol = await trigger_emergency_protocol(state, health_status, config)
    >>> protocol.status
    'escalated'

Architecture Note:
    Per ADR-007, this module follows checkpoint-based recovery patterns that
    ensure system resilience through state preservation and rollback capabilities.

References:
    - FR17: SM Agent can trigger emergency protocols when system health degrades
    - FR70: SM Agent can escalate to human when circular logic persists
    - FR71: SM Agent can coordinate rollback operations as emergency sprints
    - ADR-007: Retry with exponential backoff + SM-coordinated recovery
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.agents.sm.emergency_types import (
    Checkpoint,
    EmergencyConfig,
    EmergencyProtocol,
    EmergencyTrigger,
    EmergencyType,
    ProtocolStatus,
    RecoveryOption,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.health_types import HealthStatus
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

# =============================================================================
# In-memory checkpoint storage (MVP implementation)
# =============================================================================

_checkpoint_store: dict[str, Checkpoint] = {}
"""In-memory storage for checkpoints. For MVP only - should be persisted in production."""

# Threshold for agent idle time (seconds) to consider stuck
AGENT_STUCK_THRESHOLD_SECONDS: float = 600.0
"""Idle time threshold (10 minutes) to consider an agent stuck."""

# Threshold for consecutive gate failures to consider blocked
GATE_BLOCKED_THRESHOLD: int = 3
"""Number of consecutive gate failures to consider progress blocked."""


# =============================================================================
# Emergency Detection Functions (Task 2)
# =============================================================================


def _check_health_degradation(health_status: dict[str, Any] | HealthStatus | None) -> bool:
    """Check if health status indicates degradation (AC #1).

    Args:
        health_status: Health status dict or HealthStatus object.

    Returns:
        True if health is degraded or critical, False otherwise.
    """
    if health_status is None:
        return False

    # Handle both dict and HealthStatus object
    if hasattr(health_status, "status"):
        status = health_status.status
    elif isinstance(health_status, dict):
        status = health_status.get("status")
    else:
        return False

    return status in ("critical", "degraded")


def _check_circular_logic(
    state: dict[str, Any],
    cycle_analysis: dict[str, Any] | None,
) -> bool:
    """Check if circular logic pattern is detected (AC #1).

    Args:
        state: Current orchestration state.
        cycle_analysis: Cycle analysis result from circular detection.

    Returns:
        True if circular pattern detected, False otherwise.
    """
    if cycle_analysis is None:
        # Check state for cycle_analysis
        cycle_analysis = state.get("cycle_analysis")
        if cycle_analysis is None:
            return False

    if isinstance(cycle_analysis, dict):
        return bool(cycle_analysis.get("has_circular_pattern", False))

    # Handle CycleAnalysis object
    if hasattr(cycle_analysis, "has_circular_pattern"):
        return bool(cycle_analysis.has_circular_pattern)

    return False


def _check_gate_blocked(state: dict[str, Any]) -> bool:
    """Check if gates are blocking progress (AC #1).

    Args:
        state: Current orchestration state.

    Returns:
        True if repeated gate failures detected, False otherwise.
    """
    decisions = state.get("decisions") or []
    if not isinstance(decisions, list):
        return False

    # Count recent gate failures
    gate_failures = 0
    for decision in decisions:
        if isinstance(decision, dict) and decision.get("type") == "gate_failure":
            gate_failures += 1

    return gate_failures >= GATE_BLOCKED_THRESHOLD


def _check_agent_stuck(health_status: dict[str, Any] | HealthStatus | None) -> bool:
    """Check if any agent is stuck (high idle time) (AC #1).

    Args:
        health_status: Health status dict or HealthStatus object.

    Returns:
        True if any agent has high idle time, False otherwise.
    """
    if health_status is None:
        return False

    # Extract metrics
    if hasattr(health_status, "metrics"):
        metrics = health_status.metrics
        if hasattr(metrics, "agent_idle_times"):
            idle_times = metrics.agent_idle_times
        else:
            idle_times = {}
    elif isinstance(health_status, dict):
        metrics = health_status.get("metrics", {})
        if isinstance(metrics, dict):
            idle_times = metrics.get("agent_idle_times", {})
        else:
            idle_times = {}
    else:
        return False

    # Check if any agent exceeds threshold
    for _agent, idle_time in idle_times.items():
        if isinstance(idle_time, (int, float)) and idle_time > AGENT_STUCK_THRESHOLD_SECONDS:
            return True

    return False


def _check_system_error(state: dict[str, Any]) -> bool:
    """Check if there's an unrecoverable system error (AC #1).

    Args:
        state: Current orchestration state.

    Returns:
        True if system error detected, False otherwise.
    """
    error = state.get("error")
    if error is None:
        return False

    # Check for unrecoverable error markers
    if isinstance(error, dict):
        error_type = error.get("type", "")
        return error_type in ("unrecoverable", "critical", "fatal")

    return bool(error)


async def _detect_emergency(
    state: dict[str, Any],
    health_status: dict[str, Any] | HealthStatus | None,
) -> dict[str, Any] | None:
    """Detect emergency type from state and health status.

    Checks all emergency types in priority order and returns the first match.

    Args:
        state: Current orchestration state.
        health_status: Health status dict or HealthStatus object.

    Returns:
        Detection dict with emergency_type and reason, or None if no emergency.
    """
    # Priority 1: System error (most critical)
    if _check_system_error(state):
        return {
            "emergency_type": "system_error",
            "reason": "Unrecoverable system error detected",
        }

    # Priority 2: Health degradation
    if _check_health_degradation(health_status):
        return {
            "emergency_type": "health_degraded",
            "reason": "System health fell below critical thresholds",
        }

    # Priority 3: Circular logic
    cycle_analysis = state.get("cycle_analysis")
    if _check_circular_logic(state, cycle_analysis):
        return {
            "emergency_type": "circular_logic",
            "reason": "Agents in ping-pong loop (>3 exchanges)",
        }

    # Priority 4: Gate blocked
    if _check_gate_blocked(state):
        return {
            "emergency_type": "gate_blocked",
            "reason": "Repeated gate failures blocking progress",
        }

    # Priority 5: Agent stuck
    if _check_agent_stuck(health_status):
        return {
            "emergency_type": "agent_stuck",
            "reason": "Agent idle beyond threshold",
        }

    return None


def _create_emergency_trigger(
    state: dict[str, Any],
    health_status: dict[str, Any] | HealthStatus | None,
    detection: dict[str, Any],
) -> EmergencyTrigger:
    """Create EmergencyTrigger from detection result.

    Args:
        state: Current orchestration state.
        health_status: Health status dict or HealthStatus object.
        detection: Detection result with emergency_type and reason.

    Returns:
        EmergencyTrigger with full context.
    """
    emergency_type = detection["emergency_type"]
    reason = detection["reason"]

    # Determine severity based on type
    if emergency_type in ("system_error", "health_degraded"):
        severity = "critical"
    else:
        severity = "warning"

    # Get source agent
    source_agent = state.get("current_agent")

    # Convert health_status to dict if needed
    health_dict: dict[str, Any] | None = None
    if health_status is not None:
        if hasattr(health_status, "to_dict"):
            health_dict = health_status.to_dict()
        elif isinstance(health_status, dict):
            health_dict = health_status

    # emergency_type comes from detection dict and matches EmergencyType literal
    return EmergencyTrigger(
        emergency_type=emergency_type,
        severity=severity,
        source_agent=source_agent,
        trigger_reason=reason,
        health_status=health_dict,
    )


# =============================================================================
# Checkpointing Functions (Task 3)
# =============================================================================


def _capture_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Capture relevant state data for checkpoint (AC #2).

    Excludes sensitive data like API keys.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary with safe state snapshot.
    """
    # Keys to exclude (sensitive data)
    exclude_keys = {"api_key", "secret", "password", "token", "credential"}

    snapshot: dict[str, Any] = {}
    for key, value in state.items():
        if key.lower() not in exclude_keys and not any(
            exc in key.lower() for exc in exclude_keys
        ):
            # Deep copy simple types, reference complex ones
            if isinstance(value, (str, int, float, bool, type(None))):
                snapshot[key] = value
            elif isinstance(value, list):
                # Shallow copy lists
                snapshot[key] = list(value)
            elif isinstance(value, dict):
                # Shallow copy dicts
                snapshot[key] = dict(value)
            else:
                # For complex objects, just note the type
                snapshot[key] = f"<{type(value).__name__}>"

    return snapshot


def _create_checkpoint(
    snapshot: dict[str, Any],
    trigger_type: EmergencyType,
    metadata: dict[str, Any] | None = None,
) -> Checkpoint:
    """Create Checkpoint from state snapshot.

    Args:
        snapshot: State snapshot dictionary.
        trigger_type: Emergency type that triggered checkpoint.
        metadata: Optional additional metadata.

    Returns:
        Checkpoint with generated ID.
    """
    checkpoint_id = f"chk-{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()

    return Checkpoint(
        checkpoint_id=checkpoint_id,
        state_snapshot=snapshot,
        created_at=created_at,
        trigger_type=trigger_type,
        metadata=metadata or {},
    )


def _store_checkpoint(checkpoint: Checkpoint) -> None:
    """Store checkpoint in memory (MVP implementation).

    Args:
        checkpoint: Checkpoint to store.
    """
    _checkpoint_store[checkpoint.checkpoint_id] = checkpoint
    logger.debug(
        "checkpoint_stored",
        checkpoint_id=checkpoint.checkpoint_id,
        trigger_type=checkpoint.trigger_type,
    )


def _retrieve_checkpoint(checkpoint_id: str) -> Checkpoint | None:
    """Retrieve checkpoint by ID.

    Args:
        checkpoint_id: ID of checkpoint to retrieve.

    Returns:
        Checkpoint if found, None otherwise.
    """
    return _checkpoint_store.get(checkpoint_id)


async def checkpoint_state(
    state: dict[str, Any],
    trigger_type: EmergencyType,
    metadata: dict[str, Any] | None = None,
) -> Checkpoint:
    """Main entry point for state checkpointing (AC #2).

    Captures state snapshot, creates checkpoint, and stores it.

    Args:
        state: Current orchestration state.
        trigger_type: Emergency type triggering the checkpoint.
        metadata: Optional additional metadata.

    Returns:
        Created and stored Checkpoint.
    """
    snapshot = _capture_state_snapshot(state)
    checkpoint = _create_checkpoint(snapshot, trigger_type, metadata)
    _store_checkpoint(checkpoint)

    logger.info(
        "state_checkpointed",
        checkpoint_id=checkpoint.checkpoint_id,
        trigger_type=trigger_type,
        snapshot_keys=list(snapshot.keys()),
    )

    return checkpoint


# =============================================================================
# Recovery Evaluation Functions (Task 4)
# =============================================================================


def _evaluate_retry_option(
    trigger: EmergencyTrigger,
    recovery_attempts: int = 0,
) -> RecoveryOption:
    """Evaluate retry as a recovery option (AC #3).

    Confidence decreases with more recovery attempts.

    Args:
        trigger: Emergency trigger information.
        recovery_attempts: Number of previous recovery attempts.

    Returns:
        RecoveryOption for retry action.
    """
    # Base confidence varies by emergency type
    base_confidence = {
        "gate_blocked": 0.7,
        "agent_stuck": 0.6,
        "circular_logic": 0.4,
        "health_degraded": 0.3,
        "system_error": 0.1,
    }.get(trigger.emergency_type, 0.5)

    # Decrease confidence with each attempt
    confidence = max(0.1, base_confidence - (recovery_attempts * 0.2))

    risks = ("May fail again", "Could waste resources") if recovery_attempts > 0 else ("May fail again",)

    return RecoveryOption(
        action="retry",
        description="Retry the failed operation",
        confidence=confidence,
        risks=risks,
        estimated_impact="minimal",
    )


def _evaluate_rollback_option(
    trigger: EmergencyTrigger,
    checkpoint: Checkpoint | None,
) -> RecoveryOption | None:
    """Evaluate rollback as a recovery option (AC #3).

    Args:
        trigger: Emergency trigger information.
        checkpoint: Available checkpoint for rollback (None if no checkpoint).

    Returns:
        RecoveryOption for rollback action, or None if no checkpoint available.
    """
    if checkpoint is None:
        return None

    # Rollback confidence varies by emergency type
    # More reliable for state-related issues, less reliable for system errors
    confidence = {
        "gate_blocked": 0.8,
        "agent_stuck": 0.7,
        "circular_logic": 0.6,
        "health_degraded": 0.5,
        "system_error": 0.3,  # System errors may have corrupted state
    }.get(trigger.emergency_type, 0.5)

    return RecoveryOption(
        action="rollback",
        description=f"Rollback to checkpoint {checkpoint.checkpoint_id}",
        confidence=confidence,
        risks=("May lose recent progress", "State may be outdated"),
        estimated_impact="moderate",
    )


def _evaluate_skip_option(trigger: EmergencyTrigger) -> RecoveryOption:
    """Evaluate skip as a recovery option (AC #3).

    Args:
        trigger: Emergency trigger information.

    Returns:
        RecoveryOption for skip action.
    """
    # Skip is risky for critical issues
    confidence = {
        "gate_blocked": 0.4,
        "agent_stuck": 0.5,
        "circular_logic": 0.3,
        "health_degraded": 0.1,
        "system_error": 0.0,
    }.get(trigger.emergency_type, 0.3)

    return RecoveryOption(
        action="skip",
        description="Skip the problematic step and continue",
        confidence=confidence,
        risks=("May miss important work", "Could cause downstream issues"),
        estimated_impact="moderate",
    )


def _evaluate_escalate_option(trigger: EmergencyTrigger) -> RecoveryOption:
    """Evaluate escalation as a recovery option (AC #3, FR70).

    Escalation is always available with high confidence.

    Args:
        trigger: Emergency trigger information.

    Returns:
        RecoveryOption for escalate action.
    """
    return RecoveryOption(
        action="escalate",
        description="Escalate to human for manual intervention",
        confidence=1.0,  # Always available
        risks=(),  # No risks - human takes over
        estimated_impact="significant",
    )


def _evaluate_terminate_option(trigger: EmergencyTrigger) -> RecoveryOption:
    """Evaluate graceful termination as a recovery option (AC #3).

    Args:
        trigger: Emergency trigger information.

    Returns:
        RecoveryOption for terminate action.
    """
    # Higher confidence for more severe issues
    confidence = {
        "system_error": 0.8,
        "health_degraded": 0.6,
        "circular_logic": 0.4,
        "gate_blocked": 0.3,
        "agent_stuck": 0.2,
    }.get(trigger.emergency_type, 0.4)

    return RecoveryOption(
        action="terminate",
        description="Gracefully terminate the current sprint",
        confidence=confidence,
        risks=("All progress in current sprint may be lost",),
        estimated_impact="significant",
    )


async def _generate_recovery_options(
    state: dict[str, Any],
    trigger: EmergencyTrigger,
    checkpoint: Checkpoint | None,
    config: EmergencyConfig,
) -> list[RecoveryOption]:
    """Generate all applicable recovery options (AC #3).

    Args:
        state: Current orchestration state.
        trigger: Emergency trigger information.
        checkpoint: Available checkpoint (may be None).
        config: Emergency protocol configuration.

    Returns:
        List of RecoveryOptions sorted by confidence.
    """
    options: list[RecoveryOption] = []

    # Get current recovery attempts from state
    recovery_attempts = state.get("recovery_attempts", 0)
    if not isinstance(recovery_attempts, int):
        recovery_attempts = 0

    # Evaluate all option types
    retry_option = _evaluate_retry_option(trigger, recovery_attempts)
    options.append(retry_option)

    rollback_option = _evaluate_rollback_option(trigger, checkpoint)
    if rollback_option:
        options.append(rollback_option)

    skip_option = _evaluate_skip_option(trigger)
    options.append(skip_option)

    escalate_option = _evaluate_escalate_option(trigger)
    options.append(escalate_option)

    terminate_option = _evaluate_terminate_option(trigger)
    options.append(terminate_option)

    # Sort by confidence (highest first)
    options.sort(key=lambda o: o.confidence, reverse=True)

    logger.debug(
        "recovery_options_generated",
        option_count=len(options),
        top_action=options[0].action if options else None,
        top_confidence=options[0].confidence if options else None,
    )

    return options


def _select_best_recovery(options: list[RecoveryOption]) -> RecoveryOption | None:
    """Select the best recovery option (highest confidence).

    Args:
        options: List of recovery options.

    Returns:
        Best option, or None if no options.
    """
    if not options:
        return None

    # Select option with highest confidence
    return max(options, key=lambda o: o.confidence)


# =============================================================================
# Escalation Functions (Task 5)
# =============================================================================


def _should_escalate(
    options: list[RecoveryOption],
    config: EmergencyConfig,
) -> bool:
    """Check if escalation is needed (AC #4).

    Escalation is needed when best option confidence is below threshold.

    Args:
        options: Available recovery options.
        config: Emergency protocol configuration.

    Returns:
        True if escalation should occur, False otherwise.
    """
    if not options:
        return True

    best = _select_best_recovery(options)
    if best is None:
        return True

    # Escalate is always an option, so check non-escalate options
    non_escalate_options = [o for o in options if o.action != "escalate"]
    if not non_escalate_options:
        return True

    best_non_escalate = max(non_escalate_options, key=lambda o: o.confidence)
    return best_non_escalate.confidence < config.escalation_threshold


def _create_escalation_record(
    protocol_id: str,
    trigger: EmergencyTrigger,
    reason: str,
) -> dict[str, Any]:
    """Create escalation record for logging.

    Args:
        protocol_id: ID of the emergency protocol.
        trigger: Emergency trigger information.
        reason: Reason for escalation.

    Returns:
        Escalation record dictionary.
    """
    return {
        "protocol_id": protocol_id,
        "emergency_type": trigger.emergency_type,
        "severity": trigger.severity,
        "source_agent": trigger.source_agent,
        "reason": reason,
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _notify_escalation(
    protocol_id: str,
    trigger: EmergencyTrigger,
    reason: str,
) -> None:
    """Log escalation notification for human review (FR70).

    Args:
        protocol_id: ID of the emergency protocol.
        trigger: Emergency trigger information.
        reason: Reason for escalation.
    """
    _ = _create_escalation_record(protocol_id, trigger, reason)  # For audit trail

    logger.warning(
        "emergency_escalated",
        protocol_id=protocol_id,
        emergency_type=trigger.emergency_type,
        severity=trigger.severity,
        reason=reason,
        requires_human_intervention=True,
    )


async def escalate_emergency(
    protocol_id: str,
    trigger: EmergencyTrigger,
    reason: str,
) -> dict[str, Any]:
    """Main entry point for emergency escalation (AC #4, FR70).

    Args:
        protocol_id: ID of the emergency protocol.
        trigger: Emergency trigger information.
        reason: Reason for escalation.

    Returns:
        Escalation result dictionary.
    """
    await _notify_escalation(protocol_id, trigger, reason)

    return {
        "escalated": True,
        "protocol_id": protocol_id,
        "emergency_type": trigger.emergency_type,
        "reason": reason,
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Main Emergency Protocol Function (Task 6)
# =============================================================================


async def trigger_emergency_protocol(
    state: dict[str, Any] | YoloState,
    health_status: HealthStatus | None = None,
    config: EmergencyConfig | None = None,
) -> EmergencyProtocol:
    """Trigger emergency protocol when system health degrades (FR17, FR71).

    Main entry point for emergency handling. Detects the emergency type,
    checkpoints state, evaluates recovery options, and either auto-recovers
    or escalates to human intervention.

    This function is designed to be non-blocking - it should never
    fail the main orchestration workflow even if the protocol fails.

    Args:
        state: Current orchestration state.
        health_status: Health status that triggered emergency (optional).
        config: Emergency protocol configuration (uses defaults if None).

    Returns:
        EmergencyProtocol with full protocol outcome.

    Example:
        >>> protocol = await trigger_emergency_protocol(state, health_status)
        >>> protocol.status
        'recovering'
        >>> protocol.selected_action
        'retry'
    """
    config = config or EmergencyConfig()
    protocol_id = f"emergency-{uuid.uuid4().hex[:8]}"
    state_dict = dict(state)  # Ensure we have a dict

    logger.warning(
        "emergency_protocol_triggered",
        protocol_id=protocol_id,
        current_agent=state_dict.get("current_agent"),
    )

    # Step 1: Detect emergency type and create trigger
    detection = await _detect_emergency(state_dict, health_status)

    if detection is None:
        # No emergency detected - shouldn't happen but handle gracefully
        detection = {
            "emergency_type": "health_degraded",
            "reason": "Unknown emergency condition",
        }

    trigger = _create_emergency_trigger(state_dict, health_status, detection)

    # Step 2: Checkpoint state if configured
    checkpoint: Checkpoint | None = None
    if config.auto_checkpoint:
        try:
            checkpoint = await checkpoint_state(
                state_dict,
                trigger.emergency_type,
                {"protocol_id": protocol_id},
            )
            logger.info("state_checkpointed", checkpoint_id=checkpoint.checkpoint_id)
        except Exception as e:
            logger.error("checkpoint_failed", error=str(e))

    # Step 3: Evaluate recovery options
    recovery_options = await _generate_recovery_options(
        state=state_dict,
        trigger=trigger,
        checkpoint=checkpoint,
        config=config,
    )

    # Step 4: Select action or escalate
    selected_action = None
    escalation_reason: str | None = None
    status: ProtocolStatus = "checkpointed" if checkpoint else "active"

    if config.enable_auto_recovery and recovery_options:
        if _should_escalate(recovery_options, config):
            escalation_reason = "No recovery option meets confidence threshold"
            status = "escalated"
        else:
            best_option = _select_best_recovery(recovery_options)
            if best_option and best_option.action != "escalate":
                selected_action = best_option.action
                status = "recovering"
                logger.info(
                    "recovery_action_selected",
                    action=selected_action,
                    confidence=best_option.confidence,
                )
            else:
                escalation_reason = "Best option is escalation"
                status = "escalated"
    else:
        escalation_reason = "Auto-recovery disabled or no options available"
        status = "escalated"

    # Step 5: Handle escalation if needed
    if status == "escalated":
        await _notify_escalation(protocol_id, trigger, escalation_reason or "Unknown")

    result = EmergencyProtocol(
        protocol_id=protocol_id,
        trigger=trigger,
        status=status,
        checkpoint=checkpoint,
        recovery_options=tuple(recovery_options),
        selected_action=selected_action,
        escalation_reason=escalation_reason,
    )

    logger.warning(
        "emergency_protocol_complete",
        protocol_id=protocol_id,
        status=status,
        selected_action=selected_action,
        escalated=status == "escalated",
    )

    return result
