"""Human escalation module for SM agent (Story 10.14).

This module provides functionality for escalating unresolvable issues to
human intervention, including:

- Escalation detection from various sources (circular logic, conflicts, health)
- Request creation with actionable options
- Response integration and state updates
- Timeout handling with default actions
- Full orchestration of the escalation workflow

Example:
    >>> from yolo_developer.agents.sm.human_escalation import (
    ...     should_escalate,
    ...     create_escalation_request,
    ...     integrate_escalation_response,
    ...     handle_escalation_timeout,
    ...     manage_human_escalation,
    ... )
    >>>
    >>> # Check if escalation is needed
    >>> should, trigger = should_escalate(state, cycle_analysis, None, None)
    >>> if should:
    ...     request = create_escalation_request(state, trigger, {})
    ...     # Present to user, get response
    ...     result = integrate_escalation_response(state, request, response)

References:
    - FR70: SM Agent can escalate to human when circular logic persists
    - Story 10.6: Circular Logic Detection (escalation_triggered)
    - Story 10.7: Conflict Mediation (escalations_triggered)
    - Story 10.10: Emergency Protocols (escalate_emergency)
    - ADR-005: Inter-Agent Communication
    - ADR-007: Error Handling Strategy
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.agents.sm.human_escalation_types import (
    EscalationConfig,
    EscalationOption,
    EscalationRequest,
    EscalationResponse,
    EscalationResult,
    EscalationTrigger,
)

if TYPE_CHECKING:
    from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis
    from yolo_developer.agents.sm.conflict_types import MediationResult
    from yolo_developer.agents.sm.health_types import HealthStatus
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

_TRIGGER_SUMMARIES: dict[str, str] = {
    "circular_logic": "Circular logic detected - agents are in a repetitive loop",
    "conflict_unresolved": "Conflict could not be automatically resolved",
    "gate_blocked": "Quality gate blocked with no automatic recovery path",
    "system_error": "Critical system error requires human intervention",
    "agent_stuck": "Agent is stuck and unable to make progress",
    "user_requested": "User explicitly requested escalation",
}

_DEFAULT_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "circular_logic": [
        {
            "option_id": "opt-retry",
            "label": "Retry",
            "description": "Retry the current operation with fresh context",
            "action": "retry",
            "is_recommended": True,
        },
        {
            "option_id": "opt-skip",
            "label": "Skip",
            "description": "Skip the problematic step and continue",
            "action": "skip",
            "is_recommended": False,
        },
        {
            "option_id": "opt-abort",
            "label": "Abort",
            "description": "Abort the current sprint",
            "action": "abort",
            "is_recommended": False,
        },
    ],
    "conflict_unresolved": [
        {
            "option_id": "opt-first",
            "label": "Accept First",
            "description": "Accept the first agent's recommendation",
            "action": "accept_first",
            "is_recommended": False,
        },
        {
            "option_id": "opt-second",
            "label": "Accept Second",
            "description": "Accept the second agent's recommendation",
            "action": "accept_second",
            "is_recommended": False,
        },
        {
            "option_id": "opt-skip",
            "label": "Skip",
            "description": "Skip this conflict and continue",
            "action": "skip",
            "is_recommended": True,
        },
    ],
    "system_error": [
        {
            "option_id": "opt-retry",
            "label": "Retry",
            "description": "Retry the failed operation",
            "action": "retry",
            "is_recommended": True,
        },
        {
            "option_id": "opt-abort",
            "label": "Abort",
            "description": "Abort the current sprint safely",
            "action": "abort",
            "is_recommended": False,
        },
    ],
    "gate_blocked": [
        {
            "option_id": "opt-override",
            "label": "Override",
            "description": "Override the gate and continue (not recommended)",
            "action": "override",
            "is_recommended": False,
        },
        {
            "option_id": "opt-retry",
            "label": "Retry",
            "description": "Retry after addressing the gate requirements",
            "action": "retry",
            "is_recommended": True,
        },
        {
            "option_id": "opt-abort",
            "label": "Abort",
            "description": "Abort the current task",
            "action": "abort",
            "is_recommended": False,
        },
    ],
    "agent_stuck": [
        {
            "option_id": "opt-reset",
            "label": "Reset Agent",
            "description": "Reset the agent state and retry",
            "action": "reset",
            "is_recommended": True,
        },
        {
            "option_id": "opt-skip",
            "label": "Skip",
            "description": "Skip the current task",
            "action": "skip",
            "is_recommended": False,
        },
    ],
    "user_requested": [
        {
            "option_id": "opt-continue",
            "label": "Continue",
            "description": "Continue with current approach",
            "action": "continue",
            "is_recommended": True,
        },
        {
            "option_id": "opt-modify",
            "label": "Modify",
            "description": "Modify the approach",
            "action": "modify",
            "is_recommended": False,
        },
        {
            "option_id": "opt-abort",
            "label": "Abort",
            "description": "Abort the operation",
            "action": "abort",
            "is_recommended": False,
        },
    ],
}


# =============================================================================
# Task 2: Escalation Detection (Subtasks 2.1-2.6)
# =============================================================================


def _check_circular_escalation(
    cycle_analysis: CycleAnalysis | None,
) -> bool:
    """Check if circular logic analysis triggers escalation.

    Args:
        cycle_analysis: Result from circular logic detection, or None.

    Returns:
        True if escalation should be triggered.
    """
    if cycle_analysis is None:
        return False
    return cycle_analysis.escalation_triggered


def _check_conflict_escalation(
    mediation_result: MediationResult | None,
) -> bool:
    """Check if conflict mediation triggers escalation.

    Args:
        mediation_result: Result from conflict mediation, or None.

    Returns:
        True if escalation should be triggered.
    """
    if mediation_result is None:
        return False
    return len(mediation_result.escalations_triggered) > 0


def _check_health_escalation(
    health_status: HealthStatus | None,
) -> bool:
    """Check if health status triggers escalation.

    Args:
        health_status: Current health status, or None.

    Returns:
        True if escalation should be triggered (critical status).
    """
    if health_status is None:
        return False
    return health_status.status == "critical"


def _check_gate_blocked_escalation(
    state: YoloState,
) -> bool:
    """Check if gate blocked state triggers escalation.

    Currently checks for gate_blocked flag in state. Future implementations
    may add more sophisticated gate recovery detection.

    Args:
        state: Current orchestration state.

    Returns:
        True if gate is blocked with no recovery path.
    """
    # Check for gate_blocked flag if present in state
    # Cast to dict for flexible access to optional state keys
    state_dict: dict[str, Any] = dict(state)
    gate_blocked = state_dict.get("gate_blocked", False)
    if not gate_blocked:
        return False

    # Check if there's a recovery path
    recovery_path = state_dict.get("gate_recovery_path")
    return recovery_path is None


def _check_user_requested_escalation(
    state: YoloState,
) -> bool:
    """Check if user explicitly requested escalation.

    Args:
        state: Current orchestration state.

    Returns:
        True if escalate_to_human flag is set.
    """
    # Cast to dict for flexible access to optional state keys
    state_dict: dict[str, Any] = dict(state)
    return bool(state_dict.get("escalate_to_human", False))


def should_escalate(
    state: YoloState,
    cycle_analysis: CycleAnalysis | None,
    mediation_result: MediationResult | None,
    health_status: HealthStatus | None,
) -> tuple[bool, EscalationTrigger | None]:
    """Determine if escalation to human is needed.

    Checks multiple sources for escalation triggers and returns the
    highest priority trigger if escalation is needed.

    Priority order:
    1. Circular logic (FR70)
    2. Unresolved conflict
    3. Critical health
    4. Gate blocked
    5. User requested

    Args:
        state: Current orchestration state.
        cycle_analysis: Result from circular logic detection.
        mediation_result: Result from conflict mediation.
        health_status: Current health status.

    Returns:
        Tuple of (should_escalate, trigger) where trigger is None if no escalation.

    Example:
        >>> should, trigger = should_escalate(state, cycle_analysis, None, None)
        >>> if should:
        ...     print(f"Escalation needed: {trigger}")
    """
    # Check in priority order
    if _check_circular_escalation(cycle_analysis):
        logger.info(
            "escalation_check_circular_logic",
            escalation_triggered=True,
            reason=cycle_analysis.escalation_reason if cycle_analysis else None,
        )
        return True, "circular_logic"

    if _check_conflict_escalation(mediation_result):
        logger.info(
            "escalation_check_conflict",
            escalation_triggered=True,
            conflicts=list(mediation_result.escalations_triggered) if mediation_result else [],
        )
        return True, "conflict_unresolved"

    if _check_health_escalation(health_status):
        logger.info(
            "escalation_check_health",
            escalation_triggered=True,
            health_status=health_status.status if health_status else None,
        )
        return True, "system_error"

    if _check_gate_blocked_escalation(state):
        logger.info(
            "escalation_check_gate_blocked",
            escalation_triggered=True,
        )
        return True, "gate_blocked"

    if _check_user_requested_escalation(state):
        logger.info(
            "escalation_check_user_requested",
            escalation_triggered=True,
        )
        return True, "user_requested"

    logger.debug("escalation_check_none", escalation_triggered=False)
    return False, None


# =============================================================================
# Task 3: Escalation Request Creation (Subtasks 3.1-3.5)
# =============================================================================


def _build_escalation_summary(
    state: YoloState,
    trigger: EscalationTrigger,
) -> str:
    """Create a human-readable summary of the escalation.

    Args:
        state: Current orchestration state.
        trigger: The escalation trigger type.

    Returns:
        Human-readable summary string.
    """
    base_summary = _TRIGGER_SUMMARIES.get(trigger, "An issue requires your attention")
    current_agent = state["current_agent"]
    return f"{base_summary} (Current agent: {current_agent})"


def _build_escalation_context(
    state: YoloState,
    trigger: EscalationTrigger,
    additional_context: dict[str, Any],
) -> dict[str, Any]:
    """Gather relevant context for the escalation.

    Args:
        state: Current orchestration state.
        trigger: The escalation trigger type.
        additional_context: Additional context provided by caller.

    Returns:
        Dictionary of context information including history of attempted resolutions
        per AC #1.
    """
    context: dict[str, Any] = {
        "trigger": trigger,
        "current_agent": state["current_agent"],
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Add recent messages if available
    messages = state["messages"]
    if messages:
        # Include last few messages for context
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        context["recent_message_count"] = len(messages)
        context["recent_messages_preview"] = len(recent_messages)

    # Add decisions if available (AC #1: history of attempted resolutions)
    decisions = state["decisions"]
    if decisions:
        context["decision_count"] = len(decisions)
        # Include recent decisions as history of attempted resolutions
        recent_decisions = decisions[-5:] if len(decisions) > 5 else decisions
        context["resolution_history"] = [
            {
                "agent": getattr(d, "agent", "unknown"),
                "summary": getattr(d, "summary", str(d)),
                "rationale": getattr(d, "rationale", None),
            }
            for d in recent_decisions
        ]

    # Cast state to dict for optional fields access
    state_dict: dict[str, Any] = dict(state)

    # Include cycle analysis history if present (AC #1: history of interventions)
    cycle_analysis = state_dict.get("cycle_analysis")
    if cycle_analysis and isinstance(cycle_analysis, dict):
        context["intervention_history"] = {
            "intervention_strategy": cycle_analysis.get("intervention_strategy"),
            "intervention_message": cycle_analysis.get("intervention_message"),
            "patterns_found": cycle_analysis.get("patterns_found", []),
            "total_exchange_count": cycle_analysis.get("total_exchange_count", 0),
        }

    # Include mediation history if present (AC #1: history of resolutions)
    mediation_result = state_dict.get("mediation_result")
    if mediation_result and isinstance(mediation_result, dict):
        context["mediation_history"] = {
            "conflicts_detected": len(mediation_result.get("conflicts_detected", [])),
            "resolutions_attempted": len(mediation_result.get("resolutions", [])),
            "mediation_notes": mediation_result.get("mediation_notes"),
        }

    # Merge additional context
    context.update(additional_context)

    return context


def _build_escalation_options(
    state: YoloState,
    trigger: EscalationTrigger,
) -> tuple[EscalationOption, ...]:
    """Create available action options for the escalation.

    Args:
        state: Current orchestration state.
        trigger: The escalation trigger type.

    Returns:
        Tuple of EscalationOption objects.
    """
    option_defs = _DEFAULT_OPTIONS.get(trigger, _DEFAULT_OPTIONS["user_requested"])
    options = tuple(
        EscalationOption(
            option_id=opt["option_id"],
            label=opt["label"],
            description=opt["description"],
            action=opt["action"],
            is_recommended=opt["is_recommended"],
        )
        for opt in option_defs
    )
    return options


def _determine_recommended_option(
    options: tuple[EscalationOption, ...],
    state: YoloState,
    trigger: EscalationTrigger,
) -> str | None:
    """Select the recommended option from available options.

    Args:
        options: Available options.
        state: Current orchestration state.
        trigger: The escalation trigger type.

    Returns:
        option_id of recommended option, or None.
    """
    for opt in options:
        if opt.is_recommended:
            return opt.option_id
    # Fallback to first option if none marked as recommended
    return options[0].option_id if options else None


def create_escalation_request(
    state: YoloState,
    trigger: EscalationTrigger,
    additional_context: dict[str, Any],
) -> EscalationRequest:
    """Create an escalation request for human intervention.

    Builds a complete EscalationRequest with summary, context, and
    available options for the user to choose from.

    Args:
        state: Current orchestration state.
        trigger: What triggered the escalation.
        additional_context: Additional context to include.

    Returns:
        EscalationRequest ready to present to user.

    Example:
        >>> request = create_escalation_request(state, "circular_logic", {"exchanges": 5})
        >>> request.trigger
        'circular_logic'
    """
    request_id = f"esc-{uuid.uuid4().hex[:12]}"
    current_agent = state["current_agent"]

    summary = _build_escalation_summary(state, trigger)
    context = _build_escalation_context(state, trigger, additional_context)
    options = _build_escalation_options(state, trigger)
    recommended = _determine_recommended_option(options, state, trigger)

    request = EscalationRequest(
        request_id=request_id,
        trigger=trigger,
        agent=current_agent,
        summary=summary,
        context=context,
        options=options,
        recommended_option=recommended,
    )

    logger.info(
        "escalation_request_created",
        request_id=request_id,
        trigger=trigger,
        agent=current_agent,
        option_count=len(options),
        recommended_option=recommended,
    )

    return request


# =============================================================================
# Task 4: Response Integration (Subtasks 4.1-4.5)
# =============================================================================


def _validate_response(
    request: EscalationRequest,
    response: EscalationResponse,
) -> tuple[bool, str | None]:
    """Validate that response matches the request.

    Args:
        request: The original escalation request.
        response: The user's response.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check request_id match
    if response.request_id != request.request_id:
        return False, f"Response request_id '{response.request_id}' does not match request '{request.request_id}'"

    # Check selected option exists
    valid_option_ids = {opt.option_id for opt in request.options}
    if response.selected_option not in valid_option_ids:
        return False, f"Selected option '{response.selected_option}' not in valid options: {valid_option_ids}"

    return True, None


def _get_action_from_option(
    request: EscalationRequest,
    selected_option_id: str,
) -> str | None:
    """Get the action string from the selected option.

    Args:
        request: The escalation request.
        selected_option_id: The option ID selected by user.

    Returns:
        The action string, or None if not found.
    """
    for opt in request.options:
        if opt.option_id == selected_option_id:
            return opt.action
    return None


def _apply_resolution_action(
    state: YoloState,
    request: EscalationRequest,
    response: EscalationResponse,
) -> bool:
    """Apply the resolution action to state.

    Note: For MVP, this logs the action. Full state modification
    will be handled by the SM node based on the result.

    Args:
        state: Current orchestration state.
        request: The escalation request.
        response: User's response.

    Returns:
        True if action was applied successfully.
    """
    action = _get_action_from_option(request, response.selected_option)

    logger.info(
        "escalation_resolution_applying",
        request_id=request.request_id,
        selected_option=response.selected_option,
        action=action,
        user_rationale=response.user_rationale,
    )

    # MVP: Log the action - actual state modification handled by caller
    return True


def integrate_escalation_response(
    state: YoloState,
    request: EscalationRequest,
    response: EscalationResponse,
) -> EscalationResult:
    """Integrate a user's escalation response into the system.

    Validates the response, extracts the action, and prepares
    the result for further processing.

    Args:
        state: Current orchestration state.
        request: The original escalation request.
        response: User's response.

    Returns:
        EscalationResult with integration outcome.

    Example:
        >>> result = integrate_escalation_response(state, request, response)
        >>> if result.integration_success:
        ...     print(f"Action: {result.resolution_action}")
    """
    start_time = time.time()

    # Validate response
    is_valid, error_msg = _validate_response(request, response)
    if not is_valid:
        logger.warning(
            "escalation_response_invalid",
            request_id=request.request_id,
            error=error_msg,
        )
        duration_ms = (time.time() - start_time) * 1000
        return EscalationResult(
            request=request,
            response=response,
            status="cancelled",
            resolution_action=None,
            integration_success=False,
            duration_ms=duration_ms,
        )

    # Get action from selected option
    action = _get_action_from_option(request, response.selected_option)

    # Apply the action
    success = _apply_resolution_action(state, request, response)

    duration_ms = (time.time() - start_time) * 1000

    result = EscalationResult(
        request=request,
        response=response,
        status="resolved" if success else "cancelled",
        resolution_action=action,
        integration_success=success,
        duration_ms=duration_ms,
    )

    logger.info(
        "escalation_response_integrated",
        request_id=request.request_id,
        status=result.status,
        resolution_action=action,
        integration_success=success,
        duration_ms=duration_ms,
    )

    return result


def handle_escalation_timeout(
    request: EscalationRequest,
    config: EscalationConfig,
) -> EscalationResult:
    """Handle an escalation that timed out without user response.

    Uses the default action from config and marks the escalation
    as timed out.

    Args:
        request: The escalation request that timed out.
        config: Configuration with default action.

    Returns:
        EscalationResult with timeout status.

    Example:
        >>> config = EscalationConfig(default_action="skip")
        >>> result = handle_escalation_timeout(request, config)
        >>> result.status
        'timed_out'
    """
    logger.warning(
        "escalation_timeout",
        request_id=request.request_id,
        trigger=request.trigger,
        default_action=config.default_action,
        timeout_seconds=config.timeout_seconds,
    )

    return EscalationResult(
        request=request,
        response=None,
        status="timed_out",
        resolution_action=config.default_action,
        integration_success=False,
        duration_ms=config.timeout_seconds * 1000,
    )


# =============================================================================
# Task 5: Main Orchestration Function (Subtasks 5.1-5.4)
# =============================================================================


async def manage_human_escalation(
    state: YoloState,
    cycle_analysis: CycleAnalysis | None = None,
    mediation_result: MediationResult | None = None,
    health_status: HealthStatus | None = None,
    config: EscalationConfig | None = None,
) -> EscalationResult | None:
    """Main orchestration function for human escalation.

    Checks if escalation is needed, creates a request if so, and
    returns the request for presentation to the user. The actual
    presentation and response collection is handled by the caller.

    Note: This function creates the request but does NOT wait for
    user response. The caller (typically sm_node) handles presenting
    the request and later calling integrate_escalation_response.

    Args:
        state: Current orchestration state.
        cycle_analysis: Optional result from circular logic detection.
        mediation_result: Optional result from conflict mediation.
        health_status: Optional current health status.
        config: Optional configuration (uses defaults if not provided).

    Returns:
        EscalationResult with pending status if escalation needed,
        or None if no escalation required.

    Example:
        >>> result = await manage_human_escalation(state, cycle_analysis)
        >>> if result:
        ...     # Present result.request to user
        ...     pass
    """
    if config is None:
        config = EscalationConfig()

    # Step 1: Check if escalation is needed
    should, trigger = should_escalate(
        state,
        cycle_analysis,
        mediation_result,
        health_status,
    )

    if not should or trigger is None:
        logger.debug(
            "escalation_not_needed",
            cycle_analysis_present=cycle_analysis is not None,
            mediation_result_present=mediation_result is not None,
            health_status_present=health_status is not None,
        )
        return None

    # Step 2: Build additional context based on trigger source
    additional_context: dict[str, Any] = {}
    if trigger == "circular_logic" and cycle_analysis:
        additional_context["escalation_reason"] = cycle_analysis.escalation_reason
        additional_context["patterns_count"] = len(cycle_analysis.patterns_found)
        additional_context["exchange_count"] = cycle_analysis.total_exchange_count
    elif trigger == "conflict_unresolved" and mediation_result:
        additional_context["conflicts_triggered"] = list(
            mediation_result.escalations_triggered
        )
        additional_context["conflict_count"] = len(
            mediation_result.conflicts_detected
        )
    elif trigger == "system_error" and health_status:
        additional_context["health_summary"] = health_status.summary
        additional_context["health_status"] = health_status.status

    # Step 3: Create escalation request
    request = create_escalation_request(state, trigger, additional_context)

    # Step 4: Return result with pending status
    # The caller will present the request and later call integrate_escalation_response
    logger.info(
        "escalation_request_pending",
        request_id=request.request_id,
        trigger=trigger,
        agent=request.agent,
    )

    # Return a "pending" result - caller will update when response received
    return EscalationResult(
        request=request,
        response=None,
        status="pending",
        resolution_action=None,
        integration_success=False,
        duration_ms=0.0,
    )
