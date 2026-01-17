"""SM agent node for LangGraph orchestration (Story 10.2, 10.6, 10.7, 10.8, 10.9, 10.10, 10.13, 10.14, 10.16).

This module provides the sm_node function that integrates with the
LangGraph orchestration workflow. The SM (Scrum Master) agent serves
as the control plane for orchestration decisions.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All operations use async/await
- **Structured Logging**: Uses structlog for audit trail
- **Routing Decisions**: Determines next agent based on state analysis
- **Circular Logic Detection**: Detects agent ping-pong patterns (>3 exchanges)
- **Enhanced Circular Detection**: Topic-aware, multi-agent cycle detection (Story 10.6)
- **Conflict Mediation**: Mediates conflicts between agents (Story 10.7)
- **Handoff Management**: Manages agent handoffs with context preservation (Story 10.8)
- **Sprint Progress Tracking**: Tracks sprint progress and completion estimates (Story 10.9)
- **Emergency Protocols**: Triggers emergency protocols when health degrades (Story 10.10)
- **Context Injection**: Injects context when agents lack information (Story 10.13)
- **Human Escalation**: Creates escalation requests for human intervention (Story 10.14)
- **Telemetry Collection**: Collects health telemetry for dashboard display (Story 10.16)

Example:
    >>> from yolo_developer.agents.sm import sm_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "sm",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await sm_node(state)
    >>> result["sm_output"]["routing_decision"]
    'analyst'

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply. Per ADR-007,
    SM is the control plane for orchestration decisions.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.sm.circular_detection import detect_circular_logic
from yolo_developer.agents.sm.circular_detection_types import CycleAnalysis
from yolo_developer.agents.sm.conflict_mediation import mediate_conflicts
from yolo_developer.agents.sm.conflict_types import MediationResult
from yolo_developer.agents.sm.context_injection import manage_context_injection
from yolo_developer.agents.sm.context_injection_types import InjectionResult
from yolo_developer.agents.sm.delegation import delegate_task, routing_to_task_type
from yolo_developer.agents.sm.emergency import trigger_emergency_protocol
from yolo_developer.agents.sm.emergency_types import EmergencyConfig, EmergencyProtocol
from yolo_developer.agents.sm.handoff import manage_handoff
from yolo_developer.agents.sm.handoff_types import HandoffResult
from yolo_developer.agents.sm.health import monitor_health
from yolo_developer.agents.sm.health_types import HealthStatus
from yolo_developer.agents.sm.human_escalation import manage_human_escalation
from yolo_developer.agents.sm.human_escalation_types import EscalationResult
from yolo_developer.agents.sm.progress import track_progress
from yolo_developer.agents.sm.progress_types import SprintProgress
from yolo_developer.agents.sm.rollback import coordinate_rollback
from yolo_developer.agents.sm.rollback_types import RollbackResult
from yolo_developer.agents.sm.telemetry import get_dashboard_telemetry
from yolo_developer.agents.sm.telemetry_types import DashboardMetrics
from yolo_developer.agents.sm.types import (
    CIRCULAR_LOGIC_THRESHOLD,
    NATURAL_SUCCESSOR,
    VALID_AGENTS,
    AgentExchange,
    EscalationReason,
    RoutingDecision,
    SMOutput,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


# =============================================================================
# State Analysis Functions (Task 2)
# =============================================================================


def _analyze_current_state(state: YoloState) -> dict[str, Any]:
    """Analyze the current orchestration state.

    Evaluates workflow progress by examining state flags, agent history,
    and handoff context to inform routing decisions.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary containing state analysis results:
        - current_agent: The currently active agent
        - message_count: Number of messages in state
        - has_handoff_context: Whether handoff context is present
        - needs_architecture: Whether architecture work is needed
        - gate_blocked: Whether a gate is blocking progress
        - escalate_to_human: Whether human escalation is requested
        - decision_count: Number of decisions made

    Example:
        >>> analysis = _analyze_current_state(state)
        >>> analysis["needs_architecture"]
        True
    """
    return {
        "current_agent": state.get("current_agent", ""),
        "message_count": len(state.get("messages", [])),
        "has_handoff_context": state.get("handoff_context") is not None,
        "needs_architecture": state.get("needs_architecture", False),
        "gate_blocked": state.get("gate_blocked", False),
        "escalate_to_human": state.get("escalate_to_human", False),
        "decision_count": len(state.get("decisions", [])),
    }


def _count_recent_exchanges(state: YoloState) -> tuple[int, list[AgentExchange]]:
    """Count recent agent exchanges from message history.

    Analyzes messages to build a list of agent-to-agent exchanges
    for circular logic detection.

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (exchange_count, list of AgentExchange objects).

    Example:
        >>> count, exchanges = _count_recent_exchanges(state)
        >>> count
        2
    """
    messages = state.get("messages", [])
    exchanges: list[AgentExchange] = []
    skipped_count = 0

    prev_agent: str | None = None
    for msg in messages:
        # Extract agent from message metadata
        if hasattr(msg, "additional_kwargs"):
            agent = msg.additional_kwargs.get("agent")
            if agent and prev_agent and agent != prev_agent:
                exchange = AgentExchange(
                    source_agent=prev_agent,
                    target_agent=agent,
                    exchange_type="handoff",
                    topic="workflow_transition",
                )
                exchanges.append(exchange)
            prev_agent = agent
        else:
            skipped_count += 1

    if skipped_count > 0:
        logger.debug(
            "exchange_tracking_skipped_messages",
            skipped_count=skipped_count,
            total_messages=len(messages),
        )

    return len(exchanges), exchanges


def _detect_circular_pattern(exchanges: list[AgentExchange]) -> bool:
    """Detect circular logic pattern in exchanges.

    Checks if the same pair of agents have exchanged more than
    CIRCULAR_LOGIC_THRESHOLD times, indicating a ping-pong pattern.

    Args:
        exchanges: List of recent agent exchanges.

    Returns:
        True if circular pattern detected, False otherwise.

    Example:
        >>> _detect_circular_pattern(exchanges)
        True  # If analyst-pm exchanges happened 4+ times
    """
    if len(exchanges) < CIRCULAR_LOGIC_THRESHOLD:
        return False

    # Count exchanges between agent pairs
    pair_counts: dict[tuple[str, str], int] = {}
    for exchange in exchanges:
        # Normalize pair (order doesn't matter for circular detection)
        sorted_agents = sorted([exchange.source_agent, exchange.target_agent])
        pair: tuple[str, str] = (sorted_agents[0], sorted_agents[1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Check if any pair exceeds threshold
    for pair, count in pair_counts.items():
        if count > CIRCULAR_LOGIC_THRESHOLD:
            logger.warning(
                "circular_logic_detected",
                agent_pair=pair,
                exchange_count=count,
                threshold=CIRCULAR_LOGIC_THRESHOLD,
            )
            return True

    return False


def _check_for_escalation(state: YoloState) -> tuple[bool, EscalationReason | None]:
    """Check if escalation to human is needed.

    Examines state flags and conditions that require human intervention.

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (should_escalate, escalation_reason).

    Example:
        >>> should_escalate, reason = _check_for_escalation(state)
        >>> should_escalate
        True
        >>> reason
        'human_requested'
    """
    # Priority 1: Explicit escalation flag
    if state.get("escalate_to_human", False):
        logger.info("escalation_check", result="human_requested")
        return True, "human_requested"

    # Priority 2: Agent failure flag
    if state.get("agent_failure", False):
        logger.info("escalation_check", result="agent_failure")
        return True, "agent_failure"

    return False, None


def _check_for_circular_logic(state: YoloState) -> tuple[bool, list[AgentExchange]]:
    """Check for circular logic pattern in state.

    Analyzes message history to detect agent ping-pong patterns
    that indicate circular logic (per FR12).

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (is_circular, exchanges).

    Example:
        >>> is_circular, exchanges = _check_for_circular_logic(state)
        >>> is_circular
        False
    """
    count, exchanges = _count_recent_exchanges(state)
    is_circular = _detect_circular_pattern(exchanges)

    logger.debug(
        "circular_logic_check",
        exchange_count=count,
        is_circular=is_circular,
    )

    return is_circular, exchanges


def _get_recovery_agent(state: YoloState) -> RoutingDecision:
    """Determine which agent should handle recovery when gate is blocked.

    Analyzes state to determine the appropriate agent to route to
    for fixing gate-blocked issues.

    Args:
        state: Current orchestration state with gate_blocked=True.

    Returns:
        Agent name to route to for recovery (guaranteed valid RoutingDecision).

    Example:
        >>> recovery_agent = _get_recovery_agent(state)
        >>> recovery_agent
        'dev'  # If TEA gate blocked, route back to dev
    """
    current = state.get("current_agent", "")

    # If TEA blocked, route back to dev for fixes
    if current == "tea":
        return "dev"

    # If architect blocked, route back to PM for re-scoping
    if current == "architect":
        return "pm"

    # Default: route to analyst for re-analysis
    return "analyst"


def _get_natural_successor(current_agent: str, state: YoloState) -> RoutingDecision:
    """Get the natural successor agent in the workflow.

    Uses standard workflow progression with state-aware overrides.

    Args:
        current_agent: Currently active agent.
        state: Current orchestration state.

    Returns:
        Next agent in natural workflow progression (guaranteed valid RoutingDecision).

    Example:
        >>> _get_natural_successor("analyst", state)
        'pm'
    """
    # Check for PM-specific routing (needs_architecture flag)
    if current_agent == "pm":
        if state.get("needs_architecture", False):
            return "architect"
        return "dev"  # Skip architect if not needed

    # Use standard succession - NATURAL_SUCCESSOR is typed as dict[str, RoutingDecision]
    return NATURAL_SUCCESSOR.get(current_agent, "analyst")


def _get_next_agent(state: YoloState) -> tuple[RoutingDecision, str]:
    """Determine the next agent based on state analysis.

    Priority-based routing decision:
    1. Check for explicit escalation
    2. Check for circular logic
    3. Check for blocked gates
    4. Normal flow based on current agent

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (routing_decision, rationale).

    Example:
        >>> decision, rationale = _get_next_agent(state)
        >>> decision
        'pm'
        >>> rationale
        'Natural workflow progression from analyst to pm'
    """
    # Priority 1: Check for escalation
    should_escalate, reason = _check_for_escalation(state)
    if should_escalate:
        return "escalate", f"Escalation triggered: {reason}"

    # Priority 2: Check for circular logic
    is_circular, _exchanges = _check_for_circular_logic(state)
    if is_circular:
        return "escalate", "Circular logic detected: agents ping-ponging on same issue"

    # Priority 3: Check for blocked gates
    if state.get("gate_blocked", False):
        recovery = _get_recovery_agent(state)
        return recovery, f"Gate blocked, routing to {recovery} for recovery"

    # Priority 4: Normal flow based on current agent
    current = state.get("current_agent", "")

    # Validate current agent
    if current not in VALID_AGENTS:
        logger.warning("invalid_current_agent", agent=current)
        return "analyst", f"Unknown agent '{current}', defaulting to analyst"

    # Get natural successor
    next_agent = _get_natural_successor(current, state)
    rationale = f"Natural workflow progression from {current} to {next_agent}"

    return next_agent, rationale


def _get_routing_rationale(
    state: YoloState,
    routing_decision: RoutingDecision,
    analysis: dict[str, Any],
) -> str:
    """Generate detailed rationale for routing decision.

    Creates a human-readable explanation of why the routing
    decision was made, including relevant state context.

    Args:
        state: Current orchestration state.
        routing_decision: The chosen routing target.
        analysis: State analysis results.

    Returns:
        Detailed rationale string.

    Example:
        >>> rationale = _get_routing_rationale(state, "pm", analysis)
        >>> rationale
        'Routing to pm: Natural workflow progression...'
    """
    current = analysis.get("current_agent", "unknown")
    msg_count = analysis.get("message_count", 0)
    decision_count = analysis.get("decision_count", 0)

    base_rationale = (
        f"Current agent: {current}, messages: {msg_count}, decisions: {decision_count}. "
    )

    if routing_decision == "escalate":
        if analysis.get("escalate_to_human"):
            return base_rationale + "Human escalation explicitly requested."
        return base_rationale + "Escalation triggered due to circular logic or unresolvable issue."

    if analysis.get("gate_blocked"):
        return base_rationale + f"Gate blocked, routing to {routing_decision} for recovery."

    return base_rationale + f"Natural workflow progression to {routing_decision}."


# =============================================================================
# SM Node Function (Task 3)
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("sm_routing", blocking=False)
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node for orchestration control plane.

    Receives orchestration state and makes routing decisions based on
    state analysis. Handles escalation, circular logic detection, and
    gate-blocked recovery.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state
    - Uses tenacity for retry with exponential backoff

    Args:
        state: Current orchestration state.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        - sm_output: Serialized SMOutput with routing details
        - routing_decision: Target agent for routing (convenience key)

    Example:
        >>> state: YoloState = {
        ...     "messages": [...],
        ...     "current_agent": "analyst",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await sm_node(state)
        >>> result["sm_output"]["routing_decision"]
        'pm'
    """
    logger.info(
        "sm_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Step 1: Analyze current state (AC #2)
    analysis = _analyze_current_state(state)
    logger.debug("state_analysis", **analysis)

    # Step 2: Check for circular logic (AC #4)
    # First do basic check for backward compatibility
    is_circular, exchanges = _check_for_circular_logic(state)
    exchange_count = len(exchanges)

    # Step 2b: Enhanced circular logic detection (Story 10.6 - FR12, FR70)
    cycle_analysis: CycleAnalysis | None = None
    try:
        cycle_analysis = await detect_circular_logic(state)
        if cycle_analysis.circular_detected:
            is_circular = True
            logger.info(
                "enhanced_circular_detection",
                patterns_found=len(cycle_analysis.patterns_found),
                intervention_strategy=cycle_analysis.intervention_strategy,
                escalation_triggered=cycle_analysis.escalation_triggered,
            )
    except Exception as e:
        # Enhanced detection should never block the main workflow
        logger.error("enhanced_circular_detection_failed", error=str(e))

    # Step 3: Check for escalation (AC #4)
    should_escalate, escalation_reason = _check_for_escalation(state)

    # Override escalation if circular logic detected (either basic or enhanced)
    if is_circular and not should_escalate:
        should_escalate = True
        escalation_reason = "circular_logic"

    # Further override if enhanced detection triggered escalation
    if cycle_analysis and cycle_analysis.escalation_triggered and not should_escalate:
        should_escalate = True
        escalation_reason = "circular_logic"

    # Step 4: Determine routing decision (AC #1, #2)
    routing_decision, base_rationale = _get_next_agent(state)
    routing_rationale = _get_routing_rationale(state, routing_decision, analysis)

    # Step 5: Handle gate-blocked recovery (AC #4)
    gate_blocked = analysis.get("gate_blocked", False)
    recovery_agent = _get_recovery_agent(state) if gate_blocked else None

    # Step 6: Delegate task to target agent (Story 10.4 - FR10)
    delegation_result = None
    handoff_context = None
    if routing_decision not in ("escalate", "sm"):
        # Convert routing decision to task type for delegation
        task_type = routing_to_task_type(routing_decision)
        if task_type is not None:
            delegation_result = await delegate_task(
                state=state,
                task_type=task_type,
                task_description=base_rationale,
            )
            if delegation_result.success:
                handoff_context = delegation_result.handoff_context

            logger.debug(
                "delegation_completed",
                target_agent=routing_decision,
                task_type=task_type,
                success=delegation_result.success,
                acknowledged=delegation_result.acknowledged,
            )

    # Step 6b: Monitor health (Story 10.5 - FR11, FR67)
    health_status: HealthStatus | None = None
    try:
        health_status = await monitor_health(state)
        if not health_status.is_healthy:
            logger.warning(
                "health_degraded",
                status=health_status.status,
                alert_count=len(health_status.alerts),
            )
    except Exception as e:
        # Health monitoring should never block the main workflow
        logger.error("health_monitoring_failed", error=str(e))

    # Step 6b2: Context injection (Story 10.13 - FR69)
    # Detects context gaps and injects relevant context for agents
    injection_result: InjectionResult | None = None
    injected_context: dict[str, Any] | None = None
    try:
        injection_result, injected_context = await manage_context_injection(
            state=state,
            memory=None,  # Memory store would be passed from orchestrator in full integration
        )
        if injection_result and injection_result.injected:
            logger.info(
                "context_injection_completed",
                gap_id=injection_result.gap.gap_id,
                gap_reason=injection_result.gap.reason,
                contexts_retrieved=len(injection_result.contexts_retrieved),
                total_context_size=injection_result.total_context_size,
                duration_ms=injection_result.duration_ms,
            )
    except Exception as e:
        # Context injection should never block the main workflow
        logger.error("context_injection_failed", error=str(e))

    # Step 6b3: Emergency protocol (Story 10.10 - FR17, FR70, FR71)
    # Triggers when health status is critical or other emergency conditions
    emergency_protocol: EmergencyProtocol | None = None
    if health_status and health_status.status == "critical":
        try:
            emergency_protocol = await trigger_emergency_protocol(
                state=cast(dict[str, Any], state),
                health_status=health_status,
                config=EmergencyConfig(),
            )
            logger.warning(
                "emergency_protocol_triggered",
                protocol_id=emergency_protocol.protocol_id,
                emergency_type=emergency_protocol.trigger.emergency_type,
                status=emergency_protocol.status,
                selected_action=emergency_protocol.selected_action,
                escalation_reason=emergency_protocol.escalation_reason,
            )

            # Trigger escalation if emergency protocol decided to escalate
            if emergency_protocol.status == "escalated" and not should_escalate:
                should_escalate = True
                escalation_reason = "agent_failure"  # Closest match for emergency escalation
        except Exception as e:
            # Emergency protocol should never block the main workflow
            logger.error("emergency_protocol_failed", error=str(e))

    # Step 6b4: Rollback coordination (Story 10.15 - FR71)
    # Coordinates rollback when emergency_protocol.recommended_action == "rollback"
    rollback_result: RollbackResult | None = None
    if emergency_protocol and emergency_protocol.selected_action == "rollback":
        try:
            rollback_result = await coordinate_rollback(
                state=cast(dict[str, Any], state),
                checkpoint=emergency_protocol.checkpoint,
                emergency_protocol=emergency_protocol,
            )
            if rollback_result:
                logger.info(
                    "rollback_coordination_completed",
                    plan_id=rollback_result.plan.plan_id,
                    status=rollback_result.status,
                    rollback_complete=rollback_result.rollback_complete,
                    steps_executed=rollback_result.steps_executed,
                    steps_failed=rollback_result.steps_failed,
                )

                # If rollback escalated, trigger human escalation
                if rollback_result.status == "escalated" and not should_escalate:
                    should_escalate = True
                    escalation_reason = "agent_failure"  # Rollback failure is an agent failure
        except Exception as e:
            # Rollback coordination should never block the main workflow
            logger.error("rollback_coordination_failed", error=str(e))

    # Step 6c: Conflict mediation (Story 10.7 - FR13)
    mediation_result: MediationResult | None = None
    try:
        mediation_result = await mediate_conflicts(state)
        if mediation_result.conflicts_detected:
            logger.info(
                "conflicts_detected",
                conflict_count=len(mediation_result.conflicts_detected),
                resolved_count=len(mediation_result.resolutions),
                escalation_count=len(mediation_result.escalations_triggered),
                success=mediation_result.success,
            )

            # Check if escalation needed due to unresolved conflicts
            if mediation_result.escalations_triggered and not should_escalate:
                should_escalate = True
                escalation_reason = "conflict_unresolved"
    except Exception as e:
        # Conflict mediation should never block the main workflow
        logger.error("conflict_mediation_failed", error=str(e))

    # Step 6c2: Human escalation (Story 10.14 - FR70)
    # Creates escalation request when any escalation trigger is detected
    # Must run AFTER conflict_mediation since it uses mediation_result
    human_escalation_result: EscalationResult | None = None
    try:
        human_escalation_result = await manage_human_escalation(
            state=state,
            cycle_analysis=cycle_analysis,
            mediation_result=mediation_result,
            health_status=health_status,
        )
        if human_escalation_result:
            logger.info(
                "human_escalation_triggered",
                request_id=human_escalation_result.request.request_id,
                trigger=human_escalation_result.request.trigger,
                status=human_escalation_result.status,
                option_count=len(human_escalation_result.request.options),
            )
    except Exception as e:
        # Human escalation should never block the main workflow
        logger.error("human_escalation_failed", error=str(e))

    # Step 6d: Managed handoff (Story 10.8 - FR14, FR15)
    # Replaces direct handoff_context setting with managed handoff that:
    # - Validates context completeness for target agent
    # - Calculates and logs handoff metrics
    # - Provides audit trail via HandoffRecord
    handoff_result: HandoffResult | None = None
    current_agent = state.get("current_agent", "sm")
    if routing_decision not in ("escalate", "sm") and routing_decision != current_agent:
        try:
            handoff_result = await manage_handoff(
                state=cast(dict[str, Any], state),
                source_agent=current_agent,
                target_agent=routing_decision,
            )
            if handoff_result.success:
                # Use handoff_context from managed handoff instead of delegation
                handoff_context = (
                    handoff_result.state_updates.get("handoff_context")
                    if handoff_result.state_updates
                    else None
                )
                logger.debug(
                    "managed_handoff_completed",
                    source_agent=current_agent,
                    target_agent=routing_decision,
                    context_validated=handoff_result.context_validated,
                    warnings=handoff_result.warnings,
                )
            else:
                logger.warning(
                    "managed_handoff_failed",
                    source_agent=current_agent,
                    target_agent=routing_decision,
                    error=handoff_result.record.error_message,
                )
        except Exception as e:
            # Handoff management should never block the main workflow
            logger.error("handoff_management_failed", error=str(e))

    # Step 6e: Sprint progress tracking (Story 10.9 - FR16, FR66)
    sprint_progress: SprintProgress | None = None
    sprint_plan_dict = state.get("sprint_plan")
    # Ensure sprint_plan is a dict for type safety
    if sprint_plan_dict is not None and isinstance(sprint_plan_dict, dict):
        try:
            sprint_progress = await track_progress(
                state=state,
                sprint_plan=sprint_plan_dict,
            )
            logger.debug(
                "sprint_progress_tracked",
                sprint_id=sprint_progress.snapshot.sprint_id,
                progress_percentage=sprint_progress.snapshot.progress_percentage,
                stories_completed=sprint_progress.snapshot.stories_completed,
                total_stories=sprint_progress.snapshot.total_stories,
                estimated_completion=sprint_progress.completion_estimate.estimated_completion_time
                if sprint_progress.completion_estimate
                else None,
            )
        except Exception as e:
            # Progress tracking should never block the main workflow
            logger.error("sprint_progress_tracking_failed", error=str(e))

    # Step 6f: Telemetry collection for dashboard (Story 10.16 - FR72)
    # Collects health telemetry data for dashboard display
    telemetry_snapshot: DashboardMetrics | None = None
    try:
        # Import velocity module to get velocity metrics if available
        from yolo_developer.agents.sm.velocity import calculate_velocity_metrics
        from yolo_developer.agents.sm.velocity_types import VelocityMetrics

        # Try to get velocity from state history or calculate from sprint progress
        velocity_metrics: VelocityMetrics | None = None
        velocity_history = state.get("velocity_history", [])
        if velocity_history and isinstance(velocity_history, list):
            # Calculate aggregate velocity from history
            velocity_metrics = calculate_velocity_metrics(velocity_history)

        telemetry_snapshot = await get_dashboard_telemetry(
            state=state,
            health_status=health_status,
            velocity_metrics=velocity_metrics,
        )
        logger.debug(
            "telemetry_collected",
            velocity_display=telemetry_snapshot.velocity_display,
            cycle_time_display=telemetry_snapshot.cycle_time_display,
            health_summary=telemetry_snapshot.health_summary,
            agent_count=len(telemetry_snapshot.agent_status_table),
        )
    except Exception as e:
        # Telemetry collection should never block the main workflow
        logger.error("telemetry_collection_failed", error=str(e))

    # Step 7: Create processing notes
    processing_notes = (
        f"Analyzed state with {analysis['message_count']} messages, "
        f"{analysis['decision_count']} decisions. "
        f"Exchange count: {exchange_count}. "
        f"Circular: {is_circular}. "
        f"Gate blocked: {gate_blocked}."
    )
    if cycle_analysis and cycle_analysis.circular_detected:
        processing_notes += (
            f" Enhanced detection: {len(cycle_analysis.patterns_found)} patterns, "
            f"intervention={cycle_analysis.intervention_strategy}."
        )
    if delegation_result:
        processing_notes += f" Delegation to {routing_decision}: {'success' if delegation_result.success else 'failed'}."
    if health_status:
        processing_notes += f" Health: {health_status.status} ({len(health_status.alerts)} alerts)."
    if emergency_protocol:
        processing_notes += (
            f" Emergency: {emergency_protocol.trigger.emergency_type}, "
            f"status={emergency_protocol.status}, "
            f"action={emergency_protocol.selected_action}."
        )
    if mediation_result and mediation_result.conflicts_detected:
        processing_notes += (
            f" Mediation: {len(mediation_result.conflicts_detected)} conflicts, "
            f"{len(mediation_result.resolutions)} resolved, "
            f"{len(mediation_result.escalations_triggered)} escalated."
        )
    if handoff_result:
        processing_notes += (
            f" Handoff {current_agent}->{routing_decision}: "
            f"{'success' if handoff_result.success else 'failed'}, "
            f"validated={handoff_result.context_validated}."
        )
    if sprint_progress:
        processing_notes += (
            f" Sprint: {sprint_progress.snapshot.progress_percentage:.1f}% complete "
            f"({sprint_progress.snapshot.stories_completed}/{sprint_progress.snapshot.total_stories} stories)."
        )
    if injection_result and injection_result.injected:
        processing_notes += (
            f" Context injection: {injection_result.gap.reason}, "
            f"{len(injection_result.contexts_retrieved)} contexts, "
            f"{injection_result.total_context_size} bytes."
        )
    if human_escalation_result:
        processing_notes += (
            f" Escalation: trigger={human_escalation_result.request.trigger}, "
            f"status={human_escalation_result.status}, "
            f"options={len(human_escalation_result.request.options)}."
        )
    if rollback_result:
        processing_notes += (
            f" Rollback: plan={rollback_result.plan.plan_id}, "
            f"status={rollback_result.status}, "
            f"complete={rollback_result.rollback_complete}, "
            f"steps={rollback_result.steps_executed}/{len(rollback_result.plan.steps)}."
        )
    if telemetry_snapshot:
        processing_notes += (
            f" Telemetry: {telemetry_snapshot.health_summary}, "
            f"velocity={telemetry_snapshot.velocity_display}."
        )

    # Create SM output (AC #3 - structured format)
    output = SMOutput(
        routing_decision=routing_decision,
        routing_rationale=routing_rationale,
        circular_logic_detected=is_circular,
        escalation_triggered=should_escalate,
        escalation_reason=escalation_reason,
        exchange_count=exchange_count,
        recent_exchanges=tuple(exchanges[-10:]),  # Keep last 10 exchanges
        gate_blocked=gate_blocked,
        recovery_agent=recovery_agent,
        processing_notes=processing_notes,
        delegation_result=delegation_result.to_dict() if delegation_result else None,
        health_status=health_status.to_dict() if health_status else None,
        cycle_analysis=cycle_analysis.to_dict() if cycle_analysis else None,
        mediation_result=mediation_result.to_dict() if mediation_result else None,
        handoff_result=handoff_result.to_dict() if handoff_result else None,
        sprint_progress=sprint_progress.to_dict() if sprint_progress else None,
        emergency_protocol=emergency_protocol.to_dict() if emergency_protocol else None,
        injection_result=injection_result.to_dict() if injection_result else None,
        escalation_result=human_escalation_result.to_dict() if human_escalation_result else None,
        rollback_result=rollback_result.to_dict() if rollback_result else None,
        telemetry_snapshot=telemetry_snapshot.to_dict() if telemetry_snapshot else None,
    )

    # Create decision record (includes delegation info for audit trail - Task 4.2)
    delegation_summary = ""
    if delegation_result:
        delegation_summary = f" Delegated {delegation_result.request.task_type} task."
    decision = Decision(
        agent="sm",
        summary=f"Routing to {routing_decision}.{delegation_summary}",
        rationale=routing_rationale,
        related_artifacts=("delegation",) if delegation_result else (),
    )

    # Create output message
    message = create_agent_message(
        content=f"SM decision: route to {routing_decision}. {base_rationale}",
        agent="sm",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "sm_node_complete",
        routing_decision=routing_decision,
        circular_detected=is_circular,
        escalation_triggered=should_escalate,
        exchange_count=exchange_count,
        delegation_success=delegation_result.success if delegation_result else None,
        health_status=health_status.status if health_status else None,
        cycle_patterns_found=len(cycle_analysis.patterns_found) if cycle_analysis else 0,
        cycle_intervention=cycle_analysis.intervention_strategy if cycle_analysis else None,
        conflicts_detected=len(mediation_result.conflicts_detected) if mediation_result else 0,
        mediation_success=mediation_result.success if mediation_result else None,
        handoff_success=handoff_result.success if handoff_result else None,
        handoff_validated=handoff_result.context_validated if handoff_result else None,
        sprint_progress_percentage=sprint_progress.snapshot.progress_percentage
        if sprint_progress
        else None,
        sprint_stories_completed=sprint_progress.snapshot.stories_completed
        if sprint_progress
        else None,
        emergency_protocol_status=emergency_protocol.status if emergency_protocol else None,
        emergency_type=emergency_protocol.trigger.emergency_type if emergency_protocol else None,
        context_injection_triggered=injection_result.injected if injection_result else None,
        context_gap_reason=injection_result.gap.reason if injection_result else None,
        human_escalation_triggered=human_escalation_result is not None,
        human_escalation_trigger=human_escalation_result.request.trigger
        if human_escalation_result
        else None,
        human_escalation_status=human_escalation_result.status if human_escalation_result else None,
        rollback_triggered=rollback_result is not None,
        rollback_status=rollback_result.status if rollback_result else None,
        rollback_complete=rollback_result.rollback_complete if rollback_result else None,
        rollback_steps_executed=rollback_result.steps_executed if rollback_result else None,
        telemetry_collected=telemetry_snapshot is not None,
        telemetry_health_summary=telemetry_snapshot.health_summary if telemetry_snapshot else None,
        telemetry_velocity=telemetry_snapshot.velocity_display if telemetry_snapshot else None,
    )

    # Return ONLY updates (AC #1, #2, #3, #4)
    # Includes handoff_context for state updates (Task 7.4)
    # Includes health_status for current snapshot (Story 10.5)
    # Includes health_history for trend analysis (Story 10.5 Task 7.4)
    # health_history accumulates snapshots over time for trend analysis
    health_snapshot = None
    if health_status:
        health_snapshot = {
            "status": health_status.status,
            "is_healthy": health_status.is_healthy,
            "alert_count": len(health_status.alerts),
            "overall_churn_rate": health_status.metrics.overall_churn_rate,
            "unproductive_churn_rate": health_status.metrics.unproductive_churn_rate,
            "cycle_time_percentiles": health_status.metrics.cycle_time_percentiles,
            "agent_idle_times": health_status.metrics.agent_idle_times,
            "evaluated_at": health_status.evaluated_at,
        }

    return {
        "messages": [message],
        "decisions": [decision],
        "sm_output": output.to_dict(),
        "routing_decision": routing_decision,  # Convenience key for routing
        "handoff_context": handoff_context,  # From delegation (Story 10.4)
        "health_status": health_status.to_dict()
        if health_status
        else None,  # Current snapshot (Story 10.5)
        "health_history": [health_snapshot]
        if health_snapshot
        else [],  # For trend analysis (Task 7.4)
        "cycle_analysis": cycle_analysis.to_dict()
        if cycle_analysis
        else None,  # Enhanced detection (Story 10.6)
        "mediation_result": mediation_result.to_dict()
        if mediation_result
        else None,  # Conflict mediation (Story 10.7)
        "handoff_result": handoff_result.to_dict()
        if handoff_result
        else None,  # Handoff management (Story 10.8)
        "sprint_progress": sprint_progress.to_dict()
        if sprint_progress
        else None,  # Sprint progress tracking (Story 10.9)
        "emergency_protocol": emergency_protocol.to_dict()
        if emergency_protocol
        else None,  # Emergency protocol (Story 10.10)
        "injected_context": injected_context,  # Context injection payload (Story 10.13)
        "injection_result": injection_result.to_dict()
        if injection_result
        else None,  # Context injection result (Story 10.13)
        "escalation_result": human_escalation_result.to_dict()
        if human_escalation_result
        else None,  # Human escalation (Story 10.14)
        "rollback_result": rollback_result.to_dict()
        if rollback_result
        else None,  # Rollback coordination (Story 10.15)
        "telemetry_snapshot": telemetry_snapshot.to_dict()
        if telemetry_snapshot
        else None,  # Dashboard telemetry (Story 10.16)
    }
